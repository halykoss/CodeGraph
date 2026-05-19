#!/usr/bin/env python3
"""
Build the public CodeGraph dataset.

Reads:
  - Arrow pipeline files in output/full_pipeline, one row per annotated source
    file, plus any legacy JSON shards in the same directory;
  - Wikidata grounding caches in ./cache/wikidata_*, one JSON per concept;
  - Wikidata hierarchy files in ./hierarchy_wikidata/wikidata_parents.json and
    ./hierarchy_wikidata/wikidata_hierarchy.json.

Writes a normalized Parquet-only tree to:
    codegraph_release_v1/

Layout:
    files.parquet                       file_id, sample_id, language
    concepts_algorithms.parquet         concept_id, name, wikidata_qid, label
    concepts_algorithm_categories.parquet
    concepts_algorithm_complexities.parquet
    concepts_domains.parquet            concept_id, name, wikidata_qid, label
    concepts_paradigms.parquet          concept_id, name, wikidata_qid, label
    concepts_design_patterns.parquet    concept_id, name, wikidata_qid, label, category
    edges_file_algorithm.parquet        file_id, concept_id
    edges_algorithm_category.parquet     algorithm_concept_id, category_id
    edges_algorithm_complexity.parquet   algorithm_concept_id, complexity_id
    edges_file_domain.parquet           file_id, concept_id
    edges_file_paradigm.parquet         file_id, concept_id, confidence
    edges_file_design_pattern.parquet   file_id, concept_id, description
    wikidata_entities.parquet           qid, label, description
    wikidata_parent_of.parquet          child_qid, parent_qid, via, depth

Concept-name canonicalization uses normalize_text().
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pyarrow as pa
import pyarrow.ipc as pa_ipc
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[1]
ARROW_DIR = Path("output/full_pipeline")
CACHE_DIR = REPO / "cache"
HIER_DIR = REPO / "hierarchy_wikidata"
OUT_DIR = Path("codegraph_release_v1")
STAGING_DIR = OUT_DIR / "_flat"  # temporary, deleted after partitioning
DUCK_TMP_DIR = Path("duckdb_tmp")

# DuckDB does not allow FILE_SIZE_BYTES together with PARTITION_BY in COPY.
PART_FILE_SIZE = None

# Map: release axis -> cache subdirectory
CACHE_FOR_AXIS: Dict[str, str] = {
    "algorithms": "wikidata_algorithms",
    "domains": "wikidata_domains",
    "paradigms": "wikidata_paradigms",
    "design_patterns": "wikidata_design_patterns",
}

# Only domains have a curated stem-cluster map; for the other axes,
# normalize_text() is the only canonicalization step.
STEM_MAP_PATH = REPO / "output" / "domains" / "stem_results.json"

# Tie-break when multiple cache files collapse onto the same normalized key.
_CONF_RANK = {"high": 0, "medium": 1, "low": 2, None: 3, "": 3}
_STATUS_RANK = {"found": 0, "not_found": 1, "discarded": 2}

PARQUET_COMPRESSION = "zstd"
PARQUET_ROW_GROUP = 256_000


# ─────────────────────────── Morphological stemmer ──────────────────────────
# Mirrors analyzer/stem_domains.py: Snowball when NLTK is available, with a
# rule-based fallback otherwise. Enables matching inflectional variants.
try:
    from nltk.stem import SnowballStemmer  # type: ignore[import-not-found]
    _SNOWBALL = SnowballStemmer("english")
except Exception:
    _SNOWBALL = None

_IRREGULAR_STEMS: Dict[str, str] = {
    "data": "data", "criteria": "criterion", "analysis": "analysis",
    "analyses": "analysis", "matrices": "matrix", "indices": "index",
    "algorithms": "algorithm", "heuristics": "heuristic",
    "statistics": "statistic", "semantics": "semantic", "dynamics": "dynamic",
    "graphics": "graphic", "metrics": "metric", "robotics": "robotic",
    "economics": "economic", "mathematics": "mathematic", "physics": "physic",
}


def _simple_stem(word: str) -> str:
    w = word.lower().strip()
    if w in _IRREGULAR_STEMS:
        return _IRREGULAR_STEMS[w]
    for suf, repl, min_len in (
        ("ies", "y", 4), ("ves", "f", 4),
        ("ings", "", 5), ("ing", "", 5),
        ("tions", "", 6), ("tion", "", 5),
        ("ness", "", 5), ("ments", "", 6), ("ment", "", 5),
        ("ed", "", 4), ("es", "", 4),
    ):
        if w.endswith(suf) and len(w) >= min_len:
            return w[: -len(suf)] + repl
    if w.endswith("s") and len(w) > 2 and not w.endswith("ss"):
        return w[:-1]
    return w


def stem_token(word: str) -> str:
    w = word.lower().strip()
    if not w:
        return ""
    if w in _IRREGULAR_STEMS:
        return _IRREGULAR_STEMS[w]
    if _SNOWBALL is not None:
        return _SNOWBALL.stem(w)
    return _simple_stem(w)


MIN_STEM_LEN = 4  # per-token: shorter stems are too ambiguous for fallback


def stem_key(text: str) -> str:
    """Per-token stemmed canonical key — for fallback matching.

    Returns "" when any token stems to fewer than MIN_STEM_LEN chars, to keep
    false positives down on short ambiguous tokens (e.g. "iot" vs "ios").
    """
    tokens = text.split()
    if not tokens:
        return ""
    stems = [stem_token(t) for t in tokens]
    if any(len(s) < MIN_STEM_LEN for s in stems):
        return ""
    return " ".join(stems)


def normalize_text(text: object) -> str:
    """Canonical key normalisation, unifying the variant forms found across
    the data sources:
      - cache filenames preserve underscores/dashes from the extract step
        (e.g. `web_development.json`, `web-development.json`),
      - cache `algorithm`/`domain` fields can carry any of underscore, dash,
        or space spellings depending on the upstream extract run,
      - the LLM annotations in the Arrow files use spaces.
    This lets the release vocabulary collapse all these forms onto the same key.
    """
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    text = text.lower()
    text = text.replace("c++", "cpp").replace("c#", "csharp").replace("f#", "fsharp")
    text = re.sub(r"[-_/\\|]+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return " ".join(text.split())


def normalize_complexity(text: object) -> Optional[str]:
    """Normalize algorithm complexity labels.

    Complexity strings intentionally preserve symbols and case; only whitespace
    and the common time/space phrasings are canonicalized.
    """
    if not isinstance(text, str):
        return str(text) if text is not None else None

    text = " ".join(text.split())

    match = re.match(r"^Time:\s*(.+?)(?:,|;)?\s*Space:\s*(.+)$", text, re.IGNORECASE)
    if match:
        time_comp = match.group(1).strip()
        space_comp = match.group(2).strip()
        return f"{time_comp} time, {space_comp} space"

    match = re.match(r"^(.+?)\s+time\s+and\s+space(?:\s+complexity)?$", text, re.IGNORECASE)
    if match:
        val = match.group(1).strip()
        return f"{val} time, {val} space"

    return text


# ─────────────────────────── Stem (domain) loader ────────────────────────────


def load_domain_stem_map() -> Dict[str, str]:
    """Returns variant_to_canonical (both keys lower+trim).

    Identity mappings are excluded — only true collapses are returned.
    """
    if not STEM_MAP_PATH.exists():
        print(f"  [warn] domain stem map missing: {STEM_MAP_PATH}", file=sys.stderr)
        return {}
    with STEM_MAP_PATH.open() as f:
        data = json.load(f)
    variant_to_canonical: Dict[str, str] = {}
    for cluster in data.get("clusters", []):
        canon = normalize_text(cluster.get("canonical") or "")
        if not canon:
            continue
        for variant in cluster.get("variants", []):
            v = normalize_text(variant)
            if v and v != canon:
                variant_to_canonical[v] = canon
    return variant_to_canonical


# ─────────────────────────── Wikidata lookups ────────────────────────────────


def _better_entry(a: dict, b: dict) -> dict:
    """Pick the 'best' between two cache entries for the same normalised key.

    Order: status (found > not_found > discarded), then confidence
    (high > medium > low), then lexicographic filename for stability.
    """
    rank_a = (
        _STATUS_RANK.get(a.get("status"), 9),
        _CONF_RANK.get((a.get("confidence") or "").lower(), 3),
        a.get("_filename", ""),
    )
    rank_b = (
        _STATUS_RANK.get(b.get("status"), 9),
        _CONF_RANK.get((b.get("confidence") or "").lower(), 3),
        b.get("_filename", ""),
    )
    return a if rank_a <= rank_b else b


def load_vocab_from_cache(
    axis: str,
    variant_to_canonical: Optional[Dict[str, str]] = None,
) -> Vocab:
    """Build a closed Vocab from the Wikidata-merge cache directory.

    Cache keys are `lower(trim(raw_label))` — matching the input vocabulary
    fed into the merger by extract_algorithms_for_wikidata.py (filtered to
    concepts occurring in >200 source files). Multiple cache files that
    collapse onto the same key are deduplicated via _better_entry; for
    domains, stem-cluster variants further collapse onto their canonical.
    """
    sub = CACHE_DIR / CACHE_FOR_AXIS[axis]
    by_key: Dict[str, dict] = {}
    counts: Dict[str, int] = {}
    # Every distinct raw label encountered per normalized key (for aliases).
    raw_labels: Dict[str, List[str]] = {}
    if not sub.is_dir():
        print(f"  [warn] cache dir missing: {sub}", file=sys.stderr)
        return Vocab(variant_to_canonical)
    for fp in sub.iterdir():
        if fp.suffix != ".json":
            continue
        try:
            with fp.open() as f:
                d = json.load(f)
        except Exception:
            continue
        raw = (
            d.get("algorithm")
            or d.get("domain")
            or d.get("paradigm")
            or d.get("design_pattern")
            or d.get("concept")
            or fp.stem
        )
        key = normalize_text(raw)
        if not key:
            continue
        counts[key] = counts.get(key, 0) + 1
        raw_clean = raw.strip() if isinstance(raw, str) else str(raw)
        bucket = raw_labels.setdefault(key, [])
        if raw_clean and raw_clean not in bucket:
            bucket.append(raw_clean)
        d["_filename"] = fp.name
        d["_raw_label"] = raw_clean
        existing = by_key.get(key)
        by_key[key] = d if existing is None else _better_entry(existing, d)

    vocab = Vocab(variant_to_canonical)
    # Sort so canonical (= stem-target) entries register first; later variant
    # entries collapse into the same cid and just extend cache_spellings.
    for key, d in sorted(by_key.items()):
        target_key = (variant_to_canonical or {}).get(key, key)
        is_found = d.get("status") == "found" and bool(d.get("wikidata_id"))
        # Prefer the Wikidata label as the primary display when grounded;
        # falls back to the raw cache spelling otherwise.
        primary_display = (
            d.get("label") if is_found and d.get("label") else d.get("_raw_label")
        ) or target_key
        cid = vocab._register(
            target_key,
            primary_display,
            d.get("wikidata_id") if is_found else None,
            (d.get("label") or None) if is_found else None,
            (d.get("description") or None) if is_found else None,
            counts.get(key, 1),
        )
        vocab.add_cache_spellings(cid, raw_labels.get(key, ()))
    return vocab


# ────────────────────────── Vocabulary builder ───────────────────────────────


class Vocab:
    """Closed concept vocabulary, preloaded from a Wikidata-merge cache.

    `lookup(raw_name)` returns the concept_id if the (lower+trim) key is in
    vocab, otherwise None — meaning the concept did not survive the >200-file
    occurrence filter applied by extract_algorithms_for_wikidata.py and is
    not part of the released graph.

    For axes with a stem map (only domains today), `variant_to_canonical`
    additionally collapses cluster variants onto canonical keys.
    """

    __slots__ = (
        "name_to_id", "normalized_names", "display", "qids", "labels",
        "descriptions", "n_cache_variants", "cache_spellings",
        "variant_to_canonical", "_stem_index",
    )

    def __init__(
        self,
        variant_to_canonical: Optional[Dict[str, str]] = None,
    ) -> None:
        self.name_to_id: Dict[str, int] = {}
        self.normalized_names: List[str] = []      # canonical join key (= name_to_id reverse)
        self.display: List[str] = []               # primary display name
        self.qids: List[Optional[str]] = []
        self.labels: List[Optional[str]] = []
        self.descriptions: List[Optional[str]] = []
        self.n_cache_variants: List[int] = []
        # Distinct raw spellings found in the cache for this concept (the
        # forms that the upstream extract step counted as >200-occurrence).
        self.cache_spellings: List[List[str]] = []
        self.variant_to_canonical: Dict[str, str] = variant_to_canonical or {}
        # Per-token stemmed canonical → concept_id (or None if ambiguous).
        # Built on demand via build_stem_index(); used for morphology fallback.
        self._stem_index: Dict[str, Optional[int]] = {}

    def _register(self, key: str, display: str, qid: Optional[str],
                  label: Optional[str], description: Optional[str],
                  count: int) -> int:
        cid = self.name_to_id.get(key)
        if cid is None:
            cid = len(self.display)
            self.name_to_id[key] = cid
            self.normalized_names.append(key)
            self.display.append(display)
            self.qids.append(qid)
            self.labels.append(label)
            self.descriptions.append(description)
            self.n_cache_variants.append(count)
            self.cache_spellings.append([])
        else:
            self.n_cache_variants[cid] += count
            # prefer a grounded variant if the canonical wasn't grounded
            if self.qids[cid] is None and qid is not None:
                self.qids[cid] = qid
                self.labels[cid] = label
                self.descriptions[cid] = description
        return cid

    def add_cache_spellings(self, cid: int, spellings: Iterable[str]) -> None:
        bucket = self.cache_spellings[cid]
        for s in spellings:
            if s and s not in bucket:
                bucket.append(s)

    def build_stem_index(self) -> None:
        """Index concepts by per-token stemmed key; mark ambiguous entries."""
        idx: Dict[str, Optional[int]] = {}
        seen: Dict[str, int] = {}
        for cid, key in enumerate(self.normalized_names):
            sk = stem_key(key)
            if not sk or sk == key:
                continue
            if sk in seen and seen[sk] != cid:
                idx[sk] = None  # ambiguous → don't use for fallback
            else:
                seen[sk] = cid
                idx[sk] = cid
        # Also stem cache_spellings for richer coverage
        for cid, spellings in enumerate(self.cache_spellings):
            for s in spellings:
                sk = stem_key(normalize_text(s))
                if not sk:
                    continue
                if sk in seen and seen[sk] != cid:
                    idx[sk] = None
                else:
                    seen[sk] = cid
                    idx.setdefault(sk, cid)
        self._stem_index = {k: v for k, v in idx.items() if v is not None}

    def lookup(self, raw_name: str) -> Optional[int]:
        key = normalize_text(raw_name)
        if not key:
            return None
        key = self.variant_to_canonical.get(key, key)
        cid = self.name_to_id.get(key)
        if cid is not None:
            return cid
        return self._stem_index.get(stem_key(key))


# ─────────────────────────── Arrow iteration ─────────────────────────────────


FIELDS = ("sample_id", "language", "domains", "algorithms", "paradigms", "design_patterns")


def _coerce_list(value) -> list:
    """Normalise array-valued fields to a Python list.

    Arrow rows store them as JSON-encoded strings; JSON rows store them as
    native lists. None / unparseable → empty list.
    """
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        if not value:
            return []
        try:
            v = json.loads(value)
        except Exception:
            return []
        return v if isinstance(v, list) else []
    return []


def iter_arrow_rows(arrow_dir: Path, batch_size: int = 50_000) -> Iterable[dict]:
    """Yield rows from .arrow shards, with array fields already parsed."""
    files = sorted(arrow_dir.glob("processing_results_part_*.arrow"))
    print(f"  found {len(files)} arrow shards", flush=True)
    for i, fp in enumerate(files):
        try:
            with pa.memory_map(str(fp), "r") as src:
                try:
                    reader = pa.ipc.open_file(src)
                    table = reader.read_all()
                except pa.lib.ArrowInvalid:
                    src.seek(0)
                    reader = pa.ipc.open_stream(src)
                    table = reader.read_all()
        except Exception as e:
            print(f"  [warn] skipping {fp.name}: {e}", file=sys.stderr)
            continue
        for batch in table.to_batches(max_chunksize=batch_size):
            b = batch.to_pydict()
            n = len(b.get("sample_id", []))
            for j in range(n):
                yield {f: b.get(f, [None] * n)[j] for f in FIELDS}
        if (i + 1) % 200 == 0:
            print(f"  …read {i + 1}/{len(files)} arrow shards", flush=True)


def iter_json_rows(arrow_dir: Path) -> Iterable[dict]:
    """Yield rows from legacy .json shards (results[]) for completeness."""
    files = sorted(arrow_dir.glob("processing_results_part_*.json"))
    print(f"  found {len(files)} json shards", flush=True)
    for fp in files:
        try:
            with fp.open() as f:
                d = json.load(f)
        except Exception as e:
            print(f"  [warn] skipping {fp.name}: {e}", file=sys.stderr)
            continue
        for r in d.get("results", []):
            yield r


def json_rows_to_arrow_table(rows: Iterable[dict]) -> pa.Table:
    """Convert legacy JSON rows to the string-encoded shape of Arrow shards."""
    cols = {field: [] for field in FIELDS}
    for row in rows:
        cols["sample_id"].append(row.get("sample_id"))
        cols["language"].append(row.get("language"))
        for field in ("domains", "algorithms", "paradigms", "design_patterns"):
            cols[field].append(
                json.dumps(_coerce_list(row.get(field)), ensure_ascii=False)
            )
    return pa.table({
        "sample_id": pa.array(cols["sample_id"], pa.string()),
        "language": pa.array(cols["language"], pa.string()),
        "domains": pa.array(cols["domains"], pa.string()),
        "algorithms": pa.array(cols["algorithms"], pa.string()),
        "paradigms": pa.array(cols["paradigms"], pa.string()),
        "design_patterns": pa.array(cols["design_patterns"], pa.string()),
    })


# ───────────────────────────── Pass orchestration ────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the CodeGraph public release dataset from Arrow annotations and Wikidata caches."
    )
    parser.add_argument(
        "--arrow-dir",
        type=Path,
        default=Path("output/full_pipeline"),
        help="Directory containing processing_results_part_*.arrow shards.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=REPO / "cache",
        help="Directory containing wikidata_domains/, wikidata_algorithms/, wikidata_paradigms/, and wikidata_design_patterns/.",
    )
    parser.add_argument(
        "--hierarchy-dir",
        type=Path,
        default=REPO / "hierarchy_wikidata",
        help="Directory containing wikidata_parents.json and/or wikidata_hierarchy.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("codegraph_release_v1"),
        help="Output directory for the Parquet release tree.",
    )
    parser.add_argument(
        "--duckdb-temp-dir",
        type=Path,
        default=Path("duckdb_tmp"),
        help="Temporary spill directory for DuckDB.",
    )
    return parser.parse_args()


def main() -> None:
    global ARROW_DIR, CACHE_DIR, HIER_DIR, OUT_DIR, STAGING_DIR, DUCK_TMP_DIR

    args = parse_args()
    ARROW_DIR = args.arrow_dir
    CACHE_DIR = args.cache_dir
    HIER_DIR = args.hierarchy_dir
    OUT_DIR = args.output_dir
    STAGING_DIR = OUT_DIR / "_flat"
    DUCK_TMP_DIR = args.duckdb_temp_dir

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    STAGING_DIR.mkdir(parents=True, exist_ok=True)

    # --- 1. Closed vocab from Wikidata caches + domain stem -----------------
    print("[1/3] Building closed vocab from Wikidata-merge caches…", flush=True)
    print("  loading domain stem map…", flush=True)
    domain_variant_to_canonical = load_domain_stem_map()
    print(
        f"  domain stem: {len(domain_variant_to_canonical)} variant→canonical",
        flush=True,
    )

    vocabs: Dict[str, Vocab] = {}
    for axis in CACHE_FOR_AXIS:
        v2c = domain_variant_to_canonical if axis == "domains" else None
        vocabs[axis] = load_vocab_from_cache(axis, v2c)
        vocabs[axis].build_stem_index()
        n_grounded = sum(1 for q in vocabs[axis].qids if q)
        print(
            f"  {axis}: vocab size = {len(vocabs[axis].display):,}, "
            f"grounded = {n_grounded:,}, "
            f"stem fallback entries = {len(vocabs[axis]._stem_index):,}",
            flush=True,
        )

    # --- 2. DuckDB pipeline: materialise files + emit hive-partitioned edges -
    print(f"[2/3] DuckDB-driven extraction → {OUT_DIR}", flush=True)
    import duckdb
    import pyarrow.dataset as pad

    arrow_files = sorted(ARROW_DIR.glob("processing_results_part_*.arrow"))
    json_files = sorted(ARROW_DIR.glob("processing_results_part_*.json"))
    if not arrow_files and not json_files:
        raise RuntimeError(f"No Arrow or JSON result shards in {ARROW_DIR}")
    print(f"  {len(arrow_files)} arrow shards registered", flush=True)
    if json_files:
        print(f"  {len(json_files)} legacy json shards found", flush=True)

    con = duckdb.connect()
    con.execute("PRAGMA threads=4;")
    # Allow DuckDB to spill freely and stream PARTITION_BY writes instead of
    # buffering them in insertion order (the latter blows up RAM with 14
    # language partitions × per-thread buffers).
    con.execute("PRAGMA preserve_insertion_order=false;")
    con.execute("PRAGMA memory_limit='10GB';")
    DUCK_TMP_DIR.mkdir(parents=True, exist_ok=True)
    con.execute(f"PRAGMA temp_directory='{DUCK_TMP_DIR}';")

    # SQL twin of Python normalize_text() — collapses separators (-, _, /, |),
    # strips accents, applies c++/c#/f# expansions, removes other punctuation
    # and collapses whitespace. Used everywhere a raw concept name is matched
    # against the vocab so cache (often underscore/dash) and arrow (usually
    # spaces) forms unify.
    con.execute(r"""
        CREATE OR REPLACE MACRO norm_key(s) AS (
            TRIM(REGEXP_REPLACE(
                REGEXP_REPLACE(
                    REGEXP_REPLACE(
                        REPLACE(REPLACE(REPLACE(
                            LOWER(STRIP_ACCENTS(s)),
                            'c++', 'cpp'), 'c#', 'csharp'), 'f#', 'fsharp'),
                        '[-_/\\|]+', ' ', 'g'),
                    '[^a-z0-9 ]', '', 'g'),
                '\s+', ' ', 'g'))
        );
    """)

    # Vocab tables (small, in-memory pyarrow → DuckDB view)
    vocab_arrow_refs: List[pa.Table] = []  # keep alive for DuckDB scans
    for axis, vocab in vocabs.items():
        keys_by_id = [""] * len(vocab.display)
        for k, cid in vocab.name_to_id.items():
            keys_by_id[cid] = k
        vt = pa.table({
            "concept_id": pa.array(range(len(keys_by_id)), pa.uint32()),
            "key": pa.array(keys_by_id, pa.string()),
        })
        vocab_arrow_refs.append(vt)
        con.register(f"vocab_{axis}", vt)

    # 2a. materialise files_full chunked (per-shard ROW_NUMBER + global offset)
    files_full_dir = STAGING_DIR / "files_full"
    CHUNK = 200  # shards per chunk
    cum_offset = 0
    n_arrow_chunks = (len(arrow_files) + CHUNK - 1) // CHUNK
    n_json_chunks = len(json_files)
    n_chunks = n_arrow_chunks + n_json_chunks
    expected_chunks = [files_full_dir / f"chunk_{ci:04d}.parquet" for ci in range(n_chunks)]
    existing_chunks = sorted(files_full_dir.glob("chunk_*.parquet"))
    reuse_files_full = existing_chunks == expected_chunks

    if reuse_files_full:
        print(f"  reusing existing files_full chunks → {files_full_dir.name}/…", flush=True)
        for ci, chunk_out in enumerate(expected_chunks):
            chunk_rows = con.execute(
                f"SELECT count(*) FROM '{chunk_out}'"
            ).fetchone()[0]
            cum_offset += chunk_rows
            if (ci + 1) % 5 == 0 or ci == n_chunks - 1:
                print(
                    f"    chunk {ci + 1}/{n_chunks}: +{chunk_rows:,} rows "
                    f"(total {cum_offset:,})",
                    flush=True,
                )
    else:
        if files_full_dir.exists():
            for child in files_full_dir.iterdir():
                child.unlink()
        files_full_dir.mkdir(parents=True, exist_ok=True)
        print(f"  materialising files_full (chunked) → {files_full_dir.name}/…", flush=True)
        for ci in range(n_arrow_chunks):
            chunk_files = arrow_files[ci * CHUNK : (ci + 1) * CHUNK]
            chunk_ds = pad.dataset([str(p) for p in chunk_files], format="ipc")
            try:
                con.unregister("raw_chunk")
            except Exception:
                pass
            con.register("raw_chunk", chunk_ds)
            chunk_out = files_full_dir / f"chunk_{ci:04d}.parquet"
            con.execute(f"""
                COPY (
                    SELECT
                        CAST({cum_offset} + row_number() OVER () - 1 AS UINTEGER) AS file_id,
                        sample_id, language,
                        domains, algorithms, paradigms, design_patterns
                    FROM raw_chunk
                ) TO '{chunk_out}' (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 256000);
            """)
            chunk_rows = con.execute(
                f"SELECT count(*) FROM '{chunk_out}'"
            ).fetchone()[0]
            cum_offset += chunk_rows
            if (ci + 1) % 5 == 0 or ci == n_chunks - 1:
                print(
                    f"    chunk {ci + 1}/{n_chunks}: +{chunk_rows:,} rows "
                    f"(total {cum_offset:,})",
                    flush=True,
                )
        for ji, json_path in enumerate(json_files):
            ci = n_arrow_chunks + ji
            try:
                with json_path.open() as f:
                    json_data = json.load(f)
            except Exception as e:
                print(f"  [warn] skipping {json_path.name}: {e}", file=sys.stderr)
                continue
            rows = json_data.get("results", [])
            if not rows:
                continue
            json_table = json_rows_to_arrow_table(rows)
            try:
                con.unregister("raw_json_chunk")
            except Exception:
                pass
            con.register("raw_json_chunk", json_table)
            chunk_out = files_full_dir / f"chunk_{ci:04d}.parquet"
            con.execute(f"""
                COPY (
                    SELECT
                        CAST({cum_offset} + row_number() OVER () - 1 AS UINTEGER) AS file_id,
                        sample_id, language,
                        domains, algorithms, paradigms, design_patterns
                    FROM raw_json_chunk
                ) TO '{chunk_out}' (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 256000);
            """)
            chunk_rows = con.execute(
                f"SELECT count(*) FROM '{chunk_out}'"
            ).fetchone()[0]
            cum_offset += chunk_rows
            print(
                f"    json {ji + 1}/{n_json_chunks}: +{chunk_rows:,} rows "
                f"(total {cum_offset:,})",
                flush=True,
            )

    n_total = cum_offset
    files_full = str(files_full_dir / "chunk_*.parquet")
    files_full_chunks = sorted(files_full_dir.glob("chunk_*.parquet"))
    print(f"  materialised {n_total:,} files across {n_chunks} chunks", flush=True)

    # 2a.bis  per-axis form maps: distinct arrow surface forms → concept_id,
    # with stem-based fallback for inflectional variants the cache missed.
    print("  building form maps (with stem fallback)…", flush=True)
    AXIS_FIELD = {
        "domains":         ("domains",         False),  # list[str]
        "algorithms":      ("algorithms",      True),   # list[str|object]
        "paradigms":       ("paradigms",       True),
        "design_patterns": ("design_patterns", True),
    }

    def _distinct_forms_for_axis(col: str, is_object: bool) -> List[str]:
        """Extract distinct normalized forms without a full-dataset DISTINCT.

        A single DISTINCT over the full staged parquet glob can keep a large
        hash table pinned in DuckDB. Each staged chunk is already bounded, so
        we dedupe per chunk and merge the small normalized form set in Python.
        """
        forms = set()
        for ci, chunk_path in enumerate(files_full_chunks):
            if is_object:
                extract_sql = f"""
                    SELECT DISTINCT norm_key(COALESCE(
                        json_extract_string(item, '$.name'),
                        json_extract_string(item, '$')
                    )) AS form
                    FROM '{chunk_path}' f,
                         UNNEST(from_json(f.{col}, '["JSON"]')) AS t(item)
                    WHERE f.{col} IS NOT NULL AND f.{col} NOT IN ('[]','null')
                """
            else:
                extract_sql = f"""
                    SELECT DISTINCT norm_key(t.raw) AS form
                    FROM '{chunk_path}' f,
                         UNNEST(from_json(f.{col}, '["VARCHAR"]')) AS t(raw)
                    WHERE f.{col} IS NOT NULL AND f.{col} NOT IN ('[]','null')
                """
            forms.update(r[0] for r in con.execute(extract_sql).fetchall() if r[0])
            if (ci + 1) % 5 == 0 or ci == len(files_full_chunks) - 1:
                print(
                    f"    {col}: scanned chunk {ci + 1}/{len(files_full_chunks)} "
                    f"({len(forms):,} forms)",
                    flush=True,
                )
        return sorted(forms)

    form_map_refs: List[pa.Table] = []
    for axis, (col, is_object) in AXIS_FIELD.items():
        distinct_forms = _distinct_forms_for_axis(col, is_object)
        vocab = vocabs[axis]
        forms, cids = [], []
        direct = stem_fallback = 0
        for form in distinct_forms:
            # form is already norm_key-normalized; lookup applies same.
            cid = vocab.lookup(form)
            if cid is None:
                continue
            forms.append(form)
            cids.append(cid)
            if form in vocab.name_to_id or vocab.variant_to_canonical.get(form, form) in vocab.name_to_id:
                direct += 1
            else:
                stem_fallback += 1
        fm = pa.table({
            "form": pa.array(forms, pa.string()),
            "concept_id": pa.array(cids, pa.uint32()),
        })
        form_map_refs.append(fm)
        con.register(f"form_map_{axis}", fm)
        print(
            f"    {axis}: {len(distinct_forms):,} distinct arrow forms → "
            f"{len(forms):,} mapped ({direct:,} direct + {stem_fallback:,} via stem)",
            flush=True,
        )

    def _clean_target(target: Path) -> None:
        if not target.exists():
            return
        for child in sorted(target.rglob("*"), reverse=True):
            if child.is_file():
                child.unlink()
            elif child.is_dir():
                child.rmdir()
        target.rmdir()

    def _copy_partitioned_chunked(target: Path, select_template: str) -> None:
        """Write hive-partitioned output one staged chunk at a time.

        DuckDB can exceed its memory limit when a JSON UNNEST + JOIN is copied
        from the full 142M-row glob into many partitions. Keeping each COPY
        bounded to one staged parquet chunk preserves the release layout while
        avoiding a full-dataset partitioned write.
        """
        _clean_target(target)
        print(f"  writing {target.name}/…", flush=True)
        for ci, chunk_path in enumerate(files_full_chunks):
            select_sql = select_template.format(source=str(chunk_path))
            con.execute(f"""
                COPY (
                    {select_sql}
                ) TO '{target}'
                (
                    FORMAT PARQUET,
                    COMPRESSION ZSTD,
                    PARTITION_BY (language),
                    FILENAME_PATTERN 'chunk_{ci:04d}_{{i}}',
                    OVERWRITE_OR_IGNORE
                );
            """)
            if (ci + 1) % 5 == 0 or ci == len(files_full_chunks) - 1:
                print(
                    f"    chunk {ci + 1}/{len(files_full_chunks)}",
                    flush=True,
                )

    # 2b. files/  (file_id, sample_id) partitioned by language
    target = OUT_DIR / "files"
    _clean_target(target)
    print(f"  writing {target.name}/…", flush=True)
    con.execute(f"""
        COPY (
            SELECT file_id, sample_id, language FROM '{files_full}'
        ) TO '{target}'
        (FORMAT PARQUET, COMPRESSION ZSTD, PARTITION_BY (language), OVERWRITE_OR_IGNORE);
    """)

    # 2c. edges_file_domain  — domains is JSON list[str]
    target = OUT_DIR / "edges_file_domain"
    _copy_partitioned_chunked(target, """
        SELECT f.file_id, v.concept_id, f.language
        FROM '{source}' f,
             UNNEST(from_json(f.domains, '["VARCHAR"]')) AS t(raw)
        JOIN form_map_domains v ON v.form = norm_key(t.raw)
        WHERE f.domains IS NOT NULL AND f.domains NOT IN ('[]','null')
    """)

    # 2d. edges_file_algorithm — items may be str OR {name, category, complexity}
    target = OUT_DIR / "edges_file_algorithm"
    _copy_partitioned_chunked(target, """
        WITH u AS (
            SELECT
                f.file_id, f.language,
                COALESCE(
                    json_extract_string(item, '$.name'),
                    json_extract_string(item, '$')
                ) AS raw_name
            FROM '{source}' f,
                 UNNEST(from_json(f.algorithms, '["JSON"]')) AS t(item)
            WHERE f.algorithms IS NOT NULL AND f.algorithms NOT IN ('[]','null')
        )
        SELECT u.file_id, v.concept_id, u.language
        FROM u
        JOIN form_map_algorithms v ON v.form = norm_key(u.raw_name)
    """)

    # 2e. edges_file_paradigm — items may be str OR {name, confidence, …}
    target = OUT_DIR / "edges_file_paradigm"
    _copy_partitioned_chunked(target, """
        WITH u AS (
            SELECT
                f.file_id, f.language,
                COALESCE(
                    json_extract_string(item, '$.name'),
                    json_extract_string(item, '$')
                ) AS raw_name,
                TRY_CAST(json_extract_string(item, '$.confidence') AS FLOAT) AS confidence
            FROM '{source}' f,
                 UNNEST(from_json(f.paradigms, '["JSON"]')) AS t(item)
            WHERE f.paradigms IS NOT NULL AND f.paradigms NOT IN ('[]','null')
        )
        SELECT u.file_id, v.concept_id, u.confidence, u.language
        FROM u
        JOIN form_map_paradigms v ON v.form = norm_key(u.raw_name)
    """)

    # 2f. edges_file_design_pattern — items may be str OR {name, category, …}
    target = OUT_DIR / "edges_file_design_pattern"
    _copy_partitioned_chunked(target, """
        WITH u AS (
            SELECT
                f.file_id, f.language,
                COALESCE(
                    json_extract_string(item, '$.name'),
                    json_extract_string(item, '$')
                ) AS raw_name
            FROM '{source}' f,
                 UNNEST(from_json(f.design_patterns, '["JSON"]')) AS t(item)
            WHERE f.design_patterns IS NOT NULL
              AND f.design_patterns NOT IN ('[]','null')
        )
        SELECT u.file_id, v.concept_id, u.language
        FROM u
        JOIN form_map_design_patterns v ON v.form = norm_key(u.raw_name)
    """)

    # 2g. dp_categories: one canonical category per design-pattern concept_id
    print("  computing design_pattern categories…", flush=True)
    dp_categories: Dict[int, str] = {}
    for ci, chunk_path in enumerate(files_full_chunks):
        dp_cats_rows = con.execute(f"""
            WITH u AS (
                SELECT
                    COALESCE(
                        json_extract_string(item, '$.name'),
                        json_extract_string(item, '$')
                    ) AS raw_name,
                    json_extract_string(item, '$.category') AS category
                FROM '{chunk_path}' f,
                     UNNEST(from_json(f.design_patterns, '["JSON"]')) AS t(item)
                WHERE f.design_patterns IS NOT NULL
                  AND f.design_patterns NOT IN ('[]','null')
                  AND json_extract_string(item, '$.category') IS NOT NULL
            )
            SELECT v.concept_id, ANY_VALUE(u.category)
            FROM u
            JOIN form_map_design_patterns v ON v.form = norm_key(u.raw_name)
            GROUP BY v.concept_id
        """).fetchall()
        for cid, cat in dp_cats_rows:
            dp_categories.setdefault(cid, cat)
        if (ci + 1) % 5 == 0 or ci == len(files_full_chunks) - 1:
            print(
                f"    chunk {ci + 1}/{len(files_full_chunks)}",
                flush=True,
            )
    print(f"    {len(dp_categories):,} design-pattern categories", flush=True)

    # 2h. AlgorithmCategory / AlgorithmComplexity nodes and Algorithm edges.
    # Categories use normalize_text(); complexity preserves symbols/case via
    # normalize_complexity().
    print("  computing algorithm categories and complexities…", flush=True)
    algorithm_category_names: set = set()
    algorithm_complexity_names: set = set()
    algorithm_category_edges: set = set()
    algorithm_complexity_edges: set = set()
    for ci, chunk_path in enumerate(files_full_chunks):
        algo_attr_rows = con.execute(f"""
            WITH u AS (
                SELECT
                    COALESCE(
                        json_extract_string(item, '$.name'),
                        json_extract_string(item, '$')
                    ) AS raw_name,
                    json_extract_string(item, '$.category') AS category,
                    json_extract_string(item, '$.complexity') AS complexity
                FROM '{chunk_path}' f,
                     UNNEST(from_json(f.algorithms, '["JSON"]')) AS t(item)
                WHERE f.algorithms IS NOT NULL
                  AND f.algorithms NOT IN ('[]','null')
                  AND (
                      json_extract_string(item, '$.category') IS NOT NULL
                      OR json_extract_string(item, '$.complexity') IS NOT NULL
                  )
            )
            SELECT DISTINCT v.concept_id, u.category, u.complexity
            FROM u
            JOIN form_map_algorithms v ON v.form = norm_key(u.raw_name)
        """).fetchall()
        for algo_cid, raw_cat, raw_comp in algo_attr_rows:
            if raw_cat:
                cat = normalize_text(raw_cat)
                if cat:
                    algorithm_category_names.add(cat)
                    algorithm_category_edges.add((algo_cid, cat))
            if raw_comp:
                comp = normalize_complexity(raw_comp)
                if comp:
                    algorithm_complexity_names.add(comp)
                    algorithm_complexity_edges.add((algo_cid, comp))
        if (ci + 1) % 5 == 0 or ci == len(files_full_chunks) - 1:
            print(
                f"    chunk {ci + 1}/{len(files_full_chunks)} "
                f"({len(algorithm_category_names):,} categories, "
                f"{len(algorithm_complexity_names):,} complexities)",
                flush=True,
            )
    print(
        f"    {len(algorithm_category_names):,} algorithm categories, "
        f"{len(algorithm_complexity_names):,} algorithm complexities",
        flush=True,
    )

    con.close()

    # Drop intermediate
    if files_full_dir.exists():
        for child in files_full_dir.iterdir():
            if child.is_file():
                child.unlink()
        files_full_dir.rmdir()

    # --- 3. Concept dimension tables + Wikidata layer -----------------------
    print("[3/3] Writing concept dimensions and Wikidata layer…", flush=True)

    grounded_qids: set = set()

    for axis, vocab in vocabs.items():
        for q in vocab.qids:
            if q:
                grounded_qids.add(q)
        cols = {
            "concept_id": pa.array(range(len(vocab.display)), pa.uint32()),
            "name": pa.array(vocab.display, pa.string()),
            "normalized_name": pa.array(vocab.normalized_names, pa.string()),
            "wikidata_qid": pa.array(vocab.qids, pa.string()),
            "label": pa.array(vocab.labels, pa.string()),
            "cache_spellings": pa.array(vocab.cache_spellings, pa.list_(pa.string())),
            "n_cache_variants": pa.array(vocab.n_cache_variants, pa.uint32()),
        }
        if axis == "design_patterns":
            cols["category"] = pa.array(
                [dp_categories.get(i) for i in range(len(vocab.display))],
                pa.string(),
            )
        pq.write_table(
            pa.table(cols),
            OUT_DIR / f"concepts_{axis}.parquet",
            compression=PARQUET_COMPRESSION,
        )

    algorithm_category_list = sorted(algorithm_category_names)
    algorithm_complexity_list = sorted(algorithm_complexity_names)
    algorithm_category_id = {
        name: i for i, name in enumerate(algorithm_category_list)
    }
    algorithm_complexity_id = {
        name: i for i, name in enumerate(algorithm_complexity_list)
    }

    pq.write_table(
        pa.table({
            "category_id": pa.array(
                range(len(algorithm_category_list)),
                pa.uint32(),
            ),
            "name": pa.array(algorithm_category_list, pa.string()),
        }),
        OUT_DIR / "concepts_algorithm_categories.parquet",
        compression=PARQUET_COMPRESSION,
    )
    pq.write_table(
        pa.table({
            "complexity_id": pa.array(
                range(len(algorithm_complexity_list)),
                pa.uint32(),
            ),
            "name": pa.array(algorithm_complexity_list, pa.string()),
        }),
        OUT_DIR / "concepts_algorithm_complexities.parquet",
        compression=PARQUET_COMPRESSION,
    )

    algorithm_category_edge_rows = sorted(
        (algo_cid, algorithm_category_id[cat])
        for algo_cid, cat in algorithm_category_edges
    )
    algorithm_complexity_edge_rows = sorted(
        (algo_cid, algorithm_complexity_id[comp])
        for algo_cid, comp in algorithm_complexity_edges
    )
    pq.write_table(
        pa.table({
            "algorithm_concept_id": pa.array(
                [r[0] for r in algorithm_category_edge_rows],
                pa.uint32(),
            ),
            "category_id": pa.array(
                [r[1] for r in algorithm_category_edge_rows],
                pa.uint32(),
            ),
        }),
        OUT_DIR / "edges_algorithm_category.parquet",
        compression=PARQUET_COMPRESSION,
    )
    pq.write_table(
        pa.table({
            "algorithm_concept_id": pa.array(
                [r[0] for r in algorithm_complexity_edge_rows],
                pa.uint32(),
            ),
            "complexity_id": pa.array(
                [r[1] for r in algorithm_complexity_edge_rows],
                pa.uint32(),
            ),
        }),
        OUT_DIR / "edges_algorithm_complexity.parquet",
        compression=PARQUET_COMPRESSION,
    )

    # Wikidata entities: union of grounded QIDs + all QIDs appearing in hierarchy
    hier_path = HIER_DIR / "wikidata_hierarchy.json"
    parents_path = HIER_DIR / "wikidata_parents.json"

    wd_labels: Dict[str, str] = {}
    wd_descr: Dict[str, str] = {}
    # seed from vocab grounding
    for vocab in vocabs.values():
        for q, lab, desc in zip(vocab.qids, vocab.labels, vocab.descriptions):
            if q and q not in wd_labels:
                wd_labels[q] = lab or ""
                wd_descr[q] = desc or ""

    # parent_of edges
    p_child: List[str] = []
    p_parent: List[str] = []
    p_via: List[str] = []
    p_depth: List[int] = []

    if hier_path.exists():
        with hier_path.open() as f:
            hier = json.load(f)
        for child_qid, info in hier.items():
            child_label = info.get("label") or ""
            if child_qid not in wd_labels:
                wd_labels[child_qid] = child_label
                wd_descr[child_qid] = ""
            for step in info.get("chain", []):
                pid = step.get("id")
                if not pid:
                    continue
                p_child.append(child_qid)
                p_parent.append(pid)
                p_via.append(step.get("via") or "P279")
                p_depth.append(int(step.get("depth") or 0))
                if pid not in wd_labels:
                    wd_labels[pid] = step.get("label") or ""
                    wd_descr[pid] = ""
    elif parents_path.exists():
        with parents_path.open() as f:
            parents = json.load(f)
        for child_qid, info in parents.items():
            if child_qid not in wd_labels:
                wd_labels[child_qid] = info.get("label") or ""
                wd_descr[child_qid] = ""
            for j, step in enumerate(info.get("parents", []), start=1):
                pid = step.get("id")
                if not pid:
                    continue
                p_child.append(child_qid)
                p_parent.append(pid)
                p_via.append(step.get("via") or "P279")
                p_depth.append(j)
                if pid not in wd_labels:
                    wd_labels[pid] = step.get("label") or ""
                    wd_descr[pid] = ""

    qids_all = sorted(wd_labels)
    pq.write_table(
        pa.table({
            "qid": pa.array(qids_all, pa.string()),
            "label": pa.array([wd_labels[q] for q in qids_all], pa.string()),
            "description": pa.array([wd_descr.get(q, "") for q in qids_all], pa.string()),
        }),
        OUT_DIR / "wikidata_entities.parquet",
        compression=PARQUET_COMPRESSION,
    )
    pq.write_table(
        pa.table({
            "child_qid": pa.array(p_child, pa.string()),
            "parent_qid": pa.array(p_parent, pa.string()),
            "via": pa.array(p_via, pa.string()),
            "depth": pa.array(p_depth, pa.uint16()),
        }),
        OUT_DIR / "wikidata_parent_of.parquet",
        compression=PARQUET_COMPRESSION,
    )

    # --- 4. Cleanup staging --------------------------------------------------
    if STAGING_DIR.exists():
        print("  cleaning up staging…", flush=True)
        for child in sorted(STAGING_DIR.rglob("*"), reverse=True):
            if child.is_file():
                child.unlink()
            elif child.is_dir():
                child.rmdir()
        STAGING_DIR.rmdir()

    # Manifest
    manifest = {
        "version": "1.0.0",
        "files": int(n_total),
        "vocab_sizes": {
            **{axis: len(v.display) for axis, v in vocabs.items()},
            "algorithm_categories": len(algorithm_category_list),
            "algorithm_complexities": len(algorithm_complexity_list),
        },
        "edge_counts": {
            "algorithm_category": len(algorithm_category_edge_rows),
            "algorithm_complexity": len(algorithm_complexity_edge_rows),
        },
        "wikidata_entities": len(qids_all),
        "wikidata_parent_of_edges": len(p_child),
        "compression": PARQUET_COMPRESSION,
        "canonicalisation": {
            "all_axes": "normalize_text (NFD strip-accents, lowercase, "
                        "c++/c#/f# expansion, separator collapse, punctuation strip)",
            "algorithm_category": "normalize_text",
            "algorithm_complexity": "normalize_complexity "
                                    "(preserve symbols/case, collapse whitespace, "
                                    "standardize time/space phrasing)",
            "domains_extra": "output/domains/stem_results.json (variant→canonical "
                             "clusters from analyzer/stem_domains.py)",
            "wikidata_dedup_policy": "per normalised key, pick cache entry with "
                                     "status (found>not_found>discarded), then "
                                     "confidence (high>medium>low), then filename ASC",
            "domain_qid_fallback": "if canonical key not grounded, look up any "
                                   "cluster variant",
        },
        "layout": "hive-partitioned by language (Java/Python/C++/… subdirs); "
                  "concepts_*.parquet and wikidata_*.parquet are single global files",
        "partition_file_size_target": PART_FILE_SIZE,
        "schema": {
            "files/language=<L>/part-*.parquet": [
                "file_id (uint32)", "sample_id (str)",
            ],
            "concepts_<axis>.parquet": [
                "concept_id (uint32)",
                "name (str)               — primary display (= label if grounded)",
                "normalized_name (str)    — canonical key, joinable across axes",
                "wikidata_qid (str?)",
                "label (str?)             — Wikidata label",
                "cache_spellings (list<str>) — raw spellings from the cache",
                "n_cache_variants (uint32)   — # cache files collapsed here",
            ],
            "concepts_design_patterns.parquet (extra)": ["category (str?)"],
            "concepts_algorithm_categories.parquet": [
                "category_id (uint32)", "name (str, normalize_text)",
            ],
            "concepts_algorithm_complexities.parquet": [
                "complexity_id (uint32)", "name (str, normalize_complexity)",
            ],
            "edges_file_algorithm/language=<L>/part-*.parquet": [
                "file_id", "concept_id",
            ],
            "edges_algorithm_category.parquet": [
                "algorithm_concept_id", "category_id",
            ],
            "edges_algorithm_complexity.parquet": [
                "algorithm_concept_id", "complexity_id",
            ],
            "edges_file_domain/language=<L>/part-*.parquet": [
                "file_id", "concept_id",
            ],
            "edges_file_paradigm/language=<L>/part-*.parquet": [
                "file_id", "concept_id", "confidence (f32)",
            ],
            "edges_file_design_pattern/language=<L>/part-*.parquet": [
                "file_id", "concept_id", "description (str)",
            ],
            "wikidata_entities.parquet": ["qid", "label", "description"],
            "wikidata_parent_of.parquet": ["child_qid", "parent_qid", "via", "depth"],
        },
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print("Done.", flush=True)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
