#!/usr/bin/env python3
"""
Algorithm Stemmer — deduplication and canonicalization of algorithm names.

Reads algorithm names from TXT, JSON, Arrow, or directory inputs, then groups
variants with the same stem root and produces a canonical form for each cluster.

Output format is identical to stem_domains.py so downstream graph builders can
consume domain and algorithm canonicalization maps consistently.

Usage:
  # From a TXT list
  python analyzer/stem_algorithms.py \\
    --input-file output/algorithms/raw.txt \\
    --output output/algorithms/stem_results.json

  # From Arrow pipeline files (reads the 'algorithms' column)
  python analyzer/stem_algorithms.py \\
    --input-file output/full_pipeline/ \\
    --output output/algorithms/stem_results.json
"""

import argparse
import json
import logging
import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional NLTK
# ---------------------------------------------------------------------------
try:
    from nltk.stem import SnowballStemmer
    from nltk.corpus import stopwords
    import nltk
    nltk.download("stopwords", quiet=True)
    _STEMMER = SnowballStemmer("english")
    _STOP_WORDS: Set[str] = set(stopwords.words("english"))
    NLTK_AVAILABLE = True
    logger.info("NLTK available — using SnowballStemmer")
except ImportError:
    NLTK_AVAILABLE = False
    _STEMMER = None
    _STOP_WORDS = {
        "the", "a", "an", "and", "or", "in", "on", "at", "to",
        "for", "of", "with", "by", "from", "as", "is", "are",
    }
    logger.warning("NLTK not available — using simplified stemmer. `pip install nltk`")

# ---------------------------------------------------------------------------
# Algorithm-specific irregular forms
# (prevent over-stemming of well-known algorithm terms)
# ---------------------------------------------------------------------------
_IRREGULAR: Dict[str, str] = {
    "trees":      "tree",
    "graphs":     "graph",
    "sorts":      "sort",
    "searches":   "search",
    "traversals": "traversal",
    "algorithms": "algorithm",
    "techniques": "technique",
    "methods":    "method",
    "approaches": "approach",
    "heuristics": "heuristic",
    "matrices":   "matrix",
    "indices":    "index",
    "heaps":      "heap",
    "queues":     "queue",
    "stacks":     "stack",
    "lists":      "list",
    "arrays":     "array",
    "strings":    "string",
    "paths":      "path",
    "nodes":      "node",
    "edges":      "edge",
    "cycles":     "cycle",
    "tables":     "table",
    "maps":       "map",
    "sets":       "set",
}

# Words to strip from algorithm names before clustering
# (they add noise without changing identity)
_NOISE_WORDS = {
    "algorithm", "technique", "method", "approach",
    "based", "using", "via", "with",
}


def _simple_stem(word: str) -> str:
    w = word.lower().strip()
    if w in _IRREGULAR:
        return _IRREGULAR[w]
    if w.endswith("ies") and len(w) > 3:
        return w[:-3] + "y"
    if w.endswith("ings") and len(w) > 4:
        return w[:-4]
    if w.endswith("ing") and len(w) > 4:
        return w[:-3]
    if w.endswith("tions") and len(w) > 5:
        return w[:-4]
    if w.endswith("tion") and len(w) > 4:
        return w[:-3]
    if w.endswith("ed") and len(w) > 3:
        return w[:-2]
    if w.endswith("es") and len(w) > 3:
        return w[:-2]
    if w.endswith("s") and len(w) > 2 and not w.endswith("ss"):
        return w[:-1]
    return w


def stem_word(word: str) -> str:
    if NLTK_AVAILABLE and _STEMMER is not None:
        return _STEMMER.stem(word.lower())
    return _simple_stem(word)


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------
# Strip leading numeric tokens like "0 1 bfs" → "bfs"
_LEADING_NUMS = re.compile(r'^(\d+\s+)+')


def normalize(name: str) -> str:
    """Lightweight normalisation: remove accents, separators → space, lowercase."""
    s = name.strip()
    # Strip leading numeric tokens (artifact of some pipeline outputs)
    s = _LEADING_NUMS.sub("", s).strip()
    s = s.lower()
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    # Expand common abbreviation symbols
    s = s.replace("c++", "cpp").replace("c#", "csharp")
    # Separators → space
    s = re.sub(r"[-_/\\|]+", " ", s)
    # Remove remaining non-alphanumeric chars
    s = re.sub(r"[^a-z0-9\s]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    # Remove noise words
    words = [w for w in s.split() if w not in _NOISE_WORDS]
    return " ".join(words)


def stem_key(name: str) -> str:
    """Grouping key: stem each content word, skip stop-words and very short tokens."""
    norm = normalize(name)
    words = norm.split()
    stemmed = [
        stem_word(w) for w in words
        if w not in _STOP_WORDS and len(w) > 1
    ]
    return " ".join(stemmed) if stemmed else norm


# ---------------------------------------------------------------------------
# Canonical form selector
# ---------------------------------------------------------------------------
def pick_canonical(variants: List[str]) -> str:
    """
    Choose the canonical form from a cluster of variants.

    Priority (in order):
    1. Fewest words after normalisation (simpler = more canonical)
    2. Does not end in 's' on the last word (prefer singular)
    3. Does not contain noise words ('algorithm', 'technique', …)
    4. Alphabetical order as tiebreaker
    """
    def sort_key(v: str) -> Tuple:
        norm = normalize(v)
        words = norm.split()
        n_words = len(words)
        last = words[-1] if words else ""
        ends_s = 1 if last.endswith("s") and not last.endswith("ss") else 0
        has_noise = 1 if any(w in _NOISE_WORDS for w in v.lower().split()) else 0
        return (n_words, ends_s, has_noise, norm)

    return sorted(variants, key=sort_key)[0]


def _load_arrow_algorithms(path: Path) -> List[str]:
    """Extract algorithm names from the 'algorithms' column of an Arrow file."""
    try:
        import pyarrow.feather as feather
    except ImportError:
        raise ImportError("pip install pyarrow")
    table = feather.read_table(path, columns=["algorithms"])
    names: List[str] = []
    for value in table.column("algorithms").to_pylist():
        if not value:
            continue
        try:
            parsed = json.loads(value) if isinstance(value, str) else value
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(parsed, list):
            for item in parsed:
                if isinstance(item, str) and item:
                    names.append(item)
                elif isinstance(item, dict) and item.get("name"):
                    names.append(item["name"])
    return names


def _load_single_file(path: Path) -> List[str]:
    if path.suffix in (".arrow", ".feather"):
        return _load_arrow_algorithms(path)

    if path.suffix == ".json":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        results = data.get("results", data) if isinstance(data, dict) else data
        names: List[str] = []
        for row in results:
            if not isinstance(row, dict):
                continue
            for item in row.get("algorithms", []):
                if isinstance(item, str) and item:
                    names.append(item)
                elif isinstance(item, dict) and item.get("name"):
                    names.append(item["name"])
        return names

    # TXT: one name per line, skip comments and blank lines
    names = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                names.append(line)
    return names


def load_from_inputs(paths: List[str]) -> List[str]:
    all_names: List[str] = []
    seen: Set[str] = set()

    files: List[Path] = []
    for raw in paths:
        p = Path(raw)
        if not p.exists():
            raise FileNotFoundError(f"Path not found: {raw}")
        if p.is_dir():
            found = sorted(
                list(p.glob("*.txt")) + list(p.glob("*.json"))
                + list(p.glob("*.arrow")) + list(p.glob("*.feather"))
            )
            files.extend(found)
        else:
            files.append(p)

    for fp in files:
        try:
            names = _load_single_file(fp)
            new = [n for n in names if n not in seen]
            seen.update(new)
            all_names.extend(new)
            logger.info(f"  {fp.name}: {len(names)} names ({len(new)} new)")
        except Exception as exc:
            logger.warning(f"  {fp.name}: skipped — {exc}")

    logger.info(f"Total algorithm names loaded: {len(all_names)}")
    return all_names


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------
def build_clusters(names: List[str]) -> Dict[str, List[str]]:
    clusters: Dict[str, List[str]] = defaultdict(list)
    seen_norm: Set[str] = set()

    for name in names:
        name = name.strip()
        norm = normalize(name)
        if not norm or norm in seen_norm:
            continue
        seen_norm.add(norm)
        key = stem_key(name) or norm
        clusters[key].append(name)

    return dict(clusters)


def run_pipeline(names: List[str]) -> Dict:
    logger.info(f"Input algorithm names: {len(names)}")

    clusters = build_clusters(names)
    logger.info(f"Clusters found: {len(clusters)}")

    canonical_list: List[str] = []
    cluster_details = []
    n_merged = 0

    for key, variants in sorted(clusters.items()):
        canonical = pick_canonical(variants)
        canonical_list.append(canonical)
        merged = len(variants) > 1
        if merged:
            n_merged += 1
        cluster_details.append({
            "stem_key":  key,
            "canonical": canonical,
            "variants":  sorted(variants),
            "merged":    merged,
        })

    canonical_list.sort()
    logger.info(
        f"Canonical forms: {len(canonical_list)} "
        f"({n_merged} clusters with merged variants)"
    )

    return {
        "metadata": {
            "input_total":            len(names),
            "unique_normalized":      len(cluster_details),
            "canonical_algorithms":   len(canonical_list),
            "clusters_with_variants": n_merged,
            "reduction_pct": round(
                (1 - len(canonical_list) / max(len(names), 1)) * 100, 1
            ),
        },
        "canonical_algorithms": canonical_list,
        "clusters": cluster_details,
    }


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def save_results(results: Dict, output_path: str) -> None:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"JSON results saved to: {out}")

    txt_path = out.with_name(out.stem + "_canonical.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("# Canonical algorithm forms (one per line)\n\n")
        for name in results["canonical_algorithms"]:
            f.write(name + "\n")
    logger.info(f"Canonical TXT saved to: {txt_path}")

    merged_path = out.with_name(out.stem + "_merged_clusters.txt")
    with open(merged_path, "w", encoding="utf-8") as f:
        f.write("# Clusters with merged variants\n")
        f.write("# Format: CANONICAL  ← variant1 | variant2 | …\n\n")
        for c in results["clusters"]:
            if c["merged"]:
                others = [v for v in c["variants"] if v != c["canonical"]]
                f.write(f"{c['canonical']}\n")
                for v in others:
                    f.write(f"    ← {v}\n")
                f.write("\n")
    logger.info(f"Merged clusters saved to: {merged_path}")


def print_summary(results: Dict) -> None:
    m = results["metadata"]
    print()
    print("=" * 55)
    print("  ALGORITHM STEMMER — SUMMARY")
    print("=" * 55)
    print(f"  Input names              : {m['input_total']}")
    print(f"  Canonical forms          : {m['canonical_algorithms']}")
    print(f"  Clusters with variants   : {m['clusters_with_variants']}")
    print(f"  Reduction                : {m['reduction_pct']}%")
    print("=" * 55)

    examples = [c for c in results["clusters"] if c["merged"]][:15]
    if examples:
        print("\n  EXAMPLES OF MERGED VARIANTS:")
        for c in examples:
            others = [v for v in c["variants"] if v != c["canonical"]]
            print(f"    {c['canonical']}")
            for v in others:
                print(f"        ← {v}")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cluster algorithm name variants and extract canonical forms",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # From a filtered TXT list (recommended)
  python analyzer/stem_algorithms.py \\
    --input-file output/algorithms/stem_results_canonical_filtered.txt \\
    --output output/algorithms/stem_results.json

  # From Arrow pipeline files
  python analyzer/stem_algorithms.py \\
    --input-file output/full_pipeline/ \\
    --output output/algorithms/stem_results.json
""",
    )

    parser.add_argument("--input-file", metavar="PATH", nargs="+", required=True,
                        help="TXT / JSON / Arrow files or directories")
    parser.add_argument("--output", default="output/algorithms/stem_results.json")
    parser.add_argument("--limit", type=int, default=0,
                        help="Limit number of names processed (0 = all)")
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    names = load_from_inputs(args.input_file)

    if args.limit and args.limit > 0:
        names = names[:args.limit]

    results = run_pipeline(names)
    save_results(results, args.output)
    print_summary(results)


if __name__ == "__main__":
    main()
