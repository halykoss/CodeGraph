#!/usr/bin/env python3
"""
Domain Stemmer — clean and deduplicate domains with stemming.

Reads domains from JSON, TXT, Arrow, or directory inputs; groups variants with
the same root (plurals, truncations, -ing, -ed, ...) and produces a list of
unique canonical forms.

Usage:
  python analyzer/stem_domains.py --input-file output/wikidata/grounded_domains.json
  python analyzer/stem_domains.py --input-file output/domains/txt/filtered_domains.txt
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
    from nltk.stem import PorterStemmer, SnowballStemmer
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
        "was", "were", "be", "been", "being",
    }
    logger.warning("NLTK unavailable — using simplified stemmer. `pip install nltk`")


# ---------------------------------------------------------------------------
# Stemming helpers
# ---------------------------------------------------------------------------
_IRREGULAR: Dict[str, str] = {
    "data": "data",
    "criteria": "criterion",
    "analysis": "analysis",
    "analyses": "analysis",
    "matrices": "matrix",
    "indices": "index",
    "algorithms": "algorithm",
    "heuristics": "heuristic",
    "statistics": "statistic",
    "semantics": "semantic",
    "dynamics": "dynamic",
    "graphics": "graphic",
    "metrics": "metric",
    "robotics": "robotic",
    "economics": "economic",
    "mathematics": "mathematic",
    "physics": "physic",
}


def _simple_stem(word: str) -> str:
    """Fallback stemmer without NLTK."""
    w = word.lower().strip()
    if w in _IRREGULAR:
        return _IRREGULAR[w]
    if w.endswith("ies") and len(w) > 3:
        return w[:-3] + "y"
    if w.endswith("ves") and len(w) > 3:
        return w[:-3] + "f"
    if w.endswith("ings") and len(w) > 4:
        return w[:-4]
    if w.endswith("ing") and len(w) > 4:
        return w[:-3]
    if w.endswith("tion") and len(w) > 4:
        return w[:-3]          # "solution" → "solut" — close enough for grouping
    if w.endswith("tions") and len(w) > 5:
        return w[:-4]
    if w.endswith("ness") and len(w) > 4:
        return w[:-4]
    if w.endswith("ment") and len(w) > 4:
        return w[:-4]
    if w.endswith("ments") and len(w) > 5:
        return w[:-5]
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
_NOISE_WORDS = {"python", "using", "based", "via", "related", "specific"}


def normalize(domain: str) -> str:
    """Lightweight normalization: lowercase, remove accents, separators → spaces."""
    s = domain.lower().strip()
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    s = re.sub(r"[-_/\\|]+", " ", s)
    s = re.sub(r"[^a-z0-9\s]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    # Remove noisy words
    words = [w for w in s.split() if w not in _NOISE_WORDS]
    return " ".join(words)


def stem_key(domain: str) -> str:
    """Grouping key: every word stemmed, excluding stop words."""
    norm = normalize(domain)
    words = norm.split()
    stemmed = [stem_word(w) for w in words if w not in _STOP_WORDS and len(w) > 1]
    return " ".join(stemmed) if stemmed else norm


# ---------------------------------------------------------------------------
# Canonical form selector
# ---------------------------------------------------------------------------
def _word_count(d: str) -> int:
    return len(normalize(d).split())


def pick_canonical(variants: List[str]) -> str:
    """
    Choose the canonical form among cluster variants.

    Criteria (in order):
    1. Prefer the shortest normalized domain by word count
    2. On ties, prefer singular forms (not ending in 's')
    3. On ties, alphabetical order
    """
    def sort_key(d: str) -> Tuple:
        norm = normalize(d)
        words = norm.split()
        n_words = len(words)
        last_word = words[-1] if words else ""
        ends_s = 1 if last_word.endswith("s") and not last_word.endswith("ss") else 0
        return (n_words, ends_s, norm)

    return sorted(variants, key=sort_key)[0]


def _load_arrow_file(path: Path) -> List[str]:
    """Load domains from the 'domains' column of an Arrow/Feather file."""
    try:
        import pyarrow.feather as feather
    except ImportError:
        raise ImportError("Install pyarrow: pip install pyarrow")

    table = feather.read_table(path, columns=["domains"])
    domains: List[str] = []
    for value in table.column("domains").to_pylist():
        if not value:
            continue
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                domains.extend(str(d) for d in parsed if d)
            elif isinstance(parsed, str):
                domains.append(parsed)
        except (json.JSONDecodeError, TypeError):
            domains.append(str(value))
    return domains


def _load_single_file(path: Path) -> List[str]:
    """Load domains from a single JSON, TXT, or Arrow file."""
    if path.suffix in (".arrow", ".feather"):
        return _load_arrow_file(path)

    if path.suffix == ".json":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            domains: List[str] = []
            for item in data:
                if isinstance(item, str):
                    domains.append(item)
                elif isinstance(item, dict):
                    name = item.get("domain") or item.get("name") or item.get("label")
                    if name:
                        domains.append(name)
            return domains

        if isinstance(data, dict):
            # processing_results_part_*.json: {"statistics": {"domains_found": [...]}, "results": [...]}
            # Prefer the aggregate already computed in "statistics"
            if "statistics" in data and "domains_found" in data["statistics"]:
                return data["statistics"].get("domains_found", [])
            # merged_domains.json: {"results": [{domain: ..., status: ...}, ...]}
            if "results" in data:
                return [
                    r["domain"]
                    for r in data["results"]
                    if isinstance(r, dict) and r.get("domain")
                ]
            # {"domains": [...]}
            if "domains" in data:
                return data["domains"]

        raise ValueError(f"Unrecognized JSON format in {path}")

    # TXT: one domain per line, skip comments (#) and blank lines
    domains = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                domains.append(line)
    return domains


def load_from_inputs(paths: List[str]) -> "Counter[str]":
    """
    Load domains from one or more paths and return a Counter {domain: occurrences}.
    Each path can be:
      - a JSON or TXT file
      - a directory (loads every .json, .txt, .arrow, .feather inside it)
    """
    from collections import Counter

    counter: Counter = Counter()
    files_loaded: List[Path] = []

    for raw in paths:
        p = Path(raw)
        if not p.exists():
            raise FileNotFoundError(f"Path not found: {raw}")

        if p.is_dir():
            found = sorted(
                list(p.glob("*.json")) + list(p.glob("*.txt"))
                + list(p.glob("*.arrow")) + list(p.glob("*.feather"))
            )
            if not found:
                logger.warning(f"No JSON/TXT file found in {raw}")
            files_loaded.extend(found)
        else:
            files_loaded.append(p)

    for fp in files_loaded:
        try:
            domains = _load_single_file(fp)
            counter.update(domains)
            logger.info(f"  {fp.name}: {len(domains)} occurrences")
        except Exception as e:
            logger.warning(f"  {fp.name}: skipped — {e}")

    logger.info(f"Unique domains from {len(files_loaded)} files: {len(counter)}")
    return counter


# ---------------------------------------------------------------------------
# Core stemming pipeline
# ---------------------------------------------------------------------------
def build_clusters(domains: List[str]) -> Dict[str, List[str]]:
    """
    Group domains by stemmed key.
    Returns {stem_key: [domain1, domain2, ...]}.
    """
    clusters: Dict[str, List[str]] = defaultdict(list)
    seen_norm: Set[str] = set()

    for domain in domains:
        domain = domain.strip()
        norm = normalize(domain)
        if not norm or norm in seen_norm:
            continue
        seen_norm.add(norm)

        key = stem_key(domain)
        if not key:
            key = norm
        clusters[key].append(domain)

    return dict(clusters)


def run_pipeline(domains: List[str]) -> Dict:
    """
    Full pipeline: normalization → stemming → clustering → canonical selection.
    Returns a dictionary with structured results.
    """
    logger.info(f"Input domains: {len(domains)}")

    # Remove domains that look like URLs
    url_re = re.compile(r"https?://|www\.|\.com\b|\.org\b|\.net\b|\.io\b")
    clean_domains = [d for d in domains if not url_re.search(d.lower())]
    removed_urls = len(domains) - len(clean_domains)
    if removed_urls:
        logger.info(f"Removed {removed_urls} URL-like domains")

    clusters = build_clusters(clean_domains)
    logger.info(f"Clusters found: {len(clusters)}")

    # Compute canonical forms
    canonical_list: List[str] = []
    cluster_details = []
    n_merged = 0

    for key, variants in sorted(clusters.items()):
        canonical = pick_canonical(variants)
        canonical_list.append(canonical)
        merged = len(variants) > 1
        if merged:
            n_merged += 1
        cluster_details.append(
            {
                "stem_key": key,
                "canonical": canonical,
                "variants": sorted(variants),
                "merged": merged,
            }
        )

    canonical_list.sort()
    logger.info(
        f"Canonical forms: {len(canonical_list)} "
        f"(of which {n_merged} clusters with merged variants)"
    )

    return {
        "metadata": {
            "input_total": len(domains),
            "after_url_filter": len(clean_domains),
            "unique_normalized": sum(1 for c in cluster_details),
            "canonical_domains": len(canonical_list),
            "clusters_with_variants": n_merged,
            "reduction_pct": round(
                (1 - len(canonical_list) / max(len(clean_domains), 1)) * 100, 1
            ),
        },
        "canonical_domains": canonical_list,
        "clusters": cluster_details,
    }


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def save_results(results: Dict, output_path: str) -> None:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Complete JSON
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"JSON results saved to: {out}")

    # TXT — canonical forms only
    txt_path = out.with_name(out.stem + "_canonical.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("# Canonical domain forms (one per line)\n\n")
        for d in results["canonical_domains"]:
            f.write(d + "\n")
    logger.info(f"Canonical forms TXT saved to: {txt_path}")

    # TXT — merged cluster report
    merged_path = out.with_name(out.stem + "_merged_clusters.txt")
    with open(merged_path, "w", encoding="utf-8") as f:
        f.write("# Clusters with merged variants\n")
        f.write("# Format: CANONICAL  ← variant1 | variant2 | ...\n\n")
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
    print("  DOMAIN STEMMER — SUMMARY")
    print("=" * 55)
    print(f"  Input domains          : {m['input_total']}")
    print(f"  After URL filter          : {m['after_url_filter']}")
    print(f"  Unique canonical forms   : {m['canonical_domains']}")
    print(f"  Clusters with variants     : {m['clusters_with_variants']}")
    print(f"  Reduction                : {m['reduction_pct']}%")
    print("=" * 55)

    # Examples of clusters with variants
    examples = [c for c in results["clusters"] if c["merged"]][:10]
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
        description="Group domains with stemming and extract unique canonical forms",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # From a single JSON file
  python analyzer/stem_domains.py \\
    --input-file output/wikidata/grounded_domains.json

  # From multiple files (all parts)
  python analyzer/stem_domains.py \\
    --input-file output/processing_results_part_*.json

  # From an entire directory (loads all JSON/TXT files)
  python analyzer/stem_domains.py \\
    --input-file output/

  # From multiple sources
  python analyzer/stem_domains.py \\
    --input-file output/full_pipeline/ output/wikidata/grounded_domains.json
""",
    )

    parser.add_argument("--input-file", metavar="PATH", nargs="+", required=True,
                        help="One or more JSON/TXT/Arrow files, or a directory")
    parser.add_argument(
        "--output",
        default="output/domains/stem_results.json",
        help="Output JSON file path (default: output/domains/stem_results.json)",
    )
    parser.add_argument("--min-count", type=int, default=1,
                        help="Include only domains with at least N occurrences (default: 1)")
    parser.add_argument("--limit", type=int, default=0,
                        help="Limit the number of processed domains (0 = all)")
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    counter = load_from_inputs(args.input_file)
    domains = [d for d, cnt in counter.items() if cnt >= args.min_count]
    logger.info(f"Domains after filter min-count={args.min_count}: {len(domains)}")

    if args.limit and args.limit > 0:
        domains = domains[: args.limit]
        logger.info(f"Limited to {len(domains)} domains")

    # Pipeline
    results = run_pipeline(domains)

    # Output
    save_results(results, args.output)
    print_summary(results)


if __name__ == "__main__":
    main()
