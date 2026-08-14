"""
Fetch up to 3 Wikidata parents (P279 subclass-of, fallback P31 instance-of)
for every grounded Wikidata entity in the grounding caches.

Uses batched SPARQL (VALUES clause, 50 QIDs/request) to stay within Wikidata
rate limits.

Output: wikidata_parents.json, a dictionary keyed by QID
  {
    "Q11660": {
      "label": "artificial intelligence",
      "parents": [
        {"id": "Q2539", "label": "science", "via": "P279"},
        ...
      ]
    },
    ...
  }
"""

import json
import os
import time
import logging
from collections import defaultdict
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
REPO = Path(__file__).resolve().parents[1]
CACHE_ROOT = Path(os.getenv("CACHE_ROOT", REPO / "cache"))
CACHE_DIRS = (
    "wikidata_domains",
    "wikidata_algorithms",
    "wikidata_paradigms",
    "wikidata_design_patterns",
)

WIKIDATA_SPARQL = "https://query.wikidata.org/sparql"
HEADERS = {
    "Accept": "application/sparql-results+json",
    "User-Agent": "CodeGraphEmbeddingDataset/1.0 (https://github.com/halykoss/CodeGraph)",
}

MAX_PARENTS  = 3
BATCH_SIZE   = 50    # QIDs per SPARQL request
RETRY_LIMIT  = 3
RETRY_DELAY  = 10    # seconds between retries
REQUEST_GAP  = 1.5   # polite gap between batches
OUTPUT_FILE  = "wikidata_parents.json"

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Grounding cache input
# ---------------------------------------------------------------------------

def get_all_wikidata_entities() -> list[dict]:
    entities: dict[str, str] = {}
    for cache_dir in CACHE_DIRS:
        directory = CACHE_ROOT / cache_dir
        if not directory.is_dir():
            log.warning(f"Cache directory not found, skipping: {directory}")
            continue
        for path in sorted(directory.glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except Exception as exc:
                log.warning(f"Skipping unreadable cache file {path}: {exc}")
                continue

            qid = data.get("wikidata_id")
            if data.get("status") != "found" or not qid:
                continue

            label = data.get("label") or data.get("concept") or path.stem
            entities.setdefault(qid, label)

    return [{"id": qid, "label": label} for qid, label in sorted(entities.items())]

# ---------------------------------------------------------------------------
# Wikidata batched SPARQL
# ---------------------------------------------------------------------------

BATCH_SPARQL = """
SELECT DISTINCT ?entity ?parent ?parentLabel ?via WHERE {{
  VALUES ?entity {{ {values} }}
  {{
    ?entity wdt:P279 ?parent .
    BIND("P279" AS ?via)
  }} UNION {{
    ?entity wdt:P31 ?parent .
    BIND("P31" AS ?via)
  }}
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
}}
"""


def fetch_batch(qids: list[str]) -> dict[str, list[dict]]:
    """
    Query Wikidata for a batch of QIDs.
    Returns {qid: [{id, label, via}, ...], ...}.
    Missing QIDs map to empty lists.
    """
    values = " ".join(f"wd:{q}" for q in qids)
    sparql = BATCH_SPARQL.format(values=values)

    for attempt in range(1, RETRY_LIMIT + 1):
        try:
            resp = requests.get(
                WIKIDATA_SPARQL,
                params={"query": sparql, "format": "json"},
                headers=HEADERS,
                timeout=60,
            )
            if resp.status_code == 429:
                wait = int(resp.headers.get("Retry-After", RETRY_DELAY * attempt))
                log.warning(f"  Rate-limited, waiting {wait}s …")
                time.sleep(wait)
                continue
            resp.raise_for_status()

            bindings = resp.json()["results"]["bindings"]
            by_entity: dict[str, list[dict]] = defaultdict(list)

            for b in bindings:
                entity_id  = b["entity"]["value"].split("/")[-1]
                parent_id  = b["parent"]["value"].split("/")[-1]
                parent_lbl = b.get("parentLabel", {}).get("value", None)
                via        = b.get("via", {}).get("value", "P279")
                # Skip entries where label is just the QID (no English label)
                if parent_lbl == parent_id:
                    parent_lbl = None
                by_entity[entity_id].append({"id": parent_id, "label": parent_lbl, "via": via})

            # Cap to MAX_PARENTS per entity
            return {qid: by_entity[qid][:MAX_PARENTS] for qid in qids}

        except requests.RequestException as exc:
            log.warning(f"  Attempt {attempt}/{RETRY_LIMIT} failed: {exc}")
            if attempt < RETRY_LIMIT:
                time.sleep(RETRY_DELAY * attempt)

    # All retries exhausted — return None sentinel for each QID
    return {qid: None for qid in qids}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Resume support: load existing cache
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE) as f:
            results: dict = json.load(f)
        log.info(f"Resuming — {len(results)} entries already cached")
    else:
        results = {}

    entities = get_all_wikidata_entities()
    log.info(f"Total grounded Wikidata entities in cache: {len(entities)}")

    # Only process entities not yet in cache
    todo = [e for e in entities if e["id"] not in results]
    log.info(f"Entities still to fetch: {len(todo)}")

    # Build lookup: QID → label
    label_map = {e["id"]: e["label"] for e in todo}

    # Split into batches
    batches = [todo[i:i + BATCH_SIZE] for i in range(0, len(todo), BATCH_SIZE)]
    total_batches = len(batches)
    log.info(f"Processing {total_batches} batches of up to {BATCH_SIZE} QIDs each")

    errors = 0
    for batch_idx, batch in enumerate(batches, 1):
        qids = [e["id"] for e in batch]
        log.info(f"Batch {batch_idx}/{total_batches}  ({len(qids)} QIDs)")

        batch_results = fetch_batch(qids)

        for qid in qids:
            parents = batch_results.get(qid)
            results[qid] = {
                "label":   label_map[qid],
                "parents": parents if parents is not None else [],
                "error":   parents is None,
            }
            if parents is None:
                errors += 1

        # Checkpoint every 10 batches
        if batch_idx % 10 == 0 or batch_idx == total_batches:
            with open(OUTPUT_FILE, "w") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            log.info(f"  Checkpoint saved ({len(results)} total, {errors} errors so far)")

        time.sleep(REQUEST_GAP)

    # Final save
    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    ok    = sum(1 for v in results.values() if not v.get("error") and v["parents"])
    empty = sum(1 for v in results.values() if not v.get("error") and not v["parents"])
    err   = sum(1 for v in results.values() if v.get("error"))
    log.info(f"\nDone.")
    log.info(f"  {ok} entities with parents")
    log.info(f"  {empty} entities with no parents found")
    log.info(f"  {err} failed (network errors)")
    log.info(f"Results saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
