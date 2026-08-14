"""
Build 3-hop ancestor chains for every grounded Wikidata entity using
wikidata_parents.json as the level-1 cache, then incrementally fetching levels
2 and 3.

The graph converges upward: many entities at L0, fewer unique nodes at L1,
fewer again at L2, and few at L3. Total calls are therefore roughly
L0_batches + L1_batches + L2_batches.

Strategy per entity:
  - L1: pick the first P279 parent; if absent, the first P31 parent.
  - L2: apply the same rule to the L1 node.
  - L3: apply the same rule to the L2 node.

Output: wikidata_hierarchy.json
  {
    "Q11660": {
      "label": "artificial intelligence",
      "chain": [
        {"id": "Q120208", "label": "emerging technology",   "depth": 1, "via": "P279"},
        {"id": "Q2375005","label": "technology",            "depth": 2, "via": "P279"},
        {"id": "Q11016",  "label": "technology",            "depth": 3, "via": "P279"}
      ]
    },
    ...
  }
  the chain can be shorter than 3 if no further parents exist.
"""

import json
import os
import time
import logging
from collections import defaultdict

import requests

WIKIDATA_SPARQL = "https://query.wikidata.org/sparql"
HEADERS = {
    "Accept": "application/sparql-results+json",
    "User-Agent": "CodeGraphEmbeddingDataset/1.0 (https://github.com/halykoss/CodeGraph)",
}

L1_FILE    = "wikidata_parents.json"   # output from fetch_wikidata_parents.py
L2_FILE    = "wikidata_l2.json"        # intermediate cache for level-2 parents
L3_FILE    = "wikidata_l3.json"        # intermediate cache for level-3 parents
OUTPUT_FILE = "wikidata_hierarchy.json"

BATCH_SIZE  = 50
RETRY_LIMIT = 3
RETRY_DELAY = 10
REQUEST_GAP = 1.5

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Wikidata batched SPARQL, reused for every level
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


def fetch_batch(qids):
    """Return {qid: [{id, label, via}, ...]} for the given QIDs."""
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
                log.warning(f"  Rate-limited, waiting {wait}s ...")
                time.sleep(wait)
                continue
            resp.raise_for_status()

            by_entity: dict[str, list[dict]] = defaultdict(list)
            for b in resp.json()["results"]["bindings"]:
                eid  = b["entity"]["value"].split("/")[-1]
                pid  = b["parent"]["value"].split("/")[-1]
                plbl = b.get("parentLabel", {}).get("value", None)
                via  = b.get("via", {}).get("value", "P279")
                if plbl == pid:
                    plbl = None
                by_entity[eid].append({"id": pid, "label": plbl, "via": via})

            return {q: by_entity[q] for q in qids}

        except requests.RequestException as exc:
            log.warning(f"  Attempt {attempt}/{RETRY_LIMIT} failed: {exc}")
            if attempt < RETRY_LIMIT:
                time.sleep(RETRY_DELAY * attempt)

    return {q: None for q in qids}


def fetch_level(qids_to_fetch, cache_file):
    """
    Fetch parents for a set of QIDs, using cache_file to resume interrupted runs.
    Returns complete {qid: [parents]} dict (empty list if none found, None if error).
    """
    cache: dict = {}
    if os.path.exists(cache_file):
        with open(cache_file) as f:
            cache = json.load(f)
        log.info(f"  Loaded {len(cache)} cached entries from {cache_file}")

    todo = [q for q in qids_to_fetch if q not in cache]
    log.info(f"  {len(todo)} QIDs still to fetch (out of {len(qids_to_fetch)})")

    batches = [todo[i:i + BATCH_SIZE] for i in range(0, len(todo), BATCH_SIZE)]
    for i, batch in enumerate(batches, 1):
        result = fetch_batch(batch)
        cache.update(result)
        if i % 10 == 0 or i == len(batches):
            with open(cache_file, "w") as f:
                json.dump(cache, f, indent=2, ensure_ascii=False)
            log.info(f"  Checkpoint {i}/{len(batches)} saved to {cache_file}")
        time.sleep(REQUEST_GAP)

    # Final save
    with open(cache_file, "w") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)

    return cache


def pick_primary(parents):
    """Pick one representative parent: prefer P279 over P31."""
    if not parents:
        return None
    p279 = [p for p in parents if p.get("via") == "P279"]
    return (p279 or parents)[0]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # ---- Load L1 data -------------------------------------------------------
    if not os.path.exists(L1_FILE):
        log.error(f"{L1_FILE} not found. Run fetch_wikidata_parents.py first.")
        return

    with open(L1_FILE) as f:
        l1_data: dict = json.load(f)
    log.info(f"Loaded {len(l1_data)} entities from {L1_FILE}")

    # ---- Fetch L2: parents of L1 nodes ------------------------------------
    log.info("\n--- Level 2: parents of L1 nodes ---")
    l1_parent_qids = set()
    for entry in l1_data.values():
        for p in entry.get("parents") or []:
            l1_parent_qids.add(p["id"])
    log.info(f"Unique L1 parent QIDs to resolve: {len(l1_parent_qids)}")
    l2_data = fetch_level(sorted(l1_parent_qids), L2_FILE)

    # ---- Fetch L3: parents of L2 nodes ------------------------------------
    log.info("\n--- Level 3: parents of L2 nodes ---")
    l2_parent_qids = set()
    for parents in l2_data.values():
        for p in (parents or []):
            l2_parent_qids.add(p["id"])
    log.info(f"Unique L2 parent QIDs to resolve: {len(l2_parent_qids)}")
    l3_data = fetch_level(sorted(l2_parent_qids), L3_FILE)

    # ---- Assemble chains ---------------------------------------------------
    log.info("\n--- Assembling chains ---")
    hierarchy: dict = {}

    for qid, entry in l1_data.items():
        chain = []

        # L1
        p1 = pick_primary(entry.get("parents"))
        if p1:
            chain.append({**p1, "depth": 1})

            # L2
            p2 = pick_primary(l2_data.get(p1["id"]))
            if p2:
                chain.append({**p2, "depth": 2})

                # L3
                p3 = pick_primary(l3_data.get(p2["id"]))
                if p3:
                    chain.append({**p3, "depth": 3})

        hierarchy[qid] = {
            "label": entry.get("label"),
            "chain": chain,
        }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(hierarchy, f, indent=2, ensure_ascii=False)

    # ---- Stats -------------------------------------------------------------
    depths = [len(v["chain"]) for v in hierarchy.values()]
    for d in range(4):
        n = sum(1 for x in depths if x == d)
        log.info(f"  chain depth {d}: {n} entities")
    log.info(f"\nSaved {len(hierarchy)} entries to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
