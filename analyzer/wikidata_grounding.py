#!/usr/bin/env python3
"""
Wikidata Concept Grounding

Pipeline for each extracted concept label:
  1. Retrieve the 5 most relevant Wikidata matches
  2. Ask the LLM (via OpenRouter or vLLM) to choose the best match
  3. If there is no match, ask the LLM for alternative queries and retry

Usage:
  python analyzer/wikidata_grounding.py \\
    --input-file output/concepts/concepts_list.txt \\
    --openrouter-key sk-or-... \\
    --output output/wikidata/grounded_concepts.json

  # Quick test on the first 10 concepts:
  python analyzer/wikidata_grounding.py ... --limit 10 --verbose
"""

import json
import random
import time
import logging
import argparse
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

import re

import requests

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
# Constants
# ---------------------------------------------------------------------------
WIKIDATA_SEARCH_URL = "https://www.wikidata.org/w/api.php"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
VLLM_DEFAULT_URL = "http://localhost:8000/v1"
WIKIDATA_USER_AGENT = os.getenv(
    "WIKIDATA_USER_AGENT",
    "CodeGraphWikidataGrounder/1.0 (https://github.com/halykoss/CodeGraph)",
)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
@dataclass
class ConceptMatch:
    concept: str
    status: str = "pending"           # pending | found | not_found | discarded
    wikidata_id: Optional[str] = None
    label: Optional[str] = None
    description: Optional[str] = None
    wikidata_url: Optional[str] = None
    confidence: Optional[str] = None  # high | medium | low
    reasoning: Optional[str] = None
    search_query: Optional[str] = None   # query that actually worked
    alternatives_tried: List[str] = field(default_factory=list)
    thinking: Optional[str] = None       # chain-of-thought reasoning from LLM


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------
class WikidataGrounder:
    """
    Ground extracted concept labels in Wikidata via:
      - Wikidata search (top 5 candidates)
      - LLM selection of the best match
      - Retry with alternative queries if there is no match
    """

    def __init__(
        self,
        openrouter_api_key: str = "",
        model: str = "qwen/qwen3.5-397b-a17b",
        cache_dir: str = "cache/wikidata_grounding",
        wikidata_delay: float = 1.0,
        backend: str = "openrouter",
        vllm_url: str = VLLM_DEFAULT_URL,
    ):
        self.api_key = openrouter_api_key
        self.model = model
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.wikidata_delay = wikidata_delay
        self.backend = backend
        self.vllm_url = vllm_url.rstrip("/")
        self._last_wikidata_call: float = 0.0
        self._wikidata_headers = {"User-Agent": WIKIDATA_USER_AGENT}
        self._wikidata_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Wikidata search
    # ------------------------------------------------------------------

    def search_wikidata(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Return up to `limit` Wikidata candidates for a query.
        Every attempt, including retries, goes through the rate limiter to avoid 429 cascades."""
        params = {
            "action": "wbsearchentities",
            "format": "json",
            "language": "en",
            "search": query,
            "limit": limit,
            "type": "item",
        }
        backoff = 10.0
        for attempt in range(4):
            # Acquire a rate-limit slot; check+update atomically inside the lock.
            while True:
                with self._wikidata_lock:
                    now = time.time()
                    next_allowed = self._last_wikidata_call + self.wikidata_delay
                    if now >= next_allowed:
                        # Reserve the slot now so other threads back off.
                        self._last_wikidata_call = now
                        break
                    wait = next_allowed - now
                time.sleep(wait)
            try:
                resp = requests.get(
                    WIKIDATA_SEARCH_URL,
                    params=params,
                    headers=self._wikidata_headers,
                    timeout=15,
                )
                # Stamp completion time so delay is measured from request end.
                with self._wikidata_lock:
                    self._last_wikidata_call = time.time()
                if resp.status_code == 429:
                    retry_after = float(resp.headers.get("Retry-After", backoff))
                    wait_secs = max(retry_after, backoff)
                    logger.warning(
                        f"Wikidata 429 for '{query}', retry in {wait_secs:.0f}s (attempt {attempt+1}/4)"
                    )
                    with self._wikidata_lock:
                        self._last_wikidata_call = time.time() + wait_secs
                    time.sleep(wait_secs)
                    backoff = min(backoff * 2, 120.0)
                    continue
                resp.raise_for_status()
                return [
                    {
                        "id": r["id"],
                        "label": r.get("label", ""),
                        "description": r.get("description", "no description"),
                        "url": r.get("concepturi", f"https://www.wikidata.org/wiki/{r['id']}"),
                    }
                    for r in resp.json().get("search", [])
                ]
            except requests.exceptions.HTTPError:
                raise
            except Exception as e:
                logger.warning(f"Wikidata search error for '{query}': {e}")
                return []
        logger.warning(f"Wikidata search failed after retries for '{query}'")
        return []

    # ------------------------------------------------------------------
    # LLM calls
    # ------------------------------------------------------------------

    def _call_llm(self, prompt: str, max_tokens: int = 200) -> Tuple[str, str]:
        """Returns (thinking, content) tuple. thinking is always empty (reserved for future use)."""
        if self.backend == "vllm":
            url = f"{self.vllm_url}/chat/completions"
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
        else:
            url = OPENROUTER_URL
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "X-Title": "Wikidata Concept Grounding",
            }
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.2,
        }
        if self.backend == "vllm":
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=300)
            resp.raise_for_status()
            body = resp.json()
            content = (body["choices"][0]["message"].get("content") or "").strip()
            if not content:
                logger.debug(f"LLM raw response (content empty):\n{json.dumps(body, indent=2)}")
            else:
                logger.debug(f"LLM response:\n{content}")
            thinking = ""
            think_match = re.search(r"<think>(.*?)</think>", content, re.DOTALL)
            if think_match:
                thinking = think_match.group(1).strip()
                content = content[think_match.end():].strip()
            return thinking, content
        except requests.exceptions.HTTPError as e:
            logger.warning(f"LLM HTTP {e.response.status_code} error: {e.response.text[:600]}")
            return "", ""
        except Exception as e:
            logger.warning(f"LLM call error: {e}")
            return "", ""

    def llm_select_best(
        self, concept: str, candidates: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Ask the LLM to choose the most relevant Wikidata candidate.

        Returns:
            dict with keys: candidate (dict or None), confidence (str), reasoning (str), discard (bool)
        """
        lines = "\n".join(
            f"{i + 1}. [{c['id']}] {c['label']} — {c['description']}"
            for i, c in enumerate(candidates)
        )
        prompt = f"""You are matching a technical label to a Wikidata entity.
First, think step by step inside <think>...</think> tags (keep it brief, max 10 sentences). After </think>, output only the formatted answer below.

Label: "{concept}"

Wikidata candidates:
{lines}

Context: This label comes from a knowledge graph built from real-world software repositories. It can be an algorithm, data structure, computational method, protocol, standard, chip, library, design pattern, or any other technical concept found in software — but it can also be a plain function/method name with no technical concept worth matching.

── WHEN TO MATCH ────────────────────────────────────────────────────────────
Match when the label IS or CLEARLY SPECIALIZES the candidate concept:
- Specialization → parent type:
    "min heap" → "heap (data structure)"                ✓   "depth first search" → "graph traversal"          ✓
    "quicksort variant" → "sorting algorithm"         ✓   "levenshtein distance" → "edit distance"          ✓
- Abbreviation / camelCase → canonical name:
    "uniformlbp" → "local binary patterns"              ✓   "moving weighted avg zscore" → "standard score"   ✓
    "numericgradientchecker" → "finite difference method" ✓
- Operation/calculation on X → X itself (data structure, library, or physical/math concept):
    "array length calculation" → "array"                ✓   "string reversal" → "string (computer science)"  ✓
    "pandas operations" → "Pandas (software)"           ✓   "trajectory calculation" → "trajectory"           ✓
- Indirect or descriptive label → underlying concept:
    "insecure compare" → "timing attack"                ✓   "address calculation with wraparound" → "circular buffer" ✓
    "node evaluation with memoization" → "memoization"  ✓   "service mediation pattern" → "mediator pattern"  ✓
    "matrix inverse via geometric series" → "Neumann series" ✓

── WHEN NOT TO MATCH (use NONE) ─────────────────────────────────────────────
Use NONE — not a discard — when none of the candidates fit but the label contains a real technical concept:
- Candidate is a sibling, not the right concept:
    "a* pathfinding" → "dijkstra's concept"           ✗   "bloom filter" → "hash table"                     ✗
- Candidate is the parent category of a specific concept (retry will find the specific item):
    "semi supervised GAN" → "semi-supervised learning"  ✗   "homophily based edge addition" → "social network analysis" ✗
- Candidate is an organization/company sharing an acronym instead of the technical concept:
    "csc" → "CSC – IT Center for Science"               ✗  (csc is the cosecant function, not an IT company)
    "acm" → "Association for Computing Machinery"       ✗  (in a code graph, acm likely refers to an algorithm)
- Candidate merely shares a word without semantic overlap:
    "scale date" → "Julian day"                         ✗   "merge sort" → "divide and conquer"               ✗
    "triple state selection" → "three-state logic"      ✗   "induced subgraph mining" → "induced subgraph isomorphism problem" ✗
- Candidate is a specific subtype of a broader label (matching downward):
    "sorting algorithm" → "quicksort"                 ✗   "graph traversal" → "breadth-first search"        ✗
- Candidate is the problem, not the concept/method:
    "dijkstra" → "shortest path problem"                ✗   "knapsack dp" → "knapsack problem"                ✗
- A more specific Wikidata item clearly exists:
    "sum cumulative distribution function" → "distribution function" ✗   "great circle midpoint calculation" → "great-circle distance" ✗

── WHEN TO DISCARD ──────────────────────────────────────────────────────────
DISCARD only when the label has ZERO identifiable technical content — it is a pure CRUD accessor, domain-specific getter, or a generic word with no CS meaning:
    "findeventobyid", "getuserbyrole", "containsrecipe", "getstudyweeks", "spellutil getshortspell" → DISCARD

Do NOT discard if the label contains ANY of the following — use NONE instead:
  · a data structure name:  array, vector, list, tree, graph, queue, stack, heap, map, set, pair, trie, buffer, …
  · an algorithm keyword: sort, search, filter, hash, traverse, encode, interpolate, compress, insert, partition, …
  · an operation type:      loop, iteration, recursion, arithmetic, atomic, sum, increment, decrement, lock, …
  · a library / tool name:  pandas, numpy, lodash, …
  · a known protocol, standard, or function name: strstr, strcpy, rpartition, modbus, …
Examples of labels that must NOT be discarded:
    "vector element insertion" → NONE  (vector = data structure, insertion = operation)
    "hash64"                   → NONE  (contains "hash" — hash function or 64-bit hash)
    "rpartition"               → NONE  (Python string partition method — real string operation)

Give your answer in this EXACT format (nothing else):
CANDIDATE: [number 1-{len(candidates)} or NONE or DISCARD]
CONFIDENCE: [HIGH / MEDIUM / LOW]
REASON: [one concise sentence]"""

        thinking, response = self._call_llm(prompt, max_tokens=4000)
        best_idx: Optional[int] = None
        confidence = "low"
        reasoning = ""
        discard = False

        for line in response.splitlines():
            upper = line.upper()
            if upper.startswith("CANDIDATE:"):
                val = line.split(":", 1)[1].strip().upper()
                if "DISCARD" in val:
                    discard = True
                elif "NONE" not in val:
                    for ch in line.split(":", 1)[1].strip():
                        if ch.isdigit():
                            idx = int(ch) - 1
                            if 0 <= idx < len(candidates):
                                best_idx = idx
                            break
            elif upper.startswith("CONFIDENCE:"):
                c = line.split(":", 1)[1].strip().upper()
                confidence = "high" if "HIGH" in c else ("low" if "LOW" in c else "medium")
            elif upper.startswith("REASON:"):
                reasoning = line.split(":", 1)[1].strip()

        return {
            "candidate": candidates[best_idx] if best_idx is not None else None,
            "confidence": confidence,
            "reasoning": reasoning,
            "thinking": thinking,
            "discard": discard,
        }

    def llm_suggest_alternatives(self, concept: str) -> Tuple[bool, List[str]]:
        """
        Ask the LLM to suggest alternative Wikidata search queries.

        Returns:
            Tuple (discard, alternatives):
              discard=True if the label is not a technical concept worth matching;
              alternatives is the list of alternative queries, empty if none is useful.
        """
        prompt = f"""The Wikidata search for "{concept}" returned no useful results.
First, think step by step inside <think>...</think> tags (keep it brief, max 10 sentences). After </think>, output only the formatted answer below.

This label comes from a code knowledge graph. It can be an algorithm, data structure, protocol, chip, standard, library, design pattern, or any technical concept found in real-world software.

── HOW TO REFORMULATE ───────────────────────────────────────────────────────
Expand abbreviations and acronyms in their technical context:
    "bfs" → "breadth-first search"        "dfs" → "depth-first search"
    "dp"  → "dynamic programming"         "bst" → "binary search tree"
    "lru" → "least recently used"         "kmp" → "Knuth–Morris–Pratt concept"
    "mst" → "minimum spanning tree"

Use the inventor's canonical name:
    "dijkstra" → "Dijkstra's algorithm"   "bellman ford" → "Bellman–Ford algorithm"
    "prim"     → "Prim's algorithm"       "kruskal"      → "Kruskal's algorithm"

For protocol/chip/standard labels, search the technical name directly:
    "can bus" → "CAN bus"   "modbus" → "Modbus"   "vpc3" → "VPC3 PROFIBUS"

For implementation-technique labels, infer the canonical concept name:
    "fast power" → "exponentiation by squaring"
    "address calculation with wraparound" → "circular buffer"
    "dcm orthonormalization" → "Gram-Schmidt process"

For "X operations/processing/methods" where X is a named library, suggest the library:
    "pandas operations" → "Pandas (software)", "DataFrame"
    "numpy operations"  → "NumPy", "array (data structure)"

For "X calculation/operation/insertion/…" where X is a data structure or physical/math concept, suggest X:
    "array length calculation"   → "array (data structure)", "array"
    "vector element insertion"   → "dynamic array", "array (data structure)"
    "linked list traversal"      → "linked list"
    "trajectory calculation"     → "trajectory", "kinematics", "projectile motion"
    "digit sum comparison"       → "digit sum", "digital root"

For hash-related labels, suggest the hash concept directly:
    "hash64"  → "hash function", "64-bit hash", "non-cryptographic hash function"

For string method names, suggest the string operation concept:
    "rpartition" → "partition (string)", "string operation", "string (computer science)"

── SPECIFICITY RULES ────────────────────────────────────────────────────────
Search for the SPECIFIC concept, not its parent category:
    "minimize slsqp"  → "Sequential Least Squares Programming", NOT "optimization algorithm"
    "a star search"   → "A* search algorithm", NOT "pathfinding"
    "rsa encryption"  → "RSA (cryptosystem)", NOT "public-key cryptography"
    "homophily based edge addition" → "homophily", NOT "social network analysis"

For concept families, try both specific and parent:
    "quicksort variant"  → "quicksort" AND "sorting algorithm"
    "sliding window max" → "sliding window technique" AND "queue (data structure)"

Avoid semantically wrong alternatives (different concept, not just different name):
    "channel prefixing" → NOT "channel bonding"   "scale date" → NOT "Julian day"
    "csc" → NOT an IT organization (try "cosecant" or "trigonometric function")

Only suggest alternatives genuinely related to the label — not just acronym lookalikes:
    "vpc3" → NOT "Van Emde Boas tree"   "pcl" → NOT "priority queue" (PCL is Point Cloud Library)

── WHEN TO DISCARD ──────────────────────────────────────────────────────────
Reply DISCARD (and nothing else) only if the label is a pure CRUD accessor, domain-specific getter, or a generic word with zero technical content (e.g. "findeventobyid", "getuserbyrole", "containsrecipe").
If the label might have technical content but no good alternative exists, reply with an empty list.

Suggest up to 5 Wikidata search queries. Reply ONLY with a numbered list or DISCARD, no preamble, no explanations:
1. ...
2. ...
3. ...
4. ...
5. ..."""

        _, response = self._call_llm(prompt, max_tokens=5000)
        if response.strip().upper() == "DISCARD":
            return True, []
        alternatives: List[str] = []
        for line in response.splitlines():
            line = line.strip()
            if line and line[0].isdigit():
                clean = line.lstrip("0123456789.-) ").strip()
                if clean:
                    alternatives.append(clean)
        return False, alternatives[:5]

    # ------------------------------------------------------------------
    # Single-concept pipeline
    # ------------------------------------------------------------------

    def _cache_path(self, concept: str) -> Path:
        safe = concept.lower().replace(" ", "_").replace("/", "_")[:80]
        return self.cache_dir / f"{safe}.json"

    def _load_cache(self, concept: str) -> Optional[ConceptMatch]:
        p = self._cache_path(concept)
        if p.exists():
            try:
                return ConceptMatch(**json.loads(p.read_text()))
            except Exception:
                pass
        return None

    def _save_cache(self, result: ConceptMatch) -> None:
        p = self._cache_path(result.concept)
        p.write_text(json.dumps(asdict(result), indent=2, ensure_ascii=False))

    def merge_concept(self, concept: str) -> ConceptMatch:
        """
        Full pipeline for a single concept:
          1. Search top-5 candidates on Wikidata
          2. LLM selects the best match
          3. If there is no match, the LLM suggests alternatives and retries
        """
        cached = self._load_cache(concept)
        if cached:
            logger.info(f"Cache hit: {concept}")
            return cached

        result = ConceptMatch(concept=concept)

        # Step 1: Search Wikidata
        candidates = self.search_wikidata(concept, limit=5)

        # Step 2: LLM selects the best direct candidate
        first_pass_medium: Optional[Dict[str, Any]] = None
        if candidates:
            selection = self.llm_select_best(concept, candidates)
            if selection.get("discard"):
                result.status = "discarded"
                result.reasoning = selection.get("reasoning", "")
                self._save_cache(result)
                return result
            elif selection["candidate"] and selection["confidence"] == "high":
                c = selection["candidate"]
                result.wikidata_id = c["id"]
                result.label = c["label"]
                result.description = c["description"]
                result.wikidata_url = c["url"]
                result.confidence = selection["confidence"]
                result.reasoning = selection["reasoning"]
                result.thinking = selection.get("thinking")
                result.search_query = concept
                result.status = "found"
                self._save_cache(result)
                return result
            elif selection["candidate"] and selection["confidence"] == "medium":
                # Keep the medium-confidence result as fallback, but try alternatives first
                first_pass_medium = {**selection, "query": concept}

        # Step 3: No high-confidence match; ask the LLM for alternatives and retry
        discard, alternatives = self.llm_suggest_alternatives(concept)
        if discard:
            result.status = "discarded"
            self._save_cache(result)
            return result
        result.alternatives_tried = alternatives

        for alt in alternatives:
            alt_candidates = self.search_wikidata(alt, limit=5)
            if not alt_candidates:
                continue
            selection = self.llm_select_best(concept, alt_candidates)
            if selection["candidate"] and selection["confidence"] in ("high", "medium"):
                c = selection["candidate"]
                result.wikidata_id = c["id"]
                result.label = c["label"]
                result.description = c["description"]
                result.wikidata_url = c["url"]
                result.confidence = selection["confidence"]
                result.reasoning = selection["reasoning"]
                result.search_query = alt
                result.status = "found"
                self._save_cache(result)
                return result

        # Step 4: No useful alternative; use the first medium-confidence result if available
        if first_pass_medium and first_pass_medium["candidate"]:
            c = first_pass_medium["candidate"]
            result.wikidata_id = c["id"]
            result.label = c["label"]
            result.description = c["description"]
            result.wikidata_url = c["url"]
            result.confidence = first_pass_medium["confidence"]
            result.reasoning = first_pass_medium["reasoning"]
            result.search_query = first_pass_medium["query"]
            result.status = "found"
            self._save_cache(result)
            return result

        result.status = "not_found"
        self._save_cache(result)
        return result

    # ------------------------------------------------------------------
    # Pipeline batch
    # ------------------------------------------------------------------

    def merge_all(self, concepts: List[str], workers: int = 1) -> List[ConceptMatch]:
        """Run grounding for all concepts, with progress logging."""
        if workers <= 1:
            return self._merge_all_sequential(concepts)
        return self._merge_all_parallel(concepts, workers)

    def _merge_all_sequential(self, concepts: List[str]) -> List[ConceptMatch]:
        results: List[ConceptMatch] = []
        found = 0
        total = len(concepts)

        for i, concept in enumerate(concepts, 1):
            logger.info(f"[{i:>5}/{total}] {concept}")
            r = self.merge_concept(concept)
            results.append(r)

            if r.status == "found":
                found += 1
                alt_note = f" (via '{r.search_query}')" if r.search_query != concept else ""
                logger.info(
                    f"           -> {r.wikidata_id} | {r.label} [{r.confidence}]{alt_note}"
                )
            elif r.status == "discarded":
                logger.info(f"           -> discarded (not a technical concept)")
            else:
                tried = f", tried: {r.alternatives_tried}" if r.alternatives_tried else ""
                logger.info(f"           -> not found{tried}")

        logger.info(f"\nDone: {found}/{total} concepts matched on Wikidata")
        return results

    def _merge_all_parallel(self, concepts: List[str], workers: int) -> List[ConceptMatch]:
        total = len(concepts)
        results: List[ConceptMatch] = [None] * total  # type: ignore[list-item]
        counter = {"done": 0, "found": 0}
        lock = threading.Lock()

        def process(idx: int, concept: str) -> ConceptMatch:
            r = self.merge_concept(concept)
            with lock:
                counter["done"] += 1
                done = counter["done"]
                if r.status == "found":
                    counter["found"] += 1
                    alt_note = f" (via '{r.search_query}')" if r.search_query != concept else ""
                    logger.info(
                        f"[{done:>5}/{total}] {concept} -> {r.wikidata_id} | {r.label} [{r.confidence}]{alt_note}"
                    )
                elif r.status == "discarded":
                    logger.info(f"[{done:>5}/{total}] {concept} -> discarded")
                else:
                    tried = f", tried: {r.alternatives_tried}" if r.alternatives_tried else ""
                    logger.info(f"[{done:>5}/{total}] {concept} -> not found{tried}")
            return r

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(process, i, d): i for i, d in enumerate(concepts)}
            for future in as_completed(futures):
                idx = futures[future]
                results[idx] = future.result()

        logger.info(f"\nDone: {counter['found']}/{total} concepts matched on Wikidata")
        return results


def load_concepts_from_file(path: str) -> List[str]:
    """Read concept labels from a text file, one per line; # lines are comments."""
    concepts = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                concepts.append(line)
    logger.info(f"Loaded {len(concepts)} unique concepts from {path}")
    return concepts


def save_results(results: List[ConceptMatch], output_path: str) -> None:
    found = [r for r in results if r.status == "found"]
    discarded = [r for r in results if r.status == "discarded"]
    not_found = [r for r in results if r.status == "not_found"]
    data = {
        "metadata": {
            "total": len(results),
            "found": len(found),
            "not_found": len(not_found),
            "discarded": len(discarded),
            "match_rate": f"{len(found) / len(results) * 100:.1f}%" if results else "0%",
        },
        "results": [asdict(r) for r in results],
    }
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    logger.info(f"Results saved to {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge graph concepts with Wikidata using LLM-assisted selection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # From a concept list
  python analyzer/wikidata_grounding.py \\
    --input-file output/concepts/concepts_list.txt \\
    --openrouter-key sk-or-... \\
    --output output/wikidata/grounded_concepts.json

  # Quick test on first 20 concepts
  python analyzer/wikidata_grounding.py \\
    --input-file output/concepts/concepts_list.txt \\
    --openrouter-key sk-or-... \\
    --limit 20 --verbose
        """,
    )

    parser.add_argument(
        "--input-file",
        required=True,
        help="Path to a text file with one concept per line (# lines are comments).",
    )

    # Backend
    parser.add_argument(
        "--backend",
        choices=["openrouter", "vllm"],
        default=os.getenv("BACKEND", "openrouter"),
        help="LLM backend: openrouter (default) or vllm",
    )
    parser.add_argument("--openrouter-key", default=os.getenv("OPENROUTER_API_KEY", ""))
    parser.add_argument(
        "--vllm-url",
        default=os.getenv("VLLM_URL", VLLM_DEFAULT_URL),
        help=f"Base URL of the vLLM server (default: {VLLM_DEFAULT_URL})",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("LLM_MODEL", "qwen/qwen3.5-397b-a17b"),
        help="Model ID (OpenRouter model ID or vLLM model name)",
    )

    # Output
    parser.add_argument(
        "--output",
        default="output/wikidata/grounded_concepts.json",
        help="Output JSON file path",
    )
    parser.add_argument(
        "--cache-dir",
        default="cache/wikidata_grounding",
        help="Directory for per-concept result cache",
    )

    # Options
    parser.add_argument(
        "--wikidata-delay",
        type=float,
        default=float(os.getenv("WIKIDATA_DELAY", "1.0")),
        help="Minimum seconds between Wikidata API calls (default: 1.0)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Process only the first N concepts (useful for testing)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=int(os.getenv("WORKERS", "1")),
        help="Number of parallel workers (default: 1; max recommended: 8)",
    )
    parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    api_key = args.openrouter_key
    if args.backend == "openrouter" and not api_key:
        parser.error(
            "OpenRouter API key required. Use --openrouter-key or set OPENROUTER_API_KEY."
        )

    concepts = load_concepts_from_file(args.input_file)

    random.shuffle(concepts)

    if args.limit:
        concepts = concepts[: args.limit]
        logger.info(f"Limited to first {args.limit} concepts (after shuffle)")

    # Run the grounding pipeline.
    logger.info(f"Backend: {args.backend}" + (f" ({args.vllm_url})" if args.backend == "vllm" else ""))
    merger = WikidataGrounder(
        openrouter_api_key=api_key,
        model=args.model,
        cache_dir=args.cache_dir,
        wikidata_delay=args.wikidata_delay,
        backend=args.backend,
        vllm_url=args.vllm_url,
    )
    results = merger.merge_all(concepts, workers=args.workers)

    # Save output.
    save_results(results, args.output)


if __name__ == "__main__":
    main()
