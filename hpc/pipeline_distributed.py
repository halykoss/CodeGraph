#!/usr/bin/env python3
"""
Pipeline distribuita con Ray + vLLM.
Sostituisce StackEduCodeGraphOrchestrator di main.py per esecuzioni multi-nodo.

Esempio d'uso:
    python -m hpc.pipeline_distributed --samples 1000 --language Python --model mistralai/Devstral-Small-2505
"""

import os
import sys
import json
import glob
import time
import random
import logging
import argparse
from typing import List, Dict, Any, Optional, Set, TYPE_CHECKING

if TYPE_CHECKING:
    from src.graph_builder import GraphBuilder

from tqdm import tqdm

from utils.utils import (
    save_samples_to_file,
    load_samples_from_file,
    save_samples_arrow,
    load_samples_auto,
)
from hpc.download_optimized import collect_new_sample_metadata, download_blob_contents

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ── Default configuration ──────────────────────────────────────────────
DEFAULT_MODEL = "Qwen/Qwen3-Coder-30B-A3B-Instruct"
DEFAULT_TENSOR_PARALLEL = 4
DEFAULT_MAX_MODEL_LEN = 16384
DEFAULT_GPU_MEM_UTIL = 0.85
DEFAULT_BATCH_SIZE = 64
DEFAULT_MAX_TOKENS = 2048
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.8
DEFAULT_PROMPT_OVERHEAD_TOKENS = 500
CHUNK_SIZE = 20_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Distributed semantic graph construction with Ray + vLLM"
    )

    # Same args as main.py
    parser.add_argument(
        "--samples", "-n", type=int, default=1000,
        help="Number of samples to download and process (default: 1000)",
    )
    parser.add_argument(
        "--language", "-l", type=str, default=None,
        help="Programming language filter. If omitted, process all cached languages.",
    )
    parser.add_argument(
        "--output-dir", "-o", type=str, default="output/full_pipeline",
        help="Output directory for results (default: output/full_pipeline)",
    )
    parser.add_argument(
        "--cache-dir", "-c", type=str, default="cache",
        help="Cache directory for downloaded samples (default: cache)",
    )
    parser.add_argument(
        "--disable-cache", action="store_true",
        help="Disable caching of downloaded samples",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from existing results if available",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--download-only", action="store_true",
        help="Only download and cache samples, without GPU processing",
    )

    # Distributed / vLLM-specific arguments
    parser.add_argument(
        "--model", type=str, default=DEFAULT_MODEL,
        help=f"Hugging Face model name or path for vLLM (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--tensor-parallel", type=int, default=DEFAULT_TENSOR_PARALLEL,
        help="Tensor parallelism size per actor (default: 4)",
    )
    parser.add_argument(
        "--max-model-len", type=int, default=DEFAULT_MAX_MODEL_LEN,
        help="Maximum model context length (default: 8192)",
    )
    parser.add_argument(
        "--gpu-memory-utilization", type=float, default=DEFAULT_GPU_MEM_UTIL,
        help="GPU memory utilization for vLLM (default: 0.90)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
        help="Samples per batch sent to each vLLM actor (default: 64)",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=DEFAULT_MAX_TOKENS,
        help="Maximum output tokens per generation (default: 2048)",
    )
    parser.add_argument(
        "--temperature", type=float, default=DEFAULT_TEMPERATURE,
        help="Sampling temperature (default: 0.7)",
    )
    parser.add_argument(
        "--top-p", type=float, default=DEFAULT_TOP_P,
        help="Top-p sampling (default: 0.8)",
    )
    parser.add_argument(
        "--init-batch-size", type=int, default=4,
        help="Number of vLLM actors to initialize in parallel. "
             "Limits I/O contention on shared filesystems during model loading. "
             "(default: 4)",
    )
    parser.add_argument(
        "--local-model-dir", type=str, default=None,
        help="Path to node-local model directory (e.g., /dev/shm/hf_model). "
             "When set, actors load from local ramdisk instead of the shared "
             "filesystem, avoiding I/O contention. Pre-broadcast the model "
             "with srun before launching the pipeline.",
    )
    parser.add_argument(
        "--prompt-overhead-tokens", type=int, default=DEFAULT_PROMPT_OVERHEAD_TOKENS,
        help=f"Tokens reserved for the prompt template when computing max code "
             f"input length (default: {DEFAULT_PROMPT_OVERHEAD_TOKENS})",
    )

    return parser.parse_args()


# ── Dataset download & cache (reuses utils/utils.py) ──────────────────

def download_dataset(
    num_samples: int,
    language: Optional[str],
    cache_dir: str,
    enable_cache: bool,
    download_only: bool = False,
    download_batch_size: int = 3000,
    skip_processed_ids: Set[str] = None,
) -> List[Dict]:
    """
    Download or load cached samples. Mirrors main.py download_dataset().

    Supports incremental download: if a cache file exists with fewer samples
    than requested, downloads only the missing ones in batches to avoid OOM.

    Args:
        num_samples: Number of samples to process (will take first N from cache)
        language: Programming language filter. If None, load all cached languages.
                 Must be set when download_only=True (need to know what to fetch).
        download_only: If True, force download from internet (login nodes).
                      If False, only load from cache (compute nodes without internet).
        download_batch_size: How many samples to download per batch (default 3000
                            to avoid OOM on login nodes with limited RAM).
        skip_processed_ids: Set of blob_ids already processed (from result files).
                           Counted toward the total so we don't re-download them.
    """
    os.makedirs(cache_dir, exist_ok=True)

    # Compute nodes (no internet): ONLY load from cache
    if not download_only:
        # Scan for both JSON and Arrow cache files
        if language:
            cache_files = sorted(set(
                glob.glob(os.path.join(cache_dir, f"stack_edu_*_{language}_samples.json"))
                + glob.glob(os.path.join(cache_dir, f"stack_edu_{language}_part_*.arrow"))
            ))
            # Also check canonical JSON name (glob * doesn't match zero chars)
            cache_file = os.path.join(cache_dir, f"stack_edu_{language}_samples.json")
            if os.path.exists(cache_file) and cache_file not in cache_files:
                cache_files.append(cache_file)
        else:
            # No language filter: load ALL cache files
            cache_files = sorted(set(
                glob.glob(os.path.join(cache_dir, "stack_edu_*.json"))
                + glob.glob(os.path.join(cache_dir, "stack_edu_*.arrow"))
            ))

        if not cache_files:
            lang_msg = language or "all languages"
            logger.error(
                f"No cache file found for {lang_msg} in {cache_dir}\n"
                f"Compute nodes may not have internet access. Populate the cache on a login node "
                f"or run this module with --download-only before submitting the GPU job."
            )
            return []

        # Merge ALL cache files, dedup by blob_id
        samples = []
        seen_ids = set()
        for cf in sorted(cache_files):
            loaded = load_samples_auto(cf) or []
            new_count = 0
            for s in loaded:
                bid = s.get("blob_id")
                if bid and bid not in seen_ids:
                    samples.append(s)
                    seen_ids.add(bid)
                    new_count += 1
            logger.info(f"  Cache {os.path.basename(cf)}: {len(loaded)} samples ({new_count} new)")

        if not samples:
            logger.error(f"Failed to load samples from cache files in {cache_dir}")
            return []

        lang_msg = language or "all languages"
        random.shuffle(samples)
        logger.info(f"✓ Loaded {len(samples)} unique samples ({lang_msg}) from {len(cache_files)} cache files (requested {num_samples}), shuffled")
        return samples[:num_samples]

    # Login nodes (with internet): Incremental download and cache
    if not language:
        logger.error("--language is required for --download-only mode (need to know what to fetch from HuggingFace)")
        return []
    ARROW_PART_SIZE = 100_000  # max samples per Arrow sub-file

    # Check if we already have some samples cached (both JSON and Arrow)
    existing_samples = []
    existing_blob_ids = set()

    cache_file = os.path.join(cache_dir, f"stack_edu_{language}_samples.json")
    cache_files = sorted(set(
        glob.glob(os.path.join(cache_dir, f"stack_edu_*_{language}_samples.json"))
        + glob.glob(os.path.join(cache_dir, f"stack_edu_{language}_part_*.arrow"))
    ))
    if os.path.exists(cache_file) and cache_file not in cache_files:
        cache_files.append(cache_file)
    for cf in cache_files:
        loaded = load_samples_auto(cf) or []
        new_count = 0
        for s in loaded:
            bid = s.get("blob_id")
            if bid and bid not in existing_blob_ids:
                existing_samples.append(s)
                existing_blob_ids.add(bid)
                new_count += 1
        logger.info(f"  Cache {os.path.basename(cf)}: {len(loaded)} samples ({new_count} new)")
    if existing_samples:
        logger.info(f"  Total cached (merged): {len(existing_samples)} samples")

    # Merge processed IDs into skip set so we don't re-download them
    all_skip_ids = set(existing_blob_ids)
    if skip_processed_ids:
        all_skip_ids |= skip_processed_ids
        logger.info(
            f"  Already processed: {len(skip_processed_ids)} samples "
            f"(total to skip: {len(all_skip_ids)})"
        )

    already_have = len(all_skip_ids)
    # Track which Arrow parts are finalized (full and already written to disk)
    saved_parts = set()

    if already_have >= num_samples:
        logger.info(
            f"✓ Already have {already_have} samples "
            f"({len(existing_blob_ids)} cached + {len(skip_processed_ids or set())} processed). "
            f"Nothing to download."
        )
        if enable_cache and existing_samples:
            _save_cache_arrow_parts(existing_samples, cache_dir, language, ARROW_PART_SIZE)
        return existing_samples[:num_samples]

    # Phase 1: Scan dataset ONCE to collect all needed metadata
    target_new = num_samples - already_have
    logger.info(
        f"Need {target_new} more samples "
        f"(have {len(existing_blob_ids)} cached + {len(skip_processed_ids or set())} processed, "
        f"want {num_samples}). Scanning dataset for new IDs..."
    )

    new_metadata = collect_new_sample_metadata(
        num_samples=target_new,
        programming_language=language,
        skip_blob_ids=all_skip_ids,
    )

    if not new_metadata:
        logger.warning("No new samples found in dataset.")
        if enable_cache and existing_samples:
            _save_cache_arrow_parts(existing_samples, cache_dir, language, ARROW_PART_SIZE)
        return existing_samples[:num_samples]

    logger.info(
        f"Found {len(new_metadata)} new sample IDs. "
        f"Downloading contents in batches of {download_batch_size} with checkpoint..."
    )

    # Phase 2: Download contents in batches with checkpoint after each
    all_samples = list(existing_samples)

    for batch_start in range(0, len(new_metadata), download_batch_size):
        batch_meta = new_metadata[batch_start : batch_start + download_batch_size]
        batch_num = batch_start // download_batch_size + 1
        total_batches = (len(new_metadata) + download_batch_size - 1) // download_batch_size

        logger.info(
            f"  Batch {batch_num}/{total_batches}: downloading {len(batch_meta)} samples..."
        )

        batch_samples = download_blob_contents(
            metadata_list=batch_meta,
            batch_size=100,
            max_workers=64,
        )

        # Deduplicate against what we already have
        new_samples = [
            s for s in batch_samples
            if s.get("blob_id") not in all_skip_ids
        ]
        all_samples.extend(new_samples)
        all_skip_ids.update(s.get("blob_id") for s in new_samples)

        logger.info(
            f"  Got {len(new_samples)} new samples. "
            f"Total cached: {len(all_samples)}"
        )

        # Save checkpoint: only writes the last (incomplete) sub-file
        if enable_cache:
            saved_parts = _save_cache_arrow_parts(
                all_samples, cache_dir, language, ARROW_PART_SIZE, saved_parts
            )

    logger.info(f"✓ Download complete: {len(all_samples)} samples cached")
    return all_samples[:num_samples]


def _save_cache_arrow_parts(
    samples: list, cache_dir: str, language: str, part_size: int,
    saved_parts: set = None,
) -> set:
    """Save samples to Arrow sub-files of max `part_size` each.

    Only writes sub-files that are new or have changed (the last/incomplete one).
    Returns the set of part indices that are finalized (full and already written).
    """
    if saved_parts is None:
        saved_parts = set()

    total_parts = (len(samples) + part_size - 1) // part_size if samples else 0

    for i in range(0, len(samples), part_size):
        part_idx = i // part_size
        is_last = part_idx == total_parts - 1

        # Skip parts that are already full and saved
        if part_idx in saved_parts and not is_last:
            continue

        part = samples[i : i + part_size]
        path = os.path.join(cache_dir, f"stack_edu_{language}_part_{part_idx}.arrow")
        save_samples_arrow(part, path)

        # Mark as finalized if full
        if len(part) == part_size:
            saved_parts.add(part_idx)

    logger.info(
        f"  Cache: {len(samples)} samples across {total_parts} Arrow sub-files"
    )
    return saved_parts


# ── Resume logic (mirrors main.py:393-413) ────────────────────────────

def load_processed_ids(output_dir: str) -> Set[str]:
    """Load already-processed sample IDs from existing result chunks (JSON + Arrow)."""
    processed_ids: Set[str] = set()
    part_files = sorted(set(
        glob.glob(os.path.join(output_dir, "processing_results_part_*.json"))
        + glob.glob(os.path.join(output_dir, "processing_results_part_*.arrow"))
    ))
    logger.info(f"Scanning {len(part_files)} result files in {output_dir}...")
    for p_file in part_files:
        try:
            count_before = len(processed_ids)
            if p_file.endswith(".arrow"):
                import pyarrow.feather as feather
                table = feather.read_table(p_file, columns=["sample_id"])
                for sid in table.column("sample_id").to_pylist():
                    if sid:
                        processed_ids.add(sid)
            else:
                with open(p_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for res in data.get("results", []):
                        if "sample_id" in res:
                            processed_ids.add(res["sample_id"])
            count_new = len(processed_ids) - count_before
            logger.info(
                f"  {os.path.basename(p_file)}: {count_new} samples "
                f"(total so far: {len(processed_ids)})"
            )
        except Exception as e:
            logger.warning(f"  Impossibile caricare {p_file}: {e}")

    logger.info(
        f"Found {len(processed_ids)} unique processed samples "
        f"across {len(part_files)} files."
    )
    return processed_ids


def get_next_chunk_index(output_dir: str) -> int:
    """Determine the next chunk file index (checks both JSON and Arrow)."""
    existing = [
        f
        for f in os.listdir(output_dir)
        if f.startswith("processing_results_part_")
        and (f.endswith(".json") or f.endswith(".arrow"))
    ]
    if not existing:
        return 0
    try:
        indices = [
            int(f.replace("processing_results_part_", "").replace(".json", "").replace(".arrow", ""))
            for f in existing
        ]
        return max(indices) + 1
    except ValueError:
        return 0


# ── Graph building (mirrors main.py:159-233) ──────────────────────────

def build_graph_from_result(
    graph_builder: "GraphBuilder",
    result: Dict[str, Any],
    sample: Dict[str, Any],
) -> None:
    """
    Add nodes and edges to the graph for a single processed sample.
    Replicates main.py process_code_sample() graph-building logic.
    """
    file_node_id = result["file_node_id"]
    domains = result.get("domains", [])
    algorithms = result.get("algorithms", [])
    paradigms = result.get("paradigms", [])
    design_patterns = result.get("design_patterns", [])

    # File node
    file_metadata = {
        "language": result.get("language", "unknown"),
        "size": sample.get("size", 0),
        "repository": sample.get("repository", "unknown"),
        "file_path": sample.get("file_path", ""),
        "paradigms": paradigms,
        "design_patterns": design_patterns,
    }
    graph_builder.add_file_node(file_node_id, file_metadata)

    # Domain nodes + edges
    for domain in domains:
        graph_builder.add_domain_node(domain)
        graph_builder.add_belongs_to_domain_edge(file_node_id, domain)

    # Algorithm nodes + edges
    if isinstance(algorithms, list):
        for algorithm in algorithms:
            if isinstance(algorithm, dict):
                algo_name = algorithm.get("name", "unknown_algorithm")
                algo_metadata = {
                    "category": algorithm.get("category", ""),
                    "complexity": algorithm.get("complexity", ""),
                    "description": algorithm.get("description", ""),
                }
                graph_builder.add_algorithm_node(algo_name, algo_metadata)
                graph_builder.add_uses_algorithm_edge(file_node_id, algo_name)
            elif isinstance(algorithm, str):
                graph_builder.add_algorithm_node(algorithm)
                graph_builder.add_uses_algorithm_edge(file_node_id, algorithm)

    # Paradigm nodes + edges
    if isinstance(paradigms, list):
        for paradigm in paradigms:
            if isinstance(paradigm, dict):
                para_name = paradigm.get("name", "unknown_paradigm")
                para_metadata = {
                    "confidence": paradigm.get("confidence", ""),
                    "evidence": paradigm.get("evidence", ""),
                }
                graph_builder.add_programming_paradigm_node(para_name, para_metadata)
                graph_builder.add_uses_paradigm_edge(file_node_id, para_name)
            elif isinstance(paradigm, str):
                graph_builder.add_programming_paradigm_node(paradigm)
                graph_builder.add_uses_paradigm_edge(file_node_id, paradigm)

    # Design pattern nodes + edges
    if isinstance(design_patterns, list):
        for dp in design_patterns:
            if isinstance(dp, dict):
                dp_name = dp.get("name", "unknown_design_pattern")
                dp_metadata = {
                    "category": dp.get("category", ""),
                    "description": dp.get("description", ""),
                }
                graph_builder.add_design_pattern_node(dp_name, dp_metadata)
                graph_builder.add_uses_design_pattern_edge(file_node_id, dp_name)
            elif isinstance(dp, str):
                graph_builder.add_design_pattern_node(dp)
                graph_builder.add_uses_design_pattern_edge(file_node_id, dp)


# ── Statistics (mirrors main.py:279-299) ──────────────────────────────

def collect_stats(all_results: List[Dict], total_samples: int, errors: int) -> Dict:
    stats = {
        "total_samples": total_samples,
        "processed_samples": len(all_results),
        "errors": errors,
        "domains_found": set(),
        "algorithms_found": set(),
        "paradigms_found": set(),
        "design_patterns_found": set(),
    }

    for r in all_results:
        for d in r.get("domains", []):
            stats["domains_found"].add(d)
        for algo in r.get("algorithms", []):
            if isinstance(algo, dict) and "name" in algo:
                stats["algorithms_found"].add(algo["name"])
            elif isinstance(algo, str):
                stats["algorithms_found"].add(algo)
        for para in r.get("paradigms", []):
            if isinstance(para, dict) and "name" in para:
                stats["paradigms_found"].add(para["name"])
            elif isinstance(para, str):
                stats["paradigms_found"].add(para)
        for dp in r.get("design_patterns", []):
            if isinstance(dp, dict) and "name" in dp:
                stats["design_patterns_found"].add(dp["name"])
            elif isinstance(dp, str):
                stats["design_patterns_found"].add(dp)

    return stats


def print_statistics(stats: Dict) -> None:
    print("\n" + "=" * 50)
    print("PROCESSING STATISTICS")
    print("=" * 50)
    print(f"Total samples (all runs): {stats['total_samples']}")
    print(f"Successfully processed:   {stats['processed_samples']}")
    print(f"Errors encountered:       {stats['errors']}")
    if stats["total_samples"] > 0:
        rate = stats["processed_samples"] / stats["total_samples"] * 100
        print(f"Success rate:             {rate:.1f}%")

    for key, label in [
        ("domains_found", "domains"),
        ("algorithms_found", "algorithms"),
        ("paradigms_found", "paradigms"),
        ("design_patterns_found", "design patterns"),
    ]:
        items = sorted(x for x in stats[key] if x is not None)
        print(f"\nUnique {label} found: {len(items)}")
        preview = items[:10]
        suffix = "..." if len(items) > 10 else ""
        print(f"  {preview}{suffix}")


def save_chunk_results(
    results: List[Dict], stats: Dict, filepath: str
) -> None:
    """Save a chunk of results to Arrow format.

    Nested list fields (domains, algorithms, paradigms, design_patterns, functions)
    are serialized as JSON strings. Statistics are stored as Arrow schema metadata.
    Falls back to JSON if filepath ends with .json for backward compatibility.
    """
    if filepath.endswith(".json"):
        # Legacy JSON path
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "statistics": {
                        "total_samples": stats["total_samples"],
                        "processed_samples": stats["processed_samples"],
                        "errors": stats["errors"],
                        "domains_found": list(stats["domains_found"]),
                        "algorithms_found": list(stats["algorithms_found"]),
                        "paradigms_found": list(stats["paradigms_found"]),
                        "design_patterns_found": list(stats["design_patterns_found"]),
                    },
                    "results": results,
                },
                f,
                indent=2,
                ensure_ascii=False,
                default=str,
            )
        logger.info(f"Results saved to {filepath}")
        return

    import pyarrow as pa
    import pyarrow.feather as feather

    # Flatten nested fields to JSON strings
    list_fields = ("domains", "algorithms", "paradigms", "design_patterns", "functions")
    rows = []
    for r in results:
        row = dict(r)
        for field in list_fields:
            if field in row:
                row[field] = json.dumps(row[field], ensure_ascii=False, default=str)
        rows.append(row)

    if rows:
        keys = list(rows[0].keys())
        columns = {k: [r.get(k) for r in rows] for k in keys}
        table = pa.table(columns)
    else:
        table = pa.table({"sample_id": []})

    # Store statistics as schema metadata
    stats_meta = json.dumps({
        "total_samples": stats["total_samples"],
        "processed_samples": stats["processed_samples"],
        "errors": stats["errors"],
    }, default=str)
    meta = table.schema.metadata or {}
    meta[b"statistics"] = stats_meta.encode()
    table = table.replace_schema_metadata(meta)

    tmp_path = filepath + ".tmp"
    feather.write_feather(table, tmp_path)
    os.replace(tmp_path, filepath)
    logger.info(f"Results saved to {filepath}")


# ── Main pipeline ─────────────────────────────────────────────────────

def main():
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    print("=" * 60)
    print("Distributed Code Graph Construction (Ray + vLLM)")
    print(f"  Dataset:   HuggingFaceTB/stack-edu")
    print(f"  Samples:   {args.samples}")
    print(f"  Language:  {args.language or 'all'}")
    print(f"  Model:     {args.model}")
    print(f"  TP size:   {args.tensor_parallel}")
    print(f"  Output:    {args.output_dir}")
    if args.download_only:
        print(f"  Mode:      DOWNLOAD ONLY (no GPU processing)")
    print("=" * 60)

    # ── Download-only mode (for pre-caching on login nodes) ──────────
    if args.download_only:
        logger.info("Download-only mode...")

        # Check already processed samples to report stats
        os.makedirs(args.output_dir, exist_ok=True)
        processed_ids = load_processed_ids(args.output_dir) if args.resume else set()

        logger.info("Step 1: downloading / caching dataset...")
        samples = download_dataset(
            num_samples=args.samples,
            language=args.language,
            cache_dir=args.cache_dir,
            enable_cache=not args.disable_cache,
            download_only=True,
            skip_processed_ids=processed_ids,
        )

        if not samples:
            logger.error("No samples downloaded. Exiting.")
            return 1

        # Report how many still need processing
        if processed_ids:
            remaining = [s for s in samples if s.get("blob_id") not in processed_ids]
            logger.info(
                f"✓ {len(samples)} samples in cache, "
                f"{len(processed_ids)} already processed, "
                f"{len(remaining)} still to process"
            )
        else:
            logger.info(f"✓ Successfully downloaded and cached {len(samples)} samples")

        logger.info(f"✓ Cache location: {args.cache_dir}")
        logger.info("")
        logger.info("You can now submit the GPU job, which will read from cache:")
        logger.info(f"  sbatch hpc/run_slurm.sh")
        return 0

    # ── Step 0: Initialize Ray cluster ────────────────────────────────
    # Lazy imports: ray and GraphBuilder pull in torch/CUDA, not needed for download-only
    import ray
    from src.graph_builder import GraphBuilder

    ray.init(address="auto")
    resources = ray.cluster_resources()
    total_gpus = int(resources.get("GPU", 0))
    total_cpus = int(resources.get("CPU", 0))
    num_actors = total_gpus // args.tensor_parallel

    logger.info(f"Ray cluster: {total_cpus} CPUs, {total_gpus} GPUs")
    logger.info(f"Will create {num_actors} vLLM actors (TP={args.tensor_parallel})")

    # ── Step 1: Check already processed samples ─────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    processed_ids: Set[str] = set()

    if args.resume:
        processed_ids = load_processed_ids(args.output_dir)
        if processed_ids:
            logger.info(
                f"Found {len(processed_ids)} already processed samples. "
                "Will skip them."
            )

    # ── Step 2: Load dataset from cache (compute nodes have no internet) ──
    logger.info("Step 2: loading dataset from cache...")
    samples = download_dataset(
        num_samples=args.samples,
        language=args.language,
        cache_dir=args.cache_dir,
        enable_cache=not args.disable_cache,
        download_only=False,
    )

    if not samples:
        if processed_ids:
            logger.info(
                f"No cache file found, but {len(processed_ids)} samples "
                "already processed. Nothing to do."
            )
            return 0
        logger.error("No samples downloaded. Exiting.")
        return 1

    total_samples = len(samples)
    logger.info(f"Loaded {total_samples} samples from cache")

    # Filter out already processed samples
    samples_to_process = [
        s for s in samples if s.get("blob_id") not in processed_ids
    ]
    if len(samples_to_process) < len(samples):
        logger.info(
            f"Skipping {len(samples) - len(samples_to_process)} "
            "already processed samples."
        )

    if not samples_to_process:
        logger.info("All samples already processed. Nothing to do.")
        return 0

    # ── Step 3: Create vLLM actors (in waves to reduce I/O contention) ─
    init_batch = min(args.init_batch_size, num_actors)
    logger.info(
        f"Step 3: creating {num_actors} VLLMPipelineActor instances "
        f"in waves of {init_batch} (--init-batch-size)..."
    )

    # Import here so the import doesn't fail on machines without vllm
    from hpc.vllm_worker import VLLMPipelineActor

    actors = []
    for wave_start in range(0, num_actors, init_batch):
        wave_end = min(wave_start + init_batch, num_actors)
        wave_size = wave_end - wave_start
        logger.info(
            f"  Wave {wave_start // init_batch + 1}: "
            f"initializing actors {wave_start + 1}-{wave_end} of {num_actors}..."
        )

        wave_actors = []
        for _ in range(wave_size):
            actor = VLLMPipelineActor.remote(
                model_name=args.model,
                tp_size=args.tensor_parallel,
                max_model_len=args.max_model_len,
                gpu_memory_utilization=args.gpu_memory_utilization,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                local_model_dir=args.local_model_dir,
                prompt_overhead_tokens=args.prompt_overhead_tokens,
            )
            wave_actors.append(actor)

        # Wait for this wave to finish loading before starting the next
        health_refs = [a.health_check.remote() for a in wave_actors]
        health_results = ray.get(health_refs)
        for h in health_results:
            logger.info(f"    {h}")

        actors.extend(wave_actors)
        logger.info(
            f"  Wave {wave_start // init_batch + 1} ready "
            f"({len(actors)}/{num_actors} actors initialized)"
        )

    # ── Step 4: Process in chunks ─────────────────────────────────────
    logger.info(
        f"Step 4: processing {len(samples_to_process)} samples "
        f"in chunks of {CHUNK_SIZE}..."
    )

    graph_builder = GraphBuilder()
    chunk_start_idx = get_next_chunk_index(args.output_dir) if args.resume else 0
    all_errors = 0
    total_processed = 0
    all_run_results = []  # accumulate results in memory for final stats
    start_time = time.time()

    # Build a lookup from blob_id -> sample for graph building
    sample_lookup = {s.get("blob_id"): s for s in samples}

    for chunk_offset in range(0, len(samples_to_process), CHUNK_SIZE):
        chunk = samples_to_process[chunk_offset : chunk_offset + CHUNK_SIZE]
        chunk_idx = chunk_start_idx + (chunk_offset // CHUNK_SIZE)

        logger.info(
            f"Processing chunk {chunk_idx} ({len(chunk)} samples)..."
        )

        # Split chunk across actors in sub-batches
        futures = []
        batch_size = args.batch_size

        for actor_offset in range(0, len(chunk), batch_size):
            batch = chunk[actor_offset : actor_offset + batch_size]
            actor = actors[
                (actor_offset // batch_size) % num_actors
            ]
            futures.append(actor.process_samples.remote(batch))

        # Collect results as they complete
        chunk_results = []

        while futures:
            done, futures = ray.wait(futures, num_returns=1)
            try:
                batch_results = ray.get(done[0])
                chunk_results.extend(batch_results)

                completed = total_processed + len(chunk_results)
                remaining_batches = len(futures)
                elapsed = time.time() - start_time
                rate = completed / elapsed if elapsed > 0 else 0
                logger.info(
                    f"  Batches remaining: {remaining_batches} | "
                    f"Processed so far: {completed} | "
                    f"Rate: {rate:.1f} samples/sec"
                )
            except Exception as e:
                logger.error(f"  Actor batch failed: {e}")
                all_errors += batch_size  # approximate

        # Build graph for this chunk
        for result in chunk_results:
            sample_orig = sample_lookup.get(result["sample_id"], {})
            try:
                build_graph_from_result(graph_builder, result, sample_orig)
            except Exception as e:
                logger.error(
                    f"Error building graph for {result['sample_id']}: {e}"
                )
                all_errors += 1

        total_processed += len(chunk_results)
        all_run_results.extend(chunk_results)

        # Save chunk results
        chunk_stats = collect_stats(chunk_results, total_samples, all_errors)
        chunk_file = os.path.join(
            args.output_dir, f"processing_results_part_{chunk_idx}.arrow"
        )
        save_chunk_results(chunk_results, chunk_stats, chunk_file)

        elapsed = time.time() - start_time
        logger.info(
            f"Chunk {chunk_idx} saved. "
            f"Progress: {total_processed}/{len(samples_to_process)} "
            f"({elapsed:.0f}s elapsed)"
        )

    # ── Step 5: Print statistics (from in-memory results, no disk re-read) ──
    logger.info("Step 5: computing final statistics...")
    final_stats = collect_stats(all_run_results, total_processed, all_errors)
    print_statistics(final_stats)

    graph = graph_builder.get_graph()
    elapsed = time.time() - start_time

    print(f"\nPipeline completed!")
    print(f"  Processed {total_processed} samples in {elapsed:.0f}s")
    print(f"  Graph: {len(graph.nodes())} nodes, {len(graph.edges())} edges")
    print(f"  Output: {args.output_dir}")

    ray.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
