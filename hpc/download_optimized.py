"""
Optimized dataset download using batched S3 downloads.
Replaces sequential download_stack_edu_dataset() with parallel batches.

Two-phase approach:
  1. collect_new_sample_metadata() loads the HF table and filters by blob_id
  2. download_blob_contents() downloads S3 contents for a metadata batch
"""

import uuid
from typing import List, Dict, Set
from tqdm import tqdm
from datasets import load_dataset
import pyarrow as pa
import pyarrow.compute as pc

from utils.utils import download_contents_batch


def collect_new_sample_metadata(
    num_samples: int,
    programming_language: str = "Python",
    skip_blob_ids: Set[str] = None,
) -> List[Dict]:
    """
    Load the Hugging Face table and filter out already-known blob_ids.

    Loads the complete metadata table and filters in bulk.

    Args:
        num_samples: number of new metadata rows to collect
        programming_language: language filter
        skip_blob_ids: blob_ids to skip

    Returns:
        list of metadata dictionaries
    """
    if skip_blob_ids is None:
        skip_blob_ids = set()

    print(
        f"📥 Loading HF dataset table for {programming_language} "
        f"(skipping {len(skip_blob_ids)} known IDs)..."
    )

    dataset = load_dataset(
        "HuggingFaceTB/stack-edu",
        programming_language,
        split="train",
    )

    total_rows = len(dataset)
    print(f"  Dataset loaded: {total_rows} rows. Filtering with PyArrow...")

    # Vectorized filter using PyArrow compute (pure C, no Python loop)
    blob_col = dataset.data.column("blob_id")

    # Build mask: blob_id is valid, non-empty, and NOT in skip set
    is_valid = pc.is_valid(blob_col)
    is_nonempty = pc.not_equal(blob_col, "")
    if skip_blob_ids:
        skip_array = pa.array(list(skip_blob_ids), type=pa.string())
        is_known = pc.is_in(blob_col, value_set=skip_array)
        keep_mask = pc.and_(pc.and_(is_valid, is_nonempty), pc.invert(is_known))
    else:
        keep_mask = pc.and_(is_valid, is_nonempty)

    kept = pc.sum(keep_mask).as_py()
    skipped = total_rows - kept - pc.sum(pc.invert(pc.and_(is_valid, is_nonempty))).as_py()
    print(f"  {skipped} known IDs skipped, {kept} new IDs available")

    # Get indices of rows to keep (up to num_samples)
    keep_indices = pc.indices_nonzero(keep_mask).to_pylist()
    if len(keep_indices) > num_samples:
        keep_indices = keep_indices[:num_samples]

    filtered = dataset.select(keep_indices)
    n = len(keep_indices)
    print(f"  Selected {n} rows. Converting to metadata...")

    # Single bulk extraction: one Arrow->Python conversion per column
    cols = filtered.to_dict()
    blob_ids = cols.get("blob_id", [""] * n)
    hexshas = cols.get("hexsha", [None] * n)
    languages = cols.get("language", [""] * n)
    sizes = cols.get("size", [0] * n)
    repos = cols.get("repository_name", ["unknown"] * n)
    paths = cols.get("path", [""] * n)
    avg_lines = cols.get("avg_line_length", [0] * n)
    max_lines = cols.get("max_line_length", [0] * n)
    alphanums = cols.get("alphanum_fraction", [0] * n)
    locs = cols.get("loc", [0] * n)

    metadata_list = [None] * n
    for i in range(n):
        metadata_list[i] = {
            "blob_id": blob_ids[i],
            "id": hexshas[i] or str(uuid.uuid4()),
            "language": (languages[i] or "").lstrip("."),
            "size": sizes[i] or 0,
            "repository": repos[i] or "unknown",
            "file_path": paths[i] or "",
            "metadata": {
                "avg_line_length": avg_lines[i] or 0,
                "max_line_length": max_lines[i] or 0,
                "alphanum_fraction": alphanums[i] or 0,
                "loc": locs[i] or 0,
            },
        }

    print(f"✅ Collected {len(metadata_list)} new metadata entries")
    return metadata_list


def download_blob_contents(
    metadata_list: List[Dict],
    batch_size: int = 100,
    max_workers: int = 64,
) -> List[Dict]:
    """
    Download S3 blob contents for a list of metadata entries.

    Args:
        metadata_list: Metadata dicts from collect_new_sample_metadata()
        batch_size: Number of blob_ids per S3 download batch
        max_workers: Parallel threads per batch

    Returns:
        List of complete sample dicts (with content)
    """
    all_blob_ids = [m["blob_id"] for m in metadata_list]

    all_contents = {}
    for i in tqdm(
        range(0, len(all_blob_ids), batch_size),
        desc="Downloading batches",
        unit="batch",
    ):
        batch_ids = all_blob_ids[i : i + batch_size]
        batch_contents = download_contents_batch(batch_ids, max_workers=max_workers)
        all_contents.update(batch_contents)

    # Combine metadata with downloaded code
    samples = []
    for meta in metadata_list:
        blob_id = meta["blob_id"]
        content = all_contents.get(blob_id)

        if not content or len(content.strip()) < 50:
            continue

        samples.append({
            "id": meta["id"],
            "content": content,
            "language": meta["language"],
            "size": meta["size"],
            "blob_id": blob_id,
            "repository": meta["repository"],
            "file_path": meta["file_path"],
            "metadata": meta["metadata"],
        })

    return samples
