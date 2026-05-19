import uuid
import gzip
from datasets import load_dataset
from botocore.exceptions import ClientError
import boto3
from botocore import UNSIGNED
from botocore.config import Config
from tqdm import tqdm
import concurrent.futures
from typing import List, Dict, Optional

# Initialize S3 client
s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED, max_pool_connections=150))
bucket_name = "softwareheritage"

def download_contents(blob_id):
    """Stream content directly without storing in dataset"""
    key = f"content/{blob_id}"
    try:
        obj = s3.get_object(Bucket=bucket_name, Key=key)
        with gzip.GzipFile(fileobj=obj['Body']) as fin:
            return fin.read().decode("utf-8", errors="ignore")
    except ClientError as e:
        if e.response['Error']['Code'] == 'NoSuchKey':
            return None
        raise

def download_contents_batch(blob_ids: List[str], max_workers: int = 8) -> Dict[str, Optional[str]]:
    """
    Download multiple contents in parallel from S3
    
    Args:
        blob_ids: List of blob IDs to download
        max_workers: Maximum number of parallel downloads
        
    Returns:
        Dictionary mapping blob_ids to their content (or None if not found)
    """
    results = {}

    def download_single(blob_id):
        key = f"content/{blob_id}"
        try:
            obj = s3.get_object(Bucket=bucket_name, Key=key)
            with gzip.GzipFile(fileobj=obj['Body']) as fin:
                return blob_id, fin.read().decode("utf-8", errors="ignore")
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                return blob_id, None
            raise

    # Use ThreadPoolExecutor for I/O-bound operations
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all download tasks
        future_to_blob_id = {executor.submit(download_single, blob_id): blob_id for blob_id in blob_ids}

        # Process results as they complete
        for future in concurrent.futures.as_completed(future_to_blob_id):
            blob_id, content = future.result()
            results[blob_id] = content

    return results


def download_stack_edu_dataset(num_samples: int = 1000, programming_language: str = 'Python') -> List[Dict]:
    """
    Download samples from the HuggingFaceTB/stack-edu dataset
    
    Args:
        num_samples: Number of samples to download
        split: Dataset split to use ('train', 'test', 'validation')
        programming_language: Filter by programming language (e.g., 'Python', 'Java')
        
    Returns:
        List of code samples with metadata
    """
    print(f"📥 Loading {num_samples} samples from HuggingFaceTB/stack-edu dataset...")
    
    try:
        # Load the dataset
        dataset = load_dataset("HuggingFaceTB/stack-edu", programming_language)
        
        samples = []
        count = 0
        
        print("🔍 Processing samples...")
        for sample in tqdm(dataset["train"], desc="Downloading samples", total=num_samples):

            content = download_stack_edu_with_content([sample.get('blob_id', '')])
            if len(content) == 0:
                continue
            # Extract relevant information
            code_sample = {
                'id': sample.get('hexsha', str(uuid.uuid4())),
                'content': content[0]["content"],
                'language': sample.get('language', '').lstrip('.'),
                'size': sample.get('size', 0),
                'blob_id': sample.get('blob_id', ''),
                'repository': sample.get('repository_name', 'unknown'),
                'file_path': sample.get('path', ''),
                'metadata': {
                    'avg_line_length': sample.get('avg_line_length', 0),
                    'max_line_length': sample.get('max_line_length', 0),
                    'alphanum_fraction': sample.get('alphanum_fraction', 0),
                    'loc': sample.get('loc', 0),  # lines of code
                }
            }

            # Skip empty or very small files
            if len(code_sample["content"].strip()) < 50:
                continue
                
            samples.append(code_sample)
            count += 1
            
            if count >= num_samples:
                break
        
        print(f"✅ Successfully downloaded {len(samples)} code samples")
        return samples
        
    except Exception as e:
        print(f"❌ Error downloading dataset: {e}")
        return []


def download_stack_edu_with_content(blob_ids: List[str], max_workers: int = 8) -> List[Dict]:
    """
    Download code samples from stack-edu dataset and fetch their content from S3
    
    Args:
        blob_ids: List of blob IDs to download content for
        max_workers: Maximum number of parallel downloads
        
    Returns:
        List of code samples with full content
    """
    print(f"📥 Downloading content for {len(blob_ids)} blob IDs...")
    
    # Download contents in parallel
    contents = download_contents_batch(blob_ids, max_workers)
    
    # Combine with metadata
    samples = []
    for blob_id in tqdm(blob_ids, desc="Processing samples"):
        content = contents.get(blob_id)
        if content and len(content.strip()) > 50:  # Skip empty or very small files
            samples.append({
                'id': blob_id,
                'content': content,
                'blob_id': blob_id,
            })
    
    print(f"✅ Successfully processed {len(samples)} samples with content")
    return samples


def filter_code_by_language(samples: List[Dict], languages: List[str]) -> List[Dict]:
    """
    Filter code samples by programming language
    
    Args:
        samples: List of code samples
        languages: List of programming languages to keep (e.g., ['py', 'java', 'js'])
        
    Returns:
        Filtered list of code samples
    """
    if not languages:
        return samples
    
    filtered = []
    for sample in samples:
        lang = sample.get('language', '').lower()
        if lang in [l.lower() for l in languages]:
            filtered.append(sample)
    
    print(f"🔍 Filtered {len(filtered)} samples for languages: {languages}")
    return filtered


def save_samples_to_file(samples: List[Dict], filepath: str):
    """
    Save code samples to a JSON file (atomic write to prevent corruption).

    Writes to a temporary file first, then renames. This way if the process
    is killed during write, the original file stays intact.

    Args:
        samples: List of code samples
        filepath: Path to save the file
    """
    import json
    import os

    tmp_path = filepath + ".tmp"
    with open(tmp_path, 'w', encoding='utf-8') as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)

    # Atomic rename: replaces the old file only after the new one is fully written
    os.replace(tmp_path, filepath)

    print(f"💾 Saved {len(samples)} samples to {filepath}")


def load_samples_from_file(filepath: str) -> List[Dict]:
    """
    Load code samples from a JSON file

    Args:
        filepath: Path to the JSON file

    Returns:
        List of code samples
    """
    import json

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            samples = json.load(f)
        print(f"📂 Loaded {len(samples)} samples from {filepath}")
        return samples
    except FileNotFoundError:
        print(f"❌ File not found: {filepath}")
        return []


def save_samples_arrow(samples: List[Dict], filepath: str) -> None:
    """
    Save code samples to an Arrow (Feather) file with atomic write.

    The `metadata` field (nested dict) is serialized as a JSON string column.
    """
    import json
    import os
    import pyarrow as pa
    import pyarrow.feather as feather

    if not samples:
        return

    # Flatten: serialize 'metadata' dict as JSON string
    rows = []
    for s in samples:
        row = dict(s)
        if "metadata" in row and isinstance(row["metadata"], dict):
            row["metadata"] = json.dumps(row["metadata"], ensure_ascii=False)
        rows.append(row)

    # Build columns from keys present in the first row
    keys = list(rows[0].keys())
    columns = {k: [r.get(k) for r in rows] for k in keys}
    table = pa.table(columns)

    tmp_path = filepath + ".tmp"
    feather.write_feather(table, tmp_path)
    os.replace(tmp_path, filepath)
    print(f"💾 Saved {len(samples)} samples to {filepath}")


def load_samples_arrow(filepath: str) -> List[Dict]:
    """
    Load code samples from an Arrow (Feather) file.

    Deserializes the `metadata` JSON string column back to a dict.
    """
    import json
    import pyarrow.feather as feather

    try:
        table = feather.read_table(filepath)
    except FileNotFoundError:
        print(f"❌ File not found: {filepath}")
        return []

    rows = table.to_pydict()
    num_rows = table.num_rows
    samples = []
    keys = list(rows.keys())
    for i in range(num_rows):
        row = {k: rows[k][i] for k in keys}
        if "metadata" in row and isinstance(row["metadata"], str):
            try:
                row["metadata"] = json.loads(row["metadata"])
            except (json.JSONDecodeError, TypeError):
                pass
        samples.append(row)

    print(f"📂 Loaded {len(samples)} samples from {filepath}")
    return samples


def load_samples_auto(filepath: str) -> List[Dict]:
    """Load samples from either JSON or Arrow file based on extension."""
    if filepath.endswith(".arrow"):
        return load_samples_arrow(filepath)
    return load_samples_from_file(filepath)