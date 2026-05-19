# Distributed Extraction Pipeline

This directory contains the minimal code for distributed extraction on a SLURM
GPU cluster.

## Files

```text
hpc/
├── __init__.py
├── pipeline_distributed.py  # Ray orchestration, dataset cache, Arrow output
├── vllm_worker.py           # Ray actor that runs vLLM inference
├── download_optimized.py    # Helpers for dataset metadata and contents
├── run_slurm.sh             # SLURM entry point
├── singularity.def          # Optional container recipe
└── README.md
```

Generated files, model caches, SLURM logs, and downloaded source-code samples
belong under `$SCRATCH` or ignored local directories, not in git.

## Usage

Set the project name and submit the SLURM job:

```bash
export PROJECT_NAME=CodeGraph
PROJECT_NAME=$PROJECT_NAME \
MODEL_NAME=Qwen/Qwen3-Coder-30B-A3B-Instruct \
NUM_SAMPLES=150000 \
LANGUAGE=Python \
sbatch hpc/run_slurm.sh
```

The job starts a Ray cluster, launches one vLLM worker per allocated node, reads
or downloads Stack-Edu samples into the cache, and writes Arrow chunks under the
configured output directory.

Main environment variables:

```text
PROJECT_NAME     Project directory name under $WORK and $SCRATCH
MODEL_NAME       Hugging Face model served by vLLM
NUM_SAMPLES      Number of source files to process
LANGUAGE         Optional Stack-Edu language filter
```

Before running, adapt the SLURM account, partition, and resource settings in
`run_slurm.sh` to the target cluster.
