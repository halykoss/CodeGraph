# CodeGraph Reproducibility Package

CodeGraph builds a semantic dataset from source files. The current pipeline has
four stages:

1. download and annotate files from Stack-Edu;
2. extract normalized concept vocabularies;
3. ground concepts in Wikidata and retrieve Wikidata hierarchy edges;
4. build the final Parquet dataset.

Generated corpora, model caches, Wikidata caches, logs, and final datasets are
not tracked in this repository.

## Repository Layout

```text
.
├── main.py                         # Small local run
├── hpc/run_slurm.sh                # Generic distributed SLURM run
├── hpc/pipeline_distributed.py     # Ray + vLLM extraction, writes Arrow shards
├── scripts/extract_concept_lists.py
├── analyzer/wikidata_grounding.py
├── scripts/run_wikidata_grounding.sh
├── scripts/fetch_wikidata_parents.py
├── scripts/build_hierarchy.py
├── scripts/build_dataset.py
├── src/                            # LLM extraction and graph components
└── tests/
```

## Requirements

- Python 3.11+
- packages in `requirements.txt`
- `pyarrow` and `duckdb` for dataset construction
- an LLM backend for extraction and Wikidata disambiguation
- a SLURM GPU cluster, or an equivalent environment, for full-scale runs

Install the base environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Runtime variables can be copied from `.env.example`. Do not commit populated
`.env` files.

## End-To-End Status

The pipeline is connected end to end:

```text
Stack-Edu
  -> output/full_pipeline/processing_results_part_*.arrow or *.json
  -> output/<axis>/<axis>_list.txt
  -> cache/wikidata_<axis>/*.json
  -> hierarchy_wikidata/wikidata_parents.json
  -> hierarchy_wikidata/wikidata_hierarchy.json
  -> codegraph_release_v1/
```

The recommended full-scale path is the distributed path because it produces
Arrow shards. The local `main.py` path produces JSON shards; `build_dataset.py`
can read them as well, so it remains useful for small validation runs.

## 1. Extract Semantic Annotations

### Local Smoke Run

Use this command for a functional check:

```bash
python main.py \
  --samples 100 \
  --language Python \
  --output-dir output/full_pipeline \
  --cache-dir cache
```

Output:

```text
output/full_pipeline/processing_results_part_*.json
```

### Distributed Run

`hpc/run_slurm.sh` launches `hpc.pipeline_distributed` and writes
`processing_results_part_*.arrow`.

```bash
export PROJECT_NAME=CodeGraph
export MODEL_NAME=Qwen/Qwen3-Coder-30B-A3B-Instruct
export NUM_SAMPLES=150000
export LANGUAGE=Python

sbatch hpc/run_slurm.sh
```

`NUM_SAMPLES` is the number of files requested for the run. There is no
automatic “all of Stack-Edu” mode: set this value to the corpus size you intend
to process. Cluster configuration is described in [hpc/README.md](hpc/README.md).

## 2. Extract Concept Lists

Extract normalized, deduplicated lists from the annotation shards:

```bash
python scripts/extract_concept_lists.py \
  --input-dir output/full_pipeline \
  --output-root output
```

Output:

```text
output/domains/domains_list.txt
output/algorithms/algorithms_list.txt
output/paradigms/paradigms_list.txt
output/design_patterns/design_patterns_list.txt
```

Optional domain canonicalization is still supported and is used by
`build_dataset.py` when `output/domains/stem_results.json` exists:

```bash
bash scripts/run_stem_domains.sh --file output/full_pipeline/
```

If the file does not exist, dataset construction continues with a warning and
uses normalized names directly.

## 3. Ground Concepts In Wikidata

Configure one backend. For OpenRouter:

```bash
export BACKEND=openrouter
export OPENROUTER_API_KEY=sk-or-...
export MODEL=google/gemini-3-flash-preview
```

For a local OpenAI-compatible vLLM server:

```bash
export BACKEND=vllm
export VLLM_URL=http://localhost:8000/v1
export MODEL=/path/or/model/name
```

Run grounding for the four axes:

```bash
bash scripts/run_wikidata_grounding.sh \
  --concept-type domains \
  --input-file output/domains/domains_list.txt

bash scripts/run_wikidata_grounding.sh \
  --concept-type algorithms \
  --input-file output/algorithms/algorithms_list.txt

bash scripts/run_wikidata_grounding.sh \
  --concept-type paradigms \
  --input-file output/paradigms/paradigms_list.txt

bash scripts/run_wikidata_grounding.sh \
  --concept-type design-patterns \
  --input-file output/design_patterns/design_patterns_list.txt
```

Per-concept caches are written to:

```text
cache/wikidata_domains/
cache/wikidata_algorithms/
cache/wikidata_paradigms/
cache/wikidata_design_patterns/
```

These caches are the source consumed by `scripts/build_dataset.py`. Aggregate
files under `output/wikidata/` are useful run summaries.

For a short check:

```bash
bash scripts/run_wikidata_grounding.sh \
  --concept-type algorithms \
  --input-file output/algorithms/algorithms_list.txt \
  --test
```

## 4. Build The Wikidata Hierarchy

Parent fetching reads QIDs from the grounding caches. Run from the repository
root:

```bash
mkdir -p hierarchy_wikidata
cd hierarchy_wikidata
python ../scripts/fetch_wikidata_parents.py
python ../scripts/build_hierarchy.py
cd ..
```

Output:

```text
hierarchy_wikidata/wikidata_parents.json
hierarchy_wikidata/wikidata_hierarchy.json
```

`build_dataset.py` can use both files. If both exist, it prefers
`wikidata_hierarchy.json`.

## 5. Build The Final Dataset

Build the final Parquet tree:

```bash
python scripts/build_dataset.py \
  --arrow-dir output/full_pipeline \
  --cache-dir cache \
  --hierarchy-dir hierarchy_wikidata \
  --output-dir codegraph_release_v1 \
  --duckdb-temp-dir duckdb_tmp
```

Expected output:

```text
codegraph_release_v1/
├── files/
├── edges_file_domain/
├── edges_file_algorithm/
├── edges_file_paradigm/
├── edges_file_design_pattern/
├── concepts_algorithms.parquet
├── concepts_algorithm_categories.parquet
├── concepts_algorithm_complexities.parquet
├── concepts_domains.parquet
├── concepts_paradigms.parquet
├── concepts_design_patterns.parquet
├── edges_algorithm_category.parquet
├── edges_algorithm_complexity.parquet
├── wikidata_entities.parquet
├── wikidata_parent_of.parquet
└── manifest.json
```

`files/` and file-concept edges are partitioned by language. Concept and
Wikidata tables are global Parquet files.

## Quick Checklist

Before running `build_dataset.py`, these should exist:

```text
output/full_pipeline/processing_results_part_*.arrow
# or, for local checks:
output/full_pipeline/processing_results_part_*.json

cache/wikidata_domains/*.json
cache/wikidata_algorithms/*.json
cache/wikidata_paradigms/*.json
cache/wikidata_design_patterns/*.json

hierarchy_wikidata/wikidata_hierarchy.json
```

`hierarchy_wikidata/wikidata_hierarchy.json` is optional only in the strict
sense: without it, `wikidata_parent_of.parquet` will be empty or based only on
`wikidata_parents.json`. Generate it for a complete release.

## Tests

Run the tests:

```bash
python -m pytest tests
```

The tests do not require the generated corpus, LLM servers, Wikidata network
calls, or cluster resources.

## Generated Data

These paths are generated and should stay out of git:

```text
output/
cache/
logs/
hierarchy_wikidata/
duckdb_tmp/
codegraph_release_v1/
```

For a public release, publish generated datasets separately from the source
code.
