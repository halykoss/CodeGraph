#!/bin/bash
#SBATCH --job-name=code-graph
#SBATCH --partition=<YOUR_PARTITION>
#SBATCH --qos=normal
#SBATCH --nodes=64
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --account=<YOUR_ACCOUNT>
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# ═══════════════════════════════════════════════════════════════════════
# Distributed semantic graph construction.
# Uses Ray + vLLM across the allocated SLURM nodes.
# ═══════════════════════════════════════════════════════════════════════

set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────
PROJECT_NAME="${PROJECT_NAME:-CodeGraph}"  # Project name

# Filesystem layout.
# - $WORK: code, typically quota-limited
# - $SCRATCH: temporary data and large outputs
CODE_DIR="${WORK}/${PROJECT_NAME}"
DATA_DIR="${SCRATCH}/${PROJECT_NAME}"

# Model and processing configuration
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-Coder-30B-A3B-Instruct}"
NUM_SAMPLES="${NUM_SAMPLES:-150000}"
LANGUAGE="${LANGUAGE:-}"  # Empty = process all cached languages
PROMPT_OVERHEAD_TOKENS="${PROMPT_OVERHEAD_TOKENS:-500}"

# Modules to load
CUDA_MODULE="${CUDA_MODULE:-cuda/12.6}"

# Singularity container path. Build it before submitting the job.
CONTAINER_PATH="${CONTAINER_PATH:-${CODE_DIR}/hpc/containers/graph-codebase.sif}"

# Data directories
OUTPUT_DIR="${DATA_DIR}/output/full_pipeline"
CACHE_DIR="${DATA_DIR}/cache"
LOGS_DIR="${DATA_DIR}/logs"

# Node-local ramdisk path for model pre-copy.
LOCAL_MODEL_DIR="/dev/shm/hf_model"

# Hugging Face cache
export HF_HOME="${DATA_DIR}/.hf_cache"

# Do not set HF_HUB_OFFLINE=1 here: it can block cache resolution.
export VLLM_TARGET_DEVICE=cuda

# Redirect caches outside $HOME.
export XDG_CACHE_HOME="${FAST}/.cache"

# Ray temporary directory. Keep this path short.
RAY_TMPDIR="/tmp/ray_${SLURM_JOB_ID}"
export RAY_TMPDIR="${RAY_TMPDIR}"
export TMPDIR="${RAY_TMPDIR}"

# Ray ports
RAY_PORT=6379
RAY_DASHBOARD_PORT=8265

# ── Setup ─────────────────────────────────────────────────────────────
if [[ ! -d "${CODE_DIR}" ]]; then
    echo "ERROR: code directory not found: ${CODE_DIR}"
    echo "Clone or copy the code to \$WORK/${PROJECT_NAME}"
    exit 1
fi

mkdir -p "${LOGS_DIR}" "${OUTPUT_DIR}" "${CACHE_DIR}" "${HF_HOME}" "${XDG_CACHE_HOME}/vllm"

# Remove possibly corrupted vLLM compile caches.
rm -rf "${XDG_CACHE_HOME}/vllm/torch_compile_cache"
rm -rf "${HOME}/.cache/vllm/torch_compile_cache" 2>/dev/null

mkdir -p "${RAY_TMPDIR}"

# ── Validate container ────────────────────────────────────────────────
if [[ ! -f "${CONTAINER_PATH}" ]]; then
    echo "[$(date)] ERROR: Singularity container not found at ${CONTAINER_PATH}"
    echo "Build the container first using singularity.def"
    exit 1
fi

echo "[$(date)] Singularity container: ${CONTAINER_PATH}"

# ── Load modules ──────────────────────────────────────────────────────
echo "[$(date)] Loading CUDA module: ${CUDA_MODULE}..."
module load ${CUDA_MODULE}

# Resolve node list.
NODELIST=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
HEAD_NODE=$(echo "$NODELIST" | head -n 1)
HEAD_ADDR=$(srun --nodes=1 --ntasks=1 -w "$HEAD_NODE" hostname -i | head -n 1)

echo "══════════════════════════════════════════════════"
echo "Job ID:       $SLURM_JOB_ID"
echo "Project:      $PROJECT_NAME"
echo "Nodes:        $SLURM_JOB_NUM_NODES"
echo "Head node:    $HEAD_NODE ($HEAD_ADDR)"
echo "──────────────────────────────────────────────────"
echo "Code:         $CODE_DIR (WORK)"
echo "Data:         $DATA_DIR (SCRATCH)"
echo "Output:       $OUTPUT_DIR"
echo "Cache:        $CACHE_DIR"
echo "HF cache:     $HF_HOME"
echo "XDG cache:    $XDG_CACHE_HOME"
echo "Container:    $CONTAINER_PATH"
echo "──────────────────────────────────────────────────"
echo "Model:        $MODEL_NAME"
echo "Samples:      $NUM_SAMPLES"
echo "Language:     ${LANGUAGE:-all}"
echo "CUDA:         ${CUDA_MODULE}"
echo "Extra tokens: $PROMPT_OVERHEAD_TOKENS"
echo "══════════════════════════════════════════════════"

# ── Step 1: start Ray head node ──────────────────────────────────────
echo "[$(date)] Starting Ray head on $HEAD_NODE ..."

srun --overlap --nodes=1 --ntasks=1 -w "$HEAD_NODE" bash -c "
    mkdir -p ${RAY_TMPDIR}
    module load ${CUDA_MODULE}
    singularity exec --nv \
        --bind ${WORK}:${WORK},${SCRATCH}:${SCRATCH},${RAY_TMPDIR}:${RAY_TMPDIR},${FAST}:${FAST} \
        --env HF_HOME=${HF_HOME} \
        --env XDG_CACHE_HOME=${XDG_CACHE_HOME} \
        --env VLLM_TARGET_DEVICE=cuda \
        --env LD_PRELOAD= \
        --env RAY_TMPDIR=${RAY_TMPDIR} \
        --env TMPDIR=${RAY_TMPDIR} \
        ${CONTAINER_PATH} \
        ray start --head \
            --node-ip-address=${HEAD_ADDR} \
            --port=${RAY_PORT} \
            --dashboard-port=${RAY_DASHBOARD_PORT} \
            --num-cpus=32 \
            --num-gpus=4 \
            --block
" &

sleep 45

# ── Step 2: start Ray workers on remaining nodes ─────────────────────
echo "[$(date)] Starting Ray workers on ${SLURM_JOB_NUM_NODES} nodes ..."

WORKER_NODES=$(echo "$NODELIST" | tail -n +2)

for NODE in $WORKER_NODES; do
    srun --overlap --nodes=1 --ntasks=1 -w "$NODE" bash -c "
        mkdir -p ${RAY_TMPDIR}
        rm -rf ${XDG_CACHE_HOME}/vllm/torch_compile_cache
        module load ${CUDA_MODULE}
        NODE_IP=\$(hostname -i | head -n 1)
        singularity exec --nv \
            --bind ${WORK}:${WORK},${SCRATCH}:${SCRATCH},${RAY_TMPDIR}:${RAY_TMPDIR},${FAST}:${FAST} \
            --env HF_HOME=${HF_HOME} \
            --env XDG_CACHE_HOME=${XDG_CACHE_HOME} \
            --env HF_HUB_OFFLINE=1 \
            --env TRANSFORMERS_OFFLINE=1 \
            --env VLLM_TARGET_DEVICE=cuda \
            --env LD_PRELOAD= \
            --env RAY_TMPDIR=${RAY_TMPDIR} \
            --env TMPDIR=${RAY_TMPDIR} \
            ${CONTAINER_PATH} \
            ray start \
                --address='${HEAD_ADDR}:${RAY_PORT}' \
                --node-ip-address=\${NODE_IP} \
                --num-cpus=32 \
                --num-gpus=4 \
                --block
    " &
done

echo "[$(date)] Waiting for workers to join the cluster ..."
sleep 60

# Verify the cluster.
srun --overlap --nodes=1 --ntasks=1 -w "$HEAD_NODE" bash -c "
    module load ${CUDA_MODULE}
    singularity exec --nv \
        --bind ${WORK}:${WORK},${SCRATCH}:${SCRATCH},${RAY_TMPDIR}:${RAY_TMPDIR},${FAST}:${FAST} \
        --env RAY_ADDRESS=${HEAD_ADDR}:${RAY_PORT} \
        --env XDG_CACHE_HOME=${XDG_CACHE_HOME} \
        --env LD_PRELOAD= \
        ${CONTAINER_PATH} \
        python3 -c '
import ray
ray.init(address=\"auto\")
res = ray.cluster_resources()
cpu_count = int(res.get(\"CPU\", 0))
gpu_count = int(res.get(\"GPU\", 0))
print(f\"Cluster ready: {cpu_count} CPU, {gpu_count} GPU\")
ray.shutdown()
'
"

# ── Step 3: copy model to node-local ramdisk ─────────────────────────
echo "[$(date)] Resolving model snapshot path on head node..."

MODEL_SNAPSHOT_DIR=$(srun --overlap --nodes=1 --ntasks=1 -w "$HEAD_NODE" bash -c "
    module load ${CUDA_MODULE}
    singularity exec \
        --bind ${WORK}:${WORK},${SCRATCH}:${SCRATCH} \
        --env HF_HOME=${HF_HOME} \
        --env LD_PRELOAD= \
        ${CONTAINER_PATH} \
        python3 -c '
from huggingface_hub import snapshot_download
import os
print(snapshot_download(\"${MODEL_NAME}\", local_files_only=True, cache_dir=os.environ.get(\"HF_HOME\")))
'
" | tail -1 | tr -d '[:space:]')

echo "[$(date)] Model snapshot: ${MODEL_SNAPSHOT_DIR}"
echo "[$(date)] Copying model to ${LOCAL_MODEL_DIR} on ${SLURM_JOB_NUM_NODES} nodes..."

srun --overlap --ntasks-per-node=1 bash -c "
    mkdir -p ${LOCAL_MODEL_DIR}
    cp -rL ${MODEL_SNAPSHOT_DIR}/* ${LOCAL_MODEL_DIR}/
    echo \"  [\$(hostname)] Model copied to ${LOCAL_MODEL_DIR} (\$(du -sh ${LOCAL_MODEL_DIR} | cut -f1))\"
"

echo "[$(date)] Model copy complete."

# ── Step 4: run the distributed pipeline ─────────────────────────────
echo "[$(date)] Running distributed pipeline ..."

srun --overlap --nodes=1 --ntasks=1 -w "$HEAD_NODE" bash -c "
    module load ${CUDA_MODULE}
    singularity exec --nv \
        --bind ${WORK}:${WORK},${SCRATCH}:${SCRATCH},${RAY_TMPDIR}:${RAY_TMPDIR},${FAST}:${FAST},${LOCAL_MODEL_DIR}:${LOCAL_MODEL_DIR} \
        --env RAY_ADDRESS=${HEAD_ADDR}:${RAY_PORT} \
        --env HF_HOME=${HF_HOME} \
        --env XDG_CACHE_HOME=${XDG_CACHE_HOME} \
        --env VLLM_TARGET_DEVICE=cuda \
        --env LD_PRELOAD= \
        --env HUGGING_FACE_HUB_TOKEN=${HUGGING_FACE_HUB_TOKEN:-} \
        --pwd ${CODE_DIR} \
        ${CONTAINER_PATH} \
        python3 -m hpc.pipeline_distributed \
            --samples ${NUM_SAMPLES} \
            ${LANGUAGE:+--language ${LANGUAGE}} \
            --model ${MODEL_NAME} \
            --output-dir ${OUTPUT_DIR} \
            --cache-dir ${CACHE_DIR} \
            --local-model-dir ${LOCAL_MODEL_DIR} \
            --prompt-overhead-tokens ${PROMPT_OVERHEAD_TOKENS} \
            --resume
"

EXIT_CODE=$?

# ── Step 5: clean up Ray cluster and ramdisk ─────────────────────────
echo "[$(date)] Stopping Ray cluster and cleaning up ramdisk ..."

for NODE in $(echo "$NODELIST"); do
    srun --overlap --nodes=1 --ntasks=1 -w "$NODE" bash -c "
        singularity exec ${CONTAINER_PATH} ray stop
        rm -rf ${LOCAL_MODEL_DIR}
    " 2>/dev/null &
done
wait

# ── Step 6: move logs to data directory ──────────────────────────────
echo "[$(date)] Moving logs to ${LOGS_DIR} ..."
mv "${SLURM_SUBMIT_DIR}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out" "${LOGS_DIR}/" 2>/dev/null || true
mv "${SLURM_SUBMIT_DIR}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.err" "${LOGS_DIR}/" 2>/dev/null || true

echo "[$(date)] Job finished with exit code: $EXIT_CODE"
echo "Results saved in: ${OUTPUT_DIR}"
echo "Logs saved in: ${LOGS_DIR}"
exit $EXIT_CODE
