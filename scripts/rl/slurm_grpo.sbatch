#!/bin/bash
# GRPO baseline (grasp.rl.grpo) on a single H200: the vLLM server and the LoRA
# trainer share the card.
#   sbatch --export=ALL slurm_grpo.sbatch                       # full WQSP run
#   sbatch --export=ALL,SMOKE=1 --time=01:00:00 slurm_grpo.sbatch  # 4 steps on 16 items
#   sbatch --export=ALL,NUM_EPOCHS=0 slurm_grpo.sbatch          # base model eval only
# Submit from a login shell, else https_proxy is missing and wandb runs offline.
# Resuming is automatic: rerun with the same OUT and the trainer continues after
# the last completed step (adapter, optimizer, lr schedule).
#SBATCH --job-name=grpo
#SBATCH --partition=addlc2_gpu-h200
#SBATCH --account=ad-dlc2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=192G
#SBATCH --time=1-00:00:00
#SBATCH --output=/work/dlc2workfs1/swalter-data/grasp-cqd/logs/%x-%j.out

set -euo pipefail

REPO=${REPO:-$HOME/phd/grasp-rl}
DATA=/work/dlc2workfs1/swalter-data/grasp-cqd
CONFIG=${CONFIG:-configs/rl/grpo/wqsp.yaml}

export MODEL=${MODEL:-Qwen/Qwen2.5-7B-Instruct}
TAG=$(basename "$MODEL" | tr '[:upper:]' '[:lower:]' | tr -d '.')
case "$MODEL" in
  *Qwen2.5*)
    SERVE_FLAGS="--enable-auto-tool-choice --tool-call-parser hermes"
    ;;
  *Qwen3.5*|*Qwen3.6*|*Qwen3.8*)
    SERVE_FLAGS="--enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3"
    # multimodal hybrid-attention checkpoint: adapt the text stack only, not
    # the vision tower "all-linear" would also hit
    export LORA_TARGET_MODULES=${LORA_TARGET_MODULES:-'model\.layers\.\d+\.(mlp\.(gate|up|down)_proj|self_attn\.[qkvo]_proj|linear_attn\.(in_proj_(qkv|z|b|a)|out_proj))'}
    ;;
  *)
    echo "unknown MODEL '$MODEL', add its parser flags (and LORA_TARGET_MODULES)"
    exit 1
    ;;
esac

source /work/dlc2workfs1/swalter-env/anaconda/etc/profile.d/conda.sh
conda activate grasp-cqd
cd "$REPO"
export PYTHONPATH=$REPO/src${PYTHONPATH:+:$PYTHONPATH}

export GRASP_INDEX_DIR=$DATA
export HF_HOME=/work/dlc2workfs1/swalter-env/huggingface
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=1
export VLLM_USE_FLASHINFER_SAMPLER=0
# the trainer runs from its own env: torch 2.12 with triton 3.7.1, which fla's
# backward needs on Hopper (Qwen3.5 linear attention); vLLM 0.26 pins torch
# 2.11 and stays in grasp-cqd. tilelang cannot JIT on the nodes (no cc1plus,
# mismatched CUDA headers), so fla uses its triton kernels.
TRAIN_PYTHON=${TRAIN_PYTHON:-/work/dlc2workfs1/swalter-env/anaconda/envs/grasp-rl-train/bin/python}
export FLA_TILELANG=0
export FREEBASE_ENDPOINT=${FREEBASE_ENDPOINT:-http://tagus.informatik.privat:7002}
export WANDB_INIT_TIMEOUT=120
[ -z "${https_proxy:-}" ] && export WANDB_MODE=offline

export TRAIN_FILE=${TRAIN_FILE:-$DATA/pools/wqsp_train_pool.jsonl}
export VAL_FILE=${VAL_FILE:-$DATA/pools/wqsp_val300_pool.jsonl}
export RUN_NAME=${RUN_NAME:-grpo-wqsp-$TAG}
[ "${NUM_EPOCHS:-1}" = "0" ] && RUN_NAME=$RUN_NAME-base
OUT=${OUT:-$DATA/runs-rl/$RUN_NAME}
if [ "${SMOKE:-0}" = "1" ]; then
  RUN_NAME=$RUN_NAME-smoke
  OUT=$DATA/runs-rl/$RUN_NAME-${SLURM_JOB_ID:-manual}
  mkdir -p "$OUT"
  head -n 16 "$TRAIN_FILE" > "$OUT/train.jsonl"
  head -n 8 "$VAL_FILE" > "$OUT/val.jsonl"
  export TRAIN_FILE=$OUT/train.jsonl VAL_FILE=$OUT/val.jsonl
  export BATCH_SIZE=4 GROUP_SIZE=4 VAL_INTERVAL=2 SAVE_INTERVAL=2 WANDB_PROJECT=grasp-rl-smoke
fi
export RUN_NAME
export WANDB_DIR=$OUT
mkdir -p "$OUT" "$DATA/logs"
echo "model=$MODEL out=$OUT run=$RUN_NAME config=$CONFIG"

# a free port per job, two jobs on one node must not share a server
PORT=$(python -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])')
export VLLM_URL="http://localhost:$PORT"

vllm serve "$MODEL" \
  --port "$PORT" \
  --max-model-len 32768 \
  $SERVE_FLAGS \
  --enable-lora --max-loras 2 --max-lora-rank 32 \
  --gpu-memory-utilization "${GPU_MEM_UTIL:-0.45}" \
  > "$DATA/logs/vllm-${SLURM_JOB_ID:-manual}.log" 2>&1 &
VLLM_PID=$!
trap 'kill $VLLM_PID 2>/dev/null || true' EXIT

# wait for OUR model, not just any server answering on the port
served=""
for _ in $(seq 1 160); do
  served=$(curl -sf --noproxy '*' "$VLLM_URL/v1/models" 2>/dev/null \
    | python -c 'import json,sys; print(" ".join(m["id"] for m in json.load(sys.stdin)["data"]))' \
    2>/dev/null || true)
  case " $served " in *" $MODEL "*) break ;; esac
  kill -0 $VLLM_PID 2>/dev/null || { echo "vllm exited early, see logs/vllm-${SLURM_JOB_ID:-manual}.log"; exit 1; }
  sleep 15
done
case " $served " in
  *" $MODEL "*) echo "vllm ready on $PORT" ;;
  *) echo "vllm on $PORT serves '$served', expected '$MODEL'"; exit 1 ;;
esac

"$TRAIN_PYTHON" -m grasp.rl.grpo "$CONFIG" "$OUT"
