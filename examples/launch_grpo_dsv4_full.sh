#!/usr/bin/env bash
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Launch full-scale DeepSeek-V4-Flash-Base GRPO on Slurm.
# Wraps the canonical multinode template (ray.sub) with the env vars
# and command that the DSv4 recipe needs.
#
# Required environment variables (set before invoking):
#   CONTAINER          Path to the NeMo-RL sqsh image (or container ref).
#   ACCOUNT            Slurm account.
#   PARTITION          Slurm partition.
#   EXP_NAME           Run name; used for results/logs/checkpoint dirs.
#   HF_TOKEN           Hugging Face token (for tokenizer + dataset access).
#   MOUNTS             Comma-sep srun --container-mounts list.
#                      Must include a results/cache mount visible to all nodes.
#
# Optional environment variables:
#   CONFIG             Recipe YAML (default: 24-node H100 config below).
#   NUM_NODES          Total Slurm nodes (default: matches CONFIG; 24).
#   WALL_TIME          Slurm --time (default: 12:00:00).
#   WANDB_API_KEY      If set, wandb logger is enabled in the run.
#   HF_HOME            Hugging Face cache root inside the container.
#   EXTRA_HYDRA_ARGS   Extra "++a.b=c" overrides appended to the Hydra call.
#
# Example:
#   CONTAINER=/lustre/fsw/portfolios/coreai/users/$USER/nemo-rl.sqsh \
#   ACCOUNT=coreai_dlalgo_nemo \
#   PARTITION=batch \
#   EXP_NAME=dsv4-flash-grpo-24n-$(date +%Y%m%d) \
#   HF_TOKEN=hf_xxx \
#   MOUNTS=/lustre/fsw/portfolios/coreai/users/$USER/data:/results \
#   bash examples/launch_grpo_dsv4_full.sh

set -euo pipefail

REPO_LOCATION=${REPO_LOCATION:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}
cd "$REPO_LOCATION"

: "${CONTAINER:?Set CONTAINER (sqsh image path).}"
: "${ACCOUNT:?Set ACCOUNT (Slurm account).}"
: "${PARTITION:?Set PARTITION (Slurm partition).}"
: "${EXP_NAME:?Set EXP_NAME (used for results / log dirs).}"
: "${MOUNTS:?Set MOUNTS (srun --container-mounts list).}"
: "${HF_TOKEN:?Set HF_TOKEN (for tokenizer / dataset).}"

CONFIG=${CONFIG:-examples/configs/grpo_dsv4_base_bf16_4k_megatron_24n.yaml}
NUM_NODES=${NUM_NODES:-24}
WALL_TIME=${WALL_TIME:-12:00:00}
WANDB_API_KEY=${WANDB_API_KEY:-}
HF_HOME=${HF_HOME:-$REPO_LOCATION/.cache/hf}
EXTRA_HYDRA_ARGS=${EXTRA_HYDRA_ARGS:-}

# Compose Hydra overrides:
#  - cluster.num_nodes follows NUM_NODES (so resizing only needs one knob)
#  - results / log_dir / checkpoint_dir all key off EXP_NAME
WANDB_OVERRIDES=""
if [[ -n "$WANDB_API_KEY" ]]; then
  WANDB_OVERRIDES="++logger.wandb_enabled=true ++logger.wandb.name=$EXP_NAME ++logger.wandb.project=grpo-dsv4-flash"
fi

read -r -d '' COMMAND <<EOF || true
set -eou pipefail
cd $REPO_LOCATION

# vLLM Base FP8 monkey-patch must be applied inside the container on each
# node *before* the worker venvs spin up. The script is idempotent.
bash tools/patch_vllm_dsv4_base_fp8_quick.sh || true

export HF_HOME=$HF_HOME
export HF_TOKEN=$HF_TOKEN
export WANDB_API_KEY=${WANDB_API_KEY}
export VLLM_DSV4_BASE_FP8=1
export NRL_SWIGLU_LIMIT=10
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False

uv run --no-sync python examples/run_grpo.py \\
    --config $CONFIG \\
    ++cluster.num_nodes=$NUM_NODES \\
    ++logger.log_dir=results/$EXP_NAME \\
    ++checkpointing.checkpoint_dir=results/$EXP_NAME/ckpt \\
    $WANDB_OVERRIDES \\
    $EXTRA_HYDRA_ARGS
EOF

echo "=========================================================="
echo "Submitting GRPO full-scale run"
echo "  EXP_NAME    : $EXP_NAME"
echo "  CONFIG      : $CONFIG"
echo "  NUM_NODES   : $NUM_NODES"
echo "  CONTAINER   : $CONTAINER"
echo "  MOUNTS      : $MOUNTS"
echo "  PARTITION   : $PARTITION (account: $ACCOUNT)"
echo "  WALL_TIME   : $WALL_TIME"
echo "  WANDB       : $([[ -n $WANDB_API_KEY ]] && echo "enabled" || echo "disabled")"
echo "=========================================================="

COMMAND="$COMMAND" \
CONTAINER="$CONTAINER" \
MOUNTS="$MOUNTS" \
sbatch \
    --nodes="$NUM_NODES" \
    --account="$ACCOUNT" \
    --partition="$PARTITION" \
    --time="$WALL_TIME" \
    --job-name="$EXP_NAME" \
    --gres=gpu:8 \
    ray.sub
