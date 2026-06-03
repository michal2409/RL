#!/usr/bin/env bash
# Launch DeepSeek-V4-Flash-Base GRPO on the nemo_gym SWE dataset via the Megatron-Bridge path,
# 24-node non-colocated (16 train + 8 vLLM), on the nemo-rl-base(+apptainer) image. The repo is
# bind-mounted over /opt/nemo-rl so the run uses this checkout's code + configs; the SWE agent
# runs task .sif images via apptainer (baked into the image by docker/Dockerfile).
#
# Site-specific paths are env-overridable (defaults below match the original dev setup):
#   DATA       scratch root, bind-mounted to /results (holds venvs, dataset, mcore_cache, outputs)
#   CONTAINER  enroot/pyxis .sqsh image (must include apptainer; see docker/Dockerfile)
#   ACCOUNT    Slurm account
# Run knobs: CONFIG, NUM_NODES, EXP, PARTITION, WALL_TIME, REBUILD, REFIT_RATIO, EXTRA_OVERRIDES.
#
# Examples:
#   # 24n SWE run, override sequence length:
#   EXP=dsv4-gym-24n EXTRA_OVERRIDES='++policy.max_total_sequence_length=65536' \
#     bash examples/nemo_gym/launch_gym_24n.sh
#   # 1-node tiny integration smoke:
#   CONFIG=examples/nemo_gym/grpo_gym_dsv4_tiny_megatron.yaml NUM_NODES=1 EXP=dsv4-gym-tiny \
#     REFIT_RATIO=0.1 bash examples/nemo_gym/launch_gym_24n.sh
set -euo pipefail

# Repo root = two dirs up from this script (examples/nemo_gym/ -> repo root).
REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
DATA="${DATA:-/lustre/fs1/portfolios/coreai/projects/coreai_mlperf_training/users/mfutrega/data}"
CONTAINER="${CONTAINER:-/lustre/fs1/portfolios/coreai/projects/coreai_mlperf_training/users/mfutrega/nemo-rl-base-apptainer.sqsh}"
ACCOUNT="${ACCOUNT:-coreai_mlperf_training}"
EXP=${EXP:-dsv4-gym-24n}
PARTITION=${PARTITION:-batch}
WALL_TIME=${WALL_TIME:-3:59:00}
NUM_NODES=${NUM_NODES:-24}
CONFIG=${CONFIG:-examples/nemo_gym/grpo_gym_dsv4_megatron.yaml}
REBUILD=${REBUILD:-true}

OUTPUT="$DATA/gym-runs/$EXP"
mkdir -p "$OUTPUT/logs" "$OUTPUT/checkpoint"

export CONTAINER
# /lustre:/lustre exposes both fs1 (repo/data) and fsw (SWE dataset + SIF images).
export MOUNTS="/lustre:/lustre,$REPO:/opt/nemo-rl,$OUTPUT/logs:/logs,$OUTPUT/checkpoint:/checkpoint,$DATA:/results"

read -r -d '' COMMAND <<EOF || true
set -eou pipefail
cd /opt/nemo-rl
export HF_HOME=/results/dataset
export HF_DATASETS_CACHE=/results/dataset
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export NRL_FORCE_REBUILD_VENVS=${REBUILD}
export NEMO_RL_VENV_DIR=/results/venvs
export NRL_MEGATRON_CHECKPOINT_DIR=/results/grpo-runs/mcore_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NRL_REFIT_BUFFER_MEMORY_RATIO=${REFIT_RATIO:-0.005}
export NCCL_BOOTSTRAP_TIMEOUT=600
mkdir -p /logs/nemo_gym /results/nccl_dbg
# Re-apply the DSv4 + nemo_gym fixes that live in upstream-pinned git submodules (chunked DSA
# indexer in Megatron-LM + cryptography in the SWE-agent venvs). Idempotent; this survives a
# fresh \`git submodule update\` / clean clone. Container-level deps (apptainer) are baked by
# docker/Dockerfile instead.
bash tools/dsv4_gym_patches/apply.sh
uv run --extra mcore --extra nemo_gym python examples/nemo_gym/run_grpo_nemo_gym.py \\
    --config ${CONFIG} \\
    ++cluster.num_nodes=${NUM_NODES} \\
    ++checkpointing.enabled=false \\
    ++logger.log_dir=/logs \\
    ++logger.wandb_enabled=false ++logger.tensorboard_enabled=false ++logger.mlflow_enabled=false ${EXTRA_OVERRIDES:-}
EOF
export COMMAND

# SWE-bench rollouts spend long stretches on CPU (SIF extract, build, pytest) with GPUs idle;
# exempt the job from the idle-GPU killer.
IDLE_EXEMPT_JSON='{"OccupiedIdleGPUsJobReaper":{"exemptIdleTimeMins":"60","reason":"data_loading","description":"SWE-bench agent rollouts compile source inside SIF containers between LLM calls; GPUs idle 15-20 min per agent."}}'

cd "$REPO"
sbatch --export=ALL --nodes="$NUM_NODES" --account="$ACCOUNT" --partition="$PARTITION" \
  --time="$WALL_TIME" --job-name="$EXP" --gres=gpu:8 \
  --comment="${IDLE_EXEMPT_JSON}" \
  --output="$OUTPUT/$EXP-%j.out" \
  ray.sub
