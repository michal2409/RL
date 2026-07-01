#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Submit the non-colocated Qwen 3.5 397B-A17B SWE/OpenHands workload through
# ray.sub. The image is expected to contain this checkout; build it with
# NEMO_GYM_PREFETCH_CONFIGS set to the selected recipe to pre-bake Gym venvs.
#
# To run the MLPerf benchmark shape, set:
#   RECIPE=examples/nemo_gym/grpo_qwen35_397ba17b_swe_openhands_async_benchmark.yaml

set -euo pipefail

require_env() {
    local name="$1"
    local description="$2"

    if [[ -z "${!name:-}" ]]; then
        echo "Error: ${name} must be set." >&2
        echo "  ${description}" >&2
        exit 1
    fi
}

require_path() {
    local name="$1"
    local kind="$2"
    local description="$3"
    local path="${!name}"

    case "${kind}" in
        file) [[ -f "${path}" ]] ;;
        dir) [[ -d "${path}" ]] ;;
        path) [[ -e "${path}" ]] ;;
        *)
            echo "Internal error: unsupported path kind '${kind}'." >&2
            exit 1
            ;;
    esac || {
        echo "Error: ${name} must point to an existing ${kind}." >&2
        echo "  Current value: ${path}" >&2
        echo "  ${description}" >&2
        exit 1
    }
}

require_positive_integer() {
    local name="$1"
    local value="${!name}"
    if ! [[ "${value}" =~ ^[1-9][0-9]*$ ]]; then
        echo "Error: ${name} must be a positive integer; got '${value}'." >&2
        exit 1
    fi
}

# Optional: WANDB_API_KEY, HF_TOKEN, EXTRA_MOUNTS, and the Slurm tuning knobs
# below. Everything else is required so a malformed job fails before sbatch.
require_env EXP_NAME "Experiment name used for the job and result directory."
require_env REPO_LOCATION "Host path to this NeMo-RL checkout."
require_env CONTAINER_IMAGE_PATH "Container image passed to ray.sub."
require_env SLURM_ACCOUNT "Slurm account for the allocation."
require_env SLURM_PARTITION "Slurm partition for the allocation."
require_env GPUS_PER_NODE "Number of GPUs on each allocated node."
require_env HF_CKPT_PATH "Host path to the Hugging Face checkpoint."
require_env NRL_MEGATRON_CHECKPOINT_DIR "Host checkpoint-conversion cache directory."
require_env NEMO_GYM_SWE_TRAIN_DATA_PATH "Host path to the training JSONL."
require_env NEMO_GYM_SWE_VALIDATION_DATA_PATH "Host path to the validation JSONL."
require_env NEMO_GYM_SWE_SIF_DIR "Host directory containing task SIF images."

require_path REPO_LOCATION dir "ray.sub is submitted from this checkout."
require_path HF_CKPT_PATH dir "The checkpoint is mounted into every container."
require_path NRL_MEGATRON_CHECKPOINT_DIR dir "This may be an initially empty directory."
require_path NEMO_GYM_SWE_TRAIN_DATA_PATH file "Expected a JSONL dataset."
require_path NEMO_GYM_SWE_VALIDATION_DATA_PATH file "Expected a JSONL dataset."
require_path NEMO_GYM_SWE_SIF_DIR dir "Expected the generated SIF image directory."
require_positive_integer GPUS_PER_NODE

TRAIN_NODES="${TRAIN_NODES:-32}"
GEN_NODES="${GEN_NODES:-32}"
NODES="${NODES:-$((TRAIN_NODES + GEN_NODES))}"
require_positive_integer TRAIN_NODES
require_positive_integer GEN_NODES
require_positive_integer NODES
if (( NODES <= GEN_NODES )); then
    echo "Error: NODES (${NODES}) must exceed GEN_NODES (${GEN_NODES})." >&2
    exit 1
fi

CONTAINER_REPO_LOCATION="${CONTAINER_REPO_LOCATION:-/opt/nemo-rl}"
RECIPE="${RECIPE:-examples/nemo_gym/grpo_qwen35_397ba17b_swe_openhands_async.yaml}"
CONTAINER_INPUT_ROOT="${CONTAINER_INPUT_ROOT:-/inputs/nemo_gym}"
CONTAINER_HF_CKPT_PATH="${CONTAINER_HF_CKPT_PATH:-${HF_CKPT_PATH}}"
CONTAINER_NRL_MEGATRON_CHECKPOINT_DIR="${CONTAINER_NRL_MEGATRON_CHECKPOINT_DIR:-${CONTAINER_INPUT_ROOT}/mcore_ckpt}"
CONTAINER_NEMO_GYM_SWE_TRAIN_DATA_PATH="${CONTAINER_NEMO_GYM_SWE_TRAIN_DATA_PATH:-${CONTAINER_INPUT_ROOT}/data/train.jsonl}"
CONTAINER_NEMO_GYM_SWE_VALIDATION_DATA_PATH="${CONTAINER_NEMO_GYM_SWE_VALIDATION_DATA_PATH:-${CONTAINER_INPUT_ROOT}/data/validation.jsonl}"
CONTAINER_NEMO_GYM_SWE_SIF_DIR="${CONTAINER_NEMO_GYM_SWE_SIF_DIR:-${CONTAINER_INPUT_ROOT}/sif}"

if [[ "${HF_CKPT_PATH}" == */snapshots/* ]]; then
    HF_MODEL_CACHE_DIR="${HF_MODEL_CACHE_DIR:-$(dirname "$(dirname "${HF_CKPT_PATH}")")}"
    CONTAINER_HF_MODEL_CACHE_DIR="${CONTAINER_HF_MODEL_CACHE_DIR:-$(dirname "$(dirname "${CONTAINER_HF_CKPT_PATH}")")}"
fi

MLPERF_SUBMISSION_ORG="${MLPERF_SUBMISSION_ORG:-reference}"
MLPERF_SUBMISSION_PLATFORM="${MLPERF_SUBMISSION_PLATFORM:-reference}"
MLPERF_TARGET_ACCURACY="${MLPERF_TARGET_ACCURACY:-1.0}"
GRPO_SEED="${GRPO_SEED:-$(((RANDOM << 15) | RANDOM))}"

cd "${REPO_LOCATION}"
if [[ ! -f "${RECIPE}" ]]; then
    echo "Error: RECIPE must exist in REPO_LOCATION; got '${RECIPE}'." >&2
    exit 1
fi
OUT_DIR="$(pwd)/results/${EXP_NAME}"
HOST_HF_HOME="${HF_HOME:-$(pwd)/.cache}"
export BASE_LOG_DIR="${OUT_DIR}/logs"
mkdir -p "${OUT_DIR}/logs" "${OUT_DIR}/checkpoint" "${HOST_HF_HOME}"

EXTRA_OVERRIDES=""
if (( $# > 0 )); then
    printf -v EXTRA_OVERRIDES ' %q' "$@"
fi

# Keep credentials as runtime expansions instead of embedding their values in
# COMMAND, which is printed and propagated through the Slurm environment.
COMMAND=$(cat <<EOF
cd ${CONTAINER_REPO_LOCATION}

HF_HOME=${CONTAINER_REPO_LOCATION}/.cache \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
HF_TOKEN="\${HF_TOKEN:-}" \
WANDB_API_KEY="\${WANDB_API_KEY:-}" \
CONTAINER_HF_CKPT_PATH="${CONTAINER_HF_CKPT_PATH}" \
NRL_MEGATRON_CHECKPOINT_DIR="${CONTAINER_NRL_MEGATRON_CHECKPOINT_DIR}" \
NEMO_GYM_SWE_WORKSPACE_ROOT=/logs/nemo_gym/workspace \
NEMO_GYM_SWE_TRAIN_DATA_PATH="${CONTAINER_NEMO_GYM_SWE_TRAIN_DATA_PATH}" \
NEMO_GYM_SWE_VALIDATION_DATA_PATH="${CONTAINER_NEMO_GYM_SWE_VALIDATION_DATA_PATH}" \
NEMO_GYM_SWE_SIF_DIR="${CONTAINER_NEMO_GYM_SWE_SIF_DIR}" \
MLPERF_SUBMISSION_ORG="${MLPERF_SUBMISSION_ORG}" \
MLPERF_SUBMISSION_PLATFORM="${MLPERF_SUBMISSION_PLATFORM}" \
MLPERF_TARGET_ACCURACY="${MLPERF_TARGET_ACCURACY}" \
MLPERF_LOG_FILE=/logs/mllogger.log \
uv run python examples/nemo_gym/run_grpo_nemo_gym.py \
    --config "${RECIPE}" \
    ++logger.mlperf_enabled=true \
    ++logger.mlperf.enabled=true \
    ++cluster.num_nodes=${NODES} \
    ++cluster.gpus_per_node=${GPUS_PER_NODE} \
    ++policy.generation.colocated.resources.num_nodes=${GEN_NODES} \
    ++policy.generation.colocated.resources.gpus_per_node=${GPUS_PER_NODE} \
    ++logger.wandb.name="${EXP_NAME}" \
    ++logger.log_dir=/logs \
    ++checkpointing.checkpoint_dir=/checkpoint \
    ++grpo.seed=${GRPO_SEED}${EXTRA_OVERRIDES}
EOF
)

echo "Using grpo.seed=${GRPO_SEED}"
echo -e "Running command:\n${COMMAND}"

# ray.sub creates coordination files under BASE_LOG_DIR on the host. Mount the
# same path into the container as well as the stable /logs path used by Gym.
MOUNTS="${OUT_DIR}/logs:${OUT_DIR}/logs,${HOST_HF_HOME}:${CONTAINER_REPO_LOCATION}/.cache"
MOUNTS+=",${OUT_DIR}/checkpoint:/checkpoint,${OUT_DIR}/logs:/logs"
if [[ -n "${HF_MODEL_CACHE_DIR:-}" ]]; then
    MOUNTS+=",${HF_MODEL_CACHE_DIR}:${CONTAINER_HF_MODEL_CACHE_DIR}"
else
    MOUNTS+=",${HF_CKPT_PATH}:${CONTAINER_HF_CKPT_PATH}"
fi
MOUNTS+=",${NRL_MEGATRON_CHECKPOINT_DIR}:${CONTAINER_NRL_MEGATRON_CHECKPOINT_DIR}"
MOUNTS+=",${NEMO_GYM_SWE_TRAIN_DATA_PATH}:${CONTAINER_NEMO_GYM_SWE_TRAIN_DATA_PATH}"
MOUNTS+=",${NEMO_GYM_SWE_VALIDATION_DATA_PATH}:${CONTAINER_NEMO_GYM_SWE_VALIDATION_DATA_PATH}"
MOUNTS+=",${NEMO_GYM_SWE_SIF_DIR}:${CONTAINER_NEMO_GYM_SWE_SIF_DIR}"
if [[ -n "${EXTRA_MOUNTS:-}" ]]; then
    MOUNTS+=",${EXTRA_MOUNTS}"
fi

# Async SWE rollouts can legitimately leave the training pool idle while the
# replay buffer fills. Clusters without this reaper can override SLURM_COMMENT.
SLURM_IDLE_EXEMPT_MINS="${SLURM_IDLE_EXEMPT_MINS:-120}"
SLURM_COMMENT="${SLURM_COMMENT:-{\"OccupiedIdleGPUsJobReaper\":{\"exemptIdleTimeMins\":\"${SLURM_IDLE_EXEMPT_MINS}\",\"reason\":\"rl-rollout-warmup\"}}}"

SBATCH_ARGS=(
    --nodes="${NODES}"
    --account="${SLURM_ACCOUNT}"
    --partition="${SLURM_PARTITION}"
    --time="${SLURM_TIME:-1:0:0}"
    --job-name="${EXP_NAME}"
    --comment="${SLURM_COMMENT}"
)
[[ -n "${SLURM_QOS:-${SBATCH_QOS:-}}" ]] && SBATCH_ARGS+=(--qos="${SLURM_QOS:-${SBATCH_QOS}}")
[[ -n "${SLURM_GRES:-${SBATCH_GRES:-}}" ]] && SBATCH_ARGS+=(--gres="${SLURM_GRES:-${SBATCH_GRES}}")
[[ -n "${SLURM_SEGMENT:-${SBATCH_SEGMENT:-}}" ]] && SBATCH_ARGS+=(--segment="${SLURM_SEGMENT:-${SBATCH_SEGMENT}}")
[[ -n "${SLURM_EXCLUDE:-}" ]] && SBATCH_ARGS+=(--exclude="${SLURM_EXCLUDE}")

MOUNTS="${MOUNTS}" \
COMMAND="${COMMAND}" \
CONTAINER="${CONTAINER_IMAGE_PATH}" \
CONTAINER_WORKDIR="${CONTAINER_REPO_LOCATION}" \
GPUS_PER_NODE="${GPUS_PER_NODE}" \
sbatch "${SBATCH_ARGS[@]}" ray.sub
