#!/usr/bin/env bash
# Idempotently re-apply the DSv4 + nemo_gym fixes that live inside upstream-pinned git
# submodules (so they are NOT lost on a fresh `git submodule update` / clean clone).
#
# Run this AFTER `git submodule update --init --recursive`, and/or have the launcher call
# it before training (see data/launch_gym_24n.sh). Safe to run repeatedly — each step is
# a no-op if already applied. Container-level deps (apptainer) live in docker/Dockerfile*
# instead, so they are not handled here.
#
#   1. Megatron-LM: chunked DSA indexer (caps the O(seq^2) index-score matrix that OOMs at
#      long sequence length). Vendored as megatron-lm-dsa-chunked-indexer.patch.
#   2. Gym SWE-agent venvs: install `cryptography` (a transitive dep of kubernetes->google-auth
#      that R2E-Gym / SWE-bench pull but do not install by default; without it the agent venv
#      ImportErrors before any rollout runs).
set -euo pipefail

# Repo root = two levels up from this script (tools/dsv4_gym_patches/apply.sh -> repo root).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

MCORE_DIR="3rdparty/Megatron-LM-workspace/Megatron-LM"
DSA_FILE="$MCORE_DIR/megatron/core/transformer/experimental_attention_variant/dsa.py"
DSA_PATCH="$SCRIPT_DIR/megatron-lm-dsa-chunked-indexer.patch"

echo "[dsv4-gym-patches] repo root: $REPO_ROOT"

# ---- 1. Chunked DSA indexer ------------------------------------------------------------
if [[ ! -f "$DSA_FILE" ]]; then
  echo "[dsv4-gym-patches] WARN: $DSA_FILE not found (run 'git submodule update --init --recursive' first); skipping DSA patch."
elif grep -q "_dsa_indexer_chunk_size" "$DSA_FILE"; then
  echo "[dsv4-gym-patches] chunked DSA indexer: already applied (no-op)."
else
  echo "[dsv4-gym-patches] chunked DSA indexer: applying patch..."
  git -C "$MCORE_DIR" apply "$DSA_PATCH"
  grep -q "_dsa_indexer_chunk_size" "$DSA_FILE" && echo "[dsv4-gym-patches] chunked DSA indexer: applied OK."
fi

# ---- 2. cryptography in the SWE-agent venv setup scripts -------------------------------
# These scripts each build a venv (`uv venv` + `uv pip install -e .`) that the nemo_gym SWE
# agent runs in. Append a cryptography install if not already present. cwd inside each script
# is the cloned harness dir (which contains ./venv), so `venv/bin/python` resolves.
SWE_SETUP_DIR="3rdparty/Gym-workspace/Gym/responses_api_agents/swe_agents/setup_scripts"
CRYPTO_MARK="# dsv4-gym: cryptography (kubernetes->google-auth transitive dep, not auto-installed)"
for s in r2e_gym.sh swebench.sh swebench_multilingual.sh swe_rebench.sh; do
  f="$SWE_SETUP_DIR/$s"
  [[ -f "$f" ]] || { echo "[dsv4-gym-patches] cryptography: $s not present, skipping."; continue; }
  if ! grep -qE "uv venv|venv/bin/python" "$f"; then
    echo "[dsv4-gym-patches] cryptography: $s does not build a ./venv, skipping."
    continue
  fi
  if grep -qF "$CRYPTO_MARK" "$f"; then
    echo "[dsv4-gym-patches] cryptography in $s: already present (no-op)."
  else
    echo "[dsv4-gym-patches] cryptography in $s: appending install."
    {
      echo ""
      echo "$CRYPTO_MARK"
      echo 'uv pip install -p venv/bin/python cryptography --no-cache'
    } >> "$f"
  fi
done

echo "[dsv4-gym-patches] done."
