# DSv4 + nemo_gym: build & run from scratch

Everything needed to run DeepSeek-V4-Flash-Base GRPO on the nemo_gym SWE dataset (Megatron-Bridge
path) from a clean clone. Most fixes are committed in this repo; two live in upstream-pinned
submodules and are re-applied by `apply.sh` (see below).

## 1. Container (build once)

`docker/Dockerfile` (and `docker/Dockerfile.ngc_pytorch`) now install **apptainer 1.3.1**
(+ squashfs-tools / uidmap / libfuse3 / fakeroot). The nemo_gym SWE agent runs each task's
environment as a `.sif` via `apptainer exec` nested in the pyxis/enroot container; the base
images ship no container runtime, so this is required. No manual `--container-save` bake anymore —
just build the image.

## 2. Clone + submodules + apply submodule patches

```bash
git clone <repo> && cd RL
git checkout dsv4-flash-megatron-bridge
git submodule update --init --recursive
bash tools/dsv4_gym_patches/apply.sh        # idempotent; re-run any time
```

`apply.sh` re-applies the two fixes that live inside upstream-pinned submodules (and would be
wiped by a fresh `git submodule update`):

1. **Chunked DSA indexer** — `megatron-lm-dsa-chunked-indexer.patch` applied to
   `3rdparty/Megatron-LM-workspace/Megatron-LM/.../experimental_attention_variant/dsa.py`.
   Query-chunks the indexer's `[seqlen, index_n_heads, seqlen]` fp32 score matrix (the O(seq^2)
   OOM wall at long sequence length); env `MCORE_DSA_INDEXER_CHUNK_SIZE` (default 1024, 0=off).
   Numerically bit-exact vs upstream (see `test_dsa_chunk_equiv.py`).
2. **cryptography in the SWE-agent venvs** — appends a `cryptography` install to the venv-building
   setup scripts (`r2e_gym.sh`, `swebench.sh`, `swebench_multilingual.sh`) under
   `3rdparty/Gym-workspace/Gym/.../swe_agents/setup_scripts/`. It is a transitive dep of
   `kubernetes`->`google-auth` that R2E-Gym / SWE-bench pull but don't install, so the agent venv
   ImportErrors before any rollout otherwise.

The launcher (`data/launch_gym_24n.sh`) calls `apply.sh` automatically before `uv run`, so a run
self-heals even if you forget step 2's last line.

## 3. Repo code (already committed on `dsv4-flash-megatron-bridge`)

These are in the branch — no action needed, listed for reference:
- `nemo_rl/environments/nemo_gym.py` — `default_host`, `allow_empty_rollouts` fallback, dotenv path.
- `nemo_rl/models/generation/vllm/vllm_worker_async.py` — vLLM 0.21 `OpenAIServingRender` wiring.
- `nemo_rl/models/generation/vllm/vllm_generation.py` — NCCL channel/buffer caps (non-colocated).
- `nemo_rl/distributed/stateless_process_group.py` — `NRL_PG_STORE_TIMEOUT_S`.
- `examples/nemo_gym/grpo_gym_dsv4_megatron.yaml` (24n) and `..._tiny_megatron.yaml` (1n smoke).

## 4. Run

```bash
# 24-node SWE run
CONFIG=examples/nemo_gym/grpo_gym_dsv4_megatron.yaml NUM_NODES=24 EXP=dsv4-gym-24n \
  bash data/launch_gym_24n.sh

# 1-node tiny smoke (rollouts empty by design; validates integration)
CONFIG=examples/nemo_gym/grpo_gym_dsv4_tiny_megatron.yaml NUM_NODES=1 EXP=dsv4-gym-tiny \
  REFIT_RATIO=0.1 bash data/launch_gym_24n.sh
```

## Gotchas

- **env_vars values must be strings.** `policy.megatron_cfg.env_vars.*` feeds Ray's `env_vars`
  (`Dict[str,str]`). A CLI override like `++...MCORE_DSA_INDEXER_CHUNK_SIZE=0` makes Hydra parse an
  **int** and Ray rejects it at worker creation. Quote it: `...='"0"'`, or set it in YAML as `"0"`.
- **Quick in-container test of the `.sqsh`** needs `--no-container-mount-home` (else host `$HOME`
  shadows the baked uv python at `/opt/nemo_rl_venv/bin/python`).
- Long SWE prompts (1537+ tok) exceed small context lengths; with `allow_empty_rollouts: true` the
  rollout becomes a reward=0 empty turn. Empty-rollout batches can trip a Megatron DDP grad-ready
  assert (`param_and_grad_buffer.py:257`) on step 2 — orthogonal to the chunked indexer; it needs
  non-empty (real, long) rollouts to exercise the OOM-fix anyway.
