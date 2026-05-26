# DeepSeek-V4-Flash-Base — full-scale GRPO on Slurm (24-node H100 default)

This recipe runs GRPO on the full DeepSeek-V4-Flash-Base checkpoint
(~671B params) via Megatron-Bridge, with vLLM as the non-colocated
rollout engine. Default layout is 24 nodes × 8 H100 80GB = 192 GPUs
(16 train + 8 inference). See the YAML header for sizing rationale.

## Files

| file | purpose |
|---|---|
| `examples/configs/grpo_dsv4_base_bf16_4k_megatron_24n.yaml` | training recipe |
| `examples/launch_grpo_dsv4_full.sh` | sbatch wrapper around `ray.sub` |
| `ray.sub` | upstream NeMo-RL multinode Slurm template |
| `tools/patch_vllm_dsv4_base_fp8_quick.sh` | required vLLM Base-FP8 monkey-patch |

## Prerequisites (one-time per cluster / container)

1. **Build / pull the NeMo-RL container** with the `mcore` extra. Either
   build it locally from `docker/Dockerfile` or import the published
   image to a sqsh:
   ```bash
   enroot import docker://nvcr.io/.../nemo-rl:<tag>
   # → produces nemo-rl+<tag>.sqsh
   ```
   The Dockerfile already runs `uv sync --extra mcore`, which installs
   `fast-hadamard-transform` (needed by DSv4 DSA) thanks to the
   `pyproject.toml` update.

2. **Stage the model checkpoint** so all nodes can read it:
   ```bash
   HF_HOME=/lustre/.../hf_cache \
   hf download deepseek-ai/DeepSeek-V4-Flash-Base --max-workers 16
   ```
   Then mount `/lustre/.../hf_cache` into the container as `$HF_HOME`.

3. **(First run only) HF → mcore conversion** is automatic on the first
   step. It writes a converted checkpoint to
   `NRL_MEGATRON_CHECKPOINT_DIR` (set in the launcher) and reuses it on
   subsequent runs. Conversion takes ~10 min for the full model;
   reserving a node-shared scratch path here is critical.

## Submitting the job

The wrapper script consumes environment variables and submits via
`sbatch ray.sub`. Minimum required vars:

```bash
export CONTAINER=/lustre/fsw/portfolios/coreai/users/$USER/nemo-rl.sqsh
export ACCOUNT=coreai_dlalgo_nemo          # your Slurm account
export PARTITION=batch                      # your Slurm partition
export EXP_NAME=dsv4-flash-grpo-24n-$(date +%Y%m%d-%H%M)
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
export MOUNTS=/lustre/fsw/portfolios/coreai/users/$USER/data:/results,/lustre/fsw/portfolios/coreai/users/$USER/hf_cache:/hf_cache

bash examples/launch_grpo_dsv4_full.sh
```

Optional overrides (any subset):

```bash
export NUM_NODES=24                      # default 24
export WALL_TIME=24:00:00                # default 12:00:00
export WANDB_API_KEY=$YOUR_WANDB_KEY     # if set, wandb logger turns on
export HF_HOME=/hf_cache                 # path inside the container
export CONFIG=examples/configs/grpo_dsv4_base_bf16_4k_megatron_24n.yaml
export EXTRA_HYDRA_ARGS="++policy.megatron_cfg.optimizer.lr=1.0e-7 ++grpo.max_num_steps=2000"
```

Then:

```bash
bash examples/launch_grpo_dsv4_full.sh
# → prints the chosen settings and submits one Slurm job. The job ID is
#   in the sbatch stdout; logs land in $SLURM_SUBMIT_DIR/<JOBID>-logs/.
```

## What to expect

- **Bootstrap (per fresh container):** Ray cluster + vLLM init + Megatron
  init ≈ 5–8 min. First-ever run additionally does the HF→mcore
  conversion (~10 min); subsequent runs reuse it.
- **Per-step time (estimate):** with `train_global_batch_size=512`,
  `max_new_tokens=2048`, expect 4–8 min/step depending on rollout
  diversity and prompt length. Plan WALL_TIME accordingly
  (e.g. 24h for 200 steps).
- **GPU memory:** non-colocated layout means vLLM and Megatron each get
  their own GPUs; per-rank training memory is ~50–60 GB with this
  EP/PP setting and CPU-offloaded optimizer.

## Scaling node count

The launcher passes `NUM_NODES` straight to `sbatch --nodes` and to the
`cluster.num_nodes` Hydra override. To change layouts:

| total nodes | train (EP × PP × TP) | inference (TP, replicas) | how |
|---|---|---|---|
| **24 (default)** | 16 nodes, EP=8 PP=2 TP=1 | 8 nodes, TP=4, 16 replicas | as shipped |
| 32 | 16 nodes, EP=8 PP=2 TP=1 | 16 nodes, TP=4, 32 replicas | `NUM_NODES=32 EXTRA_HYDRA_ARGS='++policy.generation.colocated.resources.num_nodes=16'` |
| 16 | 12 nodes, EP=8 PP=1 TP=1 | 4 nodes, TP=4, 8 replicas | `NUM_NODES=16 EXTRA_HYDRA_ARGS='++policy.megatron_cfg.pipeline_model_parallel_size=1 ++policy.generation.colocated.resources.num_nodes=4'` |
| 48 | 32 nodes, EP=8 PP=2 TP=1 | 16 nodes, TP=4, 32 replicas | `NUM_NODES=48 EXTRA_HYDRA_ARGS='++policy.generation.colocated.resources.num_nodes=16'` |

Rule of thumb: `cluster.num_nodes = train_nodes + colocated.resources.num_nodes`.

## Knobs you'll likely want to override

| Hydra key | meaning | typical value |
|---|---|---|
| `++grpo.max_num_steps=N` | total GRPO iterations | 200–2000 |
| `++grpo.num_prompts_per_step=N` | prompts per global batch | 32 or 64 |
| `++grpo.num_generations_per_prompt=N` | samples per prompt for GRPO baseline | 8 or 16 |
| `++policy.train_global_batch_size=N` | must equal prompts × generations | 256 / 512 / 1024 |
| `++policy.megatron_cfg.optimizer.lr=X` | learning rate | 1e-7 to 5e-7 |
| `++policy.megatron_cfg.optimizer.use_precision_aware_optimizer=true` | FP32 master with BF16 grad accumulation | keep on |
| `++policy.megatron_cfg.activation_checkpointing=true` | trade compute for memory | keep on at full scale |
| `++policy.generation.vllm_cfg.tensor_parallel_size=N` | vLLM TP | 4 (default) or 8 |
| `++policy.generation.vllm_cfg.gpu_memory_utilization=X` | KV cache budget | 0.85 non-colocated |

## Driver < 13.0?

The 24-node YAML sets `NRL_DSV4_DISABLE_FUSED_MHC=1` in
`policy.megatron_cfg.env_vars` by default, routing the mHC proj-RMS
layer through the native (non-cuTile) path. On hosts with CUDA driver
≥ 13.0 you can switch it to `"0"` for the fused kernel:

```bash
EXTRA_HYDRA_ARGS='++policy.megatron_cfg.env_vars.NRL_DSV4_DISABLE_FUSED_MHC=0' \
  bash examples/launch_grpo_dsv4_full.sh
```

## Verifying first

Before launching the full 24-node job, the single-node smoke test
(`grpo_dsv4_base_tiny_1n_megatron.yaml`) is the cheapest sanity check
of the same code paths — it exercises EP, refit, DSA, mHC, and the
HF→mcore conversion on a single node in ~10 min. Confirm that one
passes 3 steps end-to-end before burning multi-node time.
