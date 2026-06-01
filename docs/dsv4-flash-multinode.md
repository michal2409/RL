# DeepSeek-V4-Flash-Base GRPO at multi-node scale

This document captures the fixes required to run a GRPO training job for
`deepseek-ai/DeepSeek-V4-Flash-Base` on H100 80 GB hardware at multinode
scale (24+ nodes, non-colocated layout). The single-node smoke recipe
(`grpo_dsv4_base_bf16_4k_megatron_8n_ep4.yaml`) and the GB300 16-node
recipe were already known to work; everything below is what was needed
to make 24- and 32-node H100 launches reach steady-state training.

## TL;DR — the winning configuration

Cluster: 24-32 nodes × 8 H100 80 GB.

- **Train side**: `TP=1`, `PP=8`, `EP=8`, `ETP=1`,
  `activation_checkpointing=true`, `sequence_packing.enabled=false`,
  CPU-offload optimizer.
- **vLLM side**: `tensor_parallel_size=8` (one replica per node),
  `gpu_memory_utilization=0.88`, `kv_cache_dtype=fp8_ds_mla`,
  `enforce_eager=true`, `precision=fp8`.
- **MoE**: `moe_token_dispatcher_type="flex"`, `moe_enable_deepep=false`,
  `moe_grouped_gemm` left at the Megatron-Bridge default (`true`).
- **Refit chunk buffer**: `NRL_REFIT_BUFFER_MEMORY_RATIO=0.005`
  (the default `0.02` is too large for the headroom vLLM leaves at GMU
  0.88).
- **PP layout**: explicit string
  `"Et*6|t*6|t*6|t*5|t*5|t*5|t*5|t*5,m,L"` so the 3 hash-MoE layers stay
  on stage 0 with the embedding and MTP + LM-head land on stage 7.
- **Batch sizing**: `num_prompts_per_step * num_generations_per_prompt`
  must divide BOTH the train DP shard count and the vLLM replica count
  (see "batch shape" section below).

Reference recipe: `examples/configs/grpo_dsv4_base_bf16_4k_megatron_24n.yaml`.
Reference launcher: `examples/launch_grpo_dsv4_full.sh`.

## Categorized fixes

Each fix here corresponds to one or more files in the PR. Fixes are
ordered roughly by "how surprising the failure mode was."

### 1. Pipeline-layer-layout passthrough (required for hash-MoE at PP>1)

**Problem.** DSv4 has 3 hash-MoE layers at the start of the decoder
stack. Megatron-LM asserts that an explicit `pipeline_model_parallel_layout`
must be set when hash-MoE is combined with `pipeline_model_parallel_size > 1`,
because the hash layers must all live on the first PP stage alongside
the embedding. Without a layout, the model build asserts before training
starts.

**Fix.** Plumb a new optional `pipeline_model_parallel_layout` field end
to end:

- `nemo_rl/models/policy/__init__.py` — add the field to `MegatronConfig`
  (`NotRequired[str | list | None]`).
- `nemo_rl/models/megatron/setup.py::_apply_parallelism_config` — pass
  the value through to `model_cfg.pipeline_model_parallel_layout` when set.
- `nemo_rl/models/megatron/community_import.py::import_model_from_hf_name` —
  set the same field on the model provider during HF→Megatron conversion
  so the cached `run_config.yaml` has the right layout baked in.
- `examples/configs/grpo_math_1B_megatron.yaml` — add
  `pipeline_model_parallel_layout: null` as a default so the
  `TypedDict.NotRequired` field is documented in the base config.

Layout-string syntax is parsed by Megatron-LM's
`PipelineParallelLayerLayout.from_str` (chars: `E`=embedding,
`L`=loss/lm-head, `t`=transformer, `m`=mtp; `|` separates stages;
`*N` repeats; commas are cosmetic).

### 2. PP broadcast tolerates tied weights on multiple ranks

**Problem.** With DSv4 + `PP>1`, the same logical parameter (embedding /
LM-head / MTP head) legitimately lives on more than one PP rank. The
existing `broadcast_obj_from_pp_rank` helper asserted that the object
was on exactly one rank and crashed on the second-stage discovery.

**Fix.** `nemo_rl/models/megatron/pipeline_parallel.py` —
`broadcast_obj_from_pp_rank` now picks the lowest-rank holder as the
broadcast source instead of failing. The downstream consumer is the
parameter-size accounting in `_calculate_refit_param_info`, which only
needs a single consistent scalar per logical parameter; "lowest-rank
holder" is a safe canonical choice.

### 3. MoE dispatcher: `flex` + DeepEP off (YAML only, no code)

**Problem.** The Megatron-Bridge DSv4 provider defaults to the `flex`
dispatcher with DeepEP enabled, which hangs deterministically inside the
first refit's PP-group broadcast at this scale. The plain `flex` path
(DeepEP off, `grouped_gemm` left at its default `true`) progresses
cleanly.

**Fix.** Set `moe_token_dispatcher_type: "flex"` and
`moe_enable_deepep: false` in the recipe YAML; both are already plumbed
through `_apply_moe_config`. No code change is needed.

**Not needed:** an earlier iteration also added a YAML override for
`moe_grouped_gemm` in `_apply_moe_config`. The winning config leaves it
at the bridge default (`true`), and every shipped recipe that sets it
sets it to `true` — i.e. equal to the default — so the override path was
behavior-neutral debugging scaffolding and has been removed to keep the
PR minimal. Re-add it only if a future config needs `grouped_gemm=false`
(see the `alltoall + grouped_gemm=false` dead end below, which it was
originally added to probe).

### 4. Multi-UniqueId NCCL bootstrap (required at 192+ ranks)

**Problem.** The cross-cluster `model_update_group` joins every Megatron
train worker AND every vLLM inference worker into one NCCL communicator
(192–256 ranks). The Python NCCL wrapper passes a single `UniqueId` to
`ncclCommInitRankScalable`, which forces `nbufs=1`. With a single
bootstrap root, init occasionally fails with `ncclUnhandledCudaError` —
sometimes succeeds, sometimes hangs silently — at this rank count.

**Fix.** `nemo_rl/distributed/stateless_process_group.py` — generate
`max(1, world_size // 32)` `UniqueId`s on rank 0, publish them all to
the `TCPStore`, and pass the list to `Communicator.init`. The wrapper
routes the multi-id form to `ncclCommInitRankScalable` with
`nbufs=len(list)`, so init has multiple broadcast roots and is reliable
at 192+ ranks. For small communicators (`<= 32` ranks) the original
single-`UniqueId` path is preserved exactly.

### 5. Packed-tensor 8-byte alignment (refit correctness)

**Problem.** `packed_broadcast_producer` and `packed_broadcast_consumer`
in `nemo_rl/utils/packed_tensor.py` pack many tensors of mixed dtype
(BF16, FP8, FP32 scales) into one large UInt8 buffer, then split + view
back on the consumer side. A BF16 tensor with odd `numel` leaves the
next chunk at a 2-byte boundary; the subsequent
`chunk.view(torch.float32)` fails with `storage_offset must be divisible
by 4 to view Byte as Float`. This is reachable as soon as you have FP32
scale tensors interleaved with BF16 weights — which DSv4 does because of
the FP8 KV-cache q/k scales.

**Fix.** Pad each chunk's payload up to 8 bytes (`_PACK_ALIGN = 8`) on
both producer and consumer; record the unpadded `tensor_size` alongside
the padded `aligned_size` in the meta tuple; trim back to `tensor_size`
before `view(dtype)`. 8 covers FP64/INT64 by direct match and FP32/INT32
by transitivity, so any common dtype `view` is safe.

### 6. Refit-time memory hygiene

**Problem.** Two distinct memory issues at the boundary between
training and inference:

(a) The per-expert EP `all_gather` inside Megatron-Bridge's
`gather_from_ep_ranks` allocates small NCCL scratch buffers on every
parameter. Without a sync first, those allocations fail with CUDA OOM
even when the totals look fine — PyTorch holds reserved-but-unallocated
pages from earlier training that NCCL can't see.

(b) On the vLLM side, the consumer chunk buffer (~1.6 GiB at the default
`NRL_REFIT_BUFFER_MEMORY_RATIO=0.02`) competes with KV cache + private
NCCL pools. At GMU 0.88 there is only ~1 GiB of headroom; the buffer
allocation OOMs.

**Fix.**
- `nemo_rl/models/policy/workers/megatron_policy_worker.py::prepare_refit_info`
  — call `gc.collect()` + `torch.cuda.empty_cache()` + `synchronize()`
  before iterating params, and `empty_cache()` after each yielded
  tensor.
- `nemo_rl/models/generation/vllm/vllm_backend.py::update_weights_from_collective`
  — call `empty_cache()` + `synchronize()` on entry to reclaim
  reserved-but-unallocated pages before the chunk buffer is allocated.
- `examples/launch_grpo_dsv4_full.sh` — export
  `NRL_REFIT_BUFFER_MEMORY_RATIO=0.005`, which quarters the refit chunk
  size. Smaller chunks just mean more broadcasts; correctness is
  unaffected.

### 7. vLLM `tensor_parallel_size=8`

This is the single biggest knob change. At `TP=4`, DSv4-Flash-Base FP8
plus FP8-MLA KV cache plus the refit chunk buffer does not fit in 80 GiB.
Doubling TP to 8 (one vLLM replica per node) roughly halves the per-GPU
model footprint and leaves the headroom the refit broadcast needs.

The trade-off is throughput: at TP=8 we have 8 replicas on 8 inference
nodes instead of 16 replicas at TP=4. That cuts generation throughput
roughly in half, but it's the only configuration that fits cleanly on
80 GiB.

### 8. Batch-shape divisibility

**Problem.** GRPO generates
`num_prompts_per_step * num_generations_per_prompt` samples per step.
That total must be divisible by:

- The train DP shard count (`world_size_train / (TP_train * PP_train)`).
- The vLLM replica count (`world_size_inference / TP_inference`).

For the default 24-node layout (DP=16, replicas=8) the LCM is 16. For
the 32-node launch (DP=24, replicas=8) the LCM is 24. The shipped
recipe uses `num_prompts_per_step=48`, `num_generations_per_prompt=8`
(total = 384), which works for any node count whose LCM divides 48.

**Symptom of getting this wrong.** `lm_policy.py::get_logprobs` ->
`batched_data_dict.py::shard_by_batch_size` raises
`AssertionError: Batch size (N) is not a multiple of shards (M)` after
the first refit and rollout complete. Easy to miss because it fires
late in the first step.

## Dead ends — things that turned out NOT to matter

For future debugging context, these were tried and explicitly verified
to be NOT the cause of any remaining failure. None of them is in the
PR; they're documented here so the next person doesn't repeat them.

- **Lazy `model_update_group` NCCL communicator init**: deferring the
  256-rank cross-cluster `init_nccl_communicator` from
  `init_collective` setup to the first call of
  `update_weights_from_collective` / `broadcast_weights_for_collective`.
  Verified harmless but did not change failure shape. Reverted to eager
  init.
- **`moe_token_dispatcher_type=alltoall` with `moe_enable_deepep=false`
  and `moe_grouped_gemm=false`**: hangs at a different SeqNum in the
  same PP broadcast as `flex+DeepEP`. The working combination is `flex`
  + `moe_enable_deepep=false` + default `grouped_gemm` (true).
- **`NCCL_NET_GDR_LEVEL=PXB`, `NCCL_IB_DISABLE=1`,
  `NCCL_SOCKET_NTHREADS=8`, `NCCL_NSOCKS_PERTHREAD=8`**: tried as
  escalations for the 192-rank bootstrap failure; the actual fix was
  the multi-UniqueId change in `stateless_process_group.py`, not any of
  these IB/socket knobs.
- **`NCCL_CUMEM_ENABLE=0` and `NCCL_DEBUG=INFO`**: dropped after the
  refit-hygiene `empty_cache+synchronize` changes obsoleted CUMEM=0,
  and the success path no longer needs the debug noise. Re-add CUMEM=0
  if `ncclUnhandledCudaError on cuMem buffer registration` reappears.

## Operational notes

### CUDA driver compatibility

DSv4's hyper-connection path has a fused proj-RMS cuTile kernel
(`use_fused_mhc=True`) that requires CUDA driver >= 13.0. On hosts with
older drivers the kernel launch fails or, in cross-rank settings,
silently hangs and trips the NCCL watchdog.

Workaround: set `NRL_DSV4_DISABLE_FUSED_MHC=1` (already in the recipe's
`env_vars`). The unfused native path lives in Megatron-LM's
`hyper_connection.py` and is correct, just slower.

`nemo_rl/models/megatron/community_import.py::_apply_dsv4_overrides`
honors this env var during HF→Megatron checkpoint conversion. It also
unconditionally sets `dsa_indexer_loss_coeff=0.0` because the
Megatron-Bridge provider leaves it at `None` and the DSA forward
TypeErrors on `None * tensor`.

### Wall-clock per step

At ~5.5 min/step in steady state, 1000 GRPO steps take roughly 92
wall-clock hours. The default `WALL_TIME=2:00:00` in the launcher is
intentionally short for verification runs. Bump it (e.g. `12:00:00`)
for production training.

### Scaling further

To run on >32 nodes, increase `NUM_NODES` in the launcher (which
overrides `cluster.num_nodes` via Hydra) and keep
`colocated.resources.num_nodes=8` (the inference side). Verify two
divisibility constraints before launch:

1. `world_size_train / (TP_train * PP_train * EP_train) >= 1` and
   `world_size_train` is a multiple of `TP_train * PP_train * EP_train`
   so DP is an integer.
2. `num_prompts_per_step * num_generations_per_prompt` is divisible by
   both the new train DP count and the vLLM replica count.

A higher node count buys more train DP (faster gradient step) but does
not help vLLM throughput unless inference nodes are added too.
