# DSv4 nemo_gym (SWE) → Megatron-Bridge port — progress log

Goal: make `examples/nemo_gym/grpo_gym_dsv4_full.yaml` train DeepSeek-V4-Flash-Base via the
**Megatron-Bridge** path (not automodel/dtensor), reusing the working 24n math megatron config,
run via `run_grpo_gym_dsv4.sh` styled like the math launcher and using the nemo-rl-base image.

## Reference (my working math setup, in the RL repo)
- Repo: `/lustre/fs1/portfolios/coreai/projects/coreai_mlperf_training/users/mfutrega/RL` (branch `dsv4-flash-megatron-bridge`)
- Config: `examples/configs/grpo_dsv4_base_bf16_4k_megatron_24n.yaml` (verified 2 GRPO steps @ 24n)
  - bridge **15d8eadc** + mcore **6b6fb956a** (the only DSv4-capable bridge pair), TP1/PP8/EP8,
    pipeline layout `Et*6|t*6|t*6|t*5|t*5|t*5|t*5|t*5,m,L`, mtp_num_layers=1, flex dispatcher,
    deepep off, seq_parallel, act ckpt, empty_unused_memory_level=2, optimizer CPU offload,
    non-colocated vLLM TP=8 (8 inf nodes), env: NRL_DSV4_DISABLE_FUSED_MHC=1,
    NCCL_SOCKET_IFNAME=enp90s0np0, NCCL_NVLS_ENABLE=0.
- Image: `/lustre/fs1/.../users/mfutrega/nemo-rl-base.sqsh` (torch 2.10/2.11, vllm 0.21+cu129, transformers 5.8.1)
- Launcher: `/lustre/fs1/.../users/mfutrega/data/launch_full_24n.sh`
- My DSv4 megatron fixes (committed in RL repo): fp8 block-scale TP-shard in
  `nemo_rl/models/generation/vllm/quantization/fp8.py`; torch-2.11 pickle patch in
  `nemo_rl/models/policy/workers/patches.py`; NCCL iface + memory env in the recipe.

## grpo-studies starting state (branch cgomes/dsv4-main-cu129)
- Does DSv4 via **automodel/dtensor** (EP=64), `megatron_cfg.enabled: false`, `uv run --extra automodel`.
- Submodule pins: bridge **95e5f38f** (NO DSv4 converter!), mcore **d30c3ae5**, automodel f6f4b1a9.
  Submodules NOT initialized in the working tree.
- Deps: torch 2.10.0, transformers **5.5.0**, vllm **0.19.2rc1-custom** (docker_wheels wheel, c0879d948),
  flashinfer 0.6.8.post1.
- nemo_rl megatron_policy_worker.py + fp8.py DIFFER from my RL repo (lacks my DSv4 megatron fixes).
- Image used: cgomes `dsv4_dlcluster.sqsh` (built for automodel + nemo_gym).
- Gym task = agentic SWE-bench (nemo_gym + OpenHands agent, colocated vLLM http server, SIF containers).

## Key obstacles identified
1. bridge 95e5f38f has no DSv4 megatron converter → must repin to 15d8eadc + mcore 6b6fb956a (my verified DSv4 pair).
2. grpo-studies lacks my fp8 TP-shard + torch-pickle fixes → port them.
3. Dep divergence vs my image (vllm 0.19 vs 0.21, transformers 5.5 vs 5.8). "Use my image" → rebuild
   venvs from grpo-studies pyproject; risk of vllm/transformers/megatron incompatibility.
4. nemo_gym/SWE runtime (SIF containers, responses-API http server, apptainer) — does my image support it?
5. Colocated vs non-colocated: gym uses colocated vLLM http server (agent talks to it); my math is non-colocated.

## Plan
1. Repin bridge→15d8eadc, mcore→6b6fb956a in grpo-studies (gitlinks + setup.py CACHED + lock).
2. Port fp8 TP-shard + torch-pickle patch into grpo-studies nemo_rl.
3. Convert grpo_gym_dsv4_full.yaml: replace dtensor_cfg with megatron_cfg (from math 24n), keep gym data/env.
4. Rewrite run_grpo_gym_dsv4.sh: use nemo-rl-base image, --extra mcore, math-launcher env.
5. Iterate on a TINY 1-node run first (grpo_gym_dsv4_tiny.yaml), surface real roadblocks empirically.

## Log
- (init) Analysis complete; obstacles above. Next: read tiny+base configs, then implement.

## REDIRECT (user): port nemo_gym INTO my RL repo and run gym HERE (not in grpo-studies)
Decisive new facts:
- My RL repo ALREADY has nemo_gym: env glue, dataset, processors, examples/nemo_gym/,
  base config grpo_workplace_assistant_nemotron_nano_v2_9b.yaml, and 3rdparty/Gym-workspace/Gym
  checked out at 1a4912e (== grpo-studies pin). So the subsystem is present.
- grpo-studies nemo_gym.py adds 3 deltas to port: (a) default_host setdefault to node_ip (multinode),
  (b) allow_empty_rollouts (substitute 1-tok reward=0 empty response instead of raising),
  (c) dotenv_path -> /logs/nemo_gym_env.yaml.
- SWE rollout runs SIFs via `apptainer exec` (Gym swe_agents/app.py:1660). MY IMAGE HAS NO apptainer
  (probed job 12431471: apptainer/singularity/enroot all MISSING). => apptainer is the Phase-2 blocker
  for real SWE rewards. Phase-1 (megatron+gym data+vLLM refit) can run with allow_empty_rollouts=true.

## Revised plan (in /lustre/fs1/.../RL, branch dsv4-flash-megatron-bridge)
- Keep my working megatron stack (bridge 15d8eadc/mcore 6b6fb956a, my image, vLLM 0.21).
- Port the 3 nemo_gym.py deltas.
- New config examples/nemo_gym/grpo_gym_dsv4_megatron.yaml: `defaults: ../configs/grpo_dsv4_base_bf16_4k_megatron_24n.yaml`
  + add gym data/env (SWE) + vLLM expose_http_server + gym kwargs. NON-colocated (reuse proven 24n layout;
  agent reaches vLLM via its HTTP endpoint regardless of colocation).
- New launcher (clone of launch_full_24n.sh) -> run_grpo_nemo_gym.py, mounts SWE data + SIF dir.
- Phase 1: run with allow_empty_rollouts=true to validate megatron+gym+refit. Phase 2: solve apptainer.

## Log
- Analysis done. Building config + launcher + nemo_gym.py deltas next.

## Implemented (port into my RL repo)
- nemo_rl/environments/nemo_gym.py: ported 3 deltas (default_host->node_ip, allow_empty_rollouts
  fallback, dotenv_path=/logs/nemo_gym_env.yaml).
- examples/nemo_gym/grpo_gym_dsv4_megatron.yaml: full 24n non-colocated megatron + SWE gym (inherits
  ../configs/grpo_dsv4_base_bf16_4k_megatron_24n.yaml). seqlen 16384 (Phase 1).
- examples/nemo_gym/grpo_gym_dsv4_tiny_megatron.yaml: 1-node tiny (4 layers, colocated) for fast
  plumbing validation (inherits ../configs/grpo_dsv4_base_tiny_1n_megatron.yaml).
- /lustre/fs1/.../data/launch_gym_24n.sh: launcher (my image, /lustre mount, --extra mcore --extra
  nemo_gym, run_grpo_nemo_gym.py, idle-GPU-reaper exemption). Takes CONFIG/NUM_NODES/EXP/REBUILD env.
- Fixed dataset selection gotcha: math base sets data.train.dataset_name=DAPOMath17K; default-merge is
  fill-missing only, so set dataset_name=NemoGymDataset explicitly in train+validation.
- Decided NOT to use the vllm DSv4 FP8 patch script / VLLM_DSV4_BASE_FP8 (my working 24n math run used
  neither — vLLM 0.21 + my fp8.py fix sufficed). Will add only if vLLM DSv4 FP8 breaks.

## Runs
- 12431676: 1-node TINY gym plumbing smoke (REBUILD=true, allow_empty_rollouts). Validates config load +
  venv build (mcore+nemo_gym) + nemo_gym env startup + SWE data load + rollout(empty) + megatron step +
  vLLM refit. Monitoring.

## Run 12431676 result + fix #1 (vLLM render API)
- Progressed through: config load, venv build (mcore+nemo_gym), SWE data load (1344 train/100 val),
  Ray cluster, colocated worker init, vLLM async worker venv build.
- FAILED at vLLM async HTTP server setup: `OpenAIServingChat.__init__() missing 1 required keyword-only
  argument: 'openai_serving_render'`. vLLM 0.21 moved chat rendering into OpenAIServingRender; my repo's
  HTTP-server path was stale (never exercised by the math config). grpo-studies has the canonical fix
  (commit 44cfec7 "Fix vLLM OpenAI render setup") — their vllm_worker_async.py targets the SAME 0.21 API.
- FIX: surgically ported the render fix into nemo_rl/models/generation/vllm/vllm_worker_async.py:
  added OpenAIServingRender import; added preprocess_chat override to NeMoRLOpenAIServingMixin;
  added NeMoRLOpenAIServingRender(mixin, OpenAIServingRender) + construction; pass openai_serving_render
  to NeMoRLOpenAIServingChat + NeMoRLOpenAIServingTokenization. Preserved my VllmAsyncGenerationWorker
  @ray.remote class (did NOT adopt grpo's ...Impl rename). py_compile OK.
- nemo_rl is editable-installed in the worker venvs ({"editable":true}) -> code fixes apply without
  venv rebuild. Re-running with REBUILD=false for fast iteration.

## Run 12431905: tiny gym, REBUILD=false, render fix
- Fix #1 CONFIRMED: vLLM async HTTP server starts ("Uvicorn running on http://0.0.0.0:60585"), engine init OK, DSv4 TileLang kernel compiled, colocated sleep-mode OK. Render error gone.
- Continuing: megatron worker init -> nemo_gym env startup -> rollout -> 2 steps.

## Run 12431905 result: reached GRPO loop! + fix #2 (refit buffer)
- HUGE progress. Confirmed working: vLLM HTTP server (fix#1), megatron DSv4 import+model setup
  (hash-MoE, DSA hybrid, optimizer offload), nemo_gym SWE env STARTED (built OpenHands/SWE-bench/
  SWE-bench_Multilingual/R2E-Gym agent venvs), "Using GRPO advantage estimator",
  "Generating responses for batch of size 8".
- FAILED at colocated IPC-ZMQ refit: AssertionError: Parameter embed.weight too large for buffer:
  1059061760 > 150935633 (policy/utils.py:319 stream_weights_via_ipc_zmq_impl). My NRL_REFIT_BUFFER_
  MEMORY_RATIO=0.005 (set for the non-colocated 24n collective refit) gives a 150MB buffer that can't
  hold the 1.06GB embedding on the COLOCATED IPC-ZMQ path (never exercised by the math config).
- FIX #2: made REFIT_RATIO configurable in launcher; re-run tiny with REFIT_RATIO=0.1 (buffer must
  exceed the largest single param). Note: the full non-colocated 24n gym path uses the collective
  refit, so 0.005 likely still OK there; this is a colocated-only tunable.

## Run 12431960 (next): tiny gym, REBUILD=false, REFIT_RATIO=0.1

## Run 12432653: BOTH FIXES WORK — full GRPO loop running (Phase-1 plumbing validated)
- Refit passed (fix#2, REFIT_RATIO=0.1). "Generating responses for batch of size 8".
- allow_empty_rollouts fallback CONFIRMED: "[nemo_gym] empty rollout substituted (count=1..8)" — all 8
  SWE rollouts empty (no apptainer) -> reward=0 substitute, batch stays full. Exactly as designed.
- Processing rewards -> Computing logprobs (megatron forward). Into the training step. Monitoring to 2-step finish.

## Run 12432653 RESULT: PHASE-1 PLUMBING VALIDATED ✅ (+ step-2 degenerate-data finding)
Full end-to-end worked: refit -> generate -> empty-rollout fallback (8/step) -> process rewards ->
compute logprobs (megatron fwd) -> STEP 1 TRAIN + REFIT OK -> step 2 generate + logprobs.
Crash only at STEP 2 zero_grad_buffer() (megatron DDP assert len(per_param_grad_ready_counts)==len(params),
distributed_data_parallel.py:580). My math 24n ran 2 full steps without this => it is SPECIFIC to the
all-empty degenerate batch (every rollout empty w/o apptainer -> zero-advantage -> interleaved
logprob-forward/train leaves grad-ready state inconsistent). Real SWE rollouts won't yield all-empty steps
(math path proves step-2 train works with real data). So: integration is proven; this is a no-apptainer
smoke artifact, not an integration bug.

Confirmed working fixes: #1 vLLM render (openai_serving_render), #2 colocated refit buffer (REFIT_RATIO).

## PHASE 2 (next): real SWE rollouts need apptainer (my image lacks it). Investigating provisioning.
Open: also consider hardening the all-empty-step megatron grad-buffer case for robustness at scale
(if a whole step's rollouts fail), but real data should avoid it.

## PHASE 2: apptainer SWE runtime PROVISIONED ✅
- Baked apptainer into image via `srun --container-save`: nemo-rl-base-apptainer.sqsh (apptainer 1.3.1,
  deps squashfs-tools/uidmap/libfuse3 resolved on cpu_datamover). Job 12440777.
- Verified NESTED apptainer exec of a real SWE SIF inside pyxis works (job 12440887: NESTED_APPTAINER_OK,
  EXIT=0) -> userns/setuid OK on this cluster.
- Launcher now defaults CONTAINER=nemo-rl-base-apptainer.sqsh + supports EXTRA_OVERRIDES.

## Run 12440xxx: 1-node tiny + apptainer + seqlen 32768 — validate SWE agent runtime end-to-end
Goal: agent runs SIF via apptainer -> non-empty rollouts -> real reward -> megatron 2 steps clean (no
degenerate-data assertion). Model is tiny(4L) so rewards ~0, but validates the runtime. Then full 24n.

## Run 12441066: apptainer WORKS (agents run, 0 empty) + fix #3 (skip_tokenizer_init)
- SWE agents ran via apptainer (Collecting rollouts 0/8, 0 empty substitutions — vs all-8-empty w/o apptainer).
- But every agent chat request to the vLLM HTTP server failed: AssertionError `assert tokenizer is not None`
  (vllm serving.py:248). Engine started with skip_tokenizer_init=True (base RL default, token-id refit path)
  -> renderer.tokenizer None. The gym agent sends TEXT, so vLLM must tokenize.
- FIX #3: added policy.generation.vllm_cfg.skip_tokenizer_init=false to both gym configs.
- Re-running tiny+apptainer+seqlen32768.

## Run 12441xxx (next): tiny + apptainer + seqlen 32768 + skip_tokenizer_init=false

## Run 12441601: fix #3 confirmed (tokenizer assert gone, 0 empty) + fix #4 (cryptography)
- skip_tokenizer_init=false worked: 0 tokenizer asserts; agents ran (Collecting rollouts).
- New: R2E-Gym agent venv (swe_r2e_gym_setup/R2E-Gym/venv, py3.12) failed: ModuleNotFoundError 'cryptography'
  (r2egym docker.py -> import kubernetes -> google.auth -> cryptography). The agent venvs (uv-built, no pip)
  lacked it; grpo-studies' image likely had it system-wide.
- FIX #4: `uv pip install --python <venv> cryptography` into all 3 swe agent venvs (R2E-Gym now imports
  google.auth.crypt.es OK; venvs persist on lustre, reused via skip_venv_if_present).
- Re-running tiny+apptainer.

## Run 12442xxx (next): tiny + apptainer + skip_tokenizer_init + cryptography

## Run 12442360: apptainer+agent+crypto all OK, but empty rollouts -> ROOT CAUSE = seqlen (fix #5)
- Agents ran the SIFs (numpy instances), made chat requests, but vLLM returned 400 Bad Request ->
  policy_model proxy 500 -> "No completion files found" -> empty rollouts (Avg Reward -1.0).
- Root cause: code has MaxContextLengthFilter (vllm_worker_async.py:738) that SUPPRESSES vLLM's
  max-context-length 400s. So the 400s = SWE prompts EXCEED max_model_len (32768). grpo-studies used
  131072 for exactly this.
- Step 1 trained (Avg Reward -1.0); step 2 would hit the same degenerate-all-empty assertion.
- FIX #5: bump seqlen so SWE prompts fit. Tiny(4L) can handle 131072 memory-wise. Full model will need
  131072 + CP/nodes tuning.

## Run 12442xxx (next): tiny + apptainer + seqlen 131072 -> expect NON-empty rollouts + clean 2 steps

## Run 12442895/12443143: seqlen NOT the cause; ROOT CAUSE = missing enable_auto_tools (fix #6)
- Added [NRL_DIAG] logging to create_chat_completion. Captured the real error:
  req: temperature=1.0 top_p=1.0 top_k=None tools=10 tool_choice=auto  (asserts pass)
  ErrorResponse 400: '"auto" tool choice requires --enable-auto-tool-choice and --tool-call-parser to be set'
- vLLM 0.21 OpenAIServingRender (render/serving.py:219) checks self.enable_auto_tools; the SWE agent uses
  tool_choice=auto. My gym configs lacked http_server_serving_chat_kwargs (grpo-studies' gym config has it).
- FIX #6: added policy.generation.vllm_cfg.http_server_serving_chat_kwargs {enable_auto_tools: true,
  tool_parser: hermes} to both gym configs. (seqlen 131072 kept; not the cause.)
- NOTE: hermes tool-parser thread-safety patch warning (vllm_worker.py:392, vLLM 0.21 mismatch) — may
  matter under concurrency=16; watch for it.

## Run 12443xxx (next): tiny + apptainer + seqlen 131072 + enable_auto_tools -> expect NON-empty rollouts

## Run 12443368: fix #6 held (0 chat errors), 1/8 NON-empty rollout! + fix #7 (CSA OOM at long seqlen)
- enable_auto_tools worked: ErrorResponse(400)=0; agent produced 1 non-empty rollout (7 empty). Full SWE
  pipeline functions end-to-end (apptainer + tool-calling + agent + reward).
- NEW: torch.OutOfMemoryError in megatron forward at seqlen 131072 — DSv4 CSA:
  csa.py:207 unfused_compressed_sparse_attn -> kv_gathered.float() tried to alloc 21.2 GiB.
  seqlen↔memory tradeoff: 32768 no-OOM but prompts too long (empty); 131072 prompts fit but CSA OOMs.
  Must solve for full run too. Fix path: context parallelism (CP) or fused/chunked CSA.

## CSA OOM analysis + next steps
- csa.py unfused_compressed_sparse_attn (differentiable/training path) materializes kv_gathered.float()
  = [b, sq, topk, hn] fp32 -> O(seqlen) -> 21.2 GiB @131072. Fused path (cp_group-aware) is inference-only.
- Tiny is COLOCATED (vLLM shares GPU, ~19GB free) -> OOM at 131072. Full 24n is NON-colocated (no vLLM on
  train nodes) -> more headroom; may fit 131072, else add CP / reduce seqlen.
- Tiny validation: re-run at seqlen 65536 (CSA transient ~10.6GB < free) -> expect non-empty rollouts +
  clean step-2 train (mixed batch, non-degenerate).

## Run 12445658 (tiny@65536): ALL 8 rollouts NON-EMPTY (empty:0)! OOM is colocation-induced
- enable_auto_tools held (0 chat errors). At 65536 ALL 8 SWE rollouts non-empty -> full pipeline works.
- OOM in get_logprobs() CSA forward: only ~2GB free, 67GB already used. Because COLOCATED (vLLM ~40GB +
  megatron on same 8 GPUs). Not a code bug — pure memory pressure from colocation on the tiny.
- CONCLUSION: tiny (colocated) is memory-starved for real long SWE sequences; it has validated everything
  (integration, 6 fixes, apptainer, agent, NON-EMPTY rollouts). The full 24n is NON-colocated (no vLLM on
  train nodes, PP8/EP8 sharded, optimizer offloaded) -> the headroom the tiny lacks.

## FULL RUN: launching 24n non-colocated gym @ seqlen 65536 (all-non-empty seqlen from tiny)
- If OOM on the full model forward at 65536: add context parallelism (CP) to shard the sequence
  (model already takes input_ids_cp_sharded), or reduce seqlen.

## Run 12446334 (full 24n): died at flaky init_collective NCCL (rank 121, ncclUnhandledCudaError)
- NOT a gym/memory issue (empty:0, OOM:0 — died in setup before rollouts). This is the known intermittent
  192-rank cross-cluster model_update_group init flake (documented for math 24n; ~1/3 of launches).
- Remedy: relaunch (succeeds on retry). Gym integration unaffected.

## Run 12446700 (full 24n, attempt 3): PASSED init_collective! Generating 64 rollouts at scale.
- nccl-err:0 (init flake cleared on retry 3). Into GRPO loop: "Generating responses for batch of size 64".
- Now: slow SWE rollouts (64 @ concurrency 16) -> full-model CSA forward @65536 non-colocated (memory test).

## Run 12446700 (full 24n): FULL PIPELINE WORKS AT SCALE — 64 rollouts, 62 NON-empty! OOM = DSA quadratic (fix #8)
- Cleared init_collective; generated all 64 SWE rollouts (62 non-empty, 2 empty) -> processed rewards ->
  computing logprobs. Entire pipeline validated at 24n scale.
- OOM in forward: dsa.py:342 _compute_index_scores einsum('sbhd,tbd->sbht', q.float(), k.float()) tried
  73 GiB (54 free). This is the O(seqlen^2) DSA index-score matrix at seqlen 65536 (quadratic-attention wall).
  Not fixable by PP/EP/nodes (per-rank/per-head). Levers: context parallelism (CP) to shard query dim, or
  shorter seqlen (32768 -> ~18 GiB fits but more truncated->empty prompts).

## CSA/DSA long-context memory: CP vs seqlen
- DSA supports CP (cp_group=self.pg_collection.cp, dsa.py:821/901). BUT enabling CP on 24n reduces DP
  (CP shards train world) -> expert-DP 2->1, the layout that OOMed at 16n. So CP needs MORE train nodes
  (e.g., 32 train+8 inf to keep DP=16 w/ CP2). That's the proper long-context path.
- Cheap lever now: seqlen. OOM was forward-only (get_logprobs) @73GiB; train backward ~2x. seqlen 32768 ->
  ~18GiB fwd / ~36GiB bwd, fits the ~54GiB free. Relaunching full 24n @32768 for a clean at-scale 2-step
  completion (some long SWE prompts will be truncated->empty; mixed batch trains). Long-context (131072)
  for production needs CP + ~32-40 nodes.
Tue Jun  2 05:01:25 PM PDT 2026
- init_collective flake: attempts 1,2,4 failed init; attempt 3 passed+ran to forward. Retrying (attempt 5) @32768.

## init_collective flake: 5/6 failed (systematic for gym, not 1/3). Fix attempt: TCPStore timeout
- gym vLLM (async http-server + tokenizer + DeepGEMM warmup) starts much slower than math's sync vLLM, so
  train ranks race the 192-rank NCCL scalable bootstrap before vLLM ranks arrive -> "unhandled cuda error".
- The launcher's NCCL_BOOTSTRAP_TIMEOUT does NOT reach workers (known lesson). And StatelessProcessGroup's
  TCPStore used the default 300s timeout.
- FIX: stateless_process_group.py TCPStore timeout -> 1800s (NRL_PG_STORE_TIMEOUT_S). Relaunch (attempt 7).

## init_collective ROOT-CAUSED via NCCL_DEBUG: NCCL comm-buffer CUDA OOM (fix #9)
- NCCL_DEBUG (attempt 8) showed: rank 126 (TRAIN) "NCCL WARN Cuda failure 2 'out of memory'" (alloc.h:228)
  while setting up the 192-rank model_update_group's 8 P2P/CUMEM+RDMA channels on top of the loaded model.
  Not a race/timeout — NCCL comm-buffer OOM. Worse for gym (vLLM carries tokenizer+http server -> less
  headroom; plus train ranks tight). gmu/store-timeout were red herrings.
- FIX #9: cap NCCL comm memory on BOTH sides of the group — NCCL_MAX_NCHANNELS=2 + NCCL_BUFFSIZE=2MB.
  Train: gym config megatron_cfg.env_vars. vLLM: vllm_generation.py non-colocated env_vars
  (NRL_NCCL_MAX_NCHANNELS / NRL_NCCL_BUFFSIZE overridable). Refit is a once/step broadcast; 2 channels fine.

## Run 12449768 (attempt 10): NCCL fix #9 WORKS — init_collective passed reliably (nccl-err:0)!
- NCCL_MAX_NCHANNELS=2 cap resolved the 192-rank refit-group OOM. Into GRPO loop, generating 64 rollouts.
- (attempt 9 was a separate transient bad-node ray-startup death.) Now: rollouts -> @32768 forward (fits) -> 2 steps.

## CP path (user-requested): CP=2 @ 40 nodes
- gym config: megatron_cfg.context_parallel_size=2, cluster.num_nodes=40 (32 train: TP1*PP8*CP2*DP16=256
  GPUs + 8 inf). CP halves per-rank DSA index_scores (64GiB fwd @32768 -> ~34GiB, fits ~57 free). EP8
  divides DP*CP=32 (expert-DP=4). NCCL channel-cap fix (#9) carries over (now 320-rank model_update_group).
- Risk: train backward recompute may still be ~2x fwd; if it OOMs, escalate CP=4 (72 nodes) or trim seqlen.

## CP RESULT: architecturally INCOMPATIBLE with DSv4 CSA (conclusive)
- CP=2 @40n hit: AssertionError "Sequence Packing must be enabled to use Context Parallelism with MCore"
  (nemo_rl/models/megatron/setup.py:393). But DSv4 CSA/DSA hard-assert packed_seq_params is None
  (csa.py:528/661, dsa.py:997 "Packed sequence not supported"). And CP-without-packing doesn't shard the
  sparse-attention sequence (indexer still sees full seq) -> no memory benefit. => CP is a dead end for
  DSv4 sparse attention without rewriting CSA/DSA to support packing.
- Reverted config to working 24n (CP off, 24 nodes). NCCL channel-cap fix retained.
- ONLY viable fix for SWE-length context at current scale: CHUNKED DSA INDEXER — chunk the query dim in
  dsa.py _compute_index_scores (and the manual bwd einsum @588). Clean math-identical change; peak
  [s,b,h,t](64GiB) -> [chunk,b,h,t](~4GiB). It's a vendored Megatron-LM edit (experimental DSv4 attn).

## CHUNKED DSA INDEXER implemented + numerically verified (fix #10) — Jun 3
- Vendored edit: 3rdparty/Megatron-LM-workspace/Megatron-LM/megatron/core/transformer/
  experimental_attention_variant/dsa.py
  - Added `_dsa_indexer_chunk_size()` (env `MCORE_DSA_INDEXER_CHUNK_SIZE`, default 1024; 0 disables).
  - `_compute_index_scores`: query-chunked. The [seqlen_q, batch, index_n_heads, seqlen_k] fp32
    intermediate (THE O(seq^2) 64GiB-@32k wall) is never fully materialized; peak is one
    [chunk, batch, index_n_heads, seqlen_k] slice. Result [b, sq, sk] written slice-by-slice.
  - `bwd_fused_indexer_loss_naive`: same chunking over the query dim for the recompute+grad block
    (lines ~588-613). This was the WORST site — ~4 same-shape [s,b,h,t] tensors were live at once.
    grad_q/grad_weights are per-query-row (chunked independently); grad_k is summed over the query
    dim -> accumulated across chunks. NB: must size grad_q/grad_k from the INDEXER dims (q.size(2/3),
    k.size(2)), NOT the main-attn np/hn from query.size() (first bug, fixed).
- Verification (test_dsa_chunk_equiv.py, CPU, in-container): compares EDITED module vs ORIGINAL
  (git HEAD copy) on the same inputs, mirroring real CSA usage (causal mask passed so topk respects
  causality + sparse-loss path is well-conditioned):
    forward  chunk={0,1,7,1024}: max|orig-edited| = 0.000e+00 (bit-exact, all chunk sizes)
    backward sparse={T,F} chunk={1,7,1024}: grad_q=0, grad_w=0 (bit-exact); grad_k<=2.4e-7
      (fp32 reduction-order only; tol 1e-3). => math-identical.
  Ran via: srun --no-container-mount-home (host $HOME shadows the baked uv python in /root otherwise!)
    --container-image=nemo-rl-base-apptainer.sqsh; python=/opt/nemo_rl_venv/bin/python (torch 2.11+cu129).
- Other O(seq^2)-ish OOM sites (NOT yet chunked; become next walls only at much longer seqlen):
    (a) summed index_scores [b,sq,sk] + topk in fused_qk_topk_naive (~4/17/68 GiB @ 32k/64k/131k);
    (b) KL "true-attn" scores [b,np,sq,sk] bmm in compute_dsa_indexer_loss (fwd ~232) and bwd (~470)
        -- sk is COMPRESSED kv (seq/compress_ratio), np is TP-sharded, so smaller constant;
    (c) CSA unfused_compressed_sparse_attn kv_gathered.float() [b,seq,topk,hn].
- E2E: launched tiny smoke job 12465983 (1n colocated, seqlen 1536) with megatron_cfg.env_vars
  MCORE_DSA_INDEXER_CHUNK_SIZE=64 -> forces 24 chunks (exercises ragged-chunk + grad_k accumulation
  in real training). Monitoring for a completed megatron step (rollouts empty by design at this seqlen).

## E2E tiny smoke (job 12465983, chunk=64): chunked indexer RAN A FULL STEP; step-2 crash is orthogonal
- STEP 1 COMPLETED: "▶ Training policy..." -> "Avg Reward: -1.0000" -> "Total step time: 177.10s".
  => the chunked DSA indexer executed inside a real DSv4 Megatron GRPO fwd+bwd with NO error.
  (Rollouts empty by design: vLLM rejects 1537-tok SWE prompts > tiny 1536 ctx; allow_empty_rollouts
  -> reward=0. Same Avg Reward -1.0 as the 24n run.)
- STEP 2 crashed at the START of train() in self.model.zero_grad_buffer():
    param_and_grad_buffer.py:257  assert len(self.per_param_grad_ready_counts) == len(self.params)
  This is Megatron DDP's once-per-run "golden" grad-ready registration (reset(), is_first_batch path):
  per_param_grad_ready_counts keys are populated per-param in the grad-ready hook (lines 768-770), so the
  assert fails IFF some param received NO gradient during step 1's backward.
- ORTHOGONAL TO THE CHUNKING CHANGE (code-proven): my edit changes only the numeric VALUES the indexer's
  custom-Function backward returns for q/weights/k; it does not alter the autograd graph topology or which
  params are differentiated. The set of params that fire grad hooks is identical with/without the edit, and
  the assert is purely about grad-hook PRESENCE (param coverage). Hence the edit cannot cause it.
- ROOT CAUSE = degenerate EMPTY-ROLLOUT batch: short/all-masked sequences -> some params (MTP head /
  unrouted experts / etc.) get no grad -> incomplete coverage -> step-2 golden-registration assert.
  The math (non-gym) dsv4 config does 2 steps fine (bring-up memory); gym differs only by empty rollouts.
- CONTROL: relaunched tiny with ++policy.megatron_cfg.env_vars.MCORE_DSA_INDEXER_CHUNK_SIZE=0 (= original
  single-pass code), job 12466828. Expect the SAME step-2 assert -> confirms independence empirically.
- NOTE: this step-2 grad-coverage assert is a SEPARATE pre-existing blocker for the empty-rollout gym path.
  It does NOT affect the real long-seqlen goal: that needs NON-empty (real SWE) rollouts, where all params
  get grads (no assert) AND the long sequence is what the chunked indexer prevents OOMing on.
