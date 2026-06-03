"""Numerical-equivalence test for the chunked DSA indexer.

Compares the EDITED module (query-chunked _compute_index_scores +
bwd_fused_indexer_loss_naive) against the ORIGINAL implementation extracted from
git HEAD into /tmp/dsa_orig_src.py. This isolates the chunking change from the
(unchanged) upstream KL math, so any difference is attributable to chunking alone.

The chunked path is driven by MCORE_DSA_INDEXER_CHUNK_SIZE (set per-call below).
"""
import importlib.util
import os

import torch


def _load_original():
    spec = importlib.util.spec_from_file_location(
        "dsa_orig",
        "/lustre/fs1/portfolios/coreai/projects/coreai_mlperf_training/users/mfutrega/RL/dsa_orig_src.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    import megatron.core.transformer.experimental_attention_variant.dsa as dsa
    orig = _load_original()

    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Non-power-of-two sq/sk so chunking with a non-divisor (7) hits a ragged last chunk.
    sq = sk = 23
    b, h, d = 2, 4, 8       # batch, index_n_heads, index_head_dim
    topk = 5
    np_heads, hn = 3, 6      # main attention heads / head dim (drive the KL target)

    # Keep magnitudes modest so softmax/KL stays well-conditioned (avoids spurious NaN
    # in the *shared* KL math, which would mask the comparison we care about).
    q = (torch.randn(sq, b, h, d, device=device) * 0.3).bfloat16()
    weights = (torch.randn(sq, b, h, device=device) * 0.3).bfloat16()
    k = (torch.randn(sk, b, d, device=device) * 0.3).bfloat16()
    query = (torch.randn(sq, b, np_heads, hn, device=device) * 0.3).bfloat16()
    key = (torch.randn(sk, b, np_heads, hn, device=device) * 0.3).bfloat16()
    softmax_scale = hn ** -0.5
    loss_coeff = 1.0
    grad_loss = torch.tensor(1.0, device=device)

    class _PG:
        class _Grp:
            def size(self):
                return 1
        tp = _Grp()
    pg = _PG()

    def mx(a, c):
        return (a.float() - c.float()).abs().max().item()

    # Causal mask [b, sq, sk] (fp32): 0 where key j <= query i, -inf otherwise. This mirrors
    # how CSA actually invokes the indexer (a causal mask is always passed), so the top-k
    # selection respects causality and the sparse-loss path is well-conditioned.
    causal = torch.zeros(sq, sk, device=device)
    causal = causal.masked_fill(
        torch.triu(torch.ones(sq, sk, device=device, dtype=torch.bool), diagonal=1),
        float("-inf"),
    )
    mask = causal.unsqueeze(0).expand(b, sq, sk).contiguous()

    # ---------------- forward ----------------
    os.environ["MCORE_DSA_INDEXER_CHUNK_SIZE"] = "0"
    ref_scores = orig._compute_index_scores(q, weights, k)
    assert torch.isfinite(ref_scores).all(), "reference forward produced non-finite values"

    for cs in ("0", "1", "7", "1024"):
        os.environ["MCORE_DSA_INDEXER_CHUNK_SIZE"] = cs
        got = dsa._compute_index_scores(q, weights, k)
        e = mx(ref_scores, got)
        print(f"[forward] chunk={cs:>4}  max|orig-edited|={e:.3e}")
        assert e == 0.0, f"forward mismatch at chunk={cs}: {e}"

    # topk indices from causal-masked full scores (as fused_qk_topk_naive does)
    topk_indices = (ref_scores + mask).topk(min(topk, sk), dim=-1)[1]

    # ---------------- backward ----------------
    def bwd(mod, cs, sparse_loss):
        os.environ["MCORE_DSA_INDEXER_CHUNK_SIZE"] = cs
        return mod.bwd_fused_indexer_loss_naive(
            q, weights, k, query, key, topk_indices,
            softmax_scale, loss_coeff, sparse_loss, grad_loss, pg,
            causal_mask_override=mask,
        )

    for sparse_loss in (True, False):
        gq_r, gw_r, gk_r = bwd(orig, "0", sparse_loss)
        for t, name in ((gq_r, "grad_q"), (gw_r, "grad_w"), (gk_r, "grad_k")):
            assert torch.isfinite(t).all(), f"reference {name} non-finite (sparse={sparse_loss})"

        for cs in ("1", "7", "1024"):
            gq, gw, gk = bwd(dsa, cs, sparse_loss)
            eq, ew, ek = mx(gq_r, gq), mx(gw_r, gw), mx(gk_r, gk)
            print(
                f"[backward sparse={sparse_loss} chunk={cs:>4}] "
                f"grad_q={eq:.3e} grad_w={ew:.3e} grad_k={ek:.3e}"
            )
            # grad_q / grad_w are per-query-row -> bit exact.
            assert eq == 0.0 and ew == 0.0, f"grad_q/grad_w mismatch chunk={cs}"
            # grad_k sums over the query dim in a different float reduction order ->
            # allow a tiny fp32 tolerance.
            assert ek < 1e-3, f"grad_k mismatch chunk={cs}: {ek}"

    print("ALL CHUNKED-INDEXER EQUIVALENCE CHECKS PASSED")


if __name__ == "__main__":
    main()
