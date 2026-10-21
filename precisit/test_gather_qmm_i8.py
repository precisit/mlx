"""K-G1..K-G3 for mx.gather_qmm_i8 (modelopt-to-mlx stage2/A8-KERNEL.md). Run with the fork build:
    MLX_ENABLE_TF32=0 python -I precisit/test_gather_qmm_i8.py
"""
import json, os, sys
import numpy as np
import mlx.core as mx

# The a8-ref reference is stock fp32-x gather_qmm, which MLX runs in TF32 on NAX (M5-class) hosts unless
# MLX_ENABLE_TF32=0; the reference is then not fp32 and K-G3 is meaningless (modelopt-to-mlx amendment 16b, R23).
# Required on every host, so no caller can forget it.
if os.environ.get("MLX_ENABLE_TF32") != "0":
    raise SystemExit("STOP: set MLX_ENABLE_TF32=0 (the fp32 reference is TF32 on NAX hosts otherwise)")

E2M1_X2 = np.array([0, 1, 2, 3, 4, 6, 8, 12], np.int64)


def a8_quantize(x):            # == modelopt-to-mlx stage2/arms.py:a8_quantize
    xf = x.astype(mx.float32)
    s = mx.max(mx.abs(xf), axis=-1, keepdims=True) / 127.0
    q = mx.clip(mx.round(xf / mx.where(s == 0, mx.ones_like(s), s)), -127, 127)
    return q.astype(mx.int8), s


def w2_int(wq):                 # uint32 packed nvfp4 -> 2*e2m1 as int64 [E, N, K]
    b = np.array(wq).view(np.uint8)
    nib = np.stack([b & 0x0F, b >> 4], -1).reshape(*b.shape[:-1], -1)
    v = E2M1_X2[nib & 7]
    return np.where(nib & 8, -v, v)


def e4m3(s):
    return np.array(mx.from_fp8(mx.array(np.array(s)), dtype=mx.float32)).astype(np.float64)


def reference(q, xs, wq, ws, rows_e):   # exact float64 of the defining formula; rows_e: expert per output row of y
    qn = np.array(q).astype(np.int64).reshape(-1, q.shape[-1])
    W2, S = w2_int(wq), e4m3(ws)
    K = qn.shape[-1]; nb = K // 16
    out = np.empty((qn.shape[0], W2.shape[1]), np.float64)
    parts = []
    for r in range(qn.shape[0]):
        e = rows_e[r]
        P = (qn[r].reshape(nb, 16)[None] * W2[e].reshape(-1, nb, 16)).sum(-1)       # [N, nb] exact int64
        parts.append(np.abs(P).max())
        out[r] = 0.5 * float(np.array(xs).reshape(-1)[r]) * (P * S[e]).sum(-1)
    return out, max(parts)


def case(name, K, N, E, shape_kind, unit_scales, rng):
    W = mx.array(rng.standard_normal((E, N, K)).astype(np.float32) * 0.05)
    wq, ws = mx.quantize(W, group_size=16, bits=4, mode="nvfp4")[:2]
    if unit_scales:
        ws = mx.full(ws.shape, 0x38, dtype=mx.uint8)          # E4M3 1.0
    if shape_kind == "decode":                                  # mlx-vlm unsorted decode: x [B,L,1,1,K], idx [B,L,k]
        B, L, k = 2, 1, 10
        x = mx.array((rng.standard_normal((B, L, 1, 1, K)) * np.exp(rng.standard_normal((B, L, 1, 1, K)))).astype(np.float32))
        idx = mx.array(rng.integers(0, E, (B, L, k)).astype(np.uint32)); sort = False
        rows_e = np.array(idx).reshape(-1); xrow = np.repeat(np.arange(B * L), k)
    else:                                                        # sorted prefill: x [P,1,K], idx [P] ascending
        Pn = 300
        x = mx.array((rng.standard_normal((Pn, 1, K)) * np.exp(rng.standard_normal((Pn, 1, K)))).astype(np.float32))
        idx = mx.array(np.sort(rng.integers(0, E, Pn)).astype(np.uint32)); sort = True
        rows_e = np.array(idx); xrow = np.arange(Pn)
    q, xs = a8_quantize(x)
    if unit_scales:
        xs = mx.ones_like(xs)
    y = mx.gather_qmm_i8(q, xs, wq, ws, rhs_indices=idx, sorted_indices=sort)
    yn = np.array(y).reshape(-1, N).astype(np.float64)
    qrows = np.array(q).reshape(-1, K)[xrow]; xsrows = np.array(xs).reshape(-1)[xrow]
    ref, pmax = reference(mx.array(qrows), mx.array(xsrows), wq, ws, rows_e)
    res = {"case": name, "K": K, "N": N, "shape": shape_kind, "unit_scales": unit_scales, "max_abs_block_partial": int(pmax),
           "rel_l2_vs_exact": float(np.linalg.norm(yn - ref) / np.linalg.norm(ref))}
    if unit_scales:   # K-G2: every partial and their sum are integers < 2^24 -> the fp32 kernel must be exact
        res["bit_exact_vs_int64"] = bool(np.array_equal(yn, ref))
    else:             # K-G3: vs the a8-ref path (stock gather_qmm on the dequantized int8 input, fp32)
        xq = (q.astype(mx.float32) * xs)
        yr = mx.gather_qmm(xq, wq, ws, rhs_indices=idx, transpose=True, group_size=16, bits=4, mode="nvfp4", sorted_indices=sort)
        yrn = np.array(yr.astype(mx.float32)).reshape(-1, N).astype(np.float64)
        res["rel_l2_vs_a8ref"] = float(np.linalg.norm(yn - yrn) / np.linalg.norm(yrn))
        res["a8ref_rel_l2_vs_exact"] = float(np.linalg.norm(yrn - ref) / np.linalg.norm(ref))
        res["max_abs_vs_a8ref_over_max"] = float(np.abs(yn - yrn).max() / np.abs(yrn).max())
    return res


def main():
    rng = np.random.default_rng(7)
    out, ok = [], True
    for K, N, nm in ((2560, 640, "gate/up"), (640, 2560, "down")):
        for kind in ("decode", "sorted"):
            for unit in (True, False):
                r = case(nm, K, N, 16, kind, unit, rng)
                g = r.get("bit_exact_vs_int64", True) and r.get("rel_l2_vs_a8ref", 0) <= 1e-5
                ok &= g
                print(("PASS " if g else "FAIL ") + json.dumps(r))
                out.append(r)
    print("K-G2/K-G3:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
