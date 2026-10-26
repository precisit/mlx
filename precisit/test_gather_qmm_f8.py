"""F-G1..F-G3 (and the F-G4 dump) for mx.gather_qmm_f8 (modelopt-to-mlx stage2/PROTOCOL.md amendment 17, F8 arm).
Run with the fork build:

    MLX_ENABLE_TF32=0 [MLX_GATHER_QMM_F8_ROUTE=auto|vector|nax] python -I precisit/test_gather_qmm_f8.py [--dump OUT.npz]

F-G1  operands: mx.to_fp8 against a NumPy E4M3 reference (round half to even, saturating at +-448) over every fp16
      input (as fp16 and as fp32), every tie between adjacent E4M3 values, and 2^20 random fp32 values; the staged
      weights through the kernel (one-hot x, unit scales: 2y must equal 2*e2m1 exactly).
F-G2  block partials: one nonzero NVFP4 block per row, unit block scales and x_scale 1, so 2y is the kernel's fp32
      block sum; |2y - exact| <= gamma_15 * sum|q * w2| per output, the maximum ratio reported. Includes rows built to
      make the fp32 sum inexact (forced weights on expert 0; the gate also requires that some partials are inexact,
      so it cannot pass vacuously).
F-G3  float closure: relative L2 against f8-ref (fake quantization from_fp8(q) * s in front of stock fp32 gather_qmm,
      TF32 off) <= 1e-5, max-abs reported; gate/up and down shapes, decode (unsorted) and sorted prefill.
--dump writes every case's output for the F-G4 route comparison (compare two dumps with --compare A.npz B.npz).
"""
import argparse, hashlib, json, os, sys
import numpy as np
import mlx.core as mx

if os.environ.get("MLX_ENABLE_TF32") != "0" and "--compare" not in sys.argv:
    raise SystemExit("STOP: set MLX_ENABLE_TF32=0 (the fp32 reference is TF32 on NAX hosts otherwise)")

E2M1_X2 = np.array([0, 1, 2, 3, 4, 6, 8, 12], np.int64)
U = 2.0 ** -24
GAMMA15 = 15 * U / (1 - 15 * U)


def f8_quantize(x):            # == modelopt-to-mlx stage2/arms.py:f8_quantize (amendment 17)
    xf = x.astype(mx.float32)
    s = mx.max(mx.abs(xf), axis=-1, keepdims=True) / 448.0
    q = mx.to_fp8(xf / mx.where(s == 0, mx.ones_like(s), s))
    return q, s


# ---- NumPy E4M3 (fn: no infinities, 0x7f/0xff NaN; saturating encode) ----
def e4m3_values():
    b = np.arange(256)
    s = np.where(b & 0x80, -1.0, 1.0); e = (b >> 3) & 15; m = b & 7
    v = np.where(e == 0, m / 8 * 2.0 ** -6, (1 + m / 8) * 2.0 ** (e - 7.0)) * s
    v[(b & 0x7F) == 0x7F] = np.nan
    return v


def e4m3_encode(v):            # float64 array (finite) -> uint8 codes, RNE, saturating
    a = np.abs(v)
    e = np.floor(np.log2(np.where(a > 0, a, 1.0)))
    e = np.clip(e, -6, 8)
    quantum = 2.0 ** (e - 3)
    r = a / quantum
    n = np.floor(r); frac = r - n
    n = n + ((frac > 0.5) | ((frac == 0.5) & (n % 2 == 1)))
    mag = np.minimum(n * quantum, 448.0)
    vals = e4m3_values()[:0x7F]                       # 0x00..0x7e, increasing
    code = np.searchsorted(vals, mag)
    assert np.array_equal(vals[code], mag)
    return (code | np.where(np.signbit(v), 0x80, 0)).astype(np.uint8)


def fg1_to_fp8():
    res = {}
    h = np.arange(65536, dtype=np.uint16).view(np.float16)
    fin = np.isfinite(h)
    ref = e4m3_encode(h[fin].astype(np.float64))
    for dt in ("float16", "float32"):
        q = np.array(mx.to_fp8(mx.array(h[fin]).astype(getattr(mx, dt))))
        res[f"fp16_inputs_as_{dt}_mismatches"] = int((q != ref).sum())
    nonfin = np.array(mx.to_fp8(mx.array(h[~fin]).astype(mx.float32)))
    res["nonfinite_inputs_codes"] = sorted(set(int(c) & 0x7F for c in nonfin))   # recorded only: activations are finite
    vals = e4m3_values()[:0x7F]
    ties = (vals[:-1] + vals[1:]) / 2
    ties = np.concatenate([ties, -ties]).astype(np.float32)
    assert np.array_equal(ties.astype(np.float64), np.concatenate([(vals[:-1] + vals[1:]) / 2, -(vals[:-1] + vals[1:]) / 2]))
    res["tie_mismatches"] = int((np.array(mx.to_fp8(mx.array(ties))) != e4m3_encode(ties.astype(np.float64))).sum())
    rng = np.random.default_rng(1)
    r = (rng.standard_normal(1 << 20) * np.exp2(rng.uniform(-14, 10, 1 << 20))).astype(np.float32)
    res["random_fp32_mismatches"] = int((np.array(mx.to_fp8(mx.array(r))) != e4m3_encode(r.astype(np.float64))).sum())
    res["pass"] = all(v == 0 for k, v in res.items() if k.endswith("mismatches"))
    return res


def w2_int(wq):                 # uint32 packed nvfp4 -> 2*e2m1 as int64 [E, N, K]
    b = np.array(wq).view(np.uint8)
    nib = np.stack([b & 0x0F, b >> 4], -1).reshape(*b.shape[:-1], -1)
    v = E2M1_X2[nib & 7]
    return np.where(nib & 8, -v, v)


def e4m3_f64(codes):
    return e4m3_values()[np.array(codes).astype(np.int64)]


def weights(rng, E, N, K):
    W = mx.array(rng.standard_normal((E, N, K)).astype(np.float32) * 0.05)
    return mx.quantize(W, group_size=16, bits=4, mode="nvfp4")[:2]


def run(q, xs, wq, ws, idx, sort):
    return mx.gather_qmm_f8(q, xs, wq, ws, rhs_indices=idx, sorted_indices=sort)


DUMP = {}


def fg1_weights(rng):
    out = []
    for K, N in ((256, 128), (128, 256)):
        E = 2
        wq, ws = weights(rng, E, N, K)
        ws = mx.full(ws.shape, 0x38, dtype=mx.uint8)
        q = np.zeros((K, 1, K), np.uint8); q[np.arange(K), 0, np.arange(K)] = 0x38       # E4M3 1.0, one-hot rows
        idx = mx.array(np.full(K, 1, np.uint32))
        y = np.array(run(mx.array(q), mx.ones((K, 1, 1)), wq, ws, idx, True)).reshape(K, N)
        DUMP[f"fg1_w_{K}x{N}"] = y
        ref = 0.5 * w2_int(wq)[1].T.astype(np.float64)                                       # y[k, n] = w2[n, k] / 2
        out.append({"K": K, "N": N, "exact": bool(np.array_equal(y.astype(np.float64), ref))})
    return out


def fg2(rng):
    """Expert 0's first N/2 output rows are built so that its rows' block sums are inexact in fp32: weights +12 at
    block positions 0..14 and +1 at 15, x = 448 at 0..14 and 2^-9 at 15, so the sum is 80640 + 2^-9 (fp32 ulp there is
    2^-7). Rows on expert 0 alternate that block with a mixed-sign version; rows on other experts get random finite codes."""
    out = []
    for K, N in ((2560, 640), (640, 2560)):
        E = 4; nb = K // 16; R = 320
        wq, ws = weights(rng, E, N, K)
        ws = mx.full(ws.shape, 0x38, dtype=mx.uint8)
        wb = np.array(wq).view(np.uint8).copy()                      # [E, N, K/2]; element 2j = low nibble of byte j
        blk = np.array([0x77] * 7 + [0x17], np.uint8)
        wb[0, : N // 2] = np.tile(blk, K // 16)
        wq = mx.array(wb.view(np.uint32))
        fin = np.where(np.isfinite(e4m3_values()))[0].astype(np.uint8)
        idx = np.sort(rng.integers(0, E, R)).astype(np.uint32)
        q = np.zeros((R, K), np.uint8)
        for r in range(R):
            b = r % nb
            if idx[r] == 0 and r % 2 == 0:
                blk_x = np.array([0x7E] * 15 + [0x01], np.uint8)
            elif idx[r] == 0:
                blk_x = np.array([0x7E, 0xFE] * 7 + [0x7E, 0x01], np.uint8)
                blk_x[rng.permutation(16)[:4]] |= 0x80                # some sign flips
            else:
                blk_x = rng.choice(fin, 16)
            q[r, b * 16:(b + 1) * 16] = blk_x
        y = np.array(run(mx.array(q.reshape(R, 1, K)), mx.ones((R, 1, 1)), wq, ws, mx.array(idx), True)).reshape(R, N)
        DUMP[f"fg2_{K}x{N}"] = y
        W2 = w2_int(wq); qv = e4m3_f64(q)
        worst, inexact, n_out = 0.0, 0, 0
        for r in range(R):
            b = r % nb
            terms = qv[r, b * 16:(b + 1) * 16][None, :] * W2[idx[r]][:, b * 16:(b + 1) * 16]   # [N, 16], exact
            exact = terms.sum(-1)      # float64: exact here (products <= 2^13, >= 2^-9, 53 bits)
            bound = GAMMA15 * np.abs(terms).sum(-1)
            err = np.abs(2.0 * y[r].astype(np.float64) - exact)
            inexact += int((err > 0).sum()); n_out += err.size
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = np.where(bound > 0, err / bound, np.where(err > 0, np.inf, 0.0))
            worst = max(worst, float(ratio.max()))
        out.append({"K": K, "N": N, "rows": R, "rows_on_forced_expert": int((idx == 0).sum()), "outputs": n_out,
                    "inexact_partials": inexact, "max_err_over_bound": worst, "pass": worst <= 1.0 and inexact > 0})
    return out


def fg3(rng):
    out = []
    for K, N, nm in ((2560, 640, "gate/up"), (640, 2560, "down")):
        E = 16
        wq, ws = weights(rng, E, N, K)
        for kind in ("decode", "sorted"):
            if kind == "decode":
                B, L, k = 2, 1, 10
                x = mx.array((rng.standard_normal((B, L, 1, 1, K)) * np.exp(rng.standard_normal((B, L, 1, 1, K)))).astype(np.float32))
                idx = mx.array(rng.integers(0, E, (B, L, k)).astype(np.uint32)); sort = False
            else:
                Pn = 300
                x = mx.array((rng.standard_normal((Pn, 1, K)) * np.exp(rng.standard_normal((Pn, 1, K)))).astype(np.float32))
                idx = mx.array(np.sort(rng.integers(0, E, Pn)).astype(np.uint32)); sort = True
            q, s = f8_quantize(x)
            if kind == "decode" and os.environ.get("MLX_GATHER_QMM_F8_ROUTE") == "nax":   # inputs drawn all the same
                out.append({"case": nm, "shape": kind, "skipped": "decode has no nax route", "pass": True}); continue
            y = np.array(run(q, s, wq, ws, idx, sort)).reshape(-1, N).astype(np.float64)
            DUMP[f"fg3_{nm}_{kind}"] = y.astype(np.float32)
            xr = mx.from_fp8(q, dtype=mx.float32) * s
            yr = mx.gather_qmm(xr, wq, ws, rhs_indices=idx, transpose=True, group_size=16, bits=4, mode="nvfp4", sorted_indices=sort)
            yr = np.array(yr.astype(mx.float32)).reshape(-1, N).astype(np.float64)
            rel = float(np.linalg.norm(y - yr) / np.linalg.norm(yr))
            out.append({"case": nm, "shape": kind, "K": K, "N": N, "rel_l2_vs_f8ref": rel,
                        "max_abs_vs_f8ref_over_max": float(np.abs(y - yr).max() / np.abs(yr).max()), "pass": rel <= 1e-5})
    return out


def compare(a, b):
    A, B = np.load(a), np.load(b)
    rows = []
    for k in sorted(set(A.files) | set(B.files)):
        if k not in A.files or k not in B.files:
            rows.append({"case": k, "present_in_both": False}); continue
        x, y = A[k].astype(np.float64), B[k].astype(np.float64)
        rows.append({"case": k, "identical": bool(np.array_equal(A[k], B[k])),
                     "rel_l2": float(np.linalg.norm(x - y) / max(np.linalg.norm(y), 1e-300))})
    print(json.dumps(rows, indent=1))
    print("F-G4 (routes identical):", "YES" if all(r.get("identical") for r in rows) else "NO (each route must meet F-G3)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump")
    ap.add_argument("--compare", nargs=2)
    a = ap.parse_args()
    if a.compare:
        return compare(*a.compare)
    rng = np.random.default_rng(17)
    r = {"route": os.environ.get("MLX_GATHER_QMM_F8_ROUTE", "auto"), "relaxed": os.environ.get("MLX_GATHER_QMM_F8_RELAXED", "0"),
         "device": mx.device_info().get("device_name"), "mlx": mx.__version__}
    r["F-G1_to_fp8"] = fg1_to_fp8()
    r["F-G1_weights"] = fg1_weights(rng)
    r["F-G2"] = fg2(rng)
    r["F-G3"] = fg3(rng)
    ok = r["F-G1_to_fp8"]["pass"] and all(c["exact"] for c in r["F-G1_weights"]) and \
        all(c["pass"] for c in r["F-G2"]) and all(c["pass"] for c in r["F-G3"])
    r["pass"] = ok
    if a.dump:
        np.savez(a.dump, **DUMP)
        r["dump_sha256"] = {k: hashlib.sha256(v.tobytes()).hexdigest()[:16] for k, v in DUMP.items()}
    print(json.dumps(r, indent=1))
    print("F-G1/F-G2/F-G3:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
