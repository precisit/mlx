"""Speed diagnostic for the int8 x NVFP4 NAX rescale (modelopt-to-mlx stage2/A8-KERNEL.md, speed lane step 2.4).
Readings, not a benchmark. The variant is read once per process, so one variant per run:

    MLX_GATHER_QMM_I8_NAX_RESCALE=<r> MLX_GATHER_QMM_I8_NAX_BK=<64|128> python -I precisit/diag_rescale_i8.py nax OUT.json

(The qmv mode is kept from diag_unpack_i8.py and times the vector kernel, which this branch does not change.)

Per routed-expert shape (E = 512): the stock bf16 gather_qmm (the A16 path) and gather_qmm_i8, median of 9. qmv: one
token x 10 routes (the model's decode call). nax: T tokens x 10 routes, sorted, experts drawn uniformly over 512 (as in
the model) and "dense16" (every row on 16 experts). The output sha256 lets the driver check that the exact variants
are byte-identical to variant 0 at BK 64 (the gated kernel); 8 is a timing bound with wrong numerics.
"""
import hashlib, json, os, sys, time
import numpy as np
import mlx.core as mx


def bench(f, n=9):
    f(); mx.synchronize(); ts = []
    for _ in range(n):
        t = time.perf_counter(); f(); mx.synchronize(); ts.append(time.perf_counter() - t)
    return float(np.median(ts) * 1e3)


mode, out_path = sys.argv[1], sys.argv[2]
assert mode in ("qmv", "nax"), mode
os.environ["MLX_GATHER_QMM_I8_ROUTE"] = os.environ.get("MLX_GATHER_QMM_I8_ROUTE", "vector" if mode == "qmv" else "nax")
var = {k: os.environ.get(k) for k in ("MLX_GATHER_QMM_I8_NAX_RESCALE", "MLX_GATHER_QMM_I8_NAX_BK")}
rng = np.random.default_rng(0); E = 512; rows = []
for K, N, nm in ((2560, 640, "gate/up"), (640, 2560, "down")):
    W = mx.array(rng.standard_normal((E, N, K)).astype(np.float32) * 0.05)
    wq, ws = mx.quantize(W, group_size=16, bits=4, mode="nvfp4")[:2]; mx.eval(wq, ws); del W
    cases = [("model", 1)] if mode == "qmv" else [(sp, T) for sp in ("model", "dense16") for T in (512, 2048)]
    for spread, T in cases:
        if mode == "qmv":   # 48 calls per timed eval (one per layer), each with its own 10 experts
            x = mx.random.normal((1, 1, 1, 1, K), key=mx.random.key(K)).astype(mx.bfloat16)
            idxs = [mx.array(rng.integers(0, E, (1, 1, 10)).astype(np.uint32)) for _ in range(48)]
            idx = idxs[0]; srt = False
        else:
            P = T * 10
            ex = rng.integers(0, E, P) if spread == "model" else rng.choice(E, 16, replace=False)[rng.integers(0, 16, P)]
            idx = mx.array(np.sort(ex).astype(np.uint32)); srt = True
            x = mx.random.normal((P, 1, K), key=mx.random.key(T + K)).astype(mx.bfloat16)
        xf = x.astype(mx.float32)
        s = mx.max(mx.abs(xf), axis=-1, keepdims=True) / 127.0
        q = mx.clip(mx.round(xf / s), -127, 127).astype(mx.int8)
        calls = idxs if mode == "qmv" else [idx]
        mx.eval(x, q, s, *calls)
        y = mx.concatenate([mx.gather_qmm_i8(q, s, wq, ws, rhs_indices=i, sorted_indices=srt).reshape(-1) for i in calls])
        mx.eval(y)
        r = {"proj": nm, "spread": spread, "tokens": T, "calls_per_eval": len(calls),
             "stock_bf16_ms": bench(lambda: mx.eval([mx.gather_qmm(x, wq, ws, rhs_indices=i, transpose=True, group_size=16,
                                                                  bits=4, mode="nvfp4", sorted_indices=srt) for i in calls])),
             "i8_ms": bench(lambda: mx.eval([mx.gather_qmm_i8(q, s, wq, ws, rhs_indices=i, sorted_indices=srt) for i in calls])),
             "out_sha256": hashlib.sha256(np.array(y).tobytes()).hexdigest()}
        r["i8_vs_bf16_time"] = r["i8_ms"] / r["stock_bf16_ms"]
        rows.append(r); print(json.dumps(r), flush=True)
out = {"mode": mode, "env": var, "route": os.environ["MLX_GATHER_QMM_I8_ROUTE"],
       "device": mx.device_info().get("device_name"), "mlx": mx.__version__, "label": "readings, median of 9", "rows": rows}
open(out_path, "w").write(json.dumps(out, indent=1))
