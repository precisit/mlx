"""Speed diagnostic for the NAX int8 x NVFP4 sorted-gather kernel (modelopt-to-mlx stage2/A8-KERNEL.md, speed lane
step 1.1-1.2). Readings, not a benchmark. One DIAG variant per process (MLX_GATHER_QMM_I8_NAX_DIAG is read once):

    MLX_GATHER_QMM_I8_NAX_DIAG=<d> python -I precisit/diag_nax_i8.py OUT.json

Per routed-expert shape (E = 512) and token count (x 10 routes, sorted): the stock bf16 gather_qmm (the A16 path) and
gather_qmm_i8 on the NAX route, median of 7. Two routing spreads: "model" draws experts uniformly over 512 (about
T*10/512 rows per expert, as in the model) and "dense16" puts every row on 16 experts (full row tiles). The output's
sha256 lets the driver check that the exact variants (1, 2, 3) are bit-identical to variant 0.
"""
import hashlib, json, os, sys, time
import numpy as np
import mlx.core as mx


def bench(f, n=7):
    f(); mx.synchronize(); ts = []
    for _ in range(n):
        t = time.perf_counter(); f(); mx.synchronize(); ts.append(time.perf_counter() - t)
    return float(np.median(ts) * 1e3)


diag = int(os.environ.get("MLX_GATHER_QMM_I8_NAX_DIAG", "0"))
os.environ["MLX_GATHER_QMM_I8_ROUTE"] = os.environ.get("MLX_GATHER_QMM_I8_ROUTE", "nax")
rng = np.random.default_rng(0); E = 512; rows = []
for K, N, nm in ((2560, 640, "gate/up"), (640, 2560, "down")):
    W = mx.array(rng.standard_normal((E, N, K)).astype(np.float32) * 0.05)
    wq, ws = mx.quantize(W, group_size=16, bits=4, mode="nvfp4")[:2]; mx.eval(wq, ws); del W
    for spread in ("model", "dense16"):
        for T in (512, 2048):
            P = T * 10
            ex = rng.integers(0, E, P) if spread == "model" else rng.choice(E, 16, replace=False)[rng.integers(0, 16, P)]
            idx = mx.array(np.sort(ex).astype(np.uint32))
            x = mx.random.normal((P, 1, K), key=mx.random.key(T + K)).astype(mx.bfloat16)
            q = mx.clip(mx.round(x.astype(mx.float32) * 20), -127, 127).astype(mx.int8)
            s = mx.ones((P, 1, 1), mx.float32)
            mx.eval(x, q, s, idx)
            y = mx.gather_qmm_i8(q, s, wq, ws, rhs_indices=idx, sorted_indices=True); mx.eval(y)
            r = {"proj": nm, "spread": spread, "tokens": T, "rows": P,
                 "stock_bf16_ms": bench(lambda: mx.eval(mx.gather_qmm(x, wq, ws, rhs_indices=idx, transpose=True, group_size=16,
                                                                     bits=4, mode="nvfp4", sorted_indices=True))),
                 "i8_ms": bench(lambda: mx.eval(mx.gather_qmm_i8(q, s, wq, ws, rhs_indices=idx, sorted_indices=True))),
                 "out_sha256": hashlib.sha256(np.array(y).tobytes()).hexdigest()}
            r["i8_vs_bf16_time"] = r["i8_ms"] / r["stock_bf16_ms"]
            rows.append(r); print(json.dumps(r), flush=True)
out = {"diag": diag, "route": os.environ["MLX_GATHER_QMM_I8_ROUTE"], "device": mx.device_info().get("device_name"),
       "mlx": mx.__version__, "rows": rows}
open(sys.argv[1], "w").write(json.dumps(out, indent=1))
