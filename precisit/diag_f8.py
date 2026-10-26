"""F8 arm step 0 (modelopt-to-mlx stage2/PROTOCOL.md amendment 17): op-level readings of the F8 kernel against A8's
int8 kernel and stock bf16, on the speed lane's op cases. Readings, not a benchmark. One relaxed setting per process:

    MLX_GATHER_QMM_F8_RELAXED=<0|1> python -I precisit/diag_f8.py <nax|qmv> OUT.json

Per routed-expert shape (E = 512), as diag_rescale_i8.py: the stock bf16 gather_qmm (the A16 path), gather_qmm_i8
(A8, f295ac5) and gather_qmm_f8, median of 9 each, in one process, interleaved per case. nax: T tokens x 10 routes,
sorted, experts uniform over 512 ("model") and on 16 experts ("dense16"), T 512 and 2048, routes forced to nax for
both kernels. qmv: one token x 10 routes, 48 calls per timed eval (the model's decode call), vector routes. Kernel
time only: x is quantized before timing (a8: amendment 16's rounding; f8: f8_quantize). The decision quantity is
f8_ms / i8_ms on the 8 nax cases. Outputs are hashed.
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
route = "vector" if mode == "qmv" else "nax"
os.environ["MLX_GATHER_QMM_I8_ROUTE"] = route
os.environ["MLX_GATHER_QMM_F8_ROUTE"] = route
rng = np.random.default_rng(0); E = 512; rows = []
for K, N, nm in ((2560, 640, "gate/up"), (640, 2560, "down")):
    W = mx.array(rng.standard_normal((E, N, K)).astype(np.float32) * 0.05)
    wq, ws = mx.quantize(W, group_size=16, bits=4, mode="nvfp4")[:2]; mx.eval(wq, ws); del W
    cases = [("model", 1)] if mode == "qmv" else [(sp, T) for sp in ("model", "dense16") for T in (512, 2048)]
    for spread, T in cases:
        if mode == "qmv":
            x = mx.random.normal((1, 1, 1, 1, K), key=mx.random.key(K)).astype(mx.bfloat16)
            idxs = [mx.array(rng.integers(0, E, (1, 1, 10)).astype(np.uint32)) for _ in range(48)]
            srt = False
        else:
            P = T * 10
            ex = rng.integers(0, E, P) if spread == "model" else rng.choice(E, 16, replace=False)[rng.integers(0, 16, P)]
            idxs = [mx.array(np.sort(ex).astype(np.uint32))]; srt = True
            x = mx.random.normal((P, 1, K), key=mx.random.key(T + K)).astype(mx.bfloat16)
        xf = x.astype(mx.float32)
        s8 = mx.max(mx.abs(xf), axis=-1, keepdims=True) / 127.0
        q8 = mx.clip(mx.round(xf / s8), -127, 127).astype(mx.int8)
        sf = mx.max(mx.abs(xf), axis=-1, keepdims=True) / 448.0
        qf = mx.to_fp8(xf / sf)
        mx.eval(x, q8, s8, qf, sf, *idxs)
        yi = mx.concatenate([mx.gather_qmm_i8(q8, s8, wq, ws, rhs_indices=i, sorted_indices=srt).reshape(-1) for i in idxs])
        yf = mx.concatenate([mx.gather_qmm_f8(qf, sf, wq, ws, rhs_indices=i, sorted_indices=srt).reshape(-1) for i in idxs])
        yb = mx.concatenate([mx.gather_qmm(x, wq, ws, rhs_indices=i, transpose=True, group_size=16, bits=4, mode="nvfp4",
                                           sorted_indices=srt).reshape(-1) for i in idxs]).astype(mx.float32)
        mx.eval(yi, yf, yb)
        fb = lambda: mx.eval([mx.gather_qmm(x, wq, ws, rhs_indices=i, transpose=True, group_size=16, bits=4, mode="nvfp4",
                                            sorted_indices=srt) for i in idxs])
        fi = lambda: mx.eval([mx.gather_qmm_i8(q8, s8, wq, ws, rhs_indices=i, sorted_indices=srt) for i in idxs])
        ff = lambda: mx.eval([mx.gather_qmm_f8(qf, sf, wq, ws, rhs_indices=i, sorted_indices=srt) for i in idxs])
        t = {"bf16": [], "i8": [], "f8": []}
        for _ in range(2):    # interleaved rounds; the reading is the median over both rounds' medians
            t["bf16"].append(bench(fb)); t["i8"].append(bench(fi)); t["f8"].append(bench(ff))
        rel = lambda a: float(np.linalg.norm(np.array(a) - np.array(yb)) / np.linalg.norm(np.array(yb)))
        r = {"proj": nm, "spread": spread, "tokens": T, "calls_per_eval": len(idxs),
             "stock_bf16_ms": float(np.median(t["bf16"])), "i8_ms": float(np.median(t["i8"])), "f8_ms": float(np.median(t["f8"])),
             "rounds_ms": t, "i8_rel_l2_vs_bf16": rel(yi), "f8_rel_l2_vs_bf16": rel(yf),
             "i8_out_sha256": hashlib.sha256(np.array(yi).tobytes()).hexdigest(),
             "f8_out_sha256": hashlib.sha256(np.array(yf).tobytes()).hexdigest()}
        r["i8_vs_bf16_time"] = r["i8_ms"] / r["stock_bf16_ms"]
        r["f8_vs_bf16_time"] = r["f8_ms"] / r["stock_bf16_ms"]
        r["f8_vs_i8_time"] = r["f8_ms"] / r["i8_ms"]
        rows.append(r); print(json.dumps({k: v for k, v in r.items() if k != "rounds_ms" and "sha" not in k}), flush=True)
out = {"mode": mode, "route": route, "f8_relaxed": os.environ.get("MLX_GATHER_QMM_F8_RELAXED", "0"),
       "device": mx.device_info().get("device_name"), "architecture": mx.device_info().get("architecture"),
       "mlx": mx.__version__, "label": "readings, median of 9 per round, 2 interleaved rounds", "rows": rows}
if mode == "nax":
    fi = [r["f8_vs_i8_time"] for r in rows]
    out["decision"] = {"f8_vs_i8_time_per_case": fi, "f8_faster_on_all_nax_cases": all(v < 1 for v in fi),
                       "f8_vs_i8_geomean": float(np.exp(np.mean(np.log(fi))))}
open(out_path, "w").write(json.dumps(out, indent=1))
print(json.dumps(out.get("decision", {})))
