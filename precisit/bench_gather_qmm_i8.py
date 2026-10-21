"""Readings (not a benchmark): mx.gather_qmm_i8 (current route) vs stock gather_qmm (bf16 and fp32 x), the model's
routed-expert shapes, E = 512 experts, decode (1 token x 10 routes) and sorted prefill (T x 10 routes)."""
import json, os, time
import numpy as np
import mlx.core as mx


def bench(f, n=7):
    f(); mx.synchronize(); ts = []
    for _ in range(n):
        t = time.perf_counter(); f(); mx.synchronize(); ts.append(time.perf_counter() - t)
    return float(np.median(ts) * 1e3)


rng = np.random.default_rng(0); E = 512; rows = []
for K, N, nm in ((2560, 640, "gate/up"), (640, 2560, "down")):
    W = mx.array(rng.standard_normal((E, N, K)).astype(np.float32) * 0.05)
    wq, ws = mx.quantize(W, group_size=16, bits=4, mode="nvfp4")[:2]; mx.eval(wq, ws); del W
    for T in (1, 128, 512, 2048):
        P = T * 10
        if T == 1:
            x = mx.random.normal((1, 1, 1, 1, K)).astype(mx.bfloat16); idx = mx.array(rng.integers(0, E, (1, 1, 10)).astype(np.uint32)); srt = False
        else:
            x = mx.random.normal((P, 1, K)).astype(mx.bfloat16); idx = mx.array(np.sort(rng.integers(0, E, P)).astype(np.uint32)); srt = True
        xf = x.astype(mx.float32); q = mx.clip(mx.round(xf * 20), -127, 127).astype(mx.int8); s = mx.ones(x.shape[:-1] + (1,), mx.float32)
        mx.eval(x, xf, q, s, idx)
        g = lambda xx: mx.eval(mx.gather_qmm(xx, wq, ws, rhs_indices=idx, transpose=True, group_size=16, bits=4, mode="nvfp4", sorted_indices=srt))
        r = {"proj": nm, "tokens": T, "stock_bf16_ms": bench(lambda: g(x)), "stock_fp32_ms": bench(lambda: g(xf)),
             "i8_ms": bench(lambda: mx.eval(mx.gather_qmm_i8(q, s, wq, ws, rhs_indices=idx, sorted_indices=srt)))}
        r["i8_vs_bf16"] = r["i8_ms"] / r["stock_bf16_ms"]
        rows.append(r); print(json.dumps(r))
print(json.dumps({"route": os.environ.get("MLX_GATHER_QMM_I8_ROUTE", "auto"), "device": mx.device_info().get("device_name"), "rows": rows}))
