"""K-G4 (modelopt-to-mlx stage2/A8-KERNEL.md): the same seeded operands through each route of mx.gather_qmm_i8.
Run once per route (the route is read once per process), then compare:
    MLX_GATHER_QMM_I8_ROUTE=nax      python -I precisit/kg4_routes.py dump OUT/nax.npz
    MLX_GATHER_QMM_I8_ROUTE=vector   python -I precisit/kg4_routes.py dump OUT/vector.npz
    MLX_GATHER_QMM_I8_ROUTE=fallback python -I precisit/kg4_routes.py dump OUT/fallback.npz
    python -I precisit/kg4_routes.py compare OUT/nax.npz OUT/vector.npz OUT/fallback.npz
Also writes the a8-ref (stock gather_qmm on the dequantized int8 input) and an exact float64 reference.
"""
import json, os, sys
import numpy as np
import mlx.core as mx

E2M1_X2 = np.array([0, 1, 2, 3, 4, 6, 8, 12], np.int64)


def operands(K, N, E, P, seed):
    rng = np.random.default_rng(seed)
    W = mx.array(rng.standard_normal((E, N, K)).astype(np.float32) * 0.05)
    wq, ws = mx.quantize(W, group_size=16, bits=4, mode="nvfp4")[:2]
    x = mx.array((rng.standard_normal((P, 1, K)) * np.exp(rng.standard_normal((P, 1, K)))).astype(np.float32))
    idx = mx.array(np.sort(rng.integers(0, E, P)).astype(np.uint32))
    xf = x.astype(mx.float32)
    s = mx.max(mx.abs(xf), axis=-1, keepdims=True) / 127.0
    q = mx.clip(mx.round(xf / s), -127, 127).astype(mx.int8)
    return wq, ws, q, s, idx


def exact(wq, ws, q, s, idx):
    b = np.array(wq).view(np.uint8); nib = np.stack([b & 0x0F, b >> 4], -1).reshape(*b.shape[:-1], -1)
    W2 = np.where(nib & 8, -E2M1_X2[nib & 7], E2M1_X2[nib & 7])
    S = np.array(mx.from_fp8(mx.array(np.array(ws)), dtype=mx.float32)).astype(np.float64)
    qn = np.array(q).astype(np.int64)[:, 0]; sn = np.array(s).astype(np.float64)[:, 0, 0]; e = np.array(idx)
    K = qn.shape[-1]; nb = K // 16; out = np.empty((qn.shape[0], W2.shape[1]))
    for r in range(qn.shape[0]):
        P = (qn[r].reshape(nb, 16)[None] * W2[e[r]].reshape(-1, nb, 16)).sum(-1)
        out[r] = 0.5 * sn[r] * (P * S[e[r]]).sum(-1)
    return out


CASES = [(2560, 640, 64, 2048, 1), (640, 2560, 64, 2048, 2), (2560, 640, 512, 4096, 3), (640, 2560, 512, 300, 4)]


def dump(path):
    out = {"route": os.environ.get("MLX_GATHER_QMM_I8_ROUTE", "auto")}
    for i, (K, N, E, P, seed) in enumerate(CASES):
        wq, ws, q, s, idx = operands(K, N, E, P, seed)
        y = mx.gather_qmm_i8(q, s, wq, ws, rhs_indices=idx, sorted_indices=True)
        out[f"c{i}"] = np.array(y)[:, 0]
        if out["route"] == "vector":   # reference arrays once
            yr = mx.gather_qmm(q.astype(mx.float32) * s, wq, ws, rhs_indices=idx, transpose=True, group_size=16, bits=4, mode="nvfp4", sorted_indices=True)
            out[f"c{i}_a8ref"] = np.array(yr)[:, 0]
            out[f"c{i}_exact"] = exact(wq, ws, q, s, idx)
    np.savez(path, **{k: v for k, v in out.items() if k != "route"}, route=np.array(out["route"]))
    print("wrote", path, out["route"])


def compare(paths):
    d = {str(np.load(p)["route"]): np.load(p) for p in paths}
    ref = d["vector"]; res = {}
    for i, case in enumerate(CASES):
        ex, a8 = ref[f"c{i}_exact"], ref[f"c{i}_a8ref"]
        r = {"K,N,E,P": case[:4]}
        for name, z in d.items():
            y = z[f"c{i}"].astype(np.float64)
            r[name] = {"rel_l2_vs_exact": float(np.linalg.norm(y - ex) / np.linalg.norm(ex)),
                       "rel_l2_vs_a8ref": float(np.linalg.norm(y - a8) / np.linalg.norm(a8))}
        for a in d:
            for b in d:
                if a < b:
                    r[f"{a}=={b}_bit_identical"] = bool(np.array_equal(d[a][f"c{i}"], d[b][f"c{i}"]))
        res[f"c{i}"] = r
        print(json.dumps(r))
    ok = all(v[n]["rel_l2_vs_a8ref"] <= 1e-5 for v in res.values() for n in d)
    print("K-G3 on every route:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(dump(sys.argv[2]) if sys.argv[1] == "dump" else compare(sys.argv[2:]))
