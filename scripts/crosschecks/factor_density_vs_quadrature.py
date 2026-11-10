import numpy as np, rscopulas as rs
from scipy import integrate
rng = np.random.default_rng(3)
L = lambda f, p, r="R0": {"family": f, "parameters": p, "rotation": r}
models = {
 "gauss5": [L("gaussian",[0.8]),L("gaussian",[-0.5]),L("gaussian",[0.3]),L("gaussian",[0.95]),L("gaussian",[0.1])],
 "mixed6": [L("clayton",[2.0]),L("gumbel",[1.8]),L("frank",[5.0]),L("joe",[2.5]),L("student_t",[0.6,4.0]),L("gaussian",[-0.4])],
 "rot4": [L("clayton",[3.0],"R90"),L("gumbel",[2.5],"R270"),L("bb1",[0.5,1.5],"R180"),L("clayton",[1.0],"R180")],
 "strong3": [L("clayton",[12.0]),L("gumbel",[8.0]),L("joe",[10.0])],
 "bb5": [L("bb1",[0.6,1.6]),L("bb7",[1.5,0.8]),L("bb8",[3.0,0.8]),L("bb6",[1.5,1.4]),L("frank",[-6.0])],
}
for name, links in models.items():
    try:
        m = rs.FactorCopula.from_links(links)
    except Exception as e:
        print(name, "construct failed:", e); print(rs.FactorCopula.from_links.__doc__); continue
    d = len(links)
    pcs = [rs.PairCopula.from_spec(l["family"], l["parameters"], rotation=l["rotation"]) for l in links]
    X = m.sample(40, seed=5)
    U = np.vstack([X, rng.uniform(0.001, 0.999, (25, d)), rng.uniform(0.0005, 0.02, (8, d)), 1 - rng.uniform(0.0005, 0.02, (8, d)),
                   np.column_stack([rng.uniform(0.0005,0.01,(8,1))]+[1-rng.uniform(0.0005,0.01,(8,1)) for _ in range(d-1)])])
    ours = m.log_pdf(U, clip_eps=1e-12)
    res = {}
    for order in ("factor_second", "factor_first"):
        ref = []
        for row in U:
            def integrand(v):
                vv = np.full(1, v)
                s = 0.0
                for j, pc in enumerate(pcs):
                    uj = np.full(1, row[j])
                    s += (pc.log_pdf(uj, vv, clip_eps=1e-16) if order == "factor_second" else pc.log_pdf(vv, uj, clip_eps=1e-16))[0]
                return np.exp(s)
            # split the range finely near the ends so spikes are resolved
            brk = np.unique(np.concatenate([10.0 ** -np.arange(1, 15), 1 - 10.0 ** -np.arange(1, 15), np.linspace(0.05, 0.95, 37)]))
            val = 0.0
            edges = np.concatenate([[0.0], brk, [1.0]])
            for a, b in zip(edges[:-1], edges[1:]):
                val += integrate.quad(integrand, a, b, epsabs=0, epsrel=1e-11, limit=200)[0]
            ref.append(np.log(val))
        ref = np.array(ref)
        res[order] = np.abs(ours - ref)
    best = min(res, key=lambda k: res[k].max())
    e = res[best]; i = e.argmax()
    print(f"{name:8s} convention={best:13s} max|dlog|={e.max():.2e} median={np.median(e):.1e} at u={U[i].round(4)} ours={ours[i]:.6f} | other convention max={res[[k for k in res if k!=best][0]].max():.1e} adaptive={m.adaptive_quadrature} nodes={m.quadrature_nodes}", flush=True)
