"""Vine log-density / Rosenblatt vs a textbook set-based recursion built only from pair kernels."""
import numpy as np, rscopulas as rs
from scipy import stats
rng = np.random.default_rng(5)

def make_data(n, d, seed):
    r = np.random.default_rng(seed)
    z = r.standard_normal((n, d)); f = r.standard_normal((n, 1)); w = r.chisquare(3, (n, 1)) / 3
    x = np.empty((n, d))
    for j in range(d):
        k = j % 4
        if k == 0: x[:, j] = (0.8 * f[:, 0] + 0.6 * z[:, j]) / np.sqrt(w[:, 0])
        elif k == 1: x[:, j] = np.exp(0.7 * f[:, 0]) + 0.5 * np.abs(z[:, j]) ** 1.5
        elif k == 2: x[:, j] = -np.maximum(f[:, 0], z[:, j]) + 0.3 * z[:, (j + 1) % d]
        else: x[:, j] = 0.5 * x[:, j - 1] ** 2 * np.sign(z[:, j]) + z[:, j]
    return rs.to_pseudo_obs(x) if hasattr(rs, "to_pseudo_obs") else stats.rankdata(x, axis=0) / (n + 1)

class Naive:
    def __init__(self, model):
        self.edges = {}
        for t in model.trees:
            for e in t.edges:
                st = e.state if e.state is not None else None
                kw = {}
                if e.family == "khoudraji": raise NotImplementedError
                pc = rs.PairCopula.from_spec(e.family, list(e.parameters), rotation=e.rotation, state=e.state)
                self.edges[(frozenset(e.conditioned), frozenset(e.conditioning))] = (e.conditioned, pc)
    def F(self, a, D, U, cache):
        if not D: return U[:, a]
        key = (a, D)
        if key in cache: return cache[key]
        for j in D:
            Dp = D - {j}
            hit = self.edges.get((frozenset((a, j)), Dp))
            if hit:
                (c0, c1), pc = hit
                x0, x1 = self.F(c0, Dp, U, cache), self.F(c1, Dp, U, cache)
                out = pc.cond_first_given_second(x0, x1, clip_eps=1e-12) if a == c0 else pc.cond_second_given_first(x0, x1, clip_eps=1e-12)
                cache[key] = out
                return out
        raise KeyError((a, sorted(D)))
    def log_pdf(self, U):
        cache = {}; total = np.zeros(len(U))
        for (cond, D), ((c0, c1), pc) in self.edges.items():
            total += pc.log_pdf(self.F(c0, D, U, cache), self.F(c1, D, U, cache), clip_eps=1e-12)
        return total
    def rosenblatt(self, U, order):
        cache = {}
        return np.column_stack([self.F(a, frozenset(order[:k]), U, cache) for k, a in enumerate(order)])

FAMS = ["independence","gaussian","student_t","clayton","gumbel","frank","joe","bb1","bb6","bb7","bb8","tawn1","tawn2"]
def run(name, fit, d, **kw):
    X = make_data(1500, d, seed=d)
    try:
        m = fit(X, **kw).model
    except Exception as e:
        print(name, "FIT FAILED", e); return
    fams = sorted({e.family + (":" + e.rotation if e.rotation != "R0" else "") for t in m.trees for e in t.edges})
    nv = Naive(m)
    T = np.vstack([X[:400], rng.uniform(0.001, 0.999, (300, d)), rng.uniform(1e-4, 0.03, (50, d)), 1 - rng.uniform(1e-4, 0.03, (50, d))])
    ours, ref = m.log_pdf(T, clip_eps=1e-12), nv.log_pdf(T)
    e = np.abs(ours - ref) / np.maximum(1, np.abs(ref))
    order = list(m.variable_order)
    msg = f"{name:22s} d={d} logpdf max rel {e.max():.1e}"
    try:
        R, Rn = np.asarray(m.rosenblatt(T)), nv.rosenblatt(T, order)
        msg += f" | rosenblatt vs naive {np.abs(R - Rn).max():.1e}"
        back = np.asarray(m.inverse_rosenblatt(R))
        msg += f" | round trip {np.abs(back - np.clip(T,1e-12,1-1e-12)).max():.1e}"
        S = m.sample(200000, seed=9)
        P = nv.rosenblatt(S, order)
        ks = max(stats.kstest(P[:, j], "uniform").statistic for j in range(d)) * np.sqrt(len(S))
        Z = stats.norm.ppf(np.clip(P, 1e-12, 1 - 1e-12)); C = np.corrcoef(Z.T); np.fill_diagonal(C, 0)
        C2 = np.corrcoef((Z ** 2).T); np.fill_diagonal(C2, 0)
        msg += f" | sampler PIT KS√n {ks:.2f}, max|corr|√n {np.abs(C).max()*np.sqrt(len(S)):.2f}, sq {np.abs(C2).max()*np.sqrt(len(S)):.2f}"
    except KeyError as ex:
        msg += f" | rosenblatt order lookup failed {ex}"
    print(msg, "|", ",".join(fams), flush=True)

for d in (3, 5, 7):
    run("rvine all families", rs.VineCopula.fit_r, d, family_set=FAMS)
    run("cvine all families", rs.VineCopula.fit_c, d, family_set=FAMS)
    run("dvine all families", rs.VineCopula.fit_d, d, family_set=FAMS)
run("rvine trunc 2", rs.VineCopula.fit_r, 6, family_set=FAMS, truncation_level=2)
run("rvine no rotations", rs.VineCopula.fit_r, 6, family_set=FAMS, include_rotations=False)
run("rvine gaussian only", rs.VineCopula.fit_r, 8, family_set=["gaussian"])
run("dvine t only", rs.VineCopula.fit_d, 6, family_set=["student_t"])
run("rvine tll", rs.VineCopula.fit_r, 4, family_set=["tll"])
