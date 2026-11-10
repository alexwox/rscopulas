"""Nested Archimedean sampler vs the closed-form nested CDF (2e6 draws per tree, about five minutes)."""
import numpy as np, rscopulas as rs
rng = np.random.default_rng(7)

def psi(f, t, x):
    if f == "gumbel": return np.exp(-x ** (1 / t))
    if f == "clayton": return (1 + x) ** (-1 / t)
    if f == "frank": return -np.log1p(-np.exp(-x) * (-np.expm1(-t))) / t
def psi_inv(f, t, u):
    if f == "gumbel": return (-np.log(u)) ** t
    if f == "clayton": return u ** -t - 1
    if f == "frank": return -np.log(np.expm1(-t * u) / np.expm1(-t))
def cdf(tree, U):
    if isinstance(tree, int): return U[:, tree]
    f, t = tree["family"], tree["theta"]
    return psi(f, t, sum(psi_inv(f, t, cdf(c, U)) for c in tree["children"]))
def leaves(tree):
    return [tree] if isinstance(tree, int) else [l for c in tree["children"] for l in leaves(c)]

def cases():
    for f, (a, b, c) in [("gumbel", (1.3, 2.5, 6.0)), ("clayton", (0.4, 2.0, 7.0)), ("frank", (1.5, 6.0, 18.0)),
                         ("gumbel", (1.05, 1.1, 9.0)), ("clayton", (0.05, 0.1, 12.0)), ("frank", (0.2, 0.3, 30.0))]:
        n = lambda t, ch: {"family": f, "theta": t, "children": ch}
        yield f"{f} 2-level {a},{b}", n(a, [0, n(b, [1, 2])])
        yield f"{f} two-clusters {a},{b},{c}", n(a, [n(b, [0, 1]), n(c, [2, 3])])
        yield f"{f} 3-level {a},{b},{c}", n(a, [0, n(b, [1, n(c, [2, 3])]), 4])
        yield f"{f} exchangeable {b}", n(b, [0, 1, 2])

N, K = 2_000_000, 400
allz = []
for name, tree in cases():
    m = rs.HierarchicalArchimedeanCopula.from_tree(tree)
    X = m.sample(N, seed=11)
    d = X.shape[1]
    # evaluation points: mix of uniform cube points and points near the diagonal / tails
    P = rng.uniform(0.02, 0.98, size=(K, d))
    P[: K // 4] = np.clip(rng.uniform(0.02, 0.98, size=(K // 4, 1)) + rng.normal(0, 0.05, size=(K // 4, d)), 0.01, 0.99)
    P[K // 4 : K // 2] = rng.uniform(0.9, 0.999, size=(K // 4, d))
    P[K // 2 : 5 * K // 8] = rng.uniform(0.001, 0.1, size=(K // 8, d))
    exact = cdf(tree, P)
    emp = np.array([np.all(X <= p, axis=1).mean() for p in P])
    se = np.sqrt(np.maximum(exact * (1 - exact), 1e-12) / N)
    z = (emp - exact) / se
    allz.append(z)
    # marginal uniformity
    ks = max(np.abs(np.sort(X[:, j]) - (np.arange(1, N + 1) - 0.5) / N).max() for j in range(d)) * np.sqrt(N)
    print(f"{name:34s} d={d} max|z|={np.abs(z).max():5.2f} rms z={np.sqrt((z**2).mean()):.2f} mean z={z.mean():+.2f} max abs diff={np.abs(emp-exact).max():.1e} marg KS*sqrtN={ks:.2f} exact={m.is_exact}", flush=True)
z = np.concatenate(allz)
print("overall rms z", np.sqrt((z**2).mean()), "max", np.abs(z).max(), "n", z.size)
