"""Pair fits: is our optimum global?  vs pyvinecopulib fits and vs differential evolution on our (verified) density."""
import numpy as np, pyvinecopulib as pv, rscopulas as rs, warnings, sys
from scipy.optimize import differential_evolution
warnings.filterwarnings("ignore")
F = pv.BicopFamily
PV = dict(gaussian=F.gaussian, student_t=F.student, clayton=F.clayton, gumbel=F.gumbel, frank=F.frank, joe=F.joe, bb1=F.bb1, bb6=F.bb6, bb7=F.bb7, bb8=F.bb8)
BOUNDS = dict(gaussian=[(-0.9999, 0.9999)], student_t=[(-0.9999, 0.9999), (2.0001, 60)], clayton=[(1e-6, 40)], gumbel=[(1.0, 40)], frank=[(-45, 45)],
              joe=[(1.0, 40)], bb1=[(1e-6, 8), (1.0, 8)], bb6=[(1.0, 8), (1.0, 8)], bb7=[(1.0, 8), (1e-6, 30)], bb8=[(1.0, 8), (1e-4, 1.0)],
              tawn1=[(1.0, 30), (0.0, 1.0)], tawn2=[(1.0, 30), (0.0, 1.0)])
ROTS = dict(gaussian=["R0"], student_t=["R0"], frank=["R0"])
def ll(fam, rot, p, u1, u2):
    try:
        return float(np.sum(rs.PairCopula.from_spec(fam, list(p), rotation=rot).log_pdf(u1, u2, clip_eps=1e-12)))
    except Exception:
        return -1e300
def datasets():
    gens = [("gaussian", [0.7]), ("gaussian", [-0.3]), ("student_t", [0.5, 3.0]), ("clayton", [1.5]), ("gumbel", [2.2]), ("frank", [6.0]), ("frank", [-3.0]),
            ("joe", [2.5]), ("bb1", [0.6, 1.8]), ("bb7", [2.0, 1.2]), ("bb8", [4.0, 0.7]), ("bb6", [1.5, 1.5]), ("clayton", [0.05]), ("gumbel", [9.0])]
    for i, (fam, p) in enumerate(gens):
        for n, rot in ((300, "R0"), (2000, "R90" if fam not in ROTS else "R0")):
            g = np.random.default_rng(100 + i + n)
            w = g.uniform(1e-9, 1 - 1e-9, (n, 2))
            pc = rs.PairCopula.from_spec(fam, p, rotation=rot)
            u2 = pc.inv_second_given_first(w[:, 0], w[:, 1], clip_eps=1e-12)
            X = np.column_stack([w[:, 0], u2])
            from scipy.stats import rankdata
            U = rankdata(X, axis=0) / (n + 1)
            yield f"{fam}{p}{rot} n={n}", U
only = sys.argv[1:]
bad = 0; total = 0; summary = {}
for dname, U in datasets():
    u1, u2 = np.ascontiguousarray(U[:, 0]), np.ascontiguousarray(U[:, 1])
    for fam in BOUNDS:
        if only and fam not in only: continue
        try:
            m = rs.VineCopula.fit_r(U, family_set=[fam], include_rotations=True, criterion="aic").model
        except Exception as ex:
            print(dname, fam, "FIT ERROR", ex); continue
        e = m.trees[0].edges[0]
        ours = ll(e.family, e.rotation, e.parameters, u1, u2) if e.family != "independence" else 0.0
        # global reference: DE over every rotation
        best = (0.0, "indep", ())
        for rot in ROTS.get(fam, ["R0", "R90", "R180", "R270"]):
            r = differential_evolution(lambda p: -ll(fam, rot, p, u1, u2), BOUNDS[fam], seed=1, tol=1e-10, popsize=20, maxiter=300, polish=True)
            if -r.fun > best[0]: best = (-r.fun, rot, tuple(r.x))
        theirs = None
        if fam in PV:
            b = pv.Bicop(); b.select(U, controls=pv.FitControlsBicop(family_set=[PV[fam]], selection_criterion="loglik"))
            theirs = b.loglik(U)
        total += 1
        gap = best[0] - ours
        s = summary.setdefault(fam, [0, 0.0, 0.0]); s[0] += 1; s[1] = max(s[1], gap); s[2] = max(s[2], (theirs - ours) if theirs is not None else -9)
        if gap > 1e-4 * max(1.0, abs(best[0])) + 1e-6:
            bad += 1
            print(f"GAP {dname:28s} {fam:9s} ours {ours:.6f} ({e.family} {e.rotation} {tuple(round(x,5) for x in e.parameters)})  DE {best[0]:.6f} ({best[1]} {tuple(round(x,5) for x in best[2])})  pv {theirs}", flush=True)
print("fits:", total, "with a gap:", bad)
for f, s in summary.items(): print(f"  {f:10s} n={s[0]} max(DE - ours)={s[1]:+.2e}  max(pv - ours)={s[2]:+.2e}")
