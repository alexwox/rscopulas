"""Pair kernels: rscopulas 0.5.0 vs pyvinecopulib (vinecopulib C++)."""
import itertools, sys
import numpy as np, pyvinecopulib as pv, rscopulas as rs

F = pv.BicopFamily
GRID = {
    "gaussian": (F.gaussian, [[r] for r in (-0.95, -0.5, -0.05, 0.1, 0.6, 0.9, 0.99)]),
    "student_t": (F.student, [[r, nu] for r in (-0.9, -0.3, 0.2, 0.7, 0.97) for nu in (2.1, 3.0, 4.5, 10.0, 28.0)]),
    "clayton": (F.clayton, [[t] for t in (0.05, 0.5, 2.0, 6.0, 15.0, 27.0)]),
    "gumbel": (F.gumbel, [[t] for t in (1.02, 1.3, 2.0, 5.0, 12.0, 25.0)]),
    "frank": (F.frank, [[t] for t in (-30.0, -8.0, -0.5, 0.3, 3.0, 12.0, 34.0)]),
    "joe": (F.joe, [[t] for t in (1.05, 1.5, 3.0, 8.0, 20.0)]),
    "bb1": (F.bb1, [[a, b] for a in (0.05, 0.6, 2.5, 6.0) for b in (1.02, 1.6, 3.5, 6.5)]),
    "bb6": (F.bb6, [[a, b] for a in (1.05, 1.8, 3.5, 5.5) for b in (1.02, 1.6, 3.5, 7.0)]),
    "bb7": (F.bb7, [[a, b] for a in (1.05, 1.8, 3.5, 5.5) for b in (0.05, 0.8, 3.0, 20.0)]),
    "bb8": (F.bb8, [[a, b] for a in (1.2, 2.5, 5.0, 7.5) for b in (0.1, 0.5, 0.9, 0.99)]),
}
ROT = {"R0": 0, "R90": 90, "R180": 180, "R270": 270}
ROTATABLE = {"clayton", "gumbel", "joe", "bb1", "bb6", "bb7", "bb8"}

pts = np.array([1e-6, 1e-4, 1e-3, 0.01, 0.05, 0.2, 0.35, 0.5, 0.65, 0.8, 0.95, 0.99, 0.999, 1 - 1e-4, 1 - 1e-6])
U1, U2 = [a.ravel() for a in np.meshgrid(pts, pts)]
U = np.column_stack([U1, U2])

def rel(a, b):
    return np.abs(a - b) / np.maximum(1e-300, np.maximum(np.abs(a), np.abs(b)))

worst = {}
rows = []
for fam, (pvf, plist) in GRID.items():
    for params in plist:
        for rname, rdeg in ROT.items():
            if rname != "R0" and fam not in ROTATABLE:
                continue
            ours = rs.PairCopula.from_spec(fam, params, rotation=rname)
            theirs = pv.Bicop(family=pvf, rotation=rdeg, parameters=np.array(params, float).reshape(-1, 1))
            with np.errstate(all="ignore"):
                ref_pdf = theirs.pdf(U)
                o_lp = ours.log_pdf(U1, U2, clip_eps=1e-12)
                ok = np.isfinite(ref_pdf) & (ref_pdf > 1e-250) & (ref_pdf < 1e250)
                d_lp = np.abs(o_lp - np.log(np.where(ok, ref_pdf, 1.0)))
                d_lp = np.where(ok, d_lp, 0.0)
                h1 = np.abs(ours.cond_second_given_first(U1, U2, clip_eps=1e-12) - theirs.hfunc1(U))
                h2 = np.abs(ours.cond_first_given_second(U1, U2, clip_eps=1e-12) - theirs.hfunc2(U))
                # round trip through OUR inverse of THEIR h-function value
                i1 = np.abs(ours.inv_second_given_first(U1, U2, clip_eps=1e-12) - theirs.hinv1(U))
                i2 = np.abs(ours.inv_first_given_second(U1, U2, clip_eps=1e-12) - theirs.hinv2(U))
            for name, d in (("logpdf", d_lp), ("h1", h1), ("h2", h2), ("hinv1", i1), ("hinv2", i2)):
                k = int(np.nanargmax(d)); nan = int(np.isnan(d).sum())
                rows.append((fam, tuple(params), rname, name, float(d[k]), U1[k], U2[k], nan))
for name in ("logpdf", "h1", "h2", "hinv1", "hinv2"):
    print("==", name)
    sub = [r for r in rows if r[3] == name]
    for fam in GRID:
        fs = sorted([r for r in sub if r[0] == fam], key=lambda r: -r[4])
        r = fs[0]
        med = np.median([x[4] for x in fs])
        print(f"  {fam:10s} worst {r[4]:.2e} at params={r[1]} {r[2]} u=({r[5]:g},{r[6]:g}) nan={sum(x[7] for x in fs)} median-of-max {med:.1e}")
