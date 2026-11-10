"""Multivariate copula densities vs R's `copula` package (run multivariate_density_reference.R first)."""
import pathlib
import json, numpy as np, rscopulas as rs
cases = json.load(open(pathlib.Path(__file__).with_name("multi.json")))
worst = {}
for c in cases:
    u = np.array(c["u"], float); ref = np.array([np.nan if x is None else x for x in c["logpdf"]], float)
    f, d = c["family"], c["dim"]
    if f == "clayton": m = rs.ClaytonCopula.from_params(d, c["par"])
    elif f == "gumbel": m = rs.GumbelCopula.from_params(d, c["par"])
    elif f == "frank": m = rs.FrankCopula.from_params(d, c["par"])
    elif f == "gaussian": m = rs.GaussianCopula.from_params(np.array(c["corr"]))
    else: m = rs.StudentTCopula.from_params(np.array(c["corr"]), c["par"])
    ours = m.log_pdf(u, clip_eps=1e-15)
    ok = np.isfinite(ref)
    e = np.abs(ours - ref)[ok] / np.maximum(1, np.abs(ref[ok]))
    k = (f, d)
    if e.max() >= worst.get(k, (0,))[0]: worst[k] = (e.max(), c["par"])
    if e.max() > 1e-8 or (~ok).any():
        i = np.argmax(e); print("  ", f, d, c["par"], "relerr %.2e" % e.max(), "ours", ours[ok][i], "R", ref[ok][i], "u", u[ok][i].round(5), "R-nonfinite", int((~ok).sum()))
for k, v in sorted(worst.items()): print(k, "%.2e" % v[0], v[1])
