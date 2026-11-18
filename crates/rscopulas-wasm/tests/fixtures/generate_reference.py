"""Reference values for the WebAssembly tests, computed by the Python package.

The Python package runs the same Rust core compiled natively. The Node tests
in `tests/node/` load this file and require the WebAssembly build to give the
same numbers: densities, h-functions, Rosenblatt transforms, fitted
log-likelihoods, and seeded samples.

Usage (from the repository root, with the rscopulas Python package built from
the same commit installed):

    python crates/rscopulas-wasm/tests/fixtures/generate_reference.py

Writes `reference.json` next to this script.
"""

import json
import pathlib

import numpy as np

from rscopulas import (
    ClaytonCopula,
    FrankCopula,
    GaussianCopula,
    GumbelCopula,
    PairCopula,
    StudentTCopula,
    VineCopula,
    __version__,
    to_pseudo_obs,
)

OUT = pathlib.Path(__file__).with_name("reference.json")
DIM = 3
SEED = 42


def flat(values):
    return np.asarray(values, dtype=np.float64).ravel().tolist()


def model_block(family, result, data):
    model = result.model
    return {
        "family": family,
        "json": model.to_json(),
        "loglik": result.diagnostics.loglik,
        "aic": result.diagnostics.aic,
        "bic": result.diagnostics.bic,
        "log_pdf": flat(model.log_pdf(data[:25])),
        "sample": flat(model.sample(6, seed=SEED)),
    }


def main():
    # Asymmetric dependence, so the families and the vine really differ.
    correlation = np.array([[1.0, 0.6, 0.3], [0.6, 1.0, 0.5], [0.3, 0.5, 1.0]])
    data = StudentTCopula.from_params(correlation, 4.0).sample(400, seed=7)

    models = [
        model_block("gaussian", GaussianCopula.fit(data), data),
        model_block("student_t", StudentTCopula.fit(data), data),
        model_block("clayton", ClaytonCopula.fit(data), data),
        model_block("frank", FrankCopula.fit(data), data),
        model_block("gumbel", GumbelCopula.fit(data), data),
    ]

    vines = []
    for kind, fit, options in [
        ("r", VineCopula.fit_r, {}),
        ("c", VineCopula.fit_c, {"criterion": "bic"}),
        ("d", VineCopula.fit_d, {"family_set": ["gaussian", "clayton", "gumbel"], "truncation_level": 1}),
    ]:
        result = fit(data, **options)
        block = model_block("vine", result, data)
        block["kind"] = kind
        block["options"] = {
            "kind": kind,
            **({"criterion": options["criterion"]} if "criterion" in options else {}),
            **({"familySet": options["family_set"]} if "family_set" in options else {}),
            **({"truncationLevel": options["truncation_level"]} if "truncation_level" in options else {}),
        }
        block["rosenblatt"] = flat(result.model.rosenblatt(data[:10]))
        block["inverse_rosenblatt"] = flat(result.model.inverse_rosenblatt(data[:10]))
        vines.append(block)

    u1 = np.array([1e-6, 0.03, 0.2, 0.5, 0.81, 0.97, 1 - 1e-6])
    u2 = np.array([0.4, 1e-6, 0.9, 0.5, 0.13, 1 - 1e-6, 0.6])
    pairs = []
    for family, parameters, rotation in [
        ("independence", [], "R0"),
        ("gaussian", [0.7], "R0"),
        ("student_t", [-0.4, 3.5], "R0"),
        ("clayton", [2.5], "R0"),
        ("clayton", [2.5], "R90"),
        ("frank", [-6.0], "R0"),
        ("gumbel", [1.8], "R180"),
        ("joe", [3.0], "R270"),
        ("bb1", [0.5, 1.6], "R0"),
        ("bb6", [1.5, 1.3], "R0"),
        ("bb7", [1.8, 1.2], "R180"),
        ("bb8", [4.0, 0.95], "R0"),
        ("tawn1", [3.0, 0.8], "R0"),
        ("tawn2", [3.0, 0.8], "R90"),
    ]:
        pair = PairCopula.from_spec(family, parameters, rotation=rotation)
        pairs.append({
            "family": family,
            "parameters": parameters,
            "rotation": rotation,
            "json": pair.to_json(),
            "log_pdf": flat(pair.log_pdf(u1, u2)),
            "cond_first_given_second": flat(pair.cond_first_given_second(u1, u2)),
            "cond_second_given_first": flat(pair.cond_second_given_first(u1, u2)),
            "inv_first_given_second": flat(pair.inv_first_given_second(u1, u2)),
            "inv_second_given_first": flat(pair.inv_second_given_first(u1, u2)),
        })

    raw = np.array([[3.0, 10.0], [1.0, 20.0], [2.0, 20.0], [2.0, 5.0], [9.0, -1.0]])
    payload = {
        "version": __version__,
        "dim": DIM,
        "seed": SEED,
        "data": flat(data),
        "models": models,
        "vines": vines,
        "pair_points": {"u1": u1.tolist(), "u2": u2.tolist()},
        "pairs": pairs,
        "ranks": {"raw": flat(raw), "dim": 2, "pseudo_obs": flat(to_pseudo_obs(raw))},
    }
    OUT.write_text(json.dumps(payload) + "\n")
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
