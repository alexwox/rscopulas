// The WebAssembly build against the native build.
//
// `tests/fixtures/reference.json` holds values computed by the Python
// package, which runs the same Rust core compiled natively. The two builds
// share the source but not the math library (`exp`, `ln`, `pow` come from
// the system natively and from Rust's `libm` port in WebAssembly), so
// agreement is checked to a tolerance, not bit for bit.
//
//   npx wasm-pack build crates/rscopulas-wasm --target nodejs --out-dir pkg-node
//   node --test crates/rscopulas-wasm/tests/node/

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import { test } from "node:test";

const require = createRequire(import.meta.url);
const { Copula, PairCopula, kendallTauMatrix, toPseudoObs, version } = require("../../pkg-node/rscopulas_wasm.js");
const reference = JSON.parse(readFileSync(new URL("../fixtures/reference.json", import.meta.url), "utf8"));

const data = Float64Array.from(reference.data);
const dim = reference.dim;
const head = data.subarray(0, 25 * dim);

/** Largest |got - expected| / max(1, |expected|). */
function worst(got, expected) {
  assert.equal(got.length, expected.length, "length");
  let error = 0;
  for (let i = 0; i < got.length; i += 1) {
    error = Math.max(error, Math.abs(got[i] - expected[i]) / Math.max(1, Math.abs(expected[i])));
  }
  return error;
}

function close(got, expected, tolerance, label) {
  const error = worst(got, expected);
  assert.ok(error <= tolerance, `${label}: worst error ${error} > ${tolerance}`);
}

const fitters = {
  gaussian: Copula.fitGaussian,
  student_t: Copula.fitStudentT,
  clayton: Copula.fitClayton,
  frank: Copula.fitFrank,
  gumbel: Copula.fitGumbel,
};

test("version matches the fixture", () => {
  assert.equal(version(), reference.version);
});

for (const expected of reference.models) {
  test(`${expected.family}: a model saved by Python evaluates and samples the same`, () => {
    const model = Copula.fromJson(expected.family, expected.json);
    assert.equal(model.family, expected.family);
    assert.equal(model.dim, dim);
    assert.equal(model.loglik, undefined);
    close(model.logPdf(head), expected.log_pdf, 1e-12, "logPdf");
    // Same seed, same generator: the same draws as Python.
    close(model.sample(6, reference.seed), expected.sample, 1e-12, "sample");
    // Its JSON loads back to the same model.
    const again = Copula.fromJson(expected.family, model.toJson());
    assert.deepEqual(again.logPdf(head), model.logPdf(head));
    assert.deepEqual(JSON.parse(model.toJson()), JSON.parse(expected.json));
  });

  test(`${expected.family}: fitting in WebAssembly reaches the native fit`, () => {
    const model = fitters[expected.family](data, dim);
    close([model.loglik, model.aic, model.bic], [expected.loglik, expected.aic, expected.bic], 1e-9, "scores");
    close(model.logPdf(head), expected.log_pdf, 1e-8, "logPdf of the fitted model");
    close(model.sample(6, reference.seed), expected.sample, 1e-8, "sample of the fitted model");
  });
}

for (const expected of reference.vines) {
  test(`${expected.kind}-vine: a model saved by Python evaluates, samples and transforms the same`, () => {
    const model = Copula.fromJson("vine", expected.json);
    const rows = data.subarray(0, 10 * dim);
    close(model.logPdf(head), expected.log_pdf, 1e-11, "logPdf");
    close(model.sample(6, reference.seed), expected.sample, 1e-11, "sample");
    close(model.rosenblatt(rows), expected.rosenblatt, 1e-11, "rosenblatt");
    close(model.inverseRosenblatt(rows), expected.inverse_rosenblatt, 1e-11, "inverseRosenblatt");
    // The two transforms invert each other.
    close(model.inverseRosenblatt(model.rosenblatt(rows)), rows, 1e-9, "round trip");
  });

  test(`${expected.kind}-vine: fitting in WebAssembly reaches the native fit`, () => {
    const model = Copula.fitVine(data, dim, expected.options);
    assert.equal(model.family, "vine");
    // Pair selection compares information criteria of candidate families;
    // the fitted likelihood must match, and with it the chosen structure.
    close([model.loglik], [expected.loglik], 1e-7, "loglik");
    close(model.logPdf(head), expected.log_pdf, 1e-5, "logPdf of the fitted vine");
    const [ours, theirs] = [JSON.parse(model.toJson()), JSON.parse(expected.json)];
    const families = (payload) => JSON.stringify(payload).match(/"family":"[A-Za-z0-9]+"/g);
    assert.deepEqual(families(ours), families(theirs), "same pair families on every edge");
  });
}

for (const expected of reference.pairs) {
  const label = `${expected.family} ${expected.rotation}`;
  test(`pair ${label}: kernels match the native build`, () => {
    const u1 = Float64Array.from(reference.pair_points.u1);
    const u2 = Float64Array.from(reference.pair_points.u2);
    const pair = new PairCopula(expected.family, Float64Array.from(expected.parameters), expected.rotation);
    assert.equal(pair.family, expected.family);
    assert.equal(pair.rotation, expected.rotation);
    assert.deepEqual(Array.from(pair.parameters), expected.parameters);
    close(pair.logPdf(u1, u2), expected.log_pdf, 1e-11, "logPdf");
    close(pair.condFirstGivenSecond(u1, u2), expected.cond_first_given_second, 1e-12, "condFirstGivenSecond");
    close(pair.condSecondGivenFirst(u1, u2), expected.cond_second_given_first, 1e-12, "condSecondGivenFirst");
    close(pair.invFirstGivenSecond(u1, u2), expected.inv_first_given_second, 1e-9, "invFirstGivenSecond");
    close(pair.invSecondGivenFirst(u1, u2), expected.inv_second_given_first, 1e-9, "invSecondGivenFirst");
    // Python's JSON for the same pair loads to the same kernel.
    const loaded = PairCopula.fromJson(expected.json);
    assert.deepEqual(loaded.logPdf(u1, u2), pair.logPdf(u1, u2));
    assert.deepEqual(JSON.parse(pair.toJson()), JSON.parse(expected.json));
  });
}

test("toPseudoObs matches Python's rank transform, with ties", () => {
  const { raw, dim: columns, pseudo_obs: expected } = reference.ranks;
  assert.deepEqual(Array.from(toPseudoObs(Float64Array.from(raw), columns)), expected);
  assert.throws(() => toPseudoObs(Float64Array.from([1, NaN, 3, 4]), 2), /NaN/);
});

test("kendallTauMatrix is symmetric with a unit diagonal and the right sign", () => {
  const tau = kendallTauMatrix(data, dim);
  assert.equal(tau.length, dim * dim);
  for (let i = 0; i < dim; i += 1) {
    assert.equal(tau[i * dim + i], 1);
    for (let j = 0; j < dim; j += 1) {
      assert.equal(tau[i * dim + j], tau[j * dim + i]);
    }
  }
  // The data come from a t copula with correlations 0.6, 0.3, 0.5:
  // tau = 2 / pi * asin(rho) = 0.41, 0.19, 0.33.
  close([tau[1], tau[2], tau[5]], [0.41, 0.19, 0.33], 0.06, "tau");
});

test("PairCopula.fit picks the family the data came from", () => {
  const sample = Copula.clayton(2, 3.0).sample(2000, 5);
  const u1 = sample.filter((_, index) => index % 2 === 0);
  const u2 = sample.filter((_, index) => index % 2 === 1);
  const fitted = PairCopula.fit(u1, u2, { familySet: ["gaussian", "clayton", "gumbel", "frank"] });
  assert.equal(fitted.family, "clayton");
  assert.equal(fitted.rotation, "R0");
  assert.ok(Math.abs(fitted.parameters[0] - 3.0) < 0.3, `theta = ${fitted.parameters[0]}`);
  assert.ok(fitted.aic < 0 && fitted.loglik > 0);
});
