// Behaviour that only exists on the WebAssembly side of the boundary:
// entropy from the host, errors as JavaScript exceptions, argument checks,
// and the browser (`--target web`) build.
//
//   npx wasm-pack build crates/rscopulas-wasm --target nodejs --out-dir pkg-node
//   npx wasm-pack build crates/rscopulas-wasm --target web --out-dir pkg-web
//   node --test crates/rscopulas-wasm/tests/node/

import assert from "node:assert/strict";
import { existsSync, readFileSync } from "node:fs";
import { createRequire } from "node:module";
import { test } from "node:test";

const require = createRequire(import.meta.url);
const { Copula, PairCopula } = require("../../pkg-node/rscopulas_wasm.js");

const correlation = Float64Array.from([1, 0.7, 0.7, 1]);

test("an unseeded sample draws entropy from the host", () => {
  const model = Copula.gaussian(correlation, 2);
  const [first, second] = [model.sample(50), model.sample(50)];
  assert.equal(first.length, 100);
  assert.notDeepEqual(first, second);
  assert.ok(first.every((value) => value > 0 && value < 1));
});

test("a seed reproduces the sample, and different seeds differ", () => {
  const model = Copula.studentT(correlation, 2, 4);
  assert.deepEqual(model.sample(20, 7), model.sample(20, 7));
  assert.notDeepEqual(model.sample(20, 7), model.sample(20, 8));
  // 2^53 - 1 is the largest seed a JavaScript number carries exactly.
  assert.equal(model.sample(1, Number.MAX_SAFE_INTEGER).length, 2);
});

test("a large sample has the dependence it was drawn with", () => {
  const n = 200_000;
  const sample = Copula.gaussian(correlation, 2).sample(n, 1);
  // Spearman's rho of a Gaussian copula: 6 / pi * asin(rho / 2) = 0.6829.
  let sum = 0;
  for (let i = 0; i < n; i += 1) {
    sum += (sample[2 * i] - 0.5) * (sample[2 * i + 1] - 0.5);
  }
  const spearman = (12 * sum) / n;
  assert.ok(Math.abs(spearman - 0.6829) < 0.01, `Spearman rho = ${spearman}`);
});

test("failures are thrown as Error with the core's message", () => {
  const model = Copula.clayton(2, 2.0);
  const cases = [
    [() => Copula.clayton(2, -1), /./],
    [() => Copula.gaussian(Float64Array.from([1, 0.5, 0.5]), 2), /2 x 2 correlation matrix/],
    [() => Copula.fitGaussian(Float64Array.from([0.1, 0.2, 0.3]), 2), /not a multiple of dim/],
    [() => Copula.fitGaussian(Float64Array.from([0.1, 1.2, 0.3, 0.4]), 2), /strictly inside \(0, 1\)/],
    [() => Copula.fitGaussian(new Float64Array(0), 2), /at least one observation/],
    [() => model.logPdf(Float64Array.from([0.1, 0.2, 0.3])), /not a multiple of dim/],
    [() => model.sample(0), /at least 1/],
    [() => model.sample(5, -1), /seed must be a non-negative integer/],
    [() => model.sample(5, 1.5), /seed must be a non-negative integer/],
    [() => model.rosenblatt(Float64Array.from([0.1, 0.2])), /vine copulas only/],
    [() => Copula.fromJson("clayton", "{"), /failed to deserialize Clayton copula/],
    [() => Copula.fromJson("clayton", JSON.stringify({ dim: 2, theta: -3 })), /failed to deserialize Clayton copula/],
    [() => Copula.fromJson("mystery", "{}"), /unsupported family 'mystery'/],
    [() => Copula.fitVine(Float64Array.from([0.1, 0.2, 0.3, 0.4]), 2, { kind: "x" }), /unsupported vine kind 'x'/],
    [() => Copula.fitVine(Float64Array.from([0.1, 0.2, 0.3, 0.4]), 2, { familySet: ["nope"] }), /unsupported pair family 'nope'/],
    // A misspelt option is an error, not silently ignored.
    [() => Copula.fitVine(Float64Array.from([0.1, 0.2, 0.3, 0.4]), 2, { truncationLvl: 1 }), /invalid options: unknown field `truncationLvl`/],
    [() => new PairCopula("clayton", Float64Array.from([1, 2])), /wrong number of parameters for 'clayton'/],
    [() => new PairCopula("gumbel", Float64Array.from([0.5])), /./],
    [() => new PairCopula("clayton", Float64Array.from([2]), "R45"), /unsupported rotation 'R45'/],
    [() => new PairCopula("clayton", Float64Array.from([2])).logPdf(Float64Array.from([0.1]), Float64Array.from([0.1, 0.2])), /same length/],
  ];
  for (const [call, pattern] of cases) {
    assert.throws(call, (error) => error instanceof Error && pattern.test(error.message), String(call));
  }
  // The module still works after errors.
  assert.equal(model.sample(3, 1).length, 6);
});

test("a vine fits in a single-threaded WebAssembly instance", () => {
  const truth = Copula.studentT(Float64Array.from([1, 0.6, 0.3, 0.2, 0.6, 1, 0.5, 0.4, 0.3, 0.5, 1, 0.7, 0.2, 0.4, 0.7, 1]), 4, 5);
  const sample = truth.sample(1500, 3);
  const started = performance.now();
  const vine = Copula.fitVine(sample, 4);
  const seconds = (performance.now() - started) / 1000;
  assert.equal(vine.dim, 4);
  // A vine fitted to t data must beat the Gaussian copula on the same data.
  const gaussian = Copula.fitGaussian(sample, 4);
  assert.ok(vine.loglik > gaussian.loglik, `${vine.loglik} vs ${gaussian.loglik}`);
  assert.ok(seconds < 60, `vine fit took ${seconds.toFixed(1)} s`);
  // Rosenblatt of the fitting data is close to independent uniforms.
  const independent = vine.rosenblatt(sample);
  let sum = 0;
  for (let i = 0; i < 1500; i += 1) {
    sum += (independent[4 * i] - 0.5) * (independent[4 * i + 3] - 0.5);
  }
  assert.ok(Math.abs((12 * sum) / 1500) < 0.08, "columns 0 and 3 are uncorrelated after the transform");
});

test("the browser build loads from bytes and gives the same answers", { skip: !existsSync(new URL("../../pkg-web/rscopulas_wasm.js", import.meta.url)) }, async () => {
  const web = await import("../../pkg-web/rscopulas_wasm.js");
  web.initSync({ module: readFileSync(new URL("../../pkg-web/rscopulas_wasm_bg.wasm", import.meta.url)) });
  const here = Copula.gaussian(correlation, 2);
  const there = web.Copula.gaussian(correlation, 2);
  assert.deepEqual(there.sample(10, 3), here.sample(10, 3));
  assert.deepEqual(there.logPdf(Float64Array.from([0.2, 0.3, 0.8, 0.9])), here.logPdf(Float64Array.from([0.2, 0.3, 0.8, 0.9])));
  assert.notDeepEqual(there.sample(10), there.sample(10));
});
