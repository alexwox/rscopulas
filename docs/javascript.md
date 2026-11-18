# JavaScript API (WebAssembly)

`crates/rscopulas-wasm` compiles the Rust core to WebAssembly. The same
fitting, density and sampling code that backs the Python package runs in a
browser or in Node.js, with no server.

It is not published to npm yet. Build it with
[`wasm-pack`](https://rustwasm.github.io/wasm-pack/):

```sh
rustup target add wasm32-unknown-unknown
npx wasm-pack build crates/rscopulas-wasm --release --target web --out-dir pkg-web
```

Use `--target nodejs` for Node.js and `--target bundler` for webpack or Vite.

## Conventions

- A matrix is a row-major `Float64Array` plus its number of columns: row `i`,
  column `j` is `data[i * dim + j]`.
- Failures throw an `Error` carrying the core's message.
- A seed is a non-negative integer. The same seed gives the same sample as
  the Python package.

## Example

```js
import init, { Copula, toPseudoObs } from "./pkg-web/rscopulas_wasm.js";
await init();

const dim = 3;
const u = toPseudoObs(raw, dim);
const vine = Copula.fitVine(u, dim, { criterion: "bic" });

vine.loglik;                         // fit scores
vine.logPdf(u);                      // Float64Array, one value per row
vine.sample(10_000, 42);             // flat (10000, 3) matrix
```

## Sharing models with Python

JSON is the same on both sides:

```python
payload = VineCopula.fit_r(u).model.to_json()      # Python
```

```js
const vine = Copula.fromJson("vine", payload);     // JavaScript
```

and `vine.toJson()` loads with `VineCopula.from_json` in Python.

## Limits

- A WebAssembly instance has one thread, so fits that run in parallel
  natively run in sequence. Expect about a tenth of native speed on a
  multi-core machine for vine fits and sampling. Use a Web Worker for long
  fits.
- Hierarchical Archimedean and factor copulas are not bound yet.

The full API table, an example page and the test set-up are in
[`crates/rscopulas-wasm/README.md`](../crates/rscopulas-wasm/README.md).
