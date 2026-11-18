# rscopulas for JavaScript (WebAssembly)

The `rscopulas` Rust core compiled to WebAssembly, for browsers and Node.js:
fit copulas, evaluate densities, and sample, without a server. It is the same
code that backs the Python package, and a model saved by one loads in the
other.

Not published to npm yet; build it from this repository.

## Build

```sh
rustup target add wasm32-unknown-unknown

# Node.js (CommonJS)
npx wasm-pack build crates/rscopulas-wasm --release --target nodejs --out-dir pkg-node
# Browsers, as an ES module
npx wasm-pack build crates/rscopulas-wasm --release --target web --out-dir pkg-web
# Bundlers (webpack, Vite, ...)
npx wasm-pack build crates/rscopulas-wasm --release --target bundler --out-dir pkg
```

Each output folder is an npm package with TypeScript declarations. The
WebAssembly file is 1.1 MB (380 kB gzipped).

## Use

```js
import init, { Copula, PairCopula, toPseudoObs } from "./pkg-web/rscopulas_wasm.js";
await init();                                   // browsers; not needed in Node

// A matrix is a row-major Float64Array plus its number of columns.
const dim = 3;
const u = toPseudoObs(raw, dim);                // rank transform of raw data

const vine = Copula.fitVine(u, dim, { criterion: "bic" });
console.log(vine.loglik, vine.aic, vine.bic);

const logDensity = vine.logPdf(u);              // Float64Array, one per row
const simulated = vine.sample(10_000, 42);      // seeded: reproducible
const saved = vine.toJson();
const again = Copula.fromJson("vine", saved);
```

### `Copula`

| Call | Result |
| --- | --- |
| `Copula.fitGaussian(u, dim)`, `fitStudentT`, `fitClayton`, `fitFrank`, `fitGumbel` | Fitted model |
| `Copula.fitVine(u, dim, options?)` | Fitted R-, C- or D-vine |
| `Copula.gaussian(correlation, dim)`, `studentT(correlation, dim, nu)`, `clayton(dim, theta)`, `frank(dim, theta)`, `gumbel(dim, theta)` | Model from parameters |
| `Copula.fromJson(family, json)` / `model.toJson()` | Load / save |
| `model.family`, `model.dim` | `"gaussian"`, `"student_t"`, `"clayton"`, `"frank"`, `"gumbel"`, `"vine"` |
| `model.loglik`, `model.aic`, `model.bic` | Fit scores; `undefined` unless the model was fitted |
| `model.logPdf(u)` | Log-density of each row |
| `model.sample(n, seed?)` | `n` draws, flat `(n, dim)` |
| `model.rosenblatt(u)`, `model.inverseRosenblatt(u)` | Vine only |

`options` for `fitVine` (all optional): `kind` (`"r"`, `"c"`, `"d"`),
`familySet`, `includeRotations`, `criterion` (`"aic"`, `"bic"`, `"mbicv"`),
`truncationLevel`, `independenceTestLevel`, `order`. An unknown option name is
an error.

### `PairCopula`

```js
const pair = new PairCopula("clayton", Float64Array.of(2.5), "R180");
pair.logPdf(u1, u2);
pair.condFirstGivenSecond(u1, u2);     // h-function P(U1 <= u1 | U2 = u2)
pair.invFirstGivenSecond(p, u2);       // its inverse
const best = PairCopula.fit(u1, u2, { familySet: ["gaussian", "clayton", "gumbel"] });
```

### Helpers

`toPseudoObs(raw, dim)` (ranks over `n + 1`, ties averaged),
`kendallTauMatrix(u, dim)`, `version()`.

## Things to know

- **Errors.** Invalid input throws an `Error` carrying the core's message.
- **Seeds.** A seed is a non-negative integer up to `2^53 - 1`. The same seed
  gives the same sample as the Python package. Without a seed, entropy comes
  from `crypto.getRandomValues`.
- **Sharing models with Python.** `toJson()` output loads with
  `VineCopula.from_json(...)` (and the other classes) in Python, and the other
  way round with `Copula.fromJson(family, json)`.
- **Single-threaded.** The native build fits pair copulas in parallel; a
  WebAssembly instance has one thread. On an M3 Pro a default R-vine fit on
  2000 × 5 observations takes 5.3 s here against 0.54 s natively, and
  100 000 vine draws 1.2 s against 0.12 s. Call it from a Web Worker to keep a
  page responsive.
- **Memory.** Objects returned to JavaScript hold WebAssembly memory. Call
  `model.free()` when you create many of them; otherwise the garbage collector
  reclaims them eventually.
- **Not bound yet.** Hierarchical Archimedean and factor copulas, Khoudraji
  and TLL construction from parameters, conditional sampling helpers.

## Example page

`examples/browser/index.html` draws data, fits a vine and samples from it, all
in the page:

```sh
npx wasm-pack build crates/rscopulas-wasm --release --target web --out-dir pkg-web
python3 -m http.server --directory crates/rscopulas-wasm 8000
# open http://localhost:8000/examples/browser/
```

## Tests

```sh
cargo test -p rscopulas-wasm                          # binding logic, natively
node --test "crates/rscopulas-wasm/tests/node/*.test.mjs"   # the built package
```

The Node tests compare the WebAssembly build with values computed by the
native build (`tests/fixtures/reference.json`, written by
`tests/fixtures/generate_reference.py` through the Python package):

| Quantity | Largest difference |
| --- | --- |
| Density, sample and Rosenblatt transform of a loaded model | 3e-15 |
| Pair-copula density and h-functions | 3e-15 |
| Pair-copula inverse h-functions | 5e-13 |
| Log-likelihood of a model fitted in WebAssembly (single family) | 4e-16 |
| Log-likelihood of a vine fitted in WebAssembly | 5e-10 |

The two builds share the source but not the math library (`exp`, `ln`, `pow`
come from the system natively and from Rust's `libm` port in WebAssembly),
which is why the agreement is close but not bit for bit.

## How the build is set up

- `rand` seeds itself through `getrandom`, which needs its JavaScript backend
  on `wasm32-unknown-unknown`. The core crate enables the `wasm_js` feature
  for that target and the repository's `.cargo/config.toml` passes
  `--cfg getrandom_backend="wasm_js"`. If you depend on `rscopulas` from your
  own WebAssembly crate, add the same flag to your build.
- `rayon` has no threads to spawn here and falls back to running on the
  calling thread, so the parallel code paths work unchanged.
- The GPU backends are compiled only for their own operating systems and are
  absent from this build.
