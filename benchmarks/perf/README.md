# Performance suite

`crates/rscopulas-core/examples/perf_suite.rs` times every numerical feature
of the core crate: each pair-copula kernel (density, h-function, inverse
h-function), every pair fitter, the rank statistics, the single-family
copulas, hierarchical Archimedean and factor copulas, and vine fitting,
density evaluation, sampling and Rosenblatt transforms.

It shares its models and inputs with the golden-snapshot tests
(`crates/rscopulas-core/tests/support/`), so everything that is timed is also
pinned numerically.

## Running

```sh
# Full run (about a minute), medians printed per case
cargo run --release -p rscopulas --example perf_suite

# Only some cases, with a short time budget per case
cargo run --release -p rscopulas --example perf_suite -- --filter vine_fit --quick

# Save a run, then compare a later run against it
cargo run --release -p rscopulas --example perf_suite -- --save before.json
cargo run --release -p rscopulas --example perf_suite -- --compare before.json
```

Each case runs for a fixed wall-clock budget (at least three iterations) and
reports the median iteration time. `--compare` prints the speed-up per case
and the geometric mean.

Timings move by 5–10 % with machine load and temperature. To compare two
revisions, build both and run them back to back rather than against a file
recorded on another day.

## Files

- `baseline-0.4.0.json`: release 0.4.0 (commit `bd07898`) on an Apple M3 Pro
  (12 cores).
- `after.json`: the tree after the core-math performance work, same machine.

## Profiling

A `profiling` Cargo profile (release plus line-level debug info) is defined in
the workspace manifest:

```sh
cargo build --profile profiling -p rscopulas --example perf_suite
# then sample ./target/profiling/examples/perf_suite --filter <case>
# with your profiler of choice (Instruments, `sample`, perf, samply, …)
```
