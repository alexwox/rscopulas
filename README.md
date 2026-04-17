# rscopulas

`rscopulas` is a Rust-first copula library for fitting and sampling multivariate
copulas from pseudo-observations. The repository is structured as a small
workspace from the start so the statistical core stays isolated from hardware
acceleration and language bindings.

## Workspace layout

- `crates/rscopulas-core`: domain models, fit orchestration, numerical helpers
- `crates/rscopulas-accel`: optional CPU/GPU dispatch and backend integration
- `crates/rscopulas-python`: PyO3 bindings for the high-level API

## Current status

The repository currently contains the architectural scaffold:

- validated pseudo-observation input type
- domain model skeletons for the target copula families
- execution policy and backend capability types
- error hierarchy and fit/eval/sample option types

The actual fitting, density, and sampling math is still to be implemented.

