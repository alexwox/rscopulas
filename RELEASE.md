# Release checklist

Rust and Python releases use the same source commit and version. The Rust crate
is published with `cargo publish`; GitHub Actions builds and publishes Python
wheels and the source distribution. This checklist does not itself authorize a
release: obtain the maintainer's release approval before pushing a tag or publishing.

## 1. Choose and record the version

Use a patch for compatible fixes. Public API changes need a minor bump; while
pre-1.0, breaking changes also need a minor bump and explicit migration notes.
Version 0.3.0 includes the likelihood, TLL, factor, and vine serialization changes
in [CHANGELOG.md](CHANGELOG.md) and the
[migration guide](docs/mdx/guides/migrating-to-0-3.mdx).

Keep these versions synchronized:

- `[workspace.package].version` and the `rscopulas` workspace dependency in `Cargo.toml`.
- `[project].version` in `pyproject.toml`.
- Local package entries in `Cargo.lock` and `uv.lock` (regenerate with `cargo check --workspace` and `uv lock`).

Merge the release changes through a reviewed PR, then start from a clean,
up-to-date `master`. Set `TASK_RELEASE_VERSION` to the approved version:

```bash
export TASK_RELEASE_VERSION=0.3.0
git fetch origin
git checkout master
git pull --ff-only origin master
git status --short
```

## 2. Validate the release commit

Local checks cover the current platform. The CI matrix covers Linux, macOS,
and Windows; a local run cannot replace those jobs.

```bash
cargo fmt --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --release --all-features
cargo bench --no-run
python -m pip install --upgrade maturin pytest numpy matplotlib
maturin develop --release --locked
pytest
cargo publish -p rscopulas --dry-run
```

Use an activated virtual environment for `maturin develop`. If its automatic
installer is unavailable, build and install a wheel instead:

```bash
maturin build --release --locked --out dist
python -m pip install --no-index --no-deps --force-reinstall --find-links dist "rscopulas==$TASK_RELEASE_VERSION"
python -I scripts/smoke_wheel.py
pytest
```

The Rust test command includes doctests. Install SciPy too when checking tests
that use optional SciPy references. CI installs pytest, NumPy, and matplotlib;
optional SciPy tests must use `pytest.importorskip`.

The documentation workflow independently runs `npm ci`, `npm audit`, and
`npm run build` in `demo/` on PRs and pushes to the main branches. Fix relevant
documentation failures, but a demo dependency advisory does not gate a library
release or PyPI upload.

## 3. Tag after approval and successful validation

```bash
git tag -a "v$TASK_RELEASE_VERSION" -m "Release $TASK_RELEASE_VERSION"
git push origin "v$TASK_RELEASE_VERSION"
```

The tag triggers `.github/workflows/release.yml`:

1. Check that the tag matches the project version.
2. Call reusable library CI: formatting, Clippy, Rust tests, Python tests, and
   benchmark compilation. The separate documentation workflow is not called.
3. Build glibc/musl Linux wheels for x86_64 and ARM64, macOS wheels for Intel and
   ARM64, Windows x64 wheels, and the source distribution.
4. Install and exercise compatible wheels on seven platform/libc combinations
   with `scripts/smoke_wheel.py`.
5. Publish to PyPI only after validation, builds, and smoke tests succeed.

A manual `workflow_dispatch` runs validation/builds/smoke tests without publishing
unless its selected ref is a version tag. A failed reusable CI job or wheel smoke
job blocks the PyPI publish job. Review environment approval prompts if the
`pypi` environment requires them.

## 4. Publish and verify both channels

After the release validation succeeds and the maintainer approves publication:

```bash
cargo publish -p rscopulas
```

Monitor the tag's Release workflow for PyPI publication. Install the published
Python version in a fresh environment and run the same smoke script:

```bash
python -m venv /tmp/rscopulas-release-verify
/tmp/rscopulas-release-verify/bin/python -m pip install "rscopulas==$TASK_RELEASE_VERSION"
/tmp/rscopulas-release-verify/bin/python -I scripts/smoke_wheel.py
```

Verify the published Rust and Python versions, then add the changelog to the
GitHub release notes. Do not mark a release complete merely because the tag or
build exists.

## Recovery

If validation fails before publication, fix through a PR and rerun validation.
Do not delete or move a public release tag to conceal a failed release. Use a new
version after any artifact has been published; registry versions are immutable.
If a package version already exists, check the registries before deciding which
version to release next.

Update this checklist whenever the workflows change.
