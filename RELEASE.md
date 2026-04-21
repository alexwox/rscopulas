# Release checklist

This document is the single source of truth for cutting a new `rscopulas`
release. Releases ship **two artifacts from the same source tree and tag**:

| Channel     | Artifact                          | Published by                   |
| ----------- | --------------------------------- | ------------------------------ |
| `crates.io` | `rscopulas` Rust crate            | Manually, `cargo publish`      |
| PyPI        | `rscopulas` wheels + sdist        | `release.yml` on tag push      |

The `rscopulas-python` crate has `publish = false` and is never uploaded to
`crates.io`; it only exists to produce the Python extension module.

Follow the steps **in order**. Every step has a command you can copy-paste; do
not skip the preflight block.

---

## 0. Prerequisites (one-time setup)

- **`cargo login`** with a token that can publish `rscopulas` on crates.io.
- Push access to `origin/master` on `alexwox/rscopulas`.
- A `uv` install (`brew install uv`) for `uv.lock` regeneration.
- Python 3.12+ and `maturin` (`pip install maturin`) for the local smoke test.
- `scipy` in your local Python env (some tests opt-skip, but locally we want
  full coverage): `python -m pip install scipy`.
- PyPI trusted publisher already configured for this repo (it is); the
  `pypi` GitHub environment has **manual approval** enabled — you must click
  "Review deployments → Approve" after the release workflow builds wheels.

---

## 1. Decide the version

We use semver `MAJOR.MINOR.PATCH`:

- **PATCH** (`0.2.3 → 0.2.4`): bugfix, docs, perf, no public-API change.
- **MINOR** (`0.2.x → 0.3.0`): new public API, backward-compatible additions.
- **MAJOR**: breaking changes. We are pre-1.0, so breaking changes can also go
  in a MINOR bump — document them loudly in the commit body.

Set it once in your shell so the rest of the checklist is copy-paste:

```bash
export OLD=0.2.3
export NEW=0.2.4
```

---

## 2. Preflight — run **everything** CI runs, locally

**Do not skip this block.** Every time we skipped it, CI failed on something
trivial (rustfmt newlines, missing scipy, clippy warnings) and the release
pipeline got stuck waiting for a fix commit that landed *after* the tag.

Working tree must be clean and on `master`, up to date with `origin/master`:

```bash
git fetch origin
git checkout master
git pull --ff-only origin master
git status              # must be "nothing to commit, working tree clean"
```

Then run the full CI matrix locally, **in this exact order**, and fix anything
that fails before moving on:

```bash
# 2.1 Rust formatting — this is the one that bit us in 0.2.4.
cargo fmt --check

# 2.2 Clippy (same invocation as .github/workflows/ci.yml).
cargo clippy --workspace --all-targets --all-features -- -D warnings

# 2.3 Rust tests (workspace, all targets).
cargo test --workspace

# 2.4 Rust doctests (we ship runnable examples in crates/rscopulas-core/src/lib.rs).
cargo test --workspace --doc

# 2.5 Benchmarks compile (CI builds them with --no-run on ubuntu).
cargo bench --no-run

# 2.6 Build and install the Python extension from the current tree.
maturin develop --release

# 2.7 Python tests, identical to CI.
python -m pip install --upgrade pytest numpy
pytest

# 2.8 Optional but recommended: run with scipy too so the opt-skip test runs.
python -m pip install scipy
pytest

# 2.9 Dry-run the crates.io publish to surface packaging errors now.
cargo publish -p rscopulas --dry-run
```

If any of these fail, fix on `master` (or a PR) and restart step 2.

> Rule of thumb: **if `cargo fmt --check && cargo clippy ... -D warnings &&
> cargo test --workspace && pytest && cargo publish -p rscopulas --dry-run`
> is not green locally, you are not ready to tag.**

---

## 3. Bump the version

Five files reference the version. Bump all of them:

```bash
# 3.1 Workspace version + internal path dep.
sed -i '' "s/^version = \"$OLD\"$/version = \"$NEW\"/"                                Cargo.toml
sed -i '' "s|rscopulas = { version = \"$OLD\"|rscopulas = { version = \"$NEW\"|"      Cargo.toml

# 3.2 Python package version.
sed -i '' "s/^version = \"$OLD\"$/version = \"$NEW\"/" pyproject.toml

# 3.3 Install snippet in the core README (shown on crates.io).
sed -i '' "s/rscopulas = \"$OLD\"/rscopulas = \"$NEW\"/" crates/rscopulas-core/README.md
```

(On Linux, drop the empty `''` after `-i`.)

Regenerate the lockfiles:

```bash
cargo update -p rscopulas --precise "$NEW" --workspace
uv lock
```

Sanity check — the only version-string hits for the old number should be
unrelated transitive deps (e.g. `foreign-types-macros 0.2.3`):

```bash
rg "\"$OLD\"" Cargo.toml Cargo.lock pyproject.toml uv.lock crates/rscopulas-core/README.md
rg "rscopulas = \"$OLD\"" .
```

Rebuild once more to make sure the bumped tree compiles:

```bash
cargo check --workspace
```

---

## 4. Commit, tag, push

```bash
git add Cargo.toml Cargo.lock pyproject.toml uv.lock crates/rscopulas-core/README.md
git commit -m "release $NEW"
git tag -a "v$NEW" -m "Release $NEW"

# Push commit first, then the tag. The tag push triggers release.yml.
git push origin master
git push origin "v$NEW"
```

The `release.yml` workflow will now:

1. `check-version` — verifies `v$NEW` matches `pyproject.toml`. If you forgot
   to bump, this job fails and nothing else runs. **Fix:** bump
   `pyproject.toml`, delete the tag locally and on origin, retag, repush.
2. Build wheels: linux x86_64/aarch64 × (manylinux2014, musllinux_1_2),
   macOS aarch64 + x86_64, windows x64, plus sdist.
3. Collect all artifacts, then **wait for manual approval** on the `pypi`
   GitHub environment before uploading.

---

## 5. Publish to crates.io

While the wheels are building, publish the Rust crate. The dry-run in step 2.9
already verified packaging; this is just the real upload:

```bash
cargo publish -p rscopulas
```

Verify:

```bash
# Should list $NEW as the newest version within ~30s.
curl -s https://crates.io/api/v1/crates/rscopulas | jq '.crate.max_version'
```

---

## 6. Approve the PyPI deployment

1. Open <https://github.com/alexwox/rscopulas/actions/workflows/release.yml>.
2. Click the current run (`release $NEW`).
3. If wheels finished building you will see **"pypi — Review deployments"**.
   Click it, check the box for `pypi`, click **Approve and deploy**.
4. Wait for the `publish` job to go green.

Verify:

```bash
# Should print $NEW.
curl -s https://pypi.org/pypi/rscopulas/json | jq -r '.info.version'
```

Smoke test the published wheel in a fresh venv:

```bash
python -m venv /tmp/rscopulas-verify && source /tmp/rscopulas-verify/bin/activate
pip install "rscopulas==$NEW" numpy
python -c "import rscopulas, numpy as np; rng = np.random.default_rng(0); \
  u = rng.uniform(size=(256, 3)); v = rscopulas.VineCopula.fit_c(u); \
  print(v.sample(4, seed=1))"
deactivate
```

---

## 7. Post-release

- Edit the GitHub Release page for `v$NEW` with a short changelog.
- If you noticed anything while following this checklist that was wrong or
  incomplete, **update this file in the same commit as the fix**. This
  document is only useful if it stays honest.

---

## Troubleshooting

### CI fails on master *after* I tagged

The push-to-master CI (`ci.yml`) and the tag-triggered `release.yml` are
**independent**. `release.yml` builds wheels directly with `maturin-action` —
it does not run `cargo fmt --check`, `cargo clippy`, or `pytest`. So a
`ci.yml` failure does **not** block the PyPI upload, and the wheels will be
built from the tagged commit regardless.

That said, a broken `master` is embarrassing. Push the fix as a normal
follow-up commit (no re-tag, no re-release):

```bash
# fix the thing
git add -p
git commit -m "ci: <what you fixed>"
git push origin master
```

Preflight (step 2) is designed to stop this from happening. If it happened
anyway, add the missing check to step 2 so the next release catches it.

### `check-version` failed because I forgot to bump `pyproject.toml`

```bash
# Delete the bad tag.
git tag -d "v$NEW"
git push origin ":refs/tags/v$NEW"

# Fix pyproject.toml, amend or add a new commit, retag.
# (Amend only if the release commit has NOT been pushed yet; otherwise new commit.)
git tag -a "v$NEW" -m "Release $NEW"
git push origin master
git push origin "v$NEW"
```

### `cargo publish` fails with "crate version already exists"

You already published this version. Bump to the next patch (`$NEW → $NEW+1`)
and restart from step 1. crates.io versions are immutable; there is no way to
overwrite.

### PyPI publish job keeps timing out waiting for approval

The `pypi` GitHub environment has manual approval enabled on purpose (prevents
accidental uploads). Somebody with admin on the repo needs to click Approve.
If the run has been pending for more than a day, cancel it and retrigger by
retagging (see above) — PyPI trusted publishing requires the workflow run id
to be fresh.

### The test using `scipy` is failing in CI

CI only installs `numpy`. Tests that need `scipy` must use
`pytest.importorskip("scipy.stats")` at the top of the test body so they are
skipped (not failed) when scipy is not installed. See
`python/tests/test_conditional_sampling.py::test_sample_conditional_narrow_band_preserves_conditional_distribution`
for the pattern.

### I want to run everything locally and only see failures

```bash
set -e
cargo fmt --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace
cargo test --workspace --doc
cargo bench --no-run
maturin develop --release
pytest
cargo publish -p rscopulas --dry-run
echo "PREFLIGHT OK"
```

If that script prints `PREFLIGHT OK`, you are cleared to tag.
