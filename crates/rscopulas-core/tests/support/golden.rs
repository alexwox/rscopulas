//! Golden-snapshot harness.
//!
//! A snapshot is a JSON file under `fixtures/golden/v1/` holding named vectors
//! of numbers (and a few strings) produced by the library on fixed inputs.
//! The tests recompute the same quantities and compare them against the file,
//! so any change to the numerical core that moves a result beyond its
//! tolerance fails with the exact key, index, expected and actual value.
//!
//! Regenerate after an *intentional* numerical change with
//!
//! ```text
//! RSCOPULAS_BLESS=1 cargo test -p rscopulas --release --test golden_snapshots
//! ```
//!
//! and review the resulting fixture diff like any other code change. To
//! re-record only some keys (everything else is still compared and kept
//! byte-for-byte), list comma-separated substrings of their names:
//!
//! ```text
//! RSCOPULAS_BLESS_KEYS=gaussian_d7/log_pdf,gaussian_c4/log_pdf \
//!     cargo test -p rscopulas --release --test golden_snapshots vine_evaluation
//! ```
//!
//! Every re-recording is logged with its justification in
//! `fixtures/golden/v1/CHANGES.md`.
#![allow(dead_code)]

use std::{collections::BTreeMap, fmt::Write as _, path::PathBuf};

use serde::{Deserialize, Serialize};

/// How closely a recomputed vector must match the stored one.
#[derive(Debug, Clone, Copy)]
pub struct Tol {
    pub abs: f64,
    pub rel: f64,
}

impl Tol {
    /// Closed-form evaluations (densities, h-functions): a few ulps of slack
    /// for libm differences between platforms and for algebraically
    /// equivalent rewrites, nothing more.
    pub const EVAL: Tol = Tol {
        abs: 1e-11,
        rel: 1e-9,
    };
    /// Results of numerical root finding or quadrature.
    pub const SOLVE: Tol = Tol {
        abs: 1e-9,
        rel: 1e-8,
    };
    /// Fitted parameters: optimisers stop at ~1e-6 relative likelihood
    /// tolerance, which pins parameters to a few significant digits.
    pub const FIT: Tol = Tol {
        abs: 2e-3,
        rel: 2e-3,
    };
    /// Fitted log-likelihoods.
    pub const LOGLIK: Tol = Tol {
        abs: 1e-5,
        rel: 1e-6,
    };

    fn accepts(self, expected: f64, actual: f64) -> bool {
        if expected.is_nan() || actual.is_nan() {
            return expected.is_nan() && actual.is_nan();
        }
        if expected.is_infinite() || actual.is_infinite() {
            return expected == actual;
        }
        (expected - actual).abs() <= self.abs + self.rel * expected.abs()
    }
}

/// JSON cannot carry NaN or infinities, so non-finite values are stored as
/// strings.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
enum Num {
    Finite(f64),
    Special(String),
}

impl Num {
    fn from_f64(value: f64) -> Self {
        if value.is_nan() {
            Self::Special("nan".into())
        } else if value == f64::INFINITY {
            Self::Special("inf".into())
        } else if value == f64::NEG_INFINITY {
            Self::Special("-inf".into())
        } else {
            Self::Finite(value)
        }
    }

    fn to_f64(&self) -> f64 {
        match self {
            Self::Finite(value) => *value,
            Self::Special(text) => match text.as_str() {
                "inf" => f64::INFINITY,
                "-inf" => f64::NEG_INFINITY,
                _ => f64::NAN,
            },
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
enum Entry {
    Values(Vec<Num>),
    Text(String),
}

pub struct Golden {
    name: &'static str,
    path: PathBuf,
    bless: bool,
    /// Key substrings to re-record while everything else is compared.
    bless_keys: Vec<String>,
    expected: BTreeMap<String, Entry>,
    actual: BTreeMap<String, Entry>,
    failures: Vec<String>,
}

impl Golden {
    pub fn open(name: &'static str) -> Self {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../fixtures/golden/v1")
            .join(format!("{name}.json"));
        let bless = std::env::var_os("RSCOPULAS_BLESS").is_some_and(|value| value != "0");
        let bless_keys: Vec<String> = std::env::var("RSCOPULAS_BLESS_KEYS")
            .map(|list| {
                list.split(',')
                    .map(str::trim)
                    .filter(|key| !key.is_empty())
                    .map(str::to_string)
                    .collect()
            })
            .unwrap_or_default();
        let expected = if bless {
            BTreeMap::new()
        } else {
            let bytes = std::fs::read(&path).unwrap_or_else(|error| {
                panic!(
                    "golden fixture {} is missing ({error}); run with RSCOPULAS_BLESS=1 to create it",
                    path.display()
                )
            });
            serde_json::from_slice(&bytes).expect("golden fixture should be valid JSON")
        };
        Self {
            name,
            path,
            bless,
            bless_keys,
            expected,
            actual: BTreeMap::new(),
            failures: Vec::new(),
        }
    }

    /// Whether `key` is being re-recorded rather than compared.
    fn blessing(&self, key: &str) -> bool {
        self.bless
            || self
                .bless_keys
                .iter()
                .any(|pattern| key.contains(pattern.as_str()))
    }

    fn record(&mut self, key: &str, entry: Entry) {
        assert!(
            self.actual.insert(key.to_string(), entry).is_none(),
            "duplicate golden key {key}"
        );
    }

    fn expected_values(&mut self, key: &str, len: usize) -> Option<Vec<f64>> {
        match self.expected.get(key) {
            Some(Entry::Values(values)) if values.len() == len => {
                Some(values.iter().map(Num::to_f64).collect())
            }
            Some(Entry::Values(values)) => {
                self.failures.push(format!(
                    "{key}: length changed, expected {} got {len}",
                    values.len()
                ));
                None
            }
            Some(Entry::Text(text)) => {
                self.failures
                    .push(format!("{key}: expected text {text:?}, got {len} numbers"));
                None
            }
            None => {
                self.failures
                    .push(format!("{key}: not present in the stored snapshot"));
                None
            }
        }
    }

    /// Compares `values` element-wise against the stored vector.
    pub fn values(&mut self, key: &str, values: &[f64], tol: Tol) {
        self.values_or(key, values, tol, |_, _| false);
    }

    /// Like [`Golden::values`], but an element outside `tol` is still
    /// accepted when `rescue(index, actual)` returns true. Used for inverse
    /// problems, where a different root finder may legitimately land on
    /// another point of a flat region: the rescue checks the residual of the
    /// equation instead of the location of the root.
    pub fn values_or(
        &mut self,
        key: &str,
        values: &[f64],
        tol: Tol,
        rescue: impl Fn(usize, f64) -> bool,
    ) {
        self.record(
            key,
            Entry::Values(values.iter().copied().map(Num::from_f64).collect()),
        );
        if self.blessing(key) {
            return;
        }
        let Some(expected) = self.expected_values(key, values.len()) else {
            return;
        };
        let mut mismatches = 0usize;
        let mut worst: Option<(usize, f64, f64, f64)> = None;
        for (index, (&want, &got)) in expected.iter().zip(values).enumerate() {
            if tol.accepts(want, got) || rescue(index, got) {
                continue;
            }
            mismatches += 1;
            let gap = (want - got).abs();
            if worst.is_none_or(|(_, _, _, worst_gap)| gap.is_nan() || gap > worst_gap) {
                worst = Some((index, want, got, gap));
            }
        }
        if let Some((index, want, got, gap)) = worst {
            self.failures.push(format!(
                "{key}: {mismatches}/{} values differ; worst at [{index}] expected {want:e} got {got:e} (|diff| {gap:e}, tol abs {:e} rel {:e})",
                values.len(),
                tol.abs,
                tol.rel
            ));
        }
    }

    /// Like [`Golden::values`], with a per-element allowance on top of `tol`.
    ///
    /// Used for chained computations whose rounding noise is amplified by the
    /// input itself (a conditional probability within 1e-11 of 1 has only
    /// five significant digits in its complement). `slack[i]` is how far the
    /// result moves when the inputs are nudged by a few dozen ulps: a
    /// difference within that band is indistinguishable from rounding in the
    /// inputs and says nothing about the code under test.
    pub fn values_with_slack(&mut self, key: &str, values: &[f64], tol: Tol, slack: &[f64]) {
        assert_eq!(values.len(), slack.len(), "one slack per value");
        if std::env::var_os("RSCOPULAS_GOLDEN_VERBOSE").is_some() {
            let widened = slack.iter().filter(|&&s| s > 0.0).count();
            eprintln!(
                "{}/{key}: {widened}/{} values carry conditioning slack",
                self.name,
                slack.len()
            );
        }
        self.record(
            key,
            Entry::Values(values.iter().copied().map(Num::from_f64).collect()),
        );
        if self.blessing(key) {
            return;
        }
        let Some(expected) = self.expected_values(key, values.len()) else {
            return;
        };
        let mut mismatches = 0usize;
        let mut worst: Option<(usize, f64, f64, f64)> = None;
        for (index, (&want, &got)) in expected.iter().zip(values).enumerate() {
            let gap = (want - got).abs();
            if tol.accepts(want, got) || gap <= slack[index] {
                continue;
            }
            mismatches += 1;
            if worst.is_none_or(|(_, _, _, worst_gap)| gap.is_nan() || gap > worst_gap) {
                worst = Some((index, want, got, gap));
            }
        }
        if let Some((index, want, got, gap)) = worst {
            self.failures.push(format!(
                "{key}: {mismatches}/{} values differ; worst at [{index}] expected {want:e} got {got:e} (|diff| {gap:e}, tol abs {:e} rel {:e}, conditioning slack {:e})",
                values.len(),
                tol.abs,
                tol.rel,
                slack[index]
            ));
        }
    }

    /// One-sided check for an objective that optimisers maximise: the new
    /// value may be higher than the stored one (a better optimum is not a
    /// regression) but must not fall below it by more than `tol`.
    pub fn at_least(&mut self, key: &str, value: f64, tol: Tol) {
        self.record(key, Entry::Values(vec![Num::from_f64(value)]));
        if self.blessing(key) {
            return;
        }
        let Some(expected) = self.expected_values(key, 1) else {
            return;
        };
        let want = expected[0];
        let ok = if want.is_finite() && value.is_finite() {
            value >= want - (tol.abs + tol.rel * want.abs())
        } else {
            Tol::EVAL.accepts(want, value)
        };
        if !ok {
            self.failures.push(format!(
                "{key}: objective regressed, stored {want:e} now {value:e}"
            ));
        }
    }

    /// Exact string comparison (selected families, structures, error kinds).
    pub fn text(&mut self, key: &str, text: impl Into<String>) {
        let text = text.into();
        self.record(key, Entry::Text(text.clone()));
        if self.blessing(key) {
            return;
        }
        match self.expected.get(key) {
            Some(Entry::Text(want)) if *want == text => {}
            Some(Entry::Text(want)) => self
                .failures
                .push(format!("{key}: expected {want:?}, got {text:?}")),
            Some(Entry::Values(_)) => self
                .failures
                .push(format!("{key}: expected numbers, got text {text:?}")),
            None => self
                .failures
                .push(format!("{key}: not present in the stored snapshot")),
        }
    }

    /// A string that is *read back* rather than compared: when blessing it is
    /// produced by `make` and stored; otherwise the stored string is returned
    /// and `make` is not called. Used to pin a fitted model so downstream
    /// evaluations run on exactly the stored parameters.
    pub fn pinned_text(&mut self, key: &str, make: impl FnOnce() -> String) -> String {
        let text = if self.blessing(key) {
            make()
        } else {
            match self.expected.get(key) {
                Some(Entry::Text(text)) => text.clone(),
                _ => panic!("{key}: pinned text missing from the stored snapshot"),
            }
        };
        self.record(key, Entry::Text(text.clone()));
        text
    }

    /// Records a fallible vector: errors are snapshotted by their message so
    /// "this input used to work" and "this input used to be rejected" are
    /// both pinned.
    pub fn result<E: std::fmt::Display>(
        &mut self,
        key: &str,
        result: Result<Vec<f64>, E>,
        tol: Tol,
    ) {
        match result {
            Ok(values) => self.values(key, &values, tol),
            Err(error) => self.text(key, format!("error: {error}")),
        }
    }

    /// Writes the snapshot when blessing; otherwise fails the test if any
    /// comparison failed or a stored key was not recomputed.
    pub fn finish(mut self) {
        if self.bless {
            std::fs::create_dir_all(self.path.parent().expect("fixture dir")).expect("mkdir");
            let mut text = serde_json::to_string_pretty(&self.actual).expect("serialize");
            text.push('\n');
            std::fs::write(&self.path, text).expect("write golden fixture");
            eprintln!(
                "blessed {} ({} keys)",
                self.path.display(),
                self.actual.len()
            );
            return;
        }
        if !self.bless_keys.is_empty() && self.failures.is_empty() {
            // Partial re-recording: stored entries stay byte-for-byte except
            // for the selected keys.
            let mut merged = self.expected.clone();
            let mut replaced = 0usize;
            for (key, entry) in &self.actual {
                if self.blessing(key) && merged.get(key) != Some(entry) {
                    merged.insert(key.clone(), entry.clone());
                    replaced += 1;
                }
            }
            if replaced > 0 {
                let mut text = serde_json::to_string_pretty(&merged).expect("serialize");
                text.push('\n');
                std::fs::write(&self.path, text).expect("write golden fixture");
                eprintln!("re-recorded {replaced} key(s) in {}", self.path.display());
            }
        }
        for key in self.expected.keys() {
            if !self.actual.contains_key(key) {
                self.failures.push(format!(
                    "{key}: stored in the snapshot but no longer computed"
                ));
            }
        }
        if !self.failures.is_empty() {
            let mut report = format!(
                "golden snapshot `{}` has {} mismatch(es):\n",
                self.name,
                self.failures.len()
            );
            for failure in self.failures.iter().take(40) {
                let _ = writeln!(report, "  - {failure}");
            }
            if self.failures.len() > 40 {
                let _ = writeln!(report, "  … {} more", self.failures.len() - 40);
            }
            let _ = writeln!(
                report,
                "If the change is intentional, re-run with RSCOPULAS_BLESS=1 and review the fixture diff."
            );
            panic!("{report}");
        }
    }
}
