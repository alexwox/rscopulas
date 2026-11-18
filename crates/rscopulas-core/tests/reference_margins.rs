//! Marginal models against outside references.
//!
//! `fixtures/reference/evt/fits.json` is written by
//! `scripts/reference/generate_evt_fixtures.py`:
//!
//! - `distributions`: `scipy.stats.genpareto` / `genextreme` values;
//! - `gpd`, `gev`: simulated samples with two reference fits each. `scipy` is
//!   the maximum of SciPy's log-likelihood found by a tightly converged
//!   Nelder–Mead search; `r` is `evd::fpot` / `evd::fgev`, with
//!   `extRemes` delta-method return-level errors.
//!
//! The semi-parametric margin is checked against distributions whose
//! distribution function is known in closed form.

use std::{fs, path::PathBuf};

use ndarray::Array2;
use rand::{Rng, SeedableRng, rngs::StdRng};
use serde::Deserialize;

use rscopulas::{
    CopulaModel, FitOptions, GeneralizedExtremeValue, GeneralizedPareto, GumbelHougaardCopula,
    MarginFitOptions, Margins, SampleOptions, SemiParametricMargin, Threshold,
    margins::{block_maxima, mean_excess, threshold_stability},
};

#[derive(Deserialize)]
struct Fixture {
    gpd: Vec<GpdCase>,
    gev: Vec<GevCase>,
    distributions: Distributions,
}

#[derive(Deserialize)]
struct GpdCase {
    name: String,
    excesses: Vec<f64>,
    scipy: GpdReference,
    r: GpdReference,
}

#[derive(Deserialize)]
struct GpdReference {
    scale: f64,
    shape: f64,
    loglik: f64,
    std_errors: [f64; 2],
}

#[derive(Deserialize)]
struct GevCase {
    name: String,
    maxima: Vec<f64>,
    scipy: GevReference,
    r: GevReference,
}

#[derive(Deserialize)]
struct GevReference {
    location: f64,
    scale: f64,
    shape: f64,
    loglik: f64,
    std_errors: [f64; 3],
    return_levels: std::collections::BTreeMap<String, ReturnLevel>,
}

#[derive(Deserialize)]
struct ReturnLevel {
    level: f64,
    std_error: f64,
}

#[derive(Deserialize)]
struct Distributions {
    gpd: Vec<GpdValues>,
    gev: Vec<GevValues>,
}

#[derive(Deserialize)]
struct GpdValues {
    scale: f64,
    shape: f64,
    y: Vec<f64>,
    cdf: Vec<f64>,
    survival: Vec<f64>,
    ln_pdf: Vec<Option<f64>>,
    p: Vec<f64>,
    quantile: Vec<f64>,
}

#[derive(Deserialize)]
struct GevValues {
    location: f64,
    scale: f64,
    shape: f64,
    x: Vec<f64>,
    cdf: Vec<f64>,
    ln_pdf: Vec<Option<f64>>,
    p: Vec<f64>,
    quantile: Vec<f64>,
}

fn fixture() -> Fixture {
    let path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/reference/evt/fits.json");
    serde_json::from_str(&fs::read_to_string(&path).expect("fixture should exist"))
        .expect("fixture should parse")
}

fn close(got: f64, expected: f64, rel: f64) -> bool {
    (got - expected).abs() <= rel * expected.abs().max(1e-300)
}

#[test]
fn generalized_pareto_matches_scipy() {
    for case in &fixture().distributions.gpd {
        let model = GeneralizedPareto::new(case.scale, case.shape).unwrap();
        let label = format!("gpd({}, {})", case.scale, case.shape);
        for (idx, y) in case.y.iter().enumerate() {
            assert!(
                (model.cdf(*y) - case.cdf[idx]).abs() < 1e-14 * case.cdf[idx].max(1e-3),
                "{label} cdf at {y}: {} vs {}",
                model.cdf(*y),
                case.cdf[idx]
            );
            assert!(
                (model.survival(*y) - case.survival[idx]).abs() <= 1e-12 * case.survival[idx],
                "{label} survival at {y}: {} vs {}",
                model.survival(*y),
                case.survival[idx]
            );
            match case.ln_pdf[idx] {
                Some(expected) => assert!(
                    (model.ln_pdf(*y) - expected).abs() < 1e-11 * expected.abs().max(1.0),
                    "{label} ln_pdf at {y}"
                ),
                None => assert_eq!(model.ln_pdf(*y), f64::NEG_INFINITY),
            }
        }
        for (p, expected) in case.p.iter().zip(&case.quantile) {
            let got = model.quantile(*p).unwrap();
            // 1 - 1e-9 is only known to 1e-7 relative in a double.
            assert!(close(got, *expected, 2e-7), "{label} quantile at {p}");
            // Round trip, away from the rounding of 1 - p.
            if *p < 0.9999 {
                assert!((model.cdf(got) - p).abs() < 1e-13);
            }
        }
        assert!(model.quantile(1.0).is_err());
        assert_eq!(model.quantile(0.0).unwrap(), 0.0);
    }
    assert!(GeneralizedPareto::new(0.0, 0.1).is_err());
    assert!(GeneralizedPareto::new(1.0, f64::NAN).is_err());
}

#[test]
fn generalized_extreme_value_matches_scipy() {
    for case in &fixture().distributions.gev {
        let model = GeneralizedExtremeValue::new(case.location, case.scale, case.shape).unwrap();
        let label = format!("gev({}, {}, {})", case.location, case.scale, case.shape);
        for (idx, x) in case.x.iter().enumerate() {
            assert!(
                (model.cdf(*x) - case.cdf[idx]).abs() < 1e-13,
                "{label} cdf at {x}: {} vs {}",
                model.cdf(*x),
                case.cdf[idx]
            );
            match case.ln_pdf[idx] {
                Some(expected) => assert!(
                    (model.ln_pdf(*x) - expected).abs() < 1e-11 * expected.abs().max(1.0),
                    "{label} ln_pdf at {x}: {} vs {expected}",
                    model.ln_pdf(*x)
                ),
                None => assert_eq!(model.ln_pdf(*x), f64::NEG_INFINITY),
            }
        }
        for (p, expected) in case.p.iter().zip(&case.quantile) {
            let got = model.quantile(*p).unwrap();
            assert!(
                (got - expected).abs() < 2e-7 * expected.abs().max(case.scale),
                "{label} quantile at {p}: {got} vs {expected}"
            );
        }
        assert!(model.quantile(0.0).is_err());
        assert!(model.quantile(1.0).is_err());
        // The T-block return level is the 1 - 1/T quantile.
        let level = model.return_level(50.0).unwrap();
        assert!(close(level, model.quantile(0.98).unwrap(), 1e-12));
        assert!(model.return_level(1.0).is_err());
    }
}

#[test]
fn generalized_pareto_fit_reaches_the_reference_maximum() {
    for case in &fixture().gpd {
        let fit = GeneralizedPareto::fit(&case.excesses).unwrap();
        let (scale, shape) = (fit.model.scale(), fit.model.shape());
        assert_eq!(fit.n_obs, case.excesses.len());
        // The log-likelihood must be at least the best either reference found.
        let best = case.scipy.loglik.max(case.r.loglik);
        assert!(
            fit.loglik >= best - 1e-9,
            "{}: {} < {best}",
            case.name,
            fit.loglik
        );
        assert!(fit.loglik <= best + 1e-6, "{}", case.name);
        // The reported log-likelihood is the likelihood at the reported model.
        let direct: f64 = case.excesses.iter().map(|y| fit.model.ln_pdf(*y)).sum();
        assert!((direct - fit.loglik).abs() < 1e-9);

        assert!(close(scale, case.scipy.scale, 5e-6), "{} scale", case.name);
        assert!(
            (shape - case.scipy.shape).abs() < 5e-6,
            "{} shape: {shape} vs {}",
            case.name,
            case.scipy.shape
        );
        // R's optimiser stops earlier (its log-likelihood is the lower one).
        assert!(close(scale, case.r.scale, 1e-4), "{} scale vs R", case.name);
        assert!(
            (shape - case.r.shape).abs() < 1e-4,
            "{} shape vs R",
            case.name
        );

        let cov = fit
            .covariance
            .as_ref()
            .expect("interior fits have a covariance");
        for (idx, reference) in [case.scipy.std_errors, case.r.std_errors]
            .iter()
            .enumerate()
        {
            let tol = if idx == 0 { 1e-4 } else { 2e-3 };
            assert!(
                close(cov[(0, 0)].sqrt(), reference[0], tol),
                "{} se scale",
                case.name
            );
            assert!(
                close(cov[(1, 1)].sqrt(), reference[1], tol),
                "{} se shape",
                case.name
            );
        }
    }
}

#[test]
fn generalized_extreme_value_fit_reaches_the_reference_maximum() {
    for case in &fixture().gev {
        let fit = GeneralizedExtremeValue::fit(&case.maxima).unwrap();
        let model = fit.model;
        let best = case.scipy.loglik.max(case.r.loglik);
        assert!(
            fit.loglik >= best - 1e-8,
            "{}: {} < {best}",
            case.name,
            fit.loglik
        );
        assert!(fit.loglik <= best + 1e-6, "{}", case.name);

        let tol = 2e-5 * case.scipy.scale;
        assert!(
            (model.location() - case.scipy.location).abs() < tol,
            "{} location",
            case.name
        );
        assert!(
            (model.scale() - case.scipy.scale).abs() < tol,
            "{} scale",
            case.name
        );
        assert!(
            (model.shape() - case.scipy.shape).abs() < 2e-5,
            "{} shape",
            case.name
        );
        assert!((model.location() - case.r.location).abs() < 10.0 * tol);
        assert!((model.shape() - case.r.shape).abs() < 1e-4);

        let cov = fit
            .covariance
            .as_ref()
            .expect("interior fits have a covariance");
        for (idx, reference) in [case.scipy.std_errors, case.r.std_errors]
            .iter()
            .enumerate()
        {
            let tol = if idx == 0 { 2e-4 } else { 1e-3 };
            for param in 0..3 {
                assert!(
                    close(cov[(param, param)].sqrt(), reference[param], tol),
                    "{} se {param}: {} vs {}",
                    case.name,
                    cov[(param, param)].sqrt(),
                    reference[param]
                );
            }
        }

        for (period, reference) in &case.scipy.return_levels {
            let period: f64 = period.parse().unwrap();
            let level = model.return_level(period).unwrap();
            let std_error = fit.return_level_std_error(period).unwrap().unwrap();
            assert!(
                (level - reference.level).abs() < 1e-4 * model.scale().max(reference.level.abs()),
                "{} level {period}: {level} vs {}",
                case.name,
                reference.level
            );
            assert!(
                close(std_error, reference.std_error, 1e-3),
                "{} level se {period}",
                case.name
            );
            // extRemes: independent optimiser and analytic gradient.
            let r = &case.r.return_levels[&format!("{period}")];
            assert!(
                (level - r.level).abs() < 1e-3 * model.scale().max(r.level.abs()),
                "{} level {period} vs R",
                case.name
            );
            assert!(
                close(std_error, r.std_error, 5e-3),
                "{} level se {period} vs R",
                case.name
            );
        }
    }
}

/// With a short, bounded sample the likelihood can be largest on the
/// boundary `shape = -1`, the uniform distribution on `[0, max]`. The fit
/// must never return less than that, and never less than any interior point.
#[test]
fn generalized_pareto_fit_never_loses_to_the_boundary_or_a_grid() {
    let mut rng = StdRng::seed_from_u64(31);
    let mut on_boundary = 0;
    for case in 0..400 {
        let n = [8, 20, 50][case % 3];
        let shape = -0.9 + 1.6 * rng.random::<f64>(); // -0.9 .. 0.7
        let truth = GeneralizedPareto::new(1.0 + 9.0 * rng.random::<f64>(), shape).unwrap();
        let sample: Vec<f64> = (0..n)
            .map(|_| truth.quantile(rng.random::<f64>()).unwrap())
            .collect();
        let fit = GeneralizedPareto::fit(&sample).unwrap();
        let largest = sample.iter().copied().fold(0.0, f64::max);
        let boundary = -(n as f64) * largest.ln();
        assert!(
            fit.loglik >= boundary - 1e-9,
            "case {case}: below the boundary"
        );
        if fit.model.shape() == -1.0 {
            on_boundary += 1;
            assert_eq!(fit.model.scale(), largest);
            assert!(fit.covariance.is_none());
        }

        // Brute force over (scale, shape) with shape >= -1.
        let mut best = f64::NEG_INFINITY;
        for i in 0..=120 {
            let candidate_shape = -1.0 + 3.0 * i as f64 / 120.0;
            for j in 0..=120 {
                let scale = largest * 10f64.powf(-2.0 + 3.0 * j as f64 / 120.0);
                let candidate = GeneralizedPareto::new(scale, candidate_shape).unwrap();
                let value: f64 = sample.iter().map(|y| candidate.ln_pdf(*y)).sum();
                best = best.max(value);
            }
        }
        assert!(
            fit.loglik >= best - 1e-9,
            "case {case} (n = {n}, shape {shape:.2}): {} < grid {best}",
            fit.loglik
        );
    }
    // The boundary case really occurs in this set.
    assert!(on_boundary > 10, "{on_boundary} boundary fits");
}

#[test]
fn fits_reject_bad_input() {
    assert!(GeneralizedPareto::fit(&[1.0]).is_err());
    assert!(GeneralizedPareto::fit(&[1.0, -0.5, 2.0]).is_err());
    assert!(GeneralizedPareto::fit(&[0.0, 0.0, 0.0]).is_err());
    assert!(GeneralizedPareto::fit(&[1.0, f64::NAN]).is_err());
    assert!(GeneralizedExtremeValue::fit(&[1.0, 2.0]).is_err());
    assert!(GeneralizedExtremeValue::fit(&[3.0, 3.0, 3.0, 3.0]).is_err());
    assert_eq!(
        block_maxima(&[1.0, 5.0, 2.0, 4.0, 9.0], 2).unwrap(),
        vec![5.0, 4.0]
    );
    assert!(block_maxima(&[1.0], 0).is_err());
}

/// `sign * Lomax(alpha)`: both tails are exactly generalized Pareto with
/// shape `1 / alpha`, and the distribution function is closed form.
fn double_lomax_cdf(x: f64, alpha: f64) -> f64 {
    let tail = 0.5 * (1.0 + x.abs()).powf(-alpha);
    if x < 0.0 { tail } else { 1.0 - tail }
}

fn double_lomax_sample(n: usize, alpha: f64, seed: u64) -> Vec<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n)
        .map(|_| {
            let magnitude = (1.0 - rng.random::<f64>()).powf(-1.0 / alpha) - 1.0;
            if rng.random::<bool>() {
                magnitude
            } else {
                -magnitude
            }
        })
        .collect()
}

#[test]
fn margin_body_is_the_interpolated_empirical_distribution() {
    let values = [3.0, 1.0, 4.0, 1.5, 9.0, 2.5, 6.0, 5.0, 3.5];
    let empirical = MarginFitOptions {
        lower_tail: None,
        upper_tail: None,
        ..MarginFitOptions::default()
    };
    let margin = SemiParametricMargin::fit(&values, &empirical).unwrap();
    assert_eq!(margin.n_obs(), 9);
    assert!(margin.lower_tail().is_none() && margin.upper_tail().is_none());

    // At an observation: rank / (n + 1).
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    for (rank, value) in sorted.iter().enumerate() {
        assert!((margin.cdf(*value) - (rank as f64 + 1.0) / 10.0).abs() < 1e-15);
        assert!((margin.quantile((rank as f64 + 1.0) / 10.0).unwrap() - value).abs() < 1e-13);
    }
    // Halfway between 4 and 5 (ranks 6 and 7).
    assert!((margin.cdf(4.5) - 0.65).abs() < 1e-15);
    assert!((margin.quantile(0.65).unwrap() - 4.5).abs() < 1e-13);
    // An empirical margin cannot extrapolate.
    assert_eq!(margin.cdf(100.0), 0.9);
    assert_eq!(margin.cdf(-100.0), 0.1);
    assert_eq!(margin.quantile(0.999).unwrap(), 9.0);
    assert_eq!(margin.quantile(0.001).unwrap(), 1.0);
    assert_eq!(margin.return_level(1000.0).unwrap(), 9.0);

    // Ties: the distribution function jumps to the count of values <= x.
    let tied = SemiParametricMargin::fit(&[1.0, 2.0, 2.0, 2.0, 3.0], &empirical).unwrap();
    assert!((tied.cdf(2.0) - 4.0 / 6.0).abs() < 1e-15);
    assert!((tied.cdf(1.5) - 2.5 / 6.0).abs() < 1e-15);
}

#[test]
fn margin_is_continuous_increasing_and_invertible() {
    let sample = double_lomax_sample(5_000, 3.0, 1);
    let margin = SemiParametricMargin::fit(&sample, &MarginFitOptions::default()).unwrap();
    let (lower, upper) = (margin.lower_tail().unwrap(), margin.upper_tail().unwrap());
    assert_eq!(lower.n_exceedances, 500);
    assert_eq!(upper.n_exceedances, 500);
    // Mass beyond a threshold: (exceedances + 1) / (n + 1) at an observation.
    assert!((upper.probability - 501.0 / 5001.0).abs() < 1e-15);
    assert!((lower.probability - 501.0 / 5001.0).abs() < 1e-15);

    // Continuous at both joins.
    for (threshold, expected) in [
        (lower.threshold, lower.probability),
        (upper.threshold, 1.0 - upper.probability),
    ] {
        assert!((margin.cdf(threshold) - expected).abs() < 1e-15);
        assert!((margin.cdf(threshold - 1e-9) - expected).abs() < 1e-8);
        assert!((margin.cdf(threshold + 1e-9) - expected).abs() < 1e-8);
    }

    // Strictly increasing over five orders of magnitude beyond the sample.
    let mut previous = 0.0;
    let mut x = -1.0e4;
    while x < 1.0e4 {
        let value = margin.cdf(x);
        assert!(value > previous && value < 1.0, "not increasing at {x}");
        previous = value;
        x += if x.abs() < 5.0 { 0.01 } else { x.abs() * 0.05 };
    }

    // Exact inverse, including far outside the data.
    for p in [
        1e-12,
        1e-6,
        0.003,
        0.0999,
        0.1002,
        0.37,
        0.5,
        0.9,
        0.9001,
        0.9993,
        1.0 - 1e-9,
    ] {
        let x = margin.quantile(p).unwrap();
        assert!(
            (margin.cdf(x) - p).abs() < 1e-12 * p.max(1e-3),
            "p = {p}: cdf(quantile) = {}",
            margin.cdf(x)
        );
    }
    for x in sample.iter().take(200) {
        let back = margin.quantile(margin.cdf(*x)).unwrap();
        assert!((back - x).abs() < 1e-9 * x.abs().max(1.0));
    }
    assert!(margin.quantile(0.0).is_err());
    assert!(margin.quantile(1.0).is_err());
}

#[test]
fn margin_recovers_a_known_distribution_and_extrapolates() {
    let alpha = 3.0;
    let sample = double_lomax_sample(20_000, alpha, 7);
    let margin = SemiParametricMargin::fit(&sample, &MarginFitOptions::default()).unwrap();

    // Both tails have true shape 1/3.
    for tail in [margin.lower_tail().unwrap(), margin.upper_tail().unwrap()] {
        assert!(
            (tail.gpd.shape() - 1.0 / alpha).abs() < 0.07,
            "{:?}",
            tail.gpd
        );
    }

    // Body: within sampling error of the truth (sqrt(0.25 / n) = 0.0035).
    for x in [-1.0, -0.4, 0.0, 0.3, 1.2] {
        assert!(
            (margin.cdf(x) - double_lomax_cdf(x, alpha)).abs() < 0.012,
            "x = {x}"
        );
    }

    // Beyond the largest observation an empirical distribution has nothing
    // to say. The fitted tail still gives the survival probability.
    let largest = sample.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let x = (2.0 * 2.0e-5f64).powf(-1.0 / alpha) - 1.0; // true survival 2e-5
    let got = 1.0 - margin.cdf(x);
    assert!((got / 2.0e-5).ln().abs() < 0.7, "survival at {x}: {got}");

    // Fifty times further out (survival 1e-6, with 2000 tail points) one
    // sample is not enough to judge: the log of the estimate has a standard
    // deviation of 0.7 there. Over 40 samples the error must average out,
    // which rules out a bias in the tail fit or in how the pieces are joined.
    let x = (2.0e-6f64).powf(-1.0 / alpha) - 1.0;
    let mut log_ratios = Vec::new();
    for seed in 0..40 {
        let sample = double_lomax_sample(20_000, alpha, 500 + seed);
        let margin = SemiParametricMargin::fit(&sample, &MarginFitOptions::default()).unwrap();
        log_ratios.push(((1.0 - margin.cdf(x)) / 1.0e-6).ln());
    }
    let mean = log_ratios.iter().sum::<f64>() / 40.0;
    let spread = (log_ratios.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / 39.0).sqrt();
    assert!(mean.abs() < 0.4, "mean log error {mean}, spread {spread}");
    assert!(spread > 0.3 && spread < 1.2, "spread {spread}");

    // Return level: the level exceeded once in 10^6 observations.
    let level = margin.return_level(1.0e6).unwrap();
    let truth = (2.0e-6f64).powf(-1.0 / alpha) - 1.0; // survival 1e-6
    assert!((level / truth).ln().abs() < 0.35, "{level} vs {truth}");
    assert!(level > largest);
    assert!(close(level, margin.quantile(1.0 - 1.0e-6).unwrap(), 1e-9));
    assert!(margin.return_level(0.5).is_err());
}

#[test]
fn thresholds_can_be_values_and_tails_can_be_disabled() {
    let sample = double_lomax_sample(2_000, 3.0, 3);
    let options = MarginFitOptions {
        lower_tail: None,
        upper_tail: Some(Threshold::Value(1.0)),
        ..MarginFitOptions::default()
    };
    let margin = SemiParametricMargin::fit(&sample, &options).unwrap();
    let upper = margin.upper_tail().unwrap();
    // Moved down to the nearest observation.
    let nearest = sample
        .iter()
        .copied()
        .filter(|x| *x <= 1.0)
        .fold(f64::NEG_INFINITY, f64::max);
    assert_eq!(upper.threshold, nearest);
    assert_eq!(
        upper.n_exceedances,
        sample.iter().filter(|x| **x > 1.0).count()
    );
    assert!(margin.lower_tail().is_none());

    let invalid = |options: MarginFitOptions| SemiParametricMargin::fit(&sample, &options).is_err();
    assert!(invalid(MarginFitOptions {
        upper_tail: Some(Threshold::Fraction(0.5)),
        ..MarginFitOptions::default()
    }));
    assert!(invalid(MarginFitOptions {
        upper_tail: Some(Threshold::Fraction(0.001)), // 2 exceedances < 10
        ..MarginFitOptions::default()
    }));
    assert!(invalid(MarginFitOptions {
        lower_tail: Some(Threshold::Value(5.0)),
        upper_tail: Some(Threshold::Value(-5.0)),
        ..MarginFitOptions::default()
    }));
    assert!(SemiParametricMargin::fit(&[1.0], &options).is_err());
    assert!(SemiParametricMargin::fit(&[2.0; 50], &options).is_err());
    assert!(SemiParametricMargin::fit(&[1.0, f64::INFINITY], &options).is_err());
}

fn gumbel_cdf(u: f64, v: f64, theta: f64) -> f64 {
    (-((-u.ln()).powf(theta) + (-v.ln()).powf(theta)).powf(1.0 / theta)).exp()
}

/// Raw data -> margins -> copula -> simulation -> raw scale, checked against
/// a joint tail probability known in closed form.
#[test]
fn full_pipeline_estimates_a_joint_tail_probability() {
    let theta = 2.0;
    let (alpha, rate) = (3.0, 0.5);
    let n = 20_000;
    let mut rng = StdRng::seed_from_u64(99);
    let truth = GumbelHougaardCopula::new(2, theta).unwrap();
    let uniforms = truth
        .sample(n, &mut rng, &SampleOptions::default())
        .unwrap();
    // Column 0: Lomax(alpha). Column 1: exponential(rate).
    let mut raw = Array2::zeros((n, 2));
    for row in 0..n {
        raw[(row, 0)] = (1.0 - uniforms[(row, 0)]).powf(-1.0 / alpha) - 1.0;
        raw[(row, 1)] = -(1.0 - uniforms[(row, 1)]).ln() / rate;
    }

    let margins = Margins::fit(raw.view(), &MarginFitOptions::default()).unwrap();
    assert_eq!(margins.dim(), 2);
    let data = margins.to_pseudo_obs(raw.view()).unwrap();
    let fit = GumbelHougaardCopula::fit(&data, &FitOptions::default()).unwrap();
    assert!(
        (fit.model.theta() - theta).abs() < 0.08,
        "theta = {}",
        fit.model.theta()
    );

    let simulated = fit
        .model
        .sample(1_000_000, &mut rng, &SampleOptions::default())
        .unwrap();
    let simulated_raw = margins.quantile(simulated.view()).unwrap();

    // Both variables above their true 99.95 % quantile at once.
    let p: f64 = 0.9995;
    let a = (1.0 - p).powf(-1.0 / alpha) - 1.0;
    let b = -(1.0 - p).ln() / rate;
    let exact = 1.0 - 2.0 * p + gumbel_cdf(p, p, theta); // 2.9e-4
    let count = |matrix: &Array2<f64>| {
        (0..matrix.nrows())
            .filter(|row| matrix[(*row, 0)] > a && matrix[(*row, 1)] > b)
            .count() as f64
            / matrix.nrows() as f64
    };
    let estimate = count(&simulated_raw);
    assert!(
        (estimate / exact).ln().abs() < 0.35,
        "joint tail: {estimate} vs {exact}"
    );
    // Under independence it would be 2.5e-7: the dependence is what matters.
    assert!(estimate > 100.0 * (1.0 - p) * (1.0 - p));

    // The same levels on the fitted data round-trip through the margins.
    let back = margins
        .quantile(margins.cdf(raw.view()).unwrap().view())
        .unwrap();
    for row in 0..200 {
        for col in 0..2 {
            assert!((back[(row, col)] - raw[(row, col)]).abs() < 1e-8 * raw[(row, col)].max(1.0));
        }
    }

    assert!(margins.cdf(raw.slice(ndarray::s![.., 0..1])).is_err());
    assert!(
        margins
            .quantile(Array2::from_elem((2, 2), 1.0).view())
            .is_err()
    );
}

#[test]
fn margins_survive_a_json_round_trip() {
    let sample = double_lomax_sample(600, 3.0, 5);
    let raw = Array2::from_shape_vec((300, 2), sample).unwrap();
    let margins = Margins::fit(raw.view(), &MarginFitOptions::default()).unwrap();
    let json = serde_json::to_string(&margins).unwrap();
    let restored: Margins = serde_json::from_str(&json).unwrap();
    assert_eq!(restored, margins);
    assert_eq!(
        restored.cdf(raw.view()).unwrap(),
        margins.cdf(raw.view()).unwrap()
    );
    let single = Margins::from_margins(vec![margins.margins()[0].clone()]).unwrap();
    assert_eq!(single.dim(), 1);
    assert!(Margins::from_margins(Vec::new()).is_err());

    // The deserializer accepts any numbers; `validate` does not.
    restored.validate().unwrap();
    let broken: Margins =
        serde_json::from_str(&json.replacen("\"probabilities\":[", "\"probabilities\":[0.9,", 1))
            .unwrap();
    assert!(broken.validate().is_err());
}

#[test]
fn threshold_diagnostics_follow_generalized_pareto_theory() {
    // Lomax(alpha): the excess over u is GPD(scale (1 + u) / alpha, shape
    // 1 / alpha), so the mean excess is (1 + u) / (alpha - 1) and the
    // modified scale is 1 / alpha at every threshold.
    let alpha = 4.0;
    let mut rng = StdRng::seed_from_u64(17);
    let sample: Vec<f64> = (0..200_000)
        .map(|_| (1.0 - rng.random::<f64>()).powf(-1.0 / alpha) - 1.0)
        .collect();
    let thresholds = [0.0, 0.2, 0.5, 1.0, 1.5];

    let excess = mean_excess(&sample, &thresholds).unwrap();
    for point in &excess {
        let expected = (1.0 + point.threshold) / (alpha - 1.0);
        assert!(
            (point.mean_excess - expected).abs() < 5.0 * point.std_error,
            "u = {}: {} vs {expected}",
            point.threshold,
            point.mean_excess
        );
        // Written out.
        let above: Vec<f64> = sample
            .iter()
            .filter(|x| **x > point.threshold)
            .map(|x| x - point.threshold)
            .collect();
        assert_eq!(point.n_exceedances, above.len());
        assert!(close(
            point.mean_excess,
            above.iter().sum::<f64>() / above.len() as f64,
            1e-12
        ));
    }

    let stability = threshold_stability(&sample, &thresholds).unwrap();
    for point in &stability {
        assert!(
            (point.shape - 1.0 / alpha).abs() < 4.0 * point.shape_std_error,
            "u = {}: shape {} +- {}",
            point.threshold,
            point.shape,
            point.shape_std_error
        );
        assert!(
            (point.modified_scale - 1.0 / alpha).abs() < 4.0 * point.modified_scale_std_error,
            "u = {}: modified scale {} +- {}",
            point.threshold,
            point.modified_scale,
            point.modified_scale_std_error
        );
        assert!(close(
            point.modified_scale,
            point.scale - point.shape * point.threshold,
            1e-12
        ));
        assert!(point.shape_std_error > 0.0 && point.shape_std_error < 0.05);
    }

    assert!(mean_excess(&sample, &[1.0e9]).is_err());
    assert!(threshold_stability(&sample, &[1.0e9]).is_err());
}
