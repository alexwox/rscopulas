//! Tail-dependence coefficients and estimators against outside references.
//!
//! - Closed forms: `fixtures/reference/mpmath/tail_dependence.json`, produced
//!   by `scripts/reference/generate_tail_dependence_truth.py`. That script
//!   evaluates the defining limits of each family's textbook distribution
//!   function at 1300 digits and never uses a closed-form coefficient.
//! - Student t: the same fixture (a numerical integral of the diagonal), and
//!   R `copula::lambda(tCopula(rho, df = nu))`.
//! - Estimators: a literal transcription of their definitions, and samples
//!   drawn from copulas whose coefficients are known.

use std::{fs, path::PathBuf};

use ndarray::array;
use rand::{SeedableRng, rngs::StdRng};
use serde::Deserialize;

use rscopulas::{
    ClaytonCopula, CopulaModel, FrankCopula, GaussianCopula, GumbelHougaardCopula, InputError,
    PairCopulaFamily, PairCopulaParams, PairCopulaSpec, PseudoObs, Rotation, SampleOptions,
    StudentTCopula, TailDependence,
    tail::{
        default_tail_k, empirical_tail_concentration, empirical_tail_dependence,
        empirical_tail_dependence_curve, empirical_tail_dependence_matrix,
    },
};

#[derive(Debug, Deserialize)]
struct Fixture {
    cases: Vec<Case>,
}

#[derive(Debug, Deserialize)]
struct Case {
    family: String,
    parameters: Vec<f64>,
    lower: f64,
    upper: f64,
}

fn family(name: &str) -> PairCopulaFamily {
    match name {
        "independence" => PairCopulaFamily::Independence,
        "student_t" => PairCopulaFamily::StudentT,
        "clayton" => PairCopulaFamily::Clayton,
        "frank" => PairCopulaFamily::Frank,
        "gumbel" => PairCopulaFamily::Gumbel,
        "joe" => PairCopulaFamily::Joe,
        "bb1" => PairCopulaFamily::Bb1,
        "bb6" => PairCopulaFamily::Bb6,
        "bb7" => PairCopulaFamily::Bb7,
        "bb8" => PairCopulaFamily::Bb8,
        "tawn1" => PairCopulaFamily::Tawn1,
        "tawn2" => PairCopulaFamily::Tawn2,
        other => panic!("unexpected family {other}"),
    }
}

fn spec(name: &str, parameters: &[f64], rotation: Rotation) -> PairCopulaSpec {
    let params = match parameters {
        [] => PairCopulaParams::None,
        [theta] => PairCopulaParams::One(*theta),
        [first, second] => PairCopulaParams::Two(*first, *second),
        other => panic!("unexpected parameter count {}", other.len()),
    };
    PairCopulaSpec {
        family: family(name),
        rotation,
        params,
    }
}

fn coefficients(spec: &PairCopulaSpec) -> TailDependence {
    spec.tail_dependence()
        .expect("spec should be valid")
        .expect("family should have a closed form")
}

fn fixture() -> Fixture {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../fixtures/reference/mpmath/tail_dependence.json");
    serde_json::from_str(&fs::read_to_string(&path).expect("fixture should exist"))
        .expect("fixture should parse")
}

#[test]
fn closed_forms_match_high_precision_limits() {
    let fixture = fixture();
    assert_eq!(fixture.cases.len(), 43);
    for case in &fixture.cases {
        let got = coefficients(&spec(&case.family, &case.parameters, Rotation::R0));
        assert!(
            (got.lower - case.lower).abs() < 1e-10 && (got.upper - case.upper).abs() < 1e-10,
            "{} {:?}: got ({}, {}), reference ({}, {})",
            case.family,
            case.parameters,
            got.lower,
            got.upper,
            case.lower,
            case.upper,
        );
    }
}

#[test]
fn rotations_swap_or_remove_the_coefficients() {
    for case in &fixture().cases {
        if case.family == "student_t" {
            continue;
        }
        let base = coefficients(&spec(&case.family, &case.parameters, Rotation::R0));
        let survival = coefficients(&spec(&case.family, &case.parameters, Rotation::R180));
        assert_eq!((survival.lower, survival.upper), (base.upper, base.lower));
        for rotation in [Rotation::R90, Rotation::R270] {
            let rotated = coefficients(&spec(&case.family, &case.parameters, rotation));
            assert_eq!((rotated.lower, rotated.upper), (0.0, 0.0));
        }
    }
}

/// `copula::lambda(tCopula(rho, df = nu))`, R copula 1.1.
#[test]
fn student_t_matches_r_copula() {
    let reference = [
        (0.5, 3.0, 0.3125),
        (-0.4, 2.0, 0.0772742899875456),
        (0.9, 6.0, 0.56302780369854),
        (0.0, 4.0, 0.0755868184216124),
        (0.7, 1.0, 0.612701665379258),
        (0.3, 25.0, 9.13687668396369e-04),
    ];
    for (rho, nu, expected) in reference {
        let got = coefficients(&spec("student_t", &[rho, nu], Rotation::R0));
        assert!((got.lower - expected).abs() < 1e-13, "t({rho}, {nu})");
        assert_eq!(got.lower, got.upper);

        // Rotating a t copula by 90 degrees is the t copula with -rho.
        let rotated = coefficients(&spec("student_t", &[-rho, nu], Rotation::R90));
        assert!((rotated.lower - expected).abs() < 1e-13);
    }
}

#[test]
fn gaussian_pair_is_tail_independent() {
    let got = coefficients(&spec("student_t", &[0.9, 1e9], Rotation::R0));
    assert!(got.lower < 1e-12, "t with huge nu tends to the Gaussian");
    let gaussian = PairCopulaSpec {
        family: PairCopulaFamily::Gaussian,
        rotation: Rotation::R0,
        params: PairCopulaParams::One(0.99),
    };
    assert_eq!(
        coefficients(&gaussian),
        TailDependence {
            lower: 0.0,
            upper: 0.0
        }
    );
}

#[test]
fn families_without_a_closed_form_return_none() {
    let khoudraji = PairCopulaSpec::khoudraji(
        spec("clayton", &[2.0], Rotation::R0),
        spec("gumbel", &[2.0], Rotation::R0),
        0.4,
        0.6,
    )
    .expect("khoudraji spec should be valid");
    assert_eq!(khoudraji.tail_dependence().unwrap(), None);

    let invalid = spec("clayton", &[-1.0], Rotation::R0);
    assert!(invalid.tail_dependence().is_err());
}

/// The model's own distribution function must approach the closed-form
/// coefficients. Only families whose limit is reached at a polynomial rate
/// are checked; the elliptical ones converge too slowly for doubles.
#[test]
fn tail_concentration_approaches_the_coefficients() {
    let cases: [(&str, &[f64]); 8] = [
        ("clayton", &[2.5]),
        ("gumbel", &[2.5]),
        ("joe", &[3.0]),
        ("frank", &[5.0]),
        ("bb1", &[0.5, 1.6]),
        ("bb7", &[1.8, 1.2]),
        ("tawn1", &[3.0, 0.8]),
        ("tawn2", &[3.0, 0.8]),
    ];
    for (name, parameters) in cases {
        for rotation in [Rotation::R0, Rotation::R180] {
            let spec = spec(name, parameters, rotation);
            let expected = coefficients(&spec);
            let lower = spec.tail_concentration(1e-8).unwrap();
            let upper = spec.tail_concentration(1.0 - 1e-8).unwrap();
            // The slowest limit here is the zero tail of Gumbel(2.5), which
            // vanishes like q^(2^(1/theta) - 1) = q^0.32: 3e-3 at q = 1e-8.
            assert!(
                (lower - expected.lower).abs() < 2e-2,
                "{name} {rotation:?} lower: {lower} vs {}",
                expected.lower
            );
            assert!(
                (upper - expected.upper).abs() < 2e-2,
                "{name} {rotation:?} upper: {upper} vs {}",
                expected.upper
            );
        }
    }

    let clayton = spec("clayton", &[2.0], Rotation::R0);
    // C(q, q) / q for Clayton(2) at q = 1/2: (2 * 4 - 1)^(-1/2) / (1/2).
    let exact = 2.0 / 7f64.sqrt();
    assert!((clayton.tail_concentration(0.5).unwrap() - exact).abs() < 1e-12);
    for q in [0.0, 1.0, f64::NAN, -0.1] {
        assert!(clayton.tail_concentration(q).is_err());
    }
}

#[test]
fn multivariate_models_report_pairwise_coefficients() {
    let clayton = ClaytonCopula::new(3, 2.5).unwrap().tail_dependence();
    assert!((clayton.lower[(0, 2)] - 0.757858283255199).abs() < 1e-13);
    assert_eq!(clayton.upper[(0, 2)], 0.0);
    assert_eq!(clayton.lower[(1, 1)], 1.0);
    assert_eq!(clayton.upper[(1, 1)], 1.0);

    let gumbel = GumbelHougaardCopula::new(4, 2.5).unwrap().tail_dependence();
    assert!((gumbel.upper[(3, 0)] - 0.680492089227106).abs() < 1e-13);
    assert_eq!(gumbel.lower[(3, 0)], 0.0);

    let frank = FrankCopula::new(3, 4.0).unwrap().tail_dependence();
    assert_eq!(frank.lower[(0, 1)], 0.0);
    assert_eq!(frank.upper[(0, 1)], 0.0);

    let correlation = array![[1.0, 0.5, -0.4], [0.5, 1.0, 0.0], [-0.4, 0.0, 1.0]];
    let gaussian = GaussianCopula::new(correlation.clone())
        .unwrap()
        .tail_dependence();
    assert_eq!(gaussian.lower[(0, 1)], 0.0);
    assert_eq!(gaussian.upper[(2, 2)], 1.0);

    // Same three R values as above, with nu = 3 and 2 replaced by one nu.
    let student = StudentTCopula::new(correlation, 3.0)
        .unwrap()
        .tail_dependence();
    assert!((student.lower[(0, 1)] - 0.3125).abs() < 1e-13);
    assert_eq!(student.lower, student.upper);
    assert_eq!(student.lower[(1, 0)], student.lower[(0, 1)]);
    assert!(student.lower[(0, 2)] < student.lower[(1, 2)]);
}

/// The estimator written out from its definition, with no sorting tricks.
fn brute_force(first: &[f64], second: &[f64], k: usize) -> (f64, f64) {
    let n = first.len();
    let rank = |values: &[f64], idx: usize| -> f64 {
        let below = values.iter().filter(|v| **v < values[idx]).count();
        let equal = values.iter().filter(|v| **v == values[idx]).count();
        below as f64 + (equal as f64 + 1.0) / 2.0
    };
    let mut lower = 0;
    let mut upper = 0;
    for idx in 0..n {
        let (r, s) = (rank(first, idx), rank(second, idx));
        if r <= k as f64 && s <= k as f64 {
            lower += 1;
        }
        if r > (n - k) as f64 && s > (n - k) as f64 {
            upper += 1;
        }
    }
    (lower as f64 / k as f64, upper as f64 / k as f64)
}

fn columns(sample: &ndarray::Array2<f64>) -> (Vec<f64>, Vec<f64>) {
    (sample.column(0).to_vec(), sample.column(1).to_vec())
}

#[test]
fn estimators_match_their_definitions() {
    let mut rng = StdRng::seed_from_u64(11);
    let sample = ClaytonCopula::new(2, 1.5)
        .unwrap()
        .sample(400, &mut rng, &SampleOptions::default())
        .unwrap();
    let (mut first, second) = columns(&sample);
    // Ties must not break the counts.
    first[3] = first[7];
    first[100] = first[7];

    let ks = [1, 2, 7, 20, 133, 399, 400];
    let curve = empirical_tail_dependence_curve(&first, &second, &ks).unwrap();
    for (k, got) in ks.iter().zip(&curve) {
        let (lower, upper) = brute_force(&first, &second, *k);
        assert_eq!((got.lower, got.upper), (lower, upper), "k = {k}");
        assert_eq!(
            empirical_tail_dependence(&first, &second, *k).unwrap(),
            *got
        );
    }
    // With k = n every observation is in both "tails".
    assert_eq!((curve[6].lower, curve[6].upper), (1.0, 1.0));

    // Concentration: literal counts on u = rank / (n + 1).
    let n = first.len() as f64;
    let rank = |values: &[f64], idx: usize| -> f64 {
        let below = values.iter().filter(|v| **v < values[idx]).count();
        let equal = values.iter().filter(|v| **v == values[idx]).count();
        (below as f64 + (equal as f64 + 1.0) / 2.0) / (n + 1.0)
    };
    let levels = [0.01, 0.1, 0.37, 0.5, 0.63, 0.9, 0.99];
    let got = empirical_tail_concentration(&first, &second, &levels).unwrap();
    for (q, value) in levels.iter().zip(&got) {
        let count = (0..first.len())
            .filter(|idx| {
                let (u, v) = (rank(&first, *idx), rank(&second, *idx));
                if *q <= 0.5 {
                    u <= *q && v <= *q
                } else {
                    u > *q && v > *q
                }
            })
            .count() as f64;
        let expected = if *q <= 0.5 {
            count / (n * q)
        } else {
            count / (n * (1.0 - q))
        };
        assert!((value - expected.min(1.0)).abs() < 1e-15, "q = {q}");
    }
}

#[test]
fn estimators_only_depend_on_ranks() {
    let mut rng = StdRng::seed_from_u64(5);
    let sample = GumbelHougaardCopula::new(2, 2.0)
        .unwrap()
        .sample(500, &mut rng, &SampleOptions::default())
        .unwrap();
    let (first, second) = columns(&sample);
    // A strictly increasing transform of each margin: raw-scale data.
    let raw_first: Vec<f64> = first.iter().map(|u| (-(1.0 - u).ln()).powi(3)).collect();
    let raw_second: Vec<f64> = second.iter().map(|u| 100.0 * u.ln()).collect();
    assert_eq!(
        empirical_tail_dependence(&first, &second, 22).unwrap(),
        empirical_tail_dependence(&raw_first, &raw_second, 22).unwrap()
    );
}

/// On large samples the estimates must land on the known coefficients, and
/// the empirical concentration function on the model's.
#[test]
fn estimators_recover_known_coefficients() {
    let n = 200_000;
    let k = default_tail_k(n); // 447
    let options = SampleOptions::default();

    let mut rng = StdRng::seed_from_u64(2026);
    let clayton = ClaytonCopula::new(2, 2.5).unwrap();
    let (first, second) = columns(&clayton.sample(n, &mut rng, &options).unwrap());
    let estimate = empirical_tail_dependence(&first, &second, k).unwrap();
    // Standard error is about sqrt(lambda (1 - lambda) / k) = 0.02.
    assert!(
        (estimate.lower - 0.757858283255199).abs() < 0.06,
        "{estimate:?}"
    );
    assert!(estimate.upper < 0.06, "{estimate:?}");

    let model = spec("clayton", &[2.5], Rotation::R0);
    let levels = [0.02, 0.1, 0.3, 0.5, 0.7, 0.9, 0.98];
    let observed = empirical_tail_concentration(&first, &second, &levels).unwrap();
    for (q, value) in levels.iter().zip(&observed) {
        let implied = model.tail_concentration(*q).unwrap();
        assert!(
            (value - implied).abs() < 0.02,
            "q = {q}: {value} vs {implied}"
        );
    }

    let gumbel = GumbelHougaardCopula::new(2, 2.5).unwrap();
    let (first, second) = columns(&gumbel.sample(n, &mut rng, &options).unwrap());
    let estimate = empirical_tail_dependence(&first, &second, k).unwrap();
    assert!(
        (estimate.upper - 0.680492089227106).abs() < 0.07,
        "{estimate:?}"
    );
    // The Gumbel lower coefficient is 0, but the conditional probability
    // only falls like q^0.32. At the level the estimator looks at, k / n,
    // it is still 0.14, and that is what the estimate must reproduce.
    let level = k as f64 / n as f64;
    let implied = spec("gumbel", &[2.5], Rotation::R0)
        .tail_concentration(level)
        .unwrap();
    assert!((implied - level.powf(2f64.powf(0.4) - 1.0)).abs() < 1e-9);
    assert!((estimate.lower - implied).abs() < 0.06, "{estimate:?}");

    let correlation = array![[1.0, 0.5], [0.5, 1.0]];
    let student = StudentTCopula::new(correlation, 3.0).unwrap();
    let (first, second) = columns(&student.sample(n, &mut rng, &options).unwrap());
    let estimate = empirical_tail_dependence(&first, &second, k).unwrap();
    assert!((estimate.lower - 0.3125).abs() < 0.08, "{estimate:?}");
    assert!((estimate.upper - 0.3125).abs() < 0.08, "{estimate:?}");
}

#[test]
fn matrix_estimator_is_symmetric_with_unit_diagonal() {
    let mut rng = StdRng::seed_from_u64(3);
    let sample = ClaytonCopula::new(3, 2.0)
        .unwrap()
        .sample(2_000, &mut rng, &SampleOptions::default())
        .unwrap();
    let data = PseudoObs::new(sample.clone()).unwrap();
    let k = default_tail_k(data.n_obs());
    let matrix = empirical_tail_dependence_matrix(data.as_view(), k).unwrap();
    for left in 0..3 {
        assert_eq!(matrix.lower[(left, left)], 1.0);
        assert_eq!(matrix.upper[(left, left)], 1.0);
        for right in 0..3 {
            assert_eq!(matrix.lower[(left, right)], matrix.lower[(right, left)]);
        }
    }
    let pair = empirical_tail_dependence(&sample.column(0).to_vec(), &sample.column(2).to_vec(), k)
        .unwrap();
    assert_eq!(matrix.lower[(0, 2)], pair.lower);
    assert_eq!(matrix.upper[(0, 2)], pair.upper);

    // Every entry equals the pairwise estimator, at several k and with ties.
    let mut tied = sample.clone();
    for row in 0..40 {
        tied[(row, 1)] = tied[(0, 1)];
        tied[(row + 100, 0)] = tied[(100, 0)];
    }
    for k in [1, 7, 44, 500, 2_000] {
        let matrix = empirical_tail_dependence_matrix(tied.view(), k).unwrap();
        for left in 0..3 {
            for right in 0..3 {
                if left == right {
                    continue;
                }
                let pair = empirical_tail_dependence(
                    &tied.column(left).to_vec(),
                    &tied.column(right).to_vec(),
                    k,
                )
                .unwrap();
                assert_eq!(matrix.lower[(left, right)], pair.lower, "k = {k}");
                assert_eq!(matrix.upper[(left, right)], pair.upper, "k = {k}");
            }
        }
    }
    assert!(empirical_tail_dependence_matrix(sample.view(), 0).is_err());
    assert!(empirical_tail_dependence_matrix(sample.view(), 2_001).is_err());
}

#[test]
fn estimators_reject_bad_input() {
    let first = [0.1, 0.5, 0.9];
    let second = [0.2, 0.4, 0.8];
    let invalid = |result: Result<TailDependence, rscopulas::CopulaError>| {
        matches!(
            result,
            Err(rscopulas::CopulaError::InvalidInput(
                InputError::InvalidArgument { .. } | InputError::NonFiniteValue
            ))
        )
    };
    assert!(invalid(empirical_tail_dependence(&first, &second, 0)));
    assert!(invalid(empirical_tail_dependence(&first, &second, 4)));
    assert!(invalid(empirical_tail_dependence(&first, &second[..2], 1)));
    assert!(invalid(empirical_tail_dependence(
        &first[..1],
        &second[..1],
        1
    )));
    assert!(invalid(empirical_tail_dependence(
        &[0.1, f64::NAN, 0.9],
        &second,
        1
    )));
    assert!(empirical_tail_concentration(&first, &second, &[0.0]).is_err());
    assert!(empirical_tail_concentration(&first, &second, &[1.0]).is_err());
}
