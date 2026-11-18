//! The single-family samplers against exact properties of their copulas.
//!
//! Nothing here compares with an earlier release or with another sampler:
//! every expected value is a closed form.
//!
//! - Archimedean copulas: the joint distribution function
//!   `C(u) = ψ(Σ ψ⁻¹(u_j))`, compared with the empirical one at fixed points.
//! - Elliptical copulas: the orthant probability
//!   `P(U_i ≤ 1/2, U_j ≤ 1/2) = 1/4 + asin(ρ_ij) / (2π)`, which holds for the
//!   Gaussian and the Student t copula alike.
//! - All: uniform margins (Kolmogorov–Smirnov), and the same sample whether
//!   the blocks are drawn on one thread or on many.

use ndarray::{Array2, array};
use rand::{SeedableRng, rngs::StdRng};

use rscopulas::{
    ClaytonCopula, CopulaModel, FrankCopula, GaussianCopula, GumbelHougaardCopula, SampleOptions,
    StudentTCopula,
};

const N: usize = 400_000;
/// Largest admissible |z| for a binomial proportion. With about 700
/// comparisons in this file the chance of one exceeding it is 5e-3.
const Z_MAX: f64 = 4.5;
/// Largest admissible `√n·D`; the 0.1 % point of the Kolmogorov
/// distribution is 1.95.
const KS_MAX: f64 = 2.2;

fn draw<M: CopulaModel>(model: &M, seed: u64) -> Array2<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    model
        .sample(N, &mut rng, &SampleOptions::default())
        .expect("sampling should succeed")
}

fn assert_uniform_margins(name: &str, sample: &Array2<f64>) {
    let n = sample.nrows() as f64;
    for col in 0..sample.ncols() {
        let mut column = sample.column(col).to_vec();
        column.sort_by(f64::total_cmp);
        let distance = column
            .iter()
            .enumerate()
            .map(|(idx, &value)| ((idx as f64 + 1.0) / n - value).max(value - idx as f64 / n))
            .fold(0.0_f64, f64::max);
        assert!(
            n.sqrt() * distance < KS_MAX,
            "{name}: column {col} is not uniform, sqrt(n)·D = {}",
            n.sqrt() * distance
        );
        assert!(
            column[0] > 0.0 && column[column.len() - 1] < 1.0,
            "{name}: value on the boundary"
        );
    }
}

fn assert_proportion(name: &str, observed: usize, expected: f64, n: usize) {
    let n = n as f64;
    let z = (observed as f64 / n - expected) / (expected * (1.0 - expected) / n).sqrt();
    assert!(
        z.abs() < Z_MAX,
        "{name}: observed {} expected {expected} (z = {z})",
        observed as f64 / n
    );
}

/// Points at which the joint distribution function is compared: every
/// combination of three levels in the first three coordinates, with the
/// remaining coordinates at a fixed fourth level.
fn grid(dim: usize) -> Vec<Vec<f64>> {
    let levels = [0.15, 0.5, 0.85];
    let mut points = Vec::new();
    for &a in &levels {
        for &b in &levels {
            for &c in &levels {
                let mut point = vec![0.7; dim];
                point[0] = a;
                point[1] = b;
                if dim > 2 {
                    point[2] = c;
                }
                points.push(point);
            }
        }
    }
    points
}

fn assert_joint_cdf(name: &str, sample: &Array2<f64>, cdf: impl Fn(&[f64]) -> f64) {
    for point in grid(sample.ncols()) {
        let below = sample
            .rows()
            .into_iter()
            .filter(|row| row.iter().zip(&point).all(|(value, bound)| value <= bound))
            .count();
        assert_proportion(
            &format!("{name} at {point:?}"),
            below,
            cdf(&point),
            sample.nrows(),
        );
    }
}

#[test]
fn archimedean_samplers_follow_their_distribution_functions() {
    for dim in [2usize, 3, 6] {
        for theta in [0.3, 2.5, 9.0] {
            let model = ClaytonCopula::new(dim, theta).unwrap();
            let sample = draw(&model, 11 + dim as u64);
            let name = format!("Clayton d={dim} theta={theta}");
            assert_uniform_margins(&name, &sample);
            assert_joint_cdf(&name, &sample, |u| {
                (u.iter().map(|x| x.powf(-theta) - 1.0).sum::<f64>() + 1.0).powf(-1.0 / theta)
            });
        }
        for theta in [0.5, 5.0, 25.0] {
            let model = FrankCopula::new(dim, theta).unwrap();
            let sample = draw(&model, 23 + dim as u64);
            let name = format!("Frank d={dim} theta={theta}");
            assert_uniform_margins(&name, &sample);
            assert_joint_cdf(&name, &sample, |u| {
                let total: f64 = u
                    .iter()
                    .map(|x| -((-theta * x).exp_m1() / (-theta).exp_m1()).ln())
                    .sum();
                -((-total).exp() * (-theta).exp_m1()).ln_1p() / theta
            });
        }
        for theta in [1.2, 1.8, 6.0] {
            let model = GumbelHougaardCopula::new(dim, theta).unwrap();
            let sample = draw(&model, 37 + dim as u64);
            let name = format!("Gumbel d={dim} theta={theta}");
            assert_uniform_margins(&name, &sample);
            assert_joint_cdf(&name, &sample, |u| {
                (-u.iter()
                    .map(|x| (-x.ln()).powf(theta))
                    .sum::<f64>()
                    .powf(1.0 / theta))
                .exp()
            });
        }
    }
}

fn correlations() -> Vec<Array2<f64>> {
    vec![
        array![[1.0, 0.7], [0.7, 1.0]],
        array![[1.0, -0.45], [-0.45, 1.0]],
        array![
            [1.0, 0.6, 0.3, -0.2, 0.1],
            [0.6, 1.0, 0.5, 0.0, 0.2],
            [0.3, 0.5, 1.0, 0.4, -0.3],
            [-0.2, 0.0, 0.4, 1.0, 0.25],
            [0.1, 0.2, -0.3, 0.25, 1.0]
        ],
    ]
}

fn assert_orthants(name: &str, sample: &Array2<f64>, correlation: &Array2<f64>) {
    let dim = sample.ncols();
    for i in 0..dim {
        for j in (i + 1)..dim {
            let both = sample
                .rows()
                .into_iter()
                .filter(|row| row[i] <= 0.5 && row[j] <= 0.5)
                .count();
            let expected = 0.25 + correlation[(i, j)].asin() / (2.0 * std::f64::consts::PI);
            assert_proportion(
                &format!("{name} orthant ({i}, {j})"),
                both,
                expected,
                sample.nrows(),
            );
        }
    }
}

#[test]
fn elliptical_samplers_have_exact_orthant_probabilities() {
    for (index, correlation) in correlations().into_iter().enumerate() {
        let gaussian = GaussianCopula::new(correlation.clone()).unwrap();
        let sample = draw(&gaussian, 101 + index as u64);
        let name = format!("Gaussian #{index}");
        assert_uniform_margins(&name, &sample);
        assert_orthants(&name, &sample, &correlation);
        for nu in [2.5, 5.0, 30.0] {
            let student = StudentTCopula::new(correlation.clone(), nu).unwrap();
            let sample = draw(&student, 211 + index as u64);
            let name = format!("Student t #{index} nu={nu}");
            assert_uniform_margins(&name, &sample);
            assert_orthants(&name, &sample, &correlation);
        }
    }
}

/// A Student t copula is not a Gaussian one: with few degrees of freedom the
/// two tails coincide more often. For `ρ = 0` and `ν = 2.5` the probability
/// that both coordinates are below 0.01 is 15 times the Gaussian `1e-4`:
/// `∫ Φ(q·√(w/ν))² χ²_ν(dw)` with `q = t_ν⁻¹(0.01)`, by SciPy quadrature,
/// is 1.5170656e-3.
#[test]
fn student_t_sampler_has_the_tail_of_a_t_copula() {
    let student = StudentTCopula::new(array![[1.0, 0.0], [0.0, 1.0]], 2.5).unwrap();
    let sample = draw(&student, 5);
    let both = sample
        .rows()
        .into_iter()
        .filter(|row| row[0] <= 0.01 && row[1] <= 0.01)
        .count();
    assert_proportion(
        "Student t lower corner",
        both,
        1.517_065_6e-3,
        sample.nrows(),
    );
}

#[test]
fn a_sample_does_not_depend_on_the_number_of_threads() {
    let model = ClaytonCopula::new(4, 2.0).unwrap();
    let gaussian = GaussianCopula::new(correlations().remove(2)).unwrap();
    let on_all = (draw(&model, 9), draw(&gaussian, 9));
    let single = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    let on_one = single.install(|| (draw(&model, 9), draw(&gaussian, 9)));
    assert_eq!(on_all.0, on_one.0);
    assert_eq!(on_all.1, on_one.1);

    // A shorter sample is a prefix of a longer one from the same seed up to
    // the last complete block.
    let mut rng = StdRng::seed_from_u64(9);
    let short = model
        .sample(3000, &mut rng, &SampleOptions::default())
        .unwrap();
    assert_eq!(
        short.slice(ndarray::s![..2048, ..]),
        on_all.0.slice(ndarray::s![..2048, ..])
    );
}
