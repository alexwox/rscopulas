//! Cross-model identities that pin numerical accuracy independently of any
//! stored snapshot: two different code paths that must describe the same
//! mathematical object.

#[path = "support/catalog.rs"]
mod catalog;
#[path = "support/datagen.rs"]
mod datagen;

use ndarray::Array2;
use rscopulas::{CopulaModel, EvalOptions, PseudoObs, SampleOptions, VineCopula, math};
use statrs::distribution::{Continuous, ContinuousCDF, Normal};

/// A Gaussian vine built from a correlation matrix is the Gaussian copula
/// with that matrix. The vine evaluates the density through a chain of
/// h-functions (normal CDF and quantile at every edge); the copula evaluates
/// it through one quadratic form. Agreement to 1e-13 therefore bounds the
/// error the h-function chain accumulates.
///
/// Release 0.4.0 missed this by 6e-11 on interior rows (its normal CDF went
/// through an `erfc` that is only accurate to ~1e-10 in the tails) and by
/// 2e-8 on upper-tail rows (a conditional stored as a probability near 1 has
/// lost the digits of its complement before the next tree reads it). Gaussian
/// vines are now evaluated on the normal-score scale, which has neither
/// problem.
#[test]
fn gaussian_vine_density_equals_gaussian_copula_density() {
    for dim in [3usize, 4, 7] {
        let correlation = datagen::correlation(dim);
        let copula = catalog::gaussian_copula(dim);
        // Interior points plus rows deep in both tails. Rows are kept clear of
        // the 1e-12 clip on intermediate conditionals, beyond which the vine
        // intentionally departs from the exact density.
        let mut points = datagen::uniforms(400, dim, 0x1D + dim as u64).mapv(|u| 0.02 + 0.9 * u);
        for col in 0..dim {
            points[(0, col)] = 1e-3;
            points[(1, col)] = 1.0 - 1e-4;
            points[(2, col)] = 0.9999;
        }
        let points = PseudoObs::new(points).unwrap();
        let expected = copula.log_pdf(&points, &EvalOptions::default()).unwrap();
        let order: Vec<usize> = (0..dim).collect();
        for (name, vine) in [
            (
                "d-vine",
                VineCopula::gaussian_d_vine(order.clone(), correlation.clone()).unwrap(),
            ),
            (
                "c-vine",
                VineCopula::gaussian_c_vine(order.clone(), correlation.clone()).unwrap(),
            ),
        ] {
            let got = vine.log_pdf(&points, &EvalOptions::default()).unwrap();
            let mut worst = 0.0_f64;
            for (row, (a, b)) in got.iter().zip(&expected).enumerate() {
                let deviation = (a - b).abs() / (1.0 + b.abs());
                assert!(
                    deviation < 1e-13,
                    "dim {dim} {name} row {row}: vine {a} vs copula {b}"
                );
                worst = worst.max(deviation);
            }
            println!("dim {dim} {name}: worst relative deviation {worst:e}");
        }
    }
}

/// The Rosenblatt transform of a Gaussian vine has a closed form: with the
/// variables arranged in the vine's variable order and `L` the Cholesky
/// factor of their correlation matrix, `U = Φ(L⁻¹ z)` for `z = Φ⁻¹(v)`, and
/// the inverse transform is `V = Φ(L Φ⁻¹(u))`. Both directions of the vine
/// recursion are checked against that linear algebra.
#[test]
fn gaussian_vine_rosenblatt_equals_cholesky_whitening() {
    let normal = Normal::new(0.0, 1.0).unwrap();
    for dim in [3usize, 5, 7] {
        let correlation = datagen::correlation(dim);
        let order: Vec<usize> = (0..dim).collect();
        for (name, vine) in [
            (
                "d-vine",
                VineCopula::gaussian_d_vine(order.clone(), correlation.clone()).unwrap(),
            ),
            (
                "c-vine",
                VineCopula::gaussian_c_vine(order.clone(), correlation.clone()).unwrap(),
            ),
        ] {
            let variable_order = vine.variable_order().to_vec();
            let mut permuted = Array2::zeros((dim, dim));
            for (a, &i) in variable_order.iter().enumerate() {
                for (b, &j) in variable_order.iter().enumerate() {
                    permuted[(a, b)] = correlation[(i, j)];
                }
            }
            let lower = math::cholesky(&permuted).unwrap();

            let mut points =
                datagen::uniforms(300, dim, 0x40 + dim as u64).mapv(|u| 0.02 + 0.9 * u);
            for col in 0..dim {
                points[(0, col)] = 1e-3;
                points[(1, col)] = 0.999;
            }
            let options = SampleOptions::default();
            let forward = vine.rosenblatt(points.view(), &options).unwrap();
            let inverse = vine.inverse_rosenblatt(points.view(), &options).unwrap();
            let mut worst = 0.0_f64;
            for row in 0..points.nrows() {
                let z: Vec<f64> = variable_order
                    .iter()
                    .map(|&var| normal.inverse_cdf(points[(row, var)]))
                    .collect();
                // Compared on the score scale, where the reference needs only
                // the (accurate) normal quantile: Φ⁻¹(U) = L⁻¹ z going
                // forward, Φ⁻¹(V) = L z going back.
                let whitened = math::solve_lower_triangular(&lower, &z).unwrap();
                for (idx, &var) in variable_order.iter().enumerate() {
                    let correlated: f64 = (0..=idx).map(|k| lower[(idx, k)] * z[k]).sum();
                    for (direction, got, expected) in [
                        ("forward", forward[(row, var)], whitened[idx]),
                        ("inverse", inverse[(row, var)], correlated),
                    ] {
                        // The result is a probability: one ulp of it is
                        // `eps / φ(z)` on the score scale, which is all the
                        // resolution an upper-tail value has.
                        let resolution = f64::EPSILON / normal.pdf(expected);
                        let deviation = ((normal.inverse_cdf(got) - expected).abs() - resolution)
                            .max(0.0)
                            / (1.0 + expected.abs());
                        assert!(
                            deviation < 1e-13,
                            "dim {dim} {name} {direction} row {row} var {var}: score {} vs {expected}",
                            normal.inverse_cdf(got)
                        );
                        worst = worst.max(deviation);
                    }
                }
            }
            println!("dim {dim} {name}: worst score deviation {worst:e}");
        }
    }
}
