//! Tail-dependence diagnostics.
//!
//! Two views of the same question, "do the extremes of two variables arrive
//! together?":
//!
//! - what a fitted model implies: the closed-form tail-dependence
//!   coefficients of a pair copula ([`PairCopulaSpec::tail_dependence`]) or
//!   of every pair of a multivariate copula, and the model's tail
//!   concentration function ([`PairCopulaSpec::tail_concentration`]);
//! - what the data show: rank-based estimates
//!   ([`empirical_tail_dependence`], [`empirical_tail_dependence_curve`],
//!   [`empirical_tail_dependence_matrix`], [`empirical_tail_concentration`]).
//!
//! The coefficients are
//!
//! ```text
//! lower = lim  P(U2 <= q | U1 <= q)      as q -> 0
//! upper = lim  P(U2 >  q | U1 >  q)      as q -> 1
//! ```
//!
//! and the tail concentration function is the same conditional probability
//! at a finite `q`: `C(q, q) / q` for `q <= 1/2` and
//! `(1 - 2q + C(q, q)) / (1 - q)` above it.
//!
//! The empirical functions only use the ranks of their inputs, so they accept
//! raw observations as well as pseudo-observations.

use ndarray::{Array2, ArrayView2};
use serde::{Deserialize, Serialize};

use crate::{
    ClaytonCopula, CopulaModel, FrankCopula, GaussianCopula, GumbelHougaardCopula, StudentTCopula,
    errors::{CopulaError, InputError},
    paircopula::{PairCopulaFamily, PairCopulaParams, PairCopulaSpec, Rotation},
    special::StudentT,
};

/// Clip used when a model's distribution function is evaluated for
/// [`PairCopulaSpec::tail_concentration`].
const CONCENTRATION_CLIP: f64 = 1e-12;

/// Lower and upper tail-dependence coefficients of one pair, each in `[0, 1]`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TailDependence {
    pub lower: f64,
    pub upper: f64,
}

impl TailDependence {
    const NONE: Self = Self {
        lower: 0.0,
        upper: 0.0,
    };

    fn swapped(self) -> Self {
        Self {
            lower: self.upper,
            upper: self.lower,
        }
    }
}

/// Pairwise tail-dependence coefficients of a `d`-dimensional model or
/// sample. Both matrices are symmetric with a unit diagonal.
#[derive(Debug, Clone, PartialEq)]
pub struct TailDependenceMatrix {
    pub lower: Array2<f64>,
    pub upper: Array2<f64>,
}

impl TailDependenceMatrix {
    fn constant(dim: usize, pair: TailDependence) -> Self {
        let fill = |value: f64| {
            let mut matrix = Array2::from_elem((dim, dim), value);
            for idx in 0..dim {
                matrix[(idx, idx)] = 1.0;
            }
            matrix
        };
        Self {
            lower: fill(pair.lower),
            upper: fill(pair.upper),
        }
    }
}

/// `2 - 2^(1/x)`, the upper coefficient shared by the Gumbel-type families.
fn two_minus_root(x: f64) -> f64 {
    (2.0 - 2f64.powf(1.0 / x)).clamp(0.0, 1.0)
}

/// `2^(-1/x)`, the lower coefficient shared by the Clayton-type families.
fn inverse_root(x: f64) -> f64 {
    2f64.powf(-1.0 / x).clamp(0.0, 1.0)
}

/// Both coefficients of a bivariate Student t copula (they are equal):
/// `2 * T_{nu+1}(-sqrt((nu + 1)(1 - rho) / (1 + rho)))`.
fn student_t_coefficient(rho: f64, nu: f64) -> f64 {
    let argument = -((nu + 1.0) * (1.0 - rho) / (1.0 + rho)).sqrt();
    (2.0 * StudentT::new(nu + 1.0).cdf(argument)).clamp(0.0, 1.0)
}

/// `1 + s - (1 + s^theta)^(1/theta)`: upper coefficient `2 - 2 A(1/2)` of a
/// Tawn copula whose free asymmetry parameter is `s`.
fn tawn_upper(theta: f64, shape: f64) -> f64 {
    (1.0 + shape - (1.0 + shape.powf(theta)).powf(1.0 / theta)).clamp(0.0, 1.0)
}

impl PairCopulaSpec {
    /// Closed-form tail-dependence coefficients of this pair copula.
    ///
    /// Returns `Ok(None)` for the two families that have no closed form here:
    /// Khoudraji (the coefficients depend on the tail-dependence *functions*
    /// of both base copulas) and TLL (a kernel estimate on a finite grid has
    /// no meaningful limit).
    ///
    /// Rotations by 180 degrees swap the two coefficients. Rotations by 90
    /// and 270 degrees move the dependence to the off-diagonal corners, so
    /// both coefficients are zero; the Student t copula is the exception,
    /// because rotating it is the same as negating `rho`.
    pub fn tail_dependence(&self) -> Result<Option<TailDependence>, CopulaError> {
        use PairCopulaFamily as F;
        use PairCopulaParams as P;
        self.validate()?;

        let base = match (&self.family, &self.params) {
            (F::Independence, P::None) | (F::Gaussian, P::One(_)) | (F::Frank, P::One(_)) => {
                TailDependence::NONE
            }
            (F::StudentT, P::Two(rho, nu)) => {
                let rho = match self.rotation {
                    Rotation::R0 | Rotation::R180 => *rho,
                    Rotation::R90 | Rotation::R270 => -*rho,
                };
                let value = student_t_coefficient(rho, *nu);
                return Ok(Some(TailDependence {
                    lower: value,
                    upper: value,
                }));
            }
            (F::Clayton, P::One(theta)) => TailDependence {
                lower: inverse_root(*theta),
                upper: 0.0,
            },
            (F::Gumbel | F::Joe, P::One(theta)) => TailDependence {
                lower: 0.0,
                upper: two_minus_root(*theta),
            },
            (F::Bb1, P::Two(theta, delta)) => TailDependence {
                lower: inverse_root(theta * delta),
                upper: two_minus_root(*delta),
            },
            (F::Bb6, P::Two(theta, delta)) => TailDependence {
                lower: 0.0,
                upper: two_minus_root(theta * delta),
            },
            (F::Bb7, P::Two(theta, delta)) => TailDependence {
                lower: inverse_root(*delta),
                upper: two_minus_root(*theta),
            },
            // BB8 has no tail dependence except on the boundary delta = 1,
            // where it is the Joe copula.
            (F::Bb8, P::Two(theta, delta)) => TailDependence {
                lower: 0.0,
                upper: if *delta == 1.0 {
                    two_minus_root(*theta)
                } else {
                    0.0
                },
            },
            (F::Tawn1 | F::Tawn2, P::Two(theta, shape)) => TailDependence {
                lower: 0.0,
                upper: tawn_upper(*theta, *shape),
            },
            (F::Khoudraji, _) | (F::Tll, _) => return Ok(None),
            // `validate` has already rejected every other combination.
            _ => return Ok(None),
        };

        Ok(Some(match self.rotation {
            Rotation::R0 => base,
            Rotation::R180 => base.swapped(),
            Rotation::R90 | Rotation::R270 => TailDependence::NONE,
        }))
    }

    /// The model's tail concentration function at `q` in `(0, 1)`:
    /// `P(U2 <= q | U1 <= q)` for `q <= 1/2` and `P(U2 > q | U1 > q)` above.
    ///
    /// It tends to the lower coefficient as `q -> 0` and to the upper one as
    /// `q -> 1`, so plotting it over [`empirical_tail_concentration`] shows
    /// whether the fitted family follows the data into both tails.
    pub fn tail_concentration(&self, q: f64) -> Result<f64, CopulaError> {
        if !q.is_finite() || q <= 0.0 || q >= 1.0 {
            return Err(InputError::InvalidArgument {
                reason: "tail concentration levels must lie strictly inside (0, 1)",
            }
            .into());
        }
        let joint = self.cdf(q, q, CONCENTRATION_CLIP)?;
        let value = if q <= 0.5 {
            joint / q
        } else {
            (1.0 - 2.0 * q + joint) / (1.0 - q)
        };
        Ok(value.clamp(0.0, 1.0))
    }
}

impl GaussianCopula {
    /// Pairwise tail-dependence coefficients: zero for every pair, because a
    /// Gaussian copula with `|rho| < 1` is tail independent.
    pub fn tail_dependence(&self) -> TailDependenceMatrix {
        TailDependenceMatrix::constant(self.dim(), TailDependence::NONE)
    }
}

impl StudentTCopula {
    /// Pairwise tail-dependence coefficients. Lower and upper are equal:
    /// `2 * T_{nu+1}(-sqrt((nu + 1)(1 - rho_ij) / (1 + rho_ij)))`.
    pub fn tail_dependence(&self) -> TailDependenceMatrix {
        let dim = self.dim();
        let correlation = self.correlation();
        let nu = self.degrees_of_freedom();
        let mut matrix = Array2::eye(dim);
        for left in 0..dim {
            for right in (left + 1)..dim {
                let value = student_t_coefficient(correlation[(left, right)], nu);
                matrix[(left, right)] = value;
                matrix[(right, left)] = value;
            }
        }
        TailDependenceMatrix {
            lower: matrix.clone(),
            upper: matrix,
        }
    }
}

impl ClaytonCopula {
    /// Pairwise tail-dependence coefficients: lower `2^(-1/theta)`, upper 0.
    pub fn tail_dependence(&self) -> TailDependenceMatrix {
        TailDependenceMatrix::constant(
            self.dim(),
            TailDependence {
                lower: inverse_root(self.theta()),
                upper: 0.0,
            },
        )
    }
}

impl FrankCopula {
    /// Pairwise tail-dependence coefficients: zero for every pair.
    pub fn tail_dependence(&self) -> TailDependenceMatrix {
        TailDependenceMatrix::constant(self.dim(), TailDependence::NONE)
    }
}

impl GumbelHougaardCopula {
    /// Pairwise tail-dependence coefficients: lower 0, upper `2 - 2^(1/theta)`.
    pub fn tail_dependence(&self) -> TailDependenceMatrix {
        TailDependenceMatrix::constant(
            self.dim(),
            TailDependence {
                lower: 0.0,
                upper: two_minus_root(self.theta()),
            },
        )
    }
}

/// The number of tail observations used when the caller does not choose:
/// `floor(sqrt(n))`, at least 1.
///
/// This balances the two errors of the estimator. A small `k` uses only
/// genuine extremes but averages over few points; a large `k` is stable but
/// reaches into the body of the distribution, where the conditional
/// probability is no longer the tail limit.
pub fn default_tail_k(n_obs: usize) -> usize {
    ((n_obs as f64).sqrt().floor() as usize).clamp(1, n_obs.max(1))
}

/// Sorted per-observation `max(rank1, rank2)` and `min(rank1, rank2)`.
///
/// `max <= k` exactly when both ranks are among the `k` smallest, and
/// `min > n - k` exactly when both are among the `k` largest, so every
/// estimator below is a count on one of these two sorted arrays.
struct JointRanks {
    n_obs: usize,
    sorted_max: Vec<f64>,
    sorted_min: Vec<f64>,
}

impl JointRanks {
    fn new(first: &[f64], second: &[f64]) -> Result<Self, CopulaError> {
        if first.len() != second.len() {
            return Err(InputError::InvalidArgument {
                reason: "tail estimators need two samples of the same length",
            }
            .into());
        }
        if first.len() < 2 {
            return Err(InputError::InvalidArgument {
                reason: "tail estimators need at least two observations",
            }
            .into());
        }
        if first.iter().chain(second).any(|value| !value.is_finite()) {
            return Err(InputError::NonFiniteValue.into());
        }
        let rank_first = average_ranks(first);
        let rank_second = average_ranks(second);
        let mut sorted_max = Vec::with_capacity(first.len());
        let mut sorted_min = Vec::with_capacity(first.len());
        for (a, b) in rank_first.iter().zip(&rank_second) {
            sorted_max.push(a.max(*b));
            sorted_min.push(a.min(*b));
        }
        sorted_max.sort_by(f64::total_cmp);
        sorted_min.sort_by(f64::total_cmp);
        Ok(Self {
            n_obs: first.len(),
            sorted_max,
            sorted_min,
        })
    }

    /// Observations with both ranks `<= bound`.
    fn count_both_at_most(&self, bound: f64) -> usize {
        self.sorted_max.partition_point(|value| *value <= bound)
    }

    /// Observations with both ranks `> bound`.
    fn count_both_above(&self, bound: f64) -> usize {
        self.n_obs - self.sorted_min.partition_point(|value| *value <= bound)
    }

    fn at_k(&self, k: usize) -> Result<TailDependence, CopulaError> {
        if k == 0 || k > self.n_obs {
            return Err(InputError::InvalidArgument {
                reason: "k must lie between 1 and the number of observations",
            }
            .into());
        }
        let scale = k as f64;
        Ok(TailDependence {
            lower: (self.count_both_at_most(scale) as f64 / scale).min(1.0),
            upper: (self.count_both_above((self.n_obs - k) as f64) as f64 / scale).min(1.0),
        })
    }
}

/// 1-based ranks with ties sharing the average of the ranks they span.
fn average_ranks(values: &[f64]) -> Vec<f64> {
    let mut order: Vec<usize> = (0..values.len()).collect();
    order.sort_by(|a, b| values[*a].total_cmp(&values[*b]));
    let mut ranks = vec![0.0; values.len()];
    let mut start = 0;
    while start < order.len() {
        let mut end = start + 1;
        while end < order.len() && values[order[end]] == values[order[start]] {
            end += 1;
        }
        // Positions start..end hold ranks start+1 ..= end.
        let shared = (start + 1 + end) as f64 / 2.0;
        for position in start..end {
            ranks[order[position]] = shared;
        }
        start = end;
    }
    ranks
}

/// Rank-based estimate of both tail-dependence coefficients from the `k`
/// most extreme observations in each tail (Schmidt and Stadtmüller, 2006):
///
/// ```text
/// lower = #{ i : R_i <= k      and S_i <= k     } / k
/// upper = #{ i : R_i >  n - k  and S_i >  n - k } / k
/// ```
///
/// with `R_i`, `S_i` the ranks of the two samples. See [`default_tail_k`]
/// for the usual choice of `k`, and [`empirical_tail_dependence_curve`] to
/// look for a stable range of `k` instead of trusting one value.
pub fn empirical_tail_dependence(
    first: &[f64],
    second: &[f64],
    k: usize,
) -> Result<TailDependence, CopulaError> {
    JointRanks::new(first, second)?.at_k(k)
}

/// [`empirical_tail_dependence`] at every `k` in `ks`, sharing one ranking.
///
/// Plotted against `k`, the estimates are noisy for small `k` and drift
/// toward the body of the distribution for large `k`; a plateau in between
/// is the value to read off.
pub fn empirical_tail_dependence_curve(
    first: &[f64],
    second: &[f64],
    ks: &[usize],
) -> Result<Vec<TailDependence>, CopulaError> {
    let ranks = JointRanks::new(first, second)?;
    ks.iter().map(|k| ranks.at_k(*k)).collect()
}

/// [`empirical_tail_dependence`] for every pair of columns of an `(n, d)`
/// matrix. Pass `PseudoObs::as_view()` or a matrix of raw observations.
pub fn empirical_tail_dependence_matrix(
    data: ArrayView2<'_, f64>,
    k: usize,
) -> Result<TailDependenceMatrix, CopulaError> {
    let dim = data.ncols();
    let columns: Vec<Vec<f64>> = (0..dim).map(|idx| data.column(idx).to_vec()).collect();
    let mut lower = Array2::eye(dim);
    let mut upper = Array2::eye(dim);
    for left in 0..dim {
        for right in (left + 1)..dim {
            let pair = empirical_tail_dependence(&columns[left], &columns[right], k)?;
            lower[(left, right)] = pair.lower;
            lower[(right, left)] = pair.lower;
            upper[(left, right)] = pair.upper;
            upper[(right, left)] = pair.upper;
        }
    }
    Ok(TailDependenceMatrix { lower, upper })
}

/// Empirical tail concentration function at each level in `levels`.
///
/// With `u_i = R_i / (n + 1)` and `v_i = S_i / (n + 1)`:
///
/// ```text
/// q <= 1/2 :  #{ i : u_i <= q and v_i <= q } / (n q)
/// q >  1/2 :  #{ i : u_i >  q and v_i >  q } / (n (1 - q))
/// ```
///
/// Compare with [`PairCopulaSpec::tail_concentration`] of a fitted model.
pub fn empirical_tail_concentration(
    first: &[f64],
    second: &[f64],
    levels: &[f64],
) -> Result<Vec<f64>, CopulaError> {
    let ranks = JointRanks::new(first, second)?;
    let n_obs = ranks.n_obs as f64;
    levels
        .iter()
        .map(|&q| {
            if !q.is_finite() || q <= 0.0 || q >= 1.0 {
                return Err(InputError::InvalidArgument {
                    reason: "tail concentration levels must lie strictly inside (0, 1)",
                }
                .into());
            }
            // u_i <= q  <=>  R_i <= q (n + 1).
            let bound = q * (n_obs + 1.0);
            let value = if q <= 0.5 {
                ranks.count_both_at_most(bound) as f64 / (n_obs * q)
            } else {
                ranks.count_both_above(bound) as f64 / (n_obs * (1.0 - q))
            };
            Ok(value.min(1.0))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn average_ranks_share_ties() {
        assert_eq!(
            average_ranks(&[0.3, 0.1, 0.3, 0.9]),
            vec![2.5, 1.0, 2.5, 4.0]
        );
    }

    #[test]
    fn default_k_is_the_integer_square_root() {
        assert_eq!(default_tail_k(1), 1);
        assert_eq!(default_tail_k(99), 9);
        assert_eq!(default_tail_k(100), 10);
        assert_eq!(default_tail_k(0), 1);
    }
}
