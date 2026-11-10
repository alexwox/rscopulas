use crate::{
    errors::{CopulaError, FitError},
    special::StudentT,
};

const LN_2PI: f64 = 1.837_877_066_409_345_3;

fn t_dist(nu: f64) -> Result<StudentT, CopulaError> {
    if !nu.is_finite() || nu <= 0.0 {
        return Err(FitError::Failed {
            reason: "student t pair nu must be positive",
        }
        .into());
    }
    Ok(StudentT::new(nu))
}

/// Admissible range for the degrees of freedom in pair fits. The lower end
/// keeps the quantile transform well conditioned; beyond the upper end the
/// copula is numerically indistinguishable from the Gaussian one.
pub const NU_RANGE: (f64, f64) = (2.0, 200.0);

/// Log-spaced ν grid used to warm-start the joint (ρ, ν) maximisation. The
/// grid only needs to land in the right basin — the joint polish refines ν
/// continuously afterwards — so it is coarse but spans the full [`NU_RANGE`].
pub fn candidate_nus() -> Vec<f64> {
    let min = 2.05_f64.ln();
    let max = NU_RANGE.1.ln();
    let steps = 12usize;
    (0..steps)
        .map(|idx| {
            let fraction = idx as f64 / (steps - 1) as f64;
            (min + (max - min) * fraction).exp()
        })
        .collect()
}

/// Observation-level quantities that depend on ν but not on ρ. Computing the
/// t-quantiles is by far the most expensive part of the Student-t density, so
/// a ρ search at fixed ν reuses them instead of re-inverting the CDF per
/// candidate ρ.
pub struct NuTerms {
    nu: f64,
    /// `Σ x²+y²` and `Σ xy` are not sufficient for the t likelihood, so the
    /// per-row products are kept: `sq[i] = xᵢ² + yᵢ²`, `xy[i] = xᵢ yᵢ`.
    sq: Vec<f64>,
    xy: Vec<f64>,
    /// `Σ [ln f_ν(xᵢ) + ln f_ν(yᵢ)]`, the ρ-free part of the log-likelihood.
    log_marginal_sum: f64,
    finite: bool,
}

impl NuTerms {
    pub fn new(u1: &[f64], u2: &[f64], nu: f64) -> Result<Self, CopulaError> {
        let dist = t_dist(nu)?;
        let mut sq = Vec::with_capacity(u1.len());
        let mut xy = Vec::with_capacity(u1.len());
        let mut log_marginal_sum = 0.0;
        let mut finite = true;
        for (&u, &v) in u1.iter().zip(u2) {
            let qx = dist.ppf(u);
            let qy = dist.ppf(v);
            let log_marginal = dist.ln_pdf(qx) + dist.ln_pdf(qy);
            finite &= log_marginal.is_finite();
            log_marginal_sum += log_marginal;
            sq.push(qx * qx + qy * qy);
            xy.push(qx * qy);
        }
        Ok(Self {
            nu,
            sq,
            xy,
            log_marginal_sum,
            finite,
        })
    }

    /// Sum of the copula log-density over all observations at correlation
    /// `rho`, or `-inf` when `rho` is outside `(-1, 1)` or any row is
    /// non-finite.
    pub fn loglik(&self, rho: f64) -> f64 {
        if !rho.is_finite() || rho.abs() >= 1.0 || !self.finite {
            return f64::NEG_INFINITY;
        }
        let one_minus = 1.0 - rho * rho;
        let scale = 1.0 / (one_minus * self.nu);
        let two_rho = 2.0 * rho;
        // Σ ln(1 + quad/ν) accumulated as a product in chunks: one `ln` per
        // chunk instead of one per row. Each factor is ≥ 1 and the chunk is
        // short enough that the running product cannot overflow unless a row
        // is itself astronomically large, which the finiteness check catches.
        let mut log_sum = 0.0;
        for (sq, xy) in self.sq.chunks(8).zip(self.xy.chunks(8)) {
            let mut product = 1.0_f64;
            for (&s, &c) in sq.iter().zip(xy) {
                product *= 1.0 + (s - two_rho * c) * scale;
            }
            if !product.is_finite() {
                // Fall back to per-row logs for this chunk.
                let mut chunk = 0.0;
                for (&s, &c) in sq.iter().zip(xy) {
                    chunk += (1.0 + (s - two_rho * c) * scale).ln();
                }
                log_sum += chunk;
            } else {
                log_sum += product.ln();
            }
        }
        let n = self.sq.len() as f64;
        let total = n * (-LN_2PI - 0.5 * one_minus.ln())
            - 0.5 * (self.nu + 2.0) * log_sum
            - self.log_marginal_sum;
        if total.is_finite() {
            total
        } else {
            f64::NEG_INFINITY
        }
    }
}

pub fn cdf(u: f64, v: f64, rho: f64, nu: f64) -> Result<f64, CopulaError> {
    // Student-t Plackett integral from the countermonotone limit rho=-1.
    // Unlike Gaussian copulas, rho=0 is not independence for finite nu.
    let dist = t_dist(nu)?;
    let x = dist.ppf(u);
    let y = dist.ppf(v);
    let value = super::common::integrate_1d(
        &|t| {
            let r = t.sin();
            let q = (x - y).powi(2) / (2.0 * (1.0 - r)) + (x + y).powi(2) / (2.0 * (1.0 + r));
            Ok((-0.5 * nu * (q / nu).ln_1p()).exp())
        },
        -std::f64::consts::FRAC_PI_2,
        rho.asin(),
        1e-12,
        18,
    )? / (2.0 * std::f64::consts::PI);
    let lower = (u + v - 1.0).max(0.0);
    Ok((lower + value).clamp(lower, u.min(v)))
}

fn validate(rho: f64, nu: f64) -> Result<(), CopulaError> {
    if !rho.is_finite() || rho.abs() >= 1.0 || !nu.is_finite() || nu <= 0.0 {
        return Err(FitError::Failed {
            reason: "student t pair parameters are invalid",
        }
        .into());
    }
    Ok(())
}

/// Log-density at t-quantiles `(x, y)`. The bivariate normalising constant
/// `ln Γ(ν/2 + 1) − ln Γ(ν/2) − ln(νπ)` collapses to `−ln 2π`.
#[inline]
fn log_pdf_from_quantiles(dist: &StudentT, x: f64, y: f64, rho: f64) -> f64 {
    let nu = dist.nu();
    let one_minus = 1.0 - rho * rho;
    let quad = (x * x - 2.0 * rho * x * y + y * y) / one_minus;
    let mv_log_pdf = -LN_2PI - 0.5 * one_minus.ln() - 0.5 * (nu + 2.0) * (1.0 + quad / nu).ln();
    mv_log_pdf - dist.ln_pdf(x) - dist.ln_pdf(y)
}

pub fn log_pdf(u1: f64, u2: f64, rho: f64, nu: f64) -> Result<f64, CopulaError> {
    validate(rho, nu)?;
    let dist = StudentT::new(nu);
    Ok(log_pdf_from_quantiles(
        &dist,
        dist.ppf(u1),
        dist.ppf(u2),
        rho,
    ))
}

/// Precomputed `ν`-dependent state for evaluating many points of one
/// Student t pair copula.
pub(super) struct Kernel {
    dist: StudentT,
    cond: StudentT,
    rho: f64,
    /// `sqrt((1 − ρ²) / (ν + 1))`.
    cond_scale: f64,
}

impl Kernel {
    pub(super) fn new(rho: f64, nu: f64) -> Result<Self, CopulaError> {
        validate(rho, nu)?;
        let dist = StudentT::new(nu);
        Ok(Self {
            dist,
            cond: dist.with_one_more_df(),
            rho,
            cond_scale: ((1.0 - rho * rho) / (nu + 1.0)).sqrt(),
        })
    }

    /// `(log c, h_{1|2}, h_{2|1})` from one pair of quantile evaluations.
    #[inline]
    pub(super) fn eval3(&self, u1: f64, u2: f64) -> (f64, f64, f64) {
        let nu = self.dist.nu();
        let x = self.dist.ppf(u1);
        let y = self.dist.ppf(u2);
        (
            log_pdf_from_quantiles(&self.dist, x, y, self.rho),
            self.cond
                .cdf((x - self.rho * y) / ((nu + y * y).sqrt() * self.cond_scale)),
            self.cond
                .cdf((y - self.rho * x) / ((nu + x * x).sqrt() * self.cond_scale)),
        )
    }
}

impl Kernel {
    /// `(h_{1|2}, h_{2|1})` from one pair of quantile evaluations.
    #[inline]
    pub(super) fn cond_both(&self, u1: f64, u2: f64) -> (f64, f64) {
        let nu = self.dist.nu();
        let x = self.dist.ppf(u1);
        let y = self.dist.ppf(u2);
        (
            self.cond
                .cdf((x - self.rho * y) / ((nu + y * y).sqrt() * self.cond_scale)),
            self.cond
                .cdf((y - self.rho * x) / ((nu + x * x).sqrt() * self.cond_scale)),
        )
    }

    /// `h_{2|1}⁻¹(p | u1)` followed by `h_{1|2}(u1 | ·)` at the result
    /// clamped to `[clip, 1 − clip]`. The second step reuses the t-quantile
    /// of the first unless the clamp moved it.
    #[inline]
    pub(super) fn inverse_then_cond(&self, u1: f64, p: f64, clip: f64) -> (f64, f64) {
        let nu = self.dist.nu();
        let x = self.dist.ppf(u1);
        let quantile = self.rho * x + (nu + x * x).sqrt() * self.cond_scale * self.cond.ppf(p);
        let solved = self.dist.cdf(quantile);
        let clamped = solved.clamp(clip, 1.0 - clip);
        let y = if clamped == solved {
            quantile
        } else {
            self.dist.ppf(clamped)
        };
        (
            solved,
            self.cond
                .cdf((x - self.rho * y) / ((nu + y * y).sqrt() * self.cond_scale)),
        )
    }
}

pub fn cond_first_given_second(u1: f64, u2: f64, rho: f64, nu: f64) -> Result<f64, CopulaError> {
    let dist = t_dist(nu)?;
    let cond = dist.with_one_more_df();
    let x = dist.ppf(u1);
    let y = dist.ppf(u2);
    let scale = (((nu + y * y) * (1.0 - rho * rho)) / (nu + 1.0)).sqrt();
    Ok(cond.cdf((x - rho * y) / scale))
}

pub fn cond_second_given_first(u1: f64, u2: f64, rho: f64, nu: f64) -> Result<f64, CopulaError> {
    let dist = t_dist(nu)?;
    let cond = dist.with_one_more_df();
    let x = dist.ppf(u1);
    let y = dist.ppf(u2);
    let scale = (((nu + x * x) * (1.0 - rho * rho)) / (nu + 1.0)).sqrt();
    Ok(cond.cdf((y - rho * x) / scale))
}

pub fn inv_first_given_second(p: f64, u2: f64, rho: f64, nu: f64) -> Result<f64, CopulaError> {
    let dist = t_dist(nu)?;
    let cond = dist.with_one_more_df();
    let y = dist.ppf(u2);
    let q = cond.ppf(p);
    let scale = (((nu + y * y) * (1.0 - rho * rho)) / (nu + 1.0)).sqrt();
    Ok(dist.cdf(rho * y + scale * q))
}

pub fn inv_second_given_first(u1: f64, p: f64, rho: f64, nu: f64) -> Result<f64, CopulaError> {
    let dist = t_dist(nu)?;
    let cond = dist.with_one_more_df();
    let x = dist.ppf(u1);
    let q = cond.ppf(p);
    let scale = (((nu + x * x) * (1.0 - rho * rho)) / (nu + 1.0)).sqrt();
    Ok(dist.cdf(rho * x + scale * q))
}

#[cfg(test)]
mod cdf_tests {
    #[test]
    fn cdf_matches_independent_mvtnorm_tvpack_values() {
        // R: pmvt(upper=qt(c(.1,.7),4), df=4, corr=matrix(c(1,r,r,1),2), algorithm=TVPACK(abseps=1e-12))
        for (rho, expected) in [
            (-0.9, 0.003555225659071),
            (0.0, 0.065128123319298),
            (0.8, 0.098293867703668),
            (0.99, 0.099998560803398),
        ] {
            let value = super::cdf(0.1, 0.7, rho, 4.0).unwrap();
            assert!((value - expected).abs() < 5e-12, "rho {rho}: {value}");
            assert_eq!(value, super::cdf(0.7, 0.1, rho, 4.0).unwrap());
        }
    }
}
