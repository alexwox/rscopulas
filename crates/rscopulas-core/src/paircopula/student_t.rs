use statrs::{
    distribution::{Continuous, ContinuousCDF, StudentsT},
    function::gamma::ln_gamma,
};

use crate::errors::{CopulaError, FitError};

fn t_dist(nu: f64) -> Result<StudentsT, CopulaError> {
    StudentsT::new(0.0, 1.0, nu).map_err(|_| {
        FitError::Failed {
            reason: "student t pair nu must be positive",
        }
        .into()
    })
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
    constant: f64,
    x: Vec<f64>,
    y: Vec<f64>,
    log_marginals: Vec<f64>,
}

impl NuTerms {
    pub fn new(u1: &[f64], u2: &[f64], nu: f64) -> Result<Self, CopulaError> {
        if !nu.is_finite() || nu <= 0.0 {
            return Err(FitError::Failed {
                reason: "student t pair nu must be positive",
            }
            .into());
        }
        let dist = t_dist(nu)?;
        let mut x = Vec::with_capacity(u1.len());
        let mut y = Vec::with_capacity(u1.len());
        let mut log_marginals = Vec::with_capacity(u1.len());
        for (&u, &v) in u1.iter().zip(u2) {
            let qx = dist.inverse_cdf(u);
            let qy = dist.inverse_cdf(v);
            x.push(qx);
            y.push(qy);
            log_marginals.push(dist.ln_pdf(qx) + dist.ln_pdf(qy));
        }
        let constant =
            ln_gamma((nu + 2.0) / 2.0) - ln_gamma(nu / 2.0) - (nu * std::f64::consts::PI).ln();
        Ok(Self {
            nu,
            constant,
            x,
            y,
            log_marginals,
        })
    }

    /// Sum of the copula log-density over all observations at correlation
    /// `rho`, or `-inf` when `rho` is outside `(-1, 1)` or any row is
    /// non-finite.
    pub fn loglik(&self, rho: f64) -> f64 {
        if !rho.is_finite() || rho.abs() >= 1.0 {
            return f64::NEG_INFINITY;
        }
        let one_minus = 1.0 - rho * rho;
        let shared = self.constant - 0.5 * one_minus.ln();
        let mut total = 0.0;
        for ((&x, &y), &log_marginal) in self.x.iter().zip(&self.y).zip(&self.log_marginals) {
            let quad = (x * x - 2.0 * rho * x * y + y * y) / one_minus;
            let value = shared - 0.5 * (self.nu + 2.0) * (1.0 + quad / self.nu).ln() - log_marginal;
            if !value.is_finite() {
                return f64::NEG_INFINITY;
            }
            total += value;
        }
        total
    }
}

pub fn cdf(u: f64, v: f64, rho: f64, nu: f64) -> Result<f64, CopulaError> {
    // Student-t Plackett integral from the countermonotone limit rho=-1.
    // Unlike Gaussian copulas, rho=0 is not independence for finite nu.
    let dist = t_dist(nu)?;
    let x = dist.inverse_cdf(u);
    let y = dist.inverse_cdf(v);
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

pub fn log_pdf(u1: f64, u2: f64, rho: f64, nu: f64) -> Result<f64, CopulaError> {
    if !rho.is_finite() || rho.abs() >= 1.0 || !nu.is_finite() || nu <= 0.0 {
        return Err(FitError::Failed {
            reason: "student t pair parameters are invalid",
        }
        .into());
    }

    let dist = t_dist(nu)?;
    let x = dist.inverse_cdf(u1);
    let y = dist.inverse_cdf(u2);
    let one_minus = 1.0 - rho * rho;
    let quad = (x * x - 2.0 * rho * x * y + y * y) / one_minus;
    let mv_log_pdf = ln_gamma((nu + 2.0) / 2.0)
        - ln_gamma(nu / 2.0)
        - 0.5 * one_minus.ln()
        - (nu * std::f64::consts::PI).ln()
        - 0.5 * (nu + 2.0) * (1.0 + quad / nu).ln();
    Ok(mv_log_pdf - dist.ln_pdf(x) - dist.ln_pdf(y))
}

pub fn cond_first_given_second(u1: f64, u2: f64, rho: f64, nu: f64) -> Result<f64, CopulaError> {
    let dist = t_dist(nu)?;
    let cond = t_dist(nu + 1.0)?;
    let x = dist.inverse_cdf(u1);
    let y = dist.inverse_cdf(u2);
    let scale = (((nu + y * y) * (1.0 - rho * rho)) / (nu + 1.0)).sqrt();
    Ok(cond.cdf((x - rho * y) / scale))
}

pub fn cond_second_given_first(u1: f64, u2: f64, rho: f64, nu: f64) -> Result<f64, CopulaError> {
    let dist = t_dist(nu)?;
    let cond = t_dist(nu + 1.0)?;
    let x = dist.inverse_cdf(u1);
    let y = dist.inverse_cdf(u2);
    let scale = (((nu + x * x) * (1.0 - rho * rho)) / (nu + 1.0)).sqrt();
    Ok(cond.cdf((y - rho * x) / scale))
}

pub fn inv_first_given_second(p: f64, u2: f64, rho: f64, nu: f64) -> Result<f64, CopulaError> {
    let dist = t_dist(nu)?;
    let cond = t_dist(nu + 1.0)?;
    let y = dist.inverse_cdf(u2);
    let q = cond.inverse_cdf(p);
    let scale = (((nu + y * y) * (1.0 - rho * rho)) / (nu + 1.0)).sqrt();
    Ok(dist.cdf(rho * y + scale * q))
}

pub fn inv_second_given_first(u1: f64, p: f64, rho: f64, nu: f64) -> Result<f64, CopulaError> {
    let dist = t_dist(nu)?;
    let cond = t_dist(nu + 1.0)?;
    let x = dist.inverse_cdf(u1);
    let q = cond.inverse_cdf(p);
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
