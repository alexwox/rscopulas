use serde::{Deserialize, Serialize};

use crate::{
    errors::{CopulaError, FitError},
    math::maximize_scalar,
    vine::{SelectionCriterion, VineFitOptions},
};

use super::{clayton, frank, gaussian, gumbel, rotated, student_t};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PairCopulaFamily {
    Independence,
    Gaussian,
    StudentT,
    Clayton,
    Frank,
    Gumbel,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Rotation {
    R0,
    R90,
    R180,
    R270,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum PairCopulaParams {
    None,
    One(f64),
    Two(f64, f64),
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PairCopulaSpec {
    pub family: PairCopulaFamily,
    pub rotation: Rotation,
    pub params: PairCopulaParams,
}

#[derive(Debug, Clone)]
pub struct PairFitResult {
    pub spec: PairCopulaSpec,
    pub loglik: f64,
    pub aic: f64,
    pub bic: f64,
    pub cond_on_first: Vec<f64>,
    pub cond_on_second: Vec<f64>,
}

impl PairCopulaSpec {
    pub fn independence() -> Self {
        Self {
            family: PairCopulaFamily::Independence,
            rotation: Rotation::R0,
            params: PairCopulaParams::None,
        }
    }

    pub fn swap_axes(self) -> Self {
        let rotation = match self.rotation {
            Rotation::R90 => Rotation::R270,
            Rotation::R270 => Rotation::R90,
            other => other,
        };
        Self { rotation, ..self }
    }

    pub fn parameter_count(&self) -> usize {
        match self.params {
            PairCopulaParams::None => 0,
            PairCopulaParams::One(_) => 1,
            PairCopulaParams::Two(_, _) => 2,
        }
    }

    pub fn log_pdf(&self, u1: f64, u2: f64, clip_eps: f64) -> Result<f64, CopulaError> {
        let ((x1, x2), rotation) = rotated::to_base_inputs(self.rotation, u1, u2, clip_eps);
        let base = match (self.family, self.params) {
            (PairCopulaFamily::Independence, PairCopulaParams::None) => 0.0,
            (PairCopulaFamily::Gaussian, PairCopulaParams::One(rho)) => {
                gaussian::log_pdf(x1, x2, rho)?
            }
            (PairCopulaFamily::StudentT, PairCopulaParams::Two(rho, nu)) => {
                student_t::log_pdf(x1, x2, rho, nu)?
            }
            (PairCopulaFamily::Clayton, PairCopulaParams::One(theta)) => {
                clayton::log_pdf(x1, x2, theta)?
            }
            (PairCopulaFamily::Frank, PairCopulaParams::One(theta)) => {
                frank::log_pdf(x1, x2, theta)?
            }
            (PairCopulaFamily::Gumbel, PairCopulaParams::One(theta)) => {
                gumbel::log_pdf(x1, x2, theta)?
            }
            _ => {
                return Err(FitError::Failed {
                    reason: "pair-copula family/parameter combination is invalid",
                }
                .into());
            }
        };
        Ok(rotated::from_base_log_pdf(rotation, base))
    }

    pub fn cond_first_given_second(
        &self,
        u1: f64,
        u2: f64,
        clip_eps: f64,
    ) -> Result<f64, CopulaError> {
        match self.rotation {
            Rotation::R0 => self.base_cond_first_given_second(u1, u2, clip_eps),
            Rotation::R180 => Ok(
                1.0 - self.base_cond_first_given_second(1.0 - u1, 1.0 - u2, clip_eps)?
            ),
            Rotation::R90 => Ok(
                1.0 - self.base_cond_first_given_second(1.0 - u1, u2, clip_eps)?
            ),
            Rotation::R270 => self.base_cond_first_given_second(u1, 1.0 - u2, clip_eps),
        }
        .map(|value| value.clamp(clip_eps, 1.0 - clip_eps))
    }

    pub fn cond_second_given_first(
        &self,
        u1: f64,
        u2: f64,
        clip_eps: f64,
    ) -> Result<f64, CopulaError> {
        match self.rotation {
            Rotation::R0 => self.base_cond_second_given_first(u1, u2, clip_eps),
            Rotation::R180 => Ok(
                1.0 - self.base_cond_second_given_first(1.0 - u1, 1.0 - u2, clip_eps)?
            ),
            Rotation::R90 => self.base_cond_second_given_first(1.0 - u1, u2, clip_eps),
            Rotation::R270 => Ok(
                1.0 - self.base_cond_second_given_first(u1, 1.0 - u2, clip_eps)?
            ),
        }
        .map(|value| value.clamp(clip_eps, 1.0 - clip_eps))
    }

    pub fn inv_first_given_second(
        &self,
        p: f64,
        u2: f64,
        clip_eps: f64,
    ) -> Result<f64, CopulaError> {
        let p = p.clamp(clip_eps, 1.0 - clip_eps);
        let u2 = u2.clamp(clip_eps, 1.0 - clip_eps);
        match self.rotation {
            Rotation::R0 => self.base_inv_first_given_second(p, u2, clip_eps),
            Rotation::R180 => Ok(
                1.0 - self.base_inv_first_given_second(1.0 - p, 1.0 - u2, clip_eps)?
            ),
            Rotation::R90 => Ok(
                1.0 - self.base_inv_first_given_second(1.0 - p, u2, clip_eps)?
            ),
            Rotation::R270 => self.base_inv_first_given_second(p, 1.0 - u2, clip_eps),
        }
        .map(|value| value.clamp(clip_eps, 1.0 - clip_eps))
    }

    pub fn inv_second_given_first(
        &self,
        u1: f64,
        p: f64,
        clip_eps: f64,
    ) -> Result<f64, CopulaError> {
        let p = p.clamp(clip_eps, 1.0 - clip_eps);
        let u1 = u1.clamp(clip_eps, 1.0 - clip_eps);
        match self.rotation {
            Rotation::R0 => self.base_inv_second_given_first(u1, p, clip_eps),
            Rotation::R180 => Ok(
                1.0 - self.base_inv_second_given_first(1.0 - u1, 1.0 - p, clip_eps)?
            ),
            Rotation::R90 => self.base_inv_second_given_first(1.0 - u1, p, clip_eps),
            Rotation::R270 => Ok(
                1.0 - self.base_inv_second_given_first(u1, 1.0 - p, clip_eps)?
            ),
        }
        .map(|value| value.clamp(clip_eps, 1.0 - clip_eps))
    }

    fn base_cond_first_given_second(
        &self,
        u1: f64,
        u2: f64,
        clip_eps: f64,
    ) -> Result<f64, CopulaError> {
        let u1 = u1.clamp(clip_eps, 1.0 - clip_eps);
        let u2 = u2.clamp(clip_eps, 1.0 - clip_eps);
        match (self.family, self.params) {
            (PairCopulaFamily::Independence, PairCopulaParams::None) => Ok(u1),
            (PairCopulaFamily::Gaussian, PairCopulaParams::One(rho)) => {
                gaussian::cond_first_given_second(u1, u2, rho)
            }
            (PairCopulaFamily::StudentT, PairCopulaParams::Two(rho, nu)) => {
                student_t::cond_first_given_second(u1, u2, rho, nu)
            }
            (PairCopulaFamily::Clayton, PairCopulaParams::One(theta)) => {
                clayton::cond_first_given_second(u1, u2, theta)
            }
            (PairCopulaFamily::Frank, PairCopulaParams::One(theta)) => {
                frank::cond_first_given_second(u1, u2, theta)
            }
            (PairCopulaFamily::Gumbel, PairCopulaParams::One(theta)) => {
                gumbel::cond_first_given_second(u1, u2, theta, clip_eps)
            }
            _ => Err(FitError::Failed {
                reason: "pair-copula family/parameter combination is invalid",
            }
            .into()),
        }
    }

    fn base_cond_second_given_first(
        &self,
        u1: f64,
        u2: f64,
        clip_eps: f64,
    ) -> Result<f64, CopulaError> {
        let u1 = u1.clamp(clip_eps, 1.0 - clip_eps);
        let u2 = u2.clamp(clip_eps, 1.0 - clip_eps);
        match (self.family, self.params) {
            (PairCopulaFamily::Independence, PairCopulaParams::None) => Ok(u2),
            (PairCopulaFamily::Gaussian, PairCopulaParams::One(rho)) => {
                gaussian::cond_second_given_first(u1, u2, rho)
            }
            (PairCopulaFamily::StudentT, PairCopulaParams::Two(rho, nu)) => {
                student_t::cond_second_given_first(u1, u2, rho, nu)
            }
            (PairCopulaFamily::Clayton, PairCopulaParams::One(theta)) => {
                clayton::cond_second_given_first(u1, u2, theta)
            }
            (PairCopulaFamily::Frank, PairCopulaParams::One(theta)) => {
                frank::cond_second_given_first(u1, u2, theta)
            }
            (PairCopulaFamily::Gumbel, PairCopulaParams::One(theta)) => {
                gumbel::cond_second_given_first(u1, u2, theta, clip_eps)
            }
            _ => Err(FitError::Failed {
                reason: "pair-copula family/parameter combination is invalid",
            }
            .into()),
        }
    }

    fn base_inv_first_given_second(
        &self,
        p: f64,
        u2: f64,
        clip_eps: f64,
    ) -> Result<f64, CopulaError> {
        match (self.family, self.params) {
            (PairCopulaFamily::Independence, PairCopulaParams::None) => Ok(p),
            (PairCopulaFamily::Gaussian, PairCopulaParams::One(rho)) => {
                gaussian::inv_first_given_second(p, u2, rho)
            }
            (PairCopulaFamily::StudentT, PairCopulaParams::Two(rho, nu)) => {
                student_t::inv_first_given_second(p, u2, rho, nu)
            }
            (PairCopulaFamily::Clayton, PairCopulaParams::One(theta)) => {
                clayton::inv_first_given_second(p, u2, theta)
            }
            (PairCopulaFamily::Frank, PairCopulaParams::One(theta)) => {
                frank::inv_first_given_second(p, u2, theta)
            }
            (PairCopulaFamily::Gumbel, PairCopulaParams::One(theta)) => {
                gumbel::inv_first_given_second(p, u2, theta, clip_eps)
            }
            _ => Err(FitError::Failed {
                reason: "pair-copula family/parameter combination is invalid",
            }
            .into()),
        }
    }

    fn base_inv_second_given_first(
        &self,
        u1: f64,
        p: f64,
        clip_eps: f64,
    ) -> Result<f64, CopulaError> {
        match (self.family, self.params) {
            (PairCopulaFamily::Independence, PairCopulaParams::None) => Ok(p),
            (PairCopulaFamily::Gaussian, PairCopulaParams::One(rho)) => {
                gaussian::inv_second_given_first(u1, p, rho)
            }
            (PairCopulaFamily::StudentT, PairCopulaParams::Two(rho, nu)) => {
                student_t::inv_second_given_first(u1, p, rho, nu)
            }
            (PairCopulaFamily::Clayton, PairCopulaParams::One(theta)) => {
                clayton::inv_second_given_first(u1, p, theta)
            }
            (PairCopulaFamily::Frank, PairCopulaParams::One(theta)) => {
                frank::inv_second_given_first(u1, p, theta)
            }
            (PairCopulaFamily::Gumbel, PairCopulaParams::One(theta)) => {
                gumbel::inv_second_given_first(u1, p, theta, clip_eps)
            }
            _ => Err(FitError::Failed {
                reason: "pair-copula family/parameter combination is invalid",
            }
            .into()),
        }
    }
}

pub fn fit_pair_copula(
    u1: &[f64],
    u2: &[f64],
    options: &VineFitOptions,
) -> Result<PairFitResult, CopulaError> {
    if u1.len() != u2.len() || u1.is_empty() {
        return Err(FitError::Failed {
            reason: "pair-copula fit requires equally sized non-empty inputs",
        }
        .into());
    }

    let tau = crate::stats::kendall_tau_bivariate(u1, u2)?;
    if options
        .independence_threshold
        .is_some_and(|threshold| tau.abs() <= threshold)
    {
        return finalize_pair_fit(PairCopulaSpec::independence(), u1, u2, options);
    }

    let mut best: Option<PairFitResult> = None;
    for family in &options.family_set {
        for rotation in candidate_rotations(*family, options.include_rotations) {
            let spec = match fit_family_with_rotation(*family, *rotation, u1, u2, tau) {
                Ok(spec) => spec,
                Err(_) => continue,
            };
            let fit = match finalize_pair_fit(spec, u1, u2, options) {
                Ok(fit) => fit,
                Err(_) => continue,
            };
            if let Some(current) = &best {
                if criterion_value(&fit, options.criterion) < criterion_value(current, options.criterion)
                {
                    best = Some(fit);
                }
            } else {
                best = Some(fit);
            }
        }
    }

    best.ok_or(FitError::Failed {
        reason: "pair-copula selection produced no candidate",
    }
    .into())
}

fn finalize_pair_fit(
    spec: PairCopulaSpec,
    u1: &[f64],
    u2: &[f64],
    options: &VineFitOptions,
) -> Result<PairFitResult, CopulaError> {
    let mut loglik = 0.0;
    let mut cond_on_first = Vec::with_capacity(u1.len());
    let mut cond_on_second = Vec::with_capacity(u1.len());
    for (&left, &right) in u1.iter().zip(u2.iter()) {
        loglik += spec.log_pdf(left, right, options.base.clip_eps)?;
        cond_on_first.push(spec.cond_first_given_second(left, right, options.base.clip_eps)?);
        cond_on_second.push(spec.cond_second_given_first(left, right, options.base.clip_eps)?);
    }

    let k = spec.parameter_count() as f64;
    let n = u1.len() as f64;
    Ok(PairFitResult {
        spec,
        loglik,
        aic: 2.0 * k - 2.0 * loglik,
        bic: k * n.ln() - 2.0 * loglik,
        cond_on_first,
        cond_on_second,
    })
}

fn criterion_value(fit: &PairFitResult, criterion: SelectionCriterion) -> f64 {
    match criterion {
        SelectionCriterion::Aic => fit.aic,
        SelectionCriterion::Bic => fit.bic,
    }
}

fn candidate_rotations(family: PairCopulaFamily, include_rotations: bool) -> &'static [Rotation] {
    use PairCopulaFamily as Family;
    use Rotation as Rot;

    match family {
        Family::Independence | Family::Gaussian | Family::StudentT => &[Rot::R0],
        Family::Clayton | Family::Frank | Family::Gumbel if include_rotations => {
            &[Rot::R0, Rot::R90, Rot::R180, Rot::R270]
        }
        Family::Clayton | Family::Frank | Family::Gumbel => &[Rot::R0],
    }
}

fn fit_family_with_rotation(
    family: PairCopulaFamily,
    rotation: Rotation,
    u1: &[f64],
    u2: &[f64],
    tau: f64,
) -> Result<PairCopulaSpec, CopulaError> {
    let transformed = rotated::transform_sample(rotation, u1, u2);
    let x1 = &transformed.0;
    let x2 = &transformed.1;
    let transformed_tau = crate::stats::kendall_tau_bivariate(x1, x2)?;

    let params = match family {
        PairCopulaFamily::Independence => PairCopulaParams::None,
        PairCopulaFamily::Gaussian => {
            let rho = gaussian::tau_to_rho(tau);
            PairCopulaParams::One(rho.clamp(-0.98, 0.98))
        }
        PairCopulaFamily::StudentT => {
            let mut best = None;
            let mut best_loglik = f64::NEG_INFINITY;
            let rho_seed = gaussian::tau_to_rho(tau).clamp(-0.95, 0.95);
            for nu in student_t::candidate_nus() {
                let rho = maximize_scalar(-0.98, 0.98, 80, |rho| {
                    x1.iter()
                        .zip(x2.iter())
                        .map(|(&u, &v)| student_t::log_pdf(u, v, rho, nu).unwrap_or(f64::NEG_INFINITY))
                        .sum::<f64>()
                });
                let rho = if rho.is_finite() { rho } else { rho_seed };
                let loglik = x1
                    .iter()
                    .zip(x2.iter())
                    .map(|(&u, &v)| student_t::log_pdf(u, v, rho, nu).unwrap_or(f64::NEG_INFINITY))
                    .sum::<f64>();
                if loglik > best_loglik {
                    best_loglik = loglik;
                    best = Some((rho, nu));
                }
            }
            let (rho, nu) = best.ok_or(FitError::Failed {
                reason: "student t pair fit failed",
            })?;
            PairCopulaParams::Two(rho, nu)
        }
        PairCopulaFamily::Clayton => {
            let init = clayton::theta_from_tau(transformed_tau)?;
            let upper = (init * 4.0 + 2.0).max(20.0);
            let theta = maximize_scalar(1e-6, upper, 80, |theta| {
                x1.iter()
                    .zip(x2.iter())
                    .map(|(&u, &v)| clayton::log_pdf(u, v, theta).unwrap_or(f64::NEG_INFINITY))
                    .sum::<f64>()
            });
            PairCopulaParams::One(theta)
        }
        PairCopulaFamily::Frank => {
            let init = frank::theta_from_tau(transformed_tau)?;
            let upper = (init * 4.0 + 2.0).max(20.0);
            let theta = maximize_scalar(1e-6, upper, 80, |theta| {
                x1.iter()
                    .zip(x2.iter())
                    .map(|(&u, &v)| frank::log_pdf(u, v, theta).unwrap_or(f64::NEG_INFINITY))
                    .sum::<f64>()
            });
            PairCopulaParams::One(theta)
        }
        PairCopulaFamily::Gumbel => {
            let init = gumbel::theta_from_tau(transformed_tau)?;
            let upper = (init * 4.0 + 2.0).max(20.0);
            let theta = maximize_scalar(1.0 + 1e-6, upper, 80, |theta| {
                x1.iter()
                    .zip(x2.iter())
                    .map(|(&u, &v)| gumbel::log_pdf(u, v, theta).unwrap_or(f64::NEG_INFINITY))
                    .sum::<f64>()
            });
            PairCopulaParams::One(theta)
        }
    };

    Ok(PairCopulaSpec {
        family,
        rotation,
        params,
    })
}
