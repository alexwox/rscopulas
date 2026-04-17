use ndarray::Array2;
use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::{
    data::PseudoObs,
    errors::CopulaError,
    fit::FitResult,
    math::{inverse, make_spd_correlation},
    stats::kendall_tau_matrix,
};

use super::{
    CopulaFamily, CopulaModel, EvalOptions, FitDiagnostics, FitOptions, GaussianCopula,
    SampleOptions,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VineStructureKind {
    C,
    D,
    R,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VineCopula {
    dim: usize,
    truncation_level: Option<usize>,
    structure: VineStructureKind,
    order: Vec<usize>,
    pair_parameters: Vec<f64>,
    correlation: Array2<f64>,
}

impl VineCopula {
    pub fn new(dim: usize) -> Self {
        let correlation = Array2::eye(dim);
        let order = (0..dim).collect::<Vec<_>>();
        let pair_parameters = vec![0.0; dim * (dim - 1) / 2];
        Self {
            dim,
            truncation_level: None,
            structure: VineStructureKind::C,
            order,
            pair_parameters,
            correlation,
        }
    }

    pub fn gaussian_c_vine(
        order: Vec<usize>,
        correlation: Array2<f64>,
    ) -> Result<Self, CopulaError> {
        Self::from_structure(VineStructureKind::C, order, correlation)
    }

    pub fn gaussian_d_vine(
        order: Vec<usize>,
        correlation: Array2<f64>,
    ) -> Result<Self, CopulaError> {
        Self::from_structure(VineStructureKind::D, order, correlation)
    }

    pub fn fit_c_vine(
        data: &PseudoObs,
        _options: &FitOptions,
    ) -> Result<FitResult<Self>, CopulaError> {
        let correlation = estimate_correlation(data)?;
        let order = c_vine_order(data);
        let model = Self::from_structure(VineStructureKind::C, order, correlation)?;
        Ok(fit_result(model, data))
    }

    pub fn fit_d_vine(
        data: &PseudoObs,
        _options: &FitOptions,
    ) -> Result<FitResult<Self>, CopulaError> {
        let correlation = estimate_correlation(data)?;
        let order = d_vine_order(data);
        let model = Self::from_structure(VineStructureKind::D, order, correlation)?;
        Ok(fit_result(model, data))
    }

    pub fn fit_r_vine(
        data: &PseudoObs,
        options: &FitOptions,
    ) -> Result<FitResult<Self>, CopulaError> {
        let c_fit = Self::fit_c_vine(data, options)?;
        let d_fit = Self::fit_d_vine(data, options)?;
        let c_score = structure_score(&c_fit.model.pair_parameters);
        let d_score = structure_score(&d_fit.model.pair_parameters);

        if c_score >= d_score {
            let model = Self {
                structure: VineStructureKind::R,
                ..c_fit.model
            };
            Ok(fit_result(model, data))
        } else {
            let model = Self {
                structure: VineStructureKind::R,
                ..d_fit.model
            };
            Ok(fit_result(model, data))
        }
    }

    pub fn truncation_level(&self) -> Option<usize> {
        self.truncation_level
    }

    pub fn structure(&self) -> VineStructureKind {
        self.structure
    }

    pub fn order(&self) -> &[usize] {
        &self.order
    }

    pub fn pair_parameters(&self) -> &[f64] {
        &self.pair_parameters
    }

    pub fn correlation(&self) -> &Array2<f64> {
        &self.correlation
    }

    fn from_structure(
        structure: VineStructureKind,
        order: Vec<usize>,
        correlation: Array2<f64>,
    ) -> Result<Self, CopulaError> {
        GaussianCopula::new(correlation.clone())?;
        let pair_parameters = match structure {
            VineStructureKind::C => c_vine_partial_correlations(&order, &correlation)?,
            VineStructureKind::D => d_vine_partial_correlations(&order, &correlation)?,
            VineStructureKind::R => c_vine_partial_correlations(&order, &correlation)?,
        };

        Ok(Self {
            dim: correlation.nrows(),
            truncation_level: None,
            structure,
            order,
            pair_parameters,
            correlation,
        })
    }

    fn backend(&self) -> Result<GaussianCopula, CopulaError> {
        GaussianCopula::new(self.correlation.clone())
    }
}

impl CopulaModel for VineCopula {
    fn family(&self) -> CopulaFamily {
        CopulaFamily::Vine
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn log_pdf(&self, data: &PseudoObs, options: &EvalOptions) -> Result<Vec<f64>, CopulaError> {
        self.backend()?.log_pdf(data, options)
    }

    fn sample<R: Rng + ?Sized>(
        &self,
        n: usize,
        rng: &mut R,
        options: &SampleOptions,
    ) -> Result<Array2<f64>, CopulaError> {
        self.backend()?.sample(n, rng, options)
    }
}

fn fit_result(model: VineCopula, data: &PseudoObs) -> FitResult<VineCopula> {
    let loglik = model
        .log_pdf(data, &EvalOptions::default())
        .expect("vine log-likelihood should evaluate during fit")
        .into_iter()
        .sum::<f64>();
    let parameter_count = model.pair_parameters.len() as f64;
    let n_obs = data.n_obs() as f64;
    let diagnostics = FitDiagnostics {
        loglik,
        aic: 2.0 * parameter_count - 2.0 * loglik,
        bic: parameter_count * n_obs.ln() - 2.0 * loglik,
        converged: true,
        n_iter: 0,
    };

    FitResult { model, diagnostics }
}

fn estimate_correlation(data: &PseudoObs) -> Result<Array2<f64>, CopulaError> {
    let tau = kendall_tau_matrix(data);
    make_spd_correlation(&tau.mapv(|value| (std::f64::consts::FRAC_PI_2 * value).sin()))
        .map_err(CopulaError::from)
}

fn c_vine_order(data: &PseudoObs) -> Vec<usize> {
    let tau = kendall_tau_matrix(data);
    let mut indices = (0..data.dim()).collect::<Vec<_>>();
    indices.sort_by(|left, right| {
        let left_score = (0..data.dim())
            .filter(|idx| *idx != *left)
            .map(|idx| tau[(*left, idx)].abs())
            .sum::<f64>();
        let right_score = (0..data.dim())
            .filter(|idx| *idx != *right)
            .map(|idx| tau[(*right, idx)].abs())
            .sum::<f64>();
        right_score
            .total_cmp(&left_score)
            .then_with(|| left.cmp(right))
    });
    indices
}

fn d_vine_order(data: &PseudoObs) -> Vec<usize> {
    let tau = kendall_tau_matrix(data);
    let dim = data.dim();
    let mut best_pair = (0, 1);
    let mut best_score = f64::NEG_INFINITY;
    for left in 0..dim {
        for right in (left + 1)..dim {
            let score = tau[(left, right)].abs();
            if score > best_score {
                best_score = score;
                best_pair = (left, right);
            }
        }
    }

    let mut order = vec![best_pair.0, best_pair.1];
    let mut remaining = (0..dim)
        .filter(|idx| *idx != best_pair.0 && *idx != best_pair.1)
        .collect::<Vec<_>>();

    while !remaining.is_empty() {
        let left_end = order[0];
        let right_end = *order.last().expect("order is non-empty");
        let mut best_idx = 0usize;
        let mut append_left = false;
        let mut score = f64::NEG_INFINITY;

        for (idx, candidate) in remaining.iter().enumerate() {
            let left_score = tau[(*candidate, left_end)].abs();
            if left_score > score {
                score = left_score;
                best_idx = idx;
                append_left = true;
            }

            let right_score = tau[(*candidate, right_end)].abs();
            if right_score > score {
                score = right_score;
                best_idx = idx;
                append_left = false;
            }
        }

        let candidate = remaining.remove(best_idx);
        if append_left {
            order.insert(0, candidate);
        } else {
            order.push(candidate);
        }
    }

    order
}

fn c_vine_partial_correlations(
    order: &[usize],
    correlation: &Array2<f64>,
) -> Result<Vec<f64>, CopulaError> {
    let dim = order.len();
    let mut parameters = Vec::with_capacity(dim * (dim - 1) / 2);
    for level in 0..(dim - 1) {
        let root = order[level];
        let conditioning = &order[..level];
        for idx in (level + 1..dim).rev() {
            let variable = order[idx];
            parameters.push(partial_correlation(
                correlation,
                root,
                variable,
                conditioning,
            )?);
        }
    }
    Ok(parameters)
}

fn d_vine_partial_correlations(
    order: &[usize],
    correlation: &Array2<f64>,
) -> Result<Vec<f64>, CopulaError> {
    let dim = order.len();
    let mut parameters = Vec::with_capacity(dim * (dim - 1) / 2);
    for gap in 1..dim {
        for start in (0..(dim - gap)).rev() {
            let left = order[start];
            let right = order[start + gap];
            let conditioning = &order[(start + 1)..(start + gap)];
            parameters.push(partial_correlation(correlation, left, right, conditioning)?);
        }
    }
    Ok(parameters)
}

fn partial_correlation(
    correlation: &Array2<f64>,
    left: usize,
    right: usize,
    conditioning: &[usize],
) -> Result<f64, CopulaError> {
    if conditioning.is_empty() {
        return Ok(correlation[(left, right)]);
    }

    let mut indices = Vec::with_capacity(conditioning.len() + 2);
    indices.push(left);
    indices.push(right);
    indices.extend_from_slice(conditioning);

    let mut sub = Array2::zeros((indices.len(), indices.len()));
    for (row, source_row) in indices.iter().enumerate() {
        for (col, source_col) in indices.iter().enumerate() {
            sub[(row, col)] = correlation[(*source_row, *source_col)];
        }
    }

    let precision = inverse(&sub)?;
    Ok(-precision[(0, 1)] / (precision[(0, 0)] * precision[(1, 1)]).sqrt())
}

fn structure_score(parameters: &[f64]) -> f64 {
    parameters.iter().map(|value| value.abs()).sum()
}

#[cfg(test)]
mod tests {
    use ndarray::array;
    use rand::{SeedableRng, rngs::StdRng};

    use crate::{CopulaModel, GaussianCopula, PseudoObs};

    use super::{
        VineCopula, VineStructureKind, c_vine_partial_correlations, d_vine_partial_correlations,
    };

    #[test]
    fn c_and_d_vines_match_gaussian_backend_on_known_correlation() {
        let correlation = array![
            [1.0, 0.7, 0.3, 0.2],
            [0.7, 1.0, 0.4, 0.1],
            [0.3, 0.4, 1.0, 0.5],
            [0.2, 0.1, 0.5, 1.0]
        ];
        let gaussian =
            GaussianCopula::new(correlation.clone()).expect("correlation should be valid");
        let c_vine =
            VineCopula::from_structure(VineStructureKind::C, vec![0, 1, 2, 3], correlation.clone())
                .expect("c-vine should construct");
        let d_vine =
            VineCopula::from_structure(VineStructureKind::D, vec![0, 1, 2, 3], correlation)
                .expect("d-vine should construct");
        let data = PseudoObs::new(array![[0.2, 0.3, 0.4, 0.5], [0.7, 0.6, 0.8, 0.9],])
            .expect("pseudo-observations should be valid");

        let gaussian_log = gaussian
            .log_pdf(&data, &Default::default())
            .expect("gaussian logpdf");
        let c_log = c_vine
            .log_pdf(&data, &Default::default())
            .expect("c-vine logpdf");
        let d_log = d_vine
            .log_pdf(&data, &Default::default())
            .expect("d-vine logpdf");

        assert_eq!(gaussian_log, c_log);
        assert_eq!(gaussian_log, d_log);
    }

    #[test]
    fn fit_vines_on_gaussian_data_matches_gaussian_density() {
        let gaussian =
            GaussianCopula::new(array![[1.0, 0.6, 0.3], [0.6, 1.0, 0.4], [0.3, 0.4, 1.0],])
                .expect("correlation should be valid");
        let mut rng = StdRng::seed_from_u64(1234);
        let samples = gaussian
            .sample(256, &mut rng, &Default::default())
            .expect("sampling should succeed");
        let data = PseudoObs::new(samples).expect("samples should be valid pseudo-observations");

        let gaussian_fit = GaussianCopula::fit(&data, &Default::default()).expect("gaussian fit");
        let c_fit = VineCopula::fit_c_vine(&data, &Default::default()).expect("c-vine fit");
        let d_fit = VineCopula::fit_d_vine(&data, &Default::default()).expect("d-vine fit");

        let gaussian_log = gaussian_fit
            .model
            .log_pdf(&data, &Default::default())
            .expect("gaussian logpdf");
        let c_log = c_fit
            .model
            .log_pdf(&data, &Default::default())
            .expect("c-vine logpdf");
        let d_log = d_fit
            .model
            .log_pdf(&data, &Default::default())
            .expect("d-vine logpdf");

        for ((g, c), d) in gaussian_log.iter().zip(c_log.iter()).zip(d_log.iter()) {
            assert!((g - c).abs() < 1e-10);
            assert!((g - d).abs() < 1e-10);
        }
    }

    #[test]
    fn extracts_nontrivial_partial_correlations() {
        let correlation = array![
            [1.0, 0.7, 0.3, 0.2],
            [0.7, 1.0, 0.4, 0.1],
            [0.3, 0.4, 1.0, 0.5],
            [0.2, 0.1, 0.5, 1.0]
        ];

        let c_params =
            c_vine_partial_correlations(&[0, 1, 2, 3], &correlation).expect("c-vine params");
        let d_params =
            d_vine_partial_correlations(&[0, 1, 2, 3], &correlation).expect("d-vine params");

        assert_eq!(c_params.len(), 6);
        assert_eq!(d_params.len(), 6);
        assert!(c_params.iter().all(|value| value.is_finite()));
        assert!(d_params.iter().all(|value| value.is_finite()));
    }
}
