use ndarray::Array2;
use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::{data::PseudoObs, errors::CopulaError, fit::FitResult};

use super::{CopulaFamily, CopulaModel, EvalOptions, FitOptions, SampleOptions};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HacFamily {
    Clayton,
    Frank,
    Gumbel,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HacTree {
    Leaf(usize),
    Node(HacNode),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HacNode {
    pub family: HacFamily,
    pub theta: f64,
    pub children: Vec<HacTree>,
}

impl HacNode {
    pub fn new(family: HacFamily, theta: f64, children: Vec<HacTree>) -> Self {
        Self {
            family,
            theta,
            children,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HacStructureMethod {
    GivenTree,
    AgglomerativeTau,
    AgglomerativeTauThenCollapse,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HacFitMethod {
    TauInit,
    RecursiveMle,
    FullMle,
    Smle,
    Dmle,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HacFitOptions {
    pub base: FitOptions,
    pub structure_method: HacStructureMethod,
    pub fit_method: HacFitMethod,
    pub family_set: Vec<HacFamily>,
    pub collapse_eps: f64,
    pub mc_samples: usize,
    pub allow_experimental: bool,
}

impl Default for HacFitOptions {
    fn default() -> Self {
        Self {
            base: FitOptions::default(),
            structure_method: HacStructureMethod::AgglomerativeTauThenCollapse,
            fit_method: HacFitMethod::RecursiveMle,
            family_set: vec![HacFamily::Clayton, HacFamily::Frank, HacFamily::Gumbel],
            collapse_eps: 0.05,
            mc_samples: 256,
            allow_experimental: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HierarchicalArchimedeanCopula {
    root: HacTree,
    dim: usize,
    structure_method: HacStructureMethod,
    fit_method: HacFitMethod,
    exact: bool,
    exact_loglik: bool,
    used_smle: bool,
    mc_samples: usize,
}

impl HierarchicalArchimedeanCopula {
    pub fn new(root: HacTree) -> Result<Self, CopulaError> {
        let validation = crate::hac::validate_tree(&root)?;
        Ok(Self {
            root,
            dim: validation.dim,
            structure_method: HacStructureMethod::GivenTree,
            fit_method: HacFitMethod::TauInit,
            exact: validation.exact,
            exact_loglik: validation.exact_loglik,
            used_smle: false,
            mc_samples: 0,
        })
    }

    pub fn fit(data: &PseudoObs, options: &HacFitOptions) -> Result<FitResult<Self>, CopulaError> {
        crate::hac::fit_hac(data, None, options)
    }

    pub fn fit_with_tree(
        data: &PseudoObs,
        root: HacTree,
        options: &HacFitOptions,
    ) -> Result<FitResult<Self>, CopulaError> {
        crate::hac::fit_hac(data, Some(root), options)
    }

    pub(crate) fn from_parts(
        root: HacTree,
        structure_method: HacStructureMethod,
        fit_method: HacFitMethod,
        exact: bool,
        exact_loglik: bool,
        used_smle: bool,
        mc_samples: usize,
    ) -> Result<Self, CopulaError> {
        let validation = crate::hac::validate_tree(&root)?;
        Ok(Self {
            root,
            dim: validation.dim,
            structure_method,
            fit_method,
            exact,
            exact_loglik,
            used_smle,
            mc_samples,
        })
    }

    pub fn tree(&self) -> &HacTree {
        &self.root
    }

    pub fn is_exact(&self) -> bool {
        self.exact
    }

    pub fn exact_loglik(&self) -> bool {
        self.exact_loglik
    }

    pub fn used_smle(&self) -> bool {
        self.used_smle
    }

    pub fn mc_samples(&self) -> usize {
        self.mc_samples
    }

    pub fn structure_method(&self) -> HacStructureMethod {
        self.structure_method
    }

    pub fn fit_method(&self) -> HacFitMethod {
        self.fit_method
    }

    pub fn leaf_order(&self) -> Vec<usize> {
        crate::hac::leaf_order(&self.root)
    }

    pub fn families(&self) -> Vec<HacFamily> {
        crate::hac::families_preorder(&self.root)
    }

    pub fn parameters(&self) -> Vec<f64> {
        crate::hac::parameters_preorder(&self.root)
    }
}

impl CopulaModel for HierarchicalArchimedeanCopula {
    fn family(&self) -> CopulaFamily {
        CopulaFamily::HierarchicalArchimedean
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn log_pdf(&self, data: &PseudoObs, options: &EvalOptions) -> Result<Vec<f64>, CopulaError> {
        crate::hac::log_pdf(self, data, options)
    }

    fn sample<R: Rng + ?Sized>(
        &self,
        n: usize,
        rng: &mut R,
        options: &SampleOptions,
    ) -> Result<Array2<f64>, CopulaError> {
        crate::hac::sample(self, n, rng, options)
    }
}
