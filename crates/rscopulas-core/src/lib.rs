//! Core copula modeling for validated pseudo-observations.
//!
//! `rscopulas-core` is the main Rust surface for fitting, evaluating, and
//! sampling:
//!
//! - single-family copulas such as Gaussian, Student t, Clayton, Frank, and
//!   Gumbel-Hougaard,
//! - low-level pair-copula kernels with h-functions and inverse h-functions,
//! - C-vine, D-vine, and R-vine copulas.
//!
//! The crate assumes your data is already in pseudo-observation form: finite
//! values strictly inside `(0, 1)`. If you start from raw observations, estimate
//! or transform the marginals first, then build a [`PseudoObs`] matrix.
//!
//! # Quick start
//!
//! ```no_run
//! use ndarray::array;
//! use rand::{rngs::StdRng, SeedableRng};
//! use rscopulas_core::{CopulaModel, FitOptions, GaussianCopula, PseudoObs};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let data = PseudoObs::new(array![
//!     [0.12, 0.18],
//!     [0.21, 0.25],
//!     [0.27, 0.22],
//!     [0.35, 0.42],
//!     [0.48, 0.51],
//!     [0.56, 0.49],
//!     [0.68, 0.73],
//!     [0.82, 0.79],
//! ])?;
//!
//! let fit = GaussianCopula::fit(&data, &FitOptions::default())?;
//! println!("AIC: {}", fit.diagnostics.aic);
//!
//! let log_pdf = fit.model.log_pdf(&data, &Default::default())?;
//! println!("first log density = {}", log_pdf[0]);
//!
//! let mut rng = StdRng::seed_from_u64(7);
//! let sample = fit.model.sample(4, &mut rng, &Default::default())?;
//! println!("sample = {:?}", sample);
//! # Ok(())
//! # }
//! ```
//!
//! # Fit a vine copula
//!
//! ```no_run
//! use ndarray::array;
//! use rscopulas_core::{
//!     PairCopulaFamily, PseudoObs, SelectionCriterion, VineCopula, VineFitOptions,
//! };
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let data = PseudoObs::new(array![
//!     [0.12, 0.18, 0.21],
//!     [0.21, 0.25, 0.29],
//!     [0.27, 0.22, 0.31],
//!     [0.35, 0.42, 0.39],
//!     [0.48, 0.51, 0.46],
//!     [0.56, 0.49, 0.58],
//!     [0.68, 0.73, 0.69],
//!     [0.82, 0.79, 0.76],
//! ])?;
//!
//! let options = VineFitOptions {
//!     family_set: vec![
//!         PairCopulaFamily::Independence,
//!         PairCopulaFamily::Gaussian,
//!         PairCopulaFamily::Clayton,
//!         PairCopulaFamily::Frank,
//!         PairCopulaFamily::Gumbel,
//!     ],
//!     include_rotations: true,
//!     criterion: SelectionCriterion::Aic,
//!     truncation_level: Some(1),
//!     ..VineFitOptions::default()
//! };
//!
//! let fit = VineCopula::fit_r_vine(&data, &options)?;
//! println!("structure = {:?}", fit.model.structure());
//! println!("order = {:?}", fit.model.order());
//! # Ok(())
//! # }
//! ```
//!
//! # Backend expectations
//!
//! The crate exposes explicit execution policy controls through [`ExecPolicy`]
//! and [`Device`]. Today, `Auto` is conservative and does not promise that
//! every numerically heavy path uses CUDA or Metal. If you need a deterministic
//! backend choice, prefer `ExecPolicy::Force(...)`.
//!
mod archimedean_math;
mod backend;
mod hac;

pub mod data;
pub mod domain;
pub mod errors;
pub mod fit;
pub mod math;
pub mod paircopula;
pub mod stats;
pub mod vine;

pub use data::PseudoObs;
pub use domain::{
    ClaytonCopula, Copula, CopulaFamily, CopulaModel, Device, EvalOptions, ExecPolicy,
    FitDiagnostics, FitOptions, FrankCopula, GaussianCopula, GumbelHougaardCopula, HacFamily,
    HacFitMethod, HacFitOptions, HacNode, HacStructureMethod, HacTree,
    HierarchicalArchimedeanCopula, SampleOptions, SelectionCriterion, StudentTCopula, VineCopula,
    VineEdge, VineFitOptions, VineStructure, VineStructureKind, VineTree,
};
pub use errors::{BackendError, CopulaError, FitError, InputError, NumericalError};
pub use paircopula::{PairCopulaFamily, PairCopulaParams, PairCopulaSpec, Rotation};
