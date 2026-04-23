//! Core copula modeling for validated pseudo-observations.
//!
//! `rscopulas` is the main Rust surface for fitting, evaluating, and
//! sampling:
//!
//! - single-family copulas such as Gaussian, Student t, Clayton, Frank, and
//!   Gumbel-Hougaard,
//! - low-level pair-copula kernels with h-functions and inverse h-functions,
//!   including **Khoudraji** asymmetric pair copulas,
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
//! use rscopulas::{CopulaModel, FitOptions, GaussianCopula, PseudoObs};
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
//! use rscopulas::{
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
//!         PairCopulaFamily::Khoudraji,
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
//! # Conditional sampling from a vine
//!
//! A fitted vine natively supports conditional simulation through the
//! Rosenblatt transform. Pin the conditioning column at the Rosenblatt
//! anchor position `variable_order[0]` by placing it at the end of `order`,
//! then feed its uniforms into `inverse_rosenblatt` alongside fresh random
//! uniforms for the free columns:
//!
//! ```no_run
//! use ndarray::{Array2, Axis, s};
//! use rand::{rngs::StdRng, Rng, SeedableRng};
//! use rscopulas::{PseudoObs, SampleOptions, VineCopula, VineFitOptions};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! # let data = PseudoObs::new(ndarray::array![
//! #     [0.12, 0.18, 0.21], [0.21, 0.25, 0.29], [0.27, 0.22, 0.31],
//! #     [0.35, 0.42, 0.39], [0.48, 0.51, 0.46], [0.56, 0.49, 0.58],
//! #     [0.68, 0.73, 0.69], [0.82, 0.79, 0.76],
//! # ])?;
//! let target = 2usize; // column to condition on
//! let order = vec![0, 1, target];
//! let vine = VineCopula::fit_c_vine_with_order(
//!     &data, &order, &VineFitOptions::default(),
//! )?.model;
//! assert_eq!(vine.variable_order()[0], target);
//!
//! // Build U: known column at target, fresh uniforms elsewhere.
//! let n = 1_000;
//! let d = vine.variable_order().len();
//! let known = Array2::<f64>::from_elem((n, 1), 0.9);
//! let mut rng = StdRng::seed_from_u64(0);
//! let mut u = Array2::<f64>::zeros((n, d));
//! u.column_mut(target).assign(&known.index_axis(Axis(1), 0));
//! for obs in 0..n {
//!     for var in 0..d {
//!         if var != target {
//!             u[(obs, var)] = rng.random();
//!         }
//!     }
//! }
//! let v = vine.inverse_rosenblatt(u.view(), &SampleOptions::default())?;
//! assert_eq!(v.shape(), &[n, d]);
//! # Ok(())
//! # }
//! ```
//!
//! The Python API ships a `VineCopula.sample_conditional(known, n, seed)`
//! convenience that does this plumbing for you and also supports `k >= 2`
//! known columns via a partial forward Rosenblatt pass.
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
mod cuda_backend;
mod hac;
mod metal_backend;

#[doc(hidden)]
pub mod accel;
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
    FactorCopula, FactorFitOptions, FactorFitResult, FactorLayout, FitDiagnostics, FitOptions,
    FrankCopula, GaussianCopula, GumbelHougaardCopula, HacFamily, HacFitMethod, HacFitOptions,
    HacNode, HacStructureMethod, HacTree, HierarchicalArchimedeanCopula, SampleOptions,
    SelectionCriterion, StudentTCopula, TreeAlgorithm, TreeCriterion, VineCopula, VineEdge,
    VineFitOptions, VineStructure, VineStructureKind, VineTree,
};
pub use errors::{BackendError, CopulaError, FitError, InputError, NumericalError};
pub use paircopula::{
    KhoudrajiParams, PairCopulaFamily, PairCopulaParams, PairCopulaSpec, Rotation, TllOrder,
    TllParams, tll_fit,
};
