mod bb1;
mod bb6;
mod bb7;
mod bb8;
mod clayton;
mod common;
mod frank;
mod gaussian;
mod gumbel;
mod joe;
mod khoudraji;
mod polish;
mod rotated;
mod student_t;
mod tawn;
mod tll;

pub use common::{
    KhoudrajiParams, PairCopulaFamily, PairCopulaParams, PairCopulaSpec, PairFitResult, Rotation,
    TllOrder, TllParams, fit_pair_copula,
};

/// Fit a nonparametric TLL pair copula directly from pseudo-observations.
/// Thin wrapper around the `tll::fit` module-internal function so callers can
/// build a TLL state without needing a `VineFitOptions`.
pub fn tll_fit(u1: &[f64], u2: &[f64], method: TllOrder) -> Result<TllParams, crate::errors::CopulaError> {
    tll::fit(u1, u2, method)
}

pub(crate) use common::{
    PairBatchBuffers, cond_first_given_second_batch_into, cond_second_given_first_batch_into,
    evaluate_pair_batch_into, inverse_second_given_first_batch_into,
};
pub(crate) use polish::{decode_params, encode_brackets, encode_jacobian, encode_params};
