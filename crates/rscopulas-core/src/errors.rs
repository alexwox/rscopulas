use thiserror::Error;

#[derive(Debug, Error)]
pub enum CopulaError {
    #[error(transparent)]
    InvalidInput(#[from] InputError),
    #[error(transparent)]
    FitFailed(#[from] FitError),
    #[error(transparent)]
    Numerical(#[from] NumericalError),
    #[error(transparent)]
    Backend(#[from] BackendError),
}

#[derive(Debug, Error)]
pub enum InputError {
    #[error("expected a 2D array of pseudo-observations")]
    ExpectedMatrix,
    #[error("expected at least one observation")]
    EmptyObservations,
    #[error("expected at least two dimensions, got {0}")]
    DimensionTooSmall(usize),
    #[error("all values must be finite")]
    NonFiniteValue,
    #[error(
        "pseudo-observations must lie strictly inside (0, 1); found {value} at row {row}, col {col}"
    )]
    OutOfUnitInterval { row: usize, col: usize, value: f64 },
}

#[derive(Debug, Error)]
pub enum FitError {
    #[error("family {family} does not support dimension {dim}")]
    UnsupportedDimension { family: &'static str, dim: usize },
    #[error("optimization is not implemented yet")]
    NotImplemented,
}

#[derive(Debug, Error)]
pub enum NumericalError {
    #[error("matrix is not a valid correlation matrix")]
    InvalidCorrelationMatrix,
    #[error("numerical routine not implemented yet")]
    NotImplemented,
}

#[derive(Debug, Error)]
pub enum BackendError {
    #[error("requested backend {backend} is unavailable")]
    Unavailable { backend: &'static str },
    #[error("requested operation is not supported on backend {backend}")]
    Unsupported { backend: &'static str },
}
