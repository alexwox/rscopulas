use serde::{Deserialize, Serialize};

/// Standard return value for fitting routines.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FitResult<T> {
    /// The fitted model.
    pub model: T,
    /// Likelihood-based diagnostics reported by the fitter.
    pub diagnostics: crate::domain::FitDiagnostics,
}
