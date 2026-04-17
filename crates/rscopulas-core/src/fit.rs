use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FitResult<T> {
    pub model: T,
    pub diagnostics: crate::domain::FitDiagnostics,
}
