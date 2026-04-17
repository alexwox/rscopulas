use crate::{
    domain::{Device, ExecPolicy},
    errors::{BackendError, CopulaError},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Operation {
    DensityEval,
    VineLogPdf,
    PairBatchEval,
    PairFitScoring,
    KendallTauMatrix,
    Sample,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ExecutionStrategy {
    CpuSerial,
    CpuParallel,
}

pub(crate) fn resolve_strategy(
    policy: ExecPolicy,
    operation: Operation,
    batch_len: usize,
) -> Result<ExecutionStrategy, CopulaError> {
    match policy {
        ExecPolicy::Force(Device::Cpu) => Ok(ExecutionStrategy::CpuSerial),
        ExecPolicy::Force(Device::Cuda(device)) => resolve_accelerated_device(
            rscopulas_accel::Device::Cuda(device),
            "cuda",
            operation_name(operation),
        ),
        ExecPolicy::Force(Device::Metal) => resolve_accelerated_device(
            rscopulas_accel::Device::Metal,
            "metal",
            operation_name(operation),
        ),
        ExecPolicy::Auto => {
            if cpu_parallel_supported(operation) && batch_len >= parallel_threshold(operation) {
                Ok(ExecutionStrategy::CpuParallel)
            } else {
                Ok(ExecutionStrategy::CpuSerial)
            }
        }
    }
}

pub(crate) fn parallel_try_map_range_collect<T, F>(
    len: usize,
    strategy: ExecutionStrategy,
    f: F,
) -> Result<Vec<T>, CopulaError>
where
    T: Send,
    F: Fn(usize) -> Result<T, CopulaError> + Sync + Send,
{
    match strategy {
        ExecutionStrategy::CpuSerial => (0..len).map(f).collect(),
        ExecutionStrategy::CpuParallel => rscopulas_accel::parallel_try_map_range_collect(len, f),
    }
}

fn resolve_accelerated_device(
    device: rscopulas_accel::Device,
    backend: &'static str,
    _operation: &'static str,
) -> Result<ExecutionStrategy, CopulaError> {
    if !rscopulas_accel::is_device_available(device) {
        return Err(BackendError::Unavailable { backend }.into());
    }
    Err(BackendError::Unsupported { backend }.into())
}

fn cpu_parallel_supported(operation: Operation) -> bool {
    matches!(
        operation,
        Operation::DensityEval
            | Operation::VineLogPdf
            | Operation::PairBatchEval
            | Operation::PairFitScoring
            | Operation::KendallTauMatrix
    )
}

fn parallel_threshold(operation: Operation) -> usize {
    match operation {
        Operation::DensityEval | Operation::VineLogPdf | Operation::PairBatchEval => 128,
        Operation::PairFitScoring => 2,
        Operation::KendallTauMatrix => 4,
        Operation::Sample => usize::MAX,
    }
}

fn operation_name(operation: Operation) -> &'static str {
    match operation {
        Operation::DensityEval => "density evaluation",
        Operation::VineLogPdf => "vine log-density evaluation",
        Operation::PairBatchEval => "pair batch evaluation",
        Operation::PairFitScoring => "pair fit scoring",
        Operation::KendallTauMatrix => "Kendall tau matrix",
        Operation::Sample => "sampling",
    }
}
