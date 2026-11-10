use crate::{
    accel,
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
    SpearmanRhoMatrix,
    HoeffdingDMatrix,
    Sample,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ExecutionStrategy {
    CpuSerial,
    CpuParallel,
    Cuda(u32),
    Metal,
}

pub(crate) fn resolve_strategy(
    policy: ExecPolicy,
    operation: Operation,
    batch_len: usize,
) -> Result<ExecutionStrategy, CopulaError> {
    match policy {
        ExecPolicy::Force(Device::Cpu) => Ok(cpu_strategy(operation, batch_len)),
        // GPU force requests only become device strategies for operations that
        // have an actual GPU-aware bridge. Everything else remains explicit:
        // either a deliberate CPU fallback at the call site or an
        // `Unsupported` backend error here.
        ExecPolicy::Force(Device::Cuda(device)) => {
            resolve_accelerated_device(Device::Cuda(device), operation, batch_len)
        }
        ExecPolicy::Force(Device::Metal) => {
            resolve_accelerated_device(Device::Metal, operation, batch_len)
        }
        // `Auto` stays conservative for now: it can pick CPU parallelism, but it
        // will not opportunistically jump to CUDA or Metal until the GPU parity
        // story is broader than the current narrow kernel set.
        ExecPolicy::Auto => Ok(cpu_strategy(operation, batch_len)),
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
        ExecutionStrategy::CpuParallel => accel::parallel_try_map_range_collect(len, f),
        ExecutionStrategy::Cuda(_) | ExecutionStrategy::Metal => {
            unreachable!("GPU strategies must be handled before CPU range mapping")
        }
    }
}

/// Rows per task when a row-wise evaluation is split across threads. Large
/// enough to amortise task overhead and per-task scratch allocation, small
/// enough to balance well for a few thousand rows.
pub(crate) const ROW_CHUNK: usize = 256;

/// Fills `out[row] = f(row, scratch)` for every row, serially or in parallel
/// chunks. `init` builds the scratch state once per task rather than once per
/// row, which is what keeps row kernels allocation-free.
pub(crate) fn try_fill_rows<S, I, F>(
    out: &mut [f64],
    strategy: ExecutionStrategy,
    init: I,
    f: F,
) -> Result<(), CopulaError>
where
    I: Fn() -> S + Sync + Send,
    F: Fn(usize, &mut S) -> Result<f64, CopulaError> + Sync + Send,
{
    try_fill_rows_chunked(out, strategy, ROW_CHUNK, init, f)
}

/// [`try_fill_rows`] with an explicit task size, for kernels whose rows are
/// expensive enough (a quadrature per row, say) that a few of them already
/// make a worthwhile task.
pub(crate) fn try_fill_rows_chunked<S, I, F>(
    out: &mut [f64],
    strategy: ExecutionStrategy,
    chunk_rows: usize,
    init: I,
    f: F,
) -> Result<(), CopulaError>
where
    I: Fn() -> S + Sync + Send,
    F: Fn(usize, &mut S) -> Result<f64, CopulaError> + Sync + Send,
{
    match strategy {
        ExecutionStrategy::CpuSerial => {
            let mut scratch = init();
            for (row, slot) in out.iter_mut().enumerate() {
                *slot = f(row, &mut scratch)?;
            }
            Ok(())
        }
        ExecutionStrategy::CpuParallel => {
            use rayon::prelude::*;
            let chunk_rows = chunk_rows.max(1);
            out.par_chunks_mut(chunk_rows)
                .enumerate()
                .try_for_each(|(chunk, slots)| {
                    let mut scratch = init();
                    let start = chunk * chunk_rows;
                    for (offset, slot) in slots.iter_mut().enumerate() {
                        *slot = f(start + offset, &mut scratch)?;
                    }
                    Ok(())
                })
        }
        ExecutionStrategy::Cuda(_) | ExecutionStrategy::Metal => {
            unreachable!("GPU strategies must be handled before CPU row filling")
        }
    }
}

fn resolve_accelerated_device(
    device: Device,
    operation: Operation,
    batch_len: usize,
) -> Result<ExecutionStrategy, CopulaError> {
    match operation {
        // Pair-fit uses CPU control flow around pair-batch evaluation. The outer
        // candidate loop remains a CPU strategy, while the pair-batch helper can
        // still use the forced GPU backend for supported Gaussian kernels.
        Operation::PairFitScoring => return Ok(cpu_strategy(operation, batch_len)),
        Operation::PairBatchEval | Operation::VineLogPdf => {}
        Operation::DensityEval
        | Operation::KendallTauMatrix
        | Operation::SpearmanRhoMatrix
        | Operation::HoeffdingDMatrix
        | Operation::Sample => {
            let backend = backend_name(device);
            return Err(BackendError::Unsupported { backend }.into());
        }
    }

    let accel_device = accel_device(device);
    let backend = backend_name(device);
    if !accel::is_device_available(accel_device) {
        return Err(BackendError::Unavailable { backend }.into());
    }

    Ok(match device {
        Device::Cpu => ExecutionStrategy::CpuSerial,
        Device::Cuda(ordinal) => ExecutionStrategy::Cuda(ordinal),
        Device::Metal => ExecutionStrategy::Metal,
    })
}

fn cpu_parallel_supported(operation: Operation) -> bool {
    matches!(
        operation,
        Operation::DensityEval
            | Operation::VineLogPdf
            | Operation::PairBatchEval
            | Operation::PairFitScoring
            | Operation::KendallTauMatrix
            | Operation::SpearmanRhoMatrix
            | Operation::HoeffdingDMatrix
            | Operation::Sample
    )
}

fn cpu_strategy(operation: Operation, batch_len: usize) -> ExecutionStrategy {
    if cpu_parallel_supported(operation) && batch_len >= parallel_threshold(operation) {
        ExecutionStrategy::CpuParallel
    } else {
        ExecutionStrategy::CpuSerial
    }
}

fn parallel_threshold(operation: Operation) -> usize {
    match operation {
        Operation::DensityEval | Operation::VineLogPdf | Operation::PairBatchEval => 128,
        Operation::PairFitScoring => 2,
        Operation::KendallTauMatrix => 4,
        // Spearman's ρ is O(n log n) per pair (same as Kendall's τ's merge-
        // sort). Use the same small threshold so d ≥ 4 goes parallel.
        Operation::SpearmanRhoMatrix => 4,
        // Hoeffding's D is O(n²) per pair because of the bivariate-midrank
        // loop; parallelise earlier.
        Operation::HoeffdingDMatrix => 2,
        Operation::Sample => 128,
    }
}

fn backend_name(device: Device) -> &'static str {
    match device {
        Device::Cpu => "cpu",
        Device::Cuda(_) => "cuda",
        Device::Metal => "metal",
    }
}

fn accel_device(device: Device) -> accel::Device {
    match device {
        Device::Cpu => accel::Device::Cpu,
        Device::Cuda(ordinal) => accel::Device::Cuda(ordinal),
        Device::Metal => accel::Device::Metal,
    }
}
