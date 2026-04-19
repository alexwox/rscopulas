//! Internal acceleration support for `rscopulas`.
//!
//! This module has three roles:
//! - provide CPU-parallel helpers used by the main crate
//! - report backend capability information for CUDA and Metal
//! - expose a narrow GPU facade for the first batch kernels that have true
//!   device implementations
//!
//! The current GPU surface is intentionally narrow:
//! - CUDA targets `f64` Gaussian pair batch evaluation and is the primary GPU
//!   path for the current numerics
//! - Metal targets a bounded `f32` Gaussian pair batch kernel and is only used
//!   where the reduced-precision contract is acceptable

#[cfg(feature = "cuda")]
use std::process::Command;

use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Device targets understood by the acceleration facade.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Device {
    Cpu,
    Cuda(u32),
    Metal,
}

/// Capability snapshot reported by the accel module.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComputeCapabilities {
    pub cpu_simd: bool,
    pub rayon_threads: usize,
    pub cuda: Option<CudaCaps>,
    pub metal: Option<MetalCaps>,
}

/// Runtime-reported CUDA capability summary.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CudaCaps {
    pub device_count: usize,
    pub fp64: bool,
}

/// Runtime-reported Metal capability summary.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetalCaps {
    pub device_count: usize,
    pub fp64: bool,
}

/// Narrow Gaussian pair-kernel request shared by CUDA and Metal backends.
#[derive(Debug, Clone, Copy)]
pub struct GaussianPairBatchRequest<'a> {
    pub u1: &'a [f64],
    pub u2: &'a [f64],
    pub rho: f64,
    pub clip_eps: f64,
}

/// Outputs produced by the first GPU-backed Gaussian pair kernel.
#[derive(Debug, Clone, PartialEq)]
pub struct GaussianPairBatchResult {
    pub log_pdf: Vec<f64>,
    pub cond_on_first: Vec<f64>,
    pub cond_on_second: Vec<f64>,
}

#[derive(Debug, Error)]
pub enum DispatchError {
    #[error("device {0:?} is not available")]
    DeviceUnavailable(Device),
    #[error("operation {operation} is not supported on backend {backend}")]
    OperationUnsupported {
        backend: &'static str,
        operation: &'static str,
    },
    #[error("backend {backend} failed: {reason}")]
    Runtime {
        backend: &'static str,
        reason: String,
    },
}

/// Returns the currently visible acceleration capabilities.
pub fn detect_capabilities() -> ComputeCapabilities {
    ComputeCapabilities {
        cpu_simd: cpu_simd_available(),
        rayon_threads: rayon::current_num_threads(),
        cuda: detect_cuda_caps(),
        metal: detect_metal_caps(),
    }
}

/// Returns whether the requested backend appears available on this machine.
pub fn is_device_available(device: Device) -> bool {
    let caps = detect_capabilities();
    match device {
        Device::Cpu => true,
        Device::Cuda(index) => caps
            .cuda
            .is_some_and(|cuda| usize::try_from(index).is_ok_and(|idx| idx < cuda.device_count)),
        Device::Metal => caps.metal.is_some(),
    }
}

/// Executes a CPU-parallel map over `0..len`.
pub fn parallel_try_map_range_collect<T, E, F>(len: usize, f: F) -> Result<Vec<T>, E>
where
    T: Send,
    E: Send,
    F: Fn(usize) -> Result<T, E> + Sync + Send,
{
    (0..len).into_par_iter().map(f).collect()
}

/// Executes a CPU-parallel map over `0..len`.
pub fn parallel_map_range_collect<T, F>(len: usize, f: F) -> Vec<T>
where
    T: Send,
    F: Fn(usize) -> T + Sync + Send,
{
    (0..len).into_par_iter().map(f).collect()
}

/// Runs the bounded Gaussian pair batch kernel on the requested GPU backend.
///
/// CUDA uses `f64` kernels. Metal uses an `f32` shader and therefore should be
/// reserved for operations whose tolerance budget has been explicitly accepted
/// by the caller or the surrounding dispatch policy.
pub fn evaluate_gaussian_pair_batch(
    device: Device,
    request: GaussianPairBatchRequest<'_>,
) -> Result<GaussianPairBatchResult, DispatchError> {
    match device {
        Device::Cpu => Err(DispatchError::OperationUnsupported {
            backend: "cpu",
            operation: "gaussian pair batch gpu evaluation",
        }),
        Device::Cuda(ordinal) => {
            crate::cuda_backend::evaluate_gaussian_pair_batch(ordinal, request)
        }
        Device::Metal => crate::metal_backend::evaluate_gaussian_pair_batch(request),
    }
}

fn cpu_simd_available() -> bool {
    cfg!(target_feature = "avx")
        || cfg!(target_feature = "avx2")
        || cfg!(target_feature = "sse2")
        || cfg!(target_feature = "neon")
}

fn detect_cuda_caps() -> Option<CudaCaps> {
    #[cfg(feature = "cuda")]
    {
        let output = Command::new("nvidia-smi")
            .args([
                "--query-gpu=compute_cap,name",
                "--format=csv,noheader,nounits",
            ])
            .output()
            .ok()?;
        if !output.status.success() {
            return None;
        }
        let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
        let rows = stdout
            .lines()
            .filter_map(|line| {
                let mut parts = line.split(',').map(str::trim);
                let capability = parts.next()?;
                let _name = parts.next()?;
                Some(capability.to_owned())
            })
            .collect::<Vec<_>>();
        if rows.is_empty() {
            None
        } else {
            let fp64 = rows.iter().all(|capability| {
                capability
                    .split('.')
                    .next()
                    .and_then(|major| major.parse::<u32>().ok())
                    .is_some_and(|major| major >= 6)
            });
            Some(CudaCaps {
                device_count: rows.len(),
                fp64,
            })
        }
    }
    #[cfg(not(feature = "cuda"))]
    {
        None
    }
}

fn detect_metal_caps() -> Option<MetalCaps> {
    #[cfg(all(feature = "metal", target_os = "macos"))]
    {
        let devices = metal::Device::all();
        if devices.is_empty() {
            None
        } else {
            Some(MetalCaps {
                device_count: devices.len(),
                fp64: false,
            })
        }
    }
    #[cfg(not(all(feature = "metal", target_os = "macos")))]
    {
        None
    }
}
