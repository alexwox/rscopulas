#[cfg(feature = "cuda")]
use std::process::Command;

use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Device {
    Cpu,
    Cuda(u32),
    Metal,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComputeCapabilities {
    pub cpu_simd: bool,
    pub rayon_threads: usize,
    pub cuda: Option<CudaCaps>,
    pub metal: Option<MetalCaps>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CudaCaps {
    pub device_count: usize,
    pub fp64: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetalCaps {
    pub device_count: usize,
    pub fp64: bool,
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
}

pub fn detect_capabilities() -> ComputeCapabilities {
    ComputeCapabilities {
        cpu_simd: cpu_simd_available(),
        rayon_threads: rayon::current_num_threads(),
        cuda: detect_cuda_caps(),
        metal: detect_metal_caps(),
    }
}

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

pub fn parallel_try_map_range_collect<T, E, F>(len: usize, f: F) -> Result<Vec<T>, E>
where
    T: Send,
    E: Send,
    F: Fn(usize) -> Result<T, E> + Sync + Send,
{
    (0..len).into_par_iter().map(f).collect()
}

pub fn parallel_map_range_collect<T, F>(len: usize, f: F) -> Vec<T>
where
    T: Send,
    F: Fn(usize) -> T + Sync + Send,
{
    (0..len).into_par_iter().map(f).collect()
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
            .args(["--query-gpu=name", "--format=csv,noheader"])
            .output()
            .ok()?;
        if !output.status.success() {
            return None;
        }
        let count = String::from_utf8_lossy(&output.stdout)
            .lines()
            .filter(|line| !line.trim().is_empty())
            .count();
        if count == 0 {
            None
        } else {
            Some(CudaCaps {
                device_count: count,
                fp64: true,
            })
        }
    }
    #[cfg(not(feature = "cuda"))]
    {
        None
    }
}

fn detect_metal_caps() -> Option<MetalCaps> {
    #[cfg(feature = "metal")]
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
    #[cfg(not(feature = "metal"))]
    {
        None
    }
}
