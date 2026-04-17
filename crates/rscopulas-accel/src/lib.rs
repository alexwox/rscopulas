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
    pub fp64: bool,
}

#[derive(Debug, Error)]
pub enum DispatchError {
    #[error("device {0:?} is not available")]
    DeviceUnavailable(Device),
}

pub fn detect_capabilities() -> ComputeCapabilities {
    ComputeCapabilities {
        cpu_simd: true,
        rayon_threads: std::thread::available_parallelism()
            .map(usize::from)
            .unwrap_or(1),
        cuda: None,
        metal: None,
    }
}
