use crate::accel::{Device, DispatchError, GaussianPairBatchRequest, GaussianPairBatchResult};

#[cfg(all(feature = "metal", target_os = "macos"))]
mod imp {
    use std::{mem::size_of, ptr::copy_nonoverlapping, sync::OnceLock};

    use metal::{
        CommandQueue, CompileOptions, ComputePipelineState, Device, MTLResourceOptions, MTLSize,
    };

    use crate::accel::{DispatchError, GaussianPairBatchRequest, GaussianPairBatchResult};

    const METAL_GAUSSIAN_PAIR_SRC: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct Params {
    float rho;
    float clip_eps;
    uint n;
};

inline float normcdf(float x) {
    float abs_x = fabs(x);
    float t = 1.0f / (1.0f + 0.2316419f * abs_x);
    float poly = t
        * (0.319381530f + t
            * (-0.356563782f + t
                * (1.781477937f + t
                    * (-1.821255978f + t * 1.330274429f))));
    float density = 0.3989422804014327f * exp(-0.5f * abs_x * abs_x);
    float cdf = 1.0f - density * poly;
    return x >= 0.0f ? cdf : 1.0f - cdf;
}

inline float norminv(float p) {
    const float a1 = -3.9696830e+01f;
    const float a2 = 2.2094610e+02f;
    const float a3 = -2.7592850e+02f;
    const float a4 = 1.3835775e+02f;
    const float a5 = -3.0664798e+01f;
    const float a6 = 2.5066283e+00f;

    const float b1 = -5.4476099e+01f;
    const float b2 = 1.6158584e+02f;
    const float b3 = -1.5569898e+02f;
    const float b4 = 6.6801312e+01f;
    const float b5 = -1.3280682e+01f;

    const float c1 = -7.7848940e-03f;
    const float c2 = -3.2239646e-01f;
    const float c3 = -2.4007583e+00f;
    const float c4 = -2.5497324e+00f;
    const float c5 = 4.3746643e+00f;
    const float c6 = 2.9381640e+00f;

    const float d1 = 7.7846958e-03f;
    const float d2 = 3.2246712e-01f;
    const float d3 = 2.4451342e+00f;
    const float d4 = 3.7544086e+00f;

    const float plow = 0.02425f;
    const float phigh = 1.0f - plow;

    if (p < plow) {
        float q = sqrt(-2.0f * log(p));
        return (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) /
            ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0f);
    }
    if (p > phigh) {
        float q = sqrt(-2.0f * log(1.0f - p));
        return -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) /
            ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0f);
    }
    float q = p - 0.5f;
    float r = q * q;
    return (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q /
        (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0f);
}

kernel void gaussian_pair_batch(
    const device float* u1 [[buffer(0)]],
    const device float* u2 [[buffer(1)]],
    device float* out_log_pdf [[buffer(2)]],
    device float* out_h12 [[buffer(3)]],
    device float* out_h21 [[buffer(4)]],
    constant Params& params [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.n) {
        return;
    }

    float x1 = clamp(u1[gid], params.clip_eps, 1.0f - params.clip_eps);
    float x2 = clamp(u2[gid], params.clip_eps, 1.0f - params.clip_eps);
    float z1 = norminv(x1);
    float z2 = norminv(x2);
    float one_minus = 1.0f - params.rho * params.rho;
    float scale = sqrt(one_minus);

    out_log_pdf[gid] = -0.5f * log(one_minus)
        - (params.rho * params.rho * (z1 * z1 + z2 * z2) - 2.0f * params.rho * z1 * z2)
            / (2.0f * one_minus);
    out_h12[gid] = clamp(normcdf((z1 - params.rho * z2) / scale), params.clip_eps, 1.0f - params.clip_eps);
    out_h21[gid] = clamp(normcdf((z2 - params.rho * z1) / scale), params.clip_eps, 1.0f - params.clip_eps);
}
"#;

    #[repr(C)]
    #[derive(Clone, Copy)]
    struct Params {
        rho: f32,
        clip_eps: f32,
        n: u32,
    }

    struct Runtime {
        device: Device,
        queue: CommandQueue,
        pipeline: ComputePipelineState,
    }

    fn runtime() -> Result<&'static Runtime, DispatchError> {
        static RUNTIME: OnceLock<Result<Runtime, String>> = OnceLock::new();
        RUNTIME
            .get_or_init(|| {
                let device = Device::system_default()
                    .ok_or_else(|| "no Metal device is available".to_owned())?;
                let options = CompileOptions::new();
                let library = device
                    .new_library_with_source(METAL_GAUSSIAN_PAIR_SRC, &options)
                    .map_err(|err| format!("failed to compile Metal shader: {err}"))?;
                let function = library
                    .get_function("gaussian_pair_batch", None)
                    .map_err(|err| format!("failed to load gaussian_pair_batch: {err}"))?;
                let pipeline = device
                    .new_compute_pipeline_state_with_function(&function)
                    .map_err(|err| format!("failed to build Metal pipeline: {err}"))?;
                let queue = device.new_command_queue();
                Ok(Runtime {
                    device,
                    queue,
                    pipeline,
                })
            })
            .as_ref()
            .map_err(|reason| DispatchError::Runtime {
                backend: "metal",
                reason: reason.clone(),
            })
    }

    pub(super) fn evaluate_gaussian_pair_batch(
        request: GaussianPairBatchRequest<'_>,
    ) -> Result<GaussianPairBatchResult, DispatchError> {
        let runtime = runtime()?;
        let n = request.u1.len();
        if n != request.u2.len() {
            return Err(DispatchError::Runtime {
                backend: "metal",
                reason: "gaussian pair batch inputs must have the same length".into(),
            });
        }

        let u1 = request
            .u1
            .iter()
            .map(|value| *value as f32)
            .collect::<Vec<_>>();
        let u2 = request
            .u2
            .iter()
            .map(|value| *value as f32)
            .collect::<Vec<_>>();

        let bytes_len = (n * size_of::<f32>()) as u64;
        let u1_buffer = runtime
            .device
            .new_buffer(bytes_len, MTLResourceOptions::StorageModeShared);
        let u2_buffer = runtime
            .device
            .new_buffer(bytes_len, MTLResourceOptions::StorageModeShared);
        let out_log_pdf = runtime
            .device
            .new_buffer(bytes_len, MTLResourceOptions::StorageModeShared);
        let out_h12 = runtime
            .device
            .new_buffer(bytes_len, MTLResourceOptions::StorageModeShared);
        let out_h21 = runtime
            .device
            .new_buffer(bytes_len, MTLResourceOptions::StorageModeShared);
        let params = Params {
            rho: request.rho as f32,
            clip_eps: request.clip_eps as f32,
            n: n as u32,
        };
        let params_buffer = runtime.device.new_buffer(
            size_of::<Params>() as u64,
            MTLResourceOptions::StorageModeShared,
        );

        unsafe {
            copy_nonoverlapping(
                u1.as_ptr() as *const u8,
                u1_buffer.contents() as *mut u8,
                n * size_of::<f32>(),
            );
            copy_nonoverlapping(
                u2.as_ptr() as *const u8,
                u2_buffer.contents() as *mut u8,
                n * size_of::<f32>(),
            );
            copy_nonoverlapping(
                (&params as *const Params).cast::<u8>(),
                params_buffer.contents() as *mut u8,
                size_of::<Params>(),
            );
        }

        let command_buffer = runtime.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&runtime.pipeline);
        encoder.set_buffer(0, Some(&u1_buffer), 0);
        encoder.set_buffer(1, Some(&u2_buffer), 0);
        encoder.set_buffer(2, Some(&out_log_pdf), 0);
        encoder.set_buffer(3, Some(&out_h12), 0);
        encoder.set_buffer(4, Some(&out_h21), 0);
        encoder.set_buffer(5, Some(&params_buffer), 0);

        let width = runtime.pipeline.thread_execution_width().max(1);
        let threads_per_group = MTLSize::new(width, 1, 1);
        let thread_groups = MTLSize::new((n as u64).div_ceil(width), 1, 1);
        encoder.dispatch_thread_groups(thread_groups, threads_per_group);
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        let status = command_buffer.status();
        if status != metal::MTLCommandBufferStatus::Completed {
            return Err(DispatchError::Runtime {
                backend: "metal",
                reason: format!("command buffer completed with status {status:?}"),
            });
        }

        let read_back = |buffer: &metal::Buffer| -> Vec<f64> {
            unsafe {
                std::slice::from_raw_parts(buffer.contents() as *const f32, n)
                    .iter()
                    .map(|value| f64::from(*value))
                    .collect()
            }
        };

        Ok(GaussianPairBatchResult {
            log_pdf: read_back(&out_log_pdf),
            cond_on_first: read_back(&out_h12),
            cond_on_second: read_back(&out_h21),
        })
    }
}

#[cfg(not(all(feature = "metal", target_os = "macos")))]
mod imp {
    use crate::accel::{DispatchError, GaussianPairBatchRequest, GaussianPairBatchResult};

    pub(super) fn evaluate_gaussian_pair_batch(
        _request: GaussianPairBatchRequest<'_>,
    ) -> Result<GaussianPairBatchResult, DispatchError> {
        Err(DispatchError::OperationUnsupported {
            backend: "metal",
            operation: "gaussian pair batch gpu evaluation",
        })
    }
}

pub(crate) fn evaluate_gaussian_pair_batch(
    request: GaussianPairBatchRequest<'_>,
) -> Result<GaussianPairBatchResult, DispatchError> {
    if !crate::accel::is_device_available(Device::Metal) {
        return Err(DispatchError::DeviceUnavailable(Device::Metal));
    }
    imp::evaluate_gaussian_pair_batch(request)
}
