use crate::{Device, DispatchError, GaussianPairBatchRequest, GaussianPairBatchResult};

#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
mod imp {
    use std::sync::OnceLock;

    use cudarc::{
        driver::{CudaContext, LaunchConfig},
        nvrtc::compile_ptx,
    };

    use crate::{DispatchError, GaussianPairBatchRequest, GaussianPairBatchResult};

    const CUDA_GAUSSIAN_PAIR_SRC: &str = r#"
extern "C" __device__ double normcdf(double x) {
    return 0.5 * erfc(-x * 0.70710678118654752440);
}

extern "C" __device__ double norminv(double p) {
    const double a1 = -3.969683028665376e+01;
    const double a2 = 2.209460984245205e+02;
    const double a3 = -2.759285104469687e+02;
    const double a4 = 1.383577518672690e+02;
    const double a5 = -3.066479806614716e+01;
    const double a6 = 2.506628277459239e+00;

    const double b1 = -5.447609879822406e+01;
    const double b2 = 1.615858368580409e+02;
    const double b3 = -1.556989798598866e+02;
    const double b4 = 6.680131188771972e+01;
    const double b5 = -1.328068155288572e+01;

    const double c1 = -7.784894002430293e-03;
    const double c2 = -3.223964580411365e-01;
    const double c3 = -2.400758277161838e+00;
    const double c4 = -2.549732539343734e+00;
    const double c5 = 4.374664141464968e+00;
    const double c6 = 2.938163982698783e+00;

    const double d1 = 7.784695709041462e-03;
    const double d2 = 3.224671290700398e-01;
    const double d3 = 2.445134137142996e+00;
    const double d4 = 3.754408661907416e+00;

    const double plow = 0.02425;
    const double phigh = 1.0 - plow;

    if (p < plow) {
        double q = sqrt(-2.0 * log(p));
        return (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) /
            ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0);
    }
    if (p > phigh) {
        double q = sqrt(-2.0 * log(1.0 - p));
        return -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) /
            ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0);
    }
    double q = p - 0.5;
    double r = q * q;
    return (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q /
        (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0);
}

extern "C" __global__ void gaussian_pair_batch(
    const double* u1,
    const double* u2,
    double* out_log_pdf,
    double* out_h12,
    double* out_h21,
    double rho,
    double clip_eps,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }

    double x1 = fmin(fmax(u1[idx], clip_eps), 1.0 - clip_eps);
    double x2 = fmin(fmax(u2[idx], clip_eps), 1.0 - clip_eps);
    double z1 = norminv(x1);
    double z2 = norminv(x2);
    double one_minus = 1.0 - rho * rho;
    double scale = sqrt(one_minus);

    out_log_pdf[idx] = -0.5 * log(one_minus)
        - (rho * rho * (z1 * z1 + z2 * z2) - 2.0 * rho * z1 * z2) / (2.0 * one_minus);
    out_h12[idx] = normcdf((z1 - rho * z2) / scale);
    out_h21[idx] = normcdf((z2 - rho * z1) / scale);
}
"#;

    fn gaussian_pair_ptx() -> Result<&'static cudarc::nvrtc::Ptx, DispatchError> {
        static PTX: OnceLock<Result<cudarc::nvrtc::Ptx, String>> = OnceLock::new();
        PTX.get_or_init(|| {
            compile_ptx(CUDA_GAUSSIAN_PAIR_SRC)
                .map_err(|err| format!("failed to compile CUDA gaussian pair kernel: {err}"))
        })
        .as_ref()
        .map_err(|reason| DispatchError::Runtime {
            backend: "cuda",
            reason: reason.clone(),
        })
    }

    pub(super) fn evaluate_gaussian_pair_batch(
        ordinal: u32,
        request: GaussianPairBatchRequest<'_>,
    ) -> Result<GaussianPairBatchResult, DispatchError> {
        let n = request.u1.len();
        if n != request.u2.len() {
            return Err(DispatchError::Runtime {
                backend: "cuda",
                reason: "gaussian pair batch inputs must have the same length".into(),
            });
        }

        let ctx = CudaContext::new(ordinal as usize).map_err(|err| DispatchError::Runtime {
            backend: "cuda",
            reason: format!("failed to create CUDA context: {err}"),
        })?;
        let stream = ctx.default_stream();
        let module = ctx
            .load_module(gaussian_pair_ptx()?.clone())
            .map_err(|err| DispatchError::Runtime {
                backend: "cuda",
                reason: format!("failed to load CUDA module: {err}"),
            })?;
        let function =
            module
                .load_function("gaussian_pair_batch")
                .map_err(|err| DispatchError::Runtime {
                    backend: "cuda",
                    reason: format!("failed to load gaussian_pair_batch: {err}"),
                })?;

        let u1 = stream
            .clone_htod(request.u1)
            .map_err(|err| DispatchError::Runtime {
                backend: "cuda",
                reason: format!("failed to upload u1: {err}"),
            })?;
        let u2 = stream
            .clone_htod(request.u2)
            .map_err(|err| DispatchError::Runtime {
                backend: "cuda",
                reason: format!("failed to upload u2: {err}"),
            })?;
        let mut out_log_pdf =
            stream
                .alloc_zeros::<f64>(n)
                .map_err(|err| DispatchError::Runtime {
                    backend: "cuda",
                    reason: format!("failed to allocate log_pdf output: {err}"),
                })?;
        let mut out_h12 = stream
            .alloc_zeros::<f64>(n)
            .map_err(|err| DispatchError::Runtime {
                backend: "cuda",
                reason: format!("failed to allocate h12 output: {err}"),
            })?;
        let mut out_h21 = stream
            .alloc_zeros::<f64>(n)
            .map_err(|err| DispatchError::Runtime {
                backend: "cuda",
                reason: format!("failed to allocate h21 output: {err}"),
            })?;

        let mut builder = stream.launch_builder(&function);
        builder.arg(&u1);
        builder.arg(&u2);
        builder.arg(&mut out_log_pdf);
        builder.arg(&mut out_h12);
        builder.arg(&mut out_h21);
        builder.arg(&request.rho);
        builder.arg(&request.clip_eps);
        builder.arg(&(n as i32));
        unsafe { builder.launch(LaunchConfig::for_num_elems(n as u32)) }.map_err(|err| {
            DispatchError::Runtime {
                backend: "cuda",
                reason: format!("failed to launch gaussian_pair_batch: {err}"),
            }
        })?;

        let log_pdf = stream
            .clone_dtoh(&out_log_pdf)
            .map_err(|err| DispatchError::Runtime {
                backend: "cuda",
                reason: format!("failed to download log_pdf: {err}"),
            })?;
        let cond_on_first = stream
            .clone_dtoh(&out_h12)
            .map_err(|err| DispatchError::Runtime {
                backend: "cuda",
                reason: format!("failed to download h12: {err}"),
            })?;
        let cond_on_second = stream
            .clone_dtoh(&out_h21)
            .map_err(|err| DispatchError::Runtime {
                backend: "cuda",
                reason: format!("failed to download h21: {err}"),
            })?;

        Ok(GaussianPairBatchResult {
            log_pdf,
            cond_on_first,
            cond_on_second,
        })
    }
}

#[cfg(not(all(feature = "cuda", any(target_os = "linux", target_os = "windows"))))]
mod imp {
    use crate::{DispatchError, GaussianPairBatchRequest, GaussianPairBatchResult};

    pub(super) fn evaluate_gaussian_pair_batch(
        _ordinal: u32,
        _request: GaussianPairBatchRequest<'_>,
    ) -> Result<GaussianPairBatchResult, DispatchError> {
        Err(DispatchError::OperationUnsupported {
            backend: "cuda",
            operation: "gaussian pair batch gpu evaluation",
        })
    }
}

pub(crate) fn evaluate_gaussian_pair_batch(
    ordinal: u32,
    request: GaussianPairBatchRequest<'_>,
) -> Result<GaussianPairBatchResult, DispatchError> {
    if !crate::is_device_available(Device::Cuda(ordinal)) {
        return Err(DispatchError::DeviceUnavailable(Device::Cuda(ordinal)));
    }
    imp::evaluate_gaussian_pair_batch(ordinal, request)
}
