//! Normal-score evaluation of vines whose edges are all Gaussian (or
//! independence) copulas.
//!
//! The generic evaluator works on probabilities: every edge maps its two
//! inputs to normal scores with `Φ⁻¹`, combines them, and maps the
//! conditional back with `Φ` — only for the next tree to apply `Φ⁻¹` again.
//! For a Gaussian edge that round trip is the identity, because both
//! h-functions are linear in the scores:
//!
//! ```text
//! Φ⁻¹(h_{2|1}(u₂ | u₁)) = (z₂ − ρ z₁) / √(1 − ρ²).
//! ```
//!
//! So when every edge is Gaussian the whole recursion can stay on the score
//! scale: `d` quantile evaluations on the way in, at most `d` CDF evaluations
//! on the way out, and nothing but multiply-adds for the `d(d−1)/2` edges in
//! between. Besides being an order of magnitude faster, this avoids the
//! precision lost by storing a conditional as a probability near 1.
//!
//! Clipping is preserved: where the generic path clamps a conditional
//! probability to `[clip, 1 − clip]`, this path clamps its score to
//! `[Φ⁻¹(clip), Φ⁻¹(1 − clip)]`.

use crate::{
    paircopula::{PairCopulaFamily, PairCopulaParams, PairCopulaSpec, Rotation},
    special::{norm_cdf, norm_ppf},
};

use super::{CompiledEvalStep, CompiledSampleStep, CompiledVineRuntime};

/// One edge of the recursion with its Gaussian constants precomputed.
struct Step {
    row: usize,
    col: usize,
    label: usize,
    source_from_direct: bool,
    write_indirect: bool,
    rho: f64,
    /// `1 / √(1 − ρ²)`.
    inv_scale: f64,
    /// `√(1 − ρ²)`.
    scale: f64,
    /// `−½ ln(1 − ρ²)`.
    log_norm: f64,
    /// `1 / (2 (1 − ρ²))`.
    half_inv_var: f64,
}

impl Step {
    fn new(
        (row, col, label): (usize, usize, usize),
        source_from_direct: bool,
        write_indirect: bool,
        spec: &PairCopulaSpec,
    ) -> Option<Self> {
        let rho = match (spec.family, spec.rotation, &spec.params) {
            (PairCopulaFamily::Gaussian, Rotation::R0, PairCopulaParams::One(rho))
                if rho.is_finite() && rho.abs() < 1.0 =>
            {
                *rho
            }
            (PairCopulaFamily::Independence, _, PairCopulaParams::None) => 0.0,
            _ => return None,
        };
        let one_minus = 1.0 - rho * rho;
        let scale = one_minus.sqrt();
        Some(Self {
            row,
            col,
            label,
            source_from_direct,
            write_indirect,
            rho,
            inv_scale: 1.0 / scale,
            scale,
            log_norm: -0.5 * one_minus.ln(),
            half_inv_var: 1.0 / (2.0 * one_minus),
        })
    }
}

/// Score-scale recursion for an all-Gaussian vine.
pub(super) struct GaussianChain<'a> {
    dim: usize,
    variable_order: &'a [usize],
    eval_steps: Vec<Step>,
    sample_steps: Vec<Step>,
}

/// Per-task workspace: the `d × d` direct and indirect score tables of one
/// observation, and its clipped inputs.
pub(super) struct Workspace {
    direct: Vec<f64>,
    indirect: Vec<f64>,
    inputs: Vec<f64>,
}

impl<'a> GaussianChain<'a> {
    /// `None` unless every edge is an unrotated Gaussian or an independence
    /// copula.
    pub(super) fn new(runtime: &'a CompiledVineRuntime) -> Option<Self> {
        let eval_steps = runtime
            .eval_steps
            .iter()
            .map(|step: &CompiledEvalStep| {
                Step::new(
                    (step.row, step.col, step.label),
                    step.source_from_direct,
                    step.write_indirect,
                    &step.spec,
                )
            })
            .collect::<Option<Vec<_>>>()?;
        let sample_steps = runtime
            .sample_steps
            .iter()
            .map(|step: &CompiledSampleStep| {
                Step::new(
                    (step.row, step.col, step.label),
                    step.source_from_direct,
                    step.write_indirect,
                    &step.spec,
                )
            })
            .collect::<Option<Vec<_>>>()?;
        Some(Self {
            dim: runtime.dim,
            variable_order: &runtime.variable_order,
            eval_steps,
            sample_steps,
        })
    }

    pub(super) fn workspace(&self) -> Workspace {
        Workspace {
            direct: vec![0.0; self.dim * self.dim],
            indirect: vec![0.0; self.dim * self.dim],
            inputs: vec![0.0; self.dim],
        }
    }

    /// Clips the observation (given in variable-label order through `value`)
    /// and stores its scores at `direct[slot(idx)]` for every position `idx`
    /// of the variable order.
    #[inline]
    fn load(
        &self,
        ws: &mut Workspace,
        value: impl Fn(usize) -> f64,
        clip_eps: f64,
        slot: impl Fn(usize) -> usize,
    ) {
        let (low, high) = (clip_eps, 1.0 - clip_eps);
        for (idx, &var) in self.variable_order.iter().enumerate() {
            let u = value(var).clamp(low, high);
            ws.inputs[idx] = u;
            ws.direct[slot(idx)] = norm_ppf(u);
        }
    }

    /// Runs the forward (evaluation) recursion over the steps whose column is
    /// below `col_limit`, returning the summed log-density of those edges.
    #[inline]
    fn forward(&self, ws: &mut Workspace, col_limit: usize, bounds: (f64, f64)) -> f64 {
        let d = self.dim;
        ws.indirect[0] = ws.direct[0];
        let mut total = 0.0;
        for step in &self.eval_steps {
            if step.col >= col_limit {
                break;
            }
            let source_index = step.row * d + step.label;
            let source = if step.source_from_direct {
                ws.direct[source_index]
            } else {
                ws.indirect[source_index]
            };
            let target = ws.direct[step.row * d + step.col];
            let rho = step.rho;
            total += step.log_norm
                - (rho * rho * (source * source + target * target) - 2.0 * rho * source * target)
                    * step.half_inv_var;
            ws.direct[(step.row + 1) * d + step.col] =
                ((target - rho * source) * step.inv_scale).clamp(bounds.0, bounds.1);
            if step.write_indirect {
                ws.indirect[(step.row + 1) * d + step.col] =
                    ((source - rho * target) * step.inv_scale).clamp(bounds.0, bounds.1);
            }
        }
        total
    }

    /// Log-density of one observation.
    #[inline]
    pub(super) fn log_pdf(
        &self,
        ws: &mut Workspace,
        value: impl Fn(usize) -> f64,
        clip_eps: f64,
        bounds: (f64, f64),
    ) -> f64 {
        self.load(ws, value, clip_eps, |idx| idx);
        self.forward(ws, self.dim, bounds)
    }

    /// Rosenblatt transform of one observation, restricted to the first
    /// `out.len()` positions of the variable order; `out` is indexed by
    /// position.
    #[inline]
    pub(super) fn rosenblatt(
        &self,
        ws: &mut Workspace,
        value: impl Fn(usize) -> f64,
        clip_eps: f64,
        bounds: (f64, f64),
        out: &mut [f64],
    ) {
        self.load(ws, value, clip_eps, |idx| idx);
        self.forward(ws, out.len(), bounds);
        let d = self.dim;
        // The anchor passes through unchanged; every other position is the
        // conditional accumulated on the diagonal.
        out[0] = ws.inputs[0];
        for (idx, slot) in out.iter_mut().enumerate().skip(1) {
            *slot = norm_cdf(ws.direct[idx * d + idx]).clamp(clip_eps, 1.0 - clip_eps);
        }
    }

    /// Inverse Rosenblatt transform of one observation; `out` is indexed by
    /// variable label.
    #[inline]
    pub(super) fn inverse_rosenblatt(
        &self,
        ws: &mut Workspace,
        value: impl Fn(usize) -> f64,
        clip_eps: f64,
        bounds: (f64, f64),
        out: &mut [f64],
    ) {
        let d = self.dim;
        self.load(ws, value, clip_eps, |idx| idx * d + idx);
        ws.indirect[0] = ws.direct[0];
        for step in &self.sample_steps {
            let source_index = step.row * d + step.label;
            let source = if step.source_from_direct {
                ws.direct[source_index]
            } else {
                ws.indirect[source_index]
            };
            let target = ws.direct[(step.row + 1) * d + step.col];
            let rho = step.rho;
            let solved = (step.scale * target + rho * source).clamp(bounds.0, bounds.1);
            ws.direct[step.row * d + step.col] = solved;
            if step.write_indirect {
                ws.indirect[(step.row + 1) * d + step.col] =
                    ((source - rho * solved) * step.inv_scale).clamp(bounds.0, bounds.1);
            }
        }
        out[self.variable_order[0]] = ws.inputs[0];
        for (idx, &var) in self.variable_order.iter().enumerate().skip(1) {
            out[var] = norm_cdf(ws.direct[idx]).clamp(clip_eps, 1.0 - clip_eps);
        }
    }
}

/// Score-scale images of the probability clip bounds.
pub(super) fn score_bounds(clip_eps: f64) -> (f64, f64) {
    (norm_ppf(clip_eps), norm_ppf(1.0 - clip_eps))
}
