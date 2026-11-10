use ndarray::Array2;

use crate::errors::NumericalError;

/// log(1 - exp(x)) for x <= 0, including the exact endpoint x = -infinity.
pub(crate) fn log1mexp(x: f64) -> f64 {
    if x < -std::f64::consts::LN_2 {
        (-x.exp()).ln_1p()
    } else {
        (-x.exp_m1()).ln()
    }
}

pub(crate) fn logaddexp(a: f64, b: f64) -> f64 {
    let high = a.max(b);
    if !high.is_finite() {
        return high;
    }
    high + ((a.min(b) - high).exp()).ln_1p()
}

pub(crate) fn softplus(x: f64) -> f64 {
    x.max(0.0) + (-x.abs()).exp().ln_1p()
}

/// log(1 - exp(-exp(log_x))), retaining values below the exp underflow limit.
pub(crate) fn log1mexp_neg_exp(log_x: f64) -> f64 {
    if log_x < -36.0 {
        log_x
    } else {
        log1mexp(-log_x.exp())
    }
}

/// log(-log(1 - exp(x))) without underflow in the upper copula tail.
pub(crate) fn log_neg_log1mexp(x: f64) -> f64 {
    if x < -36.0 { x } else { (-log1mexp(x)).ln() }
}

pub fn identity_correlation(dim: usize) -> Array2<f64> {
    Array2::eye(dim)
}

pub fn make_spd_correlation(matrix: &Array2<f64>) -> Result<Array2<f64>, NumericalError> {
    if matrix.nrows() != matrix.ncols() {
        return Err(NumericalError::InvalidCorrelationMatrix);
    }

    let dim = matrix.nrows();
    let mut sym = Array2::zeros((dim, dim));
    for row in 0..dim {
        sym[(row, row)] = 1.0;
        for col in (row + 1)..dim {
            let value = 0.5 * (matrix[(row, col)] + matrix[(col, row)]);
            sym[(row, col)] = value;
            sym[(col, row)] = value;
        }
    }

    if validate_correlation_matrix(&sym).is_ok() {
        return Ok(sym);
    }

    for step in 0..500 {
        let shrink = 1.0 - (step as f64 + 1.0) / 600.0;
        let mut candidate = Array2::eye(dim);
        for row in 0..dim {
            for col in (row + 1)..dim {
                let value = sym[(row, col)] * shrink;
                candidate[(row, col)] = value;
                candidate[(col, row)] = value;
            }
        }

        if validate_correlation_matrix(&candidate).is_ok() {
            return Ok(candidate);
        }
    }

    Err(NumericalError::InvalidCorrelationMatrix)
}

pub fn validate_correlation_matrix(matrix: &Array2<f64>) -> Result<(), NumericalError> {
    if matrix.nrows() != matrix.ncols() {
        return Err(NumericalError::InvalidCorrelationMatrix);
    }

    for i in 0..matrix.nrows() {
        if (matrix[(i, i)] - 1.0).abs() > 1e-12 {
            return Err(NumericalError::InvalidCorrelationMatrix);
        }

        for j in 0..matrix.ncols() {
            let value = matrix[(i, j)];
            if !value.is_finite() {
                return Err(NumericalError::InvalidCorrelationMatrix);
            }

            if (value - matrix[(j, i)]).abs() > 1e-12 {
                return Err(NumericalError::InvalidCorrelationMatrix);
            }
        }
    }

    cholesky(matrix)?;
    Ok(())
}

pub fn cholesky(matrix: &Array2<f64>) -> Result<Array2<f64>, NumericalError> {
    if matrix.nrows() != matrix.ncols() {
        return Err(NumericalError::InvalidCorrelationMatrix);
    }

    let dim = matrix.nrows();
    let mut lower = Array2::zeros((dim, dim));

    for i in 0..dim {
        for j in 0..=i {
            let mut value = matrix[(i, j)];
            for k in 0..j {
                value -= lower[(i, k)] * lower[(j, k)];
            }

            if i == j {
                if value <= 0.0 {
                    return Err(NumericalError::DecompositionFailed);
                }
                lower[(i, j)] = value.sqrt();
            } else {
                let pivot = lower[(j, j)];
                if pivot == 0.0 {
                    return Err(NumericalError::DecompositionFailed);
                }
                lower[(i, j)] = value / pivot;
            }
        }
    }

    Ok(lower)
}

pub fn log_determinant_from_cholesky(lower: &Array2<f64>) -> f64 {
    2.0 * (0..lower.nrows()).map(|i| lower[(i, i)].ln()).sum::<f64>()
}

pub fn solve_lower_triangular(
    lower: &Array2<f64>,
    rhs: &[f64],
) -> Result<Vec<f64>, NumericalError> {
    if lower.nrows() != lower.ncols() || lower.nrows() != rhs.len() {
        return Err(NumericalError::InvalidCorrelationMatrix);
    }

    let mut solution = vec![0.0; rhs.len()];
    for i in 0..rhs.len() {
        let mut value = rhs[i];
        for j in 0..i {
            value -= lower[(i, j)] * solution[j];
        }

        let pivot = lower[(i, i)];
        if pivot == 0.0 {
            return Err(NumericalError::DecompositionFailed);
        }
        solution[i] = value / pivot;
    }

    Ok(solution)
}

pub fn quadratic_form_from_cholesky(
    lower: &Array2<f64>,
    vector: &[f64],
) -> Result<f64, NumericalError> {
    let whitened = solve_lower_triangular(lower, vector)?;
    Ok(whitened.iter().map(|value| value * value).sum())
}

pub fn inverse(matrix: &Array2<f64>) -> Result<Array2<f64>, NumericalError> {
    if matrix.nrows() != matrix.ncols() {
        return Err(NumericalError::InvalidCorrelationMatrix);
    }

    let dim = matrix.nrows();
    let mut augmented = Array2::zeros((dim, 2 * dim));
    for row in 0..dim {
        for col in 0..dim {
            augmented[(row, col)] = matrix[(row, col)];
        }
        augmented[(row, dim + row)] = 1.0;
    }

    for pivot in 0..dim {
        let mut max_row = pivot;
        let mut max_value = augmented[(pivot, pivot)].abs();
        for row in (pivot + 1)..dim {
            let value = augmented[(row, pivot)].abs();
            if value > max_value {
                max_value = value;
                max_row = row;
            }
        }

        if max_value < 1e-14 {
            return Err(NumericalError::DecompositionFailed);
        }

        if max_row != pivot {
            for col in 0..(2 * dim) {
                let tmp = augmented[(pivot, col)];
                augmented[(pivot, col)] = augmented[(max_row, col)];
                augmented[(max_row, col)] = tmp;
            }
        }

        let pivot_value = augmented[(pivot, pivot)];
        for col in 0..(2 * dim) {
            augmented[(pivot, col)] /= pivot_value;
        }

        for row in 0..dim {
            if row == pivot {
                continue;
            }
            let factor = augmented[(row, pivot)];
            for col in 0..(2 * dim) {
                augmented[(row, col)] -= factor * augmented[(pivot, col)];
            }
        }
    }

    let mut result = Array2::zeros((dim, dim));
    for row in 0..dim {
        for col in 0..dim {
            result[(row, col)] = augmented[(row, dim + col)];
        }
    }

    Ok(result)
}

/// Returns the (nodes, weights) for an `n`-point Gauss–Legendre quadrature
/// rule on `[0, 1]`.
///
/// The rule is derived on the reference interval `[-1, 1]` and linearly
/// rescaled; weights absorb the Jacobian (factor of 1/2). The reference-
/// interval roots are found by Newton iteration on the `n`-th Legendre
/// polynomial, started from Tricomi's asymptotic approximation.
///
/// Integrates smooth functions on the unit interval. Accuracy depends on the
/// integrand and node count; callers must check convergence for narrow peaks.
pub fn gauss_legendre_01(n: usize) -> (Vec<f64>, Vec<f64>) {
    debug_assert!(n >= 1, "gauss_legendre_01 requires at least one node");

    let nf = n as f64;
    let mut nodes = vec![0.0; n];
    let mut weights = vec![0.0; n];

    // Legendre polynomial roots are symmetric about 0; we compute the (n+1)/2
    // non-positive ones and mirror.
    let half = n.div_ceil(2);
    for i in 0..half {
        // Tricomi's initial guess for the i-th (1-indexed) root on [-1, 1].
        let theta = std::f64::consts::PI * ((i + 1) as f64 - 0.25) / (nf + 0.5);
        let mut x = (1.0 - (nf - 1.0) / (8.0 * nf.powi(3))) * theta.cos();

        // Newton iteration: x ← x - P_n(x) / P_n'(x).
        let mut pp;
        loop {
            let (p, p_prime) = legendre_p_and_p_prime(n, x);
            pp = p_prime;
            let dx = p / p_prime;
            x -= dx;
            if dx.abs() < 1e-15 {
                break;
            }
        }

        // Map [-1, 1] → [0, 1]: node' = (1 - x)/2, weight' = 1 / ((1 - x²) P_n'(x)²)
        // (the standard weight 2 / ((1-x²) P'²) picks up a factor of 1/2 from the
        // Jacobian of the linear map). We emit ascending nodes in [0, 1], so the
        // root closest to -1 (Tricomi's first guess) lands at the smallest x₀₁.
        let node_01 = 0.5 * (1.0 - x);
        let weight_01 = 1.0 / ((1.0 - x * x) * pp * pp);

        nodes[i] = node_01;
        weights[i] = weight_01;

        // Mirror root at -x (which maps to 1 - node_01); this is the symmetric
        // partner in [0, 1]. Skip if n is odd and x == 0 (the middle root).
        let mirror = n - 1 - i;
        if mirror != i {
            nodes[mirror] = 1.0 - node_01;
            weights[mirror] = weight_01;
        }
    }

    (nodes, weights)
}

/// Evaluates `(P_n(x), P_n'(x))` via the three-term recurrence. Used by the
/// Gauss–Legendre root finder above.
fn legendre_p_and_p_prime(n: usize, x: f64) -> (f64, f64) {
    // Recurrence: P_0 = 1, P_1 = x, k P_k = (2k-1) x P_{k-1} - (k-1) P_{k-2}.
    let mut p_prev = 1.0;
    let mut p_cur = x;
    if n == 0 {
        return (1.0, 0.0);
    }
    for k in 2..=n {
        let k_f = k as f64;
        let p_next = ((2.0 * k_f - 1.0) * x * p_cur - (k_f - 1.0) * p_prev) / k_f;
        p_prev = p_cur;
        p_cur = p_next;
    }
    // P_n'(x) = n (x P_n - P_{n-1}) / (x² - 1) away from endpoints.
    let denom = x * x - 1.0;
    let p_prime = n as f64 * (x * p_cur - p_prev) / denom;
    (p_cur, p_prime)
}

pub fn maximize_scalar<F>(low: f64, high: f64, iterations: usize, f: F) -> f64
where
    F: FnMut(f64) -> f64,
{
    maximize_scalar_with_diagnostics(low, high, iterations, f).0
}

pub(crate) fn maximize_scalar_with_diagnostics<F>(
    mut low: f64,
    mut high: f64,
    iterations: usize,
    mut f: F,
) -> (f64, usize, bool)
where
    F: FnMut(f64) -> f64,
{
    let phi = (1.0 + 5.0_f64.sqrt()) / 2.0;
    let resphi = 2.0 - phi;
    let mut x1 = low + resphi * (high - low);
    let mut x2 = high - resphi * (high - low);
    let mut f1 = f(x1);
    let mut f2 = f(x2);

    let mut completed = 0;
    for _ in 0..iterations {
        if (high - low).abs() <= 1e-10 * (1.0 + low.abs().max(high.abs())) {
            break;
        }
        completed += 1;
        if f1 < f2 {
            low = x1;
            x1 = x2;
            f1 = f2;
            x2 = high - resphi * (high - low);
            f2 = f(x2);
        } else {
            high = x2;
            x2 = x1;
            f2 = f1;
            x1 = low + resphi * (high - low);
            f1 = f(x1);
        }
    }

    (
        if f1 > f2 { x1 } else { x2 },
        completed,
        f1.max(f2).is_finite() && (high - low).abs() <= 1e-10 * (1.0 + low.abs().max(high.abs())),
    )
}

/// Root of a continuous `f` on a bracket `[a, b]` with `f(a)` and `f(b)` of
/// opposite sign, by Chandrupatla's method (inverse quadratic interpolation
/// safeguarded by bisection).
///
/// `fa` and `fb` are the already-evaluated end values. The search stops once
/// the bracket is narrower than `2 (abs_tol + rel_tol |x|)` or after
/// `max_iter` evaluations, and returns the end of the final bracket with the
/// smaller residual. Smooth functions converge superlinearly — typically
/// 6–12 evaluations where bisection to the same tolerance needs 40–60.
pub(crate) fn find_root_bracketed<F>(
    bracket: (f64, f64),
    values: (f64, f64),
    abs_tol: f64,
    rel_tol: f64,
    max_iter: usize,
    mut f: F,
) -> f64
where
    F: FnMut(f64) -> f64,
{
    let (mut a, mut b) = bracket;
    let (mut fa, mut fb) = values;
    if fa == 0.0 {
        return a;
    }
    if fb == 0.0 {
        return b;
    }
    let (mut c, mut fc);
    let mut t = 0.5;
    for _ in 0..max_iter {
        let xt = a + t * (b - a);
        let ft = f(xt);
        if (ft < 0.0) == (fa < 0.0) {
            c = a;
            fc = fa;
        } else {
            c = b;
            fc = fb;
            b = a;
            fb = fa;
        }
        a = xt;
        fa = ft;
        let (xm, fm) = if fa.abs() < fb.abs() {
            (a, fa)
        } else {
            (b, fb)
        };
        if fm == 0.0 {
            return xm;
        }
        let tol = 2.0 * f64::EPSILON * xm.abs() + abs_tol + rel_tol * xm.abs();
        let t_lim = tol / (b - c).abs();
        if t_lim > 0.5 {
            return xm;
        }
        let xi = (a - b) / (c - b);
        let phi = (fa - fb) / (fc - fb);
        t = if phi * phi < xi && (1.0 - phi) * (1.0 - phi) < 1.0 - xi {
            fa / (fb - fa) * fc / (fb - fc) + (c - a) / (b - a) * fa / (fc - fa) * fb / (fc - fb)
        } else {
            0.5
        };
        t = t.clamp(t_lim, 1.0 - t_lim);
    }
    if fa.abs() < fb.abs() { a } else { b }
}

/// Coordinate-ascent maximisation of `f: Rᵖ → R` using the crate's golden-
/// section step along each axis. One "cycle" sweeps every coordinate once.
///
/// This is the multi-dim polish used by [`FactorCopula::fit`] after its
/// sequential/EM warm start. We deliberately avoid a full quasi-Newton scheme:
/// the inputs arrive close to the optimum, and the factor log-likelihood is
/// near-separable across links once Joe-style bounded-to-unconstrained
/// reparametrisations are applied, so coordinate ascent reaches a stationary
/// point in a handful of cycles without any curvature safeguarding.
///
/// The sweep stops early when a full cycle improves the objective by less than
/// `rel_tol` (measured relative to `|f|`, or absolutely when `f` is near zero).
/// If `x0` is empty or `brackets` is empty the starting point is returned as-is.
pub fn coord_ascent_maximise<F>(
    x0: &[f64],
    brackets: &[(f64, f64)],
    rel_tol: f64,
    max_cycles: usize,
    iters_per_coord: usize,
    f: F,
) -> (Vec<f64>, f64)
where
    F: Fn(&[f64]) -> f64,
{
    let (x, value, _, _) =
        coord_ascent_with_diagnostics(x0, brackets, rel_tol, max_cycles, iters_per_coord, f);
    (x, value)
}

/// Line search used along each coordinate of a coordinate ascent.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum CoordinateSearch {
    /// Golden section on the full bracket, at most this many iterations
    /// (the bracket is shrunk to 1e-10 relative).
    Golden(usize),
    /// Brent's method on the full bracket, seeded at the current value and
    /// stopped at this absolute tolerance on the coordinate. Reaches the
    /// same optimum in a third of the evaluations when the objective is
    /// smooth, and never returns a point worse than the current one.
    Brent { tol: f64, max_iter: usize },
}

pub(crate) fn coord_ascent_with_diagnostics<F>(
    x0: &[f64],
    brackets: &[(f64, f64)],
    rel_tol: f64,
    max_cycles: usize,
    iters_per_coord: usize,
    f: F,
) -> (Vec<f64>, f64, usize, bool)
where
    F: Fn(&[f64]) -> f64,
{
    coord_ascent_with_search(
        x0,
        brackets,
        rel_tol,
        max_cycles,
        CoordinateSearch::Golden(iters_per_coord),
        f,
    )
}

pub(crate) fn coord_ascent_with_search<F>(
    x0: &[f64],
    brackets: &[(f64, f64)],
    rel_tol: f64,
    max_cycles: usize,
    search: CoordinateSearch,
    f: F,
) -> (Vec<f64>, f64, usize, bool)
where
    F: Fn(&[f64]) -> f64,
{
    assert_eq!(x0.len(), brackets.len());
    let mut x = x0.to_vec();
    let mut best = f(&x);
    if x.is_empty() {
        return (x, best, 0, best.is_finite());
    }
    let mut completed = 0;
    for _ in 0..max_cycles {
        let previous = best;
        completed += 1;
        for k in 0..x.len() {
            let mut candidate = x.clone();
            let objective = |value: f64| {
                candidate[k] = value;
                f(&candidate)
            };
            let coordinate = match search {
                CoordinateSearch::Golden(iterations) => {
                    maximize_scalar(brackets[k].0, brackets[k].1, iterations, objective)
                }
                CoordinateSearch::Brent { tol, max_iter } => {
                    maximize_scalar_brent(
                        brackets[k].0,
                        brackets[k].1,
                        Some(x[k]),
                        tol,
                        max_iter,
                        objective,
                    )
                    .x
                }
            };
            candidate[k] = coordinate;
            let value = f(&candidate);
            // A bounded scalar search must never discard a better warm start.
            if value.is_finite() && value > best {
                x = candidate;
                best = value;
            }
        }
        if best.is_finite() && (best - previous).abs() <= rel_tol * previous.abs().max(1.0) {
            return (x, best, completed, true);
        }
    }
    (x, best, completed, false)
}

/// Outcome of a bounded scalar maximisation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScalarMaximum {
    /// Location of the best point found.
    pub x: f64,
    /// Objective value at `x` (`-inf` if no finite value was encountered).
    pub value: f64,
    /// Number of objective evaluations consumed, including the start point.
    pub evaluations: usize,
    /// Whether the bracket around `x` shrank below the requested tolerance
    /// before the iteration cap was reached.
    pub converged: bool,
}

/// Bounded scalar maximisation of `f` on `[low, high]` with Brent's method
/// (parabolic interpolation safeguarded by golden-section steps).
///
/// `start`, when strictly inside the bracket, seeds the search so that a good
/// warm start (for example a Kendall-τ inversion) is refined rather than
/// rediscovered; the returned value is never worse than `f(start)`. The
/// search stops once the bracket around the incumbent is narrower than `tol`
/// (absolute, plus a `1e-10·|x|` relative term) or after `max_iter`
/// additional objective evaluations. Non-finite objective values are treated
/// as arbitrarily bad, so the search backs away from invalid regions instead
/// of propagating NaNs.
pub fn maximize_scalar_brent<F>(
    low: f64,
    high: f64,
    start: Option<f64>,
    tol: f64,
    max_iter: usize,
    mut f: F,
) -> ScalarMaximum
where
    F: FnMut(f64) -> f64,
{
    // Brent's `localmin` minimises; negate the objective and map non-finite
    // values to +inf so comparisons stay total.
    let mut g = |x: f64| {
        let value = f(x);
        if value.is_finite() {
            -value
        } else {
            f64::INFINITY
        }
    };
    let (mut a, mut b) = if low <= high {
        (low, high)
    } else {
        (high, low)
    };
    let golden = 0.5 * (3.0 - 5.0_f64.sqrt());
    let mut x = match start {
        Some(seed) if seed.is_finite() && seed > a && seed < b => seed,
        _ => a + golden * (b - a),
    };
    let mut w = x;
    let mut v = x;
    let mut fx = g(x);
    let mut fw = fx;
    let mut fv = fx;
    let mut evaluations = 1usize;
    let mut d = 0.0_f64;
    let mut e = 0.0_f64;
    let mut converged = false;

    for _ in 0..max_iter {
        let xm = 0.5 * (a + b);
        let tol1 = 1e-10 * x.abs() + tol;
        let tol2 = 2.0 * tol1;
        if (x - xm).abs() <= tol2 - 0.5 * (b - a) {
            converged = true;
            break;
        }
        let mut golden_step = true;
        if e.abs() > tol1 && fx.is_finite() && fw.is_finite() && fv.is_finite() {
            // Parabola through (x, fx), (w, fw), (v, fv).
            let r = (x - w) * (fx - fv);
            let mut q = (x - v) * (fx - fw);
            let mut p = (x - v) * q - (x - w) * r;
            q = 2.0 * (q - r);
            if q > 0.0 {
                p = -p;
            } else {
                q = -q;
            }
            let previous = e;
            e = d;
            if p.abs() < (0.5 * q * previous).abs() && p > q * (a - x) && p < q * (b - x) {
                d = p / q;
                let u = x + d;
                if u - a < tol2 || b - u < tol2 {
                    d = if xm - x >= 0.0 { tol1 } else { -tol1 };
                }
                golden_step = false;
            }
        }
        if golden_step {
            e = if x >= xm { a - x } else { b - x };
            d = golden * e;
        }
        let u = if d.abs() >= tol1 {
            x + d
        } else if d >= 0.0 {
            x + tol1
        } else {
            x - tol1
        };
        let fu = g(u);
        evaluations += 1;
        if fu <= fx {
            if u >= x {
                a = x;
            } else {
                b = x;
            }
            v = w;
            fv = fw;
            w = x;
            fw = fx;
            x = u;
            fx = fu;
        } else {
            if u < x {
                a = u;
            } else {
                b = u;
            }
            if fu <= fw || w == x {
                v = w;
                fv = fw;
                w = u;
                fw = fu;
            } else if fu <= fv || v == x || v == w {
                v = u;
                fv = fu;
            }
        }
    }

    ScalarMaximum {
        x,
        value: if fx.is_finite() {
            -fx
        } else {
            f64::NEG_INFINITY
        },
        evaluations,
        converged,
    }
}

/// Tuning knobs for [`nelder_mead_maximize`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NelderMeadOptions {
    /// Edge length of the initial simplex along every coordinate.
    pub initial_step: f64,
    /// Absolute tolerance on the spread of objective values across the
    /// simplex vertices.
    pub ftol: f64,
    /// Absolute tolerance on the simplex diameter (max-norm distance from the
    /// best vertex).
    pub xtol: f64,
    /// Cap on the number of simplex iterations, summed over restarts.
    pub max_iter: usize,
    /// Number of times the simplex is rebuilt around the incumbent (with a
    /// halved step) after convergence, to escape degenerate simplices.
    pub restarts: usize,
}

/// Outcome of [`nelder_mead_maximize`].
#[derive(Debug, Clone, PartialEq)]
pub struct NelderMeadResult {
    /// Best point found (inside the bounds).
    pub x: Vec<f64>,
    /// Objective value at `x` (`-inf` if no finite value was encountered).
    pub value: f64,
    /// Simplex iterations consumed over all restarts.
    pub iterations: usize,
    /// Objective evaluations consumed over all restarts.
    pub evaluations: usize,
    /// Whether the final simplex met both tolerances before the cap.
    pub converged: bool,
}

/// Bounded Nelder–Mead maximisation of `f` starting from `x0`.
///
/// Trial points are projected onto the box `bounds`, non-finite objective
/// values are treated as arbitrarily bad, and the incumbent is never replaced
/// by a worse point, so the result is guaranteed to be at least as good as
/// `f(x0)`. After each convergence the simplex is rebuilt around the incumbent
/// with a halved step (up to `restarts` times) and the search resumes; a
/// restart that fails to improve the objective by more than `ftol` ends the
/// run early. This is intended as a polish for low-dimensional likelihoods
/// that already have a grid or moment-based warm start.
pub fn nelder_mead_maximize<F>(
    x0: &[f64],
    bounds: &[(f64, f64)],
    options: &NelderMeadOptions,
    mut f: F,
) -> NelderMeadResult
where
    F: FnMut(&[f64]) -> f64,
{
    assert_eq!(x0.len(), bounds.len(), "one bound per coordinate");
    let n = x0.len();
    let project = |x: &mut [f64]| {
        for (value, (low, high)) in x.iter_mut().zip(bounds) {
            *value = value.clamp(*low, *high);
        }
    };
    let mut evaluations = 0usize;
    let mut g = |x: &[f64]| {
        evaluations += 1;
        let value = f(x);
        if value.is_finite() {
            -value
        } else {
            f64::INFINITY
        }
    };

    let mut best_x = x0.to_vec();
    project(&mut best_x);
    let mut best_g = g(&best_x);
    if n == 0 {
        return NelderMeadResult {
            x: best_x,
            value: if best_g.is_finite() {
                -best_g
            } else {
                f64::NEG_INFINITY
            },
            iterations: 0,
            evaluations,
            converged: true,
        };
    }

    let mut iterations = 0usize;
    let mut converged = false;
    let mut step = options.initial_step.abs().max(f64::EPSILON);
    for restart in 0..=options.restarts {
        // Build the simplex around the incumbent, flipping the step direction
        // when the forward point would leave the box.
        let mut simplex: Vec<(Vec<f64>, f64)> = Vec::with_capacity(n + 1);
        simplex.push((best_x.clone(), best_g));
        for i in 0..n {
            let mut vertex = best_x.clone();
            let forward = vertex[i] + step;
            vertex[i] = if forward <= bounds[i].1 {
                forward
            } else {
                vertex[i] - step
            };
            project(&mut vertex);
            let value = g(&vertex);
            simplex.push((vertex, value));
        }
        let restart_start = best_g;
        converged = false;

        while iterations < options.max_iter {
            iterations += 1;
            simplex.sort_by(|left, right| left.1.total_cmp(&right.1));
            let f_spread = simplex[n].1 - simplex[0].1;
            let x_spread = simplex[1..]
                .iter()
                .flat_map(|(vertex, _)| {
                    vertex.iter().zip(&simplex[0].0).map(|(a, b)| (a - b).abs())
                })
                .fold(0.0_f64, f64::max);
            // An infinite spread means the validity boundary crosses the
            // simplex; once the simplex has collapsed there is nothing left
            // to resolve at this tolerance.
            if x_spread <= options.xtol && (f_spread <= options.ftol || !f_spread.is_finite()) {
                converged = true;
                break;
            }

            let mut centroid = vec![0.0; n];
            for (vertex, _) in &simplex[..n] {
                for (c, v) in centroid.iter_mut().zip(vertex) {
                    *c += v / n as f64;
                }
            }
            let worst = simplex[n].clone();
            let mut reflected: Vec<f64> = centroid
                .iter()
                .zip(&worst.0)
                .map(|(c, w)| c + (c - w))
                .collect();
            project(&mut reflected);
            let g_reflected = g(&reflected);

            if g_reflected < simplex[0].1 {
                let mut expanded: Vec<f64> = centroid
                    .iter()
                    .zip(&reflected)
                    .map(|(c, r)| c + 2.0 * (r - c))
                    .collect();
                project(&mut expanded);
                let g_expanded = g(&expanded);
                simplex[n] = if g_expanded < g_reflected {
                    (expanded, g_expanded)
                } else {
                    (reflected, g_reflected)
                };
            } else if g_reflected < simplex[n - 1].1 {
                simplex[n] = (reflected, g_reflected);
            } else {
                let (mut contracted, threshold): (Vec<f64>, f64) = if g_reflected < worst.1 {
                    (
                        centroid
                            .iter()
                            .zip(&reflected)
                            .map(|(c, r)| c + 0.5 * (r - c))
                            .collect(),
                        g_reflected,
                    )
                } else {
                    (
                        centroid
                            .iter()
                            .zip(&worst.0)
                            .map(|(c, w)| c + 0.5 * (w - c))
                            .collect(),
                        worst.1,
                    )
                };
                project(&mut contracted);
                let g_contracted = g(&contracted);
                if g_contracted <= threshold && g_contracted < worst.1 {
                    simplex[n] = (contracted, g_contracted);
                } else {
                    let best_vertex = simplex[0].0.clone();
                    for (vertex, value) in simplex.iter_mut().skip(1) {
                        for (v, b) in vertex.iter_mut().zip(&best_vertex) {
                            *v = b + 0.5 * (*v - b);
                        }
                        project(vertex);
                        *value = g(vertex);
                    }
                }
            }
        }

        simplex.sort_by(|left, right| left.1.total_cmp(&right.1));
        if simplex[0].1 < best_g {
            best_g = simplex[0].1;
            best_x = simplex[0].0.clone();
        }
        if iterations >= options.max_iter {
            break;
        }
        // Stop restarting once a rebuilt simplex no longer buys anything.
        if restart > 0 && restart_start - best_g <= options.ftol {
            break;
        }
        step *= 0.5;
    }

    NelderMeadResult {
        x: best_x,
        value: if best_g.is_finite() {
            -best_g
        } else {
            f64::NEG_INFINITY
        },
        iterations,
        evaluations,
        converged,
    }
}

/// Tuning knobs for [`quasi_newton_maximize`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QuasiNewtonOptions {
    /// Convergence threshold on the *predicted* remaining gain
    /// `½ gᵀ H⁻¹ g` of the quadratic model (absolute, in objective units).
    pub ftol: f64,
    /// Step length (per coordinate) used along a coordinate whose curvature
    /// cannot be estimated.
    pub initial_step: f64,
    /// Cap on quasi-Newton iterations.
    pub max_iter: usize,
    /// Evaluate the finite-difference stencils of each gradient and Hessian
    /// on the rayon pool. Worth it when one objective evaluation is a pass over a
    /// sample; pure overhead for cheap objectives.
    pub parallel_gradient: bool,
}

/// Outcome of [`quasi_newton_maximize`].
#[derive(Debug, Clone, PartialEq)]
pub struct QuasiNewtonResult {
    /// Best point found (inside the bounds).
    pub x: Vec<f64>,
    /// Objective value at `x` (`-inf` if no finite value was encountered).
    pub value: f64,
    /// Quasi-Newton iterations consumed.
    pub iterations: usize,
    /// Objective evaluations consumed, including gradient stencils.
    pub evaluations: usize,
    /// Whether the predicted remaining gain fell below `ftol`. When `false`
    /// the search stalled (line-search failure, non-finite stencil, or the
    /// iteration cap) and `x` is merely the best point seen; callers that
    /// need a certified optimum should continue with a derivative-free
    /// method from there.
    pub converged: bool,
}

/// Bounded quasi-Newton maximisation of a smooth `f` starting from `x0`,
/// with finite-difference derivatives.
///
/// This is the fast polish for likelihoods that already have a warm start in
/// the right basin: where Nelder–Mead needs a few hundred evaluations to
/// shrink a simplex to convergence, this needs a few dozen.
///
/// * The curvature model starts from a finite-difference Hessian at `x0`, so
///   the first step is a Newton step. Copula parameters in log/logit
///   coordinates are routinely strongly correlated and differ by orders of
///   magnitude in curvature; a scaled identity or a diagonal would call a
///   ridge converged while there is still likelihood to gain along it.
/// * Later iterations update that model by BFGS from central-difference
///   gradients (`2n` evaluations each, optionally in parallel).
/// * Bounds are handled as an active set: a coordinate on a bound with the
///   gradient pointing outwards is frozen, a step stops exactly at the first
///   bound it reaches, and the model is rebuilt whenever the active set
///   changes.
///
/// Non-finite objective values are treated as arbitrarily bad in the line
/// search, and a non-finite derivative stencil ends the search with
/// `converged = false`. The returned point is never worse than `f(x0)`.
pub fn quasi_newton_maximize<F>(
    x0: &[f64],
    bounds: &[(f64, f64)],
    options: &QuasiNewtonOptions,
    f: F,
) -> QuasiNewtonResult
where
    F: Fn(&[f64]) -> f64 + Sync,
{
    /// Largest move of any coordinate in one iteration (unconstrained units).
    const MAX_STEP: f64 = 1.0;
    /// Relative stencil widths for first and second differences.
    const GRADIENT_STEP: f64 = 6.0e-6;
    const HESSIAN_STEP: f64 = 1.0e-3;

    assert_eq!(x0.len(), bounds.len(), "one bound per coordinate");
    let n = x0.len();
    let evaluations = std::sync::atomic::AtomicUsize::new(0);
    // Minimise g = −f; non-finite objective values become +inf.
    let g = |x: &[f64]| {
        evaluations.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let value = f(x);
        if value.is_finite() {
            -value
        } else {
            f64::INFINITY
        }
    };
    let finish = |x: Vec<f64>, gx: f64, iterations: usize, converged: bool| QuasiNewtonResult {
        x,
        value: if gx.is_finite() {
            -gx
        } else {
            f64::NEG_INFINITY
        },
        iterations,
        evaluations: evaluations.load(std::sync::atomic::Ordering::Relaxed),
        converged,
    };
    // Runs `job(0..len)` serially or on the pool.
    let map = |len: usize, job: &(dyn Fn(usize) -> f64 + Sync)| -> Vec<f64> {
        if options.parallel_gradient && len > 1 {
            use rayon::prelude::*;
            (0..len).into_par_iter().map(job).collect()
        } else {
            (0..len).map(job).collect()
        }
    };

    let mut x: Vec<f64> = x0
        .iter()
        .zip(bounds)
        .map(|(value, (low, high))| value.clamp(*low, *high))
        .collect();
    let mut gx = g(&x);
    if n == 0 || !gx.is_finite() {
        return finish(x, gx, 0, n == 0 && gx.is_finite());
    }

    // Central differences, one-sided at a bound. `None` when a stencil point
    // is non-finite.
    let gradient = |x: &[f64], gx: f64| -> Option<Vec<f64>> {
        let slopes = map(n, &|i| {
            let h = GRADIENT_STEP * x[i].abs().max(1.0);
            let (low, high) = bounds[i];
            let mut point = x.to_vec();
            let up = (x[i] + h).min(high);
            let down = (x[i] - h).max(low);
            if up == down {
                return 0.0;
            }
            let value_up = if up > x[i] {
                point[i] = up;
                g(&point)
            } else {
                gx
            };
            let value_down = if down < x[i] {
                point[i] = down;
                g(&point)
            } else {
                gx
            };
            (value_up - value_down) / (up - down)
        });
        slopes
            .iter()
            .all(|slope| slope.is_finite())
            .then_some(slopes)
    };
    let pinned = |x: &[f64], grad: &[f64]| -> Vec<bool> {
        (0..n)
            .map(|i| {
                (x[i] <= bounds[i].0 && grad[i] > 0.0) || (x[i] >= bounds[i].1 && grad[i] < 0.0)
            })
            .collect()
    };

    // Inverse-Hessian model on the free coordinates (row-major, zero rows and
    // columns for frozen ones), from a finite-difference Hessian when that is
    // positive definite and from its usable diagonal otherwise.
    let model = |x: &[f64], gx: f64, grad: &[f64], frozen: &[bool]| -> Vec<f64> {
        let free: Vec<usize> = (0..n).filter(|&i| !frozen[i]).collect();
        let k = free.len();
        let mut inverse = vec![0.0; n * n];
        if k == 0 {
            return inverse;
        }
        // Step towards the side with room; second differences are symmetric
        // in the step sign, so a one-sided stencil pair is fine at a bound.
        let steps: Vec<f64> = free
            .iter()
            .map(|&i| {
                let h = HESSIAN_STEP * x[i].abs().max(1.0);
                if x[i] + h <= bounds[i].1 { h } else { -h }
            })
            .collect();
        let shifted = |moves: &[(usize, f64)]| {
            let mut point = x.to_vec();
            for &(i, delta) in moves {
                point[i] += delta;
            }
            if point
                .iter()
                .zip(bounds)
                .any(|(value, (low, high))| value < low || value > high)
            {
                f64::NAN
            } else {
                g(&point)
            }
        };
        let forward = map(k, &|a| shifted(&[(free[a], steps[a])]));
        let backward = map(k, &|a| shifted(&[(free[a], -steps[a])]));
        let pairs: Vec<(usize, usize)> = (0..k)
            .flat_map(|a| ((a + 1)..k).map(move |b| (a, b)))
            .collect();
        // Mixed partials by the symmetric four-point stencil. Its O(h²)
        // error matters: along a likelihood ridge the Hessian is nearly
        // singular, and a first-order cross term is enough to wipe out the
        // small eigenvalue that says how far the ridge still climbs.
        let cross = map(pairs.len(), &|p| {
            let (a, b) = pairs[p];
            let (i, j) = (free[a], free[b]);
            let (hi, hj) = (steps[a], steps[b]);
            let corners = [
                shifted(&[(i, hi), (j, hj)]),
                shifted(&[(i, hi), (j, -hj)]),
                shifted(&[(i, -hi), (j, hj)]),
                shifted(&[(i, -hi), (j, -hj)]),
            ];
            (corners[0] - corners[1] - corners[2] + corners[3]) / (4.0 * hi * hj)
        });

        let mut hessian = Array2::<f64>::zeros((k, k));
        for a in 0..k {
            hessian[(a, a)] = (forward[a] - 2.0 * gx + backward[a]) / (steps[a] * steps[a]);
        }
        for (p, &(a, b)) in pairs.iter().enumerate() {
            hessian[(a, b)] = cross[p];
            hessian[(b, a)] = cross[p];
        }
        if hessian.iter().all(|value| value.is_finite())
            && cholesky(&hessian).is_ok()
            && let Ok(inverted) = self::inverse(&hessian)
        {
            for a in 0..k {
                for b in 0..k {
                    inverse[free[a] * n + free[b]] = inverted[(a, b)];
                }
            }
            return inverse;
        }
        // Not a usable bowl (non-finite stencil, saddle, or flat stretch):
        // keep each coordinate's own curvature where it is clearly positive,
        // and otherwise the step that moves it by `initial_step`.
        for (a, &i) in free.iter().enumerate() {
            let slope = grad[i].abs().max(f64::MIN_POSITIVE);
            let curvature = hessian[(a, a)];
            let floor = 64.0 * f64::EPSILON * gx.abs().max(1.0) / (steps[a] * steps[a]);
            inverse[i * n + i] = if curvature.is_finite() && curvature > floor {
                (1.0 / curvature).min(MAX_STEP / slope)
            } else {
                options.initial_step / slope
            };
        }
        inverse
    };

    let Some(mut grad) = gradient(&x, gx) else {
        return finish(x, gx, 0, false);
    };
    let mut frozen = pinned(&x, &grad);
    let mut inverse = model(&x, gx, &grad, &frozen);
    // Whether `inverse` was just rebuilt (no BFGS update absorbed yet).
    let mut fresh = true;

    let mut iterations = 0usize;
    let mut converged = false;
    let mut direction = vec![0.0; n];
    let mut trial = vec![0.0; n];
    while iterations < options.max_iter {
        let mut predicted = 0.0;
        for i in 0..n {
            direction[i] = 0.0;
            if frozen[i] {
                continue;
            }
            for j in 0..n {
                direction[i] -= inverse[i * n + j] * grad[j];
            }
            predicted -= 0.5 * direction[i] * grad[i];
        }
        if predicted.is_nan() || predicted <= 0.0 {
            if (0..n).all(|i| frozen[i] || grad[i] == 0.0) {
                converged = true;
                break;
            }
            // Not a descent direction: the curvature model is unusable.
            if fresh {
                break;
            }
            inverse = model(&x, gx, &grad, &frozen);
            fresh = true;
            continue;
        }
        if predicted <= options.ftol {
            converged = true;
            break;
        }
        iterations += 1;

        // Trust region: no coordinate moves more than `MAX_STEP` at once,
        // and the step ends at the first bound it reaches.
        let longest = direction.iter().fold(0.0_f64, |acc, d| acc.max(d.abs()));
        let mut step = (MAX_STEP / longest).min(1.0);
        let mut hits_bound = false;
        for i in 0..n {
            let room = if direction[i] > 0.0 {
                (bounds[i].1 - x[i]) / direction[i]
            } else if direction[i] < 0.0 {
                (bounds[i].0 - x[i]) / direction[i]
            } else {
                f64::INFINITY
            };
            if room < step {
                step = room;
                hits_bound = true;
            }
        }
        let slope: f64 = direction.iter().zip(&grad).map(|(d, v)| d * v).sum();
        let mut accepted = None;
        for attempt in 0..40 {
            for i in 0..n {
                trial[i] = (x[i] + step * direction[i]).clamp(bounds[i].0, bounds[i].1);
            }
            if attempt == 0 && hits_bound {
                // Land exactly on the bound so the coordinate can be frozen.
                for i in 0..n {
                    let target = if direction[i] > 0.0 {
                        bounds[i].1
                    } else {
                        bounds[i].0
                    };
                    if direction[i] != 0.0
                        && (target - trial[i]).abs() <= 1e-12 * target.abs().max(1.0)
                    {
                        trial[i] = target;
                    }
                }
            }
            if trial == x {
                break;
            }
            let value = g(&trial);
            if value <= gx + 1e-4 * step * slope {
                accepted = Some(value);
                break;
            }
            step *= if value.is_finite() { 0.5 } else { 0.25 };
        }
        let Some(value) = accepted else {
            if fresh {
                // Even a Newton step from a fresh model cannot improve:
                // stationary to within the resolution of the objective.
                converged = predicted <= 1e3 * options.ftol;
                break;
            }
            inverse = model(&x, gx, &grad, &frozen);
            fresh = true;
            continue;
        };
        let Some(new_grad) = gradient(&trial, value) else {
            // Keep the improvement but stop: the stencil left the domain.
            x.copy_from_slice(&trial);
            gx = value;
            break;
        };

        let new_frozen = pinned(&trial, &new_grad);
        if new_frozen != frozen {
            // The active set changed: curvature gathered in the old subspace
            // does not carry over.
            x.copy_from_slice(&trial);
            gx = value;
            grad = new_grad;
            frozen = new_frozen;
            inverse = model(&x, gx, &grad, &frozen);
            fresh = true;
            continue;
        }

        // BFGS update of the inverse Hessian with s = Δx, y = Δgrad on the
        // free coordinates.
        let s: Vec<f64> = (0..n)
            .map(|i| if frozen[i] { 0.0 } else { trial[i] - x[i] })
            .collect();
        let y: Vec<f64> = (0..n)
            .map(|i| {
                if frozen[i] {
                    0.0
                } else {
                    new_grad[i] - grad[i]
                }
            })
            .collect();
        let sy: f64 = s.iter().zip(&y).map(|(a, b)| a * b).sum();
        let yy: f64 = y.iter().map(|v| v * v).sum();
        let ss: f64 = s.iter().map(|v| v * v).sum();
        if sy > 1e-10 * (ss * yy).sqrt() && sy > 0.0 {
            let hy: Vec<f64> = (0..n)
                .map(|i| (0..n).map(|j| inverse[i * n + j] * y[j]).sum())
                .collect();
            let yhy: f64 = y.iter().zip(&hy).map(|(a, b)| a * b).sum();
            let factor = (1.0 + yhy / sy) / sy;
            for i in 0..n {
                for j in 0..n {
                    inverse[i * n + j] += factor * s[i] * s[j] - (hy[i] * s[j] + s[i] * hy[j]) / sy;
                }
            }
            fresh = false;
        }
        x.copy_from_slice(&trial);
        gx = value;
        grad = new_grad;
    }
    finish(x, gx, iterations, converged)
}

/// Maximises a smooth likelihood from a warm start: quasi-Newton for speed,
/// coordinate probes for robustness, Nelder–Mead as the last resort.
///
/// [`quasi_newton_maximize`] converges in a few dozen evaluations but, like
/// any gradient method, declares victory wherever the gradient vanishes. In
/// the log/logit coordinates used for copula parameters that also happens on
/// plateaus — a parameter saturating towards a bound flattens the objective
/// without being optimal. After each quasi-Newton run every coordinate is
/// therefore probed at a few large offsets and at its bounds; an improving
/// probe restarts the search from there. Only if the quasi-Newton search
/// cannot certify convergence does the (several times more expensive)
/// simplex search run, from the best point found.
///
/// `tol` is the absolute objective tolerance. The result is never worse than
/// `f(x0)`; `(x, value)` is returned.
pub(crate) fn polish_maximize<F>(
    x0: &[f64],
    bounds: &[(f64, f64)],
    tol: f64,
    max_iter: usize,
    parallel: bool,
    f: F,
) -> (Vec<f64>, f64)
where
    F: Fn(&[f64]) -> f64 + Sync,
{
    /// Offsets (in unconstrained units) at which each coordinate is probed.
    const PROBE_OFFSETS: [f64; 4] = [-3.0, -1.0, 1.0, 3.0];
    /// Quasi-Newton runs before giving up on probes improving the optimum.
    const MAX_ROUNDS: usize = 4;

    let options = QuasiNewtonOptions {
        ftol: tol,
        initial_step: 0.2,
        max_iter,
        parallel_gradient: parallel,
    };
    let mut x: Vec<f64> = x0
        .iter()
        .zip(bounds)
        .map(|(value, (low, high))| value.clamp(*low, *high))
        .collect();
    let mut value = f(&x);
    let mut certified = false;
    for _ in 0..MAX_ROUNDS {
        let run = quasi_newton_maximize(&x, bounds, &options, &f);
        if run.value.is_finite() && run.value >= value {
            x = run.x;
            value = run.value;
        }
        certified = run.converged;
        if !certified {
            break;
        }
        // Probe each coordinate far from the incumbent.
        let mut probes: Vec<Vec<f64>> = Vec::new();
        for (i, &(low, high)) in bounds.iter().enumerate() {
            let mut targets: Vec<f64> = PROBE_OFFSETS
                .iter()
                .map(|offset| (x[i] + offset).clamp(low, high))
                .collect();
            targets.push(low);
            targets.push(high);
            targets.sort_by(f64::total_cmp);
            targets.dedup();
            for target in targets {
                if target != x[i] {
                    let mut probe = x.clone();
                    probe[i] = target;
                    probes.push(probe);
                }
            }
        }
        let values: Vec<f64> = if parallel {
            use rayon::prelude::*;
            probes.par_iter().map(|probe| f(probe)).collect()
        } else {
            probes.iter().map(|probe| f(probe)).collect()
        };
        let mut improved = false;
        let mut best = value;
        for (probe, probe_value) in probes.into_iter().zip(values) {
            if probe_value.is_finite() && probe_value > best + tol {
                best = probe_value;
                x = probe;
                improved = true;
            }
        }
        if !improved {
            break;
        }
        value = best;
    }
    if !certified {
        let simplex = nelder_mead_maximize(
            &x,
            bounds,
            &NelderMeadOptions {
                initial_step: 0.2,
                ftol: tol,
                xtol: 1e-6,
                max_iter,
                restarts: 1,
            },
            &f,
        );
        if simplex.value.is_finite() && simplex.value >= value {
            x = simplex.x;
            value = simplex.value;
        }
    }
    (x, value)
}

/// Numerical Hessian via central differences.
///
/// Computes
///     H_{ii} = (f(x + h e_i) + f(x − h e_i) − 2 f(x)) / h²,
///     H_{ij} = (f(x + h e_i + h e_j) − f(x + h e_i − h e_j)
///              − f(x − h e_i + h e_j) + f(x − h e_i − h e_j)) / (4 h²).
///
/// `f` should return a smooth scalar in a neighbourhood of `x`; callers that
/// pass a quadrature-integrated log-likelihood should use the *same* quadrature
/// rule as the optimiser so `f` is deterministic and continuous.
///
/// Cost: `O(p²)` evaluations of `f`. With `p ≤ 20` (the realistic factor-
/// copula case) and the factor-loglik cost this is at most a few hundred
/// calls — negligible next to the polish stage itself.
pub fn numerical_hessian<F>(x: &[f64], h: f64, f: F) -> Array2<f64>
where
    F: Fn(&[f64]) -> f64,
{
    let p = x.len();
    let mut hess = Array2::<f64>::zeros((p, p));
    if p == 0 {
        return hess;
    }

    let f_x = f(x);
    let mut scratch = x.to_vec();

    // Diagonal.
    for i in 0..p {
        let orig = scratch[i];
        scratch[i] = orig + h;
        let f_plus = f(&scratch);
        scratch[i] = orig - h;
        let f_minus = f(&scratch);
        scratch[i] = orig;
        hess[(i, i)] = (f_plus - 2.0 * f_x + f_minus) / (h * h);
    }

    // Off-diagonal.
    for i in 0..p {
        for j in (i + 1)..p {
            let oi = scratch[i];
            let oj = scratch[j];

            scratch[i] = oi + h;
            scratch[j] = oj + h;
            let f_pp = f(&scratch);
            scratch[j] = oj - h;
            let f_pm = f(&scratch);
            scratch[i] = oi - h;
            let f_mm = f(&scratch);
            scratch[j] = oj + h;
            let f_mp = f(&scratch);

            scratch[i] = oi;
            scratch[j] = oj;

            let value = (f_pp - f_pm - f_mp + f_mm) / (4.0 * h * h);
            hess[(i, j)] = value;
            hess[(j, i)] = value;
        }
    }

    hess
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    use super::{
        NelderMeadOptions, QuasiNewtonOptions, cholesky, coord_ascent_maximise, gauss_legendre_01,
        log_determinant_from_cholesky, maximize_scalar_brent, nelder_mead_maximize,
        numerical_hessian, quadratic_form_from_cholesky, quasi_newton_maximize,
    };

    #[test]
    fn brent_refines_a_warm_start_to_tolerance() {
        let f = |x: f64| -(x - 1.3).powi(2) + 0.25;
        let result = maximize_scalar_brent(-5.0, 5.0, Some(0.0), 1e-10, 200, f);
        assert!(result.converged);
        assert!((result.x - 1.3).abs() < 1e-7, "x = {}", result.x);
        assert!((result.value - 0.25).abs() < 1e-12);
        assert!(result.value >= f(0.0));
        // Golden section needs ~50 evaluations to shrink a width-10 bracket
        // to 1e-10; parabolic steps land on the optimum much earlier and the
        // remainder is spent certifying the bracket.
        assert!(
            result.evaluations < 40,
            "evaluations = {}",
            result.evaluations
        );
    }

    #[test]
    fn brent_backs_away_from_non_finite_regions() {
        let f = |x: f64| {
            if x < 0.0 {
                f64::NAN
            } else {
                -(x - 0.5).powi(2)
            }
        };
        let result = maximize_scalar_brent(-2.0, 2.0, Some(-1.0), 1e-9, 200, f);
        assert!((result.x - 0.5).abs() < 1e-6, "x = {}", result.x);
        assert!(result.value.is_finite());
    }

    #[test]
    fn brent_honours_the_iteration_cap() {
        let result = maximize_scalar_brent(-5.0, 5.0, None, 1e-12, 3, |x| -(x * x));
        assert_eq!(result.evaluations, 4);
        assert!(!result.converged);
        assert!(result.value.is_finite());
    }

    #[test]
    fn brent_clamps_an_out_of_bracket_start() {
        let result = maximize_scalar_brent(0.0, 1.0, Some(7.0), 1e-9, 100, |x| -(x - 0.25).powi(2));
        assert!((result.x - 0.25).abs() < 1e-6);
    }

    fn nm_options() -> NelderMeadOptions {
        NelderMeadOptions {
            initial_step: 0.25,
            ftol: 1e-10,
            xtol: 1e-7,
            max_iter: 500,
            restarts: 2,
        }
    }

    #[test]
    fn nelder_mead_finds_quadratic_maximum() {
        // argmax of -((x - 1.5)^2 + 2 (y + 0.5)^2 + 0.6 x y) solves the linear
        // system 2(x - 1.5) + 0.6 y = 0, 4(y + 0.5) + 0.6 x = 0.
        let f =
            |x: &[f64]| -((x[0] - 1.5).powi(2) + 2.0 * (x[1] + 0.5).powi(2) + 0.6 * x[0] * x[1]);
        let det = 2.0 * 4.0 - 0.6 * 0.6;
        let x_star = (3.0 * 4.0 - 0.6 * (-2.0)) / det;
        let y_star = (2.0 * (-2.0) - 0.6 * 3.0) / det;
        let result =
            nelder_mead_maximize(&[0.0, 0.0], &[(-5.0, 5.0), (-5.0, 5.0)], &nm_options(), f);
        assert!(result.converged);
        assert!((result.x[0] - x_star).abs() < 1e-5, "x = {:?}", result.x);
        assert!((result.x[1] - y_star).abs() < 1e-5, "x = {:?}", result.x);
        assert!(result.value >= f(&[0.0, 0.0]));
        assert!(result.iterations <= 500);
    }

    #[test]
    fn nelder_mead_respects_bounds_and_never_regresses() {
        let f = |x: &[f64]| -((x[0] - 3.0).powi(2) + (x[1] + 4.0).powi(2));
        let result =
            nelder_mead_maximize(&[0.5, 0.5], &[(-1.0, 1.0), (-1.0, 1.0)], &nm_options(), f);
        assert!(result.x[0] <= 1.0 && result.x[0] >= -1.0);
        assert!(result.x[1] <= 1.0 && result.x[1] >= -1.0);
        assert!((result.x[0] - 1.0).abs() < 1e-5, "x = {:?}", result.x);
        assert!((result.x[1] + 1.0).abs() < 1e-5, "x = {:?}", result.x);
        assert!(result.value >= f(&[0.5, 0.5]));
    }

    #[test]
    fn nelder_mead_survives_invalid_regions_and_iteration_caps() {
        let f = |x: &[f64]| {
            if x[0] + x[1] > 1.0 {
                f64::NAN
            } else {
                -((x[0] - 0.9).powi(2) + (x[1] - 0.9).powi(2))
            }
        };
        let result =
            nelder_mead_maximize(&[0.0, 0.0], &[(-2.0, 2.0), (-2.0, 2.0)], &nm_options(), f);
        assert!(result.value.is_finite());
        assert!(result.x[0] + result.x[1] <= 1.0 + 1e-9);
        assert!(
            (result.x[0] + result.x[1] - 1.0).abs() < 1e-4,
            "x = {:?}",
            result.x
        );

        let capped = nelder_mead_maximize(
            &[0.0, 0.0],
            &[(-5.0, 5.0), (-5.0, 5.0)],
            &NelderMeadOptions {
                max_iter: 5,
                ..nm_options()
            },
            |x: &[f64]| -(x[0] * x[0] + x[1] * x[1]),
        );
        assert!(capped.iterations <= 5);
        assert!(!capped.converged);
        assert!(capped.value >= -0.0);
    }

    fn qn_options() -> QuasiNewtonOptions {
        QuasiNewtonOptions {
            ftol: 1e-12,
            initial_step: 0.2,
            max_iter: 200,
            parallel_gradient: false,
        }
    }

    #[test]
    fn quasi_newton_finds_quadratic_maximum_in_few_evaluations() {
        let f =
            |x: &[f64]| -((x[0] - 1.5).powi(2) + 2.0 * (x[1] + 0.5).powi(2) + 0.6 * x[0] * x[1]);
        let det = 2.0 * 4.0 - 0.6 * 0.6;
        let x_star = (3.0 * 4.0 - 0.6 * (-2.0)) / det;
        let y_star = (2.0 * (-2.0) - 0.6 * 3.0) / det;
        let result =
            quasi_newton_maximize(&[0.0, 0.0], &[(-5.0, 5.0), (-5.0, 5.0)], &qn_options(), f);
        assert!(result.converged);
        assert!((result.x[0] - x_star).abs() < 1e-6, "x = {:?}", result.x);
        assert!((result.x[1] - y_star).abs() < 1e-6, "x = {:?}", result.x);
        assert!(
            result.evaluations < 80,
            "evaluations = {}",
            result.evaluations
        );
    }

    #[test]
    fn quasi_newton_handles_a_curved_valley() {
        // Rosenbrock, the classic test of curvature tracking.
        let f = |x: &[f64]| -((1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0] * x[0]).powi(2));
        let result = quasi_newton_maximize(
            &[-1.2, 1.0],
            &[(-5.0, 5.0), (-5.0, 5.0)],
            &QuasiNewtonOptions {
                max_iter: 500,
                ..qn_options()
            },
            f,
        );
        assert!(result.converged, "result = {result:?}");
        assert!((result.x[0] - 1.0).abs() < 1e-4, "x = {:?}", result.x);
        assert!((result.x[1] - 1.0).abs() < 1e-4, "x = {:?}", result.x);
    }

    #[test]
    fn quasi_newton_stops_on_active_bounds() {
        let f = |x: &[f64]| -((x[0] - 3.0).powi(2) + (x[1] + 4.0).powi(2));
        let result =
            quasi_newton_maximize(&[0.5, 0.5], &[(-1.0, 1.0), (-1.0, 1.0)], &qn_options(), f);
        assert!(result.converged, "result = {result:?}");
        assert_eq!(result.x, vec![1.0, -1.0]);
        assert!(result.value >= f(&[0.5, 0.5]));
    }

    #[test]
    fn quasi_newton_mixes_free_and_bound_coordinates() {
        // Optimum at x = 0.3 (interior), y at its upper bound.
        let f = |x: &[f64]| -((x[0] - 0.3).powi(2)) + 2.0 * x[1] - 0.1 * x[0] * x[1];
        let result =
            quasi_newton_maximize(&[-0.5, 0.0], &[(-1.0, 1.0), (-1.0, 1.0)], &qn_options(), f);
        assert!(result.converged, "result = {result:?}");
        assert_eq!(result.x[1], 1.0);
        assert!((result.x[0] - 0.25).abs() < 1e-6, "x = {:?}", result.x);
    }

    #[test]
    fn quasi_newton_never_regresses_and_reports_invalid_regions() {
        // NaN beyond x + y > 1: the optimum sits on the validity boundary,
        // which a gradient method cannot certify.
        let f = |x: &[f64]| {
            if x[0] + x[1] > 1.0 {
                f64::NAN
            } else {
                -((x[0] - 0.9).powi(2) + (x[1] - 0.9).powi(2))
            }
        };
        let start = [0.0, 0.0];
        let result = quasi_newton_maximize(&start, &[(-2.0, 2.0), (-2.0, 2.0)], &qn_options(), f);
        assert!(result.value.is_finite());
        assert!(result.value >= f(&start));
        assert!(result.x[0] + result.x[1] <= 1.0);

        let invalid = quasi_newton_maximize(&[5.0], &[(-9.0, 9.0)], &qn_options(), |_| f64::NAN);
        assert!(!invalid.converged);
        assert_eq!(invalid.value, f64::NEG_INFINITY);
    }

    #[test]
    fn quasi_newton_parallel_gradient_matches_serial() {
        let f = |x: &[f64]| -(x[0].exp() - 2.0 * x[0] + (x[1] - 0.4).powi(4) + x[2] * x[2]);
        let bounds = [(-3.0, 3.0); 3];
        let serial = quasi_newton_maximize(&[1.0, 1.0, 1.0], &bounds, &qn_options(), f);
        let parallel = quasi_newton_maximize(
            &[1.0, 1.0, 1.0],
            &bounds,
            &QuasiNewtonOptions {
                parallel_gradient: true,
                ..qn_options()
            },
            f,
        );
        assert_eq!(serial.x, parallel.x);
        assert_eq!(serial.value, parallel.value);
        assert!((serial.x[0] - 2.0_f64.ln()).abs() < 1e-6);
    }

    #[test]
    fn bracketed_root_converges_fast_on_smooth_functions() {
        let mut calls = 0usize;
        let root = super::find_root_bracketed((0.0, 4.0), (-2.0, 14.0), 0.0, 1e-13, 200, |x| {
            calls += 1;
            x * x - 2.0
        });
        assert!(
            (root - std::f64::consts::SQRT_2).abs() < 1e-12,
            "root = {root}"
        );
        assert!(calls <= 14, "calls = {calls}");

        // Monotone but far from polynomial: a CDF-like sigmoid.
        let g = |x: f64| 1.0 / (1.0 + (-(x - 37.5) * 3.0).exp()) - 0.25;
        let root = super::find_root_bracketed((0.0, 100.0), (g(0.0), g(100.0)), 0.0, 1e-12, 200, g);
        assert!(g(root).abs() < 1e-10, "residual = {}", g(root));
        // Exact zero at an end point is returned as is.
        assert_eq!(
            super::find_root_bracketed((1.0, 2.0), (0.0, 3.0), 0.0, 1e-12, 50, |x| x),
            1.0
        );
    }

    #[test]
    fn nelder_mead_handles_empty_input() {
        let result = nelder_mead_maximize(&[], &[], &nm_options(), |_: &[f64]| 4.0);
        assert!(result.x.is_empty());
        assert_eq!(result.value, 4.0);
        assert!(result.converged);
    }

    #[test]
    fn gauss_legendre_integrates_polynomials_exactly() {
        // An n-point rule is exact for polynomials of degree up to 2n-1.
        // Test ∫₀¹ x^k dx = 1/(k+1) for a few k using 25 nodes.
        let (nodes, weights) = gauss_legendre_01(25);
        assert_eq!(nodes.len(), 25);
        assert_eq!(weights.len(), 25);

        // Weights should sum to the interval length (1).
        let w_sum: f64 = weights.iter().sum();
        assert!((w_sum - 1.0).abs() < 1e-14, "weight sum was {w_sum}");

        for k in 0usize..10 {
            let approx: f64 = nodes
                .iter()
                .zip(weights.iter())
                .map(|(x, w)| w * x.powi(k as i32))
                .sum();
            let exact = 1.0 / (k as f64 + 1.0);
            assert!(
                (approx - exact).abs() < 1e-13,
                "k={k}: approx={approx}, exact={exact}"
            );
        }
    }

    #[test]
    fn gauss_legendre_nodes_strictly_inside_unit_interval() {
        let (nodes, weights) = gauss_legendre_01(25);
        for (x, w) in nodes.iter().zip(weights.iter()) {
            assert!(*x > 0.0 && *x < 1.0, "node {x} escaped (0, 1)");
            assert!(*w > 0.0, "weight {w} not strictly positive");
        }
        // Nodes are ascending.
        for pair in nodes.windows(2) {
            assert!(pair[0] < pair[1], "nodes not ascending: {pair:?}");
        }
    }

    #[test]
    fn cholesky_factorizes_spd_matrix() {
        let matrix = array![[1.0, 0.7], [0.7, 1.0]];
        let lower = cholesky(&matrix).expect("matrix should be SPD");

        let reconstructed = array![
            [lower[(0, 0)] * lower[(0, 0)], lower[(0, 0)] * lower[(1, 0)],],
            [
                lower[(1, 0)] * lower[(0, 0)],
                lower[(1, 0)] * lower[(1, 0)] + lower[(1, 1)] * lower[(1, 1)],
            ]
        ];

        for ((row, col), expected) in matrix.indexed_iter() {
            assert!((reconstructed[(row, col)] - expected).abs() < 1e-12);
        }

        let log_det = log_determinant_from_cholesky(&lower);
        assert!((log_det - (1.0_f64 - 0.49).ln()).abs() < 1e-12);

        let quad = quadratic_form_from_cholesky(&lower, &[1.0, -1.0]).expect("solve should work");
        assert!(quad > 0.0);
    }

    #[test]
    fn coord_ascent_finds_quadratic_maximum() {
        // f(x, y) = -((x - 1.5)^2 + 2 (y + 0.5)^2); argmax at (1.5, -0.5), f* = 0.
        let f = |x: &[f64]| -((x[0] - 1.5).powi(2) + 2.0 * (x[1] + 0.5).powi(2));
        let (x, best) =
            coord_ascent_maximise(&[0.0, 0.0], &[(-5.0, 5.0), (-5.0, 5.0)], 1e-10, 20, 80, f);
        assert!((x[0] - 1.5).abs() < 1e-6, "x[0] = {}", x[0]);
        assert!((x[1] - (-0.5)).abs() < 1e-6, "x[1] = {}", x[1]);
        assert!(best.abs() < 1e-10, "best = {best}");
    }

    #[test]
    fn coord_ascent_terminates_early_on_stationary_start() {
        let f = |x: &[f64]| -(x[0].powi(2) + x[1].powi(2));
        let (x, best) =
            coord_ascent_maximise(&[0.0, 0.0], &[(-1.0, 1.0), (-1.0, 1.0)], 1e-10, 5, 40, f);
        assert!(x[0].abs() < 1e-8);
        assert!(x[1].abs() < 1e-8);
        assert!(best.abs() < 1e-10);
    }

    #[test]
    fn numerical_hessian_matches_analytic_on_quadratic() {
        // f(x, y) = -((x - 1)^2 + 3 (y + 2)^2 + 0.5 x y).
        // Analytic Hessian = [[-2, -0.5], [-0.5, -6]].
        let f =
            |x: &[f64]| -((x[0] - 1.0).powi(2) + 3.0 * (x[1] + 2.0).powi(2) + 0.5 * x[0] * x[1]);
        let h = numerical_hessian(&[1.0, -2.0], 1e-4, f);
        assert!((h[(0, 0)] - (-2.0)).abs() < 1e-4);
        assert!((h[(1, 1)] - (-6.0)).abs() < 1e-4);
        assert!((h[(0, 1)] - (-0.5)).abs() < 1e-4);
        assert!((h[(1, 0)] - h[(0, 1)]).abs() < 1e-12);
    }
}
