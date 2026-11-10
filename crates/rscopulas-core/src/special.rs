//! Special functions on the hot paths of the crate: the standard normal and
//! Student t distributions.
//!
//! Every Gaussian or Student t kernel bottoms out in a quantile or CDF call,
//! so these are written for throughput:
//!
//! * the normal quantile is Wichura's AS 241 (`PPND16`), a branch plus one
//!   rational function, accurate to about `1e-16` relative;
//! * [`StudentT`] precomputes the two log-gamma values every density, CDF and
//!   quantile at a given `ν` needs (a generic implementation recomputes them
//!   on each call), evaluates the CDF with a continued fraction, and inverts
//!   it with Hill's closed-form approximation polished by Halley steps — two
//!   CDF evaluations per quantile instead of an iterative inverse-beta search.

const LN_SQRT_PI: f64 = 0.572_364_942_924_700_1; // ln Γ(1/2)

/// Standard normal CDF `Φ(x)` by Cody's rational Chebyshev approximations
/// (W. J. Cody, 1969, "Rational Chebyshev approximations for the error
/// function"; the algorithm behind R's `pnorm`). Relative accuracy is about
/// `1e-16` including far into the lower tail, where an `erfc`-based
/// evaluation through a generic implementation loses several digits.
// The coefficients are quoted to the digits of the published algorithm.
#[allow(clippy::excessive_precision)]
#[inline]
pub(crate) fn norm_cdf(x: f64) -> f64 {
    const A: [f64; 5] = [
        2.235_252_035_460_684,
        161.028_231_068_555_87,
        1_067.689_485_460_371,
        18_154.981_253_343_56,
        0.065_682_337_918_207_45,
    ];
    const B: [f64; 4] = [
        47.202_581_904_688_24,
        976.098_551_737_776_7,
        10_260.932_208_618_979,
        45_507.789_335_026_73,
    ];
    const C: [f64; 9] = [
        0.398_941_512_088_134_66,
        8.883_149_794_388_377,
        93.506_656_132_177_85,
        597.270_276_394_800_3,
        2_494.537_585_290_372_6,
        6_848.190_450_536_283,
        11_602.651_437_647_35,
        9_842.714_838_383_978,
        1.076_557_677_372_019_2e-8,
    ];
    const D: [f64; 8] = [
        22.266_688_044_328_117,
        235.387_901_782_625,
        1_519.377_599_407_554_7,
        6_485.558_298_266_761,
        18_615.571_640_885_097,
        34_900.952_721_145_98,
        38_912.003_286_093_27,
        19_685.429_676_859_992,
    ];
    const P: [f64; 6] = [
        0.215_898_534_057_957,
        0.127_401_161_160_247_36,
        0.022_235_277_870_649_807,
        0.001_421_619_193_227_893_4,
        2.911_287_495_116_879_3e-5,
        0.023_073_441_764_940_174,
    ];
    const Q: [f64; 5] = [
        1.284_260_096_144_911,
        0.468_238_212_480_865_1,
        0.065_988_137_868_928_56,
        0.003_782_396_332_027_582_4,
        7.297_515_550_839_662e-5,
    ];
    const FRAC_1_SQRT_2PI: f64 = 0.398_942_280_401_432_7;

    if x.is_nan() {
        return f64::NAN;
    }
    let y = x.abs();
    if y <= 0.674_489_75 {
        let (mut num, mut den) = (0.0, 0.0);
        if y > 1.11e-16 {
            let sq = x * x;
            num = A[4] * sq;
            den = sq;
            for i in 0..3 {
                num = (num + A[i]) * sq;
                den = (den + B[i]) * sq;
            }
        }
        return 0.5 + x * (num + A[3]) / (den + B[3]);
    }
    // Lower-tail mass of −|x|; the split exponential keeps exp(−y²/2)
    // accurate when y² is not exactly representable.
    let lower = if y <= 5.656_854_249_492_381 {
        let mut num = C[8] * y;
        let mut den = y;
        for i in 0..7 {
            num = (num + C[i]) * y;
            den = (den + D[i]) * y;
        }
        let ratio = (num + C[7]) / (den + D[7]);
        let trunc = (y * 16.0).trunc() / 16.0;
        let delta = (y - trunc) * (y + trunc);
        (-trunc * trunc * 0.5).exp() * (-delta * 0.5).exp() * ratio
    } else if y < 38.6 {
        let sq = 1.0 / (x * x);
        let mut num = P[5] * sq;
        let mut den = sq;
        for i in 0..4 {
            num = (num + P[i]) * sq;
            den = (den + Q[i]) * sq;
        }
        let ratio = (FRAC_1_SQRT_2PI - sq * (num + P[4]) / (den + Q[4])) / y;
        let trunc = (y * 16.0).trunc() / 16.0;
        let delta = (y - trunc) * (y + trunc);
        (-trunc * trunc * 0.5).exp() * (-delta * 0.5).exp() * ratio
    } else {
        0.0
    };
    if x > 0.0 { 1.0 - lower } else { lower }
}

/// The standard normal distribution as a value, for call sites that pass a
/// distribution around.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct StdNormal;

impl StdNormal {
    #[inline]
    pub(crate) fn cdf(&self, x: f64) -> f64 {
        norm_cdf(x)
    }

    #[inline]
    pub(crate) fn inverse_cdf(&self, p: f64) -> f64 {
        norm_ppf(p)
    }
}

/// Standard normal quantile `Φ⁻¹(p)` by Wichura's algorithm AS 241 (PPND16).
///
/// Returns `-inf` at 0, `+inf` at 1 and NaN outside `[0, 1]`.
// The coefficients are quoted to the digits of the published algorithm.
#[allow(clippy::excessive_precision)]
#[inline]
pub(crate) fn norm_ppf(p: f64) -> f64 {
    let q = p - 0.5;
    if q.abs() <= 0.425 {
        let r = 0.180625 - q * q;
        let num = ((((((2.509_080_928_730_122_7e3 * r + 3.343_057_558_358_813e4) * r
            + 6.726_577_092_700_87e4)
            * r
            + 4.592_195_393_154_987e4)
            * r
            + 1.373_169_376_550_946_1e4)
            * r
            + 1.971_590_950_306_551_3e3)
            * r
            + 1.331_416_678_917_843_8e2)
            * r
            + 3.387_132_872_796_366_5;
        let den = ((((((5.226_495_278_852_854_5e3 * r + 2.872_908_573_572_194_3e4) * r
            + 3.930_789_580_009_271e4)
            * r
            + 2.121_379_430_158_659_7e4)
            * r
            + 5.394_196_021_424_751e3)
            * r
            + 6.871_870_074_920_579e2)
            * r
            + 4.231_333_070_160_091e1)
            * r
            + 1.0;
        return q * num / den;
    }
    if !(0.0..=1.0).contains(&p) {
        return f64::NAN;
    }
    let tail = if q < 0.0 { p } else { 1.0 - p };
    if tail == 0.0 {
        return if q < 0.0 {
            f64::NEG_INFINITY
        } else {
            f64::INFINITY
        };
    }
    let mut r = (-tail.ln()).sqrt();
    let value = if r <= 5.0 {
        r -= 1.6;
        let num = ((((((7.745_450_142_783_414e-4 * r + 2.272_384_498_926_918_4e-2) * r
            + 2.417_807_251_774_506e-1)
            * r
            + 1.270_458_252_452_368_4)
            * r
            + 3.647_848_324_763_204_5)
            * r
            + 5.769_497_221_460_691)
            * r
            + 4.630_337_846_156_546)
            * r
            + 1.423_437_110_749_683_5;
        let den = ((((((1.050_750_071_644_416_9e-9 * r + 5.475_938_084_995_345e-4) * r
            + 1.519_866_656_361_645_7e-2)
            * r
            + 1.481_039_764_274_800_8e-1)
            * r
            + 6.897_673_349_851e-1)
            * r
            + 1.676_384_830_183_803_8)
            * r
            + 2.053_191_626_637_759)
            * r
            + 1.0;
        num / den
    } else {
        r -= 5.0;
        let num = ((((((2.010_334_399_292_288_1e-7 * r + 2.711_555_568_743_487_6e-5) * r
            + 1.242_660_947_388_078_4e-3)
            * r
            + 2.653_218_952_657_612_4e-2)
            * r
            + 2.965_605_718_285_048_7e-1)
            * r
            + 1.784_826_539_917_291_3)
            * r
            + 5.463_784_911_164_114)
            * r
            + 6.657_904_643_501_103_8;
        let den = ((((((2.044_263_103_389_939_7e-15 * r + 1.421_511_758_316_446e-7) * r
            + 1.846_318_317_510_054_8e-5)
            * r
            + 7.868_691_311_456_133e-4)
            * r
            + 1.487_536_129_085_061_5e-2)
            * r
            + 1.369_298_809_227_358e-1)
            * r
            + 5.998_322_065_558_88e-1)
            * r
            + 1.0;
        num / den
    };
    if q < 0.0 { -value } else { value }
}

/// Continued fraction of the regularized incomplete beta function (modified
/// Lentz). Converges quickly for `x < (a + 1) / (a + b + 2)`.
#[inline]
fn beta_continued_fraction(a: f64, b: f64, x: f64) -> f64 {
    const TINY: f64 = 1e-300;
    const EPS: f64 = 1e-16;
    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;
    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < TINY {
        d = TINY;
    }
    d = 1.0 / d;
    let mut h = d;
    for m in 1..=500 {
        let m = m as f64;
        let m2 = 2.0 * m;
        let aa = m * (b - m) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if d.abs() < TINY {
            d = TINY;
        }
        c = 1.0 + aa / c;
        if c.abs() < TINY {
            c = TINY;
        }
        d = 1.0 / d;
        h *= d * c;
        let aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if d.abs() < TINY {
            d = TINY;
        }
        c = 1.0 + aa / c;
        if c.abs() < TINY {
            c = TINY;
        }
        d = 1.0 / d;
        let delta = d * c;
        h *= delta;
        if (delta - 1.0).abs() < EPS {
            break;
        }
    }
    h
}

/// `ln[Γ(a + ½) / Γ(a)]` for `a > 0`, to a few ulps.
///
/// Differencing two log-gamma values loses ~1e-13 here (each is accurate to
/// an ulp of a number in the tens). Instead `a` is shifted above 15 with
/// `Γ(a + 1) = a Γ(a)` and the Stirling-series difference
/// `½ ln a − 1/(8a) + 1/(192a³) − 1/(640a⁵) + …` is summed there, where six
/// terms are exact to double precision.
fn ln_gamma_ratio_half(a: f64) -> f64 {
    let mut shifted = a;
    let mut numerator = 1.0_f64;
    let mut denominator = 1.0_f64;
    while shifted < 15.0 {
        numerator *= shifted;
        denominator *= shifted + 0.5;
        shifted += 1.0;
    }
    let inv = 1.0 / shifted;
    let inv2 = inv * inv;
    let series = inv
        * (-1.0 / 8.0
            + inv2
                * (1.0 / 192.0
                    + inv2
                        * (-1.0 / 640.0
                            + inv2
                                * (17.0 / 14336.0
                                    + inv2 * (-31.0 / 18432.0 + inv2 * (691.0 / 180224.0))))));
    0.5 * shifted.ln() + series + (numerator / denominator).ln()
}

/// Student t distribution with `ν` degrees of freedom, with the `ν`-only
/// constants precomputed.
#[derive(Debug, Clone, Copy)]
pub(crate) struct StudentT {
    nu: f64,
    /// `ln[Γ((ν+1)/2) / Γ(ν/2)]`.
    ln_ratio: f64,
    /// `ln Γ((ν+1)/2) − ln Γ(ν/2) − ½ ln(νπ)`: log of the density at zero.
    ln_norm: f64,
    /// `ln B(ν/2, 1/2)`.
    ln_beta: f64,
}

impl StudentT {
    /// `nu` must be finite and positive.
    pub(crate) fn new(nu: f64) -> Self {
        Self::from_ln_ratio(nu, ln_gamma_ratio_half(0.5 * nu))
    }

    fn from_ln_ratio(nu: f64, ln_ratio: f64) -> Self {
        Self {
            nu,
            ln_ratio,
            ln_norm: ln_ratio - 0.5 * (nu * std::f64::consts::PI).ln(),
            ln_beta: LN_SQRT_PI - ln_ratio,
        }
    }

    /// The distribution with `ν + 1` degrees of freedom (the conditional
    /// margin of a bivariate t). `Γ(ν/2 + 1) / Γ(ν/2 + ½) = (ν/2) / ratio`,
    /// so no further special-function call is needed.
    pub(crate) fn with_one_more_df(&self) -> Self {
        Self::from_ln_ratio(self.nu + 1.0, (0.5 * self.nu).ln() - self.ln_ratio)
    }

    #[inline]
    pub(crate) fn nu(&self) -> f64 {
        self.nu
    }

    #[inline]
    pub(crate) fn ln_pdf(&self, x: f64) -> f64 {
        self.ln_norm - 0.5 * (self.nu + 1.0) * (x * x / self.nu).ln_1p()
    }

    /// `(P(T > |x|), P(0 < T < |x|))`, each computed without cancellation in
    /// the regime where it is small.
    #[inline]
    fn tail_and_half_centre(&self, x: f64) -> (f64, f64) {
        let nu = self.nu;
        let x2 = x * x;
        if !x2.is_finite() {
            return if x.is_nan() {
                (f64::NAN, f64::NAN)
            } else {
                (0.0, 0.5)
            };
        }
        let denom = nu + x2;
        let w = x2 / denom; // argument of I_w(1/2, ν/2)
        let h = nu / denom; // argument of I_h(ν/2, 1/2); w + h = 1
        // ln[w^(1/2) h^(ν/2) / B(ν/2, 1/2)], shared by both representations.
        let ln_front = 0.5 * w.ln() + 0.5 * nu * h.ln() - self.ln_beta;
        if w * (nu + 5.0) < 3.0 {
            // Centre: I_w(1/2, ν/2) directly.
            if w == 0.0 {
                return (0.5, 0.0);
            }
            let centre = ln_front.exp() * beta_continued_fraction(0.5, 0.5 * nu, w) / 0.5;
            let half = 0.5 * centre;
            (0.5 - half, half)
        } else {
            let tail_two = ln_front.exp() * beta_continued_fraction(0.5 * nu, 0.5, h) / (0.5 * nu);
            let tail = 0.5 * tail_two;
            (tail, 0.5 - tail)
        }
    }

    #[inline]
    pub(crate) fn cdf(&self, x: f64) -> f64 {
        let (tail, half_centre) = self.tail_and_half_centre(x);
        if x <= 0.0 {
            tail
        } else if tail < 0.25 {
            1.0 - tail
        } else {
            0.5 + half_centre
        }
    }

    /// Hill's (1970, CACM algorithm 396) approximation to the positive `t`
    /// with two-sided tail probability `two_tail`.
    fn hill_two_tail(&self, two_tail: f64) -> f64 {
        let n = self.nu;
        if n < 1.0 {
            // Outside Hill's validity range: start from the exact Cauchy
            // quantile raised to the tail power; Halley polishing does the
            // rest.
            let cauchy = (two_tail * std::f64::consts::FRAC_PI_2).tan().recip();
            return cauchy.max(1e-300).powf(1.0 / n).max(cauchy);
        }
        if n == 1.0 {
            return (two_tail * std::f64::consts::FRAC_PI_2).tan().recip();
        }
        if n == 2.0 {
            return (2.0 / (two_tail * (2.0 - two_tail)) - 2.0).sqrt();
        }
        let a = 1.0 / (n - 0.5);
        let b = 48.0 / (a * a);
        let mut c = ((20700.0 * a / b - 98.0) * a - 16.0) * a + 96.36;
        let d = ((94.5 / (b + c) - 3.0) / b + 1.0) * (a * std::f64::consts::FRAC_PI_2).sqrt() * n;
        let mut y = (d * two_tail).powf(2.0 / n);
        if y > 0.05 + a {
            // Asymptotic inverse expansion about the normal quantile.
            let x = norm_ppf(0.5 * two_tail);
            y = x * x;
            if n < 5.0 {
                c += 0.3 * (n - 4.5) * (x + 0.6);
            }
            c += (((0.05 * d * x - 5.0) * x - 7.0) * x - 2.0) * x + b;
            y = (((((0.4 * y + 6.3) * y + 36.0) * y + 94.5) / c - y - 3.0) / b + 1.0) * x;
            y = a * y * y;
            y = if y > 0.002 {
                y.exp_m1()
            } else {
                0.5 * y * y + y
            };
        } else {
            y = ((1.0 / (((n + 6.0) / (n * y) - 0.089 * d - 0.822) * (n + 2.0) * 3.0)
                + 0.5 / (n + 4.0))
                * y
                - 1.0)
                * (n + 1.0)
                / (n + 2.0)
                + 1.0 / y;
        }
        (n * y).sqrt()
    }

    /// Quantile function. Returns `±inf` at the end points and NaN outside
    /// `[0, 1]`.
    pub(crate) fn ppf(&self, p: f64) -> f64 {
        if !(0.0..=1.0).contains(&p) {
            return f64::NAN;
        }
        if p == 0.5 {
            return 0.0;
        }
        let upper = p > 0.5;
        let q = if upper { 1.0 - p } else { p }; // tail mass, in (0, 0.5)
        if q == 0.0 {
            return if upper {
                f64::INFINITY
            } else {
                f64::NEG_INFINITY
            };
        }
        let nu = self.nu;
        // Solve on whichever of the tail / centre masses is the small one, so
        // the residual keeps full relative precision.
        let use_tail = q < 0.25;
        let centre_target = 0.5 - q;
        let mut x = self.hill_two_tail(2.0 * q);
        if !x.is_finite() || x <= 0.0 {
            x = 1.0;
        }
        // Bracket maintained for the bisection fallback.
        let (mut low, mut high) = (0.0_f64, f64::INFINITY);
        let mut converged = false;
        for _ in 0..60 {
            let (tail, half_centre) = self.tail_and_half_centre(x);
            // g is increasing in x in both formulations.
            let g = if use_tail {
                q - tail
            } else {
                half_centre - centre_target
            };
            if g == 0.0 {
                converged = true;
                break;
            }
            if g < 0.0 {
                low = x;
            } else {
                high = x;
            }
            let density = self.ln_pdf(x).exp();
            let newton = g / density;
            // Halley correction: g' = f and g'' = f' = −f (ν+1) x / (ν + x²).
            let curvature = (nu + 1.0) * x / (nu + x * x);
            let halley = newton / (1.0 + 0.5 * newton * curvature).max(0.5);
            let mut next = x - halley;
            if next == x {
                // The correction is below the resolution of `x`.
                converged = true;
                break;
            }
            if !(next > low && next < high) {
                // Step left the bracket: bisect (geometrically when the
                // bracket is still open-ended).
                next = if high.is_finite() {
                    0.5 * (low + high)
                } else {
                    2.0 * x
                };
            }
            let change = (next - x).abs();
            x = next;
            // Third-order convergence: once a step is this small the next
            // one would be below rounding error.
            if change <= 1e-7 * x.abs() {
                converged = true;
                break;
            }
        }
        if !converged {
            x = self.ppf_by_bisection(q, use_tail, centre_target, low, high);
        }
        if upper { x } else { -x }
    }

    /// Geometric bisection on `[low, high]`, used only when the Halley
    /// iteration fails to settle (extreme tails of very heavy-tailed `ν`).
    /// Returns `inf` when the quantile exceeds the largest finite `f64`.
    #[cold]
    fn ppf_by_bisection(
        &self,
        q: f64,
        use_tail: bool,
        centre_target: f64,
        mut low: f64,
        mut high: f64,
    ) -> f64 {
        let residual = |x: f64| {
            let (tail, half_centre) = self.tail_and_half_centre(x);
            if use_tail {
                q - tail
            } else {
                half_centre - centre_target
            }
        };
        if !high.is_finite() {
            high = low.max(1.0);
            while residual(high) < 0.0 {
                low = high;
                high *= 1e8;
                if !high.is_finite() {
                    return f64::INFINITY;
                }
            }
        }
        for _ in 0..2200 {
            let mid = if low > 0.0 {
                (low * high).sqrt()
            } else {
                0.5 * (low + high)
            };
            if !(mid > low && mid < high) {
                break;
            }
            if residual(mid) < 0.0 {
                low = mid;
            } else {
                high = mid;
            }
        }
        0.5 * (low + high)
    }
}

#[cfg(test)]
mod tests {
    use statrs::distribution::{Continuous, ContinuousCDF, Normal, StudentsT};

    use super::*;

    fn probability_grid() -> Vec<f64> {
        let mut grid = vec![
            1e-300, 1e-100, 1e-20, 1e-15, 1e-12, 1e-9, 1e-6, 1e-4, 1e-3, 0.01, 0.02425, 0.05,
            0.075, 0.1, 0.2, 0.25, 0.3, 0.4, 0.45, 0.49, 0.499999, 0.5,
        ];
        let upper: Vec<f64> = grid.iter().map(|p| 1.0 - p).filter(|p| *p < 1.0).collect();
        grid.extend(upper);
        grid.extend((1..200).map(|i| i as f64 / 200.0));
        grid
    }

    #[test]
    fn normal_quantile_matches_statrs_and_round_trips() {
        let normal = Normal::new(0.0, 1.0).unwrap();
        for p in probability_grid() {
            let z = norm_ppf(p);
            let reference = normal.inverse_cdf(p);
            assert!(
                (z - reference).abs() <= 1e-13 * (1.0 + reference.abs()),
                "p {p}: {z} vs statrs {reference}"
            );
            if p > 1e-290 {
                let back = norm_cdf(z);
                assert!(
                    (back - p).abs() <= 4e-16 + 1e-13 * p.min(1.0 - p),
                    "p {p}: Φ(Φ⁻¹) = {back}"
                );
            }
        }
        assert_eq!(norm_ppf(0.0), f64::NEG_INFINITY);
        assert_eq!(norm_ppf(1.0), f64::INFINITY);
        assert!(norm_ppf(-0.1).is_nan() && norm_ppf(1.1).is_nan() && norm_ppf(f64::NAN).is_nan());
        assert_eq!(norm_ppf(0.5), 0.0);
    }

    #[test]
    fn normal_cdf_agrees_with_statrs_to_its_accuracy() {
        // statrs evaluates Φ through an erfc that is good to ~1e-10 relative
        // in the tails; the scipy reference test below is the sharp one.
        let normal = Normal::new(0.0, 1.0).unwrap();
        for i in -400..=400 {
            let x = i as f64 / 40.0;
            let (got, reference) = (norm_cdf(x), normal.cdf(x));
            assert!(
                (got - reference).abs() <= 1e-15 + 1e-9 * reference.min(1.0 - reference),
                "cdf at {x}: {got} vs {reference}"
            );
        }
        assert_eq!(norm_cdf(0.0), 0.5);
        assert_eq!(norm_cdf(-40.0), 0.0);
        assert_eq!(norm_cdf(40.0), 1.0);
        assert!(norm_cdf(f64::NAN).is_nan());
    }

    const NUS: [f64; 14] = [
        0.3, 0.7, 1.0, 1.5, 2.0, 2.05, 2.5, 3.0, 4.0, 7.3, 15.0, 30.0, 200.0, 1000.0,
    ];

    #[test]
    fn student_t_density_matches_statrs() {
        for nu in NUS {
            let ours = StudentT::new(nu);
            let reference = StudentsT::new(0.0, 1.0, nu).unwrap();
            for i in -60..=60 {
                let x = (i as f64 / 6.0).powi(3);
                let (a, b) = (ours.ln_pdf(x), reference.ln_pdf(x));
                assert!(
                    (a - b).abs() <= 1e-12 * (1.0 + b.abs()),
                    "nu {nu} x {x}: {a} vs {b}"
                );
            }
        }
    }

    #[test]
    fn student_t_cdf_matches_statrs() {
        for nu in NUS {
            let ours = StudentT::new(nu);
            let reference = StudentsT::new(0.0, 1.0, nu).unwrap();
            for i in -80..=80 {
                let x = (i as f64 / 8.0).powi(3);
                let (a, b) = (ours.cdf(x), reference.cdf(x));
                assert!(
                    (a - b).abs() <= 1e-13 + 1e-11 * b.min(1.0 - b),
                    "nu {nu} x {x}: {a} vs {b}"
                );
            }
        }
    }

    #[test]
    fn student_t_quantile_inverts_its_own_cdf_to_full_precision() {
        for nu in NUS {
            let dist = StudentT::new(nu);
            for p in probability_grid() {
                // Below this the quantile of a heavy-tailed t overflows f64.
                if p.min(1.0 - p) < if nu < 2.0 { 1e-20 } else { 1e-250 } {
                    continue;
                }
                let x = dist.ppf(p);
                assert!(x.is_finite(), "nu {nu} p {p}: {x}");
                let back = dist.cdf(x);
                let tail = p.min(1.0 - p);
                assert!(
                    (back - p).abs() <= 4e-16 + 1e-12 * tail,
                    "nu {nu} p {p}: x {x} cdf(x) {back}"
                );
            }
            assert_eq!(dist.ppf(0.5), 0.0);
            assert_eq!(dist.ppf(0.0), f64::NEG_INFINITY);
            assert_eq!(dist.ppf(1.0), f64::INFINITY);
            assert!(dist.ppf(1.5).is_nan());
        }
    }

    #[test]
    fn student_t_quantile_is_antisymmetric_and_monotone() {
        for nu in NUS {
            let dist = StudentT::new(nu);
            let mut previous = f64::NEG_INFINITY;
            for i in 1..400 {
                let p = i as f64 / 400.0;
                let x = dist.ppf(p);
                assert!(x > previous, "nu {nu}: not monotone at p {p}");
                previous = x;
            }
            // 1 − p is exact for these, so both sides solve the same problem.
            for p in [0.0625, 0.125, 0.25, 0.375] {
                assert_eq!(dist.ppf(p), -dist.ppf(1.0 - p), "nu {nu} p {p}");
            }
        }
    }

    #[test]
    fn matches_scipy_reference_values() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../fixtures/reference/special/student_t_scipy.json"
        );
        let fixture: serde_json::Value =
            serde_json::from_slice(&std::fs::read(path).expect("fixture should exist"))
                .expect("fixture should deserialize");
        let floats = |value: &serde_json::Value| -> Vec<f64> {
            value
                .as_array()
                .expect("array")
                .iter()
                .map(|x| x.as_f64().expect("number"))
                .collect()
        };
        let probabilities = floats(&fixture["probabilities"]);
        let points = floats(&fixture["points"]);
        for row in fixture["student_t"].as_array().expect("rows") {
            let nu = row["nu"].as_f64().unwrap();
            let dist = StudentT::new(nu);
            for (p, expected) in probabilities.iter().zip(floats(&row["ppf"])) {
                let got = dist.ppf(*p);
                // SciPy itself is only good to ~4e-13 here (see the
                // closed-form test below for the sharper check).
                assert!(
                    (got - expected).abs() <= 1e-12 * expected.abs() + 1e-15,
                    "ppf nu {nu} p {p}: {got} vs scipy {expected}"
                );
            }
            for (x, expected) in points.iter().zip(floats(&row["cdf"])) {
                let got = dist.cdf(*x);
                assert!(
                    (got - expected).abs() <= 2e-13 * expected.min(1.0 - expected) + 3e-16,
                    "cdf nu {nu} x {x}: {got} vs scipy {expected}"
                );
            }
            for (x, expected) in points.iter().zip(floats(&row["logpdf"])) {
                let got = dist.ln_pdf(*x);
                assert!(
                    (got - expected).abs() <= 1e-12 * (1.0 + expected.abs()),
                    "ln_pdf nu {nu} x {x}: {got} vs scipy {expected}"
                );
            }
        }
        // Exact values for even ν, where the CDF is a finite trigonometric sum
        // (evaluated to 60 digits when the fixture was generated).
        for row in fixture["student_t_exact"].as_array().expect("rows") {
            let nu = row["nu"].as_f64().unwrap();
            let dist = StudentT::new(nu);
            for (x, expected) in floats(&row["x"]).into_iter().zip(floats(&row["cdf"])) {
                let got = dist.cdf(x);
                assert!(
                    (got - expected).abs() <= 2e-14 * expected.min(1.0 - expected) + 2e-16,
                    "exact cdf nu {nu} x {x}: {got} vs {expected}"
                );
            }
            for (p, expected) in floats(&row["p"]).into_iter().zip(floats(&row["ppf"])) {
                let got = dist.ppf(p);
                assert!(
                    (got - expected).abs() <= 2e-14 * expected.abs(),
                    "exact ppf nu {nu} p {p}: {got} vs {expected}"
                );
            }
        }
        let cdf_points = floats(&fixture["normal"]["cdf_points"]);
        for (x, expected) in cdf_points.iter().zip(floats(&fixture["normal"]["cdf"])) {
            let got = norm_cdf(*x);
            assert!(
                (got - expected).abs() <= 4e-15 * expected.min(1.0 - expected) + 2e-16,
                "norm_cdf x {x}: {got} vs scipy {expected}"
            );
        }
        for (p, expected) in probabilities.iter().zip(floats(&fixture["normal"]["ppf"])) {
            let got = norm_ppf(*p);
            assert!(
                (got - expected).abs() <= 4e-15 * (1.0 + expected.abs()),
                "norm_ppf p {p}: {got} vs scipy {expected}"
            );
        }
    }

    #[test]
    fn student_t_quantile_matches_closed_forms() {
        // ν = 1 (Cauchy) and ν = 2 have elementary quantile functions; the
        // ν = 4 value was computed to 50 digits from its closed-form CDF.
        let cauchy = StudentT::new(1.0);
        let two = StudentT::new(2.0);
        for p in [1e-9, 1e-4, 0.03, 0.2, 0.37, 0.49, 0.62, 0.9, 0.9999] {
            // −cot(πp), written to avoid cancelling p against ½.
            let exact = if p < 0.5 {
                -1.0 / (std::f64::consts::PI * p).tan()
            } else {
                1.0 / (std::f64::consts::PI * (1.0 - p)).tan()
            };
            let got = cauchy.ppf(p);
            assert!(
                (got - exact).abs() <= 1e-13 * exact.abs(),
                "cauchy p {p}: {got} vs {exact}"
            );
            let exact = (2.0 * p - 1.0) / (2.0 * p * (1.0 - p)).sqrt();
            let got = two.ppf(p);
            assert!(
                (got - exact).abs() <= 1e-13 * exact.abs(),
                "nu 2 p {p}: {got} vs {exact}"
            );
        }
        let got = StudentT::new(4.0).ppf(0.49);
        let exact = -0.026_670_618_302_639_03;
        assert!((got - exact).abs() <= 1e-14 * exact.abs(), "nu 4: {got}");
    }

    #[test]
    fn one_more_df_matches_direct_construction() {
        for nu in NUS {
            let derived = StudentT::new(nu).with_one_more_df();
            let direct = StudentT::new(nu + 1.0);
            assert!(
                (derived.ln_norm - direct.ln_norm).abs() <= 1e-14 * (1.0 + direct.ln_norm.abs())
            );
            assert!(
                (derived.ln_beta - direct.ln_beta).abs() <= 1e-14 * (1.0 + direct.ln_beta.abs())
            );
            assert_eq!(derived.nu(), nu + 1.0);
        }
    }

    #[test]
    fn gamma_ratio_matches_closed_forms() {
        // Γ(1)/Γ(½) = 1/√π and Γ(n + ½)/Γ(n) follows by recurrence.
        let mut exact = -LN_SQRT_PI; // a = ½
        let mut a = 0.5_f64;
        for _ in 0..80 {
            let got = ln_gamma_ratio_half(a);
            assert!(
                (got - exact).abs() <= 4e-15 * (1.0 + exact.abs()),
                "a {a}: {got} vs {exact}"
            );
            // ratio(a + ½) = ln a − ratio(a)
            exact = a.ln() - exact;
            a += 0.5;
        }
    }
}
