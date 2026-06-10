//! Implied volatility solver.
//!
//! Implements Jäckel's "Let's Be Rational" algorithm, which combines a
//! rational-cubic initial guess (segmented in four log-moneyness branches)
//! with Householder order-4 iteration to reach f64 precision in at most two
//! iterations across all interior inputs.
//!
//! References:
//!   - Jäckel, P. (2015). "Let's Be Rational", Wilmott Magazine, Nov 2015.
//!   - Delbourgo, R., Gregory, J. A. (1985). "Shape preserving piecewise
//!     rational interpolation", SIAM J. Sci. Stat. Comput.
//!   - Abramowitz, M., Stegun, I. (1964) §26.2.12 (asymptotic series).
//!   - Marsaglia, G. (2004). "Evaluating the Normal Distribution",
//!     J. Statistical Software 11(4) (motivation for the `erfcx` formulation).

use crate::bsm::Flag;

mod normalised_black {
    //! Normalised Black call: `b(x, s) = Φ(h+t)·exp(x/2) - Φ(h-t)·exp(-x/2)`
    //! with `h = x/s`, `t = s/2`, `x = ln(F/K)`, `s = σ·√T`.
    //!
    //! The undiscounted BSM call price is `√(F·K) · b(x, s) + intrinsic`.
    //!
    //! Four evaluation regions are dispatched by `call_and_vega` (Jäckel §3.1)
    //! to keep `b` accurate to ~1 ULP across all moneyness/vol combinations:
    //!   1. asymptotic series (very negative x, small t — divergent A&S 26.2.12
    //!      truncated at 17th order in q = (h/r)²)
    //!   2. small-t series (short expiry / low vol — 12th order in t²)
    //!   3. direct `Φ` evaluation (price dominated by first term)
    //!   4. `erfcx` formulation (default OTM tail, avoids catastrophic
    //!      cancellation between two near-equal `exp(x/2)·Φ(...)` terms)

    use crate::normal::{cdf, erfcx};

    const ONE_OVER_SQRT_TWO: f64 = std::f64::consts::FRAC_1_SQRT_2;
    const ONE_OVER_SQRT_TWO_PI: f64 = 0.398_942_280_401_432_7;

    /// `η` in Jäckel §3.1.2: x < s·η triggers the asymptotic-series region.
    const ASYMPTOTIC_EXPANSION_THRESHOLD: f64 = -10.0;

    /// `τ` in Jäckel §3.1.3: t < τ triggers the small-t-series region.
    /// `τ = 2·ε^(1/16)`; in f64 this is 2·(2⁻⁵²)^(1/16) = 2·2⁻³·²⁵ ≈ 0.21025.
    const SMALL_T_THRESHOLD: f64 = 0.210_252_854_115_171_3;

    /// `98·ε^(1/4)` ≈ 0.0120: the Taylor-vs-direct switchover for `2·sinh(x/2)`
    /// in `normalised_intrinsic`. The factor 98 = √√92897280 (last series
    /// coefficient) makes the truncation error match f64 precision.
    const INTRINSIC_TAYLOR_THRESHOLD: f64 = 0.011_962_890_625;

    /// Normalised intrinsic value.  `q = +1` for calls, `q = -1` for puts.
    ///
    /// For OTM (q·x ≤ 0) returns 0.  For ITM uses `|2·sinh(x/2)|`, replaced
    /// by its Taylor series for small `|x|` to avoid catastrophic cancellation.
    pub fn intrinsic(x: f64, q: f64) -> f64 {
        if q * x <= 0.0 {
            return 0.0;
        }
        let x2 = x * x;
        if x2 < INTRINSIC_TAYLOR_THRESHOLD {
            // Series for x·(1 + x²/24 + x⁴/1920 + x⁶/322560 + x⁸/92897280)
            // equals 2·sinh(x/2) to f64 precision in this range.
            let body = x2.mul_add(
                x2.mul_add(
                    x2.mul_add(
                        x2.mul_add(1.0 / 92_897_280.0, 1.0 / 322_560.0),
                        1.0 / 1_920.0,
                    ),
                    1.0 / 24.0,
                ),
                1.0,
            );
            let value = q.signum() * x * body;
            return value.max(0.0);
        }
        let half = (0.5 * x).exp();
        (q.signum() * (half - 1.0 / half)).max(0.0)
    }

    /// Normalised vega `∂b/∂s = (1/√(2π))·exp(-½·(x²/s² + s²/4))`.
    ///
    /// Symmetric in x; vanishes for s = 0 except at x = 0 where the limit
    /// is `1/√(2π)·exp(-s²/8) → 1/√(2π)` as s → 0.
    pub fn vega(x: f64, s: f64) -> f64 {
        let ax = x.abs();
        if ax <= 0.0 {
            return ONE_OVER_SQRT_TWO_PI * (-0.125 * s * s).exp();
        }
        if s <= 0.0 || s <= ax * f64::MIN_POSITIVE.sqrt() {
            return 0.0;
        }
        let x_over_s = x / s;
        let half_s = 0.5 * s;
        ONE_OVER_SQRT_TWO_PI * (-0.5 * (x_over_s.mul_add(x_over_s, half_s * half_s))).exp()
    }

    /// Direct evaluation `b(x, s) = Φ(h+t)·exp(x/2) - Φ(h-t)·exp(-x/2)`.
    /// Accurate when `h + t > 0.85` (first term dominates).
    pub fn call_using_norm_cdf(x: f64, s: f64) -> f64 {
        let h = x / s;
        let t = 0.5 * s;
        let b_max = (0.5 * x).exp();
        let b = cdf(h + t) * b_max - cdf(h - t) / b_max;
        b.max(0.0)
    }

    /// `erfcx`-based combined `b` and `vega` evaluation:
    /// `b = ½·exp(-½(h²+t²))·[erfcx(-(h+t)/√2) - erfcx(-(h-t)/√2)]`,
    /// `vega = (1/√(2π))·exp(-½(h²+t²))`.
    /// Accurate in the deep-OTM tail (Jäckel §3.1.4; Marsaglia 2004). The
    /// `exp(-½(h²+t²))` factor is shared between the two quantities.
    fn call_and_vega_erfcx(h: f64, t: f64) -> (f64, f64) {
        let raw_exp = (-0.5 * (h * h + t * t)).exp();
        let b = 0.5
            * raw_exp
            * (erfcx(-ONE_OVER_SQRT_TWO * (h + t)) - erfcx(-ONE_OVER_SQRT_TWO * (h - t)));
        let v = ONE_OVER_SQRT_TWO_PI * raw_exp;
        (b.max(0.0), v)
    }

    /// Asymptotic-expansion combined `b` and `vega` (very negative x, small t).
    /// 17-order series in `q = (h/r)²` with `r = (h+t)(h-t)` and `e = (t/h)²`.
    /// Derived from A&S 26.2.12; truncation gives ~1.64e-16 relative for x ≤ -10.
    /// Shares the `(1/√(2π))·exp(-½(h²+t²))` factor between the two quantities.
    #[allow(clippy::too_many_lines)]
    fn call_and_vega_asymptotic(h: f64, t: f64) -> (f64, f64) {
        let e = (t / h) * (t / h);
        let r = (h + t) * (h - t);
        let q = (h / r) * (h / r);
        // The Horner nest below is the verbatim 17-term truncation of
        // [Y(h+t) - Y(h-t)] in q with each coefficient a polynomial in e.
        // Pasted as a single expression because hand-grouping it across lines
        // is bracket-mismatch-prone; rustfmt reflows it.
        #[rustfmt::skip]
        #[allow(clippy::neg_multiply, clippy::unreadable_literal)]
        let asymp_sum: f64 = 2.0+q*(-6.0e0-2.0*e+3.0*q*(1.0e1+e*(2.0e1+2.0*e)+5.0*q*(-1.4e1+e*(-7.0e1+e*(-4.2e1-2.0*e))+7.0*q*(1.8e1+e*(1.68e2+e*(2.52e2+e*(7.2e1+2.0*e)))+9.0*q*(-2.2e1+e*(-3.3e2+e*(-9.24e2+e*(-6.6e2+e*(-1.1e2-2.0*e))))+1.1e1*q*(2.6e1+e*(5.72e2+e*(2.574e3+e*(3.432e3+e*(1.43e3+e*(1.56e2+2.0*e)))))+1.3e1*q*(-3.0e1+e*(-9.1e2+e*(-6.006e3+e*(-1.287e4+e*(-1.001e4+e*(-2.73e3+e*(-2.1e2-2.0*e))))))+1.5e1*q*(3.4e1+e*(1.36e3+e*(1.2376e4+e*(3.8896e4+e*(4.862e4+e*(2.4752e4+e*(4.76e3+e*(2.72e2+2.0*e)))))))+1.7e1*q*(-3.8e1+e*(-1.938e3+e*(-2.3256e4+e*(-1.00776e5+e*(-1.84756e5+e*(-1.51164e5+e*(-5.4264e4+e*(-7.752e3+e*(-3.42e2-2.0*e))))))))+1.9e1*q*(4.2e1+e*(2.66e3+e*(4.0698e4+e*(2.3256e5+e*(5.8786e5+e*(7.05432e5+e*(4.0698e5+e*(1.08528e5+e*(1.197e4+e*(4.2e2+2.0*e)))))))))+2.1e1*q*(-4.6e1+e*(-3.542e3+e*(-6.7298e4+e*(-4.90314e5+e*(-1.63438e6+e*(-2.704156e6+e*(-2.288132e6+e*(-9.80628e5+e*(-2.01894e5+e*(-1.771e4+e*(-5.06e2-2.0*e))))))))))+2.3e1*q*(5.0e1+e*(4.6e3+e*(1.0626e5+e*(9.614e5+e*(4.08595e6+e*(8.9148e6+e*(1.04006e7+e*(6.53752e6+e*(2.16315e6+e*(3.542e5+e*(2.53e4+e*(6.0e2+2.0*e)))))))))))+2.5e1*q*(-5.4e1+e*(-5.85e3+e*(-1.6146e5+e*(-1.77606e6+e*(-9.37365e6+e*(-2.607579e7+e*(-4.01166e7+e*(-3.476772e7+e*(-1.687257e7+e*(-4.44015e6+e*(-5.9202e5+e*(-3.51e4+e*(-7.02e2-2.0*e))))))))))))+2.7e1*q*(5.8e1+e*(7.308e3+e*(2.3751e5+e*(3.12156e6+e*(2.003001e7+e*(6.919458e7+e*(1.3572783e8+e*(1.5511752e8+e*(1.0379187e8+e*(4.006002e7+e*(8.58429e6+e*(9.5004e5+e*(4.7502e4+e*(8.12e2+2.0*e)))))))))))))+2.9e1*q*(-6.2e1+e*(-8.99e3+e*(-3.39822e5+e*(-5.25915e6+e*(-4.032015e7+e*(-1.6934463e8+e*(-4.1250615e8+e*(-6.0108039e8+e*(-5.3036505e8+e*(-2.8224105e8+e*(-8.870433e7+e*(-1.577745e7+e*(-1.472562e6+e*(-6.293e4+e*(-9.3e2-2.0*e))))))))))))))+3.1e1*q*(6.6e1+e*(1.0912e4+e*(4.74672e5+e*(8.544096e6+e*(7.71342e7+e*(3.8707344e8+e*(1.14633288e9+e*(2.07431664e9+e*(2.33360622e9+e*(1.6376184e9+e*(7.0963464e8+e*(1.8512208e8+e*(2.7768312e7+e*(2.215136e6+e*(8.184e4+e*(1.056e3+2.0*e)))))))))))))))+3.3e1*(-7.0e1+e*(-1.309e4+e*(-6.49264e5+e*(-1.344904e7+e*(-1.4121492e8+e*(-8.344518e8+e*(-2.9526756e9+e*(-6.49588632e9+e*(-9.0751353e9+e*(-8.1198579e9+e*(-4.6399188e9+e*(-1.6689036e9+e*(-3.67158792e8+e*(-4.707164e7+e*(-3.24632e6+e*(-1.0472e5+e*(-1.19e3-2.0*e)))))))))))))))))*q))))))))))))))));
        let phi = ONE_OVER_SQRT_TWO_PI * (-0.5 * (h * h + t * t)).exp();
        let b = phi * (t / r) * asymp_sum;
        (b.max(0.0), phi)
    }

    /// Small-t combined `b` and `vega` (Jäckel §3.1.3). 12th-order in t² of
    /// `[Y(h+t) - Y(h-t)]` with `Y(z) = Φ(z)/φ(z)`. Accurate to f64 precision
    /// when `h ≤ 0` and `t < τ`. The leading factor `a = 1 + h·Y(h)` is the
    /// precision bottleneck for |h| > 1. Shares the `(1/√(2π))·exp(-½(h²+t²))`
    /// factor between the two quantities.
    fn call_and_vega_small_t(h: f64, t: f64) -> (f64, f64) {
        const SQRT_TWO_PI: f64 = 2.506_628_274_631_000_5;
        let a = 1.0 + h * (0.5 * SQRT_TWO_PI) * erfcx(-ONE_OVER_SQRT_TWO * h);
        let w = t * t;
        let h2 = h * h;
        // Expansion in w (= t²); coefficients are polynomials in (a, h²).
        let exp = 2.0
            * t
            * (a + w
                * ((-1.0 + 3.0 * a + a * h2) / 6.0
                    + w * ((-7.0 + 15.0 * a + h2 * (-1.0 + 10.0 * a + a * h2)) / 120.0
                        + w * ((-57.0
                            + 105.0 * a
                            + h2 * (-18.0 + 105.0 * a + h2 * (-1.0 + 21.0 * a + a * h2)))
                            / 5040.0
                            + w * ((-561.0
                                + 945.0 * a
                                + h2 * (-285.0
                                    + 1260.0 * a
                                    + h2 * (-33.0
                                        + 378.0 * a
                                        + h2 * (-1.0 + 36.0 * a + a * h2))))
                                / 362_880.0
                                + w * ((-6555.0
                                    + 10395.0 * a
                                    + h2 * (-4680.0
                                        + 17325.0 * a
                                        + h2 * (-840.0
                                            + 6930.0 * a
                                            + h2 * (-52.0
                                                + 990.0 * a
                                                + h2 * (-1.0 + 55.0 * a + a * h2)))))
                                    / 39_916_800.0
                                    + ((-89055.0
                                        + 135_135.0 * a
                                        + h2 * (-82_845.0
                                            + 270_270.0 * a
                                            + h2 * (-20370.0
                                                + 135_135.0 * a
                                                + h2 * (-1926.0
                                                    + 25740.0 * a
                                                    + h2 * (-75.0
                                                        + 2145.0 * a
                                                        + h2 * (-1.0
                                                            + 78.0 * a
                                                            + a * h2))))))
                                        * w)
                                        / 6_227_020_800.0))))));
        let phi = ONE_OVER_SQRT_TWO_PI * (-0.5 * (h * h + t * t)).exp();
        let b = phi * exp;
        (b.max(0.0), phi)
    }

    /// Thin `b`-only wrapper over [`call_and_vega`]. Retained for test
    /// ergonomics (mpmath goldens, finite-difference vega checks) where
    /// only `b` is needed. Production LBR routes through `call_and_vega`.
    #[cfg(test)]
    pub fn call(x: f64, s: f64) -> f64 {
        call_and_vega(x, s).0
    }

    /// Combined `b(x, s)` and `∂b/∂s = vega(x, s)` evaluated together.
    ///
    /// In three of the four regions (asymptotic, small-t, erfcx) the
    /// `(1/√(2π))·exp(-½(h²+t²))` factor appears in both quantities; this
    /// dispatcher routes through the `call_and_vega_*` variants so the
    /// `exp` is computed once per call instead of twice. Region 3
    /// (`call_using_norm_cdf`) uses `exp(x/2)` instead of the symmetric
    /// factor, so b and vega are computed independently there.
    ///
    /// Used by the LBR Householder iteration where every step needs both.
    pub fn call_and_vega(x: f64, s: f64) -> (f64, f64) {
        if x > 0.0 {
            // Call-put symmetry. Vega is symmetric in x (`vega(x,s) = vega(-x,s)`).
            let (b_neg, v) = call_and_vega(-x, s);
            return (intrinsic(x, 1.0) + b_neg, v);
        }
        if s <= 0.0 {
            return (intrinsic(x, 1.0), 0.0);
        }
        // Region 1: very negative x, small t — asymptotic series.
        if x < s * ASYMPTOTIC_EXPANSION_THRESHOLD
            && 0.5 * s * s + x < s * (SMALL_T_THRESHOLD + ASYMPTOTIC_EXPANSION_THRESHOLD)
        {
            return call_and_vega_asymptotic(x / s, 0.5 * s);
        }
        // Region 2: small t — short-expiry / low-vol series.
        if 0.5 * s < SMALL_T_THRESHOLD {
            return call_and_vega_small_t(x / s, 0.5 * s);
        }
        // Region 3: b dominated by first term — direct Φ evaluation. The
        // cdf form uses `exp(x/2)` rather than `exp(-½(h²+t²))`, so b and
        // vega share no factor here. Fall back to the two-call form.
        if x + 0.5 * s * s > s * 0.85 {
            return (call_using_norm_cdf(x, s), vega(x, s));
        }
        // Region 4: default OTM tail — erfcx.
        call_and_vega_erfcx(x / s, 0.5 * s)
    }
}

/// Normalised Black call `b(x, s)` with `x = ln(F/K)`, `s = σ·√T`, exposed for
/// `bsm::price`. The undiscounted BSM call is `√(F·K)·b(x, s)`; backing the
/// public pricer with this ~1-ULP engine (the same one the IV solver uses)
/// replaces the cancellation-prone `S·Φ(d1) − K·Φ(d2)` textbook form, which
/// keeps only ~9 digits in the deep-OTM short-expiry corner. Puts use the
/// reflection `b_put(x, s) = b_call(−x, s)`.
pub(crate) fn normalised_black_call(x: f64, s: f64) -> f64 {
    // `call_and_vega` is the production path (the test-only `call` wrapper takes
    // its `.0`); the vega is discarded here — pricing needs only `b`.
    normalised_black::call_and_vega(x, s).0
}

mod lbr {
    //! The Jäckel "Let's Be Rational" core: rational-cubic initial guess
    //! segmented in four log-moneyness branches, Householder order-4 iteration.
    //!
    //! Inputs are normalised throughout:
    //!   `x = ln(F/K)`,  `s = σ·√T`,  `β = price / √(F·K)`.
    //! The convention `q = ±1` (q=+1 call, q=-1 put) is preserved.

    use super::normalised_black;
    use crate::normal::{cdf, inverse_cdf};

    /// `1/sqrt(3)`.
    const SQRT_ONE_OVER_THREE: f64 = 0.577_350_269_189_625_7;
    /// `sqrt(3)`.
    const SQRT_THREE: f64 = 1.732_050_807_568_877_2;
    /// `sqrt(pi/2)`.
    const SQRT_PI_OVER_TWO: f64 = 1.253_314_137_315_500_3;
    /// `2*pi`.
    const TWO_PI: f64 = std::f64::consts::TAU;
    /// `pi/6`.
    const PI_OVER_SIX: f64 = std::f64::consts::FRAC_PI_6;
    /// `2*pi/sqrt(27)`.
    const TWO_PI_OVER_SQRT_TWENTY_SEVEN: f64 = 1.209_199_576_156_145_2;

    // ------------------------------------------------------------------
    //  Lower / upper maps used to build the initial guess in the outer
    //  segments (β < b_l and β > b_h).  These are smooth invertible
    //  re-parameterisations of (β, s) that the rational cubic can fit.
    // ------------------------------------------------------------------

    /// `f_lower(x, s) = (2π/√27)·|x|·Φ(-z)³`  with  `z = |x|/(s√3)`.
    /// Maps near-intrinsic β to a smooth function of s (Jäckel §3.4.1).
    #[cfg(test)]
    pub fn f_lower(x: f64, s: f64) -> f64 {
        if x.abs() <= 0.0 || s <= 0.0 {
            return 0.0;
        }
        let z = SQRT_ONE_OVER_THREE * x.abs() / s;
        let phi = cdf(-z);
        TWO_PI_OVER_SQRT_TWENTY_SEVEN * x.abs() * phi * phi * phi
    }

    /// Returns `(f, df/dβ, d²f/dβ²)` for the lower map.
    /// The derivatives are reformulated to avoid divide-by-zero at `s = 0`
    /// or `x = 0` (the `COMPUTE_LOWER_MAP_DERIVATIVES_INDIVIDUALLY` variant).
    pub fn f_lower_with_derivatives(x: f64, s: f64) -> (f64, f64, f64) {
        let ax = x.abs();
        let z = SQRT_ONE_OVER_THREE * ax / s;
        let y = z * z;
        let s2 = s * s;
        let phi = cdf(-z);
        let phi_pdf = (-0.5 * z * z).exp() / (2.0_f64 * std::f64::consts::PI).sqrt();
        let fpp = PI_OVER_SIX * y / (s2 * s)
            * phi
            * (8.0 * SQRT_THREE * s * ax + (3.0 * s2 * (s2 - 8.0) - 8.0 * x * x) * phi / phi_pdf)
            * (2.0 * y + 0.25 * s2).exp();
        let (f, fp) = if s <= 0.0 {
            (0.0, 1.0)
        } else {
            let phi2 = phi * phi;
            let fp = TWO_PI * y * phi2 * (y + 0.125 * s * s).exp();
            let f = if ax <= 0.0 {
                0.0
            } else {
                TWO_PI_OVER_SQRT_TWENTY_SEVEN * ax * (phi2 * phi)
            };
            (f, fp)
        };
        (f, fp, fpp)
    }

    /// Inverse of the lower map: given `f`, return `s`.
    pub fn inv_f_lower(x: f64, f: f64) -> f64 {
        if f <= 0.0 {
            return 0.0;
        }
        let cube = f / (TWO_PI_OVER_SQRT_TWENTY_SEVEN * x.abs());
        (x / (SQRT_THREE * inverse_cdf(cube.cbrt()))).abs()
    }

    /// `f_upper(s) = Φ(-s/2)`.  Maps β near `b_max` to a smooth function of s.
    #[cfg(test)]
    pub fn f_upper(s: f64) -> f64 {
        cdf(-0.5 * s)
    }

    /// Returns `(f, df/dβ, d²f/dβ²)` for the upper map.
    pub fn f_upper_with_derivatives(x: f64, s: f64) -> (f64, f64, f64) {
        let f = cdf(-0.5 * s);
        if x.abs() <= 0.0 {
            return (f, -0.5, 0.0);
        }
        let w = (x / s) * (x / s);
        let fp = -0.5 * (0.5 * w).exp();
        let fpp = SQRT_PI_OVER_TWO * (w + 0.125 * s * s).exp() * w / s;
        (f, fp, fpp)
    }

    /// Inverse of the upper map.
    pub fn inv_f_upper(f: f64) -> f64 {
        -2.0 * inverse_cdf(f)
    }

    // ------------------------------------------------------------------
    //  Shape-preserving rational cubic interpolation
    //  (Delbourgo & Gregory 1985, SIAM J. Sci. Stat. Comput.).
    //
    //  Used in all four branches of the initial guess to produce a
    //  smooth, monotonic, convex interpolant of (β, s) (or of mapped
    //  (β, f)) between the reference points.
    // ------------------------------------------------------------------

    const MIN_RATIONAL_CUBIC_CONTROL_PARAMETER: f64 = -0.999_999_985_069_5; // -(1 - sqrt(eps))
    const MAX_RATIONAL_CUBIC_CONTROL_PARAMETER: f64 = 4.054_944_344_523_184e31; // 2 / eps^2

    /// Rational-cubic interpolation. Formula (2.4)/(2.5) of Delbourgo-Gregory.
    pub fn rational_cubic(
        x: f64,
        x_l: f64,
        x_r: f64,
        y_l: f64,
        y_r: f64,
        d_l: f64,
        d_r: f64,
        r: f64,
    ) -> f64 {
        let h = x_r - x_l;
        if h.abs() <= 0.0 {
            return 0.5 * (y_l + y_r);
        }
        let t = (x - x_l) / h;
        if r >= MAX_RATIONAL_CUBIC_CONTROL_PARAMETER {
            // Degenerate to linear.
            return y_r * t + y_l * (1.0 - t);
        }
        let omt = 1.0 - t;
        let t2 = t * t;
        let omt2 = omt * omt;
        (y_r * t2 * t
            + (r * y_r - h * d_r) * t2 * omt
            + (r * y_l + h * d_l) * t * omt2
            + y_l * omt2 * omt)
            / (1.0 + (r - 3.0) * t * omt)
    }

    fn control_param_at_left(
        x_l: f64,
        x_r: f64,
        y_l: f64,
        y_r: f64,
        d_l: f64,
        d_r: f64,
        d2_l: f64,
    ) -> f64 {
        let h = x_r - x_l;
        let numerator = 0.5 * h * d2_l + (d_r - d_l);
        if numerator.abs() < f64::MIN_POSITIVE {
            return 0.0;
        }
        let denominator = (y_r - y_l) / h - d_l;
        if denominator.abs() < f64::MIN_POSITIVE {
            return if numerator > 0.0 {
                MAX_RATIONAL_CUBIC_CONTROL_PARAMETER
            } else {
                MIN_RATIONAL_CUBIC_CONTROL_PARAMETER
            };
        }
        numerator / denominator
    }

    fn control_param_at_right(
        x_l: f64,
        x_r: f64,
        y_l: f64,
        y_r: f64,
        d_l: f64,
        d_r: f64,
        d2_r: f64,
    ) -> f64 {
        let h = x_r - x_l;
        let numerator = 0.5 * h * d2_r + (d_r - d_l);
        if numerator.abs() < f64::MIN_POSITIVE {
            return 0.0;
        }
        let denominator = d_r - (y_r - y_l) / h;
        if denominator.abs() < f64::MIN_POSITIVE {
            return if numerator > 0.0 {
                MAX_RATIONAL_CUBIC_CONTROL_PARAMETER
            } else {
                MIN_RATIONAL_CUBIC_CONTROL_PARAMETER
            };
        }
        numerator / denominator
    }

    /// Minimum control parameter that guarantees monotonicity / convexity / concavity
    /// (Delbourgo-Gregory §3.8, §3.18).
    fn min_control_param(d_l: f64, d_r: f64, slope: f64, prefer_shape: bool) -> f64 {
        let monotonic = d_l * slope >= 0.0 && d_r * slope >= 0.0;
        let convex = d_l <= slope && slope <= d_r;
        let concave = d_l >= slope && slope >= d_r;
        if !monotonic && !convex && !concave {
            return MIN_RATIONAL_CUBIC_CONTROL_PARAMETER;
        }
        let d_diff = d_r - d_l;
        let dr_ms = d_r - slope;
        let s_mdl = slope - d_l;
        let mut r1 = f64::MIN;
        let mut r2 = f64::MIN;
        if monotonic {
            if slope.abs() >= f64::MIN_POSITIVE {
                r1 = (d_r + d_l) / slope;
            } else if prefer_shape {
                r1 = MAX_RATIONAL_CUBIC_CONTROL_PARAMETER;
            }
        }
        if convex || concave {
            if s_mdl.abs() >= f64::MIN_POSITIVE && dr_ms.abs() >= f64::MIN_POSITIVE {
                r2 = (d_diff / dr_ms).abs().max((d_diff / s_mdl).abs());
            } else if prefer_shape {
                r2 = MAX_RATIONAL_CUBIC_CONTROL_PARAMETER;
            }
        } else if monotonic && prefer_shape {
            r2 = MAX_RATIONAL_CUBIC_CONTROL_PARAMETER;
        }
        r1.max(r2).max(MIN_RATIONAL_CUBIC_CONTROL_PARAMETER)
    }

    /// Convex shape-preserving control parameter at left side.
    #[allow(clippy::too_many_arguments)]
    pub fn convex_control_left(
        x_l: f64,
        x_r: f64,
        y_l: f64,
        y_r: f64,
        d_l: f64,
        d_r: f64,
        d2_l: f64,
        prefer_shape: bool,
    ) -> f64 {
        let r = control_param_at_left(x_l, x_r, y_l, y_r, d_l, d_r, d2_l);
        let r_min = min_control_param(d_l, d_r, (y_r - y_l) / (x_r - x_l), prefer_shape);
        r.max(r_min)
    }

    /// Convex shape-preserving control parameter at right side.
    #[allow(clippy::too_many_arguments)]
    pub fn convex_control_right(
        x_l: f64,
        x_r: f64,
        y_l: f64,
        y_r: f64,
        d_l: f64,
        d_r: f64,
        d2_r: f64,
        prefer_shape: bool,
    ) -> f64 {
        let r = control_param_at_right(x_l, x_r, y_l, y_r, d_l, d_r, d2_r);
        let r_min = min_control_param(d_l, d_r, (y_r - y_l) / (x_r - x_l), prefer_shape);
        r.max(r_min)
    }

    // ------------------------------------------------------------------
    //  The main routine: implied σ·√T from (β, x, q).
    //
    //  β is the normalised market price (= price / √(F·K)).  This routine
    //  always operates on the OTM form via call-put symmetry, then
    //  iterates with Householder-4 from a rational-cubic initial guess.
    // ------------------------------------------------------------------

    /// Sentinel value used internally — returned to the caller as `NaN`.
    const PRICE_ABOVE_MAX: f64 = f64::INFINITY;
    /// Hard cap on Householder iterations.  Jäckel §3.5 shows ≤ 2 suffice for
    /// all interior inputs; we allow 4 to absorb rare bracket-bisection steps
    /// at the no-arbitrage boundary.
    const MAX_ITER: u32 = 4;

    /// Householder-4 factor: `(1 + ½·halley·newton) / (1 + newton·(halley + hh3·newton/6))`.
    #[inline]
    fn householder_factor(newton: f64, halley: f64, hh3: f64) -> f64 {
        (1.0 + 0.5 * halley * newton) / (1.0 + newton * (halley + hh3 * newton / 6.0))
    }

    /// Solve `b(x, s) = β` for s, given OTM inputs (`q·x ≤ 0`).
    /// Returns `+inf` if the price is above the upper bound, `NaN` if β < 0
    /// (the caller is expected to have stripped intrinsic already).
    #[allow(clippy::too_many_lines)]
    pub fn unchecked_normalised_iv(beta: f64, x: f64, q: f64) -> f64 {
        // Step 0: map ITM → OTM by stripping intrinsic, flipping q.
        let (mut beta, mut x, mut q) = (beta, x, q);
        if q * x > 0.0 {
            beta = (beta - normalised_black::intrinsic(x, q)).max(0.0);
            q = -q;
        }
        // Step 0b: map puts to calls (negate x).
        if q < 0.0 {
            x = -x;
            // q = +1 after this point — symmetry of b means we now operate on a call.
        }

        if beta <= 0.0 {
            return 0.0;
        }
        let b_max = (0.5 * x).exp();
        if beta >= b_max {
            return PRICE_ABOVE_MAX;
        }

        // Reference points (Jäckel §3.3):
        //   s_c is the inflection (∂²b/∂s² = 0): s_c = √|2·x|
        //   s_l = s_c - b_c/v_c  (one Newton step from s_c with slope v_c, towards 0)
        //   s_h = s_c + (b_max - b_c)/v_c
        let s_c = (2.0 * x.abs()).sqrt();
        let (b_c, v_c) = normalised_black::call_and_vega(x, s_c);

        let (mut s, mut s_left, mut s_right);

        if beta < b_c {
            // Lower half: β ∈ (0, b_c).
            let s_l = s_c - b_c / v_c;
            // `v_l` is only needed in the else branch below, but computing it
            // up-front via `call_and_vega` shares the exp factor with `b_l`
            // for one fewer transcendental per IV solve in this region.
            let (b_l, v_l) = normalised_black::call_and_vega(x, s_l);
            if beta < b_l {
                // Lowest segment: β ∈ (0, b_l).  Rational-cubic in the
                // lower-map space.
                let (f_l, fp_l, fpp_l) = f_lower_with_derivatives(x, s_l);
                let r_ll = convex_control_right(0.0, b_l, 0.0, f_l, 1.0, fp_l, fpp_l, true);
                let mut f = rational_cubic(beta, 0.0, b_l, 0.0, f_l, 1.0, fp_l, r_ll);
                if f.is_nan() || f <= 0.0 {
                    // Rational cubic can produce non-positive `f` due to roundoff
                    // truncation at extreme |x|.  Fall back to quadratic through
                    // (0, 0), (b_l, f_l), f'(0)=1.
                    let t = beta / b_l;
                    f = (f_l * t + b_l * (1.0 - t)) * t;
                }
                s = inv_f_lower(x, f);
                s_left = 0.0;
                s_right = s_l;
                householder_iter_lower(beta, x, &mut s, &mut s_left, &mut s_right);
                return s;
            }
            // Lower-middle segment: β ∈ [b_l, b_c). `v_l` was computed alongside `b_l` above.
            let r_lm = convex_control_right(b_l, b_c, s_l, s_c, 1.0 / v_l, 1.0 / v_c, 0.0, false);
            s = rational_cubic(beta, b_l, b_c, s_l, s_c, 1.0 / v_l, 1.0 / v_c, r_lm);
            s_left = s_l;
            s_right = s_c;
        } else {
            // Upper half: β ∈ [b_c, b_max).
            let s_h = if v_c > f64::MIN_POSITIVE {
                s_c + (b_max - b_c) / v_c
            } else {
                s_c
            };
            // Same shared-exp argument as `s_l` above: compute `v_h` alongside `b_h`.
            let (b_h, v_h) = normalised_black::call_and_vega(x, s_h);
            if beta <= b_h {
                // Upper-middle segment: β ∈ [b_c, b_h].
                let r_hm =
                    convex_control_left(b_c, b_h, s_c, s_h, 1.0 / v_c, 1.0 / v_h, 0.0, false);
                s = rational_cubic(beta, b_c, b_h, s_c, s_h, 1.0 / v_c, 1.0 / v_h, r_hm);
                s_left = s_c;
                s_right = s_h;
            } else {
                // Highest segment: β ∈ (b_h, b_max).  Rational-cubic in
                // the upper-map space.
                let (f_h, fp_h, fpp_h) = f_upper_with_derivatives(x, s_h);
                let mut f = if fpp_h.abs() < f64::MAX.sqrt() {
                    let r_hh = convex_control_left(b_h, b_max, f_h, 0.0, fp_h, -0.5, fpp_h, true);
                    rational_cubic(beta, b_h, b_max, f_h, 0.0, fp_h, -0.5, r_hh)
                } else {
                    0.0
                };
                if f.is_nan() || f <= 0.0 {
                    let h_seg = b_max - b_h;
                    let t = (beta - b_h) / h_seg;
                    f = (f_h * (1.0 - t) + 0.5 * h_seg * t) * (1.0 - t);
                }
                s = inv_f_upper(f);
                s_left = s_h;
                s_right = f64::MAX;
                if beta > 0.5 * b_max {
                    householder_iter_upper(beta, x, b_max, &mut s, &mut s_left, &mut s_right);
                    return s;
                }
            }
        }
        // Middle segments use the direct objective g(s) = b(x,s) − β.
        householder_iter_middle(beta, x, &mut s, &mut s_left, &mut s_right);
        let _ = q; // q is consumed by the put-call mapping above; σ is sign-free.
        s
    }

    /// Iterate `g = ln(b)/ln(β) · ... ` form (lower segment, β tiny).
    fn householder_iter_lower(beta: f64, x: f64, s: &mut f64, s_left: &mut f64, s_right: &mut f64) {
        let mut ds = f64::MAX;
        let mut ds_prev = 0.0_f64;
        let mut reversal = 0;
        for iter in 0..MAX_ITER {
            if ds.abs() <= f64::EPSILON * *s {
                break;
            }
            if ds * ds_prev < 0.0 {
                reversal += 1;
            }
            if iter > 0 && (reversal >= 3 || !(*s > *s_left && *s < *s_right)) {
                *s = 0.5 * (*s_left + *s_right);
                if *s_right - *s_left <= f64::EPSILON * *s {
                    return;
                }
                reversal = 0;
                ds = 0.0;
            }
            ds_prev = ds;
            let (b, bp) = normalised_black::call_and_vega(x, *s);
            if b > beta && *s < *s_right {
                *s_right = *s;
            } else if b < beta && *s > *s_left {
                *s_left = *s;
            }
            if b <= 0.0 || bp <= 0.0 {
                ds = 0.5 * (*s_left + *s_right) - *s;
            } else {
                let ln_b = b.ln();
                let ln_beta = beta.ln();
                let bpob = bp / b;
                let h = x / *s;
                let b_halley = h * h / *s - 0.25 * *s;
                let newton = (ln_beta - ln_b) * ln_b / ln_beta / bpob;
                let halley = b_halley - bpob * (1.0 + 2.0 / ln_b);
                let b_hh3 = b_halley * b_halley - 3.0 * (h / *s) * (h / *s) - 0.25;
                let hh3 = b_hh3 + 2.0 * bpob * bpob * (1.0 + 3.0 / ln_b * (1.0 + 1.0 / ln_b))
                    - 3.0 * b_halley * bpob * (1.0 + 2.0 / ln_b);
                ds = newton * householder_factor(newton, halley, hh3);
            }
            ds = ds.max(-0.5 * *s);
            *s += ds;
        }
    }

    /// Iterate `g = ln((b_max − β)/(b_max − b))` form (upper segment).
    fn householder_iter_upper(
        beta: f64,
        x: f64,
        b_max: f64,
        s: &mut f64,
        s_left: &mut f64,
        s_right: &mut f64,
    ) {
        let mut ds = f64::MAX;
        let mut ds_prev = 0.0_f64;
        let mut reversal = 0;
        for iter in 0..MAX_ITER {
            if ds.abs() <= f64::EPSILON * *s {
                break;
            }
            if ds * ds_prev < 0.0 {
                reversal += 1;
            }
            if iter > 0 && (reversal >= 3 || !(*s > *s_left && *s < *s_right)) {
                *s = 0.5 * (*s_left + *s_right);
                if *s_right - *s_left <= f64::EPSILON * *s {
                    return;
                }
                reversal = 0;
                ds = 0.0;
            }
            ds_prev = ds;
            let (b, bp) = normalised_black::call_and_vega(x, *s);
            if b > beta && *s < *s_right {
                *s_right = *s;
            } else if b < beta && *s > *s_left {
                *s_left = *s;
            }
            if b >= b_max || bp <= f64::MIN_POSITIVE {
                ds = 0.5 * (*s_left + *s_right) - *s;
            } else {
                let b_max_minus_b = b_max - b;
                let g = ((b_max - beta) / b_max_minus_b).ln();
                let gp = bp / b_max_minus_b;
                let b_halley = (x / *s) * (x / *s) / *s - 0.25 * *s;
                let b_hh3 = b_halley * b_halley - 3.0 * (x / (*s * *s)) * (x / (*s * *s)) - 0.25;
                let newton = -g / gp;
                let halley = b_halley + gp;
                let hh3 = b_hh3 + gp * (2.0 * gp + 3.0 * b_halley);
                ds = newton * householder_factor(newton, halley, hh3);
            }
            ds = ds.max(-0.5 * *s);
            *s += ds;
        }
    }

    /// Iterate direct `g = b − β` (middle segments).
    fn householder_iter_middle(
        beta: f64,
        x: f64,
        s: &mut f64,
        s_left: &mut f64,
        s_right: &mut f64,
    ) {
        let mut ds = f64::MAX;
        let mut ds_prev = 0.0_f64;
        let mut reversal = 0;
        for iter in 0..MAX_ITER {
            if ds.abs() <= f64::EPSILON * *s {
                break;
            }
            if ds * ds_prev < 0.0 {
                reversal += 1;
            }
            if iter > 0 && (reversal >= 3 || !(*s > *s_left && *s < *s_right)) {
                *s = 0.5 * (*s_left + *s_right);
                if *s_right - *s_left <= f64::EPSILON * *s {
                    return;
                }
                reversal = 0;
                ds = 0.0;
            }
            ds_prev = ds;
            let (b, bp) = normalised_black::call_and_vega(x, *s);
            if b > beta && *s < *s_right {
                *s_right = *s;
            } else if b < beta && *s > *s_left {
                *s_left = *s;
            }
            let newton = (beta - b) / bp;
            let halley = (x / *s) * (x / *s) / *s - 0.25 * *s;
            let hh3 = halley * halley - 3.0 * (x / (*s * *s)) * (x / (*s * *s)) - 0.25;
            ds = newton * householder_factor(newton, halley, hh3);
            ds = ds.max(-0.5 * *s);
            *s += ds;
        }
    }
}

/// Solve for implied volatility given a market price.
///
/// Returns `NaN` if:
///   - `t <= 0`,
///   - `s <= 0` or `k <= 0`,
///   - the target price is not finite or is negative,
///   - the target price violates the no-arbitrage bounds, or
///   - the normalised solve returns a non-finite value (a defensive guard:
///     the `PRICE_ABOVE_MAX` sentinel maps here). The Householder iteration is
///     hard-capped at `MAX_ITER` but always returns a finite `s` — it does not
///     signal non-convergence, since ≤ 2 steps suffice across all interior β.
pub fn solve(target_price: f64, flag: Flag, s: f64, k: f64, t: f64, r: f64, q: f64) -> f64 {
    if t <= 0.0 || s <= 0.0 || k <= 0.0 || !target_price.is_finite() || target_price < 0.0 {
        return f64::NAN;
    }

    let disc_r = (-r * t).exp();
    let disc_q = (-q * t).exp();
    // Forward (Black-76 equivalent): F = S·exp((r-q)T).
    let forward = s * disc_q / disc_r;

    // No-arbitrage bounds (in BSM, i.e. discounted units).
    let (lower, upper) = match flag {
        Flag::Call => ((s * disc_q - k * disc_r).max(0.0), s * disc_q),
        Flag::Put => ((k * disc_r - s * disc_q).max(0.0), k * disc_r),
    };
    if target_price < lower - 1e-12 || target_price > upper + 1e-12 {
        return f64::NAN;
    }

    // Convert to normalised inputs:
    //   β = undiscounted_price / √(F·K),   x = ln(F/K),   q_sign = ±1.
    let undiscounted = target_price / disc_r;
    let beta = undiscounted / (forward * k).sqrt();
    let x = (forward / k).ln();
    let q_sign = match flag {
        Flag::Call => 1.0,
        Flag::Put => -1.0,
    };

    let s_norm = lbr::unchecked_normalised_iv(beta, x, q_sign);
    if !s_norm.is_finite() {
        return f64::NAN;
    }
    s_norm / t.sqrt()
}

#[cfg(test)]
mod lbr_tests {
    //! Layer 3–4 tests: lower/upper map inverses, rational cubic invariants,
    //! and full `unchecked_normalised_iv` roundtrips.
    use super::{lbr, normalised_black};
    use approx::assert_relative_eq;

    #[test]
    fn lower_map_inverse_roundtrip() {
        for &(x, s) in &[(-0.5_f64, 0.1_f64), (-1.0, 0.05), (-2.0, 0.2)] {
            let f = lbr::f_lower(x, s);
            let s_back = lbr::inv_f_lower(x, f);
            assert_relative_eq!(s_back, s, max_relative = 1e-12);
        }
    }

    #[test]
    fn upper_map_inverse_roundtrip() {
        for &s in &[0.1_f64, 0.5, 1.0, 2.0] {
            let f = lbr::f_upper(s);
            let s_back = lbr::inv_f_upper(f);
            assert_relative_eq!(s_back, s, max_relative = 1e-12);
        }
    }

    #[test]
    fn rational_cubic_endpoints() {
        // At t=0: returns y_l; at t=1: returns y_r.
        let r = 0.0;
        for &y_l in &[0.0_f64, 1.0, -5.0] {
            for &y_r in &[2.0_f64, -3.0, 7.0] {
                let left = lbr::rational_cubic(0.0, 0.0, 1.0, y_l, y_r, 0.5, 0.3, r);
                assert_relative_eq!(left, y_l, max_relative = 1e-15);
                let right = lbr::rational_cubic(1.0, 0.0, 1.0, y_l, y_r, 0.5, 0.3, r);
                assert_relative_eq!(right, y_r, max_relative = 1e-15);
            }
        }
    }

    /// Roundtrip: pick (x, s), price it via b(x,s), invert via LBR, expect s back.
    #[test]
    fn iv_roundtrip_grid() {
        let cases: &[(f64, f64)] = &[
            // (x, s) across all four iteration branches.
            (-0.1, 0.2),
            (0.0, 0.2),
            (-0.5, 0.4),
            (0.0, 2.0), // high vol — upper-middle
            (-0.5, 2.0),
            (0.5, 0.2),  // ITM call
            (-2.0, 0.4), // deep OTM
            (-3.0, 1.0), // very deep OTM
        ];
        for &(x, s) in cases {
            let beta = normalised_black::call(x, s);
            // q = +1 (call form).
            let s_back = lbr::unchecked_normalised_iv(beta, x, 1.0);
            assert_relative_eq!(s_back, s, max_relative = 1e-12, epsilon = 1e-13);
        }
    }
}

#[cfg(test)]
mod normalised_black_tests {
    //! Goldens are mpmath-computed (50 decimal digits).  `bsm::price` would
    //! otherwise be a lossy oracle in the deep-OTM short-expiry corner, where
    //! `S·Φ(d1) − K·Φ(d2)` suffers catastrophic cancellation (only ~9 digits
    //! of relative accuracy survives at e.g. h ≈ −5, t ≈ 0.01). LBR's regions
    //! exist precisely to dodge this cancellation, so the algorithm must be
    //! validated against an external oracle, not `bsm::price`, in those regimes.

    use super::normalised_black;
    use crate::bsm::{price as bsm_price, Flag};
    use approx::assert_relative_eq;

    #[test]
    fn call_matches_mpmath_goldens() {
        // (label, x, s, expected_b)
        for &(_label, x, s, expected) in &[
            ("r4_moderate_otm", -0.1_f64, 0.2_f64, 3.945_860_264_180_7e-2),
            ("r4_atm", 0.0_f64, 0.2_f64, 7.965_567_455_405_796e-2),
            ("r4_otm_high", -0.5_f64, 0.4_f64, 1.996_050_989_765_552e-2),
            (
                "r3_atm_high_vol",
                0.0_f64,
                2.0_f64,
                6.826_894_921_370_859e-1,
            ),
            ("r3_otm_long_t", -0.5_f64, 2.0_f64, 4.666_462_289_193_513e-1),
            (
                "r2_small_t_atm",
                0.0_f64,
                0.02_f64,
                7.978_712_629_263_208e-3,
            ),
            (
                "r2_small_t_otm",
                -0.095_310_179_804_324_8_f64,
                0.02_f64,
                3.663_393_972_840_609e-9,
            ),
            (
                "r1_deep_asymp",
                -1.1_f64,
                0.1_f64,
                1.707_270_160_488_868_5e-30,
            ),
            ("r1_deeper", -1.5_f64, 0.1_f64, 2.423_020_570_391_193_4e-53),
            ("itm_call", 0.5_f64, 0.2_f64, 5.056_237_971_192_526e-1),
            (
                "itm_call_high_vol",
                1.0_f64,
                0.5_f64,
                1.046_332_969_738_173_3,
            ),
        ] {
            let got = normalised_black::call(x, s);
            assert_relative_eq!(got, expected, max_relative = 1e-14);
        }
    }

    /// Sanity cross-check vs `bsm::price` at moderate inputs (away from the
    /// extreme-OTM cancellation regime, where `bsm::price` itself is lossy).
    /// Identity for OTM calls: `bsm_call = √(F·K)·exp(-rT)·b(x, s)`.
    #[test]
    fn call_agrees_with_bsm_price_at_moderate_inputs() {
        let cases: &[(f64, f64, f64, f64, f64, f64)] = &[
            (100.0, 100.0, 1.0, 0.05, 0.0, 0.20),
            (100.0, 110.0, 1.0, 0.05, 0.0, 0.20),
            (100.0, 90.0, 1.0, 0.05, 0.0, 0.20),
            (100.0, 110.0, 0.5, 0.0, 0.02, 0.30),
            (100.0, 100.0, 2.0, 0.05, 0.0, 0.50),
        ];
        for &(s_spot, k, t, r, q, sigma) in cases {
            let forward = s_spot * ((r - q) * t).exp();
            let x = (forward / k).ln();
            let s = sigma * t.sqrt();
            let b = normalised_black::call(x, s);
            let bsm = bsm_price(Flag::Call, s_spot, k, t, r, q, sigma);
            let predicted = (forward * k).sqrt() * (-r * t).exp() * b;
            assert_relative_eq!(predicted, bsm, max_relative = 1e-12);
        }
    }

    #[test]
    fn intrinsic_taylor_matches_direct() {
        // Near x = 0 the Taylor series replaces `2·sinh(x/2)` for stability;
        // verify the two formulas agree to f64 precision in the overlap.
        for &x in &[1e-3_f64, 1e-2, 0.05, 0.1] {
            let half = (0.5 * x).exp();
            let direct = (half - 1.0 / half).max(0.0);
            let series = normalised_black::intrinsic(x, 1.0);
            assert_relative_eq!(series, direct, max_relative = 1e-14);
        }
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn intrinsic_otm_returns_zero() {
        // OTM (q·x ≤ 0) must be exactly zero.
        assert_eq!(normalised_black::intrinsic(-0.5, 1.0), 0.0);
        assert_eq!(normalised_black::intrinsic(0.5, -1.0), 0.0);
    }

    /// Drift guard: `call_and_vega(x, s)` must match `(call(x, s), vega(x, s))`
    /// across all four regions of the LBR dispatch. `b` is bit-identical
    /// (both paths share the same expression). `vega` allows up to a few
    /// ULPs of drift because the standalone `vega` evaluates the exp
    /// argument via `mul_add` while the per-region helpers use a plain
    /// `h*h + t*t` (the regions disagreed with the standalone vega by
    /// 1-2 ULPs pre-refactor too — that's not new behaviour, just a
    /// pre-existing inconsistency this test now surfaces). 1e-14 is four
    /// orders of magnitude tighter than the IV roundtrip tolerance.
    #[test]
    fn call_and_vega_matches_separate() {
        // Cases drawn from each region of the four-way dispatch.
        let cases: &[(&str, f64, f64)] = &[
            ("r1_deep_asymp", -1.1, 0.1),
            ("r1_deeper", -1.5, 0.1),
            ("r2_small_t_atm", 0.0, 0.02),
            ("r2_small_t_otm", -0.095_310_179_8, 0.02),
            ("r3_atm_high_vol", 0.0, 2.0),
            ("r3_itm_call", 0.5, 0.2),
            ("r4_moderate_otm", -0.1, 0.2),
            ("r4_otm_high", -0.5, 0.4),
            ("r4_otm_long_t", -0.5, 2.0),
            ("itm_call_high_vol", 1.0, 0.5),
        ];
        for &(label, x, s) in cases {
            let (b_combined, v_combined) = normalised_black::call_and_vega(x, s);
            let b_sep = normalised_black::call(x, s);
            let v_sep = normalised_black::vega(x, s);
            // `b` is bit-identical (same expression on both sides).
            assert_relative_eq!(b_combined, b_sep, max_relative = 1e-15, epsilon = 1e-300);
            // `vega` allows 1-2 ULPs of pre-existing FMA-vs-plain drift.
            assert_relative_eq!(v_combined, v_sep, max_relative = 1e-14, epsilon = 1e-300);
            let _ = label;
        }
    }

    #[test]
    fn normalised_vega_matches_finite_difference() {
        // ∂b/∂s by central difference should match normalised_black::vega.
        let h_step = 1e-7_f64;
        for &(x, s) in &[
            (-0.1_f64, 0.3),
            (0.0, 0.5),
            (-0.5, 0.4),
            (-2.0, 0.8),
            (-5.0, 1.2),
        ] {
            let fd = (normalised_black::call(x, s + h_step)
                - normalised_black::call(x, s - h_step))
                / (2.0 * h_step);
            let analytic = normalised_black::vega(x, s);
            assert_relative_eq!(analytic, fd, max_relative = 1e-6);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bsm::price;
    use approx::assert_relative_eq;

    /// Roundtrip tolerance for LBR is bounded by the precision of `bsm::price`
    /// at the test inputs, not the algorithm itself.  At moderate moneyness
    /// both directions hit ~1e-14; tightened from the prior Newton solver's 1e-6.
    const LBR_ROUNDTRIP_TOL: f64 = 1e-13;

    #[test]
    fn roundtrip_call_atm() {
        let (s, k, t, r, q, sigma) = (100.0, 100.0, 1.0, 0.05, 0.0, 0.20);
        let p = price(Flag::Call, s, k, t, r, q, sigma);
        let iv = solve(p, Flag::Call, s, k, t, r, q);
        assert_relative_eq!(iv, sigma, max_relative = LBR_ROUNDTRIP_TOL);
    }

    #[test]
    fn roundtrip_put_otm_short_expiry() {
        let (s, k, t, r, q, sigma) = (100.0, 80.0, 0.05, 0.03, 0.01, 0.45);
        let p = price(Flag::Put, s, k, t, r, q, sigma);
        let iv = solve(p, Flag::Put, s, k, t, r, q);
        assert_relative_eq!(iv, sigma, max_relative = LBR_ROUNDTRIP_TOL);
    }

    #[test]
    fn roundtrip_call_deep_otm() {
        let (s, k, t, r, q, sigma) = (100.0, 200.0, 0.5, 0.05, 0.0, 0.30);
        let p = price(Flag::Call, s, k, t, r, q, sigma);
        let iv = solve(p, Flag::Call, s, k, t, r, q);
        assert_relative_eq!(iv, sigma, max_relative = LBR_ROUNDTRIP_TOL);
    }

    #[test]
    fn price_below_intrinsic_returns_nan() {
        let iv = solve(-1.0, Flag::Call, 100.0, 100.0, 1.0, 0.05, 0.0);
        assert!(iv.is_nan());
    }

    #[test]
    fn zero_time_returns_nan() {
        let iv = solve(5.0, Flag::Call, 100.0, 100.0, 0.0, 0.05, 0.0);
        assert!(iv.is_nan());
    }
}
