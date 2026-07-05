//! Standard normal distribution helpers and special functions used by the IV solver.
//!
//! `cdf` uses `libm::erf`/`erfc` (sign-split to avoid tail cancellation) on
//! `|x| < 4` and an `erfcx` form beyond, for high accuracy across the full range.
//!
//! `erfcx` and `inverse_cdf` are needed by the "Let's Be Rational" IV solver
//! (Jäckel, 2015) — `erfcx` for stable normalised-Black evaluation in the deep
//! out-of-the-money tail, `inverse_cdf` for the rational-cubic initial guess.
//!
//! **Why the pinned `libm` crate:** `erf`/`erfc` come from the pure-Rust `libm`
//! crate rather than the platform libm so the tail is bit-identical across
//! macOS/Linux/Windows. `erfc` is the weakest primitive under the ~1-ULP bar
//! (the crate guarantees only 4 ULP), so the dependency is pinned to an exact
//! version (see `crates/core/Cargo.toml`) and re-validated on any deliberate
//! bump by the dense `erfcx` goldens in `erfcx_dense_direct_path_goldens`. NB
//! glibc 2.43 (2026) ships CORE-MATH correctly-rounded erf/erfc, so a modern
//! Linux *system* libm is now more accurate than this vendored erfc — the pin
//! buys determinism, not peak accuracy, with `cr_erfc` as the contingency if
//! the tail ever regresses.

/// `1 / sqrt(2)` — multiply rather than divide; saves ~7 cycles per call on
/// `x86_64`.  IEEE 754 division and multiplication round identically here
/// because `1/sqrt(2)` is representable to one ULP and the magnitudes
/// involved (`d1`, `d2` for BSM-pricing typical inputs) do not lose any
/// significand bits through the rescaling.
const INV_SQRT_2: f64 = std::f64::consts::FRAC_1_SQRT_2;
/// `1 / sqrt(2 * pi)`.
const INV_SQRT_2PI: f64 = 0.398_942_280_401_432_7;
/// `1 / sqrt(pi)`.
const INV_SQRT_PI: f64 = 0.564_189_583_547_756_3;

/// Standard normal CDF: `P(Z <= x)` for `Z ~ N(0, 1)`.
///
/// For `|x| < 4`, split by sign to avoid catastrophic cancellation:
///   - `x >= 0`: `0.5·(1 + erf(x/√2))` — two non-negative terms, ~1 ULP.
///   - `x < 0`:  `0.5·erfc(|x|/√2)` — the complementary form computes the small
///     tail value directly. The naive `0.5·(1 + erf(x/√2))` here subtracts two
///     near-equal quantities (`erf(x/√2) → -1`) and bleeds precision — ~1e-12
///     relative near `x = -4`, vs the ~1 ULP it claims. `erfc` sidesteps it;
///     the residual is the tail's intrinsic conditioning (a few ULP near
///     `|x| = 4`, falling to ~1 ULP toward `x = 0`).
///
/// For `|x| >= 4` the `erf`/`erfc` arguments saturate (`libm::erf` reaches ±1
/// once `|x/√2| > ~5.93`), so we switch to the `erfcx` form
/// `cdf(x) = 0.5·exp(-x²/2)·erfcx(|x|/√2)` (`x < 0`) or `1 - same` (`x > 0`),
/// which stays accurate — conditioning-limited, a few ULP — down to
/// `f64::MIN_POSITIVE` and below.
///
/// The tail branch is marked `#[cold]` so LLVM lays out the fast path
/// contiguously — most BSM-pricing calls hit `|x| < 4` (typical `d1`, `d2`).
///
/// A branchless alternative (always-erfcx form) was measured in the F2 audit
/// (2026-05) and rejected: 14-69% slower across narrow/wide/shuffled inputs
/// because `libm` is scalar and the outer branch doesn't gate vectorisation.
/// See `bench_cdf_branch_experiment` in `benches/pricing.rs` if curious.
#[inline]
pub fn cdf(x: f64) -> f64 {
    if x.abs() < 4.0 {
        // `1 + erf(x/√2)` cancels for x < 0, so route the negative half through
        // `erfc`, which evaluates the small complementary value without the
        // subtraction. The positive half adds two non-negative terms cleanly.
        if x < 0.0 {
            return 0.5 * libm::erfc(-x * INV_SQRT_2);
        }
        return 0.5 * (1.0 + libm::erf(x * INV_SQRT_2));
    }
    cdf_tail(x)
}

#[cold]
fn cdf_tail(x: f64) -> f64 {
    // Non-finite inputs land here (`|±inf| < 4` and `|NaN| < 4` are both false),
    // so this cold branch owns their exact limits — keeping the hot `|x| < 4`
    // path free of the guard. Without it, `erfcx(±inf)` fed the modified-Lentz
    // recurrence a `NaN` that never satisfied its `|δ−1| < 1e-16` break, spinning
    // the full 200-iteration cap and returning `NaN` where the limit is 1 or 0.
    if !x.is_finite() {
        return if x.is_nan() {
            f64::NAN
        } else if x > 0.0 {
            1.0
        } else {
            0.0
        };
    }
    if x < 0.0 {
        0.5 * (-0.5 * x * x).exp() * erfcx(-x * INV_SQRT_2)
    } else {
        1.0 - 0.5 * (-0.5 * x * x).exp() * erfcx(x * INV_SQRT_2)
    }
}

/// Standard normal PDF: `(2 * pi)^(-1/2) * exp(-x^2 / 2)`.
#[inline]
pub fn pdf(x: f64) -> f64 {
    INV_SQRT_2PI * (-0.5 * x * x).exp()
}

/// Scaled complementary error function: `erfcx(z) = exp(z^2) * erfc(z)`.
///
/// The direct product `(z*z).exp() * libm::erfc(z)` agrees with high-precision
/// references (verified against mpmath at 50 digits) to ~1 ULP for the entire
/// range until `exp(z*z)` overflows in `f64`, which happens at `z*z > ~709`
/// (i.e., `z > ~26.6`).  Below that overflow cliff, the direct path is both
/// faster and at least as accurate as any alternative.
///
/// For `z >= 26.5` we fall through to the modified-Lentz continued fraction
///
/// ```text
///     sqrt(pi) * erfcx(z) = 1 / (z + (1/2) / (z + 1 / (z + (3/2) / (z + ...))))
/// ```
///
/// (`a_j = j/2`, `b_j = z`).  At such large z the CF converges in 3-4
/// iterations, so the cost is negligible.
///
/// Reference: continued fraction is Numerical Recipes §6.2 (after Henry
/// Thacher); Marsaglia (2004) motivates the use of `erfcx` in the Black
/// formula to avoid catastrophic cancellation.
///
/// Note on cross-platform behavior: `libm::erfc` is a vendored
/// implementation, bit-identical on macOS / Linux / Windows. But the
/// `(z*z).exp()` beside it — and every `.exp()`/`.ln()`/`.sqrt()` elsewhere
/// in this crate — is `std` math, i.e. the *platform's* libm, so end-to-end
/// results may differ between platforms in the last ULP. Bit-for-bit
/// cross-platform determinism is NOT guaranteed; correctness to the
/// documented ~1-ULP tolerance is. (Routing all transcendentals through
/// `libm` would buy bit-determinism at a measured perf cost — a deliberate,
/// output-changing decision deferred until something needs it.)
pub fn erfcx(z: f64) -> f64 {
    // Modified-Lentz convergence tolerance and divide-by-zero floor.
    const LENTZ_TINY: f64 = 1e-300;
    // Direct path stays below the `exp(z*z)` overflow cliff (`z*z > 709`).
    const DIRECT_LIMIT: f64 = 26.5;

    // Exact limits for non-finite z. `+inf` and `NaN` would otherwise reach the
    // continued fraction below (`+inf ≥ DIRECT_LIMIT`; `NaN < DIRECT_LIMIT` is
    // false), where the recurrence produces a `NaN` δ that never trips the break
    // — spinning all 200 iterations and returning `NaN`. `erfcx(+inf) = 0`,
    // `erfcx(−inf) = +inf`, `erfcx(NaN) = NaN`.
    if !z.is_finite() {
        return if z.is_nan() {
            f64::NAN
        } else if z > 0.0 {
            0.0
        } else {
            f64::INFINITY
        };
    }

    if z < DIRECT_LIMIT {
        // Direct product.  For very negative z, exp(z²) correctly overflows
        // to +inf as erfcx → ∞.
        return (z * z).exp() * libm::erfc(z);
    }
    let mut f = z;
    let mut c = z;
    let mut d = 0.0_f64;
    for j in 1..200 {
        let a = 0.5 * f64::from(j);
        let mut new_d = a.mul_add(d, z);
        if new_d.abs() < LENTZ_TINY {
            new_d = LENTZ_TINY;
        }
        let mut new_c = z + a / c;
        if new_c.abs() < LENTZ_TINY {
            new_c = LENTZ_TINY;
        }
        new_d = 1.0 / new_d;
        let delta = new_c * new_d;
        f *= delta;
        c = new_c;
        d = new_d;
        if (delta - 1.0).abs() < 1e-16 {
            break;
        }
    }
    INV_SQRT_PI / f
}

// Wichura (1988) AS241 coefficients. Reproduced verbatim from the paper;
// truncated to 17 significant figures (f64 mantissa is ~15.95 decimal digits).
mod as241 {
    pub(super) const A: [f64; 8] = [
        3.387_132_872_796_366_6,
        1.331_416_678_917_843_8e2,
        1.971_590_950_306_551_4e3,
        1.373_169_376_550_946e4,
        4.592_195_393_154_987e4,
        6.726_577_092_700_87e4,
        3.343_057_558_358_813e4,
        2.509_080_928_730_122_8e3,
    ];
    pub(super) const B: [f64; 7] = [
        4.231_333_070_160_091e1,
        6.871_870_074_920_579e2,
        5.394_196_021_424_751e3,
        2.121_379_430_158_659_7e4,
        3.930_789_580_009_271e4,
        2.872_908_573_572_194_3e4,
        5.226_495_278_852_854e3,
    ];
    pub(super) const C: [f64; 8] = [
        1.423_437_110_749_683_6,
        4.630_337_846_156_545,
        5.769_497_221_460_691,
        3.647_848_324_763_204_6,
        1.270_458_252_452_368_4,
        2.417_807_251_774_506e-1,
        2.272_384_498_926_918_3e-2,
        7.745_450_142_783_414e-4,
    ];
    pub(super) const D: [f64; 7] = [
        2.053_191_626_637_758_5,
        1.676_384_830_183_803_8,
        6.897_673_349_851e-1,
        1.481_039_764_274_8e-1,
        1.519_866_656_361_645_7e-2,
        5.475_938_084_995_345e-4,
        1.050_750_071_644_416_8e-9,
    ];
    pub(super) const E: [f64; 8] = [
        6.657_904_643_501_103,
        5.463_784_911_164_114,
        1.784_826_539_917_291_4,
        2.965_605_718_285_048_8e-1,
        2.653_218_952_657_612e-2,
        1.242_660_947_388_078_4e-3,
        2.711_555_568_743_487_6e-5,
        2.010_334_399_292_288e-7,
    ];
    pub(super) const F: [f64; 7] = [
        5.998_322_065_558_88e-1,
        1.369_298_809_227_358e-1,
        1.487_536_129_085_061_6e-2,
        7.868_691_311_456_133e-4,
        1.846_318_317_510_054_6e-5,
        1.421_511_758_316_446e-7,
        2.044_263_103_389_939_8e-15,
    ];
}

/// Inverse standard normal CDF.
///
/// Returns `Z` such that `cdf(Z) = u`, accurate to ~1.6e-16 across the open
/// interval `(0, 1)`.  At the closed boundaries returns `-inf` (`u <= 0`) or
/// `+inf` (`u >= 1`); for `NaN` input returns `NaN`.
///
/// Reference: Wichura, M. J. (1988). Algorithm AS 241: The percentage points
/// of the normal distribution. *Applied Statistics*, 37(3), 477–484.
pub fn inverse_cdf(u: f64) -> f64 {
    use as241::{A, B, C, D, E, F};

    if u.is_nan() {
        return f64::NAN;
    }
    if u <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if u >= 1.0 {
        return f64::INFINITY;
    }

    let q = u - 0.5;
    if q.abs() <= 0.425 {
        // Central region.
        let r = 0.180_625 - q * q;
        let num = (((((((A[7] * r + A[6]) * r + A[5]) * r + A[4]) * r + A[3]) * r + A[2]) * r
            + A[1])
            * r
            + A[0])
            * q;
        let den =
            ((((((B[6] * r + B[5]) * r + B[4]) * r + B[3]) * r + B[2]) * r + B[1]) * r + B[0]) * r
                + 1.0;
        return num / den;
    }

    // Tail.  Reflect to `u in (0, 0.5)` to keep both branches well-conditioned.
    let r_inner = if q < 0.0 { u } else { 1.0 - u };
    let r = (-r_inner.ln()).sqrt();

    let val = if r < 5.0 {
        let r = r - 1.6;
        (((((((C[7] * r + C[6]) * r + C[5]) * r + C[4]) * r + C[3]) * r + C[2]) * r + C[1]) * r
            + C[0])
            / (((((((D[6] * r + D[5]) * r + D[4]) * r + D[3]) * r + D[2]) * r + D[1]) * r + D[0])
                * r
                + 1.0)
    } else {
        let r = r - 5.0;
        (((((((E[7] * r + E[6]) * r + E[5]) * r + E[4]) * r + E[3]) * r + E[2]) * r + E[1]) * r
            + E[0])
            / (((((((F[6] * r + F[5]) * r + F[4]) * r + F[3]) * r + F[2]) * r + F[1]) * r + F[0])
                * r
                + 1.0)
    };

    if q < 0.0 {
        -val
    } else {
        val
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn cdf_symmetric_around_zero() {
        assert_relative_eq!(cdf(0.0), 0.5, epsilon = 1e-15);
    }

    #[test]
    fn cdf_complement() {
        for &x in &[-2.0, -1.0, -0.5, 0.5, 1.0, 2.0, 3.5] {
            assert_relative_eq!(cdf(x) + cdf(-x), 1.0, epsilon = 1e-15);
        }
    }

    #[test]
    fn cdf_known_values() {
        // Reference values verified against scipy.stats.norm.cdf and R pnorm.
        assert_relative_eq!(cdf(1.0), 0.841_344_746_068_542_9, epsilon = 1e-12);
        assert_relative_eq!(cdf(1.96), 0.975_002_104_851_780_8, epsilon = 1e-12);
        assert_relative_eq!(cdf(2.576), 0.995_002_467_684_265, epsilon = 1e-12);
    }

    // Negative-tail goldens: 50-digit mpmath Φ(x) rounded to f64. The
    // pre-2026-06 `0.5·(1 + erf(x/√2))` branch lost ~1e-12 relative here to
    // `1 + erf(→ -1)` cancellation (it would fail this test at `1e-14`); the
    // sign-split `erfc` form is conditioning-limited to a few ULP. Regenerate
    // with `mpmath.mp.dps=60; 0.5*erfc(-x/sqrt(2))`.
    #[test]
    fn cdf_negative_tail_matches_mpmath() {
        for &(x, expected) in &[
            (-3.99_f64, 3.303_664_762_940_242e-5),
            (-3.9_f64, 4.809_634_401_760_273_6e-5),
            (-3.5_f64, 0.000_232_629_079_035_525_04),
            (-3.0_f64, 0.001_349_898_031_630_094_6),
            (-2.5_f64, 0.006_209_665_325_776_135),
            (-2.0_f64, 0.022_750_131_948_179_21),
            (-1.5_f64, 0.066_807_201_268_858_07),
            (-1.0_f64, 0.158_655_253_931_457_05),
        ] {
            assert_relative_eq!(cdf(x), expected, max_relative = 1e-14);
        }
    }

    #[test]
    fn pdf_known_values() {
        assert_relative_eq!(pdf(0.0), INV_SQRT_2PI, epsilon = 1e-15);
        assert_relative_eq!(pdf(1.0), 0.241_970_724_519_143_4, epsilon = 1e-12);
    }

    // erfcx golden values generated with scipy.special.erfcx.
    #[test]
    fn erfcx_golden_values() {
        for &(z, expected) in &[
            (-3.0, 1.620_598_885_399_958_6e4),
            (-1.0, 5.008_980_080_762_283),
            (-0.5, 1.952_360_489_182_557),
            (0.0, 1.0),
            (0.25, 7.703_465_477_309_97e-1),
            (0.5, 6.156_903_441_929_258e-1),
            (1.0, 4.275_835_761_558_07e-1),
            (2.0, 2.553_956_763_105_058e-1),
            (3.0, 1.790_011_511_813_9e-1),
            (4.0, 1.369_994_576_250_614e-1),
            (5.0, 1.107_046_377_330_686e-1),
            (10.0, 5.614_099_274_382_259e-2),
            (25.0, 2.254_957_243_264_135_5e-2),
            (100.0, 5.641_613_782_989_433e-3),
            (1000.0, 5.641_893_014_533_876e-4),
        ] {
            assert_relative_eq!(erfcx(z), expected, epsilon = 1e-14);
        }
    }

    /// Dense `erfcx` goldens (mpmath, 60 digits) across the direct-path range
    /// `z < 26.5`, where `erfcx` evaluates `(z*z).exp() * libm::erfc(z)`. This
    /// is the libm-bump gate (see the module doc + `Cargo.toml`): `libm` is
    /// pinned exactly, and a deliberate bump that shifted deep-tail `erfc` bits
    /// would fail here. Tolerance matches `erfcx_golden_values` (1e-14): the
    /// vendored `erfc` is bit-identical cross-platform, but the neighbouring
    /// `(z*z).exp()` is std math, so ~1-2 ULP of cross-platform drift is
    /// expected — 1e-14 clears that plus erfc's 4-ULP budget with headroom.
    #[test]
    fn erfcx_dense_direct_path_goldens() {
        for &(z, expected) in &[
            (-3.0, 1.620_598_885_399_958_6e4),
            (-2.0, 1.089_409_043_899_779_7e2),
            (-1.0, 5.008_980_080_762_283e0),
            (-0.5, 1.952_360_489_182_557e0),
            (0.5, 6.156_903_441_929_259e-1),
            (1.0, 4.275_835_761_558_07e-1),
            (1.5, 3.215_854_164_543_175e-1),
            (2.0, 2.553_956_763_105_057_5e-1),
            (2.5, 2.108_063_640_611_436e-1),
            (3.0, 1.790_011_511_813_899_6e-1),
            (3.5, 1.552_936_556_088_943e-1),
            (4.0, 1.369_994_576_250_613_8e-1),
            (4.5, 1.224_848_042_738_414_2e-1),
            (5.0, 1.107_046_377_330_686_3e-1),
            (5.5, 1.009_622_183_994_990_9e-1),
            (6.0, 9.277_656_780_053_835e-2),
            (6.5, 8.580_567_010_489_461e-2),
            (7.0, 7.980_005_432_915_294e-2),
            (7.5, 7.457_369_306_287_669e-2),
            (8.0, 6.998_516_620_088_092e-2),
            (8.5, 6.592_512_249_998_035e-2),
            (9.0, 6.230_772_403_777_468e-2),
            (9.5, 5.906_467_835_256_389e-2),
            (10.0, 5.614_099_274_382_259e-2),
            (10.5, 5.349_189_974_656_412e-2),
            (11.0, 5.108_059_475_808_844_6e-2),
            (11.5, 4.887_654_689_598_227_4e-2),
            (12.0, 4.685_422_101_489_376e-2),
            (12.5, 4.499_209_900_102_792e-2),
            (13.0, 4.327_192_186_460_969_4e-2),
            (13.5, 4.167_809_676_408_815e-2),
            (14.0, 4.019_722_865_021_846e-2),
            (14.5, 3.881_774_707_464_722e-2),
            (15.0, 3.752_960_638_850_576e-2),
            (15.5, 3.632_404_305_948_543e-2),
            (16.0, 3.519_337_782_493_084e-2),
            (16.5, 3.413_085_332_191_327_6e-2),
            (17.0, 3.313_049_999_972_554e-2),
            (17.5, 3.218_702_473_823_041e-2),
            (18.0, 3.129_571_781_590_521e-2),
            (18.5, 3.045_237_479_977_461e-2),
            (19.0, 2.965_323_064_126_216_4e-2),
            (19.5, 2.889_490_381_193_821_8e-2),
            (20.0, 2.817_434_874_105_132e-2),
            (20.5, 2.748_881_515_193_487e-2),
            (21.0, 2.683_581_315_864_795_6e-2),
            (21.5, 2.621_308_319_381_898_3e-2),
            (22.0, 2.561_857_000_587_945_3e-2),
            (22.5, 2.505_040_009_801_007_6e-2),
            (23.0, 2.450_686_208_928_260_6e-2),
            (23.5, 2.398_638_956_613_400_8e-2),
            (24.0, 2.348_754_606_368_264e-2),
            (24.5, 2.300_901_187_477_818_3e-2),
            (25.0, 2.254_957_243_264_136e-2),
            (25.5, 2.210_810_805_251_982_7e-2),
            (26.0, 2.168_358_485_056_290_7e-2),
        ] {
            assert_relative_eq!(erfcx(z), expected, max_relative = 1e-14);
        }
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn erfcx_at_zero_is_one() {
        // erfcx(0) = exp(0) * erfc(0) = 1 * 1 = 1 exactly.
        assert_eq!(erfcx(0.0), 1.0);
    }

    #[test]
    fn erfcx_monotone_decreasing() {
        let mut prev = f64::INFINITY;
        for i in -20..200 {
            let z = f64::from(i) * 0.25;
            let v = erfcx(z);
            assert!(v < prev, "not strictly decreasing at z={z}: {v} >= {prev}");
            prev = v;
        }
    }

    #[test]
    fn erfcx_large_positive_z_asymptote() {
        // erfcx(z) * z * sqrt(pi) -> 1 as z -> +inf. At z = 10 the next-order
        // correction is -1/(2 z^2) = -0.005, so the leading-term asymptote
        // only matches to about 1e-3. By z = 1e6 it's exact to f64.
        assert_relative_eq!(erfcx(1e6) * 1e6 / INV_SQRT_PI, 1.0, epsilon = 1e-12);
        assert_relative_eq!(erfcx(1e8), INV_SQRT_PI / 1e8, epsilon = 1e-15);
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn cdf_non_finite_returns_exact_limits() {
        // Φ(+inf) = 1, Φ(−inf) = 0, Φ(NaN) = NaN — and no 200-iteration spin.
        assert_eq!(cdf(f64::INFINITY), 1.0);
        assert_eq!(cdf(f64::NEG_INFINITY), 0.0);
        assert!(cdf(f64::NAN).is_nan());
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn erfcx_non_finite_returns_exact_limits() {
        // erfcx(+inf) = 0, erfcx(−inf) = +inf, erfcx(NaN) = NaN.
        assert_eq!(erfcx(f64::INFINITY), 0.0);
        assert_eq!(erfcx(f64::NEG_INFINITY), f64::INFINITY);
        assert!(erfcx(f64::NAN).is_nan());
    }

    // inverse_cdf golden values from scipy.stats.norm.ppf (which uses AS241).
    #[test]
    fn inverse_cdf_golden_values() {
        for &(u, expected) in &[
            (1e-15, -7.941_345_326_170_998),
            (1e-10, -6.361_340_902_404_056),
            (1e-6, -4.753_424_308_822_899),
            (1e-3, -3.090_232_306_167_813),
            (1e-2, -2.326_347_874_040_840_7),
            (1e-1, -1.281_551_565_544_600_4),
            (0.25, -6.744_897_501_960_817e-1),
            (0.4, -2.533_471_031_357_997e-1),
            (0.5, 0.0),
            (0.6, 2.533_471_031_357_997e-1),
            (0.75, 6.744_897_501_960_817e-1),
            (0.9, 1.281_551_565_544_600_4),
            (0.99, 2.326_347_874_040_840_7),
            (0.999, 3.090_232_306_167_813),
        ] {
            let got = inverse_cdf(u);
            if expected == 0.0 {
                assert!(got.abs() < 1e-15);
            } else {
                assert_relative_eq!(got, expected, epsilon = 1e-14);
            }
        }
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn inverse_cdf_boundary_returns_signed_infinity() {
        // Comparing to f64::INFINITY exactly is well-defined.
        assert_eq!(inverse_cdf(0.0), f64::NEG_INFINITY);
        assert_eq!(inverse_cdf(1.0), f64::INFINITY);
        assert_eq!(inverse_cdf(-0.1), f64::NEG_INFINITY);
        assert_eq!(inverse_cdf(1.1), f64::INFINITY);
    }

    #[test]
    fn inverse_cdf_nan_in_nan_out() {
        assert!(inverse_cdf(f64::NAN).is_nan());
    }

    #[test]
    fn inverse_cdf_antisymmetric() {
        // Antisymmetry is exact in real arithmetic. In f64, `1 - u` rounds to
        // a slightly different value than `u` reflected, so we get equality up
        // to the f64 representation precision of `1 - u`, not algorithmic
        // precision. The bound loosens as `u` approaches 0 or 1.
        for &u in &[0.01_f64, 0.1, 0.25, 0.4] {
            assert_relative_eq!(inverse_cdf(1.0 - u), -inverse_cdf(u), epsilon = 1e-13);
        }
        for &u in &[1e-6_f64, 1e-3] {
            assert_relative_eq!(inverse_cdf(1.0 - u), -inverse_cdf(u), epsilon = 1e-10);
        }
    }

    #[test]
    fn inverse_cdf_roundtrip_with_cdf() {
        for &u in &[
            1e-6_f64,
            1e-3,
            0.01,
            0.1,
            0.25,
            0.5,
            0.75,
            0.9,
            0.99,
            1.0 - 1e-6,
        ] {
            let z = inverse_cdf(u);
            assert_relative_eq!(cdf(z), u, epsilon = 1e-10);
        }
    }
}
