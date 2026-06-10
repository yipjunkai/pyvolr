//! Black-Scholes-Merton pricing of European options with continuous dividend yield.
//!
//! References:
//!   - Hull, J. C. (2017). *Options, Futures, and Other Derivatives* (10th ed.), Ch. 17.
//!   - Merton, R. C. (1973). Theory of rational option pricing.

/// Option type. Maps to/from `i8` (`>=0` -> Call, `<0` -> Put) for FFI.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Flag {
    /// Call option: right to buy at strike.
    Call,
    /// Put option: right to sell at strike.
    Put,
}

impl Flag {
    /// Decode from the FFI representation. Non-negative -> Call, negative -> Put.
    #[inline]
    pub fn from_i8(v: i8) -> Self {
        if v >= 0 {
            Flag::Call
        } else {
            Flag::Put
        }
    }
}

/// Compute `d1` and `d2` for given inputs. Assumes `t > 0`, `sigma > 0`.
#[inline]
pub fn d1_d2(s: f64, k: f64, t: f64, r: f64, q: f64, sigma: f64) -> (f64, f64) {
    let sqrt_t = t.sqrt();
    let vol_sqrt_t = sigma * sqrt_t;
    let d1 = ((s / k).ln() + (r - q + 0.5 * sigma * sigma) * t) / vol_sqrt_t;
    let d2 = d1 - vol_sqrt_t;
    (d1, d2)
}

/// European option price under BSM with continuous dividend yield `q`.
///
/// Degenerate inputs:
///   - `t <= 0`: returns intrinsic value `max(0, +/-(s - k))`.
///   - `sigma <= 0`: returns the discounted intrinsic of the deterministic forward
///     `F = s * exp((r - q) * t)`.
pub fn price(flag: Flag, s: f64, k: f64, t: f64, r: f64, q: f64, sigma: f64) -> f64 {
    if t <= 0.0 {
        return intrinsic(flag, s, k);
    }
    let forward = s * ((r - q) * t).exp();
    let disc_r = (-r * t).exp();
    let x = (forward / k).ln();
    if sigma <= 0.0 || !x.is_finite() {
        // No f64-resolvable optionality, so the price is the discounted
        // intrinsic of the forward F = S·e^((r−q)T):
        //   - sigma <= 0: deterministic forward.
        //   - x = ln(F/K) non-finite: F or K ≤ 0, or F/K over/underflows f64
        //     (extreme moneyness) — the option is then at its intrinsic to f64.
        return disc_r * intrinsic(flag, forward, k);
    }
    // Back the public pricer with the normalised-Black engine (the same ~1-ULP
    // evaluator the IV solver uses) rather than the textbook
    // `S·e^(−qT)·Φ(d1) − K·e^(−rT)·Φ(d2)`, which cancels catastrophically in the
    // deep-OTM short-expiry corner (h ≈ −5, t ≈ 0.01) and keeps only ~9 digits.
    //
    //   undiscounted OTM value = √(F·K)·b(−|x|, s),  x = ln(F/K),  s = σ·√T
    //   price = e^(−rT)·[√(F·K)·b(−|x|, s) + intrinsic]
    //
    // Always evaluate the OTM side through the engine — `b(−|x|, s) ≤ e^(−|x|/2)
    // ≤ 1`, so it never overflows and keeps full relative precision — and add
    // the EXACT discounted intrinsic for the ITM side in price space. Folding
    // the intrinsic through the normalised `√(F·K)·b(x, s)` reconstruction (as
    // the engine does internally for ITM `x`) loses precision when the
    // normalised intrinsic `e^(x/2)` is large (deep ITM). Splitting it out also
    // makes put-call parity EXACT: C and P share the same `b(−|x|, s)` term, so
    // `C − P = e^(−rT)(F − K)` to the last bit.
    let s_norm = sigma * t.sqrt();
    let sqrt_fk = (forward * k).sqrt();
    let extrinsic = disc_r * sqrt_fk * crate::iv::normalised_black_call(-x.abs(), s_norm);
    let intrinsic_disc = match flag {
        Flag::Call if x > 0.0 => disc_r * (forward - k),
        Flag::Put if x < 0.0 => disc_r * (k - forward),
        _ => 0.0,
    };
    extrinsic + intrinsic_disc
}

#[inline]
fn intrinsic(flag: Flag, s: f64, k: f64) -> f64 {
    match flag {
        Flag::Call => (s - k).max(0.0),
        Flag::Put => (k - s).max(0.0),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn call_atm_no_div_known_value() {
        // Hull worked example: S=K=100, r=5%, sigma=20%, T=1 year -> call ~10.4506
        let p = price(Flag::Call, 100.0, 100.0, 1.0, 0.05, 0.0, 0.20);
        assert_relative_eq!(p, 10.450_583_572_185_565, epsilon = 1e-10);
    }

    #[test]
    fn put_atm_no_div_known_value() {
        let p = price(Flag::Put, 100.0, 100.0, 1.0, 0.05, 0.0, 0.20);
        assert_relative_eq!(p, 5.573_526_022_256_971, epsilon = 1e-10);
    }

    #[test]
    fn put_call_parity_holds() {
        // C - P = S*exp(-qT) - K*exp(-rT)
        let (s, k, t, r, q, sigma) = (100.0, 105.0, 0.5, 0.05, 0.02, 0.25);
        let c = price(Flag::Call, s, k, t, r, q, sigma);
        let p = price(Flag::Put, s, k, t, r, q, sigma);
        let parity_lhs = c - p;
        let parity_rhs = s * (-q * t).exp() - k * (-r * t).exp();
        assert_relative_eq!(parity_lhs, parity_rhs, epsilon = 1e-12);
    }

    #[test]
    fn zero_time_returns_intrinsic() {
        assert_relative_eq!(price(Flag::Call, 110.0, 100.0, 0.0, 0.05, 0.0, 0.2), 10.0);
        assert_relative_eq!(price(Flag::Put, 90.0, 100.0, 0.0, 0.05, 0.0, 0.2), 10.0);
        assert_relative_eq!(price(Flag::Call, 90.0, 100.0, 0.0, 0.05, 0.0, 0.2), 0.0);
    }

    /// Price goldens from 50-digit mpmath BSM, rounded to f64. These pin the
    /// engine-backed pricer's tail accuracy: the pre-2026-06 textbook
    /// `S·Φ(d1) − K·Φ(d2)` form cancelled here, keeping only ~9 digits (~1e-9
    /// relative) in the deep-OTM short-expiry corner — so this test passing at
    /// 1e-12 is itself the regression guard against reverting to it.
    ///
    /// The deepest cell (`deep_otm_put`, ~1e-201) lands at ~1.7e-13 relative,
    /// not ~1 ULP: that residual is conditioning, not solver error. The price
    /// is exponentially sensitive to `x = ln(F/K)`, and computing `x` in f64
    /// carries ~½ ULP that the deep-OTM sensitivity (~10³ here) amplifies. The
    /// mpmath oracle forms `x` at 60 digits, so it has no such rounding; given
    /// f64 inputs this is the floor. Regenerate with mpmath (mp.dps=60):
    /// `C = e^(−rT)(F·Φ(d1) − K·Φ(d2))`, `F = S·e^((r−q)T)`.
    #[test]
    fn price_matches_mpmath_goldens() {
        // (label, flag, [s, k, t, r, q, sigma], expected)
        let cases: &[(&str, Flag, [f64; 6], f64)] = &[
            (
                "benign_call",
                Flag::Call,
                [100.0, 100.0, 1.0, 0.05, 0.0, 0.2],
                10.450_583_572_185_568,
            ),
            (
                "benign_put",
                Flag::Put,
                [100.0, 105.0, 0.5, 0.05, 0.02, 0.25],
                8.923_052_137_509_15,
            ),
            (
                "deep_otm_put",
                Flag::Put,
                [100.0, 60.0, 0.02, 0.05, 0.0, 0.12],
                1.743_957_269_143_156e-201,
            ),
            (
                "very_deep_otm_call",
                Flag::Call,
                [100.0, 300.0, 0.25, 0.03, 0.0, 0.2],
                7.924_761_804_025_656e-28,
            ),
            (
                "deep_itm_call",
                Flag::Call,
                [100.0, 50.0, 0.5, 0.05, 0.0, 0.2],
                51.234_504_744_174_35,
            ),
        ];
        for &(label, flag, [s, k, t, r, q, sigma], expected) in cases {
            let got = price(flag, s, k, t, r, q, sigma);
            let rel = ((got - expected) / expected).abs();
            assert!(
                rel < 1e-12,
                "{label}: price {got:e} vs mpmath {expected:e} (rel {rel:e})"
            );
        }
    }

    /// Degenerate spot/strike: the engine path would NaN (`x = ln(F/K)` → ±inf,
    /// `√(F·K) = 0` ⇒ `0·inf`). Fuzz regressions `crash-bd4b…` (K=0) and
    /// `crash-f94d…` (S=0). Both reduce to the discounted forward intrinsic and
    /// must stay finite with parity intact.
    #[test]
    #[allow(clippy::float_cmp)]
    fn degenerate_zero_spot_or_strike() {
        let (t, r, q, sigma) = (0.5, 0.05, 0.02, 0.25);
        // K = 0: call = S·e^(−qT), put = 0.
        let s = 100.0;
        let ck = price(Flag::Call, s, 0.0, t, r, q, sigma);
        let pk = price(Flag::Put, s, 0.0, t, r, q, sigma);
        assert!(ck.is_finite() && pk.is_finite());
        assert_relative_eq!(ck, s * (-q * t).exp(), epsilon = 1e-13);
        assert_eq!(pk, 0.0);
        assert_relative_eq!(ck - pk, s * (-q * t).exp(), epsilon = 1e-13);
        // S = 0: call = 0, put = K·e^(−rT).
        let k = 100.0;
        let cs = price(Flag::Call, 0.0, k, t, r, q, sigma);
        let ps = price(Flag::Put, 0.0, k, t, r, q, sigma);
        assert!(cs.is_finite() && ps.is_finite());
        assert_eq!(cs, 0.0);
        assert_relative_eq!(ps, k * (-r * t).exp(), epsilon = 1e-13);
        assert_relative_eq!(cs - ps, -k * (-r * t).exp(), epsilon = 1e-13);
    }

    /// Extreme moneyness where `F/K` overflows f64 (`x = ln(F/K) = +inf`): the
    /// engine's `e^(x/2)` would overflow to a non-finite price. Fuzz regression
    /// `crash-5907…`. Must fall back to intrinsic, stay finite, respect parity.
    #[test]
    #[allow(clippy::float_cmp)]
    fn extreme_moneyness_stays_finite() {
        let (t, r, q, sigma) = (0.5, 0.05, 0.0, 0.20);
        let (s, k) = (1.0e5, 1.0e-305); // F/K ≈ 1e310 > f64::MAX
        let c = price(Flag::Call, s, k, t, r, q, sigma);
        let p = price(Flag::Put, s, k, t, r, q, sigma);
        assert!(c.is_finite() && p.is_finite(), "c={c} p={p}");
        // Call ≈ discounted forward (strike negligible); put is worthless.
        assert_relative_eq!(c, s * (-q * t).exp(), max_relative = 1e-12);
        assert_eq!(p, 0.0);
        let parity = s * (-q * t).exp() - k * (-r * t).exp();
        assert_relative_eq!(c - p, parity, max_relative = 1e-12);
    }

    #[test]
    fn zero_vol_returns_discounted_forward_intrinsic() {
        let (s, k, t, r, q) = (100.0, 100.0, 1.0, 0.05, 0.0);
        let p = price(Flag::Call, s, k, t, r, q, 0.0);
        // F = 100*exp(0.05) ~ 105.1271; intrinsic = 5.1271; disc = 5.1271*exp(-0.05) ~ 4.8770
        let forward = s * ((r - q) * t).exp();
        let expected = (-r * t).exp() * (forward - k).max(0.0);
        assert_relative_eq!(p, expected, epsilon = 1e-12);
    }
}
