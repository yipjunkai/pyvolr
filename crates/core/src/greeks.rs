//! Analytical Greeks for European BSM options.
//!
//! Conventions:
//!   - `vega`: change in price per unit change in `sigma` (i.e. per "1.00 vol"), not per 1% vol.
//!   - `theta`: per year (annualized). Divide by 365 (or 252) for daily theta.
//!   - `rho`: per unit change in `r`, not per 1% r.
//!
//! All return `0.0` for the degenerate `t <= 0` or `sigma <= 0` cases except
//! `delta`, which returns the limiting step value as it does in textbook treatments.

use crate::bsm::{d1_d2, Flag};
use crate::normal::{cdf, pdf};

/// First derivative of price with respect to spot. Range: `(-exp(-qT), exp(-qT))`.
pub fn delta(flag: Flag, s: f64, k: f64, t: f64, r: f64, q: f64, sigma: f64) -> f64 {
    if t <= 0.0 || sigma <= 0.0 {
        // ITM is decided by the moneyness the limiting price actually uses:
        //   - t <= 0 (expiry): price = max(0, ±(S − K)), a step on SPOT vs K.
        //   - sigma <= 0, t > 0: price = e^(−rT)·intrinsic(F, K) with the
        //     forward F = S·e^((r−q)T) (see `bsm::price`), so the step is on
        //     FORWARD moneyness and the slope is d/dS[e^(−rT)(F − K)] = e^(−qT).
        //     Deciding on spot here is wrong: for S in (K·e^(−(r−q)T), K) the
        //     option is forward-ITM yet spot-OTM, and delta must be ±e^(−qT),
        //     not 0 — finite-differencing this library's own zero-vol price
        //     exposes the inconsistency.
        let moneyness = if t > 0.0 { s * ((r - q) * t).exp() } else { s };
        let in_the_money = match flag {
            Flag::Call => moneyness > k,
            Flag::Put => moneyness < k,
        };
        let disc_q = (-q * t.max(0.0)).exp();
        return if in_the_money {
            match flag {
                Flag::Call => disc_q,
                Flag::Put => -disc_q,
            }
        } else {
            0.0
        };
    }
    let (d1, _) = d1_d2(s, k, t, r, q, sigma);
    let disc_q = (-q * t).exp();
    match flag {
        Flag::Call => disc_q * cdf(d1),
        // `-cdf(-d1)` routes via the `erfcx` tail when d1 is large positive,
        // keeping put delta f64-precise on deep-OTM strikes. The natural
        // `cdf(d1) - 1.0` form cancels catastrophically there (cdf(d1)
        // saturates to 1.0 once `1 - cdf(d1) < f64::EPSILON`, around d1 ≈ 8).
        Flag::Put => -disc_q * cdf(-d1),
    }
}

/// Second derivative of price with respect to spot. Always non-negative; identical for call and put.
pub fn gamma(s: f64, k: f64, t: f64, r: f64, q: f64, sigma: f64) -> f64 {
    if t <= 0.0 || sigma <= 0.0 {
        return 0.0;
    }
    let (d1, _) = d1_d2(s, k, t, r, q, sigma);
    let disc_q = (-q * t).exp();
    disc_q * pdf(d1) / (s * sigma * t.sqrt())
}

/// Derivative of price with respect to volatility, per unit vol (not per 1%).
/// Always non-negative; identical for call and put.
pub fn vega(s: f64, k: f64, t: f64, r: f64, q: f64, sigma: f64) -> f64 {
    if t <= 0.0 || sigma <= 0.0 {
        return 0.0;
    }
    let (d1, _) = d1_d2(s, k, t, r, q, sigma);
    let disc_q = (-q * t).exp();
    s * disc_q * pdf(d1) * t.sqrt()
}

/// Calendar theta: MINUS the derivative of price with respect to
/// time-to-expiry, per year. Typically negative for long options (value
/// decays as the clock advances).
/// Divide by 365 (or 252) for the per-day convention used by some libraries.
pub fn theta(flag: Flag, s: f64, k: f64, t: f64, r: f64, q: f64, sigma: f64) -> f64 {
    if t <= 0.0 || sigma <= 0.0 {
        return 0.0;
    }
    let (d1, d2) = d1_d2(s, k, t, r, q, sigma);
    let disc_q = (-q * t).exp();
    let disc_r = (-r * t).exp();
    let common = -s * disc_q * pdf(d1) * sigma / (2.0 * t.sqrt());
    match flag {
        Flag::Call => common - r * k * disc_r * cdf(d2) + q * s * disc_q * cdf(d1),
        Flag::Put => common + r * k * disc_r * cdf(-d2) - q * s * disc_q * cdf(-d1),
    }
}

/// Derivative of price with respect to the risk-free rate, per unit `r`.
pub fn rho(flag: Flag, s: f64, k: f64, t: f64, r: f64, q: f64, sigma: f64) -> f64 {
    if t <= 0.0 || sigma <= 0.0 {
        return 0.0;
    }
    let (_, d2) = d1_d2(s, k, t, r, q, sigma);
    let disc_r = (-r * t).exp();
    match flag {
        Flag::Call => k * t * disc_r * cdf(d2),
        Flag::Put => -k * t * disc_r * cdf(-d2),
    }
}

/// Compute all five Greeks in a single pass, sharing the `d1_d2`, `disc_q`,
/// `disc_r`, `cdf`, and `pdf` evaluations that would otherwise be repeated
/// across `delta`/`gamma`/`vega`/`theta`/`rho`. Returns `(delta, gamma, vega,
/// theta, rho)`. Numerically identical to calling the per-Greek functions
/// individually (verified by `tests::all_matches_individual_*`).
pub fn all(
    flag: Flag,
    s: f64,
    k: f64,
    t: f64,
    r: f64,
    q: f64,
    sigma: f64,
) -> (f64, f64, f64, f64, f64) {
    // Degenerate regime: delta has a non-zero limiting step; all others vanish.
    // Defer to `delta` so the textbook step value (`±disc_q` ITM, `0` OTM) is
    // computed in exactly one place.
    if t <= 0.0 || sigma <= 0.0 {
        return (delta(flag, s, k, t, r, q, sigma), 0.0, 0.0, 0.0, 0.0);
    }
    let (d1, d2) = d1_d2(s, k, t, r, q, sigma);
    let sqrt_t = t.sqrt();
    let disc_q = (-q * t).exp();
    let disc_r = (-r * t).exp();
    let pd1 = pdf(d1);

    let (delta_v, theta_v, rho_v) = match flag {
        Flag::Call => {
            // `cdf(d1)`/`cdf(d2)` only feed the call arm; the put arm uses the
            // `cdf(-d1)`/`cdf(-d2)` tail forms, so keep each inside its arm
            // rather than computing both pairs unconditionally.
            let nd1 = cdf(d1);
            let nd2 = cdf(d2);
            let common = -s * disc_q * pd1 * sigma / (2.0 * sqrt_t);
            (
                disc_q * nd1,
                common - r * k * disc_r * nd2 + q * s * disc_q * nd1,
                k * t * disc_r * nd2,
            )
        }
        Flag::Put => {
            // Use `cdf(-d1)` / `cdf(-d2)` directly throughout (delta, theta,
            // rho) so the `erfcx`-tail branch handles large `|d1|` / `|d2|`
            // without precision loss — `cdf(d1) - 1.0` cancels catastrophically
            // for deep-OTM puts (d1 large positive, cdf(d1) saturates to 1.0).
            let neg_nd1 = cdf(-d1);
            let neg_nd2 = cdf(-d2);
            let common = -s * disc_q * pd1 * sigma / (2.0 * sqrt_t);
            (
                -disc_q * neg_nd1,
                common + r * k * disc_r * neg_nd2 - q * s * disc_q * neg_nd1,
                -k * t * disc_r * neg_nd2,
            )
        }
    };
    let gamma_v = disc_q * pd1 / (s * sigma * sqrt_t);
    let vega_v = s * disc_q * pd1 * sqrt_t;
    (delta_v, gamma_v, vega_v, theta_v, rho_v)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bsm::price;
    use approx::assert_relative_eq;

    /// Verify analytical Greek matches a central finite-difference approximation.
    fn finite_diff<F: Fn(f64) -> f64>(f: F, x: f64, h: f64) -> f64 {
        (f(x + h) - f(x - h)) / (2.0 * h)
    }

    #[test]
    fn delta_matches_fd() {
        let (s, k, t, r, q, sigma) = (100.0, 105.0, 0.5, 0.05, 0.02, 0.25);
        let analytical = delta(Flag::Call, s, k, t, r, q, sigma);
        let fd = finite_diff(|x| price(Flag::Call, x, k, t, r, q, sigma), s, 0.01);
        assert_relative_eq!(analytical, fd, epsilon = 1e-6);
    }

    #[test]
    fn gamma_matches_fd() {
        let (s, k, t, r, q, sigma) = (100.0, 105.0, 0.5, 0.05, 0.02, 0.25);
        let analytical = gamma(s, k, t, r, q, sigma);
        // gamma = ∂²price/∂S²: second central difference of price wrt spot.
        let direct = (price(Flag::Call, s + 0.01, k, t, r, q, sigma)
            - 2.0 * price(Flag::Call, s, k, t, r, q, sigma)
            + price(Flag::Call, s - 0.01, k, t, r, q, sigma))
            / (0.01 * 0.01);
        assert_relative_eq!(analytical, direct, epsilon = 1e-3);
    }

    #[test]
    fn vega_matches_fd() {
        let (s, k, t, r, q, sigma) = (100.0, 105.0, 0.5, 0.05, 0.02, 0.25);
        let analytical = vega(s, k, t, r, q, sigma);
        let fd = finite_diff(|v| price(Flag::Call, s, k, t, r, q, v), sigma, 1e-4);
        assert_relative_eq!(analytical, fd, epsilon = 1e-6);
    }

    #[test]
    fn rho_matches_fd() {
        let (s, k, t, r, q, sigma) = (100.0, 105.0, 0.5, 0.05, 0.02, 0.25);
        let analytical = rho(Flag::Call, s, k, t, r, q, sigma);
        let fd = finite_diff(|x| price(Flag::Call, s, k, t, x, q, sigma), r, 1e-5);
        assert_relative_eq!(analytical, fd, epsilon = 1e-6);
    }

    #[test]
    fn theta_matches_fd_via_negative_t() {
        // theta = -dC/dT
        let (s, k, t, r, q, sigma) = (100.0, 105.0, 0.5, 0.05, 0.02, 0.25);
        let analytical = theta(Flag::Call, s, k, t, r, q, sigma);
        let fd = finite_diff(|x| price(Flag::Call, s, k, x, r, q, sigma), t, 1e-5);
        assert_relative_eq!(analytical, -fd, epsilon = 1e-5);
    }

    /// Drift guard: `all()` must agree with the per-Greek functions bit-for-bit
    /// across both option flags, including the t<=0 / sigma<=0 degenerate paths.
    /// Tolerances are tight (relative 0 or 1e-15) because `all()` shares the
    /// same subexpressions, so floating-point rounding lines up exactly.
    #[test]
    fn all_matches_individual_at_grid() {
        let grid: &[(f64, f64, f64, f64, f64, f64)] = &[
            (100.0, 100.0, 1.0, 0.05, 0.0, 0.20),
            (100.0, 105.0, 0.5, 0.05, 0.02, 0.25),
            (100.0, 80.0, 0.05, 0.03, 0.01, 0.45),
            (100.0, 200.0, 0.5, 0.05, 0.0, 0.30), // deep OTM call
            (1000.0, 100.0, 0.01, 0.05, 0.0, 0.20), // deep ITM call
        ];
        for &(s, k, t, r, q, sigma) in grid {
            for &flag in &[Flag::Call, Flag::Put] {
                let (d, g, v, th, rh) = all(flag, s, k, t, r, q, sigma);
                assert_relative_eq!(d, delta(flag, s, k, t, r, q, sigma), max_relative = 1e-15);
                assert_relative_eq!(g, gamma(s, k, t, r, q, sigma), max_relative = 1e-15);
                assert_relative_eq!(v, vega(s, k, t, r, q, sigma), max_relative = 1e-15);
                assert_relative_eq!(th, theta(flag, s, k, t, r, q, sigma), max_relative = 1e-15);
                assert_relative_eq!(rh, rho(flag, s, k, t, r, q, sigma), max_relative = 1e-15);
            }
        }
    }

    /// Deep-OTM put precision: when `d1` is large positive, `cdf(d1)` rounds
    /// to exactly `1.0` in f64 (since `1 − cdf(d1) < ε`), so the old form
    /// `cdf(d1) − 1.0` evaluates to `0.0` exactly and put delta drops its
    /// sign and magnitude. The fix routes through `−cdf(−d1)`, where the
    /// `erfcx` tail in `normal::cdf` keeps the value f64-precise.
    #[test]
    fn put_delta_deep_otm_retains_precision() {
        // S=1000, K=100 (10x OTM put), T=0.5y, σ=20% → d1 ≈ 16.5.
        // cdf(16.5) saturates to 1.0; cdf(-16.5) ≈ 1.5e-61 via erfcx.
        let pd = delta(Flag::Put, 1000.0, 100.0, 0.5, 0.05, 0.0, 0.20);
        // Must be strictly negative (the old form returned exactly 0.0).
        assert!(
            pd < 0.0,
            "put delta lost sign at deep OTM (returned {pd:e}); old `cdf(d1) - 1.0` form"
        );
        // Must be tiny but non-zero — the call delta is essentially exp(-qT)=1
        // here, so the put delta is essentially -1·(1 - cdf(d1)) ≈ -1.5e-61.
        assert!(
            pd.abs() < 1e-50,
            "put delta should be ~1e-61 at deep OTM, got {pd:e}"
        );
        assert!(pd.abs() > 0.0, "put delta underflowed to zero ({pd:e})");
    }

    /// Identity guard: `call_delta − put_delta = exp(−qT)` for all inputs
    /// (this is `d(C−P)/dS` from put-call parity, since `C−P = S·exp(−qT) − K·exp(−rT)`).
    /// The new put-delta form preserves this identity to f64 even where the
    /// old form failed (call delta is correct in both forms; only put delta
    /// changes, and the identity is now exact at deep OTM too).
    #[test]
    fn put_call_delta_parity_holds_at_deep_otm() {
        let (s, k, t, r, q, sigma) = (1000.0, 100.0, 0.5, 0.05, 0.0, 0.20);
        let cd = delta(Flag::Call, s, k, t, r, q, sigma);
        let pd = delta(Flag::Put, s, k, t, r, q, sigma);
        let disc_q = (-q * t).exp();
        // call_delta + |put_delta| = exp(-qT) (since put_delta is negative).
        assert_relative_eq!(cd - pd, disc_q, max_relative = 1e-15);
    }

    /// σ=0 delta must read FORWARD moneyness and equal the slope of this
    /// library's own zero-vol price — not spot moneyness (the old bug).
    #[test]
    #[allow(clippy::float_cmp)]
    fn delta_zero_vol_uses_forward_moneyness() {
        let h = 1e-3;
        // Spot-OTM but forward-ITM call: S=97 < K=100, F = 97·e^(0.05) ≈ 101.97
        // > 100. Old spot test (97 > 100) returned 0; true delta is e^(−qT) = 1.
        let (s, k, t, r, q) = (97.0, 100.0, 1.0, 0.05, 0.0);
        let dc = delta(Flag::Call, s, k, t, r, q, 0.0);
        assert_relative_eq!(dc, (-q * t).exp(), epsilon = 1e-15);
        let fd_c = (price(Flag::Call, s + h, k, t, r, q, 0.0)
            - price(Flag::Call, s - h, k, t, r, q, 0.0))
            / (2.0 * h);
        assert_relative_eq!(dc, fd_c, epsilon = 1e-9);
        // The put at the same point is forward-OTM ⇒ delta 0.
        assert_eq!(delta(Flag::Put, s, k, t, r, q, 0.0), 0.0);

        // Dividend case where the forward sits BELOW spot: S=K=100, r=4%, q=8%
        // ⇒ F = 100·e^(−0.04) ≈ 96.08 < K. The put is forward-ITM and must read
        // −e^(−qT); the call is forward-OTM ⇒ 0.
        let (s2, k2, t2, r2, q2): (f64, f64, f64, f64, f64) = (100.0, 100.0, 1.0, 0.04, 0.08);
        assert!(s2 * ((r2 - q2) * t2).exp() < k2);
        let dp = delta(Flag::Put, s2, k2, t2, r2, q2, 0.0);
        assert_relative_eq!(dp, -(-q2 * t2).exp(), epsilon = 1e-15);
        let fd_p = (price(Flag::Put, s2 + h, k2, t2, r2, q2, 0.0)
            - price(Flag::Put, s2 - h, k2, t2, r2, q2, 0.0))
            / (2.0 * h);
        assert_relative_eq!(dp, fd_p, epsilon = 1e-9);
        assert_eq!(delta(Flag::Call, s2, k2, t2, r2, q2, 0.0), 0.0);
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn all_matches_individual_degenerate() {
        // t == 0 and sigma == 0 paths.
        let degenerate: &[(f64, f64, f64, f64, f64, f64)] = &[
            (110.0, 100.0, 0.0, 0.05, 0.0, 0.20), // t = 0, ITM call
            (90.0, 100.0, 0.0, 0.05, 0.0, 0.20),  // t = 0, OTM call / ITM put
            (100.0, 100.0, 1.0, 0.05, 0.0, 0.0),  // sigma = 0
        ];
        for &(s, k, t, r, q, sigma) in degenerate {
            for &flag in &[Flag::Call, Flag::Put] {
                let (d, g, v, th, rh) = all(flag, s, k, t, r, q, sigma);
                assert_eq!(d, delta(flag, s, k, t, r, q, sigma));
                assert_eq!(g, gamma(s, k, t, r, q, sigma));
                assert_eq!(v, vega(s, k, t, r, q, sigma));
                assert_eq!(th, theta(flag, s, k, t, r, q, sigma));
                assert_eq!(rh, rho(flag, s, k, t, r, q, sigma));
            }
        }
    }
}
