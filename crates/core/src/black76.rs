//! Black-76 (Black 1976) pricing for European options on futures / forwards.
//!
//! Black-76 is the BSM formula evaluated at `S = F` (the forward price) with
//! the dividend yield `q` set equal to the risk-free rate `r`. This makes the
//! cost-of-carry zero, which is correct for forwards: the forward is already
//! priced for delivery at expiry, so it doesn't drift at `r`. The same
//! substitution makes price, delta, gamma, vega, and theta of Black-76
//! identical to BSM-with-`q=r`. Rho is the one Greek that diverges — in
//! Black-76 only the discount factor depends on `r` (d1/d2 don't, since the
//! `r-q` term in the drift vanishes), so `rho = -T * price` for both call
//! and put. See `rho` below.
//!
//! References:
//!   - Black, F. (1976). The pricing of commodity contracts. *Journal of
//!     Financial Economics*, 3(1-2), 167-179.
//!   - Hull, J. C. (2017). *Options, Futures, and Other Derivatives* (10th
//!     ed.), Ch. 18 (Futures Options).
//!
//! Put-call parity: `C - P = exp(-r*T) * (F - K)`.

use crate::bsm::{d1_d2, price as bsm_price, Flag};
use crate::greeks;
use crate::normal::{cdf, pdf};

/// European Black-76 option price on a futures/forward `f` with strike `k`.
///
/// Equivalent to `bsm::price(flag, f, k, t, r, r, sigma)` — both substitute
/// `S=F` and `q=r`, which is the defining specialization of Black-76.
pub fn price(flag: Flag, f: f64, k: f64, t: f64, r: f64, sigma: f64) -> f64 {
    bsm_price(flag, f, k, t, r, r, sigma)
}

/// First derivative of price with respect to the forward. Range: `(-exp(-rT), exp(-rT))`.
pub fn delta(flag: Flag, f: f64, k: f64, t: f64, r: f64, sigma: f64) -> f64 {
    greeks::delta(flag, f, k, t, r, r, sigma)
}

/// Second derivative of price with respect to the forward. Identical for call and put.
pub fn gamma(f: f64, k: f64, t: f64, r: f64, sigma: f64) -> f64 {
    greeks::gamma(f, k, t, r, r, sigma)
}

/// Derivative of price with respect to volatility, per unit vol (not per 1%).
pub fn vega(f: f64, k: f64, t: f64, r: f64, sigma: f64) -> f64 {
    greeks::vega(f, k, t, r, r, sigma)
}

/// Calendar theta: MINUS the derivative of price with respect to
/// time-to-expiry, per year (typically negative). See `greeks::theta`.
pub fn theta(flag: Flag, f: f64, k: f64, t: f64, r: f64, sigma: f64) -> f64 {
    greeks::theta(flag, f, k, t, r, r, sigma)
}

/// Derivative of price with respect to the risk-free rate, per unit `r`.
///
/// Specific to Black-76: only the discount factor `exp(-r*T)` depends on `r`
/// (the d1/d2 drift term vanishes since `q=r`), so the chain rule collapses
/// to `dC/dr = -T * exp(-r*T) * [F*N(d1) - K*N(d2)] = -T * C`. Same for puts.
/// This is NOT the same expression as `bsm::greeks::rho(flag, f, k, t, r, r, sigma)`,
/// which holds `q` fixed while differentiating in `r` and therefore picks up
/// an extra `K*t*disc_r*N(d2)`-style term that doesn't exist in Black-76.
pub fn rho(flag: Flag, f: f64, k: f64, t: f64, r: f64, sigma: f64) -> f64 {
    -t * price(flag, f, k, t, r, sigma)
}

/// Compute all five Black-76 Greeks in a single pass. Shares the `d1_d2`,
/// discount factor, `cdf`, and `pdf` evaluations across delta/gamma/vega/
/// theta, and reuses the rebuilt call/put price to derive rho (= `-T·price`).
/// Numerically identical to calling the per-Greek functions individually.
pub fn all(flag: Flag, f: f64, k: f64, t: f64, r: f64, sigma: f64) -> (f64, f64, f64, f64, f64) {
    // Degenerate regime: defer delta to its existing scalar path, and rho stays
    // `-T·price` (Black-76 rho is defined off the price, not the d1/d2 form).
    if t <= 0.0 || sigma <= 0.0 {
        let delta_v = greeks::delta(flag, f, k, t, r, r, sigma);
        let price_v = bsm_price(flag, f, k, t, r, r, sigma);
        return (delta_v, 0.0, 0.0, 0.0, -t * price_v);
    }
    // Black-76 specialises BSM with q = r, so disc_q == disc_r and the r-q
    // drift in d1 vanishes.
    let (d1, d2) = d1_d2(f, k, t, r, r, sigma);
    let sqrt_t = t.sqrt();
    let disc = (-r * t).exp();
    let pd1 = pdf(d1);

    let (delta_v, theta_v) = match flag {
        Flag::Call => {
            let nd1 = cdf(d1);
            let nd2 = cdf(d2);
            let common = -f * disc * pd1 * sigma / (2.0 * sqrt_t);
            (disc * nd1, common - r * k * disc * nd2 + r * f * disc * nd1)
        }
        Flag::Put => {
            // Route put delta/theta through `cdf(-d1)` / `cdf(-d2)` so the
            // `erfcx` tail handles deep-OTM puts without the `cdf(d1) - 1.0`
            // catastrophic cancellation (same argument as `greeks::all`).
            let neg_nd1 = cdf(-d1);
            let neg_nd2 = cdf(-d2);
            let common = -f * disc * pd1 * sigma / (2.0 * sqrt_t);
            (
                -disc * neg_nd1,
                common + r * k * disc * neg_nd2 - r * f * disc * neg_nd1,
            )
        }
    };
    let gamma_v = disc * pd1 / (f * sigma * sqrt_t);
    let vega_v = f * disc * pd1 * sqrt_t;
    // Black-76 rho is defined off the price (rho = −T·price), not the d1/d2
    // form. Route through `bsm_price` — now the normalised-Black engine — so
    // `rho_v` stays bit-equal to the standalone `rho()` path, which is also
    // `−t · bsm_price(...)`. (This supersedes the old inline `f·disc·N(d1) −
    // k·disc·N(d2)`, which mirrored the textbook pricer's arithmetic ordering.)
    let rho_v = -t * bsm_price(flag, f, k, t, r, r, sigma);
    (delta_v, gamma_v, vega_v, theta_v, rho_v)
}

// Higher-order Greeks. Each is a q = r specialization of its BSM counterpart —
// none differentiates with respect to `r`, so (unlike `rho`) there is no
// Black-76-specific divergence and the delegation is exact. See `greeks` for
// the formulas, sign conventions, and degenerate-case policy.

/// Vanna: `d(vega)/dF = d(delta)/dsigma`. Per unit vol. See `greeks::vanna`.
pub fn vanna(f: f64, k: f64, t: f64, r: f64, sigma: f64) -> f64 {
    greeks::vanna(f, k, t, r, r, sigma)
}

/// Vomma (volga): `d(vega)/dsigma`. Per unit vol. See `greeks::vomma`.
pub fn vomma(f: f64, k: f64, t: f64, r: f64, sigma: f64) -> f64 {
    greeks::vomma(f, k, t, r, r, sigma)
}

/// Charm (delta decay): `-d(delta)/dt` per year. Flag-dependent. See `greeks::charm`.
pub fn charm(flag: Flag, f: f64, k: f64, t: f64, r: f64, sigma: f64) -> f64 {
    greeks::charm(flag, f, k, t, r, r, sigma)
}

/// Speed: `d(gamma)/dF`. See `greeks::speed`.
pub fn speed(f: f64, k: f64, t: f64, r: f64, sigma: f64) -> f64 {
    greeks::speed(f, k, t, r, r, sigma)
}

/// Zomma: `d(gamma)/dsigma`. Per unit vol. See `greeks::zomma`.
pub fn zomma(f: f64, k: f64, t: f64, r: f64, sigma: f64) -> f64 {
    greeks::zomma(f, k, t, r, r, sigma)
}

/// Color (gamma decay): `-d(gamma)/dt` per year. See `greeks::color`.
pub fn color(f: f64, k: f64, t: f64, r: f64, sigma: f64) -> f64 {
    greeks::color(f, k, t, r, r, sigma)
}

/// Veta (vega decay): `-d(vega)/dt` per year. Per unit vol. See `greeks::veta`.
pub fn veta(f: f64, k: f64, t: f64, r: f64, sigma: f64) -> f64 {
    greeks::veta(f, k, t, r, r, sigma)
}

/// Ultima: `d(vomma)/dsigma`. Per unit vol. See `greeks::ultima`.
pub fn ultima(f: f64, k: f64, t: f64, r: f64, sigma: f64) -> f64 {
    greeks::ultima(f, k, t, r, r, sigma)
}

/// All eight higher-order Black-76 Greeks in a single pass. See `greeks::higher_all`.
/// Order: `(vanna, vomma, charm, speed, zomma, color, veta, ultima)`.
pub fn higher_all(
    flag: Flag,
    f: f64,
    k: f64,
    t: f64,
    r: f64,
    sigma: f64,
) -> (f64, f64, f64, f64, f64, f64, f64, f64) {
    greeks::higher_all(flag, f, k, t, r, r, sigma)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// Convenience: per-1% vega from per-unit (multiply by 0.01).
    const PCT: f64 = 0.01;

    #[test]
    fn atm_call_py_vollib_doctest() {
        // py_vollib.black doctest: F=K=100, r=0.02, t=0.5, sigma=0.2 -> 5.5811067246048118
        let p = price(Flag::Call, 100.0, 100.0, 0.5, 0.02, 0.20);
        assert_relative_eq!(p, 5.581_106_724_604_812, epsilon = 1e-10);
    }

    #[test]
    fn delta_call_py_vollib_doctest() {
        // F=49, K=50, r=0.05, t=0.3846, sigma=0.20 -> 0.45107017482201828
        let d = delta(Flag::Call, 49.0, 50.0, 0.3846, 0.05, 0.20);
        assert_relative_eq!(d, 0.451_070_174_822_018_3, epsilon = 1e-10);
    }

    #[test]
    fn gamma_py_vollib_doctest() {
        // F=49, K=50, r=0.05, t=0.3846, sigma=0.20 -> 0.0640646705882
        let g = gamma(49.0, 50.0, 0.3846, 0.05, 0.20);
        assert_relative_eq!(g, 0.064_064_670_588_2, epsilon = 1e-10);
    }

    #[test]
    fn vega_py_vollib_doctest_per_unit() {
        // py_vollib reports per-1%: 0.118317785624. Per-unit = 100x that.
        let v = vega(49.0, 50.0, 0.3846, 0.05, 0.20);
        assert_relative_eq!(v, 0.118_317_785_624 / PCT, epsilon = 1e-8);
    }

    #[test]
    fn theta_call_py_vollib_doctest_per_year() {
        // py_vollib reports per-day: -0.00816236877462. Per-year = 365x.
        let th = theta(Flag::Call, 49.0, 50.0, 0.3846, 0.05, 0.20);
        assert_relative_eq!(th, -0.008_162_368_774_62 * 365.0, epsilon = 1e-7);
    }

    #[test]
    fn rho_call_py_vollib_doctest_per_unit() {
        // py_vollib reports per-1%: -0.0074705380059582258. Per-unit = 100x.
        let r_call = rho(Flag::Call, 49.0, 50.0, 0.3846, 0.05, 0.20);
        assert_relative_eq!(r_call, -0.007_470_538_005_958_226 / PCT, epsilon = 1e-8);
    }

    #[test]
    fn rho_put_py_vollib_doctest_per_unit() {
        // py_vollib reports per-1%: -0.011243286001308292. Per-unit = 100x.
        let r_put = rho(Flag::Put, 49.0, 50.0, 0.3846, 0.05, 0.20);
        assert_relative_eq!(r_put, -0.011_243_286_001_308_292 / PCT, epsilon = 1e-8);
    }

    #[test]
    fn put_call_parity_holds() {
        // C - P = exp(-r*T) * (F - K)
        let (f, k, t, r, sigma) = (100.0, 105.0, 0.5, 0.05, 0.25);
        let c = price(Flag::Call, f, k, t, r, sigma);
        let p = price(Flag::Put, f, k, t, r, sigma);
        let lhs = c - p;
        let rhs = (-r * t).exp() * (f - k);
        assert_relative_eq!(lhs, rhs, epsilon = 1e-12);
    }

    #[test]
    fn rho_equals_minus_t_times_price() {
        // The defining property of Black-76 rho.
        for &flag in &[Flag::Call, Flag::Put] {
            let p = price(flag, 100.0, 105.0, 0.5, 0.05, 0.25);
            let rh = rho(flag, 100.0, 105.0, 0.5, 0.05, 0.25);
            assert_relative_eq!(rh, -0.5 * p, epsilon = 1e-12);
        }
    }

    #[test]
    fn zero_time_returns_intrinsic_undiscounted() {
        // exp(0) = 1; result = max(F-K, 0) for call, max(K-F, 0) for put.
        assert_relative_eq!(price(Flag::Call, 110.0, 100.0, 0.0, 0.05, 0.2), 10.0);
        assert_relative_eq!(price(Flag::Put, 90.0, 100.0, 0.0, 0.05, 0.2), 10.0);
        assert_relative_eq!(price(Flag::Call, 90.0, 100.0, 0.0, 0.05, 0.2), 0.0);
    }

    #[test]
    fn zero_vol_returns_discounted_intrinsic() {
        // With sigma=0, F IS the price at expiry — payoff = max(F-K, 0).
        let p = price(Flag::Call, 110.0, 100.0, 1.0, 0.05, 0.0);
        let expected = (-0.05_f64).exp() * 10.0;
        assert_relative_eq!(p, expected, epsilon = 1e-12);
    }

    /// Verify analytical Greek matches a central finite-difference approximation.
    fn finite_diff<F: Fn(f64) -> f64>(f: F, x: f64, h: f64) -> f64 {
        (f(x + h) - f(x - h)) / (2.0 * h)
    }

    #[test]
    fn delta_matches_fd() {
        let (f, k, t, r, sigma) = (100.0, 105.0, 0.5, 0.05, 0.25);
        let analytical = delta(Flag::Call, f, k, t, r, sigma);
        let fd = finite_diff(|x| price(Flag::Call, x, k, t, r, sigma), f, 0.01);
        assert_relative_eq!(analytical, fd, epsilon = 1e-6);
    }

    #[test]
    fn vega_matches_fd() {
        let (f, k, t, r, sigma) = (100.0, 105.0, 0.5, 0.05, 0.25);
        let analytical = vega(f, k, t, r, sigma);
        let fd = finite_diff(|v| price(Flag::Call, f, k, t, r, v), sigma, 1e-4);
        assert_relative_eq!(analytical, fd, epsilon = 1e-6);
    }

    #[test]
    fn rho_matches_fd() {
        let (f, k, t, r, sigma) = (100.0, 105.0, 0.5, 0.05, 0.25);
        let analytical_call = rho(Flag::Call, f, k, t, r, sigma);
        let fd_call = finite_diff(|x| price(Flag::Call, f, k, t, x, sigma), r, 1e-5);
        assert_relative_eq!(analytical_call, fd_call, epsilon = 1e-6);

        let analytical_put = rho(Flag::Put, f, k, t, r, sigma);
        let fd_put = finite_diff(|x| price(Flag::Put, f, k, t, x, sigma), r, 1e-5);
        assert_relative_eq!(analytical_put, fd_put, epsilon = 1e-6);
    }

    #[test]
    fn theta_matches_fd_via_negative_t() {
        // theta = -dC/dT
        let (f, k, t, r, sigma) = (100.0, 105.0, 0.5, 0.05, 0.25);
        let analytical = theta(Flag::Call, f, k, t, r, sigma);
        let fd = finite_diff(|x| price(Flag::Call, f, k, x, r, sigma), t, 1e-5);
        // theta in pyvolr's convention is `common - r*K*disc_r*N(d2) + r*F*disc_r*N(d1)`
        // — this matches `-dP/dT` only after sign and `r-q` cancellation; see bsm::greeks
        // for the convention. Here, we verify against `-dP/dT` directly.
        assert_relative_eq!(analytical, -fd, epsilon = 1e-5);
    }

    /// Drift guard: `all()` must agree with the per-Greek functions across
    /// both flags and across both regular and degenerate input regimes.
    ///
    /// Includes a deep-OTM put cell where the `f*N(d1) - k*N(d2)` cancellation
    /// inside Black-76 rho is large enough that the choice between
    /// `disc * (f*nd1 - k*nd2)` and `f*disc*nd1 - k*disc*nd2` produces visibly
    /// different f64 values. `all` mirrors `bsm::price`'s second association
    /// so `rho_v = -t * price_v` stays bit-equal to the standalone path.
    #[test]
    fn all_matches_individual_at_grid() {
        let grid: &[(f64, f64, f64, f64, f64)] = &[
            (100.0, 100.0, 1.0, 0.05, 0.20),
            (49.0, 50.0, 0.3846, 0.05, 0.20),
            (100.0, 200.0, 0.5, 0.05, 0.30),
            (1000.0, 100.0, 0.01, 0.05, 0.20),
            // Deep-OTM corner reached by the fuzz harness: price ~ 1e-15,
            // catches the `f*nd1 - k*nd2` cancellation asymmetry on rho.
            (10.0, 100.0, 0.5, 0.05, 0.20),
        ];
        for &(f, k, t, r, sigma) in grid {
            for &flag in &[Flag::Call, Flag::Put] {
                let (d, g, v, th, rh) = all(flag, f, k, t, r, sigma);
                assert_relative_eq!(d, delta(flag, f, k, t, r, sigma), max_relative = 1e-15);
                assert_relative_eq!(g, gamma(f, k, t, r, sigma), max_relative = 1e-15);
                assert_relative_eq!(v, vega(f, k, t, r, sigma), max_relative = 1e-15);
                assert_relative_eq!(th, theta(flag, f, k, t, r, sigma), max_relative = 1e-15);
                assert_relative_eq!(rh, rho(flag, f, k, t, r, sigma), max_relative = 1e-15);
            }
        }
    }

    /// Deep-OTM put precision via `black76::all`. Mirrors the BSM regression
    /// test (`greeks::tests::put_delta_deep_otm_retains_precision`): when
    /// `d1` saturates so `cdf(d1) == 1.0` exactly, the old `nd1 - 1.0` form
    /// returned `0.0` instead of the correct tiny-negative put delta. The
    /// fix routes through `-cdf(-d1)` (erfcx tail).
    #[test]
    fn all_put_delta_deep_otm_retains_precision() {
        // F=1000, K=100 (10x OTM put), T=0.5y, σ=20% → d1 ≈ 16.5.
        let (d, _, _, _, _) = all(Flag::Put, 1000.0, 100.0, 0.5, 0.05, 0.20);
        assert!(d < 0.0, "put delta lost sign at deep OTM (returned {d:e})");
        assert!(
            d.abs() < 1e-50 && d.abs() > 0.0,
            "expected ~1e-61, got {d:e}"
        );
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn all_matches_individual_degenerate() {
        let degenerate: &[(f64, f64, f64, f64, f64)] = &[
            (110.0, 100.0, 0.0, 0.05, 0.2),
            (90.0, 100.0, 0.0, 0.05, 0.2),
            (100.0, 100.0, 1.0, 0.05, 0.0),
        ];
        for &(f, k, t, r, sigma) in degenerate {
            for &flag in &[Flag::Call, Flag::Put] {
                let (d, g, v, th, rh) = all(flag, f, k, t, r, sigma);
                assert_eq!(d, delta(flag, f, k, t, r, sigma));
                assert_eq!(g, gamma(f, k, t, r, sigma));
                assert_eq!(v, vega(f, k, t, r, sigma));
                assert_eq!(th, theta(flag, f, k, t, r, sigma));
                assert_eq!(rh, rho(flag, f, k, t, r, sigma));
            }
        }
    }

    /// Each higher-order Black-76 Greek must equal its `q = r` BSM counterpart
    /// (guards the delegation arg order), and `higher_all` must match the
    /// individual functions.
    #[test]
    #[allow(clippy::float_cmp)]
    fn higher_greeks_delegate_to_bsm_with_q_eq_r() {
        let grid: &[(f64, f64, f64, f64, f64)] = &[
            (100.0, 100.0, 1.0, 0.05, 0.20),
            (49.0, 50.0, 0.3846, 0.05, 0.20),
            (100.0, 120.0, 0.5, 0.03, 0.30),
        ];
        for &(f, k, t, r, sigma) in grid {
            assert_eq!(
                vanna(f, k, t, r, sigma),
                greeks::vanna(f, k, t, r, r, sigma)
            );
            assert_eq!(
                vomma(f, k, t, r, sigma),
                greeks::vomma(f, k, t, r, r, sigma)
            );
            assert_eq!(
                speed(f, k, t, r, sigma),
                greeks::speed(f, k, t, r, r, sigma)
            );
            assert_eq!(
                zomma(f, k, t, r, sigma),
                greeks::zomma(f, k, t, r, r, sigma)
            );
            assert_eq!(
                color(f, k, t, r, sigma),
                greeks::color(f, k, t, r, r, sigma)
            );
            assert_eq!(veta(f, k, t, r, sigma), greeks::veta(f, k, t, r, r, sigma));
            assert_eq!(
                ultima(f, k, t, r, sigma),
                greeks::ultima(f, k, t, r, r, sigma)
            );
            for &flag in &[Flag::Call, Flag::Put] {
                assert_eq!(
                    charm(flag, f, k, t, r, sigma),
                    greeks::charm(flag, f, k, t, r, r, sigma)
                );
                assert_eq!(
                    higher_all(flag, f, k, t, r, sigma),
                    greeks::higher_all(flag, f, k, t, r, r, sigma)
                );
            }
        }
    }
}
