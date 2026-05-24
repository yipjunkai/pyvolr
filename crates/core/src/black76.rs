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

use crate::bsm::{price as bsm_price, Flag};
use crate::greeks;

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

/// Derivative of price with respect to time-to-expiry (per year, annualized).
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
}
