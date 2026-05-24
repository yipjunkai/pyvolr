//! Implied volatility solver.
//!
//! Uses Newton-Raphson with the Manaster-Koehler initial guess, falling back
//! to bisection when Newton stalls (tiny vega, OTM tails) or steps out of bounds.
//!
//! Reference: Manaster, S., & Koehler, G. (1982). The calculation of implied
//! variances from the Black-Scholes model: a note. *Journal of Finance*, 37(1).

use crate::bsm::{price, Flag};
use crate::greeks::vega;

const MAX_NEWTON_ITER: u32 = 50;
const MAX_BISECT_ITER: u32 = 100;
const PRICE_TOL: f64 = 1e-10;
const SIGMA_MIN: f64 = 1e-9;
const SIGMA_MAX: f64 = 5.0;
const MIN_VEGA: f64 = 1e-12;

/// Solve for implied volatility given a market price.
///
/// Returns `NaN` if:
///   - `t <= 0`,
///   - the target price is not finite or is negative,
///   - the target price violates the no-arbitrage bounds,
///   - the solver cannot bracket a root within `[SIGMA_MIN, SIGMA_MAX]`.
pub fn solve(target_price: f64, flag: Flag, s: f64, k: f64, t: f64, r: f64, q: f64) -> f64 {
    if t <= 0.0 || !target_price.is_finite() || target_price < 0.0 {
        return f64::NAN;
    }

    let disc_q = (-q * t).exp();
    let disc_r = (-r * t).exp();
    let (lower, upper) = match flag {
        Flag::Call => ((s * disc_q - k * disc_r).max(0.0), s * disc_q),
        Flag::Put => ((k * disc_r - s * disc_q).max(0.0), k * disc_r),
    };
    if target_price < lower - 1e-12 || target_price > upper + 1e-12 {
        return f64::NAN;
    }

    // Manaster-Koehler initial guess, clamped to a sensible band.
    let mk_seed = (2.0 * (s / k).ln().abs() / t).sqrt();
    let mut sigma = if mk_seed.is_finite() {
        mk_seed.clamp(0.05, 2.0)
    } else {
        0.2
    };

    // Newton-Raphson.
    let mut newton_ok = false;
    for _ in 0..MAX_NEWTON_ITER {
        let p = price(flag, s, k, t, r, q, sigma);
        let diff = p - target_price;
        if diff.abs() < PRICE_TOL {
            newton_ok = true;
            break;
        }
        let v = vega(s, k, t, r, q, sigma);
        if v < MIN_VEGA {
            break;
        }
        let new_sigma = sigma - diff / v;
        if !new_sigma.is_finite() || new_sigma <= SIGMA_MIN || new_sigma >= SIGMA_MAX {
            break;
        }
        sigma = new_sigma;
    }
    if newton_ok {
        return sigma;
    }

    // Bisection fallback.
    let mut lo = SIGMA_MIN;
    let mut hi = SIGMA_MAX;
    let target = target_price;
    let f = |x: f64| price(flag, s, k, t, r, q, x) - target;
    let mut f_lo = f(lo);
    let f_hi = f(hi);
    if f_lo.signum() == f_hi.signum() {
        // Root not bracketed in the search interval.
        return f64::NAN;
    }
    for _ in 0..MAX_BISECT_ITER {
        let mid = 0.5 * (lo + hi);
        let f_mid = f(mid);
        if f_mid.abs() < PRICE_TOL || (hi - lo) < 1e-12 {
            return mid;
        }
        if f_mid.signum() == f_lo.signum() {
            lo = mid;
            f_lo = f_mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn roundtrip_call_atm() {
        let (s, k, t, r, q, sigma) = (100.0, 100.0, 1.0, 0.05, 0.0, 0.20);
        let p = price(Flag::Call, s, k, t, r, q, sigma);
        let iv = solve(p, Flag::Call, s, k, t, r, q);
        assert_relative_eq!(iv, sigma, epsilon = 1e-8);
    }

    #[test]
    fn roundtrip_put_otm_short_expiry() {
        let (s, k, t, r, q, sigma) = (100.0, 80.0, 0.05, 0.03, 0.01, 0.45);
        let p = price(Flag::Put, s, k, t, r, q, sigma);
        let iv = solve(p, Flag::Put, s, k, t, r, q);
        assert_relative_eq!(iv, sigma, epsilon = 1e-6);
    }

    #[test]
    fn roundtrip_call_deep_otm() {
        let (s, k, t, r, q, sigma) = (100.0, 200.0, 0.5, 0.05, 0.0, 0.30);
        let p = price(Flag::Call, s, k, t, r, q, sigma);
        let iv = solve(p, Flag::Call, s, k, t, r, q);
        assert_relative_eq!(iv, sigma, epsilon = 1e-6);
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
