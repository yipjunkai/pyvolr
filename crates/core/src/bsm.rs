//! Black-Scholes-Merton pricing of European options with continuous dividend yield.
//!
//! References:
//!   - Hull, J. C. (2017). *Options, Futures, and Other Derivatives* (10th ed.), Ch. 17.
//!   - Merton, R. C. (1973). Theory of rational option pricing.

use crate::normal::cdf;

/// Option type. Maps to/from `i8` (`>=0` -> Call, `<0` -> Put) for FFI.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Flag {
    Call,
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
    if sigma <= 0.0 {
        let forward = s * ((r - q) * t).exp();
        return (-r * t).exp() * intrinsic(flag, forward, k);
    }
    let (d1, d2) = d1_d2(s, k, t, r, q, sigma);
    let disc_q = (-q * t).exp();
    let disc_r = (-r * t).exp();
    match flag {
        Flag::Call => s * disc_q * cdf(d1) - k * disc_r * cdf(d2),
        Flag::Put => k * disc_r * cdf(-d2) - s * disc_q * cdf(-d1),
    }
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
