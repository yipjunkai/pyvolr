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

pub fn delta(flag: Flag, s: f64, k: f64, t: f64, r: f64, q: f64, sigma: f64) -> f64 {
    if t <= 0.0 || sigma <= 0.0 {
        let in_the_money = match flag {
            Flag::Call => s > k,
            Flag::Put => s < k,
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
        Flag::Put => disc_q * (cdf(d1) - 1.0),
    }
}

pub fn gamma(s: f64, k: f64, t: f64, r: f64, q: f64, sigma: f64) -> f64 {
    if t <= 0.0 || sigma <= 0.0 {
        return 0.0;
    }
    let (d1, _) = d1_d2(s, k, t, r, q, sigma);
    let disc_q = (-q * t).exp();
    disc_q * pdf(d1) / (s * sigma * t.sqrt())
}

pub fn vega(s: f64, k: f64, t: f64, r: f64, q: f64, sigma: f64) -> f64 {
    if t <= 0.0 || sigma <= 0.0 {
        return 0.0;
    }
    let (d1, _) = d1_d2(s, k, t, r, q, sigma);
    let disc_q = (-q * t).exp();
    s * disc_q * pdf(d1) * t.sqrt()
}

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
        let fd = finite_diff(
            |x| {
                (price(Flag::Call, x + 0.01, k, t, r, q, sigma)
                    - 2.0 * price(Flag::Call, x, k, t, r, q, sigma)
                    + price(Flag::Call, x - 0.01, k, t, r, q, sigma))
                    / (0.01 * 0.01)
            },
            s,
            0.0,
        );
        // Use the inner expression rather than passing fd's wrapper.
        let direct = (price(Flag::Call, s + 0.01, k, t, r, q, sigma)
            - 2.0 * price(Flag::Call, s, k, t, r, q, sigma)
            + price(Flag::Call, s - 0.01, k, t, r, q, sigma))
            / (0.01 * 0.01);
        let _ = fd;
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
}
