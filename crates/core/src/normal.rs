//! Standard normal distribution helpers.
//!
//! `cdf` uses `libm::erf` for high accuracy across the full range.

/// `sqrt(2)`.
const SQRT_2: f64 = std::f64::consts::SQRT_2;
/// `1 / sqrt(2 * pi)`.
const INV_SQRT_2PI: f64 = 0.398_942_280_401_432_7;

/// Standard normal CDF: `P(Z <= x)` for `Z ~ N(0, 1)`.
#[inline]
pub fn cdf(x: f64) -> f64 {
    0.5 * (1.0 + libm::erf(x / SQRT_2))
}

/// Standard normal PDF: `(2 * pi)^(-1/2) * exp(-x^2 / 2)`.
#[inline]
pub fn pdf(x: f64) -> f64 {
    INV_SQRT_2PI * (-0.5 * x * x).exp()
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

    #[test]
    fn pdf_known_values() {
        assert_relative_eq!(pdf(0.0), INV_SQRT_2PI, epsilon = 1e-15);
        assert_relative_eq!(pdf(1.0), 0.241_970_724_519_143_4, epsilon = 1e-12);
    }
}
