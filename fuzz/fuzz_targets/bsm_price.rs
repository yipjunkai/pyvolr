//! Fuzz target for `pyvolr_core::bsm::price`.
//!
//! Goal: prove no input — including pathological ones (NaN, infinity,
//! negative time, zero volatility) — can panic the pricer or produce
//! values outside the no-arbitrage bounds for sensible inputs.

#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use pyvolr_core::bsm::{Flag, price};

#[derive(Arbitrary, Debug)]
struct Input {
    flag_is_call: bool,
    s: f64,
    k: f64,
    t: f64,
    r: f64,
    q: f64,
    sigma: f64,
}

fuzz_target!(|inp: Input| {
    let flag = if inp.flag_is_call { Flag::Call } else { Flag::Put };
    let p = price(flag, inp.s, inp.k, inp.t, inp.r, inp.q, inp.sigma);

    // Invariants for well-conditioned inputs.
    //
    // The numeric bounds keep the closed-form pricing math inside the
    // regime where f64 arithmetic faithfully reproduces the analytical
    // limits, so the strict invariants below are actually meaningful:
    //
    //   - `|r*t|`, `|q*t|` < 700: keeps `exp(-r*t)`, `exp(-q*t)` finite
    //     (exp(709.7) is the f64 overflow threshold).
    //   - `sigma * sqrt(t)` < 100: keeps `sigma^2 * t` from overflowing to
    //     +inf in the d1/d2 numerator. When it overflows, BOTH d1 and d2
    //     become +inf (instead of d1 → +inf, d2 → -inf as the true limit
    //     requires), and the formula degenerates to `s - k * disc_r` —
    //     which is wildly negative for k >> s, even though the limiting
    //     call price should approach `s * disc_q`.
    //
    // Outside this band the only invariant we still require is "the pricer
    // didn't panic", which is exercised implicitly by reaching this point.
    let well_conditioned = inp.s.is_finite()
        && inp.k.is_finite()
        && inp.t.is_finite()
        && inp.r.is_finite()
        && inp.q.is_finite()
        && inp.sigma.is_finite()
        && inp.s >= 0.0
        && inp.k >= 0.0
        && inp.t >= 0.0
        && inp.sigma >= 0.0
        && (inp.r * inp.t).abs() < 700.0
        && (inp.q * inp.t).abs() < 700.0
        && inp.sigma * inp.t.sqrt() < 100.0
        // Absolute bounds on individual parameters: the constraints above
        // bound *products* (r*t, sigma*sqrt(t)), but the BSM formula also
        // computes intermediates like `0.5 * sigma * sigma` (overflows to
        // +inf for sigma > ~1.34e154) and `r - q` (overflows when both are
        // near f64::MAX with opposite signs). Once any of these go to ±inf,
        // both d1 and d2 collapse to +inf instead of taking opposite signs,
        // and the formula degenerates to `s - k * disc_r` — wildly negative
        // for k >> s. 1e150 keeps every intermediate finite while still
        // leaving the fuzzer 150 orders of magnitude past anything real.
        && inp.r.abs() < 1e150
        && inp.q.abs() < 1e150
        && inp.sigma < 1e150;

    if well_conditioned {
        assert!(p.is_finite() || p.is_nan(), "non-finite, non-nan: p={p}");
        assert!(p >= -1e-12 || p.is_nan(), "negative price for well-conditioned inputs: p={p}");

        // Put-call parity (only meaningful if both legs computable and finite).
        let other = match flag {
            Flag::Call => Flag::Put,
            Flag::Put => Flag::Call,
        };
        let p_other = price(other, inp.s, inp.k, inp.t, inp.r, inp.q, inp.sigma);
        if p.is_finite() && p_other.is_finite() && inp.t > 0.0 {
            let (c, put) = match flag {
                Flag::Call => (p, p_other),
                Flag::Put => (p_other, p),
            };
            let parity_lhs = c - put;
            let parity_rhs = inp.s * (-inp.q * inp.t).exp() - inp.k * (-inp.r * inp.t).exp();
            let scale = (inp.s.abs() + inp.k.abs()).max(1.0);
            let err = (parity_lhs - parity_rhs).abs();
            assert!(
                err < 1e-6 * scale || err.is_nan(),
                "put-call parity violated: lhs={parity_lhs} rhs={parity_rhs} err={err}"
            );
        }
    }
});
