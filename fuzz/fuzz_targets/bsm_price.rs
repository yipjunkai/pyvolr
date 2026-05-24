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
    // We check the three places BSM intermediates can overflow:
    //   - `sigma * sigma` overflowing alone (sigma > ~1.34e154)
    //   - `sigma * sigma * t` overflowing via the d1/d2 numerator
    //   - `s * exp(-q*t)` or `k * exp(-r*t)` overflowing in the price line
    //
    // The first two are bounded by capping `sigma` and `sigma * sqrt(t)`.
    // The last is jointly determined by `s`, `k`, `r*t`, `q*t` (can't be
    // bounded with single-variable constraints), so we just compute the
    // products and require them to be finite. Outside this set the only
    // invariant we still require is "the pricer didn't panic".
    let s_disc_q = inp.s * (-inp.q * inp.t).exp();
    let k_disc_r = inp.k * (-inp.r * inp.t).exp();

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
        && inp.sigma < 1e150
        && inp.sigma * inp.t.sqrt() < 100.0
        && inp.r.abs() < 1e150
        && inp.q.abs() < 1e150
        && s_disc_q.is_finite()
        && k_disc_r.is_finite();

    if well_conditioned {
        assert!(p.is_finite() || p.is_nan(), "non-finite, non-nan: p={p}");
        // Tolerance is relative to the price scale: for huge s, k (e.g.
        // 1e175), even f64's ~15-digit precision allows absolute errors of
        // 1e160. The negative price we're guarding against is real arbitrage
        // (price << -ULP * scale), not FP roundoff at the working scale.
        let neg_tol = 1e-12 * (inp.s.abs() + inp.k.abs()).max(1.0);
        assert!(
            p >= -neg_tol || p.is_nan(),
            "negative price for well-conditioned inputs: p={p} tol={neg_tol}"
        );

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
