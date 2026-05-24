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
    // The bounds on `r * t` and `q * t` keep the discount factors
    // `exp(-r*t)` and `exp(-q*t)` finite — `exp(700) ~ 1e304` is the f64
    // overflow threshold. Without these, absurd-but-finite inputs (e.g.
    // r=-3e304 with tiny t) push the discount factor to +inf and the price
    // to +inf, which is a legitimate result the assertion below would
    // otherwise reject. Outside this band the only invariant we still
    // require is "the pricer didn't panic", which is exercised implicitly
    // by reaching this point.
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
        && (inp.q * inp.t).abs() < 700.0;

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
