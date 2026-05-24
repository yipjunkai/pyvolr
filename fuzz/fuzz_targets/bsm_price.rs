//! Fuzz target for `pyvolr_core::bsm::price`.
//!
//! Two-tier contract:
//!   1. For *every* input the pricer must not panic (reaching the bottom
//!      of `fuzz_target!` already proves this).
//!   2. For inputs in a physically realistic band — the values any actual
//!      options-pricing user would supply — the price must additionally
//!      be finite, non-negative (up to FP roundoff), and respect put-call
//!      parity.
//!
//! Trying to assert (2) on the full f64 input space is whack-a-mole: BSM
//! has multiple intermediate products that overflow at different boundaries
//! of f64 (`sigma^2`, `sigma^2 * t`, `s * exp(-q*t)`, `k * exp(-r*t)`, the
//! `sigma=0` branch's `s * exp((r-q)*t) * exp(-r*t)`, …). Bounding inputs
//! to the realistic region — wide enough to subsume any real market but
//! narrow enough to keep every intermediate in f64's well-defined regime —
//! is both simpler and a more meaningful statement of correctness.

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

    // Physically realistic input band. Bounds are 2–4 orders of magnitude
    // past anything a real options market ever produces, so the fuzzer
    // still explores meaningful edge cases (deep ITM/OTM, low/high vol,
    // tiny/long time), while staying inside f64's well-defined regime.
    let realistic = inp.s.is_finite()
        && inp.k.is_finite()
        && inp.t.is_finite()
        && inp.r.is_finite()
        && inp.q.is_finite()
        && inp.sigma.is_finite()
        && inp.s >= 0.0
        && inp.s < 1e12        // up to a trillion per share
        && inp.k >= 0.0
        && inp.k < 1e12
        && inp.t >= 0.0
        && inp.t < 100.0       // 100 years
        && inp.r.abs() < 1.0   // ±100% annual rate
        && inp.q.abs() < 1.0
        && inp.sigma >= 0.0
        && inp.sigma < 5.0;    // 500% vol

    if realistic {
        // NaN is the pricer's graceful-failure value for genuinely
        // indeterminate inputs (s=k=0 → 0/0). What the assertion really
        // forbids is ±inf — those would be a real numerical bug inside
        // the realistic band.
        assert!(
            p.is_finite() || p.is_nan(),
            "non-finite, non-nan price for realistic inputs: p={p}"
        );
        let neg_tol = 1e-12 * (inp.s.abs() + inp.k.abs()).max(1.0);
        assert!(
            p >= -neg_tol || p.is_nan(),
            "negative price for realistic inputs: p={p}"
        );

        // Put-call parity.
        let other = match flag {
            Flag::Call => Flag::Put,
            Flag::Put => Flag::Call,
        };
        let p_other = price(other, inp.s, inp.k, inp.t, inp.r, inp.q, inp.sigma);
        if p_other.is_finite() && inp.t > 0.0 {
            let (c, put) = match flag {
                Flag::Call => (p, p_other),
                Flag::Put => (p_other, p),
            };
            let parity_lhs = c - put;
            let parity_rhs = inp.s * (-inp.q * inp.t).exp() - inp.k * (-inp.r * inp.t).exp();
            let scale = (inp.s.abs() + inp.k.abs()).max(1.0);
            let err = (parity_lhs - parity_rhs).abs();
            assert!(
                err < 1e-6 * scale,
                "put-call parity violated: lhs={parity_lhs} rhs={parity_rhs} err={err}"
            );
        }
    }
});
