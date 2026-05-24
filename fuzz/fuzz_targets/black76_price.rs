//! Fuzz target for `pyvolr_core::black76::price`.
//!
//! Two-tier contract identical to `bsm_price.rs`:
//!   1. For *every* input the pricer must not panic.
//!   2. For inputs in a physically realistic band the result must
//!      additionally be finite-or-NaN, sign-correct, and respect
//!      Black-76 put-call parity: `C - P = exp(-r*T) * (F - K)`.

#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use pyvolr_core::black76::price;
use pyvolr_core::bsm::Flag;

#[derive(Arbitrary, Debug)]
struct Input {
    flag_is_call: bool,
    f: f64,
    k: f64,
    t: f64,
    r: f64,
    sigma: f64,
}

fuzz_target!(|inp: Input| {
    let flag = if inp.flag_is_call { Flag::Call } else { Flag::Put };
    let p = price(flag, inp.f, inp.k, inp.t, inp.r, inp.sigma);

    // Realistic band: same scale as bsm_price's. Black-76 has no q parameter.
    let realistic = inp.f.is_finite()
        && inp.k.is_finite()
        && inp.t.is_finite()
        && inp.r.is_finite()
        && inp.sigma.is_finite()
        && inp.f >= 0.0
        && inp.f < 1e12
        && inp.k >= 0.0
        && inp.k < 1e12
        && inp.t >= 0.0
        && inp.t < 100.0
        && inp.r.abs() < 1.0
        && inp.sigma >= 0.0
        && inp.sigma < 5.0;

    if realistic {
        assert!(
            p.is_finite() || p.is_nan(),
            "non-finite, non-nan price for realistic inputs: p={p}"
        );
        let neg_tol = 1e-12 * (inp.f.abs() + inp.k.abs()).max(1.0);
        assert!(
            p >= -neg_tol || p.is_nan(),
            "negative price for realistic inputs: p={p}"
        );

        // Put-call parity: C - P = exp(-r*T) * (F - K).
        let other = match flag {
            Flag::Call => Flag::Put,
            Flag::Put => Flag::Call,
        };
        let p_other = price(other, inp.f, inp.k, inp.t, inp.r, inp.sigma);
        if p_other.is_finite() && inp.t > 0.0 {
            let (c, put) = match flag {
                Flag::Call => (p, p_other),
                Flag::Put => (p_other, p),
            };
            let parity_lhs = c - put;
            let parity_rhs = (-inp.r * inp.t).exp() * (inp.f - inp.k);
            let scale = (inp.f.abs() + inp.k.abs()).max(1.0);
            let err = (parity_lhs - parity_rhs).abs();
            assert!(
                err < 1e-6 * scale,
                "black76 put-call parity violated: lhs={parity_lhs} rhs={parity_rhs} err={err}"
            );
        }
    }
});
