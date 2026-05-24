//! Fuzz target for `pyvolr_core::iv::solve`.
//!
//! IV inversion is the trickiest part of the math (Newton-Raphson + bisection
//! fallback). Goals:
//!   1. No input can panic, hang, or escape with a non-finite recovered sigma.
//!   2. When the solver claims convergence, the roundtrip price matches.

#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use pyvolr_core::{
    bsm::{Flag, price},
    iv::solve,
};

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
    // The IV solver is only meaningfully defined for the well-conditioned
    // input region. Anything else: just check we don't panic.
    if !(inp.s.is_finite()
        && inp.k.is_finite()
        && inp.t.is_finite()
        && inp.r.is_finite()
        && inp.q.is_finite()
        && inp.sigma.is_finite()
        && inp.s > 0.0
        && inp.k > 0.0
        && inp.t > 1e-6
        && inp.t < 100.0
        && inp.sigma > 1e-4
        && inp.sigma < 5.0
        && inp.r.abs() < 1.0
        && inp.q.abs() < 1.0)
    {
        // Still run the solver to verify no panic on weird inputs.
        let flag = if inp.flag_is_call { Flag::Call } else { Flag::Put };
        let _ = solve(0.0, flag, inp.s, inp.k, inp.t, inp.r, inp.q);
        return;
    }

    let flag = if inp.flag_is_call { Flag::Call } else { Flag::Put };

    // Forward: price the option at the chosen sigma.
    let p = price(flag, inp.s, inp.k, inp.t, inp.r, inp.q, inp.sigma);
    if !p.is_finite() || p < 0.0 {
        return;
    }

    // Inverse: recover sigma from price.
    let recovered = solve(p, flag, inp.s, inp.k, inp.t, inp.r, inp.q);

    // The solver returning NaN means "couldn't find a root" — acceptable.
    if !recovered.is_finite() {
        return;
    }

    // If the solver claims convergence, the recovered sigma must reproduce
    // the original price. (Sigma may differ from the input sigma because the
    // IV-from-price inversion is ill-conditioned at deep ITM/OTM tails;
    // checking price reproduction is the well-posed correctness statement.)
    let p_back = price(flag, inp.s, inp.k, inp.t, inp.r, inp.q, recovered);
    assert!(p_back.is_finite(), "non-finite roundtrip price for converged solve");

    let scale = p.abs().max(1e-6);
    let err = (p_back - p).abs();
    assert!(
        err < 1e-6 * scale,
        "price roundtrip drift: p={p} recovered_sigma={recovered} p_back={p_back} err={err}"
    );
});
