//! Fuzz target for `pyvolr_core::black76::all`.
//!
//! Mirror of `greeks_all.rs` for Black-76 (futures-option pricer). Differences:
//!   - No `q` parameter (the forward `f` already absorbs dividends/funding).
//!   - Put-call delta parity becomes `dC/dF - dP/dF = exp(-r*T)`, since
//!     `C - P = exp(-r*T) * (F - K)` for Black-76.

#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use pyvolr_core::black76::{all, delta, gamma, rho, theta, vega};
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

    // Tier 1: reaching here proves `all` did not panic.
    let (d, g, v, th, rh) = all(flag, inp.f, inp.k, inp.t, inp.r, inp.sigma);

    let realistic = inp.f.is_finite()
        && inp.k.is_finite()
        && inp.t.is_finite()
        && inp.r.is_finite()
        && inp.sigma.is_finite()
        && inp.f > 1e-6
        && inp.f < 1e9
        && inp.k > 1e-6
        && inp.k < 1e9
        && inp.t > 1e-6
        && inp.t < 100.0
        && inp.r.abs() < 1.0
        && inp.sigma > 1e-4
        && inp.sigma < 5.0;

    if !realistic {
        return;
    }

    // Bit-equality vs per-Greek functions. See `greeks_all.rs` for why
    // exact `==` is the right assertion here.
    let d_ind = delta(flag, inp.f, inp.k, inp.t, inp.r, inp.sigma);
    let g_ind = gamma(inp.f, inp.k, inp.t, inp.r, inp.sigma);
    let v_ind = vega(inp.f, inp.k, inp.t, inp.r, inp.sigma);
    let th_ind = theta(flag, inp.f, inp.k, inp.t, inp.r, inp.sigma);
    let rh_ind = rho(flag, inp.f, inp.k, inp.t, inp.r, inp.sigma);

    assert!(
        d == d_ind || (d.is_nan() && d_ind.is_nan()),
        "delta drift: all={d} ind={d_ind}"
    );
    assert!(
        g == g_ind || (g.is_nan() && g_ind.is_nan()),
        "gamma drift: all={g} ind={g_ind}"
    );
    assert!(
        v == v_ind || (v.is_nan() && v_ind.is_nan()),
        "vega drift: all={v} ind={v_ind}"
    );
    assert!(
        th == th_ind || (th.is_nan() && th_ind.is_nan()),
        "theta drift: all={th} ind={th_ind}"
    );
    assert!(
        rh == rh_ind || (rh.is_nan() && rh_ind.is_nan()),
        "rho drift: all={rh} ind={rh_ind}"
    );

    assert!(g.is_nan() || g >= 0.0, "negative gamma: {g}");
    assert!(v.is_nan() || v >= 0.0, "negative vega: {v}");

    // Put-call delta parity for Black-76: `dC/dF - dP/dF = exp(-r*T)`.
    let other = match flag {
        Flag::Call => Flag::Put,
        Flag::Put => Flag::Call,
    };
    let (d_other, _, _, _, _) = all(other, inp.f, inp.k, inp.t, inp.r, inp.sigma);
    if d_other.is_finite() && d.is_finite() {
        let (cd, pd) = match flag {
            Flag::Call => (d, d_other),
            Flag::Put => (d_other, d),
        };
        let lhs = cd - pd;
        let rhs = (-inp.r * inp.t).exp();
        let err = (lhs - rhs).abs();
        // `cd`/`pd` are separately-rounded products `disc·Φ(±d1)`, so `cd − pd`
        // recovers `disc·(Φ(d1)+Φ(−d1)) = disc` only to ~1 ULP *relative*, and
        // `disc = exp(−rT)` is unbounded above for r < 0 — an absolute bound
        // spuriously fails once `disc·ε > 1e-12`. Scale by the magnitude; 1e-12
        // relative still catches a gross break like the old deep-OTM
        // `cdf(d1) − 1.0` put-delta form (which broke parity by ~disc).
        assert!(
            err <= 1e-12 * rhs.max(1.0),
            "black76 put-call delta parity violated: cd-pd={lhs} exp(-rT)={rhs} err={err}"
        );
    }
});
