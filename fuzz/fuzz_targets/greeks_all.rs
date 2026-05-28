//! Fuzz target for `pyvolr_core::greeks::all`.
//!
//! `greeks::all` shares `d1_d2`, `disc_q`, `disc_r`, `cdf`, and `pdf`
//! evaluations across the five Greeks. The per-Greek functions are already
//! fuzzed indirectly via `bsm_price` (they share `d1_d2` / `cdf` / `pdf` /
//! `exp`), but `all`'s combined arithmetic ordering is a distinct code path.
//!
//! Three contracts:
//!   1. Never panic on any input.
//!   2. In a realistic band the bundled tuple must:
//!      - Be bit-equal to the per-Greek functions (the `tests::all_matches_*`
//!        grid asserts this at fixed points; the fuzzer extends it across
//!        the input space).
//!      - Satisfy gamma >= 0 and vega >= 0 (mathematical guarantees).
//!      - Respect put-call delta parity: `delta(C) - delta(P) = exp(-q*T)`.
//!        Already pinned at one deep-OTM grid point by
//!        `put_call_delta_parity_holds_at_deep_otm`; the fuzzer extends the
//!        guard across the input space, where the deep-OTM `erfcx`-tail
//!        formula on the put arm is the only path that keeps the identity
//!        f64-exact (the old `cdf(d1) - 1.0` form failed it silently).

#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use pyvolr_core::bsm::Flag;
use pyvolr_core::greeks::{all, delta, gamma, rho, theta, vega};

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

    // Tier 1: reaching here proves `all` did not panic.
    let (d, g, v, th, rh) = all(flag, inp.s, inp.k, inp.t, inp.r, inp.q, inp.sigma);

    // Same realistic band as `iv_solve.rs` — keeps every intermediate
    // (`sigma^2 * t`, `s * exp(-q*t)`, `k * exp(-r*t)`, ...) inside f64's
    // well-defined regime. Excludes `t = 0` / `sigma = 0` because the
    // degenerate branch is trivially correct: `all` delegates to `delta`
    // and zeros the other four.
    let realistic = inp.s.is_finite()
        && inp.k.is_finite()
        && inp.t.is_finite()
        && inp.r.is_finite()
        && inp.q.is_finite()
        && inp.sigma.is_finite()
        && inp.s > 1e-6
        && inp.s < 1e9
        && inp.k > 1e-6
        && inp.k < 1e9
        && inp.t > 1e-6
        && inp.t < 100.0
        && inp.r.abs() < 1.0
        && inp.q.abs() < 1.0
        && inp.sigma > 1e-4
        && inp.sigma < 5.0;

    if !realistic {
        return;
    }

    // Bit-equality vs per-Greek functions. Both paths apply the same
    // arithmetic in the same order to the same inputs (verified by
    // `tests::all_matches_individual_at_grid` at `max_relative = 1e-15`),
    // so an exact `==` is the right assertion here. Drift would mean the
    // shared-subexpression refactor silently changed a value somewhere.
    let d_ind = delta(flag, inp.s, inp.k, inp.t, inp.r, inp.q, inp.sigma);
    let g_ind = gamma(inp.s, inp.k, inp.t, inp.r, inp.q, inp.sigma);
    let v_ind = vega(inp.s, inp.k, inp.t, inp.r, inp.q, inp.sigma);
    let th_ind = theta(flag, inp.s, inp.k, inp.t, inp.r, inp.q, inp.sigma);
    let rh_ind = rho(flag, inp.s, inp.k, inp.t, inp.r, inp.q, inp.sigma);

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

    // Sign correctness — gamma and vega are non-negative for any valid BSM
    // input (gamma = disc_q * pdf(d1) / (s*sigma*sqrt(t)); vega same shape).
    assert!(g.is_nan() || g >= 0.0, "negative gamma: {g}");
    assert!(v.is_nan() || v >= 0.0, "negative vega: {v}");

    // Put-call delta parity: `dC/dS - dP/dS = exp(-q*T)` (since
    // `C - P = S*exp(-q*T) - K*exp(-r*T)`). At deep OTM (d1 large positive)
    // the old `cdf(d1) - 1.0` put-delta form returned 0 and broke this
    // identity by a full `exp(-q*T)`. The `-cdf(-d1)` form keeps it
    // f64-exact via the `erfcx` tail.
    let other = match flag {
        Flag::Call => Flag::Put,
        Flag::Put => Flag::Call,
    };
    let (d_other, _, _, _, _) = all(other, inp.s, inp.k, inp.t, inp.r, inp.q, inp.sigma);
    if d_other.is_finite() && d.is_finite() {
        let (cd, pd) = match flag {
            Flag::Call => (d, d_other),
            Flag::Put => (d_other, d),
        };
        let lhs = cd - pd;
        let rhs = (-inp.q * inp.t).exp();
        let err = (lhs - rhs).abs();
        // 1e-12 absolute is comfortably above f64 roundoff for `exp(-q*t)`
        // at any realistic `q*t` (worst case `q*t = 99`, `exp` is computed
        // to ~1 ULP).
        assert!(
            err < 1e-12,
            "put-call delta parity violated: cd-pd={lhs} exp(-qT)={rhs} err={err}"
        );
    }
});
