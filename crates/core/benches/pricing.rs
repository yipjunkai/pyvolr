// criterion_group! / criterion_main! generate undocumented items; the file
// itself is documented above.
#![allow(missing_docs)]

//! Performance benchmarks for the pyvolr Rust core.
//!
//! Drives the CI perf-gate. Each `bench_function` here is a contract: a
//! regression beyond the configured threshold in `.github/workflows/perf.yml`
//! fails the PR. Keep the input shapes representative of the README's
//! published numbers (scalar + 10k-strike vectors) so wins reported there
//! stay backed by something CI verifies.

use std::hint::black_box;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use pyvolr_core::normal::erfcx;
use pyvolr_core::{black76, bsm, greeks, iv};

const VEC_LEN: usize = 10_000;
const INV_SQRT_2: f64 = std::f64::consts::FRAC_1_SQRT_2;

/// Experimental branchless `cdf`: always uses the `exp + erfcx` formulation,
/// avoiding the `|x| < 4` data-dependent branch that `normal::cdf` takes.
///
/// The single `if x >= 0.0` left over compiles to a `csel` on aarch64 (a
/// conditional move with no branch). The cost is ~50-70% more compute per
/// call (one `erfcx` + one `exp` vs one `erf`), so branchless can only win
/// if removing the `|x| < 4` branch produces an auto-vectorisation gain
/// big enough to pay for the extra transcendentals. See
/// `bench_cdf_branch_experiment` below for the measurement.
fn cdf_branchless(x: f64) -> f64 {
    let p = 0.5 * (-0.5 * x * x).exp() * erfcx(x.abs() * INV_SQRT_2);
    if x >= 0.0 {
        1.0 - p
    } else {
        p
    }
}

/// Copy of `bsm::price`'s body with `cdf` replaced by `cdf_branchless`.
/// Used only by `bench_cdf_branch_experiment`.
fn bsm_price_branchless(
    flag: bsm::Flag,
    s: f64,
    k: f64,
    t: f64,
    r: f64,
    q: f64,
    sigma: f64,
) -> f64 {
    if t <= 0.0 {
        return match flag {
            bsm::Flag::Call => (s - k).max(0.0),
            bsm::Flag::Put => (k - s).max(0.0),
        };
    }
    if sigma <= 0.0 {
        let forward = s * ((r - q) * t).exp();
        let disc_r = (-r * t).exp();
        return disc_r
            * match flag {
                bsm::Flag::Call => (forward - k).max(0.0),
                bsm::Flag::Put => (k - forward).max(0.0),
            };
    }
    let (d1, d2) = bsm::d1_d2(s, k, t, r, q, sigma);
    let disc_q = (-q * t).exp();
    let disc_r = (-r * t).exp();
    match flag {
        bsm::Flag::Call => s * disc_q * cdf_branchless(d1) - k * disc_r * cdf_branchless(d2),
        bsm::Flag::Put => k * disc_r * cdf_branchless(-d2) - s * disc_q * cdf_branchless(-d1),
    }
}

fn bench_bsm_price_scalar(c: &mut Criterion) {
    c.bench_function("bsm_price_scalar", |b| {
        b.iter(|| {
            bsm::price(
                bsm::Flag::Call,
                black_box(100.0),
                black_box(105.0),
                black_box(0.5),
                black_box(0.05),
                black_box(0.0),
                black_box(0.20),
            );
        });
    });
}

fn bench_bsm_price_vec(c: &mut Criterion) {
    let strikes: Vec<f64> = (0..VEC_LEN)
        .map(|i| 80.0 + 40.0 * (i as f64) / (VEC_LEN as f64))
        .collect();
    let mut group = c.benchmark_group("bsm_price_vec");
    group.throughput(Throughput::Elements(VEC_LEN as u64));
    group.bench_function(BenchmarkId::from_parameter(VEC_LEN), |b| {
        b.iter(|| {
            let mut out = Vec::with_capacity(VEC_LEN);
            for &k in &strikes {
                out.push(bsm::price(
                    bsm::Flag::Call,
                    black_box(100.0),
                    black_box(k),
                    black_box(0.5),
                    black_box(0.05),
                    black_box(0.0),
                    black_box(0.20),
                ));
            }
            out
        });
    });
    group.finish();
}

fn bench_bsm_vega_vec(c: &mut Criterion) {
    let strikes: Vec<f64> = (0..VEC_LEN)
        .map(|i| 80.0 + 40.0 * (i as f64) / (VEC_LEN as f64))
        .collect();
    let mut group = c.benchmark_group("bsm_vega_vec");
    group.throughput(Throughput::Elements(VEC_LEN as u64));
    group.bench_function(BenchmarkId::from_parameter(VEC_LEN), |b| {
        b.iter(|| {
            let mut out = Vec::with_capacity(VEC_LEN);
            for &k in &strikes {
                out.push(greeks::vega(
                    black_box(100.0),
                    black_box(k),
                    black_box(0.5),
                    black_box(0.05),
                    black_box(0.0),
                    black_box(0.20),
                ));
            }
            out
        });
    });
    group.finish();
}

fn bench_iv_solve_scalar(c: &mut Criterion) {
    // ATM, mid-vol — the "middle segment" (β ∈ [b_l, b_h]) of LBR, where the
    // initial guess + Householder converges in ≤ 2 iterations.  Typical case.
    let target = bsm::price(bsm::Flag::Call, 100.0, 100.0, 0.5, 0.05, 0.0, 0.20);
    c.bench_function("iv_solve_scalar_atm", |b| {
        b.iter(|| {
            iv::solve(
                black_box(target),
                bsm::Flag::Call,
                black_box(100.0),
                black_box(100.0),
                black_box(0.5),
                black_box(0.05),
                black_box(0.0),
            );
        });
    });

    // OTM short expiry — small-t evaluator + lower-segment objective.  This
    // is the slower of the two scalar bench rows; a regression here is the
    // more diagnostic signal because it exercises more of LBR's machinery.
    let target_otm = bsm::price(bsm::Flag::Put, 100.0, 80.0, 0.05, 0.03, 0.01, 0.45);
    c.bench_function("iv_solve_scalar_otm_short", |b| {
        b.iter(|| {
            iv::solve(
                black_box(target_otm),
                bsm::Flag::Put,
                black_box(100.0),
                black_box(80.0),
                black_box(0.05),
                black_box(0.03),
                black_box(0.01),
            );
        });
    });
}

fn bench_iv_solve_vec(c: &mut Criterion) {
    let strikes: Vec<f64> = (0..VEC_LEN)
        .map(|i| 80.0 + 40.0 * (i as f64) / (VEC_LEN as f64))
        .collect();
    let targets: Vec<f64> = strikes
        .iter()
        .map(|&k| bsm::price(bsm::Flag::Call, 100.0, k, 0.5, 0.05, 0.0, 0.20))
        .collect();
    let mut group = c.benchmark_group("iv_solve_vec");
    group.throughput(Throughput::Elements(VEC_LEN as u64));
    group.bench_function(BenchmarkId::from_parameter(VEC_LEN), |b| {
        b.iter(|| {
            let mut out = Vec::with_capacity(VEC_LEN);
            for (i, &k) in strikes.iter().enumerate() {
                out.push(iv::solve(
                    black_box(targets[i]),
                    bsm::Flag::Call,
                    black_box(100.0),
                    black_box(k),
                    black_box(0.5),
                    black_box(0.05),
                    black_box(0.0),
                ));
            }
            out
        });
    });
    group.finish();
}

fn bench_bsm_greeks_all_vec(c: &mut Criterion) {
    // Locks in the F1 (combined-greeks) win: `greeks::all` shares d1_d2 /
    // disc_q / disc_r / cdf / pdf across all five Greeks, eliminating the
    // ~80% redundancy of calling delta+gamma+vega+theta+rho separately.
    let strikes: Vec<f64> = (0..VEC_LEN)
        .map(|i| 80.0 + 40.0 * (i as f64) / (VEC_LEN as f64))
        .collect();
    let mut group = c.benchmark_group("bsm_greeks_all_vec");
    group.throughput(Throughput::Elements(VEC_LEN as u64));
    group.bench_function(BenchmarkId::new("combined", VEC_LEN), |b| {
        b.iter(|| {
            let mut d = Vec::with_capacity(VEC_LEN);
            let mut g = Vec::with_capacity(VEC_LEN);
            let mut v = Vec::with_capacity(VEC_LEN);
            let mut th = Vec::with_capacity(VEC_LEN);
            let mut rh = Vec::with_capacity(VEC_LEN);
            for &k in &strikes {
                let (a, b2, c2, d2, e) = greeks::all(
                    bsm::Flag::Call,
                    black_box(100.0),
                    black_box(k),
                    black_box(0.5),
                    black_box(0.05),
                    black_box(0.0),
                    black_box(0.20),
                );
                d.push(a);
                g.push(b2);
                v.push(c2);
                th.push(d2);
                rh.push(e);
            }
            (d, g, v, th, rh)
        });
    });
    // The "individual" arm is the baseline the combined path must beat —
    // five separate Vec builds, one per Greek, the way `bs.greeks()` does
    // it pre-F1. Keep both arms so any future regression in the combined
    // path is visible as the gap closing.
    group.bench_function(BenchmarkId::new("individual", VEC_LEN), |b| {
        b.iter(|| {
            let mut d = Vec::with_capacity(VEC_LEN);
            let mut g = Vec::with_capacity(VEC_LEN);
            let mut v = Vec::with_capacity(VEC_LEN);
            let mut th = Vec::with_capacity(VEC_LEN);
            let mut rh = Vec::with_capacity(VEC_LEN);
            for &k in &strikes {
                d.push(greeks::delta(
                    bsm::Flag::Call,
                    black_box(100.0),
                    black_box(k),
                    black_box(0.5),
                    black_box(0.05),
                    black_box(0.0),
                    black_box(0.20),
                ));
                g.push(greeks::gamma(
                    black_box(100.0),
                    black_box(k),
                    black_box(0.5),
                    black_box(0.05),
                    black_box(0.0),
                    black_box(0.20),
                ));
                v.push(greeks::vega(
                    black_box(100.0),
                    black_box(k),
                    black_box(0.5),
                    black_box(0.05),
                    black_box(0.0),
                    black_box(0.20),
                ));
                th.push(greeks::theta(
                    bsm::Flag::Call,
                    black_box(100.0),
                    black_box(k),
                    black_box(0.5),
                    black_box(0.05),
                    black_box(0.0),
                    black_box(0.20),
                ));
                rh.push(greeks::rho(
                    bsm::Flag::Call,
                    black_box(100.0),
                    black_box(k),
                    black_box(0.5),
                    black_box(0.05),
                    black_box(0.0),
                    black_box(0.20),
                ));
            }
            (d, g, v, th, rh)
        });
    });
    group.finish();
}

/// F2 investigation bench (audit/mechanical-sympathy): does removing the
/// `|x| < 4` branch in `normal::cdf` help, hurt, or neither, when measured
/// through `bsm::price`?
///
/// **Verdict (2026-05, Apple M4 Pro): rejected.** Branchless is 14-69%
/// slower across all three input distributions below. `libm` is scalar
/// with its own internal branches, so removing the outer `|x|<4` test
/// does not enable auto-vectorisation; branchless just pays for two
/// transcendentals (`erfcx`+`exp`) instead of one (`erf`) on every call.
///
/// Input distributions:
/// - `narrow`: strikes 80..120 — d1 in [-1.0, +1.8], branch always taken,
///   no mispredicts possible. Tests pure per-call overhead.
/// - `wide`: strikes 20..200 — d1 in [-3.9, +11.6], one contiguous
///   boundary crossing, still highly predictable.
/// - `shuffled`: `wide` strikes in Knuth-multiplicative-permuted order.
///   Worst case for the branch predictor.
fn bench_cdf_branch_experiment(c: &mut Criterion) {
    let narrow: Vec<f64> = (0..VEC_LEN)
        .map(|i| 80.0 + 40.0 * (i as f64) / (VEC_LEN as f64))
        .collect();
    let wide: Vec<f64> = (0..VEC_LEN)
        .map(|i| 20.0 + 180.0 * (i as f64) / (VEC_LEN as f64))
        .collect();
    // Deterministic permutation: i -> (i * 2654435769) mod VEC_LEN. The
    // multiplier is coprime to 10000, so this visits each index exactly
    // once in a hash-scrambled order. Reproducible across runs.
    let shuffled: Vec<f64> = (0..VEC_LEN)
        .map(|i| {
            let j = (i.wrapping_mul(2_654_435_769)) % VEC_LEN;
            wide[j]
        })
        .collect();

    let cases: [(&str, &[f64]); 3] = [
        ("narrow", &narrow),
        ("wide", &wide),
        ("shuffled", &shuffled),
    ];
    let mut group = c.benchmark_group("cdf_branch_experiment");
    group.throughput(Throughput::Elements(VEC_LEN as u64));
    for (label, strikes) in &cases {
        group.bench_function(BenchmarkId::new("branched", *label), |b| {
            b.iter(|| {
                let mut out = Vec::with_capacity(VEC_LEN);
                for &k in *strikes {
                    out.push(bsm::price(
                        bsm::Flag::Call,
                        black_box(100.0),
                        black_box(k),
                        black_box(0.5),
                        black_box(0.05),
                        black_box(0.0),
                        black_box(0.20),
                    ));
                }
                out
            });
        });
        group.bench_function(BenchmarkId::new("branchless", *label), |b| {
            b.iter(|| {
                let mut out = Vec::with_capacity(VEC_LEN);
                for &k in *strikes {
                    out.push(bsm_price_branchless(
                        bsm::Flag::Call,
                        black_box(100.0),
                        black_box(k),
                        black_box(0.5),
                        black_box(0.05),
                        black_box(0.0),
                        black_box(0.20),
                    ));
                }
                out
            });
        });
    }
    group.finish();
}

fn bench_black76_price_scalar(c: &mut Criterion) {
    c.bench_function("black76_price_scalar", |b| {
        b.iter(|| {
            black76::price(
                bsm::Flag::Call,
                black_box(100.0),
                black_box(105.0),
                black_box(0.5),
                black_box(0.05),
                black_box(0.20),
            );
        });
    });
}

criterion_group!(
    benches,
    bench_bsm_price_scalar,
    bench_bsm_price_vec,
    bench_bsm_vega_vec,
    bench_iv_solve_scalar,
    bench_iv_solve_vec,
    bench_bsm_greeks_all_vec,
    bench_cdf_branch_experiment,
    bench_black76_price_scalar,
);
criterion_main!(benches);
