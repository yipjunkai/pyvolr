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
use pyvolr_core::{black76, bsm, greeks, iv};

const VEC_LEN: usize = 10_000;

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
    bench_black76_price_scalar,
);
criterion_main!(benches);
