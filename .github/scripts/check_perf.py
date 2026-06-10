#!/usr/bin/env python3
"""Compare two criterion baselines and fail on regressions beyond a threshold.

Usage: check_perf.py <base-name> <new-name> <threshold-fraction>

Walks `target/criterion/` for paired `estimates.json` files (one under
`<base-name>/`, one under `<new-name>/`). For each pair, computes the
percent change of the mean estimate. Exits non-zero if any benchmark
regressed by more than its ceiling (the third CLI arg by default, or a
per-benchmark override from `PER_BENCH_THRESHOLDS`).

Bootstrap behavior: when no `<base-name>/estimates.json` exists for a given
benchmark (typical on the first PR that adds the bench harness), the
benchmark is skipped, not failed.

Per-benchmark overrides: `PER_BENCH_THRESHOLDS` below lets specific benches
use a higher ceiling than the global default. Use sparingly and document
the reason; each entry encodes a deliberate accepted regression.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Per-benchmark threshold overrides (fraction). A bench name matches when it
# equals the key exactly, or when it is a child path (criterion groups expose
# `<group>/<param>` under target/criterion/).
#
# LBR IV solver (Jäckel "Let's Be Rational") replaces the prior
# Newton+bisection in PR feat/lbr-iv.  Two sources of regression:
#
# (1) IV solver:  LBR evaluates a 4-region `b(x, s)` dispatcher and runs
#     a bounded Householder-4 iteration; Newton called a single
#     `bsm::price` per iter and stopped at 1e-10 absolute (~1e-6 IV).
#     The tradeoff buys ~10^7 more IV precision and bounded worst-case
#     latency at the no-arbitrage boundary.  Local measurements (Apple
#     M4 Pro) show +30-55%; GitHub-hosted Linux runners measure +73-79%
#     on the same benches — slower libm, shared CPU.
#
# (2) Price/Greeks:  `normal::cdf` switches to an `erfcx`-based form for
#     `|z| > 4` so `bsm::price` stays f64-accurate in the deep OTM tail
#     (libm::erf saturates at +/-1 past z/sqrt(2) ≈ 5.93, which made the
#     prior cdf lossy and broke LBR's lower-map roundtrip at extreme x).
#     The added branch on the common path is amortised away in the
#     vectorised benches (bsm_price_vec is actually *faster* on CI) but
#     remains visible on the scalar paths.  CI measurements:
#       initial implementation:           bsm_price_scalar +30%
#       after #[cold] on tail:            bsm_price_scalar +30% (no win on x86_64)
#       after INV_SQRT_2 + drop
#         #[inline(never)] on tail:       bsm_price_scalar +23%
#     The remaining 23-26% is the irreducible cost on Linux x86_64 of
#     keeping bsm::price f64-accurate at deep OTM, which the differential
#     test against py_vollib requires.  Splitting cdf into fast and precise
#     variants would dodge the cost but breaks the differential at the
#     |d1| ~ 5-6 inputs in the differential grid.
#
# Ceilings are set comfortably above the CI-measured worst case to absorb
# day-to-day GitHub-runner noise.
#
# === ENGINE-REROUTE TRANSITION: ceilings below marked (T) are TEMPORARY ===
# bsm::price now evaluates the normalised-Black engine (the IV solver's ~1-ULP
# b(x, s)) instead of the textbook S*N(d1) - K*N(d2), fixing the deep-OTM
# cancellation at the cost of ~2x pricing throughput (the engine's region
# dispatch + erfcx is far heavier than two cdf calls). CI (x86_64) measured
# +90% to +148% on every price-touching bench (Greeks/IV are unaffected); M4
# shows vectorised pricing ~2.3x (10k: 152x -> 68x vs py_vollib). This is a
# DELIBERATE, accepted one-time baseline shift.
#
# The gate compares PR-head vs PR-base, so once this PR merges the engine
# becomes the base and these paths return to ~0%. The (T) ceilings exist only
# to let this transition PR through and MUST be reset (production paths -> 0.10
# default; experiment harnesses -> 0.50) in the next PR -- leaving them is the
# stale-threshold trap the 2026-06 audit flagged. Tracked as a follow-up.
PER_BENCH_THRESHOLDS: dict[str, float] = {
    # IV solver (LBR) -- NOT touched by the engine reroute; pre-existing
    # overrides for the Newton->LBR transition baked into the base. (Audit:
    # candidates for a reset to 0.10 too, but out of scope here.)
    "iv_solve_scalar_atm": 0.85,
    "iv_solve_vec": 0.80,
    # (T) Production price paths -- engine reroute, reset to 0.10 after merge:
    "bsm_price_scalar": 1.80,
    "black76_price_scalar": 1.80,
    "bsm_price_vec": 1.80,
    "parallel_bsm_price": 1.80,
    # Audit (audit/mechanical-sympathy) bench harnesses that document
    # measurement-backed decisions rather than gate production perf. Normally
    # 0.50 (variant-comparison noise); (T)-bumped because they also exercise the
    # rerouted price / changed cdf -- reset to 0.50 after merge:
    # - cdf_branch_experiment (F2 rejected): branched vs branchless cdf.
    # - bsm_price_flag_dispatch (F5 rejected): call/put branch distributions.
    "cdf_branch_experiment": 1.80,
    "bsm_price_flag_dispatch": 1.80,
    # parallel/* (F4 / F4b) experiment harness -- kept at 0.50. NB: this key
    # matches only a literal `parallel/...` group; the price sub-bench is
    # parallel_bsm_price (its own (T) entry above), iv/greeks are unaffected.
    "parallel": 0.50,
}


def threshold_for(rel: str, default: float) -> tuple[float, bool]:
    """Return (threshold, is_override) for the given benchmark path."""
    for prefix, t in PER_BENCH_THRESHOLDS.items():
        if rel == prefix or rel.startswith(prefix + "/"):
            return t, True
    return default, False


def main() -> int:
    if len(sys.argv) != 4:
        print(f"usage: {sys.argv[0]} <base> <new> <threshold>", file=sys.stderr)
        return 2
    base_name = sys.argv[1]
    new_name = sys.argv[2]
    threshold = float(sys.argv[3])

    root = Path("target/criterion")
    if not root.exists():
        print(f"::error::{root} not found; nothing to compare", file=sys.stderr)
        return 1

    failed: list[tuple[str, float, float]] = []
    compared = 0
    skipped = 0
    any_override_applied = False

    for new_est in sorted(root.rglob(f"{new_name}/estimates.json")):
        bench_dir = new_est.parent.parent
        base_est = bench_dir / base_name / "estimates.json"
        rel = bench_dir.relative_to(root).as_posix()

        if not base_est.exists():
            skipped += 1
            print(f"::notice::Skipping {rel} (no '{base_name}' baseline)")
            continue

        with new_est.open() as f:
            new_data = json.load(f)
        with base_est.open() as f:
            base_data = json.load(f)
        new_mean = float(new_data["mean"]["point_estimate"])
        base_mean = float(base_data["mean"]["point_estimate"])
        if base_mean <= 0:
            print(f"::warning::Skipping {rel} (non-positive base mean)")
            skipped += 1
            continue

        change = (new_mean - base_mean) / base_mean
        bench_threshold, overridden = threshold_for(rel, threshold)
        any_override_applied = any_override_applied or overridden
        compared += 1
        # Criterion stores times in nanoseconds.
        marker = "*" if overridden else " "
        print(
            f" {marker}{rel:47s}  {base_mean:>12.1f} ns -> {new_mean:>12.1f} ns "
            f" ({change * 100:+.2f}%, ceiling {bench_threshold * 100:.0f}%)"
        )
        if change > bench_threshold:
            failed.append((rel, change, bench_threshold))

    print(f"\nCompared {compared} benchmarks, skipped {skipped}.")
    if any_override_applied:
        print("(*) per-benchmark override applied — see PER_BENCH_THRESHOLDS in check_perf.py.")
    if failed:
        print(f"\n::error::{len(failed)} benchmark(s) exceeded their ceiling:")
        for name, change, ceiling in failed:
            print(f"  - {name}: +{change * 100:.2f}% (ceiling {ceiling * 100:.0f}%)")
        return 1
    print("All compared benchmarks within their ceilings.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
