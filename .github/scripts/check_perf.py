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
# This table stays small by design. The gate compares PR-head against PR-base,
# so an override only earns its keep while a cost is being *introduced* (head
# has it, base does not); once the PR merges, the cost is in the base and the
# bench returns to ~0%, so the override is dropped in the next PR rather than
# kept forever. Leaving a merged-in override behind is a stale-threshold trap:
# it silently masks the next real regression on that bench. Two price-path
# shifts were retired exactly this way -- the `erfcx` deep-OTM cdf tail and the
# normalised-Black price engine -- and the price benches now ride the default
# 10% ceiling again.
#
# Ceilings are set comfortably above the CI-measured worst case to absorb
# day-to-day GitHub-runner noise.
PER_BENCH_THRESHOLDS: dict[str, float] = {
    # IV solver (Jäckel "Let's Be Rational", #10): the bounded Householder
    # iteration is intrinsically more work per call than the old
    # Newton+bisection -- CI observed +77.8%/+79.4% (scalar) and +73.5%/+73.4%
    # (vec) at introduction, and the ceilings give ~5 pp margin. Like the
    # retired price overrides these are head-vs-base stale (the LBR cost is in
    # the base, so a normal PR sees ~0% here too) and are candidates for a reset
    # to the 0.10 default; kept for now, out of scope of the audit follow-up
    # that removed the engine-reroute ceilings.
    "iv_solve_scalar_atm": 0.85,
    "iv_solve_vec": 0.80,
    # iv_solve_scalar_otm_short needs no override: CI shows LBR is *faster*
    # there (-10.88%) because Newton degenerated to bisection at OTM short
    # expiry (small vega) while LBR stays at <= 2 Householder iters. The
    # default 10% ceiling guards it against future regression.
    #
    # Audit (audit/mechanical-sympathy) bench harnesses that document
    # measurement-backed decisions rather than gate production perf:
    # - cdf_branch_experiment (F2 rejected): the branchless arm is 14-69%
    #   slower by design, so comparing branched vs branchless absolutely is the
    #   point; the 50% ceiling just absorbs runner noise.
    # - bsm_price_flag_dispatch (F5 rejected): three input distributions mapping
    #   to different inner-loop branches; the relative ratio matters more than
    #   the absolute, which drifts together under noise.
    "cdf_branch_experiment": 0.50,
    "bsm_price_flag_dispatch": 0.50,
    # The F4/F4b parallel experiment harness (serial-vs-rayon across N, the input
    # to the rayon-threshold decision -- not a production gate) is deliberately
    # absent. Its groups are declared `benchmark_group("parallel/<sub>")`, but
    # criterion flattens the '/' in a group name to '_', so the on-disk paths are
    # parallel_bsm_price / parallel_greeks_all / parallel_iv_solve -- a bare
    # `parallel` prefix key matches none of them (the perf-gate log confirms they
    # run on the default 10%). Production price / greeks / iv are gated by
    # bsm_price_vec / bsm_greeks_all_vec / iv_solve_vec respectively. If a noisy
    # rayon-small-N arm ever flakes the gate, widen that sub-bench with its own
    # explicit `parallel_<sub>` key (e.g. 0.50) -- not a `parallel` prefix.
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
