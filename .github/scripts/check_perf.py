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
# The table is currently EMPTY, and that is the steady state. The gate compares
# PR-head against PR-base, so an override only earns its keep while a cost is
# being *introduced* (head has it, base does not); once the PR merges, the cost
# is in the base, the bench returns to ~0%, and the override must be dropped in
# the next PR. Leaving a merged-in override behind is a stale-threshold trap:
# it silently masks the next real regression on that bench. Three shifts were
# retired exactly this way -- the Newton->LBR IV solver (#10, ceilings 0.85/
# 0.80), the `erfcx` deep-OTM cdf tail, and the normalised-Black price engine
# (#19, ceilings 1.80) -- every bench now rides the default 10% ceiling.
#
# The gate also no longer measures the decision-record experiment harness
# (F2 branchless cdf, F5 flag dispatch, F4/F4b serial-vs-rayon): it lives in
# benches/experiments.rs, outside `--bench pricing`, so its variant-comparison
# arms (slower by design) need no ceilings here.
#
# When adding an entry: set it comfortably above the CI-measured worst case
# (runner noise is 3-8%), document the measurement, and mark the PR that will
# remove it.
PER_BENCH_THRESHOLDS: dict[str, float] = {}


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
