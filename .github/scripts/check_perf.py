#!/usr/bin/env python3
"""Compare two criterion baselines and fail on regressions beyond a threshold.

Usage: check_perf.py <base-name> <new-name> <threshold-fraction>

Walks `target/criterion/` for paired `estimates.json` files (one under
`<base-name>/`, one under `<new-name>/`). For each pair, computes the
percent change of the mean estimate. Exits non-zero if any benchmark
regressed by more than `threshold-fraction` (e.g. 0.10 == 10%).

Bootstrap behavior: when no `<base-name>/estimates.json` exists for a given
benchmark (typical on the first PR that adds the bench harness), the
benchmark is skipped, not failed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


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

    failed: list[tuple[str, float]] = []
    compared = 0
    skipped = 0

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
        compared += 1
        # Criterion stores times in nanoseconds.
        print(f"  {rel:48s}  {base_mean:>12.1f} ns -> {new_mean:>12.1f} ns  ({change * 100:+.2f}%)")
        if change > threshold:
            failed.append((rel, change))

    print(f"\nCompared {compared} benchmarks, skipped {skipped}.")
    if failed:
        print(
            f"\n::error::{len(failed)} benchmark(s) regressed by more than {threshold * 100:.0f}%:"
        )
        for name, change in failed:
            print(f"  - {name}: +{change * 100:.2f}%")
        return 1
    print(f"All compared benchmarks within {threshold * 100:.0f}% threshold.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
