"""Benchmark pyvolr against the live competitor set (2026-05).

Run in two phases against two venvs, then chart the merged result:

    # phase 1 — pure-Python and numba/QuantLib stack (Python 3.11)
    .venv-bench/bin/python bench/compare_competitors.py bench

    # phase 2 — quantforge (Python 3.12, separate venv because quantforge
    # requires >=3.12 and py_vollib needs <=3.11)
    .venv-bench312/bin/python bench/compare_competitors.py bench

    # chart — combines both phases and pops the plot
    .venv-bench/bin/python bench/compare_competitors.py chart

Each phase appends to bench/.competitor_results.json. Skipped libraries
(import failure, runtime error, would-take-forever) are logged but never
break the run.
"""

from __future__ import annotations

import argparse
import functools
import json
import statistics
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

RESULTS_PATH = Path(__file__).resolve().parent / ".competitor_results.json"

# Fixed at-the-money-ish single contract for the scalar measurements, and a
# fan of strikes for vectorised measurements. Same inputs as the existing
# bench/compare_py_vollib.py so the pyvolr numbers stay comparable.
S, K, T, R, SIGMA = 100.0, 105.0, 0.5, 0.05, 0.20
SIZES = [1, 1_000, 10_000, 100_000, 1_000_000, 10_000_000]


def _strikes(n: int) -> np.ndarray:
    return np.linspace(80.0, 120.0, n)


def bench(
    fn: Callable[[], object], *, target_seconds: float = 2.0, max_reps: int = 50_000
) -> float:
    """Return median wall time of `fn()`. Reps chosen adaptively.

    Does one warmup, uses its time to pick reps so total spend ~= target_seconds.
    If a single call already exceeds target_seconds, runs 3 reps and takes the median.
    """
    t0 = time.perf_counter()
    fn()  # warmup, discarded
    warm = time.perf_counter() - t0
    if warm <= 0:
        warm = 1e-6
    reps = min(max_reps, max(3, int(target_seconds / warm)))
    samples = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples)


def fmt(secs: float) -> str:
    if secs < 1e-3:
        return f"{secs * 1e6:7.1f} us"
    if secs < 1.0:
        return f"{secs * 1e3:7.2f} ms"
    return f"{secs:7.2f} s"


# ----------------------------------------------------------------------------
# Library adapters: each returns (name, scalar_fn, vector_fn or None).
# scalar_fn(K_value) -> price; vector_fn(strikes_array) -> array-like prices.
# vector_fn=None means "no native vector path, loop the scalar".
# ----------------------------------------------------------------------------


def adapter_pyvolr():
    from pyvolr import bs as pv

    def scalar(k):
        return pv.price("c", S=S, K=k, T=T, r=R, sigma=SIGMA)

    def vector(ks):
        return pv.price("c", S=S, K=ks, T=T, r=R, sigma=SIGMA)

    import pyvolr

    return f"pyvolr {pyvolr.__version__}", scalar, vector


def adapter_vollib():
    from vollib.black_scholes import black_scholes as v_bs

    def scalar(k):
        return v_bs(flag="c", S=S, K=k, t=T, r=R, sigma=SIGMA)

    return "vollib 1.0.7", scalar, None


def adapter_py_vollib():
    # py_vollib 1.0.7 is now a deprecated thin shim over vollib — measuring
    # it separately exposes the redirect cost, but in practice it's identical
    # to vollib. Skip to avoid a duplicate line on the chart.
    raise RuntimeError("skipped — py_vollib 1.0.7 redirects to vollib")


def adapter_py_vollib_vectorized():
    import py_vollib_vectorized  # noqa: F401 — monkeypatches py_vollib
    from py_vollib.black_scholes import black_scholes as pv_bs

    # py_vollib_vectorized monkeypatches the scalar entry point to accept
    # arrays and dispatch through numba; that's the path pandas users actually
    # take, so we exercise it directly rather than via .vectorized_*.

    def scalar(k):
        return pv_bs("c", S, k, T, R, SIGMA)

    def vector(ks):
        # Pass numpy arrays — numba kernel handles broadcasting.
        return pv_bs("c", np.full_like(ks, S), ks, T, R, SIGMA)

    return "py_vollib_vectorized 0.1.1", scalar, vector


def adapter_blackscholes():
    from blackscholes import BlackScholesCall

    def scalar(k):
        return BlackScholesCall(S, k, T, R, SIGMA).price()

    return "blackscholes 0.2.0", scalar, None


def adapter_quantlib():
    import QuantLib as ql

    discount = float(np.exp(-R * T))
    stddev = SIGMA * float(np.sqrt(T))
    forward = S / discount  # so that F * discount = S

    def scalar(k):
        # ql.blackFormula is the bare formula — no object graph, fastest
        # QuantLib path for vanilla European pricing.
        return ql.blackFormula(ql.Option.Call, k, forward, stddev, discount)

    return f"QuantLib {ql.__version__}", scalar, None


def adapter_quantforge():
    import quantforge as qf

    def scalar(k):
        return qf.black_scholes.call_price(S, k, T, R, SIGMA)

    def vector(ks):
        n = len(ks)
        return qf.black_scholes.call_price_batch(
            np.full(n, S), ks, np.full(n, T), np.full(n, R), np.full(n, SIGMA)
        )

    return f"quantforge {qf.__version__}", scalar, vector


ADAPTERS = [
    adapter_pyvolr,
    adapter_vollib,
    adapter_py_vollib,
    adapter_py_vollib_vectorized,
    adapter_blackscholes,
    adapter_quantlib,
    adapter_quantforge,
]


# Per-library cap on input size when looping scalar calls. Pure-Python libs
# take ~200ns/call; 1M calls = 200ms, 10M = 2s — still OK. But blackscholes
# constructs a class per call (~5µs), so 10M = 50s. Cap conservatively.
SCALAR_LOOP_CAP = {
    "blackscholes 0.2.0": 100_000,
    "vollib 1.0.7": 1_000_000,
    "QuantLib 1.42.1": 1_000_000,
}


def run_bench() -> None:
    """Measure every importable adapter; merge into the JSON cache."""
    if RESULTS_PATH.exists():
        results: dict[str, dict[str, float]] = json.loads(RESULTS_PATH.read_text())
    else:
        results = {}

    print(f"{'library':<32} {'size':>10} {'time/call':>12} {'thru (op/s)':>14}")
    print("-" * 72)

    for adapter in ADAPTERS:
        try:
            name, scalar_fn, vector_fn = adapter()
        except Exception as exc:
            # Adapters legitimately raise to signal skip (missing import, etc.).
            print(f"# skipped: {adapter.__name__}: {exc}")
            continue

        timings: dict[str, float] = results.get(name, {})

        for n in SIZES:
            cap = SCALAR_LOOP_CAP.get(name)
            fn: Callable[[], object]
            if vector_fn is not None:
                ks = _strikes(n)
                fn = functools.partial(vector_fn, ks)
            else:
                if cap is not None and n > cap:
                    print(f"{name:<32} {n:>10,} {'(skip > cap)':>12}")
                    continue
                if n == 1:
                    fn = functools.partial(scalar_fn, K)
                else:
                    ks = _strikes(n)
                    # Bind both the strike list and the scalar fn explicitly
                    # so the closure can't be re-bound by the outer loop.
                    fn = functools.partial(
                        lambda strikes, sf: [sf(k) for k in strikes],
                        ks,
                        scalar_fn,
                    )

            try:
                t = bench(fn, target_seconds=2.0 if n <= 100_000 else 5.0)
            except Exception as exc:
                print(f"{name:<32} {n:>10,} FAILED — {type(exc).__name__}: {exc}")
                continue
            timings[str(n)] = t
            thru = n / t
            print(f"{name:<32} {n:>10,} {fmt(t):>12} {thru:>14,.0f}")

        results[name] = timings
        RESULTS_PATH.write_text(json.dumps(results, indent=2, sort_keys=True))

    print(f"\nwrote {RESULTS_PATH}")


# Palette: pyvolr stays in the rust-orange used by the README chart; every
# competitor gets a (light, dark) pair so both theme variants stay readable.
PV_COLOR = "#CE422B"
COMPETITOR_PALETTE: dict[str, tuple[str, str]] = {
    # name-prefix:           (light theme,  dark theme)
    "quantforge": ("#1D4ED8", "#60A5FA"),
    "py_vollib_vectorized": ("#047857", "#34D399"),
    "QuantLib": ("#7C3AED", "#C4B5FD"),
    "vollib": ("#6C757D", "#9CA3AF"),
    "blackscholes": ("#B45309", "#FBBF24"),
}


def _color_for(name: str, *, theme: str) -> str:
    if name.startswith("pyvolr"):
        return PV_COLOR
    idx = 0 if theme == "light" else 1
    for key, pair in COMPETITOR_PALETTE.items():
        if name.startswith(key):
            return pair[idx]
    return "#888"


def _render(
    results: dict[str, dict[str, float]],
    out: Path | None,
    *,
    theme: str,
    metric: str,
):
    """Render one themed chart.

    metric="time": y = wall time per call (µs, log).
    metric="thru": y = throughput (options/sec, log) — higher is better.
    If `out` is None, returns the figure for an interactive plt.show().
    """
    import matplotlib.pyplot as plt

    if theme == "light":
        text_color = "#222"
        muted_color = "#888"
        grid_color = "#999"
        spine_color = "#999"
    else:
        text_color = "#E5E7EB"
        muted_color = "#9CA3AF"
        grid_color = "#4B5563"
        spine_color = "#6B7280"

    fig, ax = plt.subplots(figsize=(10, 6), dpi=120)
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("none")

    # pyvolr last (drawn on top). For the time chart, order competitors
    # slowest-at-100k first so the legend reads top-down "worst → best".
    # For the throughput chart, fastest-at-100k first (same intent, flipped axis).
    def at_100k(name: str) -> float:
        return results[name].get("100000", float("inf"))

    competitor_sort_sign = 1 if metric == "time" else -1
    names = sorted(
        results.keys(),
        key=lambda n: (not n.startswith("pyvolr"), competitor_sort_sign * -at_100k(n)),
    )

    # Visual emphasis on pyvolr: solid line, white-edged markers, full opacity,
    # thicker stroke. Competitors are de-emphasised: dashed lines, smaller
    # markers, lower opacity. The contrast makes pyvolr legible at a glance
    # even with 6 lines on a log-log chart.
    import matplotlib.patheffects as path_effects

    for name in names:
        timings = results[name]
        ns = sorted(int(k) for k in timings)
        if metric == "time":
            ys = [timings[str(n)] * 1e6 for n in ns]
        else:
            ys = [n / timings[str(n)] for n in ns]
        is_pyvolr = name.startswith("pyvolr")
        color = _color_for(name, theme=theme)
        if is_pyvolr:
            (line,) = ax.loglog(
                ns,
                ys,
                "o-",
                color=color,
                linewidth=3.2,
                markersize=11,
                markeredgewidth=1.5,
                markeredgecolor=("#ffffff" if theme == "light" else "#1f2937"),
                label=name,
                alpha=1.0,
                zorder=10,
            )
            # Subtle outline so the pyvolr line stays distinct against any
            # competitor that happens to cross it on the log-log plot.
            line.set_path_effects(
                [
                    path_effects.Stroke(
                        linewidth=4.6,
                        foreground=("#ffffff" if theme == "light" else "#111827"),
                        alpha=0.9,
                    ),
                    path_effects.Normal(),
                ]
            )
        else:
            ax.loglog(
                ns,
                ys,
                "s--",
                color=color,
                linewidth=1.6,
                markersize=6,
                label=name,
                alpha=0.65,
                zorder=3,
            )

    ax.set_xlabel(
        "Array size (number of options priced per call)",
        fontsize=12,
        labelpad=10,
        color=text_color,
    )
    if metric == "time":
        ylabel = "Wall time (µs, log scale)"
        title = "Black-Scholes call pricing: wall time vs the 2026 competitor set"
        legend_loc = "upper left"
    else:
        ylabel = "Throughput (options priced / sec, log scale)"
        title = "Black-Scholes call pricing: throughput vs the 2026 competitor set"
        legend_loc = "lower right"
    ax.set_ylabel(ylabel, fontsize=12, labelpad=10, color=text_color)
    ax.set_title(title, fontsize=15, fontweight="bold", pad=18, color=text_color)

    ax.set_xticks([1, 1_000, 10_000, 100_000, 1_000_000, 10_000_000])
    ax.set_xticklabels(["1\n(scalar)", "1k", "10k", "100k", "1M", "10M"], fontsize=11)
    ax.tick_params(colors=text_color)

    ax.grid(True, which="major", linestyle="--", alpha=0.35, color=grid_color, zorder=1)
    ax.grid(True, which="minor", linestyle=":", alpha=0.15, color=grid_color, zorder=1)

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color(spine_color)
    ax.spines["bottom"].set_color(spine_color)

    legend = ax.legend(fontsize=10, loc=legend_loc, frameon=False)
    for text in legend.get_texts():
        text.set_color(text_color)

    fig.text(
        0.5,
        0.01,
        "Apple M4 Pro · pyvolr 0.1.2 (local) · "
        "vs vollib 1.0.7, py_vollib_vectorized 0.1.1, blackscholes 0.2.0, QuantLib 1.42.1, quantforge 0.1.1",
        ha="center",
        fontsize=9,
        color=muted_color,
        style="italic",
    )

    plt.tight_layout(rect=(0, 0.03, 1, 1))
    if out is not None:
        fig.savefig(out, format="svg", transparent=True, bbox_inches="tight")
        plt.close(fig)
        return None
    return fig


def render_chart() -> None:
    if not RESULTS_PATH.exists():
        sys.exit(f"no results at {RESULTS_PATH}; run `bench` first")
    results: dict[str, dict[str, float]] = json.loads(RESULTS_PATH.read_text())

    out_dir = Path(__file__).resolve().parent.parent / "docs" / "assets"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Older runs wrote perf-competitors-{light,dark}.svg without the
    # metric in the name; remove them so the asset set is unambiguous.
    for stale in ("perf-competitors-light.svg", "perf-competitors-dark.svg"):
        (out_dir / stale).unlink(missing_ok=True)

    for metric in ("time", "thru"):
        for theme in ("light", "dark"):
            out = out_dir / f"perf-competitors-{metric}-{theme}.svg"
            _render(results, out, theme=theme, metric=metric)
            print(f"wrote {out}")

    # Preview both light variants interactively in two windows.
    import matplotlib.pyplot as plt

    _render(results, None, theme="light", metric="time")
    _render(results, None, theme="light", metric="thru")
    plt.show()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["bench", "chart"])
    args = p.parse_args()
    if args.cmd == "bench":
        run_bench()
    else:
        render_chart()


if __name__ == "__main__":
    main()
