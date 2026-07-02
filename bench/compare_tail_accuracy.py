"""Implied-vol recovery accuracy vs option-price depth (the deep-OTM tail).

One-command runner: `just accuracy` (env pins + setup live in the repo justfile).
Sweep in each venv, then chart the merged result:

    .venv-bench-entrants/bin/python bench/compare_tail_accuracy.py sweep
    .venv-bench/bin/python          bench/compare_tail_accuracy.py sweep
    .venv-bench312/bin/python       bench/compare_tail_accuracy.py sweep
    .venv-bench-entrants/bin/python bench/compare_tail_accuracy.py chart

Methodology: a ladder of well-posed deep-OTM calls (K=100, T=0.05, r=0.05,
sigma_true=0.16; S swept ~99 down to ~33) gives target prices spanning ~1e-1
down to ~1e-215. Each library inverts the same target price; the plotted
quantity is |sigma_recovered - sigma_true| / sigma_true. (sigma_true avoids
0.20 on purpose — see the note at SIGMA_TRUE for why.)

The forward map (sigma_true -> price) is pyvolr's normalised-Black engine,
which is pinned to 60-digit mpmath goldens at ~1-ULP through this region
(crates/core/src/bsm.rs price_matches_mpmath_goldens, deep_otm_put ~1e-201),
so target prices are trustworthy well past every competitor's failure point.
A library that raises, returns NaN, or returns a wrong sigma shows up as a
gap or an error cliff. Exact recoveries are floored at 1e-17 for log plotting
(raw values, including 0.0, are stored in the JSON).

This is the reproducible artifact behind the README's "correct at every
moneyness" claim; speed is compare_competitors.py / compare_new_entrants.py.
Writes bench/.tail_accuracy_results.json.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")  # numba typeof FutureWarning spam on broadcast inputs

RESULTS_PATH = Path(__file__).resolve().parent / ".tail_accuracy_results.json"

# sigma_true deliberately avoids 0.20: at least one competitor's tail failure
# mode is returning a constant ~0.2, which a 0.20-truth sweep cannot detect
# (it masqueraded as correct in the first run of this script). 0.16 is not a
# round default anywhere in the tested solvers.
K, T, R, SIGMA_TRUE = 100.0, 0.05, 0.05, 0.16
# ln-moneyness ladder: even in m = ln(S/K), which spaces log10(price) roughly
# quadratically — dense enough near every observed failure cliff (1e-50/1e-55).
# Endpoint chosen so the deepest price stays above f64 underflow at sigma=0.16.
MONEYNESS = np.linspace(-0.01, -1.12, 44)
S_LADDER = [float(K * math.exp(m)) for m in MONEYNESS]


def target_prices(results: dict) -> list[tuple[float, float]]:
    """[(S, target_price)] for the ladder.

    Computed once via pyvolr's golden-pinned forward map and persisted into the
    JSON, so venvs without pyvolr (QuantLib / quantforge phases) reuse the
    exact same f64 targets (repr() round-trips bit-exactly).
    """
    stored = results.get("_ladder", {}).get("pairs")
    if stored:
        return [(float(s), float(p)) for s, p in stored]

    from pyvolr import bs as pv

    out = []
    for s in S_LADDER:
        p = float(pv.price("c", S=s, K=K, T=T, r=R, sigma=SIGMA_TRUE))
        if np.isfinite(p) and p > 0.0:
            out.append((s, p))
    results["_ladder"] = {"pairs": [[repr(s), repr(p)] for s, p in out]}
    return out


# ----------------------------------------------------------------------------
# Adapters: each returns (name, invert) where invert(price, S) -> sigma.
# Raising / returning non-finite marks the cell as failed (None in the JSON).
# ----------------------------------------------------------------------------


def adapter_pyvolr():
    import pyvolr
    from pyvolr import bs as pv

    def invert(p: float, s: float) -> float:
        return float(pv.implied_vol(p, "c", S=s, K=K, T=T, r=R, on_error="ignore"))

    return f"pyvolr {pyvolr.__version__}", invert


def adapter_vollib():
    from importlib.metadata import version as _pkg_version

    from vollib.black_scholes.implied_volatility import implied_volatility as v_iv

    def invert(p: float, s: float) -> float:
        return float(v_iv(p, s, K, T, R, "c"))

    return f"vollib {_pkg_version('vollib')}", invert


def adapter_opengreeks():
    from importlib.metadata import version as _pkg_version

    from opengreeks import black_scholes as og

    def invert(p: float, s: float) -> float:
        return float(og.implied_volatility(p, s, K, T, R, "c"))

    return f"opengreeks {_pkg_version('opengreeks')}", invert


def adapter_fast_vollib():
    from importlib.metadata import version as _pkg_version

    import fast_vollib as fv
    import numba  # noqa: F401 — raise (=> clean skip) when the extra isn't installed

    def invert(p: float, s: float) -> float:
        out = fv.fast_implied_volatility(
            p, s, K, T, R, "c", return_as="numpy", backend="numba", on_error="ignore"
        )
        return float(np.asarray(out).ravel()[0])

    return f"fast-vollib {_pkg_version('fast-vollib')} (numba)", invert


def adapter_py_vollib_vectorized():
    import py_vollib_vectorized  # noqa: F401 — side-effect: monkeypatches py_vollib
    from py_vollib.black_scholes.implied_volatility import implied_volatility as pv_iv

    def invert(p: float, s: float) -> float:
        out = pv_iv(p, s, K, T, R, "c")
        try:
            return float(out)
        except (TypeError, ValueError):
            return float(out.iloc[0, 0])

    return "py_vollib_vectorized 0.1.1", invert


def adapter_quantlib():
    import QuantLib as ql

    discount = math.exp(-R * T)
    sqrt_t = math.sqrt(T)

    def invert(p: float, s: float) -> float:
        forward = s / discount
        stddev = ql.blackFormulaImpliedStdDev(ql.Option.Call, K, forward, p, discount)
        return stddev / sqrt_t

    return f"QuantLib {ql.__version__}", invert


def adapter_quantforge():
    import quantforge as qf

    def invert(p: float, s: float) -> float:
        return float(qf.black_scholes.implied_volatility(p, s, K, T, R, is_call=True))

    return f"quantforge {qf.__version__}", invert


ADAPTERS = [
    adapter_pyvolr,
    adapter_vollib,
    adapter_opengreeks,
    adapter_fast_vollib,
    adapter_py_vollib_vectorized,
    adapter_quantlib,
    adapter_quantforge,
]


def run_sweep() -> None:
    if RESULTS_PATH.exists():
        results: dict = json.loads(RESULTS_PATH.read_text())
    else:
        results = {}
    ladder = target_prices(results)

    for adapter in ADAPTERS:
        try:
            name, invert = adapter()
        except Exception as exc:
            print(f"# skipped: {adapter.__name__}: {exc}")
            continue
        # First venv to provide a library wins; later phases carrying an older
        # duplicate (e.g. the pyvolr/vollib pins in the legacy venvs) skip, so
        # the merged JSON holds one row per library.
        prefix = name.split(" ")[0]
        dup = next((k for k in results if k != name and k.startswith(prefix + " ")), None)
        if dup is not None:
            print(f"# skipped: {name}: {dup!r} already measured")
            continue
        row: dict[str, float | None] = {}
        n_fail = 0
        worst = 0.0
        for s, p in ladder:
            try:
                sigma = invert(p, s)
            except Exception:
                row[repr(p)] = None
                n_fail += 1
                continue
            if not np.isfinite(sigma):
                row[repr(p)] = None
                n_fail += 1
                continue
            rel = abs(sigma - SIGMA_TRUE) / SIGMA_TRUE
            row[repr(p)] = rel
            worst = max(worst, rel)
        results[name] = row
        print(f"{name:<32} cells={len(ladder)}  raised/NaN={n_fail}  worst rel err={worst:.3e}")
        RESULTS_PATH.write_text(json.dumps(results, indent=2, sort_keys=True))

    print(f"\nwrote {RESULTS_PATH}")


# Same palette as compare_competitors.py so a library keeps its color across
# the README's chart pair. (Copied, not imported: bench scripts run as files.)
PV_COLOR = "#CE422B"
COMPETITOR_PALETTE: dict[str, tuple[str, str]] = {
    "quantforge": ("#1D4ED8", "#60A5FA"),
    "py_vollib_vectorized": ("#047857", "#34D399"),
    "QuantLib": ("#7C3AED", "#C4B5FD"),
    "vollib": ("#6C757D", "#9CA3AF"),
    "blackscholes": ("#B45309", "#FBBF24"),
    "opengreeks": ("#0E7490", "#67E8F9"),
    "fast-vollib": ("#BE185D", "#F472B6"),
}

# The chart's y-axis is "correct significant digits" = -log10(rel err), so that
# higher = better (matching the throughput chart's convention). f64 carries
# ~15.95 decimal digits, so a rel err at the machine floor caps here; exact
# recoveries (raw 0.0 in the JSON) also land at the cap.
DIGITS_CAP = 16.0


def _correct_digits(rel_err: float) -> float:
    """-log10(relative error), clamped to [0, DIGITS_CAP]. Bigger is more accurate."""
    if rel_err <= 0.0:
        return DIGITS_CAP
    return max(0.0, min(DIGITS_CAP, -math.log10(rel_err)))


def _color_for(name: str, *, theme: str) -> str:
    if name.startswith("pyvolr"):
        return PV_COLOR
    idx = 0 if theme == "light" else 1
    for key, pair in COMPETITOR_PALETTE.items():
        if name.startswith(key):
            return pair[idx]
    return "#888"


def _render(results: dict, out: Path | None, *, theme: str):
    import matplotlib.pyplot as plt

    if theme == "light":
        text_color, muted_color = "#222", "#888"
        grid_color, spine_color = "#999", "#999"
    else:
        text_color, muted_color = "#E5E7EB", "#9CA3AF"
        grid_color, spine_color = "#4B5563", "#6B7280"

    fig, ax = plt.subplots(figsize=(10, 6), dpi=120)
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("none")

    # pyvolr drawn last (on top); competitors ordered by how deep they survive
    # so the legend reads top-down "fails first -> fails last".
    def survival_depth(name: str) -> float:
        cells = results[name]
        ok = [float(p) for p, e in cells.items() if e is not None and e < 1e-3]
        return min(ok) if ok else float("inf")

    names = [n for n in results if n != "_ladder"]
    names.sort(key=lambda n: (n.startswith("pyvolr"), -survival_depth(n)))

    import matplotlib.patheffects as path_effects

    for name in names:
        cells = results[name]
        pts = sorted(
            ((float(p), e) for p, e in cells.items()),
            key=lambda t: t[0],
            reverse=True,  # shallow -> deep
        )
        xs = [p for p, e in pts if e is not None]
        ys = [_correct_digits(e) for _, e in pts if e is not None]
        if not xs:
            continue
        is_pyvolr = name.startswith("pyvolr")
        color = _color_for(name, theme=theme)
        if is_pyvolr:
            (line,) = ax.semilogx(
                xs,
                ys,
                "o-",
                color=color,
                linewidth=3.2,
                markersize=9,
                markeredgewidth=1.5,
                markeredgecolor=("#ffffff" if theme == "light" else "#1f2937"),
                label=name,
                alpha=1.0,
                zorder=10,
            )
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
            ax.semilogx(
                xs,
                ys,
                "s--",
                color=color,
                linewidth=1.6,
                markersize=5,
                label=name,
                alpha=0.65,
                zorder=3,
            )

    ax.invert_xaxis()  # deeper OTM (smaller price) rightward
    ax.set_xticks([10.0**e for e in range(0, -211, -30)])
    ax.set_xlabel(
        "Target option price (log scale — deeper out-of-the-money →)",
        fontsize=12,
        labelpad=10,
        color=text_color,
    )
    ax.set_ylabel(
        "Correct digits in recovered σ  (higher is better)",  # noqa: RUF001
        fontsize=12,
        labelpad=10,
        color=text_color,
    )
    ax.set_title(
        "Implied-vol recovery accuracy into the deep-OTM tail",
        fontsize=15,
        fontweight="bold",
        pad=18,
        color=text_color,
    )
    ax.set_ylim(-0.8, 16.8)
    ax.set_yticks([0, 4, 8, 12, 16])
    # f64-exact region sits at the top now (max attainable accuracy).
    ax.axhspan(13.0, 16.8, color=grid_color, alpha=0.10, zorder=0)
    # vollib and py_vollib_vectorized sit exactly under pyvolr's line (all
    # three are LBR-lineage and stay f64-exact) — say so, since the emphasis
    # styling hides their lines beneath pyvolr's.
    ax.text(
        0.55,
        14.6,
        "pyvolr · vollib · py_vollib_vectorized\noverlap up here — all f64-exact to the end",
        fontsize=9.5,
        color=muted_color,
        style="italic",
        ha="center",
        va="center",
        transform=ax.get_yaxis_transform(),
    )
    # A wrong constant volatility scores ~0 correct digits.
    ax.text(
        0.5,
        2.4,
        "the four that break saturate at a wrong constant σ — 0 correct digits",  # noqa: RUF001
        fontsize=9.5,
        color=muted_color,
        style="italic",
        ha="center",
        va="center",
        transform=ax.get_yaxis_transform(),
    )

    ax.grid(True, which="major", linestyle="--", alpha=0.35, color=grid_color, zorder=1)
    ax.grid(True, which="minor", linestyle=":", alpha=0.15, color=grid_color, zorder=1)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color(spine_color)
    ax.spines["bottom"].set_color(spine_color)
    ax.tick_params(colors=text_color)

    legend = ax.legend(fontsize=10, loc="center right", frameon=False)
    for text in legend.get_texts():
        text.set_color(text_color)

    pyvolr_label = next(n for n in results if n.startswith("pyvolr"))
    fig.text(
        0.5,
        0.01,
        f"K=100, T=0.05, r=0.05, true σ=0.16; S swept 99→33 · {pyvolr_label} forward map "  # noqa: RUF001
        "(pinned to 60-digit mpmath goldens) · a line ending early = solver raised or returned NaN",
        ha="center",
        fontsize=8.5,
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
        sys.exit(f"no results at {RESULTS_PATH}; run `sweep` first")
    results = json.loads(RESULTS_PATH.read_text())
    out_dir = Path(__file__).resolve().parent.parent / "docs" / "assets"
    out_dir.mkdir(parents=True, exist_ok=True)
    for theme in ("light", "dark"):
        out = out_dir / f"accuracy-iv-tail-{theme}.svg"
        _render(results, out, theme=theme)
        print(f"wrote {out}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["sweep", "chart"])
    args = p.parse_args()
    if args.cmd == "sweep":
        run_sweep()
    else:
        render_chart()


if __name__ == "__main__":
    main()
