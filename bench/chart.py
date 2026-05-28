"""Render the BSM-pricing-throughput chart embedded in the README.

Writes two variants (light + dark theme) to docs/assets/:

    docs/assets/perf-light.svg
    docs/assets/perf-dark.svg

The README references both via a <picture> element so GitHub serves the
right one based on the reader's prefers-color-scheme.

Usage:
    uv pip install matplotlib
    python bench/chart.py

Update the data block below if the numbers in bench/compare_py_vollib.py
change. Both files should agree.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# (n_options, pyvolr_microseconds, py_vollib_microseconds)
# Refreshed post-audit (mechanical-sympathy F1/F3/F4); pyvolr column
# from `python bench/compare_py_vollib.py`. py_vollib column unchanged
# (same library version, baseline preserved across runs).
N = np.array([1, 1_000, 10_000, 100_000, 1_000_000])
PV = np.array([4.2, 24.6, 153.0, 1_390.0, 14_540.0])
PVOL = np.array([2.2, 2_320.0, 23_320.0, 234_910.0, 2_350_000.0])

PV_COLOR = "#CE422B"  # rust orange
PVOL_COLOR_LIGHT = "#6C757D"  # slate gray for light theme
PVOL_COLOR_DARK = "#9CA3AF"  # lighter slate for dark theme


def render(out: Path, *, theme: str) -> None:
    """Render one variant of the chart to `out`."""
    if theme == "light":
        text_color = "#222"
        muted_color = "#888"
        grid_color = "#999"
        spine_color = "#999"
        pvol_color = PVOL_COLOR_LIGHT
    else:
        text_color = "#E5E7EB"
        muted_color = "#9CA3AF"
        grid_color = "#4B5563"
        spine_color = "#6B7280"
        pvol_color = PVOL_COLOR_DARK

    fig, ax = plt.subplots(figsize=(10, 6), dpi=120)
    fig.patch.set_alpha(0.0)  # transparent — README background shows through
    ax.set_facecolor("none")

    # Shaded gap between the two series.
    ax.fill_between(N, PV, PVOL, color=PV_COLOR, alpha=0.06, zorder=0)

    ax.loglog(
        N,
        PVOL,
        "s-",
        color=pvol_color,
        linewidth=2.2,
        markersize=8,
        label="py_vollib (pure Python)",
        zorder=3,
    )
    ax.loglog(
        N,
        PV,
        "o-",
        color=PV_COLOR,
        linewidth=2.6,
        markersize=9,
        label="pyvolr (Rust core)",
        zorder=4,
    )

    # Speedup annotation under each pyvolr point.
    for ni, pi, pli in zip(N, PV, PVOL, strict=True):
        speedup = pli / pi
        if speedup < 1:
            txt = f"{speedup:.1f}×"  # noqa: RUF001 — multiplication sign is intentional for chart label
            color = pvol_color
        else:
            txt = f"{speedup:.0f}× faster"  # noqa: RUF001
            color = PV_COLOR
        ax.annotate(
            txt,
            xy=(ni, pi),
            xytext=(0, -18),
            textcoords="offset points",
            ha="center",
            fontsize=10,
            fontweight="bold",
            color=color,
        )

    ax.set_xlabel(
        "Array size (number of options priced per call)",
        fontsize=12,
        labelpad=10,
        color=text_color,
    )
    ax.set_ylabel("Wall time (µs, log scale)", fontsize=12, labelpad=10, color=text_color)
    ax.set_title(
        "Black-Scholes call pricing: pyvolr vs py_vollib",
        fontsize=15,
        fontweight="bold",
        pad=18,
        color=text_color,
    )

    ax.set_xticks(N)
    ax.set_xticklabels(["1\n(scalar)", "1k", "10k", "100k", "1M"], fontsize=11)
    ax.tick_params(colors=text_color)

    ax.grid(True, which="major", linestyle="--", alpha=0.35, color=grid_color, zorder=1)
    ax.grid(True, which="minor", linestyle=":", alpha=0.15, color=grid_color, zorder=1)

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color(spine_color)
    ax.spines["bottom"].set_color(spine_color)

    legend = ax.legend(fontsize=11, loc="upper left", frameon=False)
    for text in legend.get_texts():
        text.set_color(text_color)

    fig.text(
        0.5,
        0.01,
        "Apple M4 Pro · Python 3.10.20 · pyvolr 0.1.2 vs py_vollib 1.0.1 · "
        "IV via Jäckel “Let's Be Rational” (1e-13 precision)",
        ha="center",
        fontsize=9,
        color=muted_color,
        style="italic",
    )

    plt.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out, format="svg", transparent=True, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    out_dir = Path(__file__).resolve().parent.parent / "docs" / "assets"
    out_dir.mkdir(parents=True, exist_ok=True)
    render(out_dir / "perf-light.svg", theme="light")
    render(out_dir / "perf-dark.svg", theme="dark")
    print(f"wrote {out_dir / 'perf-light.svg'}")
    print(f"wrote {out_dir / 'perf-dark.svg'}")


if __name__ == "__main__":
    main()
