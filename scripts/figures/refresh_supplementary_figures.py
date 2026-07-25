#!/usr/bin/env python3
"""
Standalone replot script for the two appendix figures, restricted to the
SUBSCALE level (paper main analysis unit).

Reads existing CSV outputs from `compute_supplementary_behavior.py`:
  - outputs/supplementary/human_split_half_bootstrap.csv
  - outputs/supplementary/llm_vs_human_baseline.csv
  - outputs/supplementary/intercept_results.csv

Writes:
  - reports/figures/figB0_split_half_baseline.{pdf,png}
  - reports/figures/figH1_intercept_scaling.{pdf,png}

Both figures restrict to the subscale level only. Item and scale levels are
omitted intentionally; the paper's main analysis is at the subscale level
(§3.1), so the appendix figures align with that.

Usage (from project root):
  python scripts/figures/refresh_supplementary_figures.py
  # Or override paths:
  python scripts/figures/refresh_supplementary_figures.py --sup_dir my_results --fig_dir my_figs
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from psychometric_inference.paths import FIGURE_DIR, PROJECT_ROOT, SUPPLEMENTARY_OUTPUT_DIR

# Color palette consistent with the existing compute_supplementary_behavior.py / plot_figures.py style
FAMILY_TYPE_COLORS = {
    ("Qwen",  "base"):     "#085041",
    ("Qwen",  "instruct"): "#5DCAA5",
    ("Llama", "base"):     "#993C1D",
    ("Llama", "instruct"): "#F0997B",
}


def plot_split_half_subscale(boot_df: pd.DataFrame,
                              llm_df: pd.DataFrame,
                              outdir: Path):
    """1 row × 3 col: r / slope / intercept at the subscale level only."""
    metrics = [("r", "subscale_r", r"alignment $r$"),
               ("slope", "subscale_slope", "OLS slope"),
               ("intercept", "subscale_intercept", "OLS intercept")]

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.6))

    for ax, (mlabel, col, xlabel) in zip(axes, metrics):
        if col not in boot_df.columns:
            ax.text(0.5, 0.5, f"col '{col}' not in CSV",
                    ha="center", va="center", transform=ax.transAxes,
                    color="grey")
            continue

        vals = boot_df[col].dropna().values
        # Histogram (density)
        ax.hist(vals, bins=50, alpha=0.55, color="#2c7bb6",
                density=True, edgecolor="white", linewidth=0.3,
                label="Human split-half")

        # Faint per-model lines
        if llm_df is not None and col in llm_df.columns:
            for _, row in llm_df.iterrows():
                v = row[col]
                if np.isnan(v):
                    continue
                key = (row.get("family"), row.get("type"))
                color = FAMILY_TYPE_COLORS.get(key, "grey")
                ax.axvline(v, color=color, alpha=0.25, lw=0.8)

            # Highlight largest base model per family (for clarity, like
            # the original 3x3 plot but restricted to base for cleaner story)
            for family in ["Qwen", "Llama"]:
                sub = llm_df[(llm_df["family"] == family) &
                             (llm_df["type"] == "base")]
                if sub.empty:
                    continue
                largest = sub.loc[sub["size"].idxmax()]
                v = largest[col]
                if np.isnan(v):
                    continue
                color = FAMILY_TYPE_COLORS[(family, "base")]
                lbl = f"{largest['model']}"
                ax.axvline(v, color=color, lw=2, ls="--", label=lbl)

        ax.set_xlabel(xlabel)
        ax.tick_params(axis="y", labelleft=False)  # density y-axis is uninformative
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("density")
    # Single shared legend (only the dashed marker lines)
    handles, labels = axes[-1].get_legend_handles_labels()
    if handles:
        axes[-1].legend(handles, labels, fontsize=7,
                        loc="upper left", frameon=False)

    plt.suptitle(
        "Human split-half baseline (subscale level) vs LLM-vs-human alignment",
        fontsize=11, y=1.00,
    )
    plt.tight_layout()

    outdir.mkdir(parents=True, exist_ok=True)
    pdf = outdir / "figB0_split_half_baseline.pdf"
    png = outdir / "figB0_split_half_baseline.png"
    plt.savefig(pdf, bbox_inches="tight")
    plt.savefig(png, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {pdf}")
    print(f"  Saved: {png}")


def plot_intercept_subscale(df: pd.DataFrame, outdir: Path):
    """Single panel: subscale-level intercept vs model size, base/instruct × family."""
    fig, ax = plt.subplots(1, 1, figsize=(5.2, 3.8))

    col = "subscale_intercept"
    if col not in df.columns:
        ax.text(0.5, 0.5, f"col '{col}' not in CSV",
                ha="center", va="center", transform=ax.transAxes, color="grey")
    else:
        for (family, mtype), color in FAMILY_TYPE_COLORS.items():
            sub = df[(df["family"] == family) & (df["type"] == mtype)] \
                .sort_values("size")
            if not len(sub):
                continue
            valid = sub[col].notna()
            if not valid.any():
                continue
            ls = "-" if mtype == "base" else "--"
            ax.plot(sub.loc[valid, "size"], sub.loc[valid, col],
                    marker="o", ms=6, lw=1.7, ls=ls, color=color,
                    label=f"{family} {mtype}")

    ax.axhline(0, color="grey", ls=":", lw=0.8, alpha=0.6)
    ax.set_xscale("log")
    ax.set_xlabel("Parameters (B)")
    ax.set_ylabel("OLS intercept")
    ax.set_title("Subscale-level OLS intercept vs model size", fontsize=11)
    ax.legend(fontsize=8, frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    outdir.mkdir(parents=True, exist_ok=True)
    pdf = outdir / "figH1_intercept_scaling.pdf"
    png = outdir / "figH1_intercept_scaling.png"
    plt.savefig(pdf, bbox_inches="tight")
    plt.savefig(png, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {pdf}")
    print(f"  Saved: {png}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sup_dir", default=str(SUPPLEMENTARY_OUTPUT_DIR),
                   help="Directory holding human_split_half_bootstrap.csv etc.")
    p.add_argument("--fig_dir", default=str(FIGURE_DIR),
                   help="Directory to write figures into.")
    args = p.parse_args()

    sup_dir = (PROJECT_ROOT / args.sup_dir
               if not Path(args.sup_dir).is_absolute()
               else Path(args.sup_dir))
    fig_dir = (PROJECT_ROOT / args.fig_dir
               if not Path(args.fig_dir).is_absolute()
               else Path(args.fig_dir))

    boot_csv = sup_dir / "human_split_half_bootstrap.csv"
    llm_csv = sup_dir / "llm_vs_human_baseline.csv"
    intercept_csv = sup_dir / "intercept_results.csv"

    print("=" * 70)
    print("  REPLOT (subscale-only versions of split-half + intercept figures)")
    print("=" * 70)
    print(f"  sup_dir: {sup_dir}")
    print(f"  fig_dir: {fig_dir}")

    # --- split-half ---
    if boot_csv.exists():
        print(f"\n[split-half] Reading {boot_csv.name}...")
        boot_df = pd.read_csv(boot_csv)
        llm_df = pd.read_csv(llm_csv) if llm_csv.exists() else None
        if llm_df is None:
            print(f"  WARN: {llm_csv.name} not found; histogram will have no overlay.")
        plot_split_half_subscale(boot_df, llm_df, fig_dir)
    else:
        print(f"\n[split-half] SKIP: {boot_csv} not found. "
              f"Run compute_supplementary_behavior.py first (section A).")

    # --- intercept scaling ---
    if intercept_csv.exists():
        print(f"\n[intercept] Reading {intercept_csv.name}...")
        df = pd.read_csv(intercept_csv)
        plot_intercept_subscale(df, fig_dir)
    else:
        print(f"\n[intercept] SKIP: {intercept_csv} not found. "
              f"Run compute_supplementary_behavior.py first (section B).")

    print("\nDone.")


if __name__ == "__main__":
    main()
