#!/usr/bin/env python3
"""
Compare representational structure (corrected slope) vs behavioral
amplification (subscale_slope averaged across 7 scales) for all 14 models.

Usage:
    python -m psychometric_inference.mechanisms.representation_behavior
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

from psychometric_inference.model_registry import behavioral_map_by_mech_name, mech_model_tuples
from psychometric_inference.paths import BEHAVIOR_OUTPUT_DIR, MECHANISTIC_OUTPUT_DIR

MECH_ROOT = MECHANISTIC_OUTPUT_DIR

BEH_MAP = behavioral_map_by_mech_name()
MODELS = mech_model_tuples(family_case="lower")


def main():
    # Load representational scaling results
    scaling_csv = MECH_ROOT / "scaling_results_full.csv"
    if not scaling_csv.exists():
        print(f"Missing {scaling_csv}. Run scaling_plots first.")
        return
    repr_df = pd.read_csv(scaling_csv)

    rows = []
    for name, size, family, mtype in MODELS:
        beh_name = BEH_MAP.get(name)
        if not beh_name:
            continue

        beh_csv = BEHAVIOR_OUTPUT_DIR / beh_name / "per_scale_alignment.csv"
        if not beh_csv.exists():
            print(f"  No behavioral data for {name} at {beh_csv}")
            continue

        beh_df = pd.read_csv(beh_csv)
        beh_slope_subscale = beh_df["subscale_slope"].mean()
        beh_slope_item = beh_df["item_slope"].mean()
        beh_r_subscale = beh_df["subscale_r"].mean()

        # Get representational corrected slope
        rr = repr_df[repr_df["model"] == name]
        if rr.empty:
            continue
        rr = rr.iloc[0]

        rows.append({
            "model": name,
            "size": size,
            "family": family,
            "mtype": mtype,
            "repr_corrected_slope": rr.get("geom_slope_corrected"),
            "repr_mantel_r": rr.get("geom_mantel_r"),
            "repr_reliability": rr.get("mean_reliability"),
            "beh_subscale_slope": beh_slope_subscale,
            "beh_item_slope": beh_slope_item,
            "beh_subscale_r": beh_r_subscale,
        })

    df = pd.DataFrame(rows)
    print("\nCombined results:")
    print(df.to_string(index=False))

    out_csv = MECH_ROOT / "structure_vs_behavior.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")

    # Scatter plot: representational slope vs behavioral slope
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    colors = {"llama": "#d7191c", "qwen": "#2c7bb6"}
    markers = {"instruct": "o", "base": "s"}

    pairs = [
        ("repr_corrected_slope", "beh_subscale_slope",
         "Repr corrected slope", "Behavioral subscale slope", 1.0),
        ("repr_mantel_r", "beh_subscale_slope",
         "Repr Mantel r", "Behavioral subscale slope", 1.0),
        ("repr_corrected_slope", "beh_subscale_r",
         "Repr corrected slope", "Behavioral subscale alignment r", None),
    ]

    for ax, (xcol, ycol, xlab, ylab, ref_y) in zip(axes, pairs):
        df_valid = df.dropna(subset=[xcol, ycol])

        # Per-family/mtype markers
        for family in ["llama", "qwen"]:
            for mtype in ["instruct", "base"]:
                sub = df_valid[(df_valid["family"] == family) & (df_valid["mtype"] == mtype)]
                if sub.empty:
                    continue
                ax.scatter(sub[xcol], sub[ycol],
                          c=colors[family], marker=markers[mtype],
                          s=120, edgecolor="white", linewidth=1.5,
                          alpha=0.85 if mtype == "instruct" else 0.55,
                          label=f"{family.capitalize()} {mtype}")

        # Add labels for each point
        for _, row in df_valid.iterrows():
            ax.annotate(f"{row['size']}B", (row[xcol], row[ycol]),
                       fontsize=7, alpha=0.7,
                       xytext=(5, 5), textcoords="offset points")

        # Correlation line
        if len(df_valid) >= 3:
            xv = df_valid[xcol].values
            yv = df_valid[ycol].values
            r, p = pearsonr(xv, yv)
            sl, ic = np.polyfit(xv, yv, 1)
            xx = np.linspace(xv.min() - 0.1, xv.max() + 0.1, 100)
            ax.plot(xx, sl * xx + ic, "k--", lw=1, alpha=0.5,
                   label=f"r = {r:.3f}, p = {p:.4f}")

        if ref_y is not None:
            ax.axhline(ref_y, ls=":", color="gray", alpha=0.5)
        ax.axvline(0, ls=":", color="gray", alpha=0.3)
        ax.axhline(0, ls=":", color="gray", alpha=0.3)

        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
        ax.legend(fontsize=8, loc="best")
        ax.grid(alpha=0.3)

    plt.suptitle("Representational Structure vs Behavioral Amplification\n(14 models: Llama & Qwen, base & instruct)", fontsize=13)
    plt.tight_layout()
    out_plot = MECH_ROOT / "structure_vs_behavior.png"
    plt.savefig(out_plot, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_plot}")

    # Print key correlations
    print("\n" + "="*60)
    print("  KEY CORRELATIONS")
    print("="*60)
    for xcol, ycol, xlab, ylab, _ in pairs:
        dv = df.dropna(subset=[xcol, ycol])
        if len(dv) >= 3:
            r, p = pearsonr(dv[xcol], dv[ycol])
            print(f"  {xlab} ↔ {ylab}: r = {r:.3f} (p = {p:.4f}, N = {len(dv)})")


if __name__ == "__main__":
    main()
