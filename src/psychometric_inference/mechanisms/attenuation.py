#!/usr/bin/env python3
"""
Filtered attenuation correction: only use directions with Spearman-Brown > threshold.

Usage:
    python -m psychometric_inference.mechanisms.attenuation --layer 16
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

from .config import PROJECT_ROOT, ALL_SUBSCALES, DEFAULT_LAYER, RESULTS_DIR
from .geometry import (
    load_directions, compute_cosine_similarity_matrix,
    compute_human_correlation_matrix, mantel_test,
)
from .reliability import compute_split_half_reliability, disattenuate_cosine_matrix


OUTPUT_DIR = RESULTS_DIR / "reliability"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=DEFAULT_LAYER)
    parser.add_argument("--threshold", type=float, default=0.3,
                        help="Minimum Spearman-Brown reliability to include")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    layer = args.layer
    threshold = args.threshold

    raw_dir = RESULTS_DIR / "directions" / "raw_activations"
    rel = compute_split_half_reliability(raw_dir, ALL_SUBSCALES, layer)

    # Filter by reliability
    reliable = [s for s in ALL_SUBSCALES
                if s in rel and rel[s]["spearman_brown_r"] >= threshold]
    unreliable = [s for s in ALL_SUBSCALES
                  if s in rel and rel[s]["spearman_brown_r"] < threshold]

    print(f"Threshold: Spearman-Brown >= {threshold}")
    print(f"Reliable ({len(reliable)}):")
    for s in reliable:
        print(f"  {s:<35} SB = {rel[s]['spearman_brown_r']:.3f}")
    print(f"Unreliable ({len(unreliable)}):")
    for s in unreliable:
        print(f"  {s:<35} SB = {rel[s]['spearman_brown_r']:.3f}")

    # Load directions and human correlation
    sub_dir_path = RESULTS_DIR / "directions" / "subscale_directions.npz"
    sub_dirs = load_directions(str(sub_dir_path), reliable, layer)
    human_corr = compute_human_correlation_matrix("subscale")

    common = [s for s in reliable if s in sub_dirs and s in human_corr.index]

    # Raw cosine similarity
    cos_raw = compute_cosine_similarity_matrix(sub_dirs, common)
    # Corrected
    cos_corrected = disattenuate_cosine_matrix(cos_raw, rel, common)

    hum = human_corr.loc[common, common]
    n = len(common)
    idx = np.triu_indices(n, k=1)

    print(f"\n{'='*60}")
    print(f"  FILTERED ATTENUATION CORRECTION (Layer {layer})")
    print(f"  {len(common)} reliable directions, {len(idx[0])} pairs")
    print(f"{'='*60}")

    for label, cos in [("Raw", cos_raw), ("Corrected", cos_corrected)]:
        cv = cos.loc[common, common].values[idx]
        hv = hum.values[idx]
        valid = ~(np.isnan(cv) | np.isnan(hv))
        cv_v, hv_v = cv[valid], hv[valid]

        if len(hv_v) < 3:
            continue

        r, p = pearsonr(hv_v, cv_v)
        slope, intercept = np.polyfit(hv_v, cv_v, 1)
        print(f"\n  {label}:")
        print(f"    Pearson r  = {r:.4f} (p = {p:.6f})")
        print(f"    Slope      = {slope:.4f}")
        print(f"    Intercept  = {intercept:.4f}")
        print(f"    N pairs    = {len(hv_v)}")

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for ax, cos, title in [(ax1, cos_raw, "Raw"), (ax2, cos_corrected, "Corrected")]:
        cv = cos.loc[common, common].values[idx]
        hv = hum.values[idx]
        valid = ~(np.isnan(cv) | np.isnan(hv))
        cv_v, hv_v = cv[valid], hv[valid]

        ax.scatter(hv_v, cv_v, alpha=0.6, s=40, edgecolors="white", linewidths=0.5)
        if len(hv_v) >= 3:
            slope, intercept = np.polyfit(hv_v, cv_v, 1)
            r_val, _ = pearsonr(hv_v, cv_v)
            xx = np.linspace(min(hv_v) - 0.05, max(hv_v) + 0.05, 100)
            ax.plot(xx, slope * xx + intercept, "k--", lw=1, alpha=0.7,
                    label=f"slope={slope:.2f}, r={r_val:.3f}")

        lim_x = max(abs(hv_v).max(), 0.8) * 1.1
        lim_y = max(abs(cv_v).max(), 0.8) * 1.1
        ax.plot([-lim_x, lim_x], [-lim_x, lim_x], "gray", ls=":", lw=0.8, alpha=0.5, label="y=x")
        ax.set_xlabel("Human behavioral correlation")
        ax.set_ylabel(f"Cosine similarity ({title.lower()})")
        ax.set_title(f"{title} (slope={slope:.2f})")
        ax.legend(fontsize=8)
        ax.axhline(y=0, color="gray", lw=0.3)
        ax.axvline(x=0, color="gray", lw=0.3)

    plt.suptitle(
        f"Filtered Attenuation Correction (Layer {layer}, {len(common)} reliable directions, SB≥{threshold})",
        fontsize=12,
    )
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"filtered_attenuation_L{layer}_SB{threshold}.png",
                dpi=200, bbox_inches="tight")
    plt.close()

    print(f"\nSaved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()