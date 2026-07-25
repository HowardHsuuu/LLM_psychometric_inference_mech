#!/usr/bin/env python3
"""
Split-half reliability and attenuation correction for contrastive directions.

The key question: is slope < 1 real attenuation or noise artifact?

Method:
  1. For each subscale direction, split the 10 replications into two halves
  2. Compute direction from each half independently
  3. Cosine similarity between the two half-directions = split-half reliability
  4. Spearman-Brown correction → full reliability estimate
  5. Disattenuate the cosine similarity matrix using these reliabilities
  6. Re-run geometry analysis on corrected matrix

If corrected slope ≈ 1 → attenuation was noise, model faithfully encodes human structure
If corrected slope still < 1 → real attenuation
If corrected slope > 1 → amplification starts at representation level

Usage:
    python -m psychometric_inference.mechanisms.reliability
    python -m psychometric_inference.mechanisms.reliability --layer 16
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

from .config import (
    PROJECT_ROOT, ALL_SUBSCALES, SCALE_NAMES,
    TARGET_LAYERS, DEFAULT_LAYER, RESULTS_DIR,
)
from .geometry import (
    compute_cosine_similarity_matrix,
    compute_human_correlation_matrix,
    mantel_test,
    load_directions,
)


OUTPUT_DIR = RESULTS_DIR / "reliability"


def compute_split_half_reliability(
    raw_dir: Path,
    targets: list,
    layer: int,
) -> dict:
    """Compute split-half reliability for each direction.
    
    Splits the N replications × M probe questions activations into
    odd/even halves, computes direction from each half, and measures
    cosine similarity between the two directions.
    
    Returns:
        Dict mapping target_name -> {split_half_r, spearman_brown_r}
    """
    results = {}

    for target in targets:
        raw_path = raw_dir / f"{target}_raw.npz"
        if not raw_path.exists():
            continue

        data = np.load(raw_path)
        h_key = f"high_L{layer}"
        l_key = f"low_L{layer}"

        if h_key not in data or l_key not in data:
            continue

        high_acts = data[h_key]  # (n_rep * n_questions, hidden_dim)
        low_acts = data[l_key]

        n = len(high_acts)
        if n < 4:
            continue

        # Split into odd/even indices
        odd_idx = np.arange(0, n, 2)
        even_idx = np.arange(1, n, 2)

        # Compute direction from each half
        dir_odd = np.mean(high_acts[odd_idx], axis=0) - np.mean(low_acts[odd_idx], axis=0)
        dir_even = np.mean(high_acts[even_idx], axis=0) - np.mean(low_acts[even_idx], axis=0)

        # Normalize
        norm_odd = np.linalg.norm(dir_odd)
        norm_even = np.linalg.norm(dir_even)
        if norm_odd == 0 or norm_even == 0:
            continue

        dir_odd /= norm_odd
        dir_even /= norm_even

        # Split-half reliability = cosine similarity between halves
        split_half_r = np.dot(dir_odd, dir_even)

        # Spearman-Brown correction: full reliability from split-half
        # r_full = 2 * r_half / (1 + r_half)
        if split_half_r > -1:
            spearman_brown = 2 * split_half_r / (1 + split_half_r)
        else:
            spearman_brown = 0.0

        results[target] = {
            "split_half_r": float(split_half_r),
            "spearman_brown_r": float(max(spearman_brown, 0.01)),  # floor at 0.01 to avoid division issues
            "n_samples": n,
        }

    return results


def disattenuate_cosine_matrix(
    cos_sim: pd.DataFrame,
    reliabilities: dict,
    labels: list,
) -> pd.DataFrame:
    """Correct cosine similarity matrix for attenuation due to measurement error.
    
    Disattenuation formula: r_corrected = r_observed / sqrt(rel_i * rel_j)
    Same as classical test theory correction for attenuation.
    """
    n = len(labels)
    corrected = cos_sim.copy()

    for i, li in enumerate(labels):
        for j, lj in enumerate(labels):
            if i == j:
                continue
            rel_i = reliabilities.get(li, {}).get("spearman_brown_r", 1.0)
            rel_j = reliabilities.get(lj, {}).get("spearman_brown_r", 1.0)

            observed = cos_sim.loc[li, lj]
            denom = np.sqrt(rel_i * rel_j)
            if denom > 0:
                corrected.loc[li, lj] = observed / denom
            else:
                corrected.loc[li, lj] = np.nan

    # Clip to [-1, 1]
    corrected = corrected.clip(-1, 1)
    return corrected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=DEFAULT_LAYER)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    layer = args.layer

    raw_dir = RESULTS_DIR / "directions" / "raw_activations"
    if not raw_dir.exists():
        print("Raw activations not found. Run directions.py first.")
        return

    # ── Step 1: Compute split-half reliabilities ──

    print(f"{'='*60}")
    print(f"  SPLIT-HALF RELIABILITY (Layer {layer})")
    print(f"{'='*60}")

    rel_sub = compute_split_half_reliability(raw_dir, ALL_SUBSCALES, layer)
    rel_scale = compute_split_half_reliability(raw_dir, SCALE_NAMES, layer)

    print(f"\n  Subscale direction reliabilities:")
    print(f"  {'Subscale':<35} {'Split-half':>10} {'Spearman-Brown':>15}")
    print(f"  {'-'*62}")
    for sub in ALL_SUBSCALES:
        if sub in rel_sub:
            r = rel_sub[sub]
            print(f"  {sub:<35} {r['split_half_r']:>10.3f} {r['spearman_brown_r']:>15.3f}")
        else:
            print(f"  {sub:<35} {'N/A':>10} {'N/A':>15}")

    if rel_sub:
        vals = [r["spearman_brown_r"] for r in rel_sub.values()]
        print(f"\n  Mean Spearman-Brown: {np.mean(vals):.3f}")
        print(f"  Min:                 {np.min(vals):.3f}")
        print(f"  Max:                 {np.max(vals):.3f}")

    print(f"\n  Scale direction reliabilities:")
    for scale in SCALE_NAMES:
        if scale in rel_scale:
            r = rel_scale[scale]
            print(f"  {scale:<35} {r['split_half_r']:>10.3f} {r['spearman_brown_r']:>15.3f}")

    # ── Step 2: Disattenuate cosine similarity matrices ──

    print(f"\n{'='*60}")
    print(f"  ATTENUATION CORRECTION (Layer {layer})")
    print(f"{'='*60}")

    human_sub_corr = compute_human_correlation_matrix("subscale")
    human_scale_corr = compute_human_correlation_matrix("scale")

    # Subscale level
    sub_dir_path = RESULTS_DIR / "directions" / "subscale_directions.npz"
    if sub_dir_path.exists() and rel_sub:
        sub_dirs = load_directions(str(sub_dir_path), ALL_SUBSCALES, layer)
        available = [s for s in ALL_SUBSCALES if s in sub_dirs]

        cos_sim_raw = compute_cosine_similarity_matrix(sub_dirs, available)
        cos_sim_corrected = disattenuate_cosine_matrix(cos_sim_raw, rel_sub, available)

        # Compare raw vs corrected
        common = [s for s in available if s in human_sub_corr.index]

        for label, cos_mat in [("raw", cos_sim_raw), ("corrected", cos_sim_corrected)]:
            cs = cos_mat.loc[common, common]
            hm = human_sub_corr.loc[common, common]
            n = len(common)
            idx = np.triu_indices(n, k=1)
            cv = cs.values[idx]
            hv = hm.values[idx]
            valid = ~(np.isnan(cv) | np.isnan(hv))
            cv, hv = cv[valid], hv[valid]

            if len(hv) >= 3:
                r, p = pearsonr(hv, cv)
                slope, intercept = np.polyfit(hv, cv, 1)
                print(f"\n  Subscale {label}:")
                print(f"    Pearson r = {r:.4f}")
                print(f"    Slope     = {slope:.4f}")
                print(f"    Intercept = {intercept:.4f}")

        # Plot: raw vs corrected scatter
        cs_raw = cos_sim_raw.loc[common, common]
        cs_cor = cos_sim_corrected.loc[common, common]
        hm = human_sub_corr.loc[common, common]
        n = len(common)
        idx = np.triu_indices(n, k=1)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        for ax, cs, title in [(ax1, cs_raw, "Raw"), (ax2, cs_cor, "Corrected")]:
            cv = cs.values[idx]
            hv = hm.values[idx]
            valid = ~(np.isnan(cv) | np.isnan(hv))
            cv_v, hv_v = cv[valid], hv[valid]

            ax.scatter(hv_v, cv_v, alpha=0.6, s=30, edgecolors="white", linewidths=0.5)
            if len(hv_v) >= 3:
                slope, intercept = np.polyfit(hv_v, cv_v, 1)
                r_val, _ = pearsonr(hv_v, cv_v)
                xx = np.linspace(hv_v.min() - 0.05, hv_v.max() + 0.05, 100)
                ax.plot(xx, slope * xx + intercept, "k--", lw=1, alpha=0.7,
                        label=f"slope={slope:.2f}, r={r_val:.3f}")

            lim = max(abs(hv_v).max(), abs(cv_v).max()) * 1.3
            ax.plot([-lim, lim], [-lim, lim], "gray", ls=":", lw=0.8, alpha=0.5, label="y=x")
            ax.set_xlabel("Human behavioral correlation")
            ax.set_ylabel(f"Cosine similarity ({title.lower()})")
            ax.set_title(f"{title} (slope={slope:.2f})")
            ax.legend(fontsize=8)
            ax.axhline(y=0, color="gray", lw=0.3)
            ax.axvline(x=0, color="gray", lw=0.3)

        plt.suptitle(f"Attenuation Correction (Layer {layer}, {len(common)} subscales)", fontsize=13)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"attenuation_correction_L{layer}.png", dpi=200, bbox_inches="tight")
        plt.close()

        cos_sim_corrected.to_csv(OUTPUT_DIR / f"cosine_sim_corrected_L{layer}.csv")

    # Save reliability results
    rel_df = pd.DataFrame([
        {"target": k, "type": "subscale", **v} for k, v in rel_sub.items()
    ] + [
        {"target": k, "type": "scale", **v} for k, v in rel_scale.items()
    ])
    rel_df.to_csv(OUTPUT_DIR / f"reliability_L{layer}.csv", index=False)

    print(f"\nAll outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()