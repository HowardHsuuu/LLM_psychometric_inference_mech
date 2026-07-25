#!/usr/bin/env python3
"""
Follow-up analyses based on initial results.

1. Filtered A2: Re-run geometry with only validated directions (on-target |r| > 0.2)
2. Within-scale vs between-scale decomposition
3. A4 cross-construct pattern vs human correlation matrix

Usage:
    python -m psychometric_inference.mechanisms.robustness
    python -m psychometric_inference.mechanisms.robustness --layer 16
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

from .config import (
    PROJECT_ROOT, ALL_SUBSCALES, SCALE_NAMES, SUB_TO_SCALE,
    TARGET_LAYERS, DEFAULT_LAYER, RESULTS_DIR,
)
from .geometry import (
    load_directions,
    compute_cosine_similarity_matrix,
    compute_human_correlation_matrix,
    mantel_test,
)


OUTPUT_DIR = RESULTS_DIR / "followup"


# ── Analysis 1: Filtered geometry ──

def filtered_geometry(layer: int, output_dir: Path, threshold: float = 0.2):
    """Re-run A2 using only directions with on-target |r| > threshold.
    
    This removes noisy directions (mostly BigFive subscales with only 2 items)
    that systematically deflate the slope.
    """
    print(f"\n{'='*60}")
    print(f"  FILTERED GEOMETRY (Layer {layer}, |r| > {threshold})")
    print(f"{'='*60}")

    # Load A4 validation results
    val_path = RESULTS_DIR / "geometry" / f"direction_validation_L{layer}.csv"
    if not val_path.exists():
        print("  A4 validation results not found, skipping")
        return None

    val_df = pd.read_csv(val_path, index_col=0)

    # Find on-target correlations (diagonal)
    on_target = {}
    for sub in ALL_SUBSCALES:
        if sub in val_df.index and sub in val_df.columns:
            on_target[sub] = val_df.loc[sub, sub]

    # Filter by threshold
    valid_subs = [s for s, r in on_target.items() if abs(r) >= threshold]
    rejected = [s for s, r in on_target.items() if abs(r) < threshold]

    print(f"  On-target correlations:")
    for s in ALL_SUBSCALES:
        if s in on_target:
            r = on_target[s]
            status = "✓" if abs(r) >= threshold else "✗"
            print(f"    {status} {s:<35} r = {r:+.3f}")

    print(f"\n  Retained: {len(valid_subs)}/{len(on_target)}")
    print(f"  Rejected: {', '.join([s.split('_',1)[1] for s in rejected])}")

    if len(valid_subs) < 4:
        print("  Too few valid directions for meaningful analysis")
        return None

    # Load directions and compute filtered cosine similarity
    sub_dir_path = RESULTS_DIR / "directions" / "subscale_directions.npz"
    sub_dirs = load_directions(str(sub_dir_path), valid_subs, layer)
    cos_sim = compute_cosine_similarity_matrix(sub_dirs, valid_subs)

    # Load human correlation and filter
    human_corr = compute_human_correlation_matrix("subscale")
    common = [s for s in valid_subs if s in human_corr.index]
    cos_sub = cos_sim.loc[common, common]
    hum_sub = human_corr.loc[common, common]

    n = len(common)
    idx = np.triu_indices(n, k=1)
    cos_vec = cos_sub.values[idx]
    hum_vec = hum_sub.values[idx]

    valid_mask = ~(np.isnan(cos_vec) | np.isnan(hum_vec))
    cos_v = cos_vec[valid_mask]
    hum_v = hum_vec[valid_mask]

    r_mantel, p_mantel = mantel_test(cos_sub.values, hum_sub.values)
    r_pearson, p_pearson = pearsonr(hum_v, cos_v)
    slope, intercept = np.polyfit(hum_v, cos_v, 1)

    print(f"\n  Filtered results ({len(common)} constructs, {len(hum_v)} pairs):")
    print(f"    Mantel r  = {r_mantel:.4f} (p = {p_mantel:.4f})")
    print(f"    Pearson r = {r_pearson:.4f} (p = {p_pearson:.6f})")
    print(f"    Slope     = {slope:.4f}")
    print(f"    Intercept = {intercept:.4f}")

    # Compare with unfiltered
    unfilt_path = RESULTS_DIR / "geometry" / "geometry_results.csv"
    if unfilt_path.exists():
        unfilt = pd.read_csv(unfilt_path)
        unfilt_sub = unfilt[(unfilt["level"] == "subscale") & (unfilt["layer"] == layer)]
        if not unfilt_sub.empty:
            uf = unfilt_sub.iloc[0]
            print(f"\n  Comparison with unfiltered:")
            print(f"    {'Metric':<12} {'Unfiltered':>12} {'Filtered':>12} {'Change':>12}")
            print(f"    {'Mantel r':<12} {uf['mantel_r']:>12.4f} {r_mantel:>12.4f} {r_mantel - uf['mantel_r']:>+12.4f}")
            print(f"    {'Slope':<12} {uf['slope']:>12.4f} {slope:>12.4f} {slope - uf['slope']:>+12.4f}")
            print(f"    {'N pairs':<12} {int(uf['n_pairs']):>12d} {len(hum_v):>12d}")

    # Plot
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(hum_v, cos_v, alpha=0.6, s=40, edgecolors="white", linewidths=0.5)

    xx = np.linspace(hum_v.min() - 0.05, hum_v.max() + 0.05, 100)
    ax.plot(xx, slope * xx + intercept, "k--", lw=1, alpha=0.7,
            label=f"slope={slope:.2f}, r={r_pearson:.3f}")
    lim = max(abs(hum_v).max(), abs(cos_v).max()) * 1.2
    ax.plot([-lim, lim], [-lim, lim], "gray", ls=":", lw=0.8, alpha=0.5, label="y = x")

    ax.set_xlabel("Human behavioral correlation")
    ax.set_ylabel("Activation cosine similarity")
    ax.set_title(
        f"Filtered Geometry (Layer {layer}, {len(common)} validated constructs)\n"
        f"Mantel r={r_mantel:.3f}, slope={slope:.2f}"
    )
    ax.legend(fontsize=8)
    ax.axhline(y=0, color="gray", lw=0.3)
    ax.axvline(x=0, color="gray", lw=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / f"filtered_scatter_L{layer}.png", dpi=200, bbox_inches="tight")
    plt.close()

    return {
        "n_valid": len(common),
        "n_pairs": len(hum_v),
        "mantel_r": r_mantel,
        "mantel_p": p_mantel,
        "slope": slope,
        "intercept": intercept,
    }


# ── Analysis 2: Within-scale vs between-scale decomposition ──

def within_between_decomposition(layer: int, output_dir: Path):
    """Decompose A2 into within-scale and between-scale pairs.
    
    Within-scale: pairs of subscales belonging to the same parent scale
      (e.g., IRI_PT vs IRI_Fantasy)
    Between-scale: pairs from different scales
      (e.g., IRI_PT vs PANAS_NA)
    """
    print(f"\n{'='*60}")
    print(f"  WITHIN vs BETWEEN SCALE (Layer {layer})")
    print(f"{'='*60}")

    sub_dir_path = RESULTS_DIR / "directions" / "subscale_directions.npz"
    if not sub_dir_path.exists():
        print("  Directions not found, skipping")
        return None

    sub_dirs = load_directions(str(sub_dir_path), ALL_SUBSCALES, layer)
    available = [s for s in ALL_SUBSCALES if s in sub_dirs]

    cos_sim = compute_cosine_similarity_matrix(sub_dirs, available)
    human_corr = compute_human_correlation_matrix("subscale")
    common = [s for s in available if s in human_corr.index]

    cos_sub = cos_sim.loc[common, common]
    hum_sub = human_corr.loc[common, common]

    within_cos, within_hum = [], []
    between_cos, between_hum = [], []

    for i, si in enumerate(common):
        for j, sj in enumerate(common):
            if j <= i:
                continue
            c = cos_sub.loc[si, sj]
            h = hum_sub.loc[si, sj]
            if np.isnan(c) or np.isnan(h):
                continue

            scale_i = SUB_TO_SCALE.get(si, si)
            scale_j = SUB_TO_SCALE.get(sj, sj)

            if scale_i == scale_j:
                within_cos.append(c)
                within_hum.append(h)
            else:
                between_cos.append(c)
                between_hum.append(h)

    within_cos, within_hum = np.array(within_cos), np.array(within_hum)
    between_cos, between_hum = np.array(between_cos), np.array(between_hum)

    print(f"  Within-scale pairs:  {len(within_hum)}")
    print(f"  Between-scale pairs: {len(between_hum)}")

    results = {}

    for label, cv, hv in [("within", within_cos, within_hum),
                           ("between", between_cos, between_hum),
                           ("all", np.concatenate([within_cos, between_cos]),
                                  np.concatenate([within_hum, between_hum]))]:
        if len(hv) < 3:
            continue
        r, p = pearsonr(hv, cv)
        slope, intercept = np.polyfit(hv, cv, 1)
        print(f"\n  {label:>8}: r={r:.3f}, slope={slope:.3f}, n={len(hv)}")
        results[label] = {"r": r, "slope": slope, "n": len(hv)}

    # Plot
    fig, ax = plt.subplots(figsize=(7, 6))

    if len(within_hum) > 0:
        ax.scatter(within_hum, within_cos, alpha=0.7, s=50, c="#e41a1c",
                   edgecolors="white", linewidths=0.5, label=f"Within-scale (n={len(within_hum)})", zorder=5)
    ax.scatter(between_hum, between_cos, alpha=0.4, s=30, c="#377eb8",
               edgecolors="white", linewidths=0.5, label=f"Between-scale (n={len(between_hum)})")

    # Regression lines
    all_h = np.concatenate([within_hum, between_hum])
    all_c = np.concatenate([within_cos, between_cos])
    xx = np.linspace(all_h.min() - 0.05, all_h.max() + 0.05, 100)

    if len(between_hum) >= 3:
        sl_b, int_b = np.polyfit(between_hum, between_cos, 1)
        ax.plot(xx, sl_b * xx + int_b, "--", color="#377eb8", lw=1, alpha=0.6)

    if len(within_hum) >= 3:
        sl_w, int_w = np.polyfit(within_hum, within_cos, 1)
        ax.plot(xx, sl_w * xx + int_w, "--", color="#e41a1c", lw=1, alpha=0.6)

    lim = max(abs(all_h).max(), abs(all_c).max()) * 1.2
    ax.plot([-lim, lim], [-lim, lim], "gray", ls=":", lw=0.8, alpha=0.5, label="y = x")

    ax.set_xlabel("Human behavioral correlation")
    ax.set_ylabel("Activation cosine similarity")
    ax.set_title(f"Within vs Between Scale (Layer {layer})")
    ax.legend(fontsize=8)
    ax.axhline(y=0, color="gray", lw=0.3)
    ax.axvline(x=0, color="gray", lw=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / f"within_between_L{layer}.png", dpi=200, bbox_inches="tight")
    plt.close()

    return results


# ── Analysis 3: A4 cross-construct pattern as alternative geometry measure ──

def a4_as_geometry(layer: int, output_dir: Path):
    """Use the A4 validation heatmap as an alternative way to measure
    whether the model's internal structure matches human psychometric structure.
    
    The A4 heatmap (directions × behavioral subscales) can be read as:
    "when I project 272 subjects onto the Neuroticism direction, how well
    does that predict their Negative Affect score?"
    
    This 16×16 matrix is a projection-behavior correlation matrix.
    Its off-diagonal structure should mirror the human subscale correlation matrix
    if the directions are psychometrically valid.
    
    Key advantage: this doesn't depend on cosine similarity between directions
    (which is affected by direction noise). Instead it uses the directions'
    predictive validity, which is more robust.
    """
    print(f"\n{'='*60}")
    print(f"  A4 CROSS-CONSTRUCT AS GEOMETRY (Layer {layer})")
    print(f"{'='*60}")

    val_path = RESULTS_DIR / "geometry" / f"direction_validation_L{layer}.csv"
    if not val_path.exists():
        print("  A4 validation results not found, skipping")
        return None

    val_df = pd.read_csv(val_path, index_col=0)
    human_corr = compute_human_correlation_matrix("subscale")

    # The A4 matrix is (directions × behavioral subscales)
    # To compare with human correlation (subscales × subscales),
    # we use the A4 matrix as a proxy: if direction_i predicts behavior_j
    # with correlation r_ij, this is analogous to the relationship
    # between construct i and construct j.
    #
    # The A4 matrix is NOT symmetric (direction_i→behavior_j ≠ direction_j→behavior_i)
    # So we symmetrize by averaging: proxy_ij = (A4_ij + A4_ji) / 2

    common = [s for s in ALL_SUBSCALES if s in val_df.index and s in val_df.columns and s in human_corr.index]
    n = len(common)

    a4_sub = val_df.loc[common, common].values
    hum_sub = human_corr.loc[common, common].values

    # Symmetrize A4
    a4_sym = (a4_sub + a4_sub.T) / 2

    # Compare upper triangles (excluding diagonal)
    idx = np.triu_indices(n, k=1)
    a4_vec = a4_sym[idx]
    hum_vec = hum_sub[idx]

    valid = ~(np.isnan(a4_vec) | np.isnan(hum_vec))
    a4_v = a4_vec[valid]
    hum_v = hum_vec[valid]

    r_mantel, p_mantel = mantel_test(a4_sym, hum_sub)
    r_pearson, p_pearson = pearsonr(hum_v, a4_v)
    slope, intercept = np.polyfit(hum_v, a4_v, 1)

    print(f"  N constructs: {n}")
    print(f"  N pairs:      {len(hum_v)}")
    print(f"  Mantel r:     {r_mantel:.4f} (p = {p_mantel:.4f})")
    print(f"  Pearson r:    {r_pearson:.4f} (p = {p_pearson:.6f})")
    print(f"  Slope:        {slope:.4f}")
    print(f"  Intercept:    {intercept:.4f}")

    # Compare with cosine similarity approach
    cos_path = RESULTS_DIR / "geometry" / "geometry_results.csv"
    if cos_path.exists():
        cos_df = pd.read_csv(cos_path)
        cos_sub = cos_df[(cos_df["level"] == "subscale") & (cos_df["layer"] == layer)]
        if not cos_sub.empty:
            cs = cos_sub.iloc[0]
            print(f"\n  Comparison of geometry measures:")
            print(f"    {'Method':<25} {'Mantel r':>10} {'Slope':>10}")
            print(f"    {'Cosine similarity':<25} {cs['mantel_r']:>10.4f} {cs['slope']:>10.4f}")
            print(f"    {'A4 cross-construct':<25} {r_mantel:>10.4f} {slope:>10.4f}")

    # Plot
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(hum_v, a4_v, alpha=0.6, s=30, edgecolors="white", linewidths=0.5)

    xx = np.linspace(hum_v.min() - 0.05, hum_v.max() + 0.05, 100)
    ax.plot(xx, slope * xx + intercept, "k--", lw=1, alpha=0.7,
            label=f"slope={slope:.2f}, r={r_pearson:.3f}")
    lim = max(abs(hum_v).max(), abs(a4_v).max()) * 1.2
    ax.plot([-lim, lim], [-lim, lim], "gray", ls=":", lw=0.8, alpha=0.5, label="y = x")

    ax.set_xlabel("Human behavioral correlation")
    ax.set_ylabel("A4 cross-construct correlation (symmetrized)")
    ax.set_title(
        f"A4 as Geometry Measure (Layer {layer})\n"
        f"Mantel r={r_mantel:.3f}, slope={slope:.2f}"
    )
    ax.legend(fontsize=8)
    ax.axhline(y=0, color="gray", lw=0.3)
    ax.axvline(x=0, color="gray", lw=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / f"a4_geometry_scatter_L{layer}.png", dpi=200, bbox_inches="tight")
    plt.close()

    # Also plot symmetrized A4 as heatmap
    short_labels = [s.split("_", 1)[1] if "_" in s else s for s in common]
    fig, ax = plt.subplots(figsize=(9, 8))
    sns.heatmap(
        a4_sym, cmap="RdBu_r", center=0, vmin=-0.5, vmax=0.5,
        xticklabels=short_labels, yticklabels=short_labels,
        annot=True, fmt=".2f", annot_kws={"size": 6}, ax=ax,
        cbar_kws={"shrink": 0.6, "label": "Symmetrized projection-behavior r"},
    )
    ax.set_title(f"Symmetrized A4 Cross-Construct Matrix (Layer {layer})", fontsize=11)
    ax.tick_params(axis="both", labelsize=7)
    plt.tight_layout()
    plt.savefig(output_dir / f"a4_geometry_heatmap_L{layer}.png", dpi=200, bbox_inches="tight")
    plt.close()

    return {
        "n_constructs": n,
        "n_pairs": len(hum_v),
        "mantel_r": r_mantel,
        "mantel_p": p_mantel,
        "slope": slope,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=DEFAULT_LAYER)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    layer = args.layer

    print(f"Layer: {layer}")
    print(f"Output: {OUTPUT_DIR}")

    # 1. Filtered geometry
    filt_result = filtered_geometry(layer, OUTPUT_DIR)

    # 2. Within vs between
    wb_result = within_between_decomposition(layer, OUTPUT_DIR)

    # 3. A4 as geometry
    a4_result = a4_as_geometry(layer, OUTPUT_DIR)

    # Save summary
    summary = {"layer": layer}
    if filt_result:
        summary["filtered"] = filt_result
    if wb_result:
        summary["within_between"] = wb_result
    if a4_result:
        summary["a4_geometry"] = a4_result

    import json
    with open(OUTPUT_DIR / f"followup_summary_L{layer}.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nAll outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()