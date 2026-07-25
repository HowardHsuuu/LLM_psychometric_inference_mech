#!/usr/bin/env python3
"""
Analyze the geometry of extracted psychometric directions.

Core analyses:
  A2. Cosine similarity matrix of 16 subscale directions vs human subscale correlation matrix
  A3. Same at scale level (7 directions)
  A4. Direction quality validation (projection correlates with behavioral scores)

Usage:
    python -m psychometric_inference.mechanisms.geometry
    python -m psychometric_inference.mechanisms.geometry --layer 16
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr

from .config import (
    PROJECT_ROOT, ALL_SUBSCALES, SCALE_NAMES, SUB_TO_SCALE,
    TARGET_LAYERS, DEFAULT_LAYER, RESULTS_DIR, HUMAN_DIRS,
)

from psychometric_inference.scoring import compute_subscale_scores_from_csvs

OUTPUT_DIR = RESULTS_DIR / "geometry"


def load_directions(npz_path: str, targets: List[str], layer: int) -> Dict[str, np.ndarray]:
    """Load direction vectors from npz file for a specific layer."""
    data = np.load(npz_path, allow_pickle=True)
    directions = {}
    for t in targets:
        key = f"{t}_L{layer}"
        if key in data:
            directions[t] = data[key]
    return directions


def compute_cosine_similarity_matrix(
    directions: Dict[str, np.ndarray],
    labels: List[str],
) -> pd.DataFrame:
    """Compute pairwise cosine similarity between direction vectors."""
    n = len(labels)
    sim_matrix = np.zeros((n, n))

    for i, li in enumerate(labels):
        for j, lj in enumerate(labels):
            if li in directions and lj in directions:
                vi, vj = directions[li], directions[lj]
                cos = np.dot(vi, vj) / (np.linalg.norm(vi) * np.linalg.norm(vj) + 1e-10)
                sim_matrix[i, j] = cos
            elif i == j:
                sim_matrix[i, j] = 1.0

    return pd.DataFrame(sim_matrix, index=labels, columns=labels)


def compute_human_correlation_matrix(level: str = "subscale") -> pd.DataFrame:
    """Compute human behavioral correlation matrix from real data.
    
    Args:
        level: "subscale" (16x16) or "scale" (7x7)
    """
    # Compute subscale scores from human CSVs
    human_scores = compute_subscale_scores_from_csvs(
        [str(d) for d in HUMAN_DIRS],
        un_reverse=True,
    )

    if level == "subscale":
        labels = [s for s in ALL_SUBSCALES if s in human_scores.columns]
        corr = human_scores[labels].corr()
    else:
        # Aggregate to scale level by averaging subscales within each scale
        scale_scores = {}
        for scale in SCALE_NAMES:
            sub_cols = [s for s in human_scores.columns if s.startswith(scale + "_")]
            if sub_cols:
                scale_scores[scale] = human_scores[sub_cols].mean(axis=1)
        scale_df = pd.DataFrame(scale_scores)
        labels = [s for s in SCALE_NAMES if s in scale_df.columns]
        corr = scale_df[labels].corr()

    return corr


def mantel_test(
    matrix_a: np.ndarray,
    matrix_b: np.ndarray,
    n_permutations: int = 10000,
    seed: int = 42,
) -> Tuple[float, float]:
    """Mantel test: correlation between two distance/similarity matrices.
    
    Uses upper triangle only. Tests via row/column permutation.
    
    Returns:
        (observed_r, p_value)
    """
    n = matrix_a.shape[0]
    assert matrix_a.shape == matrix_b.shape

    # Extract upper triangle
    idx = np.triu_indices(n, k=1)
    a_vec = matrix_a[idx]
    b_vec = matrix_b[idx]

    # Remove NaN pairs
    valid = ~(np.isnan(a_vec) | np.isnan(b_vec))
    a_vec = a_vec[valid]
    b_vec = b_vec[valid]

    if len(a_vec) < 3:
        return np.nan, np.nan

    observed_r = np.corrcoef(a_vec, b_vec)[0, 1]

    # Permutation test
    rng = np.random.default_rng(seed)
    n_greater = 0
    for _ in range(n_permutations):
        perm = rng.permutation(n)
        a_perm = matrix_a[np.ix_(perm, perm)]
        a_perm_vec = a_perm[idx][valid]
        r_perm = np.corrcoef(a_perm_vec, b_vec)[0, 1]
        if r_perm >= observed_r:
            n_greater += 1

    p_value = (n_greater + 1) / (n_permutations + 1)
    return observed_r, p_value


def analyze_and_plot(
    cos_sim: pd.DataFrame,
    human_corr: pd.DataFrame,
    level: str,
    layer: int,
    output_dir: Path,
):
    """Run full comparison analysis and generate plots."""
    labels = [l for l in cos_sim.index if l in human_corr.index]
    cos_sub = cos_sim.loc[labels, labels]
    hum_sub = human_corr.loc[labels, labels]

    n = len(labels)
    idx = np.triu_indices(n, k=1)
    cos_vec = cos_sub.values[idx]
    hum_vec = hum_sub.values[idx]

    valid = ~(np.isnan(cos_vec) | np.isnan(hum_vec))
    cos_v = cos_vec[valid]
    hum_v = hum_vec[valid]

    # Mantel test
    r_mantel, p_mantel = mantel_test(cos_sub.values, hum_sub.values)

    # Pearson on upper triangle
    r_pearson, p_pearson = pearsonr(hum_v, cos_v)

    # Linear regression (slope tells us about amplification)
    slope, intercept = np.polyfit(hum_v, cos_v, 1)

    print(f"\n{'='*60}")
    print(f"  {level.upper()}-LEVEL GEOMETRY ANALYSIS (Layer {layer})")
    print(f"{'='*60}")
    print(f"  N constructs:    {n}")
    print(f"  N unique pairs:  {len(hum_v)}")
    print(f"  Mantel r:        {r_mantel:.4f}  (p = {p_mantel:.4f})")
    print(f"  Pearson r:       {r_pearson:.4f}  (p = {p_pearson:.6f})")
    print(f"  Slope:           {slope:.4f}")
    print(f"  Intercept:       {intercept:.4f}")
    if slope > 0:
        print(f"  Interpretation:  slope {'> 1 → amplification' if slope > 1 else '< 1 → attenuation' if slope < 1 else '≈ 1 → faithful'}")

    # ── Plot 1: Side-by-side heatmaps ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    short_labels = [l.split("_", 1)[1] if "_" in l else l for l in labels]

    sns.heatmap(hum_sub.values, cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                xticklabels=short_labels, yticklabels=short_labels,
                annot=True, fmt=".2f", annot_kws={"size": 7}, ax=ax1,
                cbar_kws={"shrink": 0.6})
    ax1.set_title("Human Behavioral Correlation", fontsize=11)
    ax1.tick_params(axis="both", labelsize=7)

    sns.heatmap(cos_sub.values, cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                xticklabels=short_labels, yticklabels=short_labels,
                annot=True, fmt=".2f", annot_kws={"size": 7}, ax=ax2,
                cbar_kws={"shrink": 0.6})
    ax2.set_title(f"Activation Cosine Similarity (Layer {layer})", fontsize=11)
    ax2.tick_params(axis="both", labelsize=7)

    plt.suptitle(f"{level.capitalize()}-level: Human Correlation vs Activation Geometry", fontsize=13)
    plt.tight_layout()
    plt.savefig(output_dir / f"{level}_heatmaps_L{layer}.png", dpi=200, bbox_inches="tight")
    plt.close()

    # ── Plot 2: Scatter plot ──
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(hum_v, cos_v, alpha=0.6, s=30, edgecolors="white", linewidths=0.5)

    # Regression line
    xx = np.linspace(hum_v.min() - 0.05, hum_v.max() + 0.05, 100)
    ax.plot(xx, slope * xx + intercept, "k--", lw=1, alpha=0.7,
            label=f"slope={slope:.2f}, r={r_pearson:.3f}")

    # Identity line
    lim = max(abs(hum_v).max(), abs(cos_v).max()) * 1.2
    ax.plot([-lim, lim], [-lim, lim], "gray", ls=":", lw=0.8, alpha=0.5, label="y = x")

    ax.set_xlabel("Human behavioral correlation")
    ax.set_ylabel("Activation cosine similarity")
    ax.set_title(
        f"{level.capitalize()}-level (Layer {layer})\n"
        f"Mantel r={r_mantel:.3f} (p={p_mantel:.4f}), slope={slope:.2f}"
    )
    ax.legend(fontsize=8)
    ax.axhline(y=0, color="gray", lw=0.3)
    ax.axvline(x=0, color="gray", lw=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / f"{level}_scatter_L{layer}.png", dpi=200, bbox_inches="tight")
    plt.close()

    # ── Plot 3: Difference heatmap ──
    diff = cos_sub.values - hum_sub.values
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(diff, cmap="RdBu_r", center=0,
                xticklabels=short_labels, yticklabels=short_labels,
                annot=True, fmt=".2f", annot_kws={"size": 7}, ax=ax,
                cbar_kws={"shrink": 0.6, "label": "Cosine sim − Human corr"})
    ax.set_title(f"Difference: Activation − Human ({level}, Layer {layer})", fontsize=11)
    ax.tick_params(axis="both", labelsize=7)
    plt.tight_layout()
    plt.savefig(output_dir / f"{level}_difference_L{layer}.png", dpi=200, bbox_inches="tight")
    plt.close()

    return {
        "level": level,
        "layer": layer,
        "n_constructs": n,
        "n_pairs": len(hum_v),
        "mantel_r": r_mantel,
        "mantel_p": p_mantel,
        "pearson_r": r_pearson,
        "pearson_p": p_pearson,
        "slope": slope,
        "intercept": intercept,
    }


def validate_directions(
    layer: int,
    output_dir: Path,
):
    """A4: Direction quality validation.
    
    For each extracted direction, project 272 subjects' activations onto it
    and correlate with the subjects' behavioral subscale scores.
    
    A good direction should correlate with its OWN subscale (on-target)
    and the pattern of cross-subscale correlations should mirror human structure.
    
    Requires: subject activations from subject_activations.py and directions from directions.py.
    """
    # Load subject activations
    act_path = RESULTS_DIR / "pca" / "subject_activations.npz"
    scores_path = RESULTS_DIR / "pca" / "human_subscale_scores.csv"
    sub_dir_path = RESULTS_DIR / "directions" / "subscale_directions.npz"

    if not all(p.exists() for p in [act_path, scores_path, sub_dir_path]):
        print("  Skipping A4 validation (need subject activations + directions)")
        print(f"    subject_activations: {'✓' if act_path.exists() else '✗'}")
        print(f"    human_subscale_scores: {'✓' if scores_path.exists() else '✗'}")
        print(f"    subscale_directions: {'✓' if sub_dir_path.exists() else '✗'}")
        return None

    act_data = np.load(act_path)
    key = f"L{layer}"
    if key not in act_data:
        print(f"  Layer {layer} not found in subject activations")
        return None

    activations = act_data[key]  # (n_subjects, hidden_dim)
    scores_df = pd.read_csv(scores_path)
    sub_dirs = load_directions(str(sub_dir_path), ALL_SUBSCALES, layer)

    if not sub_dirs:
        print("  No directions found for this layer")
        return None

    print(f"\n{'='*60}")
    print(f"  A4: DIRECTION VALIDATION (Layer {layer})")
    print(f"{'='*60}")

    available_subs = [s for s in ALL_SUBSCALES if s in sub_dirs and s in scores_df.columns]

    # For each direction, project all subjects and correlate with all subscale scores
    n_dirs = len(available_subs)
    n_subs = len([s for s in ALL_SUBSCALES if s in scores_df.columns])
    all_sub_labels = [s for s in ALL_SUBSCALES if s in scores_df.columns]

    # Build projection-vs-behavior correlation matrix
    # Rows = directions, Columns = behavioral subscales
    proj_corr = np.full((n_dirs, n_subs), np.nan)
    proj_p = np.full((n_dirs, n_subs), np.nan)

    on_target_rs = []

    for i, dir_name in enumerate(available_subs):
        direction = sub_dirs[dir_name]
        # Project all subjects onto this direction
        projections = activations @ direction  # (n_subjects,)

        for j, sub_name in enumerate(all_sub_labels):
            behavioral = scores_df[sub_name].values
            valid = ~(np.isnan(projections) | np.isnan(behavioral))
            if valid.sum() < 3:
                continue
            r, p = pearsonr(projections[valid], behavioral[valid])
            proj_corr[i, j] = r
            proj_p[i, j] = p

            # Track on-target correlation
            if dir_name == sub_name:
                on_target_rs.append(r)
                print(f"  {dir_name:<35} on-target r = {r:+.3f} (p={p:.4f})")

    # Summary stats
    if on_target_rs:
        print(f"\n  On-target correlations:")
        print(f"    Mean |r| = {np.mean(np.abs(on_target_rs)):.3f}")
        print(f"    Min  |r| = {np.min(np.abs(on_target_rs)):.3f}")
        print(f"    Max  |r| = {np.max(np.abs(on_target_rs)):.3f}")
        n_sig = sum(1 for r in on_target_rs if abs(r) > 0.1)
        print(f"    N with |r| > 0.1: {n_sig}/{len(on_target_rs)}")

    # Plot: projection-behavior correlation heatmap
    dir_short = [s.split("_", 1)[1] if "_" in s else s for s in available_subs]
    sub_short = [s.split("_", 1)[1] if "_" in s else s for s in all_sub_labels]

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(
        proj_corr, cmap="RdBu_r", center=0, vmin=-0.5, vmax=0.5,
        xticklabels=sub_short, yticklabels=dir_short,
        annot=True, fmt=".2f", annot_kws={"size": 6}, ax=ax,
        cbar_kws={"shrink": 0.6, "label": "Pearson r"},
    )
    ax.set_xlabel("Behavioral subscale score")
    ax.set_ylabel("Activation direction (projection)")
    ax.set_title(
        f"Direction Validation: Projection vs Behavior (Layer {layer})\n"
        f"Diagonal = on-target, off-diagonal = cross-construct",
        fontsize=11,
    )
    ax.tick_params(axis="x", labelsize=7, rotation=45)
    ax.tick_params(axis="y", labelsize=7)
    plt.tight_layout()
    plt.savefig(output_dir / f"direction_validation_L{layer}.png", dpi=200, bbox_inches="tight")
    plt.close()

    # Save
    proj_df = pd.DataFrame(proj_corr, index=available_subs, columns=all_sub_labels)
    proj_df.to_csv(output_dir / f"direction_validation_L{layer}.csv")

    return {
        "on_target_mean_abs_r": float(np.mean(np.abs(on_target_rs))) if on_target_rs else np.nan,
        "on_target_rs": {name: float(r) for name, r in zip(available_subs, on_target_rs)},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=DEFAULT_LAYER)
    parser.add_argument("--all_layers", action="store_true",
                        help="Run analysis for all target layers")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    layers = TARGET_LAYERS if args.all_layers else [args.layer]

    # Load human correlation matrices
    print("Computing human correlation matrices...")
    human_sub_corr = compute_human_correlation_matrix("subscale")
    human_scale_corr = compute_human_correlation_matrix("scale")

    human_sub_corr.to_csv(OUTPUT_DIR / "human_subscale_corr.csv")
    human_scale_corr.to_csv(OUTPUT_DIR / "human_scale_corr.csv")

    all_results = []

    for layer in layers:
        print(f"\n{'#'*60}")
        print(f"# LAYER {layer}")
        print(f"{'#'*60}")

        # Subscale level
        sub_dir_path = RESULTS_DIR / "directions" / "subscale_directions.npz"
        if sub_dir_path.exists():
            sub_dirs = load_directions(str(sub_dir_path), ALL_SUBSCALES, layer)
            if sub_dirs:
                cos_sim_sub = compute_cosine_similarity_matrix(sub_dirs, ALL_SUBSCALES)
                cos_sim_sub.to_csv(OUTPUT_DIR / f"subscale_cosine_sim_L{layer}.csv")
                result = analyze_and_plot(cos_sim_sub, human_sub_corr, "subscale", layer, OUTPUT_DIR)
                all_results.append(result)

        # Scale level
        scale_dir_path = RESULTS_DIR / "directions" / "scale_directions.npz"
        if scale_dir_path.exists():
            scale_dirs = load_directions(str(scale_dir_path), SCALE_NAMES, layer)
            if scale_dirs:
                cos_sim_scale = compute_cosine_similarity_matrix(scale_dirs, SCALE_NAMES)
                cos_sim_scale.to_csv(OUTPUT_DIR / f"scale_cosine_sim_L{layer}.csv")
                result = analyze_and_plot(cos_sim_scale, human_scale_corr, "scale", layer, OUTPUT_DIR)
                all_results.append(result)

        # A4: Direction validation
        val_result = validate_directions(layer, OUTPUT_DIR)

    # Summary
    if all_results:
        df = pd.DataFrame(all_results)
        df.to_csv(OUTPUT_DIR / "geometry_results.csv", index=False)
        print(f"\n{'='*60}")
        print("  SUMMARY")
        print(f"{'='*60}")
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()