#!/usr/bin/env python3
"""Partial-correlation diagnostics for cross-construct specificity.

Using subject activations and behavioral scores, this module asks whether
projection onto direction_i predicts behavior_j after controlling for
behavior_i. Activation interventions live in ``steering.py``.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

from .config import ALL_SUBSCALES, DEFAULT_LAYER, RESULTS_DIR
from .geometry import (
    load_directions, compute_human_correlation_matrix, mantel_test,
)

OUTPUT_DIR = RESULTS_DIR / "causal"


# ════════════════════════════════════════════════════════
#  PART 1: Partial Correlation
# ════════════════════════════════════════════════════════

def partial_corr(x, y, covariates):
    """Partial correlation between x and y, controlling for covariates.
    
    Residualize both x and y on covariates, then correlate residuals.
    """
    if covariates.shape[1] == 0:
        return pearsonr(x, y)

    # Add constant
    C = np.column_stack([covariates, np.ones(len(covariates))])

    # Residualize x
    beta_x = np.linalg.lstsq(C, x, rcond=None)[0]
    resid_x = x - C @ beta_x

    # Residualize y
    beta_y = np.linalg.lstsq(C, y, rcond=None)[0]
    resid_y = y - C @ beta_y

    return pearsonr(resid_x, resid_y)


def run_partial_correlation(layer: int, output_dir: Path):
    """Compute partial correlation matrix: projection onto direction_i
    predicts behavior_j, controlling for behavior_i.
    
    This tests: does the Neuroticism direction carry information about
    Negative Affect BEYOND what's explained by the subject's actual
    Neuroticism score? If yes → the direction encodes cross-construct
    relationships, not just the target construct.
    """
    print(f"\n{'='*60}")
    print(f"  PARTIAL CORRELATION ANALYSIS (Layer {layer})")
    print(f"{'='*60}")

    # Load data
    act_path = RESULTS_DIR / "pca" / "subject_activations.npz"
    scores_path = RESULTS_DIR / "pca" / "human_subscale_scores.csv"
    dir_path = RESULTS_DIR / "directions" / "subscale_directions.npz"

    if not all(p.exists() for p in [act_path, scores_path, dir_path]):
        print("  Missing data files, skipping")
        return None

    acts = np.load(act_path)[f"L{layer}"]
    scores_df = pd.read_csv(scores_path)
    all_dirs = load_directions(str(dir_path), ALL_SUBSCALES, layer)
    human_corr = compute_human_correlation_matrix("subscale")

    available = [s for s in ALL_SUBSCALES if s in all_dirs and s in scores_df.columns]
    n = len(available)
    print(f"  {n} subscales available")

    # Compute two matrices:
    # 1. Zero-order: corr(projection_i, behavior_j) — same as A4
    # 2. Partial: corr(projection_i, behavior_j | behavior_i)
    zero_order = np.full((n, n), np.nan)
    partial = np.full((n, n), np.nan)

    for i, dir_name in enumerate(available):
        direction = all_dirs[dir_name]
        proj = acts @ direction  # projection of 272 subjects onto direction_i

        for j, beh_name in enumerate(available):
            behavior_j = scores_df[beh_name].values

            # Zero-order
            r_zero, _ = pearsonr(proj, behavior_j)
            zero_order[i, j] = r_zero

            # Partial: control for behavior_i (the direction's own construct)
            if i != j:
                behavior_i = scores_df[dir_name].values
                covariates = behavior_i.reshape(-1, 1)
                r_partial, p_partial = partial_corr(proj, behavior_j, covariates)
                partial[i, j] = r_partial
            else:
                partial[i, j] = r_zero  # diagonal: no covariate to control

    # Compare partial matrix with human correlation matrix
    hum = human_corr.loc[available, available].values
    idx = np.triu_indices(n, k=1)

    # Zero-order symmetrized
    zero_sym = (zero_order + zero_order.T) / 2
    partial_sym = (partial + partial.T) / 2

    for label, mat in [("Zero-order (A4)", zero_sym), ("Partial", partial_sym)]:
        mv = mat[idx]
        hv = hum[idx]
        valid = ~(np.isnan(mv) | np.isnan(hv))
        mv_v, hv_v = mv[valid], hv[valid]
        if len(hv_v) < 3:
            continue
        r_m, p_m = mantel_test(mat, hum)
        r_p, p_p = pearsonr(hv_v, mv_v)
        slope, intercept = np.polyfit(hv_v, mv_v, 1)
        print(f"\n  {label}:")
        print(f"    Mantel r  = {r_m:.4f} (p = {p_m:.4f})")
        print(f"    Pearson r = {r_p:.4f}")
        print(f"    Slope     = {slope:.4f}")

    # Plot: partial correlation heatmap
    short = [s.split("_", 1)[1] if "_" in s else s for s in available]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    sns.heatmap(zero_sym, cmap="RdBu_r", center=0, vmin=-0.5, vmax=0.5,
                xticklabels=short, yticklabels=short,
                annot=True, fmt=".2f", annot_kws={"size": 6}, ax=ax1,
                cbar_kws={"shrink": 0.6})
    ax1.set_title("Zero-order (projection_i vs behavior_j)", fontsize=10)
    ax1.tick_params(axis="both", labelsize=7)

    sns.heatmap(partial_sym, cmap="RdBu_r", center=0, vmin=-0.5, vmax=0.5,
                xticklabels=short, yticklabels=short,
                annot=True, fmt=".2f", annot_kws={"size": 6}, ax=ax2,
                cbar_kws={"shrink": 0.6})
    ax2.set_title("Partial (projection_i vs behavior_j | behavior_i)", fontsize=10)
    ax2.tick_params(axis="both", labelsize=7)

    plt.suptitle(f"Cross-Construct Specificity via Partial Correlation (Layer {layer})", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_dir / f"partial_corr_heatmaps_L{layer}.png", dpi=200, bbox_inches="tight")
    plt.close()

    # Save
    pd.DataFrame(zero_sym, index=available, columns=available).to_csv(
        output_dir / f"zero_order_matrix_L{layer}.csv")
    pd.DataFrame(partial_sym, index=available, columns=available).to_csv(
        output_dir / f"partial_matrix_L{layer}.csv")

    return {"zero_order": zero_sym, "partial": partial_sym}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=DEFAULT_LAYER)
    parser.add_argument("--partial_only", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    run_partial_correlation(args.layer, OUTPUT_DIR)


if __name__ == "__main__":
    main()
