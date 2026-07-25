#!/usr/bin/env python3
"""
Quick permutation test: is the activation-predicted slope = 1.52 real or artifact?

Shuffles the mapping between activations and behavioral scores,
re-runs ridge CV, and checks if the predicted correlation matrix
still shows high slope. If yes → the slope is an artifact of
high-dimensional regression inflating correlations.

Usage:
    python -m psychometric_inference.mechanisms.ridge_baseline --layer 16
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import pearsonr
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import cross_val_predict

from .config import PROJECT_ROOT, ALL_SUBSCALES, DEFAULT_LAYER, RESULTS_DIR, HUMAN_DIRS
from .geometry import compute_human_correlation_matrix


import argparse


def run_ridge_predicted_corr(activations, scores_df, subscales):
    """Ridge CV predict → correlation matrix of predictions."""
    predicted = {}
    for sub in subscales:
        y = scores_df[sub].values
        valid = ~np.isnan(y)
        if valid.sum() < 20:
            continue
        X = activations[valid]
        y_v = y[valid]
        ridge = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0, 1000.0], cv=5)
        y_pred = cross_val_predict(ridge, X, y_v, cv=5)
        full_pred = np.full(len(scores_df), np.nan)
        full_pred[valid] = y_pred
        predicted[sub] = full_pred
    return pd.DataFrame(predicted).corr()


def compute_slope(pred_corr, human_corr, subscales):
    """Compute slope of predicted vs human correlation."""
    common = [s for s in subscales if s in pred_corr.columns and s in human_corr.columns]
    n = len(common)
    idx = np.triu_indices(n, k=1)
    pv = pred_corr.loc[common, common].values[idx]
    hv = human_corr.loc[common, common].values[idx]
    valid = ~(np.isnan(pv) | np.isnan(hv))
    if valid.sum() < 3:
        return np.nan, np.nan
    slope = np.polyfit(hv[valid], pv[valid], 1)[0]
    r, _ = pearsonr(hv[valid], pv[valid])
    return slope, r


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=DEFAULT_LAYER)
    parser.add_argument("--n_perms", type=int, default=100)
    args = parser.parse_args()

    act_path = RESULTS_DIR / "pca" / "subject_activations.npz"
    scores_path = RESULTS_DIR / "pca" / "human_subscale_scores.csv"

    acts = np.load(act_path)[f"L{args.layer}"]
    scores_df = pd.read_csv(scores_path)
    human_corr = compute_human_correlation_matrix("subscale")
    subscales = [s for s in ALL_SUBSCALES if s in scores_df.columns]

    # Real slope
    real_corr = run_ridge_predicted_corr(acts, scores_df, subscales)
    real_slope, real_r = compute_slope(real_corr, human_corr, subscales)
    print(f"Real: slope = {real_slope:.3f}, r = {real_r:.3f}")

    # Permutation test
    rng = np.random.default_rng(42)
    perm_slopes = []
    perm_rs = []

    for i in range(args.n_perms):
        # Shuffle rows of behavioral scores (break activation-behavior mapping)
        perm_idx = rng.permutation(len(scores_df))
        scores_perm = scores_df.iloc[perm_idx].reset_index(drop=True)

        perm_corr = run_ridge_predicted_corr(acts, scores_perm, subscales)
        s, r = compute_slope(perm_corr, human_corr, subscales)
        perm_slopes.append(s)
        perm_rs.append(r)

        if (i + 1) % 10 == 0:
            print(f"  Perm {i+1}/{args.n_perms}: slope = {s:.3f}, r = {r:.3f}")

    perm_slopes = np.array(perm_slopes)
    perm_rs = np.array(perm_rs)

    p_slope = np.mean(perm_slopes >= real_slope)
    p_r = np.mean(perm_rs >= real_r)

    print(f"\n{'='*50}")
    print(f"  PERMUTATION TEST (n={args.n_perms})")
    print(f"{'='*50}")
    print(f"  Real slope:     {real_slope:.3f}")
    print(f"  Perm slope:     {np.mean(perm_slopes):.3f} ± {np.std(perm_slopes):.3f}")
    print(f"  p-value (slope): {p_slope:.4f}")
    print(f"")
    print(f"  Real r:         {real_r:.3f}")
    print(f"  Perm r:         {np.mean(perm_rs):.3f} ± {np.std(perm_rs):.3f}")
    print(f"  p-value (r):    {p_r:.4f}")

    if p_slope < 0.05:
        print(f"\n  → Slope = {real_slope:.3f} is REAL (p = {p_slope:.4f})")
    else:
        print(f"\n  → Slope = {real_slope:.3f} may be INFLATED by ridge regression artifact")


if __name__ == "__main__":
    main()