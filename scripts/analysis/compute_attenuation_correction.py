#!/usr/bin/env python3
"""
Correction for Attenuation Analysis

Computes Cronbach's alpha for human data, then disattenuates the human
correlation matrix to estimate "true score" correlations. Compares raw
vs corrected slopes to assess how much of the observed amplification
is a reliability artifact vs genuine structural amplification.

Output:
  - Table of raw vs corrected r and slope for all models
  - Human alpha values per scale
  - CSV with all results

Usage:
    python compute_attenuation_correction.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR / "src"))

from psychometric_inference.scoring import SCORING_RULES
from psychometric_inference.model_registry import analysis_model_tuples

SCALES_MAP = [
    ("IRI", "IRI"),
    ("PANAS", "PANAS"),
    ("POM", "POM"),
    ("big_five", "BigFive"),
    ("in_inter_dependent", "SelfConst"),
    ("Life_Satisfaction", "LifeSat"),
    ("Loneliness", "Lonely"),
]

SCALE_NAMES = ["IRI", "PANAS", "POM", "BigFive", "SelfConst", "LifeSat", "Lonely"]

MODELS = analysis_model_tuples()


def cronbach_alpha(items_df):
    """Compute Cronbach's alpha from item-level data."""
    items = items_df.dropna(axis=1)
    n = items.shape[1]
    if n < 2:
        return np.nan
    item_vars = items.var(axis=0, ddof=1)
    total_var = items.sum(axis=1).var(ddof=1)
    if total_var == 0:
        return np.nan
    return (n / (n - 1)) * (1 - item_vars.sum() / total_var)


def compute_human_alphas():
    """Compute scale-level and subscale-level Cronbach's alpha from original CSVs
    (which already have reverse coding applied)."""
    scale_alphas = {}
    subscale_alphas = {}

    for scale_file, scale_short in SCALES_MAP:
        frames = []
        for ds in ["SED", "SEDC", "SEDD"]:
            fpath = BASE_DIR / "data/human" / ds / f"{scale_file}.csv"
            if fpath.exists():
                frames.append(pd.read_csv(fpath))
        if not frames:
            continue
        df = pd.concat(frames, ignore_index=True)
        q_cols = [c for c in df.columns if c.startswith("Q")]

        # Scale-level alpha (all items)
        scale_alphas[scale_short] = cronbach_alpha(df[q_cols])

        # Subscale-level alpha
        rules = SCORING_RULES.get(scale_short, {})
        for sub_name, item_nums in rules.get("subscales", {}).items():
            sub_cols = [f"Q{n}" for n in item_nums]
            valid = [c for c in sub_cols if c in df.columns]
            if len(valid) >= 2:
                subscale_alphas[f"{scale_short}_{sub_name}"] = cronbach_alpha(df[valid])

    return scale_alphas, subscale_alphas


def disattenuate_matrix(corr_matrix, alphas, scale_names):
    """Apply correction for attenuation to a correlation matrix.
    r_corrected(i,j) = r_observed(i,j) / sqrt(alpha_i * alpha_j)
    """
    corrected = corr_matrix.copy()
    n = len(scale_names)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            si = scale_names[i]
            sj = scale_names[j]
            ai = alphas.get(si, 1.0)
            aj = alphas.get(sj, 1.0)
            if ai > 0 and aj > 0:
                val = corr_matrix.iloc[i, j] / np.sqrt(ai * aj)
                # Clip to [-1, 1] (correction can exceed bounds)
                corrected.iloc[i, j] = np.clip(val, -1.0, 1.0)
            else:
                corrected.iloc[i, j] = np.nan
    return corrected


def compare_matrices(human_mat, llm_mat):
    """Compare upper triangle: return r and slope."""
    common = [c for c in human_mat.columns if c in llm_mat.columns]
    H = human_mat.loc[common, common].values
    L = llm_mat.loc[common, common].values
    n = len(common)
    mask = np.triu_indices(n, k=1)
    h_vec = H[mask]
    l_vec = L[mask]
    valid = ~(np.isnan(h_vec) | np.isnan(l_vec))
    if valid.sum() < 3:
        return np.nan, np.nan
    r = np.corrcoef(h_vec[valid], l_vec[valid])[0, 1]
    slope = np.polyfit(h_vec[valid], l_vec[valid], 1)[0]
    return r, slope


def main():
    # Compute human reliability
    scale_alphas, subscale_alphas = compute_human_alphas()

    print("=" * 60)
    print("  Human Cronbach's Alpha (scale-level)")
    print("=" * 60)
    for s in SCALE_NAMES:
        a = scale_alphas.get(s, np.nan)
        print(f"  {s:<12} alpha = {a:.3f}")

    print(f"\n{'=' * 60}")
    print("  Human Cronbach's Alpha (subscale-level)")
    print("=" * 60)
    for name, a in sorted(subscale_alphas.items()):
        print(f"  {name:<35} alpha = {a:.3f}")

    # Compute correction for each model
    results = []

    print(f"\n{'=' * 80}")
    print(f"  {'Model':<22} {'raw_r':>6} {'raw_sl':>7} {'cor_r':>6} {'cor_sl':>7} {'sl_change':>9}")
    print("=" * 80)

    for dirname, size, mtype, family in MODELS:
        h_path = BASE_DIR / "outputs/behavior" / dirname / "scale_human.csv"
        l_path = BASE_DIR / "outputs/behavior" / dirname / "scale_llm_implicit.csv"
        if not h_path.exists() or not l_path.exists():
            continue

        h_mat = pd.read_csv(h_path, index_col=0)
        l_mat = pd.read_csv(l_path, index_col=0)

        # Raw comparison
        raw_r, raw_slope = compare_matrices(h_mat, l_mat)

        # Corrected comparison
        h_corrected = disattenuate_matrix(h_mat, scale_alphas, SCALE_NAMES)
        cor_r, cor_slope = compare_matrices(h_corrected, l_mat)

        label = f"{size}B {mtype[:4]} {family}"
        sl_change = cor_slope - raw_slope if not (np.isnan(cor_slope) or np.isnan(raw_slope)) else np.nan

        print(f"  {label:<22} {raw_r:>6.3f} {raw_slope:>7.3f} {cor_r:>6.3f} {cor_slope:>7.3f} {sl_change:>+9.3f}")

        results.append({
            "dirname": dirname,
            "size": size,
            "type": mtype,
            "family": family,
            "raw_r": raw_r,
            "raw_slope": raw_slope,
            "corrected_r": cor_r,
            "corrected_slope": cor_slope,
            "slope_change": sl_change,
        })

    # Save
    df = pd.DataFrame(results)
    out_path = BASE_DIR / "outputs" / "behavior" / "attenuation_correction.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")

    # Summary
    print(f"\n{'=' * 60}")
    print("  Summary: Slope before and after correction")
    print("=" * 60)
    base_only = df[df["type"] == "base"].sort_values("size")
    print(f"\n  Base models:")
    print(f"  {'Size':<8} {'Family':<8} {'Raw slope':>10} {'Corrected':>10} {'Change':>8}")
    print(f"  {'-'*48}")
    for _, r in base_only.iterrows():
        print(f"  {r['size']:<8} {r['family']:<8} {r['raw_slope']:>10.3f} {r['corrected_slope']:>10.3f} {r['slope_change']:>+8.3f}")

    print(f"\n  Interpretation:")
    print(f"  - If corrected slope ≈ 1.0: amplification is mostly a reliability artifact")
    print(f"  - If corrected slope still > 1.0: genuine structural amplification beyond reliability")
    print(f"  - If corrected slope < 1.0: model attenuates even true-score structure")


if __name__ == "__main__":
    main()
