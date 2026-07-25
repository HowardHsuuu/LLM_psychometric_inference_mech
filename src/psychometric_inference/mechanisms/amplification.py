#!/usr/bin/env python3
"""
Locate amplification: compare three correlation matrices.

1. Human behavioral correlation matrix (ground truth)
2. Activation-predicted correlation matrix (representational level)
3. LLM behavioral output correlation matrix (output level)

If activation-predicted slope ≈ 1 but output slope > 1
→ amplification happens between representation and output.

Method:
  - 272 subjects have layer 16 activations (from subject_activations.py)
  - 272 subjects have 16 behavioral subscale scores (human ground truth)
  - Same 272 subjects have LLM-generated responses (from cross-persona experiment)
  
  For the activation-predicted matrix:
    - For each subscale, train ridge regression: activation → subscale score
    - Use cross-validated predictions to avoid overfitting
    - Compute correlation matrix of predicted scores
    
  Compare all three matrices with human ground truth.

Usage:
    python -m psychometric_inference.mechanisms.amplification \
        --llm_root data/llm_behavior/llama8b_instruct \
        --layer 16
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import cross_val_predict

from .config import (
    PROJECT_ROOT, ALL_SUBSCALES, SCALE_NAMES, SUB_TO_SCALE,
    DEFAULT_LAYER, RESULTS_DIR, HUMAN_DIRS, SCALES,
)
from .geometry import mantel_test, compute_human_correlation_matrix

from psychometric_inference.scoring import SCORING_RULES, FILENAME_TO_SCALE, compute_subscale_scores_from_csvs

logging_setup = True

OUTPUT_DIR = RESULTS_DIR / "amplification_locate"


def load_llm_subscale_scores(llm_root: str) -> pd.DataFrame:
    """Load LLM-generated response data and compute subscale scores.
    
    Expects the cross-persona experiment format:
    llm_root/persona_{scale}/ directories, each containing CSVs for all scales.
    
    For the implicit structure comparison, we need the LLM's responses on
    scales OTHER than the persona scale. This is already computed in your
    compute_behavior_structure.py — we replicate the logic here.
    """
    llm_root = Path(llm_root)
    
    # Try to load from existing implicit structure results first
    # Look for the subscale correlation matrix
    all_dfs = []
    
    for persona_dir in sorted(llm_root.glob("persona_*")):
        persona_scale = persona_dir.name.replace("persona_", "")
        
        # Load all scale CSVs from this persona rotation
        scale_dfs = {}
        item_labels = []
        
        for scale_file, scale_short in SCALES:
            csv_path = persona_dir / f"{scale_file}.csv"
            if not csv_path.exists():
                continue
            df = pd.read_csv(csv_path)
            q_cols = [c for c in df.columns if c.startswith("Q")]
            if not q_cols:
                continue
            sub = df[q_cols].copy()
            
            # Rename columns
            rename = {c: f"{scale_short}_{c}" for c in q_cols}
            sub.rename(columns=rename, inplace=True)
            
            for c in rename.values():
                item_labels.append((scale_short, c))
            
            if "Subject_ID" in df.columns:
                sub.insert(0, "Subject_ID", df["Subject_ID"])
            else:
                sub.insert(0, "Subject_ID", [f"S{i:04d}" for i in range(len(sub))])
            
            scale_dfs[scale_file] = sub
        
        if not scale_dfs:
            continue
            
        # Merge all scales for this rotation
        merged = None
        for sf, sdf in scale_dfs.items():
            if merged is None:
                merged = sdf
            else:
                # Join on Subject_ID
                cols_to_add = [c for c in sdf.columns if c != "Subject_ID"]
                for c in cols_to_add:
                    merged[c] = sdf[c].values
        
        if merged is not None:
            merged["persona_scale"] = persona_scale
            all_dfs.append(merged)
    
    if not all_dfs:
        return pd.DataFrame()
    
    # Combine all rotations
    combined = pd.concat(all_dfs, ignore_index=True)
    
    # Compute subscale scores
    q_cols = [c for c in combined.columns if "_Q" in c]
    item_labels = []
    for c in q_cols:
        parts = c.rsplit("_Q", 1)
        if len(parts) == 2:
            item_labels.append((parts[0], c))
    
    from psychometric_inference.scoring import compute_subscale_scores
    scores = compute_subscale_scores(combined[q_cols], item_labels)
    scores.insert(0, "Subject_ID", combined["Subject_ID"])
    scores["persona_scale"] = combined["persona_scale"]
    
    return scores


def compute_llm_implicit_correlation(llm_scores: pd.DataFrame) -> pd.DataFrame:
    """Compute the LLM's implicit cross-scale correlation matrix.
    
    For each persona rotation, the LLM fills in scales OTHER than the persona scale.
    The correlation between persona scale scores and filled-in scale scores
    (across subjects) gives the LLM's implicit belief about cross-scale relationships.
    
    This replicates the logic from compute_behavior_structure.py at subscale level.
    """
    available_subs = [s for s in ALL_SUBSCALES if s in llm_scores.columns]
    n = len(available_subs)
    corr_matrix = np.full((n, n), np.nan)
    
    # For each pair of subscales
    for i, sub_i in enumerate(available_subs):
        for j, sub_j in enumerate(available_subs):
            if i == j:
                corr_matrix[i, j] = 1.0
                continue
            
            scale_i = SUB_TO_SCALE.get(sub_i, sub_i)
            scale_j = SUB_TO_SCALE.get(sub_j, sub_j)
            
            # Find rotations where one is persona and the other is filled
            vals_i, vals_j = [], []
            
            for persona_scale, group in llm_scores.groupby("persona_scale"):
                persona_short = FILENAME_TO_SCALE.get(persona_scale, persona_scale)
                
                # sub_i is from persona scale, sub_j is filled (or vice versa)
                if persona_short == scale_i and scale_j != scale_i:
                    v_i = group[sub_i].values
                    v_j = group[sub_j].values
                    valid = ~(np.isnan(v_i) | np.isnan(v_j))
                    vals_i.extend(v_i[valid])
                    vals_j.extend(v_j[valid])
                elif persona_short == scale_j and scale_i != scale_j:
                    v_i = group[sub_i].values
                    v_j = group[sub_j].values
                    valid = ~(np.isnan(v_i) | np.isnan(v_j))
                    vals_i.extend(v_i[valid])
                    vals_j.extend(v_j[valid])
            
            if len(vals_i) >= 10:
                r, _ = pearsonr(vals_i, vals_j)
                corr_matrix[i, j] = r
    
    return pd.DataFrame(corr_matrix, index=available_subs, columns=available_subs)


def compute_activation_predicted_correlation(
    activations: np.ndarray,
    behavioral_scores: pd.DataFrame,
    layer: int,
) -> pd.DataFrame:
    """Predict subscale scores from activations using cross-validated ridge regression.
    
    Returns correlation matrix of CV-predicted scores.
    """
    available_subs = [s for s in ALL_SUBSCALES if s in behavioral_scores.columns]
    
    predicted = {}
    print(f"\n  Ridge regression: activation → subscale score (Layer {layer})")
    
    for sub in available_subs:
        y = behavioral_scores[sub].values
        valid = ~np.isnan(y)
        
        if valid.sum() < 20:
            continue
        
        X = activations[valid]
        y_valid = y[valid]
        
        # Cross-validated prediction
        ridge = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0, 1000.0], cv=5)
        y_pred = cross_val_predict(ridge, X, y_valid, cv=5)
        
        r, p = pearsonr(y_valid, y_pred)
        print(f"    {sub:<35} CV r = {r:.3f}")
        
        # Store full-length predictions (NaN where behavioral was NaN)
        full_pred = np.full(len(behavioral_scores), np.nan)
        full_pred[valid] = y_pred
        predicted[sub] = full_pred
    
    pred_df = pd.DataFrame(predicted)
    return pred_df.corr()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm_root", type=str, required=True,
                        help="Path to LLM cross-persona experiment data")
    parser.add_argument("--layer", type=int, default=DEFAULT_LAYER)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load three data sources ──
    
    # 1. Human behavioral correlation
    print("Loading human correlation matrix...")
    human_corr = compute_human_correlation_matrix("subscale")
    
    # 2. Activation-predicted correlation
    print("Loading activations...")
    act_path = RESULTS_DIR / "pca" / "subject_activations.npz"
    scores_path = RESULTS_DIR / "pca" / "human_subscale_scores.csv"
    
    if not act_path.exists() or not scores_path.exists():
        print("ERROR: Need subject activations from subject_activations.py")
        return
    
    acts = np.load(act_path)[f"L{args.layer}"]
    scores_df = pd.read_csv(scores_path)
    
    act_pred_corr = compute_activation_predicted_correlation(acts, scores_df, args.layer)
    
    # 3. LLM behavioral output correlation
    print(f"\nLoading LLM behavioral data from {args.llm_root}...")
    llm_scores = load_llm_subscale_scores(args.llm_root)
    
    if llm_scores.empty:
        print("ERROR: Could not load LLM data")
        return
    
    llm_corr = compute_llm_implicit_correlation(llm_scores)
    
    # ── Compare all three with human ground truth ──
    
    # Find common subscales across all three matrices
    common = [s for s in ALL_SUBSCALES 
              if s in human_corr.index and s in act_pred_corr.index and s in llm_corr.index]
    n = len(common)
    print(f"\nCommon subscales: {n}")
    
    idx = np.triu_indices(n, k=1)
    hum = human_corr.loc[common, common].values
    hum_vec = hum[idx]
    
    print(f"\n{'='*60}")
    print(f"  THREE-WAY COMPARISON (Layer {args.layer})")
    print(f"{'='*60}")
    print(f"  {'Matrix':<30} {'Mantel r':>10} {'Slope':>10} {'p':>10}")
    print(f"  {'-'*62}")
    
    results = {}
    for label, mat_df in [("Activation-predicted", act_pred_corr),
                           ("LLM behavioral output", llm_corr)]:
        mat = mat_df.loc[common, common].values
        mat_vec = mat[idx]
        
        valid = ~(np.isnan(hum_vec) | np.isnan(mat_vec))
        h_v, m_v = hum_vec[valid], mat_vec[valid]
        
        if len(h_v) < 3:
            continue
        
        r_mantel, p_mantel = mantel_test(hum, mat)
        r_pearson, _ = pearsonr(h_v, m_v)
        slope, intercept = np.polyfit(h_v, m_v, 1)
        
        print(f"  {label:<30} {r_mantel:>10.4f} {slope:>10.4f} {p_mantel:>10.4f}")
        results[label] = {
            "mantel_r": r_mantel, "slope": slope, "intercept": intercept,
            "pearson_r": r_pearson, "p": p_mantel, "n_pairs": len(h_v),
        }
    
    # ── Key comparison ──
    if "Activation-predicted" in results and "LLM behavioral output" in results:
        act_slope = results["Activation-predicted"]["slope"]
        out_slope = results["LLM behavioral output"]["slope"]
        print(f"\n  Activation-predicted slope: {act_slope:.3f}")
        print(f"  LLM output slope:          {out_slope:.3f}")
        print(f"  Ratio (output/activation):  {out_slope/act_slope:.3f}" if act_slope != 0 else "")
        
        if act_slope < 1.2 and out_slope > 1.0:
            print(f"\n  → Amplification occurs BETWEEN representation and output")
        elif act_slope > 1.0:
            print(f"\n  → Amplification already present at representation level")
        else:
            print(f"\n  → Both attenuated relative to human structure")
    
    # ── Plot: three-way scatter ──
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    for ax, (label, mat_df) in zip([ax1, ax2], 
                                     [("Activation-predicted", act_pred_corr),
                                      ("LLM behavioral output", llm_corr)]):
        mat = mat_df.loc[common, common].values
        mat_vec = mat[idx]
        valid = ~(np.isnan(hum_vec) | np.isnan(mat_vec))
        h_v, m_v = hum_vec[valid], mat_vec[valid]
        
        ax.scatter(h_v, m_v, alpha=0.5, s=30, edgecolors="white", linewidths=0.5)
        
        if len(h_v) >= 3:
            slope, intercept = np.polyfit(h_v, m_v, 1)
            r_val, _ = pearsonr(h_v, m_v)
            xx = np.linspace(h_v.min() - 0.05, h_v.max() + 0.05, 100)
            ax.plot(xx, slope * xx + intercept, "k--", lw=1, alpha=0.7,
                    label=f"slope={slope:.2f}, r={r_val:.3f}")
        
        lim = max(abs(h_v).max(), abs(m_v).max()) * 1.2
        ax.plot([-lim, lim], [-lim, lim], "gray", ls=":", lw=0.8, alpha=0.5, label="y=x")
        ax.set_xlabel("Human behavioral correlation")
        ax.set_ylabel(label)
        ax.set_title(f"{label}\n(slope={slope:.2f})")
        ax.legend(fontsize=8)
        ax.axhline(y=0, color="gray", lw=0.3)
        ax.axvline(x=0, color="gray", lw=0.3)
    
    plt.suptitle(
        f"Locating Amplification: Representation vs Output (Layer {args.layer})",
        fontsize=13,
    )
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"three_way_comparison_L{args.layer}.png",
                dpi=200, bbox_inches="tight")
    plt.close()
    
    # ── Save ──
    act_pred_corr.to_csv(OUTPUT_DIR / f"activation_predicted_corr_L{args.layer}.csv")
    llm_corr.to_csv(OUTPUT_DIR / f"llm_output_corr.csv")
    
    import json
    with open(OUTPUT_DIR / f"comparison_results_L{args.layer}.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nAll outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()