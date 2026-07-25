#!/usr/bin/env python3
"""
Control analyses to rule out confounds.

1. Semantic Similarity Control:
   Is direction geometry just item semantic similarity?
   Compute item-level semantic similarity between subscales using
   the model's embedding layer, compare with direction cosine sim
   and human correlation. If direction geometry predicts human correlation
   AFTER controlling for semantic similarity → not just semantics.

2. Regression Direction Geometry Permutation Test:
   Is regression direction cosine sim slope = 0.96 an artifact?
   Shuffle subject-behavior mapping, refit regression directions,
   check if permuted directions' cosine sim also correlates with human.

Usage:
    python -m psychometric_inference.mechanisms.controls --layer 16
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import RidgeCV

from .config import (
    PROJECT_ROOT, ALL_SUBSCALES, SCALE_NAMES, DEFAULT_LAYER,
    RESULTS_DIR, MODEL_ID, SCALES,
)
from .geometry import compute_human_correlation_matrix, mantel_test

from psychometric_inference.scoring import SCORING_RULES

OUTPUT_DIR = RESULTS_DIR / "controls"


# ════════════════════════════════════════════════════════
#  CONTROL 1: Semantic Similarity
# ════════════════════════════════════════════════════════

def compute_item_semantic_similarity(model_id: str = MODEL_ID):
    """Compute semantic similarity between subscales based on item text embeddings.
    
    For each subscale, average the embedding-layer representations of its items.
    Then compute cosine similarity between all subscale pairs.
    This captures surface-level semantic similarity of the questionnaire items.
    """
    from transformers import AutoTokenizer, AutoModel
    import torch

    print(f"\n{'='*60}")
    print(f"  SEMANTIC SIMILARITY CONTROL")
    print(f"{'='*60}")

    print(f"  Loading tokenizer: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # We only need the embedding layer, not the full model
    # Load model and extract just the embedding weights
    print(f"  Loading embedding layer...")
    model = AutoModel.from_pretrained(model_id, torch_dtype=torch.float16)
    embed_matrix = model.get_input_embeddings().weight.detach().float().cpu().numpy()
    del model
    import gc; gc.collect()

    # Load scale definitions to get item texts
    from .prompts import load_all_scale_definitions
    scale_defs = load_all_scale_definitions()

    # For each subscale, get item texts and compute mean embedding
    subscale_embeddings = {}

    for scale_file, scale_short in SCALES:
        sd = scale_defs[scale_file]
        rules = SCORING_RULES.get(scale_short, {})

        for sub_name, item_nums in rules.get("subscales", {}).items():
            sub_full = f"{scale_short}_{sub_name}"

            # Get item texts
            item_texts = []
            for item in sd["items"]:
                if item["item_number"] in item_nums:
                    item_texts.append(item["text"])

            if not item_texts:
                continue

            # Tokenize and get mean embedding
            embeddings = []
            for text in item_texts:
                token_ids = tokenizer.encode(text, add_special_tokens=False)
                if token_ids:
                    item_embed = embed_matrix[token_ids].mean(axis=0)
                    embeddings.append(item_embed)

            if embeddings:
                subscale_embeddings[sub_full] = np.mean(embeddings, axis=0)

    # Compute cosine similarity matrix
    available = [s for s in ALL_SUBSCALES if s in subscale_embeddings]
    n = len(available)
    sem_sim = np.zeros((n, n))

    for i, si in enumerate(available):
        for j, sj in enumerate(available):
            vi = subscale_embeddings[si]
            vj = subscale_embeddings[sj]
            cos = np.dot(vi, vj) / (np.linalg.norm(vi) * np.linalg.norm(vj) + 1e-10)
            sem_sim[i, j] = cos

    sem_sim_df = pd.DataFrame(sem_sim, index=available, columns=available)
    return sem_sim_df


def run_semantic_control(layer: int, output_dir: Path, model_id: str = MODEL_ID):
    """Compare direction geometry, semantic similarity, and human correlation."""

    sem_sim = compute_item_semantic_similarity(model_id)
    human_corr = compute_human_correlation_matrix("subscale")

    # Load regression direction cosine sim
    reg_cos_path = RESULTS_DIR / "regression_directions" / f"regression_cosine_sim_L{layer}.csv"
    if not reg_cos_path.exists():
        print("  Regression cosine sim not found, skipping")
        return None
    reg_cos = pd.read_csv(reg_cos_path, index_col=0)

    # Also try contrastive direction cosine sim
    from .geometry import load_directions, compute_cosine_similarity_matrix
    con_dir_path = RESULTS_DIR / "directions" / "subscale_directions.npz"
    con_cos = None
    if con_dir_path.exists():
        con_dirs = load_directions(str(con_dir_path), ALL_SUBSCALES, layer)
        available_con = [s for s in ALL_SUBSCALES if s in con_dirs]
        if available_con:
            con_cos = compute_cosine_similarity_matrix(con_dirs, available_con)

    # Find common subscales
    common = [s for s in ALL_SUBSCALES
              if s in sem_sim.index and s in human_corr.index and s in reg_cos.index]
    n = len(common)
    idx = np.triu_indices(n, k=1)

    hum_vec = human_corr.loc[common, common].values[idx]
    sem_vec = sem_sim.loc[common, common].values[idx]
    reg_vec = reg_cos.loc[common, common].values[idx]

    valid = ~(np.isnan(hum_vec) | np.isnan(sem_vec) | np.isnan(reg_vec))
    hum_v, sem_v, reg_v = hum_vec[valid], sem_vec[valid], reg_vec[valid]

    # Zero-order correlations
    r_sem_hum, p_sem_hum = pearsonr(sem_v, hum_v)
    r_reg_hum, p_reg_hum = pearsonr(reg_v, hum_v)
    r_sem_reg, p_sem_reg = pearsonr(sem_v, reg_v)

    print(f"\n  Zero-order correlations ({n} subscales, {len(hum_v)} pairs):")
    print(f"    Semantic sim ↔ Human corr:    r = {r_sem_hum:.3f} (p = {p_sem_hum:.4f})")
    print(f"    Regression cos ↔ Human corr:  r = {r_reg_hum:.3f} (p = {p_reg_hum:.4f})")
    print(f"    Semantic sim ↔ Regression cos: r = {r_sem_reg:.3f} (p = {p_sem_reg:.4f})")

    # Partial correlation: regression cos ↔ human corr, controlling for semantic sim
    from .causality import partial_corr
    r_partial, p_partial = partial_corr(reg_v, hum_v, sem_v.reshape(-1, 1))

    print(f"\n  Partial correlation (controlling for semantic similarity):")
    print(f"    Regression cos ↔ Human corr | Semantic: r = {r_partial:.3f} (p = {p_partial:.4f})")

    if r_partial > 0.3 and p_partial < 0.05:
        print(f"    → Direction geometry is NOT just semantic similarity")
    elif r_sem_hum > 0.3 and r_partial < 0.1:
        print(f"    → Direction geometry MAY be driven by semantic similarity")
    else:
        print(f"    → Partial evidence; semantic similarity partially contributes")

    # Plot
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.5))

    # Semantic sim vs human
    ax1.scatter(hum_v, sem_v, alpha=0.4, s=20)
    sl, ic = np.polyfit(hum_v, sem_v, 1)
    xx = np.linspace(hum_v.min() - 0.05, hum_v.max() + 0.05, 100)
    ax1.plot(xx, sl * xx + ic, "k--", lw=1)
    ax1.set_xlabel("Human correlation")
    ax1.set_ylabel("Item semantic similarity")
    ax1.set_title(f"Semantic sim vs Human\nr={r_sem_hum:.3f}")

    # Regression cos vs human
    ax2.scatter(hum_v, reg_v, alpha=0.4, s=20)
    sl2, ic2 = np.polyfit(hum_v, reg_v, 1)
    ax2.plot(xx, sl2 * xx + ic2, "k--", lw=1)
    ax2.set_xlabel("Human correlation")
    ax2.set_ylabel("Regression direction cosine sim")
    ax2.set_title(f"Direction geometry vs Human\nr={r_reg_hum:.3f}")

    # Semantic sim vs regression cos
    ax3.scatter(sem_v, reg_v, alpha=0.4, s=20)
    sl3, ic3 = np.polyfit(sem_v, reg_v, 1)
    xx3 = np.linspace(sem_v.min() - 0.01, sem_v.max() + 0.01, 100)
    ax3.plot(xx3, sl3 * xx3 + ic3, "k--", lw=1)
    ax3.set_xlabel("Item semantic similarity")
    ax3.set_ylabel("Regression direction cosine sim")
    ax3.set_title(f"Semantic sim vs Direction geometry\nr={r_sem_reg:.3f}")

    plt.suptitle(
        f"Semantic Similarity Control (Layer {layer})\n"
        f"Partial r (direction ↔ human | semantic) = {r_partial:.3f}, p = {p_partial:.4f}",
        fontsize=12,
    )
    plt.tight_layout()
    plt.savefig(output_dir / f"semantic_control_L{layer}.png", dpi=200, bbox_inches="tight")
    plt.close()

    # Save
    sem_sim.to_csv(output_dir / f"semantic_similarity_matrix.csv")

    return {
        "r_sem_hum": r_sem_hum,
        "r_reg_hum": r_reg_hum,
        "r_sem_reg": r_sem_reg,
        "r_partial": r_partial,
        "p_partial": p_partial,
    }


# ════════════════════════════════════════════════════════
#  CONTROL 2: Regression Direction Permutation Test
# ════════════════════════════════════════════════════════

def run_regression_direction_permutation(layer: int, output_dir: Path, n_perms: int = 100):
    """Permutation test for regression direction cosine similarity geometry.
    
    Shuffle subject-behavior mapping, refit ridge regression directions,
    compute cosine sim matrix, compare with human correlation.
    If permuted Mantel r ≈ 0 and real Mantel r = 0.97 → geometry is real.
    """
    print(f"\n{'='*60}")
    print(f"  REGRESSION DIRECTION GEOMETRY PERMUTATION TEST")
    print(f"{'='*60}")

    act_path = RESULTS_DIR / "pca" / "subject_activations.npz"
    scores_path = RESULTS_DIR / "pca" / "human_subscale_scores.csv"

    acts = np.load(act_path)[f"L{layer}"]
    scores_df = pd.read_csv(scores_path)
    human_corr = compute_human_correlation_matrix("subscale")

    subscales = [s for s in ALL_SUBSCALES if s in scores_df.columns]

    def fit_directions_and_get_geometry(activations, scores, subscales):
        """Fit ridge regression directions, compute cosine sim, compare with human."""
        directions = {}
        for sub in subscales:
            y = scores[sub].values
            valid = ~np.isnan(y)
            if valid.sum() < 20:
                continue
            X = activations[valid]
            y_v = y[valid]
            ridge = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0, 1000.0], cv=5)
            ridge.fit(X, y_v)
            w = ridge.coef_
            norm = np.linalg.norm(w)
            if norm > 0:
                directions[sub] = w / norm

        # Cosine sim matrix
        available = [s for s in subscales if s in directions]
        n = len(available)
        cos_mat = np.zeros((n, n))
        for i, si in enumerate(available):
            for j, sj in enumerate(available):
                cos_mat[i, j] = np.dot(directions[si], directions[sj])

        cos_df = pd.DataFrame(cos_mat, index=available, columns=available)

        # Compare with human
        common = [s for s in available if s in human_corr.index]
        nc = len(common)
        if nc < 4:
            return np.nan, np.nan
        idx_tri = np.triu_indices(nc, k=1)
        cv = cos_df.loc[common, common].values[idx_tri]
        hv = human_corr.loc[common, common].values[idx_tri]
        valid_mask = ~(np.isnan(cv) | np.isnan(hv))
        if valid_mask.sum() < 3:
            return np.nan, np.nan

        r_mantel, p_mantel = mantel_test(
            cos_df.loc[common, common].values,
            human_corr.loc[common, common].values,
        )
        slope = np.polyfit(hv[valid_mask], cv[valid_mask], 1)[0]
        return r_mantel, slope

    # Real geometry
    real_r, real_slope = fit_directions_and_get_geometry(acts, scores_df, subscales)
    print(f"  Real: Mantel r = {real_r:.3f}, slope = {real_slope:.3f}")

    # Permutation test
    rng = np.random.default_rng(42)
    perm_rs = []
    perm_slopes = []

    for i in range(n_perms):
        perm_idx = rng.permutation(len(scores_df))
        scores_perm = scores_df.iloc[perm_idx].reset_index(drop=True)

        pr, ps = fit_directions_and_get_geometry(acts, scores_perm, subscales)
        perm_rs.append(pr)
        perm_slopes.append(ps)

        if (i + 1) % 20 == 0:
            print(f"    Perm {i+1}/{n_perms}: r = {pr:.3f}, slope = {ps:.3f}")

    perm_rs = np.array(perm_rs)
    perm_slopes = np.array(perm_slopes)

    p_r = np.mean(perm_rs >= real_r)
    p_slope = np.mean(perm_slopes >= real_slope)

    print(f"\n  Results ({n_perms} permutations):")
    print(f"    Real Mantel r:  {real_r:.3f}")
    print(f"    Perm Mantel r:  {np.nanmean(perm_rs):.3f} ± {np.nanstd(perm_rs):.3f}")
    print(f"    p-value (r):    {p_r:.4f}")
    print(f"    Real slope:     {real_slope:.3f}")
    print(f"    Perm slope:     {np.nanmean(perm_slopes):.3f} ± {np.nanstd(perm_slopes):.3f}")
    print(f"    p-value (slope): {p_slope:.4f}")

    if p_r < 0.05:
        print(f"\n    → Regression direction geometry is REAL (p = {p_r:.4f})")
    else:
        print(f"\n    → Regression direction geometry may be artifact")

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.hist(perm_rs[~np.isnan(perm_rs)], bins=20, alpha=0.7, color="#2c7bb6", label="Permuted")
    ax1.axvline(real_r, color="red", lw=2, label=f"Real (r={real_r:.3f})")
    ax1.set_xlabel("Mantel r")
    ax1.set_ylabel("Count")
    ax1.set_title(f"Mantel r permutation (p={p_r:.4f})")
    ax1.legend()

    ax2.hist(perm_slopes[~np.isnan(perm_slopes)], bins=20, alpha=0.7, color="#2c7bb6", label="Permuted")
    ax2.axvline(real_slope, color="red", lw=2, label=f"Real (slope={real_slope:.3f})")
    ax2.set_xlabel("Slope")
    ax2.set_ylabel("Count")
    ax2.set_title(f"Slope permutation (p={p_slope:.4f})")
    ax2.legend()

    plt.suptitle("Regression Direction Geometry: Permutation Test", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_dir / f"regression_direction_permutation_L{layer}.png",
                dpi=200, bbox_inches="tight")
    plt.close()

    return {
        "real_r": real_r, "real_slope": real_slope,
        "perm_r_mean": np.nanmean(perm_rs), "perm_r_std": np.nanstd(perm_rs),
        "perm_slope_mean": np.nanmean(perm_slopes), "perm_slope_std": np.nanstd(perm_slopes),
        "p_r": p_r, "p_slope": p_slope,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=DEFAULT_LAYER)
    parser.add_argument("--model_id", type=str, default=MODEL_ID)
    parser.add_argument("--n_perms", type=int, default=100)
    parser.add_argument("--semantic_only", action="store_true")
    parser.add_argument("--permutation_only", action="store_true")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = {}

    if not args.permutation_only:
        sem_result = run_semantic_control(args.layer, OUTPUT_DIR, args.model_id)
        if sem_result:
            results["semantic_control"] = sem_result

    if not args.semantic_only:
        perm_result = run_regression_direction_permutation(
            args.layer, OUTPUT_DIR, args.n_perms
        )
        if perm_result:
            results["regression_permutation"] = perm_result

    # Save
    import json
    with open(OUTPUT_DIR / f"controls_results_L{args.layer}.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nAll outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()