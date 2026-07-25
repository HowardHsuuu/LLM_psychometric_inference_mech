#!/usr/bin/env python3
"""
Persona space PCA: unsupervised analysis of how psychometric profiles
are represented in activation space.

B1. Load 272 human subjects → build mega prompts → extract activations → PCA
B3. Dimensionality analysis (how many PCs needed)
C1. Subspace overlap with supervised directions from directions.py

Also:
D1. Subject-level RSA (activation similarity vs behavioral similarity)

Usage:
    python -m psychometric_inference.mechanisms.subject_activations                          # Full pipeline
    python -m psychometric_inference.mechanisms.subject_activations --extract_only            # Just extract activations
    python -m psychometric_inference.mechanisms.subject_activations --analyze_only             # Just analyze (activations already saved)
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.decomposition import PCA

from .config import (
    PROJECT_ROOT, ALL_SUBSCALES, SCALE_NAMES, SUB_TO_SCALE,
    TARGET_LAYERS, DEFAULT_LAYER, RESULTS_DIR, MODEL_ID,
)
from .prompts import load_all_scale_definitions, build_mega_system_prompt, load_human_subjects
from .activation_model import ActivationModel
from .geometry import mantel_test


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = RESULTS_DIR / "pca"


N_PCA_PROBES = 5  # number of probe questions per subject (fewer than directions for speed)


def extract_subject_activations(
    model: ActivationModel,
    subjects: List[Dict],
    scale_defs: dict,
    layers: List[int] = None,
    n_probes: int = N_PCA_PROBES,
) -> Dict[int, np.ndarray]:
    """Extract response-based activations for all subjects.
    
    Uses the same extraction method as directions.py (mean activation across
    generated response tokens) to ensure C1 subspace overlap comparison is valid.
    
    For each subject:
      1. Build mega system prompt from their 114-item profile
      2. Generate responses to n_probes probe questions
      3. Mean-pool response activations across all questions
    
    Args:
        model: ActivationModel instance
        subjects: List of subject dicts from load_human_subjects()
        scale_defs: From load_all_scale_definitions()
        layers: Which layers to extract
        n_probes: Number of probe questions per subject (default 5)
        
    Returns:
        Dict mapping layer_idx -> (n_subjects, hidden_dim) array
    """
    from tqdm import tqdm
    from .probes import PROBE_QUESTIONS

    if layers is None:
        layers = TARGET_LAYERS

    questions = PROBE_QUESTIONS[:n_probes]

    all_acts = {l: [] for l in layers}
    total = len(subjects) * len(questions)

    pbar = tqdm(total=total, desc="Extracting subject activations", unit="response")

    for subj in subjects:
        mega_prompt = build_mega_system_prompt(subj["item_responses"], scale_defs)

        # Collect response activations across probe questions
        subj_acts = {l: [] for l in layers}

        for question in questions:
            acts, _ = model.extract_response_activation(
                mega_prompt, question, layers=layers
            )
            for l in layers:
                if l in acts:
                    subj_acts[l].append(acts[l])
            pbar.update(1)

        # Mean across questions for this subject
        for l in layers:
            if subj_acts[l]:
                mean_act = np.mean(subj_acts[l], axis=0)
                all_acts[l].append(mean_act)

    pbar.close()

    # Stack into arrays
    return {l: np.array(all_acts[l]) for l in layers if all_acts[l]}


def run_pca_analysis(
    activations: np.ndarray,
    subscale_scores: pd.DataFrame,
    layer: int,
    output_dir: Path,
):
    """Run PCA on subject activations and correlate PCs with subscale scores.
    
    Args:
        activations: (n_subjects, hidden_dim) array
        subscale_scores: DataFrame with subscale scores for each subject
        layer: Layer index (for labeling)
        output_dir: Where to save plots
    """
    n_subjects, hidden_dim = activations.shape
    print(f"\n  PCA on {n_subjects} subjects × {hidden_dim} dims (Layer {layer})")

    # Center the data
    acts_centered = activations - activations.mean(axis=0)

    # PCA
    n_components = min(50, n_subjects - 1)
    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(acts_centered)

    # ── B3: Dimensionality analysis ──
    explained = pca.explained_variance_ratio_
    cumulative = np.cumsum(explained)

    print(f"  Variance explained by top PCs:")
    for k in [1, 2, 3, 5, 10, 20]:
        if k <= len(cumulative):
            print(f"    PC1–{k}: {cumulative[k-1]*100:.1f}%")

    n_70 = np.searchsorted(cumulative, 0.70) + 1
    n_90 = np.searchsorted(cumulative, 0.90) + 1
    print(f"  PCs for 70% variance: {n_70}")
    print(f"  PCs for 90% variance: {n_90}")

    # Plot scree
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.bar(range(1, 21), explained[:20] * 100, color="#2c7bb6", alpha=0.8)
    ax1.set_xlabel("Principal Component")
    ax1.set_ylabel("Variance Explained (%)")
    ax1.set_title(f"Scree Plot (Layer {layer})")

    ax2.plot(range(1, len(cumulative)+1), cumulative * 100, "o-", ms=3, lw=1)
    ax2.axhline(y=70, color="red", ls="--", lw=0.8, alpha=0.5, label="70%")
    ax2.axhline(y=90, color="orange", ls="--", lw=0.8, alpha=0.5, label="90%")
    ax2.set_xlabel("Number of PCs")
    ax2.set_ylabel("Cumulative Variance (%)")
    ax2.set_title(f"Cumulative Variance (Layer {layer})")
    ax2.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"scree_L{layer}.png", dpi=200, bbox_inches="tight")
    plt.close()

    # ── Correlate PCs with subscale scores ──
    common_subs = [s for s in ALL_SUBSCALES if s in subscale_scores.columns]
    n_pcs = min(10, n_components)

    corr_matrix = np.zeros((n_pcs, len(common_subs)))
    p_matrix = np.zeros_like(corr_matrix)

    for i in range(n_pcs):
        for j, sub in enumerate(common_subs):
            r, p = pearsonr(scores[:, i], subscale_scores[sub].values)
            corr_matrix[i, j] = r
            p_matrix[i, j] = p

    # Plot PC-subscale correlation heatmap
    short_labels = [s.split("_", 1)[1] if "_" in s else s for s in common_subs]
    pc_labels = [f"PC{i+1}\n({explained[i]*100:.1f}%)" for i in range(n_pcs)]

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(
        corr_matrix, cmap="RdBu_r", center=0, vmin=-0.6, vmax=0.6,
        xticklabels=short_labels, yticklabels=pc_labels,
        annot=True, fmt=".2f", annot_kws={"size": 7}, ax=ax,
        cbar_kws={"shrink": 0.6, "label": "Pearson r"},
    )
    ax.set_title(f"Activation PCs vs Behavioral Subscale Scores (Layer {layer})", fontsize=12)
    ax.tick_params(axis="x", labelsize=7, rotation=45)
    ax.tick_params(axis="y", labelsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / f"pc_subscale_corr_L{layer}.png", dpi=200, bbox_inches="tight")
    plt.close()

    # Save
    corr_df = pd.DataFrame(corr_matrix, index=[f"PC{i+1}" for i in range(n_pcs)],
                           columns=common_subs)
    corr_df.to_csv(output_dir / f"pc_subscale_correlations_L{layer}.csv")

    return {
        "pca_model": pca,
        "scores": scores,
        "explained": explained,
        "n_70pct": n_70,
        "n_90pct": n_90,
        "pc_subscale_corr": corr_matrix,
    }


def run_rsa(
    activations: np.ndarray,
    subscale_scores: pd.DataFrame,
    layer: int,
    output_dir: Path,
):
    """D1: Subject-level RSA.
    
    Compare neural similarity (cosine) with behavioral similarity
    across 272 subjects. Runs both raw and centered versions.
    
    Centered RSA subtracts the mean activation across subjects first,
    removing shared prompt structure and isolating persona-specific signal.
    """
    print(f"\n  RSA (Layer {layer})")

    # Behavioral RDM: correlation between subjects' subscale profiles
    common_subs = [s for s in ALL_SUBSCALES if s in subscale_scores.columns]
    profiles = subscale_scores[common_subs].values  # (n, 16)
    profiles_z = (profiles - profiles.mean(axis=0)) / (profiles.std(axis=0) + 1e-10)
    behavioral_sim = np.corrcoef(profiles_z)

    results = {}

    for label, acts in [("raw", activations),
                         ("centered", activations - activations.mean(axis=0))]:
        # Neural RDM: cosine similarity between subjects
        norms = np.linalg.norm(acts, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        normed = acts / norms
        neural_sim = normed @ normed.T

        r_mantel, p_mantel = mantel_test(neural_sim, behavioral_sim)
        print(f"  {label:>8} RSA: Mantel r = {r_mantel:.4f}, p = {p_mantel:.4f}")
        results[f"rsa_{label}_r"] = r_mantel
        results[f"rsa_{label}_p"] = p_mantel

        # Plot
        n = neural_sim.shape[0]
        idx = np.triu_indices(n, k=1)
        neural_vec = neural_sim[idx]
        behav_vec = behavioral_sim[idx]

        fig, ax = plt.subplots(figsize=(6, 5))
        rng = np.random.default_rng(42)
        sample_idx = rng.choice(len(neural_vec), size=min(5000, len(neural_vec)), replace=False)
        ax.scatter(behav_vec[sample_idx], neural_vec[sample_idx], alpha=0.1, s=5, color="#2c7bb6")
        ax.set_xlabel("Behavioral profile similarity")
        ax.set_ylabel(f"Neural activation similarity ({label})")
        ax.set_title(f"Subject-level RSA — {label} (Layer {layer})\nMantel r={r_mantel:.3f}, p={p_mantel:.4f}")
        plt.tight_layout()
        plt.savefig(output_dir / f"rsa_scatter_{label}_L{layer}.png", dpi=200, bbox_inches="tight")
        plt.close()

    return results


def run_subspace_overlap(
    pca_model,
    layer: int,
    output_dir: Path,
    n_pcs: int = 20,
):
    """C1: Check if supervised directions fall within the PCA subspace.
    
    For each supervised direction, compute the fraction of its variance
    captured by the top-k PCA components.
    """
    directions_path = RESULTS_DIR / "directions" / "subscale_directions.npz"
    if not directions_path.exists():
        print("  Skipping subspace overlap (no supervised directions found)")
        return None

    from .geometry import load_directions

    sub_dirs = load_directions(str(directions_path), ALL_SUBSCALES, layer)
    if not sub_dirs:
        return None

    # PCA components (n_pcs, hidden_dim)
    components = pca_model.components_[:n_pcs]

    print(f"\n  Subspace overlap: supervised dirs vs PCA top-{n_pcs} (Layer {layer})")
    results = {}
    for sub_name, direction in sub_dirs.items():
        # Project direction onto PCA subspace
        projections = components @ direction  # (n_pcs,)
        captured_variance = np.sum(projections ** 2)  # direction is unit norm
        total_variance = np.dot(direction, direction)  # = 1 since normalized
        overlap = captured_variance / total_variance
        results[sub_name] = overlap
        print(f"    {sub_name:<35} overlap = {overlap:.3f}")

    mean_overlap = np.mean(list(results.values()))
    print(f"    {'MEAN':<35} overlap = {mean_overlap:.3f}")

    return results


def run_layerwise_rsa(
    act_path: Path,
    subscale_scores: pd.DataFrame,
    output_dir: Path,
):
    """D2: Layer-wise RSA curve.
    
    Run RSA at every extracted layer to find where persona representations
    best match behavioral structure. Expect peak at middle layers (~50% depth).
    """
    act_data = np.load(act_path)
    available_layers = sorted([int(k[1:]) for k in act_data.keys() if k.startswith("L")])

    if len(available_layers) < 2:
        print("  Skipping layer-wise RSA (need activations from multiple layers)")
        return None

    print(f"\n{'='*60}")
    print(f"  D2: LAYER-WISE RSA")
    print(f"{'='*60}")

    common_subs = [s for s in ALL_SUBSCALES if s in subscale_scores.columns]
    profiles = subscale_scores[common_subs].values
    profiles_z = (profiles - profiles.mean(axis=0)) / (profiles.std(axis=0) + 1e-10)
    behavioral_sim = np.corrcoef(profiles_z)

    results = []
    for layer in available_layers:
        acts = act_data[f"L{layer}"]
        norms = np.linalg.norm(acts, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        normed = acts / norms
        neural_sim = normed @ normed.T

        r_mantel, p_mantel = mantel_test(neural_sim, behavioral_sim, n_permutations=5000)
        results.append({"layer": layer, "rsa_r": r_mantel, "rsa_p": p_mantel})
        print(f"  Layer {layer:>2}: Mantel r = {r_mantel:.4f}, p = {p_mantel:.4f}")

    # Plot
    results_df = pd.DataFrame(results)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(results_df["layer"], results_df["rsa_r"], "o-", ms=7, lw=1.5, color="#2c7bb6")

    # Mark significant layers
    sig = results_df[results_df["rsa_p"] < 0.05]
    if not sig.empty:
        ax.scatter(sig["layer"], sig["rsa_r"], s=80, facecolors="none",
                   edgecolors="red", linewidths=1.5, zorder=5, label="p < 0.05")

    ax.set_xlabel("Layer")
    ax.set_ylabel("RSA Mantel r")
    ax.set_title("Layer-wise RSA: Neural Similarity vs Behavioral Similarity")
    ax.axhline(y=0, color="gray", ls=":", lw=0.5)

    best = results_df.loc[results_df["rsa_r"].idxmax()]
    ax.annotate(f"peak: L{int(best['layer'])} (r={best['rsa_r']:.3f})",
                xy=(best["layer"], best["rsa_r"]),
                xytext=(best["layer"] + 1, best["rsa_r"] + 0.01),
                fontsize=8, color="red")

    if not sig.empty:
        ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / "layerwise_rsa.png", dpi=200, bbox_inches="tight")
    plt.close()

    results_df.to_csv(output_dir / "layerwise_rsa.csv", index=False)
    print(f"\n  Best layer: {int(best['layer'])} (r = {best['rsa_r']:.4f})")

    return results_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--extract_only", action="store_true")
    parser.add_argument("--analyze_only", action="store_true")
    parser.add_argument("--layer", type=int, default=DEFAULT_LAYER)
    parser.add_argument("--model_id", type=str, default=MODEL_ID)
    parser.add_argument("--n_probes", type=int, default=N_PCA_PROBES,
                        help="Number of probe questions per subject for extraction")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    act_path = OUTPUT_DIR / "subject_activations.npz"

    # ── Extraction phase ──
    if not args.analyze_only:
        logger.info("Loading scale definitions...")
        scale_defs = load_all_scale_definitions()

        logger.info("Loading human subjects...")
        subjects = load_human_subjects(scale_defs)
        logger.info(f"Loaded {len(subjects)} subjects")

        # Save subscale scores
        scores_df = pd.DataFrame([s["subscale_scores"] for s in subjects])
        scores_df.insert(0, "subject_id", [s["subject_id"] for s in subjects])
        scores_df.to_csv(OUTPUT_DIR / "human_subscale_scores.csv", index=False)

        logger.info(f"Loading model: {args.model_id}")
        model = ActivationModel(args.model_id)

        logger.info(f"Extracting activations for {len(subjects)} subjects ({args.n_probes} probes each)...")
        activations = extract_subject_activations(model, subjects, scale_defs, n_probes=args.n_probes)

        # Save
        np.savez_compressed(act_path, **{f"L{l}": v for l, v in activations.items()})
        logger.info(f"Saved activations: {act_path}")

        model.cleanup()

        if args.extract_only:
            return

    # ── Analysis phase ──
    logger.info("\n=== ANALYSIS ===")

    # Load activations
    act_data = np.load(act_path)
    scores_df = pd.read_csv(OUTPUT_DIR / "human_subscale_scores.csv")

    layer = args.layer
    key = f"L{layer}"
    if key not in act_data:
        print(f"Layer {layer} not found in saved activations. Available: {list(act_data.keys())}")
        return

    acts = act_data[key]  # (n_subjects, hidden_dim)
    print(f"Activations shape: {acts.shape}")

    # B1 + B3: PCA
    pca_results = run_pca_analysis(acts, scores_df, layer, OUTPUT_DIR)

    # D1: RSA
    rsa_results = run_rsa(acts, scores_df, layer, OUTPUT_DIR)

    # D2: Layer-wise RSA
    layerwise_df = run_layerwise_rsa(act_path, scores_df, OUTPUT_DIR)

    # C1: Subspace overlap
    overlap_results = run_subspace_overlap(pca_results["pca_model"], layer, OUTPUT_DIR)

    # Save summary
    summary = {
        "layer": layer,
        "n_subjects": acts.shape[0],
        "hidden_dim": acts.shape[1],
        "n_pcs_70pct": int(pca_results["n_70pct"]),
        "n_pcs_90pct": int(pca_results["n_90pct"]),
    }
    summary.update(rsa_results)
    with open(OUTPUT_DIR / f"summary_L{layer}.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nAll outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()