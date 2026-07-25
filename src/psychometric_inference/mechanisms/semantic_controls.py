#!/usr/bin/env python3
"""
Semantic similarity controls (representation + behavioral), 14 models,
two embedding sources.

For each model m and each embedding source E ∈ {model_internal, e5}:
    Build subscale-level semantic similarity matrix S_{m,E} (or S_E if shared).
    Pull human matrix H (shared) and per-model behavioral matrix B_m.
    For representation pipeline: pull contrastive direction cosine matrix D_m.

Compute upper-triangle vectors and the following statistics per model:
    Behavioral pipeline:
        r(B_m, H)            zero-order alignment (already known)
        r(S_{m,E}, B_m)      semantic-vs-behavior
        r(S_{m,E}, H)        semantic-vs-human (depends only on m for model_internal)
        partial r(B_m, H | S_{m,E})  the control statistic

    Representation pipeline (analogous, replacing B_m with D_m):
        r(D_m, H)            already known
        r(S_{m,E}, D_m)
        r(S_{m,E}, H)
        partial r(D_m, H | S_{m,E})

Outputs per (m, E) row in two CSVs (one per pipeline).
Also produces two figures, each 2-panel:
    figI3_representation_semantic_control.png
    figH2_behavioral_semantic_control.png

Usage:
    cd /path/to/repo
    python semantic_controls.py

Requires:
    - outputs/behavior/{model_name}/subscale_llm_implicit.csv  for each model
    - outputs/mechanistic/results_{model_name}/directions/subscale_directions.npz  (representation)
      OR
      outputs/mechanistic/results_{model_name}/regression_directions/regression_cosine_sim_L*.csv
    - sentence-transformers (for e5-large) — optional; skipped with warning if missing
    - HuggingFace transformers (for model-internal embedding)
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

from psychometric_inference.model_registry import semantic_control_tuples
from psychometric_inference.paths import (
    BEHAVIOR_OUTPUT_DIR,
    MECHANISTIC_OUTPUT_DIR,
    SEMANTIC_CONTROL_OUTPUT_DIR,
)

RESULTS_IMPLICIT = BEHAVIOR_OUTPUT_DIR
MECHANISTIC_RESULTS = MECHANISTIC_OUTPUT_DIR
OUTPUT_DIR = SEMANTIC_CONTROL_OUTPUT_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Subscale list (must match config.ALL_SUBSCALES)
ALL_SUBSCALES = [
    "IRI_Perspective_taking", "IRI_Fantasy", "IRI_Empathic_concern", "IRI_Personal_distress",
    "PANAS_Positive_Affect", "PANAS_Negative_Affect",
    "POM_Peace_of_Mind",
    "BigFive_Extraversion", "BigFive_Agreeableness", "BigFive_Conscientiousness",
    "BigFive_Neuroticism", "BigFive_Openness",
    "SelfConst_Independent_self", "SelfConst_Interdependent_self",
    "LifeSat_Life_Satisfaction",
    "Lonely_Loneliness",
]

SCALES = [
    ("IRI", "IRI"),
    ("PANAS", "PANAS"),
    ("POM", "POM"),
    ("big_five", "BigFive"),
    ("in_inter_dependent", "SelfConst"),
    ("Life_Satisfaction", "LifeSat"),
    ("Loneliness", "Lonely"),
]

MODELS = semantic_control_tuples()

# ════════════════════════════════════════════════════════════════════
#  Helpers
# ════════════════════════════════════════════════════════════════════

def partial_corr(x, y, covariates):
    """Partial correlation of x and y controlling for covariates (column matrix)."""
    if covariates.shape[1] == 0:
        return pearsonr(x, y)
    C = np.column_stack([covariates, np.ones(len(covariates))])
    beta_x = np.linalg.lstsq(C, x, rcond=None)[0]
    beta_y = np.linalg.lstsq(C, y, rcond=None)[0]
    rx = x - C @ beta_x
    ry = y - C @ beta_y
    return pearsonr(rx, ry)


def upper_tri_vectors(*matrices):
    """Pull aligned upper-triangle vectors from same-shape square matrices.
    Drop pairs with NaN in any matrix.
    """
    n = matrices[0].shape[0]
    for M in matrices:
        assert M.shape == (n, n)
    idx = np.triu_indices(n, k=1)
    vecs = [M[idx] for M in matrices]
    valid = ~np.any(np.stack([np.isnan(v) for v in vecs]), axis=0)
    return tuple(v[valid] for v in vecs)


def reorder_matrix(df: pd.DataFrame, order: list) -> pd.DataFrame:
    """Reorder a square DataFrame to match `order`, only keeping rows/cols present."""
    common = [s for s in order if s in df.index and s in df.columns]
    return df.loc[common, common]


# ════════════════════════════════════════════════════════════════════
#  Load human matrix
# ════════════════════════════════════════════════════════════════════

def load_human_matrix() -> pd.DataFrame:
    """Use mech.geometry helper to compute human 16x16 subscale corr."""
    from psychometric_inference.mechanisms.geometry import compute_human_correlation_matrix
    H = compute_human_correlation_matrix("subscale")
    H = reorder_matrix(H, ALL_SUBSCALES)
    return H


# ════════════════════════════════════════════════════════════════════
#  Load behavioral matrix per model
# ════════════════════════════════════════════════════════════════════

def load_behavioral_matrix(behavior_results_dir: Path) -> pd.DataFrame:
    """Load LLM behavioral subscale-level correlation matrix."""
    p = behavior_results_dir / "subscale_llm_implicit.csv"
    if not p.exists():
        raise FileNotFoundError(f"Behavioral matrix not found: {p}")
    df = pd.read_csv(p, index_col=0)
    df = reorder_matrix(df, ALL_SUBSCALES)
    return df


# ════════════════════════════════════════════════════════════════════
#  Load representation matrix per model
# ════════════════════════════════════════════════════════════════════

def load_representation_matrix(mech_dir: Path) -> tuple:
    """Load contrastive direction cosine similarity matrix for the model.

    Strategy:
      1. Read geometry/geometry_results.csv (one row per (level, layer)).
      2. Pick the subscale-level layer with the highest mantel_r.
      3. Load geometry/subscale_cosine_sim_L{best}.csv.

    Falls back to highest available layer number if geometry_results.csv is missing.

    Returns: (matrix, best_layer) or (None, None) if nothing usable.
    """
    geo_dir = mech_dir / "geometry"
    if not geo_dir.exists():
        return None, None

    # Strategy 1: pick best layer via geometry_results.csv
    geo_csv = geo_dir / "geometry_results.csv"
    best_layer = None
    if geo_csv.exists():
        try:
            gres = pd.read_csv(geo_csv)
            sub = gres[gres["level"] == "subscale"]
            if len(sub):
                best_layer = int(sub.loc[sub["mantel_r"].idxmax(), "layer"])
        except Exception as e:
            warnings.warn(f"Couldn't parse {geo_csv}: {e}")

    # Strategy 2: fall back to highest L number among available CSVs
    if best_layer is None:
        sims = sorted(geo_dir.glob("subscale_cosine_sim_L*.csv"))
        if not sims:
            return None, None
        # pick highest L
        layers = [int(p.stem.split("_L")[-1]) for p in sims]
        best_layer = max(layers)

    target = geo_dir / f"subscale_cosine_sim_L{best_layer}.csv"
    if not target.exists():
        # Final fallback: any subscale_cosine_sim_L*.csv
        sims = sorted(geo_dir.glob("subscale_cosine_sim_L*.csv"))
        if not sims:
            return None, None
        target = sims[-1]
        best_layer = int(target.stem.split("_L")[-1])

    df = pd.read_csv(target, index_col=0)
    return reorder_matrix(df, ALL_SUBSCALES), best_layer


# ════════════════════════════════════════════════════════════════════
#  Build semantic similarity matrices
# ════════════════════════════════════════════════════════════════════

def get_subscale_item_texts() -> dict[str, list[str]]:
    """Return {subscale: [item_texts]} from the project's scale definitions."""
    from psychometric_inference.mechanisms.prompts import load_all_scale_definitions
    from psychometric_inference.scoring import SCORING_RULES

    scale_defs = load_all_scale_definitions()
    out = {}
    for scale_file, scale_short in SCALES:
        sd = scale_defs[scale_file]
        rules = SCORING_RULES.get(scale_short, {})
        for sub_name, item_nums in rules.get("subscales", {}).items():
            sub_full = f"{scale_short}_{sub_name}"
            texts = [it["text"] for it in sd["items"] if it["item_number"] in item_nums]
            if texts:
                out[sub_full] = texts
    return out


def build_semantic_matrix_model_internal(model_id: str) -> pd.DataFrame:
    """Per-model semantic similarity using the model's input embedding layer.
    Replicates controls.compute_item_semantic_similarity.
    """
    import torch
    from transformers import AutoTokenizer, AutoModel
    import gc

    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id, torch_dtype=torch.float16)
    embed = model.get_input_embeddings().weight.detach().float().cpu().numpy()
    del model
    gc.collect()

    item_texts = get_subscale_item_texts()
    sub_emb = {}
    for sub, texts in item_texts.items():
        item_vecs = []
        for t in texts:
            ids = tok.encode(t, add_special_tokens=False)
            if ids:
                item_vecs.append(embed[ids].mean(axis=0))
        if item_vecs:
            sub_emb[sub] = np.mean(item_vecs, axis=0)

    available = [s for s in ALL_SUBSCALES if s in sub_emb]
    n = len(available)
    M = np.zeros((n, n))
    for i, si in enumerate(available):
        for j, sj in enumerate(available):
            vi, vj = sub_emb[si], sub_emb[sj]
            M[i, j] = np.dot(vi, vj) / (np.linalg.norm(vi) * np.linalg.norm(vj) + 1e-10)
    return pd.DataFrame(M, index=available, columns=available)


def build_semantic_matrix_e5() -> pd.DataFrame:
    """Shared semantic similarity using multilingual-e5-large sentence embeddings."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError(
            "sentence-transformers is required for the e5 embedding source. "
            "Install with: pip install sentence-transformers"
        )

    enc = SentenceTransformer("intfloat/multilingual-e5-large")
    item_texts = get_subscale_item_texts()
    # e5 expects "passage: ..." prefix for retrieval-style use; for similarity-only,
    # using a consistent prefix on both sides is fine.
    sub_emb = {}
    for sub, texts in item_texts.items():
        prefixed = [f"passage: {t}" for t in texts]
        emb = enc.encode(prefixed, convert_to_numpy=True, normalize_embeddings=True)
        sub_emb[sub] = emb.mean(axis=0)

    available = [s for s in ALL_SUBSCALES if s in sub_emb]
    n = len(available)
    M = np.zeros((n, n))
    for i, si in enumerate(available):
        for j, sj in enumerate(available):
            vi, vj = sub_emb[si], sub_emb[sj]
            M[i, j] = np.dot(vi, vj) / (np.linalg.norm(vi) * np.linalg.norm(vj) + 1e-10)
    return pd.DataFrame(M, index=available, columns=available)


# ════════════════════════════════════════════════════════════════════
#  Run all 14 models × 2 embedding sources × 2 pipelines
# ════════════════════════════════════════════════════════════════════

def compute_one(target: pd.DataFrame, sem: pd.DataFrame, hum: pd.DataFrame) -> dict:
    """Compute zero-order and partial r for one (target_matrix, semantic_matrix, human_matrix)."""
    common = [s for s in ALL_SUBSCALES
              if s in target.index and s in sem.index and s in hum.index]
    n = len(common)
    if n < 4:
        return {k: np.nan for k in
                ["n", "r_TH", "p_TH", "r_ST", "p_ST", "r_SH", "p_SH",
                 "r_partial", "p_partial"]}

    T = target.loc[common, common].values
    S = sem.loc[common, common].values
    H = hum.loc[common, common].values

    t_v, s_v, h_v = upper_tri_vectors(T, S, H)

    r_TH, p_TH = pearsonr(t_v, h_v)
    r_ST, p_ST = pearsonr(s_v, t_v)
    r_SH, p_SH = pearsonr(s_v, h_v)
    r_partial, p_partial = partial_corr(t_v, h_v, s_v.reshape(-1, 1))

    return {
        "n": n,
        "n_pairs": len(t_v),
        "r_TH": r_TH, "p_TH": p_TH,
        "r_ST": r_ST, "p_ST": p_ST,
        "r_SH": r_SH, "p_SH": p_SH,
        "r_partial": r_partial, "p_partial": p_partial,
        "drop_TH_to_partial": r_TH - r_partial,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip_e5", action="store_true",
                        help="Skip multilingual-e5-large pass (saves time)")
    parser.add_argument("--skip_internal", action="store_true",
                        help="Skip per-model input-embedding pass")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Optional subset of behavioral output directory names")
    args = parser.parse_args()

    print("Loading human correlation matrix...")
    H = load_human_matrix()
    print(f"  Human matrix shape: {H.shape}")

    # Cache shared e5 matrix once
    S_e5 = None
    if not args.skip_e5:
        e5_cache = OUTPUT_DIR / "semantic_matrix_e5.csv"
        if e5_cache.exists():
            print("\nLoading cached multilingual-e5-large semantic matrix...")
            S_e5 = pd.read_csv(e5_cache, index_col=0)
            print(f"  e5 matrix shape: {S_e5.shape}")
        else:
            print("\nBuilding shared multilingual-e5-large semantic matrix...")
            try:
                S_e5 = build_semantic_matrix_e5()
                S_e5.to_csv(e5_cache)
                print(f"  e5 matrix shape: {S_e5.shape}")
            except Exception as e:
                print(f"  e5 build FAILED: {e}; will skip e5 pass")
                S_e5 = None

    rows_beh = []
    rows_rep = []

    models_to_run = MODELS
    if args.models:
        models_to_run = [m for m in MODELS if m[1] in args.models]

    for hf_id, beh_dir, mech_dir, size, family, is_inst in models_to_run:
        print(f"\n=== {beh_dir} ({hf_id}) ===")

        # Behavioral matrix
        try:
            B = load_behavioral_matrix(RESULTS_IMPLICIT / beh_dir)
            print(f"  Behavioral B: {B.shape}")
        except Exception as e:
            print(f"  SKIP (no behavioral matrix): {e}")
            B = None

        # Representation matrix (with best layer)
        D, D_layer = load_representation_matrix(MECHANISTIC_RESULTS / f"results_{mech_dir}")
        if D is None:
            print(f"  No representation matrix found for {mech_dir}; representation pipeline skipped.")
        else:
            print(f"  Representation D: {D.shape}  (best layer: {D_layer})")

        # Semantic — model internal
        S_int = None
        if not args.skip_internal:
            cache_path = OUTPUT_DIR / f"semantic_matrix_internal__{beh_dir}.csv"
            fallback_path = OUTPUT_DIR / f"semantic_matrix_internal__{mech_dir}.csv"
            legacy_v2_path = OUTPUT_DIR / f"semantic_matrix_internal__{beh_dir.replace('_v3', '_v2')}.csv"
            try:
                if cache_path.exists():
                    S_int = pd.read_csv(cache_path, index_col=0)
                    print(f"  Model-internal semantic S cache: {S_int.shape}")
                elif fallback_path.exists():
                    S_int = pd.read_csv(fallback_path, index_col=0)
                    S_int.to_csv(cache_path)
                    print(f"  Model-internal semantic S cache copied from {fallback_path.name}: {S_int.shape}")
                elif legacy_v2_path.exists():
                    S_int = pd.read_csv(legacy_v2_path, index_col=0)
                    S_int.to_csv(cache_path)
                    print(f"  Model-internal semantic S cache copied from {legacy_v2_path.name}: {S_int.shape}")
                else:
                    S_int = build_semantic_matrix_model_internal(hf_id)
                    S_int.to_csv(cache_path)
                    print(f"  Model-internal semantic S: {S_int.shape}")
            except Exception as e:
                print(f"  Model-internal semantic FAILED ({e}); skipping internal for this model")

        # Loop both pipelines × both embedding sources
        for emb_label, S in [("model_internal", S_int), ("e5", S_e5)]:
            if S is None:
                continue
            base = {"model": beh_dir, "hf_id": hf_id, "size_B": size,
                    "family": family, "is_instruct": is_inst, "embedding": emb_label}
            if B is not None:
                rb = compute_one(B, S, H)
                rows_beh.append({**base, **rb})
                print(f"  [BEH/{emb_label}]  r(B,H)={rb['r_TH']:.3f}  partial={rb['r_partial']:.3f}  Δ={rb['drop_TH_to_partial']:+.3f}")
            if D is not None:
                rd = compute_one(D, S, H)
                rows_rep.append({**base, **rd, "best_layer": D_layer})
                print(f"  [REP/{emb_label}]  r(D,H)={rd['r_TH']:.3f}  partial={rd['r_partial']:.3f}  Δ={rd['drop_TH_to_partial']:+.3f}  (L{D_layer})")

    # Save tables
    df_beh = pd.DataFrame(rows_beh)
    df_rep = pd.DataFrame(rows_rep)
    df_beh.to_csv(OUTPUT_DIR / "behavioral_semantic_control.csv", index=False)
    df_rep.to_csv(OUTPUT_DIR / "representation_semantic_control.csv", index=False)
    print(f"\nSaved to {OUTPUT_DIR}")
    print(f"  behavioral_semantic_control.csv: {len(df_beh)} rows")
    print(f"  representation_semantic_control.csv: {len(df_rep)} rows")

    # Plots
    if len(df_beh):
        plot_two_panel(
            df_beh, OUTPUT_DIR / "figH2_behavioral_semantic_control.png",
            target_label="Behavioral", target_symbol="B",
        )
    if len(df_rep):
        plot_two_panel(
            df_rep, OUTPUT_DIR / "figI3_representation_semantic_control.png",
            target_label="Representation", target_symbol="D",
        )


# ════════════════════════════════════════════════════════════════════
#  Plotting
# ════════════════════════════════════════════════════════════════════

def plot_two_panel(df: pd.DataFrame, outpath: Path,
                   target_label: str, target_symbol: str):
    """Two panels: (left) per-model embedding, (right) e5 embedding.
    Each panel: zero-order r vs partial r, as a function of model size,
    coloured by family, marker by base/instruct.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)

    panel_labels = {
        "model_internal": "Model-internal embedding",
        "e5": "multilingual-e5-large",
    }
    panels = ["model_internal", "e5"]

    for ax, emb in zip(axes, panels):
        sub = df[df["embedding"] == emb]
        if not len(sub):
            ax.text(0.5, 0.5, f"No data\n({emb})", ha="center", va="center",
                    transform=ax.transAxes, color="grey")
            ax.set_title(panel_labels[emb])
            continue

        # Sort by size for line plotting
        sub = sub.sort_values("size_B")

        for family, color in [("Llama", "#c0392b"), ("Qwen", "#16a085")]:
            for is_inst, ls, marker in [(False, "-", "s"), (True, "--", "o")]:
                seg = sub[(sub["family"] == family) & (sub["is_instruct"] == is_inst)]
                if not len(seg):
                    continue
                lbl_kind = "Instruct" if is_inst else "Base"
                ax.plot(seg["size_B"], seg["r_TH"],
                        color=color, ls=ls, marker=marker, ms=7, alpha=0.45,
                        label=f"{family} {lbl_kind} (zero-order)")
                ax.plot(seg["size_B"], seg["r_partial"],
                        color=color, ls=ls, marker=marker, ms=7, mfc="white",
                        label=f"{family} {lbl_kind} (partial)")

        ax.set_xscale("log")
        ax.set_xlabel("Model size (B params)")
        ax.set_title(panel_labels[emb])
        ax.axhline(0, color="grey", lw=0.5, ls=":")
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel(f"$r({target_symbol},H)$  /  partial $r({target_symbol},H \\mid S)$")

    # Single legend, deduped
    handles, labels = axes[0].get_legend_handles_labels()
    seen, hh, ll = set(), [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            seen.add(l)
            hh.append(h); ll.append(l)
    fig.legend(hh, ll, loc="lower center", ncol=4, fontsize=7,
               bbox_to_anchor=(0.5, -0.05), frameon=False)

    plt.suptitle(
        f"{target_label} semantic similarity control across 14 models\n"
        f"Filled markers: zero-order $r({target_symbol},H)$. "
        f"Open markers: partial $r({target_symbol},H \\mid S)$.",
        y=1.02, fontsize=10,
    )
    plt.tight_layout()
    plt.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved figure: {outpath}")


if __name__ == "__main__":
    main()
