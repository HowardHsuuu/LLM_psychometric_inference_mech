#!/usr/bin/env python3
"""
Robustness and sensitivity analyses.
====================================

This script runs cached CPU analyses and analyses based on saved activations.
GPU-heavy behavior and activation extensions are exposed through
scripts/behavior/run_prompt_sensitivity.py and
psychometric_inference.mechanisms.steering.

Experiments (this file):

  T0 (minutes, uses existing CSV summaries):
    C1  Within-group correlation (instruct vs base, 7 each)
    C2  Partial correlation Mantel r vs beh slope | log(size)
    F1  Fixed-depth (60%) slope sensitivity vs post-hoc best layer
    F3  Variance decomposition of amplification by pair type
    D2  Cultural stratification (stable / ambiguous / specific subscales)

  T1 (CPU, hours):
    A1_minimal  Synthetic item-level ridge inflation check
    A1_full     Synthetic high-dim SNR sweep

  T2 (CPU, uses saved activations):
    A2          Alternative probes (Lasso / PLS / Linear SVR) vs Ridge
    A3          Multi-target vs single-target ridge (reduced-rank, PLS-2)
    B2          Item-level discretization-loss decomposition
    F2          Additional base-instruct comparisons (already-extracted data)

Usage
-----
    cd <repo-root>
    python scripts/analysis/compute_robustness_suite.py                   # run everything (resume)
    python scripts/analysis/compute_robustness_suite.py --only C1 C2
    python scripts/analysis/compute_robustness_suite.py --skip A1_full    # skip slow one
    python scripts/analysis/compute_robustness_suite.py --force A1_full   # force rerun even if done
    python scripts/analysis/compute_robustness_suite.py --dry-run         # list what would run
    python scripts/analysis/compute_robustness_suite.py --list            # list all experiments
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from psychometric_inference.paths import (
    HUMAN_DATA_DIR,
    MECHANISTIC_DEFAULT_RESULTS_DIR,
    MECHANISTIC_OUTPUT_DIR,
    PROJECT_ROOT,
    ROBUSTNESS_OUTPUT_DIR,
)

# ─────────────────────────────────────────────────────────────────────────
#  Paths (relative to repo root)
# ─────────────────────────────────────────────────────────────────────────

REPO_ROOT = PROJECT_ROOT

OUTPUT_ROOT = ROBUSTNESS_OUTPUT_DIR
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
DONE_DIR = OUTPUT_ROOT / ".done"
DONE_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = OUTPUT_ROOT / "run.log"

MECHANISTIC_ROOT = MECHANISTIC_OUTPUT_DIR

# ─────────────────────────────────────────────────────────────────────────
#  Model metadata (kept in sync with model_registry.py and scaling_summary.csv)
# ─────────────────────────────────────────────────────────────────────────

MODELS: List[Dict] = [
    # name                    family  mtype       size   hidden_dim  n_layers
    dict(name="llama1b_instruct", family="llama", mtype="instruct", size=1.0,  hidden_dim=2048, n_layers=16),
    dict(name="llama1b_base",     family="llama", mtype="base",     size=1.0,  hidden_dim=2048, n_layers=16),
    dict(name="llama3b_instruct", family="llama", mtype="instruct", size=3.0,  hidden_dim=3072, n_layers=28),
    dict(name="llama3b_base",     family="llama", mtype="base",     size=3.0,  hidden_dim=3072, n_layers=28),
    dict(name="llama8b_instruct", family="llama", mtype="instruct", size=8.0,  hidden_dim=4096, n_layers=32),
    dict(name="llama8b_base",     family="llama", mtype="base",     size=8.0,  hidden_dim=4096, n_layers=32),
    dict(name="qwen05b_instruct", family="qwen",  mtype="instruct", size=0.5,  hidden_dim=896,  n_layers=24),
    dict(name="qwen05b_base",     family="qwen",  mtype="base",     size=0.5,  hidden_dim=896,  n_layers=24),
    dict(name="qwen3b_instruct",  family="qwen",  mtype="instruct", size=3.0,  hidden_dim=2048, n_layers=36),
    dict(name="qwen3b_base",      family="qwen",  mtype="base",     size=3.0,  hidden_dim=2048, n_layers=36),
    dict(name="qwen7b_instruct",  family="qwen",  mtype="instruct", size=7.0,  hidden_dim=3584, n_layers=28),
    dict(name="qwen7b_base",      family="qwen",  mtype="base",     size=7.0,  hidden_dim=3584, n_layers=28),
    dict(name="qwen14b_instruct", family="qwen",  mtype="instruct", size=14.0, hidden_dim=5120, n_layers=48),
    dict(name="qwen14b_base",     family="qwen",  mtype="base",     size=14.0, hidden_dim=5120, n_layers=48),
]
MODEL_BY_NAME = {m["name"]: m for m in MODELS}

# Models with cross_persona/ (needed for B2)
MODELS_WITH_CROSS_PERSONA = [
    "llama1b_instruct", "llama3b_instruct", "llama8b_instruct", "llama8b_base",
    "qwen05b_instruct", "qwen3b_instruct", "qwen7b_instruct",
    "qwen14b_instruct", "qwen14b_base",
]

# ─────────────────────────────────────────────────────────────────────────
#  Subscale / scale naming (mirror mech.config.ALL_SUBSCALES)
# ─────────────────────────────────────────────────────────────────────────

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
SCALE_NAMES = ["IRI", "PANAS", "POM", "BigFive", "SelfConst", "LifeSat", "Lonely"]
SUB_TO_SCALE = {
    "IRI_Perspective_taking": "IRI", "IRI_Fantasy": "IRI",
    "IRI_Empathic_concern": "IRI", "IRI_Personal_distress": "IRI",
    "PANAS_Positive_Affect": "PANAS", "PANAS_Negative_Affect": "PANAS",
    "POM_Peace_of_Mind": "POM",
    "BigFive_Extraversion": "BigFive", "BigFive_Agreeableness": "BigFive",
    "BigFive_Conscientiousness": "BigFive", "BigFive_Neuroticism": "BigFive",
    "BigFive_Openness": "BigFive",
    "SelfConst_Independent_self": "SelfConst",
    "SelfConst_Interdependent_self": "SelfConst",
    "LifeSat_Life_Satisfaction": "LifeSat",
    "Lonely_Loneliness": "Lonely",
}

# D2 cultural grouping used for cross-cultural sensitivity checks
#   STABLE     — robust cross-cultural replication (Big Five E/A/C/N; IRI EC/PT;
#                PANAS NA; SWLS)
#   AMBIGUOUS  — mixed evidence (BFI Openness in East Asian factor structures;
#                PANAS PA due to dialectical PA/NA; UCLA Loneliness;
#                IRI Personal Distress)
#   SPECIFIC   — Eastern-developed or East-Asia-unique factors (POM, SCS Indep,
#                SCS Interdep, IRI Fantasy)
CULTURE_GROUPS = {
    "stable": [
        "BigFive_Extraversion", "BigFive_Agreeableness",
        "BigFive_Conscientiousness", "BigFive_Neuroticism",
        "IRI_Empathic_concern", "IRI_Perspective_taking",
        "PANAS_Negative_Affect",
        "LifeSat_Life_Satisfaction",
    ],
    "ambiguous": [
        "BigFive_Openness",
        "PANAS_Positive_Affect",
        "Lonely_Loneliness",
        "IRI_Personal_distress",
        "IRI_Fantasy",
    ],
    "specific": [
        "POM_Peace_of_Mind",
        "SelfConst_Independent_self",
        "SelfConst_Interdependent_self",
    ],
}


# ─────────────────────────────────────────────────────────────────────────
#  Logging / banner
# ─────────────────────────────────────────────────────────────────────────

logger = logging.getLogger("extra")


def setup_logger(verbose: bool = True):
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)

    fh = logging.FileHandler(LOG_FILE, mode="a")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(sh)


def banner(title: str, char: str = "=", width: int = 72):
    line = char * width
    logger.info("")
    logger.info(line)
    logger.info(f"  {title}")
    logger.info(line)


# ─────────────────────────────────────────────────────────────────────────
#  Done-flag + experiment dispatch
# ─────────────────────────────────────────────────────────────────────────

def done_path(exp_id: str) -> Path:
    return DONE_DIR / f"{exp_id}.done"


def is_done(exp_id: str) -> bool:
    return done_path(exp_id).exists()


def mark_done(exp_id: str, meta: Optional[dict] = None):
    payload = {"completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
               "meta": meta or {}}
    done_path(exp_id).write_text(json.dumps(payload, indent=2, default=str))


def clear_done(exp_id: str):
    p = done_path(exp_id)
    if p.exists():
        p.unlink()


def exp_outdir(exp_id: str) -> Path:
    d = OUTPUT_ROOT / exp_id
    d.mkdir(parents=True, exist_ok=True)
    return d


# ─────────────────────────────────────────────────────────────────────────
#  Math utilities
# ─────────────────────────────────────────────────────────────────────────

def mantel_test(matrix_a: np.ndarray, matrix_b: np.ndarray,
                n_permutations: int = 10000, seed: int = 42
                ) -> Tuple[float, float]:
    """Mantel test on upper triangle (symmetric matrices)."""
    n = matrix_a.shape[0]
    assert matrix_a.shape == matrix_b.shape, "Matrices must match"
    idx = np.triu_indices(n, k=1)
    a_vec = matrix_a[idx]
    b_vec = matrix_b[idx]
    valid = ~(np.isnan(a_vec) | np.isnan(b_vec))
    a_v = a_vec[valid]
    b_v = b_vec[valid]
    if len(a_v) < 3:
        return np.nan, np.nan
    observed = np.corrcoef(a_v, b_v)[0, 1]
    rng = np.random.default_rng(seed)
    n_ge = 0
    for _ in range(n_permutations):
        perm = rng.permutation(n)
        a_perm = matrix_a[np.ix_(perm, perm)]
        a_perm_v = a_perm[idx][valid]
        r_perm = np.corrcoef(a_perm_v, b_v)[0, 1]
        if r_perm >= observed:
            n_ge += 1
    p = (n_ge + 1) / (n_permutations + 1)
    return observed, p


def partial_corr(x: np.ndarray, y: np.ndarray, z: np.ndarray
                 ) -> Tuple[float, float]:
    """Partial correlation r(x, y | z) via residualization.
    Returns (partial_r, two-sided p-value using Fisher z with df = n - 3).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float).reshape(-1, 1) if z.ndim == 1 else np.asarray(z, dtype=float)

    valid = ~(np.isnan(x) | np.isnan(y) | np.any(np.isnan(z), axis=1))
    x = x[valid]; y = y[valid]; z = z[valid]
    n = len(x)
    if n < 4:
        return np.nan, np.nan

    # Regress x on z, y on z
    zc = np.column_stack([np.ones(n), z])
    bx, *_ = np.linalg.lstsq(zc, x, rcond=None)
    by, *_ = np.linalg.lstsq(zc, y, rcond=None)
    rx = x - zc @ bx
    ry = y - zc @ by

    r = np.corrcoef(rx, ry)[0, 1]

    # Fisher z
    if abs(r) >= 1:
        return r, 0.0
    k = z.shape[1]  # number of controls
    df = n - 2 - k
    if df < 1:
        return r, np.nan
    z_stat = np.arctanh(r) * np.sqrt(df - 1)
    from math import erfc, sqrt as _sqrt
    p = erfc(abs(z_stat) / _sqrt(2))
    return float(r), float(p)


def bootstrap_ci(values: np.ndarray, stat_fn: Callable[[np.ndarray], float],
                 n_boot: int = 5000, ci: float = 0.95, seed: int = 42
                 ) -> Tuple[float, float, float]:
    """Bootstrap CI for a univariate statistic. Returns (point, lo, hi)."""
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) < 3:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    point = stat_fn(v)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, len(v), size=len(v))
        boots[i] = stat_fn(v[idx])
    lo = np.nanpercentile(boots, (1 - ci) / 2 * 100)
    hi = np.nanpercentile(boots, (1 + ci) / 2 * 100)
    return float(point), float(lo), float(hi)


def bootstrap_pearson_r(x: np.ndarray, y: np.ndarray, n_boot: int = 5000,
                        ci: float = 0.95, seed: int = 42
                        ) -> Tuple[float, float, float]:
    """Bootstrap CI for pearson r between paired vectors."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = ~(np.isnan(x) | np.isnan(y))
    x = x[valid]; y = y[valid]
    if len(x) < 3:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    point = np.corrcoef(x, y)[0, 1]
    boots = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, len(x), size=len(x))
        boots[i] = np.corrcoef(x[idx], y[idx])[0, 1]
    lo = np.nanpercentile(boots, (1 - ci) / 2 * 100)
    hi = np.nanpercentile(boots, (1 + ci) / 2 * 100)
    return float(point), float(lo), float(hi)


def loo_correlation(x: np.ndarray, y: np.ndarray
                    ) -> Dict[str, float]:
    """Leave-one-out sensitivity of pearson r. Returns min/max/mean/std."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = ~(np.isnan(x) | np.isnan(y))
    x = x[valid]; y = y[valid]
    n = len(x)
    if n < 4:
        return dict(loo_min=np.nan, loo_max=np.nan, loo_mean=np.nan, loo_std=np.nan)
    rs = []
    for i in range(n):
        mask = np.ones(n, dtype=bool); mask[i] = False
        rs.append(np.corrcoef(x[mask], y[mask])[0, 1])
    rs = np.array(rs)
    return dict(loo_min=float(rs.min()), loo_max=float(rs.max()),
                loo_mean=float(rs.mean()), loo_std=float(rs.std()))


def pair_in_same_scale(sub_i: str, sub_j: str) -> bool:
    return SUB_TO_SCALE.get(sub_i) == SUB_TO_SCALE.get(sub_j)


# ─────────────────────────────────────────────────────────────────────────
#  Data loaders
# ─────────────────────────────────────────────────────────────────────────

def load_structure_vs_behavior() -> pd.DataFrame:
    path = MECHANISTIC_ROOT / "structure_vs_behavior.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. This file is produced by "
            f"psychometric_inference.mechanisms.representation_behavior and must exist before running C1/C2/F3/D2."
        )
    df = pd.read_csv(path)
    df["log_size"] = np.log10(df["size"])
    return df


def load_scaling_summary() -> pd.DataFrame:
    path = MECHANISTIC_ROOT / "scaling_summary.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")
    return pd.read_csv(path)


def load_human_subscale_corr() -> pd.DataFrame:
    """16x16 human subscale correlation matrix. Uses the canonical source
    baked into each model's geometry/ folder (human_subscale_corr.csv).
    """
    # Any model's human_subscale_corr is identical (same ground truth).
    # Prefer llama8b_instruct for consistency with paper main analyses.
    candidates = [
        MECHANISTIC_ROOT / "results_llama8b_instruct" / "geometry" / "human_subscale_corr.csv",
        MECHANISTIC_DEFAULT_RESULTS_DIR / "geometry" / "human_subscale_corr.csv",
    ]
    for p in candidates:
        if p.exists():
            df = pd.read_csv(p, index_col=0)
            # Reorder to canonical ALL_SUBSCALES order
            rows = [s for s in ALL_SUBSCALES if s in df.index]
            return df.loc[rows, rows]
    # Fall back: compute from psychometric_inference.scoring + real data at repo root
    logger.warning("human_subscale_corr.csv not found in outputs/mechanistic/results_*; "
                   "fallback: computing from psychometric_inference.mechanisms.geometry")
    try:
        from psychometric_inference.mechanisms.geometry import compute_human_correlation_matrix  # type: ignore
        return compute_human_correlation_matrix("subscale")
    except Exception as e:
        raise FileNotFoundError(
            f"Cannot load human 16x16 subscale correlation matrix "
            f"and fallback failed: {e}"
        )


def load_model_best_layer(model_name: str) -> int:
    """Best layer for a model from scaling_summary.csv.
    df = load_scaling_summary()
    row = df[df["model"] == model_name]
    if row.empty:
        raise ValueError(f"Model {model_name} not in scaling_summary.csv")
    return int(row.iloc[0]["best_layer"])
    """
    """Best layer for cross-persona pipeline (Mantel r maximizing).
    
    Unlike load_model_best_layer (which reads geometry best from
    scaling_summary.csv), this scans the model's cross_persona/ dir for
    all predicted_corr_matrix_L*.csv files, computes Mantel r vs human
    subscale matrix for each, and returns the layer maximizing it.
    
    This matches the "best layer" reported in paper Table 2.
    """
    cp_dir = MECHANISTIC_ROOT / f"results_{model_name}" / "cross_persona"
    if not cp_dir.exists() and model_name == "llama8b_instruct":
        cp_dir = MECHANISTIC_DEFAULT_RESULTS_DIR / "cross_persona"
    if not cp_dir.exists():
        raise ValueError(f"No cross_persona dir for {model_name}")

    human = load_human_subscale_corr()
    best_layer = None
    best_mantel = -np.inf

    for f in sorted(cp_dir.glob("predicted_corr_matrix_L*.csv")):
        try:
            layer = int(f.stem.split("_L")[-1])
        except ValueError:
            continue
        try:
            pred = pd.read_csv(f, index_col=0)
        except Exception:
            continue
        common = [s for s in ALL_SUBSCALES if s in pred.index and s in human.index]
        if len(common) < 4:
            continue
        n = len(common)
        mask = np.triu_indices(n, k=1)
        h_vec = human.loc[common, common].values[mask]
        p_vec = pred.loc[common, common].values[mask]
        valid = ~(np.isnan(h_vec) | np.isnan(p_vec))
        if valid.sum() < 3:
            continue
        r = np.corrcoef(h_vec[valid], p_vec[valid])[0, 1]
        if r > best_mantel:
            best_mantel = r
            best_layer = layer

    if best_layer is None:
        raise ValueError(f"No valid prediction file for {model_name}")
    return best_layer


def load_model_geometry_results(model_name: str) -> pd.DataFrame:
    """Layer-wise geometry results for one model (from geometry/geometry_results.csv)."""
    p = MECHANISTIC_ROOT / f"results_{model_name}" / "geometry" / "geometry_results.csv"
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)


# ─────────────────────────────────────────────────────────────────────────
#  Experiment: C1  Within-group correlation
# ─────────────────────────────────────────────────────────────────────────

def run_C1(outdir: Path) -> Dict:
    """Within-instruct / within-base correlation of repr_mantel_r vs beh_subscale_slope.

    Paper's r=0.68 (p=0.008) across all 14 models: is this just instruct/base
    dichotomy? Test within each group.
    """
    df = load_structure_vs_behavior()

    results = {}
    for grp in ["instruct", "base"]:
        sub = df[df["mtype"] == grp]
        x = sub["repr_mantel_r"].values
        y = sub["beh_subscale_slope"].values
        point_r, lo, hi = bootstrap_pearson_r(x, y, n_boot=10000)
        loo = loo_correlation(x, y)
        if len(x) >= 3:
            r_sp, p_sp = spearmanr(x, y)
        else:
            r_sp, p_sp = np.nan, np.nan
        # Analytic p
        if len(x) >= 3:
            r_an, p_an = pearsonr(x, y)
        else:
            r_an, p_an = np.nan, np.nan

        results[grp] = {
            "n": int(len(x)),
            "models": sub["model"].tolist(),
            "pearson_r": float(r_an),
            "pearson_p": float(p_an),
            "spearman_r": float(r_sp),
            "spearman_p": float(p_sp),
            "bootstrap_r": point_r,
            "bootstrap_ci_lo": lo,
            "bootstrap_ci_hi": hi,
            **loo,
        }

    # Also: Fisher's test for whether r_instruct == r_base
    r1 = results["instruct"]["pearson_r"]
    r2 = results["base"]["pearson_r"]
    n1 = results["instruct"]["n"]
    n2 = results["base"]["n"]
    if not np.isnan(r1) and not np.isnan(r2) and n1 > 3 and n2 > 3:
        z1 = np.arctanh(r1); z2 = np.arctanh(r2)
        se = np.sqrt(1 / (n1 - 3) + 1 / (n2 - 3))
        z = (z1 - z2) / se
        from math import erfc, sqrt as _sqrt
        p_diff = erfc(abs(z) / _sqrt(2))
        results["difference_test"] = {
            "fisher_z": float(z),
            "two_sided_p": float(p_diff),
            "r_instruct_minus_r_base": float(r1 - r2),
        }

    # All-14 reference
    x_all = df["repr_mantel_r"].values
    y_all = df["beh_subscale_slope"].values
    r_all, p_all = pearsonr(x_all, y_all) if len(x_all) >= 3 else (np.nan, np.nan)
    results["all_14_reference"] = {
        "n": int(len(x_all)),
        "pearson_r": float(r_all),
        "pearson_p": float(p_all),
    }

    # Save
    (outdir / "C1_within_group.json").write_text(
        json.dumps(results, indent=2, default=str))

    # CSV: one row per group
    rows = []
    for grp in ["instruct", "base"]:
        r = results[grp]
        rows.append({
            "group": grp,
            "n": r["n"],
            "pearson_r": r["pearson_r"], "pearson_p": r["pearson_p"],
            "spearman_r": r["spearman_r"], "spearman_p": r["spearman_p"],
            "boot_ci_lo": r["bootstrap_ci_lo"], "boot_ci_hi": r["bootstrap_ci_hi"],
            "loo_min": r["loo_min"], "loo_max": r["loo_max"],
        })
    rows.append({
        "group": "all_14",
        "n": results["all_14_reference"]["n"],
        "pearson_r": results["all_14_reference"]["pearson_r"],
        "pearson_p": results["all_14_reference"]["pearson_p"],
    })
    pd.DataFrame(rows).to_csv(outdir / "C1_within_group.csv", index=False)

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharex=True, sharey=True)
        for ax, grp, color in zip(axes, ["instruct", "base"], ["#c0392b", "#2471a3"]):
            sub = df[df["mtype"] == grp]
            ax.scatter(sub["repr_mantel_r"], sub["beh_subscale_slope"],
                       color=color, s=100, edgecolors="white", linewidth=1.5, zorder=3)
            for _, row in sub.iterrows():
                ax.annotate(f"{row['size']:g}B {row['family'][0].upper()}",
                            (row["repr_mantel_r"], row["beh_subscale_slope"]),
                            fontsize=7, xytext=(6, 4), textcoords="offset points")
            r = results[grp]
            ax.set_title(
                f"{grp}  (n={r['n']})\n"
                f"r = {r['pearson_r']:.3f}, p = {r['pearson_p']:.3f}\n"
                f"95% CI [{r['bootstrap_ci_lo']:.2f}, {r['bootstrap_ci_hi']:.2f}]",
                fontsize=10
            )
            ax.set_xlabel("Representational Mantel r")
            ax.grid(alpha=0.2)
        axes[0].set_ylabel("Behavioral subscale slope")
        plt.suptitle(
            f"C1: Within-group Mantel r ↔ behavioral slope\n"
            f"All 14: r = {results['all_14_reference']['pearson_r']:.3f}  (paper Fig 5)",
            fontsize=11
        )
        plt.tight_layout()
        plt.savefig(outdir / "C1_within_group.png", dpi=200, bbox_inches="tight")
        plt.close()
    except Exception as e:
        logger.warning(f"  C1 plot failed: {e}")

    # Console summary
    logger.info(f"  All 14:    r = {results['all_14_reference']['pearson_r']:.3f}, "
                f"p = {results['all_14_reference']['pearson_p']:.4f}")
    for grp in ["instruct", "base"]:
        r = results[grp]
        logger.info(f"  {grp:<8} (n={r['n']}): "
                    f"r = {r['pearson_r']:+.3f}, "
                    f"p = {r['pearson_p']:.4f}, "
                    f"95% CI [{r['bootstrap_ci_lo']:+.2f}, {r['bootstrap_ci_hi']:+.2f}]")
    if "difference_test" in results:
        d = results["difference_test"]
        logger.info(f"  Fisher z test (instruct vs base): "
                    f"Δr = {d['r_instruct_minus_r_base']:+.3f}, "
                    f"z = {d['fisher_z']:+.3f}, p = {d['two_sided_p']:.4f}")

    return results


# ─────────────────────────────────────────────────────────────────────────
#  Experiment: C2  Partial correlation controlling for log(size)
# ─────────────────────────────────────────────────────────────────────────

def run_C2(outdir: Path) -> Dict:
    """Partial r(repr_mantel_r, beh_subscale_slope | log10(size)) across 14 models.

    Also reports controls for {mtype (instruct/base dummy), family (llama/qwen dummy)}.
    Robustness question: is r=0.68 explained by model size?
    """
    df = load_structure_vs_behavior()
    x = df["repr_mantel_r"].values
    y = df["beh_subscale_slope"].values

    # Control sets
    controls = {
        "log_size": df[["log_size"]].values,
        "log_size+mtype": np.column_stack([
            df["log_size"].values,
            (df["mtype"] == "instruct").astype(int).values,
        ]),
        "log_size+mtype+family": np.column_stack([
            df["log_size"].values,
            (df["mtype"] == "instruct").astype(int).values,
            (df["family"] == "qwen").astype(int).values,
        ]),
    }

    results = {"zero_order": {}, "partial": {}}

    # Zero-order
    r0, p0 = pearsonr(x, y)
    r0_pt, r0_lo, r0_hi = bootstrap_pearson_r(x, y)
    results["zero_order"] = {
        "n": len(x), "r": float(r0), "p": float(p0),
        "boot_ci_lo": r0_lo, "boot_ci_hi": r0_hi,
    }

    for ctl_name, z in controls.items():
        pr, pp = partial_corr(x, y, z)
        results["partial"][ctl_name] = {
            "controls": ctl_name,
            "n_controls": int(z.shape[1] if z.ndim == 2 else 1),
            "partial_r": float(pr),
            "partial_p": float(pp),
        }

    # Also cross-verify using pingouin if available (independent impl)
    try:
        import pingouin as pg
        d = df[["repr_mantel_r", "beh_subscale_slope", "log_size"]].copy()
        pr_pg = pg.partial_corr(data=d, x="repr_mantel_r", y="beh_subscale_slope",
                                covar="log_size", method="pearson")
        results["partial"]["log_size"]["pingouin_r"] = float(pr_pg["r"].iloc[0])
        results["partial"]["log_size"]["pingouin_p"] = float(pr_pg["p-val"].iloc[0])
    except ImportError:
        logger.warning("  (pingouin not installed; skipping cross-verify. "
                       "`pip install pingouin` if you want it.)")

    (outdir / "C2_partial.json").write_text(
        json.dumps(results, indent=2, default=str))

    rows = [{
        "test": "zero_order",
        "controls": "",
        "r": results["zero_order"]["r"],
        "p": results["zero_order"]["p"],
        "n": results["zero_order"]["n"],
    }]
    for k, v in results["partial"].items():
        rows.append({
            "test": "partial",
            "controls": v["controls"],
            "r": v["partial_r"],
            "p": v["partial_p"],
            "n": results["zero_order"]["n"],
        })
    pd.DataFrame(rows).to_csv(outdir / "C2_partial.csv", index=False)

    logger.info(f"  Zero-order r = {results['zero_order']['r']:.3f}, "
                f"p = {results['zero_order']['p']:.4f}  "
                f"(95% CI [{results['zero_order']['boot_ci_lo']:.2f}, "
                f"{results['zero_order']['boot_ci_hi']:.2f}])")
    for ctl, v in results["partial"].items():
        logger.info(f"  Partial r | {ctl:<25} = {v['partial_r']:+.3f}, "
                    f"p = {v['partial_p']:.4f}")

    return results


# ─────────────────────────────────────────────────────────────────────────
#  Experiment: F1  Fixed-depth sensitivity
# ─────────────────────────────────────────────────────────────────────────

def run_F1(outdir: Path) -> Dict:
    """Re-report Table 2 slopes/Mantel r at fixed 60% depth vs post-hoc best layer.

    Source: each model's outputs/mechanistic/results_{name}/geometry/geometry_results.csv
    (layer-wise subscale-level mantel_r and slope).
    """
    summary = load_scaling_summary()
    rows = []
    per_model = {}

    for _, row in summary.iterrows():
        name = row["model"]
        best = int(row["best_layer"])
        n_layers = int(row["n_layers"])
        fixed_layer = round(0.60 * n_layers)

        geom = load_model_geometry_results(name)
        if geom.empty:
            rows.append({"model": name, "error": "no_geometry_results"})
            continue

        sub = geom[geom["level"] == "subscale"]
        if sub.empty:
            rows.append({"model": name, "error": "no_subscale_level"})
            continue

        avail_layers = sorted(sub["layer"].unique().tolist())
        # Find nearest available layer to 60% depth
        nearest = min(avail_layers, key=lambda x_: abs(x_ - fixed_layer))

        at_best = sub[sub["layer"] == best]
        at_fixed = sub[sub["layer"] == nearest]

        def _safe(s, col):
            return float(s[col].iloc[0]) if not s.empty else np.nan

        r_best = _safe(at_best, "mantel_r")
        slope_best = _safe(at_best, "slope")
        r_fix = _safe(at_fixed, "mantel_r")
        slope_fix = _safe(at_fixed, "slope")

        rows.append({
            "model": name,
            "family": row["family"], "mtype": row["mtype"],
            "n_layers": n_layers,
            "best_layer": best,
            "fixed_60pct_target": fixed_layer,
            "fixed_60pct_nearest": nearest,
            "mantel_r_best": r_best,
            "mantel_r_fixed": r_fix,
            "mantel_r_delta": r_fix - r_best if not np.isnan(r_fix - r_best) else np.nan,
            "slope_best": slope_best,
            "slope_fixed": slope_fix,
            "slope_delta": slope_fix - slope_best if not np.isnan(slope_fix - slope_best) else np.nan,
        })
        per_model[name] = {"layers_available": avail_layers}

    res_df = pd.DataFrame(rows)
    res_df.to_csv(outdir / "F1_fixed_depth.csv", index=False)

    # Aggregate
    if "mantel_r_delta" in res_df.columns:
        summary_stats = {
            "mantel_r_best_mean": float(res_df["mantel_r_best"].mean()),
            "mantel_r_fixed_mean": float(res_df["mantel_r_fixed"].mean()),
            "mantel_r_delta_mean": float(res_df["mantel_r_delta"].mean()),
            "mantel_r_delta_abs_mean": float(res_df["mantel_r_delta"].abs().mean()),
            "slope_best_mean": float(res_df["slope_best"].mean()),
            "slope_fixed_mean": float(res_df["slope_fixed"].mean()),
            "slope_delta_mean": float(res_df["slope_delta"].mean()),
            "slope_delta_abs_mean": float(res_df["slope_delta"].abs().mean()),
        }
    else:
        summary_stats = {}

    (outdir / "F1_fixed_depth.json").write_text(
        json.dumps({"per_model": per_model, "summary": summary_stats,
                    "rows": rows}, indent=2, default=str))

    logger.info(f"  {'model':<22} {'best L':>7} {'60% L':>7} "
                f"{'mantel(b)':>10} {'mantel(f)':>10} {'Δmantel':>10} "
                f"{'slope(b)':>10} {'slope(f)':>10}")
    for _, r in res_df.iterrows():
        if "error" in r and isinstance(r.get("error"), str):
            continue
        logger.info(
            f"  {r['model']:<22} {r['best_layer']:>7} {r['fixed_60pct_nearest']:>7} "
            f"{r['mantel_r_best']:>10.3f} {r['mantel_r_fixed']:>10.3f} "
            f"{r['mantel_r_delta']:>+10.3f} "
            f"{r['slope_best']:>10.3f} {r['slope_fixed']:>10.3f}")
    if summary_stats:
        logger.info(f"\n  Mean |Δ mantel r| across 14 models: "
                    f"{summary_stats['mantel_r_delta_abs_mean']:.3f}")
        logger.info(f"  Mean |Δ slope|     across 14 models: "
                    f"{summary_stats['slope_delta_abs_mean']:.3f}")

    return {"rows": rows, "summary": summary_stats}


# ─────────────────────────────────────────────────────────────────────────
#  Experiment: F3  Variance decomposition by pair type
# ─────────────────────────────────────────────────────────────────────────

def _pair_type(sub_i: str, sub_j: str) -> str:
    """Classify a subscale pair for F3 variance decomposition."""
    if SUB_TO_SCALE.get(sub_i) == SUB_TO_SCALE.get(sub_j):
        return "within_scale"

    # Affective constructs
    affect = {"PANAS_Positive_Affect", "PANAS_Negative_Affect",
              "POM_Peace_of_Mind", "LifeSat_Life_Satisfaction",
              "Lonely_Loneliness", "IRI_Personal_distress",
              "BigFive_Neuroticism"}
    cognitive = {"BigFive_Openness", "BigFive_Conscientiousness",
                 "IRI_Perspective_taking", "IRI_Fantasy"}
    social = {"IRI_Empathic_concern", "BigFive_Extraversion",
              "BigFive_Agreeableness",
              "SelfConst_Independent_self", "SelfConst_Interdependent_self"}

    def dom(s):
        if s in affect: return "A"
        if s in cognitive: return "C"
        if s in social: return "S"
        return "O"

    di, dj = dom(sub_i), dom(sub_j)
    key = "".join(sorted([di, dj]))
    # Named buckets for most common combinations
    bucket = {
        "AA": "affect_affect",
        "CC": "cognitive_cognitive",
        "SS": "social_social",
        "AC": "affect_cognitive",
        "AS": "affect_social",
        "CS": "cognitive_social",
    }.get(key, "other")
    return bucket


def run_F3(outdir: Path) -> Dict:
    """Per-pair amplification decomposition.

    For each model with a cross_persona/predicted_corr_matrix file, compute
    |llm_pred - human| per upper-tri pair, then aggregate by pair type.

    Source:
      outputs/mechanistic/results_{name}/cross_persona/predicted_corr_matrix_L{best}.csv
    The default aggregate output uses Llama 8B Instruct by convention.
    """
    human = load_human_subscale_corr()
    rows = []
    per_model = {}

    for m in MODELS:
        name = m["name"]
        if name not in MODELS_WITH_CROSS_PERSONA:
            continue
        best = load_model_best_layer(name)
        pred_path = (MECHANISTIC_ROOT / f"results_{name}" / "cross_persona"
                     / f"predicted_corr_matrix_L{best}.csv")
        if not pred_path.exists():
            # Fallback to legacy for llama8b_instruct
            if name == "llama8b_instruct":
                pred_path = MECHANISTIC_DEFAULT_RESULTS_DIR / "cross_persona" / f"predicted_corr_matrix_L{best}.csv"
            if not pred_path.exists():
                continue
        pred = pd.read_csv(pred_path, index_col=0)
        common = [s for s in ALL_SUBSCALES if s in pred.index and s in human.index]
        if len(common) < 4:
            continue
        H = human.loc[common, common].values
        P = pred.loc[common, common].values
        n = len(common)
        triu = np.triu_indices(n, k=1)
        for i, j in zip(*triu):
            si, sj = common[i], common[j]
            ptype = _pair_type(si, sj)
            h_r = H[i, j]; p_r = P[i, j]
            rows.append({
                "model": name,
                "family": m["family"], "mtype": m["mtype"], "size": m["size"],
                "sub_i": si, "sub_j": sj,
                "pair_type": ptype,
                "human_r": h_r,
                "pred_r": p_r,
                "amplification_ratio": (p_r / h_r) if abs(h_r) > 0.01 else np.nan,
                "signed_diff": p_r - h_r,
                "abs_diff": abs(p_r - h_r),
            })
        per_model[name] = {"best_layer": best, "n_pairs": len(triu[0])}

    if not rows:
        logger.info("  No cross_persona predicted matrices found; F3 skipped.")
        return {"error": "no_data"}

    detail = pd.DataFrame(rows)
    detail.to_csv(outdir / "F3_pair_detail.csv", index=False)

    # Aggregate: by pair type, by model, and combined
    by_type = detail.groupby("pair_type").agg(
        n=("abs_diff", "size"),
        mean_abs_diff=("abs_diff", "mean"),
        median_abs_diff=("abs_diff", "median"),
        mean_signed_diff=("signed_diff", "mean"),
        mean_amp_ratio=("amplification_ratio", "mean"),
    ).reset_index()
    by_type.to_csv(outdir / "F3_by_pair_type.csv", index=False)

    by_model_type = detail.groupby(["model", "pair_type"]).agg(
        n=("abs_diff", "size"),
        mean_abs_diff=("abs_diff", "mean"),
        mean_signed_diff=("signed_diff", "mean"),
    ).reset_index()
    by_model_type.to_csv(outdir / "F3_by_model_pair_type.csv", index=False)

    # Is amplification uniform across pair types? One-way ANOVA on signed_diff
    try:
        from scipy.stats import f_oneway
        groups = [g["signed_diff"].values for _, g in detail.groupby("pair_type")]
        groups = [g[~np.isnan(g)] for g in groups]
        groups = [g for g in groups if len(g) > 1]
        f_stat, p_val = f_oneway(*groups) if len(groups) >= 2 else (np.nan, np.nan)
    except Exception:
        f_stat, p_val = np.nan, np.nan

    summary = {
        "n_models": int(detail["model"].nunique()),
        "n_pair_observations": int(len(detail)),
        "anova_F": float(f_stat) if not np.isnan(f_stat) else None,
        "anova_p": float(p_val) if not np.isnan(p_val) else None,
        "by_pair_type": by_type.to_dict(orient="records"),
    }
    (outdir / "F3_summary.json").write_text(json.dumps(summary, indent=2, default=str))

    logger.info(f"  {'pair_type':<24} {'n':>5} {'mean |d|':>10} "
                f"{'mean signed d':>14}")
    for _, r in by_type.iterrows():
        logger.info(f"  {r['pair_type']:<24} {int(r['n']):>5} "
                    f"{r['mean_abs_diff']:>10.3f} {r['mean_signed_diff']:>+14.3f}")
    if summary["anova_p"] is not None:
        logger.info(f"\n  One-way ANOVA across pair types (signed_diff): "
                    f"F = {summary['anova_F']:.2f}, p = {summary['anova_p']:.4f}")

    return summary


# ─────────────────────────────────────────────────────────────────────────
#  Experiment: D2  Cultural stratification
# ─────────────────────────────────────────────────────────────────────────

def run_D2(outdir: Path) -> Dict:
    """Stable vs specific subscale slopes per model.

    Per model, per culture group, compute:
      - slope of predicted vs human for pairs where both subscales are in group
      - Mantel r on that subset matrix
    Primary contrast: stable vs specific.

    Note: classification is based on prior cross-cultural literature and is
    intended as a sensitivity grouping rather than a confirmatory taxonomy.
    """
    human = load_human_subscale_corr()
    rows = []

    for m in MODELS:
        name = m["name"]
        if name not in MODELS_WITH_CROSS_PERSONA:
            continue
        try:
            best = load_model_best_layer(name)
        except Exception:
            continue
        pred_path = (MECHANISTIC_ROOT / f"results_{name}" / "cross_persona"
                     / f"predicted_corr_matrix_L{best}.csv")
        if not pred_path.exists():
            if name == "llama8b_instruct":
                pred_path = MECHANISTIC_DEFAULT_RESULTS_DIR / "cross_persona" / f"predicted_corr_matrix_L{best}.csv"
            if not pred_path.exists():
                continue
        pred = pd.read_csv(pred_path, index_col=0)

        for grp_name, grp_subs in CULTURE_GROUPS.items():
            common = [s for s in grp_subs if s in pred.index and s in human.index]
            if len(common) < 3:
                continue
            H = human.loc[common, common].values
            P = pred.loc[common, common].values
            n = len(common)
            idx = np.triu_indices(n, k=1)
            hv = H[idx]; pv = P[idx]
            v = ~(np.isnan(hv) | np.isnan(pv))
            if v.sum() < 3:
                continue
            slope = np.polyfit(hv[v], pv[v], 1)[0]
            r_mantel, p_mantel = mantel_test(H, P, n_permutations=2000)
            rows.append({
                "model": name, "family": m["family"], "mtype": m["mtype"],
                "size": m["size"],
                "group": grp_name,
                "n_subscales": n,
                "n_pairs": int(v.sum()),
                "slope": float(slope),
                "mantel_r": float(r_mantel),
                "mantel_p": float(p_mantel),
            })

    if not rows:
        logger.info("  No cross_persona predicted matrices found; D2 skipped.")
        return {"error": "no_data"}

    df = pd.DataFrame(rows)
    df.to_csv(outdir / "D2_per_model_group.csv", index=False)

    # Paired test per model: stable - specific
    stable = df[df["group"] == "stable"].set_index("model")
    specific = df[df["group"] == "specific"].set_index("model")
    common_models = [m for m in stable.index if m in specific.index]
    diffs = []
    for m in common_models:
        diffs.append({
            "model": m,
            "slope_stable": stable.loc[m, "slope"],
            "slope_specific": specific.loc[m, "slope"],
            "slope_diff": stable.loc[m, "slope"] - specific.loc[m, "slope"],
            "mantel_stable": stable.loc[m, "mantel_r"],
            "mantel_specific": specific.loc[m, "mantel_r"],
            "mantel_diff": stable.loc[m, "mantel_r"] - specific.loc[m, "mantel_r"],
        })
    diff_df = pd.DataFrame(diffs)
    diff_df.to_csv(outdir / "D2_paired_diffs.csv", index=False)

    # Paired t-test
    from scipy.stats import ttest_rel, wilcoxon
    slope_p_t = slope_p_w = np.nan
    mantel_p_t = mantel_p_w = np.nan
    if len(diff_df) >= 3:
        try:
            _, slope_p_t = ttest_rel(diff_df["slope_stable"], diff_df["slope_specific"])
            _, mantel_p_t = ttest_rel(diff_df["mantel_stable"], diff_df["mantel_specific"])
        except Exception:
            pass
        try:
            _, slope_p_w = wilcoxon(diff_df["slope_stable"], diff_df["slope_specific"])
            _, mantel_p_w = wilcoxon(diff_df["mantel_stable"], diff_df["mantel_specific"])
        except Exception:
            pass

    summary = {
        "note": ("Sensitivity classification based on cross-cultural literature. "
                 "Stable: Big Five E/A/C/N, IRI EC/PT, PANAS NA, SWLS. "
                 "Specific: POM, SCS Indep/Interdep, IRI Fantasy. "
                 "Ambiguous subscales excluded from primary contrast."),
        "n_models": int(len(common_models)),
        "models": common_models,
        "paired": {
            "slope_mean_diff": float(diff_df["slope_diff"].mean()) if len(diff_df) else None,
            "slope_t_test_p": float(slope_p_t) if not np.isnan(slope_p_t) else None,
            "slope_wilcoxon_p": float(slope_p_w) if not np.isnan(slope_p_w) else None,
            "mantel_mean_diff": float(diff_df["mantel_diff"].mean()) if len(diff_df) else None,
            "mantel_t_test_p": float(mantel_p_t) if not np.isnan(mantel_p_t) else None,
            "mantel_wilcoxon_p": float(mantel_p_w) if not np.isnan(mantel_p_w) else None,
        },
    }
    (outdir / "D2_summary.json").write_text(json.dumps(summary, indent=2, default=str))

    logger.info(f"  Per-model slope (within-group pairs only):")
    logger.info(f"  {'model':<22} {'stable':>8} {'ambig':>8} {'specific':>9}")
    pivoted = df.pivot(index="model", columns="group", values="slope")
    for m in pivoted.index:
        row = pivoted.loc[m]
        logger.info(
            f"  {m:<22} "
            f"{row.get('stable', np.nan):>8.3f} "
            f"{row.get('ambiguous', np.nan):>8.3f} "
            f"{row.get('specific', np.nan):>9.3f}"
        )
    if summary["paired"]["slope_t_test_p"] is not None:
        logger.info(f"\n  Paired (stable vs specific):")
        logger.info(f"    mean Δslope = {summary['paired']['slope_mean_diff']:+.3f}  "
                    f"(paired t p = {summary['paired']['slope_t_test_p']:.4f}, "
                    f"wilcoxon p = {summary['paired']['slope_wilcoxon_p']:.4f})")
        logger.info(f"    mean Δmantel = {summary['paired']['mantel_mean_diff']:+.3f}  "
                    f"(paired t p = {summary['paired']['mantel_t_test_p']:.4f}, "
                    f"wilcoxon p = {summary['paired']['mantel_wilcoxon_p']:.4f})")

    return summary


# ─────────────────────────────────────────────────────────────────────────
#  Experiment: F1b  Adjacent-layer stability (match paper App B5 claim)
# ─────────────────────────────────────────────────────────────────────────

def run_F1b(outdir: Path) -> Dict:
    """Adjacent-layer stability: best ± 1, best ± 2 layer slope/mantel spread.

    Paper Appendix B5 claims adjacent-layer variation < 0.05.
    This verifies across all 14 models.
    """
    summary = load_scaling_summary()
    rows = []

    for _, row in summary.iterrows():
        name = row["model"]
        best = int(row["best_layer"])
        geom = load_model_geometry_results(name)
        if geom.empty:
            continue
        sub = geom[geom["level"] == "subscale"]
        if sub.empty:
            continue

        avail_layers = sorted(sub["layer"].unique().tolist())

        # Find layers within ±1, ±2 of best (that actually exist in sweep)
        def nearby(offset_max):
            candidates = [l for l in avail_layers if abs(l - best) <= offset_max]
            return candidates

        nearby1 = nearby(max(2, abs(avail_layers[1] - avail_layers[0])))  # 至少涵蓋 step size
        # Get the actual adjacent layers in sweep (best and its immediate neighbors)
        if best in avail_layers:
            idx_best = avail_layers.index(best)
        else:
            # Find nearest
            idx_best = min(range(len(avail_layers)), key=lambda i: abs(avail_layers[i] - best))
        # ±1 step in sweep
        lo1 = max(0, idx_best - 1)
        hi1 = min(len(avail_layers) - 1, idx_best + 1)
        layers_adj1 = [avail_layers[i] for i in range(lo1, hi1 + 1)]
        # ±2 step in sweep
        lo2 = max(0, idx_best - 2)
        hi2 = min(len(avail_layers) - 1, idx_best + 2)
        layers_adj2 = [avail_layers[i] for i in range(lo2, hi2 + 1)]

        def collect(layers):
            values_slope = []
            values_mantel = []
            for lyr in layers:
                srow = sub[sub["layer"] == lyr]
                if not srow.empty:
                    values_slope.append(float(srow["slope"].iloc[0]))
                    values_mantel.append(float(srow["mantel_r"].iloc[0]))
            return values_slope, values_mantel

        s1, m1 = collect(layers_adj1)
        s2, m2 = collect(layers_adj2)

        rows.append({
            "model": name,
            "family": row["family"], "mtype": row["mtype"],
            "best_layer": best,
            "layers_adj1": str(layers_adj1),
            "layers_adj2": str(layers_adj2),
            "n_adj1": len(s1), "n_adj2": len(s2),
            "slope_best": float(sub[sub["layer"] == best]["slope"].iloc[0])
                if best in avail_layers else np.nan,
            "mantel_best": float(sub[sub["layer"] == best]["mantel_r"].iloc[0])
                if best in avail_layers else np.nan,
            "slope_min_adj1": min(s1) if s1 else np.nan,
            "slope_max_adj1": max(s1) if s1 else np.nan,
            "slope_spread_adj1": (max(s1) - min(s1)) if len(s1) >= 2 else np.nan,
            "slope_min_adj2": min(s2) if s2 else np.nan,
            "slope_max_adj2": max(s2) if s2 else np.nan,
            "slope_spread_adj2": (max(s2) - min(s2)) if len(s2) >= 2 else np.nan,
            "mantel_spread_adj1": (max(m1) - min(m1)) if len(m1) >= 2 else np.nan,
            "mantel_spread_adj2": (max(m2) - min(m2)) if len(m2) >= 2 else np.nan,
        })

    df = pd.DataFrame(rows)
    df.to_csv(outdir / "F1b_adjacent_layer.csv", index=False)

    # Aggregate
    agg = {
        "n_models": int(len(df)),
        "mean_slope_spread_adj1": float(df["slope_spread_adj1"].mean()),
        "max_slope_spread_adj1": float(df["slope_spread_adj1"].max()),
        "mean_slope_spread_adj2": float(df["slope_spread_adj2"].mean()),
        "max_slope_spread_adj2": float(df["slope_spread_adj2"].max()),
        "mean_mantel_spread_adj1": float(df["mantel_spread_adj1"].mean()),
        "mean_mantel_spread_adj2": float(df["mantel_spread_adj2"].mean()),
        "paper_claim_adj_spread_lt_005": bool(df["slope_spread_adj1"].max() < 0.05),
        "cross_model_slope_spread": float(df["slope_best"].max() - df["slope_best"].min()),
    }
    (outdir / "F1b_summary.json").write_text(json.dumps(agg, indent=2, default=str))

    logger.info(f"  {'model':<22} {'best':>5} {'±1 layers':>16} "
                f"{'slope@best':>11} {'Δslope ±1':>10} {'Δslope ±2':>10}")
    for _, r in df.iterrows():
        logger.info(
            f"  {r['model']:<22} {r['best_layer']:>5} "
            f"{r['layers_adj1']:>16} "
            f"{r['slope_best']:>11.3f} "
            f"{r['slope_spread_adj1']:>10.3f} "
            f"{r['slope_spread_adj2']:>10.3f}"
        )
    logger.info(f"\n  Paper claim 'adjacent-layer variation < 0.05':")
    logger.info(f"    max Δslope(±1) across 14 models: {agg['max_slope_spread_adj1']:.3f}")
    logger.info(f"    mean Δslope(±1) across 14 models: {agg['mean_slope_spread_adj1']:.3f}")
    logger.info(f"    cross-model slope range: {agg['cross_model_slope_spread']:.3f}")
    logger.info(f"    → claim holds: {agg['paper_claim_adj_spread_lt_005']}")

    return agg


# ─────────────────────────────────────────────────────────────────────────
#  Experiment: F3b  Within-scale vs cross-scale trivial-context test
# ─────────────────────────────────────────────────────────────────────────

def run_F3b(outdir: Path) -> Dict:
    """Within-scale vs cross-scale trivial-context test.

    The persona prompt contains direct item-level answers for all subscales of
    the persona scale. Pairs where BOTH subscales share the same persona scale
    ('within_scale') therefore have direct supporting context in the prompt;
    'cross_scale' pairs have no direct context.

    Tests:
      (1) Is within-scale alignment materially better than cross-scale?
          If yes → trivial-copy confound exists. If no → LLM is inferring
          in both cases.
      (2) Report amplification (pred_r vs human_r) separately for
          within- and cross-scale pairs per model.
    """
    human = load_human_subscale_corr()
    rows = []

    for m in MODELS:
        name = m["name"]
        if name not in MODELS_WITH_CROSS_PERSONA:
            continue
        try:
            best = load_model_best_layer(name)
        except Exception:
            continue
        pred_path = (MECHANISTIC_ROOT / f"results_{name}" / "cross_persona"
                     / f"predicted_corr_matrix_L{best}.csv")
        if not pred_path.exists():
            if name == "llama8b_instruct":
                pred_path = MECHANISTIC_DEFAULT_RESULTS_DIR / "cross_persona" / f"predicted_corr_matrix_L{best}.csv"
            if not pred_path.exists():
                continue
        pred = pd.read_csv(pred_path, index_col=0)
        common = [s for s in ALL_SUBSCALES if s in pred.index and s in human.index]
        if len(common) < 4:
            continue

        n = len(common)
        for i in range(n):
            for j in range(i + 1, n):
                si, sj = common[i], common[j]
                scope = "within_scale" if SUB_TO_SCALE.get(si) == SUB_TO_SCALE.get(sj) \
                        else "cross_scale"
                h = float(human.loc[si, sj])
                p = float(pred.loc[si, sj])
                rows.append({
                    "model": name, "family": m["family"], "mtype": m["mtype"],
                    "size": m["size"],
                    "sub_i": si, "sub_j": sj,
                    "scope": scope,
                    "human_r": h, "pred_r": p,
                    "signed_diff": p - h, "abs_diff": abs(p - h),
                    "amp_ratio": (p / h) if abs(h) > 0.05 else np.nan,
                })

    if not rows:
        logger.info("  No cross_persona predicted matrices found; F3b skipped.")
        return {"error": "no_data"}

    detail = pd.DataFrame(rows)
    detail.to_csv(outdir / "F3b_pair_detail.csv", index=False)

    # Aggregate by model × scope
    by_model_scope = detail.groupby(["model", "scope"]).agg(
        n=("abs_diff", "size"),
        mean_abs_diff=("abs_diff", "mean"),
        median_abs_diff=("abs_diff", "median"),
        mean_signed_diff=("signed_diff", "mean"),
        mean_human_r=("human_r", "mean"),
        mean_pred_r=("pred_r", "mean"),
    ).reset_index()
    by_model_scope.to_csv(outdir / "F3b_by_model_scope.csv", index=False)

    # Primary test: paired per-model comparison of within vs cross |d|
    pivoted_abs = by_model_scope.pivot(index="model", columns="scope", values="mean_abs_diff")
    pivoted_sign = by_model_scope.pivot(index="model", columns="scope", values="mean_signed_diff")

    paired = []
    for m in pivoted_abs.index:
        if "within_scale" in pivoted_abs.columns and "cross_scale" in pivoted_abs.columns:
            paired.append({
                "model": m,
                "abs_within": pivoted_abs.loc[m, "within_scale"],
                "abs_cross": pivoted_abs.loc[m, "cross_scale"],
                "abs_diff_within_minus_cross": pivoted_abs.loc[m, "within_scale"] - pivoted_abs.loc[m, "cross_scale"],
                "signed_within": pivoted_sign.loc[m, "within_scale"],
                "signed_cross": pivoted_sign.loc[m, "cross_scale"],
            })
    paired_df = pd.DataFrame(paired)
    paired_df.to_csv(outdir / "F3b_paired.csv", index=False)

    from scipy.stats import ttest_rel, wilcoxon
    if len(paired_df) >= 3:
        try:
            t_stat, t_p = ttest_rel(paired_df["abs_within"], paired_df["abs_cross"])
            w_stat, w_p = wilcoxon(paired_df["abs_within"], paired_df["abs_cross"])
        except Exception:
            t_stat = t_p = w_stat = w_p = np.nan
    else:
        t_stat = t_p = w_stat = w_p = np.nan

    # Across-all-pairs aggregate (pooled)
    pooled_stats = detail.groupby("scope").agg(
        n_pairs=("abs_diff", "size"),
        mean_abs=("abs_diff", "mean"),
        median_abs=("abs_diff", "median"),
        mean_signed=("signed_diff", "mean"),
        mean_human_r=("human_r", "mean"),
        mean_pred_r=("pred_r", "mean"),
    ).to_dict(orient="index")

    # Trivial-copy verdict
    mean_within_abs = float(paired_df["abs_within"].mean()) if len(paired_df) else np.nan
    mean_cross_abs = float(paired_df["abs_cross"].mean()) if len(paired_df) else np.nan
    ratio = mean_within_abs / mean_cross_abs if mean_cross_abs > 0 else np.nan

    verdict_text = ""
    if not np.isnan(ratio):
        if ratio < 0.5 and t_p < 0.05:
            verdict_text = (
                "within-scale |d| is <50% of cross-scale AND significantly smaller "
                "→ trivial-context confound LIKELY; primary analyses should "
                "exclude within-scale pairs."
            )
        elif ratio < 0.8 and t_p < 0.05:
            verdict_text = (
                "within-scale |d| is moderately smaller than cross-scale "
                "(significantly). Some context-copy effect, not pure trivial."
            )
        else:
            verdict_text = (
                "within-scale |d| is NOT materially smaller than cross-scale "
                "(or difference not significant). Trivial-copy confound is "
                "NOT a dominant driver; LLM is inferring in both cases."
            )

    summary = {
        "n_models": int(len(paired_df)),
        "pooled_stats": pooled_stats,
        "paired_test_abs_diff": {
            "t_stat": float(t_stat) if not np.isnan(t_stat) else None,
            "t_p": float(t_p) if not np.isnan(t_p) else None,
            "wilcoxon_stat": float(w_stat) if not np.isnan(w_stat) else None,
            "wilcoxon_p": float(w_p) if not np.isnan(w_p) else None,
            "mean_abs_within": mean_within_abs,
            "mean_abs_cross": mean_cross_abs,
            "ratio_within_to_cross": float(ratio) if not np.isnan(ratio) else None,
        },
        "verdict": verdict_text,
    }
    (outdir / "F3b_summary.json").write_text(json.dumps(summary, indent=2, default=str))

    logger.info(f"  Pooled across {len(paired_df)} models:")
    for scope in ["within_scale", "cross_scale"]:
        if scope in pooled_stats:
            s = pooled_stats[scope]
            logger.info(f"    {scope:<13}: n={s['n_pairs']:>4}  "
                        f"mean |d|={s['mean_abs']:.3f}  "
                        f"signed d={s['mean_signed']:+.3f}  "
                        f"<human r>={s['mean_human_r']:+.3f}  "
                        f"<pred r>={s['mean_pred_r']:+.3f}")
    logger.info(f"\n  Paired per-model test (within vs cross |d|):")
    logger.info(f"    mean |d| within = {mean_within_abs:.3f}")
    logger.info(f"    mean |d| cross  = {mean_cross_abs:.3f}")
    logger.info(f"    ratio = {ratio:.3f}  (1.0 = no difference)")
    if not np.isnan(t_p):
        logger.info(f"    paired t: p = {t_p:.4f}, wilcoxon: p = {w_p:.4f}")
    logger.info(f"\n  Verdict: {verdict_text}")

    return summary


# ─────────────────────────────────────────────────────────────────────────
#  Experiment: D2b  D2 robustness diagnostics
# ─────────────────────────────────────────────────────────────────────────

def run_D2b(outdir: Path) -> Dict:
    """D2 robustness diagnostics — multiple sub-analyses.

    (a) Human r magnitude by culture group (does denominator issue drive slope?)
    (b) Leave-one-pair-out: identify influential pairs driving specific slope
    (c) Mantel-r based comparison (magnitude-invariant)
    (d) D2 restricted to cross-scale pairs only (removes trivial-context confound)
    """
    human = load_human_subscale_corr()

    # ─── Sub-analysis (a): Human r magnitude per group ───
    human_r_by_group = {}
    for grp_name, subs in CULTURE_GROUPS.items():
        common = [s for s in subs if s in human.index]
        n = len(common)
        if n < 2:
            continue
        idx = np.triu_indices(n, k=1)
        values = human.loc[common, common].values[idx]
        # Whether each pair is within-scale or cross-scale
        within_scale_mask = np.array([
            SUB_TO_SCALE.get(common[i]) == SUB_TO_SCALE.get(common[j])
            for i, j in zip(*idx)
        ])
        human_r_by_group[grp_name] = {
            "n_subscales": n,
            "n_pairs": int(len(values)),
            "n_within_scale_pairs": int(within_scale_mask.sum()),
            "n_cross_scale_pairs": int((~within_scale_mask).sum()),
            "mean_abs_r": float(np.nanmean(np.abs(values))),
            "median_abs_r": float(np.nanmedian(np.abs(values))),
            "min_abs_r": float(np.nanmin(np.abs(values))),
            "max_abs_r": float(np.nanmax(np.abs(values))),
            "mean_r": float(np.nanmean(values)),
        }

    # ─── Sub-analysis (b): Leave-one-pair-out per model ───
    loo_rows = []

    for m in MODELS:
        name = m["name"]
        if name not in MODELS_WITH_CROSS_PERSONA:
            continue
        try:
            best = load_model_best_layer(name)
        except Exception:
            continue
        pred_path = (MECHANISTIC_ROOT / f"results_{name}" / "cross_persona"
                     / f"predicted_corr_matrix_L{best}.csv")
        if not pred_path.exists():
            if name == "llama8b_instruct":
                pred_path = MECHANISTIC_DEFAULT_RESULTS_DIR / "cross_persona" / f"predicted_corr_matrix_L{best}.csv"
            if not pred_path.exists():
                continue
        pred = pd.read_csv(pred_path, index_col=0)

        for grp_name, grp_subs in CULTURE_GROUPS.items():
            common = [s for s in grp_subs if s in pred.index and s in human.index]
            n = len(common)
            if n < 3:
                continue
            pairs_full = []
            for i in range(n):
                for j in range(i + 1, n):
                    si, sj = common[i], common[j]
                    pairs_full.append((si, sj, float(human.loc[si, sj]),
                                       float(pred.loc[si, sj])))
            if len(pairs_full) < 3:
                continue

            # Full slope
            hv_full = np.array([p[2] for p in pairs_full])
            pv_full = np.array([p[3] for p in pairs_full])
            valid = ~(np.isnan(hv_full) | np.isnan(pv_full))
            slope_full = (np.polyfit(hv_full[valid], pv_full[valid], 1)[0]
                          if valid.sum() >= 2 else np.nan)

            # LOO: drop each pair, recompute slope
            for idx_drop, (si, sj, _, _) in enumerate(pairs_full):
                mask = np.ones(len(pairs_full), dtype=bool)
                mask[idx_drop] = False
                hv = hv_full[mask]; pv = pv_full[mask]
                v = ~(np.isnan(hv) | np.isnan(pv))
                if v.sum() >= 2:
                    slope_loo = float(np.polyfit(hv[v], pv[v], 1)[0])
                else:
                    slope_loo = np.nan
                loo_rows.append({
                    "model": name, "group": grp_name,
                    "dropped_pair": f"{si} × {sj}",
                    "dropped_human_r": pairs_full[idx_drop][2],
                    "dropped_pred_r": pairs_full[idx_drop][3],
                    "slope_full": slope_full,
                    "slope_loo": slope_loo,
                    "delta_slope": slope_full - slope_loo
                        if not (np.isnan(slope_full) or np.isnan(slope_loo)) else np.nan,
                })

    loo_df = pd.DataFrame(loo_rows)
    loo_df.to_csv(outdir / "D2b_leave_one_pair_out.csv", index=False)

    # Identify most influential pairs for specific group
    spec_loo = loo_df[loo_df["group"] == "specific"].copy()
    if not spec_loo.empty:
        # Average |delta| per dropped pair across models
        influence = spec_loo.groupby("dropped_pair").agg(
            mean_delta=("delta_slope", "mean"),
            mean_abs_delta=("delta_slope", lambda x: float(np.mean(np.abs(x)))),
            n=("delta_slope", "size"),
        ).reset_index().sort_values("mean_abs_delta", ascending=False)
        influence.to_csv(outdir / "D2b_specific_pair_influence.csv", index=False)
    else:
        influence = pd.DataFrame()

    # ─── Sub-analysis (c): Mantel r paired comparison (magnitude-invariant) ───
    # Reload D2 summary CSV
    d2_path = OUTPUT_ROOT / "D2" / "D2_paired_diffs.csv"
    d2_mantel = None
    if d2_path.exists():
        d2_df = pd.read_csv(d2_path)
        from scipy.stats import ttest_rel as _ttest, wilcoxon as _wilc
        if len(d2_df) >= 3:
            try:
                _, p_t = _ttest(d2_df["mantel_stable"], d2_df["mantel_specific"])
                _, p_w = _wilc(d2_df["mantel_stable"], d2_df["mantel_specific"])
                d2_mantel = {
                    "mean_diff_mantel": float(d2_df["mantel_diff"].mean()),
                    "t_p": float(p_t),
                    "wilcoxon_p": float(p_w),
                    "direction_stable_greater": bool(d2_df["mantel_diff"].mean() > 0),
                }
            except Exception:
                d2_mantel = None

    # ─── Sub-analysis (d): D2 restricted to cross-scale pairs only ───
    rows_cs = []
    for m in MODELS:
        name = m["name"]
        if name not in MODELS_WITH_CROSS_PERSONA:
            continue
        try:
            best = load_model_best_layer(name)
        except Exception:
            continue
        pred_path = (MECHANISTIC_ROOT / f"results_{name}" / "cross_persona"
                     / f"predicted_corr_matrix_L{best}.csv")
        if not pred_path.exists():
            if name == "llama8b_instruct":
                pred_path = MECHANISTIC_DEFAULT_RESULTS_DIR / "cross_persona" / f"predicted_corr_matrix_L{best}.csv"
            if not pred_path.exists():
                continue
        pred = pd.read_csv(pred_path, index_col=0)

        for grp_name, grp_subs in CULTURE_GROUPS.items():
            common = [s for s in grp_subs if s in pred.index and s in human.index]
            n = len(common)
            if n < 3:
                continue
            # Collect cross-scale-only pairs
            hv_cs = []; pv_cs = []
            for i in range(n):
                for j in range(i + 1, n):
                    si, sj = common[i], common[j]
                    if SUB_TO_SCALE.get(si) == SUB_TO_SCALE.get(sj):
                        continue
                    hv_cs.append(float(human.loc[si, sj]))
                    pv_cs.append(float(pred.loc[si, sj]))
            hv_cs = np.array(hv_cs); pv_cs = np.array(pv_cs)
            v = ~(np.isnan(hv_cs) | np.isnan(pv_cs))
            if v.sum() >= 2:
                slope_cs = float(np.polyfit(hv_cs[v], pv_cs[v], 1)[0])
                r_cs = float(np.corrcoef(hv_cs[v], pv_cs[v])[0, 1])
            else:
                slope_cs = np.nan; r_cs = np.nan

            rows_cs.append({
                "model": name, "family": m["family"], "mtype": m["mtype"],
                "size": m["size"], "group": grp_name,
                "n_pairs_cross_scale": int(v.sum()),
                "slope_cross_scale_only": slope_cs,
                "pearson_r_cross_scale_only": r_cs,
            })
    cs_df = pd.DataFrame(rows_cs)
    cs_df.to_csv(outdir / "D2b_cross_scale_only.csv", index=False)

    # Paired test on cross-scale-only slopes
    from scipy.stats import ttest_rel, wilcoxon
    stable_cs = cs_df[cs_df["group"] == "stable"].set_index("model")
    specific_cs = cs_df[cs_df["group"] == "specific"].set_index("model")
    common_m = [m for m in stable_cs.index if m in specific_cs.index]
    if len(common_m) >= 3:
        diff_slopes = (stable_cs.loc[common_m, "slope_cross_scale_only"] -
                       specific_cs.loc[common_m, "slope_cross_scale_only"])
        try:
            _, p_t_cs = ttest_rel(stable_cs.loc[common_m, "slope_cross_scale_only"],
                                   specific_cs.loc[common_m, "slope_cross_scale_only"])
            _, p_w_cs = wilcoxon(stable_cs.loc[common_m, "slope_cross_scale_only"],
                                  specific_cs.loc[common_m, "slope_cross_scale_only"])
        except Exception:
            p_t_cs = p_w_cs = np.nan
        cross_scale_paired = {
            "n_models": int(len(common_m)),
            "mean_slope_diff_cross_scale_only": float(diff_slopes.mean()),
            "t_p": float(p_t_cs) if not np.isnan(p_t_cs) else None,
            "wilcoxon_p": float(p_w_cs) if not np.isnan(p_w_cs) else None,
        }
    else:
        cross_scale_paired = None

    # ─── Summary ───
    summary = {
        "note": (
            "D2 robustness diagnostics. (a) human r magnitude per group; "
            "(b) leave-one-pair-out for 'specific' group to identify driver pairs; "
            "(c) paired Mantel r test (magnitude-invariant alternative to slope); "
            "(d) D2 restricted to cross-scale pairs only (removes trivial-context "
            "confound from within-scale pairs like SCS_Indep × SCS_Interdep)."
        ),
        "human_r_by_group": human_r_by_group,
        "specific_pair_influence": (
            influence.to_dict(orient="records") if not influence.empty else None
        ),
        "mantel_based_comparison": d2_mantel,
        "cross_scale_only_paired": cross_scale_paired,
    }
    (outdir / "D2b_summary.json").write_text(json.dumps(summary, indent=2, default=str))

    # Log
    logger.info(f"  (a) Human |r| magnitude by culture group:")
    for g, s in human_r_by_group.items():
        logger.info(f"    {g:<10} n_pairs={s['n_pairs']:>2}  "
                    f"mean|r|={s['mean_abs_r']:.3f}  "
                    f"median|r|={s['median_abs_r']:.3f}  "
                    f"range=[{s['min_abs_r']:.3f}, {s['max_abs_r']:.3f}]  "
                    f"(within-scale: {s['n_within_scale_pairs']}, "
                    f"cross-scale: {s['n_cross_scale_pairs']})")

    if not influence.empty:
        logger.info(f"\n  (b) Most influential pairs driving 'specific' group slope "
                    f"(top 3 by |Δslope|):")
        for _, r in influence.head(3).iterrows():
            logger.info(f"    {r['dropped_pair']:<50} "
                        f"mean |Δslope| = {r['mean_abs_delta']:.3f}  (n={r['n']})")

    if d2_mantel is not None:
        logger.info(f"\n  (c) Mantel-r-based comparison (magnitude-invariant):")
        logger.info(f"    mean Δmantel (stable - specific) = "
                    f"{d2_mantel['mean_diff_mantel']:+.3f}")
        logger.info(f"    paired t-test p = {d2_mantel['t_p']:.4f}  "
                    f"(direction: stable > specific = "
                    f"{d2_mantel['direction_stable_greater']})")

    if cross_scale_paired is not None:
        logger.info(f"\n  (d) D2 restricted to cross-scale pairs only:")
        logger.info(f"    mean slope diff (stable - specific) = "
                    f"{cross_scale_paired['mean_slope_diff_cross_scale_only']:+.3f}  "
                    f"(n={cross_scale_paired['n_models']} models)")
        if cross_scale_paired['t_p'] is not None:
            logger.info(f"    paired t-test p = {cross_scale_paired['t_p']:.4f}, "
                        f"wilcoxon p = {cross_scale_paired['wilcoxon_p']:.4f}")

    return summary


# ═════════════════════════════════════════════════════════════════════════
#                           T1: SYNTHETIC CONTROLS
# ═════════════════════════════════════════════════════════════════════════
#  Paper §7 (lines 253-259) explicitly points to "synthetic data with known
#  ground-truth correlation structures" as needed to disentangle whether the
#  1.38-1.77 representational slope range reflects genuine representational
#  amplification or pipeline artifacts.
#
#  Two companion experiments:
#    A1_minimal:  item-level synthetic (NO activation layer). Tests whether
#                 ridge regression applied to psychometric-like item data
#                 (latent + item noise) systematically inflates recovered
#                 subscale correlations relative to ground truth.
#    A1_full:     high-dim simulation (matches real hidden dimensions) +
#                 SNR sweep + multiple seeds. Maps out recovered slope as
#                 a function of (hidden_dim, SNR); real-model SNR positions
#                 are overlaid to show where they sit on the curve.
#
#  Interpretation of a positive finding:
#    if A1_minimal recovers slope >> 1.0, ridge + item aggregation alone
#    inflates — in that case the 1.38-1.77 range has a pipeline component
#    already present before activations enter the picture.
#    if A1_minimal recovers slope ≈ 1.0, pipeline is unbiased at the
#    item level — any inflation must come from the activation → prediction
#    stage (tested in A1_full).
# ═════════════════════════════════════════════════════════════════════════


def _generate_latent_subscale_scores(n_subjects: int, human_corr_matrix: pd.DataFrame,
                                     rng: np.random.Generator) -> pd.DataFrame:
    """Multivariate-normal latent subscale scores with target covariance."""
    sub_names = list(human_corr_matrix.index)
    n_sub = len(sub_names)
    C = human_corr_matrix.values
    # Nudge toward PSD if minor numerical issues
    eigvals = np.linalg.eigvalsh(C)
    if eigvals.min() < -1e-8:
        logger.warning(
            f"  human corr matrix min eigenvalue = {eigvals.min():.4f}; "
            f"projecting to nearest PSD"
        )
        w, V = np.linalg.eigh(C)
        w[w < 1e-6] = 1e-6
        C = V @ np.diag(w) @ V.T
        # renormalize to correlation matrix
        D = np.sqrt(np.diag(C))
        C = C / np.outer(D, D)
    samples = rng.multivariate_normal(mean=np.zeros(n_sub), cov=C, size=n_subjects)
    # Standardize each column (zero-mean, unit-var) to ensure ground-truth slope=1
    samples = (samples - samples.mean(axis=0)) / samples.std(axis=0, ddof=0)
    return pd.DataFrame(samples, columns=sub_names)


def _synth_items_from_latent(latent: pd.DataFrame,
                             items_per_subscale: int,
                             alpha_target: float,
                             rng: np.random.Generator
                             ) -> Tuple[pd.DataFrame, List[Tuple[str, str]]]:
    """Generate synthetic items with latent + Gaussian noise, targeting a
    given Cronbach α per subscale.

    For items (x1..xk) = latent + N(0, sigma^2):
       α = k*var(latent) / (k*var(latent) + var(noise))
    Solve for sigma given target alpha (assuming var(latent)=1):
       sigma^2 = (1/α - 1) * k
    """
    n = len(latent)
    rows = []
    labels = []
    for sub_name in latent.columns:
        k = items_per_subscale
        # var(noise) per-item such that sum of k items has given alpha
        # Reliability of sum of k items = k*Var(L) / (k*Var(L) + k*Var(noise_i))
        # If alpha target for k-item composite:  var_noise = (1/alpha - 1) * var_latent
        # (Spearman-Brown-esque approximation; close enough for simulation.)
        var_noise = (1.0 / alpha_target - 1.0)
        sigma = np.sqrt(var_noise)
        L = latent[sub_name].values
        for ki in range(k):
            item = L + rng.normal(0, sigma, size=n)
            rows.append(item)
            labels.append((sub_name, f"{sub_name}_item{ki+1}"))
    data = np.column_stack(rows)
    item_cols = [lab[1] for lab in labels]
    df = pd.DataFrame(data, columns=item_cols, index=latent.index)
    return df, labels


def _run_item_level_ridge_pipeline(items: pd.DataFrame,
                                   labels: List[Tuple[str, str]],
                                   human_corr_matrix: pd.DataFrame,
                                   reference_corr: Optional[pd.DataFrame] = None,
                                   seed: int = 42) -> Dict:
    """Mirror the paper's item-level pipeline with synthetic data.

    For each target item:
      - features = all OTHER items (leave-one-out within the item bank)
      - target = the item's response
      - fit RidgeCV with 5-fold CV
      - use cross-validated predictions as continuous item predictions

    Then average per-item predictions within each subscale to subscale score,
    compute 16×16 correlation matrix, and compare to both:
      - reference_corr (default = human_corr): the "target" structure
      - human_corr: paper's anchor (may differ from reference if the latent
        sample differs from the human population).

    Returns recovered slope / Mantel r vs reference_corr (primary) and
    vs human_corr (for paper alignment).
    """
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import KFold

    if reference_corr is None:
        reference_corr = human_corr_matrix

    n, n_items = items.shape
    item_names = list(items.columns)
    sub_of_item = {it: sub for sub, it in labels}

    # Cross-validated per-item predictions
    X_all = items.values
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    pred_matrix = np.zeros_like(X_all)
    for k_idx, target_item in enumerate(item_names):
        y = X_all[:, k_idx]
        X = np.delete(X_all, k_idx, axis=1)
        preds = np.zeros(n)
        for tr, te in kf.split(X):
            ridge = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0, 1000.0], cv=5)
            ridge.fit(X[tr], y[tr])
            preds[te] = ridge.predict(X[te])
        pred_matrix[:, k_idx] = preds

    # Aggregate to subscale scores
    subscales = list(human_corr_matrix.index)
    subscale_scores = {}
    for sub in subscales:
        cols = [i for i, it in enumerate(item_names) if sub_of_item[it] == sub]
        if cols:
            subscale_scores[sub] = pred_matrix[:, cols].mean(axis=1)
    scores_df = pd.DataFrame(subscale_scores)

    # Recovered correlation matrix
    common = [s for s in subscales if s in scores_df.columns]
    rec_corr = scores_df[common].corr().loc[common, common]
    H_human = human_corr_matrix.loc[common, common]
    H_ref = reference_corr.loc[common, common]

    def _slope_and_mantel(ref_matrix: pd.DataFrame, rec: pd.DataFrame):
        n_c = len(common)
        idx = np.triu_indices(n_c, k=1)
        rv = rec.values[idx]
        hv = ref_matrix.values[idx]
        valid = ~(np.isnan(hv) | np.isnan(rv))
        if valid.sum() < 2:
            return np.nan, np.nan, np.nan
        slope_v = float(np.polyfit(hv[valid], rv[valid], 1)[0])
        r_pearson_v = float(np.corrcoef(hv[valid], rv[valid])[0, 1])
        r_mantel_v, _ = mantel_test(ref_matrix.values, rec.values, n_permutations=200)
        return slope_v, r_pearson_v, float(r_mantel_v)

    slope_vs_ref, r_vs_ref, mantel_vs_ref = _slope_and_mantel(H_ref, rec_corr)
    slope_vs_human, r_vs_human, mantel_vs_human = _slope_and_mantel(H_human, rec_corr)

    return {
        # Primary: vs reference (sample latent corr) — what pipeline COULD recover
        "recovered_slope": slope_vs_ref,
        "recovered_pearson": r_vs_ref,
        "recovered_mantel_r": mantel_vs_ref,
        # Secondary: vs human population corr (paper's published target)
        "recovered_slope_vs_human": slope_vs_human,
        "recovered_pearson_vs_human": r_vs_human,
        "recovered_mantel_r_vs_human": mantel_vs_human,
        "n_pairs": int(np.triu_indices(len(common), k=1)[0].size),
        "recovered_corr_matrix": rec_corr.values,
    }


# ─────────────────────────────────────────────────────────────────────────
#  Experiment: A1_minimal  Item-level synthetic ridge inflation check
# ─────────────────────────────────────────────────────────────────────────

def run_A1_minimal(outdir: Path) -> Dict:
    """Item-level synthetic control — does ridge inflate recovered slope even
    without any activation layer?

    Design:
      - Ground truth: 16×16 human subscale correlation matrix
      - Generate N=272 latent subscale scores ~ MVN(0, ground truth)
      - Generate synthetic items (latent + noise) with target Cronbach α
      - Run item-level ridge pipeline (mirrors paper §5.3)
      - Compare recovered 16×16 correlation matrix vs ground truth

    Sweeps:
      - items_per_subscale: [3, 5, 7]   (paper real data ≈ 5 items/subscale)
      - alpha_target: [0.6, 0.75, 0.85] (span of realistic human reliabilities)
      - seeds: 20

    Expected behavior:
      - If ridge unbiased → recovered slope distribution centered at 1.0
      - If inflation present → distribution center > 1.0
    """
    human = load_human_subscale_corr()
    n_subjects = 272

    items_per_subscale_grid = [3, 5, 7]
    alpha_grid = [0.60, 0.75, 0.85]
    n_seeds = 20

    n_jobs = int(os.environ.get("EXTRA_N_JOBS", "1"))
    if n_jobs > 1:
        try:
            from joblib import Parallel, delayed
        except ImportError:
            logger.warning("  joblib not installed; falling back to serial")
            n_jobs = 1

    def _one_run(ipm: int, alpha_t: float, seed: int) -> Dict:
        rng = np.random.default_rng(seed)
        latent = _generate_latent_subscale_scores(n_subjects, human, rng)
        items, labels = _synth_items_from_latent(latent, ipm, alpha_t, rng)
        reference_corr = latent.corr().loc[human.index, human.index]
        res = _run_item_level_ridge_pipeline(
            items, labels, human,
            reference_corr=reference_corr, seed=seed
        )
        return {
            "items_per_subscale": ipm,
            "alpha_target": alpha_t,
            "seed": seed,
            "recovered_slope": res["recovered_slope"],
            "recovered_mantel_r": res["recovered_mantel_r"],
            "recovered_pearson": res["recovered_pearson"],
            "recovered_slope_vs_human": res["recovered_slope_vs_human"],
            "recovered_mantel_r_vs_human": res["recovered_mantel_r_vs_human"],
            "n_pairs": res["n_pairs"],
        }

    jobs = [(ipm, alpha_t, seed)
            for ipm in items_per_subscale_grid
            for alpha_t in alpha_grid
            for seed in range(n_seeds)]

    logger.info(f"  A1_minimal: {len(jobs)} runs — n_jobs = {n_jobs}")
    rows = []
    t0 = time.time()
    if n_jobs > 1:
        rows = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_one_run)(ipm, alpha_t, seed) for (ipm, alpha_t, seed) in jobs
        )
    else:
        for (ipm, alpha_t, seed) in jobs:
            rows.append(_one_run(ipm, alpha_t, seed))
    df = pd.DataFrame(rows)
    df.to_csv(outdir / "A1_minimal_all_runs.csv", index=False)

    agg = df.groupby(["items_per_subscale", "alpha_target"]).agg(
        n_seeds=("seed", "size"),
        slope_mean=("recovered_slope", "mean"),
        slope_std=("recovered_slope", "std"),
        slope_se=("recovered_slope", lambda x: float(x.std() / np.sqrt(len(x)))),
        slope_min=("recovered_slope", "min"),
        slope_max=("recovered_slope", "max"),
        mantel_mean=("recovered_mantel_r", "mean"),
        slope_vs_human_mean=("recovered_slope_vs_human", "mean"),
        slope_vs_human_std=("recovered_slope_vs_human", "std"),
    ).reset_index()
    agg.to_csv(outdir / "A1_minimal_by_grid.csv", index=False)

    overall = {
        "n_total_runs": int(len(df)),
        # vs reference (sample latent corr) — pure pipeline inflation
        "slope_vs_ref_mean": float(df["recovered_slope"].mean()),
        "slope_vs_ref_std": float(df["recovered_slope"].std()),
        "slope_vs_ref_min": float(df["recovered_slope"].min()),
        "slope_vs_ref_max": float(df["recovered_slope"].max()),
        # vs human (paper's published target)
        "slope_vs_human_mean": float(df["recovered_slope_vs_human"].mean()),
        "slope_vs_human_std": float(df["recovered_slope_vs_human"].std()),
        "slope_vs_human_max": float(df["recovered_slope_vs_human"].max()),
        "mantel_overall_mean": float(df["recovered_mantel_r"].mean()),
        "ground_truth_slope": 1.0,
        "paper_real_slope_range": [1.38, 1.77],
        "elapsed_sec": time.time() - t0,
    }

    # Interpretation verdict based on slope_vs_ref (pure pipeline bias)
    real_lo = 1.38
    real_hi = 1.77
    mn_ref = overall["slope_vs_ref_mean"]
    mn_human = overall["slope_vs_human_mean"]
    if mn_ref > 1.3:
        verdict = (
            f"SEVERE: pipeline's pure inflation (slope vs sample latent corr) = "
            f"{mn_ref:.3f}. A major portion of the real-model range [{real_lo}, "
            f"{real_hi}] could be attributable to ridge pipeline."
        )
    elif mn_ref > 1.15:
        verdict = (
            f"MODERATE: pipeline adds inflation of {mn_ref:.3f} above ground truth. "
            f"This cannot alone explain the real-model range [{real_lo}, {real_hi}] "
            f"(gap ≥ {real_lo - mn_ref:.2f}) but is a non-trivial component."
        )
    elif mn_ref > 1.05:
        verdict = (
            f"MILD: pipeline inflation = {mn_ref:.3f}, slightly above 1.0. Cannot "
            f"account for the real-model range [{real_lo}, {real_hi}]."
        )
    else:
        verdict = (
            f"NEGLIGIBLE: pipeline inflation = {mn_ref:.3f} ≈ 1.0. Ridge is "
            f"essentially unbiased; the real-model 1.38–1.77 range cannot be "
            f"attributed to pipeline effects at the item level."
        )
    overall["verdict"] = verdict
    (outdir / "A1_minimal_summary.json").write_text(
        json.dumps(overall, indent=2, default=str))

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, len(items_per_subscale_grid),
                                 figsize=(4.2 * len(items_per_subscale_grid), 4),
                                 sharey=True)
        if not isinstance(axes, np.ndarray):
            axes = [axes]
        for ax, ipm in zip(axes, items_per_subscale_grid):
            for alpha_t in alpha_grid:
                sub = df[(df["items_per_subscale"] == ipm) & (df["alpha_target"] == alpha_t)]
                ax.scatter([alpha_t] * len(sub), sub["recovered_slope"],
                           alpha=0.4, s=20)
                ax.errorbar([alpha_t], [sub["recovered_slope"].mean()],
                            yerr=[sub["recovered_slope"].std() / np.sqrt(len(sub))],
                            fmt="o", color="black", markersize=6, capsize=4)
            ax.axhline(1.0, ls="--", color="green", label="unbiased (slope=1)")
            ax.axhspan(1.38, 1.77, alpha=0.15, color="red",
                       label="paper real-model range")
            ax.set_xlabel("target Cronbach α")
            ax.set_title(f"{ipm} items per subscale")
            ax.grid(alpha=0.2)
            if ax is axes[0]:
                ax.set_ylabel("recovered slope (vs sample latent corr)")
        axes[-1].legend(loc="upper right", fontsize=8)
        plt.suptitle(
            f"A1 minimal: pipeline inflation check  "
            f"(mean slope vs ref = {overall['slope_vs_ref_mean']:.3f}, "
            f"unbiased = 1.000)",
            fontsize=11
        )
        plt.tight_layout()
        plt.savefig(outdir / "A1_minimal_slope_distribution.png",
                    dpi=200, bbox_inches="tight")
        plt.close()
    except Exception as e:
        logger.warning(f"  A1_minimal plot failed: {e}")

    logger.info(f"  {overall['n_total_runs']} runs "
                f"(items_per_subscale={items_per_subscale_grid}, "
                f"α_target={alpha_grid}, {n_seeds} seeds)")
    logger.info(f"  slope vs sample latent corr:  "
                f"mean = {overall['slope_vs_ref_mean']:.3f}  "
                f"std = {overall['slope_vs_ref_std']:.3f}  "
                f"range [{overall['slope_vs_ref_min']:.3f}, {overall['slope_vs_ref_max']:.3f}]")
    logger.info(f"  slope vs human population:    "
                f"mean = {overall['slope_vs_human_mean']:.3f}  "
                f"(paper real-model range: [1.38, 1.77])")
    logger.info(f"\n  Verdict: {verdict}")
    logger.info(f"\n  By grid (slope vs ref mean ± SE):")
    for _, r in agg.iterrows():
        logger.info(f"    items={int(r['items_per_subscale'])}, "
                    f"α={r['alpha_target']:.2f}: "
                    f"{r['slope_mean']:.3f} ± {r['slope_se']:.3f}  "
                    f"(vs human: {r['slope_vs_human_mean']:.3f}, "
                    f"mantel = {r['mantel_mean']:.3f})")

    return overall


# ─────────────────────────────────────────────────────────────────────────
#  Experiment: A1_full  High-dim simulation + SNR sweep
# ─────────────────────────────────────────────────────────────────────────

def _run_activation_level_ridge_pipeline(
    activations: np.ndarray,
    items: pd.DataFrame,
    labels: List[Tuple[str, str]],
    human_corr_matrix: pd.DataFrame,
    reference_corr: Optional[pd.DataFrame] = None,
    seed: int = 42,
) -> Dict:
    """Activation → item pipeline (mirrors paper §5.3 main analysis).

    For each target item:
      - features = activation (shape: n_subjects x hidden_dim)
      - target = item response
      - fit RidgeCV with 5-fold CV on cross-validated predictions
    Then aggregate per-item predictions to subscale scores.

    reference_corr defaults to human_corr_matrix (paper setting). For A1_full,
    caller should pass the sample latent correlation matrix as reference so
    that "slope = 1" is the true unbiased baseline.
    """
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import KFold

    if reference_corr is None:
        reference_corr = human_corr_matrix

    n = activations.shape[0]
    item_names = list(items.columns)
    sub_of_item = {it: sub for sub, it in labels}
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)

    pred_matrix = np.zeros((n, len(item_names)))
    for k_idx, target_item in enumerate(item_names):
        y = items[target_item].values
        preds = np.zeros(n)
        for tr, te in kf.split(activations):
            ridge = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0, 1000.0], cv=5)
            ridge.fit(activations[tr], y[tr])
            preds[te] = ridge.predict(activations[te])
        pred_matrix[:, k_idx] = preds

    subscales = list(human_corr_matrix.index)
    subscale_scores = {}
    for sub in subscales:
        cols = [i for i, it in enumerate(item_names) if sub_of_item[it] == sub]
        if cols:
            subscale_scores[sub] = pred_matrix[:, cols].mean(axis=1)
    scores_df = pd.DataFrame(subscale_scores)
    common = [s for s in subscales if s in scores_df.columns]
    rec_corr = scores_df[common].corr().loc[common, common]
    H_human = human_corr_matrix.loc[common, common]
    H_ref = reference_corr.loc[common, common]

    def _slope_and_mantel(ref, rec):
        n_c = len(common)
        idx = np.triu_indices(n_c, k=1)
        rv = rec.values[idx]
        hv = ref.values[idx]
        valid = ~(np.isnan(hv) | np.isnan(rv))
        if valid.sum() < 2:
            return np.nan, np.nan
        slope_v = float(np.polyfit(hv[valid], rv[valid], 1)[0])
        r_mantel_v, _ = mantel_test(ref.values, rec.values, n_permutations=200)
        return slope_v, float(r_mantel_v)

    slope_vs_ref, mantel_vs_ref = _slope_and_mantel(H_ref, rec_corr)
    slope_vs_human, mantel_vs_human = _slope_and_mantel(H_human, rec_corr)
    return {
        "recovered_slope": slope_vs_ref,
        "recovered_mantel_r": mantel_vs_ref,
        "recovered_slope_vs_human": slope_vs_human,
        "recovered_mantel_r_vs_human": mantel_vs_human,
        "n_pairs": int(np.triu_indices(len(common), k=1)[0].size),
    }


def run_A1_full(outdir: Path) -> Dict:
    """High-dim synthetic simulation + SNR sweep.

    Design:
      - Ground truth: human 16×16 correlation matrix
      - Generate 272 subjects × 16 latent subscales ~ MVN(0, ground truth)
      - Generate synthetic items = latent + Gaussian item noise (alpha = 0.75)
      - Generate synthetic activations:
           activation = latent @ W_projection.T + gaussian_noise(sigma)
        Vary sigma to span a range of empirical SNRs
      - Run activation-level item-level ridge pipeline
      - Report recovered slope as a function of (hidden_dim, SNR), 20 seeds

    Hidden dimensions mirror real model dims: [896, 2048, 4096, 5120]
    SNR range covers empirical observation (paper real model Mantel r 0.0–0.66)

    Parallelism: set env var EXTRA_N_JOBS to enable joblib parallelism.
    With 32 cores you should use EXTRA_N_JOBS=16 or 24 (leave some headroom).
    """
    human = load_human_subscale_corr()
    n_subjects = 272
    items_per_subscale = 5
    alpha_item = 0.75

    hidden_dims = [896, 2048, 4096, 5120]
    # SNR = var(signal) / var(noise). For the linear model
    #   activation = latent @ W.T + N(0, sigma^2 I)
    # with W having i.i.d. Gaussian entries of variance 1/n_sub,
    # signal variance ≈ 1 per dim.  So SNR = 1/sigma^2; sigma = 1/sqrt(SNR).
    # Sweep log-spaced SNRs from highly noisy to near-clean.
    snr_levels = [0.1, 0.3, 1.0, 3.0, 10.0]
    n_seeds = 20

    n_jobs = int(os.environ.get("EXTRA_N_JOBS", "1"))
    if n_jobs > 1:
        try:
            from joblib import Parallel, delayed
        except ImportError:
            logger.warning("  joblib not installed; falling back to serial")
            n_jobs = 1

    def _one_run(hdim: int, snr: float, seed: int) -> Dict:
        sigma = 1.0 / np.sqrt(snr)
        rng = np.random.default_rng(seed + 10000 * hidden_dims.index(hdim)
                                     + 100 * snr_levels.index(snr))
        latent = _generate_latent_subscale_scores(n_subjects, human, rng)
        items, labels = _synth_items_from_latent(latent, items_per_subscale,
                                                 alpha_item, rng)
        n_sub = len(latent.columns)
        W = rng.normal(0, 1.0 / np.sqrt(n_sub), size=(hdim, n_sub))
        activations = (latent.values @ W.T
                       + rng.normal(0, sigma, size=(n_subjects, hdim))
                       ).astype(np.float32)
        reference_corr = latent.corr().loc[human.index, human.index]
        res = _run_activation_level_ridge_pipeline(
            activations, items, labels, human,
            reference_corr=reference_corr, seed=seed,
        )
        return {
            "hidden_dim": hdim,
            "snr": snr,
            "sigma": sigma,
            "seed": seed,
            "recovered_slope": res["recovered_slope"],
            "recovered_mantel_r": res["recovered_mantel_r"],
            "recovered_slope_vs_human": res["recovered_slope_vs_human"],
            "recovered_mantel_r_vs_human": res["recovered_mantel_r_vs_human"],
            "n_pairs": res["n_pairs"],
        }

    rows = []
    t0 = time.time()
    n_total = len(hidden_dims) * len(snr_levels) * n_seeds

    # Build job list
    jobs = [(hdim, snr, seed)
            for hdim in hidden_dims
            for snr in snr_levels
            for seed in range(n_seeds)]

    logger.info(f"  A1_full: {n_total} runs "
                f"({len(hidden_dims)} hdims × {len(snr_levels)} SNR × {n_seeds} seeds) "
                f"— n_jobs = {n_jobs}")

    if n_jobs > 1:
        # Process in chunks so we can report progress
        chunk_size = max(1, n_total // 20)
        for start in range(0, n_total, chunk_size):
            chunk = jobs[start : start + chunk_size]
            out = Parallel(n_jobs=n_jobs, backend="loky")(
                delayed(_one_run)(h, s, sd) for (h, s, sd) in chunk
            )
            rows.extend(out)
            elapsed = time.time() - t0
            done_count = len(rows)
            eta = elapsed / done_count * (n_total - done_count) if done_count else 0
            logger.info(f"  ... {done_count}/{n_total}  "
                        f"elapsed {elapsed/60:.1f}min  ETA {eta/60:.1f}min")
    else:
        done_count = 0
        for (hdim, snr, seed) in jobs:
            rows.append(_one_run(hdim, snr, seed))
            done_count += 1
            if done_count % max(1, n_total // 20) == 0:
                elapsed = time.time() - t0
                eta = elapsed / done_count * (n_total - done_count)
                logger.info(f"  ... {done_count}/{n_total} "
                            f"(hdim={hdim}, snr={snr}, seed={seed})  "
                            f"elapsed {elapsed/60:.1f}min  ETA {eta/60:.1f}min")
    df = pd.DataFrame(rows)
    df.to_csv(outdir / "A1_full_all_runs.csv", index=False)

    agg = df.groupby(["hidden_dim", "snr"]).agg(
        n_seeds=("seed", "size"),
        slope_mean=("recovered_slope", "mean"),
        slope_std=("recovered_slope", "std"),
        slope_se=("recovered_slope", lambda x: float(x.std() / np.sqrt(len(x)))),
        mantel_mean=("recovered_mantel_r", "mean"),
        mantel_std=("recovered_mantel_r", "std"),
        slope_vs_human_mean=("recovered_slope_vs_human", "mean"),
        slope_vs_human_se=("recovered_slope_vs_human",
                            lambda x: float(x.std() / np.sqrt(len(x)))),
        mantel_vs_human_mean=("recovered_mantel_r_vs_human", "mean"),
    ).reset_index()
    agg.to_csv(outdir / "A1_full_by_grid.csv", index=False)

    # Empirical real-model SNR estimate:
    # Use mean_reliability as SNR proxy (reliability = var_signal / var_total),
    # SNR = reliability / (1 - reliability)
    svb = load_structure_vs_behavior()
    real_points = []
    for _, r in svb.iterrows():
        rel = r["repr_reliability"]
        if 0.01 < rel < 0.99:
            snr_est = rel / (1 - rel)
        else:
            snr_est = np.nan
        real_points.append({
            "model": r["model"],
            "size": r["size"],
            "family": r["family"],
            "mtype": r["mtype"],
            "repr_mantel_r": r["repr_mantel_r"],
            "repr_corrected_slope": r["repr_corrected_slope"],
            "reliability": rel,
            "snr_est": snr_est,
            "hidden_dim": MODEL_BY_NAME[r["model"]]["hidden_dim"]
                if r["model"] in MODEL_BY_NAME else np.nan,
        })
    pd.DataFrame(real_points).to_csv(outdir / "A1_full_real_model_positions.csv", index=False)

    # Plot: slope vs SNR curves, one line per hidden_dim
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        ax1, ax2, ax3 = axes
        colors = {896: "#8c564b", 2048: "#9467bd", 4096: "#2ca02c", 5120: "#1f77b4"}

        # Panel 1: slope vs SNR (primary — vs sample latent corr, pure pipeline bias)
        for hdim in hidden_dims:
            sub = agg[agg["hidden_dim"] == hdim]
            ax1.errorbar(sub["snr"], sub["slope_mean"], yerr=sub["slope_se"],
                         label=f"hdim={hdim}", marker="o", capsize=3,
                         color=colors[hdim])
        ax1.axhline(1.0, ls="--", color="green", alpha=0.7, label="unbiased (slope=1)")
        ax1.axhspan(1.38, 1.77, alpha=0.15, color="red",
                    label="paper real-model range")
        ax1.set_xscale("log")
        ax1.set_xlabel("SNR  (signal var / noise var)")
        ax1.set_ylabel("slope  (pipeline vs sample latent corr)")
        ax1.set_title("Pure pipeline inflation\n(1.0 = no bias; unrelated to paper direct comparison)")
        ax1.legend(fontsize=8)
        ax1.grid(alpha=0.2)

        # Panel 2: slope vs SNR (vs human — paper-style comparison) + real model overlay
        for hdim in hidden_dims:
            sub = agg[agg["hidden_dim"] == hdim]
            ax2.errorbar(sub["snr"], sub["slope_vs_human_mean"], yerr=sub["slope_vs_human_se"],
                         label=f"hdim={hdim}", marker="o", capsize=3,
                         color=colors[hdim])
        ax2.axhspan(1.38, 1.77, alpha=0.15, color="red",
                    label="paper real-model range")
        for p in real_points:
            if not np.isnan(p["snr_est"]) and not np.isnan(p["repr_corrected_slope"]):
                marker = "o" if p["mtype"] == "instruct" else "s"
                color = "#c0392b" if p["family"] == "llama" else "#2471a3"
                ax2.scatter(p["snr_est"], p["repr_corrected_slope"],
                            marker=marker, color=color, s=60, edgecolors="black",
                            linewidth=0.8, alpha=0.85, zorder=5)
        ax2.set_xscale("log")
        ax2.set_xlabel("SNR")
        ax2.set_ylabel("slope  (pipeline vs human population corr)")
        ax2.set_title("Paper-style comparison\n(real models overlaid: ○=inst, □=base)")
        ax2.legend(fontsize=8)
        ax2.grid(alpha=0.2)

        # Panel 3: Mantel r vs SNR (vs human, since paper uses this)
        for hdim in hidden_dims:
            sub = agg[agg["hidden_dim"] == hdim]
            ax3.errorbar(sub["snr"], sub["mantel_vs_human_mean"],
                         yerr=sub["mantel_std"],
                         label=f"hdim={hdim}", marker="o", capsize=3,
                         color=colors[hdim])
        for p in real_points:
            if not np.isnan(p["snr_est"]) and not np.isnan(p["repr_mantel_r"]):
                marker = "o" if p["mtype"] == "instruct" else "s"
                color = "#c0392b" if p["family"] == "llama" else "#2471a3"
                ax3.scatter(p["snr_est"], p["repr_mantel_r"],
                            marker=marker, color=color, s=60, edgecolors="black",
                            linewidth=0.8, alpha=0.85, zorder=5)
        ax3.set_xscale("log")
        ax3.set_xlabel("SNR")
        ax3.set_ylabel("Mantel r  (vs human population corr)")
        ax3.set_title("Mantel r vs SNR")
        ax3.legend(fontsize=8)
        ax3.grid(alpha=0.2)

        plt.suptitle("A1 full: high-dim synthetic pipeline with SNR sweep", fontsize=12)
        plt.tight_layout()
        plt.savefig(outdir / "A1_full_snr_curves.png", dpi=200, bbox_inches="tight")
        plt.close()
    except Exception as e:
        logger.warning(f"  A1_full plot failed: {e}")

    # Verdict / summary
    # Compute realistic-SNR stats (SNR <= 3 ~ reliability <= 0.75)
    realistic = df[df["snr"] <= 3.0]
    high_snr = df[df["snr"] > 3.0]
    summary = {
        "n_total_runs": int(len(df)),
        "grid_shape": {
            "hidden_dims": hidden_dims, "snr_levels": snr_levels,
            "n_seeds": n_seeds, "items_per_subscale": items_per_subscale,
            "alpha_item": alpha_item,
        },
        "realistic_snr_<=3": {
            "slope_vs_ref_mean": float(realistic["recovered_slope"].mean()),
            "slope_vs_ref_max": float(realistic["recovered_slope"].max()),
            "slope_vs_human_mean": float(realistic["recovered_slope_vs_human"].mean()),
            "slope_vs_human_max": float(realistic["recovered_slope_vs_human"].max()),
            "slope_vs_human_p95": float(realistic["recovered_slope_vs_human"].quantile(0.95)),
        },
        "high_snr_>3": {
            "slope_vs_ref_mean": float(high_snr["recovered_slope"].mean()) if len(high_snr) else None,
            "slope_vs_human_mean": float(high_snr["recovered_slope_vs_human"].mean()) if len(high_snr) else None,
            "slope_vs_human_max": float(high_snr["recovered_slope_vs_human"].max()) if len(high_snr) else None,
        },
        "paper_real_slope_range": [1.38, 1.77],
        "elapsed_sec": time.time() - t0,
    }

    # Verdict based on vs-ref (pure pipeline bias) + vs-human (paper comparison)
    max_realistic_ref = summary["realistic_snr_<=3"]["slope_vs_ref_max"]
    max_realistic_human = summary["realistic_snr_<=3"]["slope_vs_human_max"]
    mean_realistic_ref = summary["realistic_snr_<=3"]["slope_vs_ref_mean"]

    if mean_realistic_ref >= 1.3:
        verdict = (
            f"SEVERE: at realistic SNR (≤3), pure pipeline inflation "
            f"(slope vs sample latent corr) averages {mean_realistic_ref:.3f}. "
            f"A substantial portion of the real-model slope range [1.38, 1.77] "
            f"could be attributable to ridge pipeline alone."
        )
    elif mean_realistic_ref >= 1.15:
        verdict = (
            f"MODERATE: pure pipeline inflation at realistic SNR averages "
            f"{mean_realistic_ref:.3f}. Non-trivial but cannot alone explain "
            f"the real-model range [1.38, 1.77] — substantial portion must come "
            f"from genuine representational amplification."
        )
    else:
        verdict = (
            f"NEGLIGIBLE: pure pipeline inflation at realistic SNR averages "
            f"{mean_realistic_ref:.3f} ≈ 1.0. Ridge is essentially unbiased; "
            f"the real-model slope range [1.38, 1.77] reflects genuine "
            f"representational amplification."
        )
    summary["verdict"] = verdict
    (outdir / "A1_full_summary.json").write_text(json.dumps(summary, indent=2, default=str))

    logger.info(f"  {summary['n_total_runs']} runs over "
                f"{len(hidden_dims)} hidden_dims × {len(snr_levels)} SNR × "
                f"{n_seeds} seeds  (elapsed {summary['elapsed_sec']/60:.1f} min)")
    logger.info(f"  Realistic SNR (≤3) slope vs sample latent (pure bias):")
    logger.info(f"    mean = {summary['realistic_snr_<=3']['slope_vs_ref_mean']:.3f}, "
                f"max = {summary['realistic_snr_<=3']['slope_vs_ref_max']:.3f}")
    logger.info(f"  Realistic SNR (≤3) slope vs human (paper-style):")
    logger.info(f"    mean = {summary['realistic_snr_<=3']['slope_vs_human_mean']:.3f}, "
                f"max = {summary['realistic_snr_<=3']['slope_vs_human_max']:.3f}, "
                f"p95 = {summary['realistic_snr_<=3']['slope_vs_human_p95']:.3f}")
    if summary["high_snr_>3"]["slope_vs_ref_mean"] is not None:
        logger.info(f"  High SNR (>3) slope vs sample latent: "
                    f"mean = {summary['high_snr_>3']['slope_vs_ref_mean']:.3f}")
    logger.info(f"\n  Verdict: {verdict}")
    logger.info(f"\n  By grid (slope vs sample latent corr, mean ± SE):")
    for _, r in agg.iterrows():
        logger.info(
            f"    hdim={int(r['hidden_dim']):>5}  snr={r['snr']:>5.2f}: "
            f"ref={r['slope_mean']:.3f} ± {r['slope_se']:.3f}  "
            f"hum={r['slope_vs_human_mean']:.3f}  "
            f"mantel_hum={r['mantel_vs_human_mean']:.3f}"
        )

    return summary


# ═════════════════════════════════════════════════════════════════════════
#                    T2: USE EXISTING SAVED ACTIVATIONS
# ═════════════════════════════════════════════════════════════════════════
#  These experiments use the already-extracted `subject_activations.npz` files
#  stored in outputs/mechanistic/results_{name}/pca/ and supporting artifacts. No GPU needed.
# ═════════════════════════════════════════════════════════════════════════


# ─────────────────────────────────────────────────────────────────────────
#  Experiment: A1_adj  Pipeline-baseline-adjusted slope per real model
# ─────────────────────────────────────────────────────────────────────────

def run_A1_adj(outdir: Path) -> Dict:
    """Pipeline-baseline-adjusted item-level representational slope per real model.

    Paper reports item-level cross-persona pipeline slopes 1.38-1.77 across 9
    models (itemlevel_comparison.py Option 1 output).  These are NOT the same
    as `structure_vs_behavior.csv`'s `repr_corrected_slope` (which comes from
    a different, contrastive-direction pipeline with attenuation correction).

    This experiment:
      1. Uses the 9 published item-level pipeline slopes as 'observed'.
      2. Uses A1_full synthetic SNR grid to estimate pipeline baseline slope
         at each model's SNR (proxy: mean_reliability from scaling_summary.csv).
      3. Computes adjusted slope = observed - baseline.

    CAVEAT: the SNR proxy (mean_reliability from directions-based pipeline)
    is not the same reliability that the item-level pipeline would exhibit;
    the latter would require re-running itemlevel_comparison.py with per-
    subject predictions saved + split-half reliability. This adjustment is
    therefore approximate.  If the A1_full inflation curve is relatively flat
    across plausible SNR range the approximation is mild; if steep, results
    may shift materially after proper reliability is computed.
    """
    a1_full_csv = OUTPUT_ROOT / "A1_full" / "A1_full_all_runs.csv"
    if not a1_full_csv.exists():
        logger.error(f"  A1_full output not found at {a1_full_csv}. "
                     f"Run A1_full first.")
        return {"error": "A1_full not done"}

    a1 = pd.read_csv(a1_full_csv)

    # Hard-coded item-level pipeline slopes from
    # `python -m psychometric_inference.mechanisms.cross_persona --analyze_only` output (Option 1, continuous).
    # Layer numbers are the specific layer used by itemlevel_comparison for
    # each model; these are the layers we used for evaluating each model.
    itemlevel_slopes = {
        "qwen05b_instruct":  {"slope": 1.470, "layer":  8},
        "llama1b_instruct":  {"slope": 1.440, "layer":  8},
        "qwen3b_instruct":   {"slope": 1.384, "layer": 14},
        "llama3b_instruct":  {"slope": 1.628, "layer": 14},
        "qwen7b_instruct":   {"slope": 1.653, "layer": 18},
        "llama8b_instruct":  {"slope": 1.697, "layer": 16},
        "llama8b_base":      {"slope": 1.634, "layer": 14},
        "qwen14b_instruct":  {"slope": 1.769, "layer": 32},
        "qwen14b_base":      {"slope": 1.698, "layer": 28},
    }

    # Load scaling summary for reliability (SNR proxy)
    scaling = load_scaling_summary()
    scaling_by_model = {r["model"]: r for _, r in scaling.iterrows()}

    # Load beh_subscale_slope + repr_mantel_r from structure_vs_behavior.csv
    # (these come from the contrastive pipeline; include for correlation
    # analyses relative to paper Fig 5)
    svb = load_structure_vs_behavior()
    svb_by_model = {r["model"]: r for _, r in svb.iterrows()}

    from scipy.interpolate import interp1d

    a1_grid = a1.groupby(["hidden_dim", "snr"]).agg(
        slope_mean=("recovered_slope_vs_human", "mean"),
        slope_std=("recovered_slope_vs_human", "std"),
    ).reset_index()

    rows = []
    for name, info in itemlevel_slopes.items():
        if name not in MODEL_BY_NAME:
            continue
        m_cfg = MODEL_BY_NAME[name]
        scal = scaling_by_model.get(name)
        svb_row = svb_by_model.get(name)
        if scal is None:
            logger.warning(f"  [{name}] missing from scaling_summary.csv; skip")
            continue

        rel = scal["mean_reliability"]
        if not (0.01 < rel < 0.99):
            continue
        snr_est = rel / (1 - rel)
        log_snr = np.log10(snr_est)

        model_hdim = m_cfg["hidden_dim"]
        grid_hdims = sorted(a1_grid["hidden_dim"].unique())
        nearest_hdim = min(grid_hdims, key=lambda h: abs(h - model_hdim))

        sub = a1_grid[a1_grid["hidden_dim"] == nearest_hdim].sort_values("snr")
        log_snr_grid = np.log10(sub["snr"].values)
        slopes = sub["slope_mean"].values
        stds = sub["slope_std"].values

        if log_snr < log_snr_grid.min():
            baseline_slope = float(slopes[0])
            baseline_std = float(stds[0])
            method = "clipped_low"
        elif log_snr > log_snr_grid.max():
            baseline_slope = float(slopes[-1])
            baseline_std = float(stds[-1])
            method = "clipped_high"
        else:
            baseline_slope = float(interp1d(log_snr_grid, slopes)(log_snr))
            baseline_std = float(interp1d(log_snr_grid, stds)(log_snr))
            method = "interpolated"

        observed = float(info["slope"])
        adjusted = observed - baseline_slope

        rows.append({
            "model": name,
            "size": m_cfg["size"],
            "family": m_cfg["family"],
            "mtype": m_cfg["mtype"],
            "itemlevel_layer": info["layer"],
            "reliability_proxy": float(rel),
            "snr_proxy": float(snr_est),
            "model_hidden_dim": model_hdim,
            "nearest_grid_hdim": nearest_hdim,
            "interpolation": method,
            "observed_slope_itemlevel": observed,
            "pipeline_baseline_slope": baseline_slope,
            "pipeline_baseline_std": baseline_std,
            "adjusted_slope": adjusted,
            "beh_subscale_slope": float(svb_row["beh_subscale_slope"])
                if svb_row is not None else np.nan,
            "repr_mantel_r_contrastive": float(svb_row["repr_mantel_r"])
                if svb_row is not None else np.nan,
        })

    df = pd.DataFrame(rows)
    df.to_csv(outdir / "A1_adj_per_model.csv", index=False)

    summary = {
        "n_models": int(len(df)),
        "snr_proxy_note": (
            "SNR proxy from scaling_summary.csv mean_reliability; this is "
            "the contrastive-directions pipeline reliability, not item-level. "
            "Proper correction would require split-half reliability of the "
            "item-level predictions themselves."
        ),
        "observed_slope_range": [float(df["observed_slope_itemlevel"].min()),
                                  float(df["observed_slope_itemlevel"].max())],
        "observed_slope_spread": float(df["observed_slope_itemlevel"].max()
                                        - df["observed_slope_itemlevel"].min()),
        "baseline_slope_range": [float(df["pipeline_baseline_slope"].min()),
                                  float(df["pipeline_baseline_slope"].max())],
        "adjusted_slope_range": [float(df["adjusted_slope"].min()),
                                  float(df["adjusted_slope"].max())],
        "adjusted_slope_spread": float(df["adjusted_slope"].max()
                                        - df["adjusted_slope"].min()),
        "mean_adjusted_slope": float(df["adjusted_slope"].mean()),
        "instruct_models_adjusted_mean": float(
            df[df["mtype"] == "instruct"]["adjusted_slope"].mean())
            if (df["mtype"] == "instruct").any() else None,
        "base_models_adjusted_mean": float(
            df[df["mtype"] == "base"]["adjusted_slope"].mean())
            if (df["mtype"] == "base").any() else None,
    }

    # Correlations — adjusted slope vs behavioral slope
    if len(df) >= 3 and df["beh_subscale_slope"].notna().all():
        from scipy.stats import pearsonr as _pr
        r_obs_beh, p_obs_beh = _pr(df["observed_slope_itemlevel"],
                                    df["beh_subscale_slope"])
        r_adj_beh, p_adj_beh = _pr(df["adjusted_slope"],
                                    df["beh_subscale_slope"])
        r_obs_size, p_obs_size = _pr(np.log(df["size"]),
                                      df["observed_slope_itemlevel"])
        r_adj_size, p_adj_size = _pr(np.log(df["size"]),
                                      df["adjusted_slope"])
        summary["correlations"] = {
            "observed_slope_vs_beh_slope": {"r": float(r_obs_beh),
                                             "p": float(p_obs_beh)},
            "adjusted_slope_vs_beh_slope": {"r": float(r_adj_beh),
                                             "p": float(p_adj_beh)},
            "log_size_vs_observed_slope":  {"r": float(r_obs_size),
                                             "p": float(p_obs_size)},
            "log_size_vs_adjusted_slope":  {"r": float(r_adj_size),
                                             "p": float(p_adj_size)},
        }

    (outdir / "A1_adj_summary.json").write_text(
        json.dumps(summary, indent=2, default=str))

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

        sorted_df = df.sort_values("size")
        x = np.arange(len(sorted_df))
        w = 0.27
        ax1.bar(x - w, sorted_df["observed_slope_itemlevel"], w,
                label="observed (paper item-level)", color="#1f77b4")
        ax1.bar(x, sorted_df["pipeline_baseline_slope"], w,
                label="pipeline baseline (A1_full SNR-matched)", color="gray")
        ax1.bar(x + w, sorted_df["adjusted_slope"], w,
                label="adjusted (observed − baseline)", color="#d62728")
        ax1.axhline(0, color="black", linewidth=0.5)
        ax1.axhline(1.0, ls="--", color="green", alpha=0.5,
                    label="unbiased threshold (slope=1)")
        ax1.set_xticks(x)
        ax1.set_xticklabels(
            [f"{r['model']}\n({r['size']:g}B)"
             for _, r in sorted_df.iterrows()],
            rotation=60, ha="right", fontsize=7
        )
        ax1.set_ylabel("slope")
        ax1.set_title("Observed (paper item-level) vs pipeline baseline vs adjusted")
        ax1.legend(fontsize=8)
        ax1.grid(alpha=0.2, axis="y")

        if df["beh_subscale_slope"].notna().all():
            ax2.scatter(df["adjusted_slope"], df["beh_subscale_slope"],
                        c=["#c0392b" if f == "llama" else "#2471a3"
                           for f in df["family"]],
                        marker="o", s=80, edgecolors="black", alpha=0.85)
            for _, r in df.iterrows():
                ax2.annotate(f"{r['size']:g}B{r['mtype'][0]}",
                             (r["adjusted_slope"], r["beh_subscale_slope"]),
                             fontsize=7, xytext=(4, 4),
                             textcoords="offset points")
            if "correlations" in summary:
                c = summary["correlations"]
                r_text = (
                    f"adjusted vs beh:  r = {c['adjusted_slope_vs_beh_slope']['r']:.3f}  "
                    f"p = {c['adjusted_slope_vs_beh_slope']['p']:.4f}\n"
                    f"observed vs beh:  r = {c['observed_slope_vs_beh_slope']['r']:.3f}  "
                    f"p = {c['observed_slope_vs_beh_slope']['p']:.4f}\n"
                    f"log(size) vs adjusted:  r = {c['log_size_vs_adjusted_slope']['r']:.3f}"
                )
                ax2.text(0.05, 0.95, r_text, transform=ax2.transAxes,
                         fontsize=9, verticalalignment="top",
                         bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
            ax2.axvline(0, color="black", linewidth=0.5)
            ax2.set_xlabel("adjusted slope (observed − pipeline baseline)")
            ax2.set_ylabel("behavioral subscale slope")
            ax2.set_title("Adjusted vs behavioral slope")
            ax2.grid(alpha=0.2)

        plt.suptitle(
            f"A1_adj: pipeline-baseline-adjusted item-level slope\n"
            f"observed spread={summary['observed_slope_spread']:.2f}, "
            f"adjusted spread={summary['adjusted_slope_spread']:.2f}",
            fontsize=11
        )
        plt.tight_layout()
        plt.savefig(outdir / "A1_adj_bars_and_scatter.png",
                    dpi=200, bbox_inches="tight")
        plt.close()
    except Exception as e:
        logger.warning(f"  A1_adj plot failed: {e}")

    logger.info(f"  {len(df)} models with item-level pipeline slopes")
    logger.info(f"  {'model':<22} {'size':>5} {'rel':>6} {'SNR':>7}"
                f" {'observed':>9} {'baseline':>9} {'adjusted':>9}")
    for _, r in df.sort_values("size").iterrows():
        logger.info(
            f"  {r['model']:<22} {r['size']:>5.1f} "
            f"{r['reliability_proxy']:>6.3f} {r['snr_proxy']:>7.3f} "
            f"{r['observed_slope_itemlevel']:>9.3f} "
            f"{r['pipeline_baseline_slope']:>9.3f} "
            f"{r['adjusted_slope']:>+9.3f}"
        )
    logger.info(f"\n  Range summary:")
    logger.info(f"    observed slope  range: [{summary['observed_slope_range'][0]:.3f}, "
                f"{summary['observed_slope_range'][1]:.3f}]  "
                f"spread {summary['observed_slope_spread']:.3f}")
    logger.info(f"    baseline slope range: [{summary['baseline_slope_range'][0]:.3f}, "
                f"{summary['baseline_slope_range'][1]:.3f}]")
    logger.info(f"    ADJUSTED slope range: [{summary['adjusted_slope_range'][0]:+.3f}, "
                f"{summary['adjusted_slope_range'][1]:+.3f}]  "
                f"spread {summary['adjusted_slope_spread']:.3f}")
    logger.info(f"    mean adjusted slope: {summary['mean_adjusted_slope']:+.3f}")
    if summary.get("instruct_models_adjusted_mean") is not None:
        logger.info(f"    instruct mean adjusted: {summary['instruct_models_adjusted_mean']:+.3f}")
    if summary.get("base_models_adjusted_mean") is not None:
        logger.info(f"    base mean adjusted:     {summary['base_models_adjusted_mean']:+.3f}")
    if "correlations" in summary:
        c = summary["correlations"]
        logger.info(f"\n  Paper-relevant correlations:")
        logger.info(f"    observed slope vs beh slope:   r = {c['observed_slope_vs_beh_slope']['r']:+.3f}  "
                    f"(p = {c['observed_slope_vs_beh_slope']['p']:.4f})")
        logger.info(f"    ADJUSTED slope vs beh slope:   r = {c['adjusted_slope_vs_beh_slope']['r']:+.3f}  "
                    f"(p = {c['adjusted_slope_vs_beh_slope']['p']:.4f})")
        logger.info(f"    log(size) vs observed slope:   r = {c['log_size_vs_observed_slope']['r']:+.3f}  "
                    f"(p = {c['log_size_vs_observed_slope']['p']:.4f})")
        logger.info(f"    log(size) vs ADJUSTED slope:   r = {c['log_size_vs_adjusted_slope']['r']:+.3f}  "
                    f"(p = {c['log_size_vs_adjusted_slope']['p']:.4f})")

    logger.info(f"\n  NOTE: SNR proxy from scaling_summary.csv `mean_reliability`")
    logger.info(f"        (contrastive-directions pipeline reliability, not item-level).")
    logger.info(f"        Proper correction would require split-half reliability of")
    logger.info(f"        item-level subscale predictions. Interpret as approximate.")

    return summary


# ═════════════════════════════════════════════════════════════════════════
#             Shared item-level cross-persona pipeline
# ═════════════════════════════════════════════════════════════════════════
#
#  B2/A2/A3 all build on the paper's item-level cross-persona pipeline
#  (itemlevel_comparison.py).  We replicate the loader here and parameterize
#  over (probe, discretization).  Schema is taken from that script:
#
#    - outputs/mechanistic/results_{model}/cross_persona/rotation_{short}/activations.npz
#      keyed by f"L{layer}"
#    - outputs/mechanistic/results_{model}/cross_persona/rotation_{short}/meta.csv with
#      columns: subject_id, target_scale, item_number  (plus others ignored)
#    - Human item responses live in data/human/{SED,SEDC,SEDD}/{scale_file}.csv
#      with Subject_ID (or Scan_ID or ID) + Q{n} columns.
#    - Reverse coding / subscale membership from scoring.SCORING_RULES.
# ═════════════════════════════════════════════════════════════════════════

def _cross_persona_load_human_items() -> Dict[str, Dict[int, float]]:
    """Return {subject_id: {scale_file: {item_number: response}}} across
    all three data partitions, un-reverse-coded.
    """
    try:
        from psychometric_inference.scoring import un_reverse_df
    except ImportError:
        logger.warning("  cannot import psychometric_inference.scoring.un_reverse_df; item-level "
                       "experiments will be skipped.")
        return {}

    # Replicate SCALES from config.py
    SCALES = [
        ("IRI", "IRI"),
        ("PANAS", "PANAS"),
        ("POM", "POM"),
        ("big_five", "BigFive"),
        ("in_inter_dependent", "SelfConst"),
        ("Life_Satisfaction", "LifeSat"),
        ("Loneliness", "Lonely"),
    ]

    human_items: Dict[str, Dict[str, Dict[int, float]]] = {}
    for scale_file, _scale_short in SCALES:
        frames = []
        for ds in ["SED", "SEDC", "SEDD"]:
            csv_path = HUMAN_DATA_DIR / ds / f"{scale_file}.csv"
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                try:
                    df = un_reverse_df(df, scale_file)
                except Exception as e:
                    logger.warning(f"    un_reverse_df failed for {scale_file}/{ds}: {e}")
                frames.append(df)
        if not frames:
            continue
        combined = pd.concat(frames, ignore_index=True)
        id_col = next((c for c in ["Subject_ID", "ID", "Scan_ID"]
                       if c in combined.columns), None)
        if id_col:
            combined = combined.rename(columns={id_col: "Subject_ID"})
        else:
            combined["Subject_ID"] = [f"S{i:04d}" for i in range(len(combined))]
        combined["Subject_ID"] = combined["Subject_ID"].astype(str)

        for _, row in combined.iterrows():
            sid = str(row["Subject_ID"])
            human_items.setdefault(sid, {}).setdefault(scale_file, {})
            for col in combined.columns:
                if col.startswith("Q"):
                    try:
                        item_num = int(col[1:])
                        val = row[col]
                        if not pd.isna(val):
                            human_items[sid][scale_file][item_num] = float(val)
                    except ValueError:
                        pass
    return human_items


def _cross_persona_pipeline(
    model_name: str,
    layer: int,
    probe_factory: Callable,
    subject_ids: Optional[List[str]] = None,
    human_items: Optional[Dict] = None,
) -> Optional[Dict]:
    """Run item-level cross-persona ridge-style pipeline for one model+probe.

    Parameters
    ----------
    model_name : str
        e.g. "llama8b_instruct"
    layer : int
        Layer index to use (from itemlevel_slopes_layers)
    probe_factory : callable
        A zero-arg callable that returns a fresh scikit-learn-style estimator
        (must have fit(X, y) and predict(X)). For multi-target probes that
        expect Y as a matrix this won't work; use _cross_persona_pipeline_multi
        instead for those.

    Returns
    -------
    {
      "predictions": np.ndarray shape (n_subj, n_items_total),
      "subject_ids": list[str],
      "item_labels":  list[(scale_file, item_num, subscale_full_name)],
      "subscale_scores_df": pd.DataFrame (subjects × subscales, continuous),
      "subscale_scores_discrete_df": pd.DataFrame  (integer-rounded-then-reverse-coded),
      "pred_corr_continuous": pd.DataFrame 16×16,
      "pred_corr_discrete":   pd.DataFrame 16×16,
    }
    or None on failure.
    """
    try:
        from psychometric_inference.scoring import SCORING_RULES
        from sklearn.model_selection import cross_val_predict
    except ImportError as e:
        logger.warning(f"  missing dependency: {e}")
        return None

    SCALES = [
        ("IRI", "IRI"),
        ("PANAS", "PANAS"),
        ("POM", "POM"),
        ("big_five", "BigFive"),
        ("in_inter_dependent", "SelfConst"),
        ("Life_Satisfaction", "LifeSat"),
        ("Loneliness", "Lonely"),
    ]

    cp_dir = MECHANISTIC_ROOT / f"results_{model_name}" / "cross_persona"
    if not cp_dir.exists() and model_name == "llama8b_instruct":
        cp_dir = MECHANISTIC_DEFAULT_RESULTS_DIR / "cross_persona"
    if not cp_dir.exists():
        return None

    if human_items is None:
        human_items = _cross_persona_load_human_items()
    if not human_items:
        return None
    if subject_ids is None:
        subject_ids = sorted(human_items.keys())

    # Stage 1: per-(rotation, target_scale, item_number) ridge CV predictions.
    item_predictions: Dict[Tuple[str, str, int], Dict[str, float]] = {}

    for persona_file, persona_short in SCALES:
        rotation_dir = cp_dir / f"rotation_{persona_short}"
        act_path = rotation_dir / "activations.npz"
        meta_path = rotation_dir / "meta.csv"
        if not (act_path.exists() and meta_path.exists()):
            continue
        try:
            with np.load(act_path) as npz:
                key = f"L{layer}"
                if key not in npz.files:
                    continue
                acts = npz[key]
        except Exception:
            continue
        meta_df = pd.read_csv(meta_path)
        if "subject_id" not in meta_df.columns:
            continue
        meta_df["subject_id"] = meta_df["subject_id"].astype(str)

        for (target_scale, item_num), group in meta_df.groupby(
            ["target_scale", "item_number"]
        ):
            # Map target_scale back to scale_file
            target_file = None
            for sf, ss in SCALES:
                if ss == target_scale:
                    target_file = sf
                    break
            if target_file is None:
                continue

            X_list, y_list, sid_list = [], [], []
            for idx_pos, sid in zip(group.index.values, group["subject_id"].values):
                if idx_pos >= len(acts):
                    continue
                if sid not in human_items or target_file not in human_items[sid]:
                    continue
                v = human_items[sid][target_file].get(int(item_num))
                if v is None:
                    continue
                X_list.append(acts[idx_pos])
                y_list.append(float(v))
                sid_list.append(sid)

            if len(X_list) < 50:
                continue
            X = np.array(X_list)
            y = np.array(y_list)
            try:
                probe = probe_factory()
                y_pred = cross_val_predict(probe, X, y, cv=5)
            except Exception:
                continue
            item_predictions[(persona_short, target_file, int(item_num))] = {
                sid: float(yp) for sid, yp in zip(sid_list, y_pred)
            }

    if not item_predictions:
        logger.warning(f"  [{model_name}] no predictions; skipping")
        return None

    # Stage 2: aggregate per (subject, subscale). Exclude same-scale rotation.
    # Continuous: apply reverse-coding after averaging across rotations.
    # Discrete:  round to int first, clip, then reverse-code (like argmax).
    subscale_cont: Dict[str, Dict[str, float]] = {}
    subscale_disc: Dict[str, Dict[str, float]] = {}
    item_labels = []  # (scale_file, item_num, subscale_full)

    for scale_file, scale_short in SCALES:
        rules = SCORING_RULES.get(scale_short, {})
        rev = set(rules.get("reverse_items", []))
        max_val = rules.get("max_val", 5)

        for sub_name, item_nums in rules.get("subscales", {}).items():
            sub_full = f"{scale_short}_{sub_name}"
            cont_scores: Dict[str, float] = {}
            disc_scores: Dict[str, float] = {}
            for sid in subject_ids:
                cont_item_vals = []
                disc_item_vals = []
                for item_num in item_nums:
                    # Collect preds across rotations, excluding same-scale persona
                    preds = []
                    for persona_file, persona_short in SCALES:
                        if persona_short == scale_short:
                            continue
                        key = (persona_short, scale_file, int(item_num))
                        if key in item_predictions and sid in item_predictions[key]:
                            preds.append(item_predictions[key][sid])
                    if not preds:
                        continue
                    mean_pred = float(np.mean(preds))
                    # Continuous: reverse-code on raw continuous
                    cv = (max_val + 1) - mean_pred if item_num in rev else mean_pred
                    cont_item_vals.append(cv)
                    # Discrete: round→clip→reverse-code (like argmax)
                    rounded = int(np.clip(round(mean_pred), 1, max_val))
                    dv = (max_val + 1) - rounded if item_num in rev else rounded
                    disc_item_vals.append(dv)
                if cont_item_vals:
                    cont_scores[sid] = float(np.mean(cont_item_vals))
                if disc_item_vals:
                    disc_scores[sid] = float(np.mean(disc_item_vals))
            if cont_scores:
                subscale_cont[sub_full] = cont_scores
            if disc_scores:
                subscale_disc[sub_full] = disc_scores

    available = [s for s in ALL_SUBSCALES if s in subscale_cont]
    if len(available) < 3:
        return None

    pred_cont_df = pd.DataFrame(
        {s: [subscale_cont[s].get(sid, np.nan) for sid in subject_ids]
         for s in available}, index=subject_ids
    )
    pred_disc_df = pd.DataFrame(
        {s: [subscale_disc[s].get(sid, np.nan) for sid in subject_ids]
         for s in available}, index=subject_ids
    )

    def _corr_matrix(df_: pd.DataFrame) -> pd.DataFrame:
        n = len(df_.columns)
        M = np.zeros((n, n))
        cols = list(df_.columns)
        for i, si in enumerate(cols):
            for j, sj in enumerate(cols):
                vi = df_[si].values; vj = df_[sj].values
                v = ~(np.isnan(vi) | np.isnan(vj))
                if v.sum() > 10:
                    M[i, j] = float(np.corrcoef(vi[v], vj[v])[0, 1])
                else:
                    M[i, j] = np.nan
        return pd.DataFrame(M, index=cols, columns=cols)

    return {
        "subject_ids": subject_ids,
        "available_subscales": available,
        "subscale_scores_df": pred_cont_df,
        "subscale_scores_discrete_df": pred_disc_df,
        "pred_corr_continuous": _corr_matrix(pred_cont_df),
        "pred_corr_discrete":   _corr_matrix(pred_disc_df),
    }


# itemlevel layer per model (from itemlevel_comparison.py).
# Used by B2 / A2 / A3 to run the pipeline at the matching layer per model.
ITEMLEVEL_LAYERS = {
    "qwen05b_instruct": 8,
    "llama1b_instruct": 8,
    "qwen3b_instruct":  14,
    "llama3b_instruct": 14,
    "qwen7b_instruct":  18,
    "llama8b_instruct": 16,
    "llama8b_base":     14,
    "qwen14b_instruct": 32,
    "qwen14b_base":     28,
}


def _slope_mantel_vs_human(pred_corr: pd.DataFrame,
                            human_corr: pd.DataFrame) -> Tuple[float, float, int]:
    """OLS slope + Mantel r between pred and human 16×16 matrices.
    Returns (slope, mantel_r, n_pairs)."""
    common = [s for s in pred_corr.index if s in human_corr.index]
    if len(common) < 3:
        return np.nan, np.nan, 0
    H = human_corr.loc[common, common].values
    P = pred_corr.loc[common, common].values
    idx = np.triu_indices(len(common), k=1)
    hv = H[idx]; pv = P[idx]
    v = ~(np.isnan(hv) | np.isnan(pv))
    if v.sum() < 2:
        return np.nan, np.nan, 0
    slope = float(np.polyfit(hv[v], pv[v], 1)[0])
    mr, _ = mantel_test(H, P, n_permutations=200)
    return slope, float(mr), int(v.sum())


# ─────────────────────────────────────────────────────────────────────────
#  Experiment: B2  Continuous vs discrete (rounded) subscale slope
# ─────────────────────────────────────────────────────────────────────────

def run_B2(outdir: Path) -> Dict:
    """Continuous vs integer-discretized subscale slope per model.

    Paper itemlevel_comparison.py reports Option 1 (continuous) and Option 2
    (rounded to int then averaged) for each model, but only aggregate numbers
    make it into the paper (~1.44 continuous, ~0.83 rounded for llama8b_inst).
    This experiment reports the pair per-model across all 9 cross-persona
    models, giving a per-model view of the discretization loss.
    """
    from sklearn.linear_model import RidgeCV
    human = load_human_subscale_corr()
    human_items = _cross_persona_load_human_items()
    if not human_items:
        return {"error": "could not load human items"}

    rows = []
    for model_name, layer in ITEMLEVEL_LAYERS.items():
        logger.info(f"  [{model_name} L{layer}] running …")
        t0 = time.time()
        probe = lambda: RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0, 1000.0], cv=5)
        res = _cross_persona_pipeline(model_name, layer, probe,
                                       human_items=human_items)
        if res is None:
            continue
        slope_c, mr_c, n_c = _slope_mantel_vs_human(res["pred_corr_continuous"], human)
        slope_d, mr_d, _   = _slope_mantel_vs_human(res["pred_corr_discrete"],   human)
        rows.append({
            "model": model_name,
            "size":  MODEL_BY_NAME[model_name]["size"],
            "mtype": MODEL_BY_NAME[model_name]["mtype"],
            "family": MODEL_BY_NAME[model_name]["family"],
            "layer": layer,
            "n_pairs": n_c,
            "slope_continuous": slope_c,
            "slope_discrete":   slope_d,
            "delta_cont_to_discrete": slope_c - slope_d,
            "mantel_continuous": mr_c,
            "mantel_discrete":   mr_d,
            "elapsed_sec": time.time() - t0,
        })
        logger.info(f"    cont slope={slope_c:.3f} (mantel {mr_c:.3f})  "
                    f"disc slope={slope_d:.3f} (mantel {mr_d:.3f})  "
                    f"Δ={slope_c - slope_d:+.3f}  [{time.time()-t0:.0f}s]")

    df = pd.DataFrame(rows)
    df.to_csv(outdir / "B2_per_model.csv", index=False)

    summary = {"n_models": int(len(df))}
    if len(df):
        summary.update({
            "mean_slope_continuous": float(df["slope_continuous"].mean()),
            "mean_slope_discrete":   float(df["slope_discrete"].mean()),
            "mean_delta_cont_to_discrete": float(df["delta_cont_to_discrete"].mean()),
            "mean_mantel_continuous":      float(df["mantel_continuous"].mean()),
            "mean_mantel_discrete":        float(df["mantel_discrete"].mean()),
        })
        # Paired test on slope
        try:
            from scipy.stats import ttest_rel, wilcoxon
            t, pt = ttest_rel(df["slope_continuous"], df["slope_discrete"])
            w, pw = wilcoxon(df["slope_continuous"], df["slope_discrete"])
            summary["paired_test"] = {
                "t_p": float(pt), "wilcoxon_p": float(pw),
                "mean_diff": summary["mean_delta_cont_to_discrete"],
            }
        except Exception:
            pass

    (outdir / "B2_summary.json").write_text(
        json.dumps(summary, indent=2, default=str))

    if len(df):
        logger.info(f"\n  Summary across {len(df)} models:")
        logger.info(f"    mean slope continuous  = {summary['mean_slope_continuous']:.3f}")
        logger.info(f"    mean slope discretized = {summary['mean_slope_discrete']:.3f}")
        logger.info(f"    mean Δ (cont − disc)   = {summary['mean_delta_cont_to_discrete']:+.3f}")
        if "paired_test" in summary:
            logger.info(f"    paired t p = {summary['paired_test']['t_p']:.4f}, "
                        f"wilcoxon p = {summary['paired_test']['wilcoxon_p']:.4f}")

    return summary


# ─────────────────────────────────────────────────────────────────────────
#  Experiment: A2  Alternative probes (Ridge vs Lasso / PLS / SVR)
# ─────────────────────────────────────────────────────────────────────────

def run_A2(outdir: Path) -> Dict:
    """Alternative probes vs Ridge on the same item-level pipeline.

    Re-runs cross-persona item-level pipeline for each of 9 models using:
      - Ridge   (paper default)
      - Lasso   (L1 regularization)
      - PLS     (n_components=10, using cross_val_predict)
      - LinearSVR
    Reports slope / Mantel r (continuous branch only) for each probe.
    """
    from sklearn.linear_model import RidgeCV, LassoCV
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.svm import LinearSVR
    human = load_human_subscale_corr()
    human_items = _cross_persona_load_human_items()
    if not human_items:
        return {"error": "could not load human items"}

    probes = {
        "ridge": lambda: RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0, 1000.0], cv=5),
        "lasso": lambda: LassoCV(alphas=[0.001, 0.01, 0.1, 1.0], cv=5, max_iter=2000),
        "pls10": lambda: PLSRegression(n_components=10),
        "svr":   lambda: LinearSVR(C=1.0, max_iter=5000),
    }

    rows = []
    for model_name, layer in ITEMLEVEL_LAYERS.items():
        for probe_name, factory in probes.items():
            logger.info(f"  [{model_name} L{layer}] probe={probe_name}")
            t0 = time.time()
            res = _cross_persona_pipeline(model_name, layer, factory,
                                           human_items=human_items)
            if res is None:
                continue
            slope, mr, n_pairs = _slope_mantel_vs_human(
                res["pred_corr_continuous"], human
            )
            rows.append({
                "model": model_name,
                "size":  MODEL_BY_NAME[model_name]["size"],
                "mtype": MODEL_BY_NAME[model_name]["mtype"],
                "family": MODEL_BY_NAME[model_name]["family"],
                "layer": layer,
                "probe": probe_name,
                "slope": slope,
                "mantel_r": mr,
                "n_pairs": n_pairs,
                "elapsed_sec": time.time() - t0,
            })
            logger.info(f"    {probe_name:<6}  slope={slope:.3f}  mantel={mr:.3f}  "
                        f"[{time.time()-t0:.0f}s]")

    df = pd.DataFrame(rows)
    df.to_csv(outdir / "A2_probe_comparison.csv", index=False)

    summary = {"n_models": int(df["model"].nunique()) if len(df) else 0}
    if len(df):
        summary["mean_slope_by_probe"] = (
            df.groupby("probe")["slope"].mean().to_dict()
        )
        summary["mean_mantel_by_probe"] = (
            df.groupby("probe")["mantel_r"].mean().to_dict()
        )
        # Per-model probe spread
        spread = df.groupby("model")["slope"].agg(["min", "max"])
        spread["spread"] = spread["max"] - spread["min"]
        summary["mean_per_model_probe_spread"] = float(spread["spread"].mean())
        spread.to_csv(outdir / "A2_per_model_spread.csv")

    (outdir / "A2_summary.json").write_text(
        json.dumps(summary, indent=2, default=str))

    if len(df):
        logger.info(f"\n  Mean slope across {summary['n_models']} models by probe:")
        for probe, slope in summary["mean_slope_by_probe"].items():
            logger.info(f"    {probe:<6}: slope={slope:.3f}  "
                        f"mantel={summary['mean_mantel_by_probe'][probe]:.3f}")
        logger.info(f"  Mean per-model spread across probes: "
                    f"{summary['mean_per_model_probe_spread']:.3f}")

    return summary


# ─────────────────────────────────────────────────────────────────────────
#  Experiment: A3  Multi-target / reduced-rank vs independent ridge
# ─────────────────────────────────────────────────────────────────────────

def _cross_persona_pipeline_multi(
    model_name: str,
    layer: int,
    fit_predict_multi: Callable,   # (Xtr, Ytr, Xte) -> Pte (matrix)
    human_items: Dict,
) -> Optional[Dict]:
    """Alternative pipeline that fits one probe for all items jointly
    (multi-target). Aggregates across all rotations with target scale varying.

    This is a simplified version of the per-rotation pipeline: we stack all
    (rotation, target_scale, item) observations, and fit one multi-target
    model on the full item set. We then compute CV predictions per item,
    aggregate by subscale, and compute the 16×16 correlation matrix.
    """
    from sklearn.model_selection import KFold
    from psychometric_inference.scoring import SCORING_RULES

    SCALES = [
        ("IRI", "IRI"), ("PANAS", "PANAS"), ("POM", "POM"),
        ("big_five", "BigFive"), ("in_inter_dependent", "SelfConst"),
        ("Life_Satisfaction", "LifeSat"), ("Loneliness", "Lonely"),
    ]
    cp_dir = MECHANISTIC_ROOT / f"results_{model_name}" / "cross_persona"
    if not cp_dir.exists() and model_name == "llama8b_instruct":
        cp_dir = MECHANISTIC_DEFAULT_RESULTS_DIR / "cross_persona"
    if not cp_dir.exists():
        return None

    subject_ids = sorted(human_items.keys())

    # Build a (subject, item_id) matrix of responses and a per-(subject,item)
    # activation, where item_id = (persona_short, scale_file, item_num).
    # For multi-target we need a single activation per subject — average
    # activations across rotations for each subject (crude).
    hdim = None
    subj_act_sum: Dict[str, np.ndarray] = {}
    subj_act_n:   Dict[str, int] = {}

    for persona_file, persona_short in SCALES:
        rd = cp_dir / f"rotation_{persona_short}"
        apath = rd / "activations.npz"; mpath = rd / "meta.csv"
        if not (apath.exists() and mpath.exists()):
            continue
        try:
            with np.load(apath) as npz:
                if f"L{layer}" not in npz.files:
                    continue
                acts = npz[f"L{layer}"]
        except Exception:
            continue
        meta = pd.read_csv(mpath)
        meta["subject_id"] = meta["subject_id"].astype(str)
        if hdim is None:
            hdim = acts.shape[1]
        for _, row in meta.iterrows():
            sid = str(row["subject_id"])
            idx = row.name
            if idx >= len(acts):
                continue
            if sid not in subj_act_sum:
                subj_act_sum[sid] = acts[idx].astype(np.float64).copy()
                subj_act_n[sid] = 1
            else:
                subj_act_sum[sid] += acts[idx]
                subj_act_n[sid] += 1

    if hdim is None or not subj_act_sum:
        return None

    subjects_with_act = [sid for sid in subject_ids if sid in subj_act_sum]
    X_sub = np.array([subj_act_sum[sid] / subj_act_n[sid]
                      for sid in subjects_with_act]).astype(np.float32)

    # Build Y: for each (scale_file, item_num) available in SCORING_RULES,
    # fetch this subject's response; columns are items, in a canonical order.
    item_columns = []  # list of (scale_file, scale_short, item_num, subscale_full)
    for scale_file, scale_short in SCALES:
        rules = SCORING_RULES.get(scale_short, {})
        for sub_name, item_nums in rules.get("subscales", {}).items():
            sub_full = f"{scale_short}_{sub_name}"
            for item_num in item_nums:
                item_columns.append((scale_file, scale_short, int(item_num), sub_full))

    Y = np.full((len(subjects_with_act), len(item_columns)), np.nan)
    for ri, sid in enumerate(subjects_with_act):
        scales_for_sid = human_items.get(sid, {})
        for ci, (sf, ss, inum, _) in enumerate(item_columns):
            if sf in scales_for_sid and inum in scales_for_sid[sf]:
                Y[ri, ci] = scales_for_sid[sf][inum]

    # Only keep subjects with enough valid items
    valid_subj_mask = (~np.isnan(Y)).sum(axis=1) >= 20
    X_sub = X_sub[valid_subj_mask]
    Y = Y[valid_subj_mask]
    kept_sids = [sid for i, sid in enumerate(subjects_with_act) if valid_subj_mask[i]]
    # Impute column means for missing responses
    col_means = np.nanmean(Y, axis=0)
    col_means = np.where(np.isnan(col_means), 3.0, col_means)  # fallback
    ii, jj = np.where(np.isnan(Y))
    Y[ii, jj] = col_means[jj]

    if X_sub.shape[0] < 30:
        return None

    # CV predict
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    pred = np.zeros_like(Y)
    for tr, te in kf.split(X_sub):
        try:
            pred[te] = fit_predict_multi(X_sub[tr], Y[tr], X_sub[te])
        except Exception:
            pred[te] = Y[tr].mean(axis=0)[None, :]

    # Aggregate predictions per (subject, subscale)
    from psychometric_inference.scoring import SCORING_RULES as SR
    subscale_cont: Dict[str, Dict[str, float]] = {}
    for scale_file, scale_short in SCALES:
        rules = SR.get(scale_short, {})
        rev = set(rules.get("reverse_items", []))
        max_val = rules.get("max_val", 5)
        for sub_name, item_nums in rules.get("subscales", {}).items():
            sub_full = f"{scale_short}_{sub_name}"
            ci_list = [ci for ci, (sf, ss, inum, _) in enumerate(item_columns)
                       if sf == scale_file and inum in item_nums]
            if not ci_list:
                continue
            # reverse-coded continuous avg
            for ri, sid in enumerate(kept_sids):
                item_preds = []
                for ci in ci_list:
                    inum = item_columns[ci][2]
                    pv = float(pred[ri, ci])
                    if inum in rev:
                        pv = (max_val + 1) - pv
                    item_preds.append(pv)
                if item_preds:
                    subscale_cont.setdefault(sub_full, {})[sid] = float(np.mean(item_preds))

    available = [s for s in ALL_SUBSCALES if s in subscale_cont]
    if len(available) < 3:
        return None
    pred_df = pd.DataFrame({s: [subscale_cont[s].get(sid, np.nan) for sid in kept_sids]
                             for s in available}, index=kept_sids)
    # corr matrix
    n = len(available)
    M = np.full((n, n), np.nan)
    for i, si in enumerate(available):
        for j, sj in enumerate(available):
            vi = pred_df[si].values; vj = pred_df[sj].values
            v = ~(np.isnan(vi) | np.isnan(vj))
            if v.sum() > 10:
                M[i, j] = float(np.corrcoef(vi[v], vj[v])[0, 1])
    return {
        "pred_corr_continuous": pd.DataFrame(M, index=available, columns=available),
    }


def run_A3(outdir: Path) -> Dict:
    """Independent ridge vs PLS-2 vs reduced-rank ridge (multi-target).

    Methodological note: this experiment uses a SIMPLIFIED pipeline that
    averages per-subject activations across rotations, then fits a
    multi-target model on all items simultaneously. This differs from the
    paper's per-rotation / per-item pipeline but provides a head-to-head
    comparison between independent vs information-sharing probes on the
    same underlying data.
    """
    from sklearn.linear_model import RidgeCV
    from sklearn.cross_decomposition import PLSRegression

    human = load_human_subscale_corr()
    human_items = _cross_persona_load_human_items()
    if not human_items:
        return {"error": "could not load human items"}

    def indep_ridge(Xtr, Ytr, Xte):
        P = np.zeros((Xte.shape[0], Ytr.shape[1]))
        for j in range(Ytr.shape[1]):
            r = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0, 1000.0], cv=5)
            r.fit(Xtr, Ytr[:, j])
            P[:, j] = r.predict(Xte)
        return P

    def pls2(Xtr, Ytr, Xte):
        pls = PLSRegression(n_components=2)
        pls.fit(Xtr, Ytr)
        return pls.predict(Xte)

    def reduced_rank(Xtr, Ytr, Xte, rank=5):
        B = np.zeros((Xtr.shape[1], Ytr.shape[1]))
        for j in range(Ytr.shape[1]):
            r = RidgeCV(alphas=[1.0, 10.0, 100.0], cv=3)
            r.fit(Xtr, Ytr[:, j])
            B[:, j] = r.coef_
        U, S, Vt = np.linalg.svd(B, full_matrices=False)
        k = min(rank, len(S))
        B_rr = U[:, :k] @ np.diag(S[:k]) @ Vt[:k]
        intercept = Ytr.mean(axis=0) - Xtr.mean(axis=0) @ B_rr
        return Xte @ B_rr + intercept

    rows = []
    for model_name, layer in ITEMLEVEL_LAYERS.items():
        logger.info(f"  [{model_name} L{layer}] running 3 probes …")
        t0 = time.time()
        row = {
            "model": model_name, "size": MODEL_BY_NAME[model_name]["size"],
            "mtype": MODEL_BY_NAME[model_name]["mtype"],
            "family": MODEL_BY_NAME[model_name]["family"], "layer": layer,
        }
        for probe_name, fn in [("indep_ridge", indep_ridge),
                                ("pls_2", pls2),
                                ("reduced_rank_5", lambda a,b,c: reduced_rank(a,b,c,rank=5))]:
            res = _cross_persona_pipeline_multi(model_name, layer, fn, human_items)
            if res is None:
                row[f"slope_{probe_name}"] = np.nan
                row[f"mantel_{probe_name}"] = np.nan
                continue
            slope, mr, _ = _slope_mantel_vs_human(res["pred_corr_continuous"], human)
            row[f"slope_{probe_name}"] = slope
            row[f"mantel_{probe_name}"] = mr
        row["elapsed_sec"] = time.time() - t0
        rows.append(row)
        logger.info(f"    indep={row.get('slope_indep_ridge', np.nan):.3f}  "
                    f"pls2={row.get('slope_pls_2', np.nan):.3f}  "
                    f"rr5={row.get('slope_reduced_rank_5', np.nan):.3f}  "
                    f"[{time.time()-t0:.0f}s]")

    df = pd.DataFrame(rows)
    df.to_csv(outdir / "A3_multitarget_comparison.csv", index=False)

    summary = {"n_models": int(len(df))}
    if len(df):
        summary["mean_slopes"] = {
            "indep_ridge":     float(df["slope_indep_ridge"].mean()),
            "pls_2":           float(df["slope_pls_2"].mean()),
            "reduced_rank_5":  float(df["slope_reduced_rank_5"].mean()),
        }
    (outdir / "A3_summary.json").write_text(json.dumps(summary, indent=2, default=str))

    if len(df):
        logger.info(f"\n  Mean slopes across {len(df)} models:")
        for k, v in summary["mean_slopes"].items():
            logger.info(f"    {k:<18}: {v:.3f}")

    return summary


# ─────────────────────────────────────────────────────────────────────────
#  Experiment: F2  Additional base-instruct pair comparisons
# ─────────────────────────────────────────────────────────────────────────

def run_F2(outdir: Path) -> Dict:
    """Base-instruct pair comparisons across all 7 pairs (vs paper's 2).

    Paper §6.2 compares Qwen 7B base/instruct and Llama 8B base/instruct
    (2 pairs). We use all pairs where both base and instruct exist.

    For each pair, reports the difference in:
      - repr_mantel_r     (structure alignment)
      - repr_corrected_slope (amplification)
      - repr_reliability  (signal strength)
      - beh_subscale_slope (behavioral alignment)
    across pairs, then tests whether the structure gap is consistent.
    """
    svb = load_structure_vs_behavior()

    # Find base/instruct pairs using structure_vs_behavior.csv
    pairs = {}
    for _, row in svb.iterrows():
        key = (row["family"], row["size"])
        pairs.setdefault(key, {})[row["mtype"]] = row

    pair_rows = []
    for key, entry in pairs.items():
        if "base" in entry and "instruct" in entry:
            b = entry["base"]; i = entry["instruct"]
            pair_rows.append({
                "family": key[0],
                "size": key[1],
                "model_base": b["model"],
                "model_instruct": i["model"],
                "mantel_base": b["repr_mantel_r"],
                "mantel_instruct": i["repr_mantel_r"],
                "slope_base": b["repr_corrected_slope"],
                "slope_instruct": i["repr_corrected_slope"],
                "reliability_base": b["repr_reliability"],
                "reliability_instruct": i["repr_reliability"],
                "beh_slope_base": b["beh_subscale_slope"],
                "beh_slope_instruct": i["beh_subscale_slope"],
                "delta_mantel": i["repr_mantel_r"] - b["repr_mantel_r"],
                "delta_slope": i["repr_corrected_slope"] - b["repr_corrected_slope"],
                "delta_reliability": i["repr_reliability"] - b["repr_reliability"],
                "delta_beh_slope": i["beh_subscale_slope"] - b["beh_subscale_slope"],
            })

    df = pd.DataFrame(pair_rows).sort_values("size")
    df.to_csv(outdir / "F2_base_instruct_pairs.csv", index=False)

    from scipy.stats import ttest_rel, wilcoxon
    tests = {}
    if len(df) >= 3:
        for metric, cols in [
            ("mantel", ("mantel_instruct", "mantel_base")),
            ("slope", ("slope_instruct", "slope_base")),
            ("reliability", ("reliability_instruct", "reliability_base")),
            ("beh_slope", ("beh_slope_instruct", "beh_slope_base")),
        ]:
            try:
                a = df[cols[0]].dropna()
                b = df[cols[1]].dropna()
                if len(a) < 3 or len(b) < 3 or len(a) != len(b):
                    tests[metric] = None
                    continue
                t, p_t = ttest_rel(a, b)
                w_stat, p_w = wilcoxon(a, b)
                tests[metric] = {
                    "t_p": float(p_t),
                    "wilcoxon_p": float(p_w),
                    "mean_diff": float((a - b).mean()),
                }
            except Exception as e:
                tests[metric] = {"error": str(e)}

    summary = {
        "n_pairs": int(len(df)),
        "mean_delta_mantel": float(df["delta_mantel"].mean()) if len(df) else None,
        "mean_delta_slope": float(df["delta_slope"].mean()) if len(df) else None,
        "mean_delta_reliability": float(df["delta_reliability"].mean()) if len(df) else None,
        "mean_delta_beh_slope": float(df["delta_beh_slope"].mean()) if len(df) else None,
        "paired_tests": tests,
    }
    (outdir / "F2_summary.json").write_text(json.dumps(summary, indent=2, default=str))

    logger.info(f"  {len(df)} base-instruct pairs found:")
    logger.info(f"  {'family':<7} {'size':>6} {'Δmantel':>9} {'Δslope':>9} "
                f"{'Δreliab':>9} {'Δbeh_slope':>11}")
    for _, r in df.iterrows():
        logger.info(f"  {r['family']:<7} {r['size']:>5.1f}B  "
                    f"{r['delta_mantel']:>+9.3f} "
                    f"{r['delta_slope']:>+9.3f} "
                    f"{r['delta_reliability']:>+9.3f} "
                    f"{r['delta_beh_slope']:>+11.3f}")
    if tests:
        logger.info(f"\n  Paired tests (instruct vs base):")
        for metric, res in tests.items():
            if res and "t_p" in res:
                logger.info(f"    Δ{metric:<12} mean = {res['mean_diff']:+.3f}  "
                            f"t p = {res['t_p']:.4f}  wilcoxon p = {res['wilcoxon_p']:.4f}")

    return summary


# ═════════════════════════════════════════════════════════════════════════
#                    Registry update and dispatcher
# ═════════════════════════════════════════════════════════════════════════

@dataclass
class Experiment:
    exp_id: str
    tier: str              # T0 / T1 / T2
    needs_gpu: bool
    func: Callable[[Path], Dict]
    desc: str
    estimated_minutes: float

EXPERIMENTS: List[Experiment] = [
    Experiment("C1", "T0", False, run_C1,
               "Within-group correlation (instruct vs base)", 0.5),
    Experiment("C2", "T0", False, run_C2,
               "Partial correlation controlling for log(size)", 0.5),
    Experiment("F1", "T0", False, run_F1,
               "Fixed 60%-depth sensitivity vs post-hoc best layer", 1),
    Experiment("F3", "T0", False, run_F3,
               "Variance decomposition by pair type", 2),
    Experiment("D2", "T0", False, run_D2,
               "Cultural stratification (stable vs specific subscales)", 2),
    Experiment("F1b", "T0", False, run_F1b,
               "Adjacent-layer (best±1,±2) stability (paper App B5 claim)", 1),
    Experiment("F3b", "T0", False, run_F3b,
               "Within-scale vs cross-scale trivial-context test", 2),
    Experiment("D2b", "T0", False, run_D2b,
               "D2 robustness diagnostics (human r, LOO, Mantel, cross-scale-only)", 2),
    # T1: synthetic controls
    Experiment("A1_minimal", "T1", False, run_A1_minimal,
               "Synthetic item-level ridge inflation check", 5),
    Experiment("A1_full", "T1", False, run_A1_full,
               "Synthetic high-dim SNR sweep (4 hdims × 5 SNR × 20 seeds)", 60),
    Experiment("A1_adj", "T2", False, run_A1_adj,
               "Pipeline-baseline-adjusted slope per real model (needs A1_full)", 1),
    Experiment("B2", "T2", False, run_B2,
               "Item-level discretization-loss decomposition (cont/rounded/argmax)", 30),
    Experiment("A2", "T2", False, run_A2,
               "Alternative probes (Lasso / PLS / Linear SVR) vs Ridge", 40),
    Experiment("A3", "T2", False, run_A3,
               "Multi-target ridge / PLS-2 / reduced-rank vs independent", 20),
    Experiment("F2", "T2", False, run_F2,
               "Base-instruct pair comparisons across all pairs", 1),
]
EXP_BY_ID = {e.exp_id: e for e in EXPERIMENTS}


def run_one(exp: Experiment, force: bool = False) -> bool:
    if is_done(exp.exp_id) and not force:
        logger.info(f"[{exp.exp_id}] SKIP (already done). "
                    f"Rerun with --force {exp.exp_id} to redo.")
        return True
    banner(f"[{exp.exp_id}]  {exp.desc}  (tier={exp.tier}, "
           f"~{exp.estimated_minutes:.0f} min{'  GPU' if exp.needs_gpu else ''})")
    outdir = exp_outdir(exp.exp_id)
    t0 = time.time()
    try:
        result = exp.func(outdir)
    except Exception as e:
        elapsed = time.time() - t0
        logger.error(f"[{exp.exp_id}] FAILED after {elapsed:.1f}s: {e}")
        logger.error(traceback.format_exc())
        return False
    elapsed = time.time() - t0
    mark_done(exp.exp_id, {"elapsed_sec": elapsed})
    banner(f"[{exp.exp_id}]  DONE in {elapsed:.1f}s  →  {outdir}", char="-")
    return True


def list_experiments():
    print(f"{'ID':<6} {'Tier':<4} {'GPU':<4} {'~min':>6}  Description")
    print("-" * 72)
    for e in EXPERIMENTS:
        done_mark = "✓" if is_done(e.exp_id) else " "
        print(f"{done_mark} {e.exp_id:<4} {e.tier:<4} "
              f"{'Y' if e.needs_gpu else 'N':<4} "
              f"{e.estimated_minutes:>5.1f}  {e.desc}")


def select_experiments(only: Optional[List[str]],
                       skip: Optional[List[str]],
                       skip_gpu: bool) -> List[Experiment]:
    selected = [e for e in EXPERIMENTS]
    if only:
        bad = [x for x in only if x not in EXP_BY_ID]
        if bad:
            raise ValueError(f"Unknown experiment IDs: {bad}. "
                             f"Available: {list(EXP_BY_ID.keys())}")
        selected = [e for e in selected if e.exp_id in only]
    if skip:
        selected = [e for e in selected if e.exp_id not in skip]
    if skip_gpu:
        selected = [e for e in selected if not e.needs_gpu]
    return selected


def dry_run(selected: List[Experiment], force: Optional[List[str]]):
    total = 0.0
    print(f"{'ID':<6} {'Tier':<4} {'Status':<10} {'~min':>6}  Description")
    print("-" * 72)
    for e in selected:
        done = is_done(e.exp_id) and not (force and e.exp_id in force)
        status = "SKIP" if done else "RUN"
        if not done:
            total += e.estimated_minutes
        print(f"{e.exp_id:<6} {e.tier:<4} {status:<10} "
              f"{e.estimated_minutes:>5.1f}  {e.desc}")
    print("-" * 72)
    print(f"Total estimated time for RUN: ~{total:.1f} min")


# ─────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    ap.add_argument("--only", nargs="+", default=None,
                    help="Run only these experiment IDs (e.g., --only C1 C2)")
    ap.add_argument("--skip", nargs="+", default=None,
                    help="Skip these experiments")
    ap.add_argument("--force", nargs="*", default=None,
                    help="Force rerun these experiments (e.g., --force A1_full). "
                         "With no argument, forces all that would run.")
    ap.add_argument("--skip-gpu", action="store_true",
                    help="Skip experiments flagged needs_gpu (for CPU-only runs)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Show what would run, don't run.")
    ap.add_argument("--list", action="store_true",
                    help="List all experiments with their done status.")
    args = ap.parse_args()

    setup_logger()

    if args.list:
        list_experiments()
        return

    selected = select_experiments(args.only, args.skip, args.skip_gpu)

    force_set = set()
    if args.force is not None:
        force_set = set(args.force) if args.force else {e.exp_id for e in selected}

    if args.dry_run:
        dry_run(selected, force_set)
        return

    banner(f"compute_robustness_suite.py — running {len(selected)} experiments",
           char="#", width=72)
    logger.info(f"Output root: {OUTPUT_ROOT}")
    logger.info(f"Log file:    {LOG_FILE}")
    logger.info(f"Experiments: {[e.exp_id for e in selected]}")
    if force_set:
        logger.info(f"Force rerun: {sorted(force_set)}")

    # Clear done flags for forced ones
    for exp_id in force_set:
        clear_done(exp_id)

    # Run
    n_ok = 0
    n_fail = 0
    for e in selected:
        ok = run_one(e, force=(e.exp_id in force_set))
        if ok:
            n_ok += 1
        else:
            n_fail += 1

    banner(f"Summary: {n_ok} OK, {n_fail} FAILED, "
           f"log: {LOG_FILE}", char="#")


if __name__ == "__main__":
    main()
