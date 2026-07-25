#!/usr/bin/env python3
"""Supplementary behavioral baselines for the paper artifact.

Section A: Human split-half baseline
  - Split subjects into two halves, compute correlation matrices for each,
    then compute alignment r, slope, intercept between the two halves.
  - Repeat bootstrap splits to get a human-human alignment distribution.
  - Compare LLM-human values against this distribution.

Section B: Intercept analysis
  - For each model's existing LLM matrix, compute OLS intercept alongside slope.
  - Report item/subscale/scale levels and within/between decompositions.

Usage:
    python compute_supplementary_behavior.py
    python compute_supplementary_behavior.py --skip_b                # only Section A
    python compute_supplementary_behavior.py --skip_a                # only Section B
    python compute_supplementary_behavior.py --n_bootstrap 100       # quick split-half smoke test
"""

import argparse
import os
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR / "src"))

from psychometric_inference.model_registry import analysis_model_tuples
from psychometric_inference.scoring import (
    SCORING_RULES, FILENAME_TO_SCALE, ALL_SUBSCALES,
    compute_subscale_scores, un_reverse_df,
)

# ── Shared constants ──

SCALES = [
    ("IRI", "IRI"),
    ("PANAS", "PANAS"),
    ("POM", "POM"),
    ("big_five", "BigFive"),
    ("in_inter_dependent", "SelfConst"),
    ("Life_Satisfaction", "LifeSat"),
    ("Loneliness", "Lonely"),
]

SCALE_NAMES = ["IRI", "PANAS", "POM", "BigFive", "SelfConst", "LifeSat", "Lonely"]

HUMAN_DIRS_DEFAULT = [
    "data/human/SED",
    "data/human/SEDC",
    "data/human/SEDD",
]

MODELS = analysis_model_tuples()

# ═══════════════════════════════════════════════════════════════════════════════
#  Shared utilities
# ═══════════════════════════════════════════════════════════════════════════════

def _detect_id_col(df):
    for c in ["Subject_ID", "ID", "Scan_ID", "id"]:
        if c in df.columns:
            return c
    return None


def load_human_data(human_dirs):
    """Load all human item-level data, un-reversed, with Subject_ID."""
    scale_dfs = {}
    item_labels = []
    for scale_file, scale_short in SCALES:
        frames = []
        labs = []
        for d in human_dirs:
            fpath = os.path.join(d, f"{scale_file}.csv")
            if not os.path.exists(fpath):
                continue
            df = pd.read_csv(fpath)
            id_col = _detect_id_col(df)
            q_cols = [c for c in df.columns if c.startswith("Q")]
            sub = df[q_cols].copy()
            sub = un_reverse_df(sub, scale_file)
            rename = {c: f"{scale_short}_{c}" for c in q_cols}
            sub.rename(columns=rename, inplace=True)
            if id_col and id_col in df.columns:
                sub.insert(0, "Subject_ID", df[id_col].astype(str))
            else:
                sub.insert(0, "Subject_ID", [f"S{i:04d}" for i in range(len(sub))])
            frames.append(sub)
            labs = [(scale_short, c) for c in rename.values()]
        if not frames:
            continue
        combined = pd.concat(frames, ignore_index=True)
        scale_dfs[scale_short] = combined
        item_labels.extend(labs)

    merged = None
    for df in scale_dfs.values():
        merged = df if merged is None else merged.merge(df, on="Subject_ID", how="inner")
    item_labels = [(s, c) for s, c in item_labels if c in merged.columns]
    return merged, item_labels


def compute_three_level_matrices(data, item_labels):
    """Compute item, subscale, scale correlation matrices from subject × item data."""
    numeric = data.drop(columns=["Subject_ID"], errors="ignore")
    item_cols = [c for _, c in item_labels]
    numeric = numeric[item_cols]

    # Item-level
    item_mat = numeric.corr()

    # Subscale-level
    sub_scores = compute_subscale_scores(numeric, item_labels)
    common_subs = [s for s in ALL_SUBSCALES if s in sub_scores.columns]
    sub_mat = sub_scores[common_subs].corr()

    # Scale-level
    scale_mat = _aggregate_to_scale(sub_mat)

    return item_mat, sub_mat, scale_mat


def _aggregate_to_scale(subscale_matrix):
    """Aggregate subscale matrix to scale-level."""
    sub_to_scale = {}
    for scale_short, rules in SCORING_RULES.items():
        for sub_name in rules["subscales"]:
            sub_to_scale[f"{scale_short}_{sub_name}"] = scale_short

    n = len(SCALE_NAMES)
    scale_matrix = np.full((n, n), np.nan)
    for i, s_i in enumerate(SCALE_NAMES):
        subs_i = [s for s in ALL_SUBSCALES if sub_to_scale.get(s) == s_i]
        for j, s_j in enumerate(SCALE_NAMES):
            subs_j = [s for s in ALL_SUBSCALES if sub_to_scale.get(s) == s_j]
            vals = []
            for si in subs_i:
                for sj in subs_j:
                    if si in subscale_matrix.index and sj in subscale_matrix.columns:
                        v = subscale_matrix.loc[si, sj]
                        if not np.isnan(v):
                            vals.append(v)
            if vals:
                scale_matrix[i, j] = np.mean(vals)
    return pd.DataFrame(scale_matrix, index=SCALE_NAMES, columns=SCALE_NAMES)


def compare_upper_tri(mat_a, mat_b):
    """Compare two matrices: alignment r, slope, intercept, RMSE on upper triangle."""
    common = [c for c in mat_a.columns if c in mat_b.columns]
    A = mat_a.loc[common, common].values
    B = mat_b.loc[common, common].values
    n = len(common)
    mask = np.triu_indices(n, k=1)
    a_vec = A[mask]
    b_vec = B[mask]
    valid = ~(np.isnan(a_vec) | np.isnan(b_vec))
    a_v, b_v = a_vec[valid], b_vec[valid]

    if len(a_v) < 3:
        return {"r": np.nan, "slope": np.nan, "intercept": np.nan,
                "rmse": np.nan, "n_pairs": 0}

    r, _ = pearsonr(a_v, b_v)
    coeffs = np.polyfit(a_v, b_v, 1)
    slope, intercept = coeffs[0], coeffs[1]
    rmse = np.sqrt(np.mean((a_v - b_v) ** 2))

    return {"r": r, "slope": slope, "intercept": intercept,
            "rmse": rmse, "n_pairs": int(valid.sum())}


def _get_scale_for_col(col, item_labels=None):
    if item_labels:
        for s, c in item_labels:
            if c == col:
                return s
    for scale in SCALE_NAMES:
        if col.startswith(scale + "_"):
            return scale
    return col


def compare_within_between(mat_a, mat_b, item_labels=None):
    """Compare within-scale and between-scale pairs with slope + intercept."""
    common = [c for c in mat_a.columns if c in mat_b.columns]
    A = mat_a.loc[common, common].values
    B = mat_b.loc[common, common].values
    n = len(common)
    mask_r, mask_c = np.triu_indices(n, k=1)

    col_scales = [_get_scale_for_col(c, item_labels) for c in common]
    within = np.array([col_scales[r] == col_scales[c] for r, c in zip(mask_r, mask_c)])
    between = ~within

    results = {}
    for name, pair_mask in [("within", within), ("between", between)]:
        a_vec = A[(mask_r, mask_c)][pair_mask]
        b_vec = B[(mask_r, mask_c)][pair_mask]
        valid = ~(np.isnan(a_vec) | np.isnan(b_vec))
        a_v, b_v = a_vec[valid], b_vec[valid]

        if len(a_v) < 3:
            results[name] = {"r": np.nan, "slope": np.nan, "intercept": np.nan,
                             "rmse": np.nan, "n_pairs": 0}
            continue

        r, _ = pearsonr(a_v, b_v)
        coeffs = np.polyfit(a_v, b_v, 1)
        rmse = np.sqrt(np.mean((a_v - b_v) ** 2))
        results[name] = {"r": r, "slope": coeffs[0], "intercept": coeffs[1],
                         "rmse": rmse, "n_pairs": int(valid.sum())}

    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  Section A: Human Split-Half Baseline
# ═══════════════════════════════════════════════════════════════════════════════

def run_split_half_bootstrap(human_data, item_labels, n_boot, rng, results_dir, output_dir):
    """Bootstrap split-half: split subjects into two halves, compute matrices,
    measure alignment between halves. Repeat n_boot times."""

    print(f"\n{'='*70}")
    print(f"  SECTION A: Human Split-Half Baseline ({n_boot} iterations)")
    print(f"{'='*70}")

    subject_ids = human_data["Subject_ID"].values
    n_subj = len(subject_ids)
    half = n_subj // 2

    boot_rows = []

    for i in range(n_boot):
        if (i + 1) % 100 == 0:
            print(f"  Iteration {i+1}/{n_boot}")

        # Random split
        perm = rng.permutation(n_subj)
        idx_a, idx_b = perm[:half], perm[half:2*half]
        data_a = human_data.iloc[idx_a].reset_index(drop=True)
        data_b = human_data.iloc[idx_b].reset_index(drop=True)

        # Compute matrices for each half
        item_a, sub_a, scale_a = compute_three_level_matrices(data_a, item_labels)
        item_b, sub_b, scale_b = compute_three_level_matrices(data_b, item_labels)

        row = {"iteration": i}

        # Compare at each level
        for level, ma, mb in [("item", item_a, item_b),
                               ("subscale", sub_a, sub_b),
                               ("scale", scale_a, scale_b)]:
            metrics = compare_upper_tri(ma, mb)
            row[f"{level}_r"] = metrics["r"]
            row[f"{level}_slope"] = metrics["slope"]
            row[f"{level}_intercept"] = metrics["intercept"]
            row[f"{level}_rmse"] = metrics["rmse"]

        # Within/between for item and subscale
        for level, ma, mb, labels in [
            ("item", item_a, item_b, item_labels),
            ("subscale", sub_a, sub_b, None),
        ]:
            wb = compare_within_between(ma, mb, item_labels=labels)
            for part in ["within", "between"]:
                for metric_name in ["r", "slope", "intercept"]:
                    row[f"{level}_{part}_{metric_name}"] = wb[part][metric_name]

        boot_rows.append(row)

    boot_df = pd.DataFrame(boot_rows)
    boot_df.to_csv(output_dir / "human_split_half_bootstrap.csv", index=False)
    print(f"  Saved: human_split_half_bootstrap.csv")

    # ── Compare LLM models against baseline ──
    print(f"\n  Comparing LLM models against human split-half baseline...")

    llm_rows = []
    for dirname, size, mtype, family in MODELS:
        model_results_dir = Path(results_dir) / dirname
        if not model_results_dir.exists():
            continue

        row = {"model": dirname, "size": size, "type": mtype, "family": family}

        for level in ["item", "subscale", "scale"]:
            h_path = model_results_dir / f"{level}_human.csv"
            l_path = model_results_dir / f"{level}_llm_implicit.csv"
            if not h_path.exists() or not l_path.exists():
                continue
            h_mat = pd.read_csv(h_path, index_col=0)
            l_mat = pd.read_csv(l_path, index_col=0)

            metrics = compare_upper_tri(h_mat, l_mat)
            row[f"{level}_r"] = metrics["r"]
            row[f"{level}_slope"] = metrics["slope"]
            row[f"{level}_intercept"] = metrics["intercept"]
            row[f"{level}_rmse"] = metrics["rmse"]

            # Percentile rank against bootstrap distribution
            for metric_name in ["r", "slope", "intercept"]:
                col = f"{level}_{metric_name}"
                if col in boot_df.columns and col in row:
                    val = row[col]
                    if not np.isnan(val):
                        pctile = np.mean(boot_df[col].dropna() <= val) * 100
                        row[f"{col}_pctile"] = pctile

        llm_rows.append(row)

    llm_df = pd.DataFrame(llm_rows)
    llm_df.to_csv(output_dir / "llm_vs_human_baseline.csv", index=False)
    print(f"  Saved: llm_vs_human_baseline.csv")

    # ── Summary ──
    print(f"\n  Human split-half baseline (median [2.5th, 97.5th]):")
    for level in ["item", "subscale", "scale"]:
        for metric in ["r", "slope", "intercept"]:
            col = f"{level}_{metric}"
            if col in boot_df.columns:
                vals = boot_df[col].dropna()
                med = vals.median()
                lo, hi = vals.quantile(0.025), vals.quantile(0.975)
                print(f"    {level:<10} {metric:<10}: {med:.4f} [{lo:.4f}, {hi:.4f}]")

    # ── Plot ──
    _plot_split_half(boot_df, llm_df, output_dir)

    return boot_df, llm_df


def _plot_split_half(boot_df, llm_df, output_dir):
    """Plot bootstrap distributions with LLM values overlaid."""
    levels = ["item", "subscale", "scale"]
    metrics = ["r", "slope", "intercept"]

    fig, axes = plt.subplots(len(metrics), len(levels), figsize=(14, 10))

    for j, level in enumerate(levels):
        for i, metric in enumerate(metrics):
            ax = axes[i, j]
            col = f"{level}_{metric}"
            if col not in boot_df.columns:
                ax.set_visible(False)
                continue

            vals = boot_df[col].dropna()
            ax.hist(vals, bins=50, alpha=0.6, color="#2c7bb6", density=True,
                    label="Human split-half")

            # Overlay LLM values
            for _, row in llm_df.iterrows():
                if col in row and not np.isnan(row[col]):
                    color = "#d7191c" if row.get("type") == "instruct" else "#1a9641"
                    ax.axvline(row[col], color=color, alpha=0.3, lw=0.8)

            # Mark the largest model of each family
            for family_name, color in [("Qwen", "#1a9641"), ("Llama", "#d7191c")]:
                sub = llm_df[(llm_df["family"] == family_name)]
                if not sub.empty:
                    largest = sub.loc[sub["size"].idxmax()]
                    if col in largest and not np.isnan(largest[col]):
                        ax.axvline(largest[col], color=color, lw=2, ls="--",
                                   label=f"{largest['model']}")

            if i == 0:
                ax.set_title(f"{level.capitalize()}", fontsize=11)
            if j == 0:
                ax.set_ylabel(metric, fontsize=10)
            if i == 0 and j == 2:
                ax.legend(fontsize=6, loc="upper left")

    plt.suptitle("Human Split-Half Baseline vs LLM Alignment", fontsize=13)
    plt.tight_layout()
    fpath = output_dir / "split_half_distributions.png"
    plt.savefig(fpath, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: split_half_distributions.png")


# ═══════════════════════════════════════════════════════════════════════════════
#  Section B: Intercept Analysis
# ═══════════════════════════════════════════════════════════════════════════════

def run_intercept_analysis(results_dir, output_dir):
    """For each model, compute slope + intercept at all levels."""

    print(f"\n{'='*70}")
    print(f"  SECTION B: Intercept Analysis")
    print(f"{'='*70}")

    rows = []
    for dirname, size, mtype, family in MODELS:
        model_dir = Path(results_dir) / dirname
        if not model_dir.exists():
            continue

        row = {"model": dirname, "size": size, "type": mtype, "family": family}

        for level in ["item", "subscale", "scale"]:
            h_path = model_dir / f"{level}_human.csv"
            l_path = model_dir / f"{level}_llm_implicit.csv"
            if not h_path.exists() or not l_path.exists():
                continue

            h_mat = pd.read_csv(h_path, index_col=0)
            l_mat = pd.read_csv(l_path, index_col=0)

            # Overall
            metrics = compare_upper_tri(h_mat, l_mat)
            row[f"{level}_r"] = metrics["r"]
            row[f"{level}_slope"] = metrics["slope"]
            row[f"{level}_intercept"] = metrics["intercept"]
            row[f"{level}_rmse"] = metrics["rmse"]

            # Within/between (not for scale level)
            if level in ("item", "subscale"):
                # Determine item_labels for item level
                il = None
                if level == "item":
                    il = [(s, c) for c in h_mat.columns
                          for s in SCALE_NAMES if c.startswith(s + "_")]
                wb = compare_within_between(h_mat, l_mat, item_labels=il)
                for part in ["within", "between"]:
                    for m in ["r", "slope", "intercept", "rmse"]:
                        row[f"{level}_{part}_{m}"] = wb[part][m]

        if len(row) > 4:  # has data beyond model info
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "intercept_results.csv", index=False)
    print(f"  Saved: intercept_results.csv")

    # ── Summary ──
    print(f"\n  {'Model':<24} {'Level':<12} {'Slope':>8} {'Intercept':>10} {'r':>8}")
    print(f"  {'-'*66}")
    for _, row in df.iterrows():
        label = f"{row['size']}B {row['type'][:4]} {row['family']}"
        for level in ["item", "subscale", "scale"]:
            s = row.get(f"{level}_slope", np.nan)
            ic = row.get(f"{level}_intercept", np.nan)
            r = row.get(f"{level}_r", np.nan)
            if not np.isnan(s):
                print(f"  {label:<24} {level:<12} {s:>8.3f} {ic:>10.4f} {r:>8.3f}")
                label = ""  # only print model name once

    # ── Plot intercept scaling ──
    _plot_intercept_scaling(df, output_dir)

    return df


def _plot_intercept_scaling(df, output_dir):
    """Plot intercept vs model size."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    levels = ["item", "subscale", "scale"]
    family_colors = {
        ("Qwen", "base"): "#085041",
        ("Qwen", "instruct"): "#5DCAA5",
        ("Llama", "base"): "#993C1D",
        ("Llama", "instruct"): "#F0997B",
    }

    for ax, level in zip(axes, levels):
        for (family, mtype), color in family_colors.items():
            sub = df[(df["family"] == family) & (df["type"] == mtype)].sort_values("size")
            col = f"{level}_intercept"
            if col not in sub.columns:
                continue
            valid = sub[col].notna()
            if valid.any():
                ls = "-" if mtype == "base" else "--"
                ax.plot(sub.loc[valid, "size"], sub.loc[valid, col],
                        marker="o", ms=5, lw=1.5, ls=ls, color=color,
                        label=f"{family} {mtype}")

        ax.axhline(0, color="gray", ls=":", lw=1, alpha=0.5)
        ax.set_xlabel("Parameters (B)")
        ax.set_xscale("log")
        ax.set_title(f"{level.capitalize()}-level intercept")
        if ax == axes[0]:
            ax.set_ylabel("OLS Intercept")
            ax.legend(fontsize=7)

    plt.suptitle("Intercept vs Model Size", fontsize=13)
    plt.tight_layout()
    fpath = output_dir / "intercept_scaling.png"
    plt.savefig(fpath, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: intercept_scaling.png")


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Supplementary analyses")
    parser.add_argument("--human_dirs", nargs="+", default=None)
    parser.add_argument("--results_dir", default="outputs/behavior")
    parser.add_argument("--output_dir", default="outputs/supplementary")
    parser.add_argument("--n_bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip_a", action="store_true", help="Skip split-half baseline")
    parser.add_argument("--skip_b", action="store_true", help="Skip intercept analysis")
    args = parser.parse_args()

    human_dirs_raw = args.human_dirs or HUMAN_DIRS_DEFAULT
    human_dirs = [str(BASE_DIR / d) if not os.path.isabs(d) else d for d in human_dirs_raw]
    results_dir = args.results_dir if os.path.isabs(args.results_dir) else str(BASE_DIR / args.results_dir)
    output_dir = Path(args.output_dir) if os.path.isabs(args.output_dir) else BASE_DIR / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    print(f"{'='*70}")
    print(f"  Supplementary Analyses")
    print(f"{'='*70}")
    print(f"  Human dirs:   {human_dirs}")
    print(f"  Results dir:  {results_dir}")
    print(f"  Output dir:   {output_dir}")

    # Section A
    if not args.skip_a:
        human_data, item_labels = load_human_data(human_dirs)
        print(f"  Human data loaded: {len(human_data)} subjects × {len(item_labels)} items")
        run_split_half_bootstrap(human_data, item_labels, args.n_bootstrap,
                                 rng, results_dir, output_dir)

    # Section B
    if not args.skip_b:
        run_intercept_analysis(results_dir, output_dir)

    print(f"\n{'='*70}")
    print(f"  All done. Outputs in: {output_dir}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
