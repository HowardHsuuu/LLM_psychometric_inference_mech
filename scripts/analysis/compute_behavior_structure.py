"""
Implicit Psychometric Structure Analysis:
Compare LLM's implicit cross-scale correlation structure with human data
at three granularity levels: item-wise (114x114), subscale-wise (16x16),
and scale-wise (7x7).

For each persona round (Scale A = persona):
  - Take persona item/subscale/scale scores (what we set)
  - Take LLM-generated item/subscale/scale scores (what LLM filled)
  - Correlate across subjects -> LLM's implicit belief about relationships

Then compare with human correlation matrices at each level.

Usage:
    python compute_behavior_structure.py \
        --llm_root data/llm_behavior/qwen14b \
        --output_dir outputs/behavior/qwen14b
"""

import argparse
import os
import warnings
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

import sys

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR / "src"))

from psychometric_inference.scoring import (
    SCORING_RULES,
    FILENAME_TO_SCALE,
    SCALE_TO_FILENAME,
    ALL_SUBSCALES,
    compute_subscale_scores,
    un_reverse_df,
)

warnings.filterwarnings("ignore")

SCALES = [
    ("IRI", "IRI"),
    ("PANAS", "PANAS"),
    ("POM", "POM"),
    ("big_five", "BigFive"),
    ("in_inter_dependent", "SelfConst"),
    ("Life_Satisfaction", "LifeSat"),
    ("Loneliness", "Lonely"),
]

HUMAN_DIRS_DEFAULT = [
    "data/human/SED",
    "data/human/SEDC",
    "data/human/SEDD",
]

SCALE_NAMES = ["IRI", "PANAS", "POM", "BigFive", "SelfConst", "LifeSat", "Lonely"]


# ── Data loading ──

def _detect_id_col(df):
    for c in ["Subject_ID", "ID", "Scan_ID", "id"]:
        if c in df.columns:
            return c
    for c in df.columns:
        if not c.startswith("Q"):
            return c
    return None


def _load_scale_csv(fpath, scale_short, un_reverse=False, scale_file=None):
    df = pd.read_csv(fpath)
    id_col = _detect_id_col(df)
    q_cols = [c for c in df.columns if c.startswith("Q")]
    sub = df[q_cols].copy()
    if un_reverse and scale_file:
        sub = un_reverse_df(sub, scale_file)
    if id_col and id_col in df.columns:
        sub.insert(0, "Subject_ID", df[id_col].astype(str))
    else:
        sub.insert(0, "Subject_ID", [f"S{i:04d}" for i in range(len(sub))])
    rename = {c: f"{scale_short}_{c}" for c in q_cols}
    sub.rename(columns=rename, inplace=True)
    item_labels = [(scale_short, c) for c in rename.values()]
    return sub, item_labels


def load_human_items(human_dirs):
    scale_dfs = {}
    item_labels = []
    for scale_file, scale_short in SCALES:
        frames = []
        labs = []
        for d in human_dirs:
            fpath = os.path.join(d, f"{scale_file}.csv")
            if not os.path.exists(fpath):
                continue
            sub, labels = _load_scale_csv(fpath, scale_short, un_reverse=True, scale_file=scale_file)
            frames.append(sub)
            labs = labels
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


def load_llm_round_items(persona_dir):
    scale_dfs = {}
    item_labels = []
    for scale_file, scale_short in SCALES:
        fpath = os.path.join(persona_dir, f"{scale_file}.csv")
        if not os.path.exists(fpath):
            continue
        sub, labels = _load_scale_csv(fpath, scale_short)
        scale_dfs[scale_short] = sub
        item_labels.extend(labels)

    merged = None
    for df in scale_dfs.values():
        merged = df if merged is None else merged.merge(df, on="Subject_ID", how="inner")
    if merged is None:
        return None, []
    item_labels = [(s, c) for s, c in item_labels if c in merged.columns]
    return merged, item_labels


def load_persona_items(persona_dir, scale_file, scale_short):
    persona_path = os.path.join(persona_dir, f"{scale_file}_persona.csv")
    if not os.path.exists(persona_path):
        persona_path = os.path.join(persona_dir, f"{scale_file}.csv")
    if not os.path.exists(persona_path):
        return None, []
    return _load_scale_csv(persona_path, scale_short)


# ── Core computation ──

def _corr_vectors(vec_a, vec_b):
    valid = ~(np.isnan(vec_a) | np.isnan(vec_b))
    if valid.sum() < 3:
        return np.nan
    r, _ = pearsonr(vec_a[valid], vec_b[valid])
    return r


def compute_implicit_item_matrix(llm_root, all_item_cols, item_labels,
                                  persona_pattern="persona_{scale}"):
    n = len(all_item_cols)
    col_to_idx = {c: i for i, c in enumerate(all_item_cols)}
    corr_sum = np.zeros((n, n))
    corr_count = np.zeros((n, n))

    for scale_file, scale_short in SCALES:
        persona_dir = os.path.join(llm_root, persona_pattern.format(scale=scale_file))
        if not os.path.exists(persona_dir):
            continue

        persona_data, persona_labels = load_persona_items(persona_dir, scale_file, scale_short)
        llm_data, llm_labels = load_llm_round_items(persona_dir)
        if persona_data is None or llm_data is None:
            continue

        merged = persona_data.merge(llm_data, on="Subject_ID", suffixes=("_P", "_L"))
        print(f"  Persona={scale_short}: {len(merged)} subjects (item)")

        persona_cols = [c for _, c in persona_labels]
        llm_cols = [c for _, c in llm_labels]

        for p_col in persona_cols:
            if p_col not in col_to_idx:
                continue
            p_idx = col_to_idx[p_col]
            p_merged_col = f"{p_col}_P" if f"{p_col}_P" in merged.columns else p_col
            p_vec = pd.to_numeric(merged[p_merged_col], errors="coerce").values

            for l_col in llm_cols:
                if l_col not in col_to_idx:
                    continue
                l_idx = col_to_idx[l_col]
                l_merged_col = f"{l_col}_L" if f"{l_col}_L" in merged.columns else l_col
                l_vec = pd.to_numeric(merged[l_merged_col], errors="coerce").values

                r = _corr_vectors(p_vec, l_vec)
                if not np.isnan(r):
                    corr_sum[p_idx, l_idx] += r
                    corr_count[p_idx, l_idx] += 1

    with np.errstate(divide="ignore", invalid="ignore"):
        matrix = np.where(corr_count > 0, corr_sum / corr_count, np.nan)
    return pd.DataFrame(matrix, index=all_item_cols, columns=all_item_cols)


def compute_implicit_subscale_matrix(llm_root, persona_pattern="persona_{scale}"):
    n = len(ALL_SUBSCALES)
    corr_sum = np.zeros((n, n))
    corr_count = np.zeros((n, n))

    for scale_file, scale_short in SCALES:
        persona_dir = os.path.join(llm_root, persona_pattern.format(scale=scale_file))
        if not os.path.exists(persona_dir):
            continue

        persona_data, persona_labels = load_persona_items(persona_dir, scale_file, scale_short)
        if persona_data is None:
            continue
        p_ids = persona_data["Subject_ID"]
        p_items = persona_data.drop(columns=["Subject_ID"])
        p_scores = compute_subscale_scores(p_items, persona_labels)
        p_scores.insert(0, "Subject_ID", p_ids.values)

        llm_data, llm_labels = load_llm_round_items(persona_dir)
        if llm_data is None:
            continue
        l_ids = llm_data["Subject_ID"]
        l_items = llm_data.drop(columns=["Subject_ID"])
        l_scores = compute_subscale_scores(l_items, llm_labels)
        l_scores.insert(0, "Subject_ID", l_ids.values)

        merged = p_scores.merge(l_scores, on="Subject_ID", suffixes=("_P", "_L"))
        print(f"  Persona={scale_short}: {len(merged)} subjects (subscale)")

        for p_sub in p_scores.columns:
            if p_sub == "Subject_ID" or p_sub not in ALL_SUBSCALES:
                continue
            p_idx = ALL_SUBSCALES.index(p_sub)
            p_col = f"{p_sub}_P" if f"{p_sub}_P" in merged.columns else p_sub
            p_vec = merged[p_col].values

            for l_sub in l_scores.columns:
                if l_sub == "Subject_ID" or l_sub not in ALL_SUBSCALES:
                    continue
                l_idx = ALL_SUBSCALES.index(l_sub)
                l_col = f"{l_sub}_L" if f"{l_sub}_L" in merged.columns else l_sub
                l_vec = merged[l_col].values

                r = _corr_vectors(p_vec, l_vec)
                if not np.isnan(r):
                    corr_sum[p_idx, l_idx] += r
                    corr_count[p_idx, l_idx] += 1

    with np.errstate(divide="ignore", invalid="ignore"):
        matrix = np.where(corr_count > 0, corr_sum / corr_count, np.nan)
    return pd.DataFrame(matrix, index=ALL_SUBSCALES, columns=ALL_SUBSCALES)


def aggregate_to_scale(subscale_matrix):
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


# ── Comparison ──

def compare_upper_tri(mat_a, mat_b):
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
        return {"matrix_r": np.nan, "mantel_p": np.nan, "rmse": np.nan, "n_pairs": 0}

    r, p = pearsonr(a_v, b_v)
    rmse = np.sqrt(np.mean((a_v - b_v) ** 2))

    rng = np.random.default_rng(42)
    n_perms = 5000
    count_ge = 0
    for _ in range(n_perms):
        perm = rng.permutation(n)
        B_perm = B[np.ix_(perm, perm)]
        b_perm_vec = B_perm[mask]
        valid_p = ~(np.isnan(a_vec) | np.isnan(b_perm_vec))
        if valid_p.sum() < 3:
            continue
        perm_r = np.corrcoef(a_vec[valid_p], b_perm_vec[valid_p])[0, 1]
        if perm_r >= r:
            count_ge += 1
    mantel_p = (count_ge + 1) / (n_perms + 1)

    return {"matrix_r": r, "pearson_p": p, "mantel_p": mantel_p, "rmse": rmse, "n_pairs": int(valid.sum())}


def _get_scale_for_col(col, item_labels=None):
    """Get scale name from a column name."""
    if item_labels:
        for s, c in item_labels:
            if c == col:
                return s
    # For subscale columns like "IRI_Perspective_taking"
    for scale in SCALE_NAMES:
        if col.startswith(scale + "_"):
            return scale
    return col


def compare_within_between(mat_a, mat_b, columns, item_labels=None):
    """Decompose comparison into within-scale and between-scale pairs."""
    common = [c for c in mat_a.columns if c in mat_b.columns]
    A = mat_a.loc[common, common].values
    B = mat_b.loc[common, common].values
    n = len(common)
    mask_r, mask_c = np.triu_indices(n, k=1)

    # Build scale membership
    col_scales = []
    for c in common:
        col_scales.append(_get_scale_for_col(c, item_labels))

    within = np.array([col_scales[r] == col_scales[c] for r, c in zip(mask_r, mask_c)])
    between = ~within

    results = {}
    for name, pair_mask in [("within", within), ("between", between), ("all", np.ones(len(mask_r), dtype=bool))]:
        a_vec = A[(mask_r, mask_c)][pair_mask]
        b_vec = B[(mask_r, mask_c)][pair_mask]
        valid = ~(np.isnan(a_vec) | np.isnan(b_vec))
        a_v, b_v = a_vec[valid], b_vec[valid]

        if len(a_v) < 3:
            results[name] = {"r": np.nan, "rmse": np.nan, "n_pairs": 0, "slope": np.nan}
            continue

        r, _ = pearsonr(a_v, b_v)
        rmse = np.sqrt(np.mean((a_v - b_v) ** 2))
        slope = np.polyfit(a_v, b_v, 1)[0] if len(a_v) > 1 else np.nan
        results[name] = {"r": r, "rmse": rmse, "n_pairs": int(valid.sum()), "slope": slope}

    return results


def plot_within_between_scatter(human_mat, llm_mat, level, item_labels, wb_metrics, output_dir):
    """Scatter with within-scale (red) and between-scale (blue) colored differently."""
    common = [c for c in human_mat.columns if c in llm_mat.columns]
    H = human_mat.loc[common, common].values
    L = llm_mat.loc[common, common].values
    n = len(common)
    mask_r, mask_c = np.triu_indices(n, k=1)

    col_scales = [_get_scale_for_col(c, item_labels) for c in common]
    within = np.array([col_scales[r] == col_scales[c] for r, c in zip(mask_r, mask_c)])

    h_vec = H[(mask_r, mask_c)]
    l_vec = L[(mask_r, mask_c)]
    valid = ~(np.isnan(h_vec) | np.isnan(l_vec))

    fig, ax = plt.subplots(figsize=(8, 8))

    # Between-scale (blue)
    bw_mask = valid & ~within
    if bw_mask.sum() > 0:
        ax.scatter(h_vec[bw_mask], l_vec[bw_mask],
                   alpha=0.3 if level == "item" else 0.5,
                   s=5 if level == "item" else 25,
                   c="#2c7bb6", label=f"Between (r={wb_metrics['between']['r']:.3f}, n={wb_metrics['between']['n_pairs']})")

    # Within-scale (red)
    wi_mask = valid & within
    if wi_mask.sum() > 0:
        ax.scatter(h_vec[wi_mask], l_vec[wi_mask],
                   alpha=0.3 if level == "item" else 0.5,
                   s=5 if level == "item" else 25,
                   c="#d7191c", label=f"Within (r={wb_metrics['within']['r']:.3f}, n={wb_metrics['within']['n_pairs']})")

    ax.plot([-1, 1], [-1, 1], "k--", alpha=0.3)
    ax.set_xlabel("Human correlation", fontsize=12)
    ax.set_ylabel("LLM implicit correlation", fontsize=12)
    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1)
    ax.set_aspect("equal")
    ax.legend(fontsize=9)

    titles = {"item": "Item-wise", "subscale": "Subscale-wise", "scale": "Scale-wise"}
    ax.set_title(f"{titles[level]}: Within vs Between Scale Pairs", fontsize=13)
    plt.tight_layout()
    fpath = os.path.join(output_dir, f"{level}_04_within_between_scatter.png")
    plt.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fpath}")


# ── Visualization ──

def plot_sidebyside(human_mat, llm_mat, level, item_labels, output_dir):
    common = [c for c in human_mat.columns if c in llm_mat.columns]
    H = human_mat.loc[common, common]
    L = llm_mat.loc[common, common]

    cfg = {
        "item":     {"figsize": (24, 10), "annot": False, "labels": False, "tick": 5},
        "subscale": {"figsize": (20, 8),  "annot": True,  "labels": True,  "tick": 7},
        "scale":    {"figsize": (16, 6),  "annot": True,  "labels": True,  "tick": 9},
    }[level]

    fig, axes = plt.subplots(1, 2, figsize=cfg["figsize"])

    if level == "subscale":
        short = [s.split("_", 1)[1] for s in common]
    else:
        short = common

    for ax, mat, title in zip(axes, [H, L], ["Human (actual)", "LLM (implicit)"]):
        kw = dict(cmap="RdBu_r", vmin=-1, vmax=1, center=0, cbar_kws={"shrink": 0.6}, ax=ax)
        if cfg["annot"]:
            kw.update(annot=True, fmt=".2f", annot_kws={"size": 7 if level == "subscale" else 10})
        kw["xticklabels"] = short if cfg["labels"] else False
        kw["yticklabels"] = short if cfg["labels"] else False
        sns.heatmap(mat.values, **kw)
        ax.set_title(title, fontsize=13)
        if cfg["labels"]:
            ax.tick_params(axis="both", labelsize=cfg["tick"])

    if level == "item" and item_labels:
        for ax_i in axes:
            pos = 0
            for _, sc in SCALES:
                ni = sum(1 for s, _ in item_labels if s == sc)
                if ni == 0:
                    continue
                ax_i.axhline(y=pos, color="black", linewidth=0.5)
                ax_i.axvline(x=pos, color="black", linewidth=0.5)
                ax_i.text(-1, pos + ni / 2, sc, ha="right", va="center", fontsize=7, fontweight="bold")
                pos += ni

    titles = {"item": "Item-wise (114x114)", "subscale": "Subscale-wise (16x16)", "scale": "Scale-wise (7x7)"}
    plt.suptitle(f"Implicit Structure: Human vs LLM - {titles[level]}", fontsize=14)
    plt.tight_layout()
    fpath = os.path.join(output_dir, f"{level}_01_sidebyside.png")
    plt.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fpath}")


def plot_scatter(human_mat, llm_mat, metrics, level, output_dir):
    common = [c for c in human_mat.columns if c in llm_mat.columns]
    H = human_mat.loc[common, common].values
    L = llm_mat.loc[common, common].values
    n = len(common)
    mask = np.triu_indices(n, k=1)
    h_vec, l_vec = H[mask], L[mask]
    valid = ~(np.isnan(h_vec) | np.isnan(l_vec))

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(h_vec[valid], l_vec[valid],
               alpha=0.3 if level == "item" else 0.6,
               s=5 if level == "item" else 30, c="#2c7bb6")
    ax.plot([-1, 1], [-1, 1], "k--", alpha=0.3, label="y = x")

    if valid.sum() > 2:
        z = np.polyfit(h_vec[valid], l_vec[valid], 1)
        x_line = np.linspace(-1, 1, 100)
        ax.plot(x_line, np.polyval(z, x_line), "r-", alpha=0.7, label=f"fit (slope={z[0]:.2f})")

    ax.set_xlabel("Human correlation", fontsize=12)
    ax.set_ylabel("LLM implicit correlation", fontsize=12)
    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1)
    ax.set_aspect("equal")
    ax.legend()

    titles = {"item": "Item-wise", "subscale": "Subscale-wise", "scale": "Scale-wise"}
    ax.set_title(f"{titles[level]}: Human vs LLM\nr = {metrics['matrix_r']:.3f}, Mantel p = {metrics['mantel_p']:.4f}", fontsize=13)
    plt.tight_layout()
    fpath = os.path.join(output_dir, f"{level}_02_scatter.png")
    plt.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fpath}")


def plot_difference(human_mat, llm_mat, level, output_dir):
    common = [c for c in human_mat.columns if c in llm_mat.columns]
    diff = llm_mat.loc[common, common] - human_mat.loc[common, common]

    cfg = {
        "item":     {"figsize": (14, 12), "annot": False, "labels": False},
        "subscale": {"figsize": (10, 8),  "annot": True,  "labels": True},
        "scale":    {"figsize": (8, 7),   "annot": True,  "labels": True},
    }[level]

    fig, ax = plt.subplots(figsize=cfg["figsize"])
    short = [s.split("_", 1)[1] for s in common] if level == "subscale" else common

    kw = dict(cmap="RdBu_r", vmin=-1, vmax=1, center=0,
              cbar_kws={"shrink": 0.6, "label": "LLM - Human"}, ax=ax)
    if cfg["annot"]:
        kw.update(annot=True, fmt=".2f", annot_kws={"size": 7 if level == "subscale" else 10})
    kw["xticklabels"] = short if cfg["labels"] else False
    kw["yticklabels"] = short if cfg["labels"] else False
    sns.heatmap(diff.values, **kw)

    titles = {"item": "Item-wise", "subscale": "Subscale-wise", "scale": "Scale-wise"}
    ax.set_title(f"Difference (LLM - Human): {titles[level]}", fontsize=13)
    plt.tight_layout()
    fpath = os.path.join(output_dir, f"{level}_03_difference.png")
    plt.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fpath}")


# ── Main ──

def main():
    parser = argparse.ArgumentParser(
        description="Compare LLM implicit psychometric structure with human data "
                    "(item-wise, subscale-wise, scale-wise)"
    )
    parser.add_argument("--llm_root", required=True)
    parser.add_argument("--human_dirs", nargs="+", default=None)
    parser.add_argument("--output_dir", default="outputs/behavior")
    parser.add_argument("--persona_pattern", default="persona_{scale}")
    args = parser.parse_args()

    human_dirs = args.human_dirs or [str(BASE_DIR / d) for d in HUMAN_DIRS_DEFAULT]
    llm_root = args.llm_root if os.path.isabs(args.llm_root) else str(BASE_DIR / args.llm_root)
    output_dir = args.output_dir if os.path.isabs(args.output_dir) else str(BASE_DIR / args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Load human data
    print("Loading human data...")
    human_data, item_labels = load_human_items(human_dirs)
    all_item_cols = [c for _, c in item_labels]
    print(f"  {human_data.shape[0]} subjects x {len(all_item_cols)} items")

    # Human matrices
    print("\nComputing human matrices...")
    h_data = human_data.drop(columns=["Subject_ID"], errors="ignore")
    human_item = h_data.corr()
    human_sub = compute_subscale_scores(h_data, item_labels)
    common_subs = [s for s in ALL_SUBSCALES if s in human_sub.columns]
    human_sub_mat = human_sub[common_subs].corr()
    human_scale_mat = aggregate_to_scale(human_sub_mat)

    # LLM implicit matrices
    print("\nComputing LLM implicit item matrix (114x114)...")
    llm_item = compute_implicit_item_matrix(llm_root, all_item_cols, item_labels, args.persona_pattern)

    print("\nComputing LLM implicit subscale matrix (16x16)...")
    llm_sub_mat = compute_implicit_subscale_matrix(llm_root, args.persona_pattern)

    print("\nComputing LLM implicit scale matrix (7x7)...")
    llm_scale_mat = aggregate_to_scale(llm_sub_mat)

    # Compare & plot at each level
    all_metrics = {}
    for level, h_mat, l_mat in [
        ("item", human_item, llm_item),
        ("subscale", human_sub_mat, llm_sub_mat),
        ("scale", human_scale_mat, llm_scale_mat),
    ]:
        print(f"\n{'='*50}")
        print(f"  {level.upper()}-WISE COMPARISON")
        print(f"{'='*50}")
        metrics = compare_upper_tri(h_mat, l_mat)
        all_metrics[level] = metrics
        print(f"  Matrix r:  {metrics['matrix_r']:.4f}")
        print(f"  Mantel p:  {metrics['mantel_p']:.4f}")
        print(f"  RMSE:      {metrics['rmse']:.4f}")
        print(f"  N pairs:   {metrics['n_pairs']}")

        # Within vs between decomposition (not for scale-level, already all between)
        wb_metrics = None
        if level in ("item", "subscale"):
            wb_metrics = compare_within_between(h_mat, l_mat, list(h_mat.columns), item_labels if level == "item" else None)
            print(f"  Within-scale:  r={wb_metrics['within']['r']:.4f}  RMSE={wb_metrics['within']['rmse']:.4f}  slope={wb_metrics['within']['slope']:.2f}  (N={wb_metrics['within']['n_pairs']})")
            print(f"  Between-scale: r={wb_metrics['between']['r']:.4f}  RMSE={wb_metrics['between']['rmse']:.4f}  slope={wb_metrics['between']['slope']:.2f}  (N={wb_metrics['between']['n_pairs']})")
            all_metrics[f"{level}_within"] = wb_metrics["within"]
            all_metrics[f"{level}_between"] = wb_metrics["between"]

        l_mat.to_csv(os.path.join(output_dir, f"{level}_llm_implicit.csv"))
        h_mat.to_csv(os.path.join(output_dir, f"{level}_human.csv"))

        plot_sidebyside(h_mat, l_mat, level, item_labels, output_dir)
        plot_scatter(h_mat, l_mat, metrics, level, output_dir)
        plot_difference(h_mat, l_mat, level, output_dir)
        if wb_metrics:
            plot_within_between_scatter(h_mat, l_mat, level, item_labels if level == "item" else None, wb_metrics, output_dir)

    # Save metrics
    rows = [{"level": k, **v} for k, v in all_metrics.items()]
    pd.DataFrame(rows).to_csv(os.path.join(output_dir, "comparison_metrics.csv"), index=False)

    # Summary
    print(f"\n{'='*50}")
    print("  SUMMARY")
    print(f"{'='*50}")
    for level, m in all_metrics.items():
        if "n_pairs" in m:
            print(f"  {level:<20}  r={m.get('matrix_r', m.get('r', 0)):.3f}  RMSE={m.get('rmse', 0):.3f}  (N={m['n_pairs']})")
        else:
            print(f"  {level:<20}  r={m.get('r', 0):.3f}  (N={m.get('n_pairs', 0)})")

    print("\n== Human Scale Correlation ==")
    print(human_scale_mat.round(3).to_string())
    print("\n== LLM Implicit Scale Correlation ==")
    print(llm_scale_mat.round(3).to_string())

    report = f"""Implicit Psychometric Structure Analysis (3 levels)
=====================================================
Human data: {human_dirs}
LLM data:   {llm_root}

Summary:
"""
    for level, m in all_metrics.items():
        r_val = m.get('matrix_r', m.get('r', float('nan')))
        rmse_val = m.get('rmse', float('nan'))
        n_val = m.get('n_pairs', 0)
        mantel_val = m.get('mantel_p', '')
        mantel_str = f"  Mantel p={mantel_val:.4f}" if isinstance(mantel_val, float) and not np.isnan(mantel_val) else ""
        report += f"  {level:<20}  r={r_val:.4f}{mantel_str}  RMSE={rmse_val:.4f}  (N={n_val})\n"
    report += f"\nAll outputs saved to: {output_dir}\n"
    with open(os.path.join(output_dir, "report.txt"), "w") as f:
        f.write(report)
    print(report)


if __name__ == "__main__":
    main()
