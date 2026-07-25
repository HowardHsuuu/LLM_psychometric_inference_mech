"""
Per-Scale Alignment Analysis:
For each scale, compute how well the LLM's implicit cross-scale correlations
match human data, separately for each scale's row in the correlation matrix.

This reveals which psychological constructs the LLM understands well vs poorly.

Usage:
    python compute_scale_structure.py \
        --results_dir outputs/behavior/qwen14b \
        --output_dir outputs/behavior/qwen14b
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR / "src"))

from psychometric_inference.scoring import SCORING_RULES, ALL_SUBSCALES

SCALE_NAMES = ["IRI", "PANAS", "POM", "BigFive", "SelfConst", "LifeSat", "Lonely"]

SCALE_DISPLAY = {
    "IRI": "IRI\n(Empathy)",
    "PANAS": "PANAS\n(Affect)",
    "POM": "POM\n(Peace of Mind)",
    "BigFive": "Big Five\n(Personality)",
    "SelfConst": "SCS\n(Self-Construal)",
    "LifeSat": "SWLS\n(Life Satisfaction)",
    "Lonely": "UCLA\n(Loneliness)",
}

# Map subscale -> parent scale
SUB_TO_SCALE = {}
for scale_short, rules in SCORING_RULES.items():
    for sub_name in rules["subscales"]:
        SUB_TO_SCALE[f"{scale_short}_{sub_name}"] = scale_short


def per_scale_alignment_subscale(human_csv, llm_csv):
    """
    For each scale, extract its rows from the subscale matrix,
    take only between-scale columns, and correlate with human.

    Returns dict: scale -> {r, p, n_pairs, rmse, slope}
    """
    H = pd.read_csv(human_csv, index_col=0)
    L = pd.read_csv(llm_csv, index_col=0)
    common = [c for c in H.columns if c in L.columns]
    H = H.loc[common, common]
    L = L.loc[common, common]

    results = {}
    for scale in SCALE_NAMES:
        # Get subscales belonging to this scale
        my_subs = [s for s in common if SUB_TO_SCALE.get(s) == scale]
        other_subs = [s for s in common if SUB_TO_SCALE.get(s) != scale]

        if not my_subs or not other_subs:
            continue

        # Extract between-scale correlations for this scale's rows
        h_vals = []
        l_vals = []
        for row in my_subs:
            for col in other_subs:
                hv = H.loc[row, col]
                lv = L.loc[row, col]
                if not np.isnan(hv) and not np.isnan(lv):
                    h_vals.append(hv)
                    l_vals.append(lv)

        h_arr = np.array(h_vals)
        l_arr = np.array(l_vals)

        if len(h_arr) < 3:
            results[scale] = {"r": np.nan, "p": np.nan, "n_pairs": len(h_arr),
                              "rmse": np.nan, "slope": np.nan}
            continue

        r, p = pearsonr(h_arr, l_arr)
        rmse = np.sqrt(np.mean((h_arr - l_arr) ** 2))
        slope = np.polyfit(h_arr, l_arr, 1)[0]

        results[scale] = {"r": r, "p": p, "n_pairs": len(h_arr),
                          "rmse": rmse, "slope": slope}

    return results


def per_scale_alignment_item(human_csv, llm_csv, item_labels=None):
    """
    Same but at item level. Need to know which items belong to which scale.
    """
    H = pd.read_csv(human_csv, index_col=0)
    L = pd.read_csv(llm_csv, index_col=0)
    common = [c for c in H.columns if c in L.columns]
    H = H.loc[common, common]
    L = L.loc[common, common]

    # Infer scale from column name (e.g., "IRI_Q1" -> "IRI")
    def get_scale(col):
        for scale in SCALE_NAMES:
            if col.startswith(scale + "_"):
                return scale
        return None

    results = {}
    for scale in SCALE_NAMES:
        my_items = [c for c in common if get_scale(c) == scale]
        other_items = [c for c in common if get_scale(c) is not None and get_scale(c) != scale]

        if not my_items or not other_items:
            continue

        h_vals = []
        l_vals = []
        for row in my_items:
            for col in other_items:
                hv = H.loc[row, col]
                lv = L.loc[row, col]
                if not np.isnan(hv) and not np.isnan(lv):
                    h_vals.append(hv)
                    l_vals.append(lv)

        h_arr = np.array(h_vals)
        l_arr = np.array(l_vals)

        if len(h_arr) < 3:
            results[scale] = {"r": np.nan, "p": np.nan, "n_pairs": len(h_arr),
                              "rmse": np.nan, "slope": np.nan}
            continue

        r, p = pearsonr(h_arr, l_arr)
        rmse = np.sqrt(np.mean((h_arr - l_arr) ** 2))
        slope = np.polyfit(h_arr, l_arr, 1)[0]

        results[scale] = {"r": r, "p": p, "n_pairs": len(h_arr),
                          "rmse": rmse, "slope": slope}

    return results


def plot_per_scale_bar(results_sub, results_item, output_dir, model_name=""):
    """Bar chart showing per-scale alignment r at subscale and item level."""
    scales = [s for s in SCALE_NAMES if s in results_sub]
    r_sub = [results_sub[s]["r"] for s in scales]
    r_item = [results_item[s]["r"] for s in scales]
    labels = [SCALE_DISPLAY.get(s, s) for s in scales]

    x = np.arange(len(scales))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width/2, r_sub, width, label="Subscale-wise", color="#2c7bb6", alpha=0.8)
    bars2 = ax.bar(x + width/2, r_item, width, label="Item-wise", color="#d7191c", alpha=0.8)

    ax.set_ylabel("Alignment r (LLM vs Human)", fontsize=12)
    ax.set_xlabel("Scale", fontsize=12)
    ax.set_title(f"Per-Scale Alignment with Human Data{' — ' + model_name if model_name else ''}", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.axhline(y=np.mean(r_sub), color="#2c7bb6", linestyle="--", alpha=0.5, label=f"Mean subscale r={np.mean(r_sub):.3f}")
    ax.axhline(y=np.mean(r_item), color="#d7191c", linestyle="--", alpha=0.5, label=f"Mean item r={np.mean(r_item):.3f}")

    # Add value labels
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)

    ax.legend(loc="lower right")
    plt.tight_layout()
    fpath = os.path.join(output_dir, "per_scale_alignment_bar.png")
    plt.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fpath}")


def plot_per_scale_scatter_grid(human_csv, llm_csv, level, output_dir, model_name=""):
    """Grid of scatter plots, one per scale, showing that scale's between-scale pairs."""
    H = pd.read_csv(human_csv, index_col=0)
    L = pd.read_csv(llm_csv, index_col=0)
    common = [c for c in H.columns if c in L.columns]
    H = H.loc[common, common]
    L = L.loc[common, common]

    def get_scale(col):
        if level == "subscale":
            return SUB_TO_SCALE.get(col)
        for scale in SCALE_NAMES:
            if col.startswith(scale + "_"):
                return scale
        return None

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    for idx, scale in enumerate(SCALE_NAMES):
        ax = axes[idx]
        my = [c for c in common if get_scale(c) == scale]
        other = [c for c in common if get_scale(c) is not None and get_scale(c) != scale]

        h_vals, l_vals = [], []
        for row in my:
            for col in other:
                hv, lv = H.loc[row, col], L.loc[row, col]
                if not np.isnan(hv) and not np.isnan(lv):
                    h_vals.append(hv)
                    l_vals.append(lv)

        h_arr, l_arr = np.array(h_vals), np.array(l_vals)

        ax.scatter(h_arr, l_arr, alpha=0.4, s=15, c="#2c7bb6")
        ax.plot([-1, 1], [-1, 1], "k--", alpha=0.3)

        if len(h_arr) > 2:
            r, _ = pearsonr(h_arr, l_arr)
            z = np.polyfit(h_arr, l_arr, 1)
            x_line = np.linspace(-1, 1, 50)
            ax.plot(x_line, np.polyval(z, x_line), "r-", alpha=0.6)
            ax.set_title(f"{scale}\nr={r:.3f}, slope={z[0]:.2f}", fontsize=10)
        else:
            ax.set_title(f"{scale}\nN/A", fontsize=10)

        ax.set_xlim(-1, 1); ax.set_ylim(-1, 1)
        ax.set_aspect("equal")
        if idx >= 4:
            ax.set_xlabel("Human", fontsize=9)
        if idx % 4 == 0:
            ax.set_ylabel("LLM", fontsize=9)

    # Hide last subplot if 7 scales
    if len(SCALE_NAMES) < 8:
        axes[7].set_visible(False)

    level_str = "Subscale" if level == "subscale" else "Item"
    plt.suptitle(f"Per-Scale {level_str}-wise Alignment{' — ' + model_name if model_name else ''}", fontsize=14)
    plt.tight_layout()
    fpath = os.path.join(output_dir, f"per_scale_{level}_scatter_grid.png")
    plt.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fpath}")


def main():
    parser = argparse.ArgumentParser(description="Per-scale alignment analysis")
    parser.add_argument("--results_dir", required=True, help="Directory with *_llm_implicit.csv and *_human.csv")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--model_name", default="", help="Model name for plot titles")
    args = parser.parse_args()

    results_dir = args.results_dir
    output_dir = args.output_dir or results_dir

    sub_h = os.path.join(results_dir, "subscale_human.csv")
    sub_l = os.path.join(results_dir, "subscale_llm_implicit.csv")
    item_h = os.path.join(results_dir, "item_human.csv")
    item_l = os.path.join(results_dir, "item_llm_implicit.csv")

    print("== Per-Scale Alignment (Subscale-wise, between-scale only) ==")
    results_sub = per_scale_alignment_subscale(sub_h, sub_l)
    for scale, m in results_sub.items():
        print(f"  {scale:<12}  r={m['r']:.3f}  slope={m['slope']:.2f}  RMSE={m['rmse']:.3f}  (N={m['n_pairs']})")

    print("\n== Per-Scale Alignment (Item-wise, between-scale only) ==")
    results_item = per_scale_alignment_item(item_h, item_l)
    for scale, m in results_item.items():
        print(f"  {scale:<12}  r={m['r']:.3f}  slope={m['slope']:.2f}  RMSE={m['rmse']:.3f}  (N={m['n_pairs']})")

    # Save
    rows = []
    for scale in SCALE_NAMES:
        row = {"scale": scale}
        if scale in results_sub:
            for k, v in results_sub[scale].items():
                row[f"subscale_{k}"] = v
        if scale in results_item:
            for k, v in results_item[scale].items():
                row[f"item_{k}"] = v
        rows.append(row)
    pd.DataFrame(rows).to_csv(os.path.join(output_dir, "per_scale_alignment.csv"), index=False)

    # Plots
    plot_per_scale_bar(results_sub, results_item, output_dir, args.model_name)
    plot_per_scale_scatter_grid(sub_h, sub_l, "subscale", output_dir, args.model_name)
    plot_per_scale_scatter_grid(item_h, item_l, "item", output_dir, args.model_name)

    print(f"\nAll outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
