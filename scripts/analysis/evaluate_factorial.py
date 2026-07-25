#!/usr/bin/env python3
"""
Factorial Profile Regression Analysis

For BigFive factorial profiles (3^5 = 243 profiles), fit regression models
predicting LLM's target subscale scores from the 5 Big Five dimensions.

Includes human baseline: same regression on real human data (N=272),
directly comparing LLM vs human coefficients to test whether
amplification = over-sized coefficients.

Usage:
    python evaluate_factorial.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR / "src"))

from psychometric_inference.scoring import (
    SCORING_RULES, FILENAME_TO_SCALE, ALL_SUBSCALES,
    compute_subscale_scores, un_reverse_df,
)
from psychometric_inference.model_registry import qwen_base_experiment_tuples

SCALES = [
    ("IRI", "IRI"), ("PANAS", "PANAS"), ("POM", "POM"),
    ("big_five", "BigFive"), ("in_inter_dependent", "SelfConst"),
    ("Life_Satisfaction", "LifeSat"), ("Loneliness", "Lonely"),
]
HUMAN_DIRS = ["data/human/SED", "data/human/SEDC", "data/human/SEDD"]
BIG_FIVE_SUBS = ["Extraversion", "Agreeableness", "Conscientiousness", "Neuroticism", "Openness"]
BF_FULL = [f"BigFive_{b}" for b in BIG_FIVE_SUBS]
MODELS = qwen_base_experiment_tuples("factorial")
TARGET_SUBS = [s for s in ALL_SUBSCALES if not s.startswith("BigFive_")]


class OLSResult:
    """Small ordinary-least-squares result compatible with the fields used here."""

    def __init__(self, intercept, coef):
        self.intercept_ = float(intercept)
        self.coef_ = np.asarray(coef, dtype=float)

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return self.intercept_ + X @ self.coef_

    def score(self, X, y):
        y = np.asarray(y, dtype=float)
        pred = self.predict(X)
        ss_res = np.sum((y - pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        return np.nan if ss_tot == 0 else 1 - ss_res / ss_tot


def fit_ols(X, y):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    design = np.column_stack([np.ones(len(X)), X])
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    return OLSResult(beta[0], beta[1:])


def interaction_features(X):
    X = np.asarray(X, dtype=float)
    cols = [X]
    for i in range(X.shape[1]):
        for j in range(i + 1, X.shape[1]):
            cols.append((X[:, i] * X[:, j])[:, None])
    return np.column_stack(cols)


def _detect_id_col(df):
    for c in ["Subject_ID", "ID", "Scan_ID", "id"]:
        if c in df.columns:
            return c
    return None


def load_human_subscale_scores():
    scale_dfs = {}
    item_labels = []
    for scale_file, scale_short in SCALES:
        frames, labs = [], []
        for d in HUMAN_DIRS:
            fpath = BASE_DIR / d / f"{scale_file}.csv"
            if not fpath.exists():
                continue
            df = pd.read_csv(fpath)
            id_col = _detect_id_col(df)
            df = un_reverse_df(df, scale_file)
            q_cols = [c for c in df.columns if c.startswith("Q")]
            sub = df[q_cols].copy()
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
    if merged is None:
        return None
    item_labels = [(s, c) for s, c in item_labels if c in merged.columns]
    data = merged.drop(columns=["Subject_ID"])
    scores = compute_subscale_scores(data, item_labels)
    scores.insert(0, "Subject_ID", merged["Subject_ID"].values)
    return scores


def load_factorial_data(experiment_name):
    persona_dir = BASE_DIR / "data/llm_behavior" / experiment_name / "persona_big_five"
    if not persona_dir.exists():
        return None, None
    persona_df = pd.read_csv(persona_dir / "big_five_persona.csv")
    rows = []
    for _, row in persona_df.iterrows():
        sid = row["Subject_ID"]
        levels = {"Subject_ID": sid}
        for part in sid.split("_"):
            if "=" in part:
                name, val = part.split("=")
                levels[name] = int(val)
        rows.append(levels)
    X = pd.DataFrame(rows)[["Subject_ID"] + BIG_FIVE_SUBS]
    all_items, item_labels = [], []
    for scale_file, scale_short in SCALES:
        if scale_short == "BigFive":
            continue
        fpath = persona_dir / f"{scale_file}.csv"
        if not fpath.exists():
            continue
        df = pd.read_csv(fpath)
        q_cols = [c for c in df.columns if c.startswith("Q")]
        sub = df[q_cols].copy()
        rename = {c: f"{scale_short}_{c}" for c in q_cols}
        sub.rename(columns=rename, inplace=True)
        if "Subject_ID" in df.columns:
            sub.insert(0, "Subject_ID", df["Subject_ID"].astype(str))
        all_items.append(sub)
        for c in rename.values():
            item_labels.append((scale_short, c))
    if not all_items:
        return None, None
    merged = all_items[0]
    for df in all_items[1:]:
        merged = merged.merge(df, on="Subject_ID", how="inner")
    data = merged.drop(columns=["Subject_ID"])
    scores = compute_subscale_scores(data, item_labels)
    scores.insert(0, "Subject_ID", merged["Subject_ID"].values)
    combined = X.merge(scores, on="Subject_ID")
    X_out = combined[BIG_FIVE_SUBS].astype(float)
    Y_out = combined[[s for s in TARGET_SUBS if s in combined.columns]]
    return X_out, Y_out


def _standardized_regression(X_raw, y_raw):
    """Fit OLS and return both raw and standardized (beta) coefficients."""
    reg = fit_ols(X_raw, y_raw)
    r2 = reg.score(X_raw, y_raw)
    # Standardized coefficients: beta_j = b_j * (sd_x_j / sd_y)
    sd_x = np.std(X_raw, axis=0, ddof=1)
    sd_y = np.std(y_raw, ddof=1)
    if sd_y == 0:
        betas = np.zeros_like(reg.coef_)
    else:
        betas = reg.coef_ * sd_x / sd_y
    return reg, r2, betas


def fit_regression(X, Y):
    results = []
    for target in Y.columns:
        y = Y[target].values
        valid = ~np.isnan(y)
        if valid.sum() < 10:
            continue
        reg, r2, betas = _standardized_regression(X.values[valid], y[valid])
        row = {"target": target, "R2": r2, "intercept": reg.intercept_}
        for i, feat in enumerate(BIG_FIVE_SUBS):
            row[feat] = reg.coef_[i]
            row[f"{feat}_beta"] = betas[i]
        results.append(row)
    return pd.DataFrame(results)


def fit_human_regression(human_scores):
    X = human_scores[BF_FULL].values
    results = []
    for target in TARGET_SUBS:
        if target not in human_scores.columns:
            continue
        y = human_scores[target].values
        valid = ~(np.isnan(y) | np.any(np.isnan(X), axis=1))
        if valid.sum() < 10:
            continue
        reg, r2, betas = _standardized_regression(X[valid], y[valid])
        row = {"target": target, "R2": r2, "intercept": reg.intercept_}
        for i, feat in enumerate(BIG_FIVE_SUBS):
            row[feat] = reg.coef_[i]
            row[f"{feat}_beta"] = betas[i]
        results.append(row)
    return pd.DataFrame(results)


def fit_with_interactions(X, Y):
    results = []
    for target in Y.columns:
        y = Y[target].values
        valid = ~np.isnan(y)
        if valid.sum() < 10:
            continue
        reg_main = fit_ols(X.values[valid], y[valid])
        r2_main = reg_main.score(X.values[valid], y[valid])
        X_inter = interaction_features(X.values[valid])
        reg_inter = fit_ols(X_inter, y[valid])
        r2_inter = reg_inter.score(X_inter, y[valid])
        results.append({"target": target, "R2_main": r2_main, "R2_interaction": r2_inter, "R2_gain": r2_inter - r2_main})
    return pd.DataFrame(results)


def plot_coefficient_heatmap(coef_df, model_label, output_dir):
    targets = coef_df["target"].values
    short_targets = [t.split("_", 1)[1] if "_" in t else t for t in targets]
    beta_cols = [f"{b}_beta" for b in BIG_FIVE_SUBS]
    if beta_cols[0] in coef_df.columns:
        coefs = coef_df[beta_cols].values
        label = "Standardized coefficient (β)"
    else:
        coefs = coef_df[BIG_FIVE_SUBS].values
        label = "Regression coefficient"
    fig, ax = plt.subplots(figsize=(8, 9))
    sns.heatmap(coefs, cmap="RdBu_r", center=0, annot=True, fmt=".2f",
                annot_kws={"size": 8}, ax=ax,
                xticklabels=BIG_FIVE_SUBS, yticklabels=short_targets,
                cbar_kws={"shrink": 0.6, "label": label})
    ax.set_title(f"Big Five → Target Subscales: {model_label}\n(standardized β)", fontsize=11)
    ax.set_xlabel("Big Five dimension")
    ax.set_ylabel("Target subscale")
    plt.tight_layout()
    plt.savefig(output_dir / f"factorial_coefs_{model_label}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: factorial_coefs_{model_label}.png")


def plot_human_vs_llm_coefs(human_coefs, llm_coefs_14b, output_dir):
    common_targets = [t for t in human_coefs["target"].values if t in llm_coefs_14b["target"].values]
    h = human_coefs[human_coefs["target"].isin(common_targets)].set_index("target").loc[common_targets]
    l = llm_coefs_14b[llm_coefs_14b["target"].isin(common_targets)].set_index("target").loc[common_targets]
    short_targets = [t.split("_", 1)[1] if "_" in t else t for t in common_targets]

    beta_cols = [f"{b}_beta" for b in BIG_FIVE_SUBS]

    # Side-by-side heatmaps (standardized)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))
    h_vals_mat = h[beta_cols].values
    l_vals_mat = l[beta_cols].values
    vmax = max(abs(h_vals_mat).max(), abs(l_vals_mat).max())
    sns.heatmap(h_vals_mat, cmap="RdBu_r", center=0, vmin=-vmax, vmax=vmax,
                annot=True, fmt=".2f", annot_kws={"size": 8}, ax=ax1,
                xticklabels=BIG_FIVE_SUBS, yticklabels=short_targets, cbar_kws={"shrink": 0.6})
    ax1.set_title("Human (N=272)\nStandardized β", fontsize=11)
    ax1.set_xlabel("Big Five dimension"); ax1.set_ylabel("Target subscale")
    sns.heatmap(l_vals_mat, cmap="RdBu_r", center=0, vmin=-vmax, vmax=vmax,
                annot=True, fmt=".2f", annot_kws={"size": 8}, ax=ax2,
                xticklabels=BIG_FIVE_SUBS, yticklabels=short_targets, cbar_kws={"shrink": 0.6})
    ax2.set_title("LLM 14B (factorial)\nStandardized β", fontsize=11)
    ax2.set_xlabel("Big Five dimension")
    plt.tight_layout()
    plt.savefig(output_dir / "human_vs_llm_coefs_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: human_vs_llm_coefs_heatmap.png")

    # Scatter (standardized)
    h_vals = h[beta_cols].values.flatten()
    l_vals = l[beta_cols].values.flatten()
    valid = ~(np.isnan(h_vals) | np.isnan(l_vals))
    h_v, l_v = h_vals[valid], l_vals[valid]
    from scipy.stats import pearsonr
    r, p = pearsonr(h_v, l_v)
    slope = np.polyfit(h_v, l_v, 1)[0]

    fig, ax = plt.subplots(figsize=(6, 6))
    colors = {"Extraversion": "#e41a1c", "Agreeableness": "#377eb8",
              "Conscientiousness": "#4daf4a", "Neuroticism": "#984ea3", "Openness": "#ff7f00"}
    for i, bf in enumerate(BIG_FIVE_SUBS):
        ax.scatter(h[beta_cols[i]].values, l[beta_cols[i]].values,
                   c=colors[bf], label=bf, alpha=0.7, s=30, edgecolors="white", linewidths=0.5)
    xx = np.linspace(h_v.min() - 0.05, h_v.max() + 0.05, 100)
    ax.plot(xx, np.polyval(np.polyfit(h_v, l_v, 1), xx), "k--", lw=1, alpha=0.7)
    lim = max(abs(h_v).max(), abs(l_v).max()) * 1.2
    ax.plot([-lim, lim], [-lim, lim], "gray", ls=":", lw=0.8, alpha=0.5, label="y = x")
    ax.set_xlabel("Human standardized β"); ax.set_ylabel("LLM 14B standardized β")
    ax.set_title(f"Human vs LLM Standardized Coefficients\nr = {r:.3f}, slope = {slope:.2f}\n"
                 f"(slope > 1 = amplification, slope < 1 = attenuation)", fontsize=11)
    ax.legend(fontsize=7, loc="upper left"); ax.set_aspect("equal")
    ax.axhline(y=0, color="gray", lw=0.3); ax.axvline(x=0, color="gray", lw=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "human_vs_llm_coefs_scatter.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: human_vs_llm_coefs_scatter.png")
    return {"r": r, "p": p, "slope": slope, "n_coefs": len(h_v)}


def plot_r2_comparison(all_r2, output_dir):
    fig, axes = plt.subplots(1, len(MODELS), figsize=(5 * len(MODELS), 5), sharey=True)
    if len(MODELS) == 1: axes = [axes]
    for ax, (_, label) in zip(axes, MODELS):
        sub = all_r2[all_r2["model"] == label]
        if sub.empty: continue
        targets = sub["target"].values
        short = [t.split("_", 1)[1] if "_" in t else t for t in targets]
        x = np.arange(len(targets)); w = 0.35
        ax.barh(x - w/2, sub["R2_main"], w, label="Main effects only", color="#2c7bb6", alpha=0.8)
        ax.barh(x + w/2, sub["R2_interaction"], w, label="+ Interactions", color="#d7191c", alpha=0.8)
        ax.set_yticks(x); ax.set_yticklabels(short, fontsize=7)
        ax.set_xlabel("R²"); ax.set_title(f"Qwen {label} base", fontsize=10); ax.set_xlim(0, 1.0)
    axes[0].legend(fontsize=7, loc="lower right")
    plt.suptitle("How well do Big Five levels predict LLM target scores?", fontsize=11)
    plt.tight_layout()
    plt.savefig(output_dir / "factorial_r2_comparison.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: factorial_r2_comparison.png")


def main():
    output_dir = BASE_DIR / "outputs/behavior" / "factorial_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  Factorial Regression Analysis: Big Five → Target Subscales")
    print("=" * 60)

    # Human baseline
    print("\n  Loading human data...")
    human_scores = load_human_subscale_scores()
    human_coefs = None
    if human_scores is not None:
        print(f"    {len(human_scores)} subjects")
        human_coefs = fit_human_regression(human_scores)
        human_coefs["model"] = "Human"
        print(f"\n    Human regression (Big Five → Target):")
        print(f"    {'Target':<35} {'R2':>6}  {'E':>7} {'A':>7} {'C':>7} {'N':>7} {'O':>7}")
        print(f"    {'-'*80}")
        for _, row in human_coefs.iterrows():
            print(f"    {row['target']:<35} {row['R2']:>6.3f}  "
                  f"{row['Extraversion']:>+7.3f} {row['Agreeableness']:>+7.3f} "
                  f"{row['Conscientiousness']:>+7.3f} {row['Neuroticism']:>+7.3f} "
                  f"{row['Openness']:>+7.3f}")
        human_coefs.to_csv(output_dir / "human_coefficients.csv", index=False)
        plot_coefficient_heatmap(human_coefs, "Human", output_dir)
    else:
        print("    Could not load human data")

    # LLM models
    all_coefs, all_r2 = [], []
    for exp_name, model_label in MODELS:
        print(f"\n  Loading {exp_name}...")
        X, Y = load_factorial_data(exp_name)
        if X is None:
            print(f"    No data found"); continue
        print(f"    {len(X)} profiles, {len(Y.columns)} target subscales")
        coef_df = fit_regression(X, Y)
        coef_df["model"] = model_label
        all_coefs.append(coef_df)
        print(f"\n    Top absolute coefficients:")
        for _, row in coef_df.iterrows():
            max_bf = max(BIG_FIVE_SUBS, key=lambda b: abs(row[b]))
            print(f"      {row['target']:<35} R2={row['R2']:.3f}  strongest: {max_bf} ({row[max_bf]:+.3f})")
        r2_df = fit_with_interactions(X, Y)
        r2_df["model"] = model_label
        all_r2.append(r2_df)
        print(f"\n    Mean R2 gain from interactions: {r2_df['R2_gain'].mean():.4f}")
        plot_coefficient_heatmap(coef_df, model_label, output_dir)

    if not all_coefs:
        print("No LLM data found."); return
    all_coefs = pd.concat(all_coefs, ignore_index=True)
    all_r2 = pd.concat(all_r2, ignore_index=True)
    all_coefs.to_csv(output_dir / "factorial_coefficients.csv", index=False)
    all_r2.to_csv(output_dir / "factorial_r2.csv", index=False)
    plot_r2_comparison(all_r2, output_dir)

    # Human vs LLM comparison
    if human_coefs is not None:
        llm_14b = all_coefs[all_coefs["model"] == "14B"]
        if not llm_14b.empty:
            print(f"\n{'='*60}")
            print("  HUMAN vs LLM COEFFICIENT COMPARISON")
            print(f"{'='*60}")
            scatter_stats = plot_human_vs_llm_coefs(human_coefs, llm_14b, output_dir)
            print(f"\n  Human vs LLM 14B coefficient scatter:")
            print(f"    r = {scatter_stats['r']:.3f} (p = {scatter_stats['p']:.2e})")
            print(f"    slope = {scatter_stats['slope']:.3f}")
            print(f"    N coefficients = {scatter_stats['n_coefs']}")
            print(f"\n    slope > 1 = LLM amplifies coefficients")
            print(f"    slope < 1 = LLM attenuates coefficients")
            print(f"    slope ~ 1 = LLM matches human magnitudes")

            print(f"\n  Per-target R2 comparison:")
            print(f"    {'Target':<35} {'Human':>9} {'LLM':>9} {'Ratio':>7}")
            print(f"    {'-'*65}")
            for t in human_coefs["target"].values:
                h_r2 = human_coefs[human_coefs["target"] == t]["R2"].values
                l_r2 = llm_14b[llm_14b["target"] == t]["R2"].values
                if len(h_r2) > 0 and len(l_r2) > 0:
                    ratio = l_r2[0] / h_r2[0] if h_r2[0] > 0 else float("inf")
                    print(f"    {t:<35} {h_r2[0]:>9.3f} {l_r2[0]:>9.3f} {ratio:>7.2f}")

    # Summary
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    for _, label in MODELS:
        sub = all_r2[all_r2["model"] == label]
        if sub.empty: continue
        print(f"\n  {label}:")
        print(f"    Mean R2 (main effects): {sub['R2_main'].mean():.3f}")
        print(f"    Mean R2 (+ interactions): {sub['R2_interaction'].mean():.3f}")
        print(f"    Mean R2 gain: {sub['R2_gain'].mean():.4f}")
    print(f"\n  All outputs in: {output_dir}")


if __name__ == "__main__":
    main()
