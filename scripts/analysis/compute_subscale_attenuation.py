#!/usr/bin/env python3
"""
Subscale-level attenuation correction for the cross-persona behavioral pipeline.

Why a new script:
  compute_attenuation_correction.py runs at SCALE level (7x7), reading scale_human.csv
  / scale_llm_implicit.csv. The paper's main amplification analysis is at the
  SUBSCALE level (16x16), so the corrected slopes (1.18-1.29 for >=7B reported
  in §3.3) should also be at the subscale level.

What this script does:
  Reads subscale_human.csv / subscale_llm_implicit.csv per model, computes
  subscale-level Cronbach's alphas from the raw human item data, disattenuates
  the human matrix only (UNILATERAL: matches §3.3, NOT App H's bilateral
  description), then OLS-regresses the LLM matrix on the human matrix to get
  the corrected slope.

Outputs:
  outputs/behavior/attenuation_correction_subscale_sensitivity.csv
  reports/figures/figbeh_attenuation_subscale_sensitivity.pdf

Usage (from project root):
  python compute_subscale_attenuation.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from psychometric_inference.scoring import SCORING_RULES
from psychometric_inference.model_registry import analysis_model_tuples
from psychometric_inference.paths import BEHAVIOR_OUTPUT_DIR, FIGURE_DIR, HUMAN_DATA_DIR

# Match the layout of compute_attenuation_correction.py
SCALES_MAP = [
    ("IRI", "IRI"),
    ("PANAS", "PANAS"),
    ("POM", "POM"),
    ("big_five", "BigFive"),
    ("in_inter_dependent", "SelfConst"),
    ("Life_Satisfaction", "LifeSat"),
    ("Loneliness", "Lonely"),
]

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

MODELS = analysis_model_tuples()


def cronbach_alpha(items_df):
    """Cronbach's alpha. items_df should already have reverse coding applied
    (human CSVs in data/human do).
    """
    items = items_df.dropna(axis=1, how="all").dropna(axis=0)
    n = items.shape[1]
    if n < 2:
        return np.nan
    item_vars = items.var(axis=0, ddof=1)
    total_var = items.sum(axis=1).var(ddof=1)
    if total_var == 0:
        return np.nan
    return (n / (n - 1)) * (1 - item_vars.sum() / total_var)


def compute_subscale_alphas() -> dict:
    """Subscale-level Cronbach's alphas, keyed by full subscale name (e.g. IRI_Fantasy)."""
    out = {}
    for scale_file, scale_short in SCALES_MAP:
        frames = []
        for ds in ["SED", "SEDC", "SEDD"]:
            fpath = HUMAN_DATA_DIR / ds / f"{scale_file}.csv"
            if fpath.exists():
                frames.append(pd.read_csv(fpath))
        if not frames:
            print(f"  WARN: no human data for {scale_file}")
            continue
        df = pd.concat(frames, ignore_index=True)

        rules = SCORING_RULES.get(scale_short, {})
        for sub_name, item_nums in rules.get("subscales", {}).items():
            sub_cols = [f"Q{n}" for n in item_nums]
            valid = [c for c in sub_cols if c in df.columns]
            if len(valid) >= 2:
                a = cronbach_alpha(df[valid])
                out[f"{scale_short}_{sub_name}"] = a
    return out


def disattenuate_subscale_matrix(corr: pd.DataFrame, alphas: dict) -> pd.DataFrame:
    """Unilateral disattenuation of a subscale-level correlation matrix.

    r_corrected(i,j) = r_observed(i,j) / sqrt(alpha_i * alpha_j)

    Only off-diagonals are modified; diagonal stays at 1.0.
    Out-of-bounds values (|r| > 1 after correction) are clipped to [-1, 1].
    """
    corrected = corr.copy().astype(float)
    cols = corr.columns.tolist()
    for i, si in enumerate(cols):
        for j, sj in enumerate(cols):
            if i == j:
                continue
            ai, aj = alphas.get(si, np.nan), alphas.get(sj, np.nan)
            if not (np.isfinite(ai) and np.isfinite(aj) and ai > 0 and aj > 0):
                corrected.iloc[i, j] = np.nan
                continue
            v = corr.iloc[i, j] / np.sqrt(ai * aj)
            corrected.iloc[i, j] = float(np.clip(v, -1.0, 1.0))
    return corrected


def offdiag_pairs(mat: pd.DataFrame, common_order: list):
    """Return upper-triangle off-diagonal vector aligned to common_order."""
    M = mat.loc[common_order, common_order].values
    n = len(common_order)
    idx = np.triu_indices(n, k=1)
    return M[idx]


def regress(h_vec, l_vec):
    """OLS slope, intercept, Pearson r between aligned vectors. NaN-safe."""
    valid = ~(np.isnan(h_vec) | np.isnan(l_vec))
    if valid.sum() < 3:
        return np.nan, np.nan, np.nan
    h, l = h_vec[valid], l_vec[valid]
    slope, intercept = np.polyfit(h, l, 1)
    r = np.corrcoef(h, l)[0, 1]
    return float(slope), float(intercept), float(r)


def run_one_cut(df_models_data: list, alphas: dict, allowed_subs: list, label: str):
    """Run unilateral disattenuation on a given subscale subset.

    df_models_data: list of (dirname, size, mtype, family, h_mat, l_mat) tuples.
    Returns DataFrame of per-model raw + corrected slope/r/intercept.
    """
    rows = []
    for dirname, size, mtype, family, h_mat, l_mat in df_models_data:
        common = [s for s in allowed_subs if s in h_mat.columns and s in l_mat.columns]
        if len(common) < 4:
            continue

        h_raw = offdiag_pairs(h_mat, common)
        l_raw = offdiag_pairs(l_mat, common)
        raw_slope, raw_int, raw_r = regress(h_raw, l_raw)

        h_cor_mat = disattenuate_subscale_matrix(h_mat.loc[common, common], alphas)
        h_cor = offdiag_pairs(h_cor_mat, common)
        cor_slope, cor_int, cor_r = regress(h_cor, l_raw)

        rows.append({
            "cut": label,
            "n_subscales": len(common),
            "n_pairs": int((~np.isnan(h_raw) & ~np.isnan(l_raw)).sum()),
            "dirname": dirname,
            "size": size,
            "type": mtype,
            "family": family,
            "raw_r": raw_r,
            "raw_slope": raw_slope,
            "raw_intercept": raw_int,
            "corrected_r": cor_r,
            "corrected_slope": cor_slope,
            "corrected_intercept": cor_int,
        })
    return pd.DataFrame(rows)


def print_cut_summary(df: pd.DataFrame, label: str):
    """Print compact per-model table for one cut."""
    print(f"\n  --- Cut: {label} ({df['n_subscales'].iloc[0]} subscales, "
          f"{df['n_pairs'].iloc[0]} pairs) ---")
    print(f"  {'Model':<22} {'raw_sl':>7} {'cor_sl':>7} {'delta':>7}")
    for _, r in df.iterrows():
        lab = f"{r['size']}B {r['type'][:4]} {r['family']}"
        d = r["corrected_slope"] - r["raw_slope"]
        print(f"  {lab:<22} {r['raw_slope']:>7.3f} {r['corrected_slope']:>7.3f} {d:>+7.3f}")
    big = df[df["size"] >= 7.0]["corrected_slope"].dropna()
    if len(big):
        print(f"  >=7B corrected slope range: [{big.min():.3f}, {big.max():.3f}]   "
              f"mean={big.mean():.3f}")


def plot_sensitivity(all_results: pd.DataFrame, outdir: Path):
    """N-panel plot: raw vs corrected slope for each cut."""
    cuts = all_results["cut"].unique().tolist()
    fig, axes = plt.subplots(1, len(cuts), figsize=(4 * len(cuts), 4.2),
                             sharey=True)
    if len(cuts) == 1:
        axes = [axes]

    for ax, cut_label in zip(axes, cuts):
        df = all_results[all_results["cut"] == cut_label]
        n_sub = int(df["n_subscales"].iloc[0])
        n_pairs = int(df["n_pairs"].iloc[0])

        for family, color in [("Llama", "#c0392b"), ("Qwen", "#16a085")]:
            for mtype, marker in [("base", "s"), ("instruct", "o")]:
                seg = df[(df["family"] == family) & (df["type"] == mtype)] \
                    .sort_values("size")
                if not len(seg):
                    continue
                ls = "--" if mtype == "instruct" else "-"
                lk = mtype.capitalize()
                ax.plot(seg["size"], seg["raw_slope"], color=color, marker=marker,
                        ms=7, ls=ls, alpha=0.95,
                        label=f"{family} {lk} (raw)")
                ax.plot(seg["size"], seg["corrected_slope"], color=color,
                        marker=marker, ms=7, mfc="white", ls=ls, alpha=0.95,
                        label=f"{family} {lk} (corr)")
        ax.axhline(1.0, color="grey", lw=0.8, ls=":")
        ax.set_xscale("log")
        ax.set_xlabel("Model size (B)")
        ax.set_title(f"{cut_label}\n({n_sub} subs, {n_pairs} pairs)",
                     fontsize=10)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("OLS slope")

    # Shared legend
    h, l = axes[0].get_legend_handles_labels()
    seen = set(); hh, ll = [], []
    for hi, li in zip(h, l):
        if li not in seen:
            seen.add(li); hh.append(hi); ll.append(li)
    fig.legend(hh, ll, loc="lower center", ncol=4, fontsize=7,
               bbox_to_anchor=(0.5, -0.05), frameon=False)
    plt.suptitle("Subscale-level attenuation correction sensitivity to alpha threshold\n"
                 "Filled = raw slope; open = corrected slope (unilateral, human-side disattenuated)",
                 y=1.02, fontsize=10)
    plt.tight_layout()

    pdf_path = outdir / "figbeh_attenuation_subscale_sensitivity.pdf"
    png_path = outdir / "figbeh_attenuation_subscale_sensitivity.png"
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved: {pdf_path} (and .png)")


def main():
    print("=" * 70)
    print("  SUBSCALE-LEVEL ATTENUATION CORRECTION (sensitivity sweep)")
    print("=" * 70)

    # 1. Compute alphas
    print("\nComputing subscale-level Cronbach alphas from human data...")
    alphas = compute_subscale_alphas()
    if not alphas:
        print("ERROR: No alphas computed. Check data/human/ paths.")
        return
    print(f"  Got alphas for {len(alphas)} subscales:")
    for s in ALL_SUBSCALES:
        a = alphas.get(s, np.nan)
        scale_short = s.split("_")[0]
        sub_name = "_".join(s.split("_")[1:])
        n_items = len(SCORING_RULES.get(scale_short, {})
                      .get("subscales", {}).get(sub_name, []))
        print(f"    {s:<35} alpha = {a:.3f}   (k = {n_items} items)")

    # 2. Pre-load all model data once
    print(f"\nLoading per-model matrices...")
    df_models_data = []
    for dirname, size, mtype, family in MODELS:
        h_path = BEHAVIOR_OUTPUT_DIR / dirname / "subscale_human.csv"
        l_path = BEHAVIOR_OUTPUT_DIR / dirname / "subscale_llm_implicit.csv"
        if not h_path.exists() or not l_path.exists():
            print(f"  SKIP {dirname}: missing CSV")
            continue
        h_mat = pd.read_csv(h_path, index_col=0)
        l_mat = pd.read_csv(l_path, index_col=0)
        df_models_data.append((dirname, size, mtype, family, h_mat, l_mat))
    print(f"  Loaded {len(df_models_data)} models")

    # 3. Define cuts: all 16 + alpha-threshold filtered
    cuts = [
        ("All 16 subscales", [s for s in ALL_SUBSCALES if s in alphas]),
        ("alpha >= 0.60", [s for s in ALL_SUBSCALES if alphas.get(s, 0) >= 0.60]),
        ("alpha >= 0.70", [s for s in ALL_SUBSCALES if alphas.get(s, 0) >= 0.70]),
        ("alpha >= 0.80", [s for s in ALL_SUBSCALES if alphas.get(s, 0) >= 0.80]),
    ]
    for label, allowed in cuts:
        excluded = [s for s in ALL_SUBSCALES if s not in allowed and s in alphas]
        if excluded:
            print(f"\n  Cut '{label}' excludes {len(excluded)}: {', '.join(excluded)}")

    # 4. Run each cut
    print(f"\n{'=' * 70}")
    print(f"  RESULTS PER CUT")
    print(f"{'=' * 70}")

    all_dfs = []
    for label, allowed in cuts:
        df = run_one_cut(df_models_data, alphas, allowed, label)
        if not len(df):
            continue
        all_dfs.append(df)
        print_cut_summary(df, label)

    if not all_dfs:
        print("\nNothing to plot.")
        return

    all_results = pd.concat(all_dfs, ignore_index=True)
    out_csv = BEHAVIOR_OUTPUT_DIR / "attenuation_correction_subscale_sensitivity.csv"
    all_results.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")

    # 5. Plot
    figdir = FIGURE_DIR
    figdir.mkdir(parents=True, exist_ok=True)
    plot_sensitivity(all_results, figdir)

    # 6. Final summary
    print(f"\n{'=' * 70}")
    print(f"  FINAL SUMMARY: >=7B corrected slope range across cuts")
    print(f"{'=' * 70}")
    print(f"  {'Cut':<22} {'n_sub':>6} {'min':>6} {'max':>6} {'mean':>6}")
    for label, _ in cuts:
        df = all_results[all_results["cut"] == label]
        big = df[df["size"] >= 7.0]["corrected_slope"].dropna()
        if not len(big):
            continue
        n_sub = int(df["n_subscales"].iloc[0])
        print(f"  {label:<22} {n_sub:>6d} {big.min():>6.3f} {big.max():>6.3f} {big.mean():>6.3f}")

    print(f"\n  Paper section 3.3 claim (currently scale-level): 1.18-1.29")
    print(f"\nDone.")


if __name__ == "__main__":
    main()
