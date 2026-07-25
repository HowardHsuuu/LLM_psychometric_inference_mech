#!/usr/bin/env python3
"""
Generate all 5 paper figures from final data.

Fig 1: Behavioral scaling (alignment r + slope, 14 models)
Fig 2: Representational scaling (Mantel r + steering DC Mantel r, 14 models)
Fig 3: Amplification (scatter + 5-model scaling curve)
Fig 4: Structure-behavior link (14-model scatter)
Fig 5: Synthetic pipeline control (3 panels: SNR curve, adjustment bars, adjusted scatter)

Usage:
    cd /path/to/Psychometric Inference_mech
    python scripts/figures/make_main_figures.py --outdir reports/figures/
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import pearsonr

from psychometric_inference.paths import (
    BEHAVIOR_OUTPUT_DIR,
    FIGURE_DIR,
    MECHANISTIC_OUTPUT_DIR,
    ROBUSTNESS_OUTPUT_DIR,
)

FAMILY_COLORS = {"Llama": "#c0392b", "Qwen": "#2471a3"}
TYPE_MARKERS = {"instruct": "o", "base": "s"}
TYPE_LS = {"instruct": "-", "base": "--"}
TYPE_ALPHA = {"instruct": 1.0, "base": 0.5}

# ── Data loading ──

from psychometric_inference.model_registry import behavioral_figure_map

BEHAVIORAL_MAP = behavioral_figure_map()
MECH_ROOT = MECHANISTIC_OUTPUT_DIR


def load_behavioral():
    rows = []
    for mech_name, size, family, mtype, beh_name in BEHAVIORAL_MAP:
        csv = BEHAVIOR_OUTPUT_DIR / beh_name / "per_scale_alignment.csv"
        if not csv.exists():
            continue
        df = pd.read_csv(csv)
        rows.append({
            "model": mech_name, "size": size, "family": family, "mtype": mtype,
            "beh_slope": df["subscale_slope"].mean(),
            "beh_r": df["subscale_r"].mean(),
        })
    return pd.DataFrame(rows)


def load_scaling():
    path = MECH_ROOT / "scaling_results_full.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def load_structure_behavior():
    path = MECH_ROOT / "structure_vs_behavior.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def load_a1_full():
    """Raw A1_full runs (synthetic SNR sweep)."""
    path = ROBUSTNESS_OUTPUT_DIR / "A1_full" / "A1_full_all_runs.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def load_a1_adj():
    """Per-model pipeline-baseline-adjusted slopes."""
    path = ROBUSTNESS_OUTPUT_DIR / "A1_adj" / "A1_adj_per_model.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def make_legend_handles():
    handles = []
    for family in ["Llama", "Qwen"]:
        for mtype in ["instruct", "base"]:
            handles.append(Line2D([0], [0], color=FAMILY_COLORS[family],
                                  marker=TYPE_MARKERS[mtype], ls=TYPE_LS[mtype],
                                  alpha=TYPE_ALPHA[mtype], ms=8, lw=2,
                                  label=f"{family} {mtype}"))
    return handles


# ═══════════════════════════════════════════════════════
#  FIGURE 1: Behavioral Scaling
# ═══════════════════════════════════════════════════════

def make_fig1(outdir):
    beh = load_behavioral()
    if beh.empty:
        print("Fig 1: No data"); return

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    for ax, (metric, ylabel, refline) in zip(axes, [
        ("beh_r", "Subscale alignment $r$", None),
        ("beh_slope", "Subscale slope", 1.0),
    ]):
        for family in ["Llama", "Qwen"]:
            for mtype in ["instruct", "base"]:
                sub = beh[(beh["family"] == family) & (beh["mtype"] == mtype)].sort_values("size")
                if sub.empty: continue
                ax.plot(sub["size"], sub[metric],
                       color=FAMILY_COLORS[family], marker=TYPE_MARKERS[mtype],
                       ls=TYPE_LS[mtype], alpha=TYPE_ALPHA[mtype], ms=9, lw=2)
        if refline is not None:
            ax.axhline(refline, ls=":", color="gray", alpha=0.5, lw=1)
        ax.axhline(0, ls=":", color="gray", alpha=0.2, lw=0.5)
        ax.set_xscale("log")
        ax.set_xticks([0.5, 1, 3, 7, 8, 14])
        ax.set_xticklabels(["0.5", "1", "3", "7", "8", "14"])
        ax.xaxis.set_major_formatter(matplotlib.ticker.FixedFormatter(["0.5", "1", "3", "7", "8", "14"]))
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_xlabel("Parameters (B)", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.grid(alpha=0.15)

    axes[1].legend(handles=make_legend_handles(), fontsize=8, loc="upper left")
    axes[0].set_title("(A) Alignment with human structure", fontsize=11)
    axes[1].set_title("(B) Amplification (slope)", fontsize=11)

    plt.tight_layout()
    plt.savefig(outdir / "fig1_behavioral_scaling.png", dpi=300, bbox_inches="tight")
    plt.savefig(outdir / "fig1_behavioral_scaling.pdf", bbox_inches="tight")
    plt.close()
    print("Fig 1 saved")


# ═══════════════════════════════════════════════════════
#  FIGURE 2: Representational Scaling
# ═══════════════════════════════════════════════════════

def make_fig2(outdir):
    sc = load_scaling()
    if sc.empty:
        print("Fig 2: No data"); return

    # Single panel: geometry Mantel r across 14 models.
    # Steering panel moved to App J (non-monotonic magnitude distracts from geometry story).
    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    metric = "geom_mantel_r"
    ylabel = "Representational Mantel $r$"

    if metric not in sc.columns:
        ax.text(0.5, 0.5, f"{metric} not available", ha="center", va="center",
                transform=ax.transAxes)
        plt.tight_layout()
        plt.savefig(outdir / "fig2_representational_scaling.png", dpi=300, bbox_inches="tight")
        plt.close()
        return

    for family in ["llama", "qwen"]:
        for mtype in ["instruct", "base"]:
            sub = sc[(sc["family"] == family) & (sc["mtype"] == mtype)].sort_values("size")
            if sub.empty or sub[metric].isna().all(): continue
            fam = family.capitalize()
            ax.plot(sub["size"], sub[metric],
                   color=FAMILY_COLORS[fam], marker=TYPE_MARKERS[mtype],
                   ls=TYPE_LS[mtype], alpha=TYPE_ALPHA[mtype], ms=10, lw=2)
    ax.axhline(0, ls=":", color="gray", alpha=0.3, lw=0.8)
    ax.set_xscale("log")
    ax.set_xticks([0.5, 1, 3, 7, 8, 14])
    ax.set_xticklabels(["0.5", "1", "3", "7", "8", "14"])
    ax.xaxis.set_major_formatter(matplotlib.ticker.FixedFormatter(["0.5", "1", "3", "7", "8", "14"]))
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("Parameters (B)", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(alpha=0.15)
    ax.legend(handles=make_legend_handles(), fontsize=9, loc="upper left")

    plt.tight_layout()
    plt.savefig(outdir / "fig2_representational_scaling.png", dpi=300, bbox_inches="tight")
    plt.savefig(outdir / "fig2_representational_scaling.pdf", bbox_inches="tight")
    plt.close()
    print("Fig 2 saved")


# ═══════════════════════════════════════════════════════
#  FIGURE 3: Amplification (scatter + scaling)
# ═══════════════════════════════════════════════════════

def make_fig3(outdir):
    """Left: Llama 8B scatter. Right: 9-model repr vs beh slope scaling."""
    import json

    # Cross-persona item-level results (hardcoded from final runs)
    cp_results = pd.DataFrame([
        {"model": "Qwen 0.5B Inst", "size": 0.5, "family": "Qwen", "mtype": "instruct",
         "repr_slope": 1.470, "beh_slope": 0.53, "mantel_r": 0.734},
        {"model": "Llama 1B Inst", "size": 1.0, "family": "Llama", "mtype": "instruct",
         "repr_slope": 1.440, "beh_slope": 0.53, "mantel_r": 0.813},
        {"model": "Qwen 3B Inst", "size": 3.0, "family": "Qwen", "mtype": "instruct",
         "repr_slope": 1.384, "beh_slope": 0.64, "mantel_r": 0.808},
        {"model": "Llama 3B Inst", "size": 3.0, "family": "Llama", "mtype": "instruct",
         "repr_slope": 1.628, "beh_slope": 0.83, "mantel_r": 0.839},
        {"model": "Qwen 7B Inst", "size": 7.0, "family": "Qwen", "mtype": "instruct",
         "repr_slope": 1.653, "beh_slope": 1.25, "mantel_r": 0.817},
        {"model": "Llama 8B Inst", "size": 8.0, "family": "Llama", "mtype": "instruct",
         "repr_slope": 1.697, "beh_slope": 1.19, "mantel_r": 0.827},
        {"model": "Llama 8B Base", "size": 8.0, "family": "Llama", "mtype": "base",
         "repr_slope": 1.634, "beh_slope": 1.00, "mantel_r": 0.822},
        {"model": "Qwen 14B Inst", "size": 14.0, "family": "Qwen", "mtype": "instruct",
         "repr_slope": 1.769, "beh_slope": 1.53, "mantel_r": 0.840},
        {"model": "Qwen 14B Base", "size": 14.0, "family": "Qwen", "mtype": "base",
         "repr_slope": 1.698, "beh_slope": 1.28, "mantel_r": 0.818},
    ])

    # Try to load Llama 8B Instruct item-level predicted correlation matrix for scatter
    pred_corr_path = MECH_ROOT / "results_llama8b_instruct" / "cross_persona" / "itemlevel_pred_corr_matrix.csv"
    print(f"  Looking for item-level matrix: {pred_corr_path}")
    print(f"  Exists: {pred_corr_path.exists()}")
    if not pred_corr_path.exists():
        # Fallback to ridge-to-subscale
        pred_corr_path = MECH_ROOT / "results_llama8b_instruct" / "cross_persona" / "predicted_corr_matrix_L16.csv"
        print(f"  Fallback to: {pred_corr_path}")
        print(f"  Exists: {pred_corr_path.exists()}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    # Panel A: Scatter (predicted vs human correlations)
    ax = axes[0]
    if pred_corr_path.exists():
        pred_corr = pd.read_csv(pred_corr_path, index_col=0)

        # Load human correlation matrix
        from psychometric_inference.mechanisms.geometry import compute_human_correlation_matrix
        human_corr = compute_human_correlation_matrix("subscale")

        common = [s for s in pred_corr.index if s in human_corr.index]
        n = len(common)
        idx = np.triu_indices(n, k=1)
        pv = pred_corr.loc[common, common].values[idx]
        hv = human_corr.loc[common, common].values[idx]
        valid = ~(np.isnan(pv) | np.isnan(hv))
        pv_c, hv_c = pv[valid], hv[valid]

        slope, intercept = np.polyfit(hv_c, pv_c, 1)
        r_val, _ = pearsonr(hv_c, pv_c)

        ax.scatter(hv_c, pv_c, s=30, alpha=0.5, color=FAMILY_COLORS["Llama"],
                  edgecolors="white", linewidth=0.5, zorder=3)
        xx = np.linspace(hv_c.min() - 0.05, hv_c.max() + 0.05, 100)
        ax.plot(xx, slope * xx + intercept, "k-", lw=1.5, label=f"slope = {slope:.2f}")
        ax.plot([-1, 1], [-1, 1], color="gray", ls=":", lw=0.8, alpha=0.5)
        ax.set_xlabel("Human correlation", fontsize=11)
        ax.set_ylabel("Activation-predicted correlation", fontsize=11)
        ax.set_title(f"(A) Llama 8B Instruct\nMantel $r$ = {r_val:.2f}, slope = {slope:.2f}", fontsize=11)
        ax.legend(fontsize=9, loc="upper left")
        ax.grid(alpha=0.15)
    else:
        ax.text(0.5, 0.5, "Predicted correlation\nmatrix not found",
               ha="center", va="center", transform=ax.transAxes)

    # Panel B: Repr vs Beh slope across 5 models
    ax = axes[1]

    # Plot repr slope (solid markers) and beh slope (open markers)
    instruct = cp_results[cp_results["mtype"] == "instruct"].sort_values("size")
    base = cp_results[cp_results["mtype"] == "base"]

    # Instruct representational line
    ax.plot(instruct["size"], instruct["repr_slope"], "o-",
           color="#2c3e50", ms=10, lw=2.5, label="Representational (instruct)", zorder=4)
    # Instruct behavioral line
    ax.plot(instruct["size"], instruct["beh_slope"], "s--",
           color="#e74c3c", ms=10, lw=2.5, label="Behavioral (instruct)", zorder=4)

    # Base model points
    if not base.empty:
        ax.scatter(base["size"], base["repr_slope"], marker="^", s=120,
                  color="#2c3e50", alpha=0.5, edgecolors="white", linewidth=1.5,
                  label="Representational (base)", zorder=5)
        ax.scatter(base["size"], base["beh_slope"], marker="v", s=120,
                  color="#e74c3c", alpha=0.5, edgecolors="white", linewidth=1.5,
                  label="Behavioral (base)", zorder=5)

    # Fill between for readout attenuation (instruct only)
    ax.fill_between(instruct["size"], instruct["repr_slope"], instruct["beh_slope"],
                   alpha=0.12, color="#95a5a6", label="Readout attenuation")

    ax.axhline(1.0, ls=":", color="gray", alpha=0.5, lw=1)
    ax.set_xscale("log")
    ax.set_xticks([0.5, 1, 3, 7, 8, 14])
    ax.set_xticklabels(["0.5", "1", "3", "7", "8", "14"])
    ax.xaxis.set_major_formatter(matplotlib.ticker.FixedFormatter(["0.5", "1", "3", "7", "8", "14"]))
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("Parameters (B)", fontsize=11)
    ax.set_ylabel("Slope (vs human correlation)", fontsize=11)
    ax.set_title("(B) Representational vs behavioral amplification", fontsize=11)
    ax.legend(fontsize=8, loc="lower right")
    ax.set_ylim(0.3, 2.0)
    ax.grid(alpha=0.15)

    # Annotate gap values
    for _, row in instruct.iterrows():
        gap = row["repr_slope"] - row["beh_slope"]
        mid_y = (row["repr_slope"] + row["beh_slope"]) / 2
        ax.annotate(f"+{gap:.2f}", (row["size"], mid_y),
                   fontsize=7, ha="left", va="center", color="#7f8c8d",
                   xytext=(8, 0), textcoords="offset points")

    plt.tight_layout()
    plt.savefig(outdir / "fig3_amplification.png", dpi=300, bbox_inches="tight")
    plt.savefig(outdir / "fig3_amplification.pdf", bbox_inches="tight")
    plt.close()
    print("Fig 3 saved")


# ═══════════════════════════════════════════════════════
#  FIGURE 4: Structure-Behavior Link
# ═══════════════════════════════════════════════════════

def make_fig4(outdir):
    sb = load_structure_behavior()
    if sb.empty:
        print("Fig 4: No data"); return

    fig, ax = plt.subplots(figsize=(6.5, 5.5))

    for _, row in sb.iterrows():
        fam = row["family"].capitalize() if isinstance(row["family"], str) else row["family"]
        mt = row["mtype"]
        color = FAMILY_COLORS.get(fam, "gray")
        marker = TYPE_MARKERS.get(mt, "o")
        alpha = TYPE_ALPHA.get(mt, 0.5)
        ax.scatter(row["repr_mantel_r"], row["beh_subscale_slope"],
                  c=color, marker=marker, s=130, alpha=alpha,
                  edgecolors="white", linewidth=1.5, zorder=3)
        # Label with size
        size = row.get("size", "")
        ax.annotate(f"{size}B",
                   (row["repr_mantel_r"], row["beh_subscale_slope"]),
                   fontsize=7, alpha=0.7, xytext=(6, 4), textcoords="offset points")

    # Correlation line
    xv = sb["repr_mantel_r"].values
    yv = sb["beh_subscale_slope"].values
    valid = ~(np.isnan(xv) | np.isnan(yv))
    if valid.sum() >= 3:
        r, p = pearsonr(xv[valid], yv[valid])
        sl, ic = np.polyfit(xv[valid], yv[valid], 1)
        xx = np.linspace(xv[valid].min() - 0.05, xv[valid].max() + 0.05, 100)
        ax.plot(xx, sl * xx + ic, "k--", lw=1, alpha=0.5)
        ax.text(0.05, 0.95, f"$r$ = {r:.2f}, $p$ = {p:.3f}",
               transform=ax.transAxes, fontsize=11, va="top",
               bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    ax.axhline(1.0, ls=":", color="gray", alpha=0.5)
    ax.axvline(0, ls=":", color="gray", alpha=0.3)
    ax.set_xlabel("Representational Mantel $r$", fontsize=12)
    ax.set_ylabel("Behavioral subscale slope", fontsize=12)
    ax.grid(alpha=0.15)

    handles = []
    for family in ["Llama", "Qwen"]:
        for mtype in ["instruct", "base"]:
            handles.append(Line2D([0], [0], color=FAMILY_COLORS[family],
                                  marker=TYPE_MARKERS[mtype], ls="none",
                                  alpha=TYPE_ALPHA[mtype], ms=10,
                                  markeredgecolor="white", markeredgewidth=1,
                                  label=f"{family} {mtype}"))
    ax.legend(handles=handles, fontsize=9, loc="lower right")

    plt.tight_layout()
    plt.savefig(outdir / "fig4_structure_behavior.png", dpi=300, bbox_inches="tight")
    plt.savefig(outdir / "fig4_structure_behavior.pdf", bbox_inches="tight")
    plt.close()
    print("Fig 4 saved")


# ═══════════════════════════════════════════════════════
#  FIGURE 5: Synthetic Pipeline Control
# ═══════════════════════════════════════════════════════
#
#  Three panels:
#    A) A1_full synthetic SNR curve (4 hidden dims) + real models overlaid
#       at their estimated SNR
#    B) Per-model bars: observed / baseline / adjusted slope
#    C) Adjusted slope vs behavioral slope scatter
#
#  Data sources:
#    outputs/robustness/A1_full/A1_full_all_runs.csv
#    outputs/robustness/A1_adj/A1_adj_per_model.csv

def make_fig5(outdir):
    a1 = load_a1_full()
    adj = load_a1_adj()

    if a1.empty or adj.empty:
        print("Fig 5: need both A1_full and A1_adj outputs; check outputs/robustness/")
        return

    # Aggregate A1_full to (hdim, snr) means
    a1_grid = a1.groupby(["hidden_dim", "snr"]).agg(
        slope_mean=("recovered_slope_vs_human", "mean"),
        slope_se=("recovered_slope_vs_human",
                  lambda x: float(x.std() / np.sqrt(len(x)))),
    ).reset_index()

    # Two-row layout: top = SNR curve (full width), bottom = bars + scatter side by side.
    # The page renders the figure at one column width, so squeezing 3 panels
    # horizontally makes each tiny; stacking gives each panel enough room.
    fig = plt.figure(figsize=(11, 9))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0], hspace=0.35, wspace=0.30)
    ax_top = fig.add_subplot(gs[0, :])     # Panel A spans top row
    ax_bl = fig.add_subplot(gs[1, 0])      # Panel B bottom-left
    ax_br = fig.add_subplot(gs[1, 1])      # Panel C bottom-right

    # ── Panel A: SNR curve ──
    ax = ax_top
    hdim_colors = {896: "#8c564b", 2048: "#9467bd", 4096: "#2ca02c", 5120: "#1f77b4"}
    for hdim in sorted(a1_grid["hidden_dim"].unique()):
        sub = a1_grid[a1_grid["hidden_dim"] == hdim].sort_values("snr")
        ax.errorbar(sub["snr"], sub["slope_mean"], yerr=sub["slope_se"],
                    marker="o", ms=6, lw=1.8, capsize=3,
                    color=hdim_colors.get(int(hdim), "gray"),
                    label=f"hdim={int(hdim)}")
    ax.axhline(1.0, ls=":", color="green", alpha=0.6, lw=1.2,
               label="unbiased (slope=1)")

    # Paper observed range band
    ax.axhspan(1.38, 1.77, alpha=0.10, color="red", label="paper 9-model range")

    # Overlay real models
    for _, r in adj.iterrows():
        mt = r["mtype"]; fam = r["family"]
        fam_cap = fam.capitalize() if isinstance(fam, str) else "Qwen"
        color = FAMILY_COLORS.get(fam_cap, "gray")
        marker = TYPE_MARKERS.get(mt, "o")
        ax.scatter(r["snr_proxy"], r["observed_slope_itemlevel"],
                   color=color, marker=marker, s=110,
                   edgecolors="black", linewidth=1.0,
                   alpha=0.95, zorder=10)

    ax.set_xscale("log")
    ax.set_xlabel("SNR (signal var / noise var)", fontsize=12)
    ax.set_ylabel("Recovered slope (vs ground truth)", fontsize=11)
    ax.set_title("(A) Synthetic pipeline SNR curve (real models overlaid)",
                 fontsize=12)
    # Two-column legend at top to avoid overlap with low-SNR bump
    ax.legend(fontsize=9, loc="upper right", ncol=2)
    ax.grid(alpha=0.15)

    # ── Panel B: Per-model observed / baseline / adjusted bars ──
    ax = ax_bl
    sorted_adj = adj.sort_values("size").reset_index(drop=True)
    x = np.arange(len(sorted_adj))
    w = 0.27
    ax.bar(x - w, sorted_adj["observed_slope_itemlevel"], w,
           label="observed", color="#1f77b4", edgecolor="white", linewidth=0.5)
    ax.bar(x, sorted_adj["pipeline_baseline_slope"], w,
           label="pipeline baseline", color="#95a5a6",
           edgecolor="white", linewidth=0.5)
    ax.bar(x + w, sorted_adj["adjusted_slope"], w,
           label="adjusted (obs $-$ base)",
           color="#d62728", edgecolor="white", linewidth=0.5)

    ax.axhline(0, color="black", linewidth=0.5)
    ax.axhline(1.0, ls=":", color="green", alpha=0.5, lw=1)
    ax.set_xticks(x)

    def _label(r):
        # Very short labels to fit under bars at one-column width
        short = {"qwen05b_instruct": "Q.5I",
                 "llama1b_instruct": "L1I",
                 "qwen3b_instruct":  "Q3I",
                 "llama3b_instruct": "L3I",
                 "qwen7b_instruct":  "Q7I",
                 "llama8b_instruct": "L8I",
                 "llama8b_base":     "L8B",
                 "qwen14b_instruct": "Q14I",
                 "qwen14b_base":     "Q14B"}
        return short.get(r["model"], r["model"])

    ax.set_xticklabels([_label(r) for _, r in sorted_adj.iterrows()],
                       fontsize=9)
    ax.set_ylabel("Slope", fontsize=11)
    ax.set_title("(B) Observed vs baseline vs adjusted per model", fontsize=11)
    ax.legend(fontsize=8, loc="upper left", framealpha=0.9)
    ax.grid(alpha=0.15, axis="y")

    # ── Panel C: Adjusted slope vs behavioral slope ──
    ax = ax_br
    adj_valid = adj.dropna(subset=["beh_subscale_slope"])
    if len(adj_valid) >= 3:
        for _, r in adj_valid.iterrows():
            fam = r["family"].capitalize() if isinstance(r["family"], str) else "Qwen"
            mt = r["mtype"]
            color = FAMILY_COLORS.get(fam, "gray")
            marker = TYPE_MARKERS.get(mt, "o")
            alpha = TYPE_ALPHA.get(mt, 0.5)
            ax.scatter(r["adjusted_slope"], r["beh_subscale_slope"],
                       c=color, marker=marker, s=140, alpha=alpha,
                       edgecolors="white", linewidth=1.5, zorder=3)
            size_lbl = f"{r['size']:g}B"
            ax.annotate(size_lbl, (r["adjusted_slope"], r["beh_subscale_slope"]),
                        fontsize=8, xytext=(6, 4), textcoords="offset points",
                        alpha=0.75)

        xv = adj_valid["adjusted_slope"].values
        yv = adj_valid["beh_subscale_slope"].values
        valid = ~(np.isnan(xv) | np.isnan(yv))
        if valid.sum() >= 3:
            r_adj, p_adj = pearsonr(xv[valid], yv[valid])
            r_obs, p_obs = pearsonr(
                adj_valid["observed_slope_itemlevel"][valid].values, yv[valid]
            )
            sl, ic = np.polyfit(xv[valid], yv[valid], 1)
            xx = np.linspace(xv[valid].min() - 0.05, xv[valid].max() + 0.05, 100)
            ax.plot(xx, sl * xx + ic, "k--", lw=1, alpha=0.5)
            txt = (f"adjusted vs beh:\n"
                   f"  $r$ = {r_adj:.2f},  $p$ = {p_adj:.3f}\n"
                   f"(observed vs beh:  $r$ = {r_obs:.2f})")
            ax.text(0.05, 0.95, txt, transform=ax.transAxes,
                    fontsize=9, verticalalignment="top",
                    bbox=dict(boxstyle="round,pad=0.3",
                              facecolor="white", alpha=0.85))

    ax.axvline(0, color="gray", ls=":", alpha=0.3)
    ax.set_xlabel("Adjusted representational slope\n(observed $-$ pipeline baseline)",
                  fontsize=11)
    ax.set_ylabel("Behavioral subscale slope", fontsize=11)
    ax.set_title("(C) Adjusted slope vs behavior", fontsize=11)
    ax.grid(alpha=0.15)

    handles = []
    for family in ["Llama", "Qwen"]:
        for mtype in ["instruct", "base"]:
            handles.append(Line2D([0], [0], color=FAMILY_COLORS[family],
                                   marker=TYPE_MARKERS[mtype], ls="none",
                                   alpha=TYPE_ALPHA[mtype], ms=9,
                                   markeredgecolor="white", markeredgewidth=1,
                                   label=f"{family} {mtype}"))
    ax.legend(handles=handles, fontsize=8, loc="lower right", framealpha=0.9)

    plt.savefig(outdir / "fig5_pipeline_control.png", dpi=300, bbox_inches="tight")
    plt.savefig(outdir / "fig5_pipeline_control.pdf", bbox_inches="tight")
    plt.close()
    print("Fig 5 saved")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=str, default=str(FIGURE_DIR))
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    make_fig1(outdir)
    make_fig2(outdir)
    make_fig3(outdir)
    make_fig4(outdir)
    make_fig5(outdir)
    print(f"\nAll figures saved to {outdir}/")


if __name__ == "__main__":
    main()
