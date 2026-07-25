#!/usr/bin/env python3
"""
Paper-quality figures for LLM psychometric structure scaling.

Figure 1: 2x3 — Scaling curves (Qwen top, Llama bottom)
          with bootstrap CI shading, base + instruct
Figure 2: 2x2 — Slope (amplification), raw + corrected  
Figure 3: 1x2 — Base vs instruct scatter

Usage:
    python plot_figures.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR / "src"))

from psychometric_inference.model_registry import FAMILY_PLOT_META, SIZE_TO_LABEL, analysis_model_tuples

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "lines.linewidth": 1.3,
    "lines.markersize": 5,
    "axes.grid": True,
    "grid.alpha": 0.15,
    "grid.linewidth": 0.4,
})

FAMILY_META = FAMILY_PLOT_META

TYPE_STYLE = {
    "base":     {"ls": "-",  "marker": "o", "ms": 5.5, "lw": 1.5},
    "instruct": {"ls": "--", "marker": "s", "ms": 4.5, "lw": 1.1},
}

MODELS = analysis_model_tuples()


def load_all_metrics():
    rows = []
    for dirname, size, mtype, family in MODELS:
        fpath = BASE_DIR / "outputs" / "behavior" / dirname / "comparison_metrics.csv"
        if not fpath.exists():
            continue
        df = pd.read_csv(fpath)
        entry = {"dirname": dirname, "size": size, "type": mtype, "family": family}
        for _, r in df.iterrows():
            level = r["level"]
            entry[f"{level}_r"] = r["matrix_r"] if pd.notna(r.get("matrix_r")) else r.get("r", np.nan)
            entry[f"{level}_n"] = r.get("n_pairs", 0)
            entry[f"{level}_p"] = r.get("mantel_p", np.nan)
            if "slope" in r and pd.notna(r.get("slope")):
                entry[f"{level}_slope"] = r["slope"]
        rows.append(entry)
    return pd.DataFrame(rows)


def load_bootstrap_ci():
    fpath = BASE_DIR / "outputs" / "behavior" / "bootstrap_ci_subjects.csv"
    if fpath.exists():
        return pd.read_csv(fpath)
    return None


def load_attenuation():
    fpath = BASE_DIR / "outputs" / "behavior" / "attenuation_correction.csv"
    if fpath.exists():
        return pd.read_csv(fpath)
    return None


def _get_ci(boot_ci, model_label, level):
    """Get bootstrap CI for a model+level."""
    if boot_ci is None:
        return None, None
    match = boot_ci[(boot_ci["model"] == model_label) & (boot_ci["level"] == level)]
    if len(match) == 0:
        return None, None
    return match.iloc[0]["ci_lo"], match.iloc[0]["ci_hi"]


def _model_label(size, mtype, family):
    """Build label matching bootstrap CSV format."""
    fam_short = "Q" if family == "Qwen" else "L"
    type_short = "base" if mtype == "base" else "inst"
    # Format size: show decimal only when needed (0.5B, but 1B, 3B, 14B)
    if size == int(size):
        s = f"{int(size)}B"
    else:
        s = f"{size:.1f}B"
    return f"{s} {type_short} {fam_short}"


def _plot_family_panel(ax, df, family, col, boot_ci=None, boot_level=None):
    """Plot base + instruct for one family with optional CI shading."""
    meta = FAMILY_META[family]

    for mtype in ["base", "instruct"]:
        style = TYPE_STYLE[mtype]
        color = meta["color_base"] if mtype == "base" else meta["color_inst"]
        sub = df[(df["type"] == mtype) & (df["family"] == family)].sort_values("size")
        if col not in sub.columns:
            continue
        vals = sub[col].values.astype(float)
        sizes = sub["size"].values
        valid = ~np.isnan(vals)
        if not valid.any():
            continue

        ax.plot(sizes[valid], vals[valid],
                marker=style["marker"], color=color,
                ls=style["ls"], ms=style["ms"], lw=style["lw"],
                markeredgecolor="white", markeredgewidth=0.5,
                zorder=5 if mtype == "base" else 4)

        # Bootstrap CI shading
        if boot_ci is not None and boot_level is not None:
            ci_los, ci_his = [], []
            s_valid = []
            for s, v in zip(sizes, vals):
                if np.isnan(v):
                    continue
                label = _model_label(s, mtype, family)
                lo, hi = _get_ci(boot_ci, label, boot_level)
                if lo is not None:
                    ci_los.append(lo)
                    ci_his.append(hi)
                    s_valid.append(s)
            if ci_los:
                ax.fill_between(s_valid, ci_los, ci_his,
                               color=color, alpha=0.12, zorder=1)


def _setup_xaxis(ax, family):
    ax.set_xscale("log")
    sizes = FAMILY_META[family]["sizes"]
    ax.set_xticks(sizes)
    ax.set_xticklabels([SIZE_TO_LABEL[family][s] for s in sizes], fontsize=7)
    ax.xaxis.set_minor_formatter(ticker.NullFormatter())
    ax.set_xlim(min(sizes) * 0.7, max(sizes) * 1.4)


def fig1_scaling_curves(df, boot_ci, output_dir):
    """2x3: Qwen top, Llama bottom. item_between, subscale, scale."""
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.2),
                             gridspec_kw={"hspace": 0.5, "wspace": 0.35})

    col_specs = [
        ("item_between_r", "Item-level\n(between-scale)", "item_between"),
        ("subscale_between_r", "Subscale-level\n(between-scale)", "subscale"),
        ("scale_r", "Scale-level", "scale"),
    ]

    for row, family in enumerate(["Qwen", "Llama"]):
        for col_idx, (col, col_title, boot_level) in enumerate(col_specs):
            ax = axes[row, col_idx]
            _plot_family_panel(ax, df, family, col, boot_ci, boot_level)
            _setup_xaxis(ax, family)
            ax.set_ylim(-0.05, 1.05)
            ax.axhline(y=0, color="gray", ls=":", lw=0.5, alpha=0.3)

            if row == 0:
                ax.set_title(col_title, fontsize=8.5, pad=6)
            if col_idx == 0:
                ax.set_ylabel(f"{FAMILY_META[family]['label']}\n\nAlignment r")
            if row == 1:
                ax.set_xlabel("Parameters (B)")

    handles = [
        Line2D([0], [0], color="#555", ls="-", lw=1.5, marker="o", ms=5,
               markeredgecolor="white", markeredgewidth=0.5, label="Base"),
        Line2D([0], [0], color="#aaa", ls="--", lw=1.1, marker="s", ms=4,
               markeredgecolor="white", markeredgewidth=0.5, label="Instruct"),
    ]
    axes[0, 2].legend(handles=handles, loc="lower right", frameon=True,
                      framealpha=0.95, edgecolor="none", fontsize=7,
                      handlelength=2.5)

    for i, (row, col_idx) in enumerate([(0,0),(0,1),(0,2),(1,0),(1,1),(1,2)]):
        axes[row, col_idx].text(-0.08, 1.12, chr(65 + i),
                                transform=axes[row, col_idx].transAxes,
                                fontsize=11, fontweight="bold", va="top")

    fpath = output_dir / "fig1_scaling_curves.pdf"
    plt.savefig(fpath)
    plt.savefig(fpath.with_suffix(".png"), dpi=300)
    plt.close()
    print(f"Saved: {fpath}")


def fig2_amplification(df, att_df, output_dir):
    """2x2: Qwen top, Llama bottom. Raw slope + corrected slope overlay."""
    fig, axes = plt.subplots(2, 2, figsize=(5.0, 4.2),
                             gridspec_kw={"hspace": 0.5, "wspace": 0.4})

    col_specs = [
        ("item_between_slope", "Item-level\n(between-scale)"),
        ("subscale_between_slope", "Subscale-level\n(between-scale)"),
    ]

    for row, family in enumerate(["Qwen", "Llama"]):
        for col_idx, (col, col_title) in enumerate(col_specs):
            ax = axes[row, col_idx]

            # Plot raw slopes
            _plot_family_panel(ax, df, family, col)
            _setup_xaxis(ax, family)

            # Overlay corrected slope for base models (from attenuation_correction.csv)
            # Only for scale-level slopes which we computed
            # For now, scale-level raw vs corrected shown in separate figure

            ax.axhline(y=1.0, color="#333", ls="-", lw=0.6, alpha=0.4)
            ax.axhline(y=0, color="gray", ls=":", lw=0.5, alpha=0.3)
            ax.set_ylim(-0.05, 1.55)

            ax.text(0.97, 0.97, "amplification", transform=ax.transAxes,
                    fontsize=6, color="#999", ha="right", va="top", style="italic")
            ax.text(0.97, 0.03, "attenuation", transform=ax.transAxes,
                    fontsize=6, color="#999", ha="right", va="bottom", style="italic")

            if row == 0:
                ax.set_title(col_title, fontsize=8.5, pad=6)
            if col_idx == 0:
                ax.set_ylabel(f"{FAMILY_META[family]['label']}\n\nSlope")
            if row == 1:
                ax.set_xlabel("Parameters (B)")

    handles = [
        Line2D([0], [0], color="#555", ls="-", lw=1.5, marker="o", ms=5,
               markeredgecolor="white", markeredgewidth=0.5, label="Base"),
        Line2D([0], [0], color="#aaa", ls="--", lw=1.1, marker="s", ms=4,
               markeredgecolor="white", markeredgewidth=0.5, label="Instruct"),
    ]
    axes[0, 1].legend(handles=handles, loc="lower right", frameon=True,
                      framealpha=0.95, edgecolor="none", fontsize=7,
                      handlelength=2.5)

    for i, (r, c) in enumerate([(0,0),(0,1),(1,0),(1,1)]):
        axes[r, c].text(-0.14, 1.12, chr(65 + i), transform=axes[r, c].transAxes,
                        fontsize=11, fontweight="bold", va="top")

    fpath = output_dir / "fig2_amplification.pdf"
    plt.savefig(fpath)
    plt.savefig(fpath.with_suffix(".png"), dpi=300)
    plt.close()
    print(f"Saved: {fpath}")


def fig2b_attenuation_correction(att_df, output_dir):
    """2x2: Qwen top, Llama bottom. Scale-level slope (left) and r (right),
    raw vs corrected for both base and instruct."""
    if att_df is None:
        print("No attenuation data, skipping fig2b")
        return

    fig, axes = plt.subplots(2, 2, figsize=(5.0, 4.2),
                             gridspec_kw={"hspace": 0.5, "wspace": 0.4})

    col_specs = [
        ("raw_slope", "corrected_slope", "Scale-level slope"),
        ("raw_r", "corrected_r", "Scale-level r"),
    ]

    for row, family in enumerate(["Qwen", "Llama"]):
        meta = FAMILY_META[family]
        for col_idx, (raw_col, cor_col, col_title) in enumerate(col_specs):
            ax = axes[row, col_idx]

            for mtype in ["base", "instruct"]:
                style = TYPE_STYLE[mtype]
                color = meta["color_base"] if mtype == "base" else meta["color_inst"]
                sub = att_df[(att_df["type"] == mtype) & (att_df["family"] == family)].sort_values("size")
                if sub.empty:
                    continue

                # Raw
                ax.plot(sub["size"], sub[raw_col],
                        marker=style["marker"], color=color,
                        ls=style["ls"], ms=style["ms"], lw=style["lw"],
                        markeredgecolor="white", markeredgewidth=0.5, zorder=5)

                # Corrected (dotted, open markers)
                ax.plot(sub["size"], sub[cor_col],
                        marker=style["marker"], color=color,
                        ls=":", ms=style["ms"], lw=style["lw"] * 0.8,
                        markeredgecolor=color, markeredgewidth=0.8,
                        alpha=0.5, zorder=4, fillstyle="none")

            _setup_xaxis(ax, family)

            if "slope" in raw_col:
                ax.axhline(y=1.0, color="#333", ls="-", lw=0.6, alpha=0.4)
                ax.axhline(y=0, color="gray", ls=":", lw=0.5, alpha=0.3)
                ax.set_ylim(-0.05, 1.7)
                ax.text(0.97, 0.97, "amplification", transform=ax.transAxes,
                        fontsize=6, color="#999", ha="right", va="top", style="italic")
                ax.text(0.97, 0.03, "attenuation", transform=ax.transAxes,
                        fontsize=6, color="#999", ha="right", va="bottom", style="italic")
            else:
                ax.set_ylim(-0.05, 1.05)
                ax.axhline(y=0, color="gray", ls=":", lw=0.5, alpha=0.3)

            if row == 0:
                ax.set_title(col_title, fontsize=8.5, pad=6)
            if col_idx == 0:
                ax.set_ylabel(f"{meta['label']}\n\n{'Slope' if 'slope' in raw_col else 'Alignment r'}")
            if row == 1:
                ax.set_xlabel("Parameters (B)")

    # Legend: solid = raw, dotted/open = corrected; dark = base, light = instruct
    handles = [
        Line2D([0], [0], color="#555", ls="-", lw=1.5, marker="o", ms=5,
               markeredgecolor="white", markeredgewidth=0.5, label="Base (raw)"),
        Line2D([0], [0], color="#555", ls=":", lw=1.1, marker="o", ms=5,
               fillstyle="none", alpha=0.5, label="Base (corrected)"),
        Line2D([0], [0], color="#aaa", ls="--", lw=1.1, marker="s", ms=4,
               markeredgecolor="white", markeredgewidth=0.5, label="Instruct (raw)"),
        Line2D([0], [0], color="#aaa", ls=":", lw=0.9, marker="s", ms=4,
               fillstyle="none", alpha=0.5, label="Instruct (corrected)"),
    ]
    axes[0, 1].legend(handles=handles, loc="lower right", frameon=True,
                      framealpha=0.95, edgecolor="none", fontsize=6,
                      handlelength=2.5)

    for i, (r, c) in enumerate([(0,0),(0,1),(1,0),(1,1)]):
        axes[r, c].text(-0.14, 1.12, chr(65 + i), transform=axes[r, c].transAxes,
                        fontsize=11, fontweight="bold", va="top")

    fpath = output_dir / "fig2b_attenuation_correction.pdf"
    plt.savefig(fpath)
    plt.savefig(fpath.with_suffix(".png"), dpi=300)
    plt.close()
    print(f"Saved: {fpath}")


def fig3_base_vs_instruct(df, boot_ci, output_dir):
    """1x2: base vs instruct scatter with CI error bars."""
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.8))

    for ax, (col, title, boot_level) in zip(axes, [
        ("item_between_r", "Item-level (between-scale)", "item_between"),
        ("scale_r", "Scale-level", "scale"),
    ]):
        for family in ["Qwen", "Llama"]:
            meta = FAMILY_META[family]
            color = meta["color_base"]
            marker = "o" if family == "Qwen" else "D"
            bases = df[(df["type"] == "base") & (df["family"] == family)].set_index("size")
            insts = df[(df["type"] == "instruct") & (df["family"] == family)].set_index("size")
            common_sizes = sorted(set(bases.index) & set(insts.index))

            for s in common_sizes:
                bv = bases.loc[s, col] if col in bases.columns else np.nan
                iv = insts.loc[s, col] if col in insts.columns else np.nan
                if np.isnan(bv) or np.isnan(iv):
                    continue

                # Error bars from bootstrap
                xerr_lo, xerr_hi, yerr_lo, yerr_hi = 0, 0, 0, 0
                if boot_ci is not None and boot_level is not None:
                    b_label = _model_label(s, "base", family)
                    i_label = _model_label(s, "instruct", family)
                    blo, bhi = _get_ci(boot_ci, b_label, boot_level)
                    ilo, ihi = _get_ci(boot_ci, i_label, boot_level)
                    if blo is not None:
                        xerr_lo = max(0, bv - blo)
                        xerr_hi = max(0, bhi - bv)
                    if ilo is not None:
                        yerr_lo = max(0, iv - ilo)
                        yerr_hi = max(0, ihi - iv)

                if max(xerr_lo, xerr_hi, yerr_lo, yerr_hi) > 0:
                    ax.errorbar(bv, iv, xerr=[[xerr_lo], [xerr_hi]],
                               yerr=[[yerr_lo], [yerr_hi]],
                               fmt="none", ecolor=color, elinewidth=0.6,
                               capsize=2, capthick=0.6, alpha=0.4, zorder=3)

                ax.scatter(bv, iv, c=color, marker=marker, s=30, zorder=5,
                          edgecolors="white", linewidths=0.5)
                label = f"{int(s)}B" if s >= 1 else f"{s}B"
                ax.annotate(label, (bv, iv), fontsize=5.5, ha="left", va="bottom",
                           xytext=(3, 3), textcoords="offset points", color="#555")

        lo, hi = 0, 1.05
        ax.plot([lo, hi], [lo, hi], "k-", lw=0.5, alpha=0.3)
        ax.set_xlabel("Base model r")
        ax.set_title(title, fontsize=8.5, pad=6)
        ax.set_aspect("equal")

    axes[0].set_ylabel("Instruct model r")

    legend_handles = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=FAMILY_META["Qwen"]["color_base"],
               markersize=5.5, label="Qwen 2.5"),
        Line2D([0], [0], marker="D", color="w",
               markerfacecolor=FAMILY_META["Llama"]["color_base"],
               markersize=5, label="Llama 3"),
    ]
    axes[0].legend(handles=legend_handles, loc="upper left", frameon=True,
                   framealpha=0.95, edgecolor="none", fontsize=7)

    for i, ax in enumerate(axes):
        ax.text(-0.12, 1.08, chr(65 + i), transform=ax.transAxes,
                fontsize=11, fontweight="bold", va="top")

    plt.tight_layout(w_pad=1.5)
    fpath = output_dir / "fig3_base_vs_instruct.pdf"
    plt.savefig(fpath)
    plt.savefig(fpath.with_suffix(".png"), dpi=300)
    plt.close()
    print(f"Saved: {fpath}")


def main():
    output_dir = BASE_DIR / "reports" / "figures"
    output_dir.mkdir(exist_ok=True)

    print("Loading data...")
    df = load_all_metrics()
    boot_ci = load_bootstrap_ci()
    att_df = load_attenuation()
    print(f"  {len(df)} models, bootstrap={'yes' if boot_ci is not None else 'no'}, "
          f"attenuation={'yes' if att_df is not None else 'no'}")

    # Summary table
    print(f"\n{'Model':<25} {'it_btwn':>7} {'sub_btwn':>8} {'scale':>6} {'it_b_sl':>7} {'sb_b_sl':>7}")
    print("-" * 62)
    for _, r in df.sort_values(["family", "type", "size"]).iterrows():
        name = f"{r['size']}B {r['type'][:4]} {r['family']}"
        def fmt(c):
            v = r.get(c, np.nan)
            return f"{v:.3f}" if pd.notna(v) and not np.isnan(v) else "—"
        print(f"{name:<25} {fmt('item_between_r'):>7} {fmt('subscale_between_r'):>8} "
              f"{fmt('scale_r'):>6} {fmt('item_between_slope'):>7} {fmt('subscale_between_slope'):>7}")

    print("\nGenerating figures...")
    fig1_scaling_curves(df, boot_ci, output_dir)
    fig2_amplification(df, att_df, output_dir)
    fig2b_attenuation_correction(att_df, output_dir)
    fig3_base_vs_instruct(df, boot_ci, output_dir)
    print(f"\nAll figures saved to {output_dir}/")


if __name__ == "__main__":
    main()
