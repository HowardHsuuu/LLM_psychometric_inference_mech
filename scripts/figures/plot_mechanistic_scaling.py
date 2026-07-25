#!/usr/bin/env python3
"""
Collect scaling results across all 14 models and plot:
1. Corrected slope vs model size (base vs instruct, Llama vs Qwen)
2. Mantel r vs model size
3. Reliability vs model size

Usage:
    python scripts/figures/plot_mechanistic_scaling.py
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

from psychometric_inference.model_registry import mech_model_tuples
from psychometric_inference.paths import MECHANISTIC_OUTPUT_DIR

MECH_ROOT = MECHANISTIC_OUTPUT_DIR

MODELS = mech_model_tuples(family_case="lower")


def collect_results():
    rows = []
    for name, size, family, mtype in MODELS:
        rd = MECH_ROOT / f"results_{name}"
        if not rd.exists():
            continue

        row = dict(model=name, size=size, family=family, mtype=mtype)

        # Geometry results (raw)
        gp = rd / "geometry" / "geometry_results.csv"
        if gp.exists():
            df = pd.read_csv(gp)
            sub = df[df["level"] == "subscale"]
            if not sub.empty:
                best = sub.loc[sub["mantel_r"].idxmax()]
                bl = int(best["layer"])
                row["best_layer"] = bl
                row["geom_mantel_r"] = best["mantel_r"]
                row["geom_slope_raw"] = best["slope"]

        bl = row.get("best_layer")
        if bl is None:
            rows.append(row)
            continue

        # Reliability
        rel = rd / "reliability" / f"reliability_L{bl}.csv"
        if rel.exists():
            rdf = pd.read_csv(rel)
            sr = rdf[rdf["type"] == "subscale"]
            if not sr.empty:
                row["mean_reliability"] = sr["spearman_brown_r"].mean()
                row["n_reliable"] = int((sr["spearman_brown_r"] >= 0.3).sum())

        # Corrected cosine sim (attenuation corrected)
        cor = rd / "reliability" / f"cosine_sim_corrected_L{bl}.csv"
        if cor.exists():
            corrected = pd.read_csv(cor, index_col=0)
            # Compute Mantel & slope from corrected vs human
            from psychometric_inference.mechanisms.geometry import compute_human_correlation_matrix, mantel_test
            from psychometric_inference.mechanisms.config import ALL_SUBSCALES
            from scipy.stats import pearsonr as _pearsonr
            hc = compute_human_correlation_matrix("subscale")
            common = [s for s in corrected.index if s in hc.index]
            if len(common) >= 4:
                n = len(common)
                idx = np.triu_indices(n, k=1)
                cv = corrected.loc[common, common].values[idx]
                hv = hc.loc[common, common].values[idx]
                v = ~(np.isnan(cv) | np.isnan(hv))
                if v.sum() >= 3:
                    rm, _ = mantel_test(corrected.loc[common, common].values,
                                        hc.loc[common, common].values)
                    sl = np.polyfit(hv[v], cv[v], 1)[0]
                    r_p, _ = _pearsonr(hv[v], cv[v])
                    row["geom_slope_corrected"] = sl
                    row["geom_mantel_r_corrected"] = rm
                    row["geom_pearson_r_corrected"] = r_p

        # Controls - permutation test
        controls_json = rd / "controls" / f"controls_results_L{bl}.json"
        if controls_json.exists():
            with open(controls_json) as f:
                ctrl = json.load(f)
            if "semantic_control" in ctrl:
                row["r_partial_semantic"] = ctrl["semantic_control"].get("r_partial")
                row["p_partial_semantic"] = ctrl["semantic_control"].get("p_partial")
            if "regression_permutation" in ctrl:
                row["reg_perm_baseline"] = ctrl["regression_permutation"].get("perm_r_mean")
                row["reg_genuine_signal"] = ctrl["regression_permutation"].get("real_r", 0) - \
                                             ctrl["regression_permutation"].get("perm_r_mean", 0)

        # Steering results
        steer_csv = rd / "regression_directions" / "regression_steering_raw.csv"
        if steer_csv.exists():
            try:
                sdf = pd.read_csv(steer_csv)
                # Compute steering effect matrix (source x target)
                # Each row has: source, alpha, target, item_number, response, ev
                # Effect = slope of ev vs alpha per (source, target)
                from psychometric_inference.mechanisms.config import ALL_SUBSCALES
                from psychometric_inference.mechanisms.geometry import compute_human_correlation_matrix, mantel_test
                from scipy.stats import pearsonr as _pearsonr
                human_corr = compute_human_correlation_matrix("subscale")

                sources = sorted(sdf["steer_source"].unique())
                targets = sorted(sdf["target"].unique())
                effect_mat = np.full((len(sources), len(targets)), np.nan)
                for i, s in enumerate(sources):
                    for j, t in enumerate(targets):
                        sub_df = sdf[(sdf["steer_source"] == s) & (sdf["target"] == t)]
                        if len(sub_df) < 3:
                            continue
                        evs = sub_df.groupby("alpha")["ev"].mean().reset_index()
                        if len(evs) >= 2 and evs["alpha"].std() > 0:
                            sl_eff = np.polyfit(evs["alpha"].values, evs["ev"].values, 1)[0]
                            effect_mat[i, j] = sl_eff

                # Build human_r matrix aligned
                common = [x for x in sources if x in human_corr.index and x in targets]
                if len(common) >= 4:
                    si = [sources.index(x) for x in common]
                    ti = [targets.index(x) for x in common]
                    effect_sub = effect_mat[np.ix_(si, ti)]
                    hum_sub = human_corr.loc[common, common].values

                    # Raw signed correlation
                    hv = hum_sub.flatten()
                    ev = effect_sub.flatten()
                    v = ~(np.isnan(hv) | np.isnan(ev))
                    # Exclude diagonals
                    n_c = len(common)
                    mask = np.ones((n_c, n_c), dtype=bool)
                    np.fill_diagonal(mask, False)
                    m_flat = mask.flatten() & v
                    if m_flat.sum() >= 5:
                        r_raw, p_raw = _pearsonr(hv[m_flat], ev[m_flat])
                        row["steer_raw_r"] = r_raw
                        row["steer_raw_p"] = p_raw

                    # Double-centered Mantel
                    ef = np.where(np.isnan(effect_sub), 0, effect_sub)
                    row_mean = ef.mean(axis=1, keepdims=True)
                    col_mean = ef.mean(axis=0, keepdims=True)
                    all_mean = ef.mean()
                    dc = ef - row_mean - col_mean + all_mean
                    dc_sym = (dc + dc.T) / 2
                    try:
                        rm_steer, pm_steer = mantel_test(dc_sym, hum_sub)
                        row["steer_dc_mantel_r"] = rm_steer
                        row["steer_dc_mantel_p"] = pm_steer
                    except Exception:
                        pass
            except Exception as e:
                print(f"  steering parse failed for {name}: {e}")

        rows.append(row)

    return pd.DataFrame(rows)


def scaling_plots(df, output_path):
    """Plot scaling curves: x = size, y = key metrics, faceted by base/instruct and family."""

    fig, axes = plt.subplots(1, 4, figsize=(22, 5))

    metrics = [
        ("geom_slope_corrected", "Corrected slope (reliability)", 1.0),
        ("geom_mantel_r", "Raw Mantel r (geometry)", None),
        ("mean_reliability", "Mean split-half reliability", 0.3),
        ("steer_dc_mantel_r", "Steering double-centered Mantel r", None),
    ]

    colors = {"llama": "#d7191c", "qwen": "#2c7bb6"}
    markers = {"instruct": "o", "base": "s"}

    for ax, (metric, label, ref_line) in zip(axes, metrics):
        if metric not in df.columns:
            ax.set_title(f"{label} (not available)")
            continue
        for family in ["llama", "qwen"]:
            for mtype in ["instruct", "base"]:
                sub = df[(df["family"] == family) & (df["mtype"] == mtype)].sort_values("size")
                if sub.empty or metric not in sub.columns:
                    continue
                label_str = f"{family.capitalize()} {mtype}"
                ls = "-" if mtype == "instruct" else "--"
                alpha = 1.0 if mtype == "instruct" else 0.6
                ax.plot(sub["size"], sub[metric],
                       color=colors[family], marker=markers[mtype],
                       linestyle=ls, alpha=alpha, markersize=10,
                       label=label_str, linewidth=2)

        ax.set_xscale("log")
        ax.set_xlabel("Model size (B parameters)")
        ax.set_ylabel(label)
        ax.set_title(label)
        if ref_line is not None:
            ax.axhline(ref_line, ls=":", color="gray", alpha=0.5)
        ax.axhline(0, ls=":", color="gray", alpha=0.3)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="best")

    plt.suptitle("Scaling of Psychometric Structure in LLM Representations", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def main():
    df = collect_results()
    print("\nCollected results:")
    cols_show = [c for c in ["model","mtype","family","size","best_layer",
                               "geom_mantel_r","geom_slope_raw","geom_slope_corrected",
                               "mean_reliability","n_reliable",
                               "steer_raw_r","steer_dc_mantel_r"] if c in df.columns]
    print(df[cols_show].to_string(index=False))

    out_csv = MECH_ROOT / "scaling_results_full.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved CSV: {out_csv}")

    scaling_plots(df, MECH_ROOT / "scaling_plot.png")


if __name__ == "__main__":
    main()
