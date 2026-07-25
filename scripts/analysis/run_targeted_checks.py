#!/usr/bin/env python3
"""
Standalone rerun of D2 / F3b / D2b using the SAME pipeline as paper Table 2.

This script:
  1. Re-runs the item-level cross-persona pipeline (mirrors itemlevel_comparison.py)
     for all 9 cross-persona models at paper Table 2's hard-coded layers.
  2. Computes D2 (cultural specificity), F3b (trivial-context), D2b (LOO)
     using the resulting 16x16 subscale correlation matrices.

Usage:
    cd /path/to/Psychometric Inference_mech
    python scripts/analysis/run_targeted_checks.py
"""

from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, ttest_rel, wilcoxon
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import cross_val_predict

from psychometric_inference.paths import (
    HUMAN_DATA_DIR,
    MECHANISTIC_DEFAULT_RESULTS_DIR,
    MECHANISTIC_OUTPUT_DIR,
    ROBUSTNESS_OUTPUT_DIR,
)

MECHANISTIC_ROOT = MECHANISTIC_OUTPUT_DIR
OUT_ROOT = ROBUSTNESS_OUTPUT_DIR
OUT_ROOT.mkdir(parents=True, exist_ok=True)

from psychometric_inference.scoring import SCORING_RULES, un_reverse_df  # type: ignore
from psychometric_inference.mechanisms.cross_persona import load_human_scores, SCALES  # type: ignore
from psychometric_inference.mechanisms.geometry import compute_human_correlation_matrix  # type: ignore

# ─────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────

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

MODEL_META = {
    "qwen05b_instruct": ("qwen", "instruct", 0.5),
    "llama1b_instruct": ("llama", "instruct", 1.0),
    "qwen3b_instruct":  ("qwen", "instruct", 3.0),
    "llama3b_instruct": ("llama", "instruct", 3.0),
    "qwen7b_instruct":  ("qwen", "instruct", 7.0),
    "llama8b_instruct": ("llama", "instruct", 8.0),
    "llama8b_base":     ("llama", "base", 8.0),
    "qwen14b_instruct": ("qwen", "instruct", 14.0),
    "qwen14b_base":     ("qwen", "base", 14.0),
}

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
SUB_TO_SCALE = {}
for scale_short, rules in SCORING_RULES.items():
    for sub_name in rules.get("subscales", {}):
        SUB_TO_SCALE[f"{scale_short}_{sub_name}"] = scale_short

CULTURE_GROUPS = {
    "stable": [
        "BigFive_Extraversion", "BigFive_Agreeableness",
        "BigFive_Conscientiousness", "BigFive_Neuroticism",
        "IRI_Empathic_concern", "IRI_Perspective_taking",
        "PANAS_Negative_Affect",
        "LifeSat_Life_Satisfaction",
    ],
    "ambiguous": [
        "BigFive_Openness", "PANAS_Positive_Affect", "Lonely_Loneliness",
        "IRI_Personal_distress", "IRI_Fantasy",
    ],
    "specific": [
        "POM_Peace_of_Mind",
        "SelfConst_Independent_self", "SelfConst_Interdependent_self",
    ],
}


# ─────────────────────────────────────────────────────────────────────────
# Load human items (un-reverse-coded) for ridge target labels
# ─────────────────────────────────────────────────────────────────────────

def load_human_items():
    """{subject_id: {scale_file: {item_num: response}}}"""
    human_items = {}
    for scale_file, scale_short in SCALES:
        frames = []
        for ds in ["SED", "SEDC", "SEDD"]:
            p = HUMAN_DATA_DIR / ds / f"{scale_file}.csv"
            if p.exists():
                df = pd.read_csv(p)
                try:
                    df = un_reverse_df(df, scale_file)
                except Exception:
                    pass
                frames.append(df)
        if not frames:
            continue
        combined = pd.concat(frames, ignore_index=True)
        id_col = next((c for c in ["Subject_ID", "ID", "Scan_ID"] if c in combined.columns), None)
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
                        v = row[col]
                        if not pd.isna(v):
                            human_items[sid][scale_file][item_num] = float(v)
                    except ValueError:
                        pass
    return human_items


# ─────────────────────────────────────────────────────────────────────────
# Item-level pipeline (mirrors itemlevel_comparison.py)
# ─────────────────────────────────────────────────────────────────────────

def itemlevel_pipeline(model_name, layer, human_items, subject_ids):
    """Returns 16x16 subscale correlation matrix (continuous, paper main pipeline)."""
    cp_dir = MECHANISTIC_ROOT / f"results_{model_name}" / "cross_persona"
    if not cp_dir.exists() and model_name == "llama8b_instruct":
        cp_dir = MECHANISTIC_DEFAULT_RESULTS_DIR / "cross_persona"
    if not cp_dir.exists():
        return None

    item_predictions = {}
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
                    return None
                acts = npz[key]
        except Exception:
            return None
        meta = pd.read_csv(meta_path)
        meta["subject_id"] = meta["subject_id"].astype(str)

        for (target_scale, item_num), group in meta.groupby(["target_scale", "item_number"]):
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
                ridge = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0, 1000.0], cv=5)
                y_pred = cross_val_predict(ridge, X, y, cv=5)
            except Exception:
                continue
            item_predictions[(persona_short, target_file, int(item_num))] = {
                sid: float(yp) for sid, yp in zip(sid_list, y_pred)
            }

    if not item_predictions:
        return None

    # Aggregate per subscale
    subscale_scores = {}
    for scale_file, scale_short in SCALES:
        rules = SCORING_RULES.get(scale_short, {})
        rev = set(rules.get("reverse_items", []))
        max_val = rules.get("max_val", 5)
        for sub_name, item_nums in rules.get("subscales", {}).items():
            sub_full = f"{scale_short}_{sub_name}"
            sid_scores = {}
            for sid in subject_ids:
                item_preds = []
                for item_num in item_nums:
                    preds = []
                    for persona_file, persona_short in SCALES:
                        if persona_short == scale_short:
                            continue
                        key = (persona_short, scale_file, int(item_num))
                        if key in item_predictions and sid in item_predictions[key]:
                            preds.append(item_predictions[key][sid])
                    if preds:
                        pv = float(np.mean(preds))
                        if item_num in rev:
                            pv = (max_val + 1) - pv
                        item_preds.append(pv)
                if item_preds:
                    sid_scores[sid] = float(np.mean(item_preds))
            if sid_scores:
                subscale_scores[sub_full] = sid_scores

    available = [s for s in ALL_SUBSCALES if s in subscale_scores]
    if len(available) < 5:
        return None
    pred_df = pd.DataFrame(
        {s: [subscale_scores[s].get(sid, np.nan) for sid in subject_ids] for s in available},
        index=subject_ids,
    )
    n = len(available)
    M = np.full((n, n), np.nan)
    for i, si in enumerate(available):
        for j, sj in enumerate(available):
            vi = pred_df[si].values
            vj = pred_df[sj].values
            v = ~(np.isnan(vi) | np.isnan(vj))
            if v.sum() > 10:
                M[i, j] = float(np.corrcoef(vi[v], vj[v])[0, 1])
    return pd.DataFrame(M, index=available, columns=available)


# ─────────────────────────────────────────────────────────────────────────
# Step 1: build pred_corr matrices for all 9 models
# ─────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  Loading human data...")
    print("=" * 70)
    scores_df = load_human_scores()
    subject_ids = list(scores_df.index)
    human_corr = compute_human_correlation_matrix("subscale")
    human_items = load_human_items()
    print(f"  N subjects: {len(subject_ids)}")
    print(f"  Human corr: {human_corr.shape}")

    print("\n" + "=" * 70)
    print("  Step 1: Item-level pipeline for 9 models at paper Table 2 layers")
    print("=" * 70)
    pred_corrs = {}
    t0 = time.time()
    for model_name, layer in ITEMLEVEL_LAYERS.items():
        ts = time.time()
        print(f"  [{model_name} L{layer}] computing...", end=" ", flush=True)
        M = itemlevel_pipeline(model_name, layer, human_items, subject_ids)
        if M is None:
            print("FAILED")
            continue
        pred_corrs[model_name] = M
        # Quick slope/Mantel sanity
        common = [s for s in M.index if s in human_corr.index]
        H = human_corr.loc[common, common].values
        P = M.loc[common, common].values
        idx = np.triu_indices(len(common), k=1)
        hv, pv = H[idx], P[idx]
        v = ~(np.isnan(hv) | np.isnan(pv))
        slope = float(np.polyfit(hv[v], pv[v], 1)[0])
        mr = float(np.corrcoef(hv[v], pv[v])[0, 1])
        print(f"slope={slope:.3f} mantel={mr:.3f}  [{time.time()-ts:.0f}s]")
    print(f"  Total: {time.time()-t0:.0f}s, {len(pred_corrs)}/9 models")

    if len(pred_corrs) < 9:
        print(f"\n  WARNING: only {len(pred_corrs)}/9 models succeeded")

    # ─────────────────────────────────────────────────────────────────────
    # Step 2: D2 (cultural specificity)
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  Step 2: D2 cultural specificity")
    print("=" * 70)
    d2_outdir = OUT_ROOT / "D2"
    d2_outdir.mkdir(exist_ok=True)
    d2_rows = []
    for model_name, M in pred_corrs.items():
        family, mtype, size = MODEL_META[model_name]
        for grp_name, grp_subs in CULTURE_GROUPS.items():
            common = [s for s in grp_subs if s in M.index and s in human_corr.index]
            if len(common) < 3:
                continue
            H = human_corr.loc[common, common].values
            P = M.loc[common, common].values
            n = len(common)
            idx = np.triu_indices(n, k=1)
            hv, pv = H[idx], P[idx]
            v = ~(np.isnan(hv) | np.isnan(pv))
            if v.sum() < 3:
                continue
            slope = float(np.polyfit(hv[v], pv[v], 1)[0])
            mr = float(np.corrcoef(hv[v], pv[v])[0, 1])
            d2_rows.append({
                "model": model_name, "family": family, "mtype": mtype, "size": size,
                "group": grp_name, "n_subscales": n, "n_pairs": int(v.sum()),
                "slope": slope, "mantel_r": mr,
            })
    d2_df = pd.DataFrame(d2_rows)
    d2_df.to_csv(d2_outdir / "D2_per_model_group.csv", index=False)

    stable = d2_df[d2_df["group"] == "stable"].set_index("model")
    specific = d2_df[d2_df["group"] == "specific"].set_index("model")
    common_m = [m for m in stable.index if m in specific.index]
    diffs = []
    for m in common_m:
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
    diff_df.to_csv(d2_outdir / "D2_paired_diffs.csv", index=False)

    if len(diff_df) >= 3:
        _, slope_t = ttest_rel(diff_df["slope_stable"], diff_df["slope_specific"])
        _, slope_w = wilcoxon(diff_df["slope_stable"], diff_df["slope_specific"])
        _, mantel_t = ttest_rel(diff_df["mantel_stable"], diff_df["mantel_specific"])
        _, mantel_w = wilcoxon(diff_df["mantel_stable"], diff_df["mantel_specific"])
    else:
        slope_t = slope_w = mantel_t = mantel_w = float("nan")

    d2_summary = {
        "n_models": len(common_m),
        "models": common_m,
        "slope_mean_diff_stable_minus_specific": float(diff_df["slope_diff"].mean()),
        "slope_t_p": float(slope_t),
        "slope_wilcoxon_p": float(slope_w),
        "mantel_mean_diff_stable_minus_specific": float(diff_df["mantel_diff"].mean()),
        "mantel_t_p": float(mantel_t),
        "mantel_wilcoxon_p": float(mantel_w),
    }
    (d2_outdir / "D2_summary.json").write_text(json.dumps(d2_summary, indent=2))
    print(f"  D2: n={len(common_m)} models")
    print(f"     mean Δslope (stable - specific) = {d2_summary['slope_mean_diff_stable_minus_specific']:+.3f}")
    print(f"     paired t p = {slope_t:.4g}, wilcoxon p = {slope_w:.4g}")
    print(f"     mean Δmantel = {d2_summary['mantel_mean_diff_stable_minus_specific']:+.3f}")
    print(f"     paired t p = {mantel_t:.4g}, wilcoxon p = {mantel_w:.4g}")

    # ─────────────────────────────────────────────────────────────────────
    # Step 3: D2b (LOO + cross-scale-only)
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  Step 3: D2b leave-one-pair-out + cross-scale-only")
    print("=" * 70)
    d2b_outdir = OUT_ROOT / "D2b"
    d2b_outdir.mkdir(exist_ok=True)

    # LOO on specific group
    loo_rows = []
    for model_name, M in pred_corrs.items():
        for grp_name, grp_subs in CULTURE_GROUPS.items():
            common = [s for s in grp_subs if s in M.index and s in human_corr.index]
            n = len(common)
            if n < 3:
                continue
            pairs = []
            for i in range(n):
                for j in range(i + 1, n):
                    si, sj = common[i], common[j]
                    pairs.append((si, sj, float(human_corr.loc[si, sj]),
                                  float(M.loc[si, sj])))
            if len(pairs) < 3:
                continue
            hv_full = np.array([p[2] for p in pairs])
            pv_full = np.array([p[3] for p in pairs])
            v = ~(np.isnan(hv_full) | np.isnan(pv_full))
            slope_full = float(np.polyfit(hv_full[v], pv_full[v], 1)[0]) if v.sum() >= 2 else float("nan")
            for idx_drop, (si, sj, _, _) in enumerate(pairs):
                mask = np.ones(len(pairs), dtype=bool)
                mask[idx_drop] = False
                hv = hv_full[mask]; pv = pv_full[mask]
                vv = ~(np.isnan(hv) | np.isnan(pv))
                slope_loo = float(np.polyfit(hv[vv], pv[vv], 1)[0]) if vv.sum() >= 2 else float("nan")
                loo_rows.append({
                    "model": model_name, "group": grp_name,
                    "dropped_pair": f"{si} × {sj}",
                    "slope_full": slope_full,
                    "slope_loo": slope_loo,
                    "delta_slope": slope_full - slope_loo,
                })
    loo_df = pd.DataFrame(loo_rows)
    loo_df.to_csv(d2b_outdir / "D2b_leave_one_pair_out.csv", index=False)

    spec_loo = loo_df[loo_df["group"] == "specific"].copy()
    influence = spec_loo.groupby("dropped_pair").agg(
        mean_delta=("delta_slope", "mean"),
        mean_abs_delta=("delta_slope", lambda x: float(np.mean(np.abs(x)))),
        n=("delta_slope", "size"),
    ).reset_index().sort_values("mean_abs_delta", ascending=False)
    influence.to_csv(d2b_outdir / "D2b_specific_pair_influence.csv", index=False)

    # Cross-scale-only D2
    rows_cs = []
    for model_name, M in pred_corrs.items():
        family, mtype, size = MODEL_META[model_name]
        for grp_name, grp_subs in CULTURE_GROUPS.items():
            common = [s for s in grp_subs if s in M.index and s in human_corr.index]
            n = len(common)
            if n < 3:
                continue
            hv_cs, pv_cs = [], []
            for i in range(n):
                for j in range(i + 1, n):
                    si, sj = common[i], common[j]
                    if SUB_TO_SCALE.get(si) == SUB_TO_SCALE.get(sj):
                        continue
                    hv_cs.append(float(human_corr.loc[si, sj]))
                    pv_cs.append(float(M.loc[si, sj]))
            hv_cs = np.array(hv_cs); pv_cs = np.array(pv_cs)
            v = ~(np.isnan(hv_cs) | np.isnan(pv_cs))
            slope_cs = float(np.polyfit(hv_cs[v], pv_cs[v], 1)[0]) if v.sum() >= 2 else float("nan")
            rows_cs.append({
                "model": model_name, "family": family, "mtype": mtype, "size": size,
                "group": grp_name, "n_pairs_cross_scale": int(v.sum()),
                "slope_cross_scale_only": slope_cs,
            })
    cs_df = pd.DataFrame(rows_cs)
    cs_df.to_csv(d2b_outdir / "D2b_cross_scale_only.csv", index=False)

    stable_cs = cs_df[cs_df["group"] == "stable"].set_index("model")
    specific_cs = cs_df[cs_df["group"] == "specific"].set_index("model")
    common_m_cs = [m for m in stable_cs.index if m in specific_cs.index]
    if len(common_m_cs) >= 3:
        diff = stable_cs.loc[common_m_cs, "slope_cross_scale_only"] - specific_cs.loc[common_m_cs, "slope_cross_scale_only"]
        _, p_t_cs = ttest_rel(stable_cs.loc[common_m_cs, "slope_cross_scale_only"],
                              specific_cs.loc[common_m_cs, "slope_cross_scale_only"])
        _, p_w_cs = wilcoxon(stable_cs.loc[common_m_cs, "slope_cross_scale_only"],
                             specific_cs.loc[common_m_cs, "slope_cross_scale_only"])
        cs_paired = {
            "n_models": len(common_m_cs),
            "mean_slope_diff": float(diff.mean()),
            "t_p": float(p_t_cs),
            "wilcoxon_p": float(p_w_cs),
        }
    else:
        cs_paired = None

    d2b_summary = {
        "specific_pair_influence_top3": influence.head(3).to_dict(orient="records"),
        "cross_scale_only_paired": cs_paired,
    }
    (d2b_outdir / "D2b_summary.json").write_text(json.dumps(d2b_summary, indent=2))
    print(f"  D2b: top influential pairs in specific group (mean |Δslope|):")
    for _, r in influence.head(3).iterrows():
        print(f"     {r['dropped_pair']:<55} mean|Δ|={r['mean_abs_delta']:.3f} (n={int(r['n'])})")
    if cs_paired:
        print(f"  Cross-scale-only Δslope (stable - specific) = {cs_paired['mean_slope_diff']:+.3f}")
        print(f"     paired t p = {cs_paired['t_p']:.4g}, wilcoxon p = {cs_paired['wilcoxon_p']:.4g}")

    # ─────────────────────────────────────────────────────────────────────
    # Step 4: F3b (within-scale vs cross-scale)
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  Step 4: F3b within-scale vs cross-scale trivial-context test")
    print("=" * 70)
    f3b_outdir = OUT_ROOT / "F3b"
    f3b_outdir.mkdir(exist_ok=True)

    f3b_rows = []
    for model_name, M in pred_corrs.items():
        family, mtype, size = MODEL_META[model_name]
        common = [s for s in ALL_SUBSCALES if s in M.index and s in human_corr.index]
        if len(common) < 4:
            continue
        n = len(common)
        for i in range(n):
            for j in range(i + 1, n):
                si, sj = common[i], common[j]
                scope = "within_scale" if SUB_TO_SCALE.get(si) == SUB_TO_SCALE.get(sj) else "cross_scale"
                h = float(human_corr.loc[si, sj])
                p = float(M.loc[si, sj])
                f3b_rows.append({
                    "model": model_name, "family": family, "mtype": mtype, "size": size,
                    "sub_i": si, "sub_j": sj, "scope": scope,
                    "human_r": h, "pred_r": p,
                    "signed_diff": p - h, "abs_diff": abs(p - h),
                })
    f3b_detail = pd.DataFrame(f3b_rows)
    f3b_detail.to_csv(f3b_outdir / "F3b_pair_detail.csv", index=False)

    by_ms = f3b_detail.groupby(["model", "scope"]).agg(
        n=("abs_diff", "size"),
        mean_abs_diff=("abs_diff", "mean"),
        mean_signed_diff=("signed_diff", "mean"),
    ).reset_index()
    by_ms.to_csv(f3b_outdir / "F3b_by_model_scope.csv", index=False)

    pivoted_abs = by_ms.pivot(index="model", columns="scope", values="mean_abs_diff")
    paired_rows = []
    for m in pivoted_abs.index:
        if "within_scale" in pivoted_abs.columns and "cross_scale" in pivoted_abs.columns:
            paired_rows.append({
                "model": m,
                "abs_within": pivoted_abs.loc[m, "within_scale"],
                "abs_cross": pivoted_abs.loc[m, "cross_scale"],
            })
    paired_df = pd.DataFrame(paired_rows)
    paired_df.to_csv(f3b_outdir / "F3b_paired.csv", index=False)

    if len(paired_df) >= 3:
        _, t_p = ttest_rel(paired_df["abs_within"], paired_df["abs_cross"])
        _, w_p = wilcoxon(paired_df["abs_within"], paired_df["abs_cross"])
    else:
        t_p = w_p = float("nan")

    pooled = f3b_detail.groupby("scope").agg(
        n_pairs=("abs_diff", "size"),
        mean_abs=("abs_diff", "mean"),
        median_abs=("abs_diff", "median"),
        mean_signed=("signed_diff", "mean"),
    ).to_dict(orient="index")
    mean_within = float(paired_df["abs_within"].mean())
    mean_cross = float(paired_df["abs_cross"].mean())
    f3b_summary = {
        "n_models": len(paired_df),
        "pooled_stats": pooled,
        "paired_test": {
            "mean_abs_within": mean_within,
            "mean_abs_cross": mean_cross,
            "ratio": mean_within / mean_cross if mean_cross > 0 else None,
            "t_p": float(t_p),
            "wilcoxon_p": float(w_p),
        },
    }
    (f3b_outdir / "F3b_summary.json").write_text(json.dumps(f3b_summary, indent=2))
    print(f"  F3b: pooled across {len(paired_df)} models")
    for s, st in pooled.items():
        print(f"     {s:<13}: n={st['n_pairs']:>4}  mean|d|={st['mean_abs']:.3f}  signed={st['mean_signed']:+.3f}")
    print(f"     mean|d| within = {mean_within:.3f}")
    print(f"     mean|d| cross  = {mean_cross:.3f}")
    print(f"     ratio = {mean_within/mean_cross:.3f}")
    print(f"     paired t p = {t_p:.4g}, wilcoxon p = {w_p:.4g}")

    print("\n" + "=" * 70)
    print(f"  DONE. Outputs in {OUT_ROOT}/D2/, D2b/, F3b/")
    print("=" * 70)


if __name__ == "__main__":
    main()
