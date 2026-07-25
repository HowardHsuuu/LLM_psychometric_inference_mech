#!/usr/bin/env python3
"""
Supplementary analyses for robustness / appendix.

1. Layer-wise slope trajectory (item-level pipeline) for all Table 2 models
2. Ridge lambda sensitivity on Llama 8B Instruct
3. Random features baseline (shuffled activations)
4. Small model repr slope (Llama 1B Instruct, Qwen 0.5B/3B Instruct)

Usage:
    python scripts/analysis/compute_additional_controls.py --all
    python scripts/analysis/compute_additional_controls.py --layerwise
    python scripts/analysis/compute_additional_controls.py --lambda_sensitivity
    python scripts/analysis/compute_additional_controls.py --random_baseline
    python scripts/analysis/compute_additional_controls.py --small_models
"""

import argparse
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.linear_model import RidgeCV, Ridge
from sklearn.model_selection import cross_val_predict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from psychometric_inference.scoring import SCORING_RULES
from psychometric_inference.mechanisms.config import ALL_SUBSCALES
from psychometric_inference.mechanisms.cross_persona import load_human_scores, SCALES
from psychometric_inference.mechanisms.geometry import compute_human_correlation_matrix, mantel_test
from psychometric_inference.paths import HUMAN_DATA_DIR, MECHANISTIC_OUTPUT_DIR

MECHANISTIC_ROOT = MECHANISTIC_OUTPUT_DIR


def compute_itemlevel_slope(cp_dir, layer, scores_df, subject_ids, human_corr):
    """Compute item-level pipeline slope for a given layer."""
    human_items = {}
    from psychometric_inference.scoring import un_reverse_df
    for scale_file, scale_short in SCALES:
        dfs = []
        for ds in ["SED", "SEDC", "SEDD"]:
            csv_path = HUMAN_DATA_DIR / ds / f"{scale_file}.csv"
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                df = un_reverse_df(df, scale_file)
                dfs.append(df)
        if dfs:
            combined = pd.concat(dfs, ignore_index=True)
            id_col = next((c for c in ["Subject_ID", "ID", "Scan_ID"] if c in combined.columns), None)
            if id_col:
                combined = combined.rename(columns={id_col: "Subject_ID"})
            else:
                combined["Subject_ID"] = [f"S{i:04d}" for i in range(len(combined))]
            combined["Subject_ID"] = combined["Subject_ID"].astype(str)
            human_items[scale_file] = combined

    human_item_responses = {}
    for scale_file, scale_short in SCALES:
        if scale_file not in human_items:
            continue
        df = human_items[scale_file]
        for _, row in df.iterrows():
            sid = str(row["Subject_ID"])
            if sid not in human_item_responses:
                human_item_responses[sid] = {}
            if scale_file not in human_item_responses[sid]:
                human_item_responses[sid][scale_file] = {}
            for col in df.columns:
                if col.startswith("Q"):
                    try:
                        item_num = int(col[1:])
                        human_item_responses[sid][scale_file][item_num] = row[col]
                    except ValueError:
                        pass

    SUB_TO_SCALE = {}
    for scale_short, rules in SCORING_RULES.items():
        for sub_name in rules.get("subscales", {}):
            SUB_TO_SCALE[f"{scale_short}_{sub_name}"] = scale_short

    item_predictions = {}
    for persona_file, persona_short in SCALES:
        rotation_dir = cp_dir / f"rotation_{persona_short}"
        act_path = rotation_dir / "activations.npz"
        meta_path = rotation_dir / "meta.csv"
        if not act_path.exists():
            continue
        acts_data = np.load(act_path)
        meta_df = pd.read_csv(meta_path)
        meta_df["subject_id"] = meta_df["subject_id"].astype(str)
        layer_key = f"L{layer}"
        if layer_key not in acts_data:
            return None, None
        acts = acts_data[layer_key]

        for (target_scale, item_num), group in meta_df.groupby(["target_scale", "item_number"]):
            target_file = None
            for sf, ss in SCALES:
                if ss == target_scale:
                    target_file = sf
                    break
            if target_file is None:
                continue
            indices = group.index.values
            sids = group["subject_id"].values
            X_list, y_list, sid_list = [], [], []
            for idx_pos, sid in zip(indices, sids):
                if idx_pos >= len(acts) or sid not in human_item_responses:
                    continue
                if target_file not in human_item_responses[sid]:
                    continue
                if item_num not in human_item_responses[sid][target_file]:
                    continue
                true_val = human_item_responses[sid][target_file][item_num]
                if pd.isna(true_val):
                    continue
                X_list.append(acts[idx_pos])
                y_list.append(float(true_val))
                sid_list.append(sid)
            if len(X_list) < 50:
                continue
            X = np.array(X_list)
            y = np.array(y_list)
            ridge = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0, 1000.0], cv=5)
            try:
                y_pred = cross_val_predict(ridge, X, y, cv=5)
            except Exception:
                continue
            key = (persona_short, target_file, item_num)
            item_predictions[key] = {sid: yp for sid, yp in zip(sid_list, y_pred)}

    # Aggregate to subscale
    subscale_predictions = {}
    for scale_file, scale_short in SCALES:
        rules = SCORING_RULES.get(scale_short, {})
        rev_items = set(rules.get("reverse_items", []))
        max_val = rules.get("max_val", 5)
        for sub_name, item_nums in rules.get("subscales", {}).items():
            sub_full = f"{scale_short}_{sub_name}"
            sid_scores = {}
            for sid in subject_ids:
                item_preds = []
                for item_num in item_nums:
                    preds_for_item = []
                    for persona_file, persona_short in SCALES:
                        if persona_short == scale_short:
                            continue
                        key = (persona_short, scale_file, item_num)
                        if key in item_predictions and sid in item_predictions[key]:
                            preds_for_item.append(item_predictions[key][sid])
                    if preds_for_item:
                        pred_val = np.mean(preds_for_item)
                        if item_num in rev_items:
                            pred_val = (max_val + 1) - pred_val
                        item_preds.append(pred_val)
                if item_preds:
                    sid_scores[sid] = np.mean(item_preds)
            if sid_scores:
                subscale_predictions[sub_full] = sid_scores

    available = [s for s in ALL_SUBSCALES if s in subscale_predictions]
    if len(available) < 5:
        return None, None

    pred_df = pd.DataFrame({sub: [subscale_predictions[sub].get(sid, np.nan)
                                    for sid in subject_ids]
                             for sub in available}, index=subject_ids)
    n = len(available)
    pred_corr = np.zeros((n, n))
    for i, si in enumerate(available):
        for j, sj in enumerate(available):
            vi, vj = pred_df[si].values, pred_df[sj].values
            v = ~(np.isnan(vi) | np.isnan(vj))
            if v.sum() > 10:
                pred_corr[i, j], _ = pearsonr(vi[v], vj[v])
            else:
                pred_corr[i, j] = np.nan

    pred_corr_df = pd.DataFrame(pred_corr, index=available, columns=available)
    common = [s for s in available if s in human_corr.index]
    nc = len(common)
    idx = np.triu_indices(nc, k=1)
    pv = pred_corr_df.loc[common, common].values[idx]
    hv = human_corr.loc[common, common].values[idx]
    valid = ~(np.isnan(pv) | np.isnan(hv))
    if valid.sum() < 5:
        return None, None
    slope = np.polyfit(hv[valid], pv[valid], 1)[0]
    mr, _ = mantel_test(pred_corr_df.loc[common, common].values,
                         human_corr.loc[common, common].values)
    return slope, mr


# ═══════════════════════════════════════════════════════
#  1. LAYER-WISE SLOPE TRAJECTORY
# ═══════════════════════════════════════════════════════

def run_layerwise(outdir):
    print("=" * 60)
    print("  1. LAYER-WISE ITEM-LEVEL SLOPE TRAJECTORY")
    print("=" * 60)

    scores_df = load_human_scores()
    subject_ids = list(scores_df.index)
    human_corr = compute_human_correlation_matrix("subscale")

    models = {
        "llama3b_instruct": [8, 10, 12, 14, 16, 18, 20],
        "llama8b_instruct": [12, 14, 16, 18, 20, 22, 24],
        "llama8b_base":     [12, 14, 16, 18, 20, 22, 24],
        "qwen7b_instruct":  [8, 10, 12, 14, 16, 18, 20],
        "qwen14b_instruct": [16, 20, 24, 28, 32, 36],
        "qwen14b_base":     [16, 20, 24, 28, 32, 36],
    }

    all_results = []
    for model_name, layers in models.items():
        cp_dir = MECHANISTIC_ROOT / f"results_{model_name}" / "cross_persona"
        if not cp_dir.exists():
            print(f"  {model_name}: no data, skipping")
            continue
        print(f"\n  {model_name}:")
        for layer in layers:
            slope, mr = compute_itemlevel_slope(cp_dir, layer, scores_df, subject_ids, human_corr)
            if slope is not None:
                print(f"    L{layer}: slope = {slope:.3f}, Mantel r = {mr:.3f}")
                all_results.append({"model": model_name, "layer": layer, "slope": slope, "mantel_r": mr})
            else:
                print(f"    L{layer}: failed")

    if all_results:
        df = pd.DataFrame(all_results)
        df.to_csv(outdir / "layerwise_itemlevel_slopes.csv", index=False)

        # Plot
        fig, ax = plt.subplots(figsize=(8, 5))
        colors = {"llama3b_instruct": "#e74c3c", "llama8b_instruct": "#c0392b",
                  "llama8b_base": "#e6b0aa", "qwen7b_instruct": "#2980b9",
                  "qwen14b_instruct": "#1a5276", "qwen14b_base": "#85c1e9"}
        for model_name in models:
            sub = df[df["model"] == model_name]
            if sub.empty: continue
            n_layers_total = {"llama3b_instruct": 28, "llama8b_instruct": 32, "llama8b_base": 32,
                              "qwen7b_instruct": 28, "qwen14b_instruct": 48, "qwen14b_base": 48}
            total = n_layers_total.get(model_name, 32)
            pct = sub["layer"] / total * 100
            ls = "--" if "base" in model_name else "-"
            ax.plot(pct, sub["slope"], marker="o", ls=ls, ms=6, lw=2,
                   color=colors.get(model_name, "gray"), label=model_name, alpha=0.8)
        ax.axhline(1.0, ls=":", color="gray", alpha=0.5)
        ax.set_xlabel("Layer depth (%)", fontsize=11)
        ax.set_ylabel("Item-level pipeline slope", fontsize=11)
        ax.set_title("Layer-wise representational slope trajectory")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.15)
        plt.tight_layout()
        plt.savefig(outdir / "layerwise_itemlevel_trajectory.png", dpi=300, bbox_inches="tight")
        plt.close()
        print(f"\n  Saved to {outdir}/layerwise_itemlevel_slopes.csv")


# ═══════════════════════════════════════════════════════
#  2. RIDGE LAMBDA SENSITIVITY
# ═══════════════════════════════════════════════════════

def run_lambda_sensitivity(outdir):
    print("\n" + "=" * 60)
    print("  2. RIDGE LAMBDA SENSITIVITY (Llama 8B Instruct)")
    print("=" * 60)

    scores_df = load_human_scores()
    subject_ids = list(scores_df.index)
    human_corr = compute_human_correlation_matrix("subscale")

    cp_dir = MECHANISTIC_ROOT / "results_llama8b_instruct" / "cross_persona"
    layer = 16

    from psychometric_inference.scoring import un_reverse_df
    SUB_TO_SCALE = {}
    for scale_short, rules in SCORING_RULES.items():
        for sub_name in rules.get("subscales", {}):
            SUB_TO_SCALE[f"{scale_short}_{sub_name}"] = scale_short

    # Load all activations and item responses
    human_items = {}
    for scale_file, scale_short in SCALES:
        dfs = []
        for ds in ["SED", "SEDC", "SEDD"]:
            csv_path = HUMAN_DATA_DIR / ds / f"{scale_file}.csv"
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                df = un_reverse_df(df, scale_file)
                dfs.append(df)
        if dfs:
            combined = pd.concat(dfs, ignore_index=True)
            id_col = next((c for c in ["Subject_ID", "ID", "Scan_ID"] if c in combined.columns), None)
            if id_col:
                combined = combined.rename(columns={id_col: "Subject_ID"})
            else:
                combined["Subject_ID"] = [f"S{i:04d}" for i in range(len(combined))]
            combined["Subject_ID"] = combined["Subject_ID"].astype(str)
            human_items[scale_file] = combined

    human_item_responses = {}
    for scale_file, scale_short in SCALES:
        if scale_file not in human_items: continue
        df = human_items[scale_file]
        for _, row in df.iterrows():
            sid = str(row["Subject_ID"])
            if sid not in human_item_responses: human_item_responses[sid] = {}
            if scale_file not in human_item_responses[sid]: human_item_responses[sid][scale_file] = {}
            for col in df.columns:
                if col.startswith("Q"):
                    try:
                        human_item_responses[sid][scale_file][int(col[1:])] = row[col]
                    except ValueError:
                        pass

    # Collect all (X, y, meta) for ridge
    all_data = []  # list of (persona_short, target_file, item_num, X, y, sids)
    for persona_file, persona_short in SCALES:
        rotation_dir = cp_dir / f"rotation_{persona_short}"
        act_path = rotation_dir / "activations.npz"
        meta_path = rotation_dir / "meta.csv"
        if not act_path.exists(): continue
        acts_data = np.load(act_path)
        meta_df = pd.read_csv(meta_path)
        meta_df["subject_id"] = meta_df["subject_id"].astype(str)
        layer_key = f"L{layer}"
        if layer_key not in acts_data: continue
        acts = acts_data[layer_key]

        for (target_scale, item_num), group in meta_df.groupby(["target_scale", "item_number"]):
            target_file = None
            for sf, ss in SCALES:
                if ss == target_scale: target_file = sf; break
            if target_file is None: continue
            indices = group.index.values
            sids = group["subject_id"].values
            X_list, y_list, sid_list = [], [], []
            for idx_pos, sid in zip(indices, sids):
                if idx_pos >= len(acts) or sid not in human_item_responses: continue
                if target_file not in human_item_responses[sid]: continue
                if item_num not in human_item_responses[sid][target_file]: continue
                true_val = human_item_responses[sid][target_file][item_num]
                if pd.isna(true_val): continue
                X_list.append(acts[idx_pos])
                y_list.append(float(true_val))
                sid_list.append(sid)
            if len(X_list) >= 50:
                all_data.append((persona_short, target_file, item_num, np.array(X_list), np.array(y_list), sid_list))

    # Test different lambdas
    lambdas = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
    results = []

    for alpha in lambdas:
        item_predictions = {}
        for persona_short, target_file, item_num, X, y, sid_list in all_data:
            ridge = Ridge(alpha=alpha)
            from sklearn.model_selection import cross_val_predict as cvp
            try:
                y_pred = cvp(ridge, X, y, cv=5)
            except Exception:
                continue
            key = (persona_short, target_file, item_num)
            item_predictions[key] = {sid: yp for sid, yp in zip(sid_list, y_pred)}

        # Aggregate
        subscale_predictions = {}
        for scale_file, scale_short in SCALES:
            rules = SCORING_RULES.get(scale_short, {})
            rev_items = set(rules.get("reverse_items", []))
            max_val = rules.get("max_val", 5)
            for sub_name, item_nums in rules.get("subscales", {}).items():
                sub_full = f"{scale_short}_{sub_name}"
                sid_scores = {}
                for sid in subject_ids:
                    item_preds = []
                    for inum in item_nums:
                        preds = []
                        for pf, ps in SCALES:
                            if ps == scale_short: continue
                            key = (ps, scale_file, inum)
                            if key in item_predictions and sid in item_predictions[key]:
                                preds.append(item_predictions[key][sid])
                        if preds:
                            pv = np.mean(preds)
                            if inum in rev_items: pv = (max_val + 1) - pv
                            item_preds.append(pv)
                    if item_preds:
                        sid_scores[sid] = np.mean(item_preds)
                if sid_scores:
                    subscale_predictions[sub_full] = sid_scores

        available = [s for s in ALL_SUBSCALES if s in subscale_predictions]
        pred_df = pd.DataFrame({sub: [subscale_predictions[sub].get(sid, np.nan)
                                        for sid in subject_ids]
                                 for sub in available}, index=subject_ids)
        n = len(available)
        pred_corr = np.zeros((n, n))
        for i, si in enumerate(available):
            for j, sj in enumerate(available):
                vi, vj = pred_df[si].values, pred_df[sj].values
                v = ~(np.isnan(vi) | np.isnan(vj))
                if v.sum() > 10: pred_corr[i, j], _ = pearsonr(vi[v], vj[v])
                else: pred_corr[i, j] = np.nan

        pred_corr_df = pd.DataFrame(pred_corr, index=available, columns=available)
        common = [s for s in available if s in human_corr.index]
        idx = np.triu_indices(len(common), k=1)
        pv = pred_corr_df.loc[common, common].values[idx]
        hv = human_corr.loc[common, common].values[idx]
        valid = ~(np.isnan(pv) | np.isnan(hv))
        slope = np.polyfit(hv[valid], pv[valid], 1)[0]
        mr, _ = mantel_test(pred_corr_df.loc[common, common].values,
                             human_corr.loc[common, common].values)
        print(f"  λ = {alpha:>10.2f}: slope = {slope:.3f}, Mantel r = {mr:.3f}")
        results.append({"alpha": alpha, "slope": slope, "mantel_r": mr})

    df = pd.DataFrame(results)
    df.to_csv(outdir / "ridge_lambda_sensitivity.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogx(df["alpha"], df["slope"], "o-", ms=8, lw=2, color="#2c3e50")
    ax.axhline(1.0, ls=":", color="gray", alpha=0.5)
    ax.set_xlabel("Ridge λ", fontsize=11)
    ax.set_ylabel("Item-level pipeline slope", fontsize=11)
    ax.set_title("Ridge regularization sensitivity (Llama 8B Instruct, L16)")
    ax.grid(alpha=0.15)
    plt.tight_layout()
    plt.savefig(outdir / "ridge_lambda_sensitivity.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved to {outdir}/")


# ═══════════════════════════════════════════════════════
#  3. RANDOM FEATURES BASELINE
# ═══════════════════════════════════════════════════════

def run_random_baseline(outdir):
    print("\n" + "=" * 60)
    print("  3. RANDOM FEATURES BASELINE (Llama 8B Instruct)")
    print("=" * 60)

    scores_df = load_human_scores()
    subject_ids = list(scores_df.index)
    human_corr = compute_human_correlation_matrix("subscale")

    cp_dir = MECHANISTIC_ROOT / "results_llama8b_instruct" / "cross_persona"
    layer = 16

    # Load real activations, then shuffle
    rng = np.random.default_rng(42)
    n_iterations = 10
    real_slope, real_mr = compute_itemlevel_slope(cp_dir, layer, scores_df, subject_ids, human_corr)
    print(f"  Real activations: slope = {real_slope:.3f}, Mantel r = {real_mr:.3f}")

    # For random baseline: shuffle subject labels within each rotation
    # This breaks the subject-activation mapping while preserving activation distribution
    shuffled_slopes = []
    for it in range(n_iterations):
        # Create shuffled activations by permuting within each rotation
        shuffled_cp_dir = outdir / f"_shuffled_temp_{it}"
        shuffled_cp_dir.mkdir(parents=True, exist_ok=True)

        for persona_file, persona_short in SCALES:
            rotation_dir = cp_dir / f"rotation_{persona_short}"
            act_path = rotation_dir / "activations.npz"
            meta_path = rotation_dir / "meta.csv"
            if not act_path.exists(): continue

            acts_data = np.load(act_path)
            meta_df = pd.read_csv(meta_path)
            layer_key = f"L{layer}"
            if layer_key not in acts_data: continue
            acts = acts_data[layer_key].copy()

            # Shuffle activations (break subject-activation mapping)
            perm = rng.permutation(len(acts))
            acts_shuffled = acts[perm]

            # Save shuffled
            rot_out = shuffled_cp_dir / f"rotation_{persona_short}"
            rot_out.mkdir(parents=True, exist_ok=True)
            np.savez(rot_out / "activations.npz", **{layer_key: acts_shuffled})
            meta_df.to_csv(rot_out / "meta.csv", index=False)

        slope, mr = compute_itemlevel_slope(shuffled_cp_dir, layer, scores_df, subject_ids, human_corr)
        if slope is not None:
            print(f"  Shuffled iter {it}: slope = {slope:.3f}, Mantel r = {mr:.3f}")
            shuffled_slopes.append(slope)

        # Cleanup
        import shutil
        shutil.rmtree(shuffled_cp_dir, ignore_errors=True)

    if shuffled_slopes:
        print(f"\n  Random baseline: slope = {np.mean(shuffled_slopes):.3f} ± {np.std(shuffled_slopes):.3f}")
        print(f"  Real slope:      {real_slope:.3f}")
        print(f"  Ratio:           {real_slope / np.mean(shuffled_slopes):.1f}x")

        results = {"real_slope": real_slope, "real_mantel_r": real_mr,
                   "shuffled_mean": np.mean(shuffled_slopes), "shuffled_std": np.std(shuffled_slopes),
                   "shuffled_slopes": shuffled_slopes}
        pd.DataFrame([results]).to_csv(outdir / "random_baseline.csv", index=False)


# ═══════════════════════════════════════════════════════
#  4. SMALL MODEL REPR SLOPE
# ═══════════════════════════════════════════════════════

def run_small_models(outdir):
    print("\n" + "=" * 60)
    print("  4. SMALL MODEL CROSS-PERSONA (extraction + item-level)")
    print("=" * 60)
    print("  This requires running cross-persona extraction on small models.")
    print("  Run these commands separately:")
    print()
    print("  python -m psychometric_inference.mechanisms.pipeline --models llama1b_instruct")
    print("  python -m psychometric_inference.mechanisms.pipeline --models qwen05b_instruct")
    print("  python -m psychometric_inference.mechanisms.pipeline --models qwen3b_instruct")
    print()
    print("  Then run:")
    print("  python -m psychometric_inference.mechanisms.cross_persona --analyze_only")
    print()

    # Check if any small model results already exist
    small_models = {
        "llama1b_instruct": 8,
        "qwen05b_instruct": 6,
        "qwen3b_instruct": 10,
    }
    for model_name, default_layer in small_models.items():
        cp_dir = MECHANISTIC_ROOT / f"results_{model_name}" / "cross_persona"
        if cp_dir.exists():
            print(f"  {model_name}: data exists, computing item-level slope...")
            scores_df = load_human_scores()
            subject_ids = list(scores_df.index)
            human_corr = compute_human_correlation_matrix("subscale")
            slope, mr = compute_itemlevel_slope(cp_dir, default_layer, scores_df, subject_ids, human_corr)
            if slope is not None:
                print(f"    slope = {slope:.3f}, Mantel r = {mr:.3f}")
            else:
                print(f"    Failed at layer {default_layer}")
        else:
            print(f"  {model_name}: no data yet")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--layerwise", action="store_true")
    parser.add_argument("--lambda_sensitivity", action="store_true")
    parser.add_argument("--random_baseline", action="store_true")
    parser.add_argument("--small_models", action="store_true")
    parser.add_argument("--outdir", type=str, default="outputs/supplementary")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.all or args.layerwise:
        run_layerwise(outdir)
    if args.all or args.lambda_sensitivity:
        run_lambda_sensitivity(outdir)
    if args.all or args.random_baseline:
        run_random_baseline(outdir)
    if args.all or args.small_models:
        run_small_models(outdir)

    print("\n  Done.")


if __name__ == "__main__":
    main()
