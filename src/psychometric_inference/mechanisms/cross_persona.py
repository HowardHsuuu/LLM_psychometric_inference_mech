#!/usr/bin/env python3
"""
Single-scale mechanistic analysis: matches behavioral experiment exactly.

Setup (identical to generate_instruct_behavior.py):
  - System prompt: one scale's real human responses (e.g., IRI 28 items)
  - User prompt: one item from a TARGET scale (e.g., Loneliness item)
  - Extract decision-point activation (last token before model answers)
  - 272 subjects × 7 persona rotations × target items

Key analysis:
  - Can activation predict target scale scores (cross-scale)?
  - Activation-predicted cross-scale matrix vs behavioral output matrix
  - Where does amplification happen?

Usage:
    python -m psychometric_inference.mechanisms.cross_persona --model_id meta-llama/Llama-3.1-8B-Instruct
    python -m psychometric_inference.mechanisms.cross_persona --analyze_only
"""

import argparse
import gc
import json
import logging
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import cross_val_predict
from tqdm import tqdm

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

from psychometric_inference.scoring import SCORING_RULES, un_reverse_df
from psychometric_inference.model_registry import model_config_by_hf_id
from psychometric_inference.mechanisms.config import RESULTS_DIR, ALL_SUBSCALES, SCALE_NAMES, SUB_TO_SCALE
from psychometric_inference.questionnaire_prompts import format_item_prompt
from psychometric_inference.paths import HUMAN_DATA_DIR, MECHANISTIC_OUTPUT_DIR, questionnaire_path

MECH_ROOT = MECHANISTIC_OUTPUT_DIR

# Scale definitions: (jsonl_filename_stem, short_name)
SCALES = [
    ("IRI", "IRI"),
    ("PANAS", "PANAS"),
    ("POM", "POM"),
    ("big_five", "BigFive"),
    ("in_inter_dependent", "SelfConst"),
    ("Life_Satisfaction", "LifeSat"),
    ("Loneliness", "Lonely"),
]

HUMAN_DATASETS = ["SED", "SEDC", "SEDD"]
N_TARGET_ITEMS = 3  # items per target scale to average over


def load_scale_definition(scale_file):
    path = questionnaire_path(scale_file)
    metadata, items = None, []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line.strip())
            if obj.get("_metadata"):
                metadata = obj
            else:
                items.append(obj)
    return metadata, items


def load_human_scores():
    """Load 272 subjects' subscale scores across all scales."""
    all_scores = {}
    for scale_file, scale_short in SCALES:
        rules = SCORING_RULES[scale_short]
        for ds in HUMAN_DATASETS:
            csv_path = HUMAN_DATA_DIR / ds / f"{scale_file}.csv"
            if not csv_path.exists():
                continue
            df = pd.read_csv(csv_path)
            df = un_reverse_df(df, scale_file)
            q_cols = [c for c in df.columns if c.startswith("Q")]
            id_col = next((c for c in ["Subject_ID", "ID", "Scan_ID"] if c in df.columns), None)

            for idx, row in df.iterrows():
                sid = str(row[id_col]) if id_col else f"S{idx:04d}"
                if sid not in all_scores:
                    all_scores[sid] = {}
                for sub_name, item_nums in rules["subscales"].items():
                    full_name = f"{scale_short}_{sub_name}"
                    vals = [row[f"Q{q}"] for q in item_nums if f"Q{q}" in row and pd.notna(row[f"Q{q}"])]
                    if vals:
                        all_scores[sid][full_name] = np.mean(vals)

    scores_df = pd.DataFrame.from_dict(all_scores, orient="index")
    scores_df.index.name = "Subject_ID"
    return scores_df


def build_system_prompt_single_scale(subject_row, scale_file, scale_short, metadata, items):
    """Build system prompt with ONE scale's responses only.
    Matches generate_instruct_behavior.py / injectors.py format exactly.
    """
    rules = SCORING_RULES[scale_short]
    scale_name = metadata.get("scale_name_zh", metadata.get("scale_name", scale_file))
    instruction = metadata.get("instruction", "")
    labels = metadata.get("response_labels", [])

    label_lines = [f"  {i} = {label}" for i, label in enumerate(labels, 1)]
    response_label_desc = "Response scale:\n" + "\n".join(label_lines)

    item_lines = []
    for item in items:
        q_num = item["item_number"]
        col = f"Q{q_num}"
        if col in subject_row and pd.notna(subject_row[col]):
            resp = int(subject_row[col])
        else:
            resp = "?"
        item_lines.append(f"Q{q_num}. {item['text']} → {resp}")
    item_responses = "\n".join(item_lines)

    return (
        f"You are completing psychological questionnaires as a person "
        f"with the following responses on the {scale_name}:\n\n"
        f"{instruction}\n"
        f"{response_label_desc}\n\n"
        f"{item_responses}\n\n"
        f"Respond to all questions as this person would, maintaining consistency "
        f"with this psychological profile. Answer with just the number corresponding to your choice."
    )


def extract_activations(model, tokenizer, system_prompt, item_prompt, layers):
    """Extract last-token activation at specified layers. No generation."""
    prompt = f"{system_prompt}\n\n{item_prompt}\n\nAnswer: "
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    hooked = {}
    hooks = []
    for layer_idx in layers:
        def make_hook(li):
            def hook_fn(module, input, output):
                hs = output[0] if isinstance(output, tuple) else output
                hooked[li] = hs[0, -1, :].detach().cpu().float().numpy()
            return hook_fn
        h = model.model.layers[layer_idx].register_forward_hook(make_hook(layer_idx))
        hooks.append(h)

    with torch.no_grad():
        model(**inputs)

    for h in hooks:
        h.remove()

    return hooked


def run_extraction(model_id, output_dir, target_layers, n_target_items=N_TARGET_ITEMS):
    """Phase 1: Extract activations for all rotations × subjects × target items."""
    logger.info(f"Loading model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map="cuda"
    )
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load all scale definitions
    scale_defs = {}
    for scale_file, scale_short in SCALES:
        meta, items = load_scale_definition(scale_file)
        scale_defs[scale_file] = {"metadata": meta, "items": items, "short": scale_short}

    # Load raw human data for building prompts
    human_data = {}  # scale_file -> DataFrame (un-reversed)
    subject_ids = None
    for scale_file, scale_short in SCALES:
        dfs = []
        for ds in HUMAN_DATASETS:
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
            human_data[scale_file] = combined
            if subject_ids is None:
                subject_ids = list(combined["Subject_ID"])

    n_subjects = len(subject_ids)
    logger.info(f"Subjects: {n_subjects}")
    logger.info(f"Target layers: {target_layers}")

    # For each rotation (persona_scale), extract activations
    for persona_file, persona_short in SCALES:
        rotation_dir = output_dir / f"rotation_{persona_short}"
        done_marker = rotation_dir / ".done"
        if done_marker.exists():
            logger.info(f"  Rotation {persona_short}: already done, skip")
            continue

        rotation_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"\n  Rotation: persona = {persona_short}")

        persona_meta = scale_defs[persona_file]["metadata"]
        persona_items = scale_defs[persona_file]["items"]
        persona_df = human_data[persona_file]

        # Collect target items from OTHER scales
        # Ensure every subscale is covered: pick 1 item per subscale
        target_items_info = []
        for target_file, target_short in SCALES:
            if target_short == persona_short:
                continue
            t_meta = scale_defs[target_file]["metadata"]
            t_items = scale_defs[target_file]["items"]

            # Get subscale definitions for this scale
            rules = SCORING_RULES.get(target_short, {})
            subscale_defs = rules.get("subscales", {})

            if subscale_defs:
                # Pick 1-2 items per subscale to ensure coverage
                selected_item_nums = set()
                for sub_name, item_nums in subscale_defs.items():
                    # Take up to 2 items per subscale
                    for inum in item_nums[:2]:
                        selected_item_nums.add(inum)
                for item in t_items:
                    if item["item_number"] in selected_item_nums:
                        target_items_info.append({
                            "scale_short": target_short,
                            "item": item,
                            "instruction": t_meta["instruction"],
                        })
            else:
                # No subscale info, take first 3
                for item in t_items[:3]:
                    target_items_info.append({
                        "scale_short": target_short,
                        "item": item,
                        "instruction": t_meta["instruction"],
                    })

        total = n_subjects * len(target_items_info)
        logger.info(f"    {n_subjects} subjects × {len(target_items_info)} target items = {total}")

        # Storage: per layer, (n_subjects, n_target_items, hidden_dim)
        all_activations = {l: [] for l in target_layers}
        all_meta = []

        pbar = tqdm(total=total, desc=f"R:{persona_short}", unit="item")

        for subj_idx in range(n_subjects):
            sid = subject_ids[subj_idx]
            subj_row = persona_df[persona_df["Subject_ID"] == sid]
            if subj_row.empty:
                pbar.update(len(target_items_info))
                continue
            subj_row = subj_row.iloc[0]

            sys_prompt = build_system_prompt_single_scale(
                subj_row, persona_file, persona_short, persona_meta, persona_items
            )

            for ti in target_items_info:
                item_prompt = format_item_prompt(ti["item"], ti["instruction"])
                acts = extract_activations(model, tokenizer, sys_prompt, item_prompt, target_layers)

                for l in target_layers:
                    if l in acts:
                        all_activations[l].append(acts[l])

                all_meta.append({
                    "subject_idx": subj_idx,
                    "subject_id": sid,
                    "target_scale": ti["scale_short"],
                    "item_number": ti["item"]["item_number"],
                })
                pbar.update(1)

        pbar.close()

        # Save
        save_dict = {}
        for l in target_layers:
            if all_activations[l]:
                save_dict[f"L{l}"] = np.array(all_activations[l])
        np.savez_compressed(rotation_dir / "activations.npz", **save_dict)
        pd.DataFrame(all_meta).to_csv(rotation_dir / "meta.csv", index=False)
        done_marker.touch()
        logger.info(f"    Saved to {rotation_dir}")

    # Cleanup
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()


def run_analysis(output_dir, target_layers, default_layer):
    """Phase 2: Analyze cross-scale prediction from activations."""
    logger.info("\n" + "="*60)
    logger.info("  CROSS-PERSONA ANALYSIS")
    logger.info("="*60)

    # Load human scores
    scores_df = load_human_scores()
    subject_ids = list(scores_df.index)
    logger.info(f"  {len(subject_ids)} subjects, {len(scores_df.columns)} subscales")

    layer = default_layer
    from psychometric_inference.mechanisms.geometry import compute_human_correlation_matrix, mantel_test

    human_corr = compute_human_correlation_matrix("subscale")

    # For each rotation, predict cross-scale subscale scores from activations
    all_cv_rs = []  # list of {persona, target_subscale, cv_r}

    for persona_file, persona_short in SCALES:
        rotation_dir = output_dir / f"rotation_{persona_short}"
        act_path = rotation_dir / "activations.npz"
        meta_path = rotation_dir / "meta.csv"

        if not act_path.exists():
            logger.info(f"  Rotation {persona_short}: no data, skip")
            continue

        acts_data = np.load(act_path)
        meta_df = pd.read_csv(meta_path)
        layer_key = f"L{layer}"
        if layer_key not in acts_data:
            logger.info(f"  Rotation {persona_short}: no layer {layer}, skip")
            continue

        acts = acts_data[layer_key]  # (n_subjects * n_target_items, hidden_dim)
        logger.info(f"\n  Rotation {persona_short}: {acts.shape}")

        # Group by subject: average activation across target items per target scale
        for target_short in SCALE_NAMES:
            if target_short == persona_short:
                continue

            # Find rows for this target scale
            mask = meta_df["target_scale"] == target_short
            if mask.sum() == 0:
                continue

            target_meta = meta_df[mask]

            # Average activation per subject for this target scale
            subject_acts = []
            subject_scores = []
            for sid in subject_ids:
                subj_mask = target_meta["subject_id"] == sid
                if subj_mask.sum() == 0:
                    continue
                indices = target_meta[subj_mask].index.values
                subj_act = acts[indices].mean(axis=0)
                subject_acts.append(subj_act)

                # Get ALL target subscale scores for this subject
                if sid in scores_df.index:
                    subject_scores.append(scores_df.loc[sid])

            if len(subject_acts) < 50:
                continue

            X = np.array(subject_acts)
            scores_sub = pd.DataFrame(subject_scores)

            # Predict each subscale of the target scale
            target_subscales = [s for s in ALL_SUBSCALES if SUB_TO_SCALE.get(s) == target_short]
            for sub in target_subscales:
                if sub not in scores_sub.columns:
                    continue
                y = scores_sub[sub].values
                valid = ~np.isnan(y)
                if valid.sum() < 50:
                    continue

                ridge = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0, 1000.0], cv=5)
                try:
                    y_pred = cross_val_predict(ridge, X[valid], y[valid], cv=5)
                    r, p = pearsonr(y[valid], y_pred)
                except Exception:
                    r, p = 0.0, 1.0

                all_cv_rs.append({
                    "persona": persona_short,
                    "target_scale": target_short,
                    "target_subscale": sub,
                    "cv_r": r,
                    "p": p,
                    "n": int(valid.sum()),
                })

    results_df = pd.DataFrame(all_cv_rs)
    results_df.to_csv(output_dir / f"cross_persona_cv_rs_L{layer}.csv", index=False)

    # Print summary
    logger.info(f"\n  Cross-scale prediction (Layer {layer}):")
    logger.info(f"  {'Persona':<10} {'Target subscale':<30} {'CV r':>8} {'p':>8}")
    logger.info(f"  {'-'*58}")
    for _, row in results_df.sort_values("cv_r", ascending=False).head(20).iterrows():
        logger.info(f"  {row['persona']:<10} {row['target_subscale']:<30} {row['cv_r']:>+8.3f} {row['p']:>8.4f}")

    mean_r = results_df["cv_r"].mean()
    n_sig = (results_df["p"] < 0.05).sum()
    logger.info(f"\n  Mean CV r: {mean_r:.3f}")
    logger.info(f"  Significant (p<0.05): {n_sig}/{len(results_df)}")

    # Build cross-scale prediction matrix: aggregate by (persona_scale, target_scale)
    # Average CV r across subscales within each target scale
    scale_level = results_df.groupby(["persona", "target_scale"])["cv_r"].mean().reset_index()
    pivot = scale_level.pivot(index="persona", columns="target_scale", values="cv_r")

    logger.info(f"\n  Scale-level cross-prediction matrix:")
    logger.info(pivot.to_string())
    pivot.to_csv(output_dir / f"cross_prediction_matrix_L{layer}.csv")

    # Compare with human correlation at scale level
    human_scale_corr = compute_human_correlation_matrix("scale")

    # Flatten both matrices for comparison (off-diagonal only)
    persona_scales = [s for s in SCALE_NAMES if s in pivot.index]
    target_scales = [s for s in SCALE_NAMES if s in pivot.columns]
    common_scales = [s for s in persona_scales if s in target_scales and s in human_scale_corr.index]

    pred_vals = []
    human_vals = []
    for ps in common_scales:
        for ts in common_scales:
            if ps == ts:
                continue
            if ps in pivot.index and ts in pivot.columns:
                pv = pivot.loc[ps, ts]
                hv = human_scale_corr.loc[ps, ts]
                if not np.isnan(pv) and not np.isnan(hv):
                    pred_vals.append(pv)
                    human_vals.append(hv)

    if len(pred_vals) >= 5:
        pred_vals = np.array(pred_vals)
        human_vals = np.array(human_vals)
        r_corr, p_corr = pearsonr(human_vals, pred_vals)
        slope, intercept = np.polyfit(human_vals, pred_vals, 1)

        logger.info(f"\n  Activation-predicted vs Human correlation:")
        logger.info(f"    Pearson r: {r_corr:.3f} (p = {p_corr:.4f})")
        logger.info(f"    Slope: {slope:.3f}")

        # Plot
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.scatter(human_vals, pred_vals, s=40, alpha=0.6, edgecolors="white")
        xx = np.linspace(human_vals.min() - 0.05, human_vals.max() + 0.05, 100)
        ax.plot(xx, slope * xx + intercept, "k--", lw=1, label=f"slope={slope:.2f}, r={r_corr:.3f}")
        ax.plot([-1, 1], [-1, 1], "gray", ls=":", lw=0.8, alpha=0.5, label="y=x")
        ax.set_xlabel("Human cross-scale correlation")
        ax.set_ylabel("Activation-predicted CV r")
        ax.set_title(f"Single-Scale Setup: Cross-Scale Prediction (Layer {layer})")
        ax.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f"cross_prediction_scatter_L{layer}.png", dpi=200)
        plt.close()

    logger.info(f"\n  All outputs saved to {output_dir}")


def run_predicted_correlation_analysis(output_dir, default_layer, llm_root=None):
    """Compare activation-predicted correlation matrix vs human vs behavioral output.
    
    For each rotation (persona = Scale A):
      1. Use ridge regression to predict ALL 16 subscale scores from activations
      2. Pool predicted scores across rotations
      3. Compute correlation matrix of predicted scores
      4. Compare with human correlation matrix (and behavioral output if available)
    """
    logger.info("\n" + "="*60)
    logger.info("  PREDICTED CORRELATION MATRIX ANALYSIS")
    logger.info("="*60)

    from psychometric_inference.mechanisms.geometry import compute_human_correlation_matrix, mantel_test

    scores_df = load_human_scores()
    subject_ids = list(scores_df.index)
    human_corr = compute_human_correlation_matrix("subscale")
    layer = default_layer

    # For each rotation, predict cross-scale subscale scores
    # Store predicted scores: {subscale: [predicted values for 272 subjects]}
    # We average predictions across rotations where this subscale is a TARGET (not persona)
    predicted_per_rotation = {}  # rotation -> {subscale -> predicted_array}

    for persona_file, persona_short in SCALES:
        rotation_dir = output_dir / f"rotation_{persona_short}"
        act_path = rotation_dir / "activations.npz"
        meta_path = rotation_dir / "meta.csv"
        if not act_path.exists():
            continue

        acts_data = np.load(act_path)
        meta_df = pd.read_csv(meta_path)
        layer_key = f"L{layer}"
        if layer_key not in acts_data:
            continue
        acts = acts_data[layer_key]

        predicted_per_rotation[persona_short] = {}

        # For each target subscale (not in persona scale)
        for target_sub in ALL_SUBSCALES:
            target_scale = SUB_TO_SCALE.get(target_sub)
            if target_scale == persona_short:
                continue  # skip same-scale predictions
            if target_sub not in scores_df.columns:
                continue

            # Get activations for this target scale's items, averaged per subject
            mask = meta_df["target_scale"] == target_scale
            if mask.sum() == 0:
                continue
            target_meta = meta_df[mask]

            subject_acts = []
            subject_sids = []
            for sid in subject_ids:
                subj_mask = target_meta["subject_id"] == sid
                if subj_mask.sum() == 0:
                    continue
                indices = target_meta[subj_mask].index.values
                subj_act = acts[indices].mean(axis=0)
                subject_acts.append(subj_act)
                subject_sids.append(sid)

            if len(subject_acts) < 50:
                continue

            X = np.array(subject_acts)
            y = np.array([scores_df.loc[sid, target_sub] for sid in subject_sids])
            valid = ~np.isnan(y)
            if valid.sum() < 50:
                continue

            ridge = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0, 1000.0], cv=5)
            try:
                y_pred = cross_val_predict(ridge, X[valid], y[valid], cv=5)
                # Store predictions aligned to subject_ids
                pred_full = np.full(len(subject_ids), np.nan)
                valid_sids = [subject_sids[i] for i in range(len(subject_sids)) if valid[i]]
                for sid, yp in zip(valid_sids, y_pred):
                    idx = subject_ids.index(sid)
                    pred_full[idx] = yp
                predicted_per_rotation[persona_short][target_sub] = pred_full
            except Exception:
                continue

    # Average predicted scores across rotations
    # For each subscale, average predictions from all rotations where it was a target
    avg_predicted = {}
    for sub in ALL_SUBSCALES:
        preds = []
        for rot, rot_preds in predicted_per_rotation.items():
            if sub in rot_preds:
                preds.append(rot_preds[sub])
        if preds:
            stacked = np.array(preds)
            avg_predicted[sub] = np.nanmean(stacked, axis=0)

    available = [s for s in ALL_SUBSCALES if s in avg_predicted]
    n = len(available)
    logger.info(f"  Available subscales: {n}")

    if n < 4:
        logger.info("  Not enough subscales for correlation matrix")
        return

    # Build predicted correlation matrix
    pred_matrix = np.zeros((n, n))
    for i, si in enumerate(available):
        for j, sj in enumerate(available):
            vi = avg_predicted[si]
            vj = avg_predicted[sj]
            valid = ~(np.isnan(vi) | np.isnan(vj))
            if valid.sum() > 10:
                pred_matrix[i, j], _ = pearsonr(vi[valid], vj[valid])
            else:
                pred_matrix[i, j] = np.nan

    pred_corr = pd.DataFrame(pred_matrix, index=available, columns=available)
    pred_corr.to_csv(output_dir / f"predicted_corr_matrix_L{layer}.csv")

    # Compare with human
    common = [s for s in available if s in human_corr.index]
    nc = len(common)
    idx = np.triu_indices(nc, k=1)
    pv = pred_corr.loc[common, common].values[idx]
    hv = human_corr.loc[common, common].values[idx]
    valid_mask = ~(np.isnan(pv) | np.isnan(hv))

    if valid_mask.sum() < 5:
        logger.info("  Not enough valid pairs")
        return

    pv_clean = pv[valid_mask]
    hv_clean = hv[valid_mask]

    r_pred_human, p_pred_human = pearsonr(hv_clean, pv_clean)
    slope_pred, intercept_pred = np.polyfit(hv_clean, pv_clean, 1)
    mantel_r, mantel_p = mantel_test(
        pred_corr.loc[common, common].values,
        human_corr.loc[common, common].values,
    )

    logger.info(f"\n  Activation-predicted correlation matrix vs Human:")
    logger.info(f"    Pearson r:  {r_pred_human:.3f} (p = {p_pred_human:.4f})")
    logger.info(f"    Slope:      {slope_pred:.3f}")
    logger.info(f"    Mantel r:   {mantel_r:.3f} (p = {mantel_p:.4f})")

    # Compare with behavioral output if available
    beh_slope = None
    if llm_root:
        llm_root = Path(llm_root)
        if llm_root.exists():
            try:
                from psychometric_inference.mechanisms.amplification import (
                    load_llm_subscale_scores, compute_llm_implicit_correlation
                )
                llm_scores = load_llm_subscale_scores(str(llm_root))
                if llm_scores.empty:
                    raise ValueError("No behavioral data loaded")
                beh_corr = compute_llm_implicit_correlation(llm_scores)
                common_beh = [s for s in common if s in beh_corr.index]
                if len(common_beh) >= 4:
                    nb = len(common_beh)
                    idx_b = np.triu_indices(nb, k=1)
                    bv = beh_corr.loc[common_beh, common_beh].values[idx_b]
                    hv_b = human_corr.loc[common_beh, common_beh].values[idx_b]
                    pv_b = pred_corr.loc[common_beh, common_beh].values[idx_b]
                    valid_b = ~(np.isnan(bv) | np.isnan(hv_b) | np.isnan(pv_b))

                    if valid_b.sum() >= 5:
                        beh_slope_val = np.polyfit(hv_b[valid_b], bv[valid_b], 1)[0]
                        pred_slope_val = np.polyfit(hv_b[valid_b], pv_b[valid_b], 1)[0]
                        beh_r, _ = pearsonr(hv_b[valid_b], bv[valid_b])

                        logger.info(f"\n  Three-way comparison (vs human correlation):")
                        logger.info(f"    Activation-predicted slope: {pred_slope_val:.3f}")
                        logger.info(f"    Behavioral output slope:    {beh_slope_val:.3f}")
                        logger.info(f"    Human (reference):          1.000")
                        if pred_slope_val > 0:
                            ratio = beh_slope_val / pred_slope_val
                            logger.info(f"    Behavioral/Predicted ratio: {ratio:.2f}")
                            if ratio > 1.2:
                                logger.info(f"    → Amplification in readout (representation → output)")
                            elif ratio < 0.8:
                                logger.info(f"    → Attenuation in readout")
                            else:
                                logger.info(f"    → Amplification already at representation level")
                        beh_slope = beh_slope_val
            except Exception as e:
                logger.error(f"    Failed to load behavioral data: {e}")

    # Plot
    fig, axes = plt.subplots(1, 2 if beh_slope is None else 3, figsize=(7 * (2 if beh_slope is None else 3), 6))
    if not isinstance(axes, np.ndarray):
        axes = [axes]

    # Panel 1: predicted vs human
    ax = axes[0]
    ax.scatter(hv_clean, pv_clean, s=30, alpha=0.5, edgecolors="white")
    xx = np.linspace(hv_clean.min() - 0.05, hv_clean.max() + 0.05, 100)
    ax.plot(xx, slope_pred * xx + intercept_pred, "k--", lw=1,
            label=f"slope={slope_pred:.2f}, r={r_pred_human:.3f}")
    ax.plot([-1, 1], [-1, 1], "gray", ls=":", lw=0.8, alpha=0.5)
    ax.set_xlabel("Human correlation")
    ax.set_ylabel("Predicted correlation")
    ax.set_title("Activation-predicted vs Human")
    ax.legend(fontsize=9)

    # Panel 2: heatmap of predicted correlation matrix
    ax = axes[1]
    im = ax.imshow(pred_corr.loc[common, common].values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(common)))
    ax.set_xticklabels([s.split("_", 1)[1][:8] for s in common], rotation=90, fontsize=7)
    ax.set_yticks(range(len(common)))
    ax.set_yticklabels([s.split("_", 1)[1][:8] for s in common], fontsize=7)
    ax.set_title("Predicted correlation matrix")
    plt.colorbar(im, ax=ax, shrink=0.8)

    plt.suptitle(f"Single-Scale Predicted Correlation (Layer {layer})\n"
                 f"Mantel r = {mantel_r:.3f}, slope = {slope_pred:.3f}",
                 fontsize=12)
    plt.tight_layout()
    plt.savefig(output_dir / f"predicted_corr_analysis_L{layer}.png", dpi=200, bbox_inches="tight")
    plt.close()

    # Save summary
    summary = {
        "layer": layer,
        "n_subscales": n,
        "r_pred_human": r_pred_human,
        "p_pred_human": p_pred_human,
        "slope_pred": slope_pred,
        "mantel_r": mantel_r,
        "mantel_p": mantel_p,
    }
    if beh_slope is not None:
        summary["beh_slope"] = beh_slope
    with open(output_dir / f"predicted_corr_summary_L{layer}.json", "w") as f:
        json.dump(summary, f, indent=2)

    return slope_pred, mantel_r


def run_permutation_test(output_dir, default_layer, n_perms=100):
    """Permutation test: shuffle subject labels, refit ridge, check if slope is artifact."""
    logger.info("\n" + "="*60)
    logger.info("  PERMUTATION TEST FOR PREDICTED CORRELATION SLOPE")
    logger.info("="*60)

    from psychometric_inference.mechanisms.geometry import compute_human_correlation_matrix, mantel_test

    scores_df = load_human_scores()
    subject_ids = list(scores_df.index)
    human_corr = compute_human_correlation_matrix("subscale")
    layer = default_layer
    rng = np.random.default_rng(42)

    def compute_predicted_slope(scores_to_use):
        """Core function: predict subscale scores from activations, compute correlation matrix slope."""
        predicted = {}
        for persona_file, persona_short in SCALES:
            rotation_dir = output_dir / f"rotation_{persona_short}"
            act_path = rotation_dir / "activations.npz"
            meta_path = rotation_dir / "meta.csv"
            if not act_path.exists():
                continue
            acts_data = np.load(act_path)
            meta_df = pd.read_csv(meta_path)
            layer_key = f"L{layer}"
            if layer_key not in acts_data:
                continue
            acts = acts_data[layer_key]

            for target_sub in ALL_SUBSCALES:
                target_scale = SUB_TO_SCALE.get(target_sub)
                if target_scale == persona_short or target_sub not in scores_to_use.columns:
                    continue
                mask = meta_df["target_scale"] == target_scale
                if mask.sum() == 0:
                    continue
                target_meta = meta_df[mask]

                subject_acts, subject_sids = [], []
                for sid in subject_ids:
                    subj_mask = target_meta["subject_id"] == sid
                    if subj_mask.sum() == 0:
                        continue
                    indices = target_meta[subj_mask].index.values
                    subject_acts.append(acts[indices].mean(axis=0))
                    subject_sids.append(sid)

                if len(subject_acts) < 50:
                    continue
                X = np.array(subject_acts)
                y = np.array([scores_to_use.loc[sid, target_sub] for sid in subject_sids])
                valid = ~np.isnan(y)
                if valid.sum() < 50:
                    continue

                ridge = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0, 1000.0], cv=5)
                try:
                    y_pred = cross_val_predict(ridge, X[valid], y[valid], cv=5)
                    pred_full = np.full(len(subject_ids), np.nan)
                    valid_sids = [subject_sids[i] for i in range(len(subject_sids)) if valid[i]]
                    for sid, yp in zip(valid_sids, y_pred):
                        pred_full[subject_ids.index(sid)] = yp
                    key = f"{persona_short}_{target_sub}"
                    if key not in predicted:
                        predicted[key] = []
                    predicted[key].append(pred_full)
                except Exception:
                    continue

        # Average across rotations per subscale
        avg_pred = {}
        for sub in ALL_SUBSCALES:
            preds = []
            for key, vals in predicted.items():
                if key.endswith(f"_{sub}"):
                    preds.extend(vals)
            if preds:
                avg_pred[sub] = np.nanmean(preds, axis=0)

        available = [s for s in ALL_SUBSCALES if s in avg_pred]
        if len(available) < 4:
            return np.nan, np.nan

        n_a = len(available)
        pm = np.zeros((n_a, n_a))
        for i, si in enumerate(available):
            for j, sj in enumerate(available):
                vi, vj = avg_pred[si], avg_pred[sj]
                v = ~(np.isnan(vi) | np.isnan(vj))
                pm[i, j] = pearsonr(vi[v], vj[v])[0] if v.sum() > 10 else np.nan

        common = [s for s in available if s in human_corr.index]
        nc = len(common)
        ci = [available.index(s) for s in common]
        sub_pm = pm[np.ix_(ci, ci)]
        sub_hm = human_corr.loc[common, common].values
        idx = np.triu_indices(nc, k=1)
        pv_c = sub_pm[idx]
        hv_c = sub_hm[idx]
        v = ~(np.isnan(pv_c) | np.isnan(hv_c))
        if v.sum() < 5:
            return np.nan, np.nan
        sl = np.polyfit(hv_c[v], pv_c[v], 1)[0]
        mr, _ = mantel_test(sub_pm, sub_hm)
        return sl, mr

    # Real result
    real_slope, real_mantel = compute_predicted_slope(scores_df)
    logger.info(f"  Real: slope = {real_slope:.3f}, Mantel r = {real_mantel:.3f}")

    # Permutations
    perm_slopes, perm_mantels = [], []
    for i in range(n_perms):
        perm_idx = rng.permutation(len(scores_df))
        scores_perm = scores_df.copy()
        scores_perm.index = scores_df.index[perm_idx]
        sl, mr = compute_predicted_slope(scores_perm)
        perm_slopes.append(sl)
        perm_mantels.append(mr)
        if (i + 1) % 20 == 0:
            logger.info(f"    Perm {i+1}/{n_perms}: slope = {sl:.3f}, Mantel r = {mr:.3f}")

    perm_slopes = np.array(perm_slopes)
    perm_mantels = np.array(perm_mantels)
    p_slope = np.mean(perm_slopes >= real_slope)
    p_mantel = np.mean(perm_mantels >= real_mantel)

    logger.info(f"\n  Results ({n_perms} permutations):")
    logger.info(f"    Real slope:     {real_slope:.3f}")
    logger.info(f"    Perm slope:     {np.nanmean(perm_slopes):.3f} ± {np.nanstd(perm_slopes):.3f}")
    logger.info(f"    p-value (slope): {p_slope:.4f}")
    logger.info(f"    Real Mantel r:  {real_mantel:.3f}")
    logger.info(f"    Perm Mantel r:  {np.nanmean(perm_mantels):.3f} ± {np.nanstd(perm_mantels):.3f}")
    logger.info(f"    p-value (Mantel): {p_mantel:.4f}")

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.hist(perm_slopes[~np.isnan(perm_slopes)], bins=20, alpha=0.7, color="#2c7bb6")
    ax1.axvline(real_slope, color="red", lw=2, label=f"Real ({real_slope:.3f})")
    ax1.set_xlabel("Slope"); ax1.set_title(f"Slope permutation (p={p_slope:.4f})"); ax1.legend()
    ax2.hist(perm_mantels[~np.isnan(perm_mantels)], bins=20, alpha=0.7, color="#2c7bb6")
    ax2.axvline(real_mantel, color="red", lw=2, label=f"Real ({real_mantel:.3f})")
    ax2.set_xlabel("Mantel r"); ax2.set_title(f"Mantel permutation (p={p_mantel:.4f})"); ax2.legend()
    plt.suptitle("Cross-Persona Predicted Correlation: Permutation Test")
    plt.tight_layout()
    plt.savefig(output_dir / f"permutation_test_L{default_layer}.png", dpi=200, bbox_inches="tight")
    plt.close()

    return {"real_slope": real_slope, "perm_slope_mean": float(np.nanmean(perm_slopes)),
            "perm_slope_std": float(np.nanstd(perm_slopes)), "p_slope": p_slope,
            "real_mantel": real_mantel, "perm_mantel_mean": float(np.nanmean(perm_mantels)),
            "p_mantel": p_mantel}


def run_layerwise_analysis(output_dir, target_layers):
    """Run predicted correlation analysis across all layers."""
    logger.info("\n" + "="*60)
    logger.info("  LAYER-WISE CROSS-PERSONA ANALYSIS")
    logger.info("="*60)

    rows = []
    for layer in target_layers:
        # Check if data exists for this layer
        has_layer = False
        for _, ps in SCALES:
            act_path = output_dir / f"rotation_{ps}" / "activations.npz"
            if act_path.exists():
                data = np.load(act_path)
                if f"L{layer}" in data:
                    has_layer = True
                break
        if not has_layer:
            continue

        logger.info(f"  Layer {layer}...")
        try:
            result = run_predicted_correlation_analysis(output_dir, layer)
            if result:
                slope, mantel_r = result
                rows.append({"layer": layer, "slope": slope, "mantel_r": mantel_r})
        except Exception as e:
            logger.error(f"    Layer {layer} failed: {e}")

    if rows:
        layer_df = pd.DataFrame(rows)
        layer_df.to_csv(output_dir / "layerwise_results.csv", index=False)
        logger.info(f"\n  Layer-wise results:")
        logger.info(layer_df.to_string(index=False))

        # Plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        ax1.plot(layer_df["layer"], layer_df["slope"], "o-", color="#d7191c")
        ax1.axhline(1.0, ls=":", color="gray", alpha=0.5)
        ax1.set_xlabel("Layer"); ax1.set_ylabel("Slope"); ax1.set_title("Predicted slope by layer")
        ax2.plot(layer_df["layer"], layer_df["mantel_r"], "o-", color="#2c7bb6")
        ax2.set_xlabel("Layer"); ax2.set_ylabel("Mantel r"); ax2.set_title("Mantel r by layer")
        plt.suptitle("Cross-Persona: Layer-wise Analysis")
        plt.tight_layout()
        plt.savefig(output_dir / "layerwise_plot.png", dpi=200, bbox_inches="tight")
        plt.close()


def run_asymmetry_analysis(output_dir, default_layer, llm_root=None):
    """Compare asymmetry in activation-predicted vs behavioral cross-prediction matrices."""
    logger.info("\n" + "="*60)
    logger.info("  ASYMMETRY ANALYSIS")
    logger.info("="*60)

    # Load the scale-level cross-prediction matrix (CV r, inherently asymmetric)
    cv_path = output_dir / f"cross_persona_cv_rs_L{default_layer}.csv"
    if not cv_path.exists():
        logger.info("  No CV results found, skip")
        return

    results_df = pd.read_csv(cv_path)
    scale_level = results_df.groupby(["persona", "target_scale"])["cv_r"].mean().reset_index()
    pivot = scale_level.pivot(index="persona", columns="target_scale", values="cv_r")

    # Compute asymmetry: for each pair (A,B), asym = pivot[A,B] - pivot[B,A]
    scales = [s for s in SCALE_NAMES if s in pivot.index and s in pivot.columns]
    n = len(scales)
    asym_act = np.zeros((n, n))
    for i, si in enumerate(scales):
        for j, sj in enumerate(scales):
            if i == j:
                continue
            v1 = pivot.loc[si, sj] if si in pivot.index and sj in pivot.columns else np.nan
            v2 = pivot.loc[sj, si] if sj in pivot.index and si in pivot.columns else np.nan
            asym_act[i, j] = v1 - v2 if not (np.isnan(v1) or np.isnan(v2)) else np.nan

    logger.info(f"  Activation asymmetry (A→B minus B→A):")
    asym_df = pd.DataFrame(asym_act, index=scales, columns=scales)
    logger.info(asym_df.round(3).to_string())

    # Compare with behavioral asymmetry if available
    if llm_root:
        llm_root = Path(llm_root)
        if llm_root.exists():
            try:
                # Load behavioral per-rotation results
                from psychometric_inference.mechanisms.amplification import load_llm_subscale_scores
                llm_scores = load_llm_subscale_scores(str(llm_root))
                if not llm_scores.empty:
                    # Compute per-rotation behavioral cross-scale correlation
                    beh_pivot = {}
                    for persona_scale, group in llm_scores.groupby("persona_scale"):
                        from psychometric_inference.mechanisms.amplification import FILENAME_TO_SCALE
                        ps = FILENAME_TO_SCALE.get(persona_scale, persona_scale)
                        for ts in SCALE_NAMES:
                            if ts == ps:
                                continue
                            target_subs = [s for s in ALL_SUBSCALES if SUB_TO_SCALE.get(s) == ts]
                            cvrs = []
                            for sub in target_subs:
                                if sub in group.columns:
                                    from psychometric_inference.mechanisms.geometry import compute_human_correlation_matrix
                                    scores_df = load_human_scores()
                                    human_vals = scores_df[sub].values
                                    llm_vals = group[sub].values
                                    valid = ~(np.isnan(human_vals) | np.isnan(llm_vals))
                                    if valid.sum() > 10:
                                        r, _ = pearsonr(human_vals[valid], llm_vals[valid])
                                        cvrs.append(r)
                            if cvrs:
                                beh_pivot[(ps, ts)] = np.mean(cvrs)

                    # Build behavioral asymmetry
                    asym_beh = np.zeros((n, n))
                    for i, si in enumerate(scales):
                        for j, sj in enumerate(scales):
                            if i == j:
                                continue
                            v1 = beh_pivot.get((si, sj), np.nan)
                            v2 = beh_pivot.get((sj, si), np.nan)
                            asym_beh[i, j] = v1 - v2 if not (np.isnan(v1) or np.isnan(v2)) else np.nan

                    # Correlate asymmetry patterns
                    idx = np.triu_indices(n, k=1)
                    act_asym_vec = asym_act[idx]
                    beh_asym_vec = asym_beh[idx]
                    valid = ~(np.isnan(act_asym_vec) | np.isnan(beh_asym_vec))
                    if valid.sum() >= 5:
                        r_asym, p_asym = pearsonr(act_asym_vec[valid], beh_asym_vec[valid])
                        logger.info(f"\n  Asymmetry correlation (activation vs behavioral):")
                        logger.info(f"    r = {r_asym:.3f} (p = {p_asym:.4f})")
            except Exception as e:
                logger.error(f"  Behavioral asymmetry failed: {e}")


def run_bridging_analysis(output_dir, default_layer):
    """Bridge mega-prompt and single-scale findings.
    
    Test: does contrastive direction geometry (cosine sim between subscale
    directions from mega-prompt setup) predict cross-persona inference
    performance (CV r from single-scale setup)?
    
    If yes → the representational geometry IS the basis for cross-scale inference.
    """
    logger.info("\n" + "="*60)
    logger.info("  BRIDGING ANALYSIS: Direction Geometry → Inference Performance")
    logger.info("="*60)

    from psychometric_inference.mechanisms.geometry import compute_human_correlation_matrix, mantel_test

    # Try to find contrastive directions - first in RESULTS_DIR (if model-specific config set),
    # else look in results_llama8b_instruct (default bridging model)
    candidate_roots = [RESULTS_DIR]
    llama8b_root = MECH_ROOT / "results_llama8b_instruct"
    if llama8b_root.exists() and llama8b_root != RESULTS_DIR:
        candidate_roots.append(llama8b_root)

    cos_path = None
    for root in candidate_roots:
        for layer_try in [default_layer, 16, 18, 14, 12, 20, 22, 6, 8, 10]:
            test_path = root / "geometry" / f"subscale_cosine_sim_L{layer_try}.csv"
            if test_path.exists():
                cos_path = test_path
                bridging_root = root
                logger.info(f"  Using contrastive directions from: {cos_path}")
                break
        if cos_path is not None:
            break

    if cos_path is None:
        logger.info("  No contrastive direction cosine sim found. Run mega-prompt directions first.")
        return

    cos_sim = pd.read_csv(cos_path, index_col=0)

    # Match cross_persona layer to the one that has the contrastive directions  
    # But cross_persona results are layer-specific too — try matching layer or fall back
    layer_from_cos = int(cos_path.stem.split("L")[-1])

    # Load cross-persona CV r results - try bridging_root/cross_persona first, then output_dir
    cv_candidates = [
        bridging_root / "cross_persona" / f"cross_persona_cv_rs_L{layer_from_cos}.csv",
        output_dir / f"cross_persona_cv_rs_L{layer_from_cos}.csv",
        output_dir / f"cross_persona_cv_rs_L{default_layer}.csv",
    ]
    cv_path = None
    for cvp in cv_candidates:
        if cvp.exists():
            cv_path = cvp
            break

    if cv_path is None:
        # Try any layer
        for l in [16, 18, 14, 12, 20, 22]:
            for root in [bridging_root / "cross_persona", output_dir]:
                p = root / f"cross_persona_cv_rs_L{l}.csv"
                if p.exists():
                    cv_path = p
                    break
            if cv_path is not None:
                break

    if cv_path is None:
        logger.info("  No cross-persona CV results found.")
        return

    logger.info(f"  Using cross-persona CV from: {cv_path}")
    cv_df = pd.read_csv(cv_path)

    # Build cross-persona CV r matrix at subscale level
    # For each pair (sub_i, sub_j), find the CV r where:
    #   persona = scale containing sub_i, target_subscale = sub_j
    cv_matrix = pd.DataFrame(np.nan, index=ALL_SUBSCALES, columns=ALL_SUBSCALES)

    for _, row in cv_df.iterrows():
        persona_scale = row["persona"]
        target_sub = row["target_subscale"]
        cvr = row["cv_r"]
        # Assign to all subscales of the persona scale
        persona_subs = [s for s in ALL_SUBSCALES if SUB_TO_SCALE.get(s) == persona_scale]
        for ps in persona_subs:
            cv_matrix.loc[ps, target_sub] = cvr

    # Find common subscales
    common = [s for s in ALL_SUBSCALES if s in cos_sim.index and s in cv_matrix.index]
    n = len(common)
    logger.info(f"  Common subscales: {n}")

    if n < 4:
        logger.info("  Not enough common subscales")
        return

    # Extract upper triangle (excluding diagonal and same-scale pairs)
    cos_vals = []
    cv_vals = []
    pair_labels = []

    for i, si in enumerate(common):
        for j, sj in enumerate(common):
            if i >= j:
                continue
            # Skip same-scale pairs
            if SUB_TO_SCALE.get(si) == SUB_TO_SCALE.get(sj):
                continue

            c = cos_sim.loc[si, sj]
            # Average the two directions of CV r (si→sj and sj→si)
            v1 = cv_matrix.loc[si, sj]
            v2 = cv_matrix.loc[sj, si]
            if np.isnan(v1) and np.isnan(v2):
                continue
            v = np.nanmean([v1, v2])

            if not np.isnan(c) and not np.isnan(v):
                cos_vals.append(c)
                cv_vals.append(v)
                pair_labels.append(f"{si.split('_',1)[1][:6]}-{sj.split('_',1)[1][:6]}")

    cos_vals = np.array(cos_vals)
    cv_vals = np.array(cv_vals)

    if len(cos_vals) < 5:
        logger.info("  Not enough valid pairs")
        return

    r_bridge, p_bridge = pearsonr(cos_vals, cv_vals)
    slope_bridge, intercept_bridge = np.polyfit(cos_vals, cv_vals, 1)

    logger.info(f"\n  Direction cosine sim → Cross-persona CV r:")
    logger.info(f"    Pearson r:  {r_bridge:.3f} (p = {p_bridge:.4f})")
    logger.info(f"    Slope:      {slope_bridge:.3f}")
    logger.info(f"    N pairs:    {len(cos_vals)} (between-scale only)")

    if r_bridge > 0.3 and p_bridge < 0.05:
        logger.info(f"    → Representational geometry PREDICTS inference performance")
    else:
        logger.info(f"    → Weak or no link between geometry and inference")

    # Also compare: does direction cosine sim predict cross-persona CV r
    # BETTER than human correlation predicts CV r?
    human_corr = compute_human_correlation_matrix("subscale")
    human_vals = []
    for i, si in enumerate(common):
        for j, sj in enumerate(common):
            if i >= j or SUB_TO_SCALE.get(si) == SUB_TO_SCALE.get(sj):
                continue
            if si in human_corr.index and sj in human_corr.columns:
                h = human_corr.loc[si, sj]
                if not np.isnan(h):
                    human_vals.append(h)
                else:
                    human_vals.append(np.nan)
            else:
                human_vals.append(np.nan)

    # Align with cos_vals/cv_vals (they should be same length since we skipped nans)
    # Recompute to ensure alignment
    cos_v2, cv_v2, hum_v2 = [], [], []
    for i, si in enumerate(common):
        for j, sj in enumerate(common):
            if i >= j or SUB_TO_SCALE.get(si) == SUB_TO_SCALE.get(sj):
                continue
            c = cos_sim.loc[si, sj]
            v1 = cv_matrix.loc[si, sj]
            v2 = cv_matrix.loc[sj, si]
            v = np.nanmean([v1, v2]) if not (np.isnan(v1) and np.isnan(v2)) else np.nan
            h = human_corr.loc[si, sj] if (si in human_corr.index and sj in human_corr.columns) else np.nan
            if not any(np.isnan([c, v, h])):
                cos_v2.append(c)
                cv_v2.append(v)
                hum_v2.append(h)

    cos_v2, cv_v2, hum_v2 = np.array(cos_v2), np.array(cv_v2), np.array(hum_v2)

    if len(cos_v2) >= 5:
        r_hum_cv, p_hum_cv = pearsonr(hum_v2, cv_v2)
        r_cos_cv, p_cos_cv = pearsonr(cos_v2, cv_v2)

        logger.info(f"\n  Comparison: what predicts inference performance better?")
        logger.info(f"    Direction cosine sim → CV r:  r = {r_cos_cv:.3f} (p = {p_cos_cv:.4f})")
        logger.info(f"    Human correlation → CV r:     r = {r_hum_cv:.3f} (p = {p_hum_cv:.4f})")

    # Plot
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(cos_vals, cv_vals, s=30, alpha=0.5, edgecolors="white")
    xx = np.linspace(cos_vals.min() - 0.02, cos_vals.max() + 0.02, 100)
    ax.plot(xx, slope_bridge * xx + intercept_bridge, "k--", lw=1,
            label=f"slope={slope_bridge:.2f}, r={r_bridge:.3f}")
    ax.set_xlabel("Contrastive direction cosine similarity (mega-prompt)")
    ax.set_ylabel("Cross-persona inference CV r (single-scale)")
    ax.set_title(f"Bridging: Representational Geometry → Inference\n"
                 f"r = {r_bridge:.3f}, p = {p_bridge:.4f}")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"bridging_analysis_L{default_layer}.png", dpi=200, bbox_inches="tight")
    plt.close()

    # Save
    bridge_results = {
        "r_bridge": r_bridge, "p_bridge": p_bridge,
        "slope_bridge": slope_bridge, "n_pairs": len(cos_vals),
    }
    with open(output_dir / f"bridging_results_L{default_layer}.json", "w") as f:
        json.dump(bridge_results, f, indent=2)


def run_bridging_analysis(output_dir, default_layer):
    """Bridge mega-prompt and single-scale setups.
    
    Test: does contrastive direction cosine similarity (mega-prompt)
    predict cross-persona inference CV r (single-scale)?
    
    If yes → representational geometry is the basis for cross-scale inference.
    """
    logger.info("\n" + "="*60)
    logger.info("  BRIDGING ANALYSIS: Direction Geometry → Inference")
    logger.info("="*60)

    from psychometric_inference.mechanisms.geometry import compute_human_correlation_matrix, mantel_test

    # Look for contrastive directions in RESULTS_DIR first, then results_llama8b_instruct
    candidate_roots = [RESULTS_DIR]
    llama8b_root = MECH_ROOT / "results_llama8b_instruct"
    if llama8b_root.exists() and llama8b_root != RESULTS_DIR:
        candidate_roots.append(llama8b_root)

    cos_path = None
    for root in candidate_roots:
        for layer_try in [default_layer, 16, 18, 14, 12, 20, 22, 6, 8, 10]:
            test_path = root / "geometry" / f"subscale_cosine_sim_L{layer_try}.csv"
            if test_path.exists():
                cos_path = test_path
                logger.info(f"  Using contrastive directions from: {cos_path}")
                break
        if cos_path is not None:
            break

    if cos_path is None:
        logger.info(f"  No contrastive direction cosine sim found. Searched in: {[str(r/'geometry') for r in candidate_roots]}")
        return

    cos_sim = pd.read_csv(cos_path, index_col=0)
    logger.info(f"  Loaded contrastive cosine sim: {cos_sim.shape}")

    # Load cross-persona CV r (from single-scale setup)
    cv_path = output_dir / f"cross_persona_cv_rs_L{default_layer}.csv"
    if not cv_path.exists():
        logger.info("  No cross-persona CV results found, skip")
        return

    cv_df = pd.read_csv(cv_path)
    human_corr = compute_human_correlation_matrix("subscale")

    # Build cross-persona CV r matrix at subscale level
    # For each (source_subscale, target_subscale) pair where source and target
    # are from different scales, we have the CV r from the rotation where
    # source's parent scale is the persona.
    cv_matrix = pd.DataFrame(np.nan, index=ALL_SUBSCALES, columns=ALL_SUBSCALES)

    for _, row in cv_df.iterrows():
        persona = row["persona"]  # scale short name
        target_sub = row["target_subscale"]
        cv_r = row["cv_r"]

        # Source subscales = all subscales belonging to persona scale
        source_subs = [s for s in ALL_SUBSCALES if SUB_TO_SCALE.get(s) == persona]
        for src in source_subs:
            cv_matrix.loc[src, target_sub] = cv_r

    # Find common subscales across all three matrices
    common = [s for s in ALL_SUBSCALES
              if s in cos_sim.index and s in cv_matrix.index and s in human_corr.index]
    n = len(common)
    logger.info(f"  Common subscales: {n}")

    if n < 4:
        logger.info("  Not enough common subscales")
        return

    idx = np.triu_indices(n, k=1)

    cos_vec = cos_sim.loc[common, common].values[idx]
    cv_vec = cv_matrix.loc[common, common].values[idx]
    hum_vec = human_corr.loc[common, common].values[idx]

    # Only use cross-scale pairs (different parent scales)
    cross_scale_mask = np.array([
        SUB_TO_SCALE.get(common[i]) != SUB_TO_SCALE.get(common[j])
        for i, j in zip(*np.triu_indices(n, k=1))
    ])

    valid = cross_scale_mask & ~(np.isnan(cos_vec) | np.isnan(cv_vec) | np.isnan(hum_vec))
    logger.info(f"  Cross-scale pairs: {cross_scale_mask.sum()}, valid: {valid.sum()}")

    if valid.sum() < 10:
        logger.info("  Not enough valid pairs")
        return

    cos_v = cos_vec[valid]
    cv_v = cv_vec[valid]
    hum_v = hum_vec[valid]

    # 1. Direction cosine sim ↔ Cross-persona CV r
    r_cos_cv, p_cos_cv = pearsonr(cos_v, cv_v)

    # 2. Direction cosine sim ↔ Human correlation
    r_cos_hum, p_cos_hum = pearsonr(cos_v, hum_v)

    # 3. Cross-persona CV r ↔ Human correlation
    r_cv_hum, p_cv_hum = pearsonr(cv_v, hum_v)

    # 4. Partial correlation: direction cosine sim ↔ CV r, controlling for human correlation
    # (Does direction geometry predict inference BEYOND what human correlation predicts?)
    from psychometric_inference.mechanisms.causality import partial_corr
    r_partial, p_partial = partial_corr(cos_v, cv_v, hum_v.reshape(-1, 1))

    logger.info(f"\n  Zero-order correlations ({valid.sum()} cross-scale pairs):")
    logger.info(f"    Direction cosine sim ↔ CV r:        r = {r_cos_cv:.3f} (p = {p_cos_cv:.4f})")
    logger.info(f"    Direction cosine sim ↔ Human corr:  r = {r_cos_hum:.3f} (p = {p_cos_hum:.4f})")
    logger.info(f"    CV r ↔ Human corr:                  r = {r_cv_hum:.3f} (p = {p_cv_hum:.4f})")
    logger.info(f"\n  Partial correlation:")
    logger.info(f"    Direction cosine sim ↔ CV r | Human: r = {r_partial:.3f} (p = {p_partial:.4f})")

    if r_cos_cv > 0.3 and p_cos_cv < 0.05:
        logger.info(f"\n    → Representational geometry PREDICTS cross-scale inference")
    if r_partial > 0.1 and p_partial < 0.05:
        logger.info(f"    → Direction geometry adds information BEYOND human correlation")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    ax = axes[0]
    ax.scatter(cos_v, cv_v, s=20, alpha=0.4)
    sl, ic = np.polyfit(cos_v, cv_v, 1)
    xx = np.linspace(cos_v.min(), cos_v.max(), 100)
    ax.plot(xx, sl * xx + ic, "k--", lw=1)
    ax.set_xlabel("Direction cosine sim (mega-prompt)")
    ax.set_ylabel("Cross-persona CV r (single-scale)")
    ax.set_title(f"Geometry → Inference\nr = {r_cos_cv:.3f}")

    ax = axes[1]
    ax.scatter(hum_v, cos_v, s=20, alpha=0.4)
    sl2, ic2 = np.polyfit(hum_v, cos_v, 1)
    xx2 = np.linspace(hum_v.min(), hum_v.max(), 100)
    ax.plot(xx2, sl2 * xx2 + ic2, "k--", lw=1)
    ax.set_xlabel("Human correlation")
    ax.set_ylabel("Direction cosine sim")
    ax.set_title(f"Human → Geometry\nr = {r_cos_hum:.3f}")

    ax = axes[2]
    ax.scatter(hum_v, cv_v, s=20, alpha=0.4)
    sl3, ic3 = np.polyfit(hum_v, cv_v, 1)
    ax.plot(xx2, sl3 * xx2 + ic3, "k--", lw=1)
    ax.set_xlabel("Human correlation")
    ax.set_ylabel("Cross-persona CV r")
    ax.set_title(f"Human → Inference\nr = {r_cv_hum:.3f}")

    plt.suptitle(f"Bridging Analysis (Layer {default_layer})\n"
                 f"Partial r (geometry → inference | human) = {r_partial:.3f}, p = {p_partial:.4f}",
                 fontsize=11)
    plt.tight_layout()
    plt.savefig(output_dir / f"bridging_analysis_L{default_layer}.png", dpi=200, bbox_inches="tight")
    plt.close()

    # Save
    results = {
        "layer": default_layer,
        "n_pairs": int(valid.sum()),
        "r_cos_cv": r_cos_cv, "p_cos_cv": p_cos_cv,
        "r_cos_hum": r_cos_hum, "p_cos_hum": p_cos_hum,
        "r_cv_hum": r_cv_hum, "p_cv_hum": p_cv_hum,
        "r_partial": r_partial, "p_partial": p_partial,
    }
    with open(output_dir / f"bridging_results_L{default_layer}.json", "w") as f:
        json.dump(results, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--analyze_only", action="store_true")
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--llm_root", type=str, default=None,
                        help="Path to behavioral LLM response data for comparison")
    parser.add_argument("--n_perms", type=int, default=100)
    parser.add_argument("--bridging_only", action="store_true")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory. Default: results_{model_short}/cross_persona")
    args = parser.parse_args()

    # Model-specific defaults
    MODEL_CONFIGS = model_config_by_hf_id()
    cfg = MODEL_CONFIGS.get(args.model_id)
    if cfg:
        model_short, n_layers, target_layers, default_layer = cfg
    else:
        model_short = args.model_id.split("/")[-1].lower().replace("-", "_")
        from psychometric_inference.mechanisms.config import TARGET_LAYERS, DEFAULT_LAYER
        target_layers = TARGET_LAYERS
        default_layer = DEFAULT_LAYER

    if args.layer:
        default_layer = args.layer

    # Output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = MECH_ROOT / f"results_{model_short}" / "cross_persona"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Model: {args.model_id} ({model_short})")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Target layers: {target_layers}, default: {default_layer}")

    if args.bridging_only:
        run_bridging_analysis(output_dir, default_layer)
        return

    if not args.analyze_only:
        run_extraction(args.model_id, output_dir, target_layers)

    run_analysis(output_dir, target_layers, default_layer)
    run_predicted_correlation_analysis(output_dir, default_layer, llm_root=args.llm_root)
    run_permutation_test(output_dir, default_layer, n_perms=args.n_perms)
    run_layerwise_analysis(output_dir, target_layers)
    run_asymmetry_analysis(output_dir, default_layer, llm_root=args.llm_root)
    run_bridging_analysis(output_dir, default_layer)


if __name__ == "__main__":
    main()
