#!/usr/bin/env python3
"""Activation-geometry controls for factorial/profile-level manipulations.

This module complements the behavioral factorial controls. It extracts
decision-point activations under controlled source-scale profiles and tests
whether internal activation geometry tracks the manipulated profile dimensions.
"""

from __future__ import annotations

import argparse
import json
import logging
from itertools import product
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from tqdm import tqdm

from psychometric_inference.model_registry import MODEL_BY_MECH_NAME, ModelSpec
from psychometric_inference.paths import MECHANISTIC_OUTPUT_DIR, questionnaire_path
from psychometric_inference.pipeline.injectors import ScaleAProfile, build_system_prompt
from psychometric_inference.questionnaire_prompts import format_item_prompt
from psychometric_inference.scoring import FILENAME_TO_SCALE, SCORING_RULES

from .activation_model import ActivationModel


logger = logging.getLogger(__name__)

SCALES = [
    ("IRI", "IRI"),
    ("PANAS", "PANAS"),
    ("POM", "POM"),
    ("big_five", "BigFive"),
    ("in_inter_dependent", "SelfConst"),
    ("Life_Satisfaction", "LifeSat"),
    ("Loneliness", "Lonely"),
]

DEFAULT_MODEL = "qwen14b_instruct"
DEFAULT_SOURCE_SCALE = "big_five"
DEFAULT_MODE = "factorial"
OUTPUT_ROOT = MECHANISTIC_OUTPUT_DIR / "factorial_geometry"


def load_scale_definition(scale_file: str) -> tuple[dict, list[dict]]:
    path = questionnaire_path(scale_file)
    metadata = None
    items = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            if obj.get("_metadata"):
                metadata = obj
            else:
                items.append(obj)
    if metadata is None:
        raise ValueError(f"Missing metadata row in {path}")
    return metadata, items


def scale_short_for(scale_file: str) -> str:
    if scale_file not in FILENAME_TO_SCALE:
        raise ValueError(f"Unknown source scale {scale_file!r}")
    return FILENAME_TO_SCALE[scale_file]


def generate_random_profiles(
    scale_def: dict,
    items: list[dict],
    n_profiles: int,
    seed: int,
) -> list[ScaleAProfile]:
    rng = np.random.default_rng(seed)
    profiles = []
    for i in range(n_profiles):
        responses = []
        for item in items:
            response = int(rng.choice(item["response_options"]))
            responses.append({
                "item_number": item["item_number"],
                "text": item["text"],
                "response": response,
            })
        profiles.append(make_profile(f"RND{i:04d}", scale_def, responses))
    return profiles


def generate_factorial_profiles(
    scale_file: str,
    scale_def: dict,
    items: list[dict],
    levels: Iterable[int] | None = None,
) -> list[ScaleAProfile]:
    scale_short = scale_short_for(scale_file)
    rules = SCORING_RULES[scale_short]
    lo, hi = scale_def["response_range"]
    mid = (lo + hi) // 2
    levels_to_use = list(levels) if levels is not None else [lo, mid, hi]

    subscale_items = rules["subscales"]
    subscales = list(subscale_items)
    item_to_subscale = {
        item_number: subscale
        for subscale, item_numbers in subscale_items.items()
        for item_number in item_numbers
    }

    profiles = []
    for combo_idx, combo in enumerate(product(levels_to_use, repeat=len(subscales))):
        level_by_subscale = dict(zip(subscales, combo))
        responses = []
        for item in items:
            subscale = item_to_subscale.get(item["item_number"], subscales[0])
            responses.append({
                "item_number": item["item_number"],
                "text": item["text"],
                "response": int(level_by_subscale[subscale]),
            })
        desc = "_".join(f"{subscale}={level_by_subscale[subscale]}" for subscale in subscales)
        profiles.append(make_profile(f"FAC{combo_idx:04d}_{desc}", scale_def, responses))
    return profiles


def generate_collapsed_profiles(scale_def: dict, items: list[dict]) -> list[ScaleAProfile]:
    lo, hi = scale_def["response_range"]
    mid = (lo + hi) // 2
    profiles = []
    for value, label in [(lo, "low"), (mid, "mid"), (hi, "high")]:
        responses = [
            {"item_number": item["item_number"], "text": item["text"], "response": int(value)}
            for item in items
        ]
        profiles.append(make_profile(f"COLL_{label}_{value}", scale_def, responses))
    return profiles


def make_profile(subject_id: str, scale_def: dict, item_responses: list[dict]) -> ScaleAProfile:
    return ScaleAProfile(
        subject_id=subject_id,
        scale_name=scale_def.get("scale_name_zh", scale_def["scale_name"]),
        instruction=scale_def["instruction"],
        response_labels=scale_def["response_labels"],
        items=item_responses,
    )


def generate_profiles(
    scale_file: str,
    mode: str,
    n_random: int,
    seed: int,
    levels: Iterable[int] | None = None,
) -> list[ScaleAProfile]:
    scale_def, items = load_scale_definition(scale_file)
    if mode == "factorial":
        return generate_factorial_profiles(scale_file, scale_def, items, levels)
    if mode == "random":
        return generate_random_profiles(scale_def, items, n_random, seed)
    if mode == "collapsed":
        return generate_collapsed_profiles(scale_def, items)
    raise ValueError(f"Unknown mode: {mode}")


def profile_features(profile: ScaleAProfile, scale_file: str) -> dict:
    scale_short = scale_short_for(scale_file)
    rules = SCORING_RULES[scale_short]
    responses = {item["item_number"]: float(item["response"]) for item in profile.items}
    row = {"profile_id": profile.subject_id}
    for subscale, item_numbers in rules["subscales"].items():
        vals = [responses[item_number] for item_number in item_numbers if item_number in responses]
        row[f"{scale_short}_{subscale}"] = float(np.mean(vals)) if vals else np.nan
    for item in profile.items:
        row[f"Q{item['item_number']}"] = item["response"]
    return row


def target_subscale_for(scale_short: str, item_number: int) -> str:
    for subscale, item_numbers in SCORING_RULES[scale_short]["subscales"].items():
        if item_number in item_numbers:
            return f"{scale_short}_{subscale}"
    return scale_short


def select_target_items(
    source_scale_file: str,
    target_items_per_subscale: int,
    include_source_scale: bool = False,
) -> list[dict]:
    source_short = scale_short_for(source_scale_file)
    selected = []
    for scale_file, scale_short in SCALES:
        if scale_short == source_short and not include_source_scale:
            continue
        meta, items = load_scale_definition(scale_file)
        by_number = {item["item_number"]: item for item in items}
        for subscale, item_numbers in SCORING_RULES[scale_short]["subscales"].items():
            for item_number in item_numbers[:target_items_per_subscale]:
                item = by_number.get(item_number)
                if item is None:
                    continue
                selected.append({
                    "scale_file": scale_file,
                    "scale_short": scale_short,
                    "target_subscale": f"{scale_short}_{subscale}",
                    "item": item,
                    "instruction": meta["instruction"],
                })
    return selected


def output_dir_for(model: ModelSpec, source_scale: str, mode: str) -> Path:
    return OUTPUT_ROOT / model.mech_name / f"{source_scale}_{mode}"


def parse_layers(model: ModelSpec, layers: list[int] | None, all_layers: bool) -> list[int]:
    if layers:
        return layers
    if all_layers:
        return list(model.target_layers)
    return [model.default_layer]


def run_extraction(
    model_spec: ModelSpec,
    source_scale: str,
    mode: str,
    layers: list[int],
    target_items_per_subscale: int,
    n_random: int,
    max_profiles: int | None,
    seed: int,
    device: str,
    prompt_format: str,
    system_prompt_variant: str,
    include_source_scale: bool,
    skip_existing: bool,
) -> Path:
    outdir = output_dir_for(model_spec, source_scale, mode)
    done_marker = outdir / ".extraction_done"
    if skip_existing and done_marker.exists():
        logger.info("Factorial geometry extraction already done: %s", outdir)
        return outdir

    profiles = generate_profiles(source_scale, mode, n_random=n_random, seed=seed)
    if max_profiles is not None:
        profiles = profiles[:max_profiles]
    targets = select_target_items(source_scale, target_items_per_subscale, include_source_scale)

    outdir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([profile_features(profile, source_scale) for profile in profiles]).to_csv(
        outdir / "profiles.csv",
        index=False,
    )
    (outdir / "run_config.json").write_text(
        json.dumps({
            "model": model_spec.mech_name,
            "hf_id": model_spec.hf_id,
            "source_scale": source_scale,
            "mode": mode,
            "layers": layers,
            "target_items_per_subscale": target_items_per_subscale,
            "n_profiles": len(profiles),
            "n_target_items": len(targets),
            "prompt_format": prompt_format,
            "system_prompt_variant": system_prompt_variant,
            "include_source_scale": include_source_scale,
        }, indent=2),
        encoding="utf-8",
    )

    model = ActivationModel(model_spec.hf_id, device=device, prompt_format=prompt_format)
    activations = {layer: [] for layer in layers}
    meta_rows = []

    pbar = tqdm(total=len(profiles) * len(targets), desc=f"{model_spec.mech_name}/{source_scale}/{mode}", unit="item")
    try:
        for profile in profiles:
            system_prompt = build_system_prompt(profile, system_prompt_variant)
            for target in targets:
                item_prompt = format_item_prompt(target["item"], target["instruction"])
                acts = model.extract_prompt_activation(system_prompt, item_prompt, layers=layers)
                row_idx = len(meta_rows)
                meta_rows.append({
                    "row_idx": row_idx,
                    "profile_id": profile.subject_id,
                    "target_scale": target["scale_short"],
                    "target_subscale": target["target_subscale"],
                    "item_number": target["item"]["item_number"],
                })
                for layer in layers:
                    if layer in acts:
                        activations[layer].append(acts[layer])
                pbar.update(1)
    finally:
        pbar.close()
        model.cleanup()

    np.savez_compressed(
        outdir / "activations.npz",
        **{f"L{layer}": np.asarray(vals, dtype=np.float32) for layer, vals in activations.items()},
    )
    pd.DataFrame(meta_rows).to_csv(outdir / "meta.csv", index=False)
    done_marker.touch()
    return outdir


def cosine_matrix(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return (x @ x.T) / (norms @ norms.T + 1e-10)


def mantel_test(
    matrix_a: np.ndarray,
    matrix_b: np.ndarray,
    n_permutations: int,
    seed: int,
) -> tuple[float, float]:
    """Mantel-style matrix permutation test using upper-triangle entries."""
    if matrix_a.shape != matrix_b.shape:
        raise ValueError(f"Matrix shape mismatch: {matrix_a.shape} vs {matrix_b.shape}")
    n = matrix_a.shape[0]
    idx = np.triu_indices(n, k=1)
    a_vec = matrix_a[idx]
    b_vec = matrix_b[idx]
    valid = ~(np.isnan(a_vec) | np.isnan(b_vec))
    a_vec = a_vec[valid]
    b_vec = b_vec[valid]
    if len(a_vec) < 3:
        return np.nan, np.nan

    observed = float(np.corrcoef(a_vec, b_vec)[0, 1])
    rng = np.random.default_rng(seed)
    n_greater = 0
    for _ in range(n_permutations):
        perm = rng.permutation(n)
        a_perm = matrix_a[np.ix_(perm, perm)][idx][valid]
        r_perm = np.corrcoef(a_perm, b_vec)[0, 1]
        if r_perm >= observed:
            n_greater += 1
    return observed, (n_greater + 1) / (n_permutations + 1)


def feature_similarity(features: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    y = features[feature_cols].to_numpy(dtype=float)
    y = (y - np.nanmean(y, axis=0)) / (np.nanstd(y, axis=0) + 1e-10)
    sim = cosine_matrix(y)
    return pd.DataFrame(sim, index=features["profile_id"], columns=features["profile_id"])


def profile_level_activations(acts: np.ndarray, meta: pd.DataFrame, profile_ids: list[str]) -> np.ndarray:
    rows = []
    for pid in profile_ids:
        idx = meta.index[meta["profile_id"] == pid].to_numpy()
        rows.append(acts[idx].mean(axis=0))
    return np.asarray(rows, dtype=np.float32)


def ridge_predict_cv(
    x: np.ndarray,
    y: np.ndarray,
    alphas: list[float],
    n_splits: int,
    seed: int,
) -> tuple[np.ndarray, float]:
    rng = np.random.default_rng(seed)
    valid = ~(np.isnan(y) | np.any(np.isnan(x), axis=1))
    x_valid = x[valid]
    y_valid = y[valid]
    preds_valid = np.full(len(y_valid), np.nan)
    if len(y_valid) < max(4, n_splits):
        return np.full(len(y), np.nan), np.nan

    order = rng.permutation(len(y_valid))
    folds = np.array_split(order, n_splits)
    best_alpha = alphas[0]
    best_score = -np.inf
    preds_by_alpha = {}

    for alpha in alphas:
        alpha_preds = np.full(len(y_valid), np.nan)
        for fold in folds:
            train = np.setdiff1d(order, fold, assume_unique=False)
            x_train, x_test = x_valid[train], x_valid[fold]
            y_train = y_valid[train]

            x_mean = x_train.mean(axis=0, keepdims=True)
            x_std = x_train.std(axis=0, keepdims=True) + 1e-8
            y_mean = y_train.mean()
            y_std = y_train.std() + 1e-8

            xt = (x_train - x_mean) / x_std
            xv = (x_test - x_mean) / x_std
            yt = (y_train - y_mean) / y_std

            kernel = xt @ xt.T
            dual = np.linalg.solve(kernel + alpha * np.eye(len(train)), yt)
            alpha_preds[fold] = (xv @ xt.T @ dual) * y_std + y_mean

        score = np.corrcoef(y_valid, alpha_preds)[0, 1] if np.isfinite(alpha_preds).all() else -np.inf
        preds_by_alpha[alpha] = alpha_preds
        if np.isfinite(score) and score > best_score:
            best_alpha = alpha
            best_score = score

    preds = np.full(len(y), np.nan)
    preds[valid] = preds_by_alpha[best_alpha]
    return preds, float(best_alpha)


def analyze(outdir: Path, layers: list[int], n_permutations: int, seed: int) -> pd.DataFrame:
    act_path = outdir / "activations.npz"
    meta_path = outdir / "meta.csv"
    profiles_path = outdir / "profiles.csv"
    if not act_path.exists() or not meta_path.exists() or not profiles_path.exists():
        raise FileNotFoundError(f"Missing factorial geometry artifacts in {outdir}")

    acts_npz = np.load(act_path)
    meta = pd.read_csv(meta_path)
    profiles = pd.read_csv(profiles_path)
    profile_ids = profiles["profile_id"].astype(str).tolist()
    feature_cols = [c for c in profiles.columns if c not in {"profile_id"} and not c.startswith("Q")]
    if not feature_cols:
        raise ValueError("No source-profile feature columns found in profiles.csv")

    feature_sim = feature_similarity(profiles, feature_cols)
    feature_sim.to_csv(outdir / "profile_feature_similarity.csv")

    summary_rows = []
    for layer in layers:
        key = f"L{layer}"
        if key not in acts_npz:
            logger.warning("Layer %s missing from %s", layer, act_path)
            continue
        profile_acts = profile_level_activations(acts_npz[key], meta, profile_ids)
        centered = profile_acts - profile_acts.mean(axis=0, keepdims=True)
        act_sim = pd.DataFrame(
            cosine_matrix(centered),
            index=profile_ids,
            columns=profile_ids,
        )
        act_sim.to_csv(outdir / f"activation_profile_similarity_L{layer}.csv")

        mantel_r, mantel_p = mantel_test(
            act_sim.to_numpy(),
            feature_sim.to_numpy(),
            n_permutations=n_permutations,
            seed=seed,
        )

        readout_rows = []
        x = centered
        for feature in feature_cols:
            y = profiles[feature].to_numpy(dtype=float)
            preds, best_alpha = ridge_predict_cv(
                x,
                y,
                alphas=[0.1, 1.0, 10.0, 100.0, 1000.0],
                n_splits=min(5, len(profile_ids)),
                seed=seed,
            )
            valid = ~(np.isnan(y) | np.isnan(preds))
            if valid.sum() >= 3:
                r, p = pearsonr(y[valid], preds[valid])
                ss_res = np.sum((y[valid] - preds[valid]) ** 2)
                ss_tot = np.sum((y[valid] - y[valid].mean()) ** 2)
                r2 = np.nan if ss_tot == 0 else 1 - ss_res / ss_tot
            else:
                r, p, r2 = np.nan, np.nan, np.nan
            readout_rows.append({
                "layer": layer,
                "feature": feature,
                "cv_r": r,
                "cv_p": p,
                "cv_r2": r2,
                "best_alpha": best_alpha,
                "n_profiles": int(valid.sum()),
            })

        readout_df = pd.DataFrame(readout_rows)
        readout_df.to_csv(outdir / f"factor_readout_L{layer}.csv", index=False)

        summary_rows.append({
            "layer": layer,
            "n_profiles": len(profile_ids),
            "n_target_observations": len(meta),
            "n_profile_features": len(feature_cols),
            "profile_geometry_mantel_r": mantel_r,
            "profile_geometry_mantel_p": mantel_p,
            "mean_factor_readout_r": readout_df["cv_r"].mean(),
            "mean_factor_readout_r2": readout_df["cv_r2"].mean(),
            "feature_columns": ";".join(feature_cols),
        })

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(outdir / "geometry_summary.csv", index=False)
    return summary


def dry_run(
    model: ModelSpec,
    source_scale: str,
    mode: str,
    layers: list[int],
    target_items_per_subscale: int,
    n_random: int,
    max_profiles: int | None,
    seed: int,
    include_source_scale: bool,
) -> None:
    profiles = generate_profiles(source_scale, mode, n_random=n_random, seed=seed)
    if max_profiles is not None:
        profiles = profiles[:max_profiles]
    targets = select_target_items(source_scale, target_items_per_subscale, include_source_scale)
    outdir = output_dir_for(model, source_scale, mode)
    print("=" * 72)
    print("  FACTORIAL / PROFILE GEOMETRY")
    print("=" * 72)
    print(f"model:             {model.mech_name} ({model.hf_id})")
    print(f"source_scale:      {source_scale}")
    print(f"mode:              {mode}")
    print(f"layers:            {layers}")
    print(f"profiles:          {len(profiles)}")
    print(f"target items:      {len(targets)}")
    print(f"forward passes:    {len(profiles) * len(targets)}")
    print(f"output_dir:        {outdir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run factorial/profile activation-geometry controls")
    parser.add_argument("--model", default=DEFAULT_MODEL, choices=sorted(MODEL_BY_MECH_NAME))
    parser.add_argument("--source_scale", default=DEFAULT_SOURCE_SCALE, choices=[scale for scale, _ in SCALES])
    parser.add_argument("--mode", default=DEFAULT_MODE, choices=["factorial", "random", "collapsed"])
    parser.add_argument("--layers", nargs="+", type=int, default=None)
    parser.add_argument("--all_layers", action="store_true")
    parser.add_argument("--target_items_per_subscale", type=int, default=1)
    parser.add_argument("--n_random", type=int, default=100)
    parser.add_argument("--max_profiles", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--prompt_format", choices=["raw", "chat_template"], default="raw")
    parser.add_argument("--system_prompt_variant", choices=["questionnaire", "questionnaire_no_scale_name"], default="questionnaire")
    parser.add_argument("--include_source_scale", action="store_true")
    parser.add_argument("--n_permutations", type=int, default=1000)
    parser.add_argument("--extract_only", action="store_true")
    parser.add_argument("--analysis_only", action="store_true")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    model = MODEL_BY_MECH_NAME[args.model]
    layers = parse_layers(model, args.layers, args.all_layers)

    if args.dry_run:
        dry_run(
            model,
            args.source_scale,
            args.mode,
            layers,
            args.target_items_per_subscale,
            args.n_random,
            args.max_profiles,
            args.seed,
            args.include_source_scale,
        )
        return

    outdir = output_dir_for(model, args.source_scale, args.mode)
    if not args.analysis_only:
        outdir = run_extraction(
            model,
            args.source_scale,
            args.mode,
            layers,
            args.target_items_per_subscale,
            args.n_random,
            args.max_profiles,
            args.seed,
            args.device,
            args.prompt_format,
            args.system_prompt_variant,
            args.include_source_scale,
            args.skip_existing,
        )

    if not args.extract_only:
        summary = analyze(outdir, layers, n_permutations=args.n_permutations, seed=args.seed)
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
