#!/usr/bin/env python3
"""Steering experiments for psychometric directions.

The default path reproduces the regression-direction steering analysis:

    python -m psychometric_inference.mechanisms.steering --extract_only
    python -m psychometric_inference.mechanisms.steering --steering

The same runner can also evaluate contrastive directions or alternate
interventions without editing source code, for example:

    python -m psychometric_inference.mechanisms.steering \
        --direction_source contrastive \
        --steering \
        --alphas -1 0 1 \
        --target_items_per_subscale 1
"""

from __future__ import annotations

import argparse
import gc
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from psychometric_inference.questionnaire_prompts import format_item_prompt
from psychometric_inference.scoring import SCORING_RULES

from .config import ALL_SUBSCALES, DEFAULT_LAYER, MODEL_ID, RESULTS_DIR, SCALES

REGRESSION_OUTPUT_DIR = RESULTS_DIR / "regression_directions"
DEFAULT_ALPHAS = (-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0)


@dataclass(frozen=True)
class SteeringConfig:
    """Runtime choices that define one steering grid."""

    direction_source: str = "regression"
    intervention: str = "last_token_add"
    alphas: tuple[float, ...] = DEFAULT_ALPHAS
    target_items_per_subscale: int = 2
    system_prompt_variant: str = "default"

    @property
    def output_prefix(self) -> str:
        return self.direction_source


def parse_alphas(values: Iterable[float] | None) -> tuple[float, ...]:
    if values is None:
        return DEFAULT_ALPHAS
    alphas = tuple(float(v) for v in values)
    if len(alphas) < 3:
        raise ValueError("Steering needs at least three alpha values to estimate a slope.")
    return alphas


def extract_regression_directions(layer: int) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Fit ridge regression for each subscale and return normalized weights."""
    from scipy.stats import pearsonr
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import cross_val_predict

    act_path = RESULTS_DIR / "pca" / "subject_activations.npz"
    scores_path = RESULTS_DIR / "pca" / "human_subscale_scores.csv"

    acts = np.load(act_path)[f"L{layer}"]
    scores_df = pd.read_csv(scores_path)

    directions: dict[str, np.ndarray] = {}
    cv_rs: dict[str, float] = {}

    print(f"Extracting regression-based directions (Layer {layer})")
    print(f"  Data: {acts.shape[0]} subjects x {acts.shape[1]} dims")

    for subscale in ALL_SUBSCALES:
        if subscale not in scores_df.columns:
            continue

        y = scores_df[subscale].values
        valid = ~np.isnan(y)
        if valid.sum() < 20:
            continue

        x_valid = acts[valid]
        y_valid = y[valid]

        ridge = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0, 1000.0], cv=5)
        ridge.fit(x_valid, y_valid)

        weights = ridge.coef_
        norm = np.linalg.norm(weights)
        if norm == 0:
            continue
        directions[subscale] = weights / norm

        y_pred = cross_val_predict(ridge, x_valid, y_valid, cv=5)
        r, _ = pearsonr(y_valid, y_pred)
        cv_rs[subscale] = float(r)
        print(f"    {subscale:<35} CV r = {r:.3f}, ridge alpha = {ridge.alpha_:.1f}")

    return directions, cv_rs


def load_contrastive_directions(layer: int) -> dict[str, np.ndarray]:
    """Load contrastive high-minus-low directions extracted by directions.py."""
    from .geometry import load_directions

    path = RESULTS_DIR / "directions" / "subscale_directions.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing contrastive directions: {path}")
    return load_directions(str(path), ALL_SUBSCALES, layer)


def compute_cosine_matrix(directions: dict[str, np.ndarray], labels: list[str]) -> pd.DataFrame:
    matrix = np.zeros((len(labels), len(labels)))
    for i, left in enumerate(labels):
        for j, right in enumerate(labels):
            if left in directions and right in directions:
                matrix[i, j] = np.dot(directions[left], directions[right])
            elif i == j:
                matrix[i, j] = 1.0
    return pd.DataFrame(matrix, index=labels, columns=labels)


def save_regression_direction_artifacts(
    directions: dict[str, np.ndarray],
    cv_rs: dict[str, float],
    layer: int,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / f"regression_directions_L{layer}.npz",
        **{subscale: direction for subscale, direction in directions.items()},
    )
    pd.DataFrame({"subscale": list(cv_rs), "cv_r": list(cv_rs.values())}).to_csv(
        output_dir / f"regression_cv_rs_L{layer}.csv",
        index=False,
    )

    labels = [subscale for subscale in ALL_SUBSCALES if subscale in directions]
    cosine = compute_cosine_matrix(directions, labels)
    cosine.to_csv(output_dir / f"regression_cosine_sim_L{layer}.csv")
    summarize_direction_geometry(cosine, labels)


def summarize_direction_geometry(cosine: pd.DataFrame, labels: list[str]) -> None:
    from scipy.stats import pearsonr

    from .geometry import compute_human_correlation_matrix, mantel_test

    human_corr = compute_human_correlation_matrix("subscale")
    common = [label for label in labels if label in human_corr.index]
    if len(common) < 3:
        return

    idx = np.triu_indices(len(common), k=1)
    cosine_vec = cosine.loc[common, common].values[idx]
    human_vec = human_corr.loc[common, common].values[idx]
    valid = ~(np.isnan(cosine_vec) | np.isnan(human_vec))
    if valid.sum() < 3:
        return

    r_p, p_p = pearsonr(human_vec[valid], cosine_vec[valid])
    slope, _ = np.polyfit(human_vec[valid], cosine_vec[valid], 1)
    r_m, p_m = mantel_test(
        cosine.loc[common, common].values,
        human_corr.loc[common, common].values,
    )
    print("\n  Direction cosine similarity vs human:")
    print(f"    Mantel r  = {r_m:.4f} (p = {p_m:.4f})")
    print(f"    Slope     = {slope:.4f}")
    print(f"    Pearson r = {r_p:.4f} (p = {p_p:.6f})")


def build_midpoint_responses(scale_defs: dict[str, dict]) -> dict[str, dict[int, int]]:
    """Create a neutral profile with each questionnaire item at its midpoint."""
    responses: dict[str, dict[int, int]] = {}
    for scale_file, _ in SCALES:
        scale_def = scale_defs[scale_file]
        low, high = scale_def["metadata"]["response_range"]
        midpoint = (low + high) // 2
        responses[scale_file] = {
            item["item_number"]: midpoint for item in scale_def["items"]
        }
    return responses


def collect_target_items(
    scale_defs: dict[str, dict],
    items_per_subscale: int,
) -> dict[str, dict]:
    """Collect questionnaire items used for steering readout."""
    if items_per_subscale < 1:
        raise ValueError("--target_items_per_subscale must be >= 1")

    target_items: dict[str, dict] = {}
    for scale_file, scale_short in SCALES:
        scale_def = scale_defs[scale_file]
        rules = SCORING_RULES.get(scale_short, {})
        for sub_name, item_nums in rules.get("subscales", {}).items():
            subscale = f"{scale_short}_{sub_name}"
            selected = [
                item
                for item in scale_def["items"]
                if item["item_number"] in item_nums[:items_per_subscale]
            ]
            if selected:
                target_items[subscale] = {
                    "items": selected,
                    "instruction": scale_def["metadata"]["instruction"],
                }
    return target_items


def expected_value_from_log_probs(tokenizer, log_probs, options) -> float:
    probs = {}
    for option in options:
        token_ids = tokenizer.encode(str(option), add_special_tokens=False)
        if token_ids:
            probs[option] = math.exp(log_probs[token_ids[0]].item())

    total = sum(probs.values())
    if total <= 0:
        return float("nan")
    return float(sum(option * prob / total for option, prob in probs.items()))


def register_intervention_hook(model, layer: int, direction_tensor, alpha: float, method: str):
    """Register one residual-stream steering hook and return its handle."""

    def hook_fn(module, inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        hidden = hidden.clone()

        if method == "last_token_add":
            hidden[:, -1, :] += alpha * direction_tensor
        elif method == "all_tokens_add":
            hidden += alpha * direction_tensor
        elif method == "projection_ablate":
            unit = direction_tensor / (direction_tensor.norm() + 1e-8)
            projection = (hidden[:, -1, :] @ unit).unsqueeze(-1) * unit
            hidden[:, -1, :] -= alpha * projection
        else:
            raise ValueError(f"Unknown steering intervention: {method}")

        if isinstance(output, tuple):
            return (hidden,) + output[1:]
        return hidden

    return model.model.layers[layer].register_forward_hook(hook_fn)


def run_steering_experiment(
    directions: dict[str, np.ndarray],
    layer: int,
    model_id: str,
    output_dir: Path,
    config: SteeringConfig,
) -> pd.DataFrame:
    """Run a steering grid and return item-level expected-value readouts."""
    import torch
    from tqdm import tqdm
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from .prompts import build_mega_system_prompt, load_all_scale_definitions

    scale_defs = load_all_scale_definitions()
    target_items = collect_target_items(scale_defs, config.target_items_per_subscale)
    readable_targets = [subscale for subscale in ALL_SUBSCALES if subscale in target_items]
    steer_sources = [subscale for subscale in ALL_SUBSCALES if subscale in directions]

    baseline_prompt = build_mega_system_prompt(
        build_midpoint_responses(scale_defs),
        scale_defs,
        variant=config.system_prompt_variant,
    )

    print(f"\nLoading model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="cuda",
    )
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    n_target_items = sum(len(target_items[target]["items"]) for target in readable_targets)
    total = len(steer_sources) * len(config.alphas) * n_target_items
    print(
        f"  {len(steer_sources)} sources x {len(config.alphas)} alphas x "
        f"{n_target_items} items = {total} forward passes"
    )
    print(f"  Intervention: {config.intervention}")

    records = []
    pbar = tqdm(total=total, desc=f"{config.direction_source} steering", unit="item")

    for source in steer_sources:
        direction_tensor = torch.tensor(
            directions[source],
            dtype=torch.float16,
            device="cuda",
        )

        for alpha in config.alphas:
            handle = register_intervention_hook(
                model,
                layer,
                direction_tensor,
                alpha,
                config.intervention,
            )
            try:
                for target in readable_targets:
                    target_config = target_items[target]
                    for item in target_config["items"]:
                        item_prompt = format_item_prompt(item, target_config["instruction"])
                        prompt = f"{baseline_prompt}\n\n{item_prompt}\n\nAnswer: "
                        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

                        with torch.no_grad():
                            outputs = model(**inputs)

                        logits = outputs.logits[0, -1, :]
                        log_probs = torch.log_softmax(logits, dim=-1)
                        ev = expected_value_from_log_probs(
                            tokenizer,
                            log_probs,
                            item["response_options"],
                        )

                        records.append(
                            {
                                "direction_source": config.direction_source,
                                "intervention": config.intervention,
                                "system_prompt_variant": config.system_prompt_variant,
                                "steer_source": source,
                                "alpha": alpha,
                                "target": target,
                                "item_num": item["item_number"],
                                "ev": ev,
                            }
                        )
                        pbar.update(1)
            finally:
                handle.remove()

    pbar.close()
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / f"{config.output_prefix}_steering_raw.csv"
    df = pd.DataFrame(records)
    df.to_csv(raw_path, index=False)
    return df


def summarize_steering_results(
    df: pd.DataFrame,
    output_dir: Path,
    output_prefix: str,
) -> None:
    """Compute steering effect matrices and specificity summaries."""
    from scipy.stats import pearsonr

    from .geometry import compute_human_correlation_matrix, mantel_test

    human_corr = compute_human_correlation_matrix("subscale")
    sources = sorted(df["steer_source"].unique())
    targets = sorted(df["target"].unique())

    effect = pd.DataFrame(index=sources, columns=targets, dtype=float)
    for source in sources:
        for target in targets:
            subset = df[(df["steer_source"] == source) & (df["target"] == target)]
            if subset.empty:
                continue
            averaged = subset.groupby("alpha")["ev"].mean()
            if len(averaged) < 3:
                continue
            effect.loc[source, target] = np.polyfit(averaged.index, averaged.values, 1)[0]

    effect.to_csv(output_dir / f"{output_prefix}_steering_effect.csv")

    raw = effect.values.astype(float)
    pairs = []
    for i, source in enumerate(sources):
        for j, target in enumerate(targets):
            if source == target:
                continue
            if source not in human_corr.index or target not in human_corr.columns:
                continue
            value = raw[i, j]
            if not np.isnan(value):
                pairs.append(
                    {
                        "source": source,
                        "target": target,
                        "human_r": human_corr.loc[source, target],
                        "effect": value,
                    }
                )

    pair_df = pd.DataFrame(pairs)
    if pair_df.empty:
        print("  No valid steering pairs to summarize.")
        return

    r_raw, p_raw = pearsonr(pair_df["human_r"], pair_df["effect"])
    row_means = np.nanmean(raw, axis=1, keepdims=True)
    col_means = np.nanmean(raw, axis=0, keepdims=True)
    centered = raw - row_means - col_means + np.nanmean(raw)

    centered_pairs = []
    for i, source in enumerate(sources):
        for j, target in enumerate(targets):
            if source == target:
                continue
            if source not in human_corr.index or target not in human_corr.columns:
                continue
            value = centered[i, j]
            if not np.isnan(value):
                centered_pairs.append(
                    {"human_r": human_corr.loc[source, target], "effect": value}
                )

    centered_df = pd.DataFrame(centered_pairs)
    r_centered, p_centered = pearsonr(centered_df["human_r"], centered_df["effect"])

    print("\n" + "=" * 60)
    print(f"  {output_prefix.upper()} STEERING SPECIFICITY")
    print("=" * 60)
    print(f"  Raw signed:             r = {r_raw:+.3f} (p = {p_raw:.4f})")
    print(f"  Double-centered signed: r = {r_centered:+.3f} (p = {p_centered:.4f})")

    common = [
        subscale
        for subscale in ALL_SUBSCALES
        if subscale in sources and subscale in targets and subscale in human_corr.index
    ]
    if len(common) >= 4:
        centered_square = pd.DataFrame(centered, index=sources, columns=targets)
        centered_square = centered_square.reindex(index=common, columns=common).values.astype(float)
        centered_sym = np.nan_to_num((centered_square + centered_square.T) / 2)
        human_square = human_corr.reindex(index=common, columns=common).values
        r_m, p_m = mantel_test(centered_sym, human_square)
        print(f"  Mantel (double-centered): r = {r_m:.3f} (p = {p_m:.4f})")

    plot_steering_specificity(
        pair_df,
        centered_df,
        r_raw,
        p_raw,
        r_centered,
        p_centered,
        output_dir / f"{output_prefix}_steering_specificity.png",
    )


def plot_steering_specificity(
    pair_df: pd.DataFrame,
    centered_df: pd.DataFrame,
    r_raw: float,
    p_raw: float,
    r_centered: float,
    p_centered: float,
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.scatter(pair_df["human_r"], pair_df["effect"], alpha=0.4, s=20)
    slope, intercept = np.polyfit(pair_df["human_r"], pair_df["effect"], 1)
    x_min = min(pair_df["human_r"].min(), centered_df["human_r"].min()) - 0.05
    x_max = max(pair_df["human_r"].max(), centered_df["human_r"].max()) + 0.05
    x_values = np.linspace(x_min, x_max, 100)
    ax1.plot(x_values, slope * x_values + intercept, "k--", lw=1)
    ax1.set_title(f"Raw: r={r_raw:.3f}, p={p_raw:.4f}")
    ax1.set_xlabel("Human r")
    ax1.set_ylabel("Steering effect")
    ax1.axhline(y=0, color="gray", ls=":", lw=0.5)
    ax1.axvline(x=0, color="gray", ls=":", lw=0.5)

    ax2.scatter(centered_df["human_r"], centered_df["effect"], alpha=0.4, s=20)
    slope2, intercept2 = np.polyfit(centered_df["human_r"], centered_df["effect"], 1)
    ax2.plot(x_values, slope2 * x_values + intercept2, "k--", lw=1)
    ax2.set_title(f"Double-centered: r={r_centered:.3f}, p={p_centered:.4f}")
    ax2.set_xlabel("Human r")
    ax2.set_ylabel("Steering effect (centered)")
    ax2.axhline(y=0, color="gray", ls=":", lw=0.5)
    ax2.axvline(x=0, color="gray", ls=":", lw=0.5)

    plt.suptitle("Steering Specificity")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def resolve_directions(
    direction_source: str,
    layer: int,
    output_dir: Path,
) -> dict[str, np.ndarray]:
    if direction_source == "regression":
        directions, cv_rs = extract_regression_directions(layer)
        save_regression_direction_artifacts(directions, cv_rs, layer, output_dir)
        return directions
    if direction_source == "contrastive":
        directions = load_contrastive_directions(layer)
        labels = [subscale for subscale in ALL_SUBSCALES if subscale in directions]
        summarize_direction_geometry(compute_cosine_matrix(directions, labels), labels)
        return directions
    raise ValueError(f"Unknown direction source: {direction_source}")


def default_output_dir(direction_source: str) -> Path:
    if direction_source == "regression":
        return REGRESSION_OUTPUT_DIR
    return RESULTS_DIR / "steering" / direction_source


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run psychometric steering experiments")
    parser.add_argument("--layer", type=int, default=DEFAULT_LAYER)
    parser.add_argument("--extract_only", action="store_true")
    parser.add_argument("--steering", action="store_true")
    parser.add_argument("--model_id", type=str, default=MODEL_ID)
    parser.add_argument(
        "--direction_source",
        choices=["regression", "contrastive"],
        default="regression",
    )
    parser.add_argument(
        "--intervention",
        choices=["last_token_add", "all_tokens_add", "projection_ablate"],
        default="last_token_add",
    )
    parser.add_argument("--alphas", nargs="+", type=float, default=list(DEFAULT_ALPHAS))
    parser.add_argument("--target_items_per_subscale", type=int, default=2)
    parser.add_argument(
        "--system_prompt_variant",
        choices=["default", "no_scale_name"],
        default="default",
    )
    parser.add_argument("--output_dir", type=Path, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir = args.output_dir or default_output_dir(args.direction_source)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = SteeringConfig(
        direction_source=args.direction_source,
        intervention=args.intervention,
        alphas=parse_alphas(args.alphas),
        target_items_per_subscale=args.target_items_per_subscale,
        system_prompt_variant=args.system_prompt_variant,
    )

    directions = resolve_directions(args.direction_source, args.layer, output_dir)
    if args.extract_only:
        return

    if args.steering:
        steering_df = run_steering_experiment(
            directions,
            layer=args.layer,
            model_id=args.model_id,
            output_dir=output_dir,
            config=config,
        )
        summarize_steering_results(
            steering_df,
            output_dir=output_dir,
            output_prefix=config.output_prefix,
        )


if __name__ == "__main__":
    main()
