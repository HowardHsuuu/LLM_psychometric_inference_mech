#!/usr/bin/env python3
"""Verify key aggregate claims from the public derived-metric bundle.

This verifier deliberately recomputes checks from released matrices instead of
reading final statistic tables. That keeps ``release_artifacts`` as a compact
derived-metric cache rather than a mirror of ``outputs/``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from psychometric_inference.model_registry import PRIMARY_MODELS
from psychometric_inference.scoring import SCORING_RULES


EXPECTED = {
    "repr_corrected_slope_vs_behavior_slope_r": 0.640,
    "repr_mantel_vs_behavior_slope_r": 0.678,
    "qwen14b_instruct_best_mantel": 0.660,
}

SCALE_NAMES = ["IRI", "PANAS", "POM", "BigFive", "SelfConst", "LifeSat", "Lonely"]

SUB_TO_SCALE: dict[str, str] = {}
for scale_short, rules in SCORING_RULES.items():
    for sub_name in rules["subscales"]:
        SUB_TO_SCALE[f"{scale_short}_{sub_name}"] = scale_short


def close(actual: float, expected: float, tol: float = 0.002) -> bool:
    return abs(actual - expected) <= tol


def check_shape(path: Path, expected_shape: tuple[int, int]) -> None:
    df = pd.read_csv(path)
    if df.shape != expected_shape:
        raise AssertionError(f"{path}: expected shape {expected_shape}, got {df.shape}")
    print(f"OK shape {path}: {df.shape}")


def load_matrix(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, index_col=0)


def _scale_from_item(label: str) -> str | None:
    for scale in SCALE_NAMES:
        if label.startswith(scale + "_"):
            return scale
    return None


def _per_scale_slope(
    human: pd.DataFrame,
    llm: pd.DataFrame,
    scale_for_label,
) -> float:
    common = [c for c in human.columns if c in llm.columns]
    human = human.loc[common, common]
    llm = llm.loc[common, common]

    slopes = []
    for scale in SCALE_NAMES:
        own = [c for c in common if scale_for_label(c) == scale]
        other = [c for c in common if scale_for_label(c) is not None and scale_for_label(c) != scale]
        if not own or not other:
            continue

        h_vals = []
        l_vals = []
        for row in own:
            for col in other:
                hv = human.loc[row, col]
                lv = llm.loc[row, col]
                if not np.isnan(hv) and not np.isnan(lv):
                    h_vals.append(hv)
                    l_vals.append(lv)
        if len(h_vals) < 3:
            continue
        slopes.append(float(np.polyfit(np.asarray(h_vals), np.asarray(l_vals), 1)[0]))

    if not slopes:
        return float("nan")
    return float(np.mean(slopes))


def behavior_subscale_slope(bundle: Path, behavior_dir: str) -> float:
    human = load_matrix(bundle / "human_derived_metrics/subscale_human.csv")
    llm = load_matrix(bundle / f"model_response_metrics/{behavior_dir}/subscale_llm_implicit.csv")
    return _per_scale_slope(human, llm, lambda label: SUB_TO_SCALE.get(label))


def behavior_item_slope(bundle: Path, behavior_dir: str) -> float:
    human = load_matrix(bundle / "human_derived_metrics/item_human.csv")
    llm = load_matrix(bundle / f"model_response_metrics/{behavior_dir}/item_llm_implicit.csv")
    return _per_scale_slope(human, llm, _scale_from_item)


def matrix_slope(human: pd.DataFrame, derived: pd.DataFrame) -> float:
    common = [c for c in human.index if c in derived.index and c in derived.columns]
    human = human.loc[common, common]
    derived = derived.loc[common, common]
    h = human.to_numpy(dtype=float)
    d = derived.to_numpy(dtype=float)
    mask = np.triu(np.ones_like(h, dtype=bool), 1) & ~(np.isnan(h) | np.isnan(d))
    return float(np.polyfit(h[mask], d[mask], 1)[0])


def model_representational_metrics(bundle: Path, mech_name: str) -> tuple[int, float, float]:
    model_dir = bundle / f"activation_geometry_metrics/results_{mech_name}"
    geom = pd.read_csv(model_dir / "geometry/geometry_results.csv")
    subscale = geom[geom["level"].eq("subscale")]
    if subscale.empty:
        raise AssertionError(f"No subscale geometry rows for {mech_name}")
    best = subscale.loc[subscale["mantel_r"].idxmax()]
    best_layer = int(best["layer"])
    best_mantel = float(best["mantel_r"])

    human = load_matrix(bundle / "human_derived_metrics/subscale_human.csv")
    corrected = load_matrix(model_dir / f"reliability/cosine_sim_corrected_L{best_layer}.csv")
    corrected_slope = matrix_slope(human, corrected)
    return best_layer, best_mantel, corrected_slope


def ensure_no_output_mirror(bundle: Path) -> None:
    forbidden = [
        bundle / "statistics",
        bundle / "figures",
        bundle / "robustness",
        bundle / "behavior",
        bundle / "mechanistic",
        bundle / "semantic_controls",
        bundle / "human_aggregate",
        bundle / "behavior/summary",
        bundle / "mechanistic/summary",
    ]
    present = [p.relative_to(bundle).as_posix() for p in forbidden if p.exists()]
    if present:
        raise AssertionError(f"Output-like directories should not be in the derived-data bundle: {present}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, default=Path("release_artifacts"))
    args = parser.parse_args()
    bundle = args.bundle
    if not bundle.exists():
        raise FileNotFoundError(bundle)

    ensure_no_output_mirror(bundle)
    if (bundle / "manifest.csv").exists():
        raise AssertionError("manifest.csv should not be included in the minimal derived-metric bundle")

    expected_top_dirs = {
        "questionnaires",
        "human_derived_metrics",
        "model_response_metrics",
        "activation_geometry_metrics",
    }
    actual_top_dirs = {p.name for p in bundle.iterdir() if p.is_dir()}
    unexpected = sorted(actual_top_dirs - expected_top_dirs)
    missing = sorted(expected_top_dirs - actual_top_dirs)
    if unexpected or missing:
        raise AssertionError(f"Unexpected release directory layout; unexpected={unexpected}, missing={missing}")

    check_shape(bundle / "human_derived_metrics/subscale_human.csv", (16, 17))
    check_shape(bundle / "human_derived_metrics/scale_human.csv", (7, 8))
    check_shape(bundle / "human_derived_metrics/item_human.csv", (114, 115))

    rows = []
    for model in PRIMARY_MODELS:
        sub_slope = behavior_subscale_slope(bundle, model.behavior_dir)
        item_slope = behavior_item_slope(bundle, model.behavior_dir)
        best_layer, repr_mantel, repr_slope = model_representational_metrics(bundle, model.mech_name)
        rows.append(
            {
                "model": model.mech_name,
                "best_layer": best_layer,
                "repr_corrected_slope": repr_slope,
                "repr_mantel_r": repr_mantel,
                "beh_subscale_slope": sub_slope,
                "beh_item_slope": item_slope,
            }
        )

    df = pd.DataFrame(rows)
    r1, p1 = pearsonr(df["repr_corrected_slope"], df["beh_subscale_slope"])
    r2, p2 = pearsonr(df["repr_mantel_r"], df["beh_subscale_slope"])
    print(f"repr_corrected_slope vs behavior_slope: r={r1:.3f}, p={p1:.4f}, N={len(df)}")
    print(f"repr_mantel_r vs behavior_slope: r={r2:.3f}, p={p2:.4f}, N={len(df)}")
    if not close(r1, EXPECTED["repr_corrected_slope_vs_behavior_slope_r"]):
        raise AssertionError(f"Unexpected corrected-slope correlation: {r1}")
    if not close(r2, EXPECTED["repr_mantel_vs_behavior_slope_r"]):
        raise AssertionError(f"Unexpected Mantel correlation: {r2}")

    qwen = df[df["model"].eq("qwen14b_instruct")].iloc[0]
    print(f"qwen14b_instruct best subscale Mantel r={qwen['repr_mantel_r']:.3f}")
    if not close(float(qwen["repr_mantel_r"]), EXPECTED["qwen14b_instruct_best_mantel"]):
        raise AssertionError(f"Unexpected qwen14b_instruct Mantel r: {qwen['repr_mantel_r']}")

    n_behavior = len(list((bundle / "model_response_metrics").glob("*/subscale_llm_implicit.csv")))
    n_predicted = len(list((bundle / "activation_geometry_metrics").glob("*/amplification_locate/activation_predicted_corr_L*.csv")))
    print(f"model-response subscale correlation matrices={n_behavior}")
    print(f"activation-predicted matrices={n_predicted}")
    if n_behavior != 14:
        raise AssertionError(f"Expected 14 behavior matrices, found {n_behavior}")
    if n_predicted != 14:
        raise AssertionError(f"Expected 14 activation-predicted matrices, found {n_predicted}")

    print("RELEASE_BUNDLE_VERIFY_OK")


if __name__ == "__main__":
    main()
