#!/usr/bin/env python3
"""Compute CPU-only uncertainty diagnostics from cached aggregate matrices."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR / "src"))

from psychometric_inference.model_registry import PRIMARY_MODELS
from psychometric_inference.paths import BEHAVIOR_OUTPUT_DIR, MECHANISTIC_OUTPUT_DIR, STATISTICS_OUTPUT_DIR
from psychometric_inference.scoring import SCORING_RULES


SUB_TO_SCALE = {
    f"{scale}_{sub}": scale
    for scale, rules in SCORING_RULES.items()
    for sub in rules["subscales"]
}


def scale_for_label(label: str) -> str:
    if label in SUB_TO_SCALE:
        return SUB_TO_SCALE[label]
    if "_Q" in label:
        return label.split("_Q", 1)[0]
    return label


def upper_vectors(
    reference: pd.DataFrame,
    observed: pd.DataFrame,
    pair_scope: str = "all",
) -> tuple[np.ndarray, np.ndarray, list[tuple[str, str]]]:
    labels = [c for c in reference.columns if c in observed.columns and c in reference.index and c in observed.index]
    ref = reference.loc[labels, labels]
    obs = observed.loc[labels, labels]
    pairs = []
    x_vals = []
    y_vals = []

    for i, left in enumerate(labels):
        for j in range(i + 1, len(labels)):
            right = labels[j]
            if pair_scope == "between" and scale_for_label(left) == scale_for_label(right):
                continue
            if pair_scope == "within" and scale_for_label(left) != scale_for_label(right):
                continue
            x = ref.loc[left, right]
            y = obs.loc[left, right]
            if pd.isna(x) or pd.isna(y):
                continue
            pairs.append((left, right))
            x_vals.append(float(x))
            y_vals.append(float(y))

    return np.asarray(x_vals), np.asarray(y_vals), pairs


def slope(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or np.nanstd(x) == 0:
        return np.nan
    return float(np.polyfit(x, y, 1)[0])


def corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or np.nanstd(x) == 0 or np.nanstd(y) == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def bootstrap_interval(
    x: np.ndarray,
    y: np.ndarray,
    fn,
    n_boot: int,
    rng: np.random.Generator,
) -> tuple[float, float, float, int]:
    vals = []
    n = len(x)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        val = fn(x[idx], y[idx])
        if not np.isnan(val):
            vals.append(val)
    arr = np.asarray(vals)
    if len(arr) == 0:
        return np.nan, np.nan, np.nan, 0
    return (
        float(np.mean(arr)),
        float(np.percentile(arr, 2.5)),
        float(np.percentile(arr, 97.5)),
        int(len(arr)),
    )


def leave_one_construct(
    reference: pd.DataFrame,
    observed: pd.DataFrame,
    pair_scope: str,
    fn,
) -> tuple[float, float, float, int]:
    labels = [c for c in reference.columns if c in observed.columns and c in reference.index and c in observed.index]
    vals = []
    for holdout in labels:
        keep = [l for l in labels if l != holdout]
        if len(keep) < 4:
            continue
        x, y, _ = upper_vectors(reference.loc[keep, keep], observed.loc[keep, keep], pair_scope)
        val = fn(x, y)
        if not np.isnan(val):
            vals.append(val)
    arr = np.asarray(vals)
    if len(arr) == 0:
        return np.nan, np.nan, np.nan, 0
    return float(np.mean(arr)), float(np.min(arr)), float(np.max(arr)), int(len(arr))


def matrix_rows(
    *,
    category: str,
    model: str,
    source: Path,
    level: str,
    pair_scope: str,
    reference: pd.DataFrame,
    observed: pd.DataFrame,
    n_boot: int,
    rng: np.random.Generator,
) -> list[dict]:
    x, y, pairs = upper_vectors(reference, observed, pair_scope)
    rows = []
    if len(x) < 3:
        return rows

    for stat_name, fn in [("slope", slope), ("alignment_r", corr)]:
        point = fn(x, y)
        b_mean, b_lo, b_hi, b_n = bootstrap_interval(x, y, fn, n_boot, rng)
        loo_mean, loo_min, loo_max, loo_n = leave_one_construct(reference, observed, pair_scope, fn)
        rows.append({
            "category": category,
            "model": model,
            "level": level,
            "pair_scope": pair_scope,
            "statistic": stat_name,
            "point": point,
            "bootstrap_mean": b_mean,
            "bootstrap_ci_lo": b_lo,
            "bootstrap_ci_hi": b_hi,
            "bootstrap_n": b_n,
            "loo_mean": loo_mean,
            "loo_min": loo_min,
            "loo_max": loo_max,
            "loo_n": loo_n,
            "n_pairs": len(pairs),
            "source": str(source.relative_to(BASE_DIR)),
        })
    return rows


def read_matrix(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path, index_col=0)


def behavior_uncertainty(n_boot: int, rng: np.random.Generator, include_item: bool) -> pd.DataFrame:
    rows = []
    for model in PRIMARY_MODELS:
        root = BEHAVIOR_OUTPUT_DIR / model.behavior_dir
        specs = [
            ("subscale", "all", root / "subscale_human.csv", root / "subscale_llm_implicit.csv"),
            ("subscale", "between", root / "subscale_human.csv", root / "subscale_llm_implicit.csv"),
            ("scale", "all", root / "scale_human.csv", root / "scale_llm_implicit.csv"),
        ]
        if include_item:
            specs.insert(0, ("item", "between", root / "item_human.csv", root / "item_llm_implicit.csv"))
        for level, pair_scope, human_path, observed_path in specs:
            human = read_matrix(human_path)
            observed = read_matrix(observed_path)
            if human is None or observed is None:
                continue
            rows.extend(matrix_rows(
                category="behavior_matrix",
                model=model.mech_name,
                source=observed_path,
                level=level,
                pair_scope=pair_scope,
                reference=human,
                observed=observed,
                n_boot=n_boot,
                rng=rng,
            ))
    return pd.DataFrame(rows)


def geometry_uncertainty(n_boot: int, rng: np.random.Generator) -> pd.DataFrame:
    scaling = MECHANISTIC_OUTPUT_DIR / "scaling_results_full.csv"
    if not scaling.exists():
        return pd.DataFrame()
    scaling_df = pd.read_csv(scaling)

    rows = []
    for _, row in scaling_df.iterrows():
        model = row["model"]
        layer = int(row["best_layer"])
        root = MECHANISTIC_OUTPUT_DIR / f"results_{model}" / "geometry"
        specs = [
            ("subscale", "all", root / "human_subscale_corr.csv", root / f"subscale_cosine_sim_L{layer}.csv"),
            ("subscale", "between", root / "human_subscale_corr.csv", root / f"subscale_cosine_sim_L{layer}.csv"),
            ("scale", "all", root / "human_scale_corr.csv", root / f"scale_cosine_sim_L{layer}.csv"),
        ]
        for level, pair_scope, human_path, observed_path in specs:
            human = read_matrix(human_path)
            observed = read_matrix(observed_path)
            if human is None or observed is None:
                continue
            matrix_result = matrix_rows(
                category="geometry_matrix",
                model=model,
                source=observed_path,
                level=level,
                pair_scope=pair_scope,
                reference=human,
                observed=observed,
                n_boot=n_boot,
                rng=rng,
            )
            for result in matrix_result:
                result["best_layer"] = layer
            rows.extend(matrix_result)
    return pd.DataFrame(rows)


def partial_corr(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    valid = ~(np.isnan(x) | np.isnan(y) | np.isnan(z))
    x = x[valid]
    y = y[valid]
    z = z[valid]
    if len(x) < 4:
        return np.nan
    design = np.column_stack([np.ones_like(z), z])
    x_res = x - design @ np.linalg.lstsq(design, x, rcond=None)[0]
    y_res = y - design @ np.linalg.lstsq(design, y, rcond=None)[0]
    return corr(x_res, y_res)


def bootstrap_models(df: pd.DataFrame, x_col: str, y_col: str, n_boot: int, rng: np.random.Generator) -> tuple[float, float, float, int]:
    vals = []
    valid = df[[x_col, y_col]].dropna()
    n = len(valid)
    for _ in range(n_boot):
        sample = valid.iloc[rng.integers(0, n, size=n)]
        val = corr(sample[x_col].to_numpy(float), sample[y_col].to_numpy(float))
        if not np.isnan(val):
            vals.append(val)
    arr = np.asarray(vals)
    return float(np.mean(arr)), float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5)), int(len(arr))


def bootstrap_partial_models(df: pd.DataFrame, x_col: str, y_col: str, z_col: str, n_boot: int, rng: np.random.Generator) -> tuple[float, float, float, int]:
    vals = []
    valid = df[[x_col, y_col, z_col]].dropna()
    n = len(valid)
    for _ in range(n_boot):
        sample = valid.iloc[rng.integers(0, n, size=n)]
        val = partial_corr(
            sample[x_col].to_numpy(float),
            sample[y_col].to_numpy(float),
            sample[z_col].to_numpy(float),
        )
        if not np.isnan(val):
            vals.append(val)
    arr = np.asarray(vals)
    return float(np.mean(arr)), float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5)), int(len(arr))


def leave_one_model(df: pd.DataFrame, fn) -> tuple[float, float, float, int]:
    vals = []
    for idx in df.index:
        val = fn(df.drop(index=idx))
        if not np.isnan(val):
            vals.append(val)
    arr = np.asarray(vals)
    return float(np.mean(arr)), float(np.min(arr)), float(np.max(arr)), int(len(arr))


def representation_behavior_uncertainty(n_boot: int, rng: np.random.Generator) -> pd.DataFrame:
    path = MECHANISTIC_OUTPUT_DIR / "structure_vs_behavior.csv"
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_csv(path)
    df["log_size"] = np.log(df["size"].astype(float))
    rows = []

    for x_col in ["repr_corrected_slope", "repr_mantel_r"]:
        y_col = "beh_subscale_slope"
        valid = df[[x_col, y_col]].dropna()
        point = corr(valid[x_col].to_numpy(float), valid[y_col].to_numpy(float))
        b_mean, b_lo, b_hi, b_n = bootstrap_models(valid, x_col, y_col, n_boot, rng)
        loo_mean, loo_min, loo_max, loo_n = leave_one_model(
            valid,
            lambda d: corr(d[x_col].to_numpy(float), d[y_col].to_numpy(float)),
        )
        rows.append({
            "relationship": f"{x_col}_vs_{y_col}",
            "statistic": "pearson_r",
            "point": point,
            "bootstrap_mean": b_mean,
            "bootstrap_ci_lo": b_lo,
            "bootstrap_ci_hi": b_hi,
            "bootstrap_n": b_n,
            "loo_mean": loo_mean,
            "loo_min": loo_min,
            "loo_max": loo_max,
            "loo_n": loo_n,
            "n_models": len(valid),
            "source": str(path.relative_to(BASE_DIR)),
        })

        valid_partial = df[[x_col, y_col, "log_size"]].dropna()
        point_partial = partial_corr(
            valid_partial[x_col].to_numpy(float),
            valid_partial[y_col].to_numpy(float),
            valid_partial["log_size"].to_numpy(float),
        )
        bp_mean, bp_lo, bp_hi, bp_n = bootstrap_partial_models(
            valid_partial,
            x_col,
            y_col,
            "log_size",
            n_boot,
            rng,
        )
        lp_mean, lp_min, lp_max, lp_n = leave_one_model(
            valid_partial,
            lambda d: partial_corr(
                d[x_col].to_numpy(float),
                d[y_col].to_numpy(float),
                d["log_size"].to_numpy(float),
            ),
        )
        rows.append({
            "relationship": f"{x_col}_vs_{y_col}_partial_log_size",
            "statistic": "partial_r",
            "point": point_partial,
            "bootstrap_mean": bp_mean,
            "bootstrap_ci_lo": bp_lo,
            "bootstrap_ci_hi": bp_hi,
            "bootstrap_n": bp_n,
            "loo_mean": lp_mean,
            "loo_min": lp_min,
            "loo_max": lp_max,
            "loo_n": lp_n,
            "n_models": len(valid_partial),
            "source": str(path.relative_to(BASE_DIR)),
        })

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute uncertainty intervals from cached aggregate outputs")
    parser.add_argument("--n_boot", type=int, default=5000)
    parser.add_argument("--include_item", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=Path, default=STATISTICS_OUTPUT_DIR)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    behavior = behavior_uncertainty(args.n_boot, rng, include_item=args.include_item)
    behavior.to_csv(args.output_dir / "behavior_matrix_uncertainty.csv", index=False)

    geometry = geometry_uncertainty(args.n_boot, rng)
    geometry.to_csv(args.output_dir / "geometry_matrix_uncertainty.csv", index=False)

    relationships = representation_behavior_uncertainty(args.n_boot, rng)
    relationships.to_csv(args.output_dir / "representation_behavior_uncertainty.csv", index=False)

    print(f"behavior uncertainty rows: {len(behavior)}")
    print(f"geometry uncertainty rows: {len(geometry)}")
    print(f"representation-behavior rows: {len(relationships)}")
    print(f"wrote: {args.output_dir}")


if __name__ == "__main__":
    main()
