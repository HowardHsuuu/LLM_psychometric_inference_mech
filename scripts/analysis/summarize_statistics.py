#!/usr/bin/env python3
"""Summarize cached robustness and key aggregate statistics.

This is a CPU-only inventory script. It does not rerun model generation,
activation extraction, or bootstrap resampling.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR / "src"))

from psychometric_inference.model_registry import PRIMARY_MODELS
from psychometric_inference.paths import (
    BEHAVIOR_OUTPUT_DIR,
    MECHANISTIC_OUTPUT_DIR,
    PROMPT_VARIANT_OUTPUT_DIR,
    ROBUSTNESS_OUTPUT_DIR,
    STATISTICS_OUTPUT_DIR,
)


def partial_corr(x: pd.Series, y: pd.Series, z: pd.Series) -> tuple[float, float]:
    valid = ~(x.isna() | y.isna() | z.isna())
    x_v = x[valid].to_numpy(dtype=float)
    y_v = y[valid].to_numpy(dtype=float)
    z_v = z[valid].to_numpy(dtype=float)
    if len(x_v) < 4:
        return np.nan, np.nan

    z_design = np.column_stack([np.ones_like(z_v), z_v])
    x_res = x_v - z_design @ np.linalg.lstsq(z_design, x_v, rcond=None)[0]
    y_res = y_v - z_design @ np.linalg.lstsq(z_design, y_v, rcond=None)[0]
    return pearsonr(x_res, y_res)


def add_row(rows: list[dict], category: str, statistic: str, value, **extra) -> None:
    row = {"category": category, "statistic": statistic, "value": value}
    row.update(extra)
    rows.append(row)


def summarize_behavior(rows: list[dict]) -> None:
    behavior_rows = []
    bootstrap_path = BEHAVIOR_OUTPUT_DIR / "bootstrap_ci_subjects.csv"
    if bootstrap_path.exists():
        df = pd.read_csv(bootstrap_path)
        for _, r in df.iterrows():
            add_row(
                rows,
                "behavior_uncertainty",
                "alignment_r_bootstrap_ci",
                r["r"],
                model_dir=r["dirname"],
                level=r["level"],
                ci_lo=r["ci_lo"],
                ci_hi=r["ci_hi"],
                source=str(bootstrap_path.relative_to(BASE_DIR)),
            )

    attenuation_path = BEHAVIOR_OUTPUT_DIR / "attenuation_correction.csv"
    if attenuation_path.exists():
        df = pd.read_csv(attenuation_path)
        for _, r in df.iterrows():
            add_row(
                rows,
                "behavior_attenuation",
                "corrected_slope",
                r["corrected_slope"],
                model_dir=r["dirname"],
                raw_slope=r["raw_slope"],
                corrected_r=r["corrected_r"],
                source=str(attenuation_path.relative_to(BASE_DIR)),
            )

    for model in PRIMARY_MODELS:
        metrics_path = BEHAVIOR_OUTPUT_DIR / model.behavior_dir / "comparison_metrics.csv"
        if not metrics_path.exists():
            continue
        df = pd.read_csv(metrics_path)
        for _, r in df.iterrows():
            behavior_rows.append({
                "model": model.mech_name,
                "model_dir": model.behavior_dir,
                "level": r.get("level"),
                "alignment_r": r.get("r") if pd.notna(r.get("r", np.nan)) else r.get("matrix_r"),
                "slope": r.get("slope"),
                "n_pairs": r.get("n_pairs"),
            })
            if pd.notna(r.get("slope", np.nan)):
                add_row(
                    rows,
                    "behavior_matrix",
                    "matrix_slope_point",
                    r["slope"],
                    model=model.mech_name,
                    model_dir=model.behavior_dir,
                    level=r["level"],
                    r=r.get("r"),
                    rmse=r.get("rmse"),
                    n_pairs=r.get("n_pairs"),
                    source=str(metrics_path.relative_to(BASE_DIR)),
                )
            elif pd.notna(r.get("matrix_r", np.nan)):
                add_row(
                    rows,
                    "behavior_matrix",
                    "matrix_alignment_point",
                    r["matrix_r"],
                    model=model.mech_name,
                    model_dir=model.behavior_dir,
                    level=r["level"],
                    mantel_p=r.get("mantel_p"),
                    rmse=r.get("rmse"),
                    n_pairs=r.get("n_pairs"),
                    source=str(metrics_path.relative_to(BASE_DIR)),
                )

    behavior_df = pd.DataFrame(behavior_rows)
    if not behavior_df.empty:
        for level in ["item_between", "subscale_between", "scale"]:
            sub = behavior_df[
                (behavior_df["level"] == level)
                & behavior_df["alignment_r"].notna()
                & behavior_df["slope"].notna()
            ]
            if len(sub) >= 4:
                r_val, p_val = pearsonr(sub["alignment_r"], sub["slope"])
                add_row(
                    rows,
                    "behavior_alignment_amplification",
                    "alignment_r_vs_slope_across_models",
                    r_val,
                    level=level,
                    p=p_val,
                    n=len(sub),
                    source="outputs/behavior/*/comparison_metrics.csv",
                )


def summarize_mechanistic(rows: list[dict]) -> None:
    scaling_path = MECHANISTIC_OUTPUT_DIR / "scaling_results_full.csv"
    if scaling_path.exists():
        df = pd.read_csv(scaling_path)
        for _, r in df.iterrows():
            for stat in ["geom_mantel_r", "geom_slope_corrected", "mean_reliability"]:
                if stat in r and pd.notna(r[stat]):
                    add_row(
                        rows,
                        "mechanistic_geometry",
                        stat,
                        r[stat],
                        model=r["model"],
                        best_layer=r.get("best_layer"),
                        source=str(scaling_path.relative_to(BASE_DIR)),
                    )

    structure_path = MECHANISTIC_OUTPUT_DIR / "structure_vs_behavior.csv"
    if structure_path.exists():
        df = pd.read_csv(structure_path)
        for x_col in ["repr_corrected_slope", "repr_mantel_r"]:
            valid = df[[x_col, "beh_subscale_slope"]].dropna()
            if len(valid) >= 3:
                r, p = pearsonr(valid[x_col], valid["beh_subscale_slope"])
                add_row(
                    rows,
                    "representation_behavior",
                    f"{x_col}_vs_behavior_slope",
                    r,
                    p=p,
                    n=len(valid),
                    source=str(structure_path.relative_to(BASE_DIR)),
                )

        if "size" in df.columns:
            log_size = np.log(df["size"].astype(float))
            for x_col in ["repr_corrected_slope", "repr_mantel_r"]:
                r, p = partial_corr(df[x_col], df["beh_subscale_slope"], log_size)
                add_row(
                    rows,
                    "representation_behavior",
                    f"{x_col}_vs_behavior_slope_partial_log_size",
                    r,
                    p=p,
                    n=int((~(df[x_col].isna() | df["beh_subscale_slope"].isna())).sum()),
                    source=str(structure_path.relative_to(BASE_DIR)),
                )


def summarize_prompt_variants(rows: list[dict]) -> None:
    summary_path = PROMPT_VARIANT_OUTPUT_DIR / "summary_focus.csv"
    if summary_path.exists():
        df = pd.read_csv(summary_path)
        for _, r in df.iterrows():
            value = r.get("slope")
            stat = "behavior_matrix_slope_point" if pd.notna(value) else "behavior_matrix_alignment_point"
            if pd.isna(value):
                value = r.get("matrix_r")
            add_row(
                rows,
                "prompt_variant_behavior",
                stat,
                value,
                model=r.get("model"),
                variant=r.get("variant"),
                readout=r.get("readout"),
                level=r.get("level"),
                source=str(summary_path.relative_to(BASE_DIR)),
            )

    geometry_path = PROMPT_VARIANT_OUTPUT_DIR / "geometry_summary_focus.csv"
    if geometry_path.exists():
        df = pd.read_csv(geometry_path)
        for _, r in df.iterrows():
            add_row(
                rows,
                "prompt_variant_geometry",
                "geometry_alignment_point",
                r.get("mantel_r"),
                model=r.get("model"),
                variant=r.get("variant"),
                level=r.get("level"),
                best_layer=r.get("layer"),
                slope=r.get("slope"),
                source=str(geometry_path.relative_to(BASE_DIR)),
            )


def summarize_factorial_geometry(rows: list[dict]) -> None:
    root = MECHANISTIC_OUTPUT_DIR / "factorial_geometry"
    if not root.exists():
        return
    for summary_path in sorted(root.rglob("geometry_summary.csv")):
        try:
            df = pd.read_csv(summary_path)
        except Exception:
            continue
        rel_parts = summary_path.relative_to(root).parts
        model = rel_parts[0] if len(rel_parts) > 0 else None
        condition = rel_parts[1] if len(rel_parts) > 1 else None
        for _, r in df.iterrows():
            add_row(
                rows,
                "factorial_profile_geometry",
                "profile_geometry_mantel_r",
                r.get("profile_geometry_mantel_r"),
                model=model,
                condition=condition,
                layer=r.get("layer"),
                mantel_p=r.get("profile_geometry_mantel_p"),
                mean_factor_readout_r=r.get("mean_factor_readout_r"),
                mean_factor_readout_r2=r.get("mean_factor_readout_r2"),
                n_profiles=r.get("n_profiles"),
                source=str(summary_path.relative_to(BASE_DIR)),
            )


def summarize_uncertainty(rows: list[dict]) -> None:
    files = [
        ("behavior_matrix_uncertainty.csv", "behavior_matrix_uncertainty"),
        ("geometry_matrix_uncertainty.csv", "geometry_matrix_uncertainty"),
        ("representation_behavior_uncertainty.csv", "representation_behavior_uncertainty"),
    ]
    for name, category in files:
        path = STATISTICS_OUTPUT_DIR / name
        if not path.exists():
            continue
        df = pd.read_csv(path)
        for _, r in df.iterrows():
            value = r.get("point", np.nan)
            statistic = r.get("statistic", "point")
            extra = {
                "bootstrap_ci_lo": r.get("bootstrap_ci_lo"),
                "bootstrap_ci_hi": r.get("bootstrap_ci_hi"),
                "loo_min": r.get("loo_min"),
                "loo_max": r.get("loo_max"),
                "source": str(path.relative_to(BASE_DIR)),
            }
            for optional in ["model", "level", "pair_scope", "relationship", "n_pairs", "n_models"]:
                if optional in r:
                    extra[optional] = r.get(optional)
            add_row(rows, category, statistic, value, **extra)


def summarize_robustness_inventory() -> pd.DataFrame:
    rows = []
    done_dir = ROBUSTNESS_OUTPUT_DIR / ".done"
    done = {p.stem for p in done_dir.glob("*.done")} if done_dir.exists() else set()

    for subdir in sorted(p for p in ROBUSTNESS_OUTPUT_DIR.iterdir() if p.is_dir() and not p.name.startswith(".")):
        summary_files = sorted(subdir.glob("*summary*.json")) + sorted(subdir.glob("*.json"))
        csv_files = sorted(subdir.glob("*.csv"))
        rows.append({
            "analysis": subdir.name,
            "done_marker": subdir.name in done,
            "n_csv": len(csv_files),
            "n_json": len(summary_files),
            "summary_files": ";".join(str(p.relative_to(BASE_DIR)) for p in summary_files),
            "csv_files": ";".join(str(p.relative_to(BASE_DIR)) for p in csv_files),
        })
    return pd.DataFrame(rows)


def summarize_gaps(key_df: pd.DataFrame, inventory_df: pd.DataFrame) -> pd.DataFrame:
    factorial_controls = [
        PROMPT_VARIANT_OUTPUT_DIR.parent / "behavior/factorial_analysis/factorial_coefficients.csv",
        PROMPT_VARIANT_OUTPUT_DIR.parent / "behavior/factorial_analysis/factorial_r2.csv",
        PROMPT_VARIANT_OUTPUT_DIR.parent / "behavior/factorial_analysis/human_coefficients.csv",
    ]
    have_factorial_behavior = all(path.exists() for path in factorial_controls)
    factorial_geometry_paths = list((MECHANISTIC_OUTPUT_DIR / "factorial_geometry").rglob("geometry_summary.csv"))
    have_factorial_geometry = any(
        path.exists() and not pd.read_csv(path).empty
        for path in factorial_geometry_paths
    )

    have_prompt_behavior = False
    have_probability_readout = False
    prompt_summary = PROMPT_VARIANT_OUTPUT_DIR / "summary_focus.csv"
    if prompt_summary.exists():
        prompt_df = pd.read_csv(prompt_summary)
        have_prompt_behavior = bool(
            "variant" in prompt_df.columns
            and any(prompt_df["variant"].astype(str) != "baseline_raw")
        )
        have_probability_readout = bool(
            {"variant", "readout"}.issubset(prompt_df.columns)
            and any(
                (prompt_df["variant"].astype(str) != "baseline_raw")
                & (prompt_df["readout"].astype(str) == "expected_value")
            )
        )
    have_prompt_geometry = False
    geometry_summary = PROMPT_VARIANT_OUTPUT_DIR / "geometry_summary_focus.csv"
    if geometry_summary.exists():
        geometry_df = pd.read_csv(geometry_summary)
        have_prompt_geometry = bool(
            "variant" in geometry_df.columns
            and any(geometry_df["variant"].astype(str) != "baseline_raw")
        )
    gaps = [
        {
            "item": "profile_level_behavior_controls",
            "status": "available" if have_factorial_behavior else "not_cached_yet",
            "note": "Factorial/random/collapsed profile controls are behavioral/profile-level diagnostics; they do not include activation geometry.",
        },
        {
            "item": "factorial_profile_geometry_results",
            "status": "available" if have_factorial_geometry else "not_cached_yet",
            "note": "Run python -m psychometric_inference.mechanisms.factorial_geometry to extract/analyze activation geometry for factorial profile controls.",
        },
        {
            "item": "prompt_variant_behavior_results",
            "status": "available" if have_prompt_behavior else "not_cached_yet",
            "note": "Run scripts/behavior/run_prompt_sensitivity.py to populate outputs/prompt_variants/.",
        },
        {
            "item": "probability_readout_behavior_results",
            "status": "available" if have_probability_readout else "not_cached_yet",
            "note": "Run scripts/behavior/run_prompt_sensitivity.py with --readouts expected_value to compare probability-derived readout against argmax.",
        },
        {
            "item": "prompt_variant_geometry_results",
            "status": "available" if have_prompt_geometry else "not_cached_yet",
            "note": "Run python -m psychometric_inference.mechanisms.prompt_geometry to populate prompt/scale-name geometry robustness summaries.",
        },
        {
            "item": "behavior_matrix_slope_ci",
            "status": "available" if (STATISTICS_OUTPUT_DIR / "behavior_matrix_uncertainty.csv").exists() else "not_summarized",
            "note": "Cached aggregate-matrix bootstrap and leave-one-construct diagnostics for subscale/scale slopes.",
        },
        {
            "item": "geometry_mantel_ci",
            "status": "available" if (STATISTICS_OUTPUT_DIR / "geometry_matrix_uncertainty.csv").exists() else "not_summarized",
            "note": "Cached geometry-matrix bootstrap and leave-one-construct diagnostics for Mantel-style alignment r.",
        },
        {
            "item": "representation_behavior_ci",
            "status": "available" if (STATISTICS_OUTPUT_DIR / "representation_behavior_uncertainty.csv").exists() else "not_summarized",
            "note": "Cached model-bootstrap and leave-one-model diagnostics for representation-behavior relationships.",
        },
        {
            "item": "robustness_inventory",
            "status": "available" if not inventory_df.empty else "missing",
            "note": "Existing A/F/D robustness artifacts are inventoried in robustness_inventory.csv.",
        },
    ]
    return pd.DataFrame(gaps)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize cached aggregate statistics")
    parser.add_argument("--output_dir", type=Path, default=STATISTICS_OUTPUT_DIR)
    args = parser.parse_args()

    rows: list[dict] = []
    summarize_behavior(rows)
    summarize_mechanistic(rows)
    summarize_prompt_variants(rows)
    summarize_factorial_geometry(rows)
    summarize_uncertainty(rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    key_df = pd.DataFrame(rows)
    key_df.to_csv(args.output_dir / "key_statistics.csv", index=False)

    inventory_df = summarize_robustness_inventory()
    inventory_df.to_csv(args.output_dir / "robustness_inventory.csv", index=False)

    gaps_df = summarize_gaps(key_df, inventory_df)
    gaps_df.to_csv(args.output_dir / "statistics_gaps.csv", index=False)

    print(f"key_statistics rows: {len(key_df)}")
    print(f"robustness analyses: {len(inventory_df)}")
    print(f"wrote: {args.output_dir}")
    if not gaps_df.empty:
        print(gaps_df.to_string(index=False))


if __name__ == "__main__":
    main()
