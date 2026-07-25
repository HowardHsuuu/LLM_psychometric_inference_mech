#!/usr/bin/env python3
"""Build the public derived-metric bundle for reproducibility.

The bundle is not a mirror of ``outputs/``. It contains only analysis-ready
derived metrics that replace data we do not release in raw form: questionnaire
definitions, aggregate human metrics, aggregate model-response metrics, and
activation-geometry metrics. Final statistics, plots, controls, and broad output
dumps stay in ``outputs/`` and can be regenerated locally.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd

from psychometric_inference.model_registry import MODEL_BY_MECH_NAME, PRIMARY_MODELS


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "release_artifacts"

FORBIDDEN_PATH_PARTS = {
    "data/human",
    "human_subscale_scores.csv",
}

FORBIDDEN_COLUMNS = {
    "Subject_ID",
    "ID",
    "Group_ID",
    "Scan_ID",
    "participant",
    "participant_id",
    "worker_id",
}


class BundleBuilder:
    def __init__(self, output_dir: Path, replace: bool = False):
        self.output_dir = output_dir
        self.file_count = 0
        if output_dir.exists():
            if not replace:
                raise FileExistsError(f"{output_dir} exists; pass --replace to rebuild.")
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True)

    def _check_source(self, source: Path) -> None:
        rel = source.relative_to(ROOT).as_posix()
        for forbidden in FORBIDDEN_PATH_PARTS:
            if forbidden in rel:
                raise ValueError(f"Forbidden source path for release bundle: {rel}")

    def _target_path(self, relative_target: str) -> Path:
        target = self.output_dir / relative_target
        target.parent.mkdir(parents=True, exist_ok=True)
        return target

    def copy_file(self, source: Path, relative_target: str, category: str, note: str) -> None:
        self._check_source(source)
        if not source.exists():
            raise FileNotFoundError(source)
        target = self._target_path(relative_target)
        if source.suffix.lower() == ".csv":
            self._copy_csv(source, target)
        else:
            shutil.copy2(source, target)
        self.file_count += 1

    def write_csv(self, df: pd.DataFrame, relative_target: str, category: str, note: str) -> None:
        target = self._target_path(relative_target)
        forbidden = FORBIDDEN_COLUMNS.intersection(map(str, df.columns))
        if forbidden:
            raise ValueError(f"Forbidden subject-level columns in {relative_target}: {sorted(forbidden)}")
        df.to_csv(target, index=False)
        self.file_count += 1

    def _copy_csv(self, source: Path, target: Path) -> None:
        df = pd.read_csv(source)
        rename = {
            c: "label"
            for c in df.columns
            if str(c).startswith("Unnamed")
        }
        if rename:
            df = df.rename(columns=rename)
        forbidden = FORBIDDEN_COLUMNS.intersection(map(str, df.columns))
        if forbidden:
            raise ValueError(f"Forbidden subject-level columns in {source}: {sorted(forbidden)}")
        df.to_csv(target, index=False)

    def write_readme(self) -> None:
        readme = self.output_dir / "README.md"
        readme.write_text(
            """# Public Derived Metrics

This directory contains the minimal public derived metrics needed for aggregate
reproducibility checks.

Contents:

- `questionnaires/`: questionnaire definitions used to construct prompts and
  score responses.
- `human_derived_metrics/`: aggregate human correlation/structure/reliability
  metrics that replace raw participant rows.
- `model_response_metrics/`: aggregate correlation matrices computed from model
  questionnaire responses; these replace raw per-subject model outputs.
- `activation_geometry_metrics/`: activation-derived geometry/readout metrics
  at the analysis layer; these replace raw hidden-state tensors.

Excluded:

- participant-level human questionnaire responses and subject-level derived
  human score tables, raw LLM responses, raw hidden states, model weights,
  control-experiment dumps, rendered figures, final statistic tables, and broad
  `outputs/` mirrors.

The bundle is meant to support no-GPU checks by recomputing key aggregate
statistics from derived matrices. GPU-heavy generation and activation extraction
scripts remain in the repository, but raw human inputs and raw activations are
not part of this public bundle.

Quick verification from the repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python scripts/artifact/verify_release_bundle.py --bundle release_artifacts
```
""",
            encoding="utf-8",
        )


def copy_questionnaires(builder: BundleBuilder) -> None:
    for path in sorted((ROOT / "data/questionnaires").glob("*.jsonl")):
        builder.copy_file(
            path,
            f"questionnaires/{path.name}",
            "questionnaire_definitions",
            "Scale metadata and item text used to construct prompts.",
        )


def copy_human_aggregates(builder: BundleBuilder) -> None:
    canonical = ROOT / "outputs/behavior/qwen14b_instruct_v3"
    for name, note in [
        ("item_human.csv", "Aggregate 114x114 human item correlation matrix."),
        ("subscale_human.csv", "Aggregate 16x16 human subscale correlation matrix."),
        ("scale_human.csv", "Aggregate 7x7 human scale correlation matrix."),
    ]:
        builder.copy_file(canonical / name, f"human_derived_metrics/{name}", "human_derived_metric", note)

    for name in [
        "02_community_scale_crosstab.csv",
        "03_factor_correlations.csv",
        "03_factor_loadings.csv",
    ]:
        builder.copy_file(
            ROOT / "outputs/human_structure" / name,
            f"human_derived_metrics/structure/{name}",
            "human_derived_metric",
            "Aggregate human-structure derived data.",
        )

    for name in ["human_split_half_bootstrap.csv"]:
        builder.copy_file(
            ROOT / "outputs/supplementary" / name,
            f"human_derived_metrics/reliability/{name}",
            "human_derived_metric",
            "Aggregate human split-half reliability derived data.",
        )


def copy_behavior(builder: BundleBuilder) -> None:
    for model in PRIMARY_MODELS:
        src_dir = ROOT / "outputs/behavior" / model.behavior_dir
        for name in [
            "item_llm_implicit.csv",
            "subscale_llm_implicit.csv",
            "scale_llm_implicit.csv",
        ]:
            builder.copy_file(
                src_dir / name,
                f"model_response_metrics/{model.behavior_dir}/{name}",
                "model_response_metric",
                f"Aggregate response-correlation metric for {model.behavior_dir}.",
            )


def copy_mechanistic(builder: BundleBuilder) -> None:
    for result_dir in sorted((ROOT / "outputs/mechanistic").glob("results_*")):
        if not result_dir.is_dir() or result_dir.name == "results_default":
            continue
        model_name = result_dir.name.removeprefix("results_")
        if model_name not in MODEL_BY_MECH_NAME:
            continue
        geometry_results = result_dir / "geometry/geometry_results.csv"
        builder.copy_file(
            geometry_results,
            f"activation_geometry_metrics/{result_dir.name}/geometry/geometry_results.csv",
            "activation_geometry_metric",
            "Layerwise activation-geometry derived data.",
        )
        geom_df = pd.read_csv(geometry_results)
        subscale_rows = geom_df[geom_df["level"].eq("subscale")]
        if subscale_rows.empty:
            raise ValueError(f"No subscale geometry rows in {geometry_results}")
        best_layer = int(subscale_rows.loc[subscale_rows["mantel_r"].idxmax(), "layer"])

        selected = [
            f"geometry/subscale_cosine_sim_L{best_layer}.csv",
            f"geometry/scale_cosine_sim_L{best_layer}.csv",
            f"geometry/direction_validation_L{best_layer}.csv",
            f"reliability/reliability_L{best_layer}.csv",
            f"reliability/cosine_sim_corrected_L{best_layer}.csv",
            f"amplification_locate/activation_predicted_corr_L{best_layer}.csv",
            f"regression_directions/regression_cv_rs_L{best_layer}.csv",
            f"regression_directions/regression_cosine_sim_L{best_layer}.csv",
            "controls/semantic_similarity_matrix.csv",
        ]
        for rel in selected:
            path = result_dir / rel
            if not path.exists():
                raise FileNotFoundError(path)
            builder.copy_file(
                path,
                f"activation_geometry_metrics/{result_dir.name}/{rel}",
                "activation_geometry_metric",
                f"Selected activation-derived data for layer {best_layer}.",
            )


def audit_bundle(output_dir: Path) -> None:
    forbidden_strings = [
        "/Users/",
        "data/human/",
        "data/llm_behavior",
        "outputs/",
        "reports/figures/",
        "Group_ID",
        "Subject_ID",
        "Scan_ID",
        "human_subscale_scores",
    ]
    for path in output_dir.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {".csv", ".json", ".md"}:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if path.name == "README.md":
            continue
        hits = [s for s in forbidden_strings if s in text]
        if hits:
            raise ValueError(f"Forbidden release text in {path}: {hits}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()

    builder = BundleBuilder(args.output, replace=args.replace)
    copy_questionnaires(builder)
    copy_human_aggregates(builder)
    copy_behavior(builder)
    copy_mechanistic(builder)
    builder.write_readme()
    audit_bundle(args.output)

    print(f"Wrote release bundle: {args.output}")
    print(f"Files copied: {builder.file_count}")


if __name__ == "__main__":
    main()
