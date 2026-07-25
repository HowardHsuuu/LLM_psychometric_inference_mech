#!/usr/bin/env python3
"""Run prompt-variant geometry checks from cached code paths."""

from __future__ import annotations

import argparse
import gc
import importlib
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from psychometric_inference.model_registry import MODEL_BY_MECH_NAME, ModelSpec
from psychometric_inference.paths import PROMPT_VARIANT_OUTPUT_DIR


@dataclass(frozen=True)
class GeometryVariant:
    name: str
    system_prompt_variant: str
    prompt_format: str
    instruct_only: bool = False


VARIANTS = {
    "no_scale_name": GeometryVariant("no_scale_name", "no_scale_name", "raw"),
    "chat_template": GeometryVariant("chat_template", "default", "chat_template", instruct_only=True),
}

DEFAULT_MODELS = ("qwen14b_instruct",)
DEFAULT_VARIANTS = ("no_scale_name", "chat_template")


def results_dir(model: ModelSpec, variant: GeometryVariant) -> Path:
    return PROMPT_VARIANT_OUTPUT_DIR / model.mech_name / variant.name / "mechanistic"


def patch_config(model: ModelSpec, variant: GeometryVariant) -> None:
    import psychometric_inference.mechanisms.config as cfg

    cfg.MODEL_ID = model.hf_id
    cfg.N_LAYERS = model.n_layers
    cfg.TARGET_LAYERS = list(model.target_layers)
    cfg.DEFAULT_LAYER = model.default_layer
    cfg.RESULTS_DIR = results_dir(model, variant)
    cfg.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    for mod_name in list(sys.modules):
        if mod_name.startswith("psychometric_inference.mechanisms.") and mod_name not in {
            "psychometric_inference.mechanisms.prompt_geometry",
            "psychometric_inference.mechanisms.config",
        }:
            try:
                importlib.reload(sys.modules[mod_name])
            except Exception:
                pass


def run_module(module_path: str, argv: list[str]) -> None:
    module = importlib.import_module(module_path)
    module = importlib.reload(module)
    old_argv = sys.argv
    try:
        sys.argv = ["x"] + argv
        module.main()
    finally:
        sys.argv = old_argv


def gpu_cleanup() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def directions_done(model: ModelSpec, variant: GeometryVariant, subscales_only: bool) -> bool:
    rd = results_dir(model, variant) / "directions"
    if not (rd / "subscale_directions.npz").exists():
        return False
    if not subscales_only and not (rd / "scale_directions.npz").exists():
        return False
    return True


def run_variant(
    model: ModelSpec,
    variant: GeometryVariant,
    n_replications: int,
    n_probes: int,
    subscales_only: bool,
    skip_existing: bool,
    analysis_only: bool,
) -> None:
    patch_config(model, variant)

    if not analysis_only and not (skip_existing and directions_done(model, variant, subscales_only)):
        argv = [
            "--model_id",
            model.hf_id,
            "--n_replications",
            str(n_replications),
            "--n_probes",
            str(n_probes),
            "--prompt_format",
            variant.prompt_format,
            "--system_prompt_variant",
            variant.system_prompt_variant,
        ]
        if subscales_only:
            argv.append("--subscales_only")
        run_module("psychometric_inference.mechanisms.directions", argv)
        gpu_cleanup()

    patch_config(model, variant)
    run_module("psychometric_inference.mechanisms.geometry", ["--all_layers"])


def dry_run(
    models: list[ModelSpec],
    variants: list[GeometryVariant],
    n_replications: int,
    n_probes: int,
    subscales_only: bool,
) -> None:
    print("=" * 72)
    print("  GEOMETRY PROMPT VARIANTS")
    print("=" * 72)
    for model in models:
        for variant in variants:
            status = "skip: instruct only" if variant.instruct_only and not model.is_instruct else "ready"
            done = directions_done(model, variant, subscales_only)
            print(
                f"  {model.mech_name:<20} {variant.name:<16} "
                f"format={variant.prompt_format:<13} system={variant.system_prompt_variant:<13} "
                f"n_rep={n_replications:<3} probes={n_probes:<2} {status} {'[done]' if done else ''}"
            )


def collect_summary(models: list[ModelSpec], variants: list[GeometryVariant]) -> pd.DataFrame:
    rows = []
    for model in models:
        baseline_path = Path("outputs") / "mechanistic" / f"results_{model.mech_name}" / "geometry" / "geometry_results.csv"
        baseline_abs = Path.cwd() / baseline_path
        if baseline_abs.exists():
            df = pd.read_csv(baseline_abs)
            for _, row in df.iterrows():
                rows.append({
                    "model": model.mech_name,
                    "variant": "baseline_raw",
                    "level": row.get("level"),
                    "layer": row.get("layer"),
                    "mantel_r": row.get("mantel_r"),
                    "mantel_p": row.get("mantel_p"),
                    "slope": row.get("slope"),
                    "source": baseline_path.as_posix(),
                })

        for variant in variants:
            path = results_dir(model, variant) / "geometry" / "geometry_results.csv"
            if not path.exists():
                continue
            df = pd.read_csv(path)
            for _, row in df.iterrows():
                rows.append({
                    "model": model.mech_name,
                    "variant": variant.name,
                    "level": row.get("level"),
                    "layer": row.get("layer"),
                    "mantel_r": row.get("mantel_r"),
                    "mantel_p": row.get("mantel_p"),
                    "slope": row.get("slope"),
                    "source": path.relative_to(Path.cwd()).as_posix(),
                })

    summary = pd.DataFrame(rows)
    PROMPT_VARIANT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(PROMPT_VARIANT_OUTPUT_DIR / "geometry_summary.csv", index=False)

    if not summary.empty:
        focus_rows = []
        for (model, variant, level), group in summary.groupby(["model", "variant", "level"]):
            best = group.loc[group["mantel_r"].idxmax()]
            focus_rows.append(best)
        pd.DataFrame(focus_rows).to_csv(PROMPT_VARIANT_OUTPUT_DIR / "geometry_summary_focus.csv", index=False)
    return summary


def resolve_models(names: list[str]) -> list[ModelSpec]:
    models = []
    for name in names:
        if name not in MODEL_BY_MECH_NAME:
            valid = ", ".join(sorted(MODEL_BY_MECH_NAME))
            raise ValueError(f"Unknown model {name!r}. Valid names: {valid}")
        models.append(MODEL_BY_MECH_NAME[name])
    return models


def resolve_variants(names: list[str]) -> list[GeometryVariant]:
    variants = []
    for name in names:
        if name not in VARIANTS:
            raise ValueError(f"Unknown variant {name!r}. Valid variants: {', '.join(VARIANTS)}")
        variants.append(VARIANTS[name])
    return variants


def main() -> None:
    parser = argparse.ArgumentParser(description="Run geometry prompt variants")
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--variants", nargs="+", default=list(DEFAULT_VARIANTS))
    parser.add_argument("--n_replications", type=int, default=3)
    parser.add_argument("--n_probes", type=int, default=20)
    parser.add_argument("--include_scales", action="store_true")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--analysis_only", action="store_true")
    parser.add_argument("--summary_only", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    models = resolve_models(args.models)
    variants = resolve_variants(args.variants)
    subscales_only = not args.include_scales

    if args.dry_run:
        dry_run(models, variants, args.n_replications, args.n_probes, subscales_only)
        return

    if args.summary_only:
        summary = collect_summary(models, variants)
        print(f"geometry summary rows: {len(summary)}")
        return

    dry_run(models, variants, args.n_replications, args.n_probes, subscales_only)
    for model in models:
        for variant in variants:
            if variant.instruct_only and not model.is_instruct:
                continue
            run_variant(
                model,
                variant,
                n_replications=args.n_replications,
                n_probes=args.n_probes,
                subscales_only=subscales_only,
                skip_existing=args.skip_existing,
                analysis_only=args.analysis_only,
            )

    summary = collect_summary(models, variants)
    print(f"geometry summary rows: {len(summary)}")


if __name__ == "__main__":
    main()
