#!/usr/bin/env python3
"""Run prompt/readout sensitivity controls for the cross-persona pipeline.

The default variants are intentionally small and targeted:

- no_scale_name: remove the source scale name from the persona prompt.
- chat_template: use each instruct model's tokenizer chat template.

The default readouts compare argmax responses against probability-weighted
expected-value responses over the same valid Likert options.

Profile-level manipulation controls are handled by
scripts/behavior/generate_random_factorial.py. An optional profile_permutation
variant is available as a permutation null, but it is not the default because
the paradigm has no independent subject identity to hold fixed while swapping
the response profile.

Raw generations are local staging data. Aggregate matrices and summaries are
written under outputs/prompt_variants/.
"""

from __future__ import annotations

import argparse
import gc
import importlib
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR / "src"))
sys.path.insert(0, str(BASE_DIR / "scripts" / "analysis"))

from psychometric_inference.model_registry import MODEL_BY_MECH_NAME, PRIMARY_MODELS, ModelSpec
from psychometric_inference.paths import (
    LLM_BEHAVIOR_PROMPT_VARIANTS_DIR,
    PROMPT_VARIANT_OUTPUT_DIR,
)
from psychometric_inference.pipeline.injectors import (
    ScaleAProfile,
    build_profile_from_csv_row,
    build_system_prompt,
)
from psychometric_inference.questionnaire_prompts import format_item_prompt
from psychometric_inference.scoring import un_reverse_df


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


SCALES = [
    ("IRI", "data/questionnaires/IRI.jsonl"),
    ("PANAS", "data/questionnaires/PANAS.jsonl"),
    ("POM", "data/questionnaires/POM.jsonl"),
    ("big_five", "data/questionnaires/big_five.jsonl"),
    ("in_inter_dependent", "data/questionnaires/in_inter_dependent.jsonl"),
    ("Life_Satisfaction", "data/questionnaires/Life_Satisfaction.jsonl"),
    ("Loneliness", "data/questionnaires/Loneliness.jsonl"),
]

HUMAN_DATASETS = ["SED", "SEDC", "SEDD"]
DEFAULT_VARIANTS = ("no_scale_name", "chat_template")
DEFAULT_MODELS = ("qwen14b_instruct",)
DEFAULT_READOUTS = ("argmax", "expected_value")
READOUTS = ("argmax", "expected_value")


@dataclass(frozen=True)
class VariantSpec:
    name: str
    system_template: str
    prompt_format: str
    profile_permutation: bool = False
    instruct_only: bool = False


VARIANTS = {
    "no_scale_name": VariantSpec(
        name="no_scale_name",
        system_template="questionnaire_no_scale_name",
        prompt_format="raw",
    ),
    "chat_template": VariantSpec(
        name="chat_template",
        system_template="questionnaire",
        prompt_format="chat_template",
        instruct_only=True,
    ),
    "profile_permutation": VariantSpec(
        name="profile_permutation",
        system_template="questionnaire",
        prompt_format="raw",
        profile_permutation=True,
    ),
}


class LocalModel:
    """Loads a HF model and scores valid response options by next-token logprob."""

    def __init__(self, spec: ModelSpec, prompt_format: str, device: str = "cuda"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info("Loading %s", spec.hf_id)
        self.tokenizer = AutoTokenizer.from_pretrained(spec.hf_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            spec.hf_id,
            torch_dtype="auto",
            device_map=device,
        )
        self.model.eval()
        self.spec = spec
        self.prompt_format = prompt_format

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def build_prompt(self, item_prompt: str, system_prompt: str = "") -> str:
        if self.prompt_format == "chat_template":
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": item_prompt})
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        if system_prompt:
            return f"{system_prompt}\n\n{item_prompt}\n\nAnswer: "
        return f"{item_prompt}\n\nAnswer: "

    def get_logprobs(self, prompt: str, options: list[str]) -> dict[str, float]:
        import torch

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        logits = outputs.logits[0, -1, :]
        log_probs = torch.log_softmax(logits, dim=-1)

        result = {}
        for opt in options:
            token_ids = self.tokenizer.encode(opt, add_special_tokens=False)
            if token_ids:
                result[opt] = log_probs[token_ids[0]].item()
        return result

    def cleanup(self) -> None:
        import torch

        del self.model
        del self.tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def load_scale_definition(jsonl_path: str) -> tuple[dict, list[dict]]:
    path = BASE_DIR / jsonl_path
    metadata, items = None, []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line.strip())
            if obj.get("_metadata"):
                metadata = obj
            else:
                items.append(obj)
    if metadata is None:
        raise ValueError(f"Missing metadata row in {path}")
    return metadata, items


def load_real_profiles(
    data_sources: Iterable[str],
    scale_def: dict,
    items: list[dict],
    scale_file: str,
) -> list[ScaleAProfile]:
    profiles = []
    for src in data_sources:
        src_path = BASE_DIR / src
        df = pd.read_csv(src_path)
        df = un_reverse_df(df, scale_file)
        id_col = next((c for c in ["Subject_ID", "ID", "Scan_ID", "id"] if c in df.columns), None)
        q_cols = [c for c in df.columns if c.startswith("Q")]
        for idx, row in df.iterrows():
            sid = str(row[id_col]) if id_col else f"S{idx:04d}"
            row_dict = {c: row[c] for c in q_cols}
            profiles.append(build_profile_from_csv_row(row_dict, scale_def, items, sid))
    return profiles


def profile_to_row(profile: ScaleAProfile, subject_id: str | None = None) -> dict:
    row = {"Subject_ID": subject_id or profile.subject_id}
    row.update({f"Q{it['item_number']}": it["response"] for it in profile.items})
    return row


def deranged_profiles(
    profiles: list[ScaleAProfile],
    seed: int,
    rotation_index: int,
) -> list[ScaleAProfile]:
    if len(profiles) < 2:
        return profiles

    rng = np.random.default_rng(seed + 1009 * rotation_index)
    idx = np.arange(len(profiles))
    for _ in range(100):
        perm = rng.permutation(idx)
        if np.all(perm != idx):
            return [profiles[i] for i in perm]

    # Deterministic fallback: a cyclic shift is a valid derangement for n > 1.
    return profiles[1:] + profiles[:1]


def option_probabilities(logprobs: dict[str, float]) -> dict[str, float]:
    if not logprobs:
        return {}
    labels = list(logprobs)
    vals = np.array([logprobs[label] for label in labels], dtype=float)
    finite = np.isfinite(vals)
    if not finite.any():
        return {}
    vals = vals[finite]
    labels = [label for label, keep in zip(labels, finite) if keep]
    weights = np.exp(vals - vals.max())
    probs = weights / weights.sum()
    return {label: float(prob) for label, prob in zip(labels, probs)}


def score_from_logprobs(logprobs: dict[str, float], readout: str) -> dict:
    probs = option_probabilities(logprobs)
    best = max(logprobs, key=logprobs.get) if logprobs else None
    expected = None
    if probs:
        expected = float(sum(float(opt) * prob for opt, prob in probs.items()))

    if readout == "argmax":
        response = int(best) if best is not None else None
        raw_text = str(best) if best is not None else ""
    elif readout == "expected_value":
        response = expected
        raw_text = "" if expected is None else f"{expected:.6f}"
    else:
        raise ValueError(f"Unknown readout mode: {readout}")

    return {
        "response": response,
        "raw_text": raw_text,
        "argmax_response": int(best) if best is not None else None,
        "expected_value": expected,
        "logprobs": {k: float(v) for k, v in logprobs.items()},
        "probabilities": probs,
        "confidence": "high" if response is not None else "failed",
    }


def experiment_name(model: ModelSpec, variant: VariantSpec, readout: str) -> str:
    return f"{model.mech_name}/{variant.name}/{readout}"


def data_root_for(model: ModelSpec, variant: VariantSpec, readout: str) -> Path:
    return LLM_BEHAVIOR_PROMPT_VARIANTS_DIR / experiment_name(model, variant, readout)


def output_root_for(model: ModelSpec, variant: VariantSpec, readout: str) -> Path:
    return PROMPT_VARIANT_OUTPUT_DIR / experiment_name(model, variant, readout)


def rotation_complete(output_dir: Path, expected_n: int) -> bool:
    progress_file = output_dir / ".progress.json"
    if not progress_file.exists():
        return False
    with progress_file.open() as f:
        progress = json.load(f)
    return len(progress) >= expected_n


def run_all_rotations(
    model: LocalModel,
    model_spec: ModelSpec,
    variant: VariantSpec,
    readouts: list[str],
    max_subjects: int | None = None,
    seed: int = 42,
) -> None:
    for scale_index, (scale_a_name, scale_a_def_path) in enumerate(SCALES, 1):
        output_dirs = {
            readout: data_root_for(model_spec, variant, readout) / f"persona_{scale_a_name}"
            for readout in readouts
        }
        expected_n = max_subjects or 272
        if all(rotation_complete(output_dir, expected_n) for output_dir in output_dirs.values()):
            print(f"  Round {scale_index}/7: {scale_a_name} already done")
            continue

        print(f"\n  Round {scale_index}/7: Persona = {scale_a_name}")
        scale_a_def, scale_a_items = load_scale_definition(scale_a_def_path)
        data_sources = [f"data/human/{ds}/{scale_a_name}.csv" for ds in HUMAN_DATASETS]
        target_profiles = load_real_profiles(data_sources, scale_a_def, scale_a_items, scale_a_name)
        if max_subjects and max_subjects < len(target_profiles):
            target_profiles = target_profiles[:max_subjects]

        prompt_profiles = (
            deranged_profiles(target_profiles, seed=seed, rotation_index=scale_index)
            if variant.profile_permutation
            else target_profiles
        )

        scale_b_configs = {}
        for sb_name, sb_def_path in SCALES:
            sb_def, sb_items = load_scale_definition(sb_def_path)
            scale_b_configs[sb_name] = {"definition": sb_def, "items": sb_items}

        progress_files = {
            readout: output_dir / ".progress.json"
            for readout, output_dir in output_dirs.items()
        }
        progress = {readout: {} for readout in readouts}
        for readout, progress_file in progress_files.items():
            if progress_file.exists():
                with progress_file.open() as f:
                    progress[readout] = json.load(f)

        results = {
            readout: {name: [] for name in scale_b_configs}
            for readout in readouts
        }
        items_per_subject = sum(len(sb["items"]) for sb in scale_b_configs.values())
        pbar = tqdm(
            total=len(target_profiles) * items_per_subject,
            desc=f"{model_spec.mech_name}/{variant.name}/{scale_a_name}",
            unit="item",
        )

        for target_profile, prompt_profile in zip(target_profiles, prompt_profiles):
            sid = target_profile.subject_id
            if all(sid in progress[readout] for readout in readouts):
                for readout in readouts:
                    for sb_name in scale_b_configs:
                        results[readout][sb_name].append(progress[readout][sid].get(sb_name, []))
                pbar.update(items_per_subject)
                continue

            sys_prompt = build_system_prompt(prompt_profile, variant.system_template)
            subj_data = {readout: {} for readout in readouts}
            for sb_name, sb_config in scale_b_configs.items():
                pbar.set_postfix(subj=sid, scale=sb_name)
                item_responses = {readout: [] for readout in readouts}
                for item in sb_config["items"]:
                    item_prompt = format_item_prompt(item, sb_config["definition"]["instruction"])
                    options = [str(o) for o in item["response_options"]]
                    full_prompt = model.build_prompt(item_prompt, sys_prompt)

                    try:
                        logprobs = model.get_logprobs(full_prompt, options)
                        for readout in readouts:
                            scored = score_from_logprobs(logprobs, readout)
                            item_responses[readout].append({"item_number": item["item_number"], **scored})
                    except Exception as e:
                        for readout in readouts:
                            item_responses[readout].append({
                                "item_number": item["item_number"],
                                "response": None,
                                "raw_text": str(e),
                                "confidence": "failed",
                            })
                    pbar.update(1)

                for readout in readouts:
                    results[readout][sb_name].append(item_responses[readout])
                    subj_data[readout][sb_name] = item_responses[readout]

            for readout in readouts:
                progress[readout][sid] = subj_data[readout]
                output_dirs[readout].mkdir(parents=True, exist_ok=True)
                with progress_files[readout].open("w") as f:
                    json.dump(progress[readout], f, ensure_ascii=False)

        pbar.close()
        for readout, output_dir in output_dirs.items():
            output_dir.mkdir(parents=True, exist_ok=True)

            pd.DataFrame([profile_to_row(p) for p in target_profiles]).to_csv(
                output_dir / f"{scale_a_name}_persona.csv",
                index=False,
            )
            if variant.profile_permutation:
                pd.DataFrame([
                    {
                        "Subject_ID": target.subject_id,
                        "Prompt_Subject_ID": prompt.subject_id,
                    }
                    for target, prompt in zip(target_profiles, prompt_profiles)
                ]).to_csv(output_dir / "persona_mapping.csv", index=False)
                pd.DataFrame([
                    profile_to_row(prompt, subject_id=target.subject_id)
                    for target, prompt in zip(target_profiles, prompt_profiles)
                ]).to_csv(output_dir / f"{scale_a_name}_prompted_persona.csv", index=False)

            for sb_name, all_resp in results[readout].items():
                rows = []
                for target_profile, item_responses_for_subject in zip(target_profiles, all_resp):
                    row = {"Subject_ID": target_profile.subject_id}
                    row.update({f"Q{ir['item_number']}": ir["response"] for ir in item_responses_for_subject})
                    rows.append(row)
                pd.DataFrame(rows).to_csv(output_dir / f"{sb_name}.csv", index=False)

            print(f"  Exported {len(target_profiles)} subjects to {output_dir}")


def run_analysis(model: ModelSpec, variant: VariantSpec, readout: str) -> None:
    llm_root = data_root_for(model, variant, readout)
    output_dir = output_root_for(model, variant, readout)
    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "run_config.json").write_text(
        json.dumps({
            "model": model.mech_name,
            "hf_id": model.hf_id,
            "variant": variant.name,
            "readout": readout,
            "system_template": variant.system_template,
            "prompt_format": variant.prompt_format,
            "profile_permutation": variant.profile_permutation,
        }, indent=2),
        encoding="utf-8",
    )

    sys.argv = ["x", "--llm_root", str(llm_root), "--output_dir", str(output_dir)]
    if "compute_behavior_structure" in sys.modules:
        importlib.reload(sys.modules["compute_behavior_structure"])
    from compute_behavior_structure import main as run_implicit
    run_implicit()

    sys.argv = ["x", "--results_dir", str(output_dir), "--model_name", f"{model.mech_name}_{variant.name}_{readout}"]
    if "compute_scale_structure" in sys.modules:
        importlib.reload(sys.modules["compute_scale_structure"])
    from compute_scale_structure import main as run_per_scale
    run_per_scale()


def collect_summary(
    models: list[ModelSpec],
    variants: list[VariantSpec],
    readouts: list[str],
) -> pd.DataFrame:
    rows = []

    for model in models:
        baseline_metrics = BASE_DIR / "outputs" / "behavior" / model.behavior_dir / "comparison_metrics.csv"
        if baseline_metrics.exists():
            df = pd.read_csv(baseline_metrics)
            for _, r in df.iterrows():
                rows.append({
                    "model": model.mech_name,
                    "variant": "baseline_raw",
                    "readout": "argmax",
                    "level": r["level"],
                    "matrix_r": r.get("matrix_r"),
                    "r": r.get("r"),
                    "slope": r.get("slope"),
                    "rmse": r.get("rmse"),
                    "n_pairs": r.get("n_pairs"),
                    "source": str(baseline_metrics.relative_to(BASE_DIR)),
                })

        for variant in variants:
            for readout in readouts:
                metrics_path = output_root_for(model, variant, readout) / "comparison_metrics.csv"
                if not metrics_path.exists():
                    continue
                df = pd.read_csv(metrics_path)
                for _, r in df.iterrows():
                    rows.append({
                        "model": model.mech_name,
                        "variant": variant.name,
                        "readout": readout,
                        "level": r["level"],
                        "matrix_r": r.get("matrix_r"),
                        "r": r.get("r"),
                        "slope": r.get("slope"),
                        "rmse": r.get("rmse"),
                        "n_pairs": r.get("n_pairs"),
                        "source": str(metrics_path.relative_to(BASE_DIR)),
                    })

    summary = pd.DataFrame(rows)
    PROMPT_VARIANT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(PROMPT_VARIANT_OUTPUT_DIR / "summary.csv", index=False)

    focus = summary[summary["level"].isin(["subscale", "subscale_between", "item_between", "scale"])]
    if not focus.empty:
        focus.to_csv(PROMPT_VARIANT_OUTPUT_DIR / "summary_focus.csv", index=False)
    return summary


def resolve_models(names: list[str]) -> list[ModelSpec]:
    aliases = {}
    for m in PRIMARY_MODELS:
        aliases[m.mech_name] = m
        aliases[m.behavior_dir] = m
        aliases[m.hf_id] = m

    models = []
    for name in names:
        if name not in aliases:
            valid = ", ".join(sorted(MODEL_BY_MECH_NAME))
            raise ValueError(f"Unknown model {name!r}. Valid short names: {valid}")
        models.append(aliases[name])
    return models


def resolve_variants(names: list[str]) -> list[VariantSpec]:
    variants = []
    for name in names:
        if name not in VARIANTS:
            valid = sorted(VARIANTS)
            raise ValueError(f"Unknown variant {name!r}. Valid variants: {', '.join(valid)}")
        variants.append(VARIANTS[name])
    return variants


def variant_complete(model: ModelSpec, variant: VariantSpec, readout: str, expected_n: int) -> bool:
    root = data_root_for(model, variant, readout)
    return all(rotation_complete(root / f"persona_{scale_name}", expected_n) for scale_name, _ in SCALES)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run cross-persona prompt variants")
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--variants", nargs="+", default=list(DEFAULT_VARIANTS))
    parser.add_argument("--readouts", nargs="+", choices=READOUTS, default=list(DEFAULT_READOUTS))
    parser.add_argument("--max_subjects", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--analysis_only", action="store_true")
    parser.add_argument("--summary_only", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    models = resolve_models(args.models)
    variants = resolve_variants(args.variants)
    expected_n = args.max_subjects or 272

    print("=" * 72)
    print("  PROMPT VARIANTS")
    print("=" * 72)
    for model in models:
        for variant in variants:
            for readout in args.readouts:
                if variant.instruct_only and not model.is_instruct:
                    status = "skip: instruct only"
                else:
                    done = variant_complete(model, variant, readout, expected_n)
                    status = "done" if done else "pending"
                print(f"  {model.mech_name:<20} {variant.name:<18} {readout:<15} {status}")

    if args.dry_run:
        return
    if args.summary_only:
        collect_summary(models, variants, args.readouts)
        return

    for model in models:
        for variant in variants:
            if variant.instruct_only and not model.is_instruct:
                logger.info("Skipping %s/%s: variant is instruct-only", model.mech_name, variant.name)
                continue

            completed = {
                readout: variant_complete(model, variant, readout, expected_n)
                for readout in args.readouts
            }

            if args.analysis_only:
                for readout, complete in completed.items():
                    if complete:
                        run_analysis(model, variant, readout)
                    else:
                        logger.warning("Missing staged data for %s/%s/%s", model.mech_name, variant.name, readout)
                continue

            if args.skip_existing:
                for readout, complete in completed.items():
                    if complete:
                        logger.info("Skipping generation for %s/%s/%s", model.mech_name, variant.name, readout)
                        run_analysis(model, variant, readout)

            pending_readouts = [
                readout for readout in args.readouts
                if not (args.skip_existing and completed[readout])
            ]
            if not pending_readouts:
                continue

            t0 = time.time()
            print("\n" + "#" * 72)
            print(f"  {model.hf_id} -> {model.mech_name}/{variant.name}")
            print(f"  readouts: {', '.join(pending_readouts)}")
            print("#" * 72)

            local_model = None
            try:
                local_model = LocalModel(model, prompt_format=variant.prompt_format, device=args.device)
                run_all_rotations(
                    local_model,
                    model,
                    variant,
                    pending_readouts,
                    max_subjects=args.max_subjects,
                    seed=args.seed,
                )
                for readout in pending_readouts:
                    run_analysis(model, variant, readout)
                elapsed = (time.time() - t0) / 60
                print(f"  Finished {model.mech_name}/{variant.name} in {elapsed:.1f} min")
            finally:
                if local_model is not None:
                    local_model.cleanup()

    summary = collect_summary(models, variants, args.readouts)
    print(f"\nSaved summary rows: {len(summary)}")
    print(f"Summary: {PROMPT_VARIANT_OUTPUT_DIR / 'summary_focus.csv'}")


if __name__ == "__main__":
    main()
