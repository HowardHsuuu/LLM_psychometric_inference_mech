#!/usr/bin/env python3
"""
generate_instruct_behavior.py — Control experiment: all models use the SAME prompt format.
Both base and instruct use raw text + "Answer: " (no chat template).
This isolates the effect of instruction tuning weights from prompt format.

Only instruct models need re-running (base _v2 results are already correct).
Results stored with _v3 suffix.

Usage:
    python generate_instruct_behavior.py --skip_existing
"""

import argparse
import gc
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR / "src"))
sys.path.insert(0, str(BASE_DIR / "scripts" / "analysis"))

from psychometric_inference.scoring import un_reverse_df
from psychometric_inference.model_registry import instruct_generation_tuples
from psychometric_inference.questionnaire_prompts import format_item_prompt

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

# Only instruct models — base _v2 already used "Answer: " format
DEFAULT_MODELS = instruct_generation_tuples()


class LocalModel:
    """All models use the same 'Answer: ' prompt format — no chat template."""

    def __init__(self, model_id: str, device: str = "cuda"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info(f"Loading {model_id}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype="auto", device_map=device
        )
        self.model.eval()
        self.model_id = model_id

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        logger.info(f"Loaded: {model_id}")

    def build_prompt(self, item_prompt: str, system_prompt: str = "") -> str:
        """Always use raw text + 'Answer: ' — same as base models."""
        if system_prompt:
            return f"{system_prompt}\n\n{item_prompt}\n\nAnswer: "
        return f"{item_prompt}\n\nAnswer: "

    def get_logprobs(self, prompt: str, options: List[str]) -> Dict[str, float]:
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

    def cleanup(self):
        import torch
        logger.info(f"Cleaning up {self.model_id}")
        del self.model
        del self.tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ── Data loading ──

def load_scale_definition(jsonl_path: str):
    path = BASE_DIR / jsonl_path if not os.path.isabs(jsonl_path) else Path(jsonl_path)
    metadata, items = None, []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line.strip())
            if obj.get("_metadata"):
                metadata = obj
            else:
                items.append(obj)
    return metadata, items


def load_real_profiles(data_sources, scale_def, items, scale_file=None):
    from psychometric_inference.pipeline.injectors import build_profile_from_csv_row
    profiles = []
    for src in data_sources:
        src_path = BASE_DIR / src
        df = pd.read_csv(src_path)
        if scale_file:
            df = un_reverse_df(df, scale_file)
        id_col = None
        for c in ["Subject_ID", "ID", "Scan_ID", "id"]:
            if c in df.columns:
                id_col = c
                break
        q_cols = [c for c in df.columns if c.startswith("Q")]
        for idx, row in df.iterrows():
            sid = str(row[id_col]) if id_col else f"S{idx:04d}"
            row_dict = {c: row[c] for c in q_cols}
            profiles.append(build_profile_from_csv_row(row_dict, scale_def, items, sid))
    return profiles


# ── Core experiment loop ──

def run_all_rotations(model, experiment_name, max_subjects=None):
    from psychometric_inference.pipeline.injectors import build_system_prompt

    for i, (scale_a_name, scale_a_def_path) in enumerate(SCALES, 1):
        output_dir = BASE_DIR / "data/llm_behavior" / experiment_name / f"persona_{scale_a_name}"
        progress_file = output_dir / ".progress.json"

        expected_n = max_subjects or 272
        if progress_file.exists():
            with open(progress_file) as f:
                existing = json.load(f)
            if len(existing) >= expected_n:
                print(f"  Round {i}/7: {scale_a_name} — already done ({len(existing)} subjects)")
                continue

        print(f"\n  Round {i}/7: Persona = {scale_a_name}")

        scale_a_def, scale_a_items = load_scale_definition(scale_a_def_path)
        data_sources = [f"data/human/{ds}/{scale_a_name}.csv" for ds in HUMAN_DATASETS]
        profiles = load_real_profiles(data_sources, scale_a_def, scale_a_items, scale_file=scale_a_name)
        if max_subjects and max_subjects < len(profiles):
            profiles = profiles[:max_subjects]

        scale_b_configs = {}
        for sb_name, sb_def_path in SCALES:
            sb_def, sb_items = load_scale_definition(sb_def_path)
            scale_b_configs[sb_name] = {"definition": sb_def, "items": sb_items}

        progress = {}
        if progress_file.exists():
            with open(progress_file) as f:
                progress = json.load(f)

        results = {name: [] for name in scale_b_configs}
        items_per_subject = sum(len(sb["items"]) for sb in scale_b_configs.values())
        pbar = tqdm(total=len(profiles) * items_per_subject, desc=f"R{i} {scale_a_name}", unit="item")

        for profile in profiles:
            sid = profile.subject_id

            if sid in progress:
                for sb_name in scale_b_configs:
                    results[sb_name].append(progress[sid].get(sb_name, []))
                pbar.update(items_per_subject)
                continue

            sys_prompt = build_system_prompt(profile, "questionnaire")
            subj_data = {}

            for sb_name, sb_config in scale_b_configs.items():
                pbar.set_postfix(subj=sid, scale=sb_name)
                item_responses = []

                for item in sb_config["items"]:
                    prompt = format_item_prompt(item, sb_config["definition"]["instruction"])
                    options = [str(o) for o in item["response_options"]]
                    full_prompt = model.build_prompt(prompt, sys_prompt)

                    try:
                        logprobs = model.get_logprobs(full_prompt, options)
                        best = max(logprobs, key=logprobs.get) if logprobs else None
                        item_responses.append({
                            "item_number": item["item_number"],
                            "response": int(best) if best else None,
                            "raw_text": best or "",
                            "confidence": "high" if best else "failed",
                        })
                    except Exception as e:
                        item_responses.append({
                            "item_number": item["item_number"],
                            "response": None,
                            "raw_text": str(e),
                            "confidence": "failed",
                        })

                    pbar.update(1)

                results[sb_name].append(item_responses)
                subj_data[sb_name] = item_responses

            progress[sid] = subj_data
            output_dir.mkdir(parents=True, exist_ok=True)
            with open(progress_file, "w") as f:
                json.dump(progress, f, ensure_ascii=False)

        pbar.close()

        output_dir.mkdir(parents=True, exist_ok=True)
        rows = [{"Subject_ID": p.subject_id, **{f"Q{it['item_number']}": it["response"] for it in p.items}} for p in profiles]
        pd.DataFrame(rows).to_csv(output_dir / f"{scale_a_name}_persona.csv", index=False)

        for sb_name, all_resp in results.items():
            rows = []
            for j, item_responses in enumerate(all_resp):
                row = {"Subject_ID": profiles[j].subject_id}
                for ir in item_responses:
                    row[f"Q{ir['item_number']}"] = ir["response"]
                rows.append(row)
            pd.DataFrame(rows).to_csv(output_dir / f"{sb_name}.csv", index=False)

        print(f"  Exported {len(profiles)} subjects to {output_dir}")


def run_analysis(experiment_name):
    llm_root = str(BASE_DIR / "data/llm_behavior" / experiment_name)
    output_dir = str(BASE_DIR / "outputs/behavior" / experiment_name)
    os.makedirs(output_dir, exist_ok=True)

    import importlib

    sys.argv = ["x", "--llm_root", llm_root, "--output_dir", output_dir]
    if "compute_behavior_structure" in sys.modules:
        importlib.reload(sys.modules["compute_behavior_structure"])
    from compute_behavior_structure import main as a1
    a1()

    sys.argv = ["x", "--results_dir", output_dir, "--model_name", experiment_name]
    if "compute_scale_structure" in sys.modules:
        importlib.reload(sys.modules["compute_scale_structure"])
    from compute_scale_structure import main as a2
    a2()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_subjects", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--analysis_only", action="store_true")
    args = parser.parse_args()

    print(f"{'='*70}")
    print(f"  SCALING v3: Uniform prompt format (Answer: ) for all models")
    print(f"  Only instruct models — base _v2 results reusable")
    print(f"{'='*70}")
    for mid, name, _ in DEFAULT_MODELS:
        skip = ""
        if args.skip_existing and all(
            (BASE_DIR / "data/llm_behavior" / name / f"persona_{s[0]}").exists() for s in SCALES
        ):
            skip = " [SKIP]"
        print(f"  {mid:<45} → {name}{skip}")

    summary = []

    for model_id, experiment_name, _ in DEFAULT_MODELS:
        all_done = all(
            (BASE_DIR / "data/llm_behavior" / experiment_name / f"persona_{s[0]}").exists()
            for s in SCALES
        )

        if args.analysis_only:
            if all_done:
                run_analysis(experiment_name)
            continue

        if args.skip_existing and all_done:
            print(f"\n  Skipping {model_id} (complete)")
            run_analysis(experiment_name)
            summary.append({"model": model_id, "name": experiment_name,
                           "status": "skipped", "time_minutes": 0})
            continue

        t0 = time.time()
        print(f"\n{'#'*70}")
        print(f"  {model_id} (uniform prompt)")
        print(f"{'#'*70}")

        try:
            model = LocalModel(model_id, args.device)
            run_all_rotations(model, experiment_name, args.max_subjects)
            model.cleanup()
            run_analysis(experiment_name)
            elapsed = (time.time() - t0) / 60
            summary.append({"model": model_id, "name": experiment_name,
                           "status": "success", "time_minutes": round(elapsed, 1)})
        except Exception as e:
            logger.error(f"Failed: {model_id}: {e}", exc_info=True)
            elapsed = (time.time() - t0) / 60
            summary.append({"model": model_id, "name": experiment_name,
                           "status": f"failed: {e}", "time_minutes": round(elapsed, 1)})
            try:
                gc.collect()
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except:
                pass

    print(f"\n{'='*70}")
    print("  DONE")
    print(f"{'='*70}")
    for r in summary:
        print(f"  {r['model']:<45} {r['status']:<10} ({r['time_minutes']} min)")

    os.makedirs(BASE_DIR / "outputs/behavior", exist_ok=True)
    pd.DataFrame(summary).to_csv(BASE_DIR / "outputs/behavior" / "scaling_v3_summary.csv", index=False)


if __name__ == "__main__":
    main()
