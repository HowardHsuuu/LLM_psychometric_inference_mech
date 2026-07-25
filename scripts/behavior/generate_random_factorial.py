#!/usr/bin/env python3
"""
Run profile-level control experiments.

Tests whether LLM cross-scale structure changes under manipulated response
profiles rather than relying only on prompt format or response marginals:
  - Random: uniform random item responses
  - Factorial: systematic subscale-level combinations (lo/mid/hi)
  - Collapsed: all items set to the same value (1, mid, max)

These are profile-level controls. They do not hold a separate subject identity
fixed, because the experimental persona is defined by the source-scale response
profile itself.

Runs 3 model sizes (0.5B, 3B, 14B base) × 7 persona rotations each.
Data stored in data/llm_behavior/{name}_random/ and data/llm_behavior/{name}_factorial/.
Results in outputs/behavior/{name}_random/ etc.

Usage:
    python generate_random_factorial.py                        # Run all
    python generate_random_factorial.py --mode random           # Random only
    python generate_random_factorial.py --mode factorial         # Factorial only
    python generate_random_factorial.py --mode collapsed         # Collapsed only
    python generate_random_factorial.py --skip_existing          # Resume
    python generate_random_factorial.py --analysis_only          # Re-run analysis
    python generate_random_factorial.py --device mps             # Apple Silicon
"""

import argparse
import gc
import json
import logging
import os
import sys
import time
from itertools import product
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
from psychometric_inference.model_registry import qwen_base_control_tuples
from psychometric_inference.questionnaire_prompts import format_item_prompt

# ── Scale definitions (same paths as generate_base_behavior.py) ──

SCALES = [
    ("IRI", "data/questionnaires/IRI.jsonl"),
    ("PANAS", "data/questionnaires/PANAS.jsonl"),
    ("POM", "data/questionnaires/POM.jsonl"),
    ("big_five", "data/questionnaires/big_five.jsonl"),
    ("in_inter_dependent", "data/questionnaires/in_inter_dependent.jsonl"),
    ("Life_Satisfaction", "data/questionnaires/Life_Satisfaction.jsonl"),
    ("Loneliness", "data/questionnaires/Loneliness.jsonl"),
]

MODELS = qwen_base_control_tuples()


# ── Model (same as generate_base_behavior.py) ──

class LocalModel:
    """Loads a HF model, provides logprob scoring."""

    def __init__(self, model_id: str, is_instruct: bool, device: str = "cuda"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info(f"Loading {model_id} (instruct={is_instruct})")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype="auto", device_map=device
        )
        self.model.eval()
        self.model_id = model_id
        self.is_instruct = is_instruct

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        logger.info(f"Loaded: {model_id}, instruct={is_instruct}")

    def build_prompt(self, item_prompt: str, system_prompt: str = "") -> str:
        if self.is_instruct:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": item_prompt})
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
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


# ── Profile generators ──

def generate_random_profiles(n_profiles, scale_def, items, seed=42):
    """Generate random item-level profiles (uniform over response range)."""
    from psychometric_inference.pipeline.injectors import ScaleAProfile

    rng = np.random.default_rng(seed)
    profiles = []
    for i in range(n_profiles):
        item_responses = []
        for item in items:
            opts = item["response_options"]
            resp = int(rng.choice(opts))
            item_responses.append({
                "item_number": item["item_number"],
                "text": item["text"],
                "response": resp,
            })
        profiles.append(ScaleAProfile(
            subject_id=f"RND{i:04d}",
            scale_name=scale_def.get("scale_name_zh", scale_def["scale_name"]),
            instruction=scale_def["instruction"],
            response_labels=scale_def["response_labels"],
            items=item_responses,
        ))
    return profiles


def generate_factorial_profiles(scale_def, items, levels=None):
    """Generate factorial profiles: all subscale-level combinations of lo/mid/hi."""
    from psychometric_inference.pipeline.injectors import ScaleAProfile

    subscales = scale_def.get("subscales", [])
    if not subscales:
        # Single-subscale scale: treat all items as one group
        lo, hi = scale_def["response_range"]
        mid = (lo + hi) // 2
        levels_to_use = levels or [lo, mid, hi]
        profiles = []
        for val in levels_to_use:
            item_responses = []
            for item in items:
                item_responses.append({
                    "item_number": item["item_number"],
                    "text": item["text"],
                    "response": val,
                })
            profiles.append(ScaleAProfile(
                subject_id=f"FAC_all={val}",
                scale_name=scale_def.get("scale_name_zh", scale_def["scale_name"]),
                instruction=scale_def["instruction"],
                response_labels=scale_def["response_labels"],
                items=item_responses,
            ))
        return profiles

    lo, hi = scale_def["response_range"]
    mid = (lo + hi) // 2
    levels_to_use = levels or [lo, mid, hi]

    item_to_subscale = {}
    for sub_idx, sub in enumerate(subscales):
        for item_num in sub["items"]:
            item_to_subscale[item_num] = sub_idx

    n_subscales = len(subscales)
    combos = list(product(levels_to_use, repeat=n_subscales))

    profiles = []
    for combo_idx, combo in enumerate(combos):
        item_responses = []
        for item in items:
            sub_idx = item_to_subscale.get(item["item_number"], 0)
            resp = combo[sub_idx]
            item_responses.append({
                "item_number": item["item_number"],
                "text": item["text"],
                "response": resp,
            })
        sub_desc = "_".join(f"{subscales[i]['name']}={combo[i]}" for i in range(n_subscales))
        profiles.append(ScaleAProfile(
            subject_id=f"FAC{combo_idx:04d}_{sub_desc}",
            scale_name=scale_def.get("scale_name_zh", scale_def["scale_name"]),
            instruction=scale_def["instruction"],
            response_labels=scale_def["response_labels"],
            items=item_responses,
        ))
    return profiles


def generate_collapsed_profiles(scale_def, items):
    """Generate collapsed profiles: all items set to same value (min, mid, max).

    This is the extreme valence test — if LLM is valence-driven,
    uniform-low and uniform-high should produce very different outputs,
    while uniform-mid should produce weak/ambiguous predictions.
    """
    from psychometric_inference.pipeline.injectors import ScaleAProfile

    lo, hi = scale_def["response_range"]
    mid = (lo + hi) // 2
    profiles = []

    for val, label in [(lo, "low"), (mid, "mid"), (hi, "high")]:
        item_responses = []
        for item in items:
            item_responses.append({
                "item_number": item["item_number"],
                "text": item["text"],
                "response": val,
            })
        profiles.append(ScaleAProfile(
            subject_id=f"COLL_{label}_{val}",
            scale_name=scale_def.get("scale_name_zh", scale_def["scale_name"]),
            instruction=scale_def["instruction"],
            response_labels=scale_def["response_labels"],
            items=item_responses,
        ))
    return profiles


# ── Core experiment loop ──

def run_all_rotations(model, experiment_name, mode, n_random=100,
                      factorial_levels=None, max_profiles=None):
    """Run 7 persona rotations with generated profiles."""
    from psychometric_inference.pipeline.injectors import build_system_prompt

    for i, (scale_a_name, scale_a_def_path) in enumerate(SCALES, 1):
        output_dir = BASE_DIR / "data/llm_behavior" / experiment_name / f"persona_{scale_a_name}"
        progress_file = output_dir / ".progress.json"

        # Load Scale A definition
        scale_a_def, scale_a_items = load_scale_definition(scale_a_def_path)

        # Generate profiles based on mode
        if mode == "random":
            profiles = generate_random_profiles(n_random, scale_a_def, scale_a_items)
        elif mode == "factorial":
            profiles = generate_factorial_profiles(scale_a_def, scale_a_items, factorial_levels)
        elif mode == "collapsed":
            profiles = generate_collapsed_profiles(scale_a_def, scale_a_items)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        if max_profiles and max_profiles < len(profiles):
            profiles = profiles[:max_profiles]

        # Check if already complete
        if progress_file.exists():
            with open(progress_file) as f:
                existing = json.load(f)
            if len(existing) >= len(profiles):
                print(f"  Round {i}/7: {scale_a_name} — already done ({len(existing)} profiles)")
                continue

        print(f"\n  Round {i}/7: Persona = {scale_a_name} ({len(profiles)} {mode} profiles)")

        # Load all Scale B definitions
        scale_b_configs = {}
        for sb_name, sb_def_path in SCALES:
            sb_def, sb_items = load_scale_definition(sb_def_path)
            scale_b_configs[sb_name] = {"definition": sb_def, "items": sb_items}

        # Resume support
        progress = {}
        if progress_file.exists():
            with open(progress_file) as f:
                progress = json.load(f)

        # Main loop
        results = {name: [] for name in scale_b_configs}
        items_per_subject = sum(len(sb["items"]) for sb in scale_b_configs.values())
        pbar = tqdm(total=len(profiles) * items_per_subject,
                    desc=f"R{i} {scale_a_name}", unit="item")

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

        # Export CSVs
        output_dir.mkdir(parents=True, exist_ok=True)

        # Scale A persona data
        rows = [{"Subject_ID": p.subject_id,
                 **{f"Q{it['item_number']}": it["response"] for it in p.items}}
                for p in profiles]
        pd.DataFrame(rows).to_csv(output_dir / f"{scale_a_name}_persona.csv", index=False)

        # Scale B results
        for sb_name, all_resp in results.items():
            rows = []
            for j, item_responses in enumerate(all_resp):
                row = {"Subject_ID": profiles[j].subject_id}
                for ir in item_responses:
                    row[f"Q{ir['item_number']}"] = ir["response"]
                rows.append(row)
            pd.DataFrame(rows).to_csv(output_dir / f"{sb_name}.csv", index=False)

        print(f"  Exported {len(profiles)} profiles to {output_dir}")


def run_analysis(experiment_name):
    """Run implicit structure analysis on collected data."""
    llm_root = str(BASE_DIR / "data/llm_behavior" / experiment_name)
    output_dir = str(BASE_DIR / "outputs/behavior" / experiment_name)
    os.makedirs(output_dir, exist_ok=True)

    import importlib

    sys.argv = ["x", "--llm_root", llm_root, "--output_dir", output_dir]
    if "compute_behavior_structure" in sys.modules:
        importlib.reload(sys.modules["compute_behavior_structure"])
    from compute_behavior_structure import main as a1
    a1()


# ── Main ──

def main():
    parser = argparse.ArgumentParser(
        description="Run random/factorial/collapsed persona experiments"
    )
    parser.add_argument("--mode", default="all",
                        choices=["all", "random", "factorial", "collapsed"],
                        help="Which experiment mode to run")
    parser.add_argument("--n_random", type=int, default=100,
                        help="Number of random profiles per rotation (default: 100)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--analysis_only", action="store_true")
    parser.add_argument("--max_profiles", type=int, default=None,
                        help="Limit profiles per rotation (for testing)")
    args = parser.parse_args()

    modes = ["random", "factorial", "collapsed"] if args.mode == "all" else [args.mode]

    # Build experiment list: (model_id, model_name, is_instruct, mode, experiment_name)
    experiments = []
    for model_id, model_name, is_instruct in MODELS:
        for mode in modes:
            exp_name = f"{model_name}_{mode}"
            experiments.append((model_id, model_name, is_instruct, mode, exp_name))

    print(f"{'='*70}")
    print(f"  RANDOM / FACTORIAL / COLLAPSED EXPERIMENTS")
    print(f"  {len(experiments)} experiments ({len(MODELS)} models × {len(modes)} modes)")
    print(f"{'='*70}")
    for model_id, model_name, _, mode, exp_name in experiments:
        print(f"  {model_id:<35} {mode:<12} → {exp_name}")

    # Group by model to load each model only once
    from collections import defaultdict
    model_experiments = defaultdict(list)
    for model_id, model_name, is_instruct, mode, exp_name in experiments:
        model_experiments[(model_id, model_name, is_instruct)].append((mode, exp_name))

    summary = []

    for (model_id, model_name, is_instruct), mode_list in model_experiments.items():

        # Check if all experiments for this model are done
        if args.analysis_only:
            for mode, exp_name in mode_list:
                llm_root = BASE_DIR / "data/llm_behavior" / exp_name
                if llm_root.exists():
                    print(f"\n  Analysis: {exp_name}")
                    run_analysis(exp_name)
            continue

        # Check skip
        all_done = True
        for mode, exp_name in mode_list:
            if not all((BASE_DIR / "data/llm_behavior" / exp_name / f"persona_{s}").exists()
                       for s, _ in SCALES):
                all_done = False
                break

        if args.skip_existing and all_done:
            print(f"\n  Skipping {model_id} (all modes complete)")
            for mode, exp_name in mode_list:
                run_analysis(exp_name)
            continue

        t0 = time.time()
        print(f"\n{'#'*70}")
        print(f"  {model_id} ({'Instruct' if is_instruct else 'Base'})")
        print(f"{'#'*70}")

        try:
            model = LocalModel(model_id, is_instruct, args.device)

            for mode, exp_name in mode_list:
                print(f"\n  === Mode: {mode} → {exp_name} ===")
                run_all_rotations(
                    model, exp_name, mode,
                    n_random=args.n_random,
                    max_profiles=args.max_profiles,
                )
                run_analysis(exp_name)

            model.cleanup()
            elapsed = (time.time() - t0) / 60
            for mode, exp_name in mode_list:
                summary.append({"model": model_id, "name": exp_name,
                               "mode": mode, "status": "success",
                               "time_minutes": round(elapsed / len(mode_list), 1)})

        except Exception as e:
            logger.error(f"Failed: {model_id}: {e}", exc_info=True)
            elapsed = (time.time() - t0) / 60
            for mode, exp_name in mode_list:
                summary.append({"model": model_id, "name": exp_name,
                               "mode": mode, "status": f"failed: {e}",
                               "time_minutes": round(elapsed / len(mode_list), 1)})
            try:
                gc.collect()
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except:
                pass

    # Summary
    print(f"\n{'='*70}")
    print("  DONE")
    print(f"{'='*70}")
    for r in summary:
        print(f"  {r['model']:<35} {r['mode']:<12} {r['status']:<10} ({r['time_minutes']} min)")

    if summary:
        os.makedirs(BASE_DIR / "outputs/behavior", exist_ok=True)
        pd.DataFrame(summary).to_csv(
            BASE_DIR / "outputs/behavior" / "random_factorial_summary.csv", index=False)


if __name__ == "__main__":
    main()
