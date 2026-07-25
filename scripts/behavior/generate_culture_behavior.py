#!/usr/bin/env python3
"""
generate_culture_behavior.py — Culture-condition scaling experiment.

Runs the same scaling experiment as generate_instruct_behavior.py but with the
"Taiwanese college student" culture-framed prompt (now in injectors.py).
Both base and instruct models use the same uniform prompt format:
    raw text + "Answer: "  (no chat template)

This is designed to be compared against the no-culture baseline:
    base models    → data/llm_behavior/{name}_v2/      (from generate_base_behavior.py)
    instruct models → data/llm_behavior/{name}_v3/     (from generate_instruct_behavior.py)

All culture-condition outputs go to a separate top-level directory:
    data/llm_behavior_culture/{name}_culture/persona_{scale}/
    outputs/behavior_culture/{name}_culture/

After data collection, runs:
  Step 2: Culture-condition self-analysis (r, slope, intercept, RMSE)
  Step 3: Culture vs No-culture comparison plots + summary CSV

Usage:
    python generate_culture_behavior.py
    python generate_culture_behavior.py --skip_existing
    python generate_culture_behavior.py --analysis_only
    python generate_culture_behavior.py --max_subjects 10 --device cuda
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
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy.stats import pearsonr
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR / "src"))
sys.path.insert(0, str(BASE_DIR / "scripts" / "analysis"))

from psychometric_inference.scoring import un_reverse_df, SCORING_RULES, ALL_SUBSCALES, compute_subscale_scores
from psychometric_inference.model_registry import culture_model_tuples
from psychometric_inference.questionnaire_prompts import format_item_prompt

# ── Constants ──

SCALES = [
    ("IRI",               "data/questionnaires/IRI.jsonl"),
    ("PANAS",             "data/questionnaires/PANAS.jsonl"),
    ("POM",               "data/questionnaires/POM.jsonl"),
    ("big_five",          "data/questionnaires/big_five.jsonl"),
    ("in_inter_dependent","data/questionnaires/in_inter_dependent.jsonl"),
    ("Life_Satisfaction", "data/questionnaires/Life_Satisfaction.jsonl"),
    ("Loneliness",        "data/questionnaires/Loneliness.jsonl"),
]

SCALE_NAMES = ["IRI", "PANAS", "POM", "BigFive", "SelfConst", "LifeSat", "Lonely"]

HUMAN_DATASETS = ["SED", "SEDC", "SEDD"]
HUMAN_DIRS = [f"data/human/{d}" for d in HUMAN_DATASETS]

# LLM data root for culture condition
CULTURE_DATA_ROOT  = BASE_DIR / "data/llm_behavior_culture"
CULTURE_RESULTS_ROOT = BASE_DIR / "outputs/behavior_culture"

# No-culture baseline mapping: culture_name → (no_culture_dir, label)
# base  → _v2 (uniform "Answer: " format from generate_base_behavior.py)
# instruct → _v3 (uniform "Answer: " format from generate_instruct_behavior.py)
DEFAULT_MODELS = culture_model_tuples()


# ══════════════════════════════════════════════════════════════════
#  STEP 1 — DATA COLLECTION
# ══════════════════════════════════════════════════════════════════

class LocalModel:
    """All models use the same raw text + 'Answer: ' prompt format (no chat template).
    This is identical to generate_instruct_behavior.py — isolates weights, not format."""

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
        """Always use raw text + 'Answer: ' — same format for base and instruct."""
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
    """Load real profiles, un-reversing the already-reversed CSV scores."""
    from psychometric_inference.pipeline.injectors import build_profile_from_csv_row
    profiles = []
    for src in data_sources:
        src_path = BASE_DIR / src
        if not src_path.exists():
            logger.warning(f"Missing: {src_path}")
            continue
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


def run_all_rotations(model: LocalModel, experiment_name: str, max_subjects: int = None):
    """Run all 7 persona rotations for one model."""
    from psychometric_inference.pipeline.injectors import build_system_prompt

    output_root = CULTURE_DATA_ROOT / experiment_name
    expected_n = max_subjects or 272

    for i, (scale_a_name, scale_a_def_path) in enumerate(SCALES, 1):
        output_dir = output_root / f"persona_{scale_a_name}"
        progress_file = output_dir / ".progress.json"

        # Check if already complete
        if progress_file.exists():
            with open(progress_file) as f:
                existing = json.load(f)
            if len(existing) >= expected_n:
                print(f"  Round {i}/7: {scale_a_name} — already done ({len(existing)} subjects), skipping")
                continue

        print(f"\n  Round {i}/7: Persona = {scale_a_name}")

        scale_a_def, scale_a_items = load_scale_definition(scale_a_def_path)
        data_sources = [f"data/human/{ds}/{scale_a_name}.csv" for ds in HUMAN_DATASETS]
        profiles = load_real_profiles(data_sources, scale_a_def, scale_a_items, scale_file=scale_a_name)
        if max_subjects and max_subjects < len(profiles):
            profiles = profiles[:max_subjects]

        # All 7 scales as Scale B (self-filling included)
        scale_b_configs = {}
        for sb_name, sb_def_path in SCALES:
            sb_def, sb_items = load_scale_definition(sb_def_path)
            scale_b_configs[sb_name] = {"definition": sb_def, "items": sb_items}

        # Resume from checkpoint
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

            # Build system prompt using current injectors.py (Taiwanese college student version)
            sys_prompt = build_system_prompt(profile, "questionnaire")
            subj_data = {}

            for sb_name, sb_config in scale_b_configs.items():
                pbar.set_postfix(subj=sid, scale=sb_name)
                item_responses = []

                for item in sb_config["items"]:
                    item_prompt = format_item_prompt(item, sb_config["definition"]["instruction"])
                    options = [str(o) for o in item["response_options"]]
                    full_prompt = model.build_prompt(item_prompt, sys_prompt)

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

        # Export CSVs (same format as data/human)
        output_dir.mkdir(parents=True, exist_ok=True)
        persona_rows = [
            {"Subject_ID": p.subject_id, **{f"Q{it['item_number']}": it["response"] for it in p.items}}
            for p in profiles
        ]
        pd.DataFrame(persona_rows).to_csv(output_dir / f"{scale_a_name}_persona.csv", index=False)

        for sb_name, all_resp in results.items():
            rows = []
            for j, item_responses in enumerate(all_resp):
                row = {"Subject_ID": profiles[j].subject_id}
                for ir in item_responses:
                    row[f"Q{ir['item_number']}"] = ir["response"]
                rows.append(row)
            pd.DataFrame(rows).to_csv(output_dir / f"{sb_name}.csv", index=False)

        print(f"  Exported {len(profiles)} subjects → {output_dir}")


# ══════════════════════════════════════════════════════════════════
#  STEP 2 — CULTURE-CONDITION SELF-ANALYSIS
# ══════════════════════════════════════════════════════════════════

def run_self_analysis(experiment_name: str):
    """Run implicit structure + per-scale analysis for culture condition."""
    import importlib

    llm_root = str(CULTURE_DATA_ROOT / experiment_name)
    output_dir = str(CULTURE_RESULTS_ROOT / experiment_name)
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n  [Self-analysis] {experiment_name}")
    print(f"  LLM root:   {llm_root}")
    print(f"  Output dir: {output_dir}")

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


# ══════════════════════════════════════════════════════════════════
#  STEP 3 — CULTURE vs NO-CULTURE COMPARISON
# ══════════════════════════════════════════════════════════════════

def load_metrics_csv(results_dir: str) -> pd.DataFrame:
    """Load comparison_metrics.csv from an implicit analysis output dir."""
    p = Path(results_dir) / "comparison_metrics.csv"
    if not p.exists():
        return None
    return pd.read_csv(p)


def load_per_scale_csv(results_dir: str) -> pd.DataFrame:
    """Load per_scale_alignment.csv from an implicit analysis output dir."""
    p = Path(results_dir) / "per_scale_alignment.csv"
    if not p.exists():
        return None
    return pd.read_csv(p)


def extract_overall_metrics(df: pd.DataFrame) -> dict:
    """Extract key overall metrics (item, subscale, scale level r, slope, rmse)."""
    if df is None:
        return {}
    out = {}
    for _, row in df.iterrows():
        level = row["level"]
        for col in ["matrix_r", "rmse", "r", "slope"]:
            if col in row and not pd.isna(row[col]):
                key = f"{level}_{col}"
                out[key] = row[col]
    return out


def build_comparison_table(models_info: list) -> pd.DataFrame:
    """
    Build a flat comparison table with one row per model × metric,
    columns: model, size, family, type, metric, culture_val, noculture_val, delta.
    """
    rows = []
    for model_id, culture_name, is_instruct, noculture_name, size, family in models_info:
        mtype = "instruct" if is_instruct else "base"

        culture_dir   = str(CULTURE_RESULTS_ROOT / culture_name)
        noculture_dir = str(BASE_DIR / "outputs/behavior" / noculture_name)

        c_metrics = load_metrics_csv(culture_dir)
        n_metrics = load_metrics_csv(noculture_dir)
        c_ps      = load_per_scale_csv(culture_dir)
        n_ps      = load_per_scale_csv(noculture_dir)

        c_overall = extract_overall_metrics(c_metrics)
        n_overall = extract_overall_metrics(n_metrics)

        all_keys = set(c_overall) | set(n_overall)
        for key in sorted(all_keys):
            c_val = c_overall.get(key, np.nan)
            n_val = n_overall.get(key, np.nan)
            rows.append({
                "model_id": model_id,
                "culture_name": culture_name,
                "noculture_name": noculture_name,
                "size": size,
                "family": family,
                "type": mtype,
                "metric_source": "overall",
                "metric": key,
                "culture": c_val,
                "noculture": n_val,
                "delta": c_val - n_val if not np.isnan(c_val) and not np.isnan(n_val) else np.nan,
            })

        # Per-scale metrics
        if c_ps is not None and n_ps is not None:
            merged_ps = c_ps.merge(n_ps, on="scale", suffixes=("_c", "_n"))
            for _, ps_row in merged_ps.iterrows():
                scale = ps_row["scale"]
                for metric_base in ["subscale_r", "subscale_slope", "subscale_rmse",
                                    "item_r", "item_slope", "item_rmse"]:
                    c_col = f"{metric_base}_c"
                    n_col = f"{metric_base}_n"
                    if c_col in ps_row and n_col in ps_row:
                        c_val = ps_row[c_col]
                        n_val = ps_row[n_col]
                        rows.append({
                            "model_id": model_id,
                            "culture_name": culture_name,
                            "noculture_name": noculture_name,
                            "size": size,
                            "family": family,
                            "type": mtype,
                            "metric_source": f"per_scale_{scale}",
                            "metric": metric_base,
                            "culture": c_val,
                            "noculture": n_val,
                            "delta": c_val - n_val if not np.isnan(c_val) and not np.isnan(n_val) else np.nan,
                        })

    return pd.DataFrame(rows)


def plot_delta_by_size(comparison_df: pd.DataFrame, output_dir: str):
    """
    For each key metric, plot delta (culture − no-culture) vs model size.
    Base and instruct shown as separate line series, Qwen and Llama distinguished by marker.
    """
    os.makedirs(output_dir, exist_ok=True)

    key_metrics = [
        ("item_matrix_r",      "Item-level matrix r"),
        ("subscale_matrix_r",  "Subscale-level matrix r"),
        ("scale_matrix_r",     "Scale-level matrix r"),
        ("item_between_slope", "Item between-scale slope"),
        ("subscale_between_slope", "Subscale between-scale slope"),
        ("item_between_r",     "Item between-scale r"),
        ("subscale_between_r", "Subscale between-scale r"),
    ]

    overall_df = comparison_df[comparison_df["metric_source"] == "overall"]

    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    axes = axes.flatten()

    colors = {"base": "#2166ac", "instruct": "#d6604d"}
    markers = {"Qwen": "o", "Llama": "s"}

    for ax_idx, (metric_key, metric_label) in enumerate(key_metrics):
        if ax_idx >= len(axes):
            break
        ax = axes[ax_idx]
        sub = overall_df[overall_df["metric"] == metric_key].dropna(subset=["delta"])
        if sub.empty:
            ax.set_visible(False)
            continue

        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)

        for mtype in ["base", "instruct"]:
            for family in ["Qwen", "Llama"]:
                grp = sub[(sub["type"] == mtype) & (sub["family"] == family)].sort_values("size")
                if grp.empty:
                    continue
                label = f"{family} {mtype}"
                ax.plot(grp["size"], grp["delta"],
                        color=colors[mtype], marker=markers[family],
                        linestyle="-", linewidth=1.5, markersize=7,
                        label=label, alpha=0.85)

        ax.set_title(f"Δ {metric_label}\n(culture − no-culture)", fontsize=10)
        ax.set_xlabel("Model size (B)", fontsize=9)
        ax.set_ylabel("Delta", fontsize=9)
        ax.legend(fontsize=7, loc="best")
        ax.grid(True, alpha=0.3)

    # Hide unused axes
    for i in range(len(key_metrics), len(axes)):
        axes[i].set_visible(False)

    plt.suptitle("Culture vs No-Culture: Δ Metrics by Model Size", fontsize=14, y=1.01)
    plt.tight_layout()
    fpath = os.path.join(output_dir, "delta_by_size.png")
    plt.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fpath}")


def plot_per_scale_delta(comparison_df: pd.DataFrame, output_dir: str):
    """
    For each scale, show delta of subscale_r across models.
    Grouped bar chart: each group = one scale, bars = models sorted by size.
    """
    os.makedirs(output_dir, exist_ok=True)

    per_scale_df = comparison_df[
        (comparison_df["metric_source"].str.startswith("per_scale_")) &
        (comparison_df["metric"] == "subscale_r")
    ].copy()

    if per_scale_df.empty:
        print("  No per-scale data available for delta plot.")
        return

    per_scale_df["scale"] = per_scale_df["metric_source"].str.replace("per_scale_", "")
    per_scale_df["label"] = per_scale_df.apply(
        lambda r: f"{r['family'][0]}{r['size']}{'i' if r['type']=='instruct' else 'b'}", axis=1
    )
    per_scale_df = per_scale_df.sort_values(["family", "type", "size"])

    scales = [s for s in SCALE_NAMES if s in per_scale_df["scale"].values]
    labels = per_scale_df["label"].unique()
    x = np.arange(len(scales))
    n_models = len(labels)
    width = 0.8 / max(n_models, 1)

    fig, ax = plt.subplots(figsize=(16, 6))
    cmap = plt.get_cmap("tab20", n_models)

    for i, label in enumerate(labels):
        grp = per_scale_df[per_scale_df["label"] == label]
        deltas = []
        for scale in scales:
            row = grp[grp["scale"] == scale]
            deltas.append(row["delta"].values[0] if not row.empty else np.nan)
        offset = (i - n_models / 2 + 0.5) * width
        ax.bar(x + offset, deltas, width * 0.9, label=label, color=cmap(i), alpha=0.8)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(scales, fontsize=10)
    ax.set_ylabel("Δ subscale_r (culture − no-culture)", fontsize=11)
    ax.set_title("Per-Scale Culture Effect: Δ Subscale Alignment r", fontsize=13)
    ax.legend(fontsize=7, ncol=4, loc="upper right")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    fpath = os.path.join(output_dir, "per_scale_delta_subscale_r.png")
    plt.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fpath}")


def plot_culture_vs_noculture_scatter(comparison_df: pd.DataFrame, output_dir: str):
    """
    Scatter: x = no-culture value, y = culture value, one point per model.
    Diagonal = no change. Points above diagonal = culture improved.
    One panel per key metric.
    """
    os.makedirs(output_dir, exist_ok=True)

    key_metrics = [
        ("item_matrix_r",         "Item matrix r"),
        ("subscale_matrix_r",     "Subscale matrix r"),
        ("scale_matrix_r",        "Scale matrix r"),
        ("item_between_slope",    "Item between-scale slope"),
        ("subscale_between_slope","Subscale between-scale slope"),
    ]

    overall_df = comparison_df[comparison_df["metric_source"] == "overall"]

    fig, axes = plt.subplots(1, len(key_metrics), figsize=(5 * len(key_metrics), 5))
    if len(key_metrics) == 1:
        axes = [axes]

    colors = {"base": "#2166ac", "instruct": "#d6604d"}
    markers = {"Qwen": "o", "Llama": "s"}

    for ax, (metric_key, metric_label) in zip(axes, key_metrics):
        sub = overall_df[overall_df["metric"] == metric_key].dropna(subset=["culture", "noculture"])
        if sub.empty:
            ax.set_visible(False)
            continue

        all_vals = pd.concat([sub["culture"], sub["noculture"]]).dropna()
        vmin, vmax = all_vals.min(), all_vals.max()
        pad = (vmax - vmin) * 0.1 or 0.1
        ax.plot([vmin - pad, vmax + pad], [vmin - pad, vmax + pad], "k--", alpha=0.4, linewidth=1)

        for _, row in sub.iterrows():
            ax.scatter(row["noculture"], row["culture"],
                       color=colors[row["type"]], marker=markers[row["family"]],
                       s=70, alpha=0.85, zorder=3)

        ax.set_xlabel("No-culture", fontsize=10)
        ax.set_ylabel("Culture", fontsize=10)
        ax.set_title(metric_label, fontsize=11)
        ax.grid(True, alpha=0.3)

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#2166ac", markersize=9, label="Base / Qwen"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#2166ac", markersize=9, label="Base / Llama"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#d6604d", markersize=9, label="Instruct / Qwen"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#d6604d", markersize=9, label="Instruct / Llama"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=4, fontsize=9,
               bbox_to_anchor=(0.5, -0.08))

    plt.suptitle("Culture vs No-Culture (above diagonal = culture helped)", fontsize=13)
    plt.tight_layout()
    fpath = os.path.join(output_dir, "culture_vs_noculture_scatter.png")
    plt.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fpath}")


def run_comparison(models_info: list):
    """Run full culture vs no-culture comparison and save outputs."""
    output_dir = str(CULTURE_RESULTS_ROOT / "_comparison")
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  STEP 3: Culture vs No-Culture Comparison")
    print(f"{'='*60}")

    comparison_df = build_comparison_table(models_info)

    if comparison_df.empty:
        print("  No data available for comparison (run data collection + analysis first).")
        return

    # Save full table
    csv_path = os.path.join(output_dir, "culture_vs_noculture.csv")
    comparison_df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")

    # Save readable summary (overall metrics only, one row per model)
    overall = comparison_df[comparison_df["metric_source"] == "overall"]
    pivot = overall.pivot_table(
        index=["culture_name", "size", "family", "type"],
        columns="metric",
        values=["culture", "noculture", "delta"],
        aggfunc="first",
    )
    pivot.columns = ["_".join(c).strip() for c in pivot.columns]
    pivot.reset_index().to_csv(os.path.join(output_dir, "summary_pivot.csv"), index=False)

    # Print readable delta summary
    print("\n  Culture − No-Culture Δ (overall metrics):")
    print(f"  {'Model':<35} {'size':>5} {'type':>8}  item_r  sub_r  scale_r  item_slope  sub_slope")
    print("  " + "-" * 90)

    key_metrics_print = [
        "item_matrix_r", "subscale_matrix_r", "scale_matrix_r",
        "item_between_slope", "subscale_between_slope",
    ]
    for _, row in overall[overall["metric"] == "item_matrix_r"].iterrows():
        name = row["culture_name"]
        vals = []
        for m in key_metrics_print:
            sub = overall[(overall["culture_name"] == name) & (overall["metric"] == m)]
            vals.append(f"{sub['delta'].values[0]:+.3f}" if not sub.empty and not np.isnan(sub["delta"].values[0]) else "   N/A")
        print(f"  {name:<35} {row['size']:>5.1f} {row['type']:>8}  {'  '.join(vals)}")

    # Plots
    plot_delta_by_size(comparison_df, output_dir)
    plot_per_scale_delta(comparison_df, output_dir)
    plot_culture_vs_noculture_scatter(comparison_df, output_dir)

    print(f"\n  All comparison outputs saved to: {output_dir}")


# ══════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Culture-condition scaling: data collection + self-analysis + culture vs no-culture comparison"
    )
    parser.add_argument("--max_subjects", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip models whose data already exists")
    parser.add_argument("--analysis_only", action="store_true",
                        help="Skip data collection, only run analysis")
    parser.add_argument("--comparison_only", action="store_true",
                        help="Skip collection + self-analysis, only run culture vs no-culture comparison")
    args = parser.parse_args()

    CULTURE_DATA_ROOT.mkdir(parents=True, exist_ok=True)
    CULTURE_RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

    print(f"{'='*70}")
    print(f"  SCALING CULTURE: Taiwanese college student prompt")
    print(f"  Uniform prompt format (Answer: ) for all models")
    print(f"  Data root:    {CULTURE_DATA_ROOT}")
    print(f"  Results root: {CULTURE_RESULTS_ROOT}")
    print(f"{'='*70}")
    for mid, name, is_inst, noculture, size, family in DEFAULT_MODELS:
        print(f"  {mid:<48} → {name}  (vs {noculture})")

    summary = []

    for model_id, experiment_name, is_instruct, noculture_name, size, family in DEFAULT_MODELS:

        all_done = all(
            (CULTURE_DATA_ROOT / experiment_name / f"persona_{s[0]}").exists()
            for s in SCALES
        )

        # ── Step 1: Data collection ──
        if not args.analysis_only and not args.comparison_only:
            if args.skip_existing and all_done:
                print(f"\n  Skipping data collection for {experiment_name} (complete)")
            else:
                t0 = time.time()
                print(f"\n{'#'*70}")
                print(f"  {model_id}")
                print(f"{'#'*70}")
                try:
                    model = LocalModel(model_id, args.device)
                    run_all_rotations(model, experiment_name, args.max_subjects)
                    model.cleanup()
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
                    continue

        # ── Step 2: Self-analysis ──
        if not args.comparison_only:
            if all_done or args.analysis_only:
                try:
                    run_self_analysis(experiment_name)
                except Exception as e:
                    logger.error(f"Self-analysis failed for {experiment_name}: {e}", exc_info=True)

    # Print collection summary
    if summary:
        print(f"\n{'='*70}")
        print("  DATA COLLECTION SUMMARY")
        print(f"{'='*70}")
        for r in summary:
            print(f"  {r['model']:<50} {r['status']:<10} ({r['time_minutes']} min)")
        pd.DataFrame(summary).to_csv(
            CULTURE_RESULTS_ROOT / "collection_summary.csv", index=False
        )

    # ── Step 3: Culture vs No-culture comparison ──
    run_comparison(DEFAULT_MODELS)

    print(f"\n{'='*70}")
    print("  ALL DONE")
    print(f"  Culture data:    {CULTURE_DATA_ROOT}")
    print(f"  Culture results: {CULTURE_RESULTS_ROOT}")
    print(f"  Comparison:      {CULTURE_RESULTS_ROOT / '_comparison'}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
