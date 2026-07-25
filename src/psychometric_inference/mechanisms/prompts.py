"""
Build persona prompts for activation extraction.

Two types of prompts:
1. Real subject prompts: 272 human subjects' complete 114-item profiles
2. Contrastive prompts: high vs low on each subscale/scale with randomized backgrounds

All prompts use a unified "mega system prompt" format containing all 114 items
from all 7 scales, ensuring cross-subscale direction comparisons are valid
(same prompt context).
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .config import (
    HUMAN_DIRS, SCALES, SCALE_NAMES, ALL_SUBSCALES,
    N_REPLICATIONS,
)
from psychometric_inference.scoring import SCORING_RULES, FILENAME_TO_SCALE, un_reverse_df
from psychometric_inference.paths import questionnaire_path


# ── Scale definition loading ──

def load_all_scale_definitions() -> Dict[str, dict]:
    """Load metadata and items for all 7 scales from JSONL files.
    
    Returns:
        Dict mapping scale_file_name -> {"metadata": {...}, "items": [...]}
    """
    scales = {}
    for scale_file, scale_short in SCALES:
        jsonl_path = questionnaire_path(scale_file)
        if not jsonl_path.exists():
            raise FileNotFoundError(f"Cannot find {scale_file}.jsonl")

        metadata, items = None, []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line.strip())
                if obj.get("_metadata"):
                    metadata = obj
                else:
                    items.append(obj)

        scales[scale_file] = {
            "metadata": metadata,
            "items": items,
            "scale_short": scale_short,
        }
    return scales


# ── Mega system prompt builder ──

def build_mega_system_prompt(
    item_responses: Dict[str, Dict[int, int]],
    scale_defs: Dict[str, dict],
    variant: str = "default",
) -> str:
    """Build a system prompt containing responses to ALL 114 items across 7 scales.
    
    Args:
        item_responses: Dict mapping scale_file_name -> {item_number: response_value}
                       e.g., {"IRI": {1: 3, 2: 5, ...}, "PANAS": {1: 4, ...}, ...}
        scale_defs: Output of load_all_scale_definitions()
        variant: "default" includes scale names; "no_scale_name" omits them.
        
    Returns:
        Complete system prompt string.
    """
    sections = []

    for scale_file, scale_short in SCALES:
        sd = scale_defs[scale_file]
        meta = sd["metadata"]
        items = sd["items"]
        responses = item_responses.get(scale_file, {})

        scale_name = meta.get("scale_name_zh", meta.get("scale_name", scale_file))
        instruction = meta.get("instruction", "")
        labels = meta.get("response_labels", [])

        # Format response labels
        label_lines = []
        for i, label in enumerate(labels, 1):
            label_lines.append(f"  {i} = {label}")
        label_desc = "\n".join(label_lines)

        # Format item responses
        item_lines = []
        for item in items:
            q_num = item["item_number"]
            text = item["text"]
            resp = responses.get(q_num, "?")
            item_lines.append(f"Q{q_num}. {text} → {resp}")

        heading = "" if variant == "no_scale_name" else f"【{scale_name}】\n"
        section = (
            heading
            + f"{instruction}\n"
            + f"Response scale:\n{label_desc}\n\n"
            + "\n".join(item_lines)
        )
        sections.append(section)

    if variant not in {"default", "no_scale_name"}:
        raise ValueError(f"Unknown mega prompt variant: {variant}")

    prompt = (
        "You are completing psychological questionnaires as a person with the following responses:\n\n"
        + "\n\n".join(sections)
        + "\n\nRespond to all questions as this person would, maintaining consistency "
        "with this psychological profile."
    )
    return prompt


# ── Load real human subjects ──

def load_human_subjects(scale_defs: Dict[str, dict]) -> List[Dict]:
    """Load 272 human subjects' complete item-level responses.
    
    Returns:
        List of dicts, each: {
            "subject_id": str,
            "item_responses": {scale_file: {item_num: raw_response}},
            "subscale_scores": {subscale_name: score},
        }
    """
    # Load each scale's CSV, un-reverse, merge by row position
    all_scale_data = {}
    n_subjects = None

    for scale_file, scale_short in SCALES:
        frames = []
        for d in HUMAN_DIRS:
            fpath = d / f"{scale_file}.csv"
            if fpath.exists():
                df = pd.read_csv(fpath)
                q_cols = [c for c in df.columns if c.startswith("Q")]
                sub = df[q_cols].copy()
                # Un-reverse to get raw scores
                sub = un_reverse_df(sub, scale_file)
                frames.append(sub)

        if not frames:
            continue

        combined = pd.concat(frames, ignore_index=True)
        all_scale_data[scale_file] = combined

        if n_subjects is None:
            n_subjects = len(combined)
        else:
            assert len(combined) == n_subjects, (
                f"Mismatch: {scale_file} has {len(combined)} subjects, expected {n_subjects}"
            )

    # Build per-subject records
    subjects = []
    for i in range(n_subjects):
        item_responses = {}
        for scale_file, _ in SCALES:
            if scale_file not in all_scale_data:
                continue
            df = all_scale_data[scale_file]
            responses = {}
            for col in df.columns:
                q_num = int(col[1:])  # "Q1" -> 1
                responses[q_num] = int(df.iloc[i][col])
            item_responses[scale_file] = responses

        # Compute subscale scores from raw responses (need to apply reverse coding)
        subscale_scores = _compute_subscale_scores_for_subject(item_responses)

        subjects.append({
            "subject_id": f"S{i:04d}",
            "item_responses": item_responses,
            "subscale_scores": subscale_scores,
        })

    return subjects


def _compute_subscale_scores_for_subject(
    item_responses: Dict[str, Dict[int, int]],
) -> Dict[str, float]:
    """Compute 16 subscale scores for one subject from raw item responses."""
    scores = {}
    file_to_short = dict(SCALES)

    for scale_file, scale_short in SCALES:
        if scale_file not in item_responses:
            continue
        responses = item_responses[scale_file]
        rules = SCORING_RULES.get(scale_short)
        if rules is None:
            continue

        max_val = rules["max_val"]
        rev_items = set(rules["reverse_items"])

        for sub_name, item_nums in rules["subscales"].items():
            vals = []
            for q in item_nums:
                if q in responses:
                    v = responses[q]
                    # Apply reverse coding for subscale scoring
                    if q in rev_items:
                        v = max_val + 1 - v
                    vals.append(v)
            if vals:
                scores[f"{scale_short}_{sub_name}"] = np.mean(vals)

    return scores


# ── Contrastive condition generators ──

def generate_contrastive_prompts(
    target_subscale: str,
    scale_defs: Dict[str, dict],
    n_replications: int = N_REPLICATIONS,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict]]:
    """Generate high vs low contrastive prompts for a single subscale.
    
    For the target subscale, sets all its items to max (high) or min (low).
    All OTHER subscales' items are randomized (uniform over response range)
    to create diverse background prompts, with the same random backgrounds
    used for both high and low → proper paired contrasts.
    
    Args:
        target_subscale: e.g., "BigFive_Neuroticism"
        scale_defs: Output of load_all_scale_definitions()
        n_replications: Number of random background prompts
        seed: Random seed
        
    Returns:
        (high_prompts, low_prompts): Lists of item_responses dicts,
        matched pairwise (high_prompts[i] and low_prompts[i] share
        the same background, differing only on the target subscale).
    """
    rng = np.random.default_rng(seed)

    # Parse target subscale
    parts = target_subscale.split("_", 1)
    target_scale_short = parts[0]
    target_sub_name = parts[1]

    # Find which scale file and which items belong to target subscale
    target_scale_file = None
    for sf, ss in SCALES:
        if ss == target_scale_short:
            target_scale_file = sf
            break
    assert target_scale_file is not None, f"Cannot find scale for {target_subscale}"

    rules = SCORING_RULES[target_scale_short]
    target_item_nums = set(rules["subscales"][target_sub_name])
    reverse_items = set(rules["reverse_items"])
    lo = scale_defs[target_scale_file]["metadata"]["response_range"][0]
    hi = scale_defs[target_scale_file]["metadata"]["response_range"][1]

    high_prompts = []
    low_prompts = []

    for rep in range(n_replications):
        # Generate random background for ALL items
        bg_responses = {}
        for scale_file, scale_short in SCALES:
            sd = scale_defs[scale_file]
            meta = sd["metadata"]
            items = sd["items"]
            r_lo, r_hi = meta["response_range"]
            responses = {}
            for item in items:
                q = item["item_number"]
                responses[q] = int(rng.integers(r_lo, r_hi + 1))
            bg_responses[scale_file] = responses

        # Create high and low by overriding target subscale items
        # IMPORTANT: for reverse-coded items, high CONSTRUCT score = low RAW score
        # because the mega prompt shows raw scores with original item text.
        # e.g., BigFive Q9 "容易緊張的" is reverse-coded:
        #   raw=1 (strongly disagree) → high Neuroticism after scoring
        #   raw=7 (strongly agree) → low Neuroticism after scoring
        high_resp = {sf: dict(rs) for sf, rs in bg_responses.items()}
        low_resp = {sf: dict(rs) for sf, rs in bg_responses.items()}

        for q in target_item_nums:
            if q in reverse_items:
                # Reverse-coded: high construct = low raw, low construct = high raw
                high_resp[target_scale_file][q] = lo
                low_resp[target_scale_file][q] = hi
            else:
                # Normal: high construct = high raw
                high_resp[target_scale_file][q] = hi
                low_resp[target_scale_file][q] = lo

        high_prompts.append(high_resp)
        low_prompts.append(low_resp)

    return high_prompts, low_prompts


def generate_scale_contrastive_prompts(
    target_scale: str,
    scale_defs: Dict[str, dict],
    n_replications: int = N_REPLICATIONS,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict]]:
    """Generate high vs low contrastive prompts for an entire scale.
    
    Sets ALL subscales within the target scale to max (high) or min (low).
    Other scales' items are randomized.
    
    Args:
        target_scale: e.g., "BigFive" (scale short name)
        
    Returns:
        (high_prompts, low_prompts): Matched pairs.
    """
    rng = np.random.default_rng(seed)

    target_scale_file = None
    for sf, ss in SCALES:
        if ss == target_scale:
            target_scale_file = sf
            break
    assert target_scale_file is not None, f"Cannot find scale file for {target_scale}"

    sd = scale_defs[target_scale_file]
    lo = sd["metadata"]["response_range"][0]
    hi = sd["metadata"]["response_range"][1]
    all_target_items = {item["item_number"] for item in sd["items"]}

    # Get reverse-coded items for this scale
    target_scale_short = None
    for sf, ss in SCALES:
        if sf == target_scale_file:
            target_scale_short = ss
            break
    reverse_items = set(SCORING_RULES[target_scale_short]["reverse_items"])

    high_prompts = []
    low_prompts = []

    for rep in range(n_replications):
        bg_responses = {}
        for scale_file, scale_short in SCALES:
            sdef = scale_defs[scale_file]
            meta = sdef["metadata"]
            r_lo, r_hi = meta["response_range"]
            responses = {}
            for item in sdef["items"]:
                q = item["item_number"]
                responses[q] = int(rng.integers(r_lo, r_hi + 1))
            bg_responses[scale_file] = responses

        high_resp = {sf: dict(rs) for sf, rs in bg_responses.items()}
        low_resp = {sf: dict(rs) for sf, rs in bg_responses.items()}

        for q in all_target_items:
            if q in reverse_items:
                high_resp[target_scale_file][q] = lo
                low_resp[target_scale_file][q] = hi
            else:
                high_resp[target_scale_file][q] = hi
                low_resp[target_scale_file][q] = lo

        high_prompts.append(high_resp)
        low_prompts.append(low_resp)

    return high_prompts, low_prompts
