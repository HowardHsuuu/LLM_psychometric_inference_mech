"""
Scoring module: compute subscale scores from item-level data.

Handles reverse coding and subscale aggregation for all 7 scales.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

# ── Scoring rules ──
# Each scale: (max_value, reverse_items, subscales)
# subscales: dict of subscale_name -> list of 1-indexed item numbers

SCORING_RULES: Dict[str, dict] = {
    "IRI": {
        "max_val": 5,
        "reverse_items": [3, 4, 7, 12, 13, 14, 15, 18, 19],
        "subscales": {
            "Perspective_taking": [3, 8, 11, 15, 21, 25, 28],
            "Fantasy": [1, 5, 7, 12, 16, 23, 26],
            "Empathic_concern": [2, 4, 9, 14, 18, 20, 22],
            "Personal_distress": [6, 10, 13, 17, 19, 24, 27],
        },
    },
    "PANAS": {
        "max_val": 5,
        "reverse_items": [],
        "subscales": {
            "Positive_Affect": [1, 3, 5, 9, 10, 12, 14, 16, 17, 19],
            "Negative_Affect": [2, 4, 6, 7, 8, 11, 13, 15, 18, 20],
        },
    },
    "POM": {
        "max_val": 6,
        "reverse_items": [5, 7],
        "subscales": {
            "Peace_of_Mind": [1, 2, 3, 4, 5, 6, 7],
        },
    },
    "BigFive": {
        "max_val": 7,
        "reverse_items": [2, 6, 8, 9, 10],
        "subscales": {
            "Extraversion": [1, 6],
            "Agreeableness": [2, 7],
            "Conscientiousness": [3, 8],
            "Neuroticism": [4, 9],
            "Openness": [5, 10],
        },
    },
    "SelfConst": {
        "max_val": 7,
        "reverse_items": [],
        "subscales": {
            "Independent_self": [1, 6, 7, 9, 11, 12, 15, 17, 20, 21, 22, 23],
            "Interdependent_self": [2, 3, 4, 5, 8, 10, 13, 14, 16, 18, 19, 24],
        },
    },
    "LifeSat": {
        "max_val": 7,
        "reverse_items": [],
        "subscales": {
            "Life_Satisfaction": [1, 2, 3, 4, 5],
        },
    },
    "Lonely": {
        "max_val": 4,
        "reverse_items": [1, 5, 6, 9, 10, 15, 16, 19, 20],
        "subscales": {
            "Loneliness": list(range(1, 21)),
        },
    },
}

# Map CSV file names to internal scale short names
FILENAME_TO_SCALE = {
    "IRI": "IRI",
    "PANAS": "PANAS",
    "POM": "POM",
    "big_five": "BigFive",
    "in_inter_dependent": "SelfConst",
    "Life_Satisfaction": "LifeSat",
    "Loneliness": "Lonely",
}

SCALE_TO_FILENAME = {v: k for k, v in FILENAME_TO_SCALE.items()}


def reverse_code(value: float, max_val: int) -> float:
    """Reverse code a Likert item: new = max + 1 - old."""
    return max_val + 1 - value


def un_reverse_df(df: pd.DataFrame, scale_file: str) -> pd.DataFrame:
    """Undo reverse coding on a DataFrame loaded from data/human.

    The CSVs in data/human store already-reversed scores.
    This function converts them back to raw (original response) scores
    so the rest of the pipeline works with a consistent coding scheme.

    Args:
        df: DataFrame with Q-numbered columns (Q1, Q2, ... or {Scale}_Q1, ...).
        scale_file: CSV filename without extension (e.g. "IRI", "big_five").

    Returns:
        DataFrame with reverse items restored to raw scores.
    """
    scale_short = FILENAME_TO_SCALE.get(scale_file)
    if scale_short is None:
        return df

    rules = SCORING_RULES.get(scale_short)
    if rules is None or not rules["reverse_items"]:
        return df

    df = df.copy()
    max_val = rules["max_val"]
    rev_items = set(rules["reverse_items"])

    for col in df.columns:
        # Handle both "Q3" and "IRI_Q3" formats
        parts = col.rsplit("_Q", 1)
        if len(parts) == 2:
            try:
                item_num = int(parts[1])
            except ValueError:
                continue
        elif col.startswith("Q"):
            try:
                item_num = int(col[1:])
            except ValueError:
                continue
        else:
            continue

        if item_num in rev_items:
            df[col] = max_val + 1 - df[col]

    return df


def compute_subscale_scores(data: pd.DataFrame, item_labels: List[Tuple[str, str]]) -> pd.DataFrame:
    """
    Compute subscale scores from item-level data.

    Args:
        data: DataFrame with columns like IRI_Q1, IRI_Q2, ..., PANAS_Q1, ...
        item_labels: list of (scale_short, column_name) tuples

    Returns:
        DataFrame with subscale score columns (one per subscale).
    """
    scores = {}

    for scale_short, rules in SCORING_RULES.items():
        max_val = rules["max_val"]
        rev_items = set(rules["reverse_items"])

        # Get columns for this scale
        scale_cols = [col for s, col in item_labels if s == scale_short]
        if not scale_cols:
            continue

        # Build a working copy with reverse coding applied
        work = data[scale_cols].copy()
        for col in scale_cols:
            # Extract item number from column name (e.g., "IRI_Q3" -> 3)
            q_part = col.split("_Q")[-1]
            try:
                item_num = int(q_part)
            except ValueError:
                continue
            if item_num in rev_items:
                work[col] = reverse_code(work[col], max_val)

        # Compute each subscale
        for sub_name, item_nums in rules["subscales"].items():
            sub_cols = [f"{scale_short}_Q{n}" for n in item_nums]
            # Only use columns that exist
            valid_cols = [c for c in sub_cols if c in work.columns]
            if valid_cols:
                scores[f"{scale_short}_{sub_name}"] = work[valid_cols].mean(axis=1)

    return pd.DataFrame(scores)


def compute_subscale_scores_from_csvs(
    data_dirs: List[str],
    scales: List[Tuple[str, str]] = None,
    un_reverse: bool = False,
) -> pd.DataFrame:
    """
    Load CSVs from data directories and compute subscale scores.

    Args:
        data_dirs: list of directory paths containing scale CSVs
        scales: list of (filename, short_name) tuples. If None, uses all 7 scales.
        un_reverse: If True, undo reverse coding (for data/human).

    Returns:
        DataFrame with subscale scores for all subjects across directories.
    """
    import os

    if scales is None:
        scales = list(FILENAME_TO_SCALE.items())

    all_items = []
    item_labels = []

    for file_name, scale_short in scales:
        frames = []
        for d in data_dirs:
            fpath = os.path.join(d, f"{file_name}.csv")
            if not os.path.exists(fpath):
                continue
            df = pd.read_csv(fpath)
            q_cols = [c for c in df.columns if c.startswith("Q")]
            sub = df[q_cols].copy()
            if un_reverse:
                sub = un_reverse_df(sub, file_name)
            frames.append(sub)

        if not frames:
            continue

        combined = pd.concat(frames, ignore_index=True)
        rename = {c: f"{scale_short}_{c}" for c in combined.columns}
        combined.rename(columns=rename, inplace=True)
        all_items.append(combined)
        for c in combined.columns:
            item_labels.append((scale_short, c))

    if not all_items:
        return pd.DataFrame()

    data = pd.concat(all_items, axis=1)
    return compute_subscale_scores(data, item_labels)


# Ordered list of all 16 subscale names for consistent output
ALL_SUBSCALES = [
    "IRI_Perspective_taking",
    "IRI_Fantasy",
    "IRI_Empathic_concern",
    "IRI_Personal_distress",
    "PANAS_Positive_Affect",
    "PANAS_Negative_Affect",
    "POM_Peace_of_Mind",
    "BigFive_Extraversion",
    "BigFive_Agreeableness",
    "BigFive_Conscientiousness",
    "BigFive_Neuroticism",
    "BigFive_Openness",
    "SelfConst_Independent_self",
    "SelfConst_Interdependent_self",
    "LifeSat_Life_Satisfaction",
    "Lonely_Loneliness",
]
