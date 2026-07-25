"""
Item-wise persona injection for psychometric experiments.

Injects a real subject's item-level responses on Scale A into the system prompt,
so the LLM can role-play as that person when filling Scale B.
"""

from dataclasses import dataclass
from typing import List


@dataclass
class ScaleAProfile:
    """A subject's item-level responses on Scale A."""

    subject_id: str
    scale_name: str
    instruction: str
    response_labels: List[str]
    items: List[dict]  # each: {"item_number": int, "text": str, "response": int}


# ── Prompt templates ──

TEMPLATES = {
    "questionnaire": """You are a Taiwanese college student completing psychological questionnaires as a person with the following responses on the {scale_name}:

{instruction}
{response_label_desc}

{item_responses}

Respond to all questions as this person would, maintaining consistency with this psychological profile. Answer with just the number corresponding to your choice.""",
    "questionnaire_no_scale_name": """You are a Taiwanese college student completing psychological questionnaires as a person with the following responses:

{instruction}
{response_label_desc}

{item_responses}

Respond to all questions as this person would, maintaining consistency with this psychological profile. Answer with just the number corresponding to your choice.""",
    "describe": """For the following task, respond as a Taiwanese college student in a way that matches this personality profile.

This Taiwanese college student gave the following responses on the {scale_name}:

{instruction}
{response_label_desc}

{item_responses}

Answer the following questions as this person would. Respond with only a number.""",
    "imagine": """Imagine you are a Taiwanese college student who gave the following responses on the {scale_name}:

{instruction}
{response_label_desc}

{item_responses}

Answer the following questions as this person would. Respond with only a number.""",
    "roleplay": """Take on the role of a Taiwanese college student who gave these responses on the {scale_name}:

{instruction}
{response_label_desc}

{item_responses}

Answer as this person would. Reply with numbers only.""",
}


def build_system_prompt(
    profile: ScaleAProfile,
    template_variant: str = "questionnaire",
) -> str:
    """Build a system prompt from a ScaleAProfile."""

    template = TEMPLATES.get(template_variant)
    if template is None:
        raise ValueError(f"Unknown template variant: {template_variant}. Choose from: {list(TEMPLATES.keys())}")

    # Format response labels
    labels = profile.response_labels
    label_lines = []
    for i, label in enumerate(labels, 1):
        label_lines.append(f"  {i} = {label}")
    response_label_desc = "Response scale:\n" + "\n".join(label_lines)

    # Format item responses
    item_lines = []
    for item in profile.items:
        q_num = item["item_number"]
        text = item["text"]
        resp = item["response"]
        item_lines.append(f"Q{q_num}. {text} → {resp}")
    item_responses = "\n".join(item_lines)

    return template.format(
        scale_name=profile.scale_name,
        instruction=profile.instruction,
        response_label_desc=response_label_desc,
        item_responses=item_responses,
    )


def build_profile_from_csv_row(
    row: dict,
    scale_def: dict,
    items: List[dict],
    subject_id: str,
) -> ScaleAProfile:
    """
    Build a ScaleAProfile from a CSV row and scale definition.

    Args:
        row: dict with keys like Q1, Q2, ... and their values
        scale_def: metadata dict from JSONL first line
        items: list of item dicts from JSONL
        subject_id: identifier for this subject
    """
    item_responses = []
    for item in items:
        q_key = f"Q{item['item_number']}"
        response = row.get(q_key)
        if response is not None:
            item_responses.append({
                "item_number": item["item_number"],
                "text": item["text"],
                "response": int(float(response)),
            })

    return ScaleAProfile(
        subject_id=subject_id,
        scale_name=scale_def.get("scale_name_zh", scale_def["scale_name"]),
        instruction=scale_def["instruction"],
        response_labels=scale_def["response_labels"],
        items=item_responses,
    )
