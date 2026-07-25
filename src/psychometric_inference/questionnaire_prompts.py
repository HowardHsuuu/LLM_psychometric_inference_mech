"""Prompt builders for questionnaire item readout."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def format_item_prompt(item: dict[str, Any], instruction: str) -> str:
    """Format one questionnaire item for numeric-response readout."""
    options: Sequence[Any] = item["response_options"]
    labels = item.get("response_labels", [])
    if labels:
        option_text = "".join(
            f"  {option} = {label}\n" for option, label in zip(options, labels)
        )
    else:
        option_text = f"  Options: {list(options)}\n"

    return (
        f"{instruction}\n\n"
        f"Statement: {item['text']}\n\n"
        f"Response options:\n{option_text}\n"
        f"Please respond with a single number from {options[0]} to {options[-1]}."
    )

