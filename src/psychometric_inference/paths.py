"""Central filesystem layout for the replication artifact."""

from __future__ import annotations

from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parent
SRC_DIR = PACKAGE_DIR.parent
PROJECT_ROOT = SRC_DIR.parent

DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
REPORTS_DIR = PROJECT_ROOT / "reports"

QUESTIONNAIRE_DIR = DATA_DIR / "questionnaires"
HUMAN_DATA_DIR = DATA_DIR / "human"
LLM_BEHAVIOR_DIR = DATA_DIR / "llm_behavior"
LLM_BEHAVIOR_CULTURE_DIR = DATA_DIR / "llm_behavior_culture"
LLM_BEHAVIOR_PROMPT_VARIANTS_DIR = DATA_DIR / "llm_behavior_prompt_variants"

BEHAVIOR_OUTPUT_DIR = OUTPUTS_DIR / "behavior"
BEHAVIOR_CULTURE_OUTPUT_DIR = OUTPUTS_DIR / "behavior_culture"
PROMPT_VARIANT_OUTPUT_DIR = OUTPUTS_DIR / "prompt_variants"
STATISTICS_OUTPUT_DIR = OUTPUTS_DIR / "statistics"
SEMANTIC_CONTROL_OUTPUT_DIR = OUTPUTS_DIR / "semantic_controls"
SUPPLEMENTARY_OUTPUT_DIR = OUTPUTS_DIR / "supplementary"
ROBUSTNESS_OUTPUT_DIR = OUTPUTS_DIR / "robustness"
HUMAN_STRUCTURE_OUTPUT_DIR = OUTPUTS_DIR / "human_structure"
MECHANISTIC_OUTPUT_DIR = OUTPUTS_DIR / "mechanistic"
MECHANISTIC_DEFAULT_RESULTS_DIR = MECHANISTIC_OUTPUT_DIR / "results_default"

FIGURE_DIR = REPORTS_DIR / "figures"
MANUSCRIPT_DIR = REPORTS_DIR / "manuscript"
DOCS_DIR = REPORTS_DIR / "docs"

HUMAN_DATASETS = ("SED", "SEDC", "SEDD")
HUMAN_DIRS = [HUMAN_DATA_DIR / ds for ds in HUMAN_DATASETS]


def questionnaire_path(scale_file: str) -> Path:
    return QUESTIONNAIRE_DIR / f"{scale_file}.jsonl"
