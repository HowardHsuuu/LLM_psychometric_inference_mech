"""Configuration for mechanistic interpretability experiments."""

# ── Paths ──
from psychometric_inference.paths import (
    HUMAN_DIRS,
    MECHANISTIC_DEFAULT_RESULTS_DIR,
    MECHANISTIC_OUTPUT_DIR,
    PROJECT_ROOT,
)

MECH_ROOT = MECHANISTIC_OUTPUT_DIR
RESULTS_DIR = MECHANISTIC_DEFAULT_RESULTS_DIR
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Model ──
MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
DEVICE = "cuda"
DTYPE = "float16"
N_LAYERS = 32  # Llama 3.1 8B has 32 transformer layers

# Target layers for extraction (middle-to-upper, where persona vectors literature
# finds strongest signal — ~50-75% of depth)
TARGET_LAYERS = [12, 14, 16, 18, 20, 22, 24]  # will sweep; persona vectors found ~layer 16 for 8B
DEFAULT_LAYER = 16  # ~50% depth, consistent with CAA and persona vectors findings

# ── Generation ──
MAX_NEW_TOKENS = 128  # for probe question responses
TEMPERATURE = 0.7     # nonzero to get diverse responses across replications

# ── Scale definitions ──
SCALES = [
    ("IRI", "IRI"),
    ("PANAS", "PANAS"),
    ("POM", "POM"),
    ("big_five", "BigFive"),
    ("in_inter_dependent", "SelfConst"),
    ("Life_Satisfaction", "LifeSat"),
    ("Loneliness", "Lonely"),
]

SCALE_NAMES = ["IRI", "PANAS", "POM", "BigFive", "SelfConst", "LifeSat", "Lonely"]

# 16 subscales in canonical order (matching scoring.py)
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

# Map subscale -> parent scale
SUB_TO_SCALE = {
    "IRI_Perspective_taking": "IRI",
    "IRI_Fantasy": "IRI",
    "IRI_Empathic_concern": "IRI",
    "IRI_Personal_distress": "IRI",
    "PANAS_Positive_Affect": "PANAS",
    "PANAS_Negative_Affect": "PANAS",
    "POM_Peace_of_Mind": "POM",
    "BigFive_Extraversion": "BigFive",
    "BigFive_Agreeableness": "BigFive",
    "BigFive_Conscientiousness": "BigFive",
    "BigFive_Neuroticism": "BigFive",
    "BigFive_Openness": "BigFive",
    "SelfConst_Independent_self": "SelfConst",
    "SelfConst_Interdependent_self": "SelfConst",
    "LifeSat_Life_Satisfaction": "LifeSat",
    "Lonely_Loneliness": "Lonely",
}

# ── Contrastive extraction parameters ──
# With mega prompts (~2000 tokens) + generation (128 tokens), each forward pass
# is ~3-5s on 8B. Total calls = n_targets × n_rep × n_questions × 2 (high+low).
# 23 targets × 10 × 20 × 2 = 9,200 calls ≈ 8-13 hours. Manageable on 5090.
N_REPLICATIONS = 10      # random background prompts per high/low pair
N_PROBE_QUESTIONS = 20   # questions to ask under each condition (first 20 of 40)
