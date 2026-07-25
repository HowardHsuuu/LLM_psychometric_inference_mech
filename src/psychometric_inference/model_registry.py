"""Shared model registry for the replication artifact.

The same 14 primary models appear in behavioral generation, cached analysis,
mechanistic aggregation, bootstrap CIs, semantic controls, and figure scripts.
Keeping their names in one place prevents drift between behavior run suffixes
(`*_v2` / `*_v3`) and canonical mechanistic result directories.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    hf_id: str
    behavior_dir: str
    mech_name: str
    size_b: float
    family: str
    is_instruct: bool
    n_layers: int
    target_layers: tuple[int, ...]
    default_layer: int

    @property
    def mtype(self) -> str:
        return "instruct" if self.is_instruct else "base"

    @property
    def family_lower(self) -> str:
        return self.family.lower()

    @property
    def bootstrap_label(self) -> str:
        size = f"{int(self.size_b)}B" if self.size_b == int(self.size_b) else f"{self.size_b:.1f}B"
        fam = "Q" if self.family == "Qwen" else "L"
        typ = "inst" if self.is_instruct else "base"
        return f"{size} {typ} {fam}"

    @property
    def size_label(self) -> str:
        return f"{int(self.size_b)}B" if self.size_b == int(self.size_b) else f"{self.size_b:.1f}B"

    @property
    def culture_dir(self) -> str:
        return f"{self.mech_name}_culture"


PRIMARY_MODELS: tuple[ModelSpec, ...] = (
    ModelSpec(
        "Qwen/Qwen2.5-0.5B", "qwen05b_base_v2", "qwen05b_base",
        0.5, "Qwen", False, 24, (6, 8, 10, 12, 14, 16), 12,
    ),
    ModelSpec(
        "Qwen/Qwen2.5-0.5B-Instruct", "qwen05b_instruct_v3", "qwen05b_instruct",
        0.5, "Qwen", True, 24, (6, 8, 10, 12, 14, 16), 12,
    ),
    ModelSpec(
        "meta-llama/Llama-3.2-1B", "llama1b_base_v2", "llama1b_base",
        1.0, "Llama", False, 16, (4, 6, 8, 10, 12, 14), 8,
    ),
    ModelSpec(
        "meta-llama/Llama-3.2-1B-Instruct", "llama1b_instruct_v3", "llama1b_instruct",
        1.0, "Llama", True, 16, (4, 6, 8, 10, 12, 14), 8,
    ),
    ModelSpec(
        "Qwen/Qwen2.5-3B", "qwen3b_base_v2", "qwen3b_base",
        3.0, "Qwen", False, 36, (10, 14, 18, 22, 26), 18,
    ),
    ModelSpec(
        "Qwen/Qwen2.5-3B-Instruct", "qwen3b_instruct_v3", "qwen3b_instruct",
        3.0, "Qwen", True, 36, (10, 14, 18, 22, 26), 18,
    ),
    ModelSpec(
        "meta-llama/Llama-3.2-3B", "llama3b_base_v2", "llama3b_base",
        3.0, "Llama", False, 28, (8, 10, 12, 14, 16, 18, 20), 14,
    ),
    ModelSpec(
        "meta-llama/Llama-3.2-3B-Instruct", "llama3b_instruct_v3", "llama3b_instruct",
        3.0, "Llama", True, 28, (8, 10, 12, 14, 16, 18, 20), 14,
    ),
    ModelSpec(
        "Qwen/Qwen2.5-7B", "qwen7b_base_v2", "qwen7b_base",
        7.0, "Qwen", False, 28, (8, 10, 12, 14, 16, 18, 20), 14,
    ),
    ModelSpec(
        "Qwen/Qwen2.5-7B-Instruct", "qwen7b_instruct_v3", "qwen7b_instruct",
        7.0, "Qwen", True, 28, (8, 10, 12, 14, 16, 18, 20), 14,
    ),
    ModelSpec(
        "meta-llama/Llama-3.1-8B", "llama8b_base_v2", "llama8b_base",
        8.0, "Llama", False, 32, (12, 14, 16, 18, 20, 22, 24), 16,
    ),
    ModelSpec(
        "meta-llama/Llama-3.1-8B-Instruct", "llama8b_instruct_v3", "llama8b_instruct",
        8.0, "Llama", True, 32, (12, 14, 16, 18, 20, 22, 24), 16,
    ),
    ModelSpec(
        "Qwen/Qwen2.5-14B", "qwen14b_base_v2", "qwen14b_base",
        14.0, "Qwen", False, 48, (16, 20, 24, 28, 32, 36), 24,
    ),
    ModelSpec(
        "Qwen/Qwen2.5-14B-Instruct", "qwen14b_instruct_v3", "qwen14b_instruct",
        14.0, "Qwen", True, 48, (16, 20, 24, 28, 32, 36), 24,
    ),
)

MODEL_BY_MECH_NAME = {m.mech_name: m for m in PRIMARY_MODELS}

MECH_RUN_ORDER = (
    "llama1b_instruct", "llama1b_base",
    "llama3b_instruct", "llama3b_base",
    "llama8b_instruct", "llama8b_base",
    "qwen05b_instruct", "qwen05b_base",
    "qwen3b_instruct", "qwen3b_base",
    "qwen7b_instruct", "qwen7b_base",
    "qwen14b_instruct", "qwen14b_base",
)


FAMILY_PLOT_META = {
    "Qwen": {
        "label": "Qwen 2.5",
        "sizes": [0.5, 3.0, 7.0, 14.0],
        "color_base": "#085041",
        "color_inst": "#5DCAA5",
    },
    "Llama": {
        "label": "Llama 3",
        "sizes": [1.0, 3.0, 8.0],
        "color_base": "#993C1D",
        "color_inst": "#F0997B",
    },
}

SIZE_TO_LABEL = {
    "Qwen": {0.5: "0.5", 3.0: "3", 7.0: "7", 14.0: "14"},
    "Llama": {1.0: "1", 3.0: "3", 8.0: "8"},
}


def base_generation_tuples() -> list[tuple[str, str, bool]]:
    return [(m.hf_id, m.behavior_dir, m.is_instruct) for m in PRIMARY_MODELS if not m.is_instruct]


def instruct_generation_tuples() -> list[tuple[str, str, bool]]:
    return [(m.hf_id, m.behavior_dir, m.is_instruct) for m in PRIMARY_MODELS if m.is_instruct]


def analysis_model_tuples() -> list[tuple[str, float, str, str]]:
    return [(m.behavior_dir, m.size_b, m.mtype, m.family) for m in PRIMARY_MODELS]


def bootstrap_model_tuples() -> list[tuple[str, str]]:
    return [(m.behavior_dir, m.bootstrap_label) for m in PRIMARY_MODELS]


def culture_model_tuples() -> list[tuple[str, str, bool, str, float, str]]:
    return [(m.hf_id, m.culture_dir, m.is_instruct, m.behavior_dir, m.size_b, m.family) for m in PRIMARY_MODELS]


def semantic_control_tuples() -> list[tuple[str, str, str, float, str, bool]]:
    return [(m.hf_id, m.behavior_dir, m.mech_name, m.size_b, m.family, m.is_instruct) for m in PRIMARY_MODELS]


def mech_model_tuples(family_case: str = "lower") -> list[tuple[str, float, str, str]]:
    rows = []
    for m in PRIMARY_MODELS:
        family = m.family_lower if family_case == "lower" else m.family
        rows.append((m.mech_name, m.size_b, family, m.mtype))
    return rows


def mech_run_configs() -> list[dict]:
    rows = []
    for name in MECH_RUN_ORDER:
        m = MODEL_BY_MECH_NAME[name]
        rows.append({
            "model_id": m.hf_id,
            "name": m.mech_name,
            "n_layers": m.n_layers,
            "target_layers": list(m.target_layers),
            "default_layer": m.default_layer,
            "family": m.family_lower,
            "mtype": m.mtype,
        })
    return rows


def model_config_by_hf_id() -> dict[str, tuple[str, int, list[int], int]]:
    return {
        m.hf_id: (m.mech_name, m.n_layers, list(m.target_layers), m.default_layer)
        for m in PRIMARY_MODELS
    }


def qwen_base_control_tuples(sizes: tuple[float, ...] = (0.5, 3.0, 14.0)) -> list[tuple[str, str, bool]]:
    return [
        (m.hf_id, m.mech_name, m.is_instruct)
        for m in PRIMARY_MODELS
        if m.family == "Qwen" and not m.is_instruct and m.size_b in sizes
    ]


def qwen_base_experiment_tuples(suffix: str, sizes: tuple[float, ...] = (0.5, 3.0, 14.0)) -> list[tuple[str, str]]:
    return [
        (f"{m.mech_name}_{suffix}", m.size_label)
        for m in PRIMARY_MODELS
        if m.family == "Qwen" and not m.is_instruct and m.size_b in sizes
    ]


def behavioral_map_by_mech_name() -> dict[str, str]:
    return {m.mech_name: m.behavior_dir for m in PRIMARY_MODELS}


def behavioral_figure_map() -> list[tuple[str, float, str, str, str]]:
    return [(m.mech_name, m.size_b, m.family, m.mtype, m.behavior_dir) for m in PRIMARY_MODELS]
