# Psychometric Inference in LLMs

This repository contains the code and public derived metrics for **Tracing
Psychometric Inference in Large Language Models**.

## Quick Start

```bash
conda env create -f environment.yml
conda activate llm_psychometric
export PYTHONPATH=src
export PYTHONDONTWRITEBYTECODE=1

python3 scripts/artifact/verify_release_bundle.py --bundle release_artifacts
```

The verification should end with:

```text
RELEASE_BUNDLE_VERIFY_OK
```

It recomputes the released aggregate checks, including:

- corrected representational slope vs. behavioral slope:
  `r = 0.640`, `p = 0.0137`, `N = 14`;
- representational Mantel `r` vs. behavioral slope:
  `r = 0.678`, `p = 0.0077`, `N = 14`;
- Qwen 14B Instruct best-layer subscale Mantel:
  `r = 0.660`.

## Contents

- `src/psychometric_inference/`: package code for prompts, scoring, model
  metadata, behavioral summaries, and mechanistic analyses.
- `scripts/`: command-line entry points for behavior generation, analysis,
  figures, artifact building, and verification.
- `data/questionnaires/`: questionnaire definitions used by the experiments.
- `release_artifacts/`: compact public derived metrics for no-GPU verification.
- `outputs/` and `reports/figures/`: small generated summaries and figures.

`release_artifacts/` contains four groups:

- `questionnaires/`
- `human_derived_metrics/`
- `model_response_metrics/`
- `activation_geometry_metrics/`

These derived metrics replace raw human rows, raw model generations, and raw
activations for aggregate reproducibility checks.

## Common Commands

Rebuild and verify the public bundle:

```bash
python3 scripts/artifact/build_release_bundle.py --replace
python3 scripts/artifact/verify_release_bundle.py --bundle release_artifacts
```

Refresh cached summary tables and figures:

```bash
python3 scripts/analysis/summarize_statistics.py
python3 scripts/figures/make_main_figures.py
```

Run extension controls:

```bash
scripts/run_extension_controls.sh
scripts/run_extension_controls.sh --dry-run
```

Most scripts expose additional options through `--help`.

## Data Boundary

`.gitignore` keeps local or heavy artifacts out of the public repository:
participant-level data, raw model outputs, model-specific caches, logs, run
markers, raw tensors, model weights, and draft documents.

Small derived statistics, summary tables, and rendered figures are intended to
be versioned when they are useful for checking or documenting aggregate results.

## GPU Runs

The provided environment covers CPU verification and cached analyses. GPU-heavy
generation, activation extraction, prompt-geometry checks, and steering require
a local PyTorch/HuggingFace setup with access to the target models.
