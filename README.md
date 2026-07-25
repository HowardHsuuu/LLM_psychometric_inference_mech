# Psychometric Inference in LLMs

Replication artifact for **Tracing Psychometric Inference in Large Language
Models**.

This repository is the curated study artifact, not the full parent behavioral
study project. The replication material is organized around
`release_artifacts/`, which contains a compact public derived-metric bundle for
verifying the main aggregate statistics without raw human rows or GPU reruns.

## Repository Layout

- `src/psychometric_inference/`: importable package for scoring, prompt
  construction, model metadata, and reusable analysis code.
- `src/psychometric_inference/mechanisms/`: mechanistic library modules for
  activation extraction, direction geometry, reliability, steering, controls,
  and cross-model summaries.
- `scripts/behavior/`: behavioral generation and local cached-analysis entry
  points.
- `scripts/analysis/`: uncertainty, attenuation, human-structure, and summary
  analyses.
- `scripts/artifact/`: build and verify the public derived-metric bundle.
- `scripts/figures/`: figure generation from cached outputs.
- `data/questionnaires/`: questionnaire JSONL definitions used by the experiments.
- `release_artifacts/`: public derived metrics for no-GPU verification.

The following local/raw or heavy generated paths are ignored by git:
`data/human/`, `data/llm_behavior/`, `data/llm_behavior_culture/`,
`data/llm_behavior_prompt_variants/`, model-specific generation caches under
`outputs/behavior/`, most `outputs/behavior_culture/` caches,
`outputs/mechanistic/results_*/`, large/cache files under
`outputs/mechanistic/factorial_geometry/`, run markers, `outputs/logs/`,
`reports/docs/`, and `reports/manuscript/`.

Small final statistics, summary tables, and rendered figures under `outputs/`
and `reports/figures/` are trackable, including behavior/factorial summaries,
prompt-variant summaries, factorial-geometry summary/readout CSVs, and the
culture comparison CSV.

## Release Policy

Included in `release_artifacts/`:

- `questionnaires/`: questionnaire definitions used to construct prompts and
  score responses;
- `human_derived_metrics/`: aggregate human correlation/structure/reliability
  metrics that replace raw participant rows;
- `model_response_metrics/`: aggregate correlation matrices computed from model
  questionnaire responses, replacing raw per-subject model outputs;
- `activation_geometry_metrics/`: activation-derived geometry/readout metrics
  at the analysis layer, replacing raw hidden-state tensors.

Excluded from the public artifact:

- participant-level human questionnaire responses;
- subject-level derived human score tables;
- raw LLM responses;
- raw hidden-state tensors;
- model weights;
- control-experiment dumps;
- final statistics tables;
- rendered figures;
- broad `outputs/` mirrors, including synthetic/robustness result dumps that can
  be regenerated locally.

## Setup

```bash
conda env create -f environment.yml
conda activate llm_psychometric
```

For headless plotting:

```bash
mkdir -p /tmp/psychometric_mpl_cache /tmp/psychometric_xdg_cache/fontconfig
export MPLBACKEND=Agg
export MPLCONFIGDIR=/tmp/psychometric_mpl_cache
export XDG_CACHE_HOME=/tmp/psychometric_xdg_cache
```

## Artifact Verification

Run this from the repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python scripts/artifact/verify_release_bundle.py --bundle release_artifacts
```

Expected checks include:

- representation corrected slope vs. behavioral subscale slope:
  `r = 0.640`, `p = 0.0137`, `N = 14`;
- representation Mantel `r` vs. behavioral subscale slope:
  `r = 0.678`, `p = 0.0077`, `N = 14`;
- Qwen 14B Instruct best subscale geometry Mantel:
  `r = 0.660`.

This verification uses only `release_artifacts/`; it does not read raw human
responses, subject-level human score tables, model weights, or hidden-state
tensors.

## Maintainer Workflows

These commands require the local staging outputs that are intentionally ignored
by git.

Rebuild the public artifact from local cached outputs:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python scripts/artifact/build_release_bundle.py --replace
```

Run CPU-level cached analyses:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python -m psychometric_inference.mechanisms.pipeline --dry_run
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python scripts/analysis/compute_robustness_suite.py --list
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python -m psychometric_inference.mechanisms.representation_behavior
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python scripts/artifact/verify_synthetic_controls.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python scripts/figures/plot_figures.py
```

Behavior cached-analysis reruns:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python scripts/behavior/generate_base_behavior.py --analysis_only
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python scripts/behavior/generate_instruct_behavior.py --analysis_only
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python scripts/behavior/generate_culture_behavior.py --comparison_only
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python scripts/behavior/generate_random_factorial.py --analysis_only
```

Profile-level controls:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python scripts/behavior/generate_random_factorial.py --mode factorial --skip_existing
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python scripts/analysis/evaluate_factorial.py
```

Run all pending extension/control experiments with logs:

```bash
scripts/run_extension_controls.sh
```

By default this runs `qwen14b_instruct` to completion, rebuilds/verifies the
release artifact, and then runs `llama8b_instruct` to completion.

Factorial profile geometry controls:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python -m psychometric_inference.mechanisms.factorial_geometry --model qwen14b_instruct --source_scale big_five --mode factorial --target_items_per_subscale 1 --layers 24
```

Prompt/readout sensitivity controls:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python scripts/behavior/run_prompt_sensitivity.py --models qwen14b_instruct --variants no_scale_name chat_template --readouts argmax expected_value
```

Prompt-geometry sensitivity controls:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python -m psychometric_inference.mechanisms.prompt_geometry --models qwen14b_instruct --variants no_scale_name chat_template --n_replications 3 --n_probes 20
```

Steering variants:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python -m psychometric_inference.mechanisms.steering --direction_source regression --extract_only --layer 16
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python -m psychometric_inference.mechanisms.steering --direction_source regression --steering --layer 16 --alphas -2 -1 -0.5 0 0.5 1 2 --target_items_per_subscale 2 --intervention last_token_add
```

Cached statistic inventory:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python scripts/analysis/summarize_statistics.py
```

## Heavy Reruns

GPU-heavy activation extraction starts from:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python -m psychometric_inference.mechanisms.pipeline --models qwen14b_instruct
```

Cached mechanistic analysis can skip extraction:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python -m psychometric_inference.mechanisms.pipeline --models qwen14b_instruct --analysis_only
```

Behavioral generation entry points live under `scripts/behavior/` and require
the protected local input tables described above.
