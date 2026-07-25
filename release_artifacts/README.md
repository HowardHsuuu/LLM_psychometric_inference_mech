# Public Derived Metrics

This directory contains the minimal public derived metrics needed for aggregate
reproducibility checks.

Contents:

- `questionnaires/`: questionnaire definitions used to construct prompts and
  score responses.
- `human_derived_metrics/`: aggregate human correlation/structure/reliability
  metrics that replace raw participant rows.
- `model_response_metrics/`: aggregate correlation matrices computed from model
  questionnaire responses; these replace raw per-subject model outputs.
- `activation_geometry_metrics/`: activation-derived geometry/readout metrics
  at the analysis layer; these replace raw hidden-state tensors.

Excluded:

- participant-level human questionnaire responses and subject-level derived
  human score tables, raw LLM responses, raw hidden states, model weights,
  control-experiment dumps, rendered figures, final statistic tables, and broad
  `outputs/` mirrors.

The bundle is meant to support no-GPU checks by recomputing key aggregate
statistics from derived matrices. GPU-heavy generation and activation extraction
scripts remain in the repository, but raw human inputs and raw activations are
not part of this public bundle.

Quick verification from the repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python scripts/artifact/verify_release_bundle.py --bundle release_artifacts
```
