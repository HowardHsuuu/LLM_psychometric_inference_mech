#!/usr/bin/env python3
"""
Extract psychometric directions from LLM activation space.

For each subscale (16) and scale (7), constructs contrastive prompts
(high vs low) using mega system prompts, generates responses to probe
questions under each condition, extracts mean response activations,
and computes difference-in-means directions.

This is the persona vectors approach adapted to psychometric operationalization.

Usage:
    python -m psychometric_inference.mechanisms.directions                    # Full run
    python -m psychometric_inference.mechanisms.directions --subscales_only   # Skip scale-level
    python -m psychometric_inference.mechanisms.directions --scales_only      # Skip subscale-level
    python -m psychometric_inference.mechanisms.directions --dry_run          # Print prompts, don't run model

Output:
    outputs/mechanistic/results_<model>/directions/
    ├── subscale_directions.npz    # 16 direction vectors
    ├── scale_directions.npz      # 7 direction vectors
    ├── extraction_log.json       # metadata: prompts, layers, etc.
    └── raw_activations/          # per-condition activations for debugging
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
from tqdm import tqdm

from .config import (
    ALL_SUBSCALES, SCALE_NAMES, TARGET_LAYERS, DEFAULT_LAYER,
    N_REPLICATIONS, N_PROBE_QUESTIONS, RESULTS_DIR, MODEL_ID,
)
from .probes import PROBE_QUESTIONS as _ALL_PROBES

# Use only the configured number of probe questions
PROBE_QUESTIONS = _ALL_PROBES[:N_PROBE_QUESTIONS]
from .prompts import (
    load_all_scale_definitions,
    build_mega_system_prompt,
    generate_contrastive_prompts,
    generate_scale_contrastive_prompts,
)
from .activation_model import ActivationModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = RESULTS_DIR / "directions"


def extract_directions_for_targets(
    model: ActivationModel,
    targets: List[str],
    target_type: str,  # "subscale" or "scale"
    scale_defs: dict,
    n_replications: int = N_REPLICATIONS,
    layers: List[int] = None,
    system_prompt_variant: str = "default",
) -> Dict[str, Dict[int, np.ndarray]]:
    """Extract contrastive directions for a list of targets.
    
    For each target:
      1. Generate n_replications paired (high, low) prompts
      2. For each pair, ask all probe questions and collect response activations
      3. Mean across questions and replications → high_mean, low_mean
      4. Direction = high_mean - low_mean (then L2-normalize)
    
    Args:
        model: ActivationModel instance
        targets: List of subscale or scale names
        target_type: "subscale" or "scale"
        scale_defs: From load_all_scale_definitions()
        n_replications: Conditions per high/low pair
        layers: Which layers to extract
        
    Returns:
        Dict mapping target_name -> {layer_idx: direction_vector}
    """
    if layers is None:
        layers = TARGET_LAYERS

    all_directions = {}
    raw_dir = OUTPUT_DIR / "raw_activations"
    raw_dir.mkdir(parents=True, exist_ok=True)

    for target in targets:
        # Resumability: skip if raw activations already exist
        raw_path = raw_dir / f"{target}_raw.npz"
        if raw_path.exists():
            logger.info(f"  {target}: raw activations exist, loading cached directions")
            # Reconstruct directions from saved raw activations
            cached = np.load(raw_path)
            directions = {}
            for l in layers:
                h_key = f"high_L{l}"
                l_key = f"low_L{l}"
                if h_key in cached and l_key in cached:
                    h_mean = np.mean(cached[h_key], axis=0)
                    l_mean = np.mean(cached[l_key], axis=0)
                    direction = h_mean - l_mean
                    norm = np.linalg.norm(direction)
                    if norm > 0:
                        direction = direction / norm
                    directions[l] = direction
            all_directions[target] = directions
            continue

        logger.info(f"\n{'='*60}")
        logger.info(f"Extracting direction: {target} ({target_type})")
        logger.info(f"{'='*60}")

        # Generate contrastive prompts (seed offset by target index for diversity)
        target_seed = 42 + targets.index(target) * 1000
        if target_type == "subscale":
            high_conds, low_conds = generate_contrastive_prompts(
                target, scale_defs, n_replications, seed=target_seed
            )
        else:
            high_conds, low_conds = generate_scale_contrastive_prompts(
                target, scale_defs, n_replications, seed=target_seed
            )

        # Collect activations: shape will be (n_rep * n_questions, hidden_dim) per layer
        high_acts = {l: [] for l in layers}
        low_acts = {l: [] for l in layers}

        total_steps = n_replications * len(PROBE_QUESTIONS) * 2  # high + low
        pbar = tqdm(total=total_steps, desc=target, unit="response")

        for rep_idx in range(n_replications):
            # Build mega prompts for this replication
            high_prompt = build_mega_system_prompt(
                high_conds[rep_idx],
                scale_defs,
                variant=system_prompt_variant,
            )
            low_prompt = build_mega_system_prompt(
                low_conds[rep_idx],
                scale_defs,
                variant=system_prompt_variant,
            )

            for q_idx, question in enumerate(PROBE_QUESTIONS):
                # High condition
                acts_h, _ = model.extract_response_activation(
                    high_prompt, question, layers=layers
                )
                for l in layers:
                    if l in acts_h:
                        high_acts[l].append(acts_h[l])
                pbar.update(1)

                # Low condition
                acts_l, _ = model.extract_response_activation(
                    low_prompt, question, layers=layers
                )
                for l in layers:
                    if l in acts_l:
                        low_acts[l].append(acts_l[l])
                pbar.update(1)

        pbar.close()

        # Compute directions per layer
        directions = {}
        for l in layers:
            if not high_acts[l] or not low_acts[l]:
                continue
            h_mean = np.mean(high_acts[l], axis=0)
            l_mean = np.mean(low_acts[l], axis=0)
            direction = h_mean - l_mean
            # L2 normalize
            norm = np.linalg.norm(direction)
            if norm > 0:
                direction = direction / norm
            directions[l] = direction

        all_directions[target] = directions

        # Save raw activations for this target
        np.savez_compressed(
            raw_dir / f"{target}_raw.npz",
            **{f"high_L{l}": np.array(high_acts[l]) for l in layers if high_acts[l]},
            **{f"low_L{l}": np.array(low_acts[l]) for l in layers if low_acts[l]},
        )
        logger.info(f"  Saved raw activations for {target}")

    return all_directions


def save_directions(
    directions: Dict[str, Dict[int, np.ndarray]],
    filename: str,
    layers: List[int],
):
    """Save directions to npz file, one array per (target, layer) pair."""
    save_dict = {}
    for target, layer_dirs in directions.items():
        for l, vec in layer_dirs.items():
            save_dict[f"{target}_L{l}"] = vec

    # Also save metadata
    save_dict["_targets"] = np.array(list(directions.keys()), dtype=object)
    save_dict["_layers"] = np.array(layers)

    outpath = OUTPUT_DIR / filename
    np.savez_compressed(outpath, **save_dict)
    logger.info(f"Saved directions: {outpath}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subscales_only", action="store_true")
    parser.add_argument("--scales_only", action="store_true")
    parser.add_argument("--n_replications", type=int, default=N_REPLICATIONS)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--model_id", type=str, default=MODEL_ID)
    parser.add_argument("--n_probes", type=int, default=N_PROBE_QUESTIONS)
    parser.add_argument("--prompt_format", choices=["raw", "chat_template"], default="raw")
    parser.add_argument("--system_prompt_variant", choices=["default", "no_scale_name"], default="default")
    args = parser.parse_args()

    if args.n_probes < 1 or args.n_probes > len(_ALL_PROBES):
        raise ValueError(f"--n_probes must be in [1, {len(_ALL_PROBES)}]")

    global PROBE_QUESTIONS
    PROBE_QUESTIONS = _ALL_PROBES[:args.n_probes]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load scale definitions
    logger.info("Loading scale definitions...")
    scale_defs = load_all_scale_definitions()

    if args.dry_run:
        # Just show what would be run
        logger.info("\n=== DRY RUN ===")
        logger.info(f"Model: {args.model_id}")
        logger.info(f"Prompt format: {args.prompt_format}")
        logger.info(f"System prompt variant: {args.system_prompt_variant}")
        logger.info(f"Replications: {args.n_replications}")
        logger.info(f"Probe questions: {len(PROBE_QUESTIONS)}")
        logger.info(f"Layers: {TARGET_LAYERS}")

        if not args.scales_only:
            logger.info(f"\nSubscale directions ({len(ALL_SUBSCALES)}):")
            for s in ALL_SUBSCALES:
                logger.info(f"  {s}")
            n_calls = len(ALL_SUBSCALES) * args.n_replications * len(PROBE_QUESTIONS) * 2
            logger.info(f"  Total forward passes: {n_calls}")

        if not args.subscales_only:
            logger.info(f"\nScale directions ({len(SCALE_NAMES)}):")
            for s in SCALE_NAMES:
                logger.info(f"  {s}")
            n_calls = len(SCALE_NAMES) * args.n_replications * len(PROBE_QUESTIONS) * 2
            logger.info(f"  Total forward passes: {n_calls}")

        return

    # Load model
    logger.info(f"Loading model: {args.model_id}")
    model = ActivationModel(args.model_id, prompt_format=args.prompt_format)

    log_data = {
        "model_id": args.model_id,
        "prompt_format": args.prompt_format,
        "system_prompt_variant": args.system_prompt_variant,
        "n_replications": args.n_replications,
        "n_probes": len(PROBE_QUESTIONS),
        "layers": TARGET_LAYERS,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    # Extract subscale directions
    if not args.scales_only:
        logger.info(f"\n{'#'*60}")
        logger.info(f"# SUBSCALE DIRECTIONS ({len(ALL_SUBSCALES)} targets)")
        logger.info(f"{'#'*60}")

        sub_dirs = extract_directions_for_targets(
            model, ALL_SUBSCALES, "subscale", scale_defs,
            n_replications=args.n_replications,
            system_prompt_variant=args.system_prompt_variant,
        )
        save_directions(sub_dirs, "subscale_directions.npz", TARGET_LAYERS)
        log_data["subscale_targets"] = ALL_SUBSCALES

    # Extract scale directions
    if not args.subscales_only:
        logger.info(f"\n{'#'*60}")
        logger.info(f"# SCALE DIRECTIONS ({len(SCALE_NAMES)} targets)")
        logger.info(f"{'#'*60}")

        scale_dirs = extract_directions_for_targets(
            model, SCALE_NAMES, "scale", scale_defs,
            n_replications=args.n_replications,
            system_prompt_variant=args.system_prompt_variant,
        )
        save_directions(scale_dirs, "scale_directions.npz", TARGET_LAYERS)
        log_data["scale_targets"] = SCALE_NAMES

    # Save log
    with open(OUTPUT_DIR / "extraction_log.json", "w") as f:
        json.dump(log_data, f, indent=2, ensure_ascii=False)

    model.cleanup()
    logger.info("\nDone!")


if __name__ == "__main__":
    main()
