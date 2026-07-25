#!/usr/bin/env python3
"""Run mechanistic interpretability pipeline across models."""
import argparse, gc, importlib, logging, sys, time
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
from psychometric_inference.model_registry import behavioral_map_by_mech_name, mech_run_configs
from psychometric_inference.paths import LLM_BEHAVIOR_DIR, MECHANISTIC_OUTPUT_DIR

MECH_ROOT = MECHANISTIC_OUTPUT_DIR

MODELS = mech_run_configs()
MODEL_LOOKUP = {m["name"]: m for m in MODELS}

BEH_MAP = behavioral_map_by_mech_name()

def results_dir(name):
    return MECH_ROOT / f"results_{name}"

def patch_and_reload(m):
    """Patch config AND reload all analysis modules so their OUTPUT_DIR updates."""
    import psychometric_inference.mechanisms.config as cfg
    cfg.MODEL_ID = m["model_id"]
    cfg.N_LAYERS = m["n_layers"]
    cfg.TARGET_LAYERS = list(m["target_layers"])
    cfg.DEFAULT_LAYER = m["default_layer"]
    cfg.RESULTS_DIR = results_dir(m["name"])
    cfg.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    # Reload mechanism modules so module-level OUTPUT_DIR values pick up RESULTS_DIR.
    for mod_name in list(sys.modules.keys()):
        if mod_name.startswith("psychometric_inference.mechanisms.") and mod_name != "psychometric_inference.mechanisms.config":
            try:
                importlib.reload(sys.modules[mod_name])
            except Exception:
                pass

def phase1_done(m):
    rd = results_dir(m["name"])
    return (rd/"directions"/"subscale_directions.npz").exists() and (rd/"pca"/"subject_activations.npz").exists()

def find_best_layer(m):
    gp = results_dir(m["name"]) / "geometry" / "geometry_results.csv"
    if gp.exists():
        df = pd.read_csv(gp)
        sub = df[df["level"]=="subscale"]
        if not sub.empty:
            return int(sub.loc[sub["mantel_r"].idxmax()]["layer"])
    return m["default_layer"]

def gpu_cleanup():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    except: pass

def safe_run(label, fn):
    try:
        fn()
    except Exception as e:
        logger.error(f"    {label} FAILED: {e}", exc_info=True)

def run_step(label, module_path, argv_list):
    """Reload module, set sys.argv, call main()."""
    def _fn():
        mod = importlib.import_module(module_path)
        mod = importlib.reload(mod)  # ensure OUTPUT_DIR is fresh
        sys.argv = ["x"] + argv_list
        mod.main()
    safe_run(label, _fn)

# ── Phase 1: GPU ──
def run_phase1(m):
    patch_and_reload(m)
    mid = m["model_id"]
    rd = results_dir(m["name"])
    if not (rd/"directions"/"subscale_directions.npz").exists():
        logger.info("  Extracting contrastive directions...")
        run_step("directions", "psychometric_inference.mechanisms.directions", ["--model_id", mid])
        gpu_cleanup()
    else:
        logger.info("  Directions exist, skip")
    if not (rd/"pca"/"subject_activations.npz").exists():
        logger.info("  Extracting subject activations...")
        run_step("subject_activation_extract", "psychometric_inference.mechanisms.subject_activations", ["--model_id", mid])
        gpu_cleanup()
    else:
        logger.info("  Subject activations exist, skip")

# ── Phase 2: No GPU ──
def run_phase2(m):
    patch_and_reload(m)
    mid = m["model_id"]
    rd = results_dir(m["name"])

    logger.info("  [1/8] Geometry (all layers)")
    run_step("geometry", "psychometric_inference.mechanisms.geometry", ["--all_layers"])

    bl = find_best_layer(m)
    logger.info(f"  Best layer: {bl}")

    logger.info("  [2/8] Reliability")
    run_step("reliability", "psychometric_inference.mechanisms.reliability", ["--layer", str(bl)])

    logger.info("  [3/8] Attenuation correction (filtered)")
    run_step("attenuation", "psychometric_inference.mechanisms.attenuation", ["--layer", str(bl)])

    logger.info("  [4/8] Subject activation summaries")
    run_step("subject_activation_summary", "psychometric_inference.mechanisms.subject_activations", ["--analyze_only"])

    logger.info("  [5/8] Robustness summaries")
    run_step("robustness", "psychometric_inference.mechanisms.robustness", ["--layer", str(bl)])

    logger.info("  [6/8] Regression directions (extract only)")
    run_step("regression", "psychometric_inference.mechanisms.steering", ["--extract_only", "--layer", str(bl)])

    logger.info("  [7/8] Partial correlation")
    run_step("partial", "psychometric_inference.mechanisms.causality", ["--partial_only", "--layer", str(bl)])

    logger.info("  [8/8] Controls (semantic + permutation)")
    run_step("controls", "psychometric_inference.mechanisms.controls", ["--layer", str(bl), "--model_id", mid, "--n_perms", "100"])

    # Amplification locate if behavioral data exists
    bn = BEH_MAP.get(m["name"])
    if bn:
        lr = LLM_BEHAVIOR_DIR / bn
        if lr.exists():
            logger.info(f"  [+] Amplification locate (behavioral: {bn})")
            run_step("amplification", "psychometric_inference.mechanisms.amplification", ["--llm_root", str(lr), "--layer", str(bl)])
            logger.info("  [+] Ridge inflation permutation")
            run_step("ridge_perm", "psychometric_inference.mechanisms.ridge_baseline", ["--layer", str(bl), "--n_perms", "100"])
        else:
            logger.info(f"  [skip] No behavioral data at {lr}")

    return bl

# ── Phase 3: GPU (optional) ──
def run_phase3(m, bl):
    patch_and_reload(m)
    mid = m["model_id"]
    logger.info(f"  Steering (layer {bl})")
    run_step("steering", "psychometric_inference.mechanisms.steering", ["--steering", "--layer", str(bl), "--model_id", mid])
    gpu_cleanup()

# ── Summary ──
def collect_summary(models):
    import numpy as np
    rows = []
    for m in models:
        rd = results_dir(m["name"])
        row = dict(model=m["name"], model_id=m["model_id"], family=m["family"],
                   mtype=m["mtype"], n_layers=m["n_layers"])
        # Geometry
        gp = rd / "geometry" / "geometry_results.csv"
        if gp.exists():
            df = pd.read_csv(gp)
            sub = df[df["level"] == "subscale"]
            if not sub.empty:
                best = sub.loc[sub["mantel_r"].idxmax()]
                row.update(best_layer=int(best["layer"]),
                          geom_mantel_r=round(best["mantel_r"],4),
                          geom_slope=round(best["slope"],4))
        # Reliability
        bl = row.get("best_layer", m["default_layer"])
        rp = rd / "reliability" / f"reliability_L{bl}.csv"
        if rp.exists():
            rdf = pd.read_csv(rp)
            sr = rdf[rdf["type"] == "subscale"]
            if not sr.empty:
                row.update(mean_reliability=round(sr["spearman_brown_r"].mean(),3),
                          n_reliable=int((sr["spearman_brown_r"] >= 0.3).sum()))
        # Regression direction geometry
        rg = rd / "regression_directions" / f"regression_cosine_sim_L{bl}.csv"
        if rg.exists():
            patch_and_reload(m)
            from psychometric_inference.mechanisms.geometry import compute_human_correlation_matrix, mantel_test
            rc = pd.read_csv(rg, index_col=0)
            hc = compute_human_correlation_matrix("subscale")
            common = [s for s in rc.index if s in hc.index]
            if len(common) >= 4:
                n = len(common)
                idx = np.triu_indices(n, k=1)
                cv = rc.loc[common,common].values[idx]
                hv = hc.loc[common,common].values[idx]
                v = ~(np.isnan(cv) | np.isnan(hv))
                if v.sum() >= 3:
                    rm, _ = mantel_test(rc.loc[common,common].values, hc.loc[common,common].values)
                    sl = np.polyfit(hv[v], cv[v], 1)[0]
                    row.update(reg_mantel_r=round(rm,4), reg_slope=round(sl,4))
        # Partial correlation
        cp = rd / "causal" / f"partial_matrix_L{bl}.csv"
        if cp.exists():
            row["has_partial"] = True
        # Steering
        sp = rd / "regression_directions" / "regression_steering_raw.csv"
        if sp.exists():
            row["has_steering"] = True
        # Amplification
        ap = rd / "amplification_locate" / f"comparison_results_L{bl}.json"
        if ap.exists():
            row["has_amplification"] = True
        rows.append(row)

    summary = pd.DataFrame(rows)
    out = MECH_ROOT / "scaling_summary.csv"
    summary.to_csv(out, index=False)
    print(f"\n{'='*70}\n  CROSS-MODEL SUMMARY\n{'='*70}")
    cols_to_show = [c for c in ["model","mtype","n_layers","best_layer","geom_mantel_r",
                                 "geom_slope","mean_reliability","n_reliable",
                                 "reg_mantel_r","reg_slope"] if c in summary.columns]
    print(summary[cols_to_show].to_string(index=False))
    print(f"\nSaved to {out}")
    return summary

def main():
    p = argparse.ArgumentParser(description="Run mech interp across models")
    p.add_argument("--models", nargs="+", default=None, help="Model short names")
    p.add_argument("--skip_existing", action="store_true", help="Skip Phase 1 if done")
    p.add_argument("--analysis_only", action="store_true", help="Only Phase 2+3, skip extraction")
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--summary_only", action="store_true")
    args = p.parse_args()

    if args.models:
        models = [MODEL_LOOKUP[n] for n in args.models if n in MODEL_LOOKUP]
        if not models:
            print(f"No valid models. Available: {list(MODEL_LOOKUP.keys())}")
            return
    else:
        models = MODELS

    print(f"{'='*70}")
    print(f"  MECHANISTIC INTERP SCALING — {len(models)} models")
    print(f"{'='*70}")
    for m in models:
        status = " [Phase1 done]" if phase1_done(m) else ""
        beh = BEH_MAP.get(m["name"], "none")
        print(f"  {m['model_id']:<45} → {m['name']}{status}  (beh: {beh})")
    print()

    if args.dry_run or args.summary_only:
        if args.summary_only:
            collect_summary(models)
        return

    summary_rows = []
    for m in models:
        print(f"\n{'#'*70}")
        print(f"  {m['model_id']} ({m['name']})")
        print(f"{'#'*70}")
        t0 = time.time()
        try:
            if not args.analysis_only:
                if args.skip_existing and phase1_done(m):
                    logger.info("  Phase 1 complete, skipping")
                else:
                    run_phase1(m)
            bl = run_phase2(m)
            run_phase3(m, bl)
            elapsed = (time.time() - t0) / 60
            summary_rows.append(dict(model=m["name"], status="success", minutes=round(elapsed, 1)))
        except Exception as e:
            logger.error(f"  MODEL FAILED: {e}", exc_info=True)
            elapsed = (time.time() - t0) / 60
            summary_rows.append(dict(model=m["name"], status=f"failed", minutes=round(elapsed, 1)))
            gpu_cleanup()

    print(f"\n{'='*70}\n  RUN SUMMARY\n{'='*70}")
    for r in summary_rows:
        print(f"  {r['model']:<25} {r['status']:<15} ({r['minutes']} min)")

    collect_summary(models)

if __name__ == "__main__":
    main()
