#!/usr/bin/env python3
"""
Bootstrap confidence intervals by resampling SUBJECTS (not pairs).
All levels: item (all/within/between), subscale, scale.
Skips model+level combos already in output CSV.

Usage:
    python -W ignore compute_bootstrap_intervals.py
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR / "src"))

from psychometric_inference.scoring import SCORING_RULES, ALL_SUBSCALES, compute_subscale_scores
from psychometric_inference.model_registry import bootstrap_model_tuples

N_BOOT = 2000
CI_LEVEL = 0.95

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

HUMAN_DIRS = [
    "data/human/SED",
    "data/human/SEDC",
    "data/human/SEDD",
]

MODELS = bootstrap_model_tuples()

ALL_LEVELS = ["item", "item_within", "item_between", "subscale", "scale"]
OUT_PATH = BASE_DIR / "outputs" / "behavior" / "bootstrap_ci_subjects.csv"


def load_existing():
    if OUT_PATH.exists():
        df = pd.read_csv(OUT_PATH)
        done = set(zip(df["dirname"], df["level"]))
        return df, done
    return pd.DataFrame(), set()


def save_results(existing_df, new_rows):
    if new_rows:
        new_df = pd.DataFrame(new_rows)
        combined = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        combined = existing_df
    combined.to_csv(OUT_PATH, index=False)


def load_human_scale_data():
    from psychometric_inference.scoring import un_reverse_df
    human = {}
    for scale_file, scale_short in SCALES:
        frames = []
        for d in HUMAN_DIRS:
            fpath = BASE_DIR / d / f"{scale_file}.csv"
            if not fpath.exists():
                continue
            df = pd.read_csv(fpath)
            id_col = None
            for c in ["Subject_ID", "ID", "Scan_ID", "id"]:
                if c in df.columns:
                    id_col = c
                    break
            q_cols = [c for c in df.columns if c.startswith("Q")]
            sub = df[q_cols].copy()
            sub = un_reverse_df(sub, scale_file)
            if id_col:
                sub.insert(0, "Subject_ID", df[id_col].astype(str))
            else:
                sub.insert(0, "Subject_ID", [f"S{i:04d}" for i in range(len(sub))])
            frames.append(sub)
        if frames:
            human[scale_file] = pd.concat(frames, ignore_index=True)
    return human


def load_llm_rotation_data(llm_root):
    rotations = {}
    for scale_file, scale_short in SCALES:
        persona_dir = BASE_DIR / llm_root / f"persona_{scale_file}"
        if not persona_dir.exists():
            continue
        filled = {}
        for fill_file, fill_short in SCALES:
            fpath = persona_dir / f"{fill_file}.csv"
            if fpath.exists():
                filled[fill_file] = pd.read_csv(fpath)
        persona_path = persona_dir / f"{scale_file}_persona.csv"
        if persona_path.exists():
            filled[f"{scale_file}_persona"] = pd.read_csv(persona_path)
        rotations[scale_file] = filled
    return rotations


def _corr(a, b):
    valid = ~(np.isnan(a) | np.isnan(b))
    if valid.sum() < 3:
        return np.nan
    return np.corrcoef(a[valid], b[valid])[0, 1]


def compare_upper_tri(mat_a, mat_b, n, pair_mask=None):
    """Compare upper triangle. If pair_mask given, only use those pairs."""
    mask_r, mask_c = np.triu_indices(n, k=1)
    a_vec = mat_a[(mask_r, mask_c)]
    b_vec = mat_b[(mask_r, mask_c)]

    if pair_mask is not None:
        a_vec = a_vec[pair_mask]
        b_vec = b_vec[pair_mask]

    valid = ~(np.isnan(a_vec) | np.isnan(b_vec))
    if valid.sum() < 3:
        return np.nan
    return np.corrcoef(a_vec[valid], b_vec[valid])[0, 1]


# ── Item-level ──

def build_item_col_list(human_data):
    item_cols = []
    item_scales = []
    for scale_file, scale_short in SCALES:
        if scale_file not in human_data:
            continue
        q_cols = [c for c in human_data[scale_file].columns if c.startswith("Q")]
        for q in q_cols:
            item_cols.append(f"{scale_short}_{q}")
            item_scales.append(scale_short)
    return item_cols, item_scales


def build_within_between_masks(item_scales):
    """Build boolean masks for within-scale and between-scale pairs."""
    n = len(item_scales)
    mask_r, mask_c = np.triu_indices(n, k=1)
    within = np.array([item_scales[r] == item_scales[c] for r, c in zip(mask_r, mask_c)])
    between = ~within
    return within, between


def compute_human_item_matrix(human_data, subject_ids, item_cols):
    frames = []
    for scale_file, scale_short in SCALES:
        if scale_file not in human_data:
            continue
        df = human_data[scale_file]
        mask = df["Subject_ID"].astype(str).isin(subject_ids)
        q_cols = [c for c in df.columns if c.startswith("Q")]
        sub = df.loc[mask, q_cols].copy()
        sub.rename(columns={q: f"{scale_short}_{q}" for q in q_cols}, inplace=True)
        sub.index = range(len(sub))
        frames.append(sub)
    merged = pd.concat(frames, axis=1)
    valid_cols = [c for c in item_cols if c in merged.columns]
    return merged[valid_cols].corr().values, valid_cols


def compute_implicit_item_matrix(llm_rotations, subject_ids, item_cols):
    n = len(item_cols)
    col_to_idx = {c: i for i, c in enumerate(item_cols)}
    corr_sum = np.zeros((n, n))
    corr_count = np.zeros((n, n))

    for persona_file, persona_short in SCALES:
        if persona_file not in llm_rotations:
            continue
        rotation = llm_rotations[persona_file]
        persona_key = f"{persona_file}_persona"
        if persona_key not in rotation:
            continue

        persona_df = rotation[persona_key]
        mask = persona_df["Subject_ID"].astype(str).isin(subject_ids)
        if mask.sum() < 3:
            continue

        p_q = [c for c in persona_df.columns if c.startswith("Q")]
        p_data = persona_df.loc[mask, p_q].values.astype(float)
        p_col_names = [f"{persona_short}_{q}" for q in p_q]

        for fill_file, fill_short in SCALES:
            if fill_file not in rotation:
                continue
            fill_df = rotation[fill_file]
            fill_mask = fill_df["Subject_ID"].astype(str).isin(subject_ids)
            if fill_mask.sum() != mask.sum():
                continue

            l_q = [c for c in fill_df.columns if c.startswith("Q")]
            l_data = fill_df.loc[fill_mask, l_q].values.astype(float)
            l_col_names = [f"{fill_short}_{q}" for q in l_q]

            for pi, pc in enumerate(p_col_names):
                if pc not in col_to_idx:
                    continue
                p_idx = col_to_idx[pc]
                p_vec = p_data[:, pi]

                for li, lc in enumerate(l_col_names):
                    if lc not in col_to_idx:
                        continue
                    l_idx = col_to_idx[lc]
                    l_vec = l_data[:, li]

                    r = _corr(p_vec, l_vec)
                    if not np.isnan(r):
                        corr_sum[p_idx, l_idx] += r
                        corr_count[p_idx, l_idx] += 1

    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(corr_count > 0, corr_sum / corr_count, np.nan)


# ── Subscale-level ──

def compute_implicit_subscale(llm_rotations, subject_ids):
    n = len(ALL_SUBSCALES)
    corr_sum = np.zeros((n, n))
    corr_count = np.zeros((n, n))

    for persona_file, persona_short in SCALES:
        if persona_file not in llm_rotations:
            continue
        rotation = llm_rotations[persona_file]
        persona_key = f"{persona_file}_persona"
        if persona_key not in rotation:
            continue

        persona_df = rotation[persona_key]
        sid_col = "Subject_ID"
        mask = persona_df[sid_col].astype(str).isin(subject_ids)
        if mask.sum() < 3:
            continue

        p_q = [c for c in persona_df.columns if c.startswith("Q")]
        p_items = persona_df.loc[mask, p_q].copy().reset_index(drop=True)
        p_labels = [(persona_short, f"{persona_short}_{q}") for q in p_q]
        p_renamed = p_items.rename(columns={q: f"{persona_short}_{q}" for q in p_q})
        p_scores = compute_subscale_scores(p_renamed, p_labels)

        for fill_file, fill_short in SCALES:
            if fill_file not in rotation:
                continue
            fill_df = rotation[fill_file]
            fill_mask = fill_df[sid_col].astype(str).isin(subject_ids)
            if fill_mask.sum() != mask.sum():
                continue

            l_q = [c for c in fill_df.columns if c.startswith("Q")]
            l_items = fill_df.loc[fill_mask, l_q].copy().reset_index(drop=True)
            l_labels = [(fill_short, f"{fill_short}_{q}") for q in l_q]
            l_renamed = l_items.rename(columns={q: f"{fill_short}_{q}" for q in l_q})
            l_scores = compute_subscale_scores(l_renamed, l_labels)

            for p_sub in p_scores.columns:
                if p_sub not in ALL_SUBSCALES:
                    continue
                p_idx = ALL_SUBSCALES.index(p_sub)
                for l_sub in l_scores.columns:
                    if l_sub not in ALL_SUBSCALES:
                        continue
                    l_idx = ALL_SUBSCALES.index(l_sub)
                    r = _corr(p_scores[p_sub].values, l_scores[l_sub].values)
                    if not np.isnan(r):
                        corr_sum[p_idx, l_idx] += r
                        corr_count[p_idx, l_idx] += 1

    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(corr_count > 0, corr_sum / corr_count, np.nan)


def compute_human_subscale(human_data, subject_ids):
    frames = []
    all_labels = []
    for scale_file, scale_short in SCALES:
        if scale_file not in human_data:
            continue
        df = human_data[scale_file]
        mask = df["Subject_ID"].astype(str).isin(subject_ids)
        q_cols = [c for c in df.columns if c.startswith("Q")]
        sub = df.loc[mask, q_cols].copy()
        rename = {q: f"{scale_short}_{q}" for q in q_cols}
        sub.rename(columns=rename, inplace=True)
        sub.index = range(len(sub))
        for q in q_cols:
            all_labels.append((scale_short, f"{scale_short}_{q}"))
        frames.append(sub)
    merged = pd.concat(frames, axis=1)
    scores = compute_subscale_scores(merged, all_labels)
    return scores.corr()


def aggregate_to_scale(sub_matrix):
    mapping = {}
    for scale_short, rules in SCORING_RULES.items():
        for sub_name in rules["subscales"]:
            mapping[f"{scale_short}_{sub_name}"] = scale_short

    n = len(SCALE_NAMES)
    scale_matrix = np.full((n, n), np.nan)
    for i, si in enumerate(SCALE_NAMES):
        subs_i = [idx for idx, s in enumerate(ALL_SUBSCALES) if mapping.get(s) == si]
        for j, sj in enumerate(SCALE_NAMES):
            subs_j = [idx for idx, s in enumerate(ALL_SUBSCALES) if mapping.get(s) == sj]
            vals = []
            for ai in subs_i:
                for bj in subs_j:
                    v = sub_matrix[ai, bj]
                    if not np.isnan(v):
                        vals.append(v)
            if vals:
                scale_matrix[i, j] = np.mean(vals)
    return scale_matrix


def main():
    existing_df, done = load_existing()
    print(f"Already completed: {len(done)} model+level combos")

    print("Loading human data...")
    human_data = load_human_scale_data()

    all_sids = None
    for sf in human_data:
        sids = set(human_data[sf]["Subject_ID"].astype(str))
        all_sids = sids if all_sids is None else all_sids & sids
    all_sids = sorted(all_sids)
    n_subjects = len(all_sids)
    print(f"  {n_subjects} subjects")

    item_cols, item_scales = build_item_col_list(human_data)
    within_mask, between_mask = build_within_between_masks(item_scales)
    n_items = len(item_cols)
    print(f"  {n_items} items ({within_mask.sum()} within pairs, {between_mask.sum()} between pairs)")

    rng = np.random.default_rng(42)
    new_rows = []

    for dirname, label in MODELS:
        llm_root = f"data/llm_behavior/{dirname}"
        if not (BASE_DIR / llm_root).exists():
            continue

        levels_pending = [lv for lv in ALL_LEVELS if (dirname, lv) not in done]
        if not levels_pending:
            print(f"\n  {label}: all done, skipping")
            continue

        print(f"\n  {label} ({dirname}) — running: {', '.join(levels_pending)}")

        llm_rotations = load_llm_rotation_data(llm_root)
        if not llm_rotations:
            continue

        # Determine what to compute
        need_item = any(lv.startswith("item") for lv in levels_pending)
        need_sub = "subscale" in levels_pending or "scale" in levels_pending

        # ── Point estimates ──
        if need_item:
            h_item_vals, h_item_cols = compute_human_item_matrix(human_data, set(all_sids), item_cols)
            llm_item_vals = compute_implicit_item_matrix(llm_rotations, set(all_sids), h_item_cols)
            n_it = len(h_item_cols)
            item_r = compare_upper_tri(h_item_vals, llm_item_vals, n_it)
            item_within_r = compare_upper_tri(h_item_vals, llm_item_vals, n_it, within_mask)
            item_between_r = compare_upper_tri(h_item_vals, llm_item_vals, n_it, between_mask)
            print(f"    Point: item={item_r:.3f}  within={item_within_r:.3f}  between={item_between_r:.3f}", end="")

        if need_sub:
            llm_sub = compute_implicit_subscale(llm_rotations, set(all_sids))
            h_sub_df = compute_human_subscale(human_data, set(all_sids))
            common = [s for s in ALL_SUBSCALES if s in h_sub_df.columns]
            sub_idx = [ALL_SUBSCALES.index(s) for s in common]
            sub_r = compare_upper_tri(
                h_sub_df.loc[common, common].values,
                llm_sub[np.ix_(sub_idx, sub_idx)],
                len(common))

            llm_scale = aggregate_to_scale(llm_sub)
            h_sub_matrix = np.full((len(ALL_SUBSCALES), len(ALL_SUBSCALES)), np.nan)
            for i, si in enumerate(ALL_SUBSCALES):
                if si in h_sub_df.columns:
                    for j, sj in enumerate(ALL_SUBSCALES):
                        if sj in h_sub_df.columns:
                            h_sub_matrix[i, j] = h_sub_df.loc[si, sj]
            h_scale = aggregate_to_scale(h_sub_matrix)
            n_scales = len(SCALE_NAMES)
            scale_r = compare_upper_tri(h_scale, llm_scale, n_scales)
            print(f"  sub={sub_r:.3f}  scale={scale_r:.3f}", end="")

        print()

        # ── Bootstrap ──
        boot = {lv: np.empty(N_BOOT) for lv in levels_pending}

        for b in range(N_BOOT):
            if (b + 1) % 500 == 0:
                print(f"    bootstrap {b+1}/{N_BOOT}")

            boot_idx = rng.choice(len(all_sids), size=n_subjects, replace=True)
            boot_ids = set([all_sids[i] for i in boot_idx])

            # Item levels
            if need_item:
                h_item_b, h_item_b_cols = compute_human_item_matrix(human_data, boot_ids, item_cols)
                llm_item_b = compute_implicit_item_matrix(llm_rotations, boot_ids, h_item_b_cols)
                n_it_b = len(h_item_b_cols)
                if "item" in boot:
                    boot["item"][b] = compare_upper_tri(h_item_b, llm_item_b, n_it_b)
                if "item_within" in boot:
                    boot["item_within"][b] = compare_upper_tri(h_item_b, llm_item_b, n_it_b, within_mask)
                if "item_between" in boot:
                    boot["item_between"][b] = compare_upper_tri(h_item_b, llm_item_b, n_it_b, between_mask)

            # Subscale + Scale
            if need_sub:
                llm_sub_b = compute_implicit_subscale(llm_rotations, boot_ids)
                h_sub_b = compute_human_subscale(human_data, boot_ids)
                common_b = [s for s in ALL_SUBSCALES if s in h_sub_b.columns]
                if len(common_b) < 5:
                    if "subscale" in boot:
                        boot["subscale"][b] = np.nan
                    if "scale" in boot:
                        boot["scale"][b] = np.nan
                    continue

                if "subscale" in boot:
                    sub_idx_b = [ALL_SUBSCALES.index(s) for s in common_b]
                    boot["subscale"][b] = compare_upper_tri(
                        h_sub_b.loc[common_b, common_b].values,
                        llm_sub_b[np.ix_(sub_idx_b, sub_idx_b)],
                        len(common_b))

                if "scale" in boot:
                    llm_scale_b = aggregate_to_scale(llm_sub_b)
                    h_sub_b_matrix = np.full((len(ALL_SUBSCALES), len(ALL_SUBSCALES)), np.nan)
                    for i, si in enumerate(ALL_SUBSCALES):
                        if si in h_sub_b.columns:
                            for j, sj in enumerate(ALL_SUBSCALES):
                                if sj in h_sub_b.columns:
                                    h_sub_b_matrix[i, j] = h_sub_b.loc[si, sj]
                    h_scale_b = aggregate_to_scale(h_sub_b_matrix)
                    boot["scale"][b] = compare_upper_tri(h_scale_b, llm_scale_b, n_scales)

        # Collect results
        alpha = (1 - CI_LEVEL) / 2
        point_rs = {
            "item": item_r if need_item else None,
            "item_within": item_within_r if need_item else None,
            "item_between": item_between_r if need_item else None,
            "subscale": sub_r if need_sub else None,
            "scale": scale_r if need_sub else None,
        }
        for level in levels_pending:
            boot_vals = boot[level]
            point_r = point_rs[level]
            lo = np.nanpercentile(boot_vals, alpha * 100)
            hi = np.nanpercentile(boot_vals, (1 - alpha) * 100)
            new_rows.append({
                "model": label,
                "dirname": dirname,
                "level": level,
                "r": point_r,
                "ci_lo": lo,
                "ci_hi": hi,
                "ci_width": hi - lo,
            })
            print(f"    {level:<14} r={point_r:.3f}  95% CI [{lo:.3f}, {hi:.3f}]")

        # Save incrementally
        save_results(existing_df, new_rows)

    # Final summary
    final_df = pd.read_csv(OUT_PATH) if OUT_PATH.exists() else pd.DataFrame()
    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")
    for level in ALL_LEVELS:
        sub = final_df[final_df["level"] == level].sort_values("r", ascending=False)
        if sub.empty:
            continue
        print(f"\n  {level.upper()}:")
        print(f"  {'Model':<18} {'r':>6} {'95% CI':>18} {'width':>6}")
        print(f"  {'-'*52}")
        for _, row in sub.iterrows():
            ci_str = f"[{row['ci_lo']:.3f}, {row['ci_hi']:.3f}]"
            print(f"  {row['model']:<18} {row['r']:>6.3f} {ci_str:>18} {row['ci_width']:>6.3f}")

    print(f"\nSaved to {OUT_PATH}")


if __name__ == "__main__":
    main()
