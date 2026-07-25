"""
Psychometric structure analysis: item-level network + EFA across 7 scales.

Data: SED + SEDC + SEDD (N=272)
Scales: IRI(28), PANAS(20), POM(7), big_five(10), in_inter_dependent(24),
        Life_Satisfaction(5), Loneliness(20) → 114 items total

Outputs:
  1. Correlation heatmap (item × item, grouped by scale)
  2. Network analysis (Graphical LASSO + community detection)
  3. EFA (parallel analysis + factor loadings)
"""

import os
import sys
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.covariance import GraphicalLassoCV
import networkx as nx
from networkx.algorithms.community import louvain_communities
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity, calculate_kmo

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from psychometric_inference.scoring import un_reverse_df
from psychometric_inference.paths import HUMAN_DATA_DIR, HUMAN_STRUCTURE_OUTPUT_DIR

warnings.filterwarnings("ignore")

BASE = str(HUMAN_DATA_DIR)
OUT_DIR = str(HUMAN_STRUCTURE_OUTPUT_DIR)
os.makedirs(OUT_DIR, exist_ok=True)

DATASETS = ["SED", "SEDC", "SEDD"]
SCALES = [
    ("IRI", "IRI"),
    ("PANAS", "PANAS"),
    ("POM", "POM"),
    ("big_five", "BigFive"),
    ("in_inter_dependent", "SelfConst"),  # self-construal
    ("Life_Satisfaction", "LifeSat"),
    ("Loneliness", "Lonely"),
]

# Color palette for 7 scales
SCALE_COLORS = {
    "IRI": "#e41a1c",
    "PANAS": "#377eb8",
    "POM": "#4daf4a",
    "BigFive": "#984ea3",
    "SelfConst": "#ff7f00",
    "LifeSat": "#a65628",
    "Lonely": "#f781bf",
}


# ── Step 0: Load & merge data ──
def load_data():
    """Load and concatenate item-level data from all datasets and scales."""
    all_items = []
    item_labels = []  # (scale_short, item_name)

    for scale_file, scale_short in SCALES:
        frames = []
        for ds in DATASETS:
            fpath = os.path.join(BASE, ds, f"{scale_file}.csv")
            df = pd.read_csv(fpath)
            q_cols = [c for c in df.columns if c.startswith("Q")]
            sub = df[q_cols].copy()
            # Un-reverse: data/human stores already-reversed scores
            sub = un_reverse_df(sub, scale_file)
            frames.append(sub)

        combined = pd.concat(frames, ignore_index=True)
        # Rename columns: e.g. Q1 → IRI_Q1
        rename = {c: f"{scale_short}_{c}" for c in combined.columns}
        combined.rename(columns=rename, inplace=True)

        all_items.append(combined)
        for c in combined.columns:
            item_labels.append((scale_short, c))

    data = pd.concat(all_items, axis=1)
    print(f"Data loaded: {data.shape[0]} subjects × {data.shape[1]} items")
    return data, item_labels


# ── Step 1: Correlation heatmap ──
def plot_correlation_heatmap(data, item_labels):
    """Plot item-item correlation matrix, grouped by scale."""
    corr = data.corr()

    fig, ax = plt.subplots(figsize=(18, 15))
    sns.heatmap(
        corr,
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        center=0,
        xticklabels=False,
        yticklabels=False,
        ax=ax,
        cbar_kws={"shrink": 0.6, "label": "Pearson r"},
    )

    # Draw scale boundary lines and labels
    pos = 0
    scale_positions = []
    for scale_short in dict(SCALES).values():
        n_items = sum(1 for s, _ in item_labels if s == scale_short)
        scale_positions.append((pos, pos + n_items, scale_short))
        ax.axhline(y=pos, color="black", linewidth=0.8)
        ax.axvline(x=pos, color="black", linewidth=0.8)
        # Label
        mid = pos + n_items / 2
        ax.text(-1.5, mid, scale_short, ha="right", va="center", fontsize=9, fontweight="bold")
        ax.text(mid, -1.5, scale_short, ha="center", va="bottom", fontsize=9, fontweight="bold", rotation=45)
        pos += n_items

    ax.set_title("Item-level Correlation Matrix (N=272, 7 scales, 114 items)", fontsize=14, pad=20)
    plt.tight_layout()
    fpath = os.path.join(OUT_DIR, "01_correlation_heatmap.png")
    plt.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fpath}")


# ── Step 2: Network analysis ──
def network_analysis(data, item_labels):
    """Estimate sparse partial correlation network via Graphical LASSO, then community detection."""

    # Standardize data
    X = data.values
    X_std = (X - X.mean(axis=0)) / X.std(axis=0)

    # Fit Graphical LASSO with cross-validation
    print("\nFitting Graphical LASSO (CV for optimal lambda)...")
    model = GraphicalLassoCV(cv=5, max_iter=500)
    model.fit(X_std)

    # Precision matrix → partial correlations
    precision = model.precision_
    # partial_corr[i,j] = -precision[i,j] / sqrt(precision[i,i] * precision[j,j])
    d = np.sqrt(np.diag(precision))
    partial_corr = -precision / np.outer(d, d)
    np.fill_diagonal(partial_corr, 0)

    print(f"  Lambda (alpha) selected: {model.alpha_:.4f}")
    n_edges = np.sum(np.abs(partial_corr) > 1e-10) // 2
    max_edges = len(data.columns) * (len(data.columns) - 1) // 2
    print(f"  Non-zero edges: {n_edges} / {max_edges} ({100*n_edges/max_edges:.1f}%)")

    # Build networkx graph
    cols = list(data.columns)
    G = nx.Graph()
    for i, col in enumerate(cols):
        scale = [s for s, c in item_labels if c == col][0]
        G.add_node(col, scale=scale)

    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            w = partial_corr[i, j]
            if abs(w) > 1e-10:
                G.add_edge(cols[i], cols[j], weight=w)

    # Community detection (Louvain)
    print("\nRunning Louvain community detection...")
    communities = louvain_communities(G, weight="weight", resolution=1.0, seed=42)
    communities = sorted(communities, key=len, reverse=True)

    print(f"  Found {len(communities)} communities")
    for idx, comm in enumerate(communities):
        # Count scale membership in this community
        scale_counts = {}
        for node in comm:
            s = G.nodes[node]["scale"]
            scale_counts[s] = scale_counts.get(s, 0) + 1
        counts_str = ", ".join(f"{k}:{v}" for k, v in sorted(scale_counts.items(), key=lambda x: -x[1]))
        print(f"  Community {idx+1} ({len(comm)} items): {counts_str}")

    # Assign community to each node
    node_community = {}
    for idx, comm in enumerate(communities):
        for node in comm:
            node_community[node] = idx

    # ── Network visualization ──
    fig, axes = plt.subplots(1, 2, figsize=(24, 11))

    # Use spring layout with edge weights
    pos = nx.spring_layout(G, k=0.3, iterations=80, seed=42, weight="weight")

    # Prepare edge drawing (positive=blue, negative=red)
    edges = G.edges(data=True)
    edge_colors = []
    edge_widths = []
    for u, v, d in edges:
        w = d["weight"]
        edge_colors.append("#4477AA" if w > 0 else "#CC6677")
        edge_widths.append(min(abs(w) * 3, 2.5))

    # Panel 1: colored by original scale
    ax = axes[0]
    node_colors_scale = [SCALE_COLORS[G.nodes[n]["scale"]] for n in G.nodes()]
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color=edge_colors, width=edge_widths, alpha=0.3)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors_scale, node_size=60, edgecolors="black", linewidths=0.3)
    legend_handles = [mpatches.Patch(color=c, label=s) for s, c in SCALE_COLORS.items()]
    ax.legend(handles=legend_handles, loc="upper left", fontsize=8, framealpha=0.9)
    ax.set_title("Network colored by ORIGINAL SCALE", fontsize=12)
    ax.axis("off")

    # Panel 2: colored by detected community
    ax = axes[1]
    comm_cmap = plt.cm.Set3(np.linspace(0, 1, max(len(communities), 3)))
    node_colors_comm = [comm_cmap[node_community[n]] for n in G.nodes()]
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color=edge_colors, width=edge_widths, alpha=0.3)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors_comm, node_size=60, edgecolors="black", linewidths=0.3)
    comm_handles = [mpatches.Patch(color=comm_cmap[i], label=f"Community {i+1}") for i in range(len(communities))]
    ax.legend(handles=comm_handles, loc="upper left", fontsize=8, framealpha=0.9)
    ax.set_title("Network colored by DETECTED COMMUNITY", fontsize=12)
    ax.axis("off")

    plt.suptitle("Partial Correlation Network (Graphical LASSO, N=272, 114 items)", fontsize=14)
    plt.tight_layout()
    fpath = os.path.join(OUT_DIR, "02_network_analysis.png")
    plt.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fpath}")

    # ── Community vs Scale agreement table ──
    print("\n── Community × Scale Crosstab ──")
    rows = []
    for node in G.nodes():
        rows.append({"item": node, "scale": G.nodes[node]["scale"], "community": node_community[node] + 1})
    ct = pd.DataFrame(rows)
    crosstab = pd.crosstab(ct["community"], ct["scale"])
    print(crosstab.to_string())

    # Save crosstab
    fpath = os.path.join(OUT_DIR, "02_community_scale_crosstab.csv")
    crosstab.to_csv(fpath)
    print(f"Saved: {fpath}")

    return partial_corr, G, communities


# ── Step 3: EFA ──
def efa_analysis(data, item_labels):
    """Parallel analysis + EFA."""

    # Bartlett's test and KMO
    chi_sq, p_value = calculate_bartlett_sphericity(data)
    print(f"\nBartlett's test: chi²={chi_sq:.1f}, p={p_value:.2e}")

    kmo_all, kmo_model = calculate_kmo(data)
    print(f"KMO: {kmo_model:.3f}")

    # Parallel analysis to determine number of factors
    print("\nRunning parallel analysis...")
    n_items = data.shape[1]
    n_obs = data.shape[0]

    # Get eigenvalues from real data
    corr_matrix = data.corr().values
    real_eigenvalues = np.sort(np.linalg.eigvalsh(corr_matrix))[::-1]

    # Generate random eigenvalues (Monte Carlo)
    n_iter = 100
    random_eigenvalues = np.zeros((n_iter, n_items))
    rng = np.random.default_rng(42)
    for i in range(n_iter):
        random_data = rng.standard_normal((n_obs, n_items))
        random_corr = np.corrcoef(random_data, rowvar=False)
        random_eigenvalues[i] = np.sort(np.linalg.eigvalsh(random_corr))[::-1]

    mean_random = random_eigenvalues.mean(axis=0)
    p95_random = np.percentile(random_eigenvalues, 95, axis=0)

    # Number of factors: eigenvalue > 95th percentile of random
    n_factors = np.sum(real_eigenvalues > p95_random)
    print(f"Parallel analysis suggests {n_factors} factors (95th percentile criterion)")

    # Plot parallel analysis
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(1, min(31, n_items + 1))
    ax.plot(x, real_eigenvalues[: len(x)], "bo-", label="Real data", markersize=5)
    ax.plot(x, mean_random[: len(x)], "r--", label="Random mean", alpha=0.7)
    ax.plot(x, p95_random[: len(x)], "r:", label="Random 95th pctl", alpha=0.7)
    ax.axhline(y=1, color="gray", linestyle="--", alpha=0.5, label="Kaiser criterion (eigenvalue=1)")
    ax.axvline(x=n_factors, color="green", linestyle="--", alpha=0.5, label=f"Suggested: {n_factors} factors")
    ax.set_xlabel("Factor number")
    ax.set_ylabel("Eigenvalue")
    ax.set_title("Parallel Analysis Scree Plot")
    ax.legend()
    plt.tight_layout()
    fpath = os.path.join(OUT_DIR, "03_parallel_analysis.png")
    plt.savefig(fpath, dpi=150)
    plt.close()
    print(f"Saved: {fpath}")

    # Run EFA
    print(f"\nRunning EFA with {n_factors} factors (oblimin rotation)...")
    fa = FactorAnalyzer(n_factors=n_factors, rotation="oblimin", method="ml", is_corr_matrix=False)
    fa.fit(data)

    loadings = pd.DataFrame(
        fa.loadings_,
        index=data.columns,
        columns=[f"F{i+1}" for i in range(n_factors)],
    )

    # Add scale membership
    scale_map = {c: s for s, c in item_labels}
    loadings["Scale"] = loadings.index.map(scale_map)

    # Save full loadings
    fpath = os.path.join(OUT_DIR, "03_factor_loadings.csv")
    loadings.to_csv(fpath)
    print(f"Saved: {fpath}")

    # Print dominant factor per scale
    print("\n── Dominant factor loadings by scale ──")
    factor_cols = [f"F{i+1}" for i in range(n_factors)]
    for scale_short in dict(SCALES).values():
        mask = loadings["Scale"] == scale_short
        mean_abs = loadings.loc[mask, factor_cols].abs().mean()
        dominant = mean_abs.idxmax()
        print(f"  {scale_short:<12} → {dominant} (mean |loading| = {mean_abs[dominant]:.3f})")

    # Heatmap of loadings
    fig, ax = plt.subplots(figsize=(12, 20))
    plot_data = loadings[factor_cols].copy()

    # Add scale color bar
    sns.heatmap(
        plot_data,
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        ax=ax,
        yticklabels=True,
        cbar_kws={"shrink": 0.5, "label": "Loading"},
    )
    ax.set_title(f"EFA Factor Loadings ({n_factors} factors, oblimin rotation)", fontsize=12)
    ax.tick_params(axis="y", labelsize=5)
    plt.tight_layout()
    fpath = os.path.join(OUT_DIR, "03_factor_loadings_heatmap.png")
    plt.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fpath}")

    # Factor correlation matrix
    if fa.phi_ is not None:
        print("\n── Factor correlation matrix ──")
        factor_corr = pd.DataFrame(fa.phi_, index=factor_cols, columns=factor_cols)
        print(factor_corr.round(3).to_string())
        fpath = os.path.join(OUT_DIR, "03_factor_correlations.csv")
        factor_corr.to_csv(fpath)

    return fa, loadings, n_factors


# ── Main ──
def main():
    data, item_labels = load_data()

    print("\n" + "=" * 60)
    print("STEP 1: Correlation Heatmap")
    print("=" * 60)
    plot_correlation_heatmap(data, item_labels)

    print("\n" + "=" * 60)
    print("STEP 2: Network Analysis")
    print("=" * 60)
    partial_corr, G, communities = network_analysis(data, item_labels)

    print("\n" + "=" * 60)
    print("STEP 3: Exploratory Factor Analysis")
    print("=" * 60)
    fa, loadings, n_factors = efa_analysis(data, item_labels)

    print("\n" + "=" * 60)
    print("DONE — all outputs in:", OUT_DIR)
    print("=" * 60)


if __name__ == "__main__":
    main()
