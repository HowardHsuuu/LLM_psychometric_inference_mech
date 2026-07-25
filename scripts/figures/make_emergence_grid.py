"""
Generate figI2_emergence_grid_qwen.png — three-panel heatmap of
Qwen 3B/7B/14B Instruct activation cosine similarity matrices.

Usage: put this script at the project root and run
    python make_emergence_grid.py
"""
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = PROJECT_ROOT / "outputs" / "mechanistic"

# (model_name, best_layer, mantel_r_display)
MODELS = [
    ("qwen3b_instruct",  22, r"Qwen 3B Instruct (Mantel $r = 0.03$)"),
    ("qwen7b_instruct",  20, r"Qwen 7B Instruct (Mantel $r = 0.42$)"),
    ("qwen14b_instruct", 36, r"Qwen 14B Instruct (Mantel $r = 0.66$)"),
]

fig, axes = plt.subplots(1, 3, figsize=(21, 6))

for ax, (name, layer, title) in zip(axes, MODELS):
    csv_path = ROOT / f"results_{name}" / "geometry" / f"subscale_cosine_sim_L{layer}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing: {csv_path}")
    df = pd.read_csv(csv_path, index_col=0)
    labels = [l.split("_", 1)[1] if "_" in l else l for l in df.index]
    sns.heatmap(
        df.values,
        cmap="RdBu_r", center=0, vmin=-1, vmax=1,
        xticklabels=labels, yticklabels=labels,
        annot=True, fmt=".2f", annot_kws={"size": 6},
        ax=ax,
        cbar_kws={"shrink": 0.6},
    )
    ax.set_title(title, fontsize=13)
    ax.tick_params(axis="both", labelsize=7)

plt.suptitle(
    "Emergence of psychometric geometry across Qwen instruct model sizes",
    fontsize=15, y=1.02
)
plt.tight_layout()

out_path = PROJECT_ROOT / "reports" / "figures" / "figI2_emergence_grid_qwen.png"
out_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out_path, dpi=200, bbox_inches="tight")
plt.close()
print(f"Saved: {out_path}")
