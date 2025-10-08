"""
Render stacked heatmaps (one per θ) for a spatial analysis from an aggregated sweep CSV 
(run spatial_analysis_sweep to obtain it).
Expects columns:
theta, Kmax, rho_w, CoopFraction_tailmedian_overruns, MoransI_occ_tailmedian_overruns,
MoransI_wocc_tailmedian_overruns, ClustersAgents_tailmedian_overruns,
LargestClusterAgents_tailmedian_overruns, MeanClusterAgents_tailmedian_overruns.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

CSV_PATH = "summary_agents_clusters_and_pointprocess_MEDIANS.csv"
RESULTS_DIR = "results"

df = pd.read_csv(CSV_PATH)
df["theta"] = df["theta"].astype(float)
df["Kmax"]  = df["Kmax"].astype(int)
df["rho_w"] = df["rho_w"].astype(float)
df = df.sort_values(["theta", "Kmax", "rho_w"]).reset_index(drop=True)

thetas = sorted(df["theta"].unique())
kvals  = sorted(df["Kmax"].unique())
rvals  = sorted(df["rho_w"].unique())

def draw_one_heatmap(sub: pd.DataFrame, value_col: str, title: str, ax, cmap, fmt=".2f", vmin=None, vmax=None) -> None:
    """
    Draw a single Kmax×rho_w heatmap for one θ.
    Args:
        sub: Subset of the dataframe filtered to a single θ.
        value_col: Column name to visualize.
        title: Axes title text.
        ax: Matplotlib Axes to draw on.
        cmap: Matplotlib/Seaborn colormap string.
        fmt: Annotation number format for cells.
        vmin, vmax: Optional fixed color limits for consistency across rows.
    """
    piv = sub.pivot(index="Kmax", columns="rho_w", values=value_col)
    piv = piv.reindex(index=kvals, columns=rvals)
    sns.heatmap(
        piv, annot=True, fmt=fmt, cmap=cmap, ax=ax, cbar=True,
        vmin=vmin, vmax=vmax, linewidths=0.5, linecolor="white"
    )
    ax.set_title(title)
    ax.set_xlabel("Water Cell Density (ρ_w)")
    ax.set_ylabel("Max Water Capacity (Kmax)")
    ax.set_yticklabels([str(int(k)) for k in piv.index])
    ax.set_xticklabels([f"{c:.1f}" for c in piv.columns], rotation=0)

def _fmt_theta(t: float) -> str:
    """
    Format θ for titles.
    Args:
        t: Theta value.
    Returns:
        Short string without trailing zeros.
    """
    s = f"{t:.2f}".rstrip("0").rstrip(".")
    return s

def plot_metric_stack(df_in: pd.DataFrame, metric_col: str, pretty_name: str, fname: str, cmap: str, fmt: str=".2f", fixed_range: bool=True) -> None:
    """
    Plot a vertical stack of heatmaps (one per θ) for a given metric and save to RESULTS_DIR.
    Args:
        df_in: Aggregated dataframe.
        metric_col: Column to visualize.
        pretty_name: Human-readable title prefix.
        fname: Output filename (PNG) saved under RESULTS_DIR.
        cmap: Colormap name.
        fmt: Cell annotation number format.
        fixed_range: If True, use shared vmin/vmax across all θ rows.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    vmin = vmax = None
    if fixed_range:
        v = df_in[metric_col].to_numpy(dtype=float)
        v = v[np.isfinite(v)]
        if v.size:
            vmin = float(np.nanmin(v))
            vmax = float(np.nanmax(v))
    n_rows = len(thetas)
    fig, axes = plt.subplots(n_rows, 1, figsize=(7.5, 3.8 * n_rows))
    if n_rows == 1:
        axes = [axes]
    for i, th in enumerate(thetas):
        sub = df_in[df_in["theta"] == th]
        draw_one_heatmap(
            sub=sub,
            value_col=metric_col,
            title=f"{pretty_name} (θ = {_fmt_theta(th)})",
            ax=axes[i],
            cmap=cmap,
            fmt=fmt,
            vmin=vmin,
            vmax=vmax
        )
    plt.tight_layout(h_pad=2.0)
    plt.subplots_adjust(top=0.95, hspace=0.6)
    outpath = os.path.join(RESULTS_DIR, fname)
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Saved: {outpath}")

plot_metric_stack(
    df, "CoopFraction_tailmedian_overruns",
    "Cooperation Fraction",
    "heatmap_coop.png", "viridis", ".2f"
)

plot_metric_stack(
    df, "MoransI_occ_tailmedian_overruns",
    "Moran's I (occupied cells)",
    "heatmap_moransI_occ.png", "magma", ".3f"
)

plot_metric_stack(
    df, "MoransI_wocc_tailmedian_overruns",
    "Moran's I (occupied mask)",
    "heatmap_moransI_wocc.png", "magma", ".3f"
)

plot_metric_stack(
    df, "LargestClusterAgents_tailmedian_overruns",
    "Largest Cooperative Cluster",
    "heatmap_largest_cluster_agents.png", "viridis", ".1f"
)

plot_metric_stack(
    df, "ClustersAgents_tailmedian_overruns",
    "Number of Cooperative Clusters",
    "heatmap_cluster_count_agents.png", "crest", ".1f"
)

plot_metric_stack(
    df, "MeanClusterAgents_tailmedian_overruns",
    "Mean Cooperative Cluster Size",
    "heatmap_mean_cluster_agents.png", "viridis", ".2f"
)
