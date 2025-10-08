"""
Render 3D surface plots of the *environmental state* tail medians over (ρ_w, Kmax),
faceted by θ. For each θ:
1) Aggregate per-run tail medians of Environment_State (last TAIL_FRAC of steps),
2) Median across runs per (θ, Kmax, ρ_w),
3) Interpolate to a grid and plot a colored surface with a shared color scale.
To be able to run, run sweep_run.py first.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm, ticker, colors
from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

CSV_PATH = "water_toc_sweep_results.csv"
OUTDIR   = "results"
FIG_NAME = "env_surfaces_by_theta.png"

TAIL_FRAC = 0.25
GRID_RES  = 60
VIEW_ELEV = 28
VIEW_AZIM = 225

os.makedirs(OUTDIR, exist_ok=True)
df = pd.read_csv(CSV_PATH)

needed = {
    "theta","max_water_capacity","water_cell_density",
    "seed","Step","Environment_State",
}
missing = needed - set(df.columns)
if missing:
    raise ValueError(f"CSV missing columns: {sorted(missing)}")

for col in needed:
    df[col] = pd.to_numeric(df[col], errors="coerce")

def tail_median_per_run(g: pd.DataFrame, frac: float) -> float:
    """
    Compute the tail median of Environment_State for a single run.
    Args:
        g: DataFrame of a single (theta, Kmax, rho_w, seed) run with 'Step' and 'Environment_State'.
        frac: Fraction (0–1] of the *final* steps to include in the tail window.
    Returns:
        Median Environment_State over the tail window.
    """
    g = g.sort_values("Step")
    n = len(g)
    take = max(1, int(np.ceil(frac * n)))
    return g["Environment_State"].iloc[-take:].median()

per_run = (
    df.groupby(["theta","max_water_capacity","water_cell_density","seed"], as_index=False)
      .apply(lambda g: pd.Series({"Env_tail_median_run": tail_median_per_run(g, TAIL_FRAC)}))
)

agg = (
    per_run.groupby(["theta","max_water_capacity","water_cell_density"], as_index=False)
           .agg(Env_tail_median=("Env_tail_median_run","median"))
).sort_values(["theta","max_water_capacity","water_cell_density"]).reset_index(drop=True)

thetas = list(np.sort(agg["theta"].unique()))

vmin = float(agg["Env_tail_median"].min())
vmax = float(agg["Env_tail_median"].max())
norm = colors.Normalize(vmin=vmin, vmax=vmax)
cmap = cm.inferno

fig = plt.figure(figsize=(13, 10))
axes = [fig.add_subplot(2, 2, i + 1, projection="3d") for i in range(4)]

last_surf = None

for idx, th in enumerate(thetas):
    ax = axes[idx]
    sub = agg[agg["theta"] == th].copy()

    xi = np.linspace(sub["water_cell_density"].min(), sub["water_cell_density"].max(), GRID_RES)
    yi = np.linspace(sub["max_water_capacity"].min(), sub["max_water_capacity"].max(), GRID_RES)
    XI, YI = np.meshgrid(xi, yi)

    pts  = sub[["water_cell_density","max_water_capacity"]].values
    vals = sub["Env_tail_median"].values

    ZI = griddata(pts, vals, (XI, YI), method="cubic")
    if ZI is None or np.isnan(ZI).all():
        ZI = griddata(pts, vals, (XI, YI), method="linear")
    if ZI is None or np.isnan(ZI).all():
        ZI = griddata(pts, vals, (XI, YI), method="nearest")
    else:
        ZN = griddata(pts, vals, (XI, YI), method="nearest")
        ZI = np.where(np.isnan(ZI), ZN, ZI)

    surf = ax.plot_surface(
        XI, YI, ZI, cmap=cmap, norm=norm,
        linewidth=0, antialiased=True, rstride=1, cstride=1, alpha=0.95
    )
    last_surf = surf

    ax.set_title(fr"$\theta = {th:g}$", pad=10, fontsize=12)
    ax.set_xlabel(r"Water density $\rho_w$", labelpad=10)
    ax.set_ylabel(r"Max water capacity $K_{\max}$", labelpad=12)
    ax.set_zlabel("Environmental state", labelpad=10)
    ax.view_init(elev=VIEW_ELEV, azim=VIEW_AZIM)

    ax.yaxis.set_major_locator(ticker.MultipleLocator(10))
    ax.set_zlim(vmin, vmax)
    ax.zaxis.set_major_locator(ticker.MaxNLocator(nbins=6, prune=None))
    ax.zaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))

    ax.tick_params(axis='both', which='major', labelsize=9, pad=6)
    ax.zaxis.set_tick_params(labelsize=9, pad=4)

for j in range(len(thetas), 4):
    axes[j].set_visible(False)

if last_surf is not None:
    plt.subplots_adjust(left=0.20, right=0.97, top=0.93, bottom=0.08,
                        wspace=0.18, hspace=0.18)
    cbar_ax = fig.add_axes([0.10, 0.18, 0.02, 0.64])
    cb = fig.colorbar(last_surf, cax=cbar_ax)
    cb.ax.yaxis.set_label_position('left')
    cb.set_label("Cooperation" if "Coop" in FIG_NAME else "Environmental state",
                 rotation=90, labelpad=14, va='center')
    cb.ax.yaxis.set_ticks_position('left')
    cb.ax.tick_params(labelsize=9)
else:
    plt.subplots_adjust(left=0.08, right=0.97, top=0.93, bottom=0.08,
                        wspace=0.18, hspace=0.18)

outpath = os.path.join(OUTDIR, FIG_NAME)
plt.savefig(outpath, dpi=300, bbox_inches="tight")
plt.show()
print(f"Saved: {outpath}")
