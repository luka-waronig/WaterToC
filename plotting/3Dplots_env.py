import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import Axes3D

# 1) Load & preprocess
df = pd.read_csv("water_toc_sweep_results.csv")
df_final = (
    df.sort_values("Step")
      .groupby(["RunId","iteration"], as_index=False)
      .tail(1)
)
grouped = (
    df_final
    .groupby(["theta","water_cell_density","max_water_capacity"], as_index=False)
    .mean(numeric_only=True)
)

# 2) Prepare θ slices
thetas = sorted(grouped["theta"].unique())
fig = plt.figure(figsize=(14,10))

for idx, θ in enumerate(thetas):
    ax = fig.add_subplot(2, 2, idx+1, projection="3d")
    sub = grouped[grouped["theta"] == θ]

    # Create regular grid in (density, capacity)
    xi = np.linspace(sub["water_cell_density"].min(),
                     sub["water_cell_density"].max(), 50)
    yi = np.linspace(sub["max_water_capacity"].min(),
                     sub["max_water_capacity"].max(), 50)
    xi, yi = np.meshgrid(xi, yi)

    # Interpolate Environment_State on that grid
    points = sub[["water_cell_density","max_water_capacity"]].values
    values = sub["Environment_State"].values
    zi = griddata(points, values, (xi, yi), method="cubic")

    # Plot the smooth surface
    surf = ax.plot_surface(
        xi, yi, zi,
        cmap=cm.inferno,
        linewidth=0,
        antialiased=True,
        rstride=1, cstride=1,
        alpha=0.9
    )

    ax.set_title(f"θ = {θ:.1f}", pad=10, fontsize=12)
    ax.set_xlabel("Water Cell Density", labelpad=8)
    ax.set_ylabel("Max Water Capacity", labelpad=8)
    ax.set_zlabel("Environment State", labelpad=8)
    ax.view_init(elev=30, azim=220)  # consistent camera angle

# 3) Single colorbar
cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
fig.colorbar(surf, cax=cbar_ax, label="Environment State")

# 4) Layout tweaks
plt.subplots_adjust(left=0.07, right=0.90, top=0.92, bottom=0.08,
                    wspace=0.25, hspace=0.25)
plt.savefig("env_surfaces_by_theta.png", dpi=300)
plt.show()
