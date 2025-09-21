import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load your simulation results
df = pd.read_csv("water_toc_sweep_results.csv")

# Keep only final step per simulation run
df = df.sort_values("Step").groupby(["RunId", "iteration"]).tail(1)

# Average over the 50 iterations for each parameter set
grouped = df.groupby(["theta", "max_water_capacity", "water_cell_density"]).mean(numeric_only=True).reset_index()

# Setup subplot grid: one row per θ, two columns (Coop + Env)
thetas = sorted(grouped["theta"].unique())
n_rows = len(thetas)
fig, axes = plt.subplots(n_rows, 2, figsize=(12, 4 * n_rows))

# Ensure axes is always 2D
if n_rows == 1:
    axes = axes.reshape(1, 2)

for i, theta in enumerate(thetas):
    sub = grouped[grouped["theta"] == theta]
    
    # Pivot for Coop Fraction
    pivot_coop = sub.pivot(index="max_water_capacity", columns="water_cell_density", values="Coop_Fraction")
    sns.heatmap(pivot_coop, annot=True, fmt=".2f", cmap="viridis", ax=axes[i, 0])
    axes[i, 0].set_title(f"Coop Fraction (θ = {theta})")
    axes[i, 0].set_xlabel("Water Cell Density")
    axes[i, 0].set_ylabel("Max Water Capacity")

    # Pivot for Environment State
    pivot_env = sub.pivot(index="max_water_capacity", columns="water_cell_density", values="Environment_State")
    sns.heatmap(pivot_env, annot=True, fmt=".2f", cmap="magma", ax=axes[i, 1])
    axes[i, 1].set_title(f"Environment State (θ = {theta})")
    axes[i, 1].set_xlabel("Water Cell Density")
    axes[i, 1].set_ylabel("Max Water Capacity")

plt.tight_layout(h_pad=2.5)
plt.subplots_adjust(top=0.95, hspace=0.6, wspace=0.3)

plt.savefig("heatmap_grid_by_theta.png", dpi=300, bbox_inches="tight")
plt.show()
