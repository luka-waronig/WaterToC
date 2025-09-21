import os
import sys
#so the code can find sever.py form the parent directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev
from server import detect_limit_cycle   
from src.model import WaterToC         

# --- parameter combinations and seeds we want ---
combos = [
    (0.5, 30.0, 0.4),
    (0.5, 30.0, 0.5),
    (0.5, 40.0, 0.3),
    (2.0, 30.0, 0.5),
    (2.0, 40.0, 0.5),
]
seeds = list(range(5))

out_dir = "limit_cycles"
os.makedirs(out_dir, exist_ok=True)

#defining the parameters for our simulation
def run_model(theta, cap, density, seed, max_steps=300):
    m = WaterToC(
        height=20, width=20,
        initial_humans=50, initial_ai=50,
        human_C_allocation=0.1, human_D_allocation=0.15,
        ai_C_allocation=2, ai_D_allocation=3,
        max_water_capacity=cap,
        water_cell_density=density,
        theta=theta,
        deviation_rate=0.1,
        seed=seed
    )
    for _ in range(max_steps):
        m.step()
    return m.datacollector.get_model_vars_dataframe().reset_index()

for theta_val, cap_val, density_val in combos:
    for seed in seeds:
        df = run_model(theta_val, cap_val, density_val, seed)
        x = df["Coop_Fraction"].values
        y = df["Environment_State"].values

        has_cycle, inds, period = detect_limit_cycle(x, y)
        if not has_cycle or len(inds) < 3:
            #skip if no cycle
            continue

        #close the loop
        cx = np.append(x[inds], x[inds][0])
        cy = np.append(y[inds], y[inds][0])

        #smooth with a periodic spline
        tck, _ = splprep([cx, cy], s=1e-4, per=True)
        u = np.linspace(0, 1, 300)
        sx, sy = splev(u, tck)

        #plot the cycles standalone
        plt.figure(figsize=(5,5))
        plt.plot(sx, sy, color="forestgreen", linewidth=2)
        #auto‐scale with 5% padding
        pad_x = (sx.max() - sx.min()) * 0.05
        pad_y = (sy.max() - sy.min()) * 0.05
        plt.xlim(sx.min() - pad_x, sx.max() + pad_x)
        plt.ylim(sy.min() - pad_y, sy.max() + pad_y)

        plt.xlabel("Coop_Fraction")
        plt.ylabel("Environment_State")
        plt.title(f"Limit Cycle — θ={theta_val}, C={cap_val}, ρ={density_val}")

        fn = (
            f"limitcycle_θ{theta_val}_C{cap_val}_ρ{density_val}"
            f"_seed{seed}.png"
        )
        out_path = os.path.join(out_dir, fn)
        plt.tight_layout()
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"Saved {out_path}")
