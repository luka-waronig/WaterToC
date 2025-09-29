import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple
from water_toc.model import WaterToC  # adjust import path if needed

"""
This code executes statistical tests for autocorrelation and clustering by running the simulation with any chosen parameter set.
Outputs include heatmaps for both final cooperation fractions, as well as Moran´s I scores per parameter combination.
"""
# ======================
# Moran's I
# ======================
def _roll_sum_4(arr: np.ndarray, periodic: bool) -> np.ndarray:
    if periodic:
        return (np.roll(arr, 1, 0) + np.roll(arr, -1, 0) +
                np.roll(arr, 1, 1) + np.roll(arr, -1, 1))
    s = np.zeros_like(arr, dtype=arr.dtype)
    s[1:, :] += arr[:-1, :]
    s[:-1, :] += arr[1:, :]
    s[:, 1:] += arr[:, :-1]
    s[:, :-1] += arr[:, 1:]
    return s

def _degree_map(shape: Tuple[int, int], periodic: bool) -> np.ndarray:
    ones = np.ones(shape, dtype=np.int16)
    return _roll_sum_4(ones, periodic)

def morans_I(field: np.ndarray, periodic: bool = True) -> float:
    x = np.asarray(field, dtype=float)
    N = x.size
    mu = x.mean()
    xc = x - mu
    W_sum = float(_degree_map(x.shape, periodic).sum())
    if W_sum == 0:
        return np.nan
    nb_sum = _roll_sum_4(xc, periodic)
    num = float(np.sum(xc * nb_sum))
    den = float(np.sum(xc * xc))
    return (N / W_sum) * (num / den) if den != 0 else np.nan


# ======================
# Sweep config
# ======================
THETAS   = [0.5, 2.0, 5.0, 10.0]
#max water capacity
KMAXES   = [10, 20, 30, 40, 50]  
#water cell density        
DENSITIES= [0.2, 0.4, 0.5, 0.6, 0.7]      
#total steps per run 
STEPS      = 100    
#average over last 25%                
TAIL_FRAC  = 0.25                    
HEIGHT, WIDTH = 20, 20              
SEED       = 42                      
OUTDIR     = "sweep_outputs"
os.makedirs(OUTDIR, exist_ok=True)


def run_one(theta, kmax, density, steps=STEPS, tail_frac=TAIL_FRAC, seed=SEED):
    m = WaterToC(
        height=HEIGHT, width=WIDTH,
        max_water_capacity=kmax,
        water_cell_density=density,
        theta=theta,
        seed=seed,
        save_snapshots=False  
    )
    for _ in range(steps):
        m.step()

    df = m.datacollector.get_model_vars_dataframe().reset_index(drop=True)

    L = len(df)
    a = int((1 - tail_frac) * L)
    tail = df.iloc[a:]

    #mean cooperation over tail
    coop_fraction = float(tail["Coop_Fraction"].mean())

    #Moran's I over tail
    morans_vals = []
    for flat in tail["Coop_Map_Flat"]:
        arr = np.array(flat, dtype=float).reshape(HEIGHT, WIDTH)
        morans_vals.append(morans_I(arr, periodic=True)) 
    morans_avg = float(np.nanmean(morans_vals))

    return coop_fraction, morans_avg


def main():
    rows = []
    for theta in THETAS:
        for k in KMAXES:
            for rho in DENSITIES:
                cf, mi = run_one(theta, k, rho)
                rows.append({
                    "theta": theta,
                    "Kmax": k,
                    "rho_w": rho,
                    "CoopFraction_tailmean": cf,
                    "MoransI_tailmean": mi,
                })
                print(f"θ={theta:>4}, K={k:>2}, ρw={rho:.1f} -> Coop={cf:.3f}, MoranI={mi:.3f}")

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(OUTDIR, "spatial_sweep_results.csv"), index=False)

    #makes heatmaps per theta
    for theta in THETAS:
        sub = res[res["theta"] == theta].copy()
        cf_grid = sub.pivot(index="Kmax", columns="rho_w", values="CoopFraction_tailmean").reindex(index=KMAXES, columns=DENSITIES)
        mi_grid = sub.pivot(index="Kmax", columns="rho_w", values="MoransI_tailmean").reindex(index=KMAXES, columns=DENSITIES)

        #Coop Fraction heatmap
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        im1 = ax1.imshow(cf_grid.values, aspect="auto", origin="lower")
        ax1.set_xticks(range(len(DENSITIES))); ax1.set_xticklabels(DENSITIES)
        ax1.set_yticks(range(len(KMAXES)));   ax1.set_yticklabels(KMAXES)
        ax1.set_xlabel("Water Cell Density (ρw)")
        ax1.set_ylabel("Max Water Capacity (Kmax)")
        ax1.set_title(f"Coop Fraction (tail mean), θ = {theta}")
        cbar1 = plt.colorbar(im1, ax=ax1)
        cbar1.set_label("Coop Fraction")
        for i in range(cf_grid.shape[0]):
            for j in range(cf_grid.shape[1]):
                ax1.text(j, i, f"{cf_grid.values[i, j]:.2f}", ha="center", va="center")
        fig1.tight_layout()
        fig1.savefig(os.path.join(OUTDIR, f"heatmap_coop_theta_{theta}.png"), dpi=200)
        plt.close(fig1)

        #Moran's I heatmap 
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        im2 = ax2.imshow(mi_grid.values, aspect="auto", origin="lower")
        ax2.set_xticks(range(len(DENSITIES))); ax2.set_xticklabels(DENSITIES)
        ax2.set_yticks(range(len(KMAXES)));   ax2.set_yticklabels(KMAXES)
        ax2.set_xlabel("Water Cell Density (ρw)")
        ax2.set_ylabel("Max Water Capacity (Kmax)")
        ax2.set_title(f"Moran's I (tail mean), θ = {theta}")
        cbar2 = plt.colorbar(im2, ax=ax2)
        cbar2.set_label("Moran's I")
        for i in range(mi_grid.shape[0]):
            for j in range(mi_grid.shape[1]):
                ax2.text(j, i, f"{mi_grid.values[i, j]:.2f}", ha="center", va="center")
        fig2.tight_layout()
        fig2.savefig(os.path.join(OUTDIR, f"heatmap_moran_theta_{theta}.png"), dpi=200)
        plt.close(fig2)

    print(f"Done. CSV + heatmaps in: {OUTDIR}")


if __name__ == "__main__":
    main()
