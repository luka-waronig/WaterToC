"""
Estimate oscillation periods and damping rates from median Coop_Fraction time series
aggregated over runs, and save both a results CSV and histogram figures (first need to run sweep_run.py results).

Workflow:
1) Load per-step sweep CSV, rename key columns.
2) Compute median Coop_Fraction over runs for each (theta, Kmax, rho_w, step).
3) Trim an initial transient, detect peaks, estimate mean inter-peak period.
4) Fit an exponential envelope to peak amplitudes to estimate damping rate λ.
5) Save per-combo estimates to results/oscillation_estimates.csv and two histograms to results/.
"""

from pathlib import Path
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.optimize import curve_fit

INPUT_CSV = "water_toc_sweep_results.csv"

RESULTS_DIR       = "results"
OUT_RESULTS_CSV   = os.path.join(RESULTS_DIR, "oscillation_estimates.csv")
OUT_PERIODS_PNG   = os.path.join(RESULTS_DIR, "oscillation_periods_hist.png")
OUT_DAMPING_PNG   = os.path.join(RESULTS_DIR, "oscillation_damping_hist.png")

TRANSIENT_FRAC   = 0.10
MIN_POINTS_AFTER = 50
PEAK_DISTANCE    = 5
PEAK_PROMINENCE  = 0.005

def envelope(t: np.ndarray, A0: float, lam: float) -> np.ndarray:
    """
    Exponential decay envelope evaluated at times t.

    Args:
        t: 1D array of times (shifted so t[0]=0 at first detected peak).
        A0: Initial amplitude at t=0.
        lam: Damping rate λ (per step).

    Returns:
        A0 * exp(-lam * t), same shape as t.
    """
    return A0 * np.exp(-lam * t)

df_raw = pd.read_csv(INPUT_CSV)
df_raw = df_raw.rename(columns={
    "Step": "step",
    "max_water_capacity": "Kmax",
    "water_cell_density": "rho_w"
})

med = (
    df_raw
    .groupby(["theta", "Kmax", "rho_w", "step"], as_index=False)
    .agg(median_coop=("Coop_Fraction", "median"))
    .sort_values(["theta", "Kmax", "rho_w", "step"])
)

results = []

for (theta, kmax, rho), sub in med.groupby(["theta", "Kmax", "rho_w"]):
    sub = sub.sort_values("step")
    time_arr = sub["step"].to_numpy()
    C_arr    = sub["median_coop"].to_numpy()
    total    = len(time_arr)

    if total == 0:
        results.append({"theta": theta, "Kmax": kmax, "rho_w": rho,
                        "n_peaks": 0, "period_est": np.nan, "damping_rate": np.nan})
        continue

    N0 = int(np.floor(total * TRANSIENT_FRAC))
    if total - N0 < MIN_POINTS_AFTER:
        N0 = max(0, total - MIN_POINTS_AFTER)

    t = time_arr[N0:]
    C = C_arr[N0:]

    if len(C) < MIN_POINTS_AFTER:
        results.append({"theta": theta, "Kmax": kmax, "rho_w": rho,
                        "n_peaks": 0, "period_est": np.nan, "damping_rate": np.nan})
        continue

    peaks, props = find_peaks(C, distance=PEAK_DISTANCE, prominence=PEAK_PROMINENCE)

    if len(peaks) < 2:
        results.append({"theta": theta, "Kmax": kmax, "rho_w": rho,
                        "n_peaks": int(len(peaks)), "period_est": np.nan, "damping_rate": np.nan})
        continue

    peak_times = t[peaks]
    intervals  = np.diff(peak_times)
    T_est = float(np.mean(intervals))

    t0 = peak_times - peak_times[0]
    try:
        (A0, lam), _ = curve_fit(envelope, t0, C[peaks], p0=(float(C[peaks][0]), 0.01), maxfev=5000)
        lam = float(lam)
    except Exception:
        lam = np.nan

    results.append({
        "theta": theta, "Kmax": kmax, "rho_w": rho,
        "n_peaks": int(len(peaks)),
        "period_est": T_est,
        "damping_rate": lam
    })

res_df = pd.DataFrame(results).sort_values(["theta", "Kmax", "rho_w"])

Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
res_df.to_csv(OUT_RESULTS_CSV, index=False)
print(f"Saved: {OUT_RESULTS_CSV}\n")
print(res_df.describe(include="all"))

plt.figure(figsize=(5,3))
plt.hist(res_df["period_est"].dropna(), bins=16, density=True, alpha=0.8)
plt.xlabel("Estimated period (steps)")
plt.ylabel("Density")
plt.title("Periods from median Coop_Fraction")
plt.tight_layout()
plt.savefig(OUT_PERIODS_PNG, dpi=300)
plt.show()
print(f"Saved: {OUT_PERIODS_PNG}")

plt.figure(figsize=(5,3))
plt.hist(res_df["damping_rate"].dropna(), bins=16, density=True, alpha=0.8)
plt.xlabel("Estimated damping rate λ")
plt.ylabel("Density")
plt.title("Damping rates from median Coop_Fraction")
plt.tight_layout()
plt.savefig(OUT_DAMPING_PNG, dpi=300)
plt.show()
print(f"Saved: {OUT_DAMPING_PNG}")
