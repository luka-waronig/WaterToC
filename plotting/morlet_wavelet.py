"""
Compute wavelet-based oscillation scores from median Coop_Fraction time series and,
for each ρ_w, save exactly TWO scalogram images:
  1) scalogram_max_rho{rho}.png (highest oscillation score)
  2) scalogram_min_rho{rho}.png (lowest  oscillation score)
Additionally, save per-ρ and combined score tables.
To run, first run sweep_run.py.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pywt

CSV_FILE   = "water_toc_sweep_results.csv"
THETAS     = [1.4, 2.0, 5.0, 10.0]
KMAXES     = [10, 20, 30, 40]
RHO_LIST   = [0.2, 0.4, 0.5, 0.7]

TRANSIENT  = 10
WAVELET    = "morl"
N_FREQS    = 200
MIN_PERIOD = 5
MAX_PERIOD = 100

SCORE_TOP_PCT = 90.0

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

df_runs = pd.read_csv(CSV_FILE)
df_runs = df_runs.rename(columns={"Step": "step", "max_water_capacity": "Kmax", "water_cell_density": "rho_w"})

required_cols = {"RunId", "iteration", "theta", "Kmax", "rho_w", "seed", "step", "Coop_Fraction"}
missing = required_cols - set(df_runs.columns)
if missing:
    raise KeyError(f"Missing columns in {CSV_FILE}: {sorted(missing)}")

df_runs["theta"] = df_runs["theta"].astype(float)
df_runs["Kmax"]  = df_runs["Kmax"].astype(float)
df_runs["rho_w"] = df_runs["rho_w"].astype(float)
df_runs["step"]  = df_runs["step"].astype(int)

df_runs = df_runs[
    df_runs["theta"].isin(THETAS)
    & df_runs["Kmax"].isin(KMAXES)
    & df_runs["rho_w"].isin(RHO_LIST)
].copy()

coop = (
    df_runs
    .groupby(["theta", "Kmax", "rho_w", "step"], as_index=False)
    .agg(median=("Coop_Fraction", "median"))
    .sort_values(["theta", "Kmax", "rho_w", "step"])
)

freqs = np.linspace(1.0 / MAX_PERIOD, 1.0 / MIN_PERIOD, N_FREQS)
cf = pywt.central_frequency(WAVELET)

def compute_cwt_power(x: np.ndarray, t: np.ndarray):
    """
    Compute |CWT| power for a 1D signal.
    Args:
        x: 1D signal values.
        t: 1D sampling times (monotone).
    Returns:
        (power, dt, scales): |coefficients|, sampling step, and scales used.
    """
    dt = np.median(np.diff(t)) if t.size > 1 else 1.0
    scales = cf / (freqs * dt)
    coef, _ = pywt.cwt(x, scales, WAVELET, sampling_period=dt)
    power = np.abs(coef)
    return power, dt, scales

def oscillation_score(power: np.ndarray, top_pct: float = 90.0) -> float:
    """
    Score oscillatory strength as the mean of the top `top_pct` percent of |CWT| values.
    Args:
        power: 2D |CWT| array (n_scales × n_time).
        top_pct: Percentile threshold for strong coefficients.
    Returns:
        Scalar oscillation score (float or NaN).
    """
    flat = power.ravel()
    if flat.size == 0:
        return np.nan
    thresh = np.percentile(flat, top_pct)
    strong = flat[flat >= thresh]
    return float(np.mean(strong)) if strong.size else float(np.mean(flat))

def plot_scalogram_single(t: np.ndarray, power: np.ndarray, title: str, outfile: str) -> None:
    """
    Render a single scalogram image.
    Args:
        t: 1D time array for the x-axis.
        power: 2D |CWT| array (scales × time).
        title: Figure title.
        outfile: Output PNG path under RESULTS_DIR.
    """
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    im = ax.imshow(
        power,
        extent=[t[0], t[-1], freqs[-1], freqs[0]],
        aspect="auto",
        cmap="inferno",
        origin="upper"
    )
    ax.set_xlabel("Step")
    ax.set_ylabel("Freq (1/step)")
    ax.set_title(title, fontsize=11)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("|CWT coefficients|")
    plt.tight_layout()
    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close(fig)

all_rows = []

for rho in RHO_LIST:
    rows = []
    panels_cache = {}

    for th in THETAS:
        for kmax in KMAXES:
            panel = coop.loc[
                (coop["theta"].eq(th))
                & (coop["Kmax"].eq(kmax))
                & (coop["rho_w"].eq(rho))
            ].sort_values("step")

            if panel.empty:
                rows.append({"rho_w": rho, "theta": th, "Kmax": kmax, "n_steps": 0, "score": np.nan})
                continue

            t_all = panel["step"].to_numpy()
            x_all = panel["median"].to_numpy()
            t = t_all[TRANSIENT:] if TRANSIENT > 0 else t_all
            x = x_all[TRANSIENT:] if TRANSIENT > 0 else x_all

            if x.size < 8:
                rows.append({"rho_w": rho, "theta": th, "Kmax": kmax, "n_steps": int(x.size), "score": np.nan})
                continue

            power, _, _ = compute_cwt_power(x, t)
            score = oscillation_score(power, SCORE_TOP_PCT)

            rows.append({"rho_w": rho, "theta": th, "Kmax": kmax, "n_steps": int(x.size), "score": score})
            panels_cache[(th, kmax)] = (t, power)

    df_scores = pd.DataFrame(rows).sort_values(["rho_w", "theta", "Kmax"]).reset_index(drop=True)
    per_rho_csv = os.path.join(RESULTS_DIR, f"cwt_scores_rho{str(rho).replace('.', 'p')}.csv")
    df_scores.to_csv(per_rho_csv, index=False)
    all_rows.append(df_scores)

    valid = df_scores.dropna(subset=["score"]).copy()
    if valid.empty:
        print(f"[WARN] No valid panels for rho_w={rho}")
        continue

    idx_max = valid["score"].idxmax()
    idx_min = valid["score"].idxmin()

    rmax = valid.loc[idx_max]
    rmin = valid.loc[idx_min]

    th_max, kmax_max = rmax["theta"], rmax["Kmax"]
    t_max, power_max = panels_cache[(th_max, kmax_max)]
    out_max = os.path.join(RESULTS_DIR, f"scalogram_max_rho{str(rho).replace('.', 'p')}.png")
    title_max = fr"Scalogram — $\rho_w={rho}$, $\theta={th_max}$, $K_{{\max}}={int(kmax_max)}$  (score={rmax['score']:.3g})"
    plot_scalogram_single(t_max, power_max, title_max, out_max)

    th_min, kmax_min = rmin["theta"], rmin["Kmax"]
    t_min, power_min = panels_cache[(th_min, kmax_min)]
    out_min = os.path.join(RESULTS_DIR, f"scalogram_min_rho{str(rho).replace('.', 'p')}.png")
    title_min = fr"Scalogram — $\rho_w={rho}$, $\theta={th_min}$, $K_{{\max}}={int(kmax_min)}$  (score={rmin['score']:.3g})"
    plot_scalogram_single(t_min, power_min, title_min, out_min)

    print(f"Saved: {out_max}")
    print(f"Saved: {out_min}")

df_all = pd.concat(all_rows, ignore_index=True)
out_csv_all = os.path.join(RESULTS_DIR, "cwt_scores_all_rho.csv")
df_all.to_csv(out_csv_all, index=False)

print("Done.")
print(f"- Combined table: {out_csv_all}")
print(f"- Per-rho tables and images in: {RESULTS_DIR}")
