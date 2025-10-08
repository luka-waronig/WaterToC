"""
Classify fixed points in Coop_Fraction and Environment_State using tail variance + ADF,
aggregate by (theta, Kmax, rho_w), and render:
  1) A table of parameter combos with >50% fixed Coop_Fraction
  2) A scatter of median equilibrium (Env, Coop) colored by fraction fixed
To run the code first run sweep_run.py.
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from pathlib import Path

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def savefig(fname: str, **kwargs) -> None:
    """
    Save the current Matplotlib figure into RESULTS_DIR.
    Args:
        fname: Filename (e.g., 'figure.png').
        **kwargs: Matplotlib savefig keyword args (dpi, bbox_inches, etc.).
    """
    fpath = RESULTS_DIR / fname
    plt.savefig(fpath, **kwargs)
    print(f"Saved: {fpath}")

SWEEP_CSV = "water_toc_sweep_results.csv"
df = pd.read_csv(SWEEP_CSV).rename(columns={"Step": "step"})

TAIL_FRAC  = 0.20
VAR_THRESH = 1e-3
ADF_ALPHA  = 0.05

def safe_adf_p(x: np.ndarray) -> float:
    """
    Compute ADF p-value robustly, falling back on edge cases.
    Args:
        x: 1D array-like time series.
    Returns:
        p-value in [0,1]; returns 0.0 for constant/too-short series and 1.0 on failure.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 5 or np.allclose(x, x[0]):
        return 0.0
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return adfuller(x, maxlag=1, regression="ct")[1]
    except Exception:
        return 1.0

records = []
for (run_id, itr), grp in df.groupby(["RunId", "iteration"]):
    grp = grp.sort_values("step")
    L = len(grp)
    tail = grp.iloc[int((1 - TAIL_FRAC) * L):]

    coop_tail = tail["Coop_Fraction"].to_numpy()
    env_tail  = tail["Environment_State"].to_numpy()

    var_coop = float(np.var(coop_tail))
    var_env  = float(np.var(env_tail))

    p_coop = safe_adf_p(coop_tail)
    p_env  = safe_adf_p(env_tail)

    is_fixed_coop = (var_coop < VAR_THRESH) and (p_coop < ADF_ALPHA)
    is_fixed_env  = (var_env  < VAR_THRESH) and (p_env  < ADF_ALPHA)

    records.append({
        "theta": grp["theta"].iat[0],
        "max_water_capacity": grp["max_water_capacity"].iat[0],
        "water_cell_density": grp["water_cell_density"].iat[0],
        "eq_coop": float(coop_tail.mean()),
        "eq_env":  float(env_tail.mean()),
        "is_fixed_coop": is_fixed_coop,
        "is_fixed_env":  is_fixed_env,
    })

results = pd.DataFrame(records)

agg = (
    results
    .groupby(["theta", "max_water_capacity", "water_cell_density"], as_index=False)
    .agg(
        frac_fixed_env=("is_fixed_env", "mean"),
        frac_fixed_coop=("is_fixed_coop", "mean"),
        median_eq_env=("eq_env", "median"),
        median_eq_coop=("eq_coop", "median"),
    )
)

def fmt_table(t: pd.DataFrame) -> pd.DataFrame:
    """
    Format numeric columns for display.
    Args:
        t: Aggregated dataframe.
    Returns:
        Copy with formatted frac_fixed_coop (%) and medians rounded.
    """
    out = t.copy()
    out["frac_fixed_coop"] = (out["frac_fixed_coop"]*100).round(1)
    out["median_eq_env"]   = out["median_eq_env"].round(3)
    out["median_eq_coop"]  = out["median_eq_coop"].round(3)
    return out

def save_table_png(df_in: pd.DataFrame, cols: list[str], title: str, fname: str) -> None:
    """
    Render a simple table as a PNG.
    Args:
        df_in: Dataframe to render.
        cols: Column order to display.
        title: Figure title.
        fname: Output filename under RESULTS_DIR.
    """
    if df_in.empty:
        print("No parameter combos with >50% fixed (Coop_Fraction).")
        return
    show = fmt_table(df_in)[cols]
    fig, ax = plt.subplots(figsize=(8, 0.5*len(show)+1))
    ax.axis("off")
    tbl = ax.table(cellText=show.values, colLabels=show.columns,
                   cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.2)
    plt.title(title, pad=16)
    plt.tight_layout()
    savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()

coop_tbl = agg[agg["frac_fixed_coop"] > 0.5].copy()
save_table_png(
    coop_tbl,
    ["theta","max_water_capacity","water_cell_density",
     "frac_fixed_coop","median_eq_env","median_eq_coop"],
    "Parameter Combos with >50% Fixed (Coop_Fraction)",
    "table_fixed_coop_over50.png",
)

def scatter_env_vs_coop(df_in: pd.DataFrame, title: str, fname: str,
                        cbar_label: str="Fraction of runs fixed", vmax: float=0.7) -> None:
    """
    Scatter of median equilibrium (Env vs Coop) colored by fraction of fixed runs.
    Args:
        df_in: Aggregated dataframe with 'median_eq_env', 'median_eq_coop', 'frac_fixed_coop'.
        title: Plot title.
        fname: Output filename under RESULTS_DIR.
        cbar_label: Colorbar label text.
        vmax: Upper bound for color normalization.
    """
    x = df_in["median_eq_env"].to_numpy()
    y = df_in["median_eq_coop"].to_numpy()
    c = df_in["frac_fixed_coop"].to_numpy()

    norm = Normalize(vmin=0.0, vmax=vmax)
    fig, ax = plt.subplots(figsize=(7.5, 6))
    sc = ax.scatter(x, y, c=c, s=45, norm=norm)

    sm = ScalarMappable(norm=norm, cmap=sc.cmap)
    sm.set_array([])
    cb = plt.colorbar(sm, ax=ax)
    cb.set_label(cbar_label)
    cb.set_ticks(np.linspace(0.0, vmax, 8))

    ax.set_xlabel("Median Equilibrium Environment_State")
    ax.set_ylabel("Median Equilibrium Coop_Fraction")
    ax.set_title(title)
    plt.tight_layout()
    savefig(fname, dpi=300)
    plt.close()

scatter_env_vs_coop(
    agg,
    "Fixed-Point Attractors",
    "scatter_fraction_fixed_coop.png",
)

print("\nSaved:")
print(RESULTS_DIR / "table_fixed_coop_over50.png")
print(RESULTS_DIR / "scatter_fraction_fixed_coop.png")
