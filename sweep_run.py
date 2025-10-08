
import itertools
import pandas as pd
from datetime import datetime
import uuid
import hashlib

from water_toc.model import WaterToC

"""
Run a full parameter sweep of the WaterToC model, collecting per-step time series.
For each (theta, Kmax, rho_w) across N_RUNS seeds:
- simulate N_STEPS steps,
- extract required metrics from the model DataCollector,
- attach sweep metadata, and
- write all runs to a single CSV (OUTCSV).
"""

THETAS    = [1.4, 2.0, 5.0, 10]
KMAXES    = [10, 20, 30, 40]
DENSITIES = [0.2, 0.4, 0.5, 0.7]

N_RUNS    = 100
N_STEPS   = 100
OUTCSV    = "water_toc_sweep_results.csv"

BASE_SEED = 42
HEIGHT, WIDTH = 20, 20
INITIAL_HUMANS = 50
INITIAL_AI     = 50

META_COLS = ["RunId","iteration","theta","max_water_capacity","water_cell_density","seed","Step"]
METRIC_COLS = [
    "Coop_Fraction",
    "AI_Coop_Fraction",
    "Human_Coop_Fraction",
    "Environment_State",
    "Total_Water",
    "Avg_Water_Per_Cell",
]

def deterministic_seed(*args) -> int:
    """
    Create a deterministic integer seed from input arguments.
    Args:
        *args: Values (hashable) that define the seed (e.g., theta, Kmax, rho_w, run index).
    Returns:
        Integer seed derived from BASE_SEED and a SHA1 hash of args.
    """
    s = "_".join(map(str, args))
    return BASE_SEED + int(hashlib.sha1(s.encode()).hexdigest(), 16) % 10_000_000

def run_single_timeseries(theta, kmax, rho_w, run_idx):
    """
    Execute one simulation run and return a per-step time-series DataFrame.
    Args:
        theta (float): Feedback strength parameter.
        kmax (int): Maximum water capacity.
        rho_w (float): Water cell density.
        run_idx (int): Index of the run (0-based).
    Returns:
        pandas.DataFrame: Columns META_COLS + METRIC_COLS for all steps of this run.
    Raises:
        ValueError: If required metric columns are missing from the DataCollector.
    """
    seed = deterministic_seed(theta, kmax, rho_w, run_idx)
    model = WaterToC(
        height=HEIGHT, width=WIDTH,
        initial_humans=INITIAL_HUMANS, initial_ai=INITIAL_AI,
        max_water_capacity=kmax, water_cell_density=rho_w,
        theta=theta, deviation_rate=0.1,
        save_snapshots=False, snapshot_interval=99999,
        seed=seed
    )
    for _ in range(N_STEPS):
        model.step()
    df = model.datacollector.get_model_vars_dataframe().copy()
    if "Step" not in df.columns:
        try:
            df = df.reset_index(names="Step")
        except TypeError:
            df = df.reset_index().rename(columns={"index": "Step"})
    else:
        df = df.reset_index(drop=True)
    df["RunId"]              = str(uuid.uuid4())
    df["iteration"]          = run_idx + 1
    df["theta"]              = theta
    df["max_water_capacity"] = kmax
    df["water_cell_density"] = rho_w
    df["seed"]               = seed
    missing = [c for c in METRIC_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Expected metrics missing from DataCollector: {missing}")
    df = df[META_COLS + METRIC_COLS]
    return df

def main():
    """
    Enumerate all parameter combinations, run N_RUNS replicates each,
    concatenate per-step time series, and save to OUTCSV.
    Outputs:
        - OUTCSV (str): CSV with META_COLS + METRIC_COLS for all runs and steps.
    """
    combos = list(itertools.product(THETAS, KMAXES, DENSITIES))
    total_jobs = len(combos) * N_RUNS
    print(f"Starting sweep: {len(combos)} combos × {N_RUNS} runs = {total_jobs} sims")
    parts = []
    job = 0
    for theta, kmax, rho_w in combos:
        for r in range(N_RUNS):
            job += 1
            if job % 10 == 0 or job == 1:
                ts = datetime.now().strftime("%H:%M:%S")
                print(f"[{ts}] {job}/{total_jobs} | θ={theta}, Kmax={kmax}, ρw={rho_w}, run={r+1}")
            parts.append(run_single_timeseries(theta, kmax, rho_w, r))
    out = pd.concat(parts, ignore_index=True)
    out.to_csv(OUTCSV, index=False)
    print(f"Saved {len(out)} rows to {OUTCSV}")

if __name__ == "__main__":
    main()
