import os
import numpy as np
import pandas as pd
from typing import Tuple, Dict, List
from water_toc.model import WaterToC  

"""
Run a parameter sweep of the WaterToC model and compute tail-median spatial statistics.
For each (theta, Kmax, rho_w) and seed:
- simulate STEPS number of steps,
- aggregate tail medians for cooperation/clustering/Moran's I,
- compute torus-based Ripley K/L and pair-correlation g for cooperator points,
- save per-parameter summaries and r-binned pair-process medians across runs.
"""

THETAS    = [1.4, 2.0, 5.0, 10.0]
KMAXES    = [10, 20, 30, 40]
DENSITIES = [0.2, 0.4, 0.5, 0.7]

STEPS      = 100
TAIL_FRAC  = 0.25
HEIGHT, WIDTH = 20, 20

REPS       = 100
BASE_SEED  = 42

OUTDIR     = "sweep_outputs"
os.makedirs(OUTDIR, exist_ok=True)

R_BINS = np.arange(1.0, 8.5, 0.5)

def _roll_sum_4(arr: np.ndarray, periodic: bool) -> np.ndarray:
    """
    Sum of 4-neighborhood (von Neumann) values.
    Args:
        arr: 2D array.
        periodic: If True, wrap on both axes (torus); else use zero-flux boundaries.
    Returns:
        2D array of neighbor sums with same shape as arr.
    """
    if periodic:
        return (np.roll(arr, 1, 0) + np.roll(arr, -1, 0) +
                np.roll(arr, 1, 1) + np.roll(arr, -1, 1))
    s = np.zeros_like(arr, dtype=arr.dtype)
    s[1:, :] += arr[:-1, :]
    s[:-1, :] += arr[1:, :]
    s[:, 1:]  += arr[:, :-1]
    s[:, :-1] += arr[:, 1:]
    return s

def _roll_sum_8(arr: np.ndarray, periodic: bool) -> np.ndarray:
    """
    Sum of 8-neighborhood (Moore) values.
    Args:
        arr: 2D array.
        periodic: If True, wrap on both axes (torus); else use zero-flux boundaries for diagonals too.
    Returns:
        2D array of neighbor sums with same shape as arr.
    """
    if periodic:
        return (_roll_sum_4(arr, True) +
                np.roll(np.roll(arr, 1, 0),  1, 1) +
                np.roll(np.roll(arr, 1, 0), -1, 1) +
                np.roll(np.roll(arr, -1, 0), 1, 1) +
                np.roll(np.roll(arr, -1, 0),-1, 1))
    s = _roll_sum_4(arr, False)
    H, W = arr.shape
    for dx, dy in [(-1,-1), (-1,1), (1,-1), (1,1)]:
        xs = slice(max(0, dx), H+min(0, dx))
        ys = slice(max(0, dy), W+min(0, dy))
        s[xs, ys] += arr[xs.start-dx:xs.stop-dx, ys.start-dy:ys.stop-dy]
    return s

def degree_map_moore(shape):
    """
    Degree map under Moore neighborhood on a torus (all interior cells have degree 8).
    Args:
        shape: (H, W) grid shape.
    Returns:
        2D integer array of degrees for each cell assuming periodic boundaries.
    """
    ones = np.ones(shape, dtype=int)
    return _roll_sum_8(ones, True)

DEGREE_MAP = degree_map_moore((HEIGHT, WIDTH))

def morans_I_masked(field: np.ndarray, mask: np.ndarray, degree_map: np.ndarray) -> float:
    """
    Moran's I on a masked fraction field with Moore neighborhood on a torus.
    Args:
        field: 2D numeric array (e.g., cooperation fraction per cell).
        mask:  2D boolean array; True where cells are included.
        degree_map: 2D integer degrees per cell under Moore+torus.
    Returns:
        Moran's I (float) or NaN if undefined.
    """
    x = np.asarray(field, float)
    m = np.asarray(mask, bool)
    n_eff = int(m.sum())
    if n_eff < 2:
        return np.nan
    mu = x[m].mean()
    xc = np.zeros_like(x, dtype=float)
    xc[m] = x[m] - mu
    nb = _roll_sum_8(xc, periodic=True)
    nb[~m] = 0.0
    W_sum = float(degree_map[m].sum())
    den   = float((xc[m] * xc[m]).sum())
    num   = float((xc[m] * nb[m]).sum())
    return (n_eff / W_sum) * (num / den) if (W_sum > 0 and den > 0) else np.nan

_NEIGH_OFFSETS_MOORE = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]

def _cluster_labels_periodic(binary: np.ndarray) -> np.ndarray:
    """
    Connected-component labels on a torus using Moore neighborhood.
    Args:
        binary: 2D boolean array indicating occupied cells.
    Returns:
        2D int32 label map (0 = background).
    """
    H, W = binary.shape
    labels = np.zeros((H, W), dtype=np.int32)
    cur = 0
    for i in range(H):
        for j in range(W):
            if binary[i, j] and labels[i, j] == 0:
                cur += 1
                stack = [(i, j)]
                labels[i, j] = cur
                while stack:
                    y, x = stack.pop()
                    for dy, dx in _NEIGH_OFFSETS_MOORE:
                        yy = (y + dy) % H
                        xx = (x + dx) % W
                        if binary[yy, xx] and labels[yy, xx] == 0:
                            labels[yy, xx] = cur
                            stack.append((yy, xx))
    return labels

def agent_cluster_metrics_agents_only(agents, height: int, width: int) -> Tuple[int, int, float]:
    """
    Cluster cooperative agents (>=2 agents per component) on a torus with Moore neighborhood.
    Args:
        agents: Iterable of agent tuples; expects (.., x, y, strategy) with strategy starting by 'C' for cooperators.
        height: Grid height.
        width:  Grid width.
    Returns:
        (n_clusters_agents, largest_cluster_size_in_agents, mean_cluster_size_in_agents).
    """
    coop_counts = np.zeros((height, width), dtype=int)
    if isinstance(agents, (list, tuple)):
        for a in agents:
            if len(a) >= 5 and str(a[4]).upper().startswith("C"):
                x, y = int(a[2]), int(a[3])
                coop_counts[x, y] += 1
    mask = (coop_counts > 0)
    if not np.any(mask):
        return 0, 0, 0.0
    labels = _cluster_labels_periodic(mask)
    sizes_agents = []
    for lbl in np.unique(labels):
        if lbl == 0:
            continue
        sz = int(coop_counts[labels == lbl].sum())
        if sz >= 2:
            sizes_agents.append(sz)
    if len(sizes_agents) == 0:
        return 0, 0, 0.0
    sizes_agents = np.asarray(sizes_agents, dtype=float)
    return int(len(sizes_agents)), int(sizes_agents.max()), float(sizes_agents.mean())

def _torus_delta(dx, L):
    """
    Shortest wrapped distance along one axis on a torus.
    Args:
        dx: Pairwise differences.
        L:  Period length (axis size).
    Returns:
        Elementwise minimal wrapped distances.
    """
    dx = np.abs(dx)
    return np.minimum(dx, L - dx)

def pair_stats_for_points(points: np.ndarray, Lx: float, Ly: float, r_bins: np.ndarray):
    """
    Ripley K, L, and pair-correlation g for points on a rectangular torus.
    Args:
        points: Nx2 array of (x, y) coordinates.
        Lx:     Width (period in x).
        Ly:     Height (period in y).
        r_bins: Radii centers for estimation (uniformly spaced).
    Returns:
        (K, L, g) arrays at r_bins; NaNs if fewer than 2 points.
    """
    n = len(points)
    A = Lx * Ly
    if n < 2:
        fill = np.full_like(r_bins, np.nan, dtype=float)
        return fill, fill, fill
    dx = points[:, None, 0] - points[None, :, 0]
    dy = points[:, None, 1] - points[None, :, 1]
    dx = _torus_delta(dx, Lx)
    dy = _torus_delta(dy, Ly)
    dist = np.sqrt(dx*dx + dy*dy)
    iu = np.triu_indices(n, k=1)
    d = dist[iu]
    lam = n / A
    K = np.array([(A / (n*(n-1))) * np.sum(d <= r) * 2 for r in r_bins], dtype=float)
    L = np.sqrt(K / np.pi)
    dr = np.diff(r_bins).mean()
    edges = np.concatenate(([r_bins[0] - dr/2], r_bins + dr/2))
    hist, _ = np.histogram(d, bins=edges)
    shell_area = 2 * np.pi * r_bins * dr
    expected = lam * shell_area * n
    g = hist / expected
    return K, L, g

def run_one(theta, kmax, density, seed):
    """
    Run one simulation and aggregate tail-median spatial statistics.
    Args:
        theta: Feedback strength parameter.
        kmax:  Max water capacity.
        density: Water cell density (rho_w).
        seed:  RNG seed for model.
    Returns:
        Dict with tail-median metrics and per-step-averaged K/L/g arrays.
    """
    m = WaterToC(height=HEIGHT, width=WIDTH,
                 max_water_capacity=kmax,
                 water_cell_density=density,
                 theta=theta, seed=seed, save_snapshots=False)
    for _ in range(STEPS):
        m.step()
    df = m.datacollector.get_model_vars_dataframe().reset_index(drop=True)
    tail_start = int((1 - TAIL_FRAC) * len(df))
    tail = df.iloc[tail_start:]
    K_acc = np.zeros_like(R_BINS, dtype=float)
    L_acc = np.zeros_like(R_BINS, dtype=float)
    g_acc = np.zeros_like(R_BINS, dtype=float)
    pair_steps = 0
    I_occ_vals   = []
    I_wocc_vals  = []
    cl_cnt_agents, cl_lrg_agents, cl_mean_agents = [], [], []
    coop_fracs = []
    for row in tail.itertuples(index=False):
        flat = row.Coop_Map_Flat
        agents = row.Agent_Pos_Strats
        arr = np.array(flat, dtype=float).reshape(HEIGHT, WIDTH)
        coop_counts = np.zeros((HEIGHT, WIDTH), dtype=int)
        occ = np.zeros((HEIGHT, WIDTH), dtype=bool)
        coords = []
        if isinstance(agents, (list, tuple)) and len(agents) > 0:
            for a in agents:
                if len(a) >= 5:
                    x, y = int(a[2]), int(a[3])
                    occ[x, y] = True
                    if str(a[4]).upper().startswith("C"):
                        coop_counts[x, y] += 1
                        coords.append((float(x), float(y)))
        I_occ_vals.append(morans_I_masked(arr, occ, DEGREE_MAP))
        I_wocc_vals.append(morans_I_masked(arr, occ, DEGREE_MAP))
        n_comp_agents, largest_agents, mean_agents = agent_cluster_metrics_agents_only(
            agents, HEIGHT, WIDTH
        )
        cl_cnt_agents.append(n_comp_agents)
        cl_lrg_agents.append(largest_agents)
        cl_mean_agents.append(mean_agents)
        if len(coords) >= 2:
            pts = np.array(coords, dtype=float)
            K, L, g = pair_stats_for_points(pts, WIDTH, HEIGHT, R_BINS)
            K_acc += K; L_acc += L; g_acc += g; pair_steps += 1
        coop_fracs.append(row.Coop_Fraction)
    K_avg = (K_acc / pair_steps) if pair_steps > 0 else np.full_like(R_BINS, np.nan, dtype=float)
    L_avg = (L_acc / pair_steps) if pair_steps > 0 else np.full_like(R_BINS, np.nan, dtype=float)
    g_avg = (g_acc / pair_steps) if pair_steps > 0 else np.full_like(R_BINS, np.nan, dtype=float)
    return dict(
        CoopFraction_tailmedian=float(np.median(coop_fracs)),
        MoransI_occ_tailmedian=float(np.nanmedian(I_occ_vals)),
        MoransI_wocc_tailmedian=float(np.nanmedian(I_wocc_vals)),
        ClustersAgents_tailmedian=float(np.median(cl_cnt_agents)),
        LargestClusterAgents_tailmedian=float(np.median(cl_lrg_agents)),
        MeanClusterAgents_tailmedian=float(np.median(cl_mean_agents)),
        RipleyK=K_avg, RipleyL=L_avg, PairCorr_g=g_avg
    )

def main():
    """
    Sweep all parameter triples, run REPS seeds each, aggregate medians, and save CSVs.
    Saves:
        - summary_agents_clusters_and_pointprocess_MEDIANS.csv
        - pair_stats_ripley_MEDIANS.csv
    """
    summary_rows: List[Dict] = []
    pair_rows: List[Dict]    = []
    for th in THETAS:
        for k in KMAXES:
            for rho in DENSITIES:
                coop_vals = []
                I_occ_vals = []
                I_wocc_vals = []
                clusters_agents_vals = []
                largest_agents_vals  = []
                mean_agents_vals     = []
                K_stack, L_stack, g_stack = [], [], []
                for r in range(REPS):
                    seed = BASE_SEED + (hash((th, k, rho, r)) % 1_000_000_007)
                    res = run_one(th, k, rho, seed)
                    coop_vals.append(res["CoopFraction_tailmedian"])
                    I_occ_vals.append(res["MoransI_occ_tailmedian"])
                    I_wocc_vals.append(res["MoransI_wocc_tailmedian"])
                    clusters_agents_vals.append(res["ClustersAgents_tailmedian"])
                    largest_agents_vals.append(res["LargestClusterAgents_tailmedian"])
                    mean_agents_vals.append(res["MeanClusterAgents_tailmedian"])
                    if not np.all(np.isnan(res["PairCorr_g"])):
                        K_stack.append(res["RipleyK"])
                        L_stack.append(res["RipleyL"])
                        g_stack.append(res["PairCorr_g"])
                if len(K_stack) > 0:
                    K_med = np.median(np.vstack(K_stack), axis=0).tolist()
                    L_med = np.median(np.vstack(L_stack), axis=0).tolist()
                    g_med = np.median(np.vstack(g_stack), axis=0).tolist()
                else:
                    K_med = [np.nan]*len(R_BINS)
                    L_med = [np.nan]*len(R_BINS)
                    g_med = [np.nan]*len(R_BINS)
                summary_rows.append({
                    "theta": th, "Kmax": k, "rho_w": rho,
                    "CoopFraction_tailmedian_overruns": float(np.median(coop_vals)),
                    "MoransI_occ_tailmedian_overruns": float(np.nanmedian(I_occ_vals)),
                    "MoransI_wocc_tailmedian_overruns": float(np.nanmedian(I_wocc_vals)),
                    "ClustersAgents_tailmedian_overruns": float(np.median(clusters_agents_vals)),
                    "LargestClusterAgents_tailmedian_overruns": float(np.median(largest_agents_vals)),
                    "MeanClusterAgents_tailmedian_overruns": float(np.median(mean_agents_vals)),
                })
                for rmid, K, L, g in zip(R_BINS, K_med, L_med, g_med):
                    pair_rows.append({
                        "theta": th, "Kmax": k, "rho_w": rho,
                        "r": float(rmid), "RipleyK_median": K, "RipleyL_median": L, "g_pair_median": g
                    })
                print(f"θ={th:>4}, K={k:>2}, ρw={rho:.1f} | "
                      f"Coop~tail-med={np.median(coop_vals):.3f} | "
                      f"I_occ~tail-med={np.nanmedian(I_occ_vals):.3f} | "
                      f"Clusters(agents,≥2)~tail-med={np.median(clusters_agents_vals):.2f} | "
                      f"LargestAgents~tail-med={np.median(largest_agents_vals):.1f} | "
                      f"MeanClusterAgents~tail-med={np.median(mean_agents_vals):.2f} | "
                      f"K/L/g runs={len(K_stack)}")
    df_summary = pd.DataFrame(summary_rows)
    df_pairs   = pd.DataFrame(pair_rows)
    out_summary = os.path.join(OUTDIR, "summary_agents_clusters_and_pointprocess_MEDIANS.csv")
    out_pairs   = os.path.join(OUTDIR, "pair_stats_ripley_MEDIANS.csv")
    df_summary.to_csv(out_summary, index=False)
    df_pairs.to_csv(out_pairs, index=False)
    print(f"Saved: {out_summary}")
    print(f"Saved: {out_pairs}")

if __name__ == "__main__":
    main()
