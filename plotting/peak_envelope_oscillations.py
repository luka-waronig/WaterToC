import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.optimize import curve_fit

SWEEP_CSV = "water_toc_sweep_results.csv"
df = pd.read_csv(SWEEP_CSV)

results = []

#setting parameters
TRANSIENT_FRAC   = 0.10   
MIN_POINTS_AFTER = 50     # we need at least this many points after the cut
PEAK_DISTANCE    = 5      # min steps between successive peaks
PEAK_PROMINENCE  = 0.005  # min peak prominence

def envelope(t, A0, lam):
    return A0 * np.exp(-lam * t)

for (run_id, itr), sub in df.groupby(['RunId','iteration']):
    sub = sub.sort_values('Step')
    time_arr = sub['Step'].to_numpy()
    C_arr    = sub['Coop_Fraction'].to_numpy()
    total    = len(time_arr)
    
    # adaptive transient cut
    N0 = int(np.floor(total * TRANSIENT_FRAC))
    if total - N0 < MIN_POINTS_AFTER:
        N0 = total - MIN_POINTS_AFTER
    t = time_arr[N0:]
    C = C_arr[N0:]
    
    # skip if still too short
    if len(C) < MIN_POINTS_AFTER:
        continue
    
    # find peaks
    peaks, _ = find_peaks(
        C,
        distance=PEAK_DISTANCE,
        prominence=PEAK_PROMINENCE
    )
    if len(peaks) < 2:
        # no clear oscillation
        continue
    
    # compute mean period
    peak_times = t[peaks]
    intervals  = np.diff(peak_times)
    T_est = intervals.mean()
    
    # fit exponential envelope
    t0 = peak_times - peak_times[0]
    try:
        popt, _ = curve_fit(
            envelope, 
            t0, 
            C[peaks], 
            p0=(C[peaks][0], 0.01),
            maxfev=5000
        )
        A0, lam = popt
    except Exception:
        lam = np.nan
    
    results.append({
        'RunId': run_id,
        'iteration': itr,
        'period_est': T_est,
        'damping_rate': lam
    })

# assemble results
res_df = pd.DataFrame(results)

# quick summary
print(res_df.describe())

# plot histogram of estimated periods
plt.figure(figsize=(4,2.5))
plt.hist(res_df['period_est'].dropna(), bins=20, density=True, alpha=0.7)
plt.xlabel('Estimated Period (steps)')
plt.ylabel('Density')
plt.title('Distribution of Damped Oscillation Periods')
plt.tight_layout()
plt.show()

# plot histogram of damping rates
plt.figure(figsize=(4,2.5))
plt.hist(res_df['damping_rate'].dropna(), bins=20, density=True, alpha=0.7)
plt.xlabel('Estimated Damping Rate λ')
plt.ylabel('Density')
plt.title('Distribution of Damping Rates')
plt.tight_layout()
plt.show()
