import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt

# 1. Load and normalize
SWEEP_CSV = "water_toc_sweep_results.csv"
df = pd.read_csv(SWEEP_CSV).rename(columns={'Step':'step'})

# 2. Parameters
TAIL_FRAC   = 0.2      # use last 20% of the steps
VAR_THRESH  = 1e-3     # variance threshold for fixed point
ADF_ALPHA   = 0.05     # significance level for stationarity

records = []

# 3. Loop over each run to classify fixed points
for (run_id, itr), grp in df.groupby(['RunId','iteration']):
    grp = grp.sort_values('step')
    L = len(grp)
    tail = grp.iloc[int((1-TAIL_FRAC)*L):]
    
    coop_tail = tail['Coop_Fraction'].values
    env_tail  = tail['Environment_State'].values
    
    var_coop = np.var(coop_tail)
    adf_p    = adfuller(coop_tail, maxlag=1, regression='ct')[1]
    
    is_fixed = (var_coop < VAR_THRESH) and (adf_p < ADF_ALPHA)
    eq_coop  = coop_tail.mean()
    eq_env   = env_tail.mean()
    
    theta = grp['theta'].iat[0]
    cap   = grp['max_water_capacity'].iat[0]
    dens  = grp['water_cell_density'].iat[0]
    
    records.append({
        'theta': theta,
        'max_water_capacity': cap,
        'water_cell_density': dens,
        'is_fixed_point': is_fixed,
        'eq_coop': eq_coop,
        'eq_env': eq_env
    })

results = pd.DataFrame(records)

# 4. Aggregate per parameter combo
agg = (
    results
    .groupby(['theta','max_water_capacity','water_cell_density'])
    .agg(
        frac_fixed      = ('is_fixed_point','mean'),
        median_eq_coop  = ('eq_coop','median'),
        median_eq_env   = ('eq_env','median')
    )
    .reset_index()
)

# 5. Filter to combos where >50% of runs fixed
filtered = agg[agg['frac_fixed'] > 0.5].copy()

# 6. Format columns for readability
filtered['frac_fixed']     = (filtered['frac_fixed']*100).round(1).astype(str) + '%'
filtered['median_eq_coop'] = filtered['median_eq_coop'].round(3)
filtered['median_eq_env']  = filtered['median_eq_env'].round(3)

print("\n=== Fixed‐Point Regime (frac_fixed > 50%) ===")
print(filtered)

# 7. Render & save the filtered summary table as PNG
fig, ax = plt.subplots(figsize=(8, 0.4*len(filtered)+1))
ax.axis('off')

tbl = ax.table(
    cellText=filtered.values,
    colLabels=filtered.columns,
    cellLoc='center',
    loc='center'
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(8)
tbl.scale(1, 1.2)

plt.title("Fixed‐Point Attractors (>50% runs)", pad=20)
plt.tight_layout()
plt.savefig("fixed_point_summary_filtered.png", dpi=300, bbox_inches='tight')
print("Saved filtered table to fixed_point_summary_filtered.png")
