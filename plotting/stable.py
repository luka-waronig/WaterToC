import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.stattools import adfuller

df_raw = pd.read_csv("water_toc_sweep_results.csv")

# compute stable-state summary 
TAIL_FRAC, VAR_THRESH, ADF_ALPHA = 0.2, 1e-3, 0.05
records = []
for (theta, cap, dens, run_id, itr), sub in df_raw.groupby(
    ['theta','max_water_capacity','water_cell_density','RunId','iteration']
):
    sub = sub.sort_values('Step')
    tail = sub.iloc[int(len(sub)*TAIL_FRAC):]
    
    if len(tail) > 4: 
        
        
        C_tail = tail['Coop_Fraction'].to_numpy()
        E_tail = tail['Environment_State'].to_numpy()
        
        stable_C, stable_E = C_tail.mean(), E_tail.mean()
        
        p_adf = adfuller(C_tail, maxlag=1, regression='ct')[1]
        is_fixed = (np.var(C_tail)<VAR_THRESH) and (p_adf<ADF_ALPHA)
        
        
        records.append({
            'theta':theta, 'max_water_capacity':cap, 'water_cell_density':dens,
            'median_C':stable_C,'median_E':stable_E,'frac_fixed':is_fixed
        })


summary = (
    pd.DataFrame(records)
      .groupby(['theta','max_water_capacity','water_cell_density'])
      .agg(median_C=('median_C','median'),
           median_E=('median_E','median'),
           frac_fixed=('frac_fixed','mean'))
      .reset_index()
)


for val, cmap, title in [
    ('median_C','viridis','Stable Cooperation (C*)'),
    ('median_E','magma','Stable Environment (E*)')
]:
    g = sns.FacetGrid(
        summary,
        col="theta",
        col_wrap=2,
        height=4.5,
    )
    g.map_dataframe(
        lambda data, color: sns.heatmap(
            data.pivot(
                index='water_cell_density',
                columns='max_water_capacity',
                values=val
            ),
            annot=True, fmt=".2f", cmap=cmap,
            cbar_kws={'label': val}
        )
    )
    g.set_titles(col_template="theta = {col_name}")
    for ax in g.axes.flat:
        ax.set_xlabel("Max Water Capacity")
        ax.set_ylabel("Water Cell Density")
    g.fig.suptitle(title, fontsize=16)
    plt.subplots_adjust(top=0.9)
    plt.show()