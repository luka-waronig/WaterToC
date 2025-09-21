import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pywt
 
#settings
CSV_FILE   = "water_toc_sweep_results.csv"
THETAS     = [2.0, 5.0, 10.0]
CAPACITIES = [20, 40]
DENSITIES  = [0.2, 0.4]
TRANSIENT  = 10        
MIN_PERIOD = 5
MAX_PERIOD = 100
WAVELET    = 'morl'
N_FREQS    = 200

df = pd.read_csv(CSV_FILE)

#pre-compute time and scales
example = df[
    (df.theta==THETAS[0]) &
    (df.max_water_capacity==CAPACITIES[0]) &
    (df.water_cell_density==DENSITIES[0]) &
    (df.iteration==0)
].sort_values('Step')
time_all = example['Step'].to_numpy()
t = time_all[TRANSIENT:]
dt = t[1] - t[0]
cf = pywt.central_frequency(WAVELET)
freqs = np.linspace(1/MAX_PERIOD, 1/MIN_PERIOD, N_FREQS)
scales = cf / (freqs * dt)

for rho in DENSITIES:
    fig, axes = plt.subplots(
        nrows=len(THETAS),
        ncols=len(CAPACITIES),
        figsize=(10, 6),
        sharex=True,
        sharey=True,
        constrained_layout=False
    )

    #plot each panel
    for i, theta in enumerate(THETAS):
        for j, cap in enumerate(CAPACITIES):
            sub = df[(df.theta==theta) &
                     (df.max_water_capacity==cap) &
                     (df.water_cell_density==rho)]
            all_coefs = []
            for itr in sorted(sub.iteration.unique()):
                run_df = sub[sub.iteration==itr].sort_values('Step')
                C_tail = run_df['Coop_Fraction'].to_numpy()[TRANSIENT:]
                coef, _ = pywt.cwt(C_tail, scales, WAVELET, sampling_period=dt)
                all_coefs.append(np.abs(coef))
            median_coef = np.median(np.stack(all_coefs,axis=0), axis=0)

            ax = axes[i,j]
            im = ax.imshow(
                median_coef,
                extent=[t[0], t[-1], freqs[-1], freqs[0]],
                aspect='auto',
                cmap='inferno'
            )
            if i == len(THETAS)-1:
                ax.set_xlabel("Step")
            if j == 0:
                ax.set_ylabel("Freq (1/step)")
            ax.set_title(f"θ={theta}, cap={cap}")

    #plotting
    fig.subplots_adjust(
        left=0.08,  
        right=0.88,
        top=0.88,   
        bottom=0.08
    )

    
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("Median |CWT coefficient|")

    fig.suptitle(f"Median Morlet CWT (density = {rho})", fontsize=14)

    plt.show()
