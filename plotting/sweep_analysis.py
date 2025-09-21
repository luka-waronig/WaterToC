# sweep_analysis_new.py
import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from mesa.batchrunner import batch_run
import pandas as pd
import numpy as np
from water_toc.model import WaterToC

def main():
    # define sweep ranges
    variable_params = {
        "theta": [0.5, 2.0, 5.0, 10],
        "max_water_capacity": [10, 20, 30, 40],
        "water_cell_density": [0.2, 0.3, 0.4, 0.5],
    }

    fixed_params = {
        "height": 20,
        "width": 20,
        "initial_humans": 50,
        "initial_ai": 50,
        "human_C_allocation": 0.1,
        "human_D_allocation": 0.15,
        "ai_C_allocation": 2.0,
        "ai_D_allocation": 3.0,
        "deviation_rate": 0.1
    }

    # run simulations
    all_data = batch_run(
        WaterToC,
        parameters={**fixed_params, **variable_params},
        iterations=100,
        max_steps=100,
        data_collection_period=1, 
        number_processes=1,       
        display_progress=True
    )

    
    df = pd.DataFrame(all_data)
    df.to_csv("water_toc_sweep_results.csv", index=False)

if __name__ == "__main__":
    main()
