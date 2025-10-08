# Agent-Based Model of Tragedy of the Commons with Environmental Feedback
## (paper title: Coevolutionary Dynamics of Cooperation and Environmental Sustainability in Shared Resource Systems)
We created an interactive agent-based model simulating consumption of a shared, renewable resource using a game-theoretic framework with environmental feedback. Although its original use was to simulate a ToC scenario with water as the shared resource, it can be applicable for a variety of scenarios including simulating climate disasters, environmental sensitivity to resource consumption, or influence of environmental degradation to agent behaviour.
The primary goal of the model is to explore the socio-environmental feedback loops that lead to complex system dynamics like limit cycles or fixed-point stability. It was inspired by the Demographic Prisoner’s Dilemma on a Grid (https://mesa.readthedocs.io/stable/examples/advanced/pd_grid.html#demographic-prisoner-s-dilemma-on-a-grid). The main innovation of this model is the added environmental feedback interpolating the expected payoffs (as introduced by the oscillating ToC dynamical system studied by Weitz et al 2016), but in a stochastic heterogeneous agent-based setting with with local resource consumption and replenishment.

The model is built using Mesa 1.2.1 for the model and Solara for the interactive web-based dashboard.
While Mesa version 3.0 was available at the time of this project's finalization, version 1.2.1 was used to ensure functional correctness and maintain compatibility. Initial testing with Mesa 3.0 revealed significant, non-backward-compatible API changes relative to the 1.x series, which would have required a substantial rewrite of the existing, validated codebase. Therefore, to guarantee the stability and reproducibility of the results based on the original model implementation, version 1.2.1 was retained as the foundational dependency for this research.

## Features

* **Interactive Dashboard:** Allows users to adjust model parameters in real-time using the Solara web interface. The interface visualizes several important plots for behaviour analysis such as cooperation fractions, detects limit cycles if present, and provides important statistical summaries. It also provides users with a visualization of the grid on the last timestep.
* **Complex Agent Behavior:** Agents' strategies are determined by a game-theoretic model influenced by current environmental states.
* **Advanced Analysis:** Besides the Solara interface, the project includes scripts for generating various visualizations, including heatmaps, 3D plots, and limit cycle diagrams from simulation data.


## Key Model Parameters

The interactive dashboard allows for the adjustment of several key parameters:

| Parameter | Description |
| :--- | :--- |
| **`theta` (${\theta}$)** | The sensitivity of the environment to cooperation. Higher values mean cooperation has a stronger positive effect on resource replenishment. |
| **`deviation_rate`** | The probability (0.0 to 1.0) that an agent will act irrationally, choosing the opposite of its calculated best move. |
| **`initial_humans` / `initial_ai`** | The starting number of each agent type. |
| **`*_C_allocation`** | The amount of resource an agent consumes when it **C**ooperates. |
| **`*_D_allocation`** | The amount of resource an agent consumes when it **D**efects. |
| **`max_resource_capacity`** | The maximum amount of resource any single grid cell can hold. |
| **`resource_density`** | The fraction of grid cells that contain the resource. |

## Installation

1.  **Clone the repository:**
    ```bash
    git clone (https://github.com/slavica01/WaterToC)
    cd WaterToC
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  #or on Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required libraries:**
    ```bash
    pip install -r requirements.txt
    ```


## Usage

### Interactive Dashboard

To launch the main interactive dashboard with the model, run the following command from the project's root directory:

```bash
solara run server.py
```
The application will then be available in your browser at `http://localhost:8765`.

### Analysis Scripts

The `plotting/` directory contains scripts for offline analysis and visualization of data generated from model runs.



## File Descriptions

* `server.py`: The main Solara application. This script creates the interactive dashboard, including all UI components and plots.
* `water_toc/model.py`: Defines the core `WaterToC` Mesa model, which manages the grid, agent scheduling, and environmental state.
* `water_toc/agents.py`: Defines the `Human` and `AI` agent classes and their decision-making logic.

### Plotting & Analysis

* `plotting/`: A directory containing all scripts for data analysis and visualization.
    * `3Dplots_coop.py` / `3Dplots_env.py`: Scripts to generate 3D surface plots for cooperation and environment state gotten from parameter sweep data.
    * `sweep_analysis.py`: Main script for running parameter sweeps and generating the output CSV called `water_toc_sweep_results.csv` used by the plotting scripts.
    * `spatial_analysis_plotting.py`: Main script for running a parameter sweep for spatial plotting. It outputs a file called `summary_agents_clusters_and_pointprocess_MEDIANS.csv` and `pair_stats_ripley_MEDIANS.csv.csv`
    * `fpoint.py` :  creates a scatterplot of runs that end up in fixed points, as well as a table of runs which end up with fixed points more than 50% of the time.
* `cwd.py`: Performs a time-frequency analysis on the simulation output data (water_toc_sweep_results.csv). Its primary purpose is to identify and characterize oscillatory dynamics in the cooperation fraction using a Continuous Wavelet Transform (CWT).
* `peak_envelope_oscillations.py`: This script performs a quantitative analysis of the oscillatory dynamics found in the simulation data. Instead of visualizing the frequency content like the wavelet script, this one uses a peak-envelope analysis.
* `spatial_analysis_plotting.py`: Outputs spatial analysis measures such as Moran´s I, fractions of cooperators, largest cooperative cluster sizes, median cooperative cluster sizes, as well as the size of the largest cooperative cluster.