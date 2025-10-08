"""
Interactive Solara app to run and visualize the WaterToC agent-based model.

Purpose
-------
This app exposes model parameters via interactive controls, runs the simulation,
and renders four views for analysis and paper-ready figures:
1) Time-series of cooperation, environment, and agent counts.
2) Phase-space trajectory (cooperation vs. environment) with a 1/θ reference.
3) Optional limit-cycle fit when periodic behavior is detected.
4) A grid snapshot showing final water distribution and agents.

The layout uses a sidebar for inputs and a two-column main area for plots
(left) and explanatory text (right).
"""

import solara
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from typing import Tuple, List
from scipy.spatial.distance import cdist
from scipy.interpolate import splprep, splev
from water_toc.model import WaterToC

FIXED_HEIGHT = 20
FIXED_WIDTH = 20

initial_humans = solara.reactive(50)
initial_ai = solara.reactive(50)
human_c_allocation = solara.reactive(0.1)
human_d_allocation = solara.reactive(0.15)
ai_c_allocation = solara.reactive(2.0)
ai_d_allocation = solara.reactive(3.0)
max_water_capacity = solara.reactive(20)
water_cell_density = solara.reactive(0.3)
theta = solara.reactive(3.0)
deviation_rate = solara.reactive(0.1)
max_steps = solara.reactive(100)
seed = solara.reactive(42)

model_data = solara.reactive(None)
is_running = solara.reactive(False)
current_step = solara.reactive(0)
grid_state = solara.reactive(None)


def detect_limit_cycle(
    x: np.ndarray,
    y: np.ndarray,
    min_cycle_length: int = 5,
    tolerance: float = 0.05,
    min_points_in_cycle: int = 10
) -> Tuple[bool, List[int], float]:
    """
    Heuristically detect a repeating pattern (limit cycle) in a 2D trajectory.

    Parameters
    ----------
    x, y : arrays
        Time-aligned series (e.g., cooperation and environment).
    min_cycle_length : int
        Minimum spacing between candidate repeat indices.
    tolerance : float
        Distance threshold for matching repeated segments.
    min_points_in_cycle : int
        Minimum number of points for an accepted cycle length.

    Returns
    -------
    has_cycle : bool
        True if a plausible repeating segment is found.
    idxs : list[int]
        Indices covering one detected cycle segment.
    period : float
        Estimated cycle length in points (steps) for the detected segment.
    """
    if len(x) < min_cycle_length * 2:
        return False, [], 0.0
    n = len(x)
    start = max(0, n - min(100, n // 2))
    pts = np.column_stack([x[start:], y[start:]])
    dist = cdist(pts, pts)
    candidates = [
        (i, j, j - i)
        for i in range(len(pts) - min_cycle_length)
        for j in range(i + min_cycle_length, len(pts))
        if dist[i, j] < tolerance
    ]
    best, best_score = None, 0
    for i, j, length in candidates:
        pattern = pts[i:j]
        score = 0
        for k in range(j, len(pts) - length, length):
            seg = pts[k: k + length]
            if seg.shape == pattern.shape and np.mean(np.linalg.norm(pattern - seg, axis=1)) < tolerance:
                score += 1
        if score > best_score and length >= min_points_in_cycle:
            best_score, best = score, (i + start, j + start, length)
    if best and best_score > 0:
        i, j, length = best
        return True, list(range(i, j)), length
    return False, [], 0.0


def analyze_stability(df: pd.DataFrame, window: int = 20) -> dict:
    """
    Classify the recent system behavior using variance and linear trends.

    Parameters
    ----------
    df : DataFrame
        Model vars dataframe with columns 'Coop_Fraction' and 'Environment_State'.
    window : int
        Number of trailing steps used for the analysis.

    Returns
    -------
    dict
        Summary with stability flag, type label, variances, and trends.
    """
    if len(df) < window:
        return {"stable": False, "stability_type": "Insufficient data"}
    sub = df.iloc[-window:]
    coop, env = sub["Coop_Fraction"].values, sub["Environment_State"].values
    var_sum = np.var(coop) + np.var(env)
    t_coop = np.polyfit(range(window), coop, 1)[0]
    t_env = np.polyfit(range(window), env, 1)[0]
    trend = abs(t_coop) + abs(t_env)
    if var_sum < 1e-3 and trend < 1e-3:
        typ = "Fixed Point"
    elif var_sum > 1e-2 and trend < 5e-3:
        typ = "Limit Cycle"
    elif trend > 1e-2:
        typ = "Trending"
    else:
        typ = "Quasi-Stable"
    return {
        "stable": var_sum < 1e-2,
        "stability_type": typ,
        "cooperation_variance": np.var(coop),
        "environment_variance": np.var(env),
        "cooperation_trend": t_coop,
        "environment_trend": t_env,
    }


def run_simulation() -> pd.DataFrame:
    """
    Run the WaterToC model for the configured number of steps and
    capture both the time series (returned) and a final grid snapshot (stored in grid_state).
    """
    try:
        m = WaterToC(
            height=FIXED_HEIGHT,
            width=FIXED_WIDTH,
            initial_humans=initial_humans.value,
            initial_ai=initial_ai.value,
            human_C_allocation=human_c_allocation.value,
            human_D_allocation=human_d_allocation.value,
            ai_C_allocation=ai_c_allocation.value,
            ai_D_allocation=ai_d_allocation.value,
            max_water_capacity=max_water_capacity.value,
            water_cell_density=water_cell_density.value,
            theta=theta.value,
            deviation_rate=deviation_rate.value,
            seed=seed.value,
        )
        for i in range(max_steps.value):
            m.step()
            current_step.value = i + 1
        grid_state.value = {
            "water_levels": m.water_levels.copy(),
            "water_capacity": m.water_capacity.copy(),
            "has_water": m.has_water.copy(),
            "agents": m._get_agent_pos_strategies_list(),
            "width": m.width,
            "height": m.height,
        }
        return m.datacollector.get_model_vars_dataframe()
    except Exception as e:
        print("Simulation error:", e)
        grid_state.value = None
        return pd.DataFrame()


@solara.component
def ParameterControls():
    """
    Sidebar sliders and inputs to configure the model parameters before running.
    """
    with solara.Card("Model Parameters"):
        solara.Text(f"Grid: {FIXED_HEIGHT}×{FIXED_WIDTH}")
        solara.SliderInt("Initial humans", initial_humans, 1, 50)
        solara.SliderInt("Initial AI", initial_ai, 1, 50)
        solara.SliderFloat("Human C consumption", human_c_allocation, 0, 4, step=0.05)
        solara.SliderFloat("Human D consumption", human_d_allocation, 0, 4, step=0.05)
        solara.SliderFloat("AI C consumption", ai_c_allocation, 0, 4, step=0.05)
        solara.SliderFloat("AI D consumption", ai_d_allocation, 0, 4, step=0.05)
        solara.SliderInt("Max resource capacity", max_water_capacity, 1, 50)
        solara.SliderFloat("Resource density", water_cell_density, 0.1, 1, step=0.1)
        solara.SliderFloat("Theta (renewability)", theta, 0, 20, step=0.1)
        solara.SliderFloat("Deviation rate", deviation_rate, 0, 1, step=0.01)
        solara.SliderInt("Steps", max_steps, 10, 1000)
        solara.SliderInt("Seed", seed, 1, 1000)


@solara.component
def SimulationControls():
    """
    Run button with a simple progress indicator bound to the reactive step counter.
    """
    def on_run():
        is_running.value, current_step.value = True, 0
        model_data.value = run_simulation()
        is_running.value = False

    with solara.Card("Simulation Controls"):
        solara.Button("Run Simulation", on_click=on_run, disabled=is_running.value)
        solara.Text(f"{('Running' if is_running.value else 'Done')}: {current_step.value}/{max_steps.value}")


@solara.component
def TimeSeriesPlots():
    """
    Three stacked plots:
    1) Cooperation fractions (overall, AI, Human) with a reference line at 1/θ.
    2) Environment state over time.
    3) Counts of cooperators and defectors.
    """
    if model_data.value is None or model_data.value.empty:
        solara.Markdown("No data. Run simulation.")
        return
    df = model_data.value.reset_index()
    thr = 1.0 / theta.value if theta.value > 0 else 0.0

    fig = make_subplots(
        rows=3,
        cols=1,
        subplot_titles=["Cooperation Dynamics", "Environment (n)", "Agent Counts"],
        vertical_spacing=0.12,
    )

    fig.add_trace(
        go.Scatter(x=df.index, y=df["Coop_Fraction"], name="Overall", line=dict(color="blue", width=2)),
        row=1, col=1,
    )
    if "AI_Coop_Fraction" in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df["AI_Coop_Fraction"], name="AI", line=dict(color="orange", dash="dash", width=2)),
            row=1, col=1,
        )
    if "Human_Coop_Fraction" in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df["Human_Coop_Fraction"], name="Human", line=dict(color="green", dash="dot", width=2)),
            row=1, col=1,
        )
    fig.add_hline(
        y=thr, row=1, col=1, line_dash="dash", line_color="gray",
        annotation_text=f"1/θ = {thr:.2f}", annotation_position="bottom right",
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df["Environment_State"], name="Environment", line=dict(color="purple", width=2)),
        row=2, col=1,
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df["Cooperators"], name="Cooperators", line=dict(color="green", width=2)),
        row=3, col=1,
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df["Defectors"], name="Defectors", line=dict(color="red", width=2)),
        row=3, col=1,
    )

    fig.update_layout(
        height=1000,
        title=dict(text="WaterToC Dynamics", y=0.98, x=0.01, xanchor="left", yanchor="top"),
        margin=dict(t=150, l=60, r=60, b=80),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.05,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="rgba(0,0,0,0.2)",
            borderwidth=1,
        ),
    )
    fig.update_xaxes(title_text="Step", row=3, col=1)
    fig.update_yaxes(title_text="Fraction", row=1, col=1)
    fig.update_yaxes(title_text="n", row=2, col=1)
    fig.update_yaxes(title_text="Count", row=3, col=1)
    solara.FigurePlotly(fig)


@solara.component
def PhaseSpacePlot():
    """
    Phase portrait of cooperation (x) vs environment (y), including start/end markers
    and a vertical reference at x = 1/θ for interpretation.
    """
    if model_data.value is None or model_data.value.empty:
        return
    df = model_data.value.reset_index()
    x, y = df["Coop_Fraction"].values, df["Environment_State"].values
    thr = 1.0 / theta.value if theta.value > 0 else 0.0

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name="Trajectory", line=dict(color="lightblue", width=1)))
    fig.add_trace(go.Scatter(x=x, y=y, mode="markers", showlegend=False, marker=dict(color=df.index, colorscale="Viridis", size=4)))
    fig.add_trace(go.Scatter(x=[x[0]], y=[y[0]], mode="markers", name="Start", marker=dict(color="green", symbol="star", size=12)))
    fig.add_trace(go.Scatter(x=[x[-1]], y=[y[-1]], mode="markers", name="End", marker=dict(color="red", symbol="x", size=12)))
    fig.add_vline(x=thr, line_dash="dash", line_color="gray", annotation_text=f"1/θ={thr:.2f}", annotation_position="top")
    fig.update_layout(
        title="Phase Space",
        width=600,
        height=500,
        margin=dict(t=140, l=60, r=60, b=60),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.05,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="rgba(0,0,0,0.2)",
            borderwidth=1,
        ),
    )
    solara.FigurePlotly(fig)


@solara.component
def LimitCyclePlot():
    """
    If a limit cycle is detected in the last portion of the trajectory,
    plot a smoothed closed curve and annotate direction.
    """
    if model_data.value is None or model_data.value.empty:
        return
    df = model_data.value.reset_index()
    x, y = df["Coop_Fraction"].values, df["Environment_State"].values
    has, idxs, period = detect_limit_cycle(x, y)
    if not has:
        return
    cx, cy = x[idxs], y[idxs]
    cx = np.append(cx, cx[0])
    cy = np.append(cy, cy[0])
    tck, _ = splprep([cx, cy], s=0.001, per=True)
    u = np.linspace(0, 1, 300)
    xs, ys = splev(u, tck)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="Cycle", line=dict(color="crimson", width=2)))
    for i in np.linspace(0, len(xs) - 2, 4, dtype=int):
        fig.add_annotation(ax=xs[i], ay=ys[i], axref="x", ayref="y", x=xs[i + 1], y=ys[i + 1], showarrow=True, arrowhead=2, arrowwidth=1.5)
    fig.update_layout(title=f"Limit Cycle (~{period} steps)", width=600, height=500, margin=dict(t=140, l=60, r=60, b=60), showlegend=False)
    solara.FigurePlotly(fig)


@solara.component
def SummaryStats():
    """
    Display end-of-run and average metrics for quick inspection.
    """
    if model_data.value is None or model_data.value.empty:
        return
    df = model_data.value
    with solara.Card("Summary Statistics"):
        solara.Markdown(
            f"""
- **Final resource (n):** {df['Environment_State'].iloc[-1]:.3f}
- **Final cooperation:** {df['Coop_Fraction'].iloc[-1]*100:.1f}%
- **Average cooperation:** {df['Coop_Fraction'].mean()*100:.1f}%
"""
        )


@solara.component
def StabilityAnalysis():
    """
    Report stability classification and whether a limit cycle is present,
    with simple variance and trend diagnostics.
    """
    if model_data.value is None or model_data.value.empty:
        return
    info = analyze_stability(model_data.value)
    has, _, period = detect_limit_cycle(
        model_data.value["Coop_Fraction"].values,
        model_data.value["Environment_State"].values,
    )
    with solara.Card("Stability Analysis"):
        solara.Markdown(
            f"""
- **Type:** {info['stability_type']}
- **Stable:** {'Yes' if info['stable'] else 'No'}
- **Limit cycle:** {'Yes' if has else 'No'}
- **Cycle period:** {period if has else 'N/A'} steps
- **Coop variance:** {info['cooperation_variance']:.4f}
- **Env variance:** {info['environment_variance']:.4f}
"""
        )


@solara.component
def DescriptionText():
    """
    Plain-language overview and figure for contextualizing the app in a paper/demo.
    """
    solara.Markdown("""
## Coevolutionary Dynamics of Cooperation and Environmental Sustainability in Shared Resource Systems
""")
    solara.Markdown("""
The "Water Commons" ABM model was developed for multidisciplinary research of human–AI alignment
(which includes sustainable governance of environmental resources) and nonlinear dynamics in socio-technological systems.

Beyond academic analysis, the goal is to support quantitative, evidence-based recommendations
for public and private entities regarding daily operations and institutional decision-making, including AI governance.
""")
    solara.Image("public/ABMenv.png", width="90%")


@solara.component
def ParameterDocs():
    """
    Quick reference for users adjusting parameters in the sidebar.
    """
    with solara.Card("Parameter Reference"):
        solara.Markdown(
            """
- **Initial humans**: number of human agents at start  
- **Initial AI**: number of AI agents at start  
- **Human C consumption**: water use when humans cooperate  
- **Human D consumption**: water use when humans defect  
- **AI C consumption**: water use when AI cooperate  
- **AI D consumption**: water use when AI defect  
- **Max resource capacity**: max water per cell  
- **Resource density**: fraction of cells that have water  
- **Theta (renewability)**: resource renewal rate parameter  
- **Deviation rate**: probability agents randomize away from optimal strategy  
- **Steps**: number of simulation steps  
- **Seed**: random seed for reproducibility  
"""
        )


@solara.component
def GridView():
    """
    Final-step grid snapshot of water levels (heatmap) with agent overlays.
    Helpful for visual inspection of spatial distributions.
    """
    if grid_state.value is None:
        solara.Markdown("Run the simulation to see the grid.")
        return

    gs = grid_state.value
    W, H = gs["width"], gs["height"]
    water = gs["water_levels"]
    hasw = gs["has_water"]

    Z = np.where(hasw, water, 0.0).T
    text = np.where(hasw, np.round(water, 1).astype(str), "").T

    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(
        go.Heatmap(
            z=Z,
            x=list(range(W)),
            y=list(range(H)),
            colorscale="Blues",
            showscale=True,
            colorbar=dict(title="Water"),
            zmin=0,
            zmax=max(1.0, float(np.max(Z))),
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Heatmap(
            z=np.zeros_like(Z),
            x=list(range(W)),
            y=list(range(H)),
            colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(0,0,0,0)"]],
            showscale=False,
            text=text,
            texttemplate="%{text}",
            textfont=dict(color="black", size=10),
        ),
        row=1, col=1,
    )

    agents = gs["agents"]
    if agents:
        ids, kinds, xs, ys, strats = zip(*agents)
        xs = np.array(xs)
        ys = np.array(ys)
        kinds = np.array(kinds)
        strats = np.array(strats)

        def add_group(name, mask, symbol, color):
            if np.any(mask):
                fig.add_trace(
                    go.Scatter(
                        x=xs[mask] + 0.0,
                        y=ys[mask] + 0.0,
                        mode="markers",
                        name=name,
                        marker=dict(
                            symbol=symbol,
                            size=12,
                            line=dict(width=1, color="black"),
                            color=color,
                        ),
                        hovertemplate="(%{x}, %{y})<extra>" + name + "</extra>",
                    ),
                    row=1, col=1,
                )

        human = kinds == "Human"
        ai = kinds == "AI"
        coop = strats == "C"
        defect = strats == "D"

        add_group("Human – C", human & coop, "square", "green")
        add_group("Human – D", human & defect, "square", "red")
        add_group("AI – C", ai & coop, "circle", "green")
        add_group("AI – D", ai & defect, "circle", "red")

    fig.update_xaxes(range=[-0.5, W - 0.5], dtick=1, showgrid=True, zeroline=False, mirror=True, ticks="outside")
    fig.update_yaxes(
        range=[H - 0.5, -0.5],
        dtick=1,
        showgrid=True,
        zeroline=False,
        mirror=True,
        ticks="outside",
        scaleanchor="x",
        scaleratio=1,
    )
    fig.update_layout(
        title="Grid View (water + agents)",
        width=700,
        height=700,
        margin=dict(t=80, l=60, r=60, b=60),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="rgba(0,0,0,0.2)",
            borderwidth=1,
        ),
    )
    solara.FigurePlotly(fig)


@solara.component
def Page():
    """
    Root component: sidebar with controls and right panel for documentation;
    left panel shows the three time-series plots, the phase portrait, the limit-cycle
    plot (if any), and the grid snapshot.
    """
    solara.Title('WaterToC Mesa Model')
    with solara.Sidebar():
        ParameterControls()
        SimulationControls()
        SummaryStats()
        StabilityAnalysis()
        ParameterDocs()
    with solara.Row():
        with solara.Column(style={"width": "1200px"}):
            TimeSeriesPlots()
            PhaseSpacePlot()
            LimitCyclePlot()
            GridView()
        with solara.Column(style={"width": "500px"}):
            DescriptionText()


if __name__ == "__main__":
    Page()
