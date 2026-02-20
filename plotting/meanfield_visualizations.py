import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

"""
Mean-field reproduction of the Weitz et al. (2016) co-evolutionary model (Eq. 17)
with (1) time-series simulation and (2) a phase-plane portrait (x vs n).

State variables:
    x(t) : fraction of cooperators in the population, constrained to [0, 1]
    n(t) : environmental state (resource level / quality), constrained to [0, 1]

Parameters:
    eps   : time-scale separation for strategy dynamics (smaller eps => faster x dynamics)
    theta : strength of environmental feedback in n-dot
    R,S,T,P : payoff matrix entries for the Prisoner's Dilemma

Dynamics (Weitz Eq. 17):
    eps * xdot = x(1-x) * g(x) * (1 - 2n)
    ndot       = n(1-n) * ( -1 + (1+theta) * x )

where:
    g(x) = (P - S) + [(T - R) - (P - S)] * x

Interpretation:
- x evolves via a replicator-like term x(1-x) modulated by the environment through (1-2n).
  The sign flip at n=1/2 creates the “oscillating tragedy of the commons” mechanism.
- n follows logistic-style growth/decay with a threshold in x at x = 1/(1+theta).

Workflow
--------
1) weitz_rhs(t, y, ...) defines the ODE right-hand side (xdot, ndot).
2) integrate_traj(x0, n0, ...) integrates the ODE from one initial condition using solve_ivp.
3) plot_timeseries(...) produces the time-series x(t) and n(t) for one chosen initial condition.
4) plot_phase_portrait(...) draws:
   - a background phase portrait vector field using streamplot
   - nullclines (n = 1/2 and x = 1/(1+theta))
   - multiple trajectories starting from different initial conditions ("initials")
     with direction arrows along each trajectory.

"""

def weitz_rhs(t, y, eps=0.1, theta=2.0, R=3, S=0, T=5, P=1):
    x, n = y

    x = float(np.clip(x, 0.0, 1.0))
    n = float(np.clip(n, 0.0, 1.0))

    dPS = P - S
    dTR = T - R

    g = dPS + (dTR - dPS) * x
    xdot = (x * (1 - x) * g * (1 - 2 * n)) / eps
    ndot = n * (1 - n) * (-1 + (1 + theta) * x)
    return [xdot, ndot]


def integrate_traj(x0, n0, tmax=60, nsteps=4000, **params):
    t_eval = np.linspace(0, tmax, nsteps)
    sol = solve_ivp(
        lambda t, y: weitz_rhs(t, y, **params),
        t_span=(0, tmax),
        y0=[x0, n0],
        t_eval=t_eval,
        rtol=1e-8,
        atol=1e-10,
        method="RK45",
    )
    x = np.clip(sol.y[0], 0, 1)
    n = np.clip(sol.y[1], 0, 1)
    return sol.t, x, n


def add_arrows_along_line(ax, x, y, n_arrows=6, color="k", size=12):
    if len(x) < 5:
        return
    idxs = np.linspace(10, len(x) - 10, n_arrows).astype(int)
    for i in idxs:
        dx = x[i + 1] - x[i - 1]
        dy = y[i + 1] - y[i - 1]
        ax.annotate(
            "",
            xy=(x[i] + dx * 0.001, y[i] + dy * 0.001),
            xytext=(x[i], y[i]),
            arrowprops=dict(arrowstyle="-|>", color=color, lw=1.2),
        )


def plot_phase_portrait(
    eps=0.1, theta=2.0, R=3, S=0, T=5, P=1,
    initials=((0.9, 0.01), (0.8, 0.15), (0.7, 0.3), (0.5, 0.4), (0.4, 0.45)),
    tmax=60,
    grid_N=25,
    title="Phase portrait",
):
 
    xs = np.linspace(0, 1, grid_N)
    ns = np.linspace(0, 1, grid_N)
    X, N = np.meshgrid(xs, ns)

    U = np.zeros_like(X)
    V = np.zeros_like(N)
    for i in range(grid_N):
        for j in range(grid_N):
            u, v = weitz_rhs(0, [X[i, j], N[i, j]], eps=eps, theta=theta, R=R, S=S, T=T, P=P)
            U[i, j] = u
            V[i, j] = v


    speed = np.sqrt(U**2 + V**2) + 1e-12
    Un = U / speed
    Vn = V / speed

    fig, ax = plt.subplots(figsize=(6.2, 5.2))


    ax.streamplot(
        xs, ns, Un.T, Vn.T,
        density=1.1,
        linewidth=0.9,
        arrowsize=1.0,
        color="0.25"
    )

    ax.axhline(0.5, color="0.4", lw=1.2, ls="--")

    if theta > 0:
        x_star = 1.0 / (1.0 + theta)
        ax.axvline(x_star, color="0.4", lw=1.2, ls="--")
    else:
        x_star = None

    if x_star is not None:
        ax.plot([x_star], [0.5], marker="*", markersize=12, color="k")

    for (x0, n0) in initials:
        t, x, n = integrate_traj(x0, n0, tmax=tmax, eps=eps, theta=theta, R=R, S=S, T=T, P=P)
        ax.plot(x, n, color="k", lw=2.0)
        add_arrows_along_line(ax, x, n, n_arrows=5, color="k")

        ax.plot([x0], [n0], "o", color="k", ms=5)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Fraction cooperators, x")
    ax.set_ylabel("Environment state, n")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")

    txt = f"eps={eps}, theta={theta}, R={R}, S={S}, T={T}, P={P}"

    plt.tight_layout()
    plt.show()
    plt.savefig("phase")


def plot_timeseries(
    eps=0.1, theta=2.0, R=3, S=0, T=5, P=1,
    x0=0.9, n0=0.01,
    tmax=60,
    title="Time series plot"
):
    t, x, n = integrate_traj(x0, n0, tmax=tmax, eps=eps, theta=theta, R=R, S=S, T=T, P=P)

    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    ax.plot(t, x, lw=2.0, label="Fraction cooperators x(t)")
    ax.plot(t, n, lw=2.0, label="Environment n(t)")
    ax.set_xlabel("Time")
    ax.set_ylabel("State")
    ax.set_ylim(-0.02, 1.05)
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.show()
    plt.savefig("timeseries")


if __name__ == "__main__":
    params = dict(eps=0.1, theta=2.0, R=3, S=0, T=5, P=1)
    plot_timeseries(**params, x0=0.9, n0=0.01, tmax=60)

    initials = [(0.9, 0.01), (0.8, 0.15), (0.7, 0.3), (0.5, 0.4), (0.4, 0.45)]
    plot_phase_portrait(**params, initials=initials, tmax=60)
