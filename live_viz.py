"""
live_viz.py — Live training dashboard for the EC Maze Agent
Silent Cartographer: COSC 4368 AI Spring 2026

Run:
    python live_viz.py [--pop 60] [--gens 80] [--turns 3000] [--persist]

Layout
------
┌─────────────────────┬──────────────────────────┐
│  Agent map          │  Fitness curves           │
│  (64×64 grid,       │  (best / mean / ±1σ)      │
│   best individual)  │                           │
├─────────────────────┴──────────────────────────-┤
│  Stats bar  (gen · σ · strategy · deaths · etc) │
└──────────────────────────────────────────────────┘

The GA runs on a background thread.  The main thread owns the matplotlib
event loop and redraws every ~400 ms via FuncAnimation.
"""

from __future__ import annotations
import threading
import time
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LinearSegmentedColormap
from collections import defaultdict

# ── Import our agent module ───────────────────────────────────────────────────
from maze_agent import (
    NeuralController, EvolutionaryAgent, GeneticAlgorithm,
    evaluate_fitness, StateEncoder,
    START_CELL, GOAL_CELL, GRID_SIZE, ACTION_MAP, INVERT_MAP,
)

try:
    from environment import Action, TurnResult, MazeEnvironment
except ImportError:
    from enum import Enum
    class Action(Enum):
        MOVE_UP=0; MOVE_DOWN=1; MOVE_LEFT=2; MOVE_RIGHT=3; WAIT=4
    class TurnResult:
        def __init__(self):
            self.wall_hits=0; self.current_position=(0,0); self.is_dead=False
            self.is_confused=False; self.is_goal_reached=False
            self.teleported=False; self.actions_executed=0
    class MazeEnvironment:
        def __init__(self, maze_id): pass
        def reset(self): return START_CELL
        def step(self, actions): return TurnResult()
        def get_episode_stats(self): return {}


# ─────────────────────────────────────────────────────────────────────────────
# Shared visualisation state  (written by GA thread, read by plot thread)
# ─────────────────────────────────────────────────────────────────────────────
class SharedState:
    def __init__(self):
        self.lock           = threading.Lock()
        self.history        = []              # list of gen-dicts
        self.agent_grid     = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
        self.agent_pos      = START_CELL
        self.agent_path     = []
        self.known_walls    : set = set()
        self.known_pits     : set = set()
        self.visit_map      = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
        self.status_text    = "Initialising…"
        self.done           = False

    def update_from_agent(self, agent: EvolutionaryAgent,
                          pos, path, gen_record: dict, status: str):
        mem = agent.memory
        grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
        vmap = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)

        # Visit density (log-scaled so rarely visited cells still show)
        for (r, c), cnt in mem.visit_count.items():
            if 0 <= r < GRID_SIZE and 0 <= c < GRID_SIZE:
                vmap[r, c] = np.log1p(cnt)

        # Cell category (used for discrete colours overlay)
        for (r, c), cnt in mem.visit_count.items():
            if 0 <= r < GRID_SIZE and 0 <= c < GRID_SIZE:
                grid[r, c] = 1.0      # explored

        for (r, c) in mem.known_pits:
            if 0 <= r < GRID_SIZE and 0 <= c < GRID_SIZE:
                grid[r, c] = 2.0      # pit

        for (src, dst) in mem.known_teleports.items():
            r, c = src
            if 0 <= r < GRID_SIZE and 0 <= c < GRID_SIZE:
                grid[r, c] = 3.0      # teleport

        with self.lock:
            self.agent_grid  = grid
            self.visit_map   = vmap
            self.agent_pos   = pos
            self.agent_path  = list(path)
            self.known_walls = set(mem.known_walls)
            self.known_pits  = set(mem.known_pits)
            self.status_text = status
            if gen_record:
                self.history.append(gen_record)

    def snapshot(self):
        with self.lock:
            return dict(
                history     = list(self.history),
                agent_grid  = self.agent_grid.copy(),
                visit_map   = self.visit_map.copy(),
                agent_pos   = self.agent_pos,
                agent_path  = list(self.agent_path),
                status_text = self.status_text,
                done        = self.done,
            )


SHARED = SharedState()


# ─────────────────────────────────────────────────────────────────────────────
# Fixed fitness evaluation  (corrects the position-tracking bug)
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_fitness_fixed(controller: NeuralController,
                            env: MazeEnvironment,
                            episodes: int = 2,
                            max_turns: int = 3000,
                            epsilon: float = 0.05,
                            persist: bool = False) -> tuple[float, EvolutionaryAgent]:
    """
    Returns (mean_fitness, agent_after_last_episode).

    Fixes vs original evaluate_fitness
    -----------------------------------
    • env.reset() return value is used to initialise agent position.
    • position is updated BEFORE encode_state in plan_turn by reading
      last_result.current_position directly (agent always trusts the env).
    • unique_cells is updated from last_result, not from agent.current_pos,
      to handle cases where the agent's internal tracking drifts.
    """
    agent      = EvolutionaryAgent(controller, epsilon=epsilon,
                                   persist_memory=persist)
    total_fit  = 0.0
    last_agent = agent

    for ep in range(episodes):
        # ── reset ────────────────────────────────────────────────────────────
        agent.reset_episode()
        start_pos        = env.reset()   # trust the environment
        agent.current_pos = start_pos

        last_result  = None
        turns        = 0
        deaths       = 0
        goal_reached = False
        unique_cells : set = {start_pos}

        while turns < max_turns:
            actions     = agent.plan_turn(last_result)
            last_result = env.step(actions)
            turns      += 1

            env_pos = last_result.current_position    # ground truth from env
            unique_cells.add(env_pos)

            if last_result.is_dead:
                deaths            += 1
                agent.current_pos  = start_pos        # respawn

            if last_result.is_goal_reached:
                goal_reached = True
                break

        # ── fitness ──────────────────────────────────────────────────────────
        r, c   = last_result.current_position if last_result else start_pos
        gr, gc = GOAL_CELL
        dist   = abs(gr - r) + abs(gc - c)

        ep_fit = 0.0
        ep_fit += 200 * (126 - dist)
        ep_fit += 5   * len(unique_cells)
        ep_fit -= 300 * deaths
        ep_fit -= 10  * turns
        if goal_reached:
            ep_fit += 50_000

        total_fit += ep_fit
        last_agent = agent

    return total_fit / episodes, last_agent


# ─────────────────────────────────────────────────────────────────────────────
# GA subclass that feeds SharedState after every generation
# ─────────────────────────────────────────────────────────────────────────────
class VisualGA(GeneticAlgorithm):
    """Extends GeneticAlgorithm with live-state reporting."""

    def step(self, env, eval_episodes=2, eval_turns=3000,
             epsilon=0.05, verbose=False):

        gen_num = self.generation + 1
        SHARED.status_text = f"Gen {gen_num} — evaluating population…"

        # ── Evaluate every individual ────────────────────────────────────────
        best_ep_agent = None
        for i, ctrl in enumerate(self.population):
            fit, ep_agent = evaluate_fitness_fixed(
                ctrl, env,
                episodes  = eval_episodes,
                max_turns = eval_turns,
                epsilon   = epsilon,
            )
            self.fitness[i] = fit

            if best_ep_agent is None or fit > self.fitness[:i+1].max():
                best_ep_agent = ep_agent

            if (i + 1) % 5 == 0 or (i + 1) == self.pop_size:
                print(f"  [{i+1:3d}/{self.pop_size}]"
                      f"  fit={self.fitness[:i+1].max():+.0f}"
                      f"  mean={self.fitness[:i+1].mean():+.0f}"
                      f"  std={self.fitness[:i+1].std():.0f}")
                SHARED.status_text = (
                    f"Gen {gen_num}  [{i+1}/{self.pop_size}]"
                    f"  best={self.fitness[:i+1].max():+.0f}"
                    f"  mean={self.fitness[:i+1].mean():+.0f}"
                )

        # ── Hall of fame ─────────────────────────────────────────────────────
        best_idx = int(np.argmax(self.fitness))
        if self.fitness[best_idx] > self.best_fitness:
            self.best_fitness    = float(self.fitness[best_idx])
            self.best_individual = self.population[best_idx].clone()
            tag = "★ NEW BEST"
        else:
            tag = ""

        rec = {
            "generation": self.generation,
            "best":  float(self.fitness.max()),
            "mean":  float(self.fitness.mean()),
            "std":   float(self.fitness.std()),
            "worst": float(self.fitness.min()),
            "sigma": self.mut_sigma,
        }
        self.history.append(rec)

        print(f"\n  Gen {gen_num:3d}  best={rec['best']:+.0f}"
              f"  mean={rec['mean']:+.0f}  std={rec['std']:.0f}"
              f"  σ={self.mut_sigma:.4f}  {tag}\n")

        # ── Push best agent state to shared viz state ─────────────────────────
        if best_ep_agent is not None:
            status = (f"Gen {gen_num}  |  best={rec['best']:+.0f}"
                      f"  mean={rec['mean']:+.0f}  σ={self.mut_sigma:.4f}  {tag}")
            SHARED.update_from_agent(
                best_ep_agent,
                best_ep_agent.current_pos,
                best_ep_agent.memory.path,
                rec,
                status,
            )

        # ── Build next generation (same as base class) ────────────────────────
        sorted_idx  = np.argsort(self.fitness)[::-1]
        elite_k     = max(1, int(self.elite_frac * self.pop_size))
        new_pop     = [self.population[i].clone() for i in sorted_idx[:elite_k]]

        import random
        while len(new_pop) < self.pop_size:
            p1 = self.population[self._tournament_select()]
            if random.random() < self.crossover_prob:
                p2    = self.population[self._tournament_select()]
                child = self._uniform_crossover(p1, p2)
            else:
                child = p1.clone()
            child = self._mutate(child)
            new_pop.append(child)

        self.population = new_pop
        self.mut_sigma  = max(self.min_mut_sigma, self.mut_sigma * self.mut_decay)
        self.generation += 1
        return self.best_individual


# ─────────────────────────────────────────────────────────────────────────────
# Training thread
# ─────────────────────────────────────────────────────────────────────────────
def training_thread(args):
    env = MazeEnvironment("training")
    ga  = VisualGA(
        pop_size       = args.pop,
        init_mut_sigma = args.sigma,
        mut_decay      = args.decay,
    )
    print(f"\n[GA] {ga.pop_size} individuals × {ga.population[0].num_params} params")
    print(f"     Architecture : {ga.layer_sizes}")
    print(f"     Generations  : {args.gens}\n")

    epsilon = 0.15
    for gen in range(args.gens):
        ga.step(
            env,
            eval_episodes = args.eps_eval,
            eval_turns    = args.turns,
            epsilon       = epsilon,
        )
        epsilon = max(0.02, epsilon * 0.97)

        if ga.best_individual:
            ga.best_individual.save("best_weights.npy")

    with SHARED.lock:
        SHARED.done        = True
        SHARED.status_text = (f"Training complete — {args.gens} generations"
                               f"  |  best fitness = {ga.best_fitness:+.0f}")
    print("\n✓ Training done.")


# ─────────────────────────────────────────────────────────────────────────────
# Matplotlib live dashboard
# ─────────────────────────────────────────────────────────────────────────────

# ── Colour maps ───────────────────────────────────────────────────────────────
UNEXPLORED_CLR = "#0d1117"
EXPLORED_CLR   = "#1c3a5c"
PATH_CMAP      = LinearSegmentedColormap.from_list(
    "path", ["#1e3a5f", "#2979ff", "#00e5ff"], N=256
)
VISIT_CMAP = LinearSegmentedColormap.from_list(
    "visit", ["#0d1117", "#1a4080", "#3399ff", "#00ffcc"], N=256
)

def build_figure():
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(16, 8), facecolor="#0d1117")
    gs  = gridspec.GridSpec(
        2, 2,
        figure    = fig,
        hspace    = 0.08,
        wspace    = 0.25,
        height_ratios = [14, 1],
        left=0.04, right=0.97, top=0.93, bottom=0.06,
    )

    ax_map  = fig.add_subplot(gs[0, 0])   # agent map
    ax_fit  = fig.add_subplot(gs[0, 1])   # fitness curves
    ax_stat = fig.add_subplot(gs[1, :])   # status bar

    # ── Map axis setup ────────────────────────────────────────────────────────
    ax_map.set_facecolor(UNEXPLORED_CLR)
    ax_map.set_xlim(-0.5, GRID_SIZE - 0.5)
    ax_map.set_ylim(GRID_SIZE - 0.5, -0.5)      # row 0 at top
    ax_map.set_aspect("equal")
    ax_map.set_title("Best Individual — Known World", color="#c8d0e0",
                     fontsize=11, pad=6)
    ax_map.set_xlabel("Column", color="#7a8a9a", fontsize=8)
    ax_map.set_ylabel("Row",    color="#7a8a9a", fontsize=8)
    ax_map.tick_params(colors="#4a5a6a", labelsize=7)
    for spine in ax_map.spines.values():
        spine.set_edgecolor("#1e2a3a")

    # ── Fitness axis setup ────────────────────────────────────────────────────
    ax_fit.set_facecolor("#060b12")
    ax_fit.set_title("Population Fitness — Live", color="#c8d0e0",
                     fontsize=11, pad=6)
    ax_fit.set_xlabel("Generation", color="#7a8a9a", fontsize=9)
    ax_fit.set_ylabel("Fitness",    color="#7a8a9a", fontsize=9)
    ax_fit.tick_params(colors="#4a5a6a", labelsize=8)
    ax_fit.grid(alpha=0.12, color="#2a3a4a")
    ax_fit.axhline(0, color="#2a4060", lw=0.8, ls="--")
    for spine in ax_fit.spines.values():
        spine.set_edgecolor("#1e2a3a")

    # ── Status bar ────────────────────────────────────────────────────────────
    ax_stat.set_facecolor("#0a1520")
    ax_stat.axis("off")

    return fig, ax_map, ax_fit, ax_stat


def make_updater(fig, ax_map, ax_fit, ax_stat):
    """Return the FuncAnimation update function (closure over axes)."""

    # Persistent plot objects so we don't recreate every frame
    im_visit   = ax_map.imshow(
        np.zeros((GRID_SIZE, GRID_SIZE)), cmap=VISIT_CMAP,
        origin="upper", vmin=0, vmax=3,
        extent=[-0.5, GRID_SIZE-0.5, GRID_SIZE-0.5, -0.5],
        interpolation="nearest", zorder=1,
    )
    path_scat  = ax_map.scatter([], [], c=[], cmap=PATH_CMAP,
                                 s=12, zorder=3, vmin=0, vmax=1,
                                 linewidths=0)
    agent_dot  = ax_map.scatter([], [], color="#ff3355", s=90,
                                 zorder=5, marker="o",
                                 edgecolors="#ff99aa", linewidths=1.2)
    start_dot  = ax_map.scatter(
        [START_CELL[1]], [START_CELL[0]],
        color="#00e676", s=120, zorder=6, marker="^",
        edgecolors="white", linewidths=1
    )
    goal_dot   = ax_map.scatter(
        [GOAL_CELL[1]], [GOAL_CELL[0]],
        color="#00e5ff", s=120, zorder=6, marker="*",
        edgecolors="white", linewidths=1
    )

    # Fitness lines
    ln_best, = ax_fit.plot([], [], lw=2.2, color="#00e5ff",   label="Best",  zorder=3)
    ln_mean, = ax_fit.plot([], [], lw=1.5, color="#2979ff",   label="Mean",  zorder=2)
    ln_worst,= ax_fit.plot([], [], lw=1.0, color="#455a6a",   label="Worst", zorder=1, ls="--")
    fill_std = [ax_fit.fill_between([], [], [], color="#2979ff", alpha=0.12, zorder=0)]
    ax_fit.legend(loc="lower right", fontsize=8,
                  facecolor="#0a1520", edgecolor="#1e2a3a", labelcolor="#c8d0e0")

    stat_txt = ax_stat.text(
        0.5, 0.5, "Initialising…",
        transform=ax_stat.transAxes,
        ha="center", va="center",
        fontsize=9.5, color="#8ab4d4",
        fontfamily="monospace",
    )

    legend_patches = [
        mpatches.Patch(color="#0d1117",  label="Unexplored"),
        mpatches.Patch(color="#1a4080",  label="Explored (low)"),
        mpatches.Patch(color="#00ffcc",  label="Explored (high visits)"),
        mpatches.Patch(color="#ff0055",  label="Death pit"),
        mpatches.Patch(color="#ff9900",  label="Teleport src"),
        mpatches.Patch(color="#ff3355",  label="Agent"),
        mpatches.Patch(color="#00e676",  label="Start"),
        mpatches.Patch(color="#00e5ff",  label="Goal"),
    ]
    ax_map.legend(handles=legend_patches, loc="upper right",
                  fontsize=6.5, facecolor="#0a0f18",
                  edgecolor="#1e2a3a", labelcolor="#c0ccd8",
                  ncol=2, framealpha=0.88)

    def update(_frame):
        s = SHARED.snapshot()

        # ── Update visit heat-map ─────────────────────────────────────────────
        vmap = s["visit_map"].copy()

        # Overlay pits and teleports as bright colours
        for (r, c) in s.get("known_pits", set()):
            if 0 <= r < GRID_SIZE and 0 <= c < GRID_SIZE:
                vmap[r, c] = 10.0    # bright red via clamp in cmap

        grid = s["agent_grid"]
        tp_mask = grid == 3.0
        vmap[tp_mask] = 8.0

        # Re-colour pits distinctly via a separate overlay
        pit_overlay = np.zeros((GRID_SIZE, GRID_SIZE, 4), dtype=np.float32)
        for (r, c) in s.get("known_pits", set()):
            if 0 <= r < GRID_SIZE and 0 <= c < GRID_SIZE:
                pit_overlay[r, c] = [1.0, 0.0, 0.2, 0.85]

        im_visit.set_data(np.clip(vmap, 0, 4))
        im_visit.set_clim(0, 4)

        # ── Path scatter ──────────────────────────────────────────────────────
        path = s["agent_path"]
        if len(path) > 1:
            plen = max(1, len(path) - 1)
            cols = np.array([i / plen for i in range(len(path))])
            xs   = [p[1] for p in path]
            ys   = [p[0] for p in path]
            path_scat.set_offsets(np.column_stack([xs, ys]))
            path_scat.set_array(cols)
            path_scat.set_sizes([10] * len(path))
        else:
            path_scat.set_offsets(np.empty((0, 2)))

        # ── Agent dot ─────────────────────────────────────────────────────────
        pr, pc = s["agent_pos"]
        agent_dot.set_offsets([[pc, pr]])

        # ── Fitness curves ────────────────────────────────────────────────────
        hist = s["history"]
        if len(hist) >= 1:
            gens  = [h["generation"] + 1 for h in hist]
            bests = [h["best"]           for h in hist]
            means = [h["mean"]           for h in hist]
            stds  = [h["std"]            for h in hist]
            wsts  = [h["worst"]          for h in hist]

            ln_best.set_data(gens, bests)
            ln_mean.set_data(gens, means)
            ln_worst.set_data(gens, wsts)

            # Redraw std fill
            fill_std[0].remove()
            lo = [m - sd for m, sd in zip(means, stds)]
            hi = [m + sd for m, sd in zip(means, stds)]
            fill_std[0] = ax_fit.fill_between(gens, lo, hi,
                                               color="#2979ff", alpha=0.12, zorder=0)

            ax_fit.relim(); ax_fit.autoscale_view()

        # ── Status bar ────────────────────────────────────────────────────────
        txt = s["status_text"]
        if s["done"]:
            txt = "✓  " + txt
        stat_txt.set_text(txt)

        return [im_visit, path_scat, agent_dot, ln_best, ln_mean, ln_worst, stat_txt]

    return update


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Live EC Maze Training Dashboard")
    parser.add_argument("--pop",      type=int,   default=60)
    parser.add_argument("--gens",     type=int,   default=80)
    parser.add_argument("--sigma",    type=float, default=0.15)
    parser.add_argument("--decay",    type=float, default=0.97)
    parser.add_argument("--eps_eval", type=int,   default=2)
    parser.add_argument("--turns",    type=int,   default=3000)
    parser.add_argument("--persist",  action="store_true")
    args = parser.parse_args()

    # Start training in background
    t = threading.Thread(target=training_thread, args=(args,), daemon=True)
    t.start()

    # Build figure and animation on main thread
    fig, ax_map, ax_fit, ax_stat = build_figure()
    update_fn = make_updater(fig, ax_map, ax_fit, ax_stat)

    fig.suptitle(
        "Silent Cartographer — Neuroevolution Live Training",
        color="#c8d0e0", fontsize=13, fontweight="bold", y=0.97,
    )

    anim = FuncAnimation(
        fig, update_fn,
        interval         = 400,     # ms between redraws
        blit             = False,
        cache_frame_data = False,
    )

    plt.show()
    print("Window closed — training thread will finish naturally.")


if __name__ == "__main__":
    main()
