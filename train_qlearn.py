"""
train_qlearn.py — Fast CLI training for D* Lite + Q-Learning agent
Silent Cartographer: COSC 4368 AI Spring 2026

Usage:
    python train_qlearn.py --maze maze-alpha/MAZE_1.png --episodes 1500 --run_id qlearn_v1

Test afterwards:
    python live_viz.py --test --maze maze-beta/MAZE_1.png --weights weights_qlearn_v1.npy
"""
from __future__ import annotations
import argparse
import shutil
import numpy as np
import sys
import matplotlib
# Agg is a file-only backend — only use it for headless training, not interactive test
if '--test' not in sys.argv:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from datetime import datetime

from qlearn_agent import train_qlearning


def save_plot(history: list, run_id: str, out_path: str):
    if not history:
        return

    eps      = [h["episode"]      for h in history]
    solved_  = [int(h["goal_reached"]) for h in history]
    turns_   = [h["turns"]        for h in history]
    deaths_  = [h["deaths"]       for h in history]
    cells_   = [h["cells"]        for h in history]
    epsilon_ = [h["epsilon"]      for h in history]

    # Smooth with rolling window for readability
    def smooth(vals, w=50):
        out = []
        for i in range(len(vals)):
            start = max(0, i - w + 1)
            out.append(sum(vals[start:i+1]) / (i - start + 1))
        return out

    plt.style.use("dark_background")
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), facecolor="#0a0f18")
    fig.suptitle(f"Silent Cartographer — Q-Learning Run: {run_id}",
                 color="#c8d8e8", fontsize=13, fontweight="bold")

    def _style(ax, title, xlabel, ylabel):
        ax.set_facecolor("#060b12")
        ax.set_title(title, color="#c8d0e0", fontsize=10, pad=4)
        ax.set_xlabel(xlabel, color="#7a8a9a", fontsize=8)
        ax.set_ylabel(ylabel, color="#7a8a9a", fontsize=8)
        ax.tick_params(colors="#4a5a6a", labelsize=7)
        ax.grid(alpha=0.12, color="#2a3a4a")
        for sp in ax.spines.values(): sp.set_edgecolor("#1e2a3a")

    # Panel 1: Solve rate (rolling)
    ax = axes[0, 0]
    _style(ax, "Rolling Solve Rate (window=50)", "Episode", "Solve Rate")
    ax.plot(eps, smooth(solved_), lw=2.0, color="#00e5ff", label="Solved (smoothed)")
    ax.plot(eps, solved_, lw=0.4, color="#00e5ff", alpha=0.2)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7, facecolor="#0a1520", edgecolor="#1e2a3a", labelcolor="#c8d0e0")

    # Panel 2: Episode turns + epsilon
    ax = axes[0, 1]
    _style(ax, "Episode Turns  +  ε", "Episode", "Turns")
    ax.plot(eps, smooth(turns_), lw=1.8, color="#00e5ff", label="Turns (smoothed)")
    ax.plot(eps, turns_, lw=0.3, color="#00e5ff", alpha=0.15)
    ax2 = ax.twinx()
    ax2.plot(eps, epsilon_, lw=1.5, color="#ffaa44", ls="--", label="ε")
    ax2.set_ylabel("ε", color="#ffaa44", fontsize=8)
    ax2.tick_params(colors="#4a5a6a", labelsize=7)
    ax.legend(loc="upper right", fontsize=7, facecolor="#0a1520",
              edgecolor="#1e2a3a", labelcolor="#c8d0e0")

    # Panel 3: Deaths per episode (rolling)
    ax = axes[1, 0]
    _style(ax, "Avg Deaths (rolling)", "Episode", "Deaths")
    ax.plot(eps, smooth(deaths_), lw=1.8, color="#ff6666", label="Deaths (smoothed)")
    ax.plot(eps, deaths_, lw=0.3, color="#ff6666", alpha=0.15)
    ax.legend(fontsize=7, facecolor="#0a1520", edgecolor="#1e2a3a", labelcolor="#c8d0e0")

    # Panel 4: Cells explored per episode
    ax = axes[1, 1]
    _style(ax, "Cells Explored / Episode", "Episode", "Cells")
    ax.plot(eps, smooth(cells_), lw=1.8, color="#44bbff", label="Cells (smoothed)")
    ax.plot(eps, cells_, lw=0.3, color="#44bbff", alpha=0.15)
    ax.legend(fontsize=7, facecolor="#0a1520", edgecolor="#1e2a3a", labelcolor="#c8d0e0")

    plt.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight", facecolor="#0a0f18")
    plt.close(fig)
    print(f"  📊 Plot saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Test runner — uses LiveVisualizer with QLearningController
# ─────────────────────────────────────────────────────────────────────────────
def run_test(args):
    import maze_agent as ma
    from environment import MazeEnvironment
    from qlearn_agent import QLearningController, QLearningAgent
    from visualize import LiveVisualizer

    env = MazeEnvironment(args.maze)
    ma.configure(env.start_cell, env.goal_cell)

    ctrl = QLearningController()
    try:
        ctrl.load(args.weights)
        print(f"[TEST] loaded {args.weights}  ({ctrl.num_params} params)")
    except Exception as e:
        print(f"[ERROR] {e}"); return

    print(f"[TEST] maze={args.maze}  start={env.start_cell}  goal={env.goal_cell}")
    print(f"[TEST] {args.test_episodes} episodes  eps=0  (frozen weights)\n")

    agent = QLearningAgent(ctrl, env.goal_cell, env.start_cell,
                           epsilon=0.0, alpha=0.0)
    agent.freeze()
    agent.env = env

    viz = LiveVisualizer(env,
                         title=f"Q-Learning Test — {args.weights}",
                         update_every=5, delay=0.02)

    # Collective memory across test episodes
    test_known_pits:  set = set()
    test_known_walls: set = set()

    for ep in range(1, args.test_episodes + 1):
        agent.reset_episode()
        agent.memory._shared_pits  = set(test_known_pits)
        agent.memory._shared_walls = set(test_known_walls)

        sp = env.reset()
        agent.current_pos = sp
        agent._last_dist  = (abs(env.goal_cell[0]-sp[0]) +
                             abs(env.goal_cell[1]-sp[1]))
        agent._best_dist_ever = agent._last_dist

        viz.reset_episode(env)

        last_result  = None
        turns = deaths = 0
        goal_reached   = False

        while turns < args.turns:
            actions     = agent.plan_turn(last_result)
            last_result = env.step(actions)
            turns      += 1

            if last_result.is_dead:   deaths += 1
            if last_result.is_goal_reached:
                goal_reached = True
                break

            # Build known-hazards dict for visualizer
            known = {(r, c): 'death' for (r, c) in agent.memory._shared_pits}
            for src in agent.memory.known_teleports:
                known[src] = 'teleport'

            viz.update(
                known       = known,
                current_pos = agent.current_pos,
                path        = list(agent.memory.path[-500:]),
                episode     = ep,
                turn        = turns,
                goal_pos    = env.goal_cell,
                start_pos   = env.start_cell,
                extra_stats = {
                    "Deaths":      deaths,
                    "Turns":       turns,
                    "Known pits":  len(agent.memory._shared_pits),
                    "Status":      "✓ SOLVED" if goal_reached else "Running…",
                },
            )

        test_known_pits  |= agent.memory._shared_pits
        test_known_walls |= agent.memory._shared_walls

        tag = "✓ SOLVED" if goal_reached else "✗ TIMEOUT"
        print(f"  Ep {ep}/{args.test_episodes}  [{tag}]"
              f"  turns={turns}  deaths={deaths}"
              f"  known_pits={len(test_known_pits)}")

    viz.close()


def main():
    p = argparse.ArgumentParser(description="D* Lite + Q-Learning training")
    p.add_argument("--maze",          default="maze-alpha/MAZE_1.png")
    p.add_argument("--test",          action="store_true",
                   help="Run test mode using LiveVisualizer")
    p.add_argument("--weights",       default="best_weights.npy",
                   help="Weights file for --test mode")
    p.add_argument("--test_episodes", type=int, default=5)
    p.add_argument("--episodes",      type=int,   default=3000)
    p.add_argument("--turns",         type=int,   default=5000)
    p.add_argument("--alpha",         type=float, default=0.001,
                   help="Learning rate")
    p.add_argument("--gamma",         type=float, default=0.999,
                   help="Discount factor (high = long-horizon credit)")
    p.add_argument("--lambda_",       type=float, default=0.99,
                   help="Eligibility trace decay (high = deeper credit assignment)")
    p.add_argument("--eps_start",     type=float, default=1.0)
    p.add_argument("--eps_end",       type=float, default=0.05)
    p.add_argument("--eps_decay",     type=float, default=0.9998,
                   help="Slow decay — maintain D* Lite exploration for longer")
    p.add_argument("--stall_window",  type=int,   default=0,
                   help="Early stop turns (0=disabled — TD-lambda needs full episodes)")
    p.add_argument("--verbose_every", type=int,   default=100)
    p.add_argument("--run_id",        default=None)
    args = p.parse_args()

    if args.test:
        run_test(args)
        return

    run_id       = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    weights_file = f"weights_{run_id}.npy"
    alias_file   = f"best_weights_{run_id}.npy"
    plot_file    = f"training_{run_id}.png"

    try:
        ctrl, history = train_qlearning(
            maze_path     = args.maze,
            n_episodes    = args.episodes,
            max_turns     = args.turns,
            alpha         = args.alpha,
            gamma         = args.gamma,
            lambda_       = args.lambda_,
            epsilon_start = args.eps_start,
            epsilon_end   = args.eps_end,
            epsilon_decay = args.eps_decay,
            stall_window  = args.stall_window,
            verbose_every = args.verbose_every,
            weights_path  = weights_file,
        )
    except KeyboardInterrupt:
        print("\n\n  ⚠ Interrupted — saving current weights...")
        # ctrl may not be bound if interrupted very early
        print("  Use --run_id to resume from saved weights.")
        return

    shutil.copy2(weights_file, alias_file)
    save_plot(history, run_id, plot_file)
    print(f"\nTo test:  python live_viz.py --test --maze <maze> --weights {weights_file}")


if __name__ == "__main__":
    main()