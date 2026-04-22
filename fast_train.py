"""
train.py — Fast headless training for Silent Cartographer
COSC 4368 AI Spring 2026

Drops all live visualization. Parallelizes fitness evaluation across CPU cores.
Prints per-generation stats to console. Saves training curves to a PNG at the end.

Usage:
    python train.py --maze maze-alpha/MAZE_1.png --pop 80 --gens 100 --run_id dstar_v4
    python train.py --maze maze-alpha/MAZE_1.png --pop 80 --gens 100 --workers 8

Speed gains vs live_viz.py:
  - No queue/IPC overhead
  - No live replay episodes
  - Parallel eval: ~4-8x faster depending on CPU core count
"""
from __future__ import annotations
import argparse
import os
import shutil
import time
import random
import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — no display needed
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from datetime import datetime
from multiprocessing import Pool, cpu_count

import maze_agent as ma
from maze_agent import (NeuralController, GeneticAlgorithm,
                        evaluate_fitness, GRID_SIZE,
                        PHASE_EXPLORE, PHASE_OPTIMIZE)
from environment import MazeEnvironment


# ─────────────────────────────────────────────────────────────────────────────
# Worker function — must be top-level for multiprocessing pickle
# ─────────────────────────────────────────────────────────────────────────────
def _eval_worker(args):
    """Evaluate one individual. Runs in a separate process."""
    (flat_weights, layer_sizes, maze_path,
     goal_cell, start_cell, episodes, max_turns, epsilon, phase,
     phase_gen, transition_gens, seed_pits) = args

    # Suppress worker stdout (CNN loader spam)
    import sys as _sys, os as _os
    _sys.stdout = open(_os.devnull, 'w')

    try:
        env = MazeEnvironment(maze_path)
        env._phase_gen       = phase_gen
        env._transition_gens = transition_gens

        ctrl = NeuralController(layer_sizes)
        ctrl.set_flat_weights(flat_weights)

        fit, agent = evaluate_fitness(
            ctrl, env,
            goal_cell=goal_cell, start_cell=start_cell,
            episodes=episodes, max_turns=max_turns,
            epsilon=epsilon, persist=True,
            seed_pits=seed_pits,
            verbose=False, step_q=None,
            phase=phase,
        )
        stats = env.get_episode_stats()

        # Only return pit discoveries — walls are maze-specific and should be
        # rediscovered each episode. Accumulating walls across generations causes
        # D* Lite to cut off paths when pit approach-walls combine with real walls.
        discovered_pits = frozenset(agent.memory._shared_pits)

        return (fit, agent.goal_reached, stats["turns_taken"], stats["deaths"],
                len(agent.memory.visit_count),
                discovered_pits,
                getattr(agent, "total_wall_hits", 0))
    finally:
        _sys.stdout = _sys.__stdout__


# ─────────────────────────────────────────────────────────────────────────────
# Fast GA — parallelized fitness evaluation
# ─────────────────────────────────────────────────────────────────────────────
class FastGA(GeneticAlgorithm):

    def __init__(self, *args, workers=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.workers             = workers or max(1, cpu_count() - 1)
        self.gens_since_improvement = 0
        self.last_recorded_best     = -float("inf")
        # Collective pit memory — agents don't re-die at known pits each gen.
        # Walls are NOT shared: accumulating them cuts off D* Lite paths.
        self.shared_pits: frozenset = frozenset()

    def step(self, maze_path, goal_cell, start_cell,
             eval_episodes=1, eval_turns=10_000, epsilon=0.05):

        gen_num = self.generation + 1
        t0      = time.time()

        print(f"\n{'─'*60}")
        print(f"  Gen {gen_num:3d} START  σ={self.mut_sigma:.4f}"
              f"  pop={self.pop_size}  phase=[{self.phase}]"
              f"  workers={self.workers}")
        print(f"{'─'*60}")

        # Build argument list for parallel workers
        # Pass collective memory so every individual starts with known pit/wall locations
        worker_args = [
            (ctrl.get_flat_weights(), ctrl.layer_sizes, maze_path,
             goal_cell, start_cell, eval_episodes, eval_turns, epsilon,
             self.phase, self.phase_gen, self.transition_gens,
             self.shared_pits)
            for ctrl in self.population
        ]

        # Parallel evaluation
        with Pool(processes=self.workers) as pool:
            try:
                results = pool.map(_eval_worker, worker_args)
            except KeyboardInterrupt:
                pool.terminate()
                pool.join()
                raise

        # Unpack results and aggregate collective pit/wall knowledge
        gen_solver_turns  = []
        gen_solver_deaths = []
        gen_new_cells     = []
        gen_solvers       = 0
        all_pits          = set(self.shared_pits)
        all_turns_list    = []
        all_deaths_list   = []
        all_wall_hits_list= []

        for i, (fit, solved, turns, deaths, cells, new_pits, wall_hits) in enumerate(results):
            self.fitness[i] = fit
            gen_new_cells.append(cells)
            all_pits          |= new_pits
            all_turns_list.append(turns)
            all_deaths_list.append(deaths)
            all_wall_hits_list.append(wall_hits)
            if solved:
                gen_solvers += 1
                gen_solver_turns.append(turns)
                gen_solver_deaths.append(deaths)

        # Update collective pit memory only — walls stay ephemeral
        prev_pit_count = len(self.shared_pits)
        self.shared_pits = frozenset(all_pits)
        new_pit_count = len(self.shared_pits)
        if new_pit_count > prev_pit_count:
            print(f"  🗺 Collective memory: {new_pit_count} pits known"
                  f"  (+{new_pit_count - prev_pit_count} new)")

        avg_solve_turns  = int(sum(gen_solver_turns)  / len(gen_solver_turns))  if gen_solver_turns  else 0
        avg_solve_deaths = sum(gen_solver_deaths) / len(gen_solver_deaths) if gen_solver_deaths else 0.0
        gen_best_cells   = max(gen_new_cells) if gen_new_cells else 0
        avg_turns_all    = sum(all_turns_list)    / max(len(all_turns_list),    1)
        avg_deaths_all   = sum(all_deaths_list)   / max(len(all_deaths_list),   1)
        avg_wall_hits    = sum(all_wall_hits_list) / max(len(all_wall_hits_list), 1)
        elapsed          = time.time() - t0

        # Track best individual
        best_idx = int(np.argmax(self.fitness))
        tag = ""
        if self.fitness[best_idx] > self.best_fitness:
            self.best_fitness    = float(self.fitness[best_idx])
            self.best_individual = self.population[best_idx].clone()
            tag = "★ NEW BEST"

        rec = {
            "generation":       self.generation,
            "best":             float(self.fitness.max()),
            "mean":             float(self.fitness.mean()),
            "std":              float(self.fitness.std()),
            "worst":            float(self.fitness.min()),
            "sigma":            self.mut_sigma,
            "solvers":          gen_solvers,
            "new_cells":        gen_best_cells,
            "avg_solve_turns":  avg_solve_turns,
            "avg_solve_deaths": avg_solve_deaths,
            "avg_turns_all":    avg_turns_all,
            "avg_deaths_all":   avg_deaths_all,
            "avg_wall_hits":    avg_wall_hits,
            "success_rate":     gen_solvers / max(self.pop_size, 1),
            "phase":            self.phase,
            "elapsed_s":        elapsed,
        }
        self.history.append(rec)

        # Stagnation detection → sigma restart (optimize phase only)
        stagnation_tag = ""
        if rec["best"] > self.last_recorded_best + 100:
            self.last_recorded_best     = rec["best"]
            self.gens_since_improvement = 0
        else:
            self.gens_since_improvement += 1

        if self.gens_since_improvement >= 15 and self.phase == PHASE_OPTIMIZE:
            self.gens_since_improvement = 0
            old_sigma      = self.mut_sigma
            self.mut_sigma = min(0.20, self.mut_sigma * 2.5)
            stagnation_tag = f"🔄 σ {old_sigma:.4f}→{self.mut_sigma:.4f}"
            print(f"  🔄 STAGNATION RESTART  {stagnation_tag}")

        print(f"\n{'='*60}")
        print(f"  Gen {gen_num:3d} END  best={rec['best']:+.0f}"
              f"  mean={rec['mean']:+.0f}  std={rec['std']:.0f}"
              f"  σ={self.mut_sigma:.4f}  [{elapsed:.1f}s]")
        print(f"           solvers={gen_solvers}/{self.pop_size}"
              f"  avg_turns={avg_solve_turns}"
              f"  avg_deaths={avg_solve_deaths:.2f}"
              f"  cells={gen_best_cells}"
              f"  [{self.phase}]  {tag}")
        if stagnation_tag:
            print(f"           {stagnation_tag}")
        print(f"{'='*60}")

        # Phase switch check
        self._maybe_switch_phase(gen_solvers)

        # Build next generation
        sorted_idx  = np.argsort(self.fitness)[::-1]
        elite_k     = max(1, int(self.elite_frac * self.pop_size))
        new_pop     = [self.population[i].clone() for i in sorted_idx[:elite_k]]
        while len(new_pop) < self.pop_size:
            p1 = self.population[self._tournament_select()]
            if random.random() < self.crossover_prob:
                child = self._uniform_crossover(p1, self.population[self._tournament_select()])
            else:
                child = p1.clone()
            new_pop.append(self._mutate(child))

        # Random immigrants
        n_immigrants = max(1, int(self.immigrant_frac * self.pop_size))
        inject_start = max(elite_k, self.pop_size - n_immigrants)
        for k in range(self.pop_size - inject_start):
            new_pop[inject_start + k] = NeuralController(self.layer_sizes)

        self.population = new_pop
        self.mut_sigma  = max(self.min_mut_sigma, self.mut_sigma * self.mut_decay)
        self.generation += 1
        return self.best_individual



# ─────────────────────────────────────────────────────────────────────────────
# Metrics Report — Required + Bonus, PowerPoint-ready PNGs
# ─────────────────────────────────────────────────────────────────────────────
def save_metrics_report(history: list, run_id: str, pop_size: int = 80):
    """
    Generate two publication-quality metric charts for the project report.
      - metrics_required_{run_id}.png  (4 required metrics)
      - metrics_bonus_{run_id}.png     (4 bonus metrics)
    """
    if len(history) < 2:
        print("  [metrics] need ≥2 generations to plot — skipping")
        return

    gens   = [h["generation"] + 1           for h in history]
    phase  = [h.get("phase", "explore")     for h in history]
    n      = len(gens)

    # ── Raw series ──────────────────────────────────────────────────────────
    success_rate  = [h.get("success_rate",  h["solvers"] / pop_size) for h in history]
    solve_turns   = [h.get("avg_solve_turns",  0)   for h in history]
    death_rate    = [h.get("avg_deaths_all",
                           h.get("avg_solve_deaths", 0.0))           for h in history]
    path_len      = solve_turns   # turns ≈ path length in discrete grid

    cells         = [h.get("new_cells", 0)       for h in history]
    turns_all     = [max(h.get("avg_turns_all",
                              h.get("avg_solve_turns", 1)), 1)        for h in history]
    wall_hits     = [h.get("avg_wall_hits", 0.0) for h in history]
    best_fit      = [h["best"]                   for h in history]

    explore_eff   = [c / t for c, t in zip(cells, turns_all)]
    map_complete  = [c / 4096 * 100 for c in cells]
    replan_eff    = [max(0.0, (1.0 - w / t) * 100)
                     for w, t in zip(wall_hits, turns_all)]
    learn_rate    = []
    for i in range(n):
        prev  = best_fit[max(0, i - 5)]
        delta = (best_fit[i] - prev) / max(5, 1)
        learn_rate.append(max(-100, min(100, delta / 1000)))

    # ── Shared style helpers ─────────────────────────────────────────────────
    BG      = "#0d1117"
    PANEL   = "#161b22"
    GRID_C  = "#21262d"
    TEXT    = "#e6edf3"
    SUB     = "#8b949e"
    ACCENT  = ["#58a6ff", "#3fb950", "#ff7b72", "#ffa657",
               "#d2a8ff", "#79c0ff", "#56d364", "#f78166"]

    def _ax(ax, title, xlabel, ylabel, ylim=None):
        ax.set_facecolor(PANEL)
        ax.set_title(title, color=TEXT, fontsize=11, fontweight="bold", pad=8)
        ax.set_xlabel(xlabel, color=SUB, fontsize=8)
        ax.set_ylabel(ylabel, color=SUB, fontsize=8)
        ax.tick_params(colors=SUB, labelsize=8)
        ax.grid(alpha=0.20, color=GRID_C, linewidth=0.8)
        for sp in ax.spines.values():
            sp.set_edgecolor(GRID_C)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=10))
        if ylim is not None:
            ax.set_ylim(*ylim)

    def _phase_shade(ax, gens, phase):
        """Shade optimize-phase gens in a subtle gold tint."""
        in_opt = False
        start  = None
        for i, (g, p) in enumerate(zip(gens, phase)):
            if p == "optimize" and not in_opt:
                in_opt = True; start = g - 0.5
            elif p != "optimize" and in_opt:
                ax.axvspan(start, g - 0.5, alpha=0.06, color="#ffa657", zorder=0)
                in_opt = False
        if in_opt:
            ax.axvspan(start, gens[-1] + 0.5, alpha=0.06, color="#ffa657", zorder=0)

    def _smooth(vals, w=3):
        out = []
        for i in range(len(vals)):
            sl = vals[max(0, i-w+1):i+1]
            out.append(sum(sl)/len(sl))
        return out

    # ════════════════════════════════════════════════════════════════════════
    # Figure 1 — Required Metrics
    # ════════════════════════════════════════════════════════════════════════
    fig1, axes1 = plt.subplots(2, 2, figsize=(14, 8), facecolor=BG)
    fig1.suptitle(
        f"Silent Cartographer — Required Metrics  (run: {run_id})",
        color=TEXT, fontsize=13, fontweight="bold", y=0.98)
    fig1.subplots_adjust(hspace=0.38, wspace=0.32,
                         left=0.07, right=0.97, top=0.92, bottom=0.09)

    # ── 1. Success Rate ──────────────────────────────────────────────────────
    ax = axes1[0, 0]
    _ax(ax, "1. Success Rate", "Generation", "Rate (fraction)", ylim=(0, 1.05))
    _phase_shade(ax, gens, phase)
    ax.fill_between(gens, success_rate, alpha=0.15, color=ACCENT[0])
    ax.plot(gens, success_rate,          lw=0.6, color=ACCENT[0], alpha=0.4)
    ax.plot(gens, _smooth(success_rate), lw=2.2, color=ACCENT[0], label="Success rate")
    ax.axhline(1.0, color=GRID_C, lw=0.8, ls="--")
    # Annotate final value
    ax.annotate(f"{success_rate[-1]:.0%}",
                xy=(gens[-1], success_rate[-1]),
                xytext=(gens[-1]-max(1,n//6), success_rate[-1]+0.05),
                color=ACCENT[0], fontsize=9, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=ACCENT[0], lw=1.2))
    ax.legend(fontsize=8, facecolor=PANEL, edgecolor=GRID_C, labelcolor=TEXT)

    # ── 2. Average Path Length (≈ turns for solved episodes) ─────────────────
    ax = axes1[0, 1]
    _ax(ax, "2. Average Path Length (solved)", "Generation", "Turns")
    _phase_shade(ax, gens, phase)
    valid = [(g, t) for g, t in zip(gens, path_len) if t > 0]
    if valid:
        vg, vt = zip(*valid)
        ax.scatter(vg, vt, s=18, color=ACCENT[1], alpha=0.5, zorder=3)
        if len(vg) >= 3:
            ax.plot(vg, _smooth(list(vt), w=5), lw=2.2, color=ACCENT[1],
                    label="Avg path length (turns)")
        ax.legend(fontsize=8, facecolor=PANEL, edgecolor=GRID_C, labelcolor=TEXT)
    ax.text(0.5, 0.5, "No solvers yet" if not valid else "",
            transform=ax.transAxes, ha="center", va="center",
            color=SUB, fontsize=10, alpha=0.6)

    # ── 3. Average Turns to Solution ─────────────────────────────────────────
    ax = axes1[1, 0]
    _ax(ax, "3. Avg Turns to Solution", "Generation", "Turns")
    _phase_shade(ax, gens, phase)
    # All agents (background) + solvers (foreground)
    ax.fill_between(gens, turns_all, alpha=0.10, color=ACCENT[3])
    ax.plot(gens, turns_all, lw=1.0, color=ACCENT[3], ls="--",
            alpha=0.6, label="Avg all agents")
    if valid:
        vg, vt = zip(*valid)
        ax.plot(vg, _smooth(list(vt), w=3), lw=2.2, color=ACCENT[2],
                label="Avg solvers only")
    ax.legend(fontsize=8, facecolor=PANEL, edgecolor=GRID_C, labelcolor=TEXT)

    # ── 4. Death Rate ──────────────────────────────────────────────────────
    ax = axes1[1, 1]
    _ax(ax, "4. Death Rate (avg deaths / episode)", "Generation", "Deaths per episode")
    _phase_shade(ax, gens, phase)
    ax.fill_between(gens, death_rate, alpha=0.15, color=ACCENT[2])
    ax.plot(gens, death_rate,          lw=0.6, color=ACCENT[2], alpha=0.4)
    ax.plot(gens, _smooth(death_rate), lw=2.2, color=ACCENT[2], label="Death rate")
    ax.legend(fontsize=8, facecolor=PANEL, edgecolor=GRID_C, labelcolor=TEXT)

    out1 = f"metrics_required_{run_id}.png"
    fig1.savefig(out1, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig1)
    print(f"  📊 Required metrics → {out1}")

    # ════════════════════════════════════════════════════════════════════════
    # Figure 2 — Bonus Metrics
    # ════════════════════════════════════════════════════════════════════════
    fig2, axes2 = plt.subplots(2, 2, figsize=(14, 8), facecolor=BG)
    fig2.suptitle(
        f"Silent Cartographer — Bonus Metrics  (run: {run_id})",
        color=TEXT, fontsize=13, fontweight="bold", y=0.98)
    fig2.subplots_adjust(hspace=0.38, wspace=0.32,
                         left=0.07, right=0.97, top=0.92, bottom=0.09)

    # ── 5. Exploration Efficiency ──────────────────────────────────────────
    ax = axes2[0, 0]
    _ax(ax, "5. Exploration Efficiency", "Generation", "Unique cells / turn")
    _phase_shade(ax, gens, phase)
    ax.fill_between(gens, explore_eff, alpha=0.15, color=ACCENT[4])
    ax.plot(gens, explore_eff,          lw=0.6, color=ACCENT[4], alpha=0.4)
    ax.plot(gens, _smooth(explore_eff), lw=2.2, color=ACCENT[4],
            label="Cells per turn")
    ax.legend(fontsize=8, facecolor=PANEL, edgecolor=GRID_C, labelcolor=TEXT)

    # ── 6. Map Completeness ────────────────────────────────────────────────
    ax = axes2[0, 1]
    _ax(ax, "6. Map Completeness", "Generation", "% of 64×64 explored", ylim=(0, 105))
    _phase_shade(ax, gens, phase)
    ax.fill_between(gens, map_complete, alpha=0.15, color=ACCENT[5])
    ax.plot(gens, map_complete,          lw=0.6, color=ACCENT[5], alpha=0.4)
    ax.plot(gens, _smooth(map_complete), lw=2.2, color=ACCENT[5],
            label="Map complete %")
    ax.axhline(100, color=GRID_C, lw=0.8, ls="--")
    ax.legend(fontsize=8, facecolor=PANEL, edgecolor=GRID_C, labelcolor=TEXT)

    # ── 7. Replanning Efficiency ───────────────────────────────────────────
    ax = axes2[1, 0]
    _ax(ax, "7. Replanning Efficiency", "Generation",
        "% turns without wall hit", ylim=(0, 105))
    _phase_shade(ax, gens, phase)
    ax.fill_between(gens, replan_eff, alpha=0.15, color=ACCENT[6])
    ax.plot(gens, replan_eff,          lw=0.6, color=ACCENT[6], alpha=0.4)
    ax.plot(gens, _smooth(replan_eff), lw=2.2, color=ACCENT[6],
            label="Replan efficiency %")
    ax.axhline(100, color=GRID_C, lw=0.8, ls="--")
    ax.legend(fontsize=8, facecolor=PANEL, edgecolor=GRID_C, labelcolor=TEXT)

    # ── 8. Learning Efficiency ─────────────────────────────────────────────
    ax = axes2[1, 1]
    _ax(ax, "8. Learning Efficiency", "Generation",
        "Fitness Δ per gen (×10³, 5-gen rolling)")
    _phase_shade(ax, gens, phase)
    ax.axhline(0, color=GRID_C, lw=0.8, ls="--")
    pos = [max(0, v) for v in learn_rate]
    neg = [min(0, v) for v in learn_rate]
    ax.fill_between(gens, pos, alpha=0.25, color=ACCENT[1])
    ax.fill_between(gens, neg, alpha=0.25, color=ACCENT[2])
    ax.plot(gens, learn_rate, lw=2.0, color=ACCENT[7], label="Learning rate")
    ax.legend(fontsize=8, facecolor=PANEL, edgecolor=GRID_C, labelcolor=TEXT)

    out2 = f"metrics_bonus_{run_id}.png"
    fig2.savefig(out2, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig2)
    print(f"  📊 Bonus metrics    → {out2}")

    # ── Print summary table ────────────────────────────────────────────────
    final = history[-1]
    best_solve = min((h["avg_solve_turns"] for h in history if h["avg_solve_turns"] > 0),
                     default=0)
    print(f"""
  ┌─────────────────────────────────────────────┐
  │  Metrics Summary  ({run_id})
  ├─────────────────────────────────────────────┤
  │  Success rate    : {final.get("success_rate", final["solvers"]/pop_size):.1%}  (final gen)
  │  Best solve      : {best_solve} turns
  │  Avg path length : {final.get("avg_solve_turns", 0)} turns (final gen)
  │  Death rate      : {final.get("avg_deaths_all", 0):.2f} deaths/episode
  │  Map complete    : {final.get("new_cells", 0)/4096*100:.1f}%
  │  Replan eff      : {max(0,(1-final.get("avg_wall_hits",0)/max(final.get("avg_turns_all",1),1))*100):.1f}%
  └─────────────────────────────────────────────┘""")


# ─────────────────────────────────────────────────────────────────────────────
# Plot training history
# ─────────────────────────────────────────────────────────────────────────────
def save_training_plot(history, run_id, out_path):
    if not history:
        return

    gens     = [h["generation"] + 1 for h in history]
    best_    = [h["best"]             for h in history]
    mean_    = [h["mean"]             for h in history]
    worst_   = [h["worst"]            for h in history]
    std_     = [h["std"]              for h in history]
    solvers_ = [h["solvers"]          for h in history]
    sigma_   = [h["sigma"]            for h in history]
    cells_   = [h.get("new_cells", 0) for h in history]
    turns_   = [h.get("avg_solve_turns", 0)   for h in history]
    deaths_  = [h.get("avg_solve_deaths", 0.0) for h in history]
    elapsed_ = [h.get("elapsed_s", 0) for h in history]

    plt.style.use("dark_background")
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), facecolor="#0a0f18")
    fig.suptitle(f"Silent Cartographer — Training Run: {run_id}",
                 color="#c8d8e8", fontsize=13, fontweight="bold")

    def _style(ax, title, xlabel, ylabel):
        ax.set_facecolor("#060b12")
        ax.set_title(title, color="#c8d0e0", fontsize=10, pad=4)
        ax.set_xlabel(xlabel, color="#7a8a9a", fontsize=8)
        ax.set_ylabel(ylabel, color="#7a8a9a", fontsize=8)
        ax.tick_params(colors="#4a5a6a", labelsize=7)
        ax.grid(alpha=0.12, color="#2a3a4a")
        for sp in ax.spines.values(): sp.set_edgecolor("#1e2a3a")
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    # ── Panel 1: Population Fitness ──
    ax = axes[0, 0]
    _style(ax, "Population Fitness", "Generation", "Fitness")
    lo = [m - s for m, s in zip(mean_, std_)]
    hi = [m + s for m, s in zip(mean_, std_)]
    ax.fill_between(gens, lo, hi, color="#2979ff", alpha=0.12)
    ax.plot(gens, worst_, lw=0.8, color="#455a6a", ls="--", label="Worst")
    ax.plot(gens, mean_,  lw=1.5, color="#2979ff", label="Mean")
    ax.plot(gens, best_,  lw=2.2, color="#00e5ff", label="Best")
    ax.axhline(0, color="#2a4060", lw=0.8, ls="--")
    # Mark phase switch and sigma restarts
    for i, h in enumerate(history):
        if h.get("phase") == PHASE_OPTIMIZE and (i == 0 or history[i-1].get("phase") == PHASE_EXPLORE):
            ax.axvline(gens[i], color="#ffaa00", lw=1.2, ls="--", alpha=0.7)
            ax.text(gens[i] + 0.3, ax.get_ylim()[1] * 0.95, "⚡", color="#ffaa00", fontsize=8)
    for i in range(1, len(sigma_)):
        if sigma_[i] > sigma_[i-1] * 1.5:
            ax.axvline(gens[i], color="#ff6699", lw=0.8, ls=":", alpha=0.6)
    ax.legend(loc="lower right", fontsize=7, facecolor="#0a1520",
              edgecolor="#1e2a3a", labelcolor="#c8d0e0")

    # ── Panel 2: Solvers + Sigma ──
    ax = axes[0, 1]
    _style(ax, "Solvers / Gen  +  σ", "Generation", "Solvers")
    colors = ["#00e5aa" if s > 0 else "#1e3a2a" for s in solvers_]
    ax.bar(gens, solvers_, color=colors, alpha=0.75, width=0.7)
    for g, s in zip(gens, solvers_):
        if s > 0:
            ax.text(g, s + 0.15, str(s), ha="center", va="bottom",
                    fontsize=5.5, color="#00e5aa")
    ax2 = ax.twinx()
    ax2.plot(gens, sigma_, lw=1.5, color="#ffaa44", ls="--")
    ax2.set_ylabel("σ", color="#ffaa44", fontsize=8)
    ax2.tick_params(colors="#4a5a6a", labelsize=7)
    ax2.set_ylim(0, max(max(sigma_) * 1.2, 0.05))

    # ── Panel 3: Solve Speed + Deaths ──
    ax = axes[1, 0]
    _style(ax, "Avg Solve Turns  +  Avg Deaths", "Generation", "Avg Turns")
    ax.plot(gens, turns_, lw=1.8, color="#00e5ff", label="Avg turns")
    ax3 = ax.twinx()
    ax3.plot(gens, deaths_, lw=1.5, color="#ff6666", ls="--", label="Avg deaths")
    ax3.set_ylabel("Avg Deaths", color="#ff6666", fontsize=8)
    ax3.tick_params(colors="#4a5a6a", labelsize=7)
    ax.legend(loc="upper right", fontsize=7, facecolor="#0a1520",
              edgecolor="#1e2a3a", labelcolor="#c8d0e0")

    # ── Panel 4: Cells Explored + Gen Time ──
    ax = axes[1, 1]
    _style(ax, "Cells Explored  +  Gen Time (s)", "Generation", "Cells")
    ax.plot(gens, cells_, lw=1.8, color="#44bbff", label="Cells explored")
    ax4 = ax.twinx()
    ax4.plot(gens, elapsed_, lw=1.2, color="#aaaaaa", ls=":", label="Gen time (s)")
    ax4.set_ylabel("Time (s)", color="#aaaaaa", fontsize=8)
    ax4.tick_params(colors="#4a5a6a", labelsize=7)
    ax.legend(loc="upper left", fontsize=7, facecolor="#0a1520",
              edgecolor="#1e2a3a", labelcolor="#c8d0e0")

    plt.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight", facecolor="#0a0f18")
    plt.close(fig)
    print(f"\n  📊 Training plot saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description="Silent Cartographer — Fast Headless Training")
    p.add_argument("--maze",     default="maze-alpha/MAZE_1.png")
    p.add_argument("--turns",    type=int,   default=10_000)
    p.add_argument("--pop",      type=int,   default=80)
    p.add_argument("--gens",     type=int,   default=100)
    p.add_argument("--sigma",    type=float, default=0.20)
    p.add_argument("--decay",    type=float, default=0.995)
    p.add_argument("--eps_eval", type=int,   default=3)
    p.add_argument("--phase_k",  type=int,   default=3)
    p.add_argument("--workers",  type=int,   default=None,
                   help="Parallel workers (default: CPU count - 1)")
    p.add_argument("--run_id",   default=None)
    args = p.parse_args()

    # Configure maze
    env0 = MazeEnvironment(args.maze)
    ma.configure(env0.start_cell, env0.goal_cell, GRID_SIZE)
    print(f"[TRAIN] maze={args.maze}")
    print(f"[TRAIN] start={env0.start_cell}  goal={env0.goal_cell}")

    run_id       = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    weights_file = f"weights_{run_id}.npy"
    alias_file   = f"best_weights_{run_id}.npy"
    plot_file    = f"training_{run_id}.png"

    workers = args.workers or max(1, cpu_count() - 1)
    print(f"[TRAIN] pop={args.pop}  gens={args.gens}  workers={workers}")
    print(f"[TRAIN] weights → {weights_file}\n")

    ga = FastGA(
        pop_size        = args.pop,
        init_mut_sigma  = args.sigma,
        mut_decay       = args.decay,
        phase_switch_k  = args.phase_k,
        immigrant_frac  = 0.05,
        workers         = workers,
    )

    print(f"[GA]  {ga.pop_size} individuals × {ga.population[0].num_params} params\n")

    epsilon   = 0.70
    t_total   = time.time()

    try:
        for gen_i in range(args.gens):
            ga.step(
                maze_path    = args.maze,
                goal_cell    = ma.GOAL_CELL,
                start_cell   = ma.START_CELL,
                eval_episodes= args.eps_eval,
                eval_turns   = args.turns,
                epsilon      = epsilon,
            )
            epsilon = max(0.05, epsilon * 0.94)

            if ga.best_individual:
                ga.best_individual.save(weights_file)
                shutil.copy2(weights_file, alias_file)

    except KeyboardInterrupt:
        print(f"\n\n  ⚠ Interrupted at gen {ga.generation}  —  saving current best...")
        if ga.best_individual:
            ga.best_individual.save(weights_file)
            shutil.copy2(weights_file, alias_file)
            print(f"  ✓ Weights saved → {weights_file}")
        print(f"  Best fitness so far: {ga.best_fitness:+.0f}\n")
        if ga.history:
            save_training_plot(ga.history, run_id, plot_file)
            save_metrics_report(ga.history, run_id, pop_size=args.pop)
        import os as _os; _os.kill(_os.getpid(), 9)  # force-kill all workers

    total_time = time.time() - t_total
    print(f"\n{'━'*60}")
    completed = ga.generation
    print(f"  {'✓ Training complete' if completed == args.gens else f'⚠ Stopped at gen {completed}/{args.gens}'}  ({total_time/60:.1f} min)")
    print(f"  Best fitness : {ga.best_fitness:+.0f}")
    print(f"  Weights      : {weights_file}")
    print(f"{'━'*60}\n")

    if ga.history:
        save_training_plot(ga.history, run_id, plot_file)
        save_metrics_report(ga.history, run_id, pop_size=args.pop)


if __name__ == "__main__":
    # Required for multiprocessing on Windows
    from multiprocessing import freeze_support
    freeze_support()
    main()