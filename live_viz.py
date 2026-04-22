"""
live_viz.py — D* Lite + Neuroevolution dashboard
Silent Cartographer: COSC 4368 AI Spring 2026

Run:
    python live_viz.py --maze maze-alpha/MAZE_1.png [--pop 80] [--gens 100]
"""
from __future__ import annotations
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
from matplotlib.animation import FuncAnimation
from multiprocessing import Process, Queue
import queue as _queue

from environment import Action, TurnResult, MazeEnvironment
from visualize import LiveVisualizer
import maze_agent as ma
from maze_agent import (NeuralController, EvolutionaryAgent,
                        GeneticAlgorithm, evaluate_fitness, GRID_SIZE,
                        PHASE_EXPLORE, PHASE_OPTIMIZE, NAVIGATOR)

CELL_SIZE = 16
MARK      = 5

def _get_fire_rotation_states(env):
    """
    Return the 4 fire-rotation state sets for an environment.
    Works with both:
      - environment.py  (has _fire_rotation_states pre-computed)
      - hazardDemo.py   (only has rotate_fire_clusters(); we compute manually)
    """
    if hasattr(env, "_fire_rotation_states"):
        return list(env._fire_rotation_states)
    # Compute by stepping through 4 rotations, then restore original state
    saved_pits     = set(env.death_pits)
    saved_clusters = [list(c) for c in env.fire_clusters]
    saved_pivots   = list(env.fire_cluster_pivots)
    states = []
    for _ in range(4):
        states.append(set(env.death_pits))
        if hasattr(env, "rotate_fire_clusters"):
            env.rotate_fire_clusters()
    # Restore
    env.death_pits           = saved_pits
    env.fire_clusters        = saved_clusters
    env.fire_cluster_pivots  = saved_pivots
    for r, c in env.death_pits:
        if env.is_cell_in_bounds(r, c):
            env.grid[r][c] = True
    return states


# ─────────────────────────────────────────────────────────────────────────────
# Training worker
# ─────────────────────────────────────────────────────────────────────────────
def training_worker(maze_path: str, args_dict: dict, state_q: Queue):
    import maze_agent as ma
    from environment import MazeEnvironment
    from maze_agent import (NeuralController, GeneticAlgorithm,
                             evaluate_fitness, GRID_SIZE,
                             PHASE_EXPLORE, PHASE_OPTIMIZE)

    env = MazeEnvironment(maze_path)
    ma.configure(env.start_cell, env.goal_cell, GRID_SIZE)
    print(f"[TRAIN] start={env.start_cell}  goal={env.goal_cell}"
          f"  navigator={ma.NAVIGATOR}")

    # Send display metadata to main process — eliminates the need for
    # MazeEnvironment (and CNN) in the display process entirely.
    try:
        state_q.put({
            "type":             "init",
            "start_cell":       list(env.start_cell),
            "goal_cell":        list(env.goal_cell),
            "fire_states":      [list(map(list, s)) for s in _get_fire_rotation_states(env)],
            "teleport_purple":  list(map(list, env.loader.teleport_purple)),
            "teleport_orange":  list(map(list, env.loader.teleport_orange)),
            "teleport_green":   list(map(list, env.loader.teleport_green)),
            "teleport_red":     list(map(list, getattr(env.loader, "teleport_red", []))),
            "cell_size":        env.CELL_SIZE,
            "fire_rot_idx":     getattr(env, "_fire_rot_idx", 0),
        })
    except Exception:
        pass

    class _VisualGA(GeneticAlgorithm):
        def step(self, env, eval_episodes=1, eval_turns=10_000,
                 epsilon=0.05, persist=False):
            import random, numpy as np
            gen_num            = self.generation + 1
            best_agent         = None
            gen_solvers        = 0
            gen_solver_turns   = []
            gen_solver_deaths  = []
            gen_best_new_cells = 0

            print(f"\n{'─'*60}")
            print(f"  Gen {gen_num:3d} START  σ={self.mut_sigma:.4f}"
                  f"  pop={self.pop_size}  phase=[{self.phase}]"
                  f"  nav={ma.NAVIGATOR}")
            print(f"{'─'*60}")

            gen_new_cells_list  = []
            gen_wall_hits_list  = []
            gen_turns_list      = []
            # Collective pit memory: once any agent dies on a fire pit, all
            # subsequent agents this generation start knowing about it.
            # Knowledge still comes from real deaths — spec compliant (SS4.1.2).
            gen_shared_pits:       set  = set(self.persistent_pits)
            gen_shared_walls:      set  = set(self.persistent_walls)
            gen_shared_teleports:  dict = dict(self.persistent_teleports)

            for i, ctrl in enumerate(self.population):
                _is_immigrant = i in getattr(self, "_immigrant_indices", set())
                fit, ep_agent = evaluate_fitness(
                    ctrl, env,
                    goal_cell=ma.GOAL_CELL, start_cell=ma.START_CELL,
                    episodes=eval_episodes, max_turns=eval_turns,
                    epsilon=epsilon, persist=persist,
                    seed_pits=gen_shared_pits,
                    seed_walls=gen_shared_walls,
                    seed_teleports=None if _is_immigrant else gen_shared_teleports,
                    phase=self.phase, step_q=None,
                )
                self.fitness[i] = fit
                agent_new = len(ep_agent.memory.visit_count)
                gen_new_cells_list.append(agent_new)
                gen_wall_hits_list.append(getattr(ep_agent, 'total_wall_hits', 0))
                gen_turns_list.append(getattr(ep_agent, 'total_turns', eval_turns))

                # Accumulate pit AND wall discoveries for remaining agents
                if persist and hasattr(ep_agent.memory, '_shared_pits'):
                    gen_shared_pits        |= ep_agent.memory._shared_pits
                    gen_shared_walls       |= ep_agent.memory._shared_walls
                    gen_shared_teleports.update(ep_agent.memory._shared_teleports)
                else:
                    gen_shared_pits  |= ep_agent.memory.known_pits
                    gen_shared_walls |= ep_agent.memory.known_walls
                    gen_shared_teleports.update(ep_agent.memory.known_teleports)

                # Cumulative map: every cell ever visited by any agent, ever
                self.cumulative_cell_set.update(ep_agent.memory.visit_count.keys())

                if ep_agent.goal_reached:
                    gen_solvers += 1
                    stats = env.get_episode_stats()
                    gen_solver_turns.append(stats["turns_taken"])
                    gen_solver_deaths.append(stats["deaths"])

                if best_agent is None or fit >= self.fitness[:i+1].max():
                    best_agent = ep_agent

                if (i+1) % 5 == 0 or (i+1) == self.pop_size:
                    avg_t_all = int(sum(gen_turns_list[:i+1]) / (i+1))
                    avg_t_sol = (int(sum(gen_solver_turns) / len(gen_solver_turns))
                                 if gen_solver_turns else None)
                    avg_d = (sum(gen_solver_deaths) / len(gen_solver_deaths)
                             if gen_solver_deaths else 0.0)
                    turns_str = (f"avg_turns={avg_t_all}"
                                 + (f"(solved={avg_t_sol})" if avg_t_sol else ""))
                    print(f"  [{i+1:3d}/{self.pop_size}]"
                          f"  best={self.fitness[:i+1].max():+.0f}"
                          f"  mean={self.fitness[:i+1].mean():+.0f}"
                          f"  std={self.fitness[:i+1].std():.0f}"
                          f"  solvers={gen_solvers}"
                          f"  {turns_str}"
                          f"  avg_deaths={avg_d:.2f}"
                          f"  map_cells={len(self.cumulative_cell_set)}"
                          f"  [{self.phase}]")

            gen_best_new_cells = len(self.cumulative_cell_set)
            avg_solve_turns    = (int(sum(gen_solver_turns) / len(gen_solver_turns))
                                  if gen_solver_turns else 0)
            avg_solve_deaths   = (sum(gen_solver_deaths) / len(gen_solver_deaths)
                                  if gen_solver_deaths else 0.0)
            avg_wall_hits      = (sum(gen_wall_hits_list) / len(gen_wall_hits_list)
                                  if gen_wall_hits_list else 0.0)
            avg_turns_all      = (sum(gen_turns_list) / len(gen_turns_list)
                                  if gen_turns_list else eval_turns)

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
                "new_cells":        gen_best_new_cells,
                "avg_solve_turns":  avg_solve_turns,
                "avg_solve_deaths": avg_solve_deaths,
                "avg_wall_hits":    avg_wall_hits,
                "avg_turns_all":    avg_turns_all,
                "phase":            self.phase,
            }
            self.history.append(rec)

            # ── Stagnation detection → sigma restart ─────────────────────────
            # Only active in optimize phase — stagnation in explore phase is
            # expected while the agent is still finding the goal.
            stagnation_tag = ""
            if rec["best"] > self.last_recorded_best + 100:
                self.last_recorded_best     = rec["best"]
                self.gens_since_improvement = 0
            else:
                self.gens_since_improvement += 1

            if (self.gens_since_improvement >= 15 and
                    self.phase == PHASE_OPTIMIZE):
                self.gens_since_improvement = 0
                old_sigma      = self.mut_sigma
                self.mut_sigma = min(0.20, self.mut_sigma * 2.5)
                stagnation_tag = (f"🔄 SIGMA RESTART "
                                  f"{old_sigma:.4f}→{self.mut_sigma:.4f}")
                print(f"\n  {'━'*44}\n"
                      f"  🔄 STAGNATION RESTART\n"
                      f"     σ {old_sigma:.4f} → {self.mut_sigma:.4f}\n"
                      f"     no improvement for 15 gens\n"
                      f"  {'━'*44}\n")

            print(f"{'='*60}")
            print(f"  Gen {gen_num:3d} END  best={rec['best']:+.0f}"
                  f"  mean={rec['mean']:+.0f}  std={rec['std']:.0f}"
                  f"  σ={self.mut_sigma:.4f}")
            avg_turns_display = f"avg_turns={int(avg_turns_all)}"
            if avg_solve_turns > 0:
                avg_turns_display += f"  solved_avg={avg_solve_turns}"
            print(f"           solvers={gen_solvers}/{self.pop_size}"
                  f"  {avg_turns_display}"
                  f"  avg_deaths={avg_solve_deaths:.2f}"
                  f"  map_cells={gen_best_new_cells}"
                  f"  [{self.phase}]  {tag}")
            if stagnation_tag:
                print(f"           {stagnation_tag}")
            print(f"{'='*60}\n")

            if best_agent is not None:
                mem  = best_agent.memory
                vmap = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
                for (r, c), cnt in mem.visit_count.items():
                    if 0 <= r < GRID_SIZE and 0 <= c < GRID_SIZE:
                        vmap[r, c] = np.log1p(cnt)
            else:
                vmap = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)

            phase_label = "🔍 EXPLORE" if self.phase == PHASE_EXPLORE else "⚡ OPTIMIZE"
            try:
                state_q.put_nowait({
                    "type":       "gen",
                    "visit_map":  vmap,
                    "agent_pos":  best_agent.current_pos if best_agent else ma.START_CELL,
                    "agent_path": list(best_agent.memory.path[-5000:]) if best_agent else [],
                    "history":    list(self.history),
                    "status":     (f"Gen {gen_num}  |  best={rec['best']:+.0f}"
                                   f"  mean={rec['mean']:+.0f}"
                                   f"  σ={self.mut_sigma:.4f}"
                                   f"  solvers={gen_solvers}/{self.pop_size}"
                                   f"  avg_t={int(avg_turns_all)}"
                                   f"  avg_d={avg_solve_deaths:.1f}"
                                   f"  {phase_label}"
                                   + (f"  {stagnation_tag}" if stagnation_tag else "")
                                   + (f"  {tag}" if tag else "")),
                    "done": False,
                })
            except _queue.Full:
                pass

            self.persistent_pits       = set(gen_shared_pits)
            self.persistent_walls      = set(gen_shared_walls)
            self.persistent_teleports  = dict(gen_shared_teleports)

            self._maybe_switch_phase(gen_solvers, env=env)

            sorted_idx  = np.argsort(self.fitness)[::-1]
            elite_k     = max(1, int(self.elite_frac * self.pop_size))
            new_pop     = [self.population[i].clone() for i in sorted_idx[:elite_k]]
            while len(new_pop) < self.pop_size:
                p1 = self.population[self._tournament_select()]
                if random.random() < self.crossover_prob:
                    child = self._uniform_crossover(
                        p1, self.population[self._tournament_select()])
                else:
                    child = p1.clone()
                new_pop.append(self._mutate(child))

            n_immigrants = max(1, int(self.immigrant_frac * self.pop_size))
            inject_start = max(elite_k, self.pop_size - n_immigrants)
            self._immigrant_indices = set()
            for k in range(self.pop_size - inject_start):
                new_pop[inject_start + k] = NeuralController(self.layer_sizes)
                self._immigrant_indices.add(inject_start + k)

            self.population = new_pop
            self.mut_sigma  = max(self.min_mut_sigma, self.mut_sigma * self.mut_decay)
            self.generation += 1
            return self.best_individual

    ga = _VisualGA(
        pop_size=args_dict["pop"],
        init_mut_sigma=args_dict["sigma"],
        mut_decay=args_dict["decay"],
        phase_switch_k=args_dict["phase_k"],
        immigrant_frac=0.05,   # raised from 0.02 for better diversity
    )
    # Stagnation tracking + cross-generation pit memory
    ga.gens_since_improvement = 0
    ga.last_recorded_best     = -float("inf")
    ga.persistent_pits        = set()
    ga.persistent_walls       = set()
    ga.persistent_teleports   = dict()
    ga._immigrant_indices     = set()   # tagged each gen after pop rebuild
    ga.cumulative_cell_set    = set()   # all cells ever visited by any agent

    print(f"[GA]  {ga.pop_size} individuals × "
          f"{ga.population[0].num_params} params"
          f"  phase_switch_k={ga.phase_switch_k}"
          f"  immigrant_frac={ga.immigrant_frac}\n")

    from datetime import datetime
    import shutil
    run_id       = args_dict.get("run_id") or datetime.now().strftime("%Y%m%d_%H%M%S")
    weights_file = f"weights_{run_id}.npy"
    alias_file   = f"best_weights_{run_id}.npy"
    print(f"[TRAIN] weights → {weights_file}  (alias → {alias_file})\n")

    # Warm-start: seed population from existing weights + mutations.
    # Agents inherit navigation skills from a previous maze; they still
    # need to discover this maze's specific layout from scratch.
    _init_w = args_dict.get("init_weights")
    if _init_w:
        try:
            import numpy as _np
            flat = _np.load(_init_w)
            if flat.size == ga.population[0].num_params:
                for ind in ga.population:
                    ind.set_flat_weights(flat.copy())
                    ga._mutate(ind, mutation_rate=0.15)
                print(f"[WARM-START] population seeded from {_init_w}")
            else:
                print(f"[WARM-START] skipped — param mismatch ({flat.size} vs {ga.population[0].num_params})")
        except Exception as _e:
            print(f"[WARM-START] failed: {_e}")

    epsilon   = 0.70   # D* Lite handles 70% of turns; NN handles fire timing
    for gen_i in range(args_dict["gens"]):
        ga.step(env,
                eval_episodes=args_dict["eps_eval"],
                eval_turns=args_dict["turns"],
                epsilon=epsilon,
                persist=args_dict["persist"])
        epsilon = max(0.05, epsilon * 0.94)

        if ga.best_individual:
            ga.best_individual.save(weights_file)
            shutil.copy2(weights_file, alias_file)
            # Always replay best agent so visualization stays live every generation
            print(f"\n  ── Live replay (gen {gen_i+1},"
                  f" fitness={ga.best_fitness:+.0f} [{ga.phase}]) ──")
            evaluate_fitness(
                ga.best_individual, env,
                goal_cell=ma.GOAL_CELL, start_cell=ma.START_CELL,
                episodes=1, max_turns=args_dict["turns"],
                epsilon=0.0, verbose=False,
                step_q=state_q, step_interval=50,
                phase=ga.phase,
            )

    try:
        state_q.put({"type": "done", "done": True,
                     "status": f"✓ Done — {args_dict['gens']} gens"
                               f"  best={ga.best_fitness:+.0f}"
                               f"  saved → {weights_file}  [{ga.phase}]",
                     "history": list(ga.history)})
    except Exception:
        pass
    print(f"✓ Training done.  Weights saved to {weights_file}")


# ─────────────────────────────────────────────────────────────────────────────
# Test — single window, PIL rendering, no LiveVisualizer (avoids Windows hang)
# ─────────────────────────────────────────────────────────────────────────────
def _run_test_direct(args, nav_name: str):
    """
    Test mode: single combined window (map left, results right).
    Uses same embedded PIL rendering as training to avoid the two-figure
    matplotlib event-loop hang on Windows TkAgg.
    """
    from PIL import Image as _PIL, ImageDraw as _PILDraw
    from maze_agent import NeuralController, EvolutionaryAgent, PHASE_OPTIMIZE

    env = MazeEnvironment(args.maze)
    ma.configure(env.start_cell, env.goal_cell)

    # ── Load weights ──────────────────────────────────────────────────────────
    ctrl = NeuralController()
    try:
        flat = np.load(args.weights)
        if flat.size != ctrl.num_params:
            print(f"\n[ERROR] Weight mismatch: file={flat.size}"
                  f"  NeuralController expects {ctrl.num_params}")
            return
        ctrl.set_flat_weights(flat)
        print(f"[TEST] loaded {args.weights}  ({flat.size} params)")
    except Exception as e:
        print(f"[ERROR] {e}"); return

    # Count reachable cells via BFS from start — true navigable maze size
    from collections import deque as _deque
    _visited_bfs = {env.start_cell}
    _bfs_q = _deque([env.start_cell])
    while _bfs_q:
        _br, _bc = _bfs_q.popleft()
        for _nb in env.adj[_br][_bc]:
            if _nb not in _visited_bfs:
                _visited_bfs.add(_nb)
                _bfs_q.append(_nb)
    _total_passable = len(_visited_bfs)
    print(f"[TEST] maze={args.maze}  start={env.start_cell}  goal={env.goal_cell}")
    print(f"[TEST] {args.test_episodes} episodes  eps=0  (frozen)\n")

    # ── PIL rendering setup (same approach as build_and_run) ──────────────────
    maze_pil = _PIL.open(args.maze).convert("RGB")
    img_w, img_h = maze_pil.size
    cs = img_w // GRID_SIZE

    # Pre-collect metadata from env
    fire_states  = [set(map(tuple, s)) for s in _get_fire_rotation_states(env)]
    tp_purple    = [(r, c) for r, c in env.loader.teleport_purple]
    tp_orange    = [(r, c) for r, c in env.loader.teleport_orange]
    tp_green     = [(r, c) for r, c in env.loader.teleport_green]
    tp_red       = [(r, c) for r, c in getattr(env.loader, "teleport_red", [])]
    arrow_up     = list(env.arrow_up)
    arrow_left   = list(env.arrow_left)
    fire_rot     = [0]   # mutable for closure
    # Fallback: show any teleporter pads the loader colour-missed
    _tp_shown = set(map(tuple, env.loader.teleport_purple)) \
              | set(map(tuple, env.loader.teleport_orange))  \
              | set(map(tuple, env.loader.teleport_green))   \
              | set(map(tuple, getattr(env.loader, "teleport_red", [])))
    tp_extra  = [(r,c) for (r,c) in env.teleport_map if (r,c) not in _tp_shown]

    def _xy(r, c):
        return (min(c*cs + cs//2, img_w-1), min(r*cs + cs//2, img_h-1))

    def _pdot(draw, r, c, colour, dr=3):
        x, y = _xy(r, c)
        draw.ellipse([x-dr, y-dr, x+dr, y+dr], fill=colour)

    def _pagent(draw, r, c):
        _pdot(draw, r, c, (0, 0, 0), dr=6)
        _pdot(draw, r, c, (255, 255, 255), dr=4)


    def _parrow(draw, r, c, direction):
        """Draw a small directional arrow at cell (r,c). direction: 'up' or 'left'."""
        x, y = _xy(r, c)
        col = (0, 200, 255)  # cyan
        if direction == 'up':
            draw.line([(x, y+4), (x, y-4)], fill=col, width=2)
            draw.line([(x, y-4), (x-3, y-1)], fill=col, width=2)
            draw.line([(x, y-4), (x+3, y-1)], fill=col, width=2)
        else:  # left
            draw.line([(x+4, y), (x-4, y)], fill=col, width=2)
            draw.line([(x-4, y), (x-1, y-3)], fill=col, width=2)
            draw.line([(x-4, y), (x-1, y+3)], fill=col, width=2)

    def _ppath(draw, path):
        if not path:
            return
        if len(path) == 1:
            _pdot(draw, *path[0], (33, 150, 243)); return
        # Break path at teleport jumps (consecutive cells > 2 apart)
        seg = [path[0]]
        for i in range(1, len(path)):
            try: r0, c0 = path[i-1][0], path[i-1][1]; r1, c1 = path[i][0], path[i][1]
            except (TypeError, IndexError): seg.append(path[i]); continue
            # Not adjacent in maze = teleport or respawn jump
            if (r1,c1) not in env.adj[r0][c0] and (r0,c0) != (r1,c1):
                if len(seg) >= 2:
                    draw.line([_xy(r,c) for r,c in seg], fill=(33,150,243), width=2)
                elif seg:
                    _pdot(draw, *seg[0], (33,150,243))
                seg = [path[i]]
            else:
                seg.append(path[i])
        if len(seg) >= 2:
            draw.line([_xy(r,c) for r,c in seg], fill=(33,150,243), width=2)
        elif seg:
            _pdot(draw, *seg[0], (33,150,243))

    _painted   = [maze_pil.copy()]
    _paint_idx = [0]

    def _reset_painted():
        _painted[0]   = maze_pil.copy()
        _paint_idx[0] = 0

    def _compose(agent_path, agent_pos, known_pits, deaths_list=None):
        if agent_path and len(agent_path) > _paint_idx[0]:
            draw = _PILDraw.Draw(_painted[0])
            _ppath(draw, agent_path[max(0, _paint_idx[0]-1):])
            _paint_idx[0] = len(agent_path)
        disp = _painted[0].copy()
        draw = _PILDraw.Draw(disp)
        for r, c in tp_purple: _pdot(draw, r, c, (160, 50, 255), 4)
        for r, c in tp_orange: _pdot(draw, r, c, (255, 140,   0), 4)
        for r, c in tp_green:  _pdot(draw, r, c, ( 30, 200,  70), 4)
        for r, c in tp_red:    _pdot(draw, r, c, (220,   0, 255), 4)
        for r, c in tp_extra:  _pdot(draw, r, c, (  0, 220, 220), 4)
        for r, c in arrow_up:   _parrow(draw, r, c, "up")
        for r, c in arrow_left: _parrow(draw, r, c, "left")
        _pdot(draw, *env.start_cell, (  0, 230, 100), 6)
        _pdot(draw, *env.goal_cell,  (  0, 220, 255), 7)
        for r, c in known_pits: _pdot(draw, r, c, (180,   0,   0), 4)
        for r, c in (fire_states[fire_rot[0]] if fire_states else []):
            _pdot(draw, r, c, (255, 40, 40), 5)
        # Death markers — skull shape
        for entry in (deaths_list or []):
            dr, dc, dep = entry
            x, y = _xy(dr, dc)
            col = (255, 230, 0)
            rh = 7  # skull head radius
            # Cranium
            draw.ellipse([x-rh, y-rh-1, x+rh, y+rh-3], fill=col)
            # Eye sockets
            draw.ellipse([x-5, y-6, x-2, y-3], fill=(0, 0, 0))
            draw.ellipse([x+2, y-6, x+5, y-3], fill=(0, 0, 0))
            # Nose
            draw.ellipse([x-1, y-2, x+1, y],   fill=(0, 0, 0))
            # Jaw / teeth
            draw.rectangle([x-rh+1, y+rh-5, x+rh-1, y+rh-3], fill=col)
            for _tx in [x-4, x, x+4]:
                draw.rectangle([_tx-1, y+rh-3, _tx+1, y+rh], fill=(0, 0, 0))
            draw.text((x+rh+2, y-rh), str(dep), fill=col)
        if agent_pos: _pagent(draw, *agent_pos)
        return np.array(disp)

    # ── Single combined window ────────────────────────────────────────────────
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(16, 8), facecolor="#0a0f18")
    fig.suptitle(f"Silent Cartographer — TEST  ({nav_name})",
                 color="#c8d8e8", fontsize=12, fontweight="bold", y=0.98)

    gs = gridspec.GridSpec(2, 2, figure=fig,
                           height_ratios=[14, 1], width_ratios=[6, 5],
                           hspace=0.06, wspace=0.20,
                           left=0.02, right=0.97, top=0.93, bottom=0.04)
    ax_map  = fig.add_subplot(gs[0, 0])
    ax_stat = fig.add_subplot(gs[1, 0])
    ax_res  = fig.add_subplot(gs[0, 1])

    ax_map.axis("off")
    ax_map.set_title("Discovered Map", color="#c8d0e0", fontsize=10, pad=4)
    im_map = ax_map.imshow(np.array(maze_pil), origin="upper",
                           interpolation="nearest")

    ax_res.set_facecolor("#060b12"); ax_res.axis("off")
    ax_res.set_title("Test Results", color="#c8d0e0", fontsize=10, pad=4)
    for sp in ax_res.spines.values(): sp.set_edgecolor("#1e2a3a")

    ax_stat.set_facecolor("#0a1520"); ax_stat.axis("off")
    stat_txt = ax_stat.text(0.5, 0.5, "Running…", transform=ax_stat.transAxes,
                             ha="center", va="center", fontsize=9,
                             color="#8ab4d4", fontfamily="monospace")

    def _draw_table(results, metrics=None):
        ax_res.cla(); ax_res.set_facecolor("#060b12"); ax_res.axis("off")
        ax_res.set_title("Test Results", color="#c8d0e0", fontsize=10, pad=4)
        n_k   = args.test_episodes
        row_h = min(0.075, 0.58 / max(n_k, 1))
        hdr_y = 0.97
        cols_x   = [0.03, 0.19, 0.40, 0.55, 0.70, 0.85]
        cols_hdr = ["Ep", "Result", "Steps", "Deaths", "Cells", "Fitness"]
        for x, t in zip(cols_x, cols_hdr):
            ax_res.text(x, hdr_y, t, color="#7a8a9a", fontsize=8, va="top",
                        fontweight="bold")
        ax_res.axhline(hdr_y - 0.03, color="#1e2a3a", lw=0.8)
        for i, r in enumerate(results):
            y     = hdr_y - 0.065 - i * row_h
            color = "#00e5aa" if r["solved"] else "#ff4466"
            label = "✓ SOLVED" if r["solved"] else "✗ TIMEOUT"
            vals  = [str(r["episode"]), label, str(r["turns"]),
                     str(r["deaths"]), str(r["explored"]), f"{r['fitness']:+.0f}"]
            tc    = ["#c8d0e0", color, "#c8d0e0",
                     "#ff9966" if r["deaths"] > 0 else "#c8d0e0",
                     "#c8d0e0", "#c8d0e0"]
            for x, t, col in zip(cols_x, vals, tc):
                ax_res.text(x, y, t, color=col, fontsize=8, va="top")
        sy = hdr_y - 0.065 - n_k * row_h - 0.05
        if results:
            n   = len(results)
            n_s = sum(r["solved"]  for r in results)
            at  = sum(r["turns"]   for r in results) / n
            ad  = sum(r["deaths"]  for r in results) / n
            bt  = min((r["turns"] for r in results if r["solved"]), default=None)
            ax_res.axhline(sy + 0.02, color="#1e2a3a", lw=0.8)
            rc  = "#00e5aa" if n_s == n else ("#ffaa00" if n_s > 0 else "#ff4466")
            s   = f"Rate: {n_s}/{n}   Avg steps: {at:.0f}   Avg deaths: {ad:.2f}"
            if bt is not None: s += f"   Best: {bt} steps"
            ax_res.text(0.03, sy, s, color=rc, fontsize=8, va="top", fontweight="bold")
        # ── Metrics panel ────────────────────────────────────────────────────
        if metrics:
            my = sy - 0.07
            ax_res.axhline(my + 0.03, color="#2a3a4a", lw=0.8)
            my -= 0.01
            # Required
            ax_res.text(0.03, my, "REQUIRED METRICS", color="#5a7090",
                        fontsize=7, va="top", fontweight="bold")
            my -= 0.055
            req_rows = [
                ("Success Rate",         f"{metrics['success_rate']:.1f}%",         "#00e5aa"),
                ("Avg Path Length",      f"{metrics['avg_path_length']:.0f} cells",  "#c8d0e0"),
                ("Avg Turns to Solution",f"{metrics['avg_turns_to_sol']:.0f} turns", "#c8d0e0"),
                ("Death Rate",           f"{metrics['death_rate']:.2f} / ep",         "#ff9966"),
            ]
            for lbl, val, col in req_rows:
                ax_res.text(0.03, my, lbl, color="#8a9aaa", fontsize=7.5, va="top")
                ax_res.text(0.97, my, val, color=col, fontsize=7.5, va="top", ha="right")
                my -= 0.048
            # Bonus
            my -= 0.01
            ax_res.text(0.03, my, "BONUS METRICS", color="#5a7090",
                        fontsize=7, va="top", fontweight="bold")
            my -= 0.055
            bon_rows = [
                ("Exploration Efficiency", f"{metrics['explore_eff']:.4f} cells/turn","#44bbff"),
                ("Map Completeness",       f"{metrics['map_complete']:.1f}%",          "#44bbff"),
                ("Replanning Efficiency",  f"{metrics['replan_eff']:.1f}%",            "#44bbff"),
                ("Learning Efficiency",    f"{metrics['learn_eff']}",                  "#44bbff"),
            ]
            for lbl, val, col in bon_rows:
                ax_res.text(0.03, my, lbl, color="#8a9aaa", fontsize=7.5, va="top")
                ax_res.text(0.97, my, val, color=col, fontsize=7.5, va="top", ha="right")
                my -= 0.048

    fig.canvas.draw()
    plt.pause(0.05)

    # ── Episode loop — mirrors training: evaluate_fitness + accumulated seeds ──
    import queue as _lq

    results            = []
    death_positions    = []  # (r, c, ep) cumulative
    episode_data       = []  # per-episode {path, deaths, agent_pos, seed_pits, result}
    seed_pits:  set    = set()
    seed_walls: set    = set()
    cumulative_explored: set = set()  # unique cells ever visited across all episodes

    DISPLAY_EVERY = 100

    for ep in range(args.test_episodes):
        _reset_painted()
        print(f"  ── Test episode {ep+1}/{args.test_episodes} ──"
              f"  (known: {len(seed_pits)} pits, {len(seed_walls)//2} walls)")

        step_q = _lq.Queue()

        fit, agent = evaluate_fitness(
            ctrl, env,
            goal_cell=ma.GOAL_CELL, start_cell=ma.START_CELL,
            episodes=1, max_turns=args.turns,
            epsilon=0.0, persist=True,
            seed_pits=seed_pits, seed_walls=seed_walls,
            phase=PHASE_OPTIMIZE,
            early_stop=False,
            legacy_pit_walls=getattr(args, "legacy_pits", False),
            step_q=step_q, step_interval=DISPLAY_EVERY,
        )

        # Accumulate discoveries for next episode
        # New pits = cells discovered this episode (each = one death location)
        _new_pits = agent.memory._shared_pits - seed_pits
        seed_pits  |= agent.memory._shared_pits
        seed_walls |= agent.memory._shared_walls
        cumulative_explored.update(agent.memory.visit_count.keys())
        for (_dr, _dc) in _new_pits:
            death_positions.append((_dr, _dc, ep + 1))

        ep_stats     = env.get_episode_stats()
        goal_reached = agent.goal_reached
        died         = ep_stats["deaths"] > 0

        # Record all unique death positions this episode
        for pos in getattr(agent.memory, '_death_positions', []):
            death_positions.append((*pos, ep + 1))

        results.append({
            "episode":  ep + 1,
            "solved":   goal_reached,
            "turns":    ep_stats["turns_taken"],
            "deaths":   ep_stats["deaths"],
            "explored": ep_stats["cells_explored"],
            "fitness":  fit,
            "died":     died and not goal_reached,
        })

        # Store full per-episode data for navigation (after results.append)
        episode_data.append({
            'path':      list(agent.memory.path),
            'deaths':    list(death_positions),
            'agent_pos': agent.current_pos,
            'seed_pits': set(seed_pits),
            'result':    results[-1],
        })

        tag = "✓ SOLVED" if goal_reached else ("✗ DEATH" if died else "✗ TIMEOUT")
        print(f"  [{tag}]  turns={ep_stats['turns_taken']}"
              f"  deaths={ep_stats['deaths']}"
              f"  explored={ep_stats['cells_explored']}"
              f"  pits_known={len(seed_pits)}  fitness={fit:+.0f}")

        last_path = list(agent.memory.path[-2000:])
        while True:
            try:
                msg = step_q.get_nowait()
                if "agent_path" in msg:
                    last_path = msg["agent_path"]
            except _lq.Empty:
                break

        fire_rot[0] = (fire_rot[0] + 1) % 4
        im_map.set_data(_compose(last_path, agent.current_pos,
                                 seed_pits, death_positions))
        _draw_table(results)
        n_s = sum(r["solved"] for r in results)
        stat_txt.set_text(
            f"Ep {ep+1}/{args.test_episodes}  [{tag}]  "
            f"rate={n_s}/{ep+1}  steps={ep_stats['turns_taken']}"
            f"  deaths={ep_stats['deaths']}")
        fig.canvas.draw_idle()
        plt.pause(0.2)

        _reset_painted()

    # Final summary
    n   = len(results)
    n_s = sum(r["solved"]  for r in results)
    at  = sum(r["turns"]   for r in results) / n
    ad  = sum(r["deaths"]  for r in results) / n
    bt  = min((r["turns"] for r in results if r["solved"]), default=None)
    print("\n" + "━"*50)
    print(f"  TEST COMPLETE  ({n} episodes)")
    print(f"  Goal rate  : {n_s}/{n}  ({100*n_s/n:.0f}%)")
    print(f"  Avg steps  : {at:.0f}")
    print(f"  Avg deaths : {ad:.2f}")
    if bt is not None: print(f"  Best solve : {bt} steps")
    print("━"*50 + "\n")

    # ── Compute all metrics ──────────────────────────────────────────────────
    solved_r    = [r for r in results if r["solved"]]
    total_turns = max(sum(r["turns"]  for r in results), 1)
    total_deaths= sum(r["deaths"] for r in results)
    n_explored  = len(cumulative_explored)

    avg_path_len    = (sum(r["explored"] for r in solved_r) / len(solved_r)) if solved_r else 0.0
    avg_turns_sol   = (sum(r["turns"]    for r in solved_r) / len(solved_r)) if solved_r else 0.0
    explore_eff     = n_explored / total_turns
    map_complete    = n_explored / max(_total_passable, 1) * 100
    # Replanning efficiency: clean (0-death) solves as fraction of all solves
    clean_solves    = sum(1 for r in solved_r if r["deaths"] == 0)
    replan_eff      = (clean_solves / len(solved_r) * 100) if solved_r else 0.0
    # Learning efficiency: % improvement in turns from first to last solved ep
    if len(solved_r) >= 2:
        learn_eff = f"{(solved_r[0]['turns'] - solved_r[-1]['turns']) / solved_r[0]['turns'] * 100:+.1f}%"
    elif len(solved_r) == 1:
        learn_eff = "N/A (1 solve)"
    else:
        learn_eff = "N/A (0 solves)"

    metrics = {
        "success_rate":    n_s / n * 100,
        "avg_path_length": avg_path_len,
        "avg_turns_to_sol":avg_turns_sol,
        "death_rate":      total_deaths / n,
        "explore_eff":     explore_eff,
        "map_complete":    map_complete,
        "replan_eff":      replan_eff,
        "learn_eff":       learn_eff,
    }

    # ── Terminal metrics report ───────────────────────────────────────────────
    print("\n" + "━"*54)
    print("  PERFORMANCE METRICS")
    print("━"*54)
    print("  REQUIRED")
    print(f"  {'Success Rate':<28}: {n_s}/{n} ({metrics['success_rate']:.1f}%)")
    print(f"  {'Avg Path Length':<28}: {avg_path_len:.0f} cells (solved eps)")
    print(f"  {'Avg Turns to Solution':<28}: {avg_turns_sol:.0f} turns")
    print(f"  {'Death Rate':<28}: {metrics['death_rate']:.2f} deaths/episode")
    print("  BONUS")
    print(f"  {'Exploration Efficiency':<28}: {explore_eff:.4f} cells/turn")
    print(f"  {'Map Completeness':<28}: {map_complete:.1f}% ({n_explored}/{_total_passable} cells)")
    print(f"  {'Replanning Efficiency':<28}: {replan_eff:.1f}% clean solves")
    print(f"  {'Learning Efficiency':<28}: {learn_eff}")
    print("━"*54 + "\n")

    _draw_table(results, metrics)

    # ── Per-episode navigation ─────────────────────────────────────────────
    # ← → arrow keys to step through episodes; each shows full path + deaths
    _view_ep = [len(episode_data) - 1]

    def _compose_full(ep_idx):
        """Paint a complete map for one episode from scratch."""
        ep   = episode_data[ep_idx]
        path = ep["path"]
        img  = maze_pil.copy()
        draw = _PILDraw.Draw(img)
        # Draw path with teleport-jump detection
        if path:
            seg = [path[0]]
            for i in range(1, len(path)):
                try: r0, c0 = path[i-1][0], path[i-1][1]; r1, c1 = path[i][0], path[i][1]
                except (TypeError, IndexError): seg.append(path[i]); continue
                # Not adjacent in maze = teleport or respawn jump
                if (r1,c1) not in env.adj[r0][c0] and (r0,c0) != (r1,c1):
                    if len(seg) >= 2:
                        draw.line([_xy(r,c) for r,c in seg], fill=(33,150,243), width=2)
                    elif seg:
                        _pdot(draw, *seg[0], (33,150,243))
                    seg = [path[i]]
                else:
                    seg.append(path[i])
            if len(seg) >= 2:
                draw.line([_xy(r,c) for r,c in seg], fill=(33,150,243), width=2)
            elif seg:
                _pdot(draw, *seg[0], (33,150,243))
        for r,c in tp_purple: _pdot(draw, r, c, (160, 50,255), 4)
        for r,c in tp_orange: _pdot(draw, r, c, (255,140,  0), 4)
        for r,c in tp_green:  _pdot(draw, r, c, ( 30,200, 70), 4)
        for r,c in tp_red:    _pdot(draw, r, c, (220,  0,255), 4)
        for r,c in tp_extra:  _pdot(draw, r, c, (  0,220,220), 4)
        for r,c in arrow_up:   _parrow(draw, r, c, "up")
        for r,c in arrow_left: _parrow(draw, r, c, "left")
        _pdot(draw, *env.start_cell, (  0,230,100), 6)
        _pdot(draw, *env.goal_cell,  (  0,220,255), 7)
        # Fire pits known at this episode
        for r,c in ep["seed_pits"]: _pdot(draw, r, c, (180,0,0), 4)
        # Death markers (skull): yellow = this episode, orange = earlier
        for (dr,dc,dep) in ep["deaths"]:
            col = (255,230,0) if dep == ep_idx+1 else (255,140,0)
            x,y = _xy(dr,dc)
            rh  = 7
            draw.ellipse([x-rh, y-rh-1, x+rh, y+rh-3], fill=col)
            draw.ellipse([x-5,  y-6,    x-2,   y-3],   fill=(0,0,0))
            draw.ellipse([x+2,  y-6,    x+5,   y-3],   fill=(0,0,0))
            draw.ellipse([x-1,  y-2,    x+1,   y],     fill=(0,0,0))
            draw.rectangle([x-rh+1, y+rh-5, x+rh-1, y+rh-3], fill=col)
            for _tx in [x-4, x, x+4]:
                draw.rectangle([_tx-1, y+rh-3, _tx+1, y+rh], fill=(0,0,0))
            draw.text((x+rh+2, y-rh), f"E{dep}", fill=col)
        if ep["agent_pos"]: _pagent(draw, *ep["agent_pos"])
        return np.array(img)

    def _navigate(ep_idx):
        r   = episode_data[ep_idx]["result"]
        tag = "✓ SOLVED" if r["solved"] else ("✗ DEATH" if r.get("died") else "✗ TIMEOUT")
        ax_map.set_title(
            f"Ep {ep_idx+1}/{len(episode_data)}  [{tag}]"
            f"  steps={r['turns']}  deaths={r['deaths']}"
            f"  ◄ ► arrows to browse",
            color="#c8d0e0", fontsize=9, pad=4)
        im_map.set_data(_compose_full(ep_idx))
        fig.canvas.draw_idle()

    def _on_key(event):
        if event.key == "left"  and _view_ep[0] > 0:
            _view_ep[0] -= 1; _navigate(_view_ep[0])
        elif event.key == "right" and _view_ep[0] < len(episode_data)-1:
            _view_ep[0] += 1; _navigate(_view_ep[0])

    fig.canvas.mpl_connect("key_press_event", _on_key)
    _navigate(_view_ep[0])   # start on last episode

    stat_txt.set_text(f"Done  |  rate: {n_s}/{n}  avg: {at:.0f} steps"
                      f"  avg_d: {ad:.2f}" + (f"  best: {bt}" if bt else "")
                      + "   ◄ ► arrows to browse episodes")
    fig.canvas.draw_idle()
    plt.show(block=True)


# ─────────────────────────────────────────────────────────────────────────────
# Legacy test_worker — kept for training worker's evaluate replay only
# ─────────────────────────────────────────────────────────────────────────────
def test_worker(maze_path: str, args_dict: dict, state_q: Queue):
    import maze_agent as ma
    from environment import MazeEnvironment
    from maze_agent import NeuralController, evaluate_fitness, PHASE_OPTIMIZE

    env = MazeEnvironment(maze_path)
    ma.configure(env.start_cell, env.goal_cell)

    weights_path = args_dict.get("weights", "best_weights.npy")
    n_episodes   = args_dict.get("test_episodes", 5)

    ctrl = NeuralController()
    try:
        flat = np.load(weights_path)
        if flat.size != ctrl.num_params:
            msg = (f"Weight size mismatch: file has {flat.size} params "
                   f"but NeuralController expects {ctrl.num_params}.\n"
                   f"  → For Q-learning weights use: "
                   f"python train_qlearn.py --test --weights {weights_path}")
            print(f"\n[ERROR] {msg}")
            state_q.put({"type": "test_result", "error": msg, "done": True,
                         "results": [], "status": f"ERROR: wrong weight type — use train_qlearn.py --test"}); return
        ctrl.set_flat_weights(flat)
        print(f"[TEST] loaded {weights_path}  ({flat.size} params)")
    except Exception as e:
        state_q.put({"type": "test_result", "error": str(e), "done": True,
                     "results": [], "status": f"ERROR: {e}"}); return

    print(f"[TEST] maze={maze_path}  start={env.start_cell}  goal={env.goal_cell}")
    print(f"[TEST] {n_episodes} episodes  ε=0  phase=optimize\n")

    results = []
    # Collective memory across test episodes — episode 2 knows where episode 1 died
    test_known_pits:  set = set()
    test_known_walls: set = set()

    for ep in range(n_episodes):
        print(f"  ── Test episode {ep+1}/{n_episodes} ──")
        if test_known_pits:
            print(f"     (starting with {len(test_known_pits)} known pits"
                  f" from previous episodes)")
        fit, agent = evaluate_fitness(
            ctrl, env,
            goal_cell=ma.GOAL_CELL, start_cell=ma.START_CELL,
            episodes=1, max_turns=args_dict["turns"],
            epsilon=0.0, verbose=False,
            seed_pits=test_known_pits, seed_walls=test_known_walls,
            persist=True,
            step_q=state_q, step_interval=15,
            phase=PHASE_OPTIMIZE,
        )
        # Accumulate discoveries for next episode
        test_known_pits  |= agent.memory._shared_pits
        test_known_walls |= agent.memory._shared_walls
        ep_stats = env.get_episode_stats()
        wall_hits = getattr(agent, "total_wall_hits", 0)
        if agent.goal_reached:
            print(f"  ✓ GOAL in {ep_stats['turns_taken']} turns!"
                  f"  deaths={ep_stats['deaths']}  walls={wall_hits}")
        solved_str = "SUCCESS ✓" if agent.goal_reached else "TIMEOUT ✗"
        print(f"  {solved_str}  turns={ep_stats['turns_taken']}"
              f"  deaths={ep_stats['deaths']}  walls={wall_hits}"
              f"  explored={ep_stats['cells_explored']}"
              f"  fitness={fit:+.0f}  [{PHASE_OPTIMIZE}]")
        print(f"  env_stats: {ep_stats}")
        result = {
            "episode":  ep + 1,
            "solved":   agent.goal_reached,
            "turns":    ep_stats["turns_taken"],
            "deaths":   ep_stats["deaths"],
            "explored": ep_stats["cells_explored"],
            "fitness":  fit,
        }
        results.append(result)

        n_solved   = sum(r["solved"] for r in results)
        avg_turns  = sum(r["turns"]  for r in results) / len(results)
        avg_deaths = sum(r["deaths"] for r in results) / len(results)
        solved_str = "SOLVED" if agent.goal_reached else "TIMEOUT"
        status = (f"Test {ep+1}/{n_episodes}  [{solved_str}]"
                  f"  |  rate: {n_solved}/{ep+1}"
                  f"  steps: {ep_stats['turns_taken']}"
                  f"  deaths: {ep_stats['deaths']}"
                  f"  avg_steps: {avg_turns:.0f}"
                  f"  avg_deaths: {avg_deaths:.2f}")
        print(f"  {status}\n")

        mem  = agent.memory
        vmap = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
        for (r, c), cnt in mem.visit_count.items():
            if 0 <= r < GRID_SIZE and 0 <= c < GRID_SIZE:
                vmap[r, c] = np.log1p(cnt)
        try:
            state_q.put_nowait({"type": "test_result", "visit_map": vmap,
                                 "agent_pos": agent.current_pos,
                                 "agent_path": list(mem.path),
                                 "results": list(results),
                                 "status": status, "done": False})
        except Exception:
            pass

    n_solved   = sum(r["solved"]  for r in results)
    avg_turns  = sum(r["turns"]   for r in results) / len(results)
    avg_deaths = sum(r["deaths"]  for r in results) / len(results)
    best_turns = min((r["turns"]  for r in results if r["solved"]), default=None)
    min_deaths = min((r["deaths"] for r in results if r["solved"]), default=None)

    print(f"\n{'━'*50}")
    print(f"  TEST COMPLETE  ({n_episodes} episodes)")
    print(f"  Goal rate  : {n_solved}/{n_episodes}  ({100*n_solved/n_episodes:.0f}%)")
    print(f"  Avg steps  : {avg_turns:.0f}")
    print(f"  Avg deaths : {avg_deaths:.2f}")
    if best_turns  is not None: print(f"  Best solve : {best_turns} steps")
    if min_deaths  is not None: print(f"  Min deaths : {min_deaths} (on solved runs)")
    print(f"{'━'*50}\n")

    done_status = (f"✓ Test done  |  rate: {n_solved}/{n_episodes}"
                   f"  avg: {avg_turns:.0f} steps  avg_d: {avg_deaths:.2f}"
                   + (f"  best: {best_turns} steps" if best_turns is not None else ""))
    try:
        state_q.put({"type": "test_result", "results": results,
                     "status": done_status, "done": True})
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Rendering helpers
# ─────────────────────────────────────────────────────────────────────────────
def precompute_fire_rotations(env): return _get_fire_rotation_states(env)

def _mark(ov, r, c, rgba, half=MARK):
    h, w = ov.shape[:2]
    py = r * CELL_SIZE + CELL_SIZE // 2
    px = c * CELL_SIZE + CELL_SIZE // 2
    ov[max(0,py-half):min(h,py+half+1),
       max(0,px-half):min(w,px+half+1)] = rgba

def build_static_overlay(env, img_h, img_w):
    ov = np.zeros((img_h, img_w, 4), dtype=np.uint8)
    for r, c in env.loader.teleport_purple: _mark(ov, r, c, (160,  50, 255, 200))
    for r, c in env.loader.teleport_orange: _mark(ov, r, c, (255, 140,   0, 200))
    for r, c in env.loader.teleport_green:  _mark(ov, r, c, ( 30, 200,  70, 200))
    sr, sc = env.start_cell; _mark(ov, sr, sc, (  0, 230, 100, 230), half=8)
    gr, gc = env.goal_cell;  _mark(ov, gr, gc, (  0, 220, 255, 230), half=8)
    return ov

def compose_map_frame(maze_rgb, stat_ov, fire_pits, visit_map,
                      path, agent_pos, img_h, img_w):
    frame = np.empty((img_h, img_w, 4), dtype=np.uint8)
    frame[:, :, :3] = maze_rgb[:img_h, :img_w]
    frame[:, :,  3] = 255
    smask = stat_ov[:, :, 3] > 0
    frame[smask] = stat_ov[smask]

    ov64 = np.zeros((GRID_SIZE, GRID_SIZE, 4), dtype=np.uint8)
    vmax = visit_map.max() if visit_map.max() > 0 else 1.0
    vis  = visit_map > 0
    if vis.any():
        v = (visit_map[vis] / vmax).clip(0, 1)
        ov64[vis, 0] = ( 20        * v).astype(np.uint8)
        ov64[vis, 1] = ( 80 + 150  * v).astype(np.uint8)
        ov64[vis, 2] = (180 +  75  * v).astype(np.uint8)
        ov64[vis, 3] = (160 * np.minimum(v * 3, 1)).astype(np.uint8)

    if path:
        n    = max(1, len(path) - 1)
        rows = np.array([p[0] for p in path], dtype=np.int32)
        cols = np.array([p[1] for p in path], dtype=np.int32)
        mask = (rows >= 0) & (rows < GRID_SIZE) & (cols >= 0) & (cols < GRID_SIZE)
        rows, cols = rows[mask], cols[mask]
        if len(rows):
            idx = np.where(mask)[0]; t = idx / n
            ov64[rows, cols, 0] = ( 30 + 100 * t).astype(np.uint8)
            ov64[rows, cols, 1] = ( 80 + 175 * t).astype(np.uint8)
            ov64[rows, cols, 2] = 255
            ov64[rows, cols, 3] = 220

    big   = np.repeat(np.repeat(ov64, CELL_SIZE, axis=0), CELL_SIZE, axis=1)
    h_f   = min(big.shape[0], img_h); w_f = min(big.shape[1], img_w)
    alpha = big[:h_f, :w_f, 3:4].astype(np.float32) / 255.0
    frame[:h_f, :w_f, :3] = (
        frame[:h_f, :w_f, :3] * (1 - alpha) +
        big[:h_f, :w_f, :3]   * alpha
    ).astype(np.uint8)

    if fire_pits:
        fp = np.array(list(fire_pits), dtype=np.int32)
        fm = ((fp[:, 0] >= 0) & (fp[:, 0] < GRID_SIZE) &
              (fp[:, 1] >= 0) & (fp[:, 1] < GRID_SIZE))
        fp = fp[fm]
        if len(fp):
            fmask = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)
            fmask[fp[:, 0], fp[:, 1]] = True
            fbig = np.repeat(np.repeat(fmask, CELL_SIZE, 0), CELL_SIZE, 1)
            frame[:h_f, :w_f][fbig[:h_f, :w_f]] = [255, 40, 40, 230]

    ar, ac = agent_pos
    py = ar * CELL_SIZE + CELL_SIZE // 2
    px = ac * CELL_SIZE + CELL_SIZE // 2
    frame[max(0,py-4):min(img_h,py+5),
          max(0,px-4):min(img_w,px+5)] = [255, 34, 68, 255]
    return frame



def build_and_run(args, navigator_name: str = None):
    nav_name = navigator_name or NAVIGATOR

    if getattr(args, "test", False):
        _run_test_direct(args, nav_name)
        return

    import queue as _q
    from PIL import Image as _PIL, ImageDraw as _PILDraw

    args_dict = vars(args)

    # Load maze image directly — no MazeEnvironment, no CNN in display process
    maze_pil  = _PIL.open(args.maze).convert("RGB")
    img_w, img_h = maze_pil.size   # PIL uses (width, height)

    # ── PIL rendering helpers (LiveVisualizer style) ──────────────────────────
    cs = img_w // GRID_SIZE   # cell size in pixels (typically 16)

    def _xy(r, c):
        """Grid cell → pixel centre."""
        return (min(c*cs + cs//2, img_w-1), min(r*cs + cs//2, img_h-1))

    def _pdot(draw, r, c, colour, dr=3):
        x, y = _xy(r, c)
        draw.ellipse([x-dr, y-dr, x+dr, y+dr], fill=colour)

    def _pagent(draw, r, c):
        _pdot(draw, r, c, (0, 0, 0),       dr=6)
        _pdot(draw, r, c, (255,255,255),    dr=4)

    def _ppath(draw, path, colour=(33,150,243)):
        if len(path) < 2:
            if path: _pdot(draw, *path[0], colour)
            return
        draw.line([_xy(r,c) for r,c in path], fill=colour, width=2)

    # Metadata from worker init message — display defaults until message arrives
    _meta = {
        "start_cell":      (GRID_SIZE-1, GRID_SIZE//2),
        "goal_cell":       (0, GRID_SIZE//2),
        "fire_states":     [set() for _ in range(4)],
        "fire_rot":        0,
        "tp_purple":       [],
        "tp_orange":       [],
        "tp_green":        [],
        "tp_red":          [],
    }

    # Persistent painted image — accumulates path across one episode
    _painted   = [maze_pil.copy()]
    _paint_idx = [0]

    def _reset_painted():
        img  = maze_pil.copy()
        draw = _PILDraw.Draw(img)
        # White-out cells that are on fire in EVERY rotation
        # (permanent fire displayed as static white; rotating fire shown live)
        if _meta["fire_states"] and all(_meta["fire_states"]):
            always = set.intersection(*_meta["fire_states"])
            for r, c in always:
                draw.rectangle([c*cs+2, r*cs+2, (c+1)*cs-2, (r+1)*cs-2],
                                fill=(255,255,255))
        _painted[0]   = img
        _paint_idx[0] = 0

    def _compose(agent_path, agent_pos, known_pits=None):
        """Incrementally paint new path then overlay transient elements."""
        if agent_path and len(agent_path) > _paint_idx[0]:
            draw = _PILDraw.Draw(_painted[0])
            seg  = agent_path[max(0, _paint_idx[0]-1):]
            _ppath(draw, seg)
            _paint_idx[0] = len(agent_path)

        disp  = _painted[0].copy()
        draw  = _PILDraw.Draw(disp)

        # Teleporters
        for r,c in _meta["tp_purple"]: _pdot(draw, r, c, (160, 50,255), 4)
        for r,c in _meta["tp_orange"]: _pdot(draw, r, c, (255,140,  0), 4)
        for r,c in _meta["tp_green"]:  _pdot(draw, r, c, ( 30,200, 70), 4)
        for r,c in _meta["tp_red"]:    _pdot(draw, r, c, (220,  0,255), 4)

        # Start / goal
        _pdot(draw, *_meta["start_cell"], (  0,230,100), 6)
        _pdot(draw, *_meta["goal_cell"],  (  0,220,255), 7)

        # Known pits (discovered by agent)
        for r,c in (known_pits or []):
            _pdot(draw, r, c, (180, 0, 0), 4)

        # Current fire pits (rotating)
        fire_now = _meta["fire_states"][_meta["fire_rot"]] if _meta["fire_states"] else []
        for r,c in fire_now:
            _pdot(draw, r, c, (255, 40, 40), 5)

        # Agent
        if agent_pos:
            _pagent(draw, *agent_pos)

        return np.array(disp)

    # ── Build single window ───────────────────────────────────────────────────
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(17, 9), facecolor="#0a0f18")
    fig.suptitle(f"Silent Cartographer — {nav_name} + Neuroevolution",
                 color="#c8d8e8", fontsize=13, fontweight="bold", y=0.98)

    gs = gridspec.GridSpec(2, 2, figure=fig,
                           height_ratios=[15, 1], width_ratios=[7, 5],
                           hspace=0.06, wspace=0.25,
                           left=0.02, right=0.97, top=0.93, bottom=0.05)
    ax_map  = fig.add_subplot(gs[0, 0])
    ax_stat = fig.add_subplot(gs[1, :])

    gs_right = gridspec.GridSpecFromSubplotSpec(
        3, 1, subplot_spec=gs[0, 1], hspace=0.55, height_ratios=[3, 2, 2])
    ax_fitness = fig.add_subplot(gs_right[0])
    ax_solver  = fig.add_subplot(gs_right[1])
    ax_metrics = fig.add_subplot(gs_right[2])

    ax_map.axis("off")
    ax_map.set_title(f"{nav_name} — Discovered Map", color="#c8d0e0",
                     fontsize=11, pad=6)
    im_map = ax_map.imshow(np.array(maze_pil), origin="upper",
                           interpolation="nearest")
    legend_p = [
        mpatches.Patch(color=(0.13, 0.59, 0.95), label="Path"),
        mpatches.Patch(color=(1.00, 0.16, 0.26), label="Fire pit"),
        mpatches.Patch(color=(0.63, 0.20, 1.00), label="Teleport purple"),
        mpatches.Patch(color=(1.00, 0.55, 0.00), label="Teleport orange"),
        mpatches.Patch(color=(0.12, 0.78, 0.27), label="Teleport green"),
        mpatches.Patch(color=(0.00, 0.90, 0.39), label="Start"),
        mpatches.Patch(color=(0.00, 0.86, 1.00), label="Goal"),
    ]
    ax_map.legend(handles=legend_p, loc="upper right", fontsize=6.5, ncol=2,
                  facecolor="#080e1a", edgecolor="#1e2a3a",
                  labelcolor="#c0ccd8", framealpha=0.90)

    ax_stat.set_facecolor("#0a1520"); ax_stat.axis("off")
    stat_txt = ax_stat.text(0.5, 0.5, f"{nav_name} initialising...",
                             transform=ax_stat.transAxes,
                             ha="center", va="center", fontsize=9.5,
                             color="#8ab4d4", fontfamily="monospace")

    def _style_ax(ax, title, xlabel, ylabel):
        ax.set_facecolor("#060b12")
        for sp in ax.spines.values(): sp.set_edgecolor("#1e2a3a")
        ax.set_title(title, color="#c8d0e0", fontsize=9, pad=4)
        ax.set_xlabel(xlabel, color="#7a8a9a", fontsize=8)
        ax.set_ylabel(ylabel, color="#7a8a9a", fontsize=8)
        ax.tick_params(colors="#4a5a6a", labelsize=7)
        ax.grid(alpha=0.12, color="#2a3a4a")
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    _style_ax(ax_fitness, "Population Fitness", "Generation", "Fitness")
    ax_fitness.set_xlim(1, max(2, args.gens))
    ax_fitness.axhline(0, color="#2a4060", lw=0.8, ls="--")
    ln_best,  = ax_fitness.plot([], [], lw=2.2, color="#00e5ff", label="Best",  zorder=3)
    ln_mean,  = ax_fitness.plot([], [], lw=1.5, color="#2979ff", label="Mean",  zorder=2)
    ln_worst, = ax_fitness.plot([], [], lw=1.0, color="#455a6a", label="Worst", zorder=1, ls="--")
    fill_std  = [ax_fitness.fill_between([], [], [], color="#2979ff", alpha=0.12)]
    ax_fitness.legend(loc="lower right", fontsize=7,
                      facecolor="#0a1520", edgecolor="#1e2a3a", labelcolor="#c8d0e0")

    _style_ax(ax_solver, f"Solvers / Gen + sigma  ({nav_name})", "Generation", "Solvers")
    ax_solver.set_xlim(0.5, max(2, args.gens) + 0.5)
    ax_solver.set_ylim(0, max(2, args_dict.get("pop", 80)))
    ax_sigma     = ax_solver.twinx()
    ax_sigma.tick_params(colors="#4a5a6a", labelsize=7)
    ax_sigma.set_ylabel("sigma", color="#ffaa44", fontsize=8)
    ax_sigma.set_ylim(0, 0.25)
    bar_solvers  = [None]
    ln_sigma,    = ax_sigma.plot([], [], lw=1.5, color="#ffaa44", ls="--", zorder=2)
    ax_newcells  = ax_solver.twinx()
    ax_newcells.spines["right"].set_position(("outward", 40))
    ax_newcells.tick_params(colors="#5588aa", labelsize=6)
    ax_newcells.set_ylabel("Map cells (cumul.)", color="#44bbff", fontsize=7)
    ln_newcells, = ax_newcells.plot([], [], lw=1.8, color="#44bbff",
                                    alpha=0.9, zorder=1, drawstyle="steps-post")
    cum_txt = ax_solver.text(0.98, 0.95, "", transform=ax_solver.transAxes,
                              ha="right", va="top", fontsize=7,
                              color="#aaccdd", fontfamily="monospace")
    ax_solver.set_ylabel("Solvers", color="#00e5aa", fontsize=8)

    _style_ax(ax_metrics, "Performance Metrics", "Generation", "%")
    ax_metrics.set_xlim(0.5, max(2, args.gens) + 0.5)
    ax_metrics.set_ylim(0, 105)
    ax_metrics_r = ax_metrics.twinx()
    ax_metrics_r.tick_params(colors="#4a5a6a", labelsize=6)
    ax_metrics_r.set_ylabel("cells/turn", color="#ffaa44", fontsize=7)
    ln_mapcomp, = ax_metrics.plot([], [], lw=1.8, color="#00e5ff",  label="Map %")
    ln_replan,  = ax_metrics.plot([], [], lw=1.5, color="#00e5aa",  label="Replan %")
    ln_learn,   = ax_metrics.plot([], [], lw=1.2, color="#ff6699",  label="Learn", ls="--")
    ln_explore, = ax_metrics_r.plot([], [], lw=1.5, color="#ffaa44", label="Explore", ls="-.")
    ax_metrics.legend(loc="upper left", fontsize=5.5, ncol=2,
                      facecolor="#0a1520", edgecolor="#1e2a3a", labelcolor="#c8d0e0")

    fig.canvas.draw()
    plt.pause(0.05)

    # ── Worker process ────────────────────────────────────────────────────────
    state_q  = Queue(maxsize=400)
    proc = Process(target=training_worker,
                   args=(args.maze, args_dict, state_q), daemon=True)
    proc.start()

    state = {
        "agent_pos":  _meta["start_cell"],
        "agent_path": [],
        "history":    [],
        "status":     "Waiting for worker...",
        "done":       False,
    }
    _cache = {"hist_len": 0, "fire_tick": 0, "phase_marked": False,
              "solver_bars_drawn": 0}

    try:
        while True:
            # Drain queue
            step_msg = last_msg = None
            while True:
                try:
                    msg = state_q.get_nowait()
                    mtype = msg.get("type", "gen")
                    if mtype == "step":
                        step_msg = msg
                    else:
                        last_msg = msg
                except _queue.Empty:
                    break

            if step_msg:
                state["agent_pos"]  = step_msg["agent_pos"]
                state["agent_path"] = step_msg["agent_path"]

            if last_msg:
                mtype = last_msg.get("type", "gen")

                if mtype == "init":
                    _meta["start_cell"]  = tuple(last_msg["start_cell"])
                    _meta["goal_cell"]   = tuple(last_msg["goal_cell"])
                    _meta["fire_states"] = [set(map(tuple, s))
                                            for s in last_msg["fire_states"]]
                    _meta["tp_purple"]   = [tuple(x) for x in last_msg["teleport_purple"]]
                    _meta["tp_orange"]   = [tuple(x) for x in last_msg["teleport_orange"]]
                    _meta["tp_green"]    = [tuple(x) for x in last_msg["teleport_green"]]
                    _meta["tp_red"]      = [tuple(x) for x in last_msg.get("teleport_red", [])]
                    _reset_painted()
                    state["agent_pos"] = _meta["start_cell"]

                elif mtype == "gen":
                    # New generation — reset path accumulation
                    _reset_painted()
                    _paint_idx[0] = 0
                    if "agent_pos"  in last_msg: state["agent_pos"]  = last_msg["agent_pos"]
                    if "agent_path" in last_msg: state["agent_path"] = last_msg["agent_path"]
                    if "history"    in last_msg: state["history"]    = last_msg["history"]
                    state["status"] = last_msg.get("status", state["status"])
                    state["done"]   = last_msg.get("done",   False)

                else:
                    if "agent_pos"  in last_msg: state["agent_pos"]  = last_msg["agent_pos"]
                    if "agent_path" in last_msg: state["agent_path"] = last_msg["agent_path"]
                    if "history"    in last_msg: state["history"]    = last_msg["history"]
                    state["status"] = last_msg.get("status", state["status"])
                    state["done"]   = last_msg.get("done",   False)

            # Advance fire rotation
            _cache["fire_tick"] += 1
            if _cache["fire_tick"] >= 2:
                _cache["fire_tick"] = 0
                _meta["fire_rot"] = (_meta["fire_rot"] + 1) % 4

            txt = state["status"]
            stat_txt.set_text(("OK  " if state["done"] else "") + txt)
            im_map.set_data(_compose(state["agent_path"], state["agent_pos"]))

            # ── Metrics charts ────────────────────────────────────────────────
            try:
                hist  = state["history"]
                best_ = [h["best"] for h in hist]
                if len(hist) != _cache["hist_len"]:
                    _cache["hist_len"] = len(hist)
                    gens = [h["generation"] + 1 for h in hist]

                    ln_best.set_data( gens, best_)
                    ln_mean.set_data( gens, [h["mean"]  for h in hist])
                    ln_worst.set_data(gens, [h["worst"] for h in hist])
                    fill_std[0].remove()
                    lo = [h["mean"] - h["std"] for h in hist]
                    hi = [h["mean"] + h["std"] for h in hist]
                    fill_std[0] = ax_fitness.fill_between(
                        gens, lo, hi, color="#2979ff", alpha=0.12)
                    ax_fitness.relim(); ax_fitness.autoscale_view()

                    if not _cache["phase_marked"]:
                        for h in hist:
                            if h.get("phase") == PHASE_OPTIMIZE:
                                sg = h["generation"] + 1
                                ax_fitness.axvline(sg, color="#ffaa00",
                                                   lw=1.2, ls="--", alpha=0.7)
                                ax_fitness.text(sg+0.2,
                                                ax_fitness.get_ylim()[1]*0.95,
                                                "OPTIMIZE", color="#ffaa00",
                                                fontsize=7, va="top")
                                ax_solver.axvline(sg, color="#ffaa00",
                                                  lw=1.0, ls="--", alpha=0.5)
                                _cache["phase_marked"] = True; break

                    sc_ = [h["solvers"]              for h in hist]
                    sv_ = [h["sigma"]                for h in hist]
                    nc_ = [h.get("new_cells", 0)     for h in hist]
                    at_ = [h.get("avg_solve_turns",  0)   for h in hist]
                    ad_ = [h.get("avg_solve_deaths", 0.0) for h in hist]
                    wh_ = [h.get("avg_wall_hits",    0.0) for h in hist]
                    ta_ = [h.get("avg_turns_all",    1.0) for h in hist]

                    for gi, (g, sv) in enumerate(zip(gens, sv_)):
                        prev = sv_[gi-1] if gi > 0 else sv
                        if sv > prev * 1.5:
                            ax_solver.axvline(g, color="#ff6699", lw=1.0, ls=":", alpha=0.8)
                            ax_fitness.axvline(g, color="#ff6699", lw=0.8, ls=":", alpha=0.5)

                    if len(sc_) != _cache["solver_bars_drawn"]:
                        _cache["solver_bars_drawn"] = len(sc_)
                        if bar_solvers[0] is not None: bar_solvers[0].remove()
                        colors = ["#00e5aa" if s > 0 else "#1e3a2a" for s in sc_]
                        bar_solvers[0] = ax_solver.bar(
                            gens, sc_, color=colors, alpha=0.7, width=0.7)
                        for g, s in zip(gens, sc_):
                            if s > 0:
                                ax_solver.text(g, s+0.2, str(s), ha="center",
                                               va="bottom", fontsize=6, color="#00e5aa")
                        ax_solver.set_ylim(0, max(max(sc_, default=0)*1.3+1, 5))
                        ln_sigma.set_data(gens, sv_)
                        ax_sigma.set_ylim(0, max(max(sv_, default=0.25)*1.2, 0.05))
                        ln_newcells.set_data(gens, nc_)
                        if nc_: ax_newcells.set_ylim(0, max(max(nc_)*1.2, 50))
                        cum_txt.set_text(
                            f"cumulative: {sum(sc_)}  "
                            f"cells: {nc_[-1] if nc_ else 0}  "
                            f"avg_solve: {at_[-1] if at_ else 0}t  "
                            f"avg_deaths: {ad_[-1] if ad_ else 0.0:.2f}")
                        ax_solver.relim(); ax_solver.autoscale_view(scaley=False)

                        map_comp    = [c / 4096 * 100 for c in nc_]
                        explore_eff = [c / max(t, 1) for c, t in zip(nc_, ta_)]
                        replan_eff  = [max(0.0, (1.0 - w/max(t,1))*100)
                                       for w, t in zip(wh_, ta_)]
                        learn_rate  = []
                        for gi in range(len(best_)):
                            prev  = best_[max(0, gi-10)]
                            delta = (best_[gi] - prev) / 10
                            learn_rate.append(max(-100, min(100, delta/500)))
                        ln_mapcomp.set_data(gens, map_comp)
                        ln_replan.set_data( gens, replan_eff)
                        ln_learn.set_data(  gens, learn_rate)
                        ln_explore.set_data(gens, explore_eff)
                        if explore_eff:
                            ax_metrics_r.set_ylim(0, max(max(explore_eff)*1.3, 0.1))
                        ax_metrics.relim(); ax_metrics.autoscale_view(scaley=False)
            except Exception as e:
                print(f"[metrics] {e}")

            fig.canvas.draw_idle()
            plt.pause(0.4)

            if state["done"]:
                plt.pause(2.0); break
            if not plt.fignum_exists(fig.number):
                break

    except KeyboardInterrupt:
        pass
    finally:
        proc.terminate()
        try: plt.close(fig)
        except Exception: pass


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description="Silent Cartographer — D* Lite")
    p.add_argument("--maze",  default="maze-alpha/MAZE_1.png")
    p.add_argument("--turns", type=int,   default=10_000)
    p.add_argument("--test",  action="store_true")
    p.add_argument("--weights",       default="best_weights.npy")
    p.add_argument("--test_episodes", type=int, default=5)
    p.add_argument("--test_epsilon",   type=float, default=0.5)
    p.add_argument("--pop",      type=int,   default=80)
    p.add_argument("--gens",     type=int,   default=100)
    p.add_argument("--sigma",    type=float, default=0.20)
    p.add_argument("--decay",    type=float, default=0.995)
    p.add_argument("--eps_eval", type=int,   default=3)
    p.add_argument("--persist",  action="store_true")
    p.add_argument("--phase_k",  type=int,   default=3)
    p.add_argument("--run_id",   default=None)
    p.add_argument("--init_weights", default=None,
                   help="Warm-start population from these weights + mutations")
    p.add_argument("--legacy_pits", action="store_true",
                   help="Re-enable v3 pit-wall seeding (for backward compat)")
    args = p.parse_args()
    build_and_run(args, navigator_name="D* Lite")

if __name__ == "__main__":
    main()