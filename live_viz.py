"""
live_viz.py — Live EC training dashboard
Silent Cartographer: COSC 4368 AI Spring 2026

  • Training runs in a separate PROCESS (bypasses Python's GIL entirely)
  • Map rendered as a single pre-composed numpy RGBA array
  • Queue-based communication: training process → display process
  • Fire pits animate using env's own rotate_fire_clusters() — always correct
  • Agent traversal streams live during best-individual replay

Run:
    python live_viz.py --maze MAZE_1.png [--pop 60] [--gens 80] [--turns 10000]
"""
from __future__ import annotations
import argparse, time
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
import maze_agent as ma
from maze_agent import (NeuralController, EvolutionaryAgent,
                        GeneticAlgorithm, evaluate_fitness, GRID_SIZE,
                        PHASE_EXPLORE, PHASE_OPTIMIZE)

CELL_SIZE = 16
MARK      = 5


# ─────────────────────────────────────────────────────────────────────────────
# Training worker  (runs in its own process)
# ─────────────────────────────────────────────────────────────────────────────
def training_worker(maze_path: str, args_dict: dict, state_q: Queue):
    import maze_agent as ma
    from environment import MazeEnvironment
    from maze_agent import (NeuralController, GeneticAlgorithm,
                             evaluate_fitness, GRID_SIZE,
                             PHASE_EXPLORE, PHASE_OPTIMIZE)

    env = MazeEnvironment(maze_path)
    ma.configure(env.start_cell, env.goal_cell, GRID_SIZE)
    print(f"[TRAIN] start={env.start_cell}  goal={env.goal_cell}")

    class _VisualGA(GeneticAlgorithm):
        def step(self, env, eval_episodes=1, eval_turns=10_000,
                 epsilon=0.05, persist=False):
            import random
            import numpy as np
            gen_num            = self.generation + 1
            best_agent         = None
            gen_solvers        = 0
            gen_best_new_cells = 0

            print(f"\n{'─'*60}")
            print(f"  Gen {gen_num:3d} START  σ={self.mut_sigma:.4f}"
                  f"  pop={self.pop_size}  phase=[{self.phase}]"
                  f"  goal_known={ma.GOAL_KNOWN}")
            print(f"{'─'*60}")

            gen_new_cells_list = []
            for i, ctrl in enumerate(self.population):
                fit, ep_agent = evaluate_fitness(
                    ctrl, env,
                    goal_cell=ma.GOAL_CELL, start_cell=ma.START_CELL,
                    episodes=eval_episodes, max_turns=eval_turns,
                    epsilon=epsilon, persist=persist,
                    phase=self.phase,
                    step_q=None,
                )
                self.fitness[i] = fit
                agent_new = len(ep_agent.memory.visit_count)
                gen_new_cells_list.append(agent_new)

                if ep_agent.goal_reached:
                    gen_solvers += 1

                if best_agent is None or fit >= self.fitness[:i+1].max():
                    best_agent = ep_agent

                if (i+1) % 5 == 0 or (i+1) == self.pop_size:
                    print(f"  [{i+1:3d}/{self.pop_size}]"
                          f"  best={self.fitness[:i+1].max():+.0f}"
                          f"  mean={self.fitness[:i+1].mean():+.0f}"
                          f"  std={self.fitness[:i+1].std():.0f}"
                          f"  solvers={gen_solvers}"
                          f"  new_cells(best)={max(gen_new_cells_list[:i+1])}"
                          f"  [{self.phase}]"
                          f"  goal_known={ma.GOAL_KNOWN}")

            gen_best_new_cells = max(gen_new_cells_list) if gen_new_cells_list else 0

            best_idx = int(np.argmax(self.fitness))
            tag = ""
            if self.fitness[best_idx] > self.best_fitness:
                self.best_fitness    = float(self.fitness[best_idx])
                self.best_individual = self.population[best_idx].clone()
                tag = "★ NEW BEST"

            rec = {
                "generation": self.generation,
                "best":       float(self.fitness.max()),
                "mean":       float(self.fitness.mean()),
                "std":        float(self.fitness.std()),
                "worst":      float(self.fitness.min()),
                "sigma":      self.mut_sigma,
                "solvers":    gen_solvers,
                "new_cells":  gen_best_new_cells,
                "phase":      self.phase,
                "goal_known": ma.GOAL_KNOWN,
            }
            self.history.append(rec)
            print(f"\n{'='*60}")
            print(f"  Gen {gen_num:3d} END  best={rec['best']:+.0f}"
                  f"  mean={rec['mean']:+.0f}  std={rec['std']:.0f}"
                  f"  σ={self.mut_sigma:.4f}")
            print(f"           solvers={gen_solvers}/{self.pop_size}"
                  f"  new_cells(best)={gen_best_new_cells}"
                  f"  [{self.phase}]  goal_known={ma.GOAL_KNOWN}  {tag}")
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
            goal_label  = "🎯 GOAL KNOWN" if ma.GOAL_KNOWN else "❓ EXPLORING"

            snapshot = {
                "type":       "gen",
                "visit_map":  vmap,
                "agent_pos":  best_agent.current_pos if best_agent else ma.START_CELL,
                "agent_path": list(best_agent.memory.path[-5000:]) if best_agent else [],
                "history":    list(self.history),
                "status":     (f"Gen {gen_num}  |  best={rec['best']:+.0f}"
                               f"  mean={rec['mean']:+.0f}"
                               f"  σ={self.mut_sigma:.4f}"
                               f"  solvers={gen_solvers}/{self.pop_size}"
                               f"  {phase_label}  {goal_label}  {tag}"),
                "done":       False,
            }
            try:
                state_q.put_nowait(snapshot)
            except _queue.Full:
                pass

            # Phase transition check
            self._maybe_switch_phase(gen_solvers, env=env)

            # ── Breed next generation ─────────────────────────────────────────
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

            # Random immigrants injected into bottom (non-elite) slots only
            n_immigrants = max(1, int(self.immigrant_frac * self.pop_size))
            inject_start = max(elite_k, self.pop_size - n_immigrants)
            for k in range(self.pop_size - inject_start):
                new_pop[inject_start + k] = NeuralController(self.layer_sizes)

            self.population = new_pop
            self.mut_sigma  = max(self.min_mut_sigma,
                                  self.mut_sigma * self.mut_decay)
            self.generation += 1
            return self.best_individual

    ga = _VisualGA(
        pop_size=args_dict["pop"],
        init_mut_sigma=args_dict["sigma"],
        mut_decay=args_dict["decay"],
        phase_switch_k=args_dict["phase_k"],
    )
    print(f"[GA]  {ga.pop_size} individuals × "
          f"{ga.population[0].num_params} params"
          f"  phase_switch_k={ga.phase_switch_k}\n")

    from datetime import datetime
    import shutil

    run_id       = args_dict.get("run_id") or datetime.now().strftime("%Y%m%d_%H%M%S")
    weights_file = f"weights_{run_id}.npy"
    print(f"[TRAIN] weights will be saved to  {weights_file}  (alias → best_weights.npy)\n")

    epsilon   = 0.15
    last_best = -float("inf")
    for gen_i in range(args_dict["gens"]):
        ga.step(env,
                eval_episodes=args_dict["eps_eval"],
                eval_turns=args_dict["turns"],
                epsilon=epsilon,
                persist=args_dict["persist"])
        epsilon = max(0.02, epsilon * 0.97)

        if ga.best_individual:
            ga.best_individual.save(weights_file)
            shutil.copy2(weights_file, "best_weights.npy")

            if ga.best_fitness > last_best:
                last_best = ga.best_fitness
                print(f"\n  ── Live replay (gen {gen_i+1},"
                      f" fitness={ga.best_fitness:+.0f}"
                      f" [{ga.phase}]"
                      f" goal_known={ma.GOAL_KNOWN}) ──")
                evaluate_fitness(
                    ga.best_individual, env,
                    goal_cell=ma.GOAL_CELL, start_cell=ma.START_CELL,
                    episodes=1, max_turns=args_dict["turns"],
                    epsilon=0.0, verbose=True,
                    step_q=state_q, step_interval=100,
                    phase=ga.phase,
                )

    try:
        state_q.put({"type": "done",
                     "done": True,
                     "status": f"✓ Done — {args_dict['gens']} gens"
                               f"  best={ga.best_fitness:+.0f}"
                               f"  saved → {weights_file}"
                               f"  [{ga.phase}]"
                               f"  goal_known={ma.GOAL_KNOWN}",
                     "history": list(ga.history)})
    except Exception:
        pass
    print(f"✓ Training done.  Weights saved to {weights_file}")


# ─────────────────────────────────────────────────────────────────────────────
# Test worker  (runs in its own process — no GA, pure evaluation)
# ─────────────────────────────────────────────────────────────────────────────
def test_worker(maze_path: str, args_dict: dict, state_q: Queue):
    """
    Load saved weights and run N clean episodes (epsilon=0, optimize phase).
    In test mode the goal location is always known (we're evaluating a trained agent).
    """
    import maze_agent as ma
    from environment import MazeEnvironment
    from maze_agent import NeuralController, evaluate_fitness, PHASE_OPTIMIZE

    env = MazeEnvironment(maze_path)
    ma.configure(env.start_cell, env.goal_cell)
    # In test mode the agent already trained — goal is known from the start
    ma.set_goal_known(True)

    weights_path = args_dict.get("weights", "best_weights.npy")
    n_episodes   = args_dict.get("test_episodes", 5)

    ctrl = NeuralController()
    try:
        ctrl.load(weights_path)
        print(f"[TEST] loaded weights from {weights_path}")
    except Exception as e:
        print(f"[TEST] ERROR loading weights: {e}")
        state_q.put({"type": "test_result", "error": str(e), "done": True,
                     "results": [], "status": f"ERROR: {e}"})
        return

    print(f"[TEST] maze={maze_path}  start={env.start_cell}  goal={env.goal_cell}")
    print(f"[TEST] {n_episodes} episodes  epsilon=0  phase=optimize\n")

    results = []
    for ep in range(n_episodes):
        print(f"  ── Test episode {ep+1}/{n_episodes} ──")
        fit, agent = evaluate_fitness(
            ctrl, env,
            goal_cell=ma.GOAL_CELL, start_cell=ma.START_CELL,
            episodes=1, max_turns=args_dict["turns"],
            epsilon=0.0, verbose=True,
            step_q=state_q, step_interval=15,
            phase=PHASE_OPTIMIZE,
        )

        ep_stats = env.get_episode_stats()
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
                  f"  avg turns: {avg_turns:.0f}"
                  f"  avg deaths: {avg_deaths:.1f}")
        print(f"  {status}\n")

        mem  = agent.memory
        vmap = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
        for (r, c), cnt in mem.visit_count.items():
            if 0 <= r < GRID_SIZE and 0 <= c < GRID_SIZE:
                vmap[r, c] = np.log1p(cnt)

        try:
            state_q.put_nowait({
                "type":       "test_result",
                "visit_map":  vmap,
                "agent_pos":  agent.current_pos,
                "agent_path": list(mem.path),
                "results":    list(results),
                "status":     status,
                "done":       False,
            })
        except Exception:
            pass

    n_solved   = sum(r["solved"]  for r in results)
    avg_turns  = sum(r["turns"]   for r in results) / len(results)
    avg_deaths = sum(r["deaths"]  for r in results) / len(results)
    best_turns = min((r["turns"]  for r in results if r["solved"]), default=None)

    print(f"\n{'━'*50}")
    print(f"  TEST COMPLETE  ({n_episodes} episodes)")
    print(f"  Goal rate   : {n_solved}/{n_episodes}  ({100*n_solved/n_episodes:.0f}%)")
    print(f"  Avg turns   : {avg_turns:.0f}")
    print(f"  Avg deaths  : {avg_deaths:.2f}")
    if best_turns:
        print(f"  Best solve  : {best_turns} turns")
    print(f"{'━'*50}\n")

    done_status = (f"✓ Test done  |  goal rate: {n_solved}/{n_episodes}"
                   f"  avg turns: {avg_turns:.0f}"
                   f"  avg deaths: {avg_deaths:.2f}"
                   + (f"  best: {best_turns}t" if best_turns else ""))
    try:
        state_q.put({"type": "test_result", "results": results,
                     "status": done_status, "done": True})
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Fire rotation: precompute all 4 states using env's own correct logic
# ─────────────────────────────────────────────────────────────────────────────
def precompute_fire_rotations(env) -> list:
    return list(env._fire_rotation_states)


# ─────────────────────────────────────────────────────────────────────────────
# Pixel / overlay helpers
# ─────────────────────────────────────────────────────────────────────────────
def _mark(ov, r, c, rgba, half=MARK):
    h, w = ov.shape[:2]
    py   = r * CELL_SIZE + CELL_SIZE // 2
    px   = c * CELL_SIZE + CELL_SIZE // 2
    ov[max(0, py-half):min(h, py+half+1),
       max(0, px-half):min(w, px+half+1)] = rgba


def build_static_overlay(env, img_h, img_w):
    ov = np.zeros((img_h, img_w, 4), dtype=np.uint8)
    for r, c in env.loader.teleport_purple: _mark(ov, r, c, (160,  50, 255, 200))
    for r, c in env.loader.teleport_orange: _mark(ov, r, c, (255, 140,   0, 200))
    for r, c in env.loader.teleport_green:  _mark(ov, r, c, ( 30, 200,  70, 200))
    sr, sc = env.start_cell; _mark(ov, sr, sc, (  0, 230, 100, 230), half=8)
    gr, gc = env.goal_cell;  _mark(ov, gr, gc, (  0, 220, 255, 230), half=8)
    return ov


def compose_map_frame(maze_rgb, stat_ov, fire_pits,
                      visit_map, path, agent_pos, img_h, img_w):
    frame = np.empty((img_h, img_w, 4), dtype=np.uint8)
    frame[:, :, :3] = maze_rgb[:img_h, :img_w]
    frame[:, :,  3] = 255

    smask = stat_ov[:, :, 3] > 0
    frame[smask] = stat_ov[smask]

    ov64    = np.zeros((GRID_SIZE, GRID_SIZE, 4), dtype=np.uint8)
    vmax    = visit_map.max() if visit_map.max() > 0 else 1.0
    visited = visit_map > 0
    if visited.any():
        v = (visit_map[visited] / vmax).clip(0, 1)
        ov64[visited, 0] = ( 20         * v).astype(np.uint8)
        ov64[visited, 1] = ( 80 + 150*v    ).astype(np.uint8)
        ov64[visited, 2] = (180 +  75*v    ).astype(np.uint8)
        ov64[visited, 3] = (160 * np.minimum(v * 3, 1)).astype(np.uint8)

    if path:
        n    = max(1, len(path) - 1)
        rows = np.array([p[0] for p in path], dtype=np.int32)
        cols = np.array([p[1] for p in path], dtype=np.int32)
        mask = (rows >= 0) & (rows < GRID_SIZE) & (cols >= 0) & (cols < GRID_SIZE)
        rows, cols = rows[mask], cols[mask]
        if len(rows):
            idx = np.where(mask)[0]
            t   = idx / n
            ov64[rows, cols, 0] = (30  + 100 * t).astype(np.uint8)
            ov64[rows, cols, 1] = (80  + 175 * t).astype(np.uint8)
            ov64[rows, cols, 2] = 255
            ov64[rows, cols, 3] = 220

    big   = np.repeat(np.repeat(ov64, CELL_SIZE, axis=0), CELL_SIZE, axis=1)
    h_f   = min(big.shape[0], img_h)
    w_f   = min(big.shape[1], img_w)
    alpha = big[:h_f, :w_f, 3:4].astype(np.float32) / 255.0
    frame[:h_f, :w_f, :3] = (
        frame[:h_f, :w_f, :3] * (1 - alpha) +
        big[:h_f, :w_f, :3]   * alpha
    ).astype(np.uint8)

    if fire_pits:
        fp = np.array(list(fire_pits), dtype=np.int32)
        fmask_cells = ((fp[:, 0] >= 0) & (fp[:, 0] < GRID_SIZE) &
                       (fp[:, 1] >= 0) & (fp[:, 1] < GRID_SIZE))
        fp = fp[fmask_cells]
        if len(fp):
            fmask = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)
            fmask[fp[:, 0], fp[:, 1]] = True
            fbig = np.repeat(np.repeat(fmask, CELL_SIZE, axis=0), CELL_SIZE, axis=1)
            frame[:h_f, :w_f][fbig[:h_f, :w_f]] = [255, 40, 40, 230]

    ar, ac = agent_pos
    py = ar * CELL_SIZE + CELL_SIZE // 2
    px = ac * CELL_SIZE + CELL_SIZE // 2
    frame[max(0, py-4):min(img_h, py+5),
          max(0, px-4):min(img_w, px+5)] = [255, 34, 68, 255]

    return frame


# ─────────────────────────────────────────────────────────────────────────────
# Display  (shared by train and test modes)
# ─────────────────────────────────────────────────────────────────────────────
def build_and_run(args):
    is_test  = getattr(args, "test", False)
    mode_str = "Test" if is_test else "Training"

    maze_rgb     = np.array(Image.open(args.maze).convert("RGB"))
    img_h, img_w = maze_rgb.shape[:2]

    print(f"Loading maze ({mode_str} mode) from {args.maze}…")
    _env0   = MazeEnvironment(args.maze)
    stat_ov = build_static_overlay(_env0, img_h, img_w)

    _fire = {"rotations": precompute_fire_rotations(_env0), "rot": 0}

    state = {
        "visit_map":    np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32),
        "agent_pos":    _env0.start_cell,
        "agent_path":   [],
        "fire_pits":    _fire["rotations"][0],
        "history":      [],
        "test_results": [],
        "status":       "Initialising…",
        "done":         False,
    }

    state_q   = Queue(maxsize=400)
    args_dict = vars(args)

    target = test_worker if is_test else training_worker
    proc   = Process(target=target, args=(args.maze, args_dict, state_q), daemon=True)
    proc.start()

    # ── Figure ───────────────────────────────────────────────────────────────
    plt.style.use("dark_background")
    title_suffix = "— TEST EVALUATION" if is_test else "— Neuroevolution Live Training"
    fig = plt.figure(figsize=(17, 8.5), facecolor="#0a0f18")
    fig.suptitle(f"Silent Cartographer {title_suffix}",
                 color="#c8d8e8", fontsize=13, fontweight="bold", y=0.98)

    gs_outer = gridspec.GridSpec(2, 2, figure=fig,
                                 hspace=0.06, wspace=0.25,
                                 height_ratios=[15, 1],
                                 left=0.03, right=0.97, top=0.93, bottom=0.05)
    ax_map  = fig.add_subplot(gs_outer[0, 0])
    ax_stat = fig.add_subplot(gs_outer[1, :])

    gs_right = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=gs_outer[0, 1],
        hspace=0.45, height_ratios=[3, 2])
    ax_right  = fig.add_subplot(gs_right[0])
    ax_solver = fig.add_subplot(gs_right[1])

    ax_map.axis("off")
    ax_map.set_title("Agent — Known World", color="#c8d0e0", fontsize=11, pad=6)

    init_frame = compose_map_frame(
        maze_rgb, stat_ov, state["fire_pits"],
        state["visit_map"], [], state["agent_pos"], img_h, img_w
    )
    im_map = ax_map.imshow(init_frame, origin="upper",
                           interpolation="nearest", zorder=0)

    legend_p = [
        mpatches.Patch(color=(0.08, 0.31, 0.70), label="Explored (low)"),
        mpatches.Patch(color=(0.00, 1.00, 0.80), label="Explored (high)"),
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

    ax_right.set_facecolor("#060b12")
    for sp in ax_right.spines.values(): sp.set_edgecolor("#1e2a3a")

    if not is_test:
        ax_right.set_title("Population Fitness — Live", color="#c8d0e0",
                            fontsize=11, pad=6)
        ax_right.set_xlabel("Generation", color="#7a8a9a", fontsize=9)
        ax_right.set_ylabel("Fitness",    color="#7a8a9a", fontsize=9)
        ax_right.tick_params(colors="#4a5a6a", labelsize=8)
        ax_right.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax_right.set_xlim(1, max(2, args.gens))
        ax_right.grid(alpha=0.12, color="#2a3a4a")
        ax_right.axhline(0, color="#2a4060", lw=0.8, ls="--")

        ln_best,  = ax_right.plot([], [], lw=2.2, color="#00e5ff", label="Best",  zorder=3)
        ln_mean,  = ax_right.plot([], [], lw=1.5, color="#2979ff", label="Mean",  zorder=2)
        ln_worst, = ax_right.plot([], [], lw=1.0, color="#455a6a", label="Worst", zorder=1, ls="--")
        fill_std  = [ax_right.fill_between([], [], [], color="#2979ff", alpha=0.12, zorder=0)]
        ax_right.legend(loc="lower right", fontsize=8,
                        facecolor="#0a1520", edgecolor="#1e2a3a", labelcolor="#c8d0e0")

        ax_solver.set_facecolor("#060b12")
        for sp in ax_solver.spines.values(): sp.set_edgecolor("#1e2a3a")
        ax_solver.set_title("Solvers / Gen  +  σ", color="#c8d0e0", fontsize=9, pad=4)
        ax_solver.set_xlabel("Generation", color="#7a8a9a", fontsize=8)
        ax_solver.tick_params(colors="#4a5a6a", labelsize=7)
        ax_solver.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=20))
        ax_solver.set_xlim(0.5, max(2, args.gens) + 0.5)
        ax_solver.set_ylim(0, max(2, args_dict.get("pop", 60)))
        ax_solver.grid(alpha=0.10, color="#2a3a4a")
        ax_solver.axhline(0, color="#2a4060", lw=0.6, ls="--")

        ax_sigma = ax_solver.twinx()
        ax_sigma.tick_params(colors="#4a5a6a", labelsize=7)
        ax_sigma.set_ylabel("σ", color="#ffaa44", fontsize=8)
        ax_sigma.set_ylim(0, 0.25)

        bar_solvers = [None]
        ln_sigma,   = ax_sigma.plot([], [], lw=1.5, color="#ffaa44",
                                    label="σ", ls="--", zorder=2)

        ax_newcells = ax_solver.twinx()
        ax_newcells.spines["right"].set_position(("outward", 40))
        ax_newcells.tick_params(colors="#5588aa", labelsize=6)
        ax_newcells.set_ylabel("New cells", color="#44bbff", fontsize=7)
        ln_newcells, = ax_newcells.plot([], [], lw=1.2, color="#44bbff",
                                        label="new cells", ls="-", alpha=0.8, zorder=1)

        cum_txt = ax_solver.text(0.98, 0.95, "", transform=ax_solver.transAxes,
                                 ha="right", va="top", fontsize=7,
                                 color="#aaccdd", fontfamily="monospace")
        ax_solver.set_ylabel("Solvers", color="#00e5aa", fontsize=8)
    else:
        ax_solver.set_visible(False)
        ax_right.set_title("Test Results — Per Episode", color="#c8d0e0",
                            fontsize=11, pad=6)
        ax_right.tick_params(left=False, bottom=False,
                             labelleft=False, labelbottom=False)
        ax_right.set_xlim(0, 1); ax_right.set_ylim(0, 1)

    ax_stat.set_facecolor("#0a1520"); ax_stat.axis("off")
    stat_txt = ax_stat.text(0.5, 0.5, "Initialising…",
                            transform=ax_stat.transAxes,
                            ha="center", va="center", fontsize=9.5,
                            color="#8ab4d4", fontfamily="monospace")

    _cache = {"hist_len": 0, "fire_tick": 0, "phase_marked": False,
              "goal_marked": False, "test_len": 0, "solver_bars_drawn": 0}

    def _render_test_table(results):
        ax_right.cla()
        ax_right.set_facecolor("#060b12")
        ax_right.set_title("Test Results — Per Episode", color="#c8d0e0",
                            fontsize=11, pad=6)
        ax_right.tick_params(left=False, bottom=False,
                             labelleft=False, labelbottom=False)
        ax_right.set_xlim(0, 1); ax_right.set_ylim(0, 1)
        for sp in ax_right.spines.values(): sp.set_edgecolor("#1e2a3a")

        n     = len(results)
        n_k   = args_dict.get("test_episodes", 10)
        row_h = min(0.075, 0.85 / max(n_k, 1))
        hdr_y = 0.93
        ax_right.text(0.08, hdr_y, "Ep",     color="#7a8a9a", fontsize=8, va="top")
        ax_right.text(0.22, hdr_y, "Result", color="#7a8a9a", fontsize=8, va="top")
        ax_right.text(0.46, hdr_y, "Turns",  color="#7a8a9a", fontsize=8, va="top")
        ax_right.text(0.64, hdr_y, "Deaths", color="#7a8a9a", fontsize=8, va="top")
        ax_right.text(0.82, hdr_y, "Cells",  color="#7a8a9a", fontsize=8, va="top")
        ax_right.axhline(hdr_y - 0.02, color="#1e2a3a", lw=0.8)

        for i, r in enumerate(results):
            y     = hdr_y - 0.06 - i * row_h
            color = "#00e5aa" if r["solved"] else "#ff4466"
            label = "✓ SOLVED" if r["solved"] else "✗ TIMEOUT"
            ax_right.text(0.08, y, str(r["episode"]),  color="#c8d0e0", fontsize=7.5, va="top")
            ax_right.text(0.22, y, label,              color=color,     fontsize=7.5, va="top")
            ax_right.text(0.46, y, str(r["turns"]),    color="#c8d0e0", fontsize=7.5, va="top")
            ax_right.text(0.64, y, str(r["deaths"]),   color="#c8d0e0", fontsize=7.5, va="top")
            ax_right.text(0.82, y, str(r["explored"]), color="#c8d0e0", fontsize=7.5, va="top")

        if n > 0:
            n_solved   = sum(r["solved"] for r in results)
            avg_turns  = sum(r["turns"]  for r in results) / n
            avg_deaths = sum(r["deaths"] for r in results) / n
            best_t     = min((r["turns"] for r in results if r["solved"]), default=None)
            summary_y  = hdr_y - 0.06 - n_k * row_h - 0.05
            ax_right.axhline(summary_y + 0.03, color="#1e2a3a", lw=0.8)
            rate_color = ("#00e5aa" if n_solved == n
                          else "#ffaa00" if n_solved > 0 else "#ff4466")
            ax_right.text(0.08, summary_y,
                          f"Rate: {n_solved}/{n}   Avg turns: {avg_turns:.0f}"
                          f"   Avg deaths: {avg_deaths:.1f}"
                          + (f"   Best: {best_t}t" if best_t else ""),
                          color=rate_color, fontsize=8, va="top", fontweight="bold")

    def update(_frame):
        step_msg = None
        last_msg = None
        while True:
            try:
                msg   = state_q.get_nowait()
                mtype = msg.get("type", "gen")
                if mtype == "step":
                    step_msg = msg
                else:
                    last_msg = msg
            except _queue.Empty:
                break

        if step_msg is not None:
            state["agent_pos"]  = step_msg["agent_pos"]
            state["agent_path"] = step_msg["agent_path"]

        if last_msg is not None:
            mtype = last_msg.get("type", "gen")
            if "visit_map" in last_msg:
                state["visit_map"]  = last_msg["visit_map"]
                state["agent_pos"]  = last_msg["agent_pos"]
                state["agent_path"] = last_msg["agent_path"]
            if mtype == "test_result" and "results" in last_msg:
                state["test_results"] = last_msg["results"]
            if mtype == "gen" and "history" in last_msg:
                state["history"] = last_msg["history"]
            state["status"] = last_msg.get("status", state["status"])
            state["done"]   = last_msg.get("done",   False)

        txt = state["status"]
        stat_txt.set_text("✓  " + txt if state["done"] else txt)

        _cache["fire_tick"] += 1
        if _cache["fire_tick"] >= 5:
            _cache["fire_tick"] = 0
            _fire["rot"] = (_fire["rot"] + 1) % 4
            state["fire_pits"] = _fire["rotations"][_fire["rot"]]

        frame = compose_map_frame(
            maze_rgb, stat_ov, state["fire_pits"],
            state["visit_map"], state["agent_path"],
            state["agent_pos"], img_h, img_w
        )
        im_map.set_data(frame)

        if is_test:
            tr = state["test_results"]
            if len(tr) != _cache["test_len"]:
                _cache["test_len"] = len(tr)
                _render_test_table(tr)
        else:
            try:
                hist = state["history"]
                if len(hist) != _cache["hist_len"]:
                    _cache["hist_len"] = len(hist)
                    gens = [h["generation"] + 1 for h in hist]
                    ln_best.set_data( gens, [h["best"]  for h in hist])
                    ln_mean.set_data( gens, [h["mean"]  for h in hist])
                    ln_worst.set_data(gens, [h["worst"] for h in hist])
                    fill_std[0].remove()
                    lo = [h["mean"] - h["std"] for h in hist]
                    hi = [h["mean"] + h["std"] for h in hist]
                    fill_std[0] = ax_right.fill_between(gens, lo, hi,
                                                        color="#2979ff", alpha=0.12, zorder=0)
                    ax_right.relim(); ax_right.autoscale_view()

                    # Mark phase switch
                    if not _cache["phase_marked"]:
                        for h in hist:
                            if h.get("phase") == PHASE_OPTIMIZE:
                                switch_gen = h["generation"] + 1
                                ax_right.axvline(switch_gen, color="#ffaa00",
                                                 lw=1.2, ls="--", alpha=0.7, zorder=4)
                                ax_right.text(switch_gen + 0.2,
                                              ax_right.get_ylim()[1] * 0.95,
                                              "⚡ optimize", color="#ffaa00",
                                              fontsize=7, va="top")
                                ax_solver.axvline(switch_gen, color="#ffaa00",
                                                  lw=1.0, ls="--", alpha=0.5, zorder=4)
                                _cache["phase_marked"] = True
                                break

                    # Mark goal discovery
                    if not _cache["goal_marked"]:
                        for h in hist:
                            if h.get("goal_known"):
                                disc_gen = h["generation"] + 1
                                ax_right.axvline(disc_gen, color="#00ff99",
                                                 lw=1.2, ls=":", alpha=0.8, zorder=4)
                                ax_right.text(disc_gen + 0.2,
                                              ax_right.get_ylim()[0] * 0.95
                                              if ax_right.get_ylim()[0] < 0 else 0,
                                              "🎯 found", color="#00ff99",
                                              fontsize=7, va="bottom")
                                _cache["goal_marked"] = True
                                break

                    solver_counts  = [h["solvers"]   for h in hist]
                    sigma_vals     = [h["sigma"]     for h in hist]
                    new_cell_vals  = [h.get("new_cells", 0) for h in hist]
                    cumulative     = sum(solver_counts)
                    pop_size       = args_dict.get("pop", 60)

                    if len(solver_counts) != _cache["solver_bars_drawn"]:
                        _cache["solver_bars_drawn"] = len(solver_counts)
                        if bar_solvers[0] is not None:
                            bar_solvers[0].remove()
                        colors = ["#00e5aa" if s > 0 else "#1e3a2a" for s in solver_counts]
                        bar_solvers[0] = ax_solver.bar(
                            gens, solver_counts, color=colors, alpha=0.7,
                            width=0.7, zorder=2)
                        for g, s in zip(gens, solver_counts):
                            if s > 0:
                                ax_solver.text(g, s + 0.2, str(s),
                                               ha="center", va="bottom",
                                               fontsize=6, color="#00e5aa")
                        ax_solver.set_ylim(0, max(max(solver_counts, default=0) * 1.3 + 1, 5))
                        ln_sigma.set_data(gens, sigma_vals)
                        ax_sigma.set_ylim(0, max(max(sigma_vals, default=0.25) * 1.2, 0.05))
                        ln_newcells.set_data(gens, new_cell_vals)
                        if new_cell_vals:
                            ax_newcells.set_ylim(0, max(max(new_cell_vals) * 1.2, 50))
                        cum_txt.set_text(
                            f"cumulative: {cumulative}  "
                            f"cells: {new_cell_vals[-1] if new_cell_vals else 0}")
                        ax_solver.relim(); ax_solver.autoscale_view(scaley=False)

            except Exception as e:
                print(f"[display] chart update error: {e}")

        return [im_map, stat_txt]

    anim = FuncAnimation(fig, update, interval=500,
                         blit=False, cache_frame_data=False)
    plt.show()
    proc.terminate()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(
        description="Silent Cartographer — Live Dashboard (train or test)")

    # Shared
    p.add_argument("--maze",  default="MAZE_1.png", help="Maze image file")
    p.add_argument("--turns", type=int, default=10_000, help="Max turns per episode")

    # Mode
    p.add_argument("--test", action="store_true",
                   help="Test mode: evaluate saved weights, no GA")

    # Test-only
    p.add_argument("--weights",       default="best_weights.npy",
                   help="[test] weights file to load")
    p.add_argument("--test_episodes", type=int, default=10,
                   help="[test] number of evaluation episodes")

    # Train-only
    p.add_argument("--pop",      type=int,   default=60)
    p.add_argument("--gens",     type=int,   default=80)
    p.add_argument("--sigma",    type=float, default=0.20)
    p.add_argument("--decay",    type=float, default=0.995)
    p.add_argument("--eps_eval", type=int,   default=1)
    p.add_argument("--persist",  action="store_true",
                   help="[train] shared memory across episodes")
    p.add_argument("--phase_k",  type=int, default=5,
                   help="[train] cumulative solvers to trigger OPTIMIZE phase")
    p.add_argument("--run_id",   default=None,
                   help="[train] tag for this run's weight file")

    args = p.parse_args()
    build_and_run(args)

if __name__ == "__main__":
    main()