"""
qlearn_agent.py — D* Lite + Linear Q-Learning agent
Silent Cartographer: COSC 4368 AI Spring 2026

Architecture
────────────────────────────────────────────────────
  • D* Lite     : optimal incremental replanning (identical to GA agent)
  • Linear Q-fn : Q(s,a) = θ_a · φ(s)  —  5 actions × 42 features = 210 params
  • Online TD(0) updates every step (no population, no episodes-then-select)
  • Same 5-layer safety stack as GA agent (fire, anti-stall, confusion)

Why linear FA instead of tabular:
  State space is continuous/high-dim (42 features). Tabular Q needs a discrete
  state. Linear FA gives proper Q-learning convergence guarantees and is fast
  to train — full run in minutes vs hours for GA.

Training:
    python train_qlearn.py --maze maze-alpha/MAZE_1.png --episodes 1500 --run_id qlearn_v1

Testing (uses same live_viz.py --test interface):
    python live_viz.py --test --maze maze-beta/MAZE_1.png --weights weights_qlearn_v1.npy
"""
from __future__ import annotations
import numpy as np
import random
from typing import List, Tuple, Optional, Set

from environment import Action, TurnResult, MazeEnvironment
from maze_agent import (
    EvolutionaryAgent, StateEncoder, DStarLite, AgentMemory,
    GRID_SIZE, START_CELL, GOAL_CELL, DIRECTIONS, ACTION_MAP,
    MOVE_ACTIONS, INVERT_MAP, PHASE_EXPLORE, PHASE_OPTIMIZE,
    MAX_FIRE_WAIT, configure,
)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Linear Q-function controller
#    Q(s, a) = θ_a · φ(s)
#    Compatible with NeuralController interface so live_viz.py --test works.
# ─────────────────────────────────────────────────────────────────────────────
class QLearningController:
    """
    Linear function approximation Q-learning.
    Stores one weight vector per action: theta[a] ∈ R^n_features.
    """
    N_FEATURES = 42
    N_ACTIONS  = 5

    def __init__(self, n_features: int = N_FEATURES, n_actions: int = N_ACTIONS):
        self.n_features = n_features
        self.n_actions  = n_actions
        # Small random init — avoids all-zero gradient on first update
        self.theta  = np.random.randn(n_actions, n_features) * 0.01
        # Eligibility traces for TD(λ)
        self.traces = np.zeros((n_actions, n_features), dtype=np.float64)

    def reset_traces(self):
        """Call at the start of each episode."""
        self.traces[:] = 0.0

    # ── Q-value interface ────────────────────────────────────────────────────

    def q_values(self, state: np.ndarray) -> np.ndarray:
        """Q(s, a) for all actions. Shape: (n_actions,)"""
        return self.theta @ state

    def best_action(self, state: np.ndarray, mask: np.ndarray) -> int:
        """Greedy action index, respecting mask."""
        q = self.q_values(state).copy()
        q[mask == 0] = -np.inf
        return int(np.argmax(q))

    # ── TD(λ) update with replacing eligibility traces ────────────────────
    #
    # TD(0) updates only θ[a] for the current step.
    # TD(λ) updates ALL θ[a] for ALL past steps, weighted by how recently
    # they were visited. This propagates the goal reward back through the
    # entire trajectory in a single episode — critical for long-horizon mazes.
    #
    # Replacing traces: e[a,s] = φ(s) (reset rather than accumulate)
    # This avoids trace explosion in loops and is more stable.

    def update(self, state: np.ndarray, action_idx: int, reward: float,
               next_state: np.ndarray, next_mask: np.ndarray,
               done: bool, alpha: float = 0.005, gamma: float = 0.95,
               lambda_: float = 0.9):
        """
        TD(λ) update with replacing eligibility traces.
        δ  = r + γ max_a' Q(s',a') - Q(s,a)
        e[a] ← φ(s)   (replacing trace for taken action)
        θ   ← θ + α·δ·e   (update ALL actions via their traces)
        e   ← γλ·e         (decay all traces)
        """
        q_sa = self.theta[action_idx] @ state
        if done:
            td_error = reward - q_sa
        else:
            q_next = self.q_values(next_state).copy()
            q_next[next_mask == 0] = -np.inf
            best_next = q_next.max() if (next_mask > 0).any() else 0.0
            td_error  = reward + gamma * best_next - q_sa

        # Replacing traces: set current (action, state) trace to φ(s)
        self.traces[action_idx] = state

        # Update all weights via eligibility traces
        self.theta += alpha * td_error * self.traces

        # Decay traces (reset fully on terminal)
        if done:
            self.traces[:] = 0.0
        else:
            self.traces *= gamma * lambda_

    # ── NeuralController-compatible interface ─────────────────────────────────
    # live_viz.py --test and evaluate_fitness call ctrl.forward(state)

    def forward(self, state: np.ndarray) -> np.ndarray:
        """Softmax of Q-values — makes this a drop-in for NeuralController."""
        q = self.q_values(state)
        q = q - q.max()
        e = np.exp(q)
        s = e.sum()
        return e / s if s > 0 else np.ones(self.n_actions) / self.n_actions

    # ── Serialization ─────────────────────────────────────────────────────────

    def get_flat_weights(self) -> np.ndarray:
        return self.theta.ravel().copy()

    def set_flat_weights(self, flat: np.ndarray):
        self.theta = flat.reshape(self.n_actions, self.n_features).copy()

    @property
    def num_params(self) -> int:
        return self.theta.size

    def clone(self) -> "QLearningController":
        c = QLearningController(self.n_features, self.n_actions)
        c.theta = self.theta.copy()
        return c

    def save(self, path: str):
        np.save(path, self.get_flat_weights())

    def load(self, path: str):
        self.set_flat_weights(np.load(path))


# ─────────────────────────────────────────────────────────────────────────────
# 2. Reward function
# ─────────────────────────────────────────────────────────────────────────────
def compute_reward(result: TurnResult, prev_pos: tuple, prev_dist: int,
                   new_dist: int, is_new_cell: bool,
                   best_dist_ever: int = 9999) -> float:
    """
    Dense reward shaping with best-ever distance bonus.

    The best-distance bonus creates a trail of positive signal from start
    to goal: every time the agent reaches a new closest point to the goal,
    it gets a strong reward. Combined with high γ and λ, this signal
    propagates back through the entire trajectory via eligibility traces.
    """
    if result.is_goal_reached:
        return +2000.0

    r = -0.5   # step penalty

    if result.is_dead:
        r -= 100.0
        return r

    # Best-ever distance bonus — strong signal for reaching new frontiers.
    # This creates a reward trail from start to goal that traces can follow.
    if new_dist < best_dist_ever:
        r += 20.0 + (best_dist_ever - new_dist) * 2.0   # bigger bonus for bigger records

    # Regular distance progress
    dist_delta = prev_dist - new_dist
    if dist_delta > 0:
        r += 3.0 * dist_delta
    elif dist_delta < 0:
        r -= 1.0 * abs(dist_delta)

    # New cell bonus
    if is_new_cell:
        r += 1.0

    # Wall hit penalty
    if result.wall_hits:
        r -= 3.0 * result.wall_hits

    return r


# ─────────────────────────────────────────────────────────────────────────────
# 3. Q-Learning Agent
#    Extends EvolutionaryAgent, overrides plan_turn to add TD updates.
#    All 5 safety layers (D* Lite → NN/Q → fire → anti-stall → confusion)
#    are identical to the GA agent.
# ─────────────────────────────────────────────────────────────────────────────
class QLearningAgent(EvolutionaryAgent):
    """
    Online Q-learning agent. Weights update every step via TD(0).
    For inference-only (live_viz --test), set alpha=0.0 or call freeze().
    """

    def __init__(self, controller: QLearningController,
                 goal_cell=None, start_cell=None,
                 epsilon: float = 1.0,
                 alpha:   float = 0.005,
                 gamma:   float = 0.95,
                 lambda_: float = 0.9):
        super().__init__(controller, goal_cell, start_cell,
                         epsilon=epsilon, persist_memory=False)
        self.controller: QLearningController = controller
        self.alpha   = alpha
        self.gamma   = gamma
        self.lambda_ = lambda_
        self._frozen = False   # set True for test/inference

        # State stored between steps for TD update
        self._last_state:      Optional[np.ndarray] = None
        self._last_action_idx: Optional[int]        = None
        self._last_mask:       Optional[np.ndarray] = None
        self._last_dist:       int                  = 9999
        self._best_dist_ever:  int                  = 9999   # best distance across whole episode
        self._unique_cells:    set                  = set()

    def freeze(self):
        """Stop learning — use for test/inference mode."""
        self._frozen = True

    def reset_episode(self):
        super().reset_episode()
        self._last_state      = None
        self._last_action_idx = None
        self._last_mask       = None
        self._last_dist       = 9999
        self._unique_cells    = set()
        self.controller.reset_traces()   # clear eligibility traces for new episode
        self._best_dist_ever = 9999

    def plan_turn(self, last_result: Optional[TurnResult]) -> List[Action]:
        """
        Extended plan_turn that performs TD update before choosing next action.
        Everything after the update is identical to the GA agent's safety stack.
        """
        # ── Memory update + D* Lite notification ─────────────────────────────
        new_walls: Set[Tuple] = set()
        if last_result is not None:
            if last_result.is_dead:
                self.current_pos      = self.start_cell
                self._fire_wait_count = 0
            else:
                self.current_pos = last_result.current_position
            if self.last_action is not None:
                new_walls = self.memory.update(
                    self.prev_pos, self.last_action,
                    last_result, self.last_intended,
                    goal_cell=self.goal_cell)

        if new_walls:
            self.dstar.notify_new_walls(new_walls)

        current_fire = self._current_fire
        always_fire  = self._always_fire

        # One-time always-fire registration
        if not self._always_fire_registered and always_fire:
            permanent_walls: Set[Tuple] = set()
            for fr, fc in always_fire:
                for dr, dc in DIRECTIONS:
                    pr, pc = fr - dr, fc - dc
                    if 0 <= pr < GRID_SIZE and 0 <= pc < GRID_SIZE:
                        wk = (pr, pc, dr, dc)
                        if wk not in self.memory.known_walls:
                            self.memory.known_walls.add(wk)
                            permanent_walls.add(wk)
            if permanent_walls:
                self.dstar.notify_new_walls(permanent_walls)
            self._always_fire_registered = True

        fire_rot_idx = self.env._fire_rot_idx if self.env else 0
        r2, c2       = self.current_pos

        # ── D* Lite hint ──────────────────────────────────────────────────────
        dstar_action = self.dstar.next_action(self.current_pos)

        # ── Encode current state ──────────────────────────────────────────────
        state = self.encoder.encode(self.current_pos, self.memory,
                                    current_fire, fire_rot_idx,
                                    dstar_hint=dstar_action)

        # ── Build action mask ─────────────────────────────────────────────────
        mask = np.ones(len(ACTION_MAP), dtype=np.float32)
        for ai, (dr, dc) in enumerate(DIRECTIONS):
            nr, nc = r2+dr, c2+dc
            if ((r2, c2, dr, dc) in self.memory.known_walls
                    or (nr, nc) in self.memory.known_pits
                    or (nr, nc) in always_fire):
                mask[ai] = 0.0

        # ── TD update for previous transition ────────────────────────────────
        if (not self._frozen
                and last_result is not None
                and self._last_state is not None
                and self._last_action_idx is not None):

            gc = self.goal_cell
            new_pos   = last_result.current_position
            new_dist  = abs(gc[0]-new_pos[0]) + abs(gc[1]-new_pos[1])
            is_new    = new_pos not in self._unique_cells
            reward    = compute_reward(last_result, self.prev_pos,
                                       self._last_dist, new_dist, is_new,
                                       self._best_dist_ever)
            if new_dist < self._best_dist_ever:
                self._best_dist_ever = new_dist
            done      = last_result.is_goal_reached or last_result.is_dead

            self.controller.update(
                state       = self._last_state,
                action_idx  = self._last_action_idx,
                reward      = reward,
                next_state  = state,
                next_mask   = mask,
                done        = done,
                alpha       = self.alpha,
                gamma       = self.gamma,
                lambda_     = self.lambda_,
            )
            self._last_dist = new_dist
            self._unique_cells.add(new_pos)

        # ── Step 2: Q-learning action selection ───────────────────────────────
        probs = self.controller.forward(state)
        masked_probs = probs * mask
        if masked_probs.sum() > 1e-9:
            probs = masked_probs / masked_probs.sum()

        if self.epsilon > 0 and random.random() < self.epsilon:
            # ε-greedy: follow D* Lite on epsilon turns (same as GA agent)
            if (dstar_action is not None
                    and dstar_action in MOVE_ACTIONS
                    and mask[MOVE_ACTIONS.index(dstar_action)] > 0):
                action_idx = MOVE_ACTIONS.index(dstar_action)
            else:
                valid = [ai for ai in range(len(ACTION_MAP)) if mask[ai] > 0]
                action_idx = random.choice(valid) if valid else 0
        else:
            # Greedy: pick action with highest Q-value
            action_idx = self.controller.best_action(state, mask)

        intended = ACTION_MAP[action_idx]

        # ── Step 3: Hardcoded fire safety ─────────────────────────────────────
        if intended in MOVE_ACTIONS:
            idx = MOVE_ACTIONS.index(intended)
            dr, dc = DIRECTIONS[idx]
            nr, nc = r2+dr, c2+dc
            if (nr, nc) in current_fire and (nr, nc) not in always_fire:
                if (self._fire_wait_count < MAX_FIRE_WAIT
                        and self._cell_clears_within(nr, nc, MAX_FIRE_WAIT)):
                    self._fire_wait_count += 1
                    intended   = Action.WAIT
                    action_idx = ACTION_MAP.index(Action.WAIT)
                else:
                    self._fire_wait_count = 0
                    safe = [ai for ai, (adr, adc) in enumerate(DIRECTIONS)
                            if mask[ai] > 0
                            and (r2+adr, c2+adc) not in current_fire]
                    if safe:
                        action_idx = max(safe, key=lambda ai: probs[ai])
                        intended   = ACTION_MAP[action_idx]
                    elif (dstar_action is not None
                          and mask[MOVE_ACTIONS.index(dstar_action)] > 0):
                        intended   = dstar_action
                        action_idx = MOVE_ACTIONS.index(dstar_action)
            else:
                self._fire_wait_count = 0

        # ── Step 4: Anti-stall ────────────────────────────────────────────────
        if intended == Action.WAIT:
            if dstar_action is not None:
                hint_idx = MOVE_ACTIONS.index(dstar_action)
                if mask[hint_idx] > 0:
                    intended   = dstar_action
                    action_idx = hint_idx
                else:
                    valid_moves = [ai for ai, _ in enumerate(DIRECTIONS) if mask[ai] > 0]
                    if valid_moves:
                        action_idx = random.choice(valid_moves)
                        intended   = ACTION_MAP[action_idx]
            else:
                valid_moves = [ai for ai, _ in enumerate(DIRECTIONS) if mask[ai] > 0]
                if valid_moves:
                    action_idx = random.choice(valid_moves)
                    intended   = ACTION_MAP[action_idx]

        # ── Step 5: Confusion inversion ───────────────────────────────────────
        actual = INVERT_MAP[intended] if self.memory.is_confused else intended

        # Store state/action for next TD update
        self._last_state      = state
        self._last_action_idx = action_idx
        self._last_mask       = mask.copy()

        self.prev_pos      = self.current_pos
        self.last_action   = actual
        self.last_intended = intended
        return [actual]


# ─────────────────────────────────────────────────────────────────────────────
# 4. Training loop
# ─────────────────────────────────────────────────────────────────────────────
def train_qlearning(maze_path: str,
                    n_episodes:    int   = 3000,
                    max_turns:     int   = 5000,
                    alpha:         float = 0.001,
                    gamma:         float = 0.999,
                    lambda_:       float = 0.99,
                    epsilon_start: float = 1.0,
                    epsilon_end:   float = 0.05,
                    epsilon_decay: float = 0.9998,
                    stall_window:  int   = 0,
                    verbose_every: int   = 100,
                    weights_path:  str   = "weights_qlearn.npy"):
    """
    Train a Q-learning agent online. Updates weights every step.
    Returns (controller, history).
    """
    from environment import MazeEnvironment
    import time

    env = MazeEnvironment(maze_path)
    configure(env.start_cell, env.goal_cell, GRID_SIZE)
    print(f"[QLEARN] maze={maze_path}")
    print(f"[QLEARN] start={env.start_cell}  goal={env.goal_cell}")
    print(f"[QLEARN] episodes={n_episodes}  max_turns={max_turns}")
    print(f"[QLEARN] α={alpha}  γ={gamma}  ε: {epsilon_start}→{epsilon_end}\n")

    ctrl    = QLearningController()
    agent   = QLearningAgent(ctrl, env.goal_cell, env.start_cell,
                             epsilon=epsilon_start, alpha=alpha,
                             gamma=gamma, lambda_=lambda_)
    agent.env = env

    epsilon = epsilon_start
    history = []
    best_solve_turns = float('inf')
    t0 = time.time()

    for ep in range(1, n_episodes + 1):
        agent.reset_episode()
        agent.epsilon = epsilon
        sp = env.reset()
        agent.current_pos = sp
        agent._last_dist  = abs(env.goal_cell[0]-sp[0]) + abs(env.goal_cell[1]-sp[1])

        last_result       = None
        turns = deaths = wall_hits = 0
        goal_reached      = False
        unique_cells: set = {sp}
        last_new_turn     = 0

        while turns < max_turns:
            actions     = agent.plan_turn(last_result)
            last_result = env.step(actions)
            turns      += 1

            prev_size = len(unique_cells)
            unique_cells.add(last_result.current_position)
            if len(unique_cells) > prev_size:
                last_new_turn = turns

            wall_hits += last_result.wall_hits
            if last_result.is_dead:
                deaths += 1
            if last_result.is_goal_reached:
                goal_reached = True
                break

            # Early stopping: disabled when stall_window=0
            if stall_window > 0 and turns - last_new_turn > stall_window:
                break

        epsilon = max(epsilon_end, epsilon * epsilon_decay)

        rec = {
            "episode":      ep,
            "goal_reached": goal_reached,
            "turns":        turns,
            "deaths":       deaths,
            "wall_hits":    wall_hits,
            "cells":        len(unique_cells),
            "epsilon":      epsilon,
        }
        history.append(rec)

        if goal_reached and turns < best_solve_turns:
            best_solve_turns = turns
            ctrl.save(weights_path)

        if ep % verbose_every == 0 or ep == 1:
            solved_recent = sum(1 for h in history[-verbose_every:] if h["goal_reached"])
            avg_turns     = sum(h["turns"]  for h in history[-verbose_every:]) / min(ep, verbose_every)
            avg_deaths    = sum(h["deaths"] for h in history[-verbose_every:]) / min(ep, verbose_every)
            elapsed       = time.time() - t0
            print(f"  Ep {ep:5d}/{n_episodes}"
                  f"  solved(last {verbose_every}): {solved_recent:3d}"
                  f"  avg_t: {avg_turns:6.0f}"
                  f"  avg_d: {avg_deaths:.2f}"
                  f"  ε: {epsilon:.3f}"
                  f"  best: {best_solve_turns if best_solve_turns < float('inf') else '—'}"
                  f"  [{elapsed:.0f}s]")

    total = time.time() - t0
    n_solved = sum(1 for h in history if h["goal_reached"])
    print(f"\n{'━'*60}")
    print(f"  ✓ Q-learning done  ({n_episodes} eps  {total/60:.1f} min)")
    print(f"  Solved: {n_solved}/{n_episodes} ({100*n_solved/n_episodes:.1f}%)")
    print(f"  Best solve: {best_solve_turns if best_solve_turns < float('inf') else 'never'} turns")
    print(f"  Weights: {weights_path}")
    print(f"{'━'*60}\n")

    # Save final weights (best episode weights already saved above)
    if not goal_reached:
        ctrl.save(weights_path)

    return ctrl, history