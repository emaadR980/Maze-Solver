"""
evolutionary_maze_agent.py
Silent Cartographer: Maze Navigation Project — COSC 4368 AI Spring 2026

Approach: Neuroevolution via Genetic Algorithm
  • Each individual  = weight vector of a small MLP (the agent's "brain")
  • Phenotype        = navigation policy: encoded_state → action
  • Fitness          = goal bonus + proximity reward + exploration − death penalty
  • Selection        = tournament selection with elitism
  • Variation        = uniform crossover + adaptive Gaussian mutation
  • Memory           = each individual runs with episode-local memory (pure EC)
                       optional shared map mode for hybrid EC+memory approach

Usage:
    python evolutionary_maze_agent.py --mode train
    python evolutionary_maze_agent.py --mode test  --weights best_weights.npy
    python evolutionary_maze_agent.py --mode plot   --history training_history.pkl
"""

from __future__ import annotations
import numpy as np
import random
import copy
import pickle
import argparse
import time
from collections import defaultdict
from typing import List, Tuple, Dict, Optional

# ─────────────────────────────────────────────────────────────────────────────
# Stub imports — replace with actual environment when running
# ─────────────────────────────────────────────────────────────────────────────
try:
    from environment import Action, TurnResult, MazeEnvironment
except ImportError:
    # ── Minimal stubs so the file parses and can be inspected offline ──────────
    from enum import Enum

    class Action(Enum):
        MOVE_UP    = 0
        MOVE_DOWN  = 1
        MOVE_LEFT  = 2
        MOVE_RIGHT = 3
        WAIT       = 4

    class TurnResult:
        def __init__(self):
            self.wall_hits: int              = 0
            self.current_position            = (0, 0)
            self.is_dead: bool               = False
            self.is_confused: bool           = False
            self.is_goal_reached: bool       = False
            self.teleported: bool            = False
            self.actions_executed: int       = 0

    class MazeEnvironment:
        def __init__(self, maze_id: str): pass
        def reset(self): return (0, 31)
        def step(self, actions): return TurnResult()
        def get_episode_stats(self): return {}

# ─────────────────────────────────────────────────────────────────────────────
# Constants — update these to match your confirmed pixel analysis
# ─────────────────────────────────────────────────────────────────────────────
GRID_SIZE   = 64
START_CELL  = (0,  31)
GOAL_CELL   = (63, 32)

# Direction vectors indexed to match Action enum (UP, DOWN, LEFT, RIGHT)
DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
ACTION_MAP  = [Action.MOVE_UP, Action.MOVE_DOWN, Action.MOVE_LEFT, Action.MOVE_RIGHT]
INVERT_MAP  = {
    Action.MOVE_UP:    Action.MOVE_DOWN,
    Action.MOVE_DOWN:  Action.MOVE_UP,
    Action.MOVE_LEFT:  Action.MOVE_RIGHT,
    Action.MOVE_RIGHT: Action.MOVE_LEFT,
}

# ─────────────────────────────────────────────────────────────────────────────
# 1. Neural Network Controller
# ─────────────────────────────────────────────────────────────────────────────
class NeuralController:
    """
    Lightweight feedforward MLP.
    Default architecture: 22 → 64 → 32 → 4  (ReLU hidden, linear output)

    The 22-dimensional state vector is described in StateEncoder below.
    Output = logits over [UP, DOWN, LEFT, RIGHT].
    """

    DEFAULT_LAYERS = [22, 64, 32, 4]

    def __init__(self, layer_sizes: List[int] = None):
        self.layer_sizes = layer_sizes or self.DEFAULT_LAYERS
        self.weights: List[np.ndarray] = []
        self.biases:  List[np.ndarray] = []
        self._init_weights()

    def _init_weights(self):
        self.weights.clear(); self.biases.clear()
        for i in range(len(self.layer_sizes) - 1):
            fan_in  = self.layer_sizes[i]
            fan_out = self.layer_sizes[i + 1]
            # He initialisation (good for ReLU networks)
            w = np.random.randn(fan_in, fan_out) * np.sqrt(2.0 / fan_in)
            b = np.zeros(fan_out)
            self.weights.append(w)
            self.biases.append(b)

    # ── Forward pass ──────────────────────────────────────────────────────────
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Return softmax probabilities over 4 actions."""
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            x = x @ w + b
            if i < len(self.weights) - 1:
                x = np.maximum(0.0, x)          # ReLU
        x = x - x.max()                         # numerical stability
        e = np.exp(x)
        return e / e.sum()                       # softmax

    # ── Weight serialisation ──────────────────────────────────────────────────
    def get_flat_weights(self) -> np.ndarray:
        parts = [w.ravel() for w in self.weights] + [b.ravel() for b in self.biases]
        return np.concatenate(parts)

    def set_flat_weights(self, flat: np.ndarray):
        idx = 0
        for i in range(len(self.weights)):
            sz = self.weights[i].size
            self.weights[i] = flat[idx:idx + sz].reshape(self.weights[i].shape)
            idx += sz
        for i in range(len(self.biases)):
            sz = self.biases[i].size
            self.biases[i] = flat[idx:idx + sz].copy()
            idx += sz

    @property
    def num_params(self) -> int:
        return (sum(w.size for w in self.weights) +
                sum(b.size for b in self.biases))

    def clone(self) -> "NeuralController":
        c = NeuralController(self.layer_sizes)
        c.set_flat_weights(self.get_flat_weights().copy())
        return c

    def save(self, path: str):
        np.save(path, self.get_flat_weights())
        print(f"[NeuralController] Saved {self.num_params} parameters → {path}")

    def load(self, path: str):
        flat = np.load(path)
        self.set_flat_weights(flat)
        print(f"[NeuralController] Loaded {self.num_params} parameters ← {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Agent Memory  (per-episode, resets each episode)
# ─────────────────────────────────────────────────────────────────────────────
class AgentMemory:
    """
    Stores everything an agent discovers within (and optionally across) episodes.
    Set persist=True to carry knowledge across episodes (hybrid EC+memory mode).
    """

    def __init__(self, persist: bool = False):
        self.persist = persist
        self.reset_episode()
        if persist:
            self._shared_walls:    set  = set()   # (r,c,dr,dc) confirmed walls
            self._shared_pits:     set  = set()   # (r,c) confirmed pits
            self._shared_visits:   Dict = defaultdict(int)
            self._shared_teleports: Dict = {}     # (r,c) → (r2,c2)

    # ── episode reset ─────────────────────────────────────────────────────────
    def reset_episode(self):
        if not self.persist:
            self.known_walls:    set  = set()
            self.known_pits:     set  = set()
            self.visit_count:    Dict = defaultdict(int)
            self.known_teleports: Dict = {}
        else:
            self.known_walls     = self._shared_walls
            self.known_pits      = self._shared_pits
            self.visit_count     = self._shared_visits
            self.known_teleports = self._shared_teleports

        self.path:        List[Tuple] = []
        self.is_confused: bool        = False
        self.confused_turns_left: int = 0

    # ── update from TurnResult ────────────────────────────────────────────────
    def update(self, prev_pos: Tuple, action: Action, result: TurnResult,
               intended_action: Action):
        """
        Parse a TurnResult and update the internal map.

        prev_pos        – position before the action
        action          – the Action actually sent to the environment
        result          – TurnResult returned by env.step()
        intended_action – pre-confusion action (same as action unless confused)
        """
        new_pos = result.current_position
        self.path.append(new_pos)
        self.visit_count[new_pos] += 1

        # ── confusion tracking ─────────────────────────────────────────────
        if result.is_confused:
            self.is_confused         = True
            self.confused_turns_left = 2     # current + next turn
        if self.confused_turns_left > 0:
            self.confused_turns_left -= 1
            if self.confused_turns_left == 0:
                self.is_confused = False

        # ── wall discovery ─────────────────────────────────────────────────
        # If position didn't change despite a move action, we hit a wall.
        if (new_pos == prev_pos and
                action not in (Action.WAIT,) and
                not result.is_dead):
            dr, dc = DIRECTIONS[ACTION_MAP.index(intended_action)]
            self.known_walls.add((prev_pos[0], prev_pos[1], dr, dc))

        # ── death pit discovery ────────────────────────────────────────────
        if result.is_dead:
            # The pit is at the cell we tried to move into (or teleport dest)
            if new_pos != prev_pos:
                self.known_pits.add(new_pos)

        # ── teleport discovery ─────────────────────────────────────────────
        if result.teleported and new_pos != prev_pos:
            self.known_teleports[prev_pos] = new_pos


# ─────────────────────────────────────────────────────────────────────────────
# 3. State Encoder
# ─────────────────────────────────────────────────────────────────────────────
class StateEncoder:
    """
    Converts (current position, memory) → 22-dim float32 feature vector.

    Dimensions
    ----------
    [0]  row  / (GRID_SIZE-1)              – normalised row
    [1]  col  / (GRID_SIZE-1)              – normalised column
    [2]  Δrow / (GRID_SIZE-1)              – row delta to goal (signed)
    [3]  Δcol / (GRID_SIZE-1)              – col delta to goal (signed)
    [4]  Manhattan dist / (2*(GRID_SIZE-1)) – distance to goal
    [5–8]  is_wall_known[N,S,W,E]          – known wall flags
    [9–12] is_pit_known[N,S,W,E]           – known pit flags
    [13–16] visit_count[N,S,W,E] / 10      – neighbour visit density (capped)
    [17]   visit_count_self / 10            – own visit count (capped)
    [18]   is_confused                      – currently confused?
    [19]   cells_visited_ratio              – exploration progress
    [20]   is_at_start                      – back at spawn?
    [21]   is_teleport_known               – current cell is a known teleport src
    """

    DIM = 22

    def encode(self, pos: Tuple[int, int], mem: AgentMemory) -> np.ndarray:
        r, c         = pos
        gr, gc       = GOAL_CELL
        g_norm       = 2 * (GRID_SIZE - 1)

        # Basic position / goal features
        f = [
            r / (GRID_SIZE - 1),
            c / (GRID_SIZE - 1),
            (gr - r) / (GRID_SIZE - 1),
            (gc - c) / (GRID_SIZE - 1),
            (abs(gr - r) + abs(gc - c)) / g_norm,
        ]

        # Per-direction features
        for dr, dc in DIRECTIONS:       # N, S, W, E
            nr, nc = r + dr, c + dc
            in_bounds = (0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE)

            # Wall?  Out-of-bounds counts as wall.
            wall = (not in_bounds) or ((r, c, dr, dc) in mem.known_walls)
            pit  = in_bounds and (nr, nc) in mem.known_pits
            vis  = min(mem.visit_count[(nr, nc)] / 10.0, 1.0) if in_bounds else 0.0

            f += [float(wall), float(pit), vis]

        # Self & global features
        f += [
            min(mem.visit_count[pos] / 10.0, 1.0),
            float(mem.is_confused),
            min(len(mem.path) / (GRID_SIZE * GRID_SIZE), 1.0),
            float(pos == START_CELL),
            float(pos in mem.known_teleports),
        ]

        return np.array(f, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Evolutionary Agent  (implements the spec's Agent interface)
# ─────────────────────────────────────────────────────────────────────────────
class EvolutionaryAgent:
    """
    Agent driven by a NeuralController evolved via the genetic algorithm.

    Submits exactly 1 action per turn (safe for unknown mazes).
    Optionally uses persistent memory across episodes (persist_memory=True).
    """

    def __init__(self, controller: NeuralController,
                 epsilon: float = 0.0,
                 persist_memory: bool = False):
        self.controller      = controller
        self.epsilon         = epsilon          # exploration noise during training
        self.encoder         = StateEncoder()
        self.memory          = AgentMemory(persist=persist_memory)
        self.current_pos     = START_CELL
        self.prev_pos        = None
        self.last_action     = None
        self.last_intended   = None

    # ── Agent interface (spec §6.1) ──────────────────────────────────────────
    def plan_turn(self, last_result: Optional[TurnResult]) -> List[Action]:
        if last_result is not None:
            # Update position and map first
            if last_result.is_dead:
                self.current_pos = START_CELL
            else:
                self.current_pos = last_result.current_position

            if self.last_action is not None:
                self.memory.update(
                    self.prev_pos, self.last_action, last_result, self.last_intended
                )

        # Encode state
        state  = self.encoder.encode(self.current_pos, self.memory)
        probs  = self.controller.forward(state)

        # Stochastic during training, greedy during evaluation
        if self.epsilon > 0 and random.random() < self.epsilon:
            action_idx = random.randrange(4)
        else:
            action_idx = int(np.argmax(probs))

        intended = ACTION_MAP[action_idx]

        # Apply confusion inversion
        actual = INVERT_MAP[intended] if self.memory.is_confused else intended

        self.prev_pos      = self.current_pos
        self.last_action   = actual
        self.last_intended = intended

        return [actual]   # 1 action per turn; safe for any maze

    def reset_episode(self):
        """Called at the start of each new episode (spec §6.1)."""
        self.memory.reset_episode()
        self.current_pos   = START_CELL
        self.prev_pos      = None
        self.last_action   = None
        self.last_intended = None


# ─────────────────────────────────────────────────────────────────────────────
# 5. Fitness Evaluation
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_fitness(controller: NeuralController,
                     env: MazeEnvironment,
                     episodes: int = 2,
                     max_turns: int = 2000,
                     epsilon: float = 0.05,
                     persist: bool = False,
                     verbose: bool = False) -> float:
    """
    Run `episodes` episodes with a given controller and return mean fitness.

    Fitness components
    ------------------
    +50 000  if goal reached (once only, first success)
    – 10 * turns_taken                      speed reward: fewer turns = better
    +   2 * len(unique_cells_visited)       exploration bonus
    –  300 * deaths                         safety penalty
    + 200 * (126 – final_manhattan_dist)    proximity reward  (126 = max dist)

    The weights are intentionally large so that success dominates exploration.
    """
    agent      = EvolutionaryAgent(controller, epsilon=epsilon, persist_memory=persist)
    total_fit  = 0.0
    success_ep = 0

    for ep in range(episodes):
        agent.reset_episode()
        env.reset()

        last_result  = None
        turns        = 0
        deaths       = 0
        goal_reached = False
        unique_cells : set = set()

        while turns < max_turns:
            actions     = agent.plan_turn(last_result)
            last_result = env.step(actions)
            turns      += 1

            pos = last_result.current_position
            unique_cells.add(pos)

            if last_result.is_dead:
                deaths += 1
                agent.current_pos = START_CELL

            if last_result.is_goal_reached:
                goal_reached = True
                success_ep  += 1
                break

        # ── compute fitness ──────────────────────────────────────────────────
        r, c = agent.current_pos
        gr, gc = GOAL_CELL
        dist   = abs(gr - r) + abs(gc - c)

        ep_fit = 0.0
        ep_fit += 200 * (126 - dist)           # proximity (always)
        ep_fit += 2 * len(unique_cells)         # exploration
        ep_fit -= 300 * deaths                  # safety
        ep_fit -= 10 * turns                    # speed (penalise slow)
        if goal_reached:
            ep_fit += 50_000                    # BIG success bonus

        total_fit += ep_fit

        if verbose:
            tag = "✓" if goal_reached else "✗"
            print(f"    {tag} ep{ep+1}: turns={turns:4d}  dist={dist:3d}"
                  f"  deaths={deaths:2d}  explored={len(unique_cells):3d}"
                  f"  fit={ep_fit:+.0f}")

    return total_fit / episodes


# ─────────────────────────────────────────────────────────────────────────────
# 6. Genetic Algorithm
# ─────────────────────────────────────────────────────────────────────────────
class GeneticAlgorithm:
    """
    Standard (μ, λ)-style GA with elitism.

    Parameters
    ----------
    pop_size       : number of individuals
    layer_sizes    : NN architecture (default [22, 64, 32, 4])
    elite_frac     : fraction of top individuals kept unchanged each generation
    tournament_k   : tournament size for parent selection
    crossover_prob : probability of applying crossover vs cloning parent1
    init_mut_sigma : initial Gaussian mutation standard deviation
    mut_decay      : multiplicative decay of sigma per generation
    min_mut_sigma  : minimum sigma (prevents premature convergence)
    """

    def __init__(self,
                 pop_size:      int   = 60,
                 layer_sizes:   List  = None,
                 elite_frac:    float = 0.10,
                 tournament_k:  int   = 4,
                 crossover_prob: float = 0.70,
                 init_mut_sigma: float = 0.15,
                 mut_decay:      float = 0.97,
                 min_mut_sigma:  float = 0.02):

        self.pop_size       = pop_size
        self.layer_sizes    = layer_sizes or NeuralController.DEFAULT_LAYERS
        self.elite_frac     = elite_frac
        self.tournament_k   = tournament_k
        self.crossover_prob = crossover_prob
        self.mut_sigma      = init_mut_sigma
        self.mut_decay      = mut_decay
        self.min_mut_sigma  = min_mut_sigma

        # Initialise random population
        self.population: List[NeuralController] = [
            NeuralController(self.layer_sizes) for _ in range(pop_size)
        ]
        self.fitness   = np.full(pop_size, -np.inf)
        self.generation = 0

        # Hall of fame
        self.best_individual: Optional[NeuralController] = None
        self.best_fitness    = -np.inf
        self.history: List[dict] = []

    # ── Selection ────────────────────────────────────────────────────────────
    def _tournament_select(self) -> int:
        """Return index of winner from a random tournament."""
        candidates = random.sample(range(self.pop_size), self.tournament_k)
        return max(candidates, key=lambda i: self.fitness[i])

    # ── Crossover ────────────────────────────────────────────────────────────
    def _uniform_crossover(self, p1: NeuralController,
                            p2: NeuralController) -> NeuralController:
        """Uniform crossover: each gene independently chosen from p1 or p2."""
        w1   = p1.get_flat_weights()
        w2   = p2.get_flat_weights()
        mask = np.random.rand(len(w1)) > 0.5
        child_w = np.where(mask, w1, w2)
        child   = NeuralController(self.layer_sizes)
        child.set_flat_weights(child_w)
        return child

    def _blend_crossover(self, p1: NeuralController,
                          p2: NeuralController,
                          alpha: float = 0.5) -> NeuralController:
        """BLX-α: offspring gene in [min-α·range, max+α·range]."""
        w1   = p1.get_flat_weights()
        w2   = p2.get_flat_weights()
        lo   = np.minimum(w1, w2)
        hi   = np.maximum(w1, w2)
        rng  = hi - lo
        child_w = lo - alpha * rng + np.random.rand(len(w1)) * (1 + 2*alpha) * rng
        child   = NeuralController(self.layer_sizes)
        child.set_flat_weights(child_w)
        return child

    # ── Mutation ─────────────────────────────────────────────────────────────
    def _mutate(self, ctrl: NeuralController,
                mutation_rate: float = 0.08) -> NeuralController:
        """
        Gaussian mutation.
        Each weight is independently perturbed with probability `mutation_rate`.
        σ decays across generations (self.mut_sigma).
        """
        w    = ctrl.get_flat_weights().copy()
        mask = np.random.rand(len(w)) < mutation_rate
        w   += mask * np.random.randn(len(w)) * self.mut_sigma
        ctrl.set_flat_weights(w)
        return ctrl

    # ── One generation ────────────────────────────────────────────────────────
    def step(self, env: MazeEnvironment,
             eval_episodes: int = 2,
             eval_turns:    int = 2000,
             epsilon:       float = 0.05,
             verbose:       bool  = False) -> NeuralController:
        """
        Evaluate the current population, build the next generation, return best.
        """
        print(f"\n{'─'*60}")
        print(f"  Generation {self.generation + 1} / pop_size={self.pop_size}"
              f"  σ={self.mut_sigma:.4f}")
        print(f"{'─'*60}")

        # ── Evaluate fitness ─────────────────────────────────────────────────
        for i, ctrl in enumerate(self.population):
            self.fitness[i] = evaluate_fitness(
                ctrl, env,
                episodes=eval_episodes, max_turns=eval_turns,
                epsilon=epsilon, verbose=False
            )
            if (i + 1) % 10 == 0 or (i + 1) == self.pop_size:
                print(f"  [{i+1:3d}/{self.pop_size}] "
                      f"best_so_far={self.fitness[:i+1].max():+.0f}  "
                      f"mean={self.fitness[:i+1].mean():+.0f}")

        # ── Track best ───────────────────────────────────────────────────────
        best_idx = int(np.argmax(self.fitness))
        if self.fitness[best_idx] > self.best_fitness:
            self.best_fitness    = float(self.fitness[best_idx])
            self.best_individual = self.population[best_idx].clone()
            tag = "★ NEW BEST"
        else:
            tag = ""

        rec = {
            'generation': self.generation,
            'best':   float(self.fitness.max()),
            'mean':   float(self.fitness.mean()),
            'std':    float(self.fitness.std()),
            'worst':  float(self.fitness.min()),
            'sigma':  self.mut_sigma,
        }
        self.history.append(rec)

        print(f"\n  Gen {self.generation+1:3d}  best={rec['best']:+.0f}"
              f"  mean={rec['mean']:+.0f}  std={rec['std']:.0f}  {tag}")

        # ── Build next generation ────────────────────────────────────────────
        sorted_idx   = np.argsort(self.fitness)[::-1]     # best → worst
        elite_k      = max(1, int(self.elite_frac * self.pop_size))
        new_pop      = [self.population[i].clone() for i in sorted_idx[:elite_k]]

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

        # ── Decay mutation sigma ─────────────────────────────────────────────
        self.mut_sigma = max(self.min_mut_sigma, self.mut_sigma * self.mut_decay)
        self.generation += 1

        return self.best_individual

    # ── Full training loop ───────────────────────────────────────────────────
    def train(self, env: MazeEnvironment,
              generations:    int   = 50,
              eval_episodes:  int   = 2,
              eval_turns:     int   = 2000,
              init_epsilon:   float = 0.10,
              epsilon_decay:  float = 0.97,
              save_path:      str   = "best_weights.npy",
              history_path:   str   = "training_history.pkl",
              verbose:        bool  = False) -> NeuralController:
        """
        Run the full EC training loop.

        epsilon decays from init_epsilon (encourage exploration early) to ~0.
        Best individual is saved to disk after every generation.
        """
        epsilon = init_epsilon
        for gen in range(generations):
            self.step(
                env,
                eval_episodes=eval_episodes,
                eval_turns=eval_turns,
                epsilon=epsilon,
                verbose=verbose,
            )
            epsilon = max(0.01, epsilon * epsilon_decay)

            # Save best weights every generation (safe checkpoint)
            if self.best_individual is not None:
                self.best_individual.save(save_path)

            # Persist history
            with open(history_path, "wb") as fh:
                pickle.dump(self.history, fh)

        print(f"\n{'='*60}")
        print(f"  Training complete.  Best fitness: {self.best_fitness:+.0f}")
        print(f"  Weights saved to    {save_path}")
        print(f"  History saved to    {history_path}")
        print(f"{'='*60}")
        return self.best_individual


# ─────────────────────────────────────────────────────────────────────────────
# 7. Final Evaluation Helper
# ─────────────────────────────────────────────────────────────────────────────
def run_evaluation(weights_path: str,
                   maze_id:     str   = "testing",
                   num_episodes: int  = 5,
                   max_turns:    int  = 10_000) -> dict:
    """
    Load evolved weights and evaluate on the specified maze.
    Returns the metrics dict required by the spec's Evaluator.
    """
    env = MazeEnvironment(maze_id)

    ctrl = NeuralController()
    ctrl.load(weights_path)

    agent  = EvolutionaryAgent(ctrl, epsilon=0.0, persist_memory=True)

    successes, turns_list, deaths_list, path_lengths, explored_cells = [], [], [], [], []

    for ep in range(num_episodes):
        agent.reset_episode()
        pos = env.reset()
        agent.current_pos = pos

        last_result = None
        turns       = 0
        deaths      = 0
        goal        = False
        visited     = set([pos])

        while turns < max_turns:
            actions     = agent.plan_turn(last_result)
            last_result = env.step(actions)
            turns      += 1
            visited.add(last_result.current_position)

            if last_result.is_dead:
                deaths += 1
                agent.current_pos = START_CELL

            if last_result.is_goal_reached:
                goal = True
                break

        successes.append(int(goal))
        turns_list.append(turns)
        deaths_list.append(deaths)
        path_lengths.append(len(agent.memory.path))
        explored_cells.append(len(visited))

        status = "SUCCESS ✓" if goal else "TIMEOUT ✗"
        print(f"  Episode {ep+1}: {status}  turns={turns:5d}  deaths={deaths:2d}"
              f"  explored={len(visited):3d}")

    stats = env.get_episode_stats()
    metrics = {
        "success_rate":           sum(successes) / num_episodes,
        "avg_turns":              np.mean([t for s, t in zip(successes, turns_list) if s]),
        "avg_deaths":             float(np.mean(deaths_list)),
        "avg_path_length":        float(np.mean(path_lengths)),
        "exploration_efficiency": float(np.mean(explored_cells)) / (GRID_SIZE**2),
        "env_stats":              stats,
    }

    print(f"\n  ── Final Metrics ──────────────────────────")
    for k, v in metrics.items():
        if k != "env_stats":
            print(f"  {k:<28}: {v:.4f}" if isinstance(v, float) else f"  {k:<28}: {v}")
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# 8. Learning Curve Plotter
# ─────────────────────────────────────────────────────────────────────────────
def plot_learning_curves(history_path: str = "training_history.pkl"):
    """Visualise fitness progression across generations."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    except ImportError:
        print("matplotlib not installed — skipping plot")
        return

    with open(history_path, "rb") as fh:
        history = pickle.load(fh)

    gens  = [h["generation"] + 1 for h in history]
    best  = [h["best"]           for h in history]
    mean  = [h["mean"]           for h in history]
    std   = [h["std"]            for h in history]
    sigma = [h["sigma"]          for h in history]

    fig = plt.figure(figsize=(14, 5))
    gs  = gridspec.GridSpec(1, 2, figure=fig, hspace=0.4)

    # ── Fitness ──────────────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(gens, best,  lw=2, color="#00c896", label="Best fitness")
    ax1.plot(gens, mean,  lw=1.5, color="#4a90d9", label="Mean fitness")
    ax1.fill_between(gens,
                     [m - s for m, s in zip(mean, std)],
                     [m + s for m, s in zip(mean, std)],
                     color="#4a90d9", alpha=0.15, label="±1 std")
    ax1.axhline(0, color="gray", lw=0.8, ls="--")
    ax1.set_xlabel("Generation");  ax1.set_ylabel("Fitness")
    ax1.set_title("Population Fitness over Generations")
    ax1.legend(loc="lower right");  ax1.grid(alpha=0.3)

    # ── Mutation sigma ────────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1])
    ax2.plot(gens, sigma, lw=2, color="#e87040", label="Mutation σ")
    ax2.set_xlabel("Generation");  ax2.set_ylabel("σ  (mutation strength)")
    ax2.set_title("Adaptive Mutation Strength")
    ax2.legend();  ax2.grid(alpha=0.3)

    plt.suptitle("Neuroevolution — Silent Cartographer Maze", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig("learning_curves.png", dpi=150, bbox_inches="tight")
    print("Saved learning_curves.png")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# 9. CLI entry point
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="EC Maze Agent — Silent Cartographer")
    parser.add_argument("--mode",     choices=["train", "test", "plot"], default="train")
    parser.add_argument("--weights",  default="best_weights.npy",
                        help="Path to load/save evolved weights")
    parser.add_argument("--history",  default="training_history.pkl",
                        help="Path to save training history")
    parser.add_argument("--maze",     default="training",
                        help="Maze ID: 'training' or 'testing'")
    # GA hyper-parameters
    parser.add_argument("--pop",      type=int,   default=60,   help="Population size")
    parser.add_argument("--gens",     type=int,   default=50,   help="Generations")
    parser.add_argument("--sigma",    type=float, default=0.15, help="Initial mutation σ")
    parser.add_argument("--decay",    type=float, default=0.97, help="σ decay per generation")
    parser.add_argument("--eps_eval", type=int,   default=2,    help="Episodes per fitness eval")
    parser.add_argument("--turns",    type=int,   default=2000, help="Max turns per eval episode")
    parser.add_argument("--persist",  action="store_true",
                        help="Enable shared memory across episodes (hybrid mode)")
    args = parser.parse_args()

    if args.mode == "train":
        env = MazeEnvironment(args.maze)
        ga  = GeneticAlgorithm(
            pop_size       = args.pop,
            init_mut_sigma = args.sigma,
            mut_decay      = args.decay,
        )
        print(f"\n[GA] {ga.pop_size} individuals × {ga.population[0].num_params} params each")
        print(f"     Architecture : {ga.layer_sizes}")
        print(f"     Generations  : {args.gens}")
        print(f"     Maze         : {args.maze}\n")

        best = ga.train(
            env,
            generations   = args.gens,
            eval_episodes = args.eps_eval,
            eval_turns    = args.turns,
            save_path     = args.weights,
            history_path  = args.history,
        )

    elif args.mode == "test":
        metrics = run_evaluation(
            weights_path  = args.weights,
            maze_id       = args.maze,
            num_episodes  = 5,
            max_turns     = 10_000,
        )

    elif args.mode == "plot":
        plot_learning_curves(args.history)


if __name__ == "__main__":
    main()