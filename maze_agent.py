"""
maze_agent.py
Silent Cartographer: Maze Navigation Project — COSC 4368 AI Spring 2026

Neuroevolution agent.  START_CELL / GOAL_CELL are set at runtime from the
real MazeEnvironment so they always match the image being used.

Two-phase training
──────────────────
  EXPLORE  (default)  — reward broad exploration and approaching the goal.
                         Deaths / wall hits penalised lightly so the agent is
                         not afraid to probe unknown territory.
  OPTIMIZE            — triggered automatically once enough cumulative
                         individuals have solved the maze.  Fitness is almost
                         entirely determined by *whether* the goal was reached
                         and *how cleanly* (turns, deaths, wall hits).
                         Agents that don't reach the goal score near-zero so
                         selection pressure is entirely on efficiency.
"""

from __future__ import annotations
import numpy as np
import random
import pickle
import argparse
from collections import defaultdict
from typing import List, Tuple, Dict, Optional

from environment import Action, TurnResult, MazeEnvironment

# ── Runtime-configurable maze constants (set by live_viz / main) ──────────────
GRID_SIZE  = 64
START_CELL = (0,  0)    # overwritten by configure()
GOAL_CELL  = (63, 63)   # overwritten by configure()

DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
ACTION_MAP  = [Action.MOVE_UP, Action.MOVE_DOWN, Action.MOVE_LEFT, Action.MOVE_RIGHT,
               Action.WAIT]   # WAIT added — 5th output of the neural net
MOVE_ACTIONS = ACTION_MAP[:4] # directional subset for wall/BFS logic
INVERT_MAP  = {
    Action.MOVE_UP:    Action.MOVE_DOWN,
    Action.MOVE_DOWN:  Action.MOVE_UP,
    Action.MOVE_LEFT:  Action.MOVE_RIGHT,
    Action.MOVE_RIGHT: Action.MOVE_LEFT,
    Action.WAIT:       Action.WAIT,     # wait is symmetric under confusion
}

# ── Training phases ───────────────────────────────────────────────────────────
PHASE_EXPLORE  = "explore"
PHASE_OPTIMIZE = "optimize"

def configure(start: Tuple[int,int], goal: Tuple[int,int], grid_size: int = 64):
    """Call once after creating the environment."""
    global START_CELL, GOAL_CELL, GRID_SIZE
    START_CELL = start
    GOAL_CELL  = goal
    GRID_SIZE  = grid_size


# ─────────────────────────────────────────────────────────────────────────────
# 1. Neural Network Controller
# ─────────────────────────────────────────────────────────────────────────────
class NeuralController:
    # 31 inputs: pos(2) + goal(3) + 4dirs×4(16) + fire_timer(1) + 4dirs×tp_benefit(4) + misc(5)
    # 5 outputs: UP DOWN LEFT RIGHT WAIT
    DEFAULT_LAYERS = [31, 64, 32, 5]

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
            w = np.random.randn(fan_in, fan_out) * np.sqrt(2.0 / fan_in)
            b = np.zeros(fan_out)
            self.weights.append(w)
            self.biases.append(b)

    def forward(self, x: np.ndarray) -> np.ndarray:
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            x = x @ w + b
            if i < len(self.weights) - 1:
                x = np.maximum(0.0, x)
        x = x - x.max()
        e = np.exp(x)
        return e / e.sum()

    def get_flat_weights(self) -> np.ndarray:
        return np.concatenate(
            [w.ravel() for w in self.weights] + [b.ravel() for b in self.biases]
        )

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

    def load(self, path: str):
        self.set_flat_weights(np.load(path))


# ─────────────────────────────────────────────────────────────────────────────
# 2. Agent Memory
# ─────────────────────────────────────────────────────────────────────────────
class AgentMemory:
    def __init__(self, persist: bool = False):
        self.persist = persist
        if persist:
            self._shared_walls:     set  = set()
            self._shared_pits:      set  = set()
            self._shared_visits:    Dict = defaultdict(int)
            self._shared_teleports: Dict = {}
        self.reset_episode()

    def reset_episode(self):
        if not self.persist:
            self.known_walls:     set  = set()
            self.known_pits:      set  = set()
            self.visit_count:     Dict = defaultdict(int)
            self.known_teleports: Dict = {}
        else:
            self.known_walls     = self._shared_walls
            self.known_pits      = self._shared_pits
            self.visit_count     = self._shared_visits
            self.known_teleports = self._shared_teleports

        self.path:                List = []
        self.is_confused:         bool = False
        self.confused_turns_left: int  = 0

    def update(self, prev_pos, action, result: TurnResult, intended_action):
        new_pos = result.current_position
        self.path.append(new_pos)
        self.visit_count[new_pos] += 1

        if result.is_confused:
            self.is_confused         = True
            self.confused_turns_left = 2
        if self.confused_turns_left > 0:
            self.confused_turns_left -= 1
            if self.confused_turns_left == 0:
                self.is_confused = False

        if (new_pos == prev_pos and
                action not in (Action.WAIT,) and
                not result.is_dead):
            idx = ACTION_MAP.index(intended_action)
            dr, dc = DIRECTIONS[idx]
            self.known_walls.add((prev_pos[0], prev_pos[1], dr, dc))

        if result.is_dead and new_pos != prev_pos:
            self.known_pits.add(new_pos)

        if result.teleported:
            if new_pos == prev_pos:
                # Teleporter loops back to source (or to an already-visited
                # position with no escape).  Record the direction as a wall so
                # the agent never tries it again — this breaks the infinite loop
                # where the agent hammers the same teleporter forever.
                idx = ACTION_MAP.index(intended_action)
                dr, dc = DIRECTIONS[idx]
                self.known_walls.add((prev_pos[0], prev_pos[1], dr, dc))
            else:
                self.known_teleports[prev_pos] = new_pos


# ─────────────────────────────────────────────────────────────────────────────
# 3. State Encoder  (22 features)
#
#   [0]   row / (gs-1)
#   [1]   col / (gs-1)
#   [2]   (goal_row - row) / (gs-1)       direction to goal
#   [3]   (goal_col - col) / (gs-1)
#   [4]   manhattan_dist / (2*(gs-1))
#   [5-20] 4 dirs × (wall, known_pit, current_fire, visit)
#   [21]  turns_until_fire_rotation / 5.0
#   [22-25] 4 dirs × teleport_benefit
#             = distance_saved / (2*(gs-1)) if next cell is a known teleporter
#               that brings the agent closer to goal, else 0.0
#             Gives the NN a direct signal to prefer goal-shortcutting teleporters.
#   [26]  visit_count[pos] / 10
#   [27]  is_confused
#   [28]  path_len / (gs²)
#   [29]  pos == start_cell
#   [30]  pos in known_teleports  (standing on a teleporter right now)
# ─────────────────────────────────────────────────────────────────────────────
class StateEncoder:
    DIM = 31

    def __init__(self, goal_cell:  Tuple[int,int] = None,
                       grid_size:  int             = None,
                       start_cell: Tuple[int,int]  = None):
        self.goal_cell  = goal_cell  or GOAL_CELL
        self.grid_size  = grid_size  or GRID_SIZE
        self.start_cell = start_cell or START_CELL

    def encode(self, pos: Tuple[int,int], mem: AgentMemory,
               current_fire: frozenset = None,
               fire_turn_counter: int  = 0) -> np.ndarray:
        """
        Encode state to a 31-dimensional feature vector.

        current_fire       — env.death_pits at this moment (frozenset of (r,c))
        fire_turn_counter  — env._fire_turn_counter (0-4; fire rotates at 5)
        """
        r, c   = pos
        gr, gc = self.goal_cell
        gs     = self.grid_size
        g_norm = 2 * (gs - 1)
        cur_dist = abs(gr - r) + abs(gc - c)

        if current_fire is None:
            current_fire = frozenset()

        f = [
            r / (gs - 1),
            c / (gs - 1),
            (gr - r) / (gs - 1),
            (gc - c) / (gs - 1),
            cur_dist / g_norm,
        ]

        # 4 dirs × (wall, pit, fire, visit)
        for dr, dc in DIRECTIONS:
            nr, nc    = r + dr, c + dc
            in_bounds = (0 <= nr < gs and 0 <= nc < gs)
            wall      = (not in_bounds) or ((r, c, dr, dc) in mem.known_walls)
            pit       = in_bounds and (nr, nc) in mem.known_pits
            on_fire   = in_bounds and (nr, nc) in current_fire
            vis       = min(mem.visit_count[(nr, nc)] / 10.0, 1.0) if in_bounds else 0.0
            f += [float(wall), float(pit), float(on_fire), vis]

        # Fire rotation countdown
        f.append((5 - fire_turn_counter) / 5.0)

        # 4 dirs × teleport_benefit
        # Positive when stepping onto a known teleporter would bring us closer to goal.
        # Encoded as fraction of total possible distance saved (so range 0-1).
        for dr, dc in DIRECTIONS:
            nr, nc = r + dr, c + dc
            tp_dest = mem.known_teleports.get((nr, nc))
            if tp_dest is not None:
                dest_dist = abs(gr - tp_dest[0]) + abs(gc - tp_dest[1])
                benefit   = max(0, cur_dist - dest_dist) / g_norm
            else:
                benefit = 0.0
            f.append(benefit)

        f += [
            min(mem.visit_count[pos] / 10.0, 1.0),
            float(mem.is_confused),
            min(len(mem.path) / (gs * gs), 1.0),
            float(pos == self.start_cell),
            float(pos in mem.known_teleports),   # standing on a teleporter
        ]

        return np.array(f, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Evolutionary Agent
# ─────────────────────────────────────────────────────────────────────────────
class EvolutionaryAgent:
    def __init__(self, controller:     NeuralController,
                       goal_cell:      Tuple[int,int] = None,
                       start_cell:     Tuple[int,int] = None,
                       epsilon:        float = 0.0,
                       persist_memory: bool  = False):
        self.controller       = controller
        self.goal_cell        = goal_cell  or GOAL_CELL
        self.start_cell       = start_cell or START_CELL
        self.epsilon          = epsilon
        self.encoder          = StateEncoder(self.goal_cell, GRID_SIZE, self.start_cell)
        self.memory           = AgentMemory(persist=persist_memory)
        self.current_pos      = self.start_cell
        self.prev_pos         = None
        self.last_action      = None
        self.last_intended    = None
        self.stuck_count      = 0
        self.recent_positions: List = []
        self._forced_path: List = []
        self.goal_reached: bool = False
        # Set externally by evaluate_fitness so plan_turn can query fire state
        self.env: Optional[MazeEnvironment] = None

    # ── Fire helpers ──────────────────────────────────────────────────────────
    @property
    def _current_fire(self) -> frozenset:
        if self.env is None:
            return frozenset()
        return frozenset(self.env.death_pits)

    @property
    def _always_fire(self) -> frozenset:
        """Cells that are on fire in ALL 4 rotation states — permanently blocked."""
        if self.env is None:
            return frozenset()
        return frozenset.intersection(*self.env._fire_rotation_states)

    def _turns_until_cell_clear(self, r: int, c: int) -> int:
        """
        How many turns until (r,c) is not on fire.
        Returns 0 if the cell is clear right now.
        Returns 5 if the cell is always on fire (never clears within one cycle).
        """
        if self.env is None:
            return 0
        counter = self.env._fire_turn_counter    # 0-4; fire rotates when it hits 5
        states  = self.env._fire_rotation_states  # list of 4 frozensets
        cur_idx = self.env._fire_rot_idx          # which state we're currently in

        for wait in range(5):
            check_idx = (cur_idx + (1 if (counter + wait) >= 5 else 0) +
                         ((counter + wait) // 5)) % 4
            # Simpler: walk forward state by state
            state_idx = (cur_idx + (wait + counter) // 5) % 4
            if (r, c) not in states[state_idx]:
                return wait + max(0, 5 - counter - wait)
        return 5  # fire here through the whole rotation — don't wait

    def plan_turn(self, last_result):
        if last_result is not None:
            if last_result.is_dead:
                self.current_pos = self.start_cell
                self._forced_path.clear()
            else:
                self.current_pos = last_result.current_position

            if last_result.wall_hits > 0 and last_result.current_position == self.prev_pos:
                self.stuck_count += 1
                self._forced_path.clear()
            else:
                self.stuck_count = 0

            if self.last_action is not None:
                self.memory.update(self.prev_pos, self.last_action,
                                last_result, self.last_intended)

        # Oscillation detection
        self.recent_positions.append(self.current_pos)
        if len(self.recent_positions) > 20:
            self.recent_positions.pop(0)
        if len(self.recent_positions) == 20:
            rows = [p[0] for p in self.recent_positions]
            cols = [p[1] for p in self.recent_positions]
            if max(rows) - min(rows) <= 4 and max(cols) - min(cols) <= 4:
                self.recent_positions.clear()
                self._forced_path.clear()
                self.stuck_count = 99

        # Snapshot fire state for this turn
        current_fire      = self._current_fire
        always_fire       = self._always_fire
        fire_turn_counter = self.env._fire_turn_counter if self.env else 0

        # If we have a forced path queued, follow it (with fire check)
        if self._forced_path:
            intended = self._forced_path[0]
            # Check if the next step in the queued path leads into current fire
            if intended != Action.WAIT:
                idx  = MOVE_ACTIONS.index(intended)
                dr, dc = DIRECTIONS[idx]
                nr, nc = self.current_pos[0] + dr, self.current_pos[1] + dc
                if (nr, nc) in current_fire and (nr, nc) not in always_fire:
                    # Fire is temporary — wait for it to rotate rather than step in
                    intended = Action.WAIT
                else:
                    self._forced_path.pop(0)
            else:
                self._forced_path.pop(0)

            actual = INVERT_MAP[intended] if self.memory.is_confused else intended
            self.prev_pos      = self.current_pos
            self.last_action   = actual
            self.last_intended = intended
            return [actual]

        state = self.encoder.encode(self.current_pos, self.memory,
                                    current_fire, fire_turn_counter)
        probs = self.controller.forward(state)

        if self.stuck_count > 2:
            self.stuck_count = 0
            self.recent_positions.clear()
            full_path = self._bfs_path_to_nearest_unvisited()
            if full_path:
                self._forced_path = full_path[1:]
                intended = full_path[0]
            else:
                r, c = self.current_pos
                open_dirs = [
                    a for a, (dr, dc) in zip(MOVE_ACTIONS, DIRECTIONS)
                    if (r, c, dr, dc) not in self.memory.known_walls
                       and (r + dr, c + dc) not in always_fire
                ]
                intended = random.choice(open_dirs) if open_dirs else random.choice(MOVE_ACTIONS)
        else:
            if self.epsilon > 0 and random.random() < self.epsilon:
                action_idx = random.randrange(len(ACTION_MAP))
            else:
                action_idx = int(np.argmax(probs))
            intended = ACTION_MAP[action_idx]

        # ── Teleporter shortcut override ─────────────────────────────────────
        # If a neighboring cell is a known teleporter that saves meaningful
        # distance to the goal AND the NN didn't already pick that direction,
        # override with the teleporter step.  Threshold: saves at least 10%
        # of current distance — avoids micro-detours.
        if intended != Action.WAIT:
            r, c   = self.current_pos
            gr, gc_goal = self.goal_cell
            cur_dist = abs(gr - r) + abs(gc_goal - c)
            best_tp_action = None
            best_tp_saving = cur_dist * 0.10   # must save at least 10%

            for act, (dr, dc) in zip(MOVE_ACTIONS, DIRECTIONS):
                nr, nc   = r + dr, c + dc
                tp_dest  = self.memory.known_teleports.get((nr, nc))
                if tp_dest is None:
                    continue
                dest_dist = abs(gr - tp_dest[0]) + abs(gc_goal - tp_dest[1])
                saving    = cur_dist - dest_dist
                if saving > best_tp_saving:
                    # Make sure the step to the teleporter is not blocked
                    if (r, c, dr, dc) not in self.memory.known_walls:
                        if (nr, nc) not in self._always_fire:
                            best_tp_saving = saving
                            best_tp_action = act

            if best_tp_action is not None:
                intended = best_tp_action

        # ── Fire-aware WAIT override ──────────────────────────────────────────
        # If the NN picked a directional move that steps into temporary fire,
        # substitute WAIT so the agent holds position until the fire rotates.
        # We skip this for always_fire cells (waiting is pointless) and for
        # the WAIT action itself.
        if intended != Action.WAIT:
            idx = MOVE_ACTIONS.index(intended) if intended in MOVE_ACTIONS else -1
            if idx >= 0:
                dr, dc = DIRECTIONS[idx]
                nr, nc = self.current_pos[0] + dr, self.current_pos[1] + dc
                if (nr, nc) in current_fire and (nr, nc) not in always_fire:
                    intended = Action.WAIT   # hold until fire moves

        actual = INVERT_MAP[intended] if self.memory.is_confused else intended
        self.prev_pos      = self.current_pos
        self.last_action   = actual
        self.last_intended = intended
        return [actual]

    def reset_episode(self):
        self.memory.reset_episode()
        self.current_pos      = self.start_cell
        self.prev_pos         = None
        self.last_action      = None
        self.last_intended    = None
        self.stuck_count      = 0
        self.recent_positions = []
        self._forced_path     = []
        self.goal_reached     = False

    def _bfs_path_to_nearest_unvisited(self) -> List[Action]:
        """
        BFS to nearest unvisited cell, teleporter-aware.

        When the frontier reaches a cell that is a known teleport source, the BFS
        also expands from the teleport destination (at the same path cost as
        stepping onto the teleporter).  This means BFS will find paths that use
        teleporters as shortcuts when they lead to unvisited or closer regions.
        """
        from collections import deque
        start = self.current_pos
        visited_bfs = {start}
        queue = deque([(start, [])])
        gs    = self.encoder.grid_size
        gr, gc = self.encoder.goal_cell

        while queue:
            pos, path = queue.popleft()
            r, c = pos
            for action, (dr, dc) in zip(MOVE_ACTIONS, DIRECTIONS):
                nr, nc = r + dr, c + dc
                if not (0 <= nr < gs and 0 <= nc < gs): continue
                if (r, c, dr, dc) in self.memory.known_walls: continue
                if (nr, nc) in self.memory.known_pits: continue
                nxt = (nr, nc)
                if nxt in visited_bfs: continue
                visited_bfs.add(nxt)
                new_path = path + [action]

                if self.memory.visit_count[nxt] == 0:
                    return new_path

                # If nxt is a known teleporter, also expand BFS from destination.
                # This lets BFS discover paths through teleporters without extra cost.
                tp_dest = self.memory.known_teleports.get(nxt)
                if tp_dest is not None and tp_dest not in visited_bfs:
                    visited_bfs.add(tp_dest)
                    if self.memory.visit_count[tp_dest] == 0:
                        return new_path   # path to teleporter reaches unvisited dest
                    if len(new_path) < 100:
                        queue.append((tp_dest, new_path))

                if len(new_path) < 100:
                    queue.append((nxt, new_path))
        return []


# ─────────────────────────────────────────────────────────────────────────────
# 5. Fitness Functions
#
#   EXPLORE  — prioritise finding new ground and inching toward the goal.
#              Deaths are a mild disincentive so the agent isn't paralysed by
#              caution, but exploration and proximity dominate.
#
#   OPTIMIZE — once the maze is known to be solvable, gate most of the fitness
#              on actually reaching the goal and doing so cleanly.  Agents that
#              don't solve it get only a token proximity score so selection
#              pressure is almost entirely on efficiency: low turns, zero
#              deaths, no wall thrashing.
# ─────────────────────────────────────────────────────────────────────────────
def _fitness_explore(goal_reached: bool, turns: int, deaths: int,
                     wall_hits: int, unique_cells: set,
                     gc: Tuple[int,int]) -> float:
    """
    Phase 1 — exploration-first fitness.

    Each time the agent sets a new personal best distance to the goal it earns
    a large bonus, rewarding progressive penetration of the maze.  Exploration
    breadth (unique cells) is also heavily rewarded.  Deaths and wall hits are
    penalised lightly so the agent explores freely rather than hovering near
    the start.
    """
    fit = 0.0

    # Progressive approach bonuses — reward every new closest-distance record
    min_dist = 2 * (GRID_SIZE - 1)   # worst possible Manhattan distance
    for pos in unique_cells:
        d = abs(gc[0] - pos[0]) + abs(gc[1] - pos[1])
        if d < min_dist:
            min_dist = d
            fit += 500              # bonus per new record

    fit += 300 * ((2 * (GRID_SIZE - 1)) - min_dist)   # final proximity reward
    fit += 10  * len(unique_cells)                      # exploration breadth
    fit -= 50  * deaths                                 # mild death penalty
    fit -= 5   * wall_hits                              # mild wall penalty
    fit -= 1   * turns                                  # tiny speed incentive

    if goal_reached:
        fit += 100_000              # large goal bonus — still the main prize

    return fit


def _fitness_optimize(goal_reached: bool, turns: int, deaths: int, wall_hits: int, unique_cells: set, gc: Tuple[int,int]) -> float:
    """
    Phase 2 — efficiency-first fitness.

    Priority order (explicit):
      1. Fewest moves   — primary signal, largest per-unit weight
      2. Fewest deaths  — each death costs ~30 turns-equivalent
      3. Fewest walls   — each wall costs ~0.25 turns-equivalent

    All solvers beat all non-solvers (500k base gap).
    No exploration bonus — detours cost turns and are directly penalised.

    Scale reference for a typical solved episode:
      300 turns × 100 = 30 000 turns cost
      1 death   × 3000 =  3 000 death cost  (= 30 extra turns)
      50 walls  ×  25  =  1 250 wall cost   (= 12.5 extra turns)

    Non-solver consolation uses the explore formula (capped at 49k) so the
    population keeps a meaningful gradient toward the goal even after the
    phase switch.  The hard cap guarantees every real solver outscores every
    consolation score regardless of proximity.
    """
    if goal_reached:
        fit  = 500_000              # all solvers beat all non-solvers
        #
        # Turn coefficient MUST satisfy: 500k - c×max_turns > consolation_cap (49k)
        # → c < (500k - 49k) / max_turns = 451k / 10k = 45.1
        # Using c=40 guarantees even a 10 000-turn solver scores 100 000 > 49 000.
        #
        fit -= 40    * turns        # priority 1 — minimize moves  (1 turn  = 40 pts)
        fit -= 3_000 * deaths       # priority 2 — each death ≈ 75 turns-equivalent
        fit -= 25    * wall_hits    # priority 3 — each wall  ≈ 0.6 turns-equivalent
    else:
        # Use the explore formula as consolation — it provides a rich gradient
        # (proximity + progressive approach bonuses + mild death/wall penalties)
        # that keeps non-solvers pushing toward the goal rather than staying
        # frozen near the start to avoid death penalties.
        # Hard-capped at 49k so no non-solver can ever beat an actual solver
        # (worst-case solver: 500k - 100×10000 = negative, but typical is >>49k).
        fit = _fitness_explore(False, turns, deaths, wall_hits, unique_cells, gc)
        fit = min(fit, 49_000)

    return fit


# ─────────────────────────────────────────────────────────────────────────────
# 6. Fitness Evaluation
# ─────────────────────────────────────────────────────────────────────────────
ACTION_SYMBOLS = {
    Action.MOVE_UP:    "↑",
    Action.MOVE_DOWN:  "↓",
    Action.MOVE_LEFT:  "←",
    Action.MOVE_RIGHT: "→",
    Action.WAIT:       "·",
}

def evaluate_fitness(controller:    NeuralController,
                     env:           MazeEnvironment,
                     goal_cell:     Tuple[int,int] = None,
                     start_cell:    Tuple[int,int] = None,
                     episodes:      int   = 1,
                     max_turns:     int   = 10_000,
                     epsilon:       float = 0.05,
                     persist:       bool  = False,
                     verbose:       bool  = False,
                     step_q                = None,
                     step_interval: int   = 50,
                     phase:         str   = PHASE_EXPLORE,
                     ) -> Tuple[float, EvolutionaryAgent]:
    """
    Evaluate a controller over `episodes` episodes.

    phase  — PHASE_EXPLORE (default) or PHASE_OPTIMIZE.  Controls which
             fitness formula is used.  Pass the GeneticAlgorithm's current
             phase so fitness pressure automatically adapts mid-run.

    When step_q is provided, streams agent position every step_interval turns
    so the display can animate the agent moving in real-time.
    """
    import time
    gc = goal_cell  or GOAL_CELL
    sc = start_cell or START_CELL

    fitness_fn = _fitness_optimize if phase == PHASE_OPTIMIZE else _fitness_explore

    agent     = EvolutionaryAgent(controller, gc, sc, epsilon, persist)
    agent.env = env   # gives plan_turn access to live fire state
    total_fit = 0.0

    for ep in range(episodes):
        agent.reset_episode()
        sp = env.reset()
        agent.current_pos = sp

        last_result  = None
        turns        = 0
        deaths       = 0
        wall_hits    = 0
        goal_reached = False
        unique_cells: set = {sp}

        if verbose:
            print(f"\n  ── Episode {ep+1}/{episodes}  start={sp}  goal={gc}"
                  f"  phase={phase} ──")

        while turns < max_turns:
            actions     = agent.plan_turn(last_result)
            last_result = env.step(actions)
            turns      += 1
            unique_cells.add(last_result.current_position)
            wall_hits   += last_result.wall_hits

            if turns % 50 == 0:
                time.sleep(0)

            if step_q is not None and turns % step_interval == 0:
                try:
                    step_q.put_nowait({
                        "type":       "step",
                        "agent_pos":  agent.current_pos,
                        "agent_path": list(agent.memory.path[-300:]),
                    })
                except Exception:
                    pass

            if verbose:
                sym    = ACTION_SYMBOLS.get(actions[0], "?")
                is_key = (last_result.wall_hits or last_result.is_dead or
                          last_result.teleported or last_result.is_confused or
                          last_result.is_goal_reached or turns % 50 == 0)
                if is_key:
                    extras = []
                    if last_result.wall_hits:   extras.append(f"WALL×{last_result.wall_hits}")
                    if last_result.is_dead:     extras.append("💀 DEAD→respawn")
                    if last_result.teleported:  extras.append(f"🌀 TELEPORT→{last_result.current_position}")
                    if last_result.is_confused: extras.append("😵 CONFUSED")
                    tag  = "  " + " ".join(extras) if extras else ""
                    dist = abs(gc[0]-last_result.current_position[0]) + abs(gc[1]-last_result.current_position[1])
                    print(f"  t{turns:05d} {sym}  pos={last_result.current_position}"
                          f"  dist={dist:3d}{tag}")

            if last_result.is_dead:
                deaths += 1

            if last_result.is_goal_reached:
                goal_reached = True
                if verbose:
                    print(f"  ✓ GOAL reached in {turns} turns!"
                          f"  deaths={deaths}  walls={wall_hits}")
                break

        agent.goal_reached = goal_reached   # expose for GA phase tracking

        ep_fit = fitness_fn(goal_reached, turns, deaths, wall_hits, unique_cells, gc)

        if verbose:
            stats  = env.get_episode_stats()
            status = "SUCCESS ✓" if goal_reached else "TIMEOUT ✗"
            print(f"  {status}  turns={turns}  deaths={deaths}"
                  f"  walls={wall_hits}  explored={len(unique_cells)}"
                  f"  fitness={ep_fit:+.0f}  [{phase}]")
            print(f"  env_stats: {stats}")

        total_fit += ep_fit

    return total_fit / episodes, agent


def replay_best(weights_path: str, maze_path: str, max_turns: int = 10_000):
    """Load saved weights and run one verbose episode."""
    from environment import MazeEnvironment as _Env
    env = _Env(maze_path)
    configure(env.start_cell, env.goal_cell)
    print(f"[replay] maze={maze_path}  start={START_CELL}  goal={GOAL_CELL}")

    ctrl = NeuralController()
    ctrl.load(weights_path)

    fit, agent = evaluate_fitness(
        ctrl, env,
        goal_cell  = GOAL_CELL,
        start_cell = START_CELL,
        episodes   = 1,
        max_turns  = max_turns,
        epsilon    = 0.0,
        verbose    = True,
        phase      = PHASE_OPTIMIZE,   # replay always judges by efficiency
    )
    print(f"\n[replay] final fitness={fit:+.0f}"
          f"  path_length={len(agent.memory.path)}"
          f"  unique_cells={len(agent.memory.visit_count)}")


# ─────────────────────────────────────────────────────────────────────────────
# 7. Genetic Algorithm
# ─────────────────────────────────────────────────────────────────────────────
class GeneticAlgorithm:
    """
    Evolutionary loop with automatic two-phase training.

    Phase transition
    ────────────────
    The GA starts in PHASE_EXPLORE.  Every generation it counts how many
    individuals reached the goal.  Once the *cumulative* solver count across
    all generations reaches `phase_switch_k`, it switches to PHASE_OPTIMIZE
    and logs a banner.  The switch is permanent for the rest of the run.

    `phase_switch_k` (default 5) can be tuned:
      - Lower (2-3) → switch earlier, risk switching before the population
        has a stable path to goal.
      - Higher (8-10) → longer exploration before optimising, better for
        large / heavily hazarded mazes.
    """

    def __init__(self,
                 pop_size:        int   = 60,
                 layer_sizes:     List  = None,
                 elite_frac:      float = 0.10,
                 tournament_k:    int   = 4,
                 crossover_prob:  float = 0.70,
                 init_mut_sigma:  float = 0.15,
                 mut_decay:       float = 0.97,
                 min_mut_sigma:   float = 0.02,
                 phase_switch_k:  int   = 5):
        self.pop_size       = pop_size
        self.layer_sizes    = layer_sizes or NeuralController.DEFAULT_LAYERS
        self.elite_frac     = elite_frac
        self.tournament_k   = tournament_k
        self.crossover_prob = crossover_prob
        self.mut_sigma      = init_mut_sigma
        self.mut_decay      = mut_decay
        self.min_mut_sigma  = min_mut_sigma
        self.phase_switch_k = phase_switch_k

        self.population: List[NeuralController] = [
            NeuralController(self.layer_sizes) for _ in range(pop_size)
        ]
        self.fitness     = np.full(pop_size, -np.inf)
        self.generation  = 0

        self.best_individual: Optional[NeuralController] = None
        self.best_fitness    = -np.inf
        self.history: List[dict] = []

        # Phase tracking
        self.phase               = PHASE_EXPLORE
        self.cumulative_solvers  = 0   # total goal-reaching individuals so far

    @property
    def is_optimizing(self) -> bool:
        return self.phase == PHASE_OPTIMIZE

    def _maybe_switch_phase(self, gen_solvers: int):
        """Accumulate solver count; flip to OPTIMIZE once threshold is met."""
        if self.phase == PHASE_OPTIMIZE:
            return
        self.cumulative_solvers += gen_solvers
        if self.cumulative_solvers >= self.phase_switch_k:
            self.phase = PHASE_OPTIMIZE
            # Boost sigma so the population can adapt to the new fitness landscape.
            # After many explore generations sigma may have decayed to ~0.05;
            # bumping it back up prevents the population from being frozen when
            # the fitness scale changes completely.
            self.mut_sigma = max(self.mut_sigma, 0.10)
            print(
                f"\n  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"  🎯  PHASE SWITCH → OPTIMIZE\n"
                f"      {self.cumulative_solvers} cumulative solvers reached goal.\n"
                f"      Fitness now rewards efficiency: turns, deaths, wall hits.\n"
                f"      σ reset to {self.mut_sigma:.3f} to aid landscape adaptation.\n"
                f"  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            )

    def _tournament_select(self) -> int:
        candidates = random.sample(range(self.pop_size), self.tournament_k)
        return max(candidates, key=lambda i: self.fitness[i])

    def _uniform_crossover(self, p1, p2):
        w1, w2 = p1.get_flat_weights(), p2.get_flat_weights()
        mask   = np.random.rand(len(w1)) > 0.5
        child  = NeuralController(self.layer_sizes)
        child.set_flat_weights(np.where(mask, w1, w2))
        return child

    def _mutate(self, ctrl, mutation_rate: float = 0.08):
        w    = ctrl.get_flat_weights().copy()
        mask = np.random.rand(len(w)) < mutation_rate
        w   += mask * np.random.randn(len(w)) * self.mut_sigma
        ctrl.set_flat_weights(w)
        return ctrl