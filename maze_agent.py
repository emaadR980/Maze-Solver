#maze_agent.py
"""
maze_agent.py
Silent Cartographer: Maze Navigation Project — COSC 4368 AI Spring 2026

Trained on ONE maze, designed to generalize to unseen mazes.

Generalization design
─────────────────────
  • State: local observations + goal direction only — no global map dependency.
  • Progress signals (stall, revisit pressure) guide recovery without
    memorizing specific corridors.
  • Local structure features (dead-end, corridor, open count) teach primitives
    that transfer across layouts.
  • Random immigrants each generation prevent premature convergence.
  • Phase regression: if solvers drop to 0, revert to EXPLORE to rebuild gradient.
"""

from __future__ import annotations
import numpy as np
import random
from collections import defaultdict, deque
from typing import List, Tuple, Dict, Optional

from environment import Action, TurnResult, MazeEnvironment

# ── Runtime-configurable maze constants ───────────────────────────────────────
GRID_SIZE  = 64
START_CELL = (0,  0)
GOAL_KNOWN = False

DIRECTIONS   = [(-1, 0), (1, 0), (0, -1), (0, 1)]
ACTION_MAP   = [Action.MOVE_UP, Action.MOVE_DOWN, Action.MOVE_LEFT,
                Action.MOVE_RIGHT, Action.WAIT]
MOVE_ACTIONS = ACTION_MAP[:4]
INVERT_MAP   = {
    Action.MOVE_UP:    Action.MOVE_DOWN,
    Action.MOVE_DOWN:  Action.MOVE_UP,
    Action.MOVE_LEFT:  Action.MOVE_RIGHT,
    Action.MOVE_RIGHT: Action.MOVE_LEFT,
    Action.WAIT:       Action.WAIT,
}

PHASE_EXPLORE  = "explore"
PHASE_OPTIMIZE = "optimize"
MAX_FIRE_WAIT  = 2  # max turns to wait for rotating fire to clear

def set_goal_known(v: bool = True):
    global GOAL_KNOWN
    GOAL_KNOWN = v

def configure(start: Tuple[int,int], goal: Tuple[int,int], grid_size: int = 64):
    global START_CELL, GOAL_CELL, GRID_SIZE
    START_CELL = start
    GOAL_CELL  = goal
    GRID_SIZE  = grid_size



# ─────────────────────────────────────────────────────────────────────────────
# 1. Neural Network Controller
# ─────────────────────────────────────────────────────────────────────────────
class NeuralController:
    # 39 inputs — see StateEncoder for full breakdown
    # 5 outputs: UP DOWN LEFT RIGHT WAIT
    DEFAULT_LAYERS = [37, 64, 32, 5]

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
        s = e.sum()
        return e / s if s > 0 else np.ones_like(e) / len(e)

    def get_flat_weights(self) -> np.ndarray:
        return np.concatenate(
            [w.ravel() for w in self.weights] + [b.ravel() for b in self.biases]
        )

    def set_flat_weights(self, flat: np.ndarray):
        idx = 0
        for i in range(len(self.weights)):
            sz = self.weights[i].size
            self.weights[i] = flat[idx:idx+sz].reshape(self.weights[i].shape)
            idx += sz
        for i in range(len(self.biases)):
            sz = self.biases[i].size
            self.biases[i] = flat[idx:idx+sz].copy()
            idx += sz

    @property
    def num_params(self) -> int:
        return sum(w.size for w in self.weights) + sum(b.size for b in self.biases)

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

        self.path:                          List = []
        self.is_confused:                   bool = False
        self.confused_turns_left:           int  = 0
        # Progress signals used by StateEncoder
        self.turns_since_new_cell:          int  = 0
        self.turns_since_dist_decrease:     int  = 0
        self.last_goal_dist:                int  = 9999
        self.last_wall_hit:                 bool = False

    def update(self, prev_pos, action, result: TurnResult, intended_action,
               goal_cell: Tuple[int,int] = None):
        new_pos = result.current_position
        self.path.append(new_pos)
        self.last_wall_hit = result.wall_hits > 0

        was_new = self.visit_count[new_pos] == 0
        self.visit_count[new_pos] += 1

        if was_new:
            self.turns_since_new_cell = 0
        else:
            self.turns_since_new_cell = min(self.turns_since_new_cell + 1, 100)

        # Only track goal-distance progress once the goal has been discovered
        if goal_cell is not None and GOAL_KNOWN:
            d = abs(goal_cell[0] - new_pos[0]) + abs(goal_cell[1] - new_pos[1])
            if d < self.last_goal_dist:
                self.last_goal_dist = d
                self.turns_since_dist_decrease = 0
            else:
                self.turns_since_dist_decrease = min(
                    self.turns_since_dist_decrease + 1, 100)

        if result.is_confused:
            self.is_confused         = True
            self.confused_turns_left = 2
        if self.confused_turns_left > 0:
            self.confused_turns_left -= 1
            if self.confused_turns_left == 0:
                self.is_confused = False

        if new_pos == prev_pos and action != Action.WAIT and not result.is_dead:
            try:
                idx = MOVE_ACTIONS.index(intended_action)
                dr, dc = DIRECTIONS[idx]
                self.known_walls.add((prev_pos[0], prev_pos[1], dr, dc))
            except ValueError:
                pass

        if result.is_dead and new_pos != prev_pos:
            self.known_pits.add(new_pos)

        if result.teleported:
            if new_pos == prev_pos:
                try:
                    idx = MOVE_ACTIONS.index(intended_action)
                    dr, dc = DIRECTIONS[idx]
                    self.known_walls.add((prev_pos[0], prev_pos[1], dr, dc))
                except ValueError:
                    pass
            else:
                self.known_teleports[prev_pos] = new_pos


# ─────────────────────────────────────────────────────────────────────────────
# 3. State Encoder  (39 features — all local / relative, no global map)
#
# POSITION / GOAL  [0-2]  (absolute row/col removed — overfitting channel)
#   [0]  (goal_row - row) / (gs-1)   ← goal-relative only
#   [1]  (goal_col - col) / (gs-1)
#   [2]  manhattan_dist / (2*(gs-1))
#
# LOCAL NEIGHBOURHOOD  [5-20]  4 dirs × (wall, pit, fire, visit_frac)
#   Generalizes: reads local obstacles, not memorized map positions.
#
# FIRE TIMER  [21]
#
# TELEPORT BENEFIT  [22-25]  4 dirs × distance_saved_frac
#
# PROGRESS / STALL SIGNALS  [26-29]  — key for cross-maze generalization
#   [26]  turns_since_dist_decrease / 100
#   [27]  turns_since_new_cell / 100
#   [28]  last_wall_hit
#   [29]  revisit_intensity = visit_count[pos] / 20
#
# LOCAL STRUCTURE  [30-32]  — teaches layout-independent primitives
#   [30]  open_neighbor_count / 4
#   [31]  is_dead_end  (1 open neighbour)
#   [32]  is_corridor  (2 open, opposite)
#
# MISC  [33-38]
#   [33]  dist_from_start / (2*(gs-1))
#   [34]  visit_count[pos] / 10
#   [35]  is_confused
#   [36]  path_len / (gs²)
#   [37]  pos == start_cell
#   [38]  pos in known_teleports
# ─────────────────────────────────────────────────────────────────────────────
class StateEncoder:
    DIM = 37   # 3 goal + 16 neighbourhood + 1 fire + 4 teleport + 13 misc

    def __init__(self, goal_cell=None, grid_size=None, start_cell=None):
        self.goal_cell  = goal_cell  or GOAL_CELL
        self.grid_size  = grid_size  or GRID_SIZE
        self.start_cell = start_cell or START_CELL

    def encode(self, pos, mem, current_fire=None, fire_rot_idx=0):
        r, c   = pos
        sr, sc = self.start_cell
        gs     = self.grid_size
        g_norm = 2 * (gs - 1)

        if current_fire is None:
            current_fire = frozenset()

        # ── Goal features: populated only once the goal has been discovered ──
        if GOAL_KNOWN:
            gr, gc_g = self.goal_cell
            cur_dist = abs(gr - r) + abs(gc_g - c)
            f = [
                (gr - r)   / (gs - 1),
                (gc_g - c) / (gs - 1),
                cur_dist   / g_norm,
            ]
        else:
            cur_dist = 0
            f = [0.0, 0.0, 0.0]

        # ── Local neighbourhood: 4 dirs × (wall, pit, fire, visit_frac) ──────
        open_count = 0
        open_dirs  = []
        for di, (dr, dc) in enumerate(DIRECTIONS):
            nr, nc    = r + dr, c + dc
            in_bounds = (0 <= nr < gs and 0 <= nc < gs)
            wall      = (not in_bounds) or ((r, c, dr, dc) in mem.known_walls)
            pit       = in_bounds and (nr, nc) in mem.known_pits
            on_fire   = in_bounds and (nr, nc) in current_fire
            vis       = min(mem.visit_count[(nr, nc)] / 10.0, 1.0) if in_bounds else 0.0
            f += [float(wall), float(pit), float(on_fire), vis]
            if not wall and not pit:
                open_count += 1
                open_dirs.append(di)

        f.append(fire_rot_idx / 3.0)

        # ── Teleport benefit: only meaningful once goal is known ──────────────
        for dr, dc in DIRECTIONS:
            nr, nc = r + dr, c + dc
            if GOAL_KNOWN and g_norm > 0:
                tp_dest = mem.known_teleports.get((nr, nc))
                if tp_dest is not None:
                    gr, gc_g  = self.goal_cell
                    dest_dist = abs(gr - tp_dest[0]) + abs(gc_g - tp_dest[1])
                    benefit   = max(0.0, cur_dist - dest_dist) / g_norm
                else:
                    benefit = 0.0
            else:
                benefit = 0.0
            f.append(benefit)

        # ── Progress / stall signals ──────────────────────────────────────────
        f.append(min(mem.turns_since_dist_decrease, 100) / 100.0)
        f.append(min(mem.turns_since_new_cell,      100) / 100.0)
        f.append(float(mem.last_wall_hit))
        f.append(min(mem.visit_count[pos], 20) / 20.0)

        # ── Local structure ───────────────────────────────────────────────────
        f.append(open_count / 4.0)
        f.append(float(open_count == 1))
        is_corridor = False
        if open_count == 2:
            d0, d1   = open_dirs
            dr0, dc0 = DIRECTIONS[d0]; dr1, dc1 = DIRECTIONS[d1]
            is_corridor = (dr0 + dr1 == 0 and dc0 + dc1 == 0)
        f.append(float(is_corridor))

        # ── Misc ──────────────────────────────────────────────────────────────
        f.append((abs(r - sr) + abs(c - sc)) / g_norm)
        f.append(min(mem.visit_count[pos] / 10.0, 1.0))
        f.append(float(mem.is_confused))
        f.append(min(len(mem.path) / (gs * gs), 1.0))
        f.append(float(pos == self.start_cell))
        f.append(float(pos in mem.known_teleports))

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
        self.recent_positions: deque = deque(maxlen=20)
        self._forced_path:    List = []
        self._fire_wait_count: int = 0
        self.goal_reached:    bool = False
        self._bfs_frustration: int = 0   # increments each BFS call that yields no new cell
        self._random_escape_turns: int = 0  # turns of forced random walk remaining
        self.env: Optional[MazeEnvironment] = None
        self._always_fire_cache: Optional[frozenset] = None
        self._current_fire_cache: frozenset = frozenset()
        self._last_fire_rot_idx: int = -1

    @property
    def _current_fire(self) -> frozenset:
        # Cached per rotation index: fire only changes every 5 turns
        if self.env is None:
            return frozenset()
        rot_idx = self.env._fire_rot_idx
        if rot_idx != self._last_fire_rot_idx:
            self._current_fire_cache = frozenset(self.env.death_pits)
            self._last_fire_rot_idx  = rot_idx
        return self._current_fire_cache

    @property
    def _always_fire(self) -> frozenset:
        # Computed once per episode: intersection of all 4 rotation states never changes
        if self._always_fire_cache is None:
            if self.env is None:
                return frozenset()
            states = self.env._fire_rotation_states
            self._always_fire_cache = (frozenset.intersection(*states)
                                       if states else frozenset())
        return self._always_fire_cache

    # def _cell_clears_within(self, r: int, c: int, max_turns: int) -> bool:
    #     """True if (r,c) leaves the fire rotation within max_turns."""
    #     if self.env is None:
    #         return True
    #     cur_idx = self.env._fire_rot_idx
    #     counter = self.env._fire_turn_counter
    #     states  = self.env._fire_rotation_states
    #     for wait in range(max_turns + 1):
    #         state_idx = (cur_idx + (counter + wait) // 5) % 4
    #         if (r, c) not in states[state_idx]:
    #             return True
    #     return False
    def _cell_clears_within(self, r: int, c: int, max_turns: int) -> bool:
        """True if (r,c) is not on fire at some point within the next max_turns turns."""
        if self.env is None:
            return True

        cur_idx = self.env._fire_rot_idx
        states  = self.env._fire_rotation_states

        for wait in range(max_turns + 1):
            state_idx = (cur_idx + wait) % 4
            if (r, c) not in states[state_idx]:
                return True
        return False

    def _bfs_path_to_nearest_unvisited(self, max_depth: int) -> List[Action]:
        """
        Pure exploration BFS before goal is discovered.
        Goal-biased BFS (progressive) once goal is known.
        """
        start       = self.current_pos
        visited_bfs = {start}
        queue       = deque([(start, [])])
        gs          = self.encoder.grid_size

        best_progressive: Optional[Tuple[int, List[Action]]] = None
        best_any:         Optional[Tuple[int, List[Action]]] = None
        least_visited:    Optional[Tuple[int, int, List[Action]]] = None

        if GOAL_KNOWN:
            gr, gc_g = self.encoder.goal_cell
            cur_dist = abs(gr - start[0]) + abs(gc_g - start[1])
        else:
            gr = gc_g = cur_dist = None   # not used

        while queue:
            pos, path = queue.popleft()

            if best_progressive is not None and len(path) >= 40:
                break
            if len(path) >= max_depth:
                continue

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
                vc       = self.memory.visit_count[nxt]

                if GOAL_KNOWN:
                    nxt_dist = abs(gr - nr) + abs(gc_g - nc)
                else:
                    nxt_dist = len(new_path)   # path length as neutral tiebreaker

                if vc == 0:
                    if best_any is None or nxt_dist < best_any[0]:
                        best_any = (nxt_dist, new_path)
                    if GOAL_KNOWN and nxt_dist < cur_dist:
                        if best_progressive is None or nxt_dist < best_progressive[0]:
                            best_progressive = (nxt_dist, new_path)
                else:
                    if least_visited is None or vc < least_visited[0]:
                        least_visited = (vc, nxt_dist, new_path)

                # Teleporter expansion
                tp_dest = self.memory.known_teleports.get(nxt)
                if tp_dest is not None and tp_dest not in visited_bfs:
                    visited_bfs.add(tp_dest)
                    tp_vc = self.memory.visit_count[tp_dest]
                    if GOAL_KNOWN:
                        tp_dist = abs(gr - tp_dest[0]) + abs(gc_g - tp_dest[1])
                    else:
                        tp_dist = len(new_path)
                    if tp_vc == 0:
                        if best_any is None or tp_dist < best_any[0]:
                            best_any = (tp_dist, new_path)
                        if GOAL_KNOWN and tp_dist < cur_dist:
                            if best_progressive is None or tp_dist < best_progressive[0]:
                                best_progressive = (tp_dist, new_path)
                    else:
                        if least_visited is None or tp_vc < least_visited[0]:
                            least_visited = (tp_vc, tp_dist, new_path)
                    if len(new_path) < max_depth:
                        queue.append((tp_dest, new_path))

                queue.append((nxt, new_path))

        if best_progressive:
            return best_progressive[1]
        if best_any:
            return best_any[1]
        if least_visited and len(least_visited[2]) >= 5:
            return least_visited[2]
        if least_visited:
            return least_visited[2]
        return []

    def plan_turn(self, last_result):
        # ── Update from last result ───────────────────────────────────────────
        if last_result is not None:
            if last_result.is_dead:
                self.current_pos = self.start_cell
                self._forced_path.clear()
                self._fire_wait_count = 0
            else:
                self.current_pos = last_result.current_position

            if (last_result.wall_hits > 0 and
                    last_result.current_position == self.prev_pos):
                self.stuck_count += 1
                self._forced_path.clear()
            else:
                self.stuck_count = 0

            if self.last_action is not None:
                self.memory.update(
                    self.prev_pos, self.last_action,
                    last_result, self.last_intended,
                    goal_cell=self.goal_cell,
                )

        # ── Oscillation / stall detection ─────────────────────────────────────
        self.recent_positions.append(self.current_pos)
        if len(self.recent_positions) == 20:
            rows = [p[0] for p in self.recent_positions]
            cols = [p[1] for p in self.recent_positions]
            if max(rows) - min(rows) <= 5 and max(cols) - min(cols) <= 5:
                self.recent_positions.clear()
                self._forced_path.clear()
                self.stuck_count = 99

        # Stall: no new cell visited OR no distance progress for a while
        if (self.memory.turns_since_new_cell > 30 or
                (GOAL_KNOWN and self.memory.turns_since_dist_decrease > 120)):
            if not self._forced_path:
                self.stuck_count = max(self.stuck_count, 3)

        current_fire      = self._current_fire
        always_fire       = self._always_fire
        fire_rot_idx = self.env._fire_rot_idx if self.env else 0
        # ── Discovery vs exploit regime knobs ─────────────────────────────────────
        if GOAL_KNOWN:
            bfs_max_depth         = 60   # shallower search once goal is known
            forced_prefix         = 1    # only commit to 1 BFS-guided step
            escape_turns          = 4    # short random escape
            bfs_frustration_limit = 1    # tolerate fewer stale BFS suggestions
        else:
            bfs_max_depth         = 150  # stronger exploration before goal discovery
            forced_prefix         = 3    # allow longer BFS guidance
            escape_turns          = 10   # longer breakout walk
            bfs_frustration_limit = 3

        # ── Priority 1: Forced BFS path (fire-safe) ───────────────────────────
        if self._forced_path:
            intended = self._forced_path[0]
            if intended != Action.WAIT:
                idx    = MOVE_ACTIONS.index(intended)
                dr, dc = DIRECTIONS[idx]
                nr, nc = self.current_pos[0] + dr, self.current_pos[1] + dc
                if (nr, nc) in current_fire and (nr, nc) not in always_fire:
                    if (self._fire_wait_count < MAX_FIRE_WAIT and
                            self._cell_clears_within(nr, nc, MAX_FIRE_WAIT)):
                        self._fire_wait_count += 1
                        intended = Action.WAIT
                    else:
                        self._forced_path.clear()
                        self._fire_wait_count = 0
                        self.stuck_count = 99
                        intended = Action.WAIT
                else:
                    self._forced_path.pop(0)
                    self._fire_wait_count = 0
            else:
                self._forced_path.pop(0)

            actual = INVERT_MAP[intended] if self.memory.is_confused else intended
            self.prev_pos = self.current_pos
            self.last_action = actual; self.last_intended = intended
            return [actual]

        # ── Priority 2: BFS recovery when stuck ───────────────────────────────
        if self._random_escape_turns > 0:
            # Forced random walk to break out of tight loops BFS can't escape
            self._random_escape_turns -= 1
            r, c = self.current_pos
            open_dirs = [
                a for a, (dr, dc) in zip(MOVE_ACTIONS, DIRECTIONS)
                if (r, c, dr, dc) not in self.memory.known_walls
                and (r+dr, c+dc) not in self.memory.known_pits
                and (r+dr, c+dc) not in always_fire
            ]
            intended = (random.choice(open_dirs) if open_dirs
                        else random.choice(MOVE_ACTIONS))

        elif self.stuck_count > 2:
            self.stuck_count = 0
            self.recent_positions.clear()
            full_path = self._bfs_path_to_nearest_unvisited(bfs_max_depth)

            # Check if BFS target is a well-worn cell (visit count > 15)
            # If so, BFS is just sending us back into the same pocket
            bfs_target_fresh = False
            if full_path:
                r2, c2 = self.current_pos
                for act, (dr, dc) in zip(MOVE_ACTIONS, DIRECTIONS):
                    if act == full_path[0]:
                        dest = (r2+dr, c2+dc)
                        if self.memory.visit_count[dest] < 15:
                            bfs_target_fresh = True
                        break

            if full_path and (bfs_target_fresh or self._bfs_frustration < bfs_frustration_limit):
                self._forced_path = full_path[1:1 + forced_prefix]
                intended = full_path[0]
                if not bfs_target_fresh:
                    self._bfs_frustration += 1
                else:
                    self._bfs_frustration = 0
            else:
                # BFS keeps sending us to visited cells — force random walk for 30 turns
                self._bfs_frustration = 0
                self._random_escape_turns = escape_turns
                r, c = self.current_pos
                open_dirs = [
                    a for a, (dr, dc) in zip(MOVE_ACTIONS, DIRECTIONS)
                    if (r, c, dr, dc) not in self.memory.known_walls
                    and (r+dr, c+dc) not in self.memory.known_pits
                    and (r+dr, c+dc) not in always_fire
                ]
                intended = (random.choice(open_dirs) if open_dirs
                            else random.choice(MOVE_ACTIONS))
        else:
            # ── Priority 3: Neural network policy ────────────────────────────
            state = self.encoder.encode(self.current_pos, self.memory, current_fire, fire_rot_idx)
            probs = self.controller.forward(state)

            # ── Action masking: zero out known walls to avoid wasting wall budget ──
            r2, c2 = self.current_pos
            mask = np.ones(len(ACTION_MAP), dtype=np.float32)
            for ai, (dr, dc) in enumerate(DIRECTIONS):
                if (r2, c2, dr, dc) in self.memory.known_walls:
                    mask[ai] = 0.0
            # Only apply mask if at least one action remains valid
            masked = probs * mask
            if masked.sum() > 1e-9:
                probs = masked / masked.sum()

            if self.epsilon > 0 and random.random() < self.epsilon:
                # Epsilon: pick randomly from non-wall, non-pit actions only
                valid = [ai for ai, (dr, dc) in enumerate(DIRECTIONS)
                         if (r2, c2, dr, dc) not in self.memory.known_walls
                         and (r2+dr, c2+dc) not in self.memory.known_pits]
                if valid:
                    action_idx = random.choice(valid)
                else:
                    action_idx = random.randrange(len(ACTION_MAP))
            else:
                action_idx = int(np.argmax(probs))
            intended = ACTION_MAP[action_idx]

        # ── Priority 4: Teleporter shortcut (≥15% saving) ────────────────────
        if GOAL_KNOWN and intended != Action.WAIT:
            r, c     = self.current_pos
            gr, gc_g = self.goal_cell
            cur_dist = abs(gr - r) + abs(gc_g - c)
            if cur_dist > 0:
                best_tp_action = None
                threshold = cur_dist * 0.30
                for act, (dr, dc) in zip(MOVE_ACTIONS, DIRECTIONS):
                    nr, nc  = r + dr, c + dc
                    tp_dest = self.memory.known_teleports.get((nr, nc))
                    if tp_dest is None: continue
                    saving = cur_dist - (abs(gr-tp_dest[0]) + abs(gc_g-tp_dest[1]))
                    if saving > threshold:
                        if (r,c,dr,dc) not in self.memory.known_walls:
                            if (nr,nc) not in always_fire:
                                threshold      = saving
                                best_tp_action = act
                if best_tp_action is not None:
                    intended = best_tp_action

        # ── Priority 5: Fire-aware WAIT (with side-step fallback) ─────────────
        if intended != Action.WAIT and intended in MOVE_ACTIONS:
            idx    = MOVE_ACTIONS.index(intended)
            dr, dc = DIRECTIONS[idx]
            nr, nc = self.current_pos[0] + dr, self.current_pos[1] + dc
            if (nr, nc) in current_fire and (nr, nc) not in always_fire:
                if (self._fire_wait_count < MAX_FIRE_WAIT and
                        self._cell_clears_within(nr, nc, MAX_FIRE_WAIT)):
                    self._fire_wait_count += 1
                    intended = Action.WAIT
                else:
                    self._fire_wait_count = 0
                    r2, c2 = self.current_pos
                    alts = [
                        a for a, (adr, adc) in zip(MOVE_ACTIONS, DIRECTIONS)
                        if a != intended
                        and (r2,c2,adr,adc) not in self.memory.known_walls
                        and (r2+adr,c2+adc) not in self.memory.known_pits
                        and (r2+adr,c2+adc) not in always_fire
                        and (r2+adr,c2+adc) not in current_fire
                    ]
                    if alts:
                        intended = random.choice(alts)
            else:
                self._fire_wait_count = 0

        actual = INVERT_MAP[intended] if self.memory.is_confused else intended
        self.prev_pos = self.current_pos
        self.last_action = actual; self.last_intended = intended
        return [actual]

    def reset_episode(self):
        self.memory.reset_episode()
        self.current_pos         = self.start_cell
        self.prev_pos            = None
        self.last_action         = None
        self.last_intended       = None
        self.stuck_count         = 0
        self.recent_positions    = deque(maxlen=20)
        self._forced_path        = []
        self._fire_wait_count    = 0
        self.goal_reached        = False
        self._bfs_frustration    = 0
        self._random_escape_turns = 0
        self._always_fire_cache  = None
        self._current_fire_cache = frozenset()
        self._last_fire_rot_idx  = -1


# ─────────────────────────────────────────────────────────────────────────────
# 5. Fitness Functions
# ─────────────────────────────────────────────────────────────────────────────
def _fitness_explore(goal_reached: bool, turns: int, deaths: int,
                     wall_hits: int, unique_cells: set,
                     gc: Tuple[int,int]) -> float:
    if not unique_cells:
        return -10_000.0

    max_dist = 2 * (GRID_SIZE - 1)
    fit      = 0.0
    min_dist = max_dist

    for pos in unique_cells:
        d = abs(gc[0]-pos[0]) + abs(gc[1]-pos[1])
        if d < min_dist:
            min_dist = d
            fit += 500              # progressive approach bonus per new record

    # Linear proximity
    fit += 300 * (max_dist - min_dist)

    # Exponential proximity bonus — breaks the dist~57 local maximum.
    # Gives a massive selective advantage to agents that get genuinely close.
    # At dist=60: ~0.  At dist=40: ~750.  At dist=20: ~6000.  At dist=5: ~96000.
    if min_dist < max_dist * 0.65:
        fit += 1500 * (max_dist / max(min_dist, 1)) - 1500

    # New cell bonus: count cells visited exactly once (genuine exploration)
    # This is approximated here as total unique — but strong signal regardless
    fit += 10  * len(unique_cells)                      # breadth (was 8, raised for exploration)
    fit -= 3.0 * max(0, turns - len(unique_cells))      # loop/revisit penalty — doubled (was 4.0)
    fit -= 15  * wall_hits                              # wall penalty
    fit -= 750 * deaths                                 # death hurts more — stops risky fire rushing
    fit -= 0.5 * turns                                  # mild speed pressure (was 1.0)

    if goal_reached:
        fit += 120_000
    return fit


def _fitness_optimize(goal_reached: bool, turns: int, deaths: int,wall_hits: int, unique_cells: set,
                      gc: Tuple[int,int], phase_blend: float = 1.0) -> float:
    """
    Smooth blend: phase_blend 0→explore, 1→optimize.
    All solvers beat all non-solvers (500k base; c=35 ensures 10k-turn solver
    scores 150k which is safely above any non-solver consolation score).
    """
    if goal_reached:
        opt  = 500_000
        opt -= 80   * turns
        opt -= 5_000 * deaths
        opt -= 80   * wall_hits
    else:
        opt = 0
        if unique_cells:
            min_dist = min(abs(gc[0]-r)+abs(gc[1]-c) for r,c in unique_cells)
        else:
            min_dist = 2*(GRID_SIZE-1)
        opt  = 300 * ((2*(GRID_SIZE-1)) - min_dist)
        opt += 5   * len(unique_cells)
        opt -= 200 * deaths
        opt -= 3   * wall_hits

    exp = _fitness_explore(goal_reached, turns, deaths, wall_hits, unique_cells, gc)
    return phase_blend * opt + (1.0 - phase_blend) * exp


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
    import time as _time
    gc = goal_cell  or GOAL_CELL
    sc = start_cell or START_CELL

    if phase == PHASE_OPTIMIZE:
        pg    = getattr(env, "_phase_gen",       0)
        tg    = getattr(env, "_transition_gens", 20)
        blend = min(1.0, pg / max(tg, 1))
        def fitness_fn(*args): return _fitness_optimize(*args, phase_blend=blend)
    else:
        fitness_fn = _fitness_explore

    agent     = EvolutionaryAgent(controller, gc, sc, epsilon, persist)
    agent.env = env
    total_fit = 0.0

    for ep in range(episodes):
        agent.reset_episode()
        sp = env.reset()
        agent.current_pos = sp

        last_result  = None
        turns = deaths = wall_hits = 0
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
                _time.sleep(0)

            if step_q is not None and turns % step_interval == 0:
                try:
                    step_q.put_nowait({
                        "type":       "step",
                        "agent_pos":  agent.current_pos,
                        "agent_path": list(agent.memory.path[-5000:]),
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
                    dist = abs(gc[0]-last_result.current_position[0])+abs(gc[1]-last_result.current_position[1])
                    print(f"  t{turns:05d} {sym}  pos={last_result.current_position}"
                          f"  dist={dist:3d}{tag}")

            if last_result.is_dead:
                deaths += 1
            if last_result.is_goal_reached:
                goal_reached = True
                if not GOAL_KNOWN:
                    set_goal_known(True)
                    print(f"\n  🎯 GOAL DISCOVERED at {gc}! "
                          f"All future agents will navigate directly.\n")
                if verbose:
                    print(f"  ✓ GOAL in {turns} turns!  deaths={deaths}  walls={wall_hits}")
                break

        agent.goal_reached = goal_reached
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
    from environment import MazeEnvironment as _Env
    env = _Env(maze_path)
    configure(env.start_cell, env.goal_cell)
    env._phase_gen = 999; env._transition_gens = 20
    print(f"[replay] maze={maze_path}  start={START_CELL}  goal={GOAL_CELL}")
    ctrl = NeuralController()
    ctrl.load(weights_path)
    fit, agent = evaluate_fitness(ctrl, env,
        goal_cell=GOAL_CELL, start_cell=START_CELL,
        episodes=1, max_turns=max_turns, epsilon=0.0,
        verbose=True, phase=PHASE_OPTIMIZE)
    print(f"\n[replay] fitness={fit:+.0f}  path={len(agent.memory.path)}"
          f"  unique={len(agent.memory.visit_count)}")


# ─────────────────────────────────────────────────────────────────────────────
# 7. Genetic Algorithm  (random immigrants for diversity)
# ─────────────────────────────────────────────────────────────────────────────
class GeneticAlgorithm:
    """
    Two-phase GA with smooth blend transition and random immigrants.

    Random immigrants: each generation 5% of the new population is replaced
    with freshly randomized controllers, continuously injecting diversity
    and preventing premature convergence on training-maze-specific paths.

    Phase regression: if a generation produces zero solvers during OPTIMIZE,
    temporarily revert to EXPLORE to rebuild the goal gradient.
    """

    def __init__(self,
                 pop_size:        int   = 100,
                 layer_sizes:     List  = None,
                 elite_frac:      float = 0.40,   # was 0.05 — keep top 15% so solvers survive
                 tournament_k:    int   = 4,       # was 3 — stronger selection pressure
                 crossover_prob:  float = 0.70,
                 init_mut_sigma:  float = 0.20,
                 mut_decay:       float = 0.997,  # slow — sigma stays high long enough to evolve
                 min_mut_sigma:   float = 0.08,   # raised — always enough diversity to escape local max
                 phase_switch_k:  int   = 1,
                 immigrant_frac:  float = 0.00,   # was 0.05 — fewer immigrants, less dilution
                 transition_gens: int   = 20):
        self.pop_size        = pop_size
        self.layer_sizes     = layer_sizes or NeuralController.DEFAULT_LAYERS
        self.elite_frac      = elite_frac
        self.tournament_k    = tournament_k
        self.crossover_prob  = crossover_prob
        self.mut_sigma       = init_mut_sigma
        self.mut_decay       = mut_decay
        self.min_mut_sigma   = min_mut_sigma
        self.phase_switch_k  = phase_switch_k
        self.immigrant_frac  = immigrant_frac
        self.transition_gens = transition_gens
        self.phase_gen       = 0

        self.population: List[NeuralController] = [
            NeuralController(self.layer_sizes) for _ in range(pop_size)
        ]
        self.fitness     = np.full(pop_size, -np.inf)
        self.generation  = 0

        self.best_individual: Optional[NeuralController] = None
        self.best_fitness    = -np.inf
        self.history: List[dict] = []

        self.phase              = PHASE_EXPLORE
        self.cumulative_solvers = 0

    @property
    def is_optimizing(self) -> bool:
        return self.phase == PHASE_OPTIMIZE

    def _maybe_switch_phase(self, gen_solvers: int, env=None):
        if self.phase == PHASE_OPTIMIZE:
            # if gen_solvers == 0:
            #     # Regression — temporarily revert for gradient recovery
            #     self.phase = PHASE_EXPLORE
            #     print("  [phase] Zero solvers — reverting to EXPLORE for gradient recovery\n")
            # else:
            self.phase_gen += 1
            if env is not None:
                env._phase_gen       = self.phase_gen
                env._transition_gens = self.transition_gens
            return

        self.cumulative_solvers += gen_solvers
        if self.cumulative_solvers >= self.phase_switch_k:
            self.phase     = PHASE_OPTIMIZE
            self.phase_gen = 0
            self.mut_sigma = min(self.mut_sigma, 0.03)
            self.min_mut_sigma = 0.01
            if env is not None:
                env._phase_gen       = 0
                env._transition_gens = self.transition_gens
            print(
                f"\n  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"  🎯  PHASE SWITCH → OPTIMIZE\n"
                f"      cumulative solvers: {self.cumulative_solvers}\n"
                f"      blend over {self.transition_gens} generations\n"
                f"      σ bumped to {self.mut_sigma:.3f}\n"
                f"  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
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

    def _mutate(self, ctrl, mutation_rate: float = 0.10):
        w    = ctrl.get_flat_weights().copy()
        mask = np.random.rand(len(w)) < mutation_rate
        w   += mask * np.random.randn(len(w)) * self.mut_sigma
        ctrl.set_flat_weights(w)
        return ctrl