#maze_agent.py  —  D* Lite navigator
"""
maze_agent.py
Silent Cartographer: Maze Navigation Project — COSC 4368 AI Spring 2026

Navigation architecture
────────────────────────
  • D* Lite: optimal incremental replanning as walls are discovered.
    Treats unknown cells as free (optimistic), replans cheaply on discovery.
  • Neural network: handles fire timing, teleporter evaluation, confusion.
  • GA evolves NN weights. Fitness rewards goal-reaching speed heavily.
"""

from __future__ import annotations
import heapq
import numpy as np
import random
from collections import defaultdict, deque
from typing import List, Tuple, Dict, Optional, Set

from environment import Action, TurnResult, MazeEnvironment

GRID_SIZE  = 64
START_CELL = (63, 31)
GOAL_CELL  = (0,  31)

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
MAX_FIRE_WAIT  = 3
NAVIGATOR      = "D* Lite"   # shown in viz title

def configure(start: Tuple[int,int], goal: Tuple[int,int], grid_size: int = 64):
    global START_CELL, GOAL_CELL, GRID_SIZE
    START_CELL = start
    GOAL_CELL  = goal
    GRID_SIZE  = grid_size


# ─────────────────────────────────────────────────────────────────────────────
# 1. Neural Network Controller
# ─────────────────────────────────────────────────────────────────────────────
class NeuralController:
    DEFAULT_LAYERS = [43, 64, 32, 5]

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
            [w.ravel() for w in self.weights] + [b.ravel() for b in self.biases])

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

    def save(self, path: str): np.save(path, self.get_flat_weights())
    def load(self, path: str): self.set_flat_weights(np.load(path))


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

        self.path:                      List = []
        self.is_confused:               bool = False
        self.confused_turns_left:       int  = 0
        self.turns_since_new_cell:      int  = 0
        self.turns_since_dist_decrease: int  = 0
        self.last_goal_dist:            int  = 9999
        self.last_wall_hit:             bool = False

    def update(self, prev_pos, action, result: TurnResult, intended_action,
               goal_cell: Tuple[int,int] = None) -> Set[Tuple]:
        """Returns set of newly discovered wall keys for D* Lite incremental replan."""
        new_walls: Set[Tuple] = set()
        new_pos = result.current_position
        self.path.append(new_pos)
        self.last_wall_hit = result.wall_hits > 0

        was_new = self.visit_count[new_pos] == 0
        self.visit_count[new_pos] += 1
        if was_new:
            self.turns_since_new_cell = 0
        else:
            self.turns_since_new_cell = min(self.turns_since_new_cell + 1, 100)

        if goal_cell is not None:
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
                wk = (prev_pos[0], prev_pos[1], dr, dc)
                if wk not in self.known_walls:
                    self.known_walls.add(wk)
                    new_walls.add(wk)
            except ValueError:
                pass

        if result.is_dead and new_pos != prev_pos:
            self.known_pits.add(new_pos)
            pr, pc = new_pos
            for dr, dc in DIRECTIONS:
                fr, fc = pr - dr, pc - dc
                if 0 <= fr < GRID_SIZE and 0 <= fc < GRID_SIZE:
                    wk = (fr, fc, dr, dc)
                    if wk not in self.known_walls:
                        self.known_walls.add(wk)
                        new_walls.add(wk)

        if result.teleported:
            if new_pos == prev_pos:
                try:
                    idx = MOVE_ACTIONS.index(intended_action)
                    dr, dc = DIRECTIONS[idx]
                    wk = (prev_pos[0], prev_pos[1], dr, dc)
                    if wk not in self.known_walls:
                        self.known_walls.add(wk)
                        new_walls.add(wk)
                except ValueError:
                    pass
            else:
                # Correctly store the teleporter SOURCE cell (the cell the agent
                # stepped INTO), not prev_pos (the cell before the move).
                try:
                    idx = MOVE_ACTIONS.index(intended_action)
                    tdr, tdc = DIRECTIONS[idx]
                    tp_src = (prev_pos[0] + tdr, prev_pos[1] + tdc)
                    if 0 <= tp_src[0] < GRID_SIZE and 0 <= tp_src[1] < GRID_SIZE:
                        self.known_teleports[tp_src] = new_pos
                    else:
                        self.known_teleports[prev_pos] = new_pos
                except ValueError:
                    self.known_teleports[prev_pos] = new_pos

        return new_walls


# ─────────────────────────────────────────────────────────────────────────────
# 3. State Encoder  (42 features)
#
# [0-36]  identical local/goal/fire features as before
# [37-41] D* Lite hint — one-hot (UP, DOWN, LEFT, RIGHT, NONE)
#         The NN learns when to follow the hint and when to deviate.
#         This is the genuinely learned behavior the GA selects on.
# ─────────────────────────────────────────────────────────────────────────────
class StateEncoder:
    DIM = 43

    def __init__(self, goal_cell=None, grid_size=None, start_cell=None):
        self.goal_cell  = goal_cell  or GOAL_CELL
        self.grid_size  = grid_size  or GRID_SIZE
        self.start_cell = start_cell or START_CELL

    def encode(self, pos, mem, current_fire=None, fire_rot_idx=0,
               dstar_hint=None, fire_wait_hint=0.0) -> np.ndarray:
        r, c   = pos
        gr, gc = self.goal_cell
        sr, sc = self.start_cell
        gs     = self.grid_size
        g_norm = 2 * (gs - 1)
        cur_dist = abs(gr - r) + abs(gc - c)
        if current_fire is None:
            current_fire = frozenset()

        f = [(gr-r)/(gs-1), (gc-c)/(gs-1), cur_dist/g_norm]

        open_count = 0; open_dirs = []
        for di, (dr, dc) in enumerate(DIRECTIONS):
            nr, nc    = r+dr, c+dc
            in_bounds = 0 <= nr < gs and 0 <= nc < gs
            wall      = (not in_bounds) or ((r,c,dr,dc) in mem.known_walls)
            pit       = in_bounds and (nr,nc) in mem.known_pits
            on_fire   = in_bounds and (nr,nc) in current_fire
            vis       = min(mem.visit_count[(nr,nc)]/10.0, 1.0) if in_bounds else 0.0
            f += [float(wall), float(pit), float(on_fire), vis]
            if not wall and not pit:
                open_count += 1; open_dirs.append(di)

        f.append(fire_rot_idx / 3.0)

        for dr, dc in DIRECTIONS:
            nr, nc  = r+dr, c+dc
            tp_dest = mem.known_teleports.get((nr,nc))
            benefit = 0.0
            if tp_dest is not None:
                dest_dist = abs(gr-tp_dest[0]) + abs(gc-tp_dest[1])
                benefit   = max(0.0, cur_dist - dest_dist) / g_norm
            f.append(benefit)

        f.append(min(mem.turns_since_dist_decrease, 100) / 100.0)
        f.append(min(mem.turns_since_new_cell, 100)       / 100.0)
        f.append(float(mem.last_wall_hit))
        f.append(min(mem.visit_count[pos], 20) / 20.0)
        f.append(open_count / 4.0)
        f.append(float(open_count == 1))
        is_corridor = False
        if open_count == 2:
            d0, d1 = open_dirs
            dr0,dc0 = DIRECTIONS[d0]; dr1,dc1 = DIRECTIONS[d1]
            is_corridor = (dr0+dr1 == 0 and dc0+dc1 == 0)
        f.append(float(is_corridor))
        f.append((abs(r-sr)+abs(c-sc)) / g_norm)
        f.append(min(mem.visit_count[pos]/10.0, 1.0))
        f.append(float(mem.is_confused))
        f.append(min(len(mem.path)/(gs*gs), 1.0))
        f.append(float(pos == self.start_cell))
        f.append(float(pos in mem.known_teleports))

        # D* Lite hint — one-hot: UP DOWN LEFT RIGHT NONE
        hint_vec = [0.0, 0.0, 0.0, 0.0, 0.0]
        if dstar_hint is not None and dstar_hint in MOVE_ACTIONS:
            hint_vec[MOVE_ACTIONS.index(dstar_hint)] = 1.0
        else:
            hint_vec[4] = 1.0   # no path known
        f.extend(hint_vec)

        # Feature 42: exact turns until D* Lite's next cell clears / 3
        # 0.0 = path is clear right now
        # 0.33 = need to wait 1 turn
        # 0.67 = need to wait 2 turns
        # 1.0  = need to wait 3+ turns (permanent fire handled separately)
        f.append(float(min(fire_wait_hint, 1.0)))

        return np.array(f, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# 4. D* Lite
# ─────────────────────────────────────────────────────────────────────────────
class DStarLite:
    INF = float('inf')

    def __init__(self, start, goal, grid_size):
        self.s_goal = goal
        self.gs     = grid_size
        self.known_walls:      set  = set()
        self.known_teleports:  dict = {}
        self._init(start)

    def _init(self, start):
        self.s_start = start
        self.s_last  = start
        self.km      = 0.0
        self.g       = defaultdict(lambda: DStarLite.INF)
        self.rhs     = defaultdict(lambda: DStarLite.INF)
        self._heap:    List = []
        self._in_heap: Dict = {}
        self.rhs[self.s_goal] = 0.0
        self._heap_insert(self.s_goal)
        self.compute_shortest_path()

    def _h(self, u):
        return float(abs(u[0]-self.s_start[0]) + abs(u[1]-self.s_start[1]))

    def _key(self, u):
        g_rhs = min(self.g[u], self.rhs[u])
        return (g_rhs + self._h(u) + self.km, g_rhs)

    def _heap_insert(self, u):
        k = self._key(u)
        heapq.heappush(self._heap, (k, u))
        self._in_heap[u] = k

    def _heap_remove(self, u):
        self._in_heap.pop(u, None)

    def _heap_top(self):
        while self._heap:
            k, u = self._heap[0]
            if self._in_heap.get(u) == k:
                return k, u
            heapq.heappop(self._heap)
        return None

    def _succ(self, u):
        r, c = u
        for action, (dr, dc) in zip(MOVE_ACTIONS, DIRECTIONS):
            nr, nc = r+dr, c+dc
            if 0 <= nr < self.gs and 0 <= nc < self.gs:
                if (r,c,dr,dc) not in self.known_walls:
                    # If neighbor is a known teleporter, route to its destination
                    # (stepping onto it costs 1 action but lands at destination)
                    dest = self.known_teleports.get((nr, nc), (nr, nc))
                    yield action, dest

    def _pred(self, u):
        r, c = u
        for dr, dc in DIRECTIONS:
            pr, pc = r-dr, c-dc
            if 0 <= pr < self.gs and 0 <= pc < self.gs:
                if (pr,pc,dr,dc) not in self.known_walls:
                    yield (pr, pc)
        # If u is a teleporter destination, any neighbor of the source
        # can also precede u (by stepping onto the teleporter)
        for tp_src, tp_dst in self.known_teleports.items():
            if tp_dst == u:
                sr, sc = tp_src
                for tdr, tdc in DIRECTIONS:
                    pr, pc = sr-tdr, sc-tdc
                    if 0 <= pr < self.gs and 0 <= pc < self.gs:
                        if (pr,pc,tdr,tdc) not in self.known_walls:
                            yield (pr, pc)

    def _update_vertex(self, u):
        if u != self.s_goal:
            best = self.INF
            for _, v in self._succ(u):
                val = 1.0 + self.g[v]
                if val < best: best = val
            self.rhs[u] = best
        self._heap_remove(u)
        if self.g[u] != self.rhs[u]:
            self._heap_insert(u)

    def compute_shortest_path(self):
        while True:
            top = self._heap_top()
            if top is None: break
            k_top, u = top
            k_start  = self._key(self.s_start)
            if not (k_top < k_start or
                    self.rhs[self.s_start] != self.g[self.s_start]):
                break
            heapq.heappop(self._heap)
            k_new = self._key(u)
            if k_top < k_new:
                heapq.heappush(self._heap, (k_new, u))
                self._in_heap[u] = k_new
            elif self.g[u] > self.rhs[u]:
                self.g[u] = self.rhs[u]
                del self._in_heap[u]
                for pred in self._pred(u):
                    self._update_vertex(pred)
            else:
                self.g[u] = self.INF
                del self._in_heap[u]
                self._update_vertex(u)
                for pred in self._pred(u):
                    self._update_vertex(pred)

    def notify_new_walls(self, new_walls: Set[Tuple]):
        if not new_walls: return
        self.km    += self._h(self.s_last)
        self.s_last = self.s_start
        affected: set = set()
        for r, c, dr, dc in new_walls:
            affected.add((r, c))
            nr, nc = r+dr, c+dc
            if 0 <= nr < self.gs and 0 <= nc < self.gs:
                affected.add((nr, nc))
        for v in affected:
            self._update_vertex(v)
        self.compute_shortest_path()

    def notify_new_teleports(self, new_teleports: dict):
        """Call when new teleporter source→destination mappings are discovered."""
        if not new_teleports: return
        self.known_teleports.update(new_teleports)
        self.km    += self._h(self.s_last)
        self.s_last = self.s_start
        affected: set = set()
        for tp_src, tp_dst in new_teleports.items():
            # Destination cost changed; neighbors of source now have a new path
            affected.add(tp_dst)
            sr, sc = tp_src
            for tdr, tdc in DIRECTIONS:
                pr, pc = sr-tdr, sc-tdc
                if 0 <= pr < self.gs and 0 <= pc < self.gs:
                    affected.add((pr, pc))
        for v in affected:
            self._update_vertex(v)
        self.compute_shortest_path()

    def next_action(self, current_pos,
                    visit_counts: dict = None,
                    visit_penalty: float = 0.0) -> Optional[Action]:
        """
        visit_counts: optional dict mapping cell→visit_count from AgentMemory.
        visit_penalty: how much each prior visit adds to the cost (0 = pure D* Lite).
        A small penalty (e.g. 0.05) nudges D* Lite toward less-visited cells when
        path costs are tied or very close, encouraging exploration without sacrificing
        optimality on clearly shorter paths.
        """
        self.s_start = current_pos
        self.compute_shortest_path()
        if self.g[self.s_start] == self.INF:
            return None
        best_action = None
        best_cost   = self.INF
        r, c        = self.s_start
        for action, (dr, dc) in zip(MOVE_ACTIONS, DIRECTIONS):
            nr, nc = r+dr, c+dc
            if not (0 <= nr < self.gs and 0 <= nc < self.gs): continue
            if (r,c,dr,dc) in self.known_walls: continue
            dest = self.known_teleports.get((nr, nc), (nr, nc))
            visits = visit_counts.get(dest, 0) if visit_counts else 0
            cost = 1.0 + self.g[dest] + visit_penalty * visits
            if cost < best_cost:
                best_cost   = cost
                best_action = action
        return best_action

    def reset(self, start, known_walls: set, known_teleports: dict = None):
        self.known_walls     = known_walls
        self.known_teleports = dict(known_teleports or {})
        self._init(start)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Evolutionary Agent
# ─────────────────────────────────────────────────────────────────────────────
class EvolutionaryAgent:
    def __init__(self, controller, goal_cell=None, start_cell=None,
                 epsilon=0.0, persist_memory=False):
        self.controller  = controller
        self.goal_cell   = goal_cell  or GOAL_CELL
        self.start_cell  = start_cell or START_CELL
        self.epsilon     = epsilon
        self.encoder     = StateEncoder(self.goal_cell, GRID_SIZE, self.start_cell)
        self.memory      = AgentMemory(persist=persist_memory)
        self.dstar       = DStarLite(self.start_cell, self.goal_cell, GRID_SIZE)

        self.current_pos      = self.start_cell
        self.prev_pos         = None
        self.last_action      = None
        self.last_intended    = None
        self._fire_wait_count = 0
        self.goal_reached     = False

        self.env: Optional[MazeEnvironment] = None
        self._current_fire_cache: frozenset = frozenset()
        self._last_fire_rot_idx:  int       = -1

    @property
    def _current_fire(self) -> frozenset:
        if self.env is None: return frozenset()
        rot_idx = self.env._fire_rot_idx
        if rot_idx != self._last_fire_rot_idx:
            self._current_fire_cache = frozenset(self.env.death_pits)
            self._last_fire_rot_idx  = rot_idx
        return self._current_fire_cache

    def _turns_until_clear(self, r, c) -> int:
        """Exact turns until (r,c) is not on fire. 0=clear now, 4=permanent."""
        if self.env is None: return 0
        cur_idx = self.env._fire_rot_idx
        states  = self.env._fire_rotation_states
        for wait in range(4):
            if (r, c) not in states[(cur_idx + wait) % 4]:
                return wait
        return 4   # all 4 rotation states have fire → permanent

    def _cell_clears_within(self, r, c, max_turns) -> bool:
        if self.env is None: return True
        cur_idx = self.env._fire_rot_idx
        states  = self.env._fire_rotation_states
        for wait in range(max_turns + 1):
            if (r, c) not in states[(cur_idx + wait) % 4]:
                return True
        return False

    def plan_turn(self, last_result) -> List[Action]:
        new_walls: Set[Tuple] = set()
        if last_result is not None:
            if last_result.is_dead:
                self.current_pos = self.start_cell
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

        # Sync newly discovered teleporters into D* Lite so it can route through them
        new_tps = {k: v for k, v in self.memory.known_teleports.items()
                   if k not in self.dstar.known_teleports}
        if new_tps:
            self.dstar.notify_new_teleports(new_tps)

        current_fire = self._current_fire

        fire_rot_idx = self.env._fire_rot_idx if self.env else 0
        r2, c2       = self.current_pos

        # ── Step 1: Mask — only discovered obstacles ──────────────────────────
        # Only walls and pits the agent has physically discovered are masked.
        # Fire cells are NOT pre-masked: the agent must learn to avoid them
        # through experience, as required by the spec (§3.1.1, §4.1.2).
        mask = np.ones(len(ACTION_MAP), dtype=np.float32)
        for ai, (dr, dc) in enumerate(DIRECTIONS):
            nr, nc = r2+dr, c2+dc
            if ((r2, c2, dr, dc) in self.memory.known_walls
                    or (nr, nc) in self.memory.known_pits):
                mask[ai] = 0.0

        # ── Step 2: D* Lite — primary navigator with soft visit-count penalty ──
        # visit_penalty > 0 during high-epsilon turns: nudges D* Lite away from
        # heavily-visited cells, diversifying paths across episodes without
        # overriding clear shortcuts (penalty is tiny vs real path cost differences).
        visit_pen    = 0.05 if self.epsilon > 0.10 else 0.0
        dstar_action = self.dstar.next_action(self.current_pos,
                                              visit_counts=self.memory.visit_count,
                                              visit_penalty=visit_pen)

        # ── Step 3: Is transient fire blocking D* Lite's next cell? ──────────
        fire_blocked = False
        fire_nr = fire_nc = None
        if dstar_action is not None and dstar_action in MOVE_ACTIONS:
            d_idx       = MOVE_ACTIONS.index(dstar_action)
            d_dr, d_dc  = DIRECTIONS[d_idx]
            fire_nr, fire_nc = r2 + d_dr, c2 + d_dc
            if (fire_nr, fire_nc) in current_fire:
                fire_blocked = True

        # ── Step 4: Choose intended action ───────────────────────────────────
        if not fire_blocked:
            self._fire_wait_count = 0
            if dstar_action is not None:
                d_idx = MOVE_ACTIONS.index(dstar_action)
                if mask[d_idx] > 0:
                    # Exploration override: small random chance to deviate from D* Lite.
                    # Lets the agent discover off-path teleporters during training/test.
                    # Rate is epsilon/10 so it rarely interferes with goal-directed nav.
                    if self.epsilon > 0 and random.random() < self.epsilon * 0.10:
                        valid = [ai for ai in range(len(DIRECTIONS)) if mask[ai] > 0]
                        intended = ACTION_MAP[random.choice(valid)] if valid else dstar_action
                    else:
                        intended = dstar_action
                else:
                    valid = [ai for ai in range(len(DIRECTIONS)) if mask[ai] > 0]
                    intended = ACTION_MAP[random.choice(valid)] if valid else Action.WAIT
            else:
                valid = [ai for ai in range(len(DIRECTIONS)) if mask[ai] > 0]
                intended = ACTION_MAP[random.choice(valid)] if valid else Action.WAIT

        else:
            # Fire blocking D* Lite's path — two cases:
            #   Transient fire (clears within MAX_FIRE_WAIT): NN decides wait vs detour.
            #   Permanent fire (never clears): follow D* Lite, accept death.
            #      memory.update() registers the pit; D* Lite replans next turn.
            #      This is how the spec intends hazard discovery to work (SS4.1.2).

            if self._cell_clears_within(fire_nr, fire_nc, MAX_FIRE_WAIT):
                # Transient fire: NN decides timing with fire-wait hint
                wait_needed     = self._turns_until_clear(fire_nr, fire_nc)
                fire_wait_hint  = wait_needed / 3.0
                state    = self.encoder.encode(self.current_pos, self.memory,
                                               current_fire, fire_rot_idx,
                                               dstar_hint=dstar_action,
                                               fire_wait_hint=fire_wait_hint)
                probs    = self.controller.forward(state)
                masked_p = probs * mask
                if masked_p.sum() > 1e-9:
                    probs = masked_p / masked_p.sum()

                if self.epsilon > 0 and random.random() < self.epsilon:
                    # Reference: wait exactly until fire clears, then follow D* Lite
                    if self._fire_wait_count < wait_needed:
                        self._fire_wait_count += 1
                        intended = Action.WAIT
                    else:
                        # Fire is now clear — step through it
                        self._fire_wait_count = 0
                        intended = dstar_action
                else:
                    intended = ACTION_MAP[int(np.argmax(probs))]
                    # Guard: prevent NN from accidentally entering active transient fire
                    # (would register it as a permanent pit — wrong for rotating cells)
                    if intended in MOVE_ACTIONS:
                        ti = MOVE_ACTIONS.index(intended)
                        tdr, tdc = DIRECTIONS[ti]
                        if (r2+tdr, c2+tdc) in current_fire:
                            self._fire_wait_count += 1
                            intended = Action.WAIT
            else:
                # Permanent fire — step into it to learn it's a pit.
                # But ONLY if we don't already know it: re-dying on a known pit
                # just wastes turns and score. If it's already known, D* Lite
                # should have been given wall info; pick any valid direction instead.
                self._fire_wait_count = 0
                if (fire_nr, fire_nc) not in self.memory.known_pits:
                    intended = dstar_action   # first encounter: die and learn
                else:
                    # Already know this pit. D* Lite is confused (no walls seeded).
                    # Force register walls now so it replans correctly next turn.
                    pr, pc = fire_nr, fire_nc
                    extra_walls: Set[Tuple] = set()
                    for edr, edc in DIRECTIONS:
                        fr, fc = pr - edr, pc - edc
                        if 0 <= fr < GRID_SIZE and 0 <= fc < GRID_SIZE:
                            wk = (fr, fc, edr, edc)
                            if wk not in self.memory.known_walls:
                                self.memory.known_walls.add(wk)
                                extra_walls.add(wk)
                    if extra_walls:
                        self.dstar.notify_new_walls(extra_walls)
                    valid = [ai for ai in range(len(DIRECTIONS)) if mask[ai] > 0]
                    intended = ACTION_MAP[random.choice(valid)] if valid else Action.WAIT

        # ── Step 5: Confusion inversion (physics constraint) ──────────────────
        actual = INVERT_MAP[intended] if self.memory.is_confused else intended
        self.prev_pos = self.current_pos
        self.last_action = actual; self.last_intended = intended
        return [actual]

    def reset_episode(self):
        self.memory.reset_episode()
        self.dstar.reset(self.start_cell, self.memory.known_walls,
                         self.memory.known_teleports)
        self.current_pos         = self.start_cell
        self.prev_pos            = None
        self.last_action         = None
        self.last_intended       = None
        self._fire_wait_count    = 0
        self.goal_reached        = False
        self._current_fire_cache = frozenset()
        self._last_fire_rot_idx  = -1


# ─────────────────────────────────────────────────────────────────────────────
# 6. Fitness Functions  (tighter turn penalty: 50 per turn)
# ─────────────────────────────────────────────────────────────────────────────
def _fitness_explore(goal_reached, turns, deaths, wall_hits, unique_cells, gc):
    if not unique_cells:
        return -10_000.0
    max_dist = 2 * (GRID_SIZE - 1)
    fit = 0.0; min_dist = max_dist
    for pos in unique_cells:
        d = abs(gc[0]-pos[0]) + abs(gc[1]-pos[1])
        if d < min_dist:
            min_dist = d
            fit += 500
    fit += 300 * (max_dist - min_dist)
    if min_dist < max_dist * 0.65:
        fit += 1500 * (max_dist / max(min_dist, 1)) - 1500
    fit += 10  * len(unique_cells)
    fit -= 3.0 * max(0, turns - len(unique_cells))
    fit -= 15  * wall_hits
    fit -= 750 * deaths
    fit -= 0.5 * turns
    if goal_reached:
        fit += 120_000
    return fit


def _fitness_optimize(goal_reached, turns, deaths, wall_hits, unique_cells, gc,
                      phase_blend=1.0):
    if goal_reached:
        opt  = 500_000
        opt -= 50    * turns      # tighter: 10k-turn solve scores 0, 5k = 250k
        opt -= 500   * deaths     # reduced: first-encounter pit deaths unavoidable
        opt -= 20    * wall_hits
    else:
        min_dist = (min(abs(gc[0]-r)+abs(gc[1]-c) for r,c in unique_cells)
                    if unique_cells else 2*(GRID_SIZE-1))
        opt  = 300 * ((2*(GRID_SIZE-1)) - min_dist)
        opt += 5   * len(unique_cells)
        opt -= 100 * deaths
        opt -= 3   * wall_hits
    exp = _fitness_explore(goal_reached, turns, deaths, wall_hits, unique_cells, gc)
    return phase_blend * opt + (1.0 - phase_blend) * exp


# ─────────────────────────────────────────────────────────────────────────────
# 7. Fitness Evaluation
# ─────────────────────────────────────────────────────────────────────────────
ACTION_SYMBOLS = {
    Action.MOVE_UP:"↑", Action.MOVE_DOWN:"↓",
    Action.MOVE_LEFT:"←", Action.MOVE_RIGHT:"→", Action.WAIT:"·",
}

def evaluate_fitness(controller, env, goal_cell=None, start_cell=None,
                     episodes=1, max_turns=10_000, epsilon=0.05,
                     persist=False, seed_pits=None, seed_walls=None,
                     verbose=False, step_q=None,
                     step_interval=50, phase=PHASE_EXPLORE):
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

    # Pre-seed with collective knowledge from previous generations/episodes.
    # With persist=True, _shared_pits/_shared_walls carry across all episodes.
    # With persist=False, only the first episode gets the seeded knowledge.
    if seed_pits or seed_walls:
        pits  = set(seed_pits  or set())
        walls = set(seed_walls or set())
        if persist:
            agent.memory._shared_pits  = pits
            agent.memory._shared_walls = walls
        else:
            agent.memory.known_pits  = pits
            agent.memory.known_walls = walls
    total_fit = 0.0

    for ep in range(episodes):
        agent.reset_episode()
        sp = env.reset()
        agent.current_pos = sp

        last_result  = None
        turns = deaths = wall_hits = 0
        goal_reached = False
        unique_cells: set = {sp}
        last_new_cell_turn = 0   # early stopping: track last time we found a new cell

        if verbose:
            print(f"\n  ── Episode {ep+1}/{episodes}  start={sp}  goal={gc}"
                  f"  phase={phase} ──")

        while turns < max_turns:
            actions     = agent.plan_turn(last_result)
            last_result = env.step(actions)
            turns      += 1
            prev_size   = len(unique_cells)
            unique_cells.add(last_result.current_position)
            wall_hits   += last_result.wall_hits

            if len(unique_cells) > prev_size:
                last_new_cell_turn = turns

            # Early stopping: if stuck for 300 turns with no new cells, abort.
            # Solvers and actively exploring agents are never affected.
            # Only kills genuinely stuck individuals, saving ~minutes per gen.
            if not verbose and turns - last_new_cell_turn > 300:
                break

            if turns % 50 == 0:
                _time.sleep(0)

            if step_q is not None and turns % step_interval == 0:
                try:
                    step_q.put_nowait({
                        "type":       "step",
                        "agent_pos":  agent.current_pos,
                        "agent_path": list(agent.memory.path[-5000:]),
                        "turn":       turns,
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
                    tag  = "  "+" ".join(extras) if extras else ""
                    dist = abs(gc[0]-last_result.current_position[0]) + \
                           abs(gc[1]-last_result.current_position[1])
                    print(f"  t{turns:05d} {sym}  pos={last_result.current_position}"
                          f"  dist={dist:3d}{tag}")

            if last_result.is_dead:
                deaths += 1
            if last_result.is_goal_reached:
                goal_reached = True
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

    # Store episode stats on agent so callers can compute performance metrics
    agent.total_wall_hits = wall_hits
    agent.total_turns     = turns
    return total_fit / episodes, agent


def replay_best(weights_path: str, maze_path: str, max_turns: int = 10_000):
    from environment import MazeEnvironment as _Env
    env = _Env(maze_path)
    configure(env.start_cell, env.goal_cell)
    env._phase_gen = 999; env._transition_gens = 20
    ctrl = NeuralController()
    ctrl.load(weights_path)
    fit, agent = evaluate_fitness(ctrl, env,
        goal_cell=GOAL_CELL, start_cell=START_CELL,
        episodes=1, max_turns=max_turns, epsilon=0.0,
        verbose=True, phase=PHASE_OPTIMIZE)
    print(f"\n[replay] fitness={fit:+.0f}  path={len(agent.memory.path)}"
          f"  unique={len(agent.memory.visit_count)}")


# ─────────────────────────────────────────────────────────────────────────────
# 8. Genetic Algorithm
# ─────────────────────────────────────────────────────────────────────────────
class GeneticAlgorithm:
    def __init__(self, pop_size=100, layer_sizes=None,
                 elite_frac=0.15, tournament_k=4, crossover_prob=0.70,
                 init_mut_sigma=0.20, mut_decay=0.995, min_mut_sigma=0.08,
                 phase_switch_k=12, immigrant_frac=0.02, transition_gens=20):
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
        self.population      = [NeuralController(self.layer_sizes) for _ in range(pop_size)]
        self.fitness         = np.full(pop_size, -np.inf)
        self.generation      = 0
        self.best_individual: Optional[NeuralController] = None
        self.best_fitness    = -np.inf
        self.history: List[dict] = []
        self.phase              = PHASE_EXPLORE
        self.cumulative_solvers = 0

    @property
    def is_optimizing(self): return self.phase == PHASE_OPTIMIZE

    def _maybe_switch_phase(self, gen_solvers, env=None):
        if self.phase == PHASE_OPTIMIZE:
            self.phase_gen += 1
            if env is not None:
                env._phase_gen       = self.phase_gen
                env._transition_gens = self.transition_gens
            return
        self.cumulative_solvers += gen_solvers
        if self.cumulative_solvers >= self.phase_switch_k:
            self.phase     = PHASE_OPTIMIZE
            self.phase_gen = 0
            self.mut_sigma = max(self.mut_sigma, 0.08)
            if env is not None:
                env._phase_gen       = 0
                env._transition_gens = self.transition_gens
            print(f"\n  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                  f"  🎯  PHASE SWITCH → OPTIMIZE\n"
                  f"      cumulative solvers: {self.cumulative_solvers}\n"
                  f"      blend over {self.transition_gens} generations\n"
                  f"      σ bumped to {self.mut_sigma:.3f}\n"
                  f"  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

    def _tournament_select(self):
        candidates = random.sample(range(self.pop_size), self.tournament_k)
        return max(candidates, key=lambda i: self.fitness[i])

    def _uniform_crossover(self, p1, p2):
        w1, w2 = p1.get_flat_weights(), p2.get_flat_weights()
        mask   = np.random.rand(len(w1)) > 0.5
        child  = NeuralController(self.layer_sizes)
        child.set_flat_weights(np.where(mask, w1, w2))
        return child

    def _mutate(self, ctrl, mutation_rate=0.10):
        w    = ctrl.get_flat_weights().copy()
        mask = np.random.rand(len(w)) < mutation_rate
        w   += mask * np.random.randn(len(w)) * self.mut_sigma
        ctrl.set_flat_weights(w)
        return ctrl