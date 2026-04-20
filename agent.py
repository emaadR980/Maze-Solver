from hazardDemo import Action, TurnResult
from collections import defaultdict, deque
import random, pickle
from typing import List, Optional, Tuple

MOVE_ACTIONS = [Action.MOVE_UP, Action.MOVE_DOWN, Action.MOVE_LEFT, Action.MOVE_RIGHT]

DELTAS: dict = {
    Action.MOVE_UP:    (-1,  0),
    Action.MOVE_DOWN:  ( 1,  0),
    Action.MOVE_LEFT:  ( 0, -1),
    Action.MOVE_RIGHT: ( 0,  1),
    Action.WAIT:       ( 0,  0),
}
INVERT: dict = {
    Action.MOVE_UP:    Action.MOVE_DOWN,
    Action.MOVE_DOWN:  Action.MOVE_UP,
    Action.MOVE_LEFT:  Action.MOVE_RIGHT,
    Action.MOVE_RIGHT: Action.MOVE_LEFT,
    Action.WAIT:       Action.WAIT,
}


class MazeAgent:
    def __init__(self):
        self.known: dict = {}
        self.wall_edges: set = set()
        self.open_edges: set = set()
        self.current_pos: Optional[Tuple[int,int]] = None
        self.goal_pos:    Optional[Tuple[int,int]] = None
        self.start_pos:   Optional[Tuple[int,int]] = None
        self.confused_turns_left: int = 0  # turns remaining where agent must compensate
        self.env = None   # set externally to the live MazeEnvironment

        self.q_table = defaultdict(lambda: defaultdict(float))
        self.alpha   = 0.1
        self.gamma   = 0.95
        self.epsilon     = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.975

        self.episode     = 0
        self.visit_count = defaultdict(int)

        self._planned_path: List[Action] = []
        self._prev_pos:     Optional[Tuple[int,int]] = None
        self._last_actions: List[Action] = []  # intended (pre-confusion) actions
        self._was_confused:  bool = False       # whether confusion was active when _last_actions was sent
        self._recent_positions: deque = deque(maxlen=8)
        self._stagnation_turns: int = 0
        self._respawn_wait_turns: int = 0
        self._fire_cooldown_until: dict = {}
        self._fire_death_counts = defaultdict(int)

    def reset_episode(self):
        self.confused_turns_left = 0
        self._was_confused  = False
        self._planned_path  = []
        self._prev_pos      = None
        self._last_actions  = []
        self._recent_positions.clear()
        self._stagnation_turns = 0
        self._respawn_wait_turns = 0
        self._fire_cooldown_until.clear()
        self._fire_death_counts = defaultdict(int)
        self.visit_count    = defaultdict(int)
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.episode += 1

    def plan_turn(self, last_result: Optional[TurnResult]) -> List[Action]:
        if last_result is not None:
            self._process_result(last_result)

        if self.current_pos is None:
            return [Action.WAIT]

        if self._respawn_wait_turns > 0:
            self._respawn_wait_turns -= 1
            self._record_send([Action.WAIT])
            return [Action.WAIT]

        if self._planned_path:
            batch = self._pop_batch()
            if batch:
                return self._apply_confusion(batch)

        path = self._make_plan()
        if not path:
            if self._stagnation_turns >= 20:
                escape = self._escape_action()
                if escape is not None:
                    self._record_send([escape])
                    return self._apply_confusion([escape])
            act = self._ql_action()
            self._record_send([act])
            return self._apply_confusion([act])

        take = self._safe_batch_len(path)
        if take <= 0:
            self._planned_path = []
            act = self._ql_action()
            self._record_send([act])
            return self._apply_confusion([act])
        self._planned_path = path[take:]
        batch = path[:take]
        batch = self._break_oscillation(batch)
        self._record_send(batch)
        return self._apply_confusion(batch)

    def update_q(self, old_state, action: Action, reward: float, new_state):
        old_q    = self.q_table[old_state][action.value]
        max_next = max(self.q_table[new_state].values(), default=0.0)
        self.q_table[old_state][action.value] = (
            old_q + self.alpha * (reward + self.gamma * max_next - old_q)
        )

    def compute_reward(self, result: TurnResult, old_pos: Tuple[int,int]) -> float:
        r = 0.0
        if result.is_goal_reached:
            r += 500.0
        if result.is_dead:
            r -= 100.0
        if result.wall_hits > 0:
            r -= 5.0 * result.wall_hits
        r -= 1.0 * self.visit_count.get(result.current_position, 0)
        if self.goal_pos and not result.is_dead:
            old_d = abs(old_pos[0]-self.goal_pos[0]) + abs(old_pos[1]-self.goal_pos[1])
            new_d = abs(result.current_position[0]-self.goal_pos[0]) + abs(result.current_position[1]-self.goal_pos[1])
            r += (old_d - new_d) * 3.0
        return r

    def state(self) -> tuple:
        return (*self.current_pos, self.confused_turns_left > 0)

    def save(self, path: str = "agent.pkl"):
        data = {
            "q_table":     dict(self.q_table),
            "known":       self.known,
            "wall_edges":  self.wall_edges,
            "open_edges":  self.open_edges,
            "goal_pos":    self.goal_pos,
            "start_pos":   self.start_pos,
            "visit_count": dict(self.visit_count),
            "episode":     self.episode,
            "epsilon":     self.epsilon,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        print(f"Saved  (ep={self.episode}, ε={self.epsilon:.3f}, "
              f"goal={self.goal_pos}, known={len(self.known)} cells, "
              f"wall_edges={len(self.wall_edges)//2})")

    def load(self, path: str = "agent.pkl"):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.q_table     = defaultdict(lambda: defaultdict(float), data.get("q_table", {}))
        raw_known        = data.get("known", {})
        self.known       = {k: v for k, v in raw_known.items() if v != 'wall'}
        self.wall_edges  = data.get("wall_edges", set())
        self.open_edges  = data.get("open_edges", set())
        self.goal_pos    = data.get("goal_pos", None)
        self.start_pos   = data.get("start_pos", None)
        self.visit_count = defaultdict(int, data.get("visit_count", {}))
        self.episode     = data.get("episode", 0)
        self.epsilon     = data.get("epsilon", self.epsilon_min)
        self._planned_path     = []
        self._prev_pos         = None
        self._last_actions     = []
        self._recent_positions = deque(maxlen=8)
        self._stagnation_turns = 0
        self._respawn_wait_turns = 0
        self._fire_cooldown_until = {}
        self._fire_death_counts = defaultdict(int)
        self.confused_turns_left = 0
        self._was_confused = False
        print(f"Loaded (ep={self.episode}, ε={self.epsilon:.3f}, "
              f"goal={self.goal_pos}, known={len(self.known)} cells, "
              f"wall_edges={len(self.wall_edges)//2})")

    def _process_result(self, result: TurnResult):
        if result.is_dead:
            if self.env is None or not getattr(self.env, "rotate_fire_enabled", False):
                self.known[result.current_position] = 'death'
            else:
                self._remember_fire_death(result.current_position)
                if self.known.get(result.current_position) == 'death':
                    self.known.pop(result.current_position)
            self._planned_path = []
            self.confused_turns_left = 0
            self._stagnation_turns = 0
            self._respawn_wait_turns = 5 if self.env is not None and self.env.death_pits else random.randint(1, 4)
            # Respawn at start
            if self.start_pos is not None:
                self.current_pos = self.start_pos
            return

        self.current_pos = result.current_position

        if result.is_goal_reached:
            self.goal_pos = self.current_pos
            self._planned_path = []

        if result.teleported or result.is_confused:
            self._planned_path = []

        if result.is_confused:
            self.confused_turns_left = 1   # environment inverts 1 more full turn (T+1); compensate for that
        elif self.confused_turns_left > 0:
            self.confused_turns_left -= 1

        def _actual(intended: Action) -> Action:
            return intended

        # Label current cell
        if result.is_confused:
            self.known[self.current_pos] = 'confusion'
        elif result.teleported:
            if len(self._last_actions) == 1 and self._prev_pos is not None:
                dr, dc = DELTAS[_actual(self._last_actions[0])]
                src_pad = (self._prev_pos[0]+dr, self._prev_pos[1]+dc)
                self._remember_open_edge(self._prev_pos, src_pad)
                self.known[src_pad] = 'teleport'
            self.known[self.current_pos] = 'teleport'
        else:
            if self.known.get(self.current_pos) not in ('death', 'confusion', 'teleport'):
                self.known[self.current_pos] = 'empty'
            if (len(self._last_actions) == 1
                    and result.wall_hits == 0
                    and self._prev_pos is not None
                    and self.current_pos is not None):
                actual = _actual(self._last_actions[0])
                dr, dc = DELTAS[actual]
                expected = (self._prev_pos[0]+dr, self._prev_pos[1]+dc)
                if actual is not Action.WAIT and self.current_pos == expected:
                    self._remember_open_edge(self._prev_pos, self.current_pos)

        if (len(self._last_actions) == 1
                and result.wall_hits == 1
                and self._prev_pos is not None
                and not result.is_dead):
            actual = _actual(self._last_actions[0])
            dr, dc = DELTAS[actual]
            wall_cell = (self._prev_pos[0]+dr, self._prev_pos[1]+dc)
            self.wall_edges.add((self._prev_pos, wall_cell))
            self.wall_edges.add((wall_cell, self._prev_pos))  # symmetric

        if result.wall_hits > 0:
            self._planned_path = []

        self.visit_count[self.current_pos] += 1
        self._recent_positions.append(self.current_pos)
        if self.visit_count[self.current_pos] > 1:
            self._stagnation_turns += 1
        else:
            self._stagnation_turns = 0

    def _fire_phase_sets(self):
        clusters = [list(c) for c in self.env.fire_clusters]
        pivots = list(getattr(self.env, "fire_cluster_pivots", []))
        if not pivots:
            pivots = [self.env.cluster_orientation_and_pivot(cluster)[1] for cluster in clusters]
        phases = []
        for _ in range(4):
            pits = set()
            for cl in clusters:
                pits.update(cell for cell in cl if self.env.is_cell_in_bounds(*cell))
            phases.append(pits)
            clusters = [
                self.env.rotate_fire_cluster(cluster, pivot)
                for cluster, pivot in zip(clusters, pivots)
            ]
        return phases

    def _fire_safe_len(self, path: List[Action]) -> int:
        if self.env is None or not self.env.fire_clusters:
            return len(path)
        phases = self._fire_phase_sets()
        pos = self.current_pos
        rotates_after_this_turn = (self.env.turn_count + 1) % 5 == 0
        for k, action in enumerate(path):
            if action is Action.WAIT:
                if k > 0:
                    return k
                if rotates_after_this_turn and pos in phases[1]:
                    return 0
                return 1

            dr, dc = DELTAS[action]
            pos = (pos[0] + dr, pos[1] + dc)
            if self.env is not None and pos in self.env.teleport_map:
                pos = self.env.teleport_map[pos]
            if pos in phases[0]:
                return k
            if rotates_after_this_turn and pos in phases[1]:
                return k
        return len(path)

    def _fire_safe_after_turn(self, pos: Tuple[int,int],
                              phase_deaths: List[set],
                              phase: int,
                              turns_in_cycle: int) -> bool:
        if pos in phase_deaths[phase]:
            return False
        if turns_in_cycle == 0:
            return pos not in phase_deaths[(phase + 1) % 4]
        return True

    def _fire_safe_for_turns(self, pos: Tuple[int,int], turns: int) -> bool:
        if self.env is None or not self.env.fire_clusters:
            return True
        phases = self._fire_phase_sets()
        phase = 0
        turns_in_cycle = self.env.turn_count % 5
        if pos in phases[phase]:
            return False

        for _ in range(max(0, turns)):
            turns_in_cycle = (turns_in_cycle + 1) % 5
            if turns_in_cycle == 0:
                phase = (phase + 1) % 4
            if pos in phases[phase]:
                return False
        return True

    def _turns_until_fire_rotation(self) -> int:
        if self.env is None:
            return 1
        turns = self.env.turn_count % 5
        return 5 if turns == 0 else 5 - turns

    def _fire_safe_next_turn(self, pos: Tuple[int,int]) -> bool:
        return self._fire_safe_for_turns(pos, 1)

    def _fire_safe_until_next_rotation(self, pos: Tuple[int,int]) -> bool:
        return self._fire_safe_for_turns(pos, self._turns_until_fire_rotation())

    def _remember_fire_death(self, pos: Tuple[int,int]) -> None:
        if self.env is None:
            return

        cluster_cells = {pos}
        for cluster in getattr(self.env, "fire_clusters", []):
            in_bounds = {
                cell for cell in cluster
                if self.env.is_cell_in_bounds(*cell)
            }
            if pos in in_bounds:
                cluster_cells = in_bounds
                break

        self._fire_death_counts[pos] += 1
        backoff_turns = 20 * min(4, self._fire_death_counts[pos])
        until = self.env.turn_count + backoff_turns
        for cell in cluster_cells:
            self._fire_cooldown_until[cell] = max(
                self._fire_cooldown_until.get(cell, 0),
                until,
            )

    def _cell_on_fire_cooldown(self, cell: Tuple[int,int]) -> bool:
        if self.env is None:
            return False
        until = self._fire_cooldown_until.get(cell)
        if until is None:
            return False
        if until <= self.env.turn_count:
            self._fire_cooldown_until.pop(cell, None)
            return False
        return True

    def _make_plan(self) -> List[Action]:
        pos = self.current_pos
        fire_active = self.env is not None and bool(self.env.death_pits)

        teleport_path = self._plan_to_beneficial_teleport()
        if teleport_path:
            return teleport_path

        if fire_active and self.goal_pos is not None:
            dynamic_path = self._plan_dynamic_to_goal()
            if dynamic_path:
                return dynamic_path

        if self.goal_pos is not None:
            for avoid in ({'death', 'confusion'}, {'death'}):
                path = self._bfs(pos, self.goal_pos, avoid=avoid)
                if path:
                    safe = self._fire_safe_len(path)
                    if safe > 0:
                        return path[:safe]

        if fire_active and self.goal_pos is not None:
            dynamic_path = self._plan_dynamic_to_goal()
            if dynamic_path:
                return dynamic_path

        target = self._frontier_target()
        if target is not None:
            path = self._bfs(pos, target, avoid={'death'})
            if path:
                safe = self._fire_safe_len(path)
                if safe > 0:
                    return path[:safe]

        if self.env is not None and self.goal_pos is not None:
            dynamic_path = self._plan_dynamic_to_goal()
            if dynamic_path:
                return dynamic_path

        return []

    def _plan_to_beneficial_teleport(self) -> List[Action]:
        if self.env is None or self.current_pos is None or self.goal_pos is None:
            return []
        if not self.env.teleport_map:
            return []

        current_to_goal = self._manhattan(self.current_pos, self.goal_pos)
        best: Optional[Tuple[int, List[Action]]] = None

        for src, dest in self.env.teleport_map.items():
            if src == dest:
                continue
            if self.known.get(src) in {'death', 'confusion'}:
                continue
            if self.known.get(dest) in {'death', 'confusion'}:
                continue
            if dest in self.env.death_pits:
                continue

            dest_to_goal = self._manhattan(dest, self.goal_pos)
            shortcut_gain = current_to_goal - dest_to_goal
            if shortcut_gain < 8 and dest_to_goal > 10:
                continue

            path_to_src = self._bfs_known_to_entry(self.current_pos, src, avoid={'death', 'confusion'})
            if not path_to_src:
                continue

            safe = self._fire_safe_len(path_to_src)
            if safe <= 0:
                continue

            score = len(path_to_src) + dest_to_goal
            if best is None or score < best[0]:
                best = (score, path_to_src[:safe])

        return best[1] if best is not None else []

    def _plan_dynamic_to_goal(self) -> List[Action]:
        if self.env is None or self.current_pos is None or self.goal_pos is None:
            return []

        env = self.env
        start_pos = self.current_pos
        goal = self.goal_pos
        h = env.loader.maze_height_cells
        w = env.loader.maze_width_cells

        clusters = [list(cluster) for cluster in env.fire_clusters]
        pivots = list(getattr(env, "fire_cluster_pivots", []))
        if not pivots:
            pivots = [env.cluster_orientation_and_pivot(cluster)[1] for cluster in clusters]
        phase_deaths = []
        for _ in range(4):
            pits = set()
            for cluster in clusters:
                pits.update(cell for cell in cluster if env.is_cell_in_bounds(*cell))
            phase_deaths.append(pits)
            clusters = [
                env.rotate_fire_cluster(cluster, pivot)
                for cluster, pivot in zip(clusters, pivots)
            ]

        turns_in_cycle_start = env.turn_count % 5
        start_state = (start_pos[0], start_pos[1], 0, turns_in_cycle_start)
        parent = {start_state: None}
        queue = deque([start_state])

        found = None
        max_expansions = 120000
        expansions = 0

        while queue and expansions < max_expansions:
            expansions += 1
            r, c, phase, turns_in_cycle = queue.popleft()
            if (r, c) == goal:
                found = (r, c, phase, turns_in_cycle)
                break

            for intended in (*MOVE_ACTIONS, Action.WAIT):
                dr, dc = DELTAS[intended]
                nr, nc = r + dr, c + dc

                if intended is not Action.WAIT and not (0 <= nr < h and 0 <= nc < w):
                    continue
                if intended is Action.WAIT:
                    nr, nc = r, c

                if intended is not Action.WAIT and ((r, c), (nr, nc)) in self.wall_edges:
                    continue

                if intended is not Action.WAIT and (nr, nc) in env.teleport_map:
                    nr, nc = env.teleport_map[(nr, nc)]

                next_turns = (turns_in_cycle + 1) % 5
                if not self._fire_safe_after_turn((nr, nc), phase_deaths, phase, next_turns):
                    continue
                next_phase = (phase + 1) % 4 if next_turns == 0 else phase
                nxt_state = (nr, nc, next_phase, next_turns)
                if nxt_state in parent:
                    continue
                parent[nxt_state] = ((r, c, phase, turns_in_cycle), intended)
                queue.append(nxt_state)

        if found is None:
            return []

        actions: List[Action] = []
        node = found
        while parent[node] is not None:
            prev, intended = parent[node]
            actions.append(intended)
            node = prev
        actions.reverse()
        return actions



    def _in_bounds(self, r: int, c: int) -> bool:
        if self.env is None:
            return True
        return 0 <= r < self.env.loader.maze_height_cells and 0 <= c < self.env.loader.maze_width_cells

    def _manhattan(self, a: Tuple[int,int], b: Tuple[int,int]) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _bfs_to_entry(self, start: Tuple, goal: Tuple, avoid: set) -> List[Action]:
        if start == goal:
            return []
        visited: dict = {start: None}
        queue = deque([start])
        while queue:
            cur = queue.popleft()
            r, c = cur
            for action in MOVE_ACTIONS:
                dr, dc = DELTAS[action]
                nxt = (r+dr, c+dc)
                if nxt in visited:
                    continue
                if not self._in_bounds(nxt[0], nxt[1]):
                    continue
                if (cur, nxt) in self.wall_edges:
                    continue

                ct = self.known.get(nxt)
                if ct in avoid and nxt != goal:
                    continue
                if self.env is not None and nxt in self.env.death_pits:
                    continue

                visited[nxt] = (cur, action)
                if nxt == goal:
                    queue.clear()
                    break

                queue.append(nxt)

        if goal not in visited or visited[goal] is None:
            return []
        path: List[Action] = []
        node = goal
        while visited[node] is not None:
            parent, action = visited[node]
            path.append(action)
            node = parent
        return list(reversed(path))

    def _bfs_known_to_entry(self, start: Tuple, goal: Tuple, avoid: set) -> List[Action]:
        if start == goal:
            return []
        visited: dict = {start: None}
        queue = deque([start])
        while queue:
            cur = queue.popleft()
            r, c = cur
            for action in MOVE_ACTIONS:
                dr, dc = DELTAS[action]
                nxt = (r+dr, c+dc)
                if nxt in visited:
                    continue
                if (cur, nxt) not in self.open_edges:
                    continue
                if not self._in_bounds(nxt[0], nxt[1]):
                    continue
                if (cur, nxt) in self.wall_edges:
                    continue

                ct = self.known.get(nxt)
                if nxt != goal:
                    if ct in avoid:
                        continue
                    if ct != 'empty':
                        continue
                elif ct in avoid:
                    continue

                if self.env is not None and nxt in self.env.death_pits:
                    continue
                if self._cell_on_fire_cooldown(nxt):
                    continue

                visited[nxt] = (cur, action)
                if nxt == goal:
                    queue.clear()
                    break
                queue.append(nxt)

        if goal not in visited or visited[goal] is None:
            return []
        path: List[Action] = []
        node = goal
        while visited[node] is not None:
            parent, action = visited[node]
            path.append(action)
            node = parent
        return list(reversed(path))

    def _bfs(self, start: Tuple, goal: Tuple, avoid: set) -> List[Action]:
        if start == goal:
            return []
        visited: dict = {start: None}
        queue = deque([start])
        while queue:
            cur = queue.popleft()
            if cur == goal:
                break
            r, c = cur
            for action, (dr, dc) in DELTAS.items():
                nxt = (r+dr, c+dc)
                if nxt in visited:
                    continue
                if not self._in_bounds(nxt[0], nxt[1]):
                    continue
                if (cur, nxt) in self.wall_edges:
                    continue
                effective = nxt
                if self.env is not None and nxt in self.env.teleport_map:
                    effective = self.env.teleport_map[nxt]

                if effective in visited:
                    continue

                ct = self.known.get(effective)
                if ct in avoid:
                    continue
                if self.env is not None and effective in self.env.death_pits:
                    continue
                if self._cell_on_fire_cooldown(effective):
                    continue
                visited[effective] = (cur, action)
                queue.append(effective)

        if goal not in visited or visited[goal] is None:
            return []
        path: List[Action] = []
        node = goal
        while visited[node] is not None:
            parent, action = visited[node]
            path.append(action)
            node = parent
        return list(reversed(path))

    def _frontier_target(self) -> Optional[Tuple[int,int]]:
        pos = self.current_pos
        visited = {pos}
        queue = deque([pos])
        while queue:
            cur = queue.popleft()
            r, c = cur
            for _, (dr, dc) in DELTAS.items():
                nxt = (r+dr, c+dc)
                if nxt in visited:
                    continue
                if not self._in_bounds(nxt[0], nxt[1]):
                    continue
                if (cur, nxt) in self.wall_edges:
                    continue
                ct = self.known.get(nxt)
                if ct == 'death':
                    continue
                if self._cell_on_fire_cooldown(nxt):
                    continue
                visited.add(nxt)
                if ct is None:
                    return nxt
                queue.append(nxt)
        return None

    def _known_safe_prefix(self, path: List[Action]) -> int:
        pos = self.current_pos
        count = 0
        for action in path:
            if action is Action.WAIT:
                break
            dr, dc = DELTAS[action]
            nxt = (pos[0]+dr, pos[1]+dc)
            if (pos, nxt) not in self.open_edges:
                break
            if self.known.get(nxt) != 'empty':
                break
            pos = nxt
            count += 1
        return count

    def _safe_batch_len(self, path: List[Action]) -> int:
        if path and path[0] is Action.WAIT:
            return 1 if self._fire_safe_len(path[:1]) > 0 else 0
        fire_safe = self._fire_safe_len(path)
        if fire_safe <= 0:
            return 0
        safe = self._known_safe_prefix(path)
        if safe <= 1:
            return 1
        return min(5, safe, fire_safe)

    def _remember_open_edge(self, a: Tuple[int,int], b: Tuple[int,int]):
        self.open_edges.add((a, b))
        self.open_edges.add((b, a))

    def _ql_action(self) -> Action:
        if random.random() < self.epsilon:
            choices = self._valid_actions()
            return random.choice(choices) if choices else Action.WAIT
        s = self.state()
        valid = self._valid_actions()
        if not valid:
            return Action.WAIT
        best_q = max(self.q_table[s][a.value] for a in valid)
        best = [a for a in valid if self.q_table[s][a.value] == best_q]
        return random.choice(best)

    def _valid_actions(self) -> List[Action]:
        if self.current_pos is None:
            return []
        pos = self.current_pos
        r, c = pos
        valid: List[Action] = []
        for action in MOVE_ACTIONS:
            dr, dc = DELTAS[action]
            nxt = (r + dr, c + dc)
            if not self._in_bounds(nxt[0], nxt[1]):
                continue
            if (pos, nxt) in self.wall_edges:
                continue
            if self.env is not None and nxt in self.env.death_pits:
                continue
            if not self._fire_safe_until_next_rotation(nxt):
                continue
            if self._cell_on_fire_cooldown(nxt):
                continue
            if self.known.get(nxt) == 'death':
                continue
            valid.append(action)
        if not valid and self._fire_safe_next_turn(pos):
            valid.append(Action.WAIT)
        return valid

    def _escape_action(self) -> Optional[Action]:
        if self.current_pos is None:
            return None
        current = self.current_pos
        candidates = []
        for action in MOVE_ACTIONS:
            dr, dc = DELTAS[action]
            nxt = (current[0] + dr, current[1] + dc)
            if not self._in_bounds(nxt[0], nxt[1]):
                continue
            if (current, nxt) in self.wall_edges:
                continue
            if self.known.get(nxt) == 'death':
                continue
            is_fire = self.env is not None and nxt in self.env.death_pits
            if is_fire:
                continue
            if not self._fire_safe_until_next_rotation(nxt):
                continue
            if self._cell_on_fire_cooldown(nxt):
                continue
            candidates.append((self.visit_count.get(nxt, 0), random.random(), action))
        if not candidates:
            return None
        candidates.sort(key=lambda item: (item[0], item[1]))
        return candidates[0][2]

    def _break_oscillation(self, batch: List[Action]) -> List[Action]:
        if not batch or len(batch) != 1 or self.current_pos is None:
            return batch
        if len(self._recent_positions) < 4:
            return batch

        p0, p1, p2, p3 = list(self._recent_positions)[-4:]
        if not (p0 == p2 and p1 == p3 and p0 != p1):
            return batch

        current = self.current_pos
        intended = batch[0]
        dr, dc = DELTAS[intended]
        intended_next = (current[0] + dr, current[1] + dc)
        if intended_next != p2:
            return batch

        alternatives = []
        for action in self._valid_actions():
            if action == intended:
                continue
            adr, adc = DELTAS[action]
            nxt = (current[0] + adr, current[1] + adc)
            if nxt == p2:
                continue
            alternatives.append(action)
        if not alternatives:
            return batch

        alternatives.sort(key=lambda a: self.visit_count.get((current[0] + DELTAS[a][0], current[1] + DELTAS[a][1]), 0))
        return [alternatives[0]]

    def _pop_batch(self) -> List[Action]:
        take = self._safe_batch_len(self._planned_path)
        if take <= 0:
            self._planned_path = []
            return []
        batch = self._planned_path[:take]
        self._planned_path = self._planned_path[take:]
        self._record_send(batch)
        return batch

    def _record_send(self, intended: List[Action]):
        self._prev_pos     = self.current_pos
        self._last_actions = list(intended)
        self._was_confused = self.confused_turns_left > 0

    def _apply_confusion(self, actions: List[Action]) -> List[Action]:
        if self.confused_turns_left > 0:
            return [INVERT[a] for a in actions]
        return actions
