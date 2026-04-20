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
        self._learned_fire_phases = [set() for _ in range(4)]
        self._fire_phase = 0
        self._learned_teleports: dict = {}
        self._waiting_to_cross_fire = False
        self._fire_cross_blocked_until: dict = {}
        self._pending_fire_probe: List[Action] = []
        self._probe_after_wait = False
        self._sealed_cells: set = set()
        self._sealed_refresh_in = 0
        self._blocked_fire_bursts: set = set()

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
        self._fire_phase = 0
        self._waiting_to_cross_fire = False
        self._fire_cross_blocked_until.clear()
        self._pending_fire_probe = []
        self._probe_after_wait = False
        self._sealed_cells.clear()
        self._sealed_refresh_in = 0
        self._blocked_fire_bursts.clear()
        self.visit_count    = defaultdict(int)
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.episode += 1

    def plan_turn(self, last_result: Optional[TurnResult]) -> List[Action]:
        if last_result is not None:
            self._process_result(last_result)

        if self.current_pos is None:
            return [Action.WAIT]

        if self._sealed_refresh_in <= 0:
            self._sealed_cells = self._compute_sealed_cells()
            self._sealed_refresh_in = 25
        else:
            self._sealed_refresh_in -= 1

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
            probe = self._fire_probe_batch()
            if probe:
                self._record_send(probe)
                return self._apply_confusion(probe)
            if not self._fire_safe_next_turn(self.current_pos):
                escape = self._escape_action()
                if escape is not None:
                    self._record_send([escape])
                    return self._apply_confusion([escape])
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
            crossing = self._fire_crossing_batch(path)
            if crossing:
                self._record_send(crossing)
                return self._apply_confusion(crossing)
            probe = self._fire_probe_batch()
            if probe:
                self._record_send(probe)
                return self._apply_confusion(probe)
            self._planned_path = []
            if not self._fire_safe_next_turn(self.current_pos):
                escape = self._escape_action()
                if escape is not None:
                    self._waiting_to_cross_fire = False
                    self._record_send([escape])
                    return self._apply_confusion([escape])
            act = self._ql_action()
            self._waiting_to_cross_fire = False
            self._record_send([act])
            return self._apply_confusion([act])
        self._planned_path = path[take:]
        batch = path[:take]
        batch = self._break_oscillation(batch)
        self._waiting_to_cross_fire = False
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
            "learned_fire_phases": [set(phase) for phase in self._learned_fire_phases],
            "learned_teleports": dict(self._learned_teleports),
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
        self.known       = {k: v for k, v in raw_known.items() if v not in ('wall', 'death')}
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
        learned_fire = data.get("learned_fire_phases")
        if learned_fire and len(learned_fire) == 4:
            self._learned_fire_phases = [set(phase) for phase in learned_fire]
        else:
            self._learned_fire_phases = [set() for _ in range(4)]
        self._fire_phase = 0
        self._learned_teleports = data.get("learned_teleports", {})
        self._waiting_to_cross_fire = False
        self._fire_cross_blocked_until = {}
        self._pending_fire_probe = []
        self._probe_after_wait = False
        self._sealed_cells = set()
        self._sealed_refresh_in = 0
        self._blocked_fire_bursts = set()
        self.confused_turns_left = 0
        self._was_confused = False
        print(f"Loaded (ep={self.episode}, ε={self.epsilon:.3f}, "
              f"goal={self.goal_pos}, known={len(self.known)} cells, "
              f"wall_edges={len(self.wall_edges)//2})")

    def _process_result(self, result: TurnResult):
        self._fire_phase = (self._fire_phase + 1) % 4

        if result.is_dead:
            failed_phase = (self._fire_phase - 1) % 4
            if self._prev_pos is not None and len(self._last_actions) >= 2:
                self._blocked_fire_bursts.add((
                    self._prev_pos,
                    tuple(self._last_actions),
                    failed_phase,
                ))
            self._learn_edges_from_result(result)
            self._waiting_to_cross_fire = False
            self._pending_fire_probe = []
            self._probe_after_wait = False
            if "rotating fire" in result.last_event:
                self._remember_fire_death(result.current_position, self._fire_phase)
            else:
                self._remember_fire_death(result.current_position, (self._fire_phase - 1) % 4)
            self._planned_path = []
            self.confused_turns_left = 0
            self._stagnation_turns = 0
            self._respawn_wait_turns = random.randint(1, 4)
            # Respawn at start
            if self.start_pos is not None:
                self.current_pos = self.start_pos
            return

        self._learn_edges_from_result(result)
        self.current_pos = result.current_position

        if result.is_goal_reached:
            self.goal_pos = self.current_pos
            self._planned_path = []

        if result.teleported or result.is_confused:
            self._planned_path = []
            self._waiting_to_cross_fire = False
            self._pending_fire_probe = []
            self._probe_after_wait = False

        if result.is_confused:
            self.confused_turns_left = 1   # environment inverts 1 more full turn (T+1); compensate for that
        elif self.confused_turns_left > 0:
            self.confused_turns_left -= 1

        def _actual(intended: Action) -> Action:
            return intended

        if not result.teleported and not result.is_confused:
            for visited_pos in result.positions_visited:
                if self.known.get(visited_pos) not in ('death', 'confusion', 'teleport'):
                    self.known[visited_pos] = 'empty'

        # Label current cell
        if result.is_confused:
            self.known[self.current_pos] = 'confusion'
        elif result.teleported:
            if len(self._last_actions) == 1 and self._prev_pos is not None:
                dr, dc = DELTAS[_actual(self._last_actions[0])]
                src_pad = (self._prev_pos[0]+dr, self._prev_pos[1]+dc)
                self._remember_open_edge(self._prev_pos, src_pad)
                self.known[src_pad] = 'teleport'
                self._learned_teleports[src_pad] = self.current_pos
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

        if result.wall_hits > 0:
            self._planned_path = []
            self._waiting_to_cross_fire = False
            self._pending_fire_probe = []
            self._probe_after_wait = False

        self.visit_count[self.current_pos] += 1
        self._recent_positions.append(self.current_pos)
        if self.visit_count[self.current_pos] > 1:
            self._stagnation_turns += 1
        else:
            self._stagnation_turns = 0

    def _learn_edges_from_result(self, result: TurnResult) -> None:
        if self._prev_pos is None or not self._last_actions:
            return
        if result.teleported or result.is_confused:
            return

        pos = self._prev_pos
        visited = deque(result.positions_visited)
        for action in self._last_actions:
            if action is Action.WAIT:
                continue
            dr, dc = DELTAS[action]
            expected = (pos[0] + dr, pos[1] + dc)
            if visited and visited[0] == expected:
                visited.popleft()
                self._remember_open_edge(pos, expected)
                pos = expected
            elif (pos, expected) not in self.open_edges:
                self.wall_edges.add((pos, expected))
                self.wall_edges.add((expected, pos))

    def _fire_phase_sets(self):
        return [
            set(self._learned_fire_phases[(self._fire_phase + offset) % 4])
            for offset in range(4)
        ]

    def _fire_safe_len_from(self, start: Tuple[int,int], path: List[Action]) -> int:
        if not any(self._learned_fire_phases):
            return len(path)
        phases = self._fire_phase_sets()
        pos = start
        current_fire = phases[0]
        next_fire = phases[1]
        max_safe = 0
        for k, action in enumerate(path):
            if action is Action.WAIT:
                if pos in current_fire:
                    return max_safe
                if pos not in next_fire:
                    max_safe = k + 1
                return max_safe

            dr, dc = DELTAS[action]
            pos = (pos[0] + dr, pos[1] + dc)
            if pos in self._learned_teleports:
                pos = self._learned_teleports[pos]
            if pos in current_fire:
                return max_safe
            if pos not in next_fire:
                max_safe = k + 1
        return max_safe

    def _fire_safe_len(self, path: List[Action]) -> int:
        return self._fire_safe_len_from(self.current_pos, path)

    def _fire_safe_after_turn(self, pos: Tuple[int,int],
                              phase_deaths: List[set],
                              phase: int) -> bool:
        if pos in phase_deaths[phase]:
            return False
        return pos not in phase_deaths[(phase + 1) % 4]

    def _fire_safe_for_turns(self, pos: Tuple[int,int], turns: int) -> bool:
        if not any(self._learned_fire_phases):
            return True
        phases = self._fire_phase_sets()
        phase = 0
        if pos in phases[phase]:
            return False

        for _ in range(max(0, turns)):
            phase = (phase + 1) % 4
            if pos in phases[phase]:
                return False
        return True

    def _turns_until_fire_rotation(self) -> int:
        return 1

    def _fire_safe_next_turn(self, pos: Tuple[int,int]) -> bool:
        return self._fire_safe_for_turns(pos, 1)

    def _fire_safe_until_next_rotation(self, pos: Tuple[int,int]) -> bool:
        return self._fire_safe_for_turns(pos, self._turns_until_fire_rotation())

    def _remember_fire_death(self, pos: Tuple[int,int], phase: int) -> None:
        self._learned_fire_phases[phase].add(pos)
        self._fire_death_counts[pos] += 1
        if self.known.get(pos) == 'death':
            self.known.pop(pos)
        self._fire_cooldown_until.pop(pos, None)

    def _cell_on_fire_cooldown(self, cell: Tuple[int,int]) -> bool:
        return False

    def _is_known_traversable(self, cell: Tuple[int,int]) -> bool:
        if self.known.get(cell) in {'death', 'confusion'}:
            return False
        if self.known.get(cell) is None:
            return False
        return not self._is_permanent_fire(cell)

    def _has_ordinary_frontier(self, cell: Tuple[int,int]) -> bool:
        r, c = cell
        for action in MOVE_ACTIONS:
            dr, dc = DELTAS[action]
            nxt = (r + dr, c + dc)
            if (cell, nxt) in self.wall_edges:
                continue
            if self.known.get(nxt) is not None:
                continue
            if self._is_learned_fire(nxt):
                if not self._is_permanent_fire(nxt):
                    return True
                continue
            return True
        return False

    def _is_useful_teleport_source(self, cell: Tuple[int,int]) -> bool:
        dest = self._learned_teleports.get(cell)
        if dest is None or dest == cell:
            return False
        if self.known.get(dest) in {'death', 'confusion'}:
            return False
        if self.goal_pos is None:
            return True
        return self._manhattan(dest, self.goal_pos) < self._manhattan(cell, self.goal_pos)

    def _compute_sealed_cells(self) -> set:
        known_cells = {
            cell for cell in self.known
            if self._is_known_traversable(cell)
        }
        if not known_cells:
            return set()

        useful = set()
        for cell in known_cells:
            if cell == self.current_pos:
                continue
            if cell == self.goal_pos:
                useful.add(cell)
            elif self._has_ordinary_frontier(cell):
                useful.add(cell)
            elif self._has_fire_probe_candidate(cell):
                useful.add(cell)
            elif self._is_useful_teleport_source(cell):
                useful.add(cell)

        active = set(useful)
        queue = deque(useful)
        while queue:
            cur = queue.popleft()
            r, c = cur
            for action in MOVE_ACTIONS:
                dr, dc = DELTAS[action]
                nxt = (r + dr, c + dc)
                if nxt in active or nxt not in known_cells:
                    continue
                if (cur, nxt) not in self.open_edges:
                    continue
                if (cur, nxt) in self.wall_edges:
                    continue
                active.add(nxt)
                queue.append(nxt)

        return known_cells - active

    def _is_sealed_for_planning(self, cell: Tuple[int,int],
                                start: Tuple[int,int] = None,
                                goal: Tuple[int,int] = None) -> bool:
        if cell == start or cell == goal or cell == self.current_pos:
            return False
        return cell in self._sealed_cells

    def _make_plan(self) -> List[Action]:
        pos = self.current_pos

        teleport_path = self._plan_to_beneficial_teleport()
        if teleport_path:
            return teleport_path

        if self.goal_pos is not None:
            for avoid in ({'death', 'confusion'}, {'death'}):
                path = self._bfs(pos, self.goal_pos, avoid=avoid)
                if path:
                    return path

        skipped_targets = set()
        for _ in range(20):
            target = self._frontier_target(skipped_targets)
            if target is None:
                break
            path = self._bfs(pos, target, avoid={'death'})
            if not path:
                skipped_targets.add(target)
                continue
            if self._crossing_unavailable(path):
                skipped_targets.add(target)
                continue
            return path

        fire_probe_path = self._path_to_fire_probe_source()
        if fire_probe_path:
            return fire_probe_path

        return []

    def _plan_to_beneficial_teleport(self) -> List[Action]:
        if self.current_pos is None or self.goal_pos is None or not self._learned_teleports:
            return []

        current_to_goal = self._manhattan(self.current_pos, self.goal_pos)
        best: Optional[Tuple[int, List[Action]]] = None

        for src, dest in self._learned_teleports.items():
            if src == dest:
                continue
            if self.known.get(src) in {'death', 'confusion'}:
                continue
            if self.known.get(dest) in {'death', 'confusion'}:
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

    def _in_bounds(self, r: int, c: int) -> bool:
        return True

    def _manhattan(self, a: Tuple[int,int], b: Tuple[int,int]) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

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
                if self._is_sealed_for_planning(nxt, start, goal):
                    continue

                ct = self.known.get(nxt)
                if nxt != goal:
                    if ct in avoid:
                        continue
                    if ct != 'empty':
                        continue
                elif ct in avoid:
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
            for action in MOVE_ACTIONS:
                dr, dc = DELTAS[action]
                nxt = (r+dr, c+dc)
                if nxt in visited:
                    continue
                if not self._in_bounds(nxt[0], nxt[1]):
                    continue
                if (cur, nxt) in self.wall_edges:
                    continue
                edge_known = (cur, nxt) in self.open_edges
                if not edge_known and nxt != goal:
                    continue
                if self._is_sealed_for_planning(nxt, start, goal):
                    continue
                effective = nxt
                if nxt in self._learned_teleports:
                    effective = self._learned_teleports[nxt]

                if effective in visited:
                    continue
                if self._is_sealed_for_planning(effective, start, goal):
                    continue

                ct = self.known.get(effective)
                if ct in avoid:
                    continue
                if effective == goal and self._is_learned_fire(effective):
                    continue
                if self._cell_on_fire_cooldown(effective):
                    continue
                if not edge_known and self.known.get(nxt) is not None:
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

    def _frontier_target(self, excluded: set = None) -> Optional[Tuple[int,int]]:
        excluded = excluded or set()
        pos = self.current_pos
        visited = {pos}
        queue = deque([pos])
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
                if self._is_sealed_for_planning(nxt, pos, None):
                    continue
                ct = self.known.get(nxt)
                if ct == 'death':
                    continue
                if self._is_learned_fire(nxt) and (self._is_permanent_fire(nxt) or ct is None):
                    continue
                if self._cell_on_fire_cooldown(nxt):
                    continue
                if ct is None and nxt not in excluded:
                    return nxt
                if (cur, nxt) not in self.open_edges:
                    continue
                visited.add(nxt)
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
        safe = self._known_safe_prefix(path)
        if safe <= 0:
            if not self._fire_safe_next_turn(self.current_pos):
                return 0
            return 1 if self._fire_safe_len(path[:1]) == 1 else 0

        for take in range(min(5, safe, len(path)), 0, -1):
            if self._fire_safe_len(path[:take]) == take:
                return take
        return 0

    def _fire_crossing_batch(self, path: List[Action]) -> List[Action]:
        if not path or not any(self._learned_fire_phases):
            self._waiting_to_cross_fire = False
            return []

        burst_len = min(2, len(path), self._known_safe_prefix(path))
        if burst_len < 2:
            self._waiting_to_cross_fire = False
            return []

        burst = path[:burst_len]
        key = (self.current_pos, tuple(burst))
        blocked_until = self._fire_cross_blocked_until.get(key, -1)
        if blocked_until > self.episode:
            self._waiting_to_cross_fire = False
            return []

        burst_positions = self._simulate_positions(self.current_pos, burst)
        if any(self._is_permanent_fire(pos) for pos in burst_positions):
            self._waiting_to_cross_fire = False
            self._fire_cross_blocked_until[key] = self.episode + 1
            return []

        if self._fire_safe_len(burst) == burst_len:
            self._planned_path = path[burst_len:]
            self._waiting_to_cross_fire = False
            self._fire_cross_blocked_until.pop(key, None)
            return burst

        if self._waiting_to_cross_fire:
            self._planned_path = path[burst_len:]
            self._waiting_to_cross_fire = False
            return burst

        if self._fire_safe_len([Action.WAIT]) == 1:
            self._planned_path = path
            self._waiting_to_cross_fire = True
            return [Action.WAIT]

        self._waiting_to_cross_fire = False
        return []

    def _fire_probe_batch(self) -> List[Action]:
        if not any(self._learned_fire_phases) or self.current_pos is None:
            self._pending_fire_probe = []
            self._probe_after_wait = False
            return []

        if self._probe_after_wait:
            self._probe_after_wait = False
            burst = self._choose_fire_probe_burst(self.current_pos)
            if burst and self._fire_safe_len(burst) == len(burst):
                return burst
            return []

        if self._pending_fire_probe:
            burst = self._pending_fire_probe
            self._pending_fire_probe = []
            if self._fire_safe_len(burst) == len(burst):
                return burst
            return []

        burst = self._choose_fire_probe_burst(self.current_pos)
        if burst and self._fire_safe_len(burst) == len(burst):
            return burst

        if not self._has_fire_probe_candidate(self.current_pos):
            return []

        if self._fire_safe_len([Action.WAIT]) == 1:
            self._probe_after_wait = True
            return [Action.WAIT]
        return []

    def _has_fire_probe_candidate(self, current: Tuple[int,int], aggressive: bool = False) -> bool:
        return bool(self._choose_fire_probe_burst(current, require_timing=False, aggressive=aggressive))

    def _known_component_has_frontier(self, start: Tuple[int,int]) -> bool:
        if self.known.get(start) is None:
            return True

        visited = {start}
        queue = deque([start])
        while queue:
            cur = queue.popleft()
            r, c = cur
            for action in MOVE_ACTIONS:
                dr, dc = DELTAS[action]
                nxt = (r + dr, c + dc)
                if (cur, nxt) in self.wall_edges:
                    continue
                if self._is_permanent_fire(nxt):
                    continue
                if self.known.get(nxt) is None:
                    if not self._is_learned_fire(nxt):
                        return True
                    continue
                if (cur, nxt) not in self.open_edges:
                    continue
                if nxt in visited:
                    continue
                if self.known.get(nxt) in {'death', 'confusion'}:
                    continue
                if self._is_learned_fire(nxt):
                    continue
                visited.add(nxt)
                queue.append(nxt)

        return False

    def _same_known_component(self, start: Tuple[int,int], target: Tuple[int,int]) -> bool:
        if start == target:
            return True
        if self.known.get(target) is None:
            return False

        visited = {start}
        queue = deque([start])
        while queue:
            cur = queue.popleft()
            r, c = cur
            for action in MOVE_ACTIONS:
                dr, dc = DELTAS[action]
                nxt = (r + dr, c + dc)
                if nxt in visited:
                    continue
                if (cur, nxt) not in self.open_edges:
                    continue
                if (cur, nxt) in self.wall_edges:
                    continue
                if self.known.get(nxt) in {'death', 'confusion'}:
                    continue
                if self._is_learned_fire(nxt):
                    continue
                if nxt == target:
                    return True
                visited.add(nxt)
                queue.append(nxt)

        return False

    def _choose_fire_probe_burst(self, current: Tuple[int,int],
                                 require_timing: bool = True,
                                 aggressive: bool = False) -> List[Action]:
        candidates = []
        recent_positions = set(self._recent_positions)
        for first in MOVE_ACTIONS:
            dr1, dc1 = DELTAS[first]
            fire_cell = (current[0] + dr1, current[1] + dc1)
            if (current, fire_cell) in self.wall_edges:
                continue
            if not self._is_learned_fire(fire_cell) or self._is_permanent_fire(fire_cell):
                continue
            fire_unknown = self.known.get(fire_cell) is None

            for second in MOVE_ACTIONS:
                dr2, dc2 = DELTAS[second]
                landing = (fire_cell[0] + dr2, fire_cell[1] + dc2)
                if landing == current:
                    continue
                if (fire_cell, landing) in self.wall_edges:
                    continue
                if self._is_permanent_fire(landing):
                    continue
                same_component = self._same_known_component(current, landing)
                if not aggressive:
                    if (self.known.get(landing) is not None
                            and same_component
                            and not fire_unknown):
                        continue
                    if (self.known.get(landing) is not None
                            and not fire_unknown
                            and not self._known_component_has_frontier(landing)):
                        continue
                elif self._is_sealed_for_planning(landing, current, None):
                    continue
                burst = [first, second]
                if (current, tuple(burst), self._fire_phase) in self._blocked_fire_bursts:
                    continue
                if require_timing and self._fire_safe_len_from(current, burst) != len(burst):
                    continue
                known_penalty = 0 if self.known.get(landing) is None else (2 if fire_unknown else 10)
                same_component_penalty = 40 if aggressive and same_component and not fire_unknown else 0
                recent_penalty = 20 if aggressive and landing in recent_positions else 0
                visit_penalty = self.visit_count.get(landing, 0)
                goal_score = self._manhattan(landing, self.goal_pos) if self.goal_pos else 0
                candidates.append((
                    known_penalty + same_component_penalty + recent_penalty,
                    visit_penalty,
                    goal_score,
                    random.random(),
                    burst,
                ))

        if not candidates:
            return []
        candidates.sort(key=lambda item: item[:4])
        return candidates[0][4]

    def _path_to_fire_probe_source(self, aggressive: bool = False) -> List[Action]:
        if self.current_pos is None or not any(self._learned_fire_phases):
            return []

        visited = {self.current_pos: None}
        queue = deque([self.current_pos])

        while queue:
            cur = queue.popleft()
            if self._has_fire_probe_candidate(cur, aggressive=aggressive):
                if cur == self.current_pos:
                    return []
                path = []
                node = cur
                while visited[node] is not None:
                    parent, action = visited[node]
                    path.append(action)
                    node = parent
                path.reverse()
                return path

            r, c = cur
            for action in MOVE_ACTIONS:
                dr, dc = DELTAS[action]
                nxt = (r + dr, c + dc)
                if nxt in visited:
                    continue
                if (cur, nxt) not in self.open_edges:
                    continue
                if (cur, nxt) in self.wall_edges:
                    continue
                if self.known.get(nxt) in {'death', 'confusion'}:
                    continue
                if self._is_sealed_for_planning(nxt, self.current_pos, None):
                    continue
                if self._is_learned_fire(nxt) and (self._is_permanent_fire(nxt) or self.known.get(nxt) is None):
                    continue
                visited[nxt] = (cur, action)
                queue.append(nxt)

        return []

    def _crossing_unavailable(self, path: List[Action]) -> bool:
        if not path or not any(self._learned_fire_phases):
            return False
        burst_len = min(2, len(path), self._known_safe_prefix(path))
        if burst_len < 2:
            return False
        burst = path[:burst_len]
        key = (self.current_pos, tuple(burst))
        if self._fire_cross_blocked_until.get(key, -1) > self.episode:
            return True
        return any(self._is_permanent_fire(pos) for pos in self._simulate_positions(self.current_pos, burst))

    def _simulate_positions(self, start: Tuple[int,int], actions: List[Action]) -> List[Tuple[int,int]]:
        pos = start
        positions = []
        for action in actions:
            dr, dc = DELTAS[action]
            pos = (pos[0] + dr, pos[1] + dc)
            positions.append(pos)
        return positions

    def _is_permanent_fire(self, pos: Tuple[int,int]) -> bool:
        return all(pos in phase_cells for phase_cells in self._learned_fire_phases)

    def _is_learned_fire(self, pos: Tuple[int,int]) -> bool:
        return any(pos in phase_cells for phase_cells in self._learned_fire_phases)

    def _remember_open_edge(self, a: Tuple[int,int], b: Tuple[int,int]):
        self.wall_edges.discard((a, b))
        self.wall_edges.discard((b, a))
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
        known_open: List[Action] = []
        for action in MOVE_ACTIONS:
            dr, dc = DELTAS[action]
            nxt = (r + dr, c + dc)
            if not self._in_bounds(nxt[0], nxt[1]):
                continue
            if (pos, nxt) in self.wall_edges:
                continue
            if not self._fire_safe_until_next_rotation(nxt):
                continue
            if self._cell_on_fire_cooldown(nxt):
                continue
            if self.known.get(nxt) == 'death':
                continue
            valid.append(action)
            if (pos, nxt) in self.open_edges:
                known_open.append(action)
        unsealed = [
            action for action in valid
            if not self._is_sealed_for_planning(
                (pos[0] + DELTAS[action][0], pos[1] + DELTAS[action][1]),
                pos,
                None,
            )
        ]
        if unsealed:
            valid = unsealed
            known_open = [action for action in known_open if action in valid]
        if valid and not self._fire_safe_next_turn(pos) and known_open:
            return known_open
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
            if not self._fire_safe_until_next_rotation(nxt):
                continue
            if self._cell_on_fire_cooldown(nxt):
                continue
            sealed_penalty = 1000 if self._is_sealed_for_planning(nxt, current, None) else 0
            candidates.append((sealed_penalty, self.visit_count.get(nxt, 0), random.random(), action))
        if not candidates:
            return None
        candidates.sort(key=lambda item: (item[0], item[1], item[2]))
        return candidates[0][3]

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
            crossing = self._fire_crossing_batch(self._planned_path)
            if crossing:
                self._record_send(crossing)
                return crossing
            self._planned_path = []
            return []
        batch = self._planned_path[:take]
        self._planned_path = self._planned_path[take:]
        self._waiting_to_cross_fire = False
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
