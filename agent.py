from hazardDemo import Action, TurnResult
from collections import defaultdict, deque
import heapq
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

    def reset_episode(self):
        self.confused_turns_left = 0
        self._was_confused  = False
        self._planned_path  = []
        self._prev_pos      = None
        self._last_actions  = []
        self._recent_positions.clear()
        self._stagnation_turns = 0
        self._respawn_wait_turns = 0
        self.visit_count    = defaultdict(int)
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.episode += 1
        self.known = {pos: ct for pos, ct in self.known.items()
                      if ct == 'empty'}

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

        if self.env is not None and self.env.death_pits:
            take = 1
        else:
            safe_count = self._known_safe_prefix(path)
            take = min(5, safe_count) if safe_count > 1 else 1
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
        self.confused_turns_left = 0
        self._was_confused = False
        print(f"Loaded (ep={self.episode}, ε={self.epsilon:.3f}, "
              f"goal={self.goal_pos}, known={len(self.known)} cells, "
              f"wall_edges={len(self.wall_edges)//2})")

    def _process_result(self, result: TurnResult):
        if result.is_dead:
            if self.env is None:
                self.known[result.current_position] = 'death'
            self._planned_path = []
            self.confused_turns_left = 0
            self._stagnation_turns = 0
            # Avoid phase-locking with rotating fire cycles.
            # A fixed 5-turn wait repeats the exact same fire phase and can cause infinite death loops.
            if self.env is not None and self.env.rotate_fire_enabled:
                self._respawn_wait_turns = random.randint(1, 4)
            else:
                self._respawn_wait_turns = random.randint(1, 2)
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
                self.known[src_pad] = 'teleport'
            self.known[self.current_pos] = 'teleport'
        else:
            if self.known.get(self.current_pos) not in ('death', 'confusion', 'teleport'):
                self.known[self.current_pos] = 'empty'

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
        phases = []
        for _ in range(4):
            pits = set()
            for cl in clusters:
                pits.update(cl)
            phases.append(pits)
            clusters = [self.env.rotate_fire_cluster(cl) for cl in clusters]
        return phases

    def _fire_safe_len(self, path: List[Action]) -> int:
        if self.env is None or not self.env.fire_clusters:
            return len(path)
        phases = self._fire_phase_sets()
        pos = self.current_pos
        phase = 0
        turns_in_cycle = self.env.turn_count % 5   # position within the current 5-turn window
        for k, action in enumerate(path):
            dr, dc = DELTAS[action]
            pos = (pos[0] + dr, pos[1] + dc)
            if pos in phases[phase]:
                return k
            turns_in_cycle = (turns_in_cycle + 1) % 5
            if turns_in_cycle == 0:
                phase = (phase + 1) % 4
        return len(path)

    def _make_plan(self) -> List[Action]:
        pos = self.current_pos
        fire_active = self.env is not None and bool(self.env.death_pits)

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

    def _plan_dynamic_to_goal(self) -> List[Action]:
        if self.env is None or self.current_pos is None or self.goal_pos is None:
            return []

        env = self.env
        start_pos = self.current_pos
        goal = self.goal_pos
        h = env.loader.maze_height_cells
        w = env.loader.maze_width_cells

        clusters = [list(cluster) for cluster in env.fire_clusters]
        phase_deaths = []
        for _ in range(4):
            pits = set()
            for cluster in clusters:
                pits.update(cluster)
            phase_deaths.append(pits)
            clusters = [env.rotate_fire_cluster(cluster) for cluster in clusters]

        turns_in_cycle_start = env.turn_count % 5
        start_state = (start_pos[0], start_pos[1], 0, turns_in_cycle_start)
        parent = {start_state: None}
        best_cost = {start_state: 0.0}
        pq = [(self._goal_heuristic(start_pos, goal), 0.0, start_state)]

        found = None
        max_expansions = 120000
        expansions = 0

        while pq and expansions < max_expansions:
            expansions += 1
            _, cost_so_far, state = heapq.heappop(pq)
            if cost_so_far != best_cost.get(state):
                continue
            r, c, phase, turns_in_cycle = state
            if (r, c) == goal:
                found = state
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

                if (nr, nc) in phase_deaths[phase]:
                    continue

                next_turns = (turns_in_cycle + 1) % 5
                next_phase = (phase + 1) % 4 if next_turns == 0 else phase

                # Fire rotates after every 5th turn, and the environment can kill
                # the agent immediately if the rotated pit lands on its position.
                if next_turns == 0 and (nr, nc) in phase_deaths[next_phase]:
                    continue

                nxt_state = (nr, nc, next_phase, next_turns)
                transition_cost = 1.0 + self._dynamic_risk_cost(
                    current=(r, c),
                    nxt=(nr, nc),
                    intended=intended,
                    phase=phase,
                    next_phase=next_phase,
                    phase_deaths=phase_deaths,
                )
                new_cost = cost_so_far + transition_cost
                if new_cost >= best_cost.get(nxt_state, float('inf')):
                    continue
                best_cost[nxt_state] = new_cost
                parent[nxt_state] = ((r, c, phase, turns_in_cycle), intended)
                priority = new_cost + self._goal_heuristic((nr, nc), goal)
                heapq.heappush(pq, (priority, new_cost, nxt_state))

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

    def _goal_heuristic(self, pos: Tuple[int, int], goal: Tuple[int, int]) -> float:
        return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])

    def _dynamic_risk_cost(
        self,
        current: Tuple[int, int],
        nxt: Tuple[int, int],
        intended: Action,
        phase: int,
        next_phase: int,
        phase_deaths: List[set],
    ) -> float:
        """Soft penalties for routes that are safe but likely to lead to bad fire timing."""
        risk = 0.0

        if intended is Action.WAIT:
            risk += 4.0

        risk += 0.15 * self.visit_count.get(nxt, 0)

        current_pits = phase_deaths[phase]
        next_pits = phase_deaths[next_phase]

        def manhattan_to_nearest(cell: Tuple[int, int], pits: set) -> Optional[int]:
            if not pits:
                return None
            return min(abs(cell[0] - pr) + abs(cell[1] - pc) for pr, pc in pits)

        dist_now = manhattan_to_nearest(nxt, current_pits)
        dist_next = manhattan_to_nearest(nxt, next_pits)

        if dist_now is not None:
            if dist_now == 1:
                risk += 6.0
            elif dist_now == 2:
                risk += 2.5

        if dist_next is not None:
            if dist_next == 0:
                risk += 1000.0
            elif dist_next == 1:
                risk += 10.0
            elif dist_next == 2:
                risk += 4.0

        if nxt == current:
            risk += 2.0

        return risk



    def _in_bounds(self, r: int, c: int) -> bool:
        if self.env is None:
            return True
        return 0 <= r < self.env.loader.maze_height_cells and 0 <= c < self.env.loader.maze_width_cells

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
                visited.add(nxt)
                if ct is None:
                    return nxt
                queue.append(nxt)
        return None

    def _known_safe_prefix(self, path: List[Action]) -> int:
        pos = self.current_pos
        count = 0
        for action in path:
            dr, dc = DELTAS[action]
            pos = (pos[0]+dr, pos[1]+dc)
            if self.known.get(pos) != 'empty':
                break
            count += 1
        return count

    def _ql_action(self) -> Action:
        if random.random() < self.epsilon:
            choices = self._valid_actions()
            return random.choice(choices) if choices else random.choice(MOVE_ACTIONS)
        s = self.state()
        valid = self._valid_actions()
        if not valid:
            valid = MOVE_ACTIONS
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
            if (pos, nxt) in self.wall_edges:
                continue
            if self.env is not None and nxt in self.env.death_pits:
                continue
            if self.known.get(nxt) == 'death':
                continue
            valid.append(action)
        return valid

    def _escape_action(self) -> Optional[Action]:
        if self.current_pos is None:
            return None
        current = self.current_pos
        allow_fire = (self._stagnation_turns > 60
                      and self.env is not None
                      and bool(self.env.death_pits))
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
            if is_fire and not allow_fire:
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
        if self.env is not None and self.env.death_pits:
            take = 1
        else:
            safe = self._known_safe_prefix(self._planned_path)
            take = min(5, safe) if safe > 0 else 1
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
