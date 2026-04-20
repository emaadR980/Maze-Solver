import sys
from enum import Enum
from typing import List, Tuple, Optional
from maze import MazeLoader
from collections import deque
from PIL import Image

class Action(Enum):
    MOVE_UP = 0
    MOVE_DOWN = 1
    MOVE_LEFT = 2
    MOVE_RIGHT = 3
    WAIT = 4


class TurnResult:
    def __init__(self):
        self.wall_hits: int = 0
        self.current_position: Tuple[int, int] = (0, 0)
        self.is_dead: bool = False
        self.is_confused: bool = False
        self.is_goal_reached: bool = False
        self.teleported: bool = False
        self.actions_executed: int = 0
        self.positions_visited: List[Tuple[int, int]] = []  # every position after each action
        self.last_event: str = ""  # human-readable description of the last hazard that fired

    def __repr__(self):
        parts = [f"pos={self.current_position}"]
        if self.wall_hits: parts.append(f"wall_hits={self.wall_hits}")
        if self.is_dead: parts.append("DEAD")
        if self.is_confused: parts.append("CONFUSED")
        if self.teleported: parts.append(f"TELEPORTED")
        if self.is_goal_reached: parts.append("GOAL!")
        parts.append(f"actions={self.actions_executed}")
        return "TurnResult(" + ", ".join(parts) + ")"


class MazeEnvironment:
    CELL_SIZE = 16

    # direction vectors in (row, col) = (dy, dx) form
    DELTAS = {
        Action.MOVE_UP: (-1,  0),
        Action.MOVE_DOWN: ( 1,  0),
        Action.MOVE_LEFT: ( 0, -1),
        Action.MOVE_RIGHT: ( 0,  1),
        Action.WAIT: ( 0,  0),
    }

    # invert controls when confusion trap
    INVERT = {
        Action.MOVE_UP: Action.MOVE_DOWN,
        Action.MOVE_DOWN: Action.MOVE_UP,
        Action.MOVE_LEFT: Action.MOVE_RIGHT,
        Action.MOVE_RIGHT: Action.MOVE_LEFT,
        Action.WAIT: Action.WAIT,
    }

    def __init__(self, maze_image_path: str, rotate_fire: bool = False):
        self.loader = MazeLoader(maze_image_path)
        self.loader.detect_hazards()

        h_cells = self.loader.maze_height_cells
        w_cells = self.loader.maze_width_cells
        self.grid = [[False] * w_cells for _ in range(h_cells)]
        for r in range(h_cells):
            for c in range(w_cells):
                py = r * self.CELL_SIZE + self.CELL_SIZE // 2
                px = c * self.CELL_SIZE + self.CELL_SIZE // 2
                py = min(py, self.loader.maze_array.shape[0] - 1)
                px = min(px, self.loader.maze_array.shape[1] - 1)
                self.grid[r][c] = bool(self.loader.maze_array[py, px])

        sp = self.loader.start_pos
        gp = self.loader.goal_pos
        self.start_cell = self.loader.pixel_to_cell(sp[0], sp[1])
        self.goal_cell = self.loader.pixel_to_cell(gp[0], gp[1])

        self.death_pits = set(map(tuple, self.loader.death_pits))
        self.initial_death_pits = set(self.death_pits)
        self.fire_clusters: List[List[Tuple[int, int]]] = self.group_clusters(self.death_pits)
        self.initial_fire_clusters = [list(cluster) for cluster in self.fire_clusters]
        self.confusion_pads = set(map(tuple, self.loader.confusion_pads))

        all_hazard_cells = (
            self.death_pits
            | self.confusion_pads
            | set(map(tuple, self.loader.teleport_purple))
            | set(map(tuple, self.loader.teleport_orange))
            | set(map(tuple, self.loader.teleport_green))
            | set(map(tuple, self.loader.teleport_red))
        )
        for r, c in all_hazard_cells:
            self.grid[r][c] = True

        self.teleport_map: dict = {}
        for group in [self.loader.teleport_purple, self.loader.teleport_orange, self.loader.teleport_green, self.loader.teleport_red]:
            pads = [tuple(p) for p in group]
            for i in range(0, len(pads) - 1, 2):
                self.teleport_map[pads[i]]     = pads[i + 1]
                self.teleport_map[pads[i + 1]] = pads[i]
            if len(pads) % 2 == 1:          # unpaired pad → maps to itself (no-op)
                self.teleport_map[pads[-1]] = pads[-1]

        # episode state
        self.agent_pos: Tuple[int, int] = self.start_cell
        self.turn_count: int  = 0
        self.death_count: int  = 0
        self.confused_count: int  = 0
        self.teleport_count: int  = 0
        self.confused_turns_left: int = 0   # turns of confusion remaining
        self.cells_explored: set  = set()
        self.episode_active: bool = True
        self.freeze_fire: bool = False   # set True during demo navigation to pause rotation
        self.rotate_fire_enabled: bool = rotate_fire

    def group_clusters(self, cells: set, max_gap: int = 1) -> List[List[Tuple[int, int]]]:
        remaining = set(cells)
        clusters = []
        while remaining:
            seed = next(iter(remaining))
            cluster = []
            queue = [seed]
            remaining.discard(seed)
            while queue:
                cell = queue.pop()
                cluster.append(cell)
                for other in list(remaining):
                    if abs(other[0] - cell[0]) <= max_gap and abs(other[1] - cell[1]) <= max_gap:
                        remaining.discard(other)
                        queue.append(other)
            clusters.append(cluster)
        return clusters

    def cluster_orientation_and_pivot(self, cluster: List[Tuple[int, int]]) -> Tuple[str, Tuple[int, int]]:
        rows = [r for r, _ in cluster]
        cols = [c for _, c in cluster]

        min_r, max_r = min(rows), max(rows)
        min_c, max_c = min(cols), max(cols)

        row_counts = {r: 0 for r in range(min_r, max_r + 1)}
        col_counts = {c: 0 for c in range(min_c, max_c + 1)}
        for r, c in cluster:
            row_counts[r] = row_counts.get(r, 0) + 1
            col_counts[c] = col_counts.get(c, 0) + 1

        width = max_c - min_c + 1
        height = max_r - min_r + 1

        if width >= height:
            if row_counts.get(max_r, 0) <= row_counts.get(min_r, 0):
                pivot_candidates = [cell for cell in cluster if cell[0] == max_r]
                return "v", min(pivot_candidates, key=lambda cell: abs(cell[1] - sum(cols) / len(cols)))
            pivot_candidates = [cell for cell in cluster if cell[0] == min_r]
            return "^", min(pivot_candidates, key=lambda cell: abs(cell[1] - sum(cols) / len(cols)))

        if col_counts.get(min_c, 0) <= col_counts.get(max_c, 0):
            pivot_candidates = [cell for cell in cluster if cell[1] == min_c]
            return "<", min(pivot_candidates, key=lambda cell: abs(cell[0] - sum(rows) / len(rows)))
        
        pivot_candidates = [cell for cell in cluster if cell[1] == max_c]
        return ">", min(pivot_candidates, key=lambda cell: abs(cell[0] - sum(rows) / len(rows)))

    def rotate_fire_cluster(self, cluster: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        h = self.loader.maze_height_cells
        w = self.loader.maze_width_cells

        _orientation, pivot = self.cluster_orientation_and_pivot(cluster)
        pr, pc = pivot

        rotated = []
        seen = set()
        for r, c in cluster:
            dr = r - pr
            dc = c - pc
            nr = pr + dc
            nc = pc - dr
            if 0 <= nr < h and 0 <= nc < w:
                cell = (nr, nc)
                if cell not in seen:
                    seen.add(cell)
                    rotated.append(cell)

        return sorted(rotated)

    # rotate all clusters
    def rotate_fire_clusters(self):
        if self.freeze_fire:
            return
        # restore original passability for old pit cells
        for r, c in self.death_pits:
            py = r * self.CELL_SIZE + self.CELL_SIZE // 2
            px = c * self.CELL_SIZE + self.CELL_SIZE // 2
            py = min(py, self.loader.maze_array.shape[0] - 1)
            px = min(px, self.loader.maze_array.shape[1] - 1)
            self.grid[r][c] = bool(self.loader.maze_array[py, px])

        new_clusters = []
        new_death_pits = set()
        for cluster in self.fire_clusters:
            rotated = self.rotate_fire_cluster(cluster)
            new_clusters.append(rotated)
            new_death_pits.update(rotated)

        self.fire_clusters = new_clusters
        self.death_pits = new_death_pits

        for r, c in self.death_pits:
            self.grid[r][c] = True

    def reset(self) -> Tuple[int, int]:
        self.agent_pos = self.start_cell
        self.turn_count = 0
        self.death_count = 0
        self.confused_count = 0
        self.teleport_count = 0
        self.confused_turns_left = 0
        self.cells_explored = set()
        self.episode_active = True

        self.death_pits = set(self.initial_death_pits)
        self.fire_clusters = [list(cluster) for cluster in self.initial_fire_clusters]

        for r, c in self.death_pits:
            self.grid[r][c] = True

        return self.agent_pos

    def is_passable(self, row: int, col: int) -> bool:
        h = self.loader.maze_height_cells
        w = self.loader.maze_width_cells
        if not (0 <= row < h and 0 <= col < w):
            return False
        return self.grid[row][col]

    def is_move_passable(self, from_r: int, from_c: int, to_r: int, to_c: int,
                         from_hazard: bool = False, to_hazard: bool = False) -> bool:
        h = self.loader.maze_height_cells
        w = self.loader.maze_width_cells
        if not (0 <= to_r < h and 0 <= to_c < w):
            return False

        # Hazard cells (teleport / confusion / fire) contain emoji pixels that
        # darken their interior, so we must not use the interior pixel of a hazard
        # cell to judge whether a wall exists there.  Instead only check the pixel
        # on the NON-hazard side of the shared boundary.
        from_is_hazard = from_hazard or (from_r, from_c) in self.death_pits \
            or (from_r, from_c) in self.confusion_pads \
            or (from_r, from_c) in self.teleport_map
        to_is_hazard   = to_hazard   or (to_r,   to_c)   in self.death_pits \
            or (to_r,   to_c)   in self.confusion_pads \
            or (to_r,   to_c)   in self.teleport_map

        ma    = self.loader.maze_array
        CELL  = self.CELL_SIZE
        BORDER = 1

        dr = to_r - from_r

        if dr == 0:  # horizontal move
            wall_x = max(from_c, to_c) * CELL + BORDER
            y = min(from_r * CELL + CELL // 2 + BORDER, ma.shape[0] - 1)
            # x_left  = pixel inside from-cell;  x_right = pixel inside to-cell
            x_left  = min(wall_x - 1, ma.shape[1] - 1)
            x_right = min(wall_x + 1, ma.shape[1] - 1)
            # If moving right, x_left is from-cell side, x_right is to-cell side
            if from_c < to_c:
                from_px, to_px = x_left, x_right
            else:
                from_px, to_px = x_right, x_left
            check_from = (not from_is_hazard)
            check_to   = (not to_is_hazard)
            if check_from and not bool(ma[y, from_px]):
                return False
            if check_to and not bool(ma[y, to_px]):
                return False
            # If both sides are hazards we can't tell – assume passable unless
            # the exact boundary pixel (wall_x) is dark.
            if not check_from and not check_to:
                bx = min(wall_x, ma.shape[1] - 1)
                return bool(ma[y, bx])
            return True

        else:  # vertical move
            wall_y = max(from_r, to_r) * CELL + BORDER
            x = min(from_c * CELL + CELL // 2 + BORDER, ma.shape[1] - 1)
            # y_top = pixel inside upper cell; y_bottom = pixel inside lower cell
            y_top    = min(wall_y - 1, ma.shape[0] - 1)
            y_bottom = min(wall_y + 1, ma.shape[0] - 1)
            # If moving down, from-cell is y_top side, to-cell is y_bottom side
            if from_r < to_r:
                from_py, to_py = y_top, y_bottom
            else:
                from_py, to_py = y_bottom, y_top
            check_from = (not from_is_hazard)
            check_to   = (not to_is_hazard)
            if check_from and not bool(ma[from_py, x]):
                return False
            if check_to and not bool(ma[to_py, x]):
                return False
            if not check_from and not check_to:
                by = min(wall_y, ma.shape[0] - 1)
                return bool(ma[by, x])
            return True

    def step(self, actions: List[Action]) -> TurnResult:
        if not actions or len(actions) > 5:
            raise ValueError("Must submit 1-5 actions per turn.")

        result = TurnResult()
        result.current_position = self.agent_pos

        currently_confused = self.confused_turns_left > 0

        for i, raw_action in enumerate(actions):
            action = self.INVERT[raw_action] if currently_confused else raw_action

            if action == Action.WAIT:
                result.actions_executed += 1
                continue

            dr, dc = self.DELTAS[action]
            r, c = self.agent_pos
            nr, nc = r + dr, c + dc

            _src_hazard = ((r, c) in self.death_pits or (r, c) in self.confusion_pads
                           or (r, c) in self.teleport_map)
            _dst_hazard = ((nr, nc) in self.death_pits or (nr, nc) in self.confusion_pads
                           or (nr, nc) in self.teleport_map)
            if not self.is_move_passable(r, c, nr, nc,
                                         from_hazard=_src_hazard, to_hazard=_dst_hazard):
                result.wall_hits += 1
                result.actions_executed += 1
                continue

            self.agent_pos = (nr, nc)
            self.cells_explored.add(self.agent_pos)
            result.actions_executed += 1
            result.positions_visited.append(self.agent_pos)

            if self.agent_pos in self.teleport_map:
                src = self.agent_pos
                dest = self.teleport_map[self.agent_pos]
                self.agent_pos = dest
                result.teleported = True
                result.current_position = self.agent_pos
                result.last_event = f"TELEPORT {src}->{dest}"
                self.teleport_count += 1

            if self.agent_pos in self.confusion_pads:
                result.is_confused = True
                self.confused_turns_left = 2
                self.confused_count += 1
                currently_confused = True
                result.last_event = f"CONFUSED at {self.agent_pos}"

            if self.agent_pos in self.death_pits:
                result.is_dead = True
                self.death_count += 1
                result.current_position = self.agent_pos # position of pit
                result.last_event = f"DEATH at {self.agent_pos}"
                self.agent_pos = self.start_cell
                break

            if self.agent_pos == self.goal_cell:
                result.is_goal_reached = True
                result.current_position = self.agent_pos
                self.episode_active = False
                break

        if self.confused_turns_left > 0:
            self.confused_turns_left -= 1

        self.turn_count += 1

        if self.rotate_fire_enabled and self.turn_count % 5 == 0:
            self.rotate_fire_clusters()

            if not result.is_dead and self.agent_pos in self.death_pits:
                result.is_dead = True
                self.death_count += 1
                result.current_position = self.agent_pos
                result.last_event = f"DEATH by rotating fire at {self.agent_pos}"
                self.agent_pos = self.start_cell

        if not result.is_dead:
            result.current_position = self.agent_pos
        return result

    def get_episode_stats(self) -> dict:
        return {
            "turns_taken": self.turn_count,
            "deaths": self.death_count,
            "confused": self.confused_count,
            "cells_explored": len(self.cells_explored),
            "goal_reached": not self.episode_active,
        }

# draw red square over every death pit cell and save to output_path
def visualize_fire_pits(env: MazeEnvironment, output_path: str, base_image_path: str):
    img = Image.open(base_image_path).convert("RGB")
    pixels = img.load()
    marker = 4
    h, w = env.loader.h, env.loader.w

    for r, c in env.death_pits:
        py = r * env.CELL_SIZE + env.CELL_SIZE // 2
        px = c * env.CELL_SIZE + env.CELL_SIZE // 2
        for dy in range(-marker, marker + 1):
            for dx in range(-marker, marker + 1):
                ny, nx = py + dy, px + dx
                if 0 <= ny < h and 0 <= nx < w:
                    pixels[nx, ny] = (255, 0, 0)

    img.save(output_path)
    print(f"  Saved {output_path}  ({len(env.death_pits)} fire pits drawn)")

# scripted to try each hazard
class DemoAgent:
    def __init__(self, env: MazeEnvironment):
        self.env = env

    def path_to(self, target: Tuple[int, int], avoid: set = None) -> List[Action]:
        start = self.env.agent_pos
        if start == target:
            return []

        avoid = avoid or set()
        visited = {start: None}
        queue = deque([start])

        while queue:
            r, c = queue.popleft()
            if (r, c) == target:
                break
            for action, (dr, dc) in MazeEnvironment.DELTAS.items():
                if action == Action.WAIT:
                    continue
                nxt = (r + dr, c + dc)
                if nxt in visited or nxt in avoid:
                    continue
                src_hazard = ((r, c) in self.env.death_pits or (r, c) in self.env.confusion_pads
                              or (r, c) in self.env.teleport_map)
                dst_hazard = (nxt in self.env.death_pits or nxt in self.env.confusion_pads
                              or nxt in self.env.teleport_map)
                if self.env.is_move_passable(r, c, nxt[0], nxt[1],
                                             from_hazard=src_hazard, to_hazard=dst_hazard):
                    visited[nxt] = ((r, c), action)
                    queue.append(nxt)

        if target not in visited:
            return []

        actions = []
        node = target
        while visited[node] is not None:
            node, act = visited[node]
            actions.append(act)
            
        return list(reversed(actions))

    def walk(self, actions: List[Action], label: str = ""):
        last = None
        for i in range(0, len(actions), 5):
            result = self.env.step(actions[i:i+5])
            last = result
            if label:
                print(f"  [{label}] {result}")
            if result.is_dead or result.is_goal_reached or result.teleported:
                return result
        return last

    def walk_to(self, target: Tuple[int, int], label: str = "", avoid: set = None):
        actions = self.path_to(target, avoid=avoid)
        if not actions:
            return
        return self.walk(actions, label)

    def reachable_step_to(self, target: Tuple[int, int], avoid: set = None) -> Optional[Tuple[Tuple[int, int], Action]]:
        avoid = avoid or set()
        r, c = target
        candidates = [
            ((r - 1, c), Action.MOVE_DOWN),
            ((r + 1, c), Action.MOVE_UP),
            ((r, c - 1), Action.MOVE_RIGHT),
            ((r, c + 1), Action.MOVE_LEFT),
        ]

        current_adj = None
        for adj, step_action in candidates:
            if adj == self.env.agent_pos:
                current_adj = (adj, step_action)
                break
        if current_adj is not None:
            candidates = [current_adj] + [item for item in candidates if item != current_adj]

        for adj, step_action in candidates:
            if not self.env.is_passable(adj[0], adj[1]):
                continue
            if adj in avoid:
                continue
            adj_hazard = (adj in self.env.death_pits or adj in self.env.confusion_pads
                          or adj in self.env.teleport_map)
            tgt_hazard = (target in self.env.death_pits or target in self.env.confusion_pads
                          or target in self.env.teleport_map)
            if not self.env.is_move_passable(adj[0], adj[1], r, c,
                                             from_hazard=adj_hazard, to_hazard=tgt_hazard):
                continue
            path = self.path_to(adj, avoid=avoid | {target})
            if path or self.env.agent_pos == adj:
                return adj, step_action
        return None

    def find_reachable_probe(self, hazard_cells, avoid: set = None) -> Optional[Tuple[Tuple[int, int], Tuple[int, int], Action]]:
        avoid = avoid or set()
        for target in hazard_cells:
            step_info = self.reachable_step_to(target, avoid=avoid)
            if step_info is not None:
                adj, step_action = step_info
                return target, adj, step_action
        return None

    def find_any_probe(self, hazard_cells) -> Optional[Tuple[Tuple[int, int], Tuple[int, int], Action]]:
        for target in hazard_cells:
            r, c = target
            candidates = [
                ((r - 1, c), Action.MOVE_DOWN),
                ((r + 1, c), Action.MOVE_UP),
                ((r, c - 1), Action.MOVE_RIGHT),
                ((r, c + 1), Action.MOVE_LEFT),
            ]
            for adj, step_action in candidates:
                if not self.env.is_passable(adj[0], adj[1]):
                    continue
                adj_hazard = (adj in self.env.death_pits or adj in self.env.confusion_pads
                              or adj in self.env.teleport_map)
                tgt_hazard = (target in self.env.death_pits or target in self.env.confusion_pads
                              or target in self.env.teleport_map)
                if self.env.is_move_passable(adj[0], adj[1], r, c,
                                             from_hazard=adj_hazard, to_hazard=tgt_hazard):
                    return target, adj, step_action
        return None

    def find_reachable_hazard(self, hazard_cells, avoid: set = None) -> Optional[Tuple[int, int]]:
        """Return the first hazard cell in hazard_cells that has a reachable adjacent cell
        from which the agent can actually step onto the hazard (no wall between them)."""
        avoid = avoid or set()
        for target in hazard_cells:
            if self.reachable_step_to(target, avoid=avoid) is not None:
                return target
        return None

    # navigate to cell adjacent to target w the same optional avoidance
    def step_onto(self, target: Tuple[int, int], avoid: set = None) -> TurnResult:
        step_info = self.reachable_step_to(target, avoid=avoid)
        if step_info is not None:
            adj, step_action = step_info
            if self.env.agent_pos != adj:
                self.walk_to(adj, avoid=(avoid or set()) | {target})
            if self.env.agent_pos == adj:
                return self.env.step([step_action])
        return self.env.step([Action.WAIT])

    def run_demo(self):
        sep = "─" * 20

        print(f"{sep}")
        print(f"  Start cell : {self.env.start_cell}")
        print(f"  Goal cell  : {self.env.goal_cell}")
        print(f"  Death pits : {len(self.env.death_pits)}")
        print(f"  Confusion  : {len(self.env.confusion_pads)}")
        print(f"  Teleports  : {len(self.env.teleport_map)} pads")
        print(f"{sep}\n")

        all_hazards = (self.env.death_pits | self.env.confusion_pads | set(self.env.teleport_map.keys()))

        # death pit demo
        if self.env.death_pits:
            avoid_non_death = self.env.confusion_pads | set(self.env.teleport_map.keys())
            pit = self.find_reachable_hazard(self.env.death_pits, avoid=avoid_non_death)
            if pit is None:
                pit = next(iter(self.env.death_pits))  # fallback
            print(f"Death Pit @ cell {pit}")
            print(f"  Expected : is_dead=True, agent respawns at start")
            print(f"  Before   : {self.env.agent_pos}")

            self.env.freeze_fire = True
            result = self.walk(self.path_to(pit, avoid=avoid_non_death), label="->pit")
            self.env.freeze_fire = False
            if result is None:
                result = TurnResult()
            print(f"  Result   : {result}")
            
            ok = "ok" if result.is_dead else "nope"
            print(f"  is_dead={result.is_dead} {ok}  /  "
                  f"respawned at {self.env.agent_pos}  "
                  f"(start={self.env.start_cell})")
            print()
        else:
            print("No death pits detected.\n")

        # confusion demo
        if self.env.confusion_pads:
            probe = self.find_reachable_probe(self.env.confusion_pads, avoid=self.env.death_pits)
            repositioned = False
            if probe is None:
                probe = self.find_any_probe(self.env.confusion_pads)
                repositioned = probe is not None
            if probe is None:
                pad = None
            else:
                pad, adj, probe_action = probe
        else:
            pad = None
            repositioned = False

        if pad is not None:
            print(f"Confusion Pad @ cell {pad}")
            print(f"  Expected : is_confused=True, MOVE_UP executes as MOVE_DOWN")
            if repositioned:
                self.env.agent_pos = adj
                print(f"  Repositioned next to pad at {adj} for direct verification")
            # clear any leftover confusion from navigation
            self.env.confused_turns_left = 0
            self.env.freeze_fire = True
            if repositioned:
                result = self.env.step([probe_action])
            else:
                result = self.step_onto(pad, avoid=self.env.death_pits)
            self.env.freeze_fire = False
            # ensure confusion is registered even if pad was hit mid navigation
            if result.is_confused or self.env.confused_turns_left > 0:
                self.env.confused_turns_left = max(self.env.confused_turns_left, 1)
            else:
                self.env.confused_turns_left = 0
                
            print(f"  Result   : {result}")
            ok = "ok" if result.is_confused else "nope"
            print(f"  is_confused={result.is_confused} {ok}  /  "
                  f"confused_turns_left={self.env.confused_turns_left}")

            # show inversion -> send MOVE_UP, expect row to increase (DOWN)
            pos_before = self.env.agent_pos
            self.env.step([Action.MOVE_UP])
            dr = self.env.agent_pos[0] - pos_before[0]
            if dr > 0:
                verdict = "moved DOWN (correct)"
            elif dr < 0:
                verdict = "moved UP (inversion not active)"
            else:
                verdict = "no row change (blocked both directions)"
            print(f"  Sent MOVE_UP while confused -> row change={dr:+d}  {verdict}")
            print()
        elif self.env.confusion_pads:
            print("No reachable confusion pads detected.\n")
        else:
            print("No confusion pads detected.\n")

        # teleport demo
        if self.env.teleport_map:
            self.env.agent_pos = self.env.start_cell
            self.env.confused_turns_left = 0
            teleport_pads = set(self.env.teleport_map.keys())
            src = None
            dst = None
            src_adj = None
            src_action = None
            repositioned = False
            for candidate in teleport_pads:
                candidate_dst = self.env.teleport_map[candidate]
                candidate_avoid = self.env.death_pits | self.env.confusion_pads | (teleport_pads - {candidate})
                probe = self.find_reachable_probe([candidate], avoid=candidate_avoid)
                if probe is not None:
                    src, src_adj, src_action = probe
                    dst = candidate_dst
                    break
            if src is None:
                for candidate in teleport_pads:
                    probe = self.find_any_probe([candidate])
                    if probe is not None:
                        src, src_adj, src_action = probe
                        dst = self.env.teleport_map[candidate]
                        repositioned = True
                        break
            if src is None:
                print("No reachable teleport pads detected.\n")
            else:
                print(f"Teleport Pad @ cell {src} -> dest {dst}")
                print(f"  Expected : teleported=True, position jumps to {dst}")
                if repositioned:
                    self.env.agent_pos = src_adj
                    print(f"  Repositioned next to pad at {src_adj} for direct verification")
                teleport_avoid = self.env.death_pits | self.env.confusion_pads | (teleport_pads - {src})
                self.env.freeze_fire = True
                if repositioned:
                    result = self.env.step([src_action])
                else:
                    result = self.step_onto(src, avoid=teleport_avoid)
                self.env.freeze_fire = False
                print(f"  Result   : {result}")
                ok = "ok" if result.teleported else "nope"
                match_ok = "ok" if self.env.agent_pos == dst else "nope"
                print(f"  teleported={result.teleported} {ok}  /  "
                      f"landed at {self.env.agent_pos}  expected {dst} {match_ok}")
                print()
        else:
            print("No teleport pads detected.\n")

        # final status
        stats = self.env.get_episode_stats()
        print(f"{sep}")
        print("  EPISODE STATS")
        print(f"{sep}")
        for k, v in stats.items():
            print(f"  {k:<20}: {v}")
        print(f"{sep}\n")

if __name__ == "__main__":
    MAZE_PATH = sys.argv[1] if len(sys.argv) > 1 else "maze-alpha/MAZE_1.png"

    print("Loading maze with hazards")
    env = MazeEnvironment(MAZE_PATH)
    env.reset()

    agent = DemoAgent(env)
    agent.run_demo()

    # fire rotation visualization
    print("Visualizing fire pit rotation...")
    env.reset()
    env.confused_turns_left = 0

    visualize_fire_pits(env, "fire_before.png", MAZE_PATH)
    env.step([Action.WAIT] * 5)  # one full turn triggers rotation
    visualize_fire_pits(env, "fire_after.png", MAZE_PATH)