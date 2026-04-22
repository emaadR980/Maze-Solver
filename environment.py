#Environment.py
from enum import Enum
from typing import List, Tuple, Optional

from collections import deque
from PIL import Image
from maze import MazeLoader          # colour-based loader — no CNN needed


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
        self.arrow_pushed: bool = False

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

    DELTAS = {
        Action.MOVE_UP:    (-1,  0),
        Action.MOVE_DOWN:  ( 1,  0),
        Action.MOVE_LEFT:  ( 0, -1),
        Action.MOVE_RIGHT: ( 0,  1),
        Action.WAIT:       ( 0,  0),
    }

    INVERT = {
        Action.MOVE_UP:    Action.MOVE_DOWN,
        Action.MOVE_DOWN:  Action.MOVE_UP,
        Action.MOVE_LEFT:  Action.MOVE_RIGHT,
        Action.MOVE_RIGHT: Action.MOVE_LEFT,
        Action.WAIT:       Action.WAIT,
    }

    def __init__(self, maze_image_path: str):
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
        self.goal_cell  = self.loader.pixel_to_cell(gp[0], gp[1])

        self.confusion_pads      = set(map(tuple, self.loader.confusion_pads))
        self.death_pits          = set(map(tuple, self.loader.death_pits))
        # Pass 1 — remove explicit overlap (colour-detected confusion pads)
        self.death_pits         -= self.confusion_pads
        print(f"[ENV] death_pits={len(self.death_pits)}"
              f"  confusion_pads={len(self.confusion_pads)}"
)

        self.initial_death_pits  = set(self.death_pits)
        self.fire_clusters = self.group_clusters(self.death_pits, max_gap=3)
        self.initial_fire_clusters = [list(c) for c in self.fire_clusters]
        self._fire_rotation_states = self._precompute_fire_states()
        self._fire_rot_idx = 0

        # Arrow pads (must be before all_hazard_cells)
        self.arrow_up   = set(map(tuple, getattr(self.loader, "arrow_up",   [])))
        self.arrow_left = set(map(tuple, getattr(self.loader, "arrow_left", [])))
        print(f"  arrow_up={len(self.arrow_up)}  arrow_left={len(self.arrow_left)}")

        all_hazard_cells = (
            self.death_pits | self.confusion_pads
            | set(map(tuple, self.loader.teleport_purple))
            | set(map(tuple, self.loader.teleport_orange))
            | set(map(tuple, self.loader.teleport_green))
            | set(map(tuple, getattr(self.loader, "teleport_red", [])))
            | self.arrow_up | self.arrow_left
        )
        for r, c in all_hazard_cells:
            self.grid[r][c] = True

        self.teleport_map: dict = {}
        for group in [self.loader.teleport_purple,
                      self.loader.teleport_orange,
                      self.loader.teleport_green,
                      getattr(self.loader, "teleport_red", [])]:
            pads = [tuple(p) for p in group]
            for i, pad in enumerate(pads):
                dest = pads[(i + 1) % len(pads)] if len(pads) > 1 else pad
                self.teleport_map[pad] = dest

        self.agent_pos           : Tuple[int, int] = self.start_cell
        self.turn_count          : int  = 0
        self.death_count         : int  = 0
        self.confused_count      : int  = 0
        self.confused_turns_left : int  = 0
        self.cells_explored      : set  = set()
        self.episode_active      : bool = True
        self._fire_turn_counter  : int  = 0

        # ── BUILD ADJACENCY ───────────────────────────────────────────────────
        self._build_adjacency()

    def _edge_is_open(self, r: int, c: int, nr: int, nc: int) -> bool:
        """Check a shared cell edge using a band scan instead of one center pixel."""
        ma     = self.loader.maze_array
        cell   = self.CELL_SIZE
        margin = 2
        min_open = max(6, int((cell - 2 * margin) * 0.6))

        if nr == r + 1 and nc == c:
            by = min((r + 1) * cell, ma.shape[0] - 1)
            x0 = c * cell + margin
            x1 = (c + 1) * cell - margin
            open_samples = 0
            total = 0
            for bx in range(x0, x1):
                bx = min(max(bx, 0), ma.shape[1] - 1)
                above = min(max(by - 2, 0), ma.shape[0] - 1)
                below = min(max(by + 2, 0), ma.shape[0] - 1)
                total += 1
                if bool(ma[by, bx]) and bool(ma[above, bx]) and bool(ma[below, bx]):
                    open_samples += 1
            return open_samples >= min_open

        if nr == r and nc == c + 1:
            bx = min((c + 1) * cell, ma.shape[1] - 1)
            y0 = r * cell + margin
            y1 = (r + 1) * cell - margin
            open_samples = 0
            total = 0
            for by in range(y0, y1):
                by = min(max(by, 0), ma.shape[0] - 1)
                left  = min(max(bx - 2, 0), ma.shape[1] - 1)
                right = min(max(bx + 2, 0), ma.shape[1] - 1)
                total += 1
                if bool(ma[by, bx]) and bool(ma[by, left]) and bool(ma[by, right]):
                    open_samples += 1
            return open_samples >= min_open

        return False

    # ── Adjacency from boundary pixels ───────────────────────────────────────
    def _build_adjacency(self):
        h  = self.loader.maze_height_cells
        w  = self.loader.maze_width_cells

        self.adj = [[set() for _ in range(w)] for _ in range(h)]
        for r in range(h):
            for c in range(w):
                if r + 1 < h:
                    if self._edge_is_open(r, c, r + 1, c):
                        self.adj[r    ][c].add((r + 1, c))
                        self.adj[r + 1][c].add((r,     c))
                if c + 1 < w:
                    if self._edge_is_open(r, c, r, c + 1):
                        self.adj[r][c    ].add((r, c + 1))
                        self.adj[r][c + 1].add((r, c    ))

    def group_clusters(self, cells: set, max_gap: int = 8) -> List[List[Tuple[int, int]]]:
        remaining = set(cells)
        clusters  = []
        while remaining:
            seed    = next(iter(remaining))
            cluster = []
            queue   = [seed]
            remaining.discard(seed)
            while queue:
                cell = queue.pop()
                cluster.append(cell)
                for other in list(remaining):
                    if abs(other[0]-cell[0]) <= max_gap and abs(other[1]-cell[1]) <= max_gap:
                        remaining.discard(other)
                        queue.append(other)
            clusters.append(cluster)
        return clusters

    # ── Border-aware apex pivot ───────────────────────────────────────────────
    def _find_pivot(self, cluster: list) -> Tuple[int, int]:
        if len(cluster) <= 1:
            return cluster[0]

        h = self.loader.maze_height_cells
        w = self.loader.maze_width_cells
        cluster_set = set(map(tuple, cluster))

        border_cells = [
            (r, c) for r, c in cluster
            if r == 0 or r == h - 1 or c == 0 or c == w - 1
        ]
        if border_cells:
            def _nbr_count(rc):
                r, c = rc
                return sum(
                    1 for dr in (-1, 0, 1) for dc in (-1, 0, 1)
                    if (dr, dc) != (0, 0) and (r + dr, c + dc) in cluster_set
                )
            return max(border_cells, key=_nbr_count)

        best_cell, best_score = None, -float('inf')
        for r, c in cluster:
            nbrs = [
                (r + dr, c + dc)
                for dr in (-1, 0, 1) for dc in (-1, 0, 1)
                if (dr, dc) != (0, 0) and (r + dr, c + dc) in cluster_set
            ]
            if len(nbrs) != 2:
                continue
            (r1, c1), (r2, c2) = nbrs
            d1r, d1c = r1 - r, c1 - c
            d2r, d2c = r2 - r, c2 - c
            dot   = d1r * d2r + d1c * d2c
            denom = ((d1r**2 + d1c**2) ** 0.5) * ((d2r**2 + d2c**2) ** 0.5)
            cos_a = dot / denom if denom > 0 else -1.0
            if cos_a > best_score:
                best_score = cos_a
                best_cell  = (r, c)

        if best_cell is not None:
            return best_cell

        rows = [r for r, _ in cluster]
        cols = [c for _, c in cluster]
        cr   = sum(rows) / len(rows)
        cc   = sum(cols) / len(cols)
        return min(cluster, key=lambda cell: (cell[0] - cr) ** 2 + (cell[1] - cc) ** 2)

    def _rotate_cluster_90(self, cluster: list, origin_cluster: list) -> list:
        if len(origin_cluster) <= 1:
            return sorted(cluster)

        h  = self.loader.maze_height_cells
        w  = self.loader.maze_width_cells
        pr, pc = self._find_pivot(origin_cluster)

        rotated = []
        for r, c in cluster:
            dr = r - pr
            dc = c - pc
            nr = pr + dc
            nc = pc - dr
            if not (0 <= nr < h and 0 <= nc < w):
                return sorted(origin_cluster)
            rotated.append((nr, nc))
        return sorted(rotated)

    def _precompute_fire_states(self) -> list:
        states    = [frozenset(self.death_pits)]
        originals = [list(c) for c in self.initial_fire_clusters]
        current   = [list(c) for c in self.initial_fire_clusters]

        for _ in range(3):
            next_clusters = []
            next_pits     = set()
            for orig, cur in zip(originals, current):
                rotated = self._rotate_cluster_90(cur, orig)
                next_clusters.append(rotated)
                next_pits.update(rotated)
            current = next_clusters
            states.append(frozenset(next_pits))

        return states

    def reset(self) -> Tuple[int, int]:
        self.agent_pos           = self.start_cell
        self.turn_count          = 0
        self.death_count         = 0
        self.confused_count      = 0
        self.confused_turns_left = 0
        self.cells_explored      = set()
        self.episode_active      = True
        self._fire_turn_counter  = 0
        self._fire_rot_idx       = 0
        for r, c in self.death_pits:
            py = min(r*self.CELL_SIZE + self.CELL_SIZE//2, self.loader.maze_array.shape[0]-1)
            px = min(c*self.CELL_SIZE + self.CELL_SIZE//2, self.loader.maze_array.shape[1]-1)
            self.grid[r][c] = bool(self.loader.maze_array[py, px])
        self.death_pits    = set(self.initial_death_pits)
        self.fire_clusters = [list(c) for c in self.initial_fire_clusters]
        for r, c in self.death_pits:
            self.grid[r][c] = True
        return self.agent_pos

    def is_passable(self, row: int, col: int) -> bool:
        h = self.loader.maze_height_cells
        w = self.loader.maze_width_cells
        if not (0 <= row < h and 0 <= col < w):
            return False
        return self.grid[row][col]

    def is_cell_in_bounds(self, r: int, c: int) -> bool:
        return (0 <= r < self.loader.maze_height_cells
                and 0 <= c < self.loader.maze_width_cells)

    def step(self, actions: List[Action]) -> TurnResult:
        if not actions or len(actions) > 5:
            raise ValueError("Must submit 1-5 actions per turn.")

        result = TurnResult()
        result.current_position = self.agent_pos
        currently_confused      = self.confused_turns_left > 0

        for i, raw_action in enumerate(actions):
            action = self.INVERT[raw_action] if currently_confused else raw_action

            if action == Action.WAIT:
                result.actions_executed += 1
                continue

            dr, dc = self.DELTAS[action]
            r,  c  = self.agent_pos
            nr, nc = r + dr, c + dc

            # Use adjacency graph for wall detection
            if (nr, nc) not in self.adj[r][c]:
                result.wall_hits += 1
                result.actions_executed += 1
                continue

            self.agent_pos = (nr, nc)
            self.cells_explored.add(self.agent_pos)
            result.actions_executed += 1

            if self.agent_pos in self.teleport_map:
                self.agent_pos          = self.teleport_map[self.agent_pos]
                result.teleported       = True
                result.current_position = self.agent_pos

            # Arrow pad: force one step in arrow direction, uses full turn
            if self.agent_pos in self.arrow_up or self.agent_pos in self.arrow_left:
                _ar, _ac = self.agent_pos
                _fdr, _fdc = (-1, 0) if self.agent_pos in self.arrow_up else (0, -1)
                _fnr, _fnc = _ar + _fdr, _ac + _fdc
                if (_fnr, _fnc) in self.adj[_ar][_ac]:
                    self.agent_pos = (_fnr, _fnc)
                    self.cells_explored.add(self.agent_pos)
                    result.arrow_pushed = True
                    result.current_position = self.agent_pos
                    # Full hazard chain on landing cell
                    if self.agent_pos in self.teleport_map:
                        self.agent_pos          = self.teleport_map[self.agent_pos]
                        result.teleported       = True
                        result.current_position = self.agent_pos
                    if self.agent_pos in self.confusion_pads:
                        result.is_confused       = True
                        self.confused_turns_left = 2
                        self.confused_count     += 1
                        currently_confused       = True
                    if self.agent_pos in self.death_pits:
                        result.is_dead          = True
                        self.death_count       += 1
                        result.current_position = self.agent_pos
                        self.agent_pos          = self.start_cell
                    elif self.agent_pos == self.goal_cell:
                        result.is_goal_reached  = True
                        self.episode_active     = False
                # Arrow always uses the full turn regardless of whether push succeeded
                break

            if self.agent_pos in self.confusion_pads:
                result.is_confused       = True
                self.confused_turns_left = 2
                self.confused_count     += 1
                currently_confused       = True

            if self.agent_pos in self.death_pits:
                result.is_dead          = True
                self.death_count       += 1
                result.current_position = self.agent_pos
                self.agent_pos          = self.start_cell
                break

            if self.agent_pos == self.goal_cell:
                result.is_goal_reached  = True
                result.current_position = self.agent_pos
                self.episode_active     = False
                break

        if not result.is_dead:
            result.current_position = self.agent_pos

        if self.confused_turns_left > 0:
            self.confused_turns_left -= 1

        # Fire rotates 90° every turn
        self._fire_rot_idx = (self._fire_rot_idx + 1) % 4

        for r, c in self.death_pits:
            py = min(r * self.CELL_SIZE + self.CELL_SIZE // 2, self.loader.maze_array.shape[0] - 1)
            px = min(c * self.CELL_SIZE + self.CELL_SIZE // 2, self.loader.maze_array.shape[1] - 1)
            self.grid[r][c] = bool(self.loader.maze_array[py, px])

        self.death_pits = set(self._fire_rotation_states[self._fire_rot_idx])

        for r, c in self.death_pits:
            self.grid[r][c] = True

        self.turn_count += 1
        return result

    def get_episode_stats(self) -> dict:
        return {
            "turns_taken":    self.turn_count,
            "deaths":         self.death_count,
            "confused":       self.confused_count,
            "cells_explored": len(self.cells_explored),
            "goal_reached":   not self.episode_active,
        }


def visualize_fire_pits(env: MazeEnvironment, output_path: str, base_image_path: str):
    img    = Image.open(base_image_path).convert("RGB")
    pixels = img.load()
    marker = 4
    h, w   = env.loader.h, env.loader.w
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


class DemoAgent:
    def __init__(self, env: MazeEnvironment):
        self.env = env

    def path_to(self, target, avoid=None):
        start   = self.env.agent_pos
        if start == target: return []
        avoid   = avoid or set()
        visited = {start: None}
        queue   = deque([start])
        while queue:
            r, c = queue.popleft()
            if (r, c) == target: break
            for nxt in self.env.adj[r][c]:
                if nxt not in visited and nxt not in avoid:
                    visited[nxt] = ((r, c),
                        next(a for a, (dr, dc) in MazeEnvironment.DELTAS.items()
                             if (r+dr, c+dc) == nxt))
                    queue.append(nxt)
        if target not in visited: return []
        actions = []
        node    = target
        while visited[node] is not None:
            node, act = visited[node]
            actions.append(act)
        return list(reversed(actions))

    def walk(self, actions, label=""):
        last = None
        for i in range(0, len(actions), 5):
            result = self.env.step(actions[i:i+5])
            last   = result
            if label: print(f"  [{label}] {result}")
            if result.is_dead or result.is_goal_reached: return result
        return last

    def walk_to(self, target, label="", avoid=None):
        actions = self.path_to(target, avoid=avoid)
        if actions: return self.walk(actions, label)

    def step_onto(self, target, avoid=None):
        r, c = target
        for adj, step_action in [
            ((r-1,c), Action.MOVE_DOWN), ((r+1,c), Action.MOVE_UP),
            ((r,c-1), Action.MOVE_RIGHT),((r,c+1), Action.MOVE_LEFT),
        ]:
            if not self.env.is_passable(adj[0], adj[1]): continue
            if self.env.agent_pos != adj: self.walk_to(adj, avoid=avoid)
            if self.env.agent_pos == adj: return self.env.step([step_action])
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
        all_hazards = self.env.death_pits | self.env.confusion_pads | set(self.env.teleport_map)

        if self.env.death_pits:
            pit    = next(iter(self.env.death_pits))
            print(f"Death Pit @ {pit}")
            result = self.step_onto(pit, avoid=all_hazards - {pit})
            print(f"  {result}")
            print(f"  is_dead={'ok' if result.is_dead else 'nope'}  respawned={self.env.agent_pos}\n")

        if self.env.confusion_pads:
            pad = next(iter(self.env.confusion_pads))
            print(f"Confusion Pad @ {pad}")
            self.env.confused_turns_left = 0
            result = self.step_onto(pad, avoid=all_hazards - {pad})
            print(f"  {result}")
            pos_b = self.env.agent_pos
            self.env.step([Action.MOVE_UP])
            dr = self.env.agent_pos[0] - pos_b[0]
            print(f"  MOVE_UP while confused -> row Δ={dr:+d}"
                  f"  {'correct ✓' if dr > 0 else 'inversion not active ✗'}\n")

        if self.env.teleport_map:
            src = next(iter(self.env.teleport_map))
            dst = self.env.teleport_map[src]
            print(f"Teleport @ {src} -> {dst}")
            self.env.confused_turns_left = 0
            result = self.step_onto(src, avoid=self.env.death_pits | self.env.confusion_pads)
            print(f"  {result}")
            print(f"  landed={self.env.agent_pos}  expected={dst}"
                  f"  {'ok ✓' if self.env.agent_pos == dst else 'mismatch ✗'}\n")

        stats = self.env.get_episode_stats()
        print(f"{sep}\n  EPISODE STATS")
        for k, v in stats.items(): print(f"  {k:<20}: {v}")
        print(sep)


if __name__ == "__main__":
    MAZE_PATH = "maze-alpha/MAZE_1.png"
    env = MazeEnvironment(MAZE_PATH)
    env.reset()
    DemoAgent(env).run_demo()

    print("\nVisualizing fire pit rotation...")
    env.reset()
    visualize_fire_pits(env, "fire_before.png", MAZE_PATH)
    env.step([Action.WAIT])
    visualize_fire_pits(env, "fire_after.png", MAZE_PATH)
