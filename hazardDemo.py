from enum import Enum
from typing import List, Tuple, Optional
from maze import MazeLoader


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

        # convert pixel start/goal to cell coords
        sp = self.loader.start_pos # (pixel_y, pixel_x)
        gp = self.loader.goal_pos
        self.start_cell = self.loader.pixel_to_cell(sp[0], sp[1])
        self.goal_cell = self.loader.pixel_to_cell(gp[0], gp[1])

        # hazard sets cell coords as tuples for fast lookup
        self.death_pits    = set(map(tuple, self.loader.death_pits))
        self.confusion_pads = set(map(tuple, self.loader.confusion_pads))

        # hazard cells contain colored emojis whose dark pixels make them appear as
        # walls in the grayscale maze_array so force all detected hazard cells passable
        all_hazard_cells = (
            self.death_pits
            | self.confusion_pads
            | set(map(tuple, self.loader.teleport_purple))
            | set(map(tuple, self.loader.teleport_orange))
            | set(map(tuple, self.loader.teleport_green))
        )
        for r, c in all_hazard_cells:
            self.grid[r][c] = True

        # teleport pads group by color pair[0] -> pair[1] destination
        self.teleport_map: dict = {}
        for group in [self.loader.teleport_purple, self.loader.teleport_orange, self.loader.teleport_green]:
            pads = [tuple(p) for p in group]
            for i, pad in enumerate(pads):
                dest = pads[(i + 1) % len(pads)] if len(pads) > 1 else pad
                self.teleport_map[pad] = dest

        # episode state
        self.agent_pos: Tuple[int, int] = self.start_cell
        self.turn_count: int  = 0
        self.death_count: int  = 0
        self.confused_count: int  = 0
        self.confused_turns_left: int = 0   # turns of confusion remaining
        self.cells_explored: set  = set()
        self.episode_active: bool = True

    def reset(self) -> Tuple[int, int]: # reset episode and return to start
        self.agent_pos = self.start_cell
        self.turn_count = 0
        self.death_count = 0
        self.confused_count = 0
        self.confused_turns_left = 0
        self.cells_explored = set()
        self.episode_active = True
        return self.agent_pos

    def _is_passable(self, row: int, col: int) -> bool: # check if cell is within bounds and not a wall
        h = self.loader.maze_height_cells
        w = self.loader.maze_width_cells
        if not (0 <= row < h and 0 <= col < w):
            return False
        return self.grid[row][col]

    def step(self, actions: List[Action]) -> TurnResult: # execute 5 actions sequentially
        if not actions or len(actions) > 5:
            raise ValueError("Must submit 1-5 actions per turn.")

        result = TurnResult()
        result.current_position = self.agent_pos

        # confusion carries over from previous turn
        currently_confused = self.confused_turns_left > 0

        for i, raw_action in enumerate(actions):
            # apply confusion if active
            action = self.INVERT[raw_action] if currently_confused else raw_action

            if action == Action.WAIT:
                result.actions_executed += 1
                continue

            dr, dc = self.DELTAS[action]
            r, c = self.agent_pos
            nr, nc = r + dr, c + dc

            # check wall
            if not self._is_passable(nr, nc):
                result.wall_hits += 1
                result.actions_executed += 1
                continue

            # move agent
            self.agent_pos = (nr, nc)
            self.cells_explored.add(self.agent_pos)
            result.actions_executed += 1

            # teleport check
            if self.agent_pos in self.teleport_map:
                dest = self.teleport_map[self.agent_pos]
                self.agent_pos = dest
                result.teleported = True
                result.current_position = self.agent_pos
                # continue executing remaining actions from new position

            # confusion check
            if self.agent_pos in self.confusion_pads:
                result.is_confused = True
                self.confused_turns_left = 2 # rest of this turn + next full turn
                self.confused_count += 1
                currently_confused = True # affects remaining actions this turn

            # death pit check
            if self.agent_pos in self.death_pits:
                result.is_dead = True
                self.death_count += 1
                result.current_position = self.agent_pos # position of pit
                self.agent_pos = self.start_cell # respawn
                break # remaining actions ignored

            # goal check
            if self.agent_pos == self.goal_cell:
                result.is_goal_reached = True
                result.current_position = self.agent_pos
                self.episode_active = False
                break

        result.current_position = self.agent_pos

        # decrement confusion counter at end of turn
        if self.confused_turns_left > 0:
            self.confused_turns_left -= 1

        self.turn_count += 1
        return result

    def get_episode_stats(self) -> dict:
        return {
            "turns_taken": self.turn_count,
            "deaths": self.death_count,
            "confused": self.confused_count,
            "cells_explored": len(self.cells_explored),
            "goal_reached": not self.episode_active,
        }


# scripted to try each hazard
class DemoAgent:
    def __init__(self, env: MazeEnvironment):
        self.env = env

    # BFS pathfinding to target cell optionally avoiding certain cells
    def path_to(self, target: Tuple[int, int], avoid: set = None) -> List[Action]: 
        from collections import deque
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
                if (nxt not in visited
                        and nxt not in avoid
                        and self.env._is_passable(nxt[0], nxt[1])):
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

    # walk through sequence of actions and print each turnresult
    def walk(self, actions: List[Action], label: str = ""):
        last = None
        for i in range(0, len(actions), 5):
            result = self.env.step(actions[i:i+5])
            last = result
            if label:
                print(f"  [{label}] {result}")
            if result.is_dead or result.is_goal_reached:
                return result
        return last

    # navigate to target avoiding specified cells
    def walk_to(self, target: Tuple[int, int], label: str = "", avoid: set = None):
        actions = self.path_to(target, avoid=avoid)
        if not actions:
            return
        return self.walk(actions, label)

    # navigate to cell adjacent to target w the same optional avoidance
    def step_onto(self, target: Tuple[int, int], avoid: set = None) -> TurnResult:
        r, c = target
        candidates = [
            ((r - 1, c), Action.MOVE_DOWN),
            ((r + 1, c), Action.MOVE_UP),
            ((r, c - 1), Action.MOVE_RIGHT),
            ((r, c + 1), Action.MOVE_LEFT),
        ]
        
        for adj, step_action in candidates:
            if not self.env._is_passable(adj[0], adj[1]):
                continue
            if self.env.agent_pos != adj:
                self.walk_to(adj, avoid=avoid)
            # only step if navigation succeeded
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
            pit = next(iter(self.env.death_pits))
            print(f"Death Pit @ cell {pit}")
            print(f"  Expected : is_dead=True, agent respawns at start")
            print(f"  Before   : {self.env.agent_pos}")
            
            # avoid other hazards during navigation
            result = self.step_onto(pit, avoid=all_hazards - {pit})
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
            pad = next(iter(self.env.confusion_pads))
            print(f"Confusion Pad @ cell {pad}")
            print(f"  Expected : is_confused=True, MOVE_UP executes as MOVE_DOWN")
            # clear any leftover confusion from navigation
            self.env.confused_turns_left = 0
            result = self.step_onto(pad, avoid=all_hazards - {pad})
            # ensure confusion is registered even if pad was hit mid navigation
            if result.is_confused or self.env.confused_turns_left > 0:
                self.env.confused_turns_left = max(self.env.confused_turns_left, 1)
                confused_active = True
            else:
                confused_active = False
                
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
        else:
            print("No confusion pads detected.\n")

        # teleport demo
        if self.env.teleport_map:
            src = next(iter(self.env.teleport_map))
            dst = self.env.teleport_map[src]
            print(f"Teleport Pad @ cell {src} -> dest {dst}")
            print(f"  Expected : teleported=True, position jumps to {dst}")
            # clear confusion so navigation is reliable
            self.env.confused_turns_left = 0
            result = self.step_onto(src, avoid=self.env.death_pits | self.env.confusion_pads)
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
    print("Loading maze with hazards")
    env = MazeEnvironment("MAZE_1.png")
    env.reset()

    agent = DemoAgent(env)
    agent.run_demo()