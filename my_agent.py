import os
import numpy as np
from typing import List, Tuple, Optional, Dict, Set
from hazardDemo import Action, TurnResult, MazeEnvironment

# Spec §2.1.1: maze is always 64×64 cells
GRID_H, GRID_W = 64, 64
SAVE_PATH = "my_agent_qtable.npy"

# Action → (row_delta, col_delta)
DELTAS: Dict[Action, Tuple[int, int]] = {
    Action.MOVE_UP:    (-1,  0),
    Action.MOVE_DOWN:  ( 1,  0),
    Action.MOVE_LEFT:  ( 0, -1),
    Action.MOVE_RIGHT: ( 0,  1),
}


class MyAgent:
    """
    Q-Learning maze agent — spec-compliant implementation.

    Implements the required Agent interface (§6.1):
      • self.memory  – dict holding all persistent knowledge
      • plan_turn()  – returns 1-5 actions
      • reset_episode() – resets per-episode state, keeps learning

    Learning strategy:
      • Tabular Q-learning with epsilon-greedy exploration
      • Epsilon decays per-turn across all episodes (global decay)
      • Walls, teleport routes discovered through play and remembered
      • Q-table saved to disk so learning survives between Python sessions
      • Confusion compensation: pre-inverts actions when confused so the
        environment's inversion cancels out (net = intended direction)
    """

    ACTIONS = [Action.MOVE_UP, Action.MOVE_DOWN, Action.MOVE_LEFT, Action.MOVE_RIGHT]

    INVERT: Dict[Action, Action] = {
        Action.MOVE_UP:    Action.MOVE_DOWN,
        Action.MOVE_DOWN:  Action.MOVE_UP,
        Action.MOVE_LEFT:  Action.MOVE_RIGHT,
        Action.MOVE_RIGHT: Action.MOVE_LEFT,
    }

    # ── Reward shaping ──────────────────────────────────────────────────────
    REWARD_GOAL  =  100.0   # reached the goal exit
    REWARD_DEATH =  -50.0   # stepped on a death pit
    REWARD_WALL  =   -1.0   # bumped into a wall
    REWARD_STEP  =   -0.1   # small penalty per step (encourages shorter paths)

    # ── Q-learning hyperparameters ───────────────────────────────────────────
    ALPHA         = 0.20    # learning rate
    GAMMA         = 0.99    # discount factor
    EPSILON_START = 1.00    # start fully random
    EPSILON_MIN   = 0.05    # never go below 5% exploration
    EPSILON_DECAY = 0.9999  # per-turn multiplier (~30 k turns to reach min)

    # ────────────────────────────────────────────────────────────────────────

    def __init__(self, env: MazeEnvironment = None):
        """
        env is optional so the instructor's Evaluator can call MyAgent()
        with no arguments (spec §6.1). When provided, maze dimensions and
        start/goal cells are taken directly from the environment.
        """

        # ── Required by spec §6.1 ─────────────────────────────────────────
        # Evaluator and Visualizer access agent.memory to inspect state.
        self.memory: dict = {}

        # ── Maze dimensions ───────────────────────────────────────────────
        # Spec §2.1.1 guarantees 64×64; use env values if available.
        self.h: int = env.loader.maze_height_cells if env else GRID_H
        self.w: int = env.loader.maze_width_cells  if env else GRID_W

        # ── Known positions ───────────────────────────────────────────────
        # Learned from env if given; otherwise inferred during first episode.
        self.start_cell: Optional[Tuple[int, int]] = env.start_cell if env else None
        self.goal_cell:  Optional[Tuple[int, int]] = env.goal_cell  if env else None

        # ── Q-table (persistent across episodes AND Python sessions) ──────
        self.q_table: np.ndarray = np.zeros((self.h, self.w, len(self.ACTIONS)))
        self._load_qtable()

        # ── Exploration rate (persists across episodes) ────────────────────
        self.epsilon: float = self.EPSILON_START

        # ── Discovered hazard/structure knowledge (persistent) ────────────
        # Walls: cells the agent tried to enter and was blocked.
        self.walls: Set[Tuple[int, int]] = set()

        # Teleport map: step-on cell → arrival cell (learned from play).
        self.teleport_map: Dict[Tuple[int, int], Tuple[int, int]] = {}

        # ── Per-episode state ─────────────────────────────────────────────
        self._init_episode()

    # ──────────────────────────────────────────────────────────────────────
    # Lifecycle
    # ──────────────────────────────────────────────────────────────────────

    def _init_episode(self):
        """Reset variables that belong to a single episode."""
        self.current_pos: Optional[Tuple[int, int]] = self.start_cell

        # Confusion flag: True means next submitted actions must be pre-inverted
        self.confused_next_turn: bool = False

        # Stored for Q-update on the following call to plan_turn
        self.last_state:      Optional[Tuple[int, int]] = None
        self.last_action_idx: Optional[int]             = None

        # Used by run_agent.py's simulate_steps() for path visualisation
        self.last_intended_actions: List[Action] = []

        # Flag to help infer start_cell on the first result when env=None
        self._need_start_inference: bool = (self.start_cell is None)

    def reset_episode(self):
        """
        Called at the start of every new episode (spec §6.1).
        Resets per-episode counters. Q-table, walls, teleport_map, and
        epsilon are intentionally kept so each episode builds on prior work.
        """
        self._init_episode()
        self._sync_memory()

    # ──────────────────────────────────────────────────────────────────────
    # Persistence
    # ──────────────────────────────────────────────────────────────────────

    def _load_qtable(self):
        """Restore Q-table from disk if a previous session saved one."""
        if os.path.exists(SAVE_PATH):
            try:
                saved = np.load(SAVE_PATH)
                # Only load if dimensions match (handles maze changes)
                if saved.shape == self.q_table.shape:
                    self.q_table = saved
            except Exception:
                pass  # corrupt file — start fresh

    def save(self):
        """Persist Q-table to disk. Call this at the end of training."""
        np.save(SAVE_PATH, self.q_table)

    # ──────────────────────────────────────────────────────────────────────
    # Memory dict (spec §6.1 / Visualizer §6.3)
    # ──────────────────────────────────────────────────────────────────────

    def _sync_memory(self):
        """Keep self.memory up to date with agent's current knowledge."""
        self.memory = {
            "q_table":      self.q_table,
            "walls":        self.walls,
            "teleport_map": self.teleport_map,
            "start_cell":   self.start_cell,
            "goal_cell":    self.goal_cell,
            "epsilon":      self.epsilon,
            "current_pos":  self.current_pos,
        }

    # ──────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────

    def in_bounds(self, cell: Tuple[int, int]) -> bool:
        r, c = cell
        return 0 <= r < self.h and 0 <= c < self.w

    def neighbor(self, pos: Tuple[int, int], action: Action) -> Tuple[int, int]:
        """Cell reached by taking action from pos (used by run_agent.py)."""
        dr, dc = DELTAS.get(action, (0, 0))
        return (pos[0] + dr, pos[1] + dc)

    # ──────────────────────────────────────────────────────────────────────
    # Q-learning core
    # ──────────────────────────────────────────────────────────────────────

    def _choose_action(self) -> int:
        """Epsilon-greedy action selection."""
        if np.random.random() < self.epsilon:
            return np.random.randint(0, len(self.ACTIONS))
        r, c = self.current_pos
        return int(np.argmax(self.q_table[r, c]))

    def _update_q(self,
                  state:      Tuple[int, int],
                  action_idx: int,
                  reward:     float,
                  next_state: Tuple[int, int],
                  done:       bool) -> None:
        """
        Bellman equation update:
            Q(s,a) ← Q(s,a) + α [ r + γ·max Q(s',a') − Q(s,a) ]
        """
        r,  c  = state
        nr, nc = next_state
        current_q = self.q_table[r, c, action_idx]
        target = reward if done else reward + self.GAMMA * float(np.max(self.q_table[nr, nc]))
        self.q_table[r, c, action_idx] += self.ALPHA * (target - current_q)

    # ──────────────────────────────────────────────────────────────────────
    # Start-cell inference (used when env=None, i.e. evaluator mode)
    # ──────────────────────────────────────────────────────────────────────

    def _try_infer_start(self, result: TurnResult) -> None:
        """
        Infer start_cell from the very first action+result pair.
        Called only when start_cell is still unknown (env not provided).

        Cases:
          wall hit  → agent stayed at start → start = result.current_position
          moved ok  → start = result.current_position - delta(action taken)
          died      → start still unknown (we're back there but don't know coords)
        """
        if not self._need_start_inference:
            return
        if self.last_action_idx is None:
            return  # no action recorded yet

        action = self.ACTIONS[self.last_action_idx]

        if result.wall_hits > 0:
            # Agent didn't move; current_position IS start
            self.start_cell  = result.current_position
            self.current_pos = self.start_cell
            self._need_start_inference = False

        elif not result.is_dead and not result.teleported:
            # Moved cleanly; back-calculate start from arrival position
            dr, dc = DELTAS[action]
            pos = result.current_position
            self.start_cell  = (pos[0] - dr, pos[1] - dc)
            self._need_start_inference = False
        # If dead on first step: start still unknown; leave flag True to retry

    # ──────────────────────────────────────────────────────────────────────
    # Main planning method (spec §6.1)
    # ──────────────────────────────────────────────────────────────────────

    def plan_turn(self, last_result: Optional[TurnResult]) -> List[Action]:
        """
        Process the previous turn's result, update Q-table and internal maps,
        then choose and return the next action.

        Args:
            last_result: TurnResult from env.step(), or None on the very first
                         turn of the first episode.

        Returns:
            List containing exactly 1 Action (1-5 are allowed by spec;
            we use 1 to keep each MDP step clean for Q-learning).
        """

        if last_result is None:
            # ── Very first turn ────────────────────────────────────────────
            # Agent is at start but may not know coordinates if env wasn't given.
            if self.start_cell is not None:
                self.current_pos = self.start_cell
            # else: current_pos stays None until _try_infer_start() resolves it

        else:
            # ── Try to infer start_cell if still unknown ───────────────────
            self._try_infer_start(last_result)

            # ── Assign reward ──────────────────────────────────────────────
            if last_result.is_goal_reached:
                reward = self.REWARD_GOAL
            elif last_result.is_dead:
                reward = self.REWARD_DEATH
            elif last_result.wall_hits > 0:
                reward = self.REWARD_WALL
            else:
                reward = self.REWARD_STEP

            # ── Update wall map ────────────────────────────────────────────
            # If a wall was hit, the cell in the intended direction is a wall.
            if (last_result.wall_hits > 0
                    and self.last_action_idx is not None
                    and self.current_pos is not None):
                wall_cell = self.neighbor(self.current_pos, self.ACTIONS[self.last_action_idx])
                if self.in_bounds(wall_cell):
                    self.walls.add(wall_cell)

            # ── Update teleport map ────────────────────────────────────────
            if (last_result.teleported
                    and self.last_action_idx is not None
                    and self.current_pos is not None):
                action   = self.ACTIONS[self.last_action_idx]
                tp_src   = self.neighbor(self.current_pos, action)
                tp_dst   = last_result.current_position
                if self.in_bounds(tp_src):
                    self.teleport_map[tp_src] = tp_dst

            # ── Determine next position ────────────────────────────────────
            # Deaths respawn at start; everything else: use reported position.
            if last_result.is_dead:
                next_pos = self.start_cell      # may still be None if unknown
            else:
                next_pos = last_result.current_position

            # ── Q-table Bellman update ─────────────────────────────────────
            # Skip if we don't have valid state coordinates yet.
            if (self.last_state      is not None
                    and self.last_action_idx is not None
                    and next_pos             is not None):
                self._update_q(
                    self.last_state, self.last_action_idx,
                    reward, next_pos,
                    done=last_result.is_goal_reached,
                )

            # ── Advance internal position ──────────────────────────────────
            self.current_pos        = next_pos
            self.confused_next_turn = last_result.is_confused

            # ── Decay exploration rate ─────────────────────────────────────
            self.epsilon = max(self.EPSILON_MIN, self.epsilon * self.EPSILON_DECAY)

        # ── Choose next action ─────────────────────────────────────────────
        if self.current_pos is None:
            # Position still unknown — pick randomly until we learn start
            intended_idx = np.random.randint(0, len(self.ACTIONS))
        else:
            intended_idx = self._choose_action()

        intended = self.ACTIONS[intended_idx]

        # Store for the Q-update on the next call
        self.last_state            = self.current_pos
        self.last_action_idx       = intended_idx
        self.last_intended_actions = [intended]

        # Confusion compensation: pre-invert so env's inversion cancels out
        submitted = self.INVERT[intended] if self.confused_next_turn else intended

        self._sync_memory()
        return [submitted]
