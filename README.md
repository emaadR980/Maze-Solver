# Maze-Solver

Maze navigation agent using D\* Lite incremental replanning and a GA-evolved neural network for fire timing. The agent navigates a 64×64 maze blind, with only the goal cell known at start. All hazards (fire pits, confusion pads, teleporters, arrow pads) are discovered through experience. AI was used to aid development during brainstorming, debugging, researching, and providing an initial scaffolding. It was also used to format this README!

---

## Architecture

| Component | Role |
|---|---|
| **D\* Lite** | Primary navigator. Treats unknown cells as free (optimistic), replans incrementally as walls and hazards are discovered. Routes through teleporters once found. |
| **Neural Network** | 43 → 64 → 32 → 5, ReLU + softmax. Called only when D\* Lite's next cell is on fire. Decides wait vs. detour using fire rotation timing. |
| **Genetic Algorithm** | Evolves NN weights. Tournament selection, uniform crossover, Gaussian mutation. Two-phase fitness: explore → optimize. |

---

## Files

| File | Description |
|---|---|
| `maze_agent.py` | Agent, memory, D\* Lite, NN, GA, fitness functions |
| `live_viz.py` | Training dashboard + test runner with PIL visualization |
| `environment.py` | Maze environment — fire rotation, hazards, adjacency graph |
| `maze.py` | Colour-based maze loader (no CNN) |

---

## Training

```bash
python live_viz.py --maze maze-alpha/MAZE_1.png --persist --run_id my_run
```

### Training Parameters

| Flag | Default | Description |
|---|---|---|
| `--maze` | `maze-alpha/MAZE_1.png` | Maze image to train on |
| `--pop` | `80` | Population size |
| `--gens` | `100` | Number of generations |
| `--eps_eval` | `3` | Episodes per agent per generation |
| `--turns` | `10000` | Max turns per episode |
| `--sigma` | `0.20` | Initial mutation sigma |
| `--decay` | `0.995` | Sigma decay per generation |
| `--phase_k` | `3` | Cumulative solvers needed to switch to optimize phase |
| `--persist` | `False` | Carry pit/wall/teleport knowledge across episodes within a generation |
| `--run_id` | timestamp | Name for saved weights files (`weights_<run_id>.npy`, `best_weights_<run_id>.npy`) |
| `--init_weights` | `None` | Warm-start population from existing weights + mutations |

### Recommended Training Commands

**Maze-alpha (standard):**
```bash
python live_viz.py --maze maze-alpha/MAZE_1.png --pop 80 --gens 60 --eps_eval 3 --turns 7000 --persist --run_id dstar_v4_alpha --phase_k 3
```

**Maze-gamma (harder, warm-start from alpha):**
```bash
python live_viz.py --maze maze-gamma/MAZE_1.png --pop 80 --gens 60 --eps_eval 3 --turns 7000 --persist --run_id dstar_v4_gamma --phase_k 1 --init_weights best_weights_dstar_v4_alpha.npy
```

---

## Testing

```bash
python live_viz.py --test --maze maze-beta/MAZE_1.png --weights best_weights_dstar_v4_alpha.npy
```

### Test Parameters

| Flag | Default | Description |
|---|---|---|
| `--test` | `False` | Run in test mode |
| `--maze` | `maze-alpha/MAZE_1.png` | Maze to test on |
| `--weights` | `best_weights.npy` | Path to `.npy` weights file |
| `--test_episodes` | `5` | Number of test episodes |
| `--turns` | `10000` | Max turns per episode |
| `--legacy_pits` | `False` | v3 compat: permanent D\* Lite walls from deaths (use with v3 weights) |

### Test Output

The test window shows a live map with path tracing, skull death markers, fire pit locations, and teleporter/arrow pads. Use **← →** arrow keys to browse episodes after completion.

Metrics reported:

| Metric | Description |
|---|---|
| Success Rate | Episodes where goal was reached |
| Avg Path Length | Mean cells explored on solved episodes |
| Avg Turns to Solution | Mean turns on solved episodes |
| Death Rate | Deaths per episode |
| Exploration Efficiency | Unique cells / total turns |
| Map Completeness | % of reachable cells visited |
| Replanning Efficiency | % of solves with 0 deaths |
| Learning Efficiency | Turn improvement from first to last solve |

---

## Weights Compatibility

| Weights | Mode | Notes |
|---|---|---|
| `best_weights_dstar_v3.npy` | `--legacy_pits` required | Permanent pit walls, always-block mask |
| `best_weights_dstar_v4*.npy` | Default (no flag needed) | Fire-rotation-aware, cluster learning on death |

---

## Hazard Handling

| Hazard | Behavior |
|---|---|
| **Fire pit** | On first death, learns full cluster rotation mask (all 4 states). Permanent cells (pivot points) get D\* Lite walls. Rotating cells are timed precisely — agent crosses when rotation index is clear. |
| **Confusion pad** | Detected via `TurnResult.is_confused`. Agent tracks `confused_turns_left` and inverts all outgoing actions to compensate. |
| **Teleporter** | Detected when position jumps unexpectedly vs. intended action. D\* Lite routes through beneficial teleporters as 1-step shortcuts once source→destination is known. |
| **Arrow pad** | Detected via `TurnResult.arrow_pushed`. Agent records pad cell and direction. D\* Lite assigns cost 2 to known arrow cells (uses full turn, unintended landing). |

## AI prompt Example
**Prompt:** "The fire traps are visually moving off their pivot point. Please diagnose"

**Response:** "The problem is the centroid is recomputed from the already-rotated cluster each frame, so it drifts as cells get clipped at boundaries. Fix: compute the pivot once when clusters arrive and keep it fixed across all rotations."
