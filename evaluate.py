import sys
from hazardDemo import MazeEnvironment
from agent import MazeAgent

MAX_EPISODES = 5
MAX_TURNS    = 10_000

maze_path  = sys.argv[1] if len(sys.argv) > 1 else "maze-alpha/MAZE_1.png"
maze_label = sys.argv[2] if len(sys.argv) > 2 else maze_path.replace(".png","")
print(f"Evaluating on {maze_path}  (label: {maze_label})")

env   = MazeEnvironment(maze_path)
agent = MazeAgent()
agent.load("agent.pkl")
agent.goal_pos = env.goal_cell
agent.epsilon = 0.0
agent.env = env

agent.known = {}
agent.wall_edges = set()

episode_results = []

for ep in range(MAX_EPISODES):
    pos = env.reset()
    agent.current_pos = pos
    agent.start_pos   = pos
    agent.reset_episode()
    agent.epsilon = 0.0
    last_result  = None
    path_length  = 0

    for turn in range(MAX_TURNS):
        actions = agent.plan_turn(last_result)
        result  = env.step(actions)

        path_length += result.actions_executed
        last_result  = result

        if result.is_goal_reached or turn == MAX_TURNS - 1:
            break

    stats = env.get_episode_stats()
    episode_results.append({
        "episode":        ep + 1,
        "success":        stats["goal_reached"],
        "turns":          stats["turns_taken"],
        "deaths":         stats["deaths"],
        "confused":       stats["confused"],
        "cells_explored": stats["cells_explored"],
        "path_length":    path_length,
    })

    status = "SUCCESS" if stats["goal_reached"] else "FAILED"
    print(f"Episode {ep+1}: {status} | "
          f"turns={stats['turns_taken']} | "
          f"deaths={stats['deaths']} | "
          f"explored={stats['cells_explored']}")

# ── compute final metrics ──────────────────────────────────────────────
print("\n" + "="*45)
print("FINAL EVALUATION METRICS")
print("="*45)

total     = len(episode_results)
successes = [r for r in episode_results if r["success"]]

success_rate = len(successes) / total
print(f"Success Rate        : {success_rate:.1%}  ({len(successes)}/{total})")

if successes:
    avg_path  = sum(r["path_length"] for r in successes) / len(successes)
    avg_turns = sum(r["turns"]       for r in successes) / len(successes)
    print(f"Avg Path Length     : {avg_path:.1f}  (lower is better)")
    print(f"Avg Turns           : {avg_turns:.1f}  (lower is better)")
else:
    avg_turns = None
    print("Avg Path Length     : N/A (no successes)")
    print("Avg Turns           : N/A (no successes)")

total_deaths = sum(r["deaths"] for r in episode_results)
total_turns  = sum(r["turns"]  for r in episode_results)
death_rate   = total_deaths / total_turns if total_turns > 0 else 0
print(f"Death Rate          : {death_rate:.4f}  (lower is better)")

total_visited = sum(r["path_length"]    for r in episode_results)
total_unique  = sum(r["cells_explored"] for r in episode_results)
explore_eff   = total_unique / total_visited if total_visited > 0 else 0
print(f"Exploration Efficiency: {explore_eff:.3f}  (higher is better, max 1.0)")

navigable = sum(1 for r in range(env.loader.maze_height_cells)
                  for c in range(env.loader.maze_width_cells)
                  if env.grid[r][c])
best_explored    = max(r["cells_explored"] for r in episode_results)
map_completeness = best_explored / navigable if navigable > 0 else 0
print(f"Map Completeness    : {map_completeness:.1%}  (best single episode)")

total_confused = sum(r["confused"] for r in episode_results)
print(f"Times Confused      : {total_confused}")

print("\n" + "="*55)
print("BENCHMARK COMPARISON")
print("="*55)
print(f"{'Metric':<25} {'Baseline':>10} {'You':>10} {'Target':>10}")
print("-"*55)
you_turns = f"{avg_turns:.0f}" if avg_turns is not None else "N/A"
print(f"{'Success Rate':<25} {'~5%':>10} {f'{success_rate:.0%}':>10} {'>80%':>10}")
print(f"{'Avg Turns':<25} {'~8000':>10} {you_turns:>10} {'<1000':>10}")
print(f"{'Death Rate':<25} {'~0.15':>10} {f'{death_rate:.3f}':>10} {'<0.05':>10}")
