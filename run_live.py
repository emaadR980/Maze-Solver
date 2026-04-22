import sys
from hazardDemo import MazeEnvironment
from agent import MazeAgent
from visualize import LiveVisualizer

MAZE_PATH    = sys.argv[1] if len(sys.argv) > 1 else "maze-alpha/MAZE_1.png"
NUM_EPISODES = int(sys.argv[2]) if len(sys.argv) > 2 else 3
UPDATE_EVERY = int(sys.argv[3]) if len(sys.argv) > 3 else 1
MAX_TURNS    = 10_000
DEBUG_HAZARDS = (len(sys.argv) > 5 and sys.argv[5].lower() in {"1", "true", "yes", "debug"})

env   = MazeEnvironment(MAZE_PATH, rotate_fire=True)
agent = MazeAgent()
agent.goal_pos = env.goal_cell
agent.epsilon = 0.0

agent.known = {}
agent.wall_edges = set()
agent.open_edges = set()

STEP_DELAY = float(sys.argv[4]) if len(sys.argv) > 4 else 0.05

if UPDATE_EVERY > 1:
    print("[run_live] Note: UPDATE_EVERY > 1 can make rotating fire visuals look stale.")

viz = LiveVisualizer(
    env          = env,
    title        = f"Maze Agent – {MAZE_PATH}",
    update_every = UPDATE_EVERY,
    delay        = STEP_DELAY,
)

for ep in range(NUM_EPISODES):
    pos = env.reset()
    agent.current_pos = pos
    agent.start_pos   = pos
    agent.reset_episode()
    agent.epsilon = 0.0
    last_result = None
    path = [pos]
    actions_executed_total = 0

    for turn in range(MAX_TURNS):
        actions     = agent.plan_turn(last_result)
        result      = env.step(actions)
        last_result = result
        actions_executed_total += result.actions_executed

        if DEBUG_HAZARDS and (result.teleported or result.is_dead or result.is_confused or result.wall_hits > 0):
            print(
                f"[live] turn={env.turn_count} pos={result.current_position} "
                f"event={result.last_event or '-'} teleported={result.teleported} "
                f"dead={result.is_dead} confused={result.is_confused} wall_hits={result.wall_hits} "
                f"actions={[a.name for a in actions]} visited={result.positions_visited}"
            )

        if result.is_dead:
            path = [agent.start_pos or result.current_position]
        else:
            for pos in result.positions_visited:
                if not path or path[-1] != pos:
                    path.append(pos)

        viz.update(
            known       = agent.known,
            current_pos = result.current_position,
            path        = path,
            episode     = ep + 1,
            turn        = env.turn_count,
            goal_pos    = agent.goal_pos,
            start_pos   = agent.start_pos,
            extra_stats = {
                "pos":        str(result.current_position),
                "turns":      env.turn_count,
                "actions":    actions_executed_total,
                "last batch": len(actions),
                "deaths":     env.death_count,
                "teleports":  env.teleport_count,
                "confused":   env.confused_count,
                "explored":   len(env.cells_explored),
                "ε":          f"{agent.epsilon:.3f}",
                "event":      result.last_event or "—",
            },
        )

        if result.is_goal_reached:
            print(f"Episode {ep+1}: SUCCESS in {env.turn_count} turns "
                  f"({actions_executed_total} actions)")
            break
    else:
        print(f"Episode {ep+1}: timeout after {MAX_TURNS} turns")

    # Pause so you can see the final state, then clear for next episode
    import time; time.sleep(1.5)
    viz.reset_episode()

viz.close()
print("Done.")
