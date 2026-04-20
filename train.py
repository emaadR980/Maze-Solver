from hazardDemo import MazeEnvironment
from agent import MazeAgent

MAX_EPISODES = 500
MAX_TURNS    = 10_000
MAZE_PATH    = "maze-alpha/MAZE_1.png"

env   = MazeEnvironment(MAZE_PATH)
agent = MazeAgent()
agent.env = env

successes = 0

for ep in range(MAX_EPISODES):
    pos = env.reset()
    agent.current_pos = pos
    agent.start_pos   = pos
    agent.goal_pos    = env.goal_cell
    agent.reset_episode()
    last_result = None

    for turn in range(MAX_TURNS):
        old_pos   = agent.current_pos
        old_state = agent.state()

        actions = agent.plan_turn(last_result)
        # debug to see why getting timeouts over and over
        if ep == 2 and turn % 1000 == 0:
            frontier = agent._frontier_target()
            plan = agent._make_plan()
            tp_pads = len(agent.env.teleport_map) if agent.env else 0
            known_tp = sum(1 for v in agent.known.values() if v == 'teleport')
            unlabeled = [pad for pad in agent.env.teleport_map if agent.known.get(pad) != 'teleport']
            print(f"  [dbg] unlabeled tp pads: {unlabeled}")
            print(f"  [dbg] teleport_map: {agent.env.teleport_map}")
            print(f"  [dbg t={turn}] pos={agent.current_pos} "
                  f"frontier={frontier} plan={len(plan)} "
                  f"stag={agent._stagnation_turns} confused={agent.confused_turns_left} "
                  f"wall_edges={len(agent.wall_edges)//2} known={len(agent.known)}"
                  f"known_tp={known_tp}/{tp_pads}")
        result  = env.step(actions)

        reward    = agent.compute_reward(result, old_pos)
        new_state = agent.state()
        agent.update_q(old_state, actions[0], reward, new_state)

        last_result = result

        if result.is_goal_reached:
            agent.goal_pos = env.goal_cell
            successes += 1
            print(f"Ep {ep+1:>3}: SUCCESS in {turn+1:>5} turns | "
                  f"ε={agent.epsilon:.3f} | "
                  f"known={len(agent.known)} cells | "
                  f"cumulative wins={successes}")
            break
        if turn == MAX_TURNS - 1:
            print(f"Ep {ep+1:>3}: timeout          | "
                  f"ε={agent.epsilon:.3f} | "
                  f"known={len(agent.known)} cells")

agent.save("agent.pkl")
print(f"\nTraining complete.  Total successes: {successes}/{MAX_EPISODES}")
