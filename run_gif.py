"""
run_gif.py — Run the maze agent and save one animated GIF per episode.

Usage:
    python run_gif.py [MAZE_PATH] [NUM_EPISODES] [OUTPUT_DIR] [FPS] [CAPTURE_EVERY] [MAX_TURNS]

Args (positional, all optional):
    MAZE_PATH      Path to maze PNG          (default: maze-alpha/MAZE_1.png)
    NUM_EPISODES   Number of episodes        (default: 3)
    OUTPUT_DIR     Folder for output GIFs    (default: gifs)
    FPS            Playback speed in GIF     (default: 8)
    CAPTURE_EVERY  Record every Nth turn     (default: 1 — every turn)
                   Increase to shrink file size for long episodes.
    MAX_TURNS      Turns before timeout      (default: 10000)

Output:
    OUTPUT_DIR/<maze_name>_ep01.gif, _ep02.gif, ...
"""
import sys
import os
from hazardDemo import MazeEnvironment
from agent import MazeAgent
from visualize import GifRenderer

MAZE_PATH     = sys.argv[1] if len(sys.argv) > 1 else "maze-alpha/MAZE_1.png"
NUM_EPISODES  = int(sys.argv[2]) if len(sys.argv) > 2 else 3
OUTPUT_DIR    = sys.argv[3] if len(sys.argv) > 3 else "gifs"
FPS           = int(sys.argv[4]) if len(sys.argv) > 4 else 8
CAPTURE_EVERY = int(sys.argv[5]) if len(sys.argv) > 5 else 1
MAX_TURNS     = int(sys.argv[6]) if len(sys.argv) > 6 else 10_000

os.makedirs(OUTPUT_DIR, exist_ok=True)
maze_name = os.path.splitext(os.path.basename(MAZE_PATH))[0]

env = MazeEnvironment(MAZE_PATH, rotate_fire=True)

agent = MazeAgent()
agent.goal_pos   = env.goal_cell
agent.epsilon    = 0.0
agent.known      = {}
agent.wall_edges = set()
agent.open_edges = set()

renderer = GifRenderer(env, capture_every=CAPTURE_EVERY)

print(f"Recording {NUM_EPISODES} episode(s) of '{maze_name}' → {OUTPUT_DIR}/")
print(f"  fps={FPS}  capture_every={CAPTURE_EVERY}  max_turns={MAX_TURNS}\n")

for ep in range(NUM_EPISODES):
    pos = env.reset()
    agent.current_pos = pos
    agent.start_pos   = pos
    agent.reset_episode()
    agent.epsilon = 0.0
    last_result   = None
    path          = [pos]

    for turn in range(MAX_TURNS):
        actions     = agent.plan_turn(last_result)
        result      = env.step(actions)
        last_result = result

        if result.is_dead:
            path = [agent.start_pos or result.current_position]
        else:
            for p in result.positions_visited:
                if not path or path[-1] != p:
                    path.append(p)

        renderer.update(
            known       = agent.known,
            current_pos = result.current_position,
            path        = path,
            episode     = ep + 1,
            turn        = env.turn_count,
            goal_pos    = agent.goal_pos,
            start_pos   = agent.start_pos,
        )

        if result.is_goal_reached:
            print(f"Episode {ep + 1}: SUCCESS in {env.turn_count} turns")
            break
    else:
        print(f"Episode {ep + 1}: timeout after {MAX_TURNS} turns")

    gif_path = os.path.join(OUTPUT_DIR, f"{maze_name}_ep{ep + 1:02d}.gif")
    renderer.save_episode_gif(gif_path, fps=FPS)
    renderer.reset_episode()

print(f"\nDone — GIFs saved to ./{OUTPUT_DIR}/")
