from PIL import Image, ImageDraw
from hazardDemo import MazeEnvironment
from my_agent import MyAgent

MAX_TURNS    = 10_000   # spec §5.4
MAX_EPISODES = 5        # spec §5.4 (test phase limit)
MAZE_PATH    = "MAZE_1.png"
CELL         = 16       # pixels per cell


def simulate_steps(agent: MyAgent, prev_pos, result) -> list:
    """
    Reconstruct every cell the agent physically occupied during one turn.

    Single-action turns: read result flags directly.
    Multi-action turns:  simulate step-by-step using agent's known walls
                         and teleport_map so the path stays accurate.
    """
    intended = agent.last_intended_actions
    if not intended:
        return [result.current_position]

    # ── Single-action turn ─────────────────────────────────────────────────
    if len(intended) == 1:
        if result.wall_hits > 0:
            return [prev_pos]           # blocked — agent didn't move
        return [result.current_position]

    # ── Multi-action turn ──────────────────────────────────────────────────
    cells = []
    pos   = prev_pos

    for action in intended:
        nxt = agent.neighbor(pos, action)

        if nxt in agent.walls or not agent.in_bounds(nxt):
            cells.append(pos)           # wall hit — stay
            continue

        pos = nxt

        if pos in agent.teleport_map:   # apply known teleport
            pos = agent.teleport_map[pos]

        cells.append(pos)

        if pos == result.current_position:
            break                       # reached the reported final position

    # Guard: if simulation diverged (unknown teleport, rotating pit), fix end
    if not cells or cells[-1] != result.current_position:
        cells.append(result.current_position)

    return cells


def save_visualization(env: MazeEnvironment, path_cells: list, output_path: str):
    """
    Render the agent's episode onto the maze image.

    Colours:
      Dark (near-black) = unexplored corridor cells
      Original maze     = explored cells (no overlay)
      Green line        = exact sequence of moves, connected in order
      Cyan circle       = start cell
      Red  circle       = goal cell
    """
    img  = Image.open(MAZE_PATH).convert("RGB")
    draw = ImageDraw.Draw(img)

    explored = set(path_cells)

    # ── 1. Darken unexplored cells; leave explored cells as the original maze
    for r in range(env.loader.maze_height_cells):
        for c in range(env.loader.maze_width_cells):
            if not env.grid[r][c]:
                continue                         # real wall — leave black
            if (r, c) not in explored:
                x0 = c * CELL + 1
                y0 = r * CELL + 1
                x1 = c * CELL + CELL - 1
                y1 = r * CELL + CELL - 1
                draw.rectangle([x0, y0, x1, y1], fill=(15, 15, 15))

    # ── 2. Green line — only draw where a real open passage exists ──────────
    # The environment checks only the destination cell's centre pixel, so it
    # can let the agent cross visual wall lines.  We check the BOUNDARY pixel
    # between two adjacent cells before drawing the segment: if that pixel is
    # black (wall line) we skip the segment so the line stays inside corridors.
    maze_array = env.loader.maze_array   # True = open, False = wall

    for i in range(len(path_cells) - 1):
        r0, c0 = path_cells[i]
        r1, c1 = path_cells[i + 1]

        if abs(r1 - r0) + abs(c1 - c0) != 1:
            continue                                    # teleport/respawn — skip

        # Boundary pixel is at the shared edge between the two cells
        if r0 == r1:                                   # horizontal move
            bnd_y = r0 * CELL + CELL // 2
            bnd_x = max(c0, c1) * CELL                 # column boundary
        else:                                          # vertical move
            bnd_y = max(r0, r1) * CELL                 # row boundary
            bnd_x = c0 * CELL + CELL // 2

        # Clamp to image dimensions
        bnd_y = min(bnd_y, maze_array.shape[0] - 1)
        bnd_x = min(bnd_x, maze_array.shape[1] - 1)

        if not maze_array[bnd_y, bnd_x]:
            continue                                    # wall pixel — skip segment

        draw.line(
            [(c0 * CELL + CELL // 2, r0 * CELL + CELL // 2),
             (c1 * CELL + CELL // 2, r1 * CELL + CELL // 2)],
            fill=(0, 220, 80), width=2,
        )

    # ── 3. Start (cyan) and goal (red) markers ────────────────────────────
    for cell, color in [
        (env.start_cell, (0, 240, 255)),
        (env.goal_cell,  (255, 60,  60)),
    ]:
        if cell:
            r, c = cell
            px = c * CELL + CELL // 2
            py = r * CELL + CELL // 2
            draw.ellipse([px - 5, py - 5, px + 5, py + 5],
                         fill=color, outline=(0, 0, 0), width=1)

    img.save(output_path)
    print(f"  Visualization saved → {output_path}  ({len(path_cells)} steps)")


def run_episode(env: MazeEnvironment, agent: MyAgent, episode: int):
    start = env.reset()
    agent.reset_episode()

    print(f"\n── Episode {episode + 1} ──  start={start}  goal={env.goal_cell}")
    last_result = None
    path_cells  = [env.start_cell]

    for turn in range(MAX_TURNS):
        prev_pos = agent.current_pos
        actions  = agent.plan_turn(last_result)
        result   = env.step(actions)

        step_cells = simulate_steps(agent, prev_pos, result)
        path_cells.extend(step_cells)

        print(f"  Turn {turn + 1:04d} | {result}")
        last_result = result

        if result.is_goal_reached:
            print(f"  Goal reached on turn {turn + 1}!")
            break
    else:
        print(f"  Timeout after {MAX_TURNS} turns.")

    stats = env.get_episode_stats()
    return stats, path_cells


def main():
    env   = MazeEnvironment(MAZE_PATH)
    agent = MyAgent(env)               # pass env so start/goal are known immediately

    all_stats = []

    for ep in range(MAX_EPISODES):
        stats, path = run_episode(env, agent, ep)
        all_stats.append(stats)

        print("\n  Episode stats:")
        for k, v in stats.items():
            print(f"    {k:<22}: {v}")

        save_visualization(env, path, f"agent_path_ep{ep + 1}.png")

    # Save Q-table so the next session starts from learned knowledge
    agent.save()
    print(f"\n  Q-table saved → my_agent_qtable.npy")

    # ── Summary ───────────────────────────────────────────────────────────
    successes = sum(1 for s in all_stats if s["goal_reached"])
    print(f"\n{'='*45}")
    print(f"  Success rate : {successes}/{MAX_EPISODES} episodes")
    if successes:
        winning = [s for s in all_stats if s["goal_reached"]]
        print(f"  Avg turns    : {sum(s['turns_taken'] for s in winning) / successes:.1f}")
        print(f"  Avg deaths   : {sum(s['deaths'] for s in winning) / successes:.1f}")
    total_deaths = sum(s["deaths"] for s in all_stats)
    total_turns  = sum(s["turns_taken"] for s in all_stats)
    print(f"  Death rate   : {total_deaths / max(total_turns, 1):.4f}")
    print(f"{'='*45}")

    print("\n── Verification ─────────────────────────────────")
    for i, s in enumerate(all_stats):
        status = "SOLVED" if s["goal_reached"] else "FAILED"
        print(f"  Episode {i+1}: {status:6s} | "
              f"turns={s['turns_taken']:5d} | "
              f"deaths={s['deaths']:3d} | "
              f"confused={s['confused']:3d} | "
              f"cells explored={s['cells_explored']}")
    print(f"\n  agent_path_ep<N>.png: cyan=start  red=goal  blue=explored  green=path")


if __name__ == "__main__":
    main()
