"""
live_viz_astar.py — A* + Neuroevolution dashboard
Silent Cartographer: COSC 4368 AI Spring 2026

Run:
    python live_viz_astar.py --maze maze-alpha/MAZE_1.png [--pop 80] [--gens 100]
"""
from __future__ import annotations
import argparse
import sys

# ── Swap the agent module so all live_viz logic uses A* ──────────────────────
import maze_agent_astar
import sys
sys.modules["maze_agent"] = maze_agent_astar   # makes `import maze_agent` resolve to A*

# Now import live_viz — it will import maze_agent which is now maze_agent_astar
import importlib
import live_viz as _lv
importlib.reload(_lv)   # re-evaluate with the swapped module


def main():
    p = argparse.ArgumentParser(description="Silent Cartographer — A*")
    p.add_argument("--maze",  default="maze-alpha/MAZE_1.png")
    p.add_argument("--turns", type=int,   default=10_000)
    p.add_argument("--test",  action="store_true")
    p.add_argument("--weights",       default="best_weights_astar.npy")
    p.add_argument("--test_episodes", type=int, default=10)
    p.add_argument("--pop",      type=int,   default=80)
    p.add_argument("--gens",     type=int,   default=100)
    p.add_argument("--sigma",    type=float, default=0.20)
    p.add_argument("--decay",    type=float, default=0.995)
    p.add_argument("--eps_eval", type=int,   default=3)
    p.add_argument("--persist",  action="store_true")
    p.add_argument("--phase_k",  type=int,   default=3)
    p.add_argument("--run_id",   default=None)
    args = p.parse_args()

    # Override weights default for A* run so they don't clash with D* Lite weights
    if args.run_id is None:
        from datetime import datetime
        args.run_id = f"astar_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    _lv.build_and_run(args, navigator_name="A*")

if __name__ == "__main__":
    main()