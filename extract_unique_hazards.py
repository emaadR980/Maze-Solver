"""
extract_unique_hazards.py
Saves the single best-scoring example of each hazard label from a maze image.
Useful for visually comparing what each hazard looks like and diagnosing misclassifications.

Usage:
    python extract_unique_hazards.py --maze MAZE_1.png
    python extract_unique_hazards.py --maze MAZE_2.png
"""

import os
import argparse
import numpy as np
import cv2
from maze import MazeLoader, load_templates_from_dir, CvTemplateHazards


LABELS = ["death_pit", "confusion", "teleport_orange", "teleport_green", "teleport_purple"]


def extract(maze_path: str, out_dir: str, detect_threshold: float):
    os.makedirs(out_dir, exist_ok=True)
    print(f"Output directory: {os.path.abspath(out_dir)}")
    print(f"Loading maze: {maze_path}")

    loader = MazeLoader(maze_path, template_threshold=detect_threshold)
    size   = (loader.INNER_SIZE, loader.INNER_SIZE)

    templates_by_label = load_templates_from_dir("templates")
    matcher = CvTemplateHazards(templates_by_label, size=size)

    h = loader.maze_height_cells
    w = loader.maze_width_cells

    # best: label -> (score, r, c, patch)
    best = {lab: (-1.0, -1, -1, None) for lab in LABELS}

    print(f"Scanning {h}x{w} cells...")
    for r in range(h):
        for c in range(w):
            patch = loader.cell_interior_bgr(r, c)
            _, _, scores = matcher.classify(patch)
            for label in LABELS:
                s = scores.get(label, -1.0)
                if s > best[label][0]:
                    best[label] = (s, r, c, cv2.resize(patch, size, interpolation=cv2.INTER_NEAREST))

    maze_name = os.path.splitext(os.path.basename(maze_path))[0]

    print("\nBest match per label:")
    for label in LABELS:
        score, r, c, patch = best[label]
        if patch is None or score < detect_threshold:
            print(f"  {label}: nothing found above threshold")
            continue
        fname = f"{maze_name}_{label}_best_r{r}_c{c}_s{score:.2f}.png"
        fpath = os.path.join(out_dir, fname)
        cv2.imwrite(fpath, patch)
        print(f"  {label}: score={score:.3f} at ({r},{c}) -> {fname}")

    print(f"\nSaved to '{os.path.abspath(out_dir)}'")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--maze",      default="MAZE_1.png")
    p.add_argument("--out",       default="unique_hazards")
    p.add_argument("--threshold", type=float, default=0.25)
    args = p.parse_args()
    extract(args.maze, args.out, args.threshold)


if __name__ == "__main__":
    main()