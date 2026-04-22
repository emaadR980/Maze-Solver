
"""
rotation_debug.py
Diagnose fire cluster rotation — find apex pivot, show all 4 states zoomed.
Run standalone: python rotation_debug.py
"""
import numpy as np
import matplotlib.pyplot as plt
from environment import MazeEnvironment

MAZE = "maze-alpha/MAZE_1.png"
env  = MazeEnvironment(MAZE)

print(f"Total fire cells : {len(env.death_pits)}")
print(f"Clusters (gap=3): {len(env.initial_fire_clusters)}")


# ─────────────────────────────────────────────────────────────────────────────
# Apex finder — the bend cell has the highest dot-product among 2-neighbour
# cells (arm cells score ~-1 cos, apex scores ~0 cos)
# ─────────────────────────────────────────────────────────────────────────────
def find_pivot(cluster, h=64, w=64):
    """
    Border-aware pivot selection:
      1. If any cell is on the grid border, use the border cell with the most
         in-cluster neighbours — that's the corner where the arm meets the wall.
         Rotating around it swings the free arm *inward*, keeping everything valid.
      2. Otherwise fall back to the geometric apex (highest cos angle).
    """
    cluster_set = set(map(tuple, cluster))

    # ── 1. Border-cell check ────────────────────────────────────────────────
    border_cells = [(r, c) for r, c in cluster
                    if r == 0 or r == h - 1 or c == 0 or c == w - 1]

    if border_cells:
        def neighbour_count(rc):
            r, c = rc
            return sum(
                1 for dr in (-1, 0, 1) for dc in (-1, 0, 1)
                if (dr, dc) != (0, 0) and (r + dr, c + dc) in cluster_set
            )
        return max(border_cells, key=neighbour_count)

    # ── 2. Geometric apex (original logic) ──────────────────────────────────
    if len(cluster) <= 1:
        return cluster[0]

    best_cell, best_score = None, -float('inf')
    for r, c in cluster:
        nbrs = [(r + dr, c + dc)
                for dr in (-1, 0, 1) for dc in (-1, 0, 1)
                if (dr, dc) != (0, 0) and (r + dr, c + dc) in cluster_set]

        if len(nbrs) != 2:
            continue

        (r1, c1), (r2, c2) = nbrs
        d1r, d1c = r1 - r, c1 - c
        d2r, d2c = r2 - r, c2 - c
        dot   = d1r * d2r + d1c * d2c
        denom = ((d1r**2 + d1c**2)**0.5) * ((d2r**2 + d2c**2)**0.5)
        cos_a = dot / denom if denom > 0 else -1.0

        if cos_a > best_score:
            best_score = cos_a
            best_cell  = (r, c)

    if best_cell is None:
        rows = [r for r, _ in cluster]; cols = [c for _, c in cluster]
        cr = sum(rows) / len(rows);     cc = sum(cols) / len(cols)
        best_cell = min(cluster, key=lambda cell: (cell[0]-cr)**2 + (cell[1]-cc)**2)

    return best_cell


# ─────────────────────────────────────────────────────────────────────────────
# Rotation using apex pivot — compute all 4 states fresh
# ─────────────────────────────────────────────────────────────────────────────
def rotate_cluster_90(cluster, origin_cluster, pr, pc, h=64, w=64):
    if len(origin_cluster) <= 1:
        return list(cluster)
    rotated = []
    for r, c in cluster:
        dr = r - pr; dc = c - pc
        nr = pr + dc; nc = pc - dr
        if not (0 <= nr < h and 0 <= nc < w):
            return list(origin_cluster)   # out of bounds — stay
        rotated.append((nr, nc))
    return rotated


def precompute_states(cluster):
    """Return 4 frozensets: state 0=original, 1/2/3 = 90°/180°/270° CW."""
    pivot    = find_pivot(cluster)
    pr, pc   = pivot
    states   = [frozenset(cluster)]
    current  = list(cluster)
    for _ in range(3):
        current = rotate_cluster_90(current, cluster, pr, pc)
        states.append(frozenset(current))
    return states, pivot


# ─────────────────────────────────────────────────────────────────────────────
# Per-cluster text report
# ─────────────────────────────────────────────────────────────────────────────
multi = [c for c in env.initial_fire_clusters if len(c) > 1]
print(f"Multi-cell clusters: {len(multi)}\n")

all_states  = []
all_pivots  = []

for i, cluster in enumerate(multi):
    states, pivot = precompute_states(cluster)
    all_states.append(states)
    all_pivots.append(pivot)
    rows = [r for r, _ in cluster]
    cols = [c for _, c in cluster]
    sizes = [len(s) for s in states]
    print(f"C{i:2d}: {len(cluster)} cells  "
          f"bbox=rows[{min(rows)}-{max(rows)}] cols[{min(cols)}-{max(cols)}]  "
          f"pivot={pivot}  state_sizes={sizes}")


# ─────────────────────────────────────────────────────────────────────────────
# Zoomed figure — each row = one cluster, 4 columns = 4 rotation states
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(len(multi), 4,
                          figsize=(12, max(4, len(multi) * 2)))
if len(multi) == 1:
    axes = [axes]

for ci, (cluster, states, (pr, pc)) in enumerate(
        zip(multi, all_states, all_pivots)):

    # Compute overall bounding box across all 4 states so view is stable
    all_cells = [cell for s in states for cell in s]
    rows_all  = [r for r, _ in all_cells]
    cols_all  = [c for _, c in all_cells]
    pad  = 2
    rmin = max(0,  min(rows_all) - pad)
    rmax = min(63, max(rows_all) + pad)
    cmin = max(0,  min(cols_all) - pad)
    cmax = min(63, max(cols_all) + pad)

    for si in range(4):
        ax          = axes[ci][si]
        state_cells = states[si]

        grid = np.ones((rmax-rmin+1, cmax-cmin+1, 3))
        for r, c in state_cells:
            if rmin <= r <= rmax and cmin <= c <= cmax:
                grid[r-rmin, c-cmin] = [1.0, 0.1, 0.1]

        ax.imshow(grid, interpolation="nearest", origin="upper")

        # Yellow star on pivot (stays fixed across all states)
        if rmin <= pr <= rmax and cmin <= pc <= cmax:
            ax.plot(pc-cmin, pr-rmin, "*",
                    color="yellow", markersize=14,
                    markeredgecolor="black", markeredgewidth=0.8)

        ax.set_title(f"C{ci} S{si} ({len(cluster)}c)", fontsize=7)
        ax.set_xticks([]); ax.set_yticks([])

plt.suptitle("Apex-pivot rotation — yellow★=apex/pivot (fixed), red=fire",
             fontsize=10)
plt.tight_layout()
plt.savefig("rotation_debug.png", dpi=120, bbox_inches="tight")
plt.show()
print("Saved rotation_debug.png")
