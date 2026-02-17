import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random
from collections import defaultdict
import threading
import time

img_path = 'MAZE_1.png'
img_file = Image.open(img_path)
rgb_array = np.array(img_file)
gray      = np.array(img_file.convert('L'))

IMG_SIZE  = 1026
GRID_SIZE = 64
CELL_SIZE = 16
OFFSET    = 1    # ← The image has a 1px outer border; walls sit at OFFSET + n*CELL_SIZE

print(f"Grid: {GRID_SIZE}x{GRID_SIZE}  |  {CELL_SIZE}px/cell  |  {OFFSET}px border offset")

# ── EDGE-BASED wall detection ─────────────────────────────────────────────────
# Walls are BLACK LINES drawn at pixel boundaries (multiples of CELL_SIZE).
# To check a wall between cells we sample pixels ON the boundary line.

def has_wall_south(r, c):
    """Is there a wall between (r,c) and (r+1,c)?"""
    if r + 1 >= GRID_SIZE: return True
    y        = (r + 1) * CELL_SIZE
    x_center = c * CELL_SIZE + CELL_SIZE // 2
    pixels   = gray[y, max(0, x_center-4):min(IMG_SIZE, x_center+5)]
    return int(pixels.min()) < 128

def has_wall_east(r, c):
    """Is there a wall between (r,c) and (r,c+1)?"""
    if c + 1 >= GRID_SIZE: return True
    x        = (c + 1) * CELL_SIZE
    y_center = r * CELL_SIZE + CELL_SIZE // 2
    pixels   = gray[max(0, y_center-4):min(IMG_SIZE, y_center+5), x]
    return int(pixels.min()) < 128

# Pre-build adjacency list for every cell
print("Building adjacency graph...")
ADJ = [[[] for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
for r in range(GRID_SIZE):
    for c in range(GRID_SIZE):
        if not has_wall_south(r, c):
            ADJ[r][c].append((r+1, c))
            ADJ[r+1][c].append((r, c))
        if not has_wall_east(r, c):
            ADJ[r][c].append((r, c+1))
            ADJ[r][c+1].append((r, c))

total_edges = sum(len(ADJ[r][c]) for r in range(GRID_SIZE) for c in range(GRID_SIZE))
print(f"Total directed edges: {total_edges}  (perfect maze ≈ 8192)")

def cell_nbrs(r, c):
    return ADJ[r][c]

# ── Confirmed positions from pixel analysis ───────────────────────────────────
START_CELL = (0,  31)   # top opening at pixel x=498
GOAL_CELL  = (63, 32)   # bottom opening at pixel x=527

FIRE_CELLS = {
    (2,17),(7,30),(8,7),(9,6),(9,8),(10,5),(10,9),(11,4),(11,10),
    (18,16),(21,34),(22,33),(23,32),(24,31),(25,32),(26,33),(27,34),
    (31,5),(31,11),(32,6),(32,10),(33,7),(33,9),(34,8),
    (39,28),(42,47),(43,48),(44,49),(45,50),(46,49),(47,48),(48,47),
    (58,3),(59,2),(59,55),(60,1),(61,0),(62,1),(63,2)
}

TELEPORT_PAIRS = [
    ((35, 31), (11, 55)),   # green pair
    ((46,  9), (54, 26)),   # purple pair
]

print(f"Start: {START_CELL}  Goal: {GOAL_CELL}")
print(f"Fire cells: {len(FIRE_CELLS)}  |  Teleport pairs: {len(TELEPORT_PAIRS)}")
print(f"Start neighbors: {cell_nbrs(*START_CELL)}")
print(f"Goal  neighbors: {cell_nbrs(*GOAL_CELL)}")

# ── Fire rotation ─────────────────────────────────────────────────────────────
FIRE_LIST = list(FIRE_CELLS)

def get_fire_cells(turn):
    out = set()
    cx = cy = GRID_SIZE // 2
    for r, c in FIRE_LIST:
        dr, dc = r - cx, c - cy
        for _ in range(turn % 4):
            dr, dc = dc, -dr
        nr, nc = cx + dr, cy + dc
        if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE:
            out.add((int(nr), int(nc)))
    return out

# ── Corridor walker ───────────────────────────────────────────────────────────
def walk_corridor(pos, came_from, fire_cells, tp_pairs, goal):
    path, current, prev = [pos], pos, came_from
    for _ in range(GRID_SIZE * GRID_SIZE):
        r, c  = current
        exits = [n for n in cell_nbrs(r, c) if n != prev]

        if len(exits) == 0: return current, path, 'dead_end'
        if len(exits) >  1: return current, path, 'junction'

        nxt = exits[0]
        if nxt in fire_cells: return current, path, 'fire_blocked'

        prev, current = current, nxt
        path.append(current)

        for tp1, tp2 in tp_pairs:
            if current == tp1: return tp2, path + [tp2], 'teleport'
            if current == tp2: return tp1, path + [tp1], 'teleport'

        if current == goal: return current, path, 'goal'

    return current, path, 'timeout'

# ── Agent ─────────────────────────────────────────────────────────────────────
class JunctionAgent:
    S_GREEDY = 0; S_WALL = 1; S_EXPLORE = 2; S_QLEARN = 3

    def __init__(self, start, goal, tp_pairs):
        self.start = start; self.goal = goal; self.tp_pairs = tp_pairs
        self.lock = threading.Lock()
        self._reset_memory(); self._reset_episode()

    def _reset_memory(self):
        self.q_table         = defaultdict(lambda: defaultdict(float))
        self.visit_count     = defaultdict(int)
        self.dead_ends       = set()
        self.danger_zones    = set()
        self.known_cells     = {self.start}
        self.success_paths   = []
        self.turn            = 0; self.episode       = 0
        self.total_steps     = 0; self.success_count = 0
        self.teleport_count  = 0; self.is_training   = True
        self.strategy_scores = {s: 0.25 for s in range(4)}
        self.current_strategy= self.S_GREEDY
        self.strategy_cd     = 30
        self.alpha = 0.15; self.gamma = 0.95; self.epsilon = 0.4

    def _reset_episode(self):
        self.position  = self.start; self.came_from = None
        self.cell_path = [self.start]
        self.best_dist = self._d(self.start)

    def snapshot(self):
        with self.lock:
            return dict(
                position=self.position, cell_path=list(self.cell_path),
                known_cells=set(self.known_cells), dead_ends=set(self.dead_ends),
                episode=self.episode, total_steps=self.total_steps,
                best_dist=self.best_dist, teleport_count=self.teleport_count,
                success_count=self.success_count, epsilon=self.epsilon,
                current_strategy=self.current_strategy,
                strategy_scores=dict(self.strategy_scores),
            )

    def _d(self, p): return abs(p[0]-self.goal[0]) + abs(p[1]-self.goal[1])

    def _score(self, pos):
        s, dist, visits = self.current_strategy, self._d(pos), self.visit_count.get(pos,0)
        novel = pos not in self.visit_count
        if   s == self.S_GREEDY:  score = -dist*10 + (500 if novel else 0)
        elif s == self.S_WALL:    score = -dist*5   - visits*20
        elif s == self.S_EXPLORE: score = (2000 if novel else -visits*100) - dist
        else:
            score = self.q_table[(self.position[0],self.position[1])][(pos[0],pos[1])]*10
            score += (300 if novel else 0) - dist*2
        if pos in self.dead_ends:    score -= 50000
        if pos in self.danger_zones: score -= 10000
        return score

    def _update_q(self, op, np_, r):
        os_, ns_ = (op[0],op[1]), (np_[0],np_[1])
        mq = max(self.q_table[ns_].values()) if self.q_table[ns_] else 0
        oq = self.q_table[os_][ns_]
        self.q_table[os_][ns_] = oq + self.alpha*(r + self.gamma*mq - oq)

    def _reward(self, op, np_, reason):
        r = (self._d(op) - self._d(np_)) * 5
        if reason == 'goal':         r += 50000
        if reason == 'dead_end':     r -= 1000
        if reason == 'teleport':     r += 500
        if reason == 'fire_blocked': r -= 200
        if np_ not in self.visit_count: r += 100
        r -= self.visit_count.get(np_, 0) * 20
        if np_ in self.danger_zones: r -= 500
        return r

    def take_turn(self):
        fire_cells = get_fire_cells(self.turn)
        end_pos, seg, reason = walk_corridor(
            self.position, self.came_from, fire_cells, self.tp_pairs, self.goal)

        with self.lock:
            self.cell_path.extend(seg[1:])
            self.total_steps += len(seg)-1; self.turn += len(seg)-1
            for p in seg:
                self.known_cells.add(p)
                self.visit_count[p] = self.visit_count.get(p,0)+1
            old_pos = self.position
            self.position  = end_pos
            self.came_from = seg[-2] if len(seg)>=2 else self.came_from
            if self._d(end_pos) < self.best_dist: self.best_dist = self._d(end_pos)
            if reason == 'teleport':
                self.teleport_count += 1; print(f"  🌀 Teleport → {end_pos}")
            if reason == 'dead_end':
                self.dead_ends.add(end_pos)

        r = self._reward(old_pos, end_pos, reason)
        self._update_q(old_pos, end_pos, r)
        self.strategy_cd -= 1
        if self.strategy_cd <= 0: self._switch_strategy(); self.strategy_cd = 30

        if reason == 'goal' or end_pos == self.goal: return 'goal'

        # fire_blocked = only exit is on fire right now → treat like a dead end
        # agent will backtrack and try other junctions
        if reason == 'fire_blocked':
            with self.lock:
                self.dead_ends.add(end_pos)

        exits = [n for n in cell_nbrs(*end_pos) if n != self.came_from]
        if not exits:
            # Genuine dead end or fully blocked — backtrack via came_from
            if self.came_from:
                with self.lock:
                    cf = self.came_from
                    self.position  = cf
                    self.came_from = seg[-3] if len(seg) >= 3 else None
                    self.cell_path.append(cf)
                    self.visit_count[cf] = self.visit_count.get(cf, 0) + 1
                return 'continue'
            return 'stuck'
        chosen = (random.choice(exits) if random.random() < self.epsilon
                  else max(exits, key=self._score))
        with self.lock:
            self.came_from = end_pos; self.position = chosen
            self.cell_path.append(chosen)
            self.visit_count[chosen] = self.visit_count.get(chosen,0)+1
        return 'continue'

    def _switch_strategy(self):
        total = sum(self.strategy_scores.values()) or 1
        rand, c = random.random()*total, 0
        for s, sc in self.strategy_scores.items():
            c += sc
            if rand <= c: self.current_strategy = s; break

    def end_episode(self, success):
        with self.lock:
            s = self.current_strategy
            if success:
                self.strategy_scores[s] = self.strategy_scores.get(s,0)+0.5
                self.success_paths.append(self.cell_path[:]); self.success_count += 1
            else:
                self.strategy_scores[s] = max(0.1, self.strategy_scores.get(s,0)-0.1)
                for p in self.cell_path[-5:]: self.danger_zones.add(p)
            total = sum(self.strategy_scores.values()) or 1
            for k in self.strategy_scores: self.strategy_scores[k] /= total
            self.epsilon = max(0.05, self.epsilon*0.97); self.episode += 1

    def reset_episode(self):
        with self.lock: self._reset_episode()

    def sname(self, s=None):
        s = s if s is not None else self.current_strategy
        return {self.S_GREEDY:"Greedy",self.S_WALL:"Wall-Follow",
                self.S_EXPLORE:"Explorer",self.S_QLEARN:"Q-Learn"}[s]

# ── Build & run agent ──────────────────────────────────────────────────────────
agent = JunctionAgent(START_CELL, GOAL_CELL, TELEPORT_PAIRS)

def train():
    for ep in range(500):
        if not agent.is_training: break
        agent.reset_episode()
        ep_steps = 0
        while ep_steps < 30000:
            if not agent.is_training: return
            result   = agent.take_turn()
            ep_steps = len(agent.cell_path)
            if result == 'goal':
                agent.end_episode(True)
                print(f"✓ Ep {ep+1}: SUCCESS in {len(agent.cell_path)} steps!"); time.sleep(0.1); break
            elif result == 'stuck':
                agent.end_episode(False); print(f"✗ Ep {ep+1}: stuck"); break
            time.sleep(0.001)
        else:
            agent.end_episode(False); print(f"✗ Ep {ep+1}: timeout")
    agent.is_training = False
    print(f"\n🏆 Done! {agent.success_count}/500  teleports={agent.teleport_count}")

threading.Thread(target=train, daemon=True).start()

# ── Static base image ──────────────────────────────────────────────────────────
def cell_to_px(r, c): return OFFSET + r*CELL_SIZE + CELL_SIZE//2, OFFSET + c*CELL_SIZE + CELL_SIZE//2

def fill_cell(ov, r, c, color, pad=1):
    py, px = cell_to_px(r, c)
    h = CELL_SIZE//2 - pad
    ov[max(0,py-h):min(IMG_SIZE,py+h+1),
       max(0,px-h):min(IMG_SIZE,px+h+1)] = color

# White background + original maze walls drawn correctly
base_img = np.ones((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8) * 255
base_img[gray < 128] = [0, 0, 0]   # walls exactly as in original image

# Teleport and fire on base
tp_colors = [[0,200,100], [160,60,220]]
for r, c in FIRE_CELLS:
    fill_cell(base_img, r, c, [255,110,0], pad=2)
for i, (tp1, tp2) in enumerate(TELEPORT_PAIRS):
    col = tp_colors[i]
    fill_cell(base_img, tp1[0], tp1[1], col, pad=1)
    fill_cell(base_img, tp2[0], tp2[1], col, pad=1)

fill_cell(base_img, GOAL_CELL[0],  GOAL_CELL[1],  [0,255,255], pad=0)
fill_cell(base_img, START_CELL[0], START_CELL[1], [0,220,0],   pad=0)

# ── Visualisation ──────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13,13))
im = ax.imshow(base_img.copy())
ax.set_title(f"64×64 Maze  |  Start (green) → Goal (cyan)  |  Edge-wall navigation",
             fontsize=12)
ax.axis('off')

info_txt  = ax.text(8,  18, '', fontsize=10, color='black',
                    bbox=dict(boxstyle='round',fc='white',alpha=0.92,ec='gray'))
strat_txt = ax.text(8, 180, '', fontsize=9, color='white',
                    bbox=dict(boxstyle='round',fc='#1a1aff',alpha=0.85))

def draw(_frame):
    s  = agent.snapshot()
    ov = base_img.copy()

    # Explored cells — light blue fill (inside cell only, walls stay black)
    for (r, c) in s['known_cells']:
        fill_cell(ov, r, c, [215,235,255], pad=2)

    # Dead ends — pink
    for (r, c) in s['dead_ends']:
        fill_cell(ov, r, c, [255,195,195], pad=2)

    # Redraw hazards on top
    for r, c in FIRE_CELLS:
        fill_cell(ov, r, c, [255,110,0], pad=2)
    for i, (tp1, tp2) in enumerate(TELEPORT_PAIRS):
        col = tp_colors[i]
        fill_cell(ov, tp1[0], tp1[1], col, pad=1)
        fill_cell(ov, tp2[0], tp2[1], col, pad=1)

    # Path gradient dark→bright blue
    path = s['cell_path']
    plen = max(1, len(path)-1)
    for i, (r, c) in enumerate(path):
        t = i/plen
        fill_cell(ov, r, c, [int(30+60*t), int(80+130*t), 255], pad=2)

    # Agent (red), Goal (cyan), Start (lime) always on top
    fill_cell(ov, s['position'][0], s['position'][1], [255,0,0],   pad=0)
    fill_cell(ov, GOAL_CELL[0],  GOAL_CELL[1],  [0,255,255], pad=0)
    fill_cell(ov, START_CELL[0], START_CELL[1], [0,220,0],   pad=0)

    im.set_array(ov)

    dist = agent._d(s['position'])
    info_txt.set_text(
        f"Episode  : {s['episode']+1}\n"
        f"Steps    : {s['total_steps']} cells\n"
        f"Position : {s['position']}\n"
        f"Distance : {dist}  (best {s['best_dist']})\n"
        f"Dead-ends: {len(s['dead_ends'])}\n"
        f"Teleports: {s['teleport_count']}\n"
        f"Successes: {s['success_count']}"
    )
    scores = sorted(s['strategy_scores'].items(), key=lambda k:k[1], reverse=True)
    strat_txt.set_text(
        f"Strategy : {agent.sname(s['current_strategy'])}\n"
        f"ε        : {s['epsilon']:.3f}\n\n" +
        '\n'.join(f"{agent.sname(k)}: {v:.1%}" for k,v in scores)
    )
    return [im, info_txt, strat_txt]

anim = FuncAnimation(fig, draw, interval=50, blit=False, cache_frame_data=False)
plt.tight_layout()
plt.show()