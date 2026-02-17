import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random
from collections import defaultdict
import threading
import time
from scipy.ndimage import distance_transform_edt, label, center_of_mass

img_path = 'MAZE_1.png'
img_file = Image.open(img_path)

rgb_array = np.array(img_file)
gray_img = img_file.convert('L')
walls = np.array(gray_img) < 128

red_channel = rgb_array[:, :, 0]
green_channel = rgb_array[:, :, 1]
blue_channel = rgb_array[:, :, 2]

fire_hazards = (red_channel > 150) & (green_channel > 80) & (green_channel < 180) & (blue_channel < 100)
passable = ~walls

# Detect teleport pads
green_teleports = (red_channel < 100) & (green_channel > 150) & (blue_channel < 100)
yellow_teleports = (red_channel > 200) & (green_channel > 200) & (blue_channel < 100)
purple_teleports = (red_channel > 150) & (green_channel < 100) & (blue_channel > 150)
cyan_markers = (red_channel < 100) & (green_channel > 150) & (blue_channel > 150)

print(f"Maze dimensions: {rgb_array.shape}")
print(f"Found {np.sum(fire_hazards)} fire hazard pixels")
print(f"Found {np.sum(yellow_teleports)} yellow teleport pixels")
print(f"Found {np.sum(green_teleports)} green teleport pixels")
print(f"Found {np.sum(purple_teleports)} purple teleport pixels")
print(f"Found {np.sum(cyan_markers)} cyan marker pixels")

# Compute skeleton/centerline of the maze
print("Computing maze centerline...")
distance_map = distance_transform_edt(passable)
centerline_threshold = 2.0
centerline = distance_map >= centerline_threshold

print(f"Centerline has {np.sum(centerline)} cells")
print(f"Total passable area: {np.sum(passable)} cells")

def find_teleport_pairs(teleport_mask):
    """Find pairs of teleport locations"""
    labeled, num = label(teleport_mask)
    if num >= 2:
        centers = center_of_mass(teleport_mask, labeled, range(1, num+1))
        tp1 = tuple(map(int, centers[0]))
        tp2 = tuple(map(int, centers[1]))
        return tp1, tp2
    return None, None

yellow_tp1, yellow_tp2 = find_teleport_pairs(yellow_teleports)
green_tp1, green_tp2 = find_teleport_pairs(green_teleports)
purple_tp1, purple_tp2 = find_teleport_pairs(purple_teleports)

print(f"Yellow teleports: {yellow_tp1} <-> {yellow_tp2}")
print(f"Green teleports: {green_tp1} <-> {green_tp2}")
print(f"Purple teleports: {purple_tp1} <-> {purple_tp2}")

def find_openings(maze):
    h, w = maze.shape
    top_openings = [(0, j) for j in range(w) if maze[0, j]]
    bottom_openings = [(h-1, j) for j in range(w) if maze[h-1, j]]
    left_openings = [(i, 0) for i in range(h) if maze[i, 0]]
    right_openings = [(i, w-1) for i in range(h) if maze[i, w-1]]
    return top_openings + bottom_openings + left_openings + right_openings

openings = find_openings(passable)

# Start from bottom, goal at top
start = openings[-1] if len(openings) > 1 else openings[0]
goal = openings[0] if openings else None

print(f"Start (bottom): {start}, Goal (top/cyan): {goal}")

def get_fire_positions(fire_mask, turn):
    fire_positions = set()
    h, w = fire_mask.shape
    center_y, center_x = h // 2, w // 2
    
    for i in range(h):
        for j in range(w):
            if fire_mask[i, j]:
                y_offset = i - center_y
                x_offset = j - center_x
                for _ in range(turn % 4):
                    y_offset, x_offset = x_offset, -y_offset
                new_y = center_y + y_offset
                new_x = center_x + x_offset
                if 0 <= new_y < h and 0 <= new_x < w:
                    fire_positions.add((int(new_y), int(new_x)))
    return fire_positions

def find_nearest_centerline(pos, centerline_mask, max_search=10):
    y, x = pos
    if centerline_mask[y, x]:
        return pos
    for radius in range(1, max_search + 1):
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                ny, nx = y + dy, x + dx
                if (0 <= ny < centerline_mask.shape[0] and 
                    0 <= nx < centerline_mask.shape[1] and
                    centerline_mask[ny, nx]):
                    return (ny, nx)
    return pos

def check_teleport(pos, teleport_pairs):
    """Check if position is on a teleport pad and return destination"""
    for tp1, tp2 in teleport_pairs:
        if tp1 and tp2:
            # Check if within 5x5 area around teleport center
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    test_pos = (pos[0] + dy, pos[1] + dx)
                    if test_pos == tp1:
                        return tp2
                    if test_pos == tp2:
                        return tp1
    return None

class CenterlineFollowingAgent:
    NORTH = (-1, 0)
    SOUTH = (1, 0)
    WEST = (0, -1)
    EAST = (0, 1)
    DIRECTIONS = [NORTH, EAST, SOUTH, WEST]
    DIR_NAMES = ['North', 'East', 'South', 'West']
    
    STRATEGY_GREEDY = 0
    STRATEGY_WALL_FOLLOW = 1
    STRATEGY_EXPLORE = 2
    STRATEGY_Q_LEARN = 3
    
    def __init__(self, start_pos, goal_pos, passable_mask, centerline_mask, distance_map, fire_mask, teleport_pairs):
        self.start = find_nearest_centerline(start_pos, centerline_mask)
        self.goal = find_nearest_centerline(goal_pos, centerline_mask)
        
        self.passable_mask = passable_mask
        self.centerline_mask = centerline_mask
        self.distance_map = distance_map
        self.fire_mask = fire_mask
        self.teleport_pairs = teleport_pairs
        
        self.position = self.start
        self.facing = 0
        self.turn = 0
        self.episode = 0
        self.total_steps = 0
        
        self.path = [self.start]
        self.visited = {self.start}
        self.current_episode_path = [self.start]
        
        self.q_table = defaultdict(lambda: defaultdict(float))
        self.visit_count = defaultdict(int)
        self.success_paths = []
        self.danger_zones = set()
        self.known_walls = set()
        self.known_passable = {self.start}
        self.teleport_used_count = 0
        
        self.strategy_scores = {
            self.STRATEGY_GREEDY: 0.25,
            self.STRATEGY_WALL_FOLLOW: 0.25,
            self.STRATEGY_EXPLORE: 0.25,
            self.STRATEGY_Q_LEARN: 0.25
        }
        self.current_strategy = self.STRATEGY_GREEDY
        self.strategy_switch_countdown = 100
        
        self.alpha = 0.1
        self.gamma = 0.95
        self.epsilon = 0.3
        self.best_distance = abs(self.start[0] - self.goal[0]) + abs(self.start[1] - self.goal[1])
        
        self.is_training = True
        self.success_count = 0
        
        print(f"Agent initialized at: {self.start}")
        print(f"Goal (cyan) at: {self.goal}")
    
    def get_state_key(self):
        region_size = 50
        region_y = self.position[0] // region_size
        region_x = self.position[1] // region_size
        return (region_y, region_x)
    
    def perceive_surroundings(self):
        y, x = self.position
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                ny, nx = y + dy, x + dx
                if 0 <= ny < self.passable_mask.shape[0] and 0 <= nx < self.passable_mask.shape[1]:
                    if not self.passable_mask[ny, nx]:
                        self.known_walls.add((ny, nx))
                    else:
                        self.known_passable.add((ny, nx))
    
    def is_valid_move(self, direction, fire_positions):
        dy, dx = direction
        ny, nx = self.position[0] + dy, self.position[1] + dx
        new_pos = (ny, nx)
        
        if not (0 <= ny < self.passable_mask.shape[0] and 0 <= nx < self.passable_mask.shape[1]):
            return False, None, 0
        
        if new_pos in self.known_walls or not self.passable_mask[ny, nx]:
            self.known_walls.add(new_pos)
            return False, None, 0
        
        if new_pos in fire_positions:
            return False, None, 0
        
        centerline_score = self.distance_map[ny, nx]
        return True, new_pos, centerline_score
    
    def get_valid_moves(self, fire_positions):
        valid = []
        for direction in self.DIRECTIONS:
            is_valid, new_pos, centerline_score = self.is_valid_move(direction, fire_positions)
            if is_valid:
                valid.append((direction, new_pos, centerline_score))
        return valid
    
    def strategy_greedy(self, valid_moves):
        if not valid_moves:
            return None
        best_move = None
        best_score = -float('inf')
        for direction, new_pos, centerline_score in valid_moves:
            dist = abs(new_pos[0] - self.goal[0]) + abs(new_pos[1] - self.goal[1])
            score = centerline_score * 100 - dist
            if new_pos in self.danger_zones:
                score -= 10000
            if new_pos not in self.visited:
                score += 500
            if score > best_score:
                best_score = score
                best_move = direction
        return best_move
    
    def strategy_wall_follow(self, valid_moves):
        if not valid_moves:
            return None
        priorities = [(self.facing + 1) % 4, self.facing, (self.facing - 1) % 4, (self.facing + 2) % 4]
        best_move = None
        best_centerline = -1
        for dir_idx in priorities:
            direction = self.DIRECTIONS[dir_idx]
            for valid_dir, new_pos, centerline_score in valid_moves:
                if valid_dir == direction:
                    if centerline_score > best_centerline:
                        best_centerline = centerline_score
                        best_move = valid_dir
                        self.facing = dir_idx
        return best_move if best_move else (valid_moves[0][0] if valid_moves else None)
    
    def strategy_explore(self, valid_moves):
        if not valid_moves:
            return None
        scored_moves = []
        for direction, new_pos, centerline_score in valid_moves:
            score = centerline_score * 100
            if new_pos not in self.visited:
                score += 10000
            else:
                score += 1000 - self.visit_count[new_pos] * 10
            dist = abs(new_pos[0] - self.goal[0]) + abs(new_pos[1] - self.goal[1])
            score -= dist
            scored_moves.append((score, direction))
        scored_moves.sort(reverse=True)
        return scored_moves[0][1]
    
    def strategy_q_learn(self, valid_moves):
        if not valid_moves or random.random() < self.epsilon:
            if valid_moves:
                weights = [score for _, _, score in valid_moves]
                total = sum(weights)
                if total > 0:
                    weights = [w / total for w in weights]
                    choice_idx = random.choices(range(len(valid_moves)), weights=weights)[0]
                    return valid_moves[choice_idx][0]
            return None
        state = self.get_state_key()
        best_move = None
        best_score = -float('inf')
        for direction, new_pos, centerline_score in valid_moves:
            action_key = self.DIRECTIONS.index(direction)
            q_value = self.q_table[state][action_key]
            combined_score = q_value + centerline_score * 10
            if combined_score > best_score:
                best_score = combined_score
                best_move = direction
        if best_move is None:
            return self.strategy_greedy(valid_moves)
        return best_move
    
    def choose_move(self, valid_moves):
        if self.current_strategy == self.STRATEGY_GREEDY:
            return self.strategy_greedy(valid_moves)
        elif self.current_strategy == self.STRATEGY_WALL_FOLLOW:
            return self.strategy_wall_follow(valid_moves)
        elif self.current_strategy == self.STRATEGY_EXPLORE:
            return self.strategy_explore(valid_moves)
        else:
            return self.strategy_q_learn(valid_moves)
    
    def update_q_value(self, old_pos, action, new_pos, reward):
        old_state = (old_pos[0] // 50, old_pos[1] // 50)
        new_state = (new_pos[0] // 50, new_pos[1] // 50)
        action_idx = self.DIRECTIONS.index(action)
        max_next_q = max(self.q_table[new_state].values()) if self.q_table[new_state] else 0
        old_q = self.q_table[old_state][action_idx]
        self.q_table[old_state][action_idx] = old_q + self.alpha * (reward + self.gamma * max_next_q - old_q)
    
    def calculate_reward(self, old_pos, new_pos, teleported=False):
        old_dist = abs(old_pos[0] - self.goal[0]) + abs(old_pos[1] - self.goal[1])
        new_dist = abs(new_pos[0] - self.goal[0]) + abs(new_pos[1] - self.goal[1])
        reward = 0
        if teleported:
            reward += 100
        if new_dist < old_dist:
            reward += 10
        else:
            reward -= 5
        centerline_quality = self.distance_map[new_pos]
        reward += centerline_quality * 2
        if new_pos not in self.visited:
            reward += 5
        reward -= self.visit_count[new_pos] * 2
        if new_pos in self.danger_zones:
            reward -= 50
        return reward
    
    def take_turn(self):
        self.perceive_surroundings()
        fire_positions = get_fire_positions(self.fire_mask, self.turn)
        valid_moves = self.get_valid_moves(fire_positions)
        
        if not valid_moves:
            return False
        
        chosen_direction = self.choose_move(valid_moves)
        if chosen_direction is None:
            return False
        
        old_pos = self.position
        dy, dx = chosen_direction
        new_pos = (self.position[0] + dy, self.position[1] + dx)
        
        self.position = new_pos
        self.path.append(new_pos)
        self.current_episode_path.append(new_pos)
        self.visited.add(new_pos)
        self.visit_count[new_pos] += 1
        
        teleported = False
        teleport_dest = check_teleport(new_pos, self.teleport_pairs)
        if teleport_dest:
            print(f"  🌀 TELEPORTED from {new_pos} to {teleport_dest}!")
            self.position = teleport_dest
            self.path.append(teleport_dest)
            self.current_episode_path.append(teleport_dest)
            self.visited.add(teleport_dest)
            self.teleport_used_count += 1
            teleported = True
            new_pos = teleport_dest
        
        reward = self.calculate_reward(old_pos, new_pos, teleported)
        self.update_q_value(old_pos, chosen_direction, new_pos, reward)
        
        self.total_steps += 1
        dist = abs(new_pos[0] - self.goal[0]) + abs(new_pos[1] - self.goal[1])
        if dist < self.best_distance:
            self.best_distance = dist
        
        self.turn += 1
        self.strategy_switch_countdown -= 1
        if self.strategy_switch_countdown <= 0:
            self.switch_strategy()
            self.strategy_switch_countdown = 100
        
        return True
    
    def switch_strategy(self):
        total = sum(self.strategy_scores.values())
        rand = random.random() * total
        cumsum = 0
        for strategy, score in self.strategy_scores.items():
            cumsum += score
            if rand <= cumsum:
                self.current_strategy = strategy
                break
    
    def get_strategy_name(self, strategy=None):
        if strategy is None:
            strategy = self.current_strategy
        names = {
            self.STRATEGY_GREEDY: "Greedy+Center",
            self.STRATEGY_WALL_FOLLOW: "Wall+Center",
            self.STRATEGY_EXPLORE: "Explore+Center",
            self.STRATEGY_Q_LEARN: "Q-Learn+Center"
        }
        return names.get(strategy, "Unknown")
    
    def end_episode(self, success):
        if success:
            self.strategy_scores[self.current_strategy] += 0.5
            self.success_paths.append(self.current_episode_path.copy())
            self.success_count += 1
        else:
            self.strategy_scores[self.current_strategy] = max(0.1, self.strategy_scores[self.current_strategy] - 0.1)
            for pos in self.current_episode_path[-20:]:
                self.danger_zones.add(pos)
        total = sum(self.strategy_scores.values())
        for s in self.strategy_scores:
            self.strategy_scores[s] /= total
        self.epsilon = max(0.05, self.epsilon * 0.95)
        self.episode += 1
    
    def reset_episode(self):
        self.position = self.start
        self.current_episode_path = [self.start]
        self.visited = {self.start}
        self.turn = 0
        self.best_distance = abs(self.start[0] - self.goal[0]) + abs(self.start[1] - self.goal[1])

# Create teleport pairs
teleport_pairs = []
if yellow_tp1 and yellow_tp2:
    teleport_pairs.append((yellow_tp1, yellow_tp2))
if green_tp1 and green_tp2:
    teleport_pairs.append((green_tp1, green_tp2))
if purple_tp1 and purple_tp2:
    teleport_pairs.append((purple_tp1, purple_tp2))

print(f"\n🎓 Starting agent training...")
print(f"Teleport pairs: {len(teleport_pairs)}")
agent = CenterlineFollowingAgent(start, goal, passable, centerline, distance_map, fire_hazards, teleport_pairs)

def train_agent():
    max_episodes = 50
    max_steps_per_episode = 10000
    
    for episode in range(max_episodes):
        if not agent.is_training:
            break
        agent.reset_episode()
        
        for step in range(max_steps_per_episode):
            if not agent.is_training:
                return
            if agent.position == agent.goal:
                agent.end_episode(success=True)
                print(f"✓ Episode {episode+1}: SUCCESS in {len(agent.current_episode_path)} steps!")
                time.sleep(0.5)
                break
            moved = agent.take_turn()
            if not moved:
                agent.end_episode(success=False)
                print(f"✗ Episode {episode+1}: Stuck at step {step}")
                break
            time.sleep(0.001)
        else:
            agent.end_episode(success=False)
    
    agent.is_training = False
    print(f"\n🏆 Training complete! {agent.success_count}/{max_episodes} successes")

training_thread = threading.Thread(target=train_agent, daemon=True)
training_thread.start()

# VISUALIZATION with WHITE background
fig, ax = plt.subplots(figsize=(14, 14))
im = ax.imshow(rgb_array)
ax.set_title("Maze Agent Training (Bottom→Top/Cyan Goal)", fontsize=16)
ax.axis('off')

info_text = ax.text(10, 20, '', fontsize=10, color='black',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='black'))

strategy_text = ax.text(10, 140, '', fontsize=9, color='white',
                       bbox=dict(boxstyle='round', facecolor='blue', alpha=0.8))

def update_frame(frame):
    if not agent.path:
        return [im, info_text, strategy_text]
    
    y, x = agent.position
    
    # Create WHITE background overlay
    overlay = np.ones_like(rgb_array, dtype=float) * 255  # Start with white
    
    # Show walls in black
    overlay[walls] = [0, 0, 0]
    
    # Show centerline in very light gray
    overlay[centerline] = [240, 240, 240]
    
    # Show explored passable areas in light blue
    for ey, ex in agent.known_passable:
        if 0 <= ey < overlay.shape[0] and 0 <= ex < overlay.shape[1]:
            overlay[ey, ex] = [230, 240, 255]
    
    # Show current episode path in blue
    for py, px in agent.current_episode_path:
        overlay[py, px] = [100, 150, 255]
    
    # Show teleport pads
    overlay[yellow_teleports] = [255, 255, 0]  # Yellow
    overlay[green_teleports] = [0, 200, 0]  # Green
    overlay[purple_teleports] = [200, 0, 200]  # Purple
    
    # Current position (RED - larger)
    for dy in range(-6, 7):
        for dx in range(-6, 7):
            if 0 <= y+dy < overlay.shape[0] and 0 <= x+dx < overlay.shape[1]:
                overlay[y+dy, x+dx] = [255, 0, 0]
    
    # Goal (CYAN - larger)
    gy, gx = agent.goal
    for dy in range(-6, 7):
        for dx in range(-6, 7):
            if 0 <= gy+dy < overlay.shape[0] and 0 <= gx+dx < overlay.shape[1]:
                overlay[gy+dy, gx+dx] = [0, 255, 255]
    
    im.set_array(overlay.astype('uint8'))
    
    dist = abs(y - agent.goal[0]) + abs(x - agent.goal[1])
    info_text.set_text(
        f'Episode: {agent.episode + 1}\n'
        f'Total Steps: {agent.total_steps}\n'
        f'Position: ({y}, {x})\n'
        f'Distance to Goal: {dist}\n'
        f'Teleports Used: {agent.teleport_used_count}\n'
        f'Successes: {agent.success_count}'
    )
    
    strategy_info = '\n'.join([
        f'{agent.get_strategy_name(s)}: {score:.1%}'
        for s, score in sorted(agent.strategy_scores.items(), key=lambda x: x[1], reverse=True)
    ])
    strategy_text.set_text(
        f'Strategy: {agent.get_strategy_name()}\n'
        f'ε: {agent.epsilon:.3f}\n\n{strategy_info}'
    )
    
    return [im, info_text, strategy_text]

anim = FuncAnimation(fig, update_frame, interval=50, blit=True, cache_frame_data=False)
plt.tight_layout()
plt.show()

training_thread.join()

print(f"\nFinal Stats:")
print(f"  Episodes: {agent.episode}")
print(f"  Successes: {agent.success_count}")
print(f"  Teleports: {agent.teleport_used_count}")