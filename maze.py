import os
import sys
from collections import deque
import numpy as np
from PIL import Image
import cv2
import glob

# use opencv to load templates and do template matching to find hazards in the maze
class CvTemplateHazards:

    def __init__(self, templates_by_label, size=(14, 14)):
        self.size = tuple(size)
        self.templates = {}  # label -> list[BGR template (14,14,3)]
        self.label_mean_bgr = {}  # label -> avg foreground BGR prototype

        for label, paths in templates_by_label.items():
            arrs = []
            means = []
            for p in paths:
                t = cv2.imread(p, cv2.IMREAD_COLOR)  # read in bgr
                if t is None:
                    continue
                t = cv2.resize(t, self.size, interpolation=cv2.INTER_NEAREST)
                arrs.append(t)
                mean_bgr = self.foreground_mean_bgr(t)
                if mean_bgr is not None:
                    means.append(mean_bgr)
            if arrs:
                self.templates[label] = arrs
            if means:
                self.label_mean_bgr[label] = np.mean(np.vstack(means), axis=0)

        if not self.templates:
            raise RuntimeError("no templates loaded. Check template folder.")

    @staticmethod
    def mask_foreground(bgr14):
        hsv = cv2.cvtColor(bgr14, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        mask = ((v < 250) | (s > 20)).astype(np.uint8) * 255
        return mask

    @staticmethod
    def foreground_mean_bgr(bgr14):
        mask = CvTemplateHazards.mask_foreground(bgr14) > 0
        if int(mask.sum()) == 0:
            return None
        return bgr14[mask].mean(axis=0)

    def color_distances(self, bgr14):
        mean_bgr = self.foreground_mean_bgr(bgr14)
        if mean_bgr is None:
            return {}
        return {
            label: float(np.linalg.norm(mean_bgr - proto))
            for label, proto in self.label_mean_bgr.items()
        }

    def classify(self, bgr14):
        patch = cv2.resize(bgr14, self.size, interpolation=cv2.INTER_NEAREST)

        if cv2.countNonZero(self.mask_foreground(patch)) < 30:
            return None, -1.0, {}

        scores_by_label = {}
        best_label = None
        best_score = -1.0

        for label, tmpl_list in self.templates.items():
            label_best = -1.0
            for tmpl in tmpl_list:
                patch_m = patch.copy()
                tmpl_m = tmpl.copy()

                tmpl_mask = self.mask_foreground(tmpl_m)
                if cv2.countNonZero(tmpl_mask) < 30:
                    continue

                bg = (tmpl_mask == 0)
                patch_m[bg] = 255
                tmpl_m[bg] = 255

                res = cv2.matchTemplate(patch_m, tmpl_m, cv2.TM_CCOEFF_NORMED)
                score = float(res[0, 0])
                if score > label_best:
                    label_best = score

            scores_by_label[label] = label_best
            if label_best > best_score:
                best_score = label_best
                best_label = label

        return best_label, best_score, scores_by_label

# load templates from ./templates directory
def load_templates_from_dir(template_dir="templates"):
    labels = ["confusion", "death_pit", "teleport_orange", "teleport_green", "teleport_purple", "teleport_red", "arrow_up", "arrow_left"]
    templates_by_label = {}

    for lab in labels:
        patterns = [
            os.path.join(template_dir, f"{lab}_*.png"),
            os.path.join(template_dir, f"{lab}_*.jpg"),
            os.path.join(template_dir, f"{lab}_*.jpeg"),
        ]
        files = []
        for pat in patterns:
            files.extend(glob.glob(pat))
        files = sorted(files)

        if files:
            templates_by_label[lab] = files

    if not templates_by_label:
        raise RuntimeError(f"No templates found in '{template_dir}'.")

    return templates_by_label


class MazeLoader:
    CELL_SIZE = 16
    WALL_THICKNESS = 2
    INNER_SIZE = CELL_SIZE - 2 * WALL_THICKNESS  # 14x14px

    def __init__(self, image_path, template_dir="templates", template_threshold=0.70):
        self.image_path = image_path
        self.img = Image.open(image_path).convert("RGB")
        self.rgb_array = np.array(self.img)
        self.h, self.w = self.rgb_array.shape[:2]

        gray_img = self.img.convert("L")
        self.maze_array = np.array(gray_img) > 128

        self.maze_height_cells = self.h // self.CELL_SIZE
        self.maze_width_cells = self.w // self.CELL_SIZE

        self.start_pos = self.find_middle_opening("bottom")
        self.goal_pos = self.find_middle_opening("top")

        self.death_pits = []
        self.confusion_pads = []
        self.teleport_purple = []
        self.teleport_orange = []
        self.teleport_green = []
        self.teleport_red = []
        self.arrow_up = []
        self.arrow_left = []

        self.bgr = cv2.cvtColor(self.rgb_array, cv2.COLOR_RGB2BGR)

        templates_by_label = load_templates_from_dir(template_dir)
        self.matcher = CvTemplateHazards(templates_by_label, size=(self.INNER_SIZE, self.INNER_SIZE))
        self.template_threshold = float(template_threshold)

    def find_middle_opening(self, edge):
        mid = self.w // 2
        search_range = 100
        row = 0 if edge == "top" else self.h - 1

        openings = []
        for j in range(max(0, mid - search_range), min(self.w, mid + search_range)):
            if self.maze_array[row, j]:
                openings.append((row, j))

        return openings[len(openings) // 2] if openings else None

    def cell_interior_bgr(self, cell_row, cell_col):
        y0 = cell_row * self.CELL_SIZE + self.WALL_THICKNESS
        x0 = cell_col * self.CELL_SIZE + self.WALL_THICKNESS
        return self.bgr[y0:y0 + self.INNER_SIZE, x0:x0 + self.INNER_SIZE]

    def classify_cell_template(self, cell_row, cell_col):
        patch = self.cell_interior_bgr(cell_row, cell_col)
        label, score, scores = self.matcher.classify(patch)
        color_dists = self.matcher.color_distances(patch)

        fallback = self.color_fallback_label(scores, color_dists)

        if label is None:
            return fallback
        if score < self.template_threshold:
            return fallback

        if label == "death_pit":
            fg_mask = ~((patch[:,:,0]>235) & (patch[:,:,1]>235) & (patch[:,:,2]>235))
            fg_count = int(fg_mask.sum())
            if fg_count > 0:
                mean_r = float(patch[:,:,2][fg_mask].mean())  # BGR: channel 2 = R
                mean_g = float(patch[:,:,1][fg_mask].mean())  # BGR: channel 1 = G
                gr_ratio = mean_g / mean_r if mean_r > 0 else 1.0
            else:
                gr_ratio = 1.0

            runner_up = sorted(
                ((k, v_) for k, v_ in scores.items() if k != "death_pit"),
                key=lambda x: -x[1]
            )
            best_runner_label = runner_up[0][0] if runner_up else None
            best_runner_score = runner_up[0][1] if runner_up else -1.0

            override = False
            if gr_ratio >= 0.76:
                override = True
            elif gr_ratio >= 0.68:
                if best_runner_score >= score - 0.12:
                    override = True

            if override:
                if fallback is not None:
                    return fallback
                if best_runner_label and best_runner_score >= self.template_threshold - 0.10:
                    return best_runner_label
                return None

        if label == "teleport_red":
            fg_mask = ~((patch[:,:,0]>235) & (patch[:,:,1]>235) & (patch[:,:,2]>235))
            fg_count = int(fg_mask.sum())
            if fg_count > 0:
                mean_r = float(patch[:,:,2][fg_mask].mean())
                mean_g = float(patch[:,:,1][fg_mask].mean())
                gr_ratio = mean_g / mean_r if mean_r > 0 else 1.0
            else:
                gr_ratio = 1.0
            if gr_ratio >= 0.40:
                dp_score = scores.get("death_pit", -1.0)
                if dp_score >= self.template_threshold:
                    return "death_pit"
                return None

        if label == "death_pit" and fallback is not None:
            return fallback

        return label

    def color_fallback_label(self, scores, color_dists):
        if not color_dists:
            return None

        fallback_rules = [
            ("confusion", 20.0, 0.42),
            ("teleport_orange", 28.0, 0.42),
            ("teleport_green", 28.0, 0.32),
        ]

        best_label = None
        best_score = -1.0
        for label, max_dist, min_score in fallback_rules:
            score = scores.get(label, -1.0)
            dist = color_dists.get(label, float("inf"))
            if dist <= max_dist and score >= min_score and score > best_score:
                best_label = label
                best_score = score

        return best_label

    def detect_hazards(self):
        print(f"searching {self.maze_height_cells}×{self.maze_width_cells} cells for hazards")

        detected = 0
        for r in range(self.maze_height_cells):
            for c in range(self.maze_width_cells):
                lab = self.classify_cell_template(r, c)
                if not lab:
                    continue

                detected += 1
                if lab == "death_pit":
                    self.death_pits.append((r, c))
                elif lab == "confusion":
                    self.confusion_pads.append((r, c))
                elif lab == "teleport_purple":
                    self.teleport_purple.append((r, c))
                elif lab == "teleport_orange":
                    self.teleport_orange.append((r, c))
                elif lab == "teleport_green":
                    self.teleport_green.append((r, c))
                elif lab == "teleport_red":
                    self.teleport_red.append((r, c))
                elif lab == "arrow_up":
                    self.arrow_up.append((r,c))
                elif lab == "arrow_left":
                    self.arrow_left.append((r,c))

        print(f"Found {detected} hazardous cells")
        return self.get_hazard_summary()

    def get_hazard_summary(self):
        return {
            "death_pits": len(self.death_pits),
            "confusion": len(self.confusion_pads),
            "teleport_purple": len(self.teleport_purple),
            "teleport_orange": len(self.teleport_orange),
            "teleport_green": len(self.teleport_green),
            "teleport_red": len(self.teleport_red),
            "arrow_up": len(self.arrow_up),
            "arrow_left": len(self.arrow_left),
            "start_pos": self.start_pos,
            "goal_pos": self.goal_pos,
        }

    def cell_to_pixel(self, cell_row, cell_col):
        py = cell_row * self.CELL_SIZE + self.CELL_SIZE // 2
        px = cell_col * self.CELL_SIZE + self.CELL_SIZE // 2
        return py, px

    def pixel_to_cell(self, pixel_y, pixel_x):
        r = min(max(pixel_y // self.CELL_SIZE, 0), self.maze_height_cells - 1)
        c = min(max(pixel_x // self.CELL_SIZE, 0), self.maze_width_cells - 1)
        return (r, c)

    def visualize_hazards(self, output_path, base_image_path=None, rotating_pits=None):
        viz_img = Image.open(base_image_path).convert("RGB") if base_image_path else self.img.copy()
        pixels = viz_img.load()
        marker = 4

        rotating_pits = set(rotating_pits) if rotating_pits else set()

        for r, c in self.death_pits:
            color = (255, 140, 30) if (r, c) in rotating_pits else (180, 0, 0)
            py, px = self.cell_to_pixel(r, c)
            for dy in range(-marker, marker + 1):
                for dx in range(-marker, marker + 1):
                    ny, nx = py + dy, px + dx
                    if 0 <= ny < self.h and 0 <= nx < self.w:
                        pixels[nx, ny] = color

        for r, c in self.confusion_pads:
            py, px = self.cell_to_pixel(r, c)
            for dy in range(-marker, marker + 1):
                for dx in range(-marker, marker + 1):
                    ny, nx = py + dy, px + dx
                    if 0 <= ny < self.h and 0 <= nx < self.w:
                        pixels[nx, ny] = (255, 255, 0)

        for r, c in self.teleport_purple:
            py, px = self.cell_to_pixel(r, c)
            for dy in range(-marker, marker + 1):
                for dx in range(-marker, marker + 1):
                    ny, nx = py + dy, px + dx
                    if 0 <= ny < self.h and 0 <= nx < self.w:
                        pixels[nx, ny] = (128, 0, 255)

        for r, c in self.teleport_orange:
            py, px = self.cell_to_pixel(r, c)
            for dy in range(-marker, marker + 1):
                for dx in range(-marker, marker + 1):
                    ny, nx = py + dy, px + dx
                    if 0 <= ny < self.h and 0 <= nx < self.w:
                        pixels[nx, ny] = (255, 140, 0)

        for r, c in self.teleport_green:
            py, px = self.cell_to_pixel(r, c)
            for dy in range(-marker, marker + 1):
                for dx in range(-marker, marker + 1):
                    ny, nx = py + dy, px + dx
                    if 0 <= ny < self.h and 0 <= nx < self.w:
                        pixels[nx, ny] = (0, 255, 0)

        for r, c in self.teleport_red:
            py, px = self.cell_to_pixel(r, c)
            for dy in range(-marker, marker + 1):
                for dx in range(-marker, marker + 1):
                    ny, nx = py + dy, px + dx
                    if 0 <= ny < self.h and 0 <= nx < self.w:
                        pixels[nx, ny] = (220, 0, 255)

        for pos in [self.start_pos, self.goal_pos]:
            if not pos:
                continue
            py, px = pos
            for dy in range(-6, 7):
                for dx in range(-6, 7):
                    ny, nx = py + dy, px + dx
                    if 0 <= ny < self.h and 0 <= nx < self.w:
                        pixels[nx, ny] = (0, 255, 255)

        viz_img.save(output_path)
        print(f"Saved hazard visualization to {output_path}")


# solving maze with BFS ignoring the hazards
def solve_maze_bfs(maze, start, end):
    visited = np.zeros_like(maze, dtype=bool)
    parent = {}
    queue = deque([start])
    visited[start] = True
    parent[start] = None

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while queue:
        cur = queue.popleft()
        if cur == end:
            path = []
            while cur is not None:
                path.append(cur)
                cur = parent[cur]
            return path[::-1]

        y, x = cur
        for dy, dx in directions:
            ny, nx = y + dy, x + dx
            if 0 <= ny < maze.shape[0] and 0 <= nx < maze.shape[1] and maze[ny, nx] and not visited[ny, nx]:
                visited[ny, nx] = True
                parent[(ny, nx)] = cur
                queue.append((ny, nx))

    return None


def main():
    maze_path = sys.argv[1] if len(sys.argv) > 1 else "maze-alpha/MAZE_1.png"
    loader = MazeLoader(maze_path, template_dir="templates")

    print(f"\nImage: {loader.w}×{loader.h}")
    print(f"Grid: {loader.maze_width_cells}×{loader.maze_height_cells} cells")
    print(f"Start: {loader.start_pos}")
    print(f"Goal: {loader.goal_pos}")

    summary = loader.detect_hazards()

    print("\nHazard results:")
    print(f"- Death pits: {summary['death_pits']}")
    print(f"- Confusion pads: {summary['confusion']}")
    print(f"- Purple teleports: {summary['teleport_purple']}")
    print(f"- Orange teleports: {summary['teleport_orange']}")
    print(f"- Green teleports: {summary['teleport_green']}")
    print(f"- Red teleports: {summary['teleport_red']}")

    total = (summary["death_pits"] + summary["confusion"] + summary["teleport_purple"] + summary["teleport_orange"] + summary["teleport_green"] + summary["teleport_red"])
    print(f"Total hazardous cells: {total}")

    loader.visualize_hazards("maze_detected_hazards.png", base_image_path=maze_path)

    print("\nBFS")
    path = solve_maze_bfs(loader.maze_array, loader.start_pos, loader.goal_pos)
    if path:
        print(f"Path length: {len(path)}")
        sol = loader.img.copy()
        pix = sol.load()
        for y, x in path:
            pix[x, y] = (255, 0, 0)
        sol.save("maze_solution.png")

if __name__ == "__main__":
    main()