import os
from collections import deque
import numpy as np
from PIL import Image
import cv2
import glob


class CvTemplateHazards:
    def __init__(self, templates_by_label, size=(14, 14)):
        self.size = tuple(size)
        self.templates = {}
        for label, paths in templates_by_label.items():
            arrs = []
            for p in paths:
                t = cv2.imread(p, cv2.IMREAD_COLOR)
                if t is None:
                    continue
                t = cv2.resize(t, self.size, interpolation=cv2.INTER_NEAREST)
                arrs.append(t)
            if arrs:
                self.templates[label] = arrs
        if not self.templates:
            raise RuntimeError("no templates loaded. Check template folder.")

    @staticmethod
    def mask_foreground(bgr14):
        hsv = cv2.cvtColor(bgr14, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        mask = ((v < 250) | (s > 20)).astype(np.uint8) * 255
        return mask

    def classify(self, bgr14):
        patch = cv2.resize(bgr14, self.size, interpolation=cv2.INTER_NEAREST)
        mask  = self.mask_foreground(patch)
        if cv2.countNonZero(mask) < 15:
            return None, -1.0, {}
        scores_by_label = {}
        best_label = None
        best_score = -1.0
        for label, tmpl_list in self.templates.items():
            label_best = -1.0
            for tmpl in tmpl_list:
                pm = patch.copy(); tm = tmpl.copy()
                bg = (mask == 0)
                pm[bg] = 255; tm[bg] = 255
                res   = cv2.matchTemplate(pm, tm, cv2.TM_CCOEFF_NORMED)
                score = float(res[0, 0])
                if score > label_best:
                    label_best = score
            scores_by_label[label] = label_best
            if label_best > best_score:
                best_score = label_best
                best_label = label
        return best_label, best_score, scores_by_label


def load_templates_from_dir(template_dir="templates"):
    labels = ["confusion", "death_pit", "teleport_orange", "teleport_green", "teleport_purple"]
    templates_by_label = {}
    for lab in labels:
        files = []
        for pat in [f"{lab}_*.png", f"{lab}_*.jpg", f"{lab}_*.jpeg"]:
            files.extend(glob.glob(os.path.join(template_dir, pat)))
        if files:
            templates_by_label[lab] = sorted(files)
    if not templates_by_label:
        raise RuntimeError(f"No templates found in '{template_dir}'.")
    return templates_by_label


class MazeLoader:
    CELL_SIZE      = 16
    WALL_THICKNESS = 2
    INNER_SIZE     = CELL_SIZE - 2 * WALL_THICKNESS  # 12×12 px

    def __init__(self, image_path, template_dir="templates", template_threshold=0.55):
        self.image_path = image_path
        self.img        = Image.open(image_path).convert("RGB")
        self.rgb_array  = np.array(self.img)
        self.h, self.w  = self.rgb_array.shape[:2]

        gray_img        = self.img.convert("L")
        self.maze_array = np.array(gray_img) > 128

        self.maze_height_cells = self.h // self.CELL_SIZE
        self.maze_width_cells  = self.w // self.CELL_SIZE

        self.start_pos = self.find_middle_opening("bottom")
        self.goal_pos  = self.find_middle_opening("top")

        self.death_pits      = []
        self.confusion_pads  = []
        self.teleport_purple = []
        self.teleport_orange = []
        self.teleport_green  = []

        self.bgr = cv2.cvtColor(self.rgb_array, cv2.COLOR_RGB2BGR)

        templates_by_label     = load_templates_from_dir(template_dir)
        self.matcher           = CvTemplateHazards(templates_by_label,
                                                   size=(self.INNER_SIZE, self.INNER_SIZE))
        self.template_threshold = float(template_threshold)

    def find_middle_opening(self, edge):
        mid          = self.w // 2
        search_range = 100
        row          = 0 if edge == "top" else self.h - 1
        openings     = []
        for j in range(max(0, mid - search_range), min(self.w, mid + search_range)):
            if self.maze_array[row, j]:
                openings.append((row, j))
        return openings[len(openings) // 2] if openings else None

    def cell_interior_bgr(self, cell_row, cell_col):
        y0 = cell_row * self.CELL_SIZE + self.WALL_THICKNESS
        x0 = cell_col * self.CELL_SIZE + self.WALL_THICKNESS
        return self.bgr[y0:y0 + self.INNER_SIZE, x0:x0 + self.INNER_SIZE]

    # ── Color-based primary classifier ──────────────────────────────────────
    def _classify_by_color(self, patch_bgr) -> str | None:
        """
        Primary classifier using HSV color analysis.
        OpenCV HSV: H 0-179, S 0-255, V 0-255

        Hue ranges (approximate):
          Red/orange flame  :  0–20  (fire emoji has HIGH std — varies red→orange→yellow)
          Orange pad        :  8–20  (uniform solid orange — LOW std)
          Yellow smiley     : 20–35  (confusion pads)
          Green             : 55–95
          Purple/magenta    :125–160
        """
        hsv = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2HSV)
        h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

        # Foreground mask: exclude near-white pixels
        fg = ~((s < 35) & (v > 195))
        n  = int(fg.sum())
        if n < 8:
            return None

        hf = h[fg].astype(np.float32)
        sf = s[fg].astype(np.float32)
        vf = v[fg].astype(np.float32)

        mean_h = float(hf.mean())
        std_h  = float(hf.std())
        mean_s = float(sf.mean())
        mean_v = float(vf.mean())

        # ── Purple teleport: hue ~125–160 ───────────────────────────────────
        if 120 <= mean_h <= 165 and mean_s > 60:
            return "teleport_purple"

        # ── Green teleport: hue ~55–95 ──────────────────────────────────────
        if 55 <= mean_h <= 100 and mean_s > 60:
            return "teleport_green"

        # ── Warm colours (red / orange / yellow): hue 0–45 ──────────────────
        if mean_h <= 45 or mean_h >= 165:
            if mean_s < 50:
                return None      # desaturated — probably empty

            # Fire emoji: flame has RED base + ORANGE body + YELLOW tip
            # → large hue standard deviation
            if std_h > 9:
                return "death_pit"

            # Solid yellow → confusion pads (two smiley faces)
            if mean_h >= 22:
                return "confusion"

            # Solid orange → teleport pad
            return "teleport_orange"

        return None

    def classify_cell_template(self, cell_row, cell_col):
        patch = self.cell_interior_bgr(cell_row, cell_col)

        # 1. Try color-based classification first (more reliable)
        color_label = self._classify_by_color(patch)
        if color_label is not None:
            return color_label

        # 2. Fallback to template matching for anything the color
        #    classifier can't place (e.g. unusual renders)
        label, score, _ = self.matcher.classify(patch)
        if label is None or score < self.template_threshold:
            return None
        return label

    def detect_hazards(self):
        print(f"searching {self.maze_height_cells}×{self.maze_width_cells} cells for hazards")
        detected = 0
        for r in range(self.maze_height_cells):
            for c in range(self.maze_width_cells):
                lab = self.classify_cell_template(r, c)
                if not lab:
                    continue
                detected += 1
                if   lab == "death_pit":       self.death_pits.append((r, c))
                elif lab == "confusion":       self.confusion_pads.append((r, c))
                elif lab == "teleport_purple": self.teleport_purple.append((r, c))
                elif lab == "teleport_orange": self.teleport_orange.append((r, c))
                elif lab == "teleport_green":  self.teleport_green.append((r, c))
        print(f"Found {detected} hazardous cells")
        return self.get_hazard_summary()

    def get_hazard_summary(self):
        return {
            "death_pits":      len(self.death_pits),
            "confusion":       len(self.confusion_pads),
            "teleport_purple": len(self.teleport_purple),
            "teleport_orange": len(self.teleport_orange),
            "teleport_green":  len(self.teleport_green),
            "start_pos":       self.start_pos,
            "goal_pos":        self.goal_pos,
        }

    def cell_to_pixel(self, cell_row, cell_col):
        py = cell_row * self.CELL_SIZE + self.CELL_SIZE // 2
        px = cell_col * self.CELL_SIZE + self.CELL_SIZE // 2
        return py, px

    def pixel_to_cell(self, pixel_y, pixel_x):
        r = min(max(pixel_y // self.CELL_SIZE, 0), self.maze_height_cells - 1)
        c = min(max(pixel_x // self.CELL_SIZE, 0), self.maze_width_cells  - 1)
        return (r, c)

    def visualize_hazards(self, output_path, base_image_path=None):
        viz_img = Image.open(base_image_path).convert("RGB") if base_image_path else self.img.copy()
        pixels  = viz_img.load()
        marker  = 4
        colour_map = {
            "death_pits":      ((255,   0,   0), self.death_pits),
            "confusion":       ((255, 255,   0), self.confusion_pads),
            "teleport_purple": ((128,   0, 255), self.teleport_purple),
            "teleport_orange": ((255, 140,   0), self.teleport_orange),
            "teleport_green":  ((  0, 255,   0), self.teleport_green),
        }
        for _name, (colour, cells) in colour_map.items():
            for r, c in cells:
                py, px = self.cell_to_pixel(r, c)
                for dy in range(-marker, marker + 1):
                    for dx in range(-marker, marker + 1):
                        ny, nx = py + dy, px + dx
                        if 0 <= ny < self.h and 0 <= nx < self.w:
                            pixels[nx, ny] = colour
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


def solve_maze_bfs(maze, start, end):
    visited    = np.zeros_like(maze, dtype=bool)
    parent     = {}
    queue      = deque([start])
    visited[start] = True
    parent[start]  = None
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    while queue:
        cur = queue.popleft()
        if cur == end:
            path = []
            while cur is not None:
                path.append(cur); cur = parent[cur]
            return path[::-1]
        y, x = cur
        for dy, dx in directions:
            ny, nx = y + dy, x + dx
            if (0 <= ny < maze.shape[0] and 0 <= nx < maze.shape[1]
                    and maze[ny, nx] and not visited[ny, nx]):
                visited[ny, nx] = True
                parent[(ny, nx)] = cur
                queue.append((ny, nx))
    return None


def main():
    import sys
    maze_path = sys.argv[1] if len(sys.argv) > 1 else "maze-alpha/MAZE_1.png"
    loader    = MazeLoader(maze_path)

    print(f"\nImage: {loader.w}×{loader.h}")
    print(f"Grid:  {loader.maze_width_cells}×{loader.maze_height_cells} cells")
    print(f"Start: {loader.start_pos}")
    print(f"Goal:  {loader.goal_pos}")

    summary = loader.detect_hazards()
    print("\nHazard results:")
    for k, v in summary.items():
        print(f"  {k:<20}: {v}")

    out = maze_path.replace(".png", "_detected_hazards.png")
    loader.visualize_hazards(out, base_image_path=maze_path)
    print(f"\nSaved {out}")

    path = solve_maze_bfs(loader.maze_array, loader.start_pos, loader.goal_pos)
    if path:
        sol = loader.img.copy(); pix = sol.load()
        for y, x in path:
            pix[x, y] = (255, 0, 0)
        sol_out = maze_path.replace(".png", "_solution.png")
        sol.save(sol_out)
        print(f"BFS path length: {len(path)}  →  {sol_out}")


if __name__ == "__main__":
    main()