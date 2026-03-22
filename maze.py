import os
from collections import deque
import numpy as np
from PIL import Image
import cv2
import glob

# use opencv to load templates and do template matching to find hazards in the maze 
class CvTemplateHazards:
    """
    Template matching on 14x14 cell interiors using OpenCV.
    Uses a foreground mask to reduce background (white) influence.
    """

    def __init__(self, templates_by_label, size=(14, 14)):
        self.size = tuple(size)
        self.templates = {}  # label -> list[BGR template (14,14,3)]

        for label, paths in templates_by_label.items():
            arrs = []
            for p in paths:
                t = cv2.imread(p, cv2.IMREAD_COLOR)  # read in bgr
                if t is None:
                    continue
                t = cv2.resize(t, self.size, interpolation=cv2.INTER_NEAREST)
                arrs.append(t)
            if arrs:
                self.templates[label] = arrs

        if not self.templates:
            raise RuntimeError("no templates loaded. Check template folder.")

    @staticmethod # simple operation that doesnt need self, just following convention i saw on reddit while searching
    # mask pixels that are not plain white, keep darker/saturated pixels
    # helps identify the hazards by getting rid of the white
    def mask_foreground(bgr14):
        hsv = cv2.cvtColor(bgr14, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        # keep if not near white or has decent saturation
        mask = ((v < 250) | (s > 20)).astype(np.uint8) * 255
        return mask

    # returns best label, best score, and dict of all scores by label
    # used to classify a cell patch as one of the hazard types or None if no good match
    def classify(self, bgr14):
        patch = cv2.resize(bgr14, self.size, interpolation=cv2.INTER_NEAREST)
        mask = self.mask_foreground(patch)

        if cv2.countNonZero(mask) < 15:
            return None, -1.0, {}

        scores_by_label = {}
        best_label = None
        best_score = -1.0

        # loop through each label and its templates and keep the best score for that label
        for label, tmpl_list in self.templates.items():
            label_best = -1.0
            for tmpl in tmpl_list:
                patch_m = patch.copy()
                tmpl_m = tmpl.copy()

                bg = (mask == 0)
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
# uses same naming format at make_templates.py
def load_templates_from_dir(template_dir="templates"):
    labels = ["confusion", "death_pit", "teleport_orange", "teleport_green", "teleport_purple"]
    templates_by_label = {}

    for lab in labels:
        patterns = [ # patterns to match image extensions (computer was being weird this worked)
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

    # templates not loaded
    if not templates_by_label:
        raise RuntimeError(f"No templates found in '{template_dir}'.")

    return templates_by_label


# load maze, detect hazards, solve with BFS
class MazeLoader:
    CELL_SIZE = 16
    WALL_THICKNESS = 2
    INNER_SIZE = CELL_SIZE - 2 * WALL_THICKNESS  # 14x14px

    def __init__(self, image_path, template_dir="templates", template_threshold=0.60):
        self.image_path = image_path
        self.img = Image.open(image_path).convert("RGB")
        self.rgb_array = np.array(self.img)
        self.h, self.w = self.rgb_array.shape[:2]

        # binary maze array for navigation where true = open and false = wall
        gray_img = self.img.convert("L")
        self.maze_array = np.array(gray_img) > 128

        self.maze_height_cells = self.h // self.CELL_SIZE
        self.maze_width_cells = self.w // self.CELL_SIZE

        self.start_pos = self.find_middle_opening("bottom")
        self.goal_pos = self.find_middle_opening("top")

        # hazard locations
        self.death_pits = []
        self.confusion_pads = []
        self.teleport_purple = []
        self.teleport_orange = []
        self.teleport_green = []

        # convert to bgr for opencv template matching
        self.bgr = cv2.cvtColor(self.rgb_array, cv2.COLOR_RGB2BGR)

        # load templates and initialize matcher
        templates_by_label = load_templates_from_dir(template_dir)
        self.matcher = CvTemplateHazards(templates_by_label, size=(self.INNER_SIZE, self.INNER_SIZE))
        self.template_threshold = float(template_threshold)

    # locate the openinngs on top/bottom (for start/end)
    def find_middle_opening(self, edge):
        mid = self.w // 2
        search_range = 100
        row = 0 if edge == "top" else self.h - 1

        openings = []
        for j in range(max(0, mid - search_range), min(self.w, mid + search_range)):
            if self.maze_array[row, j]:
                openings.append((row, j))

        return openings[len(openings) // 2] if openings else None

    # extract the inside 14x14px for template matching
    def cell_interior_bgr(self, cell_row, cell_col):
        y0 = cell_row * self.CELL_SIZE + self.WALL_THICKNESS
        x0 = cell_col * self.CELL_SIZE + self.WALL_THICKNESS
        return self.bgr[y0:y0 + self.INNER_SIZE, x0:x0 + self.INNER_SIZE]

    # classify cell using template matching
    def classify_cell_template(self, cell_row, cell_col):
        patch = self.cell_interior_bgr(cell_row, cell_col)
        label, score, scores = self.matcher.classify(patch)

        if label is None:
            return None
        if score < self.template_threshold:
            return None

        # tie breaker to distinguish between orange tp and death pit
        if label == "teleport_orange" and "death_pit" in scores:
            death_score = scores["death_pit"]
            orange_score = scores.get("teleport_orange", -1.0)

            # if death is within this margin of orange, assume death
            MARGIN = 0.03

            if death_score >= orange_score - MARGIN:
                # fire emoji is a bit darker than the orange tp pads, using this to fix some misclassifications
                hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
                mean_v = float(np.mean(hsv[..., 2])) / 255.0
                if mean_v <= 0.90:
                    return "death_pit"

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

        print(f"Found {detected} hazardous cells")
        return self.get_hazard_summary()

    def get_hazard_summary(self):
        return {
            "death_pits": len(self.death_pits),
            "confusion": len(self.confusion_pads),
            "teleport_purple": len(self.teleport_purple),
            "teleport_orange": len(self.teleport_orange),
            "teleport_green": len(self.teleport_green),
            "start_pos": self.start_pos,
            "goal_pos": self.goal_pos,
        }

    def cell_to_pixel(self, cell_row, cell_col):
        py = cell_row * self.CELL_SIZE + self.CELL_SIZE // 2
        px = cell_col * self.CELL_SIZE + self.CELL_SIZE // 2
        return py, px

    def pixel_to_cell(self, pixel_y, pixel_x): # convert pixel to cell coords
        r = min(max(pixel_y // self.CELL_SIZE, 0), self.maze_height_cells - 1)
        c = min(max(pixel_x // self.CELL_SIZE, 0), self.maze_width_cells - 1)
        return (r, c)

    # draw markers where hazards were detected
    def visualize_hazards(self, output_path, base_image_path=None):
        viz_img = Image.open(base_image_path).convert("RGB") if base_image_path else self.img.copy()
        pixels = viz_img.load()
        marker = 4

        # Death pits labeled with red
        for r, c in self.death_pits:
            py, px = self.cell_to_pixel(r, c)
            for dy in range(-marker, marker + 1):
                for dx in range(-marker, marker + 1):
                    ny, nx = py + dy, px + dx
                    if 0 <= ny < self.h and 0 <= nx < self.w:
                        pixels[nx, ny] = (255, 0, 0)

        # confusion pads labeled with yellow
        for r, c in self.confusion_pads:
            py, px = self.cell_to_pixel(r, c)
            for dy in range(-marker, marker + 1):
                for dx in range(-marker, marker + 1):
                    ny, nx = py + dy, px + dx
                    if 0 <= ny < self.h and 0 <= nx < self.w:
                        pixels[nx, ny] = (255, 255, 0)

        # purple tp pads
        for r, c in self.teleport_purple:
            py, px = self.cell_to_pixel(r, c)
            for dy in range(-marker, marker + 1):
                for dx in range(-marker, marker + 1):
                    ny, nx = py + dy, px + dx
                    if 0 <= ny < self.h and 0 <= nx < self.w:
                        pixels[nx, ny] = (128, 0, 255)

        # orange tp pads
        for r, c in self.teleport_orange:
            py, px = self.cell_to_pixel(r, c)
            for dy in range(-marker, marker + 1):
                for dx in range(-marker, marker + 1):
                    ny, nx = py + dy, px + dx
                    if 0 <= ny < self.h and 0 <= nx < self.w:
                        pixels[nx, ny] = (255, 140, 0)

        # green tp pads
        for r, c in self.teleport_green:
            py, px = self.cell_to_pixel(r, c)
            for dy in range(-marker, marker + 1):
                for dx in range(-marker, marker + 1):
                    ny, nx = py + dy, px + dx
                    if 0 <= ny < self.h and 0 <= nx < self.w:
                        pixels[nx, ny] = (0, 255, 0)

        # start/end labeled with cyan
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
    loader = MazeLoader("MAZE_1.png", template_dir="templates", template_threshold=0.55)

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

    total = (summary["death_pits"] + summary["confusion"] + summary["teleport_purple"] + summary["teleport_orange"] + summary["teleport_green"])
    print(f"Total hazardous cells: {total}")

    loader.visualize_hazards("maze_detected_hazards.png", base_image_path="MAZE_1.png")

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