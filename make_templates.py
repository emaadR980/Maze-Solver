# allows you to click on hazards in the maze and save them as template

import os
import sys
import cv2

CELL = 16
WALL = 2
INNER = CELL - 2 * WALL  # 14

OUT_DIR = "templates"
os.makedirs(OUT_DIR, exist_ok=True)

IMG_PATH = sys.argv[1] if len(sys.argv) > 1 else "maze-alpha/MAZE_1.png"

img = cv2.imread(IMG_PATH, cv2.IMREAD_COLOR)
if img is None:
    raise FileNotFoundError(f"Could not read {IMG_PATH}")

print(f"Using source image: {IMG_PATH}")

display = img.copy()

count = {
    "confusion": 0,
    "death_pit": 0,

    "teleport_orange_pad": 0,
    "teleport_orange_dest": 0,

    "teleport_green_pad": 0,
    "teleport_green_dest": 0,

    "teleport_purple_pad": 0,
    "teleport_purple_dest": 0,

    "teleport_red_pad": 0,
    "teleport_red_dest": 0,
}

current_label = "teleport_orange_pad"

def crop_cell_interior(x, y):
    cell_c = x // CELL
    cell_r = y // CELL
    x0 = cell_c * CELL + WALL
    y0 = cell_r * CELL + WALL
    patch = img[y0:y0 + INNER, x0:x0 + INNER].copy()
    return patch, cell_r, cell_c

def on_mouse(event, x, y, flags, param):
    global display
    if event == cv2.EVENT_LBUTTONDOWN:
        patch, r, c = crop_cell_interior(x, y)
        fname = f"{current_label}_{count[current_label]:02d}_r{r}_c{c}.png"
        out_path = os.path.join(OUT_DIR, fname)
        cv2.imwrite(out_path, patch)
        count[current_label] += 1

        cv2.circle(display, (x, y), 6, (0, 0, 255), 2)
        print(f"Saved {out_path} from cell (r={r}, c={c})")

cv2.namedWindow("maze")
cv2.setMouseCallback("maze", on_mouse)

print("Controls:")
print("  1 = teleport_orange_pad     2 = teleport_orange_dest")
print("  3 = teleport_green_pad      4 = teleport_green_dest")
print("  5 = teleport_purple_pad     6 = teleport_purple_dest")
print("  7 = teleport_red_pad        8 = teleport_red_dest")
print("  9 = death_pit               0 = confusion")
print("  q = quit")

while True:
    cv2.imshow("maze", display)
    k = cv2.waitKey(10) & 0xFF

    if k == ord("q"):
        break
    elif k == ord("1"):
        current_label = "teleport_orange_pad"; print("Label: teleport_orange_pad")
    elif k == ord("2"):
        current_label = "teleport_orange_dest"; print("Label: teleport_orange_dest")
    elif k == ord("3"):
        current_label = "teleport_green_pad"; print("Label: teleport_green_pad")
    elif k == ord("4"):
        current_label = "teleport_green_dest"; print("Label: teleport_green_dest")
    elif k == ord("5"):
        current_label = "teleport_purple_pad"; print("Label: teleport_purple_pad")
    elif k == ord("6"):
        current_label = "teleport_purple_dest"; print("Label: teleport_purple_dest")
    elif k == ord("7"):
        current_label = "teleport_red_pad"; print("Label: teleport_red_pad")
    elif k == ord("8"):
        current_label = "teleport_red_dest"; print("Label: teleport_red_dest")
    elif k == ord("9"):
        current_label = "death_pit"; print("Label: death_pit")
    elif k == ord("0"):
        current_label = "confusion"; print("Label: confusion")

cv2.destroyAllWindows()
print("Done. Templates saved in ./templates/")