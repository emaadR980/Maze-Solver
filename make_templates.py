import os, sys
import cv2

CELL  = 16
WALL  = 2
INNER = CELL - 2 * WALL   # 14

OUT_DIR = "templates"
os.makedirs(OUT_DIR, exist_ok=True)

IMG_PATH = sys.argv[1] if len(sys.argv) > 1 else "maze-gamma/MAZE_1.png"

img = cv2.imread(IMG_PATH, cv2.IMREAD_COLOR)
if img is None:
    raise FileNotFoundError(f"Could not read {IMG_PATH}")

print(f"Source image : {IMG_PATH}")

display = img.copy()
count = {
    "confusion":       0,
    "death_pit":       0,
    "teleport_orange": 0,
    "teleport_green":  0,
    "teleport_purple": 0,
    "teleport_red":    0,
    "arrow_up":        0,
    "arrow_left":      0,
}

current_label = "death_pit"


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
        fname    = f"{current_label}_{count[current_label]:02d}_r{r}_c{c}.png"
        out_path = os.path.join(OUT_DIR, fname)
        cv2.imwrite(out_path, patch)
        count[current_label] += 1
        cv2.circle(display, (x, y), 6, (0, 0, 255), 2)
        print(f"Saved {out_path}  (cell r={r}, c={c})  "
              f"[total {current_label}: {count[current_label]}]")


cv2.namedWindow("maze")
cv2.setMouseCallback("maze", on_mouse)

LABEL_MAP = {
    ord("1"): "teleport_orange",
    ord("2"): "death_pit",
    ord("3"): "confusion",
    ord("4"): "teleport_green",
    ord("5"): "teleport_purple",
    ord("6"): "teleport_red",
    ord("7"): "arrow_up",
    ord("8"): "arrow_left",
}

print(f"\nActive label : {current_label}")
print("  1=teleport_orange  2=death_pit    3=confusion")
print("  4=teleport_green   5=teleport_purple  6=teleport_red")
print("  7=arrow_up         8=arrow_left")
print("  q=quit\n")

while True:
    # Show active label in window title
    cv2.setWindowTitle("maze", f"maze  |  label={current_label}  "
                                f"(saved: {count[current_label]})")
    cv2.imshow("maze", display)
    k = cv2.waitKey(10) & 0xFF
    if k == ord("q"):
        break
    elif k in LABEL_MAP:
        current_label = LABEL_MAP[k]
        print(f"Label → {current_label}")

cv2.destroyAllWindows()
print("\nDone. Templates saved in ./templates/")
print("Counts:", {k: v for k, v in count.items() if v > 0})