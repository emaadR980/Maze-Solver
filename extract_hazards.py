import cv2, os
from maze import MazeLoader

loader = MazeLoader("maze-beta/MAZE_1.png")  # your new maze filename
loader.detect_hazards()

os.makedirs("templates", exist_ok=True)

# Print all detected death pits so you can verify
print("Detected death pits:", loader.death_pits)

# Extract and save a template from each detected fire cell
for i, (r, c) in enumerate(loader.death_pits):
    patch = loader.cell_interior_bgr(r, c)
    path = f"templates/death_pit_{i:02d}_r{r}_c{c}.png"
    cv2.imwrite(path, patch)
    print(f"Saved {path}")