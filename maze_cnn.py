"""
maze_cnn.py — importable module extracted from maze_cnn.ipynb
Provides CnnMazeLoader and MazeCellCNN for use in environment.py.

Usage in environment.py:
    from maze_cnn import CnnMazeLoader
    self.loader = CnnMazeLoader(maze_image_path)
"""
import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from maze import MazeLoader

CELL_SIZE  = 16
WALL_THICK = 2
INNER_SIZE = CELL_SIZE - 2 * WALL_THICK  # 12
MODEL_PATH = "maze_cnn.pt"

CLASS_NAMES = [
    "path", "wall", "death_pit", "confusion",
    "tp_orange", "tp_green", "tp_purple", "tp_red",
    "start", "goal"
]
NUM_CLASSES = len(CLASS_NAMES)

LABEL_PATH, LABEL_WALL, LABEL_FIRE  = 0, 1, 2
LABEL_CONF, LABEL_TPO, LABEL_TPG    = 3, 4, 5
LABEL_TPP,  LABEL_TPR               = 6, 7
LABEL_START, LABEL_GOAL             = 8, 9


class MazeCellCNN(nn.Module):
    """Tiny CNN: (B,3,12,12) → (B, NUM_CLASSES)."""

    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,  32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 3 * 3, 128), nn.ReLU(inplace=True), nn.Dropout(0.4),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.head(self.features(x))

    def predict(self, patch_bgr: np.ndarray):
        """Classify a single 12×12 BGR patch. Returns (class_idx, confidence)."""
        rgb = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        t   = torch.from_numpy(np.transpose(rgb, (2, 0, 1))).unsqueeze(0)
        with torch.no_grad():
            probs = F.softmax(self(t), dim=1)
            conf, cls = probs.max(dim=1)
        return int(cls.item()), float(conf.item())

    def save(self, path: str = MODEL_PATH):
        torch.save(self.state_dict(), path)
        print(f"[MazeCellCNN] saved → {path}")

    @classmethod
    def load(cls, path: str = MODEL_PATH) -> "MazeCellCNN":
        m = cls()
        m.load_state_dict(torch.load(path, map_location="cpu"))
        m.eval()
        return m


class CnnMazeLoader(MazeLoader):
    """
    Drop-in replacement for MazeLoader.
    Uses the trained CNN (maze_cnn.pt) for hazard detection.
    Falls back to the color classifier on low confidence.
    Also detects tp_red (red teleport pads) as a new hazard type.

    Two known CNN failure modes are corrected by _color_veto():
      1. tp_red classified as death_pit — both are red, model confuses them.
         Discriminator: tp_red fills the cell solidly (high red pixel fraction,
         low dark fraction); fire pits are sparse red/orange on a white background.
      2. Confusion-pad bleed — cells immediately adjacent to a confusion pad
         sometimes pick up its yellow-orange swirl, causing a false fire call.
         Discriminator: predominantly orange/yellow with very little true red.
    """

    _CLS_MAP = {
        LABEL_FIRE: "death_pit",
        LABEL_CONF: "confusion",
        LABEL_TPO:  "teleport_orange",
        LABEL_TPG:  "teleport_green",
        LABEL_TPP:  "teleport_purple",
        LABEL_TPR:  "teleport_red",
    }
    # Raised from 0.55 → 0.65 to reduce marginal false positives overall.
    CONF_THRESHOLD = 0.65

    def __init__(self, image_path: str, model_path: str = MODEL_PATH, **kwargs):
        super().__init__(image_path, **kwargs)
        self.teleport_red = []   # new hazard list not in base MazeLoader

        if os.path.exists(model_path):
            self._cnn = MazeCellCNN.load(model_path)
            print(f"[CnnMazeLoader] CNN loaded from {model_path}")
        else:
            self._cnn = None
            print(f"[CnnMazeLoader] {model_path} not found — color classifier only")

    # ── Color-based veto ──────────────────────────────────────────────────────
    @staticmethod
    def _color_veto(patch_bgr: np.ndarray, cnn_label: int) -> int:
        """
        Inspect pixel colors in HSV to correct two CNN misclassification patterns.
        Returns the (possibly corrected) label index.

        HSV hue is in OpenCV scale 0-180 (half of 360°):
          red   : hue  0-10  or  170-180  (wraps at 0)
          orange: hue 10-25
          yellow: hue 25-40
        """
        hsv = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
        h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        total = h.size

        # Chromatic red pixels (hue wraps near 0 and 180)
        red_mask    = ((h <= 10) | (h >= 170)) & (s > 120) & (v > 80)
        # Orange-yellow pixels (confusion pad / bleed range)
        orange_mask = (h > 10) & (h < 40)     & (s > 100) & (v > 80)
        # Dark / background pixels
        dark_mask   = v < 50

        red_frac    = red_mask.sum()    / total
        orange_frac = orange_mask.sum() / total
        dark_frac   = dark_mask.sum()   / total

        if cnn_label == LABEL_FIRE:
            # ── tp_red vs fire ───────────────────────────────────────────────
            # tp_red: solid block — high red coverage, low background fraction.
            # fire:   sparse orange-red cluster on a mostly-white cell.
            if red_frac > 0.30 and dark_frac < 0.25 and orange_frac < 0.10:
                return LABEL_TPR

            # ── confusion-pad bleed vs fire ──────────────────────────────────
            # Bleed from an adjacent confusion pad leaves orange/yellow pixels
            # but almost no true red.  Treat as plain path.
            if orange_frac > 0.20 and red_frac < 0.12:
                return LABEL_PATH

        # tp_red second-pass: CNN said tp_red but the patch is mostly dark/orange
        # (shouldn't really happen but guards the detect_hazards loop too).
        if cnn_label == LABEL_TPR:
            if red_frac < 0.20:
                return LABEL_PATH

        return cnn_label

    # ── classify_cell_template ────────────────────────────────────────────────
    def classify_cell_template(self, r: int, c: int):
        if self._cnn is not None:
            patch      = self.cell_interior_bgr(r, c)
            cls, conf  = self._cnn.predict(patch)

            if cls in self._CLS_MAP and conf >= self.CONF_THRESHOLD:
                cls = self._color_veto(patch, cls)   # apply color correction
                if cls in self._CLS_MAP:
                    return self._CLS_MAP[cls]
                # veto downgraded to LABEL_PATH → fall through to base classifier

        return super().classify_cell_template(r, c)

    # ── detect_hazards ────────────────────────────────────────────────────────
    def detect_hazards(self):
        summary = super().detect_hazards()

        # Normalise to plain tuples so set-membership and list.remove() work
        # regardless of whether the base loader stored numpy arrays or lists.
        self.death_pits = [tuple(cell) for cell in self.death_pits]

        # Second pass for tp_red (not in parent's bucket logic)
        self.teleport_red = []
        if self._cnn is not None:
            for r in range(self.maze_height_cells):
                for c in range(self.maze_width_cells):
                    patch      = self.cell_interior_bgr(r, c)
                    cls, conf  = self._cnn.predict(patch)
                    if conf >= self.CONF_THRESHOLD:
                        cls = self._color_veto(patch, cls)
                    if cls == LABEL_TPR and conf >= self.CONF_THRESHOLD:
                        self.teleport_red.append((r, c))

        # ── Spatial cleanup 1: remove fire pits that overlap teleport pads ───
        # The CNN sometimes double-classifies an orange/purple/green teleport
        # pad as a death_pit.  Teleport identity wins — evict from death_pits.
        teleport_cells = (
            set(map(tuple, self.teleport_orange))
            | set(map(tuple, self.teleport_purple))
            | set(map(tuple, self.teleport_green))
            | set(map(tuple, self.teleport_red))
        )
        tp_overlap = [cell for cell in self.death_pits if cell in teleport_cells]
        if tp_overlap:
            print(f"[CnnMazeLoader] removing {len(tp_overlap)} death_pit(s) "
                  f"overlapping teleport pads: {tp_overlap}")
            for cell in tp_overlap:
                self.death_pits.remove(cell)

        # ── Spatial cleanup 2: remove fire pits adjacent to confusion pads ───
        # A death_pit cell that shares an edge with a confusion pad is almost
        # certainly a colour-bleed artefact from the swirl graphic, not a real
        # hazard.  The maze never intentionally places fire directly touching a
        # confusion pad.
        confusion_set = set(map(tuple, self.confusion_pads))
        if confusion_set and self.death_pits:
            neighbours = {(r + dr, c + dc)
                          for r, c in confusion_set
                          for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))}
            bleed = [cell for cell in self.death_pits if cell in neighbours]
            if bleed:
                print(f"[CnnMazeLoader] removing {len(bleed)} bleed fire-pit(s) "
                      f"adjacent to confusion pads: {bleed}")
                for cell in bleed:
                    self.death_pits.remove(cell)
            summary["death_pits"] = len(self.death_pits)

        summary["teleport_red"] = len(self.teleport_red)
        return summary
