from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Dict

_DOT_R = 3

_PATH_COLOUR    = (33,  150, 243)   # blue
_AGENT_COLOUR   = (255, 255, 255)   # white
_GOAL_COLOUR    = (0,   230, 118)   # green
_START_COLOUR   = (255, 140,   0)   # orange
_HAZARD_COLOUR  = {
    'death':          (180,   0,   0),   # dark red  — static death pit
    'death_rotating': (255, 140,  30),   # orange    — rotating fire pit
    'confusion':      (253, 216,  53),   # yellow
    'teleport':       (171,  71, 188),   # purple
    'teleport_red':   (220,   0, 255),   # violet/magenta — red teleport pad
}


def _cell_centre(env, r: int, c: int) -> Tuple[int, int]:
    cell = env.CELL_SIZE
    px = min(c * cell + cell // 2, env.loader.w - 1)
    py = min(r * cell + cell // 2, env.loader.h - 1)
    return px, py     # PIL / matplotlib: x = col-direction, y = row-direction


def _dot(draw: ImageDraw.ImageDraw, xy: Tuple[int,int],
         colour: Tuple[int,int,int], r: int = _DOT_R) -> None:
    x, y = xy
    draw.ellipse([x - r, y - r, x + r, y + r], fill=colour)


def _agent_dot(draw: ImageDraw.ImageDraw, xy: Tuple[int, int]) -> None:
    _dot(draw, xy, (0, 0, 0), r=6)
    _dot(draw, xy, _AGENT_COLOUR, r=4)


def _draw_path(
    draw:   ImageDraw.ImageDraw,
    env,
    path:   List[Tuple[int,int]],
    colour: Tuple[int,int,int] = _PATH_COLOUR,
    width:  int = 2,
) -> None:
    """Draw lines connecting consecutive cell centres so the path looks continuous."""
    if len(path) < 2:
        if path:
            _dot(draw, _cell_centre(env, *path[0]), colour)
        return
    pts = [_cell_centre(env, r, c) for r, c in path]
    draw.line(pts, fill=colour, width=width)

class LiveVisualizer:
    def __init__(self, env, title: str = "Maze Agent – Live",
                 update_every: int = 3,
                 delay: float = 0.05):
        self.env          = env
        self.update_every = update_every
        self.delay        = delay
        self._step        = 0
        self._title       = title
        self._last_paint_idx = 0

        self._painted: Image.Image = env.loader.img.copy().convert("RGB")

        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.im = self.ax.imshow(np.array(self._painted), origin='upper')
        self.ax.axis('off')

        self._agent_artist = None

        self.stats_text = self.ax.text(
            0.01, 0.01, '', transform=self.ax.transAxes,
            fontsize=8.5, color='white', verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.65), zorder=10,
        )

        self.ax.set_title(title, fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.05)

    def reset_episode(self, env=None) -> None:
        if env:
            self.env = env
        self._painted = self.env.loader.img.copy().convert("RGB")
        self._last_paint_idx = 0

    def update(
        self,
        known:       Dict[Tuple[int,int], str],
        current_pos: Optional[Tuple[int,int]],
        path:        List[Tuple[int,int]],
        episode:     int,
        turn:        int,
        goal_pos:    Optional[Tuple[int,int]] = None,
        start_pos:   Optional[Tuple[int,int]] = None,
        extra_stats: Optional[Dict] = None,
    ) -> None:
        self._step += 1
        if self._step % self.update_every != 0:
            return

        draw = ImageDraw.Draw(self._painted)

        new_segment = path[max(0, self._last_paint_idx - 1):]
        _draw_path(draw, self.env, new_segment)
        self._last_paint_idx = len(path)

        for (r, c), ct in known.items():
            col = _HAZARD_COLOUR.get(ct)
            if col:
                _dot(draw, _cell_centre(self.env, r, c), col, r=_DOT_R + 1)

        if start_pos:
            _dot(draw, _cell_centre(self.env, *start_pos), _START_COLOUR, r=5)
        if goal_pos:
            _dot(draw, _cell_centre(self.env, *goal_pos), _GOAL_COLOUR, r=6)

        display = self._painted.copy()
        disp_draw = ImageDraw.Draw(display)
        rotating = set()
        for cl in self.env.fire_clusters:
            rotating.update(cl)
        for r, c in self.env.death_pits:
            color = _HAZARD_COLOUR['death_rotating'] if (r, c) in rotating else _HAZARD_COLOUR['death']
            _dot(disp_draw, _cell_centre(self.env, r, c), color, r=_DOT_R + 2)
        if current_pos:
            _agent_dot(disp_draw, _cell_centre(self.env, *current_pos))

        self.im.set_data(np.array(display))

        self.ax.set_title(f"Episode {episode}  |  Turn {turn}",
                          fontsize=13, fontweight='bold')
        if extra_stats:
            self.stats_text.set_text(
                "\n".join(f"{k}: {v}" for k, v in extra_stats.items()))

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        plt.pause(self.delay)

    def close(self):
        plt.ioff()
        plt.close(self.fig)
