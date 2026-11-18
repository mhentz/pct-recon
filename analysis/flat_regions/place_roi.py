"""Place the flat-region ROIs by dragging them on the ground truths.

Each ground truth is shown with a 100x100 ROI, red while it overlaps an insert and
green once it lies in a flat region. When the window is closed, the slices of the
final ROIs are printed, to be copied into constants.py.

Usage, from the repository root:

    PYTHONPATH=. python analysis/flat_regions/place_roi.py
"""

from pathlib import Path
from typing import TypedDict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches
from matplotlib.backend_bases import MouseEvent

from analysis.flat_regions.constants import GT_DIR

plt.style.use(str(Path(__file__).resolve().parent.parent / "style.txt"))


class RoiPhantom(TypedDict):
    """Ground truth image of one phantom and the side of its square ROI in pixels."""

    name: str
    gt: Path
    size: int


PHANTOMS: list[RoiPhantom] = [
    {"name": "Catphan", "gt": GT_DIR / "ctp528_gt_880pix.npy", "size": 100},
    {"name": "Gammex", "gt": GT_DIR / "gammex_gt_700pix.npy", "size": 100},
    {"name": "Spiral", "gt": GT_DIR / "spiral_gt_880pix.npy", "size": 100},
]


class DraggableRect:
    """ROI rectangle on a ground truth image, moved with the mouse."""

    def __init__(self, rect: patches.Rectangle, gt: np.ndarray, ax: plt.Axes, name: str) -> None:
        """Attach the mouse handlers to a rectangle drawn on a ground truth image.

        Args:
            rect: Rectangle to drag.
            gt: Ground truth image the rectangle is drawn on.
            ax: Axes holding the image.
            name: Phantom name shown in the title.
        """
        self.rect = rect
        self.gt = gt
        self.ax = ax
        self.name = name
        self.press: tuple[float, float, float, float] | None = None
        self.bg = None
        self._connect()

    def _connect(self) -> None:
        self.cid_press = self.rect.figure.canvas.mpl_connect("button_press_event", self.on_press)
        self.cid_release = self.rect.figure.canvas.mpl_connect("button_release_event", self.on_release)
        self.cid_motion = self.rect.figure.canvas.mpl_connect("motion_notify_event", self.on_motion)

    def on_press(self, event: MouseEvent) -> None:
        """Start a drag if the press is inside the rectangle."""
        if event.inaxes != self.ax:
            return
        contains, _ = self.rect.contains(event)
        if not contains or event.xdata is None or event.ydata is None:
            return
        x0, y0 = self.rect.get_xy()
        self.press = (x0, y0, event.xdata, event.ydata)

    def on_motion(self, event: MouseEvent) -> None:
        """Move the rectangle with the mouse during a drag."""
        if self.press is None or event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return
        x0, y0, xpress, ypress = self.press
        dx = event.xdata - xpress
        dy = event.ydata - ypress
        self.rect.set_xy((x0 + dx, y0 + dy))
        self._update_title()
        self.rect.figure.canvas.draw_idle()

    def on_release(self, event: MouseEvent) -> None:  # noqa: ARG002
        """End the drag."""
        if self.press is None:
            return
        self.press = None
        self._update_title()
        self.rect.figure.canvas.draw_idle()

    def _update_title(self) -> None:
        """Colour the rectangle and set the title from the ROI position and flatness."""
        x, y = self.rect.get_xy()
        col, row = round(x), round(y)
        size = int(self.rect.get_width())
        roi = self.gt[row : row + size, col : col + size]
        std = roi.std() if roi.size > 0 else float("nan")
        is_flat = std == 0.0
        status = "FLAT" if is_flat else f"NOT FLAT  std={std:.4f}"

        colour = "green" if is_flat else "red"
        self.rect.set_edgecolor(colour)
        self.rect.set_facecolor(colour)

        self.ax.set_title(
            f"{self.name}  [{row}:{row + size}, {col}:{col + size}]  ({status})",
            color="green" if is_flat else "red",
            fontweight="bold",
        )

    def get_slice(self) -> str:
        """Return the ROI as a ``(slice(...), slice(...))`` string for constants.py."""
        x, y = self.rect.get_xy()
        col, row = round(x), round(y)
        size = int(self.rect.get_width())
        return f"(slice({row}, {row + size}), slice({col}, {col + size}))"


def main() -> None:
    """Show the three ground truths, let the ROIs be dragged and print their final slices."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Drag each ROI rectangle, then close this window", fontsize=13)

    rects: list[DraggableRect] = []

    for ax, ph in zip(axes, PHANTOMS, strict=False):
        gt = np.load(ph["gt"])
        ax.imshow(gt, cmap="gray", vmin=0.0, vmax=1.5)
        ax.set_xticks([])
        ax.set_yticks([])

        # Start at the image centre
        h, w = gt.shape
        size = ph["size"]
        r0, c0 = h // 2 - size // 2, w // 2 - size // 2

        rect = patches.Rectangle(
            (c0, r0),
            size,
            size,
            linewidth=2,
            edgecolor="red",
            facecolor="red",
            alpha=0.25,
            linestyle="--",
        )
        ax.add_patch(rect)

        dr = DraggableRect(rect, gt, ax, ph["name"])
        dr._update_title()  # noqa: SLF001
        rects.append(dr)

    plt.tight_layout()
    plt.show()

    print("ROI slices for constants.py:")
    for dr in rects:
        print(f"  {dr.name:10s}  {dr.get_slice()}")


if __name__ == "__main__":
    main()
