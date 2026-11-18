"""Flat (homogeneous) ROI definitions and profile endpoints for each phantom.

These regions lie inside the phantom body but away from any inserts, edges, or
features, so they should contain a nearly uniform RSP value (~1.0 for
water-equivalent background).
"""

from pathlib import Path
from typing import TypedDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
GT_DIR = PROJECT_ROOT / "phantoms" / "ground_truths"


# ---------------------------------------------------------------------------
#  Catphan (ctp528, 880x880)
#  Line-pair inserts are in the upper ring. The flat ROI is in the centre
#  of the phantom.
# ---------------------------------------------------------------------------

CATPHAN_FLAT_ROI = (slice(390, 490), slice(390, 490))
CATPHAN_PROFILE_START = (50, 0)  # row, col within the ROI crop
CATPHAN_PROFILE_END = (50, 99)


# ---------------------------------------------------------------------------
#  Gammex (700x700, pixel size 0.5 mm)
#  Inserts are arranged in two rings. The flat ROI is in the phantom
#  centre, between the inner inserts.
#  50x50 pixels = 25x25 mm (same physical size as the other phantoms).
# ---------------------------------------------------------------------------

GAMMEX_FLAT_ROI = (slice(325, 375), slice(325, 375))
GAMMEX_PROFILE_START = (25, 0)
GAMMEX_PROFILE_END = (25, 49)


# ---------------------------------------------------------------------------
#  Spiral (880x880)
#  Spiral features are concentrated around the centre. The flat ROI is in the
#  gap between spiral arms, on the right-hand side of the phantom.
# ---------------------------------------------------------------------------

SPIRAL_FLAT_ROI = (slice(361, 461), slice(679, 779))
SPIRAL_PROFILE_START = (50, 0)
SPIRAL_PROFILE_END = (50, 99)


class Phantom(TypedDict):
    """Flat region of one phantom: where it is and the profile drawn through it."""

    name: str
    setup_prefix: str
    gt_path: Path
    roi: tuple[slice, slice]
    profile_start: tuple[int, int]
    profile_end: tuple[int, int]


PHANTOMS: list[Phantom] = [
    {
        "name": "Catphan",
        "setup_prefix": "ctp528",
        "gt_path": GT_DIR / "ctp528_gt_880pix.npy",
        "roi": CATPHAN_FLAT_ROI,
        "profile_start": CATPHAN_PROFILE_START,
        "profile_end": CATPHAN_PROFILE_END,
    },
    {
        "name": "Gammex",
        "setup_prefix": "gammex",
        "gt_path": GT_DIR / "gammex_gt_700pix.npy",
        "roi": GAMMEX_FLAT_ROI,
        "profile_start": GAMMEX_PROFILE_START,
        "profile_end": GAMMEX_PROFILE_END,
    },
    {
        "name": "Spiral",
        "setup_prefix": "spiral",
        "gt_path": GT_DIR / "spiral_gt_880pix.npy",
        "roi": SPIRAL_FLAT_ROI,
        "profile_start": SPIRAL_PROFILE_START,
        "profile_end": SPIRAL_PROFILE_END,
    },
]
