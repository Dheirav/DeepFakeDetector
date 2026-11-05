"""Input quality gate, in front of the mask-head model.

The abstain rule looks at the model's own confidence, and that does not move
when the input is degraded: across the compression grid, coverage stays at 50 to
55 percent while accuracy-when-answered falls from 0.958 to 0.705
(results/mask_head_clip448_balanced/degradation.json). A confident answer on a
tiny, heavily compressed image is therefore as likely as on a clean one, and
wrong far more often. This looks at the input instead.

It has to run on the raw upload. The model's own pre-processing squashes every
image to 512x512 and re-saves it as JPEG q90, which erases both the original
resolution and the original JPEG history.

Thresholds are read off the 7-cell degradation grid (n = 600 per cell, so about
two points of noise on accuracy-when-answered), where size and JPEG quality were
lowered together and so cannot be separated:

    clean 512 q90         0.958        256 / q80   0.846
    448 / q90             0.922        256 / q60   0.794
    384 / q75             0.901        224 / q50   0.705
    320 / q85             0.830

They separate the cells above into three bands. They are coarse on purpose and
are not fitted; with a different model, rerun mask_head_degradation.py and
re-read them. Blur is not gated: the grid never measured it, and a threshold
with no measurement behind it would be a guess.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

OK, WARN, POOR = "ok", "warn", "poor"

# Short side in pixels, and estimated JPEG quality.
WARN_SHORT_SIDE, POOR_SHORT_SIDE = 448, 256
WARN_JPEG_Q, POOR_JPEG_Q = 85, 65

# accuracy-when-answered at tau 0.9 for the cells in each band, same file as above
_WARN_RANGE = "0.83 to 0.92"
_POOR_RANGE = "0.71 to 0.79"
_CLEAN = "0.96"

# JPEG Annex K luminance table at quality 50, row-major.
_STD_LUMA = np.array([
    16, 11, 10, 16, 24, 40, 51, 61,
    12, 12, 14, 19, 26, 58, 60, 55,
    14, 13, 16, 24, 40, 57, 69, 56,
    14, 17, 22, 29, 51, 87, 80, 62,
    18, 22, 37, 56, 68, 109, 103, 77,
    24, 35, 55, 64, 81, 104, 113, 92,
    49, 64, 78, 87, 103, 121, 120, 101,
    72, 92, 95, 98, 112, 100, 103, 99])


def _table_for_quality(q: int) -> np.ndarray:
    scale = 5000 / q if q < 50 else 200 - 2 * q
    return np.clip((_STD_LUMA * scale + 50) // 100, 1, 255)


def estimate_jpeg_quality(tables: Optional[dict]) -> Optional[int]:
    """Quality 1 to 100 the luminance table was scaled from, or None.

    Compares sorted values so it does not depend on whether the library hands
    the table back in zigzag or natural order. Returns None when no standard
    scaling fits well: many cameras and editors write their own tables, and
    guessing a number for those would put a false reading in front of the user.
    """
    if not tables or 0 not in tables:
        return None
    obs = np.sort(np.asarray(list(tables[0]), dtype=float))
    if obs.size != 64:
        return None
    best_q, best_err = None, float("inf")
    for q in range(1, 101):
        err = np.abs(np.sort(_table_for_quality(q)) - obs).mean()
        if err < best_err:
            best_q, best_err = q, err
    return best_q if best_err <= 1.5 else None


def read_source(source) -> Tuple[int, Optional[int]]:
    """(short side in pixels, estimated JPEG quality or None) of a path or an
    open file object, without decoding the pixels. A file object is rewound."""
    im = Image.open(source)
    short = min(im.size)
    q = estimate_jpeg_quality(getattr(im, "quantization", None)) if im.format == "JPEG" else None
    # Never im.close() a caller's file object: Pillow closes the stream with it,
    # and the UI reads the upload again afterwards. A path is closed on collection.
    if hasattr(source, "seek"):
        source.seek(0)
    return short, q


@dataclass
class QualityReport:
    level: str
    short_side: int
    jpeg_quality: Optional[int]
    reasons: List[str] = field(default_factory=list)

    @property
    def note(self) -> Optional[str]:
        if self.level == OK:
            return None
        what = "; ".join(self.reasons)
        if self.level == POOR:
            return (f"Input quality too low to trust a verdict ({what}). On the degradation "
                    f"grid, where size and JPEG quality were lowered together, accuracy when "
                    f"the model answered fell to {_POOR_RANGE} at this level, against "
                    f"{_CLEAN} on clean input. The verdict is withheld.")
        return (f"Reduced input quality ({what}). On the degradation grid, accuracy when the "
                f"model answered was {_WARN_RANGE} at comparable levels, against {_CLEAN} on "
                f"clean input. Treat the verdict with more caution.")


def assess(short_side: int, jpeg_quality: Optional[int] = None) -> QualityReport:
    found = []
    if short_side < POOR_SHORT_SIDE:
        found.append((POOR, f"short side {short_side}px, under {POOR_SHORT_SIDE}px"))
    elif short_side < WARN_SHORT_SIDE:
        found.append((WARN, f"short side {short_side}px, under {WARN_SHORT_SIDE}px"))
    if jpeg_quality is not None:
        if jpeg_quality < POOR_JPEG_Q:
            found.append((POOR, f"JPEG quality about {jpeg_quality}, under {POOR_JPEG_Q}"))
        elif jpeg_quality < WARN_JPEG_Q:
            found.append((WARN, f"JPEG quality about {jpeg_quality}, under {WARN_JPEG_Q}"))
    rank = {OK: 0, WARN: 1, POOR: 2}
    level = max((lv for lv, _ in found), key=rank.get, default=OK)
    return QualityReport(level, short_side, jpeg_quality, [why for _, why in found])


def gate(verdict: str, report: QualityReport, abstain_label: str) -> Tuple[str, Optional[str]]:
    """Apply the report to a verdict. Poor input withholds it; reduced input
    keeps it and returns a note to show beside it."""
    if report.level == POOR:
        return abstain_label, report.note
    return verdict, report.note
