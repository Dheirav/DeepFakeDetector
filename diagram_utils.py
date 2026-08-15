"""
Minimal Matplotlib-based diagram utilities (no external Graphviz binaries).

This module is intentionally dependency-light (matplotlib only) and produces
high-resolution (300 DPI) PNGs suitable for thesis figures.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple


@dataclass(frozen=True)
class NodeStyle:
    facecolor: str = "#FFFFFF"
    edgecolor: str = "#222222"
    linewidth: float = 1.2
    boxstyle: str = "round,pad=0.25,rounding_size=0.08"
    fontsize: int = 10


@dataclass(frozen=True)
class EdgeStyle:
    color: str = "#222222"
    linewidth: float = 1.1
    arrowsize: float = 12
    linestyle: str = "solid"  # "solid" or "dashed"
    fontsize: int = 8


def _mpl():
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    return plt, FancyArrowPatch, FancyBboxPatch


def new_figure(figsize: Tuple[float, float] = (12.0, 6.0), dpi: int = 300):
    plt, _, _ = _mpl()
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return fig, ax


def draw_node(
    ax,
    node_id: str,
    label: str,
    xy: Tuple[float, float],
    wh: Tuple[float, float] = (0.18, 0.10),
    style: Optional[NodeStyle] = None,
):
    _, _, FancyBboxPatch = _mpl()
    style = style or NodeStyle()
    x, y = xy
    w, h = wh
    rect = FancyBboxPatch(
        (x - w / 2, y - h / 2),
        w,
        h,
        boxstyle=style.boxstyle,
        linewidth=style.linewidth,
        edgecolor=style.edgecolor,
        facecolor=style.facecolor,
        zorder=2,
    )
    ax.add_patch(rect)
    ax.text(
        x,
        y,
        label,
        ha="center",
        va="center",
        fontsize=style.fontsize,
        color="#111111",
        zorder=3,
    )
    return {"id": node_id, "x": x, "y": y, "w": w, "h": h}


def _anchor(n, side: str) -> Tuple[float, float]:
    if side == "right":
        return (n["x"] + n["w"] / 2, n["y"])
    if side == "left":
        return (n["x"] - n["w"] / 2, n["y"])
    if side == "top":
        return (n["x"], n["y"] + n["h"] / 2)
    if side == "bottom":
        return (n["x"], n["y"] - n["h"] / 2)
    raise ValueError(f"Unknown side: {side}")


def draw_edge(
    ax,
    src,
    dst,
    label: Optional[str] = None,
    src_side: str = "right",
    dst_side: str = "left",
    style: Optional[EdgeStyle] = None,
):
    _, FancyArrowPatch, _ = _mpl()
    style = style or EdgeStyle()
    x1, y1 = _anchor(src, src_side)
    x2, y2 = _anchor(dst, dst_side)
    arrow = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle="-|>",
        mutation_scale=style.arrowsize,
        linewidth=style.linewidth,
        linestyle=style.linestyle,
        color=style.color,
        zorder=1,
        shrinkA=4,
        shrinkB=4,
    )
    ax.add_patch(arrow)

    if label:
        # place label at midpoint with a slight vertical offset
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(
            mx,
            my + 0.02,
            label,
            ha="center",
            va="bottom",
            fontsize=style.fontsize,
            color="#111111",
            zorder=4,
        )


def draw_group_label(ax, label: str, xy: Tuple[float, float], fontsize: int = 11):
    ax.text(xy[0], xy[1], label, ha="center", va="center", fontsize=fontsize, color="#111111")


def save_png(fig, path: str, dpi: int = 300):
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.25)

