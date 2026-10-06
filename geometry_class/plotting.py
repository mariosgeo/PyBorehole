"""
plotting.py
===========

Drawing helpers shared by the run scripts.
"""

from typing import Optional, Sequence

import numpy as np
import matplotlib.patches as patches


def draw_tools_in_borehole(
    ax,
    depth_a: float,
    borehole_diameter: float,
    backfill_color,
    formation_color,
    tool_diameter: Optional[float] = None,
    normal_spacings: Sequence[float] = (),
    lateral_spacings: Sequence[float] = (),
    lateral_mn: float = 0.0,
    em_tx_depth: Optional[float] = None,
    em_spacings: Sequence[float] = (),
    em_bucking: Sequence[Optional[float]] = (),
):
    """To-scale side view of the ERT sonde and the EM tool in the borehole at one depth.

    The ERT sonde (left hole) is drawn with its current electrode A at
    ``depth_a`` and the potential electrodes above it; the EM tool (right hole)
    with its transmitter at ``em_tx_depth`` and the receivers below it.
    Depths are positive downward and plotted as negative y, like the logs.
    Returns the (top, bottom) depth range shown.
    """
    radius = borehole_diameter / 2.0
    gap = 1.6 * borehole_diameter            # distance between the two hole drawings
    x_ert, x_em = 0.0, borehole_diameter + gap

    # Depth window: everything on both tools plus a margin
    depths = [depth_a] + [depth_a - s for s in normal_spacings]
    depths += [depth_a - s - lateral_mn / 2.0 for s in lateral_spacings]
    if em_tx_depth is not None:
        depths += [em_tx_depth] + [em_tx_depth + s for s in em_spacings]
    top, bottom = min(depths) - 0.25, max(depths) + 0.25

    # Formation, then the two backfilled holes
    ax.add_patch(patches.Rectangle((x_ert - radius - 0.6 * gap, -bottom), x_em - x_ert + 2 * radius + 1.2 * gap,
                                   bottom - top, facecolor=formation_color, edgecolor="none", zorder=0))
    for x in (x_ert, x_em):
        ax.add_patch(patches.Rectangle((x - radius, -bottom), 2 * radius, bottom - top,
                                       facecolor=backfill_color, edgecolor="k", lw=0.8, zorder=1))

    label_box = dict(facecolor="white", alpha=0.85, edgecolor="none", boxstyle="round,pad=0.15")

    # ERT sonde: insulating body and ring electrodes (drawn on both sides of the body)
    t = 0.0 if tool_diameter is None else tool_diameter / 2.0
    if tool_diameter is not None:
        ax.add_patch(patches.Rectangle((x_ert - t, -bottom), 2 * t, bottom - top,
                                       facecolor="0.55", edgecolor="0.25", lw=0.6, zorder=2))
    ring_h, ring_w = 0.03, max(0.25 * radius, 0.01)

    def ring(depth, label, color):
        for side in (-1, 1):
            x0 = x_ert + side * t - (ring_w if side < 0 else 0.0)
            ax.add_patch(patches.Rectangle((x0, -depth - ring_h / 2), ring_w, ring_h,
                                           facecolor=color, edgecolor="k", lw=0.4, zorder=3))
        ax.text(x_ert - radius - 0.08 * gap, -depth, label, ha="right", va="center", fontsize=7,
                bbox=label_box, zorder=4)

    ring(depth_a, "A", "#d62728")
    for s in normal_spacings:
        ring(depth_a - s, f'M ({s / 0.0254:.0f}" normal)', "#1f77b4")
    for s in lateral_spacings:
        ring(depth_a - s + lateral_mn / 2.0, "M (lateral)", "#2ca02c")
        ring(depth_a - s - lateral_mn / 2.0, "N (lateral)", "#2ca02c")
    ax.text(x_ert, -top + 0.05, "ERT sonde", ha="center", va="bottom", fontsize=8, fontweight="bold")

    # EM tool: coils on the axis (the non-conductive tool body is not modelled)
    if em_tx_depth is not None:
        coil_w = 0.7 * radius

        def coil(depth, label, color):
            ax.add_patch(patches.Ellipse((x_em, -depth), 2 * coil_w, 0.035, facecolor="none",
                                         edgecolor=color, lw=2.0, zorder=3))
            ax.text(x_em + radius + 0.08 * gap, -depth, label, ha="left", va="center", fontsize=7,
                    bbox=label_box, zorder=4)

        coil(em_tx_depth, "TX", "#d62728")
        for s, b in zip(em_spacings, list(em_bucking) + [None] * (len(em_spacings) - len(em_bucking))):
            coil(em_tx_depth + s, f"RX ({s:g} m)", "#1f77b4")
            if b is not None:
                coil(em_tx_depth + b, f"bucking ({b:g} m)", "#9467bd")
        ax.text(x_em, -top + 0.05, "EM tool", ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_xlim(x_ert - radius - 0.6 * gap, x_em + radius + 0.6 * gap)
    ax.set_ylim(-bottom, -top + 0.25)
    ax.set_aspect("equal")
    ax.set_xticks([x_ert - radius, x_ert + radius])
    ax.set_xticklabels([f"{-radius:.2f}", f"{radius:.2f}"], fontsize=7)
    ax.set_xlabel("x (m), to scale", fontsize=8)
    ax.tick_params(axis="y", labelsize=8)
    ax.set_title("Tools in the hole", fontsize=10)
    return top, bottom
