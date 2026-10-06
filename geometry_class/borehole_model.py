"""
borehole_model.py
=================

Resistivity model shared by the ERT (``Geometry``) and EM (``EMGeometry``)
simulators, so both solvers see exactly the same earth.

Coordinates: ``r`` is the radial distance from the borehole axis (m) and
``depth`` is positive downward (m); the surface is depth 0.

The model consists of

* a 1-D layered formation ``[top_depth, bottom_depth, resistivity]``; the
  last layer extends to infinite depth,
* concentric borehole zones (backfill, casing, gravel pack, ...), each
  occupying the annulus between the previous zone's outer radius and its own
  outer radius over a depth interval,
* air above the surface (used by EM only; ERT has a no-flux surface instead).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class BoreholeZone:
    """One concentric borehole zone.

    Attributes:
        r_outer (float): Outer radius of the zone (m).  The zone fills the
            annulus from the previous zone's outer radius (or the axis) to here.
        resistivity (float): Zone resistivity (Ohm.m).
        top (float): Top depth of the zone (m, positive downward).
        bottom (Optional[float]): Bottom depth (m).  ``None`` = borehole bottom.
    """
    r_outer: float
    resistivity: float
    top: float = 0.0
    bottom: Optional[float] = None


def make_zones(
    borehole_diameter: float,
    borehole_length: float,
    backfill_resistivity: float,
    zones: Optional[Sequence] = None,
) -> List[BoreholeZone]:
    """Return a sorted, validated list of ``BoreholeZone``.

    With ``zones=None`` the whole borehole is a single backfill zone.  Zones
    may be given as ``BoreholeZone`` objects or tuples
    ``(r_outer, resistivity[, top[, bottom]])``.
    """
    radius = borehole_diameter / 2.0
    if zones is None:
        zones = [BoreholeZone(radius, backfill_resistivity, 0.0, borehole_length)]

    out = []
    for z in zones:
        if not isinstance(z, BoreholeZone):
            z = BoreholeZone(*z)
        bottom = borehole_length if z.bottom is None else float(z.bottom)
        out.append(BoreholeZone(float(z.r_outer), float(z.resistivity), float(z.top), bottom))
    out.sort(key=lambda z: z.r_outer)

    radii = [z.r_outer for z in out]
    if len(set(radii)) != len(radii):
        raise ValueError("Borehole zones must have distinct outer radii.")
    if radii[-1] > radius * (1 + 1e-9):
        raise ValueError(
            f"Outermost borehole zone (r={radii[-1]} m) exceeds the borehole radius ({radius} m)."
        )
    for z in out:
        if z.resistivity <= 0:
            raise ValueError("Zone resistivities must be positive.")
        if not (0.0 <= z.top < z.bottom <= borehole_length + 1e-9):
            raise ValueError(f"Zone depth interval [{z.top}, {z.bottom}] is outside the borehole.")
    return out


def formation_resistivity(depth, layers: Optional[np.ndarray], background: float) -> np.ndarray:
    """Formation resistivity at ``depth`` (array, positive downward).

    The last layer extends to infinite depth.  Without layers the formation is
    a homogeneous half-space of resistivity ``background``.
    """
    depth = np.atleast_1d(np.asarray(depth, dtype=float))
    rho = np.full(depth.shape, float(background))
    if layers is None or len(layers) == 0:
        return rho
    rho[:] = float(layers[-1, 2])
    # Paint from the bottom layer up so the upper layer wins on a shared interface
    for top, bottom, res in layers[::-1, :3]:
        rho[(depth >= top) & (depth < bottom)] = res
    return rho


def resistivity_rz(
    r,
    depth,
    layers: Optional[np.ndarray],
    background: float,
    zones: Sequence[BoreholeZone],
    include_borehole: bool = True,
    air_resistivity: Optional[float] = None,
) -> np.ndarray:
    """Resistivity at points ``(r, depth)``.

    ``include_borehole=False`` replaces every borehole zone by the formation,
    which gives the "no borehole" reference used to isolate the borehole effect.
    ``air_resistivity`` is applied above the surface (depth < 0) if given.
    """
    r, depth = np.broadcast_arrays(np.atleast_1d(np.asarray(r, dtype=float)),
                                   np.atleast_1d(np.asarray(depth, dtype=float)))
    rho = formation_resistivity(depth, layers, background)
    if include_borehole:
        r_inner = 0.0
        for z in zones:
            inside = (r >= r_inner) & (r < z.r_outer) & (depth >= z.top) & (depth < z.bottom)
            rho[inside] = z.resistivity
            r_inner = z.r_outer
    if air_resistivity is not None:
        rho[depth < 0] = air_resistivity
    return rho
