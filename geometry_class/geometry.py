import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict
from enum import IntEnum
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import pygimli as pg
import pygimli.physics.ert as ert
import warnings
import pyvista as pv

from .borehole_model import make_zones, resistivity_rz


class BoundaryMarkers(IntEnum):
    """Markers for boundary conditions."""
    NEUMANN = -1  # Free surface (top): no current flows into the air
    ROBIN = -2    # Mixed/Robin boundary (far-field)
    INTERNAL = 0  # Internal boundary


ELECTRODE_NODE_MARKER = -99  # pg.core.MARKER_NODE_ELECTRODE


def _polygon_radius(r: float, n_segments: int) -> float:
    """Circumradius of a regular n-gon with the same area as a circle of radius r.

    An inscribed polygon underestimates the borehole cross-section (8 segments:
    -10 %), which biases the apparent resistivity; equal area removes that bias.
    """
    return r * np.sqrt(np.pi / (0.5 * n_segments * np.sin(2.0 * np.pi / n_segments)))


def merge_protocols(*datas: pg.DataContainerERT) -> pg.DataContainerERT:
    """Merge ERT protocols built on the same PLC into one (shared sensors are deduplicated).

    Simulating one merged protocol costs one forward run instead of one per
    protocol.  The column ``protocol`` holds the index of the source protocol;
    ``depthref`` and ``spacing`` are carried over when present.
    """
    positions, index = [], {}
    cols = {k: [] for k in ("a", "b", "m", "n", "protocol", "depthref", "spacing")}
    for p_i, d in enumerate(datas):
        sensor = {}
        for i in range(d.sensorCount()):
            pos = d.sensorPosition(i)
            key = tuple(round(float(pos[c]), 6) for c in range(3))
            if key not in index:
                index[key] = len(positions)
                positions.append(list(key))
            sensor[i] = index[key]
        tokens = d.tokenList()
        for j in range(d.size()):
            for k in "abmn":
                e = int(d(k)[j])
                cols[k].append(-1 if e < 0 else sensor[e])
            cols["protocol"].append(p_i)
            for k in ("depthref", "spacing"):
                cols[k].append(float(d[k][j]) if k in tokens else np.nan)
    out = pg.DataContainerERT()
    out.setSensorPositions(positions)
    out.resize(len(cols["a"]))
    for k in "abmn":
        out.set(k, cols[k])
    out.set("valid", np.ones(len(cols["a"]), dtype=int))
    for k in ("protocol", "depthref", "spacing"):
        out[k] = np.array(cols[k], dtype=float)
    return out


@dataclass
class Geometry:
    """
    Define the geometry, build the mesh and run 3D ERT simulations for a
    borehole tool in a (backfilled) borehole through a 1D layered earth.

    The PLC is built from concentric regular polygons (equal-area to the true
    circles) around the borehole axis:

        borehole zones (backfill, casing, ...) -> near zone -> mid zone -> world

    cut by horizontal planes at every layer interface, zone top/bottom and the
    borehole bottom.  Every (ring, slab) piece is its own region, so layer
    interfaces and zone boundaries are exact mesh faces and each cell gets its
    resistivity from its region.

    Attributes:
        borehole_length (float): Borehole depth (m).
        borehole_diameter (float): Borehole diameter (m).
        layer_1d_geometry (Optional[np.ndarray]): Rows ``[top_depth, bottom_depth, resistivity]``.
            The last layer extends to the bottom of the model.
        background_resistivity (float): Formation resistivity when no layers are given.
            Legacy: also used as the borehole fill when ``backfill_resistivity`` is None.
        backfill_resistivity (Optional[float]): Resistivity of the borehole backfill (Ohm.m).
        borehole_zones (Optional[list]): Concentric zones ``(r_outer, resistivity[, top[, bottom]])``
            or ``BoreholeZone`` objects; overrides ``backfill_resistivity`` (e.g. casing + annulus).
        long_spacing (float): Largest electrode spacing (m); sizes the refined mid zone.
        n_segments (int): Number of polygon segments for every ring.
        near_radius (Optional[float]): Radius of the finely meshed zone around the borehole.
            Defaults to 10 x borehole radius.
        mid_radius (Optional[float]): Radius of the medium zone. Defaults to
            max(3 x long_spacing, 2 x near_radius).
        world_radius (float): Radius of the outer (mixed BC) boundary (m).
        world_depth (Optional[float]): Depth of the model bottom. Defaults to
            borehole_length + world_radius.
        borehole_area, near_area, mid_area, world_area (Optional[float]): Maximum
            tetrahedron volumes (m^3) per zone; defaults follow from the zone sizes.
        p2 (bool): Use quadratic (P2) elements.  Strongly recommended: P1 leaves a
            few-percent bias for high borehole/formation contrasts.
        tool_diameter (Optional[float]): Diameter of an insulating tool body (mandrel) centred
            in the hole, over the full borehole length.  Electrodes then sit on its surface
            and the geometric factor is calibrated numerically (see ``run``).
        tool_resistivity (float): Resistivity of the tool body (Ohm.m).
        ring_azimuths (int): With a tool body, electrodes are rings: the measured potential
            is averaged over this many points around the tool (1 = point electrodes).
            Point electrodes are up to ~20 % off for short spacings in resistive backfill.
    """
    borehole_length: float
    borehole_diameter: float
    layer_1d_geometry: Optional[np.ndarray] = None
    background_resistivity: float = 5.0
    backfill_resistivity: Optional[float] = None
    borehole_zones: Optional[list] = None
    rhomap: Optional[List[List[float]]] = None
    long_spacing: float = 1.0
    n_segments: int = 24
    near_radius: Optional[float] = None
    mid_radius: Optional[float] = None
    world_radius: float = 100.0
    world_depth: Optional[float] = None
    borehole_area: Optional[float] = None
    near_area: Optional[float] = None
    mid_area: Optional[float] = None
    world_area: float = 0.0
    p2: bool = True
    tool_diameter: Optional[float] = None
    tool_resistivity: float = 1e6
    ring_azimuths: int = 4
    _regions: Dict[int, Tuple[float, float]] = field(default_factory=dict, init=False, repr=False)  # marker -> (r, depth)
    _p2_cache: tuple = field(default=(None, None), init=False, repr=False)

    def __post_init__(self):
        if self.layer_1d_geometry is not None and len(self.layer_1d_geometry) > 0:
            self.layer_1d_geometry = np.asarray(self.layer_1d_geometry, dtype=float)
            if self.layer_1d_geometry[0, 0] != 0:
                self.layer_1d_geometry[0, 0] = 0.0
                warnings.warn("The top value of the first layer did not start from 0. It has been forced to 0.")

        if self.backfill_resistivity is None and self.borehole_zones is None:
            self.backfill_resistivity = self.background_resistivity
            warnings.warn(
                "backfill_resistivity not given: using background_resistivity "
                f"({self.background_resistivity} Ohm.m) as the borehole fill."
            )
        self.zones = make_zones(self.borehole_diameter, self.borehole_length,
                                self.backfill_resistivity, self.borehole_zones)

        radius = self.borehole_diameter / 2.0
        self.tool_radius = None
        if self.ring_azimuths < 1:
            raise ValueError("ring_azimuths must be >= 1.")
        if self.tool_diameter is not None:
            self.tool_radius = self.tool_diameter / 2.0
            if not (0.0 < self.tool_radius < self.zones[0].r_outer):
                raise ValueError("tool_diameter must be positive and smaller than the innermost "
                                 "borehole zone diameter.")
        if self.near_radius is None:
            self.near_radius = 10.0 * radius
        if self.mid_radius is None:
            self.mid_radius = max(3.0 * self.long_spacing, 2.0 * self.near_radius)
        if self.world_depth is None:
            self.world_depth = self.borehole_length + self.world_radius
        if not (radius < self.near_radius < self.mid_radius < self.world_radius):
            raise ValueError("Radii must satisfy borehole radius < near_radius < mid_radius < world_radius.")

        # Target edge length h -> max volume of a regular tetrahedron h^3 / (6 sqrt 2)
        tet = lambda h: h**3 / (6.0 * np.sqrt(2.0))
        self._auto_borehole_area = self.borehole_area is None
        if self.borehole_area is None:
            self.borehole_area = tet(radius / 1.6)
        if self.near_area is None:
            self.near_area = tet(self.near_radius / 3.5)
        if self.mid_area is None:
            self.mid_area = tet(self.mid_radius / 6.0)
        if self.n_segments < 12:
            warnings.warn("n_segments < 12 gives a poor approximation of the borehole wall.")

    # ------------------------------------------------------------------
    def set_borehole(self, backfill_resistivity: Optional[float] = None, borehole_zones: Optional[list] = None):
        """Change the borehole fill on an existing geometry/mesh (e.g. a backfill sweep).

        The zone radii and depth intervals must stay the same, since they are
        built into the mesh; only resistivities may change.
        """
        new = make_zones(self.borehole_diameter, self.borehole_length,
                         backfill_resistivity, borehole_zones)
        if [(z.r_outer, z.top, z.bottom) for z in new] != [(z.r_outer, z.top, z.bottom) for z in self.zones]:
            raise ValueError("set_borehole can only change resistivities; rebuild the geometry for new radii.")
        self.backfill_resistivity, self.borehole_zones, self.zones = backfill_resistivity, borehole_zones, new
        if self.rhomap is not None:
            self.rhomap = self._rhomap()

    def resistivity_at(self, r, depth, include_borehole: bool = True) -> np.ndarray:
        """Resistivity at (r, depth) for this model (depth positive downward).

        The tool body is part of the measurement, not of the borehole, so it is
        kept in the ``include_borehole=False`` reference.
        """
        rho = resistivity_rz(r, depth, self.layer_1d_geometry, self.background_resistivity,
                             self.zones, include_borehole)
        if self.tool_radius is not None:
            r, depth = np.broadcast_arrays(np.atleast_1d(r), np.atleast_1d(depth))
            rho[(r < self.tool_radius) & (depth >= 0.0) & (depth < self.borehole_length)] = self.tool_resistivity
        return rho

    def make_basic_geometry(self, vtk_filename: str = "plc.vtk") -> pg.core.Mesh:
        """
        Create the 3D PLC: concentric polygonal rings cut by horizontal planes.

        Boundary markers: -1 on the surface (no-flux), -2 on the outer wall and
        bottom (mixed), 0 on internal faces.

        Args:
            vtk_filename (str, optional): Export the PLC to VTK. Defaults to 'plc.vtk'.
        """
        n = self.n_segments
        L = self.borehole_length
        refined_bottom = min(L + self.mid_radius, self.world_depth)

        # Horizontal planes (depths): surface, layer interfaces, zone limits,
        # borehole bottom, end of the refined column and model bottom
        levels = {0.0, L, refined_bottom, self.world_depth}
        if self.layer_1d_geometry is not None:
            levels.update(float(d) for d in self.layer_1d_geometry[:, :2].ravel())
        for z in self.zones:
            levels.update((z.top, z.bottom))
        levels = np.array(sorted(d for d in levels if 0.0 <= d <= self.world_depth))
        levels = levels[np.concatenate([[True], np.diff(levels) > 1e-6])]

        inner = ([self.tool_radius] if self.tool_radius is not None else []) + [z.r_outer for z in self.zones]
        radii = inner + [self.near_radius, self.mid_radius, self.world_radius]
        # Sides per ring: a thin tool body gets n/3 sides so its facets are not
        # millimetre-wide strips along the whole hole (which made TetGen fill the
        # annulus with ~10^6 cells); every count divides the next ring's count.
        n_sides = [n] * len(radii)
        if self.tool_radius is not None and n % 3 == 0 and n >= 24:
            n_sides[0] = n // 3
        poly_radii = [_polygon_radius(r, m) for r, m in zip(radii, n_sides)]
        n_zones, last_j = len(inner), len(radii) - 1
        self._levels = levels
        self._tool_poly_radius = poly_radii[0] if self.tool_radius is not None else None
        self._n_tool_sides = n_sides[0]

        # Rings only exist where they are needed (otherwise TetGen fills thin, deep
        # columns with tiny tetrahedra): borehole rings down to the borehole bottom,
        # near/mid rings down to the end of the refined column, the world ring
        # everywhere.  present[k] = rings existing at level k.
        limit = [L] * n_zones + [refined_bottom, refined_bottom, np.inf]
        present = [[j for j in range(len(radii)) if d <= limit[j] + 1e-9] for d in levels]

        plc = pg.Mesh(3, isGeometry=True)
        centre, rings = [], []
        for k, d in enumerate(levels):
            centre.append(plc.createNode([0.0, 0.0, -d]))
            rings.append({j: [plc.createNode([poly_radii[j] * np.cos(a), poly_radii[j] * np.sin(a), -d])
                              for a in 2.0 * np.pi * np.arange(n_sides[j]) / n_sides[j]]
                          for j in present[k]})

        last_k = len(levels) - 1
        for k in range(len(levels)):
            marker = int(BoundaryMarkers.NEUMANN if k == 0
                         else BoundaryMarkers.ROBIN if k == last_k else BoundaryMarkers.INTERNAL)
            js = present[k]
            ring0 = rings[k][js[0]]
            for i in range(len(ring0)):
                # Triangle fan around the axis node, so the axis meets the plane at a vertex
                plc.createTriangleFace(centre[k], ring0[i], ring0[(i + 1) % len(ring0)], marker)
            for j0, j1 in zip(js[:-1], js[1:]):
                a, b = rings[k][j0], rings[k][j1]
                ratio = len(b) // len(a)
                for i in range(len(a)):
                    i1 = (i + 1) % len(a)
                    if ratio == 1:
                        plc.createQuadrangleFace(a[i], a[i1], b[i1], b[i], marker)
                        continue
                    # Coarser inner ring: fan from a[i] over its outer arc, then close to a[i+1]
                    for t in range(ratio):
                        plc.createTriangleFace(a[i], b[i * ratio + t], b[(i * ratio + t + 1) % len(b)], marker)
                    plc.createTriangleFace(a[i], b[(i1 * ratio) % len(b)], a[i1], marker)
        for k in range(len(levels) - 1):
            for j in present[k + 1]:
                marker = int(BoundaryMarkers.ROBIN if j == last_j else BoundaryMarkers.INTERNAL)
                top, bot = rings[k][j], rings[k + 1][j]
                for i in range(len(top)):
                    i1 = (i + 1) % len(top)
                    plc.createQuadrangleFace(top[i], top[i1], bot[i1], bot[i], marker)

        # One region per (ring slot, slab), its marker placed between the inner
        # polygon's corners and the outer polygon's edge midpoints
        self._regions = {}
        self._zone_markers = set()
        marker = 0
        for k in range(len(levels) - 1):
            d_mid = 0.5 * (levels[k] + levels[k + 1])
            js = present[k + 1]
            for idx, j in enumerate(js):
                j_in = js[idx - 1] if idx > 0 else None
                r_in = radii[j_in] if j_in is not None else 0.0
                rp_in = poly_radii[j_in] if j_in is not None else 0.0
                r_mid = 0.5 * (r_in + radii[j])
                rp_mid = 0.5 * (rp_in + poly_radii[j] * np.cos(np.pi / n_sides[j]))
                if self.tool_radius is not None and j == 0:
                    area = self.near_area          # inside the insulating tool body
                elif j < n_zones:
                    area = self.borehole_area
                elif j == last_j or len(js) == 1:
                    area = self.world_area
                elif j == n_zones:
                    area = self.near_area
                else:
                    area = self.mid_area
                marker += 1
                self._regions[marker] = (r_mid, d_mid)
                if j < n_zones:
                    self._zone_markers.add(marker)
                plc.addRegionMarker([rp_mid, 0.0, -d_mid], marker=marker, area=area)

        if vtk_filename:
            plc.exportVTK(vtk_filename)
        return plc

    def make_array(
        self,
        geom: pg.core.Mesh,
        abmn_order: tuple,
        short_spacing: float,
        long_spacing: float,
        measuring_spacing: float,
        protocol_filename: str = "protocol.dat",
        electrode_offset: Optional[float] = None,
    ) -> pg.DataContainerERT:
        """
        Generate an electrode array, move it along the borehole, create a protocol, and add nodes to the PLC.

        Args:
            geom (pg.core.Mesh): The PLC geometry where electrode nodes will be added.
            abmn_order (tuple): The order of A, B, M, N electrodes (e.g., (1, 4, 2, 3) for Wenner).
            short_spacing (float): The minimum 'a' spacing for the array.
            long_spacing (float): The maximum 'a' spacing for the array.
            measuring_spacing (float): The step size to move the array down the borehole.
            protocol_filename (str): Output filename for the protocol.
            electrode_offset (Optional[float]): Depth of the first electrode (m). ``None``
                (default) = 0, unless an electrode would then sit exactly on a layer
                interface or zone boundary; the grid is then shifted by half a step,
                because a point electrode on a resistivity jump converges poorly
                (up to ~5 % error in the no-borehole reference).

        Returns:
            pg.DataContainerERT: The generated data container with the protocol.
        """
        if not (long_spacing > short_spacing):
            raise ValueError("long_spacing must be greater than short_spacing")
        if long_spacing > self.long_spacing * (1 + 1e-9):
            warnings.warn(f"long_spacing={long_spacing} exceeds Geometry.long_spacing="
                          f"{self.long_spacing}; the refined mid zone may be too small.")

        grid = lambda off: off + measuring_spacing * np.arange(
            int(np.floor((self.borehole_length - off) / measuring_spacing + 1e-9)) + 1)
        electrode_offset = self._choose_offset(grid, measuring_spacing, electrode_offset)
        depths = grid(electrode_offset)
        if self.tool_radius is not None:
            depths = depths[(depths > 0.0) & (depths < self.borehole_length)]
        z_positions = -depths

        self._refine_for_spacing(geom, short_spacing)

        # Create the ERT DataContainer
        data = pg.DataContainerERT()
        data.setSensorPositions(self._add_electrode_nodes(geom, depths))

        # Map A, B, M, N to their relative index positions (0-based)
        a_pos, b_pos, m_pos, n_pos = [idx - 1 for idx in abmn_order]

        # Use only the short and long spacings
        a_spacings = [short_spacing, long_spacing]

        a_idx, b_idx, m_idx, n_idx = [], [], [], []

        for a in a_spacings:
            # Number of index steps between adjacent logical array positions
            a_steps = int(round(a / measuring_spacing))
            if a_steps == 0:
                continue

            max_rel_idx = max(a_pos, b_pos, m_pos, n_pos) * a_steps

            # Move the array from top to bottom
            for start_idx in range(len(z_positions) - max_rel_idx):
                a_idx.append(int(start_idx + a_pos * a_steps))
                b_idx.append(int(start_idx + b_pos * a_steps))
                m_idx.append(int(start_idx + m_pos * a_steps))
                n_idx.append(int(start_idx + n_pos * a_steps))

        # Register standard arrays for pyGIMLi
        data.resize(len(a_idx))
        data.set("a", a_idx)
        data.set("b", b_idx)
        data.set("m", m_idx)
        data.set("n", n_idx)
        data.set("valid", np.ones(len(a_idx), dtype=int))

        if protocol_filename:
            data.save(protocol_filename)

        return data

    # ------------------------------------------------------------------
    def _avoid_depths(self) -> Tuple[np.ndarray, float]:
        """Depths electrodes must not sit on, and the minimum distance to keep."""
        if self.tool_radius is not None:
            # Off-axis electrodes may not touch any horizontal plane of the PLC
            return np.asarray(self._levels), 1e-3
        return self._interface_depths(), 1e-6

    def _choose_offset(self, make_depths, step: float, electrode_offset: Optional[float]) -> float:
        """Pick an electrode-grid offset so that no electrode sits on an interface."""
        avoid, tol = self._avoid_depths()
        clash = lambda off: [d for d in make_depths(off) if np.any(np.abs(avoid - d) < tol)]
        if electrode_offset is not None:
            if clash(electrode_offset):
                warnings.warn(f"Electrodes at depths {clash(electrode_offset)} sit on a resistivity "
                              "interface; results there converge poorly.")
            return electrode_offset
        for frac in (0.0, 0.5, 0.25, 0.75, 0.125, 0.375):
            if not clash(frac * step):
                if frac:
                    warnings.warn(f"Electrodes would sit on layer/zone interfaces; electrode grid "
                                  f"shifted by {frac * step:g} m.")
                return frac * step
        warnings.warn("Could not keep all electrodes off interfaces; pass electrode_offset.")
        return 0.0

    def _refine_for_spacing(self, geom: pg.core.Mesh, short_spacing: float):
        """Small spacings need borehole cells no larger than the spacing."""
        h_bh = (self.borehole_area * 6.0 * np.sqrt(2.0)) ** (1.0 / 3.0)
        if self._auto_borehole_area and short_spacing < h_bh:
            self.borehole_area = short_spacing**3 / (6.0 * np.sqrt(2.0))
            for rm in geom.regionMarkers():
                if rm.marker() in self._zone_markers:
                    rm.setArea(self.borehole_area)

    def _add_electrode_nodes(self, geom: pg.core.Mesh, depths) -> List[List[float]]:
        """Add electrode nodes to the PLC and return their positions.

        Without a tool body the electrodes are on the borehole axis, chained by
        edges through every axis node (reusing the plane-centre nodes), which
        forces TetGen to keep them and refines the mesh along the tool path.
        With a tool body they sit in the backfill, 2 mm (at most) from the tool
        surface; a point electrode there stands in for a ring electrode.
        """
        if self.tool_radius is None:
            positions = [[0.0, 0.0, -d] for d in depths]
            for pos in positions:
                geom.createNodeWithCheck(pos).setMarker(ELECTRODE_NODE_MARKER)
            axis = sorted((nd for nd in geom.nodes()
                           if abs(nd.pos()[0]) < 1e-9 and abs(nd.pos()[1]) < 1e-9
                           and -self.borehole_length - 1e-9 <= nd.pos()[2] <= 1e-9),
                          key=lambda nd: -nd.pos()[2])
            for n0, n1 in zip(axis[:-1], axis[1:]):
                geom.createEdge(n0, n1)
            return positions

        gap = min(2e-3, 0.1 * (_polygon_radius(self.zones[0].r_outer, self.n_segments) - self._tool_poly_radius))
        self._electrode_radius = self._tool_poly_radius + gap
        positions = [[self._electrode_radius, 0.0, -d] for d in depths]
        for d in depths:
            for x, y in self._ring_xy():
                geom.createNodeWithCheck([x, y, -d]).setMarker(ELECTRODE_NODE_MARKER)
        return positions

    def _ring_xy(self) -> np.ndarray:
        """(x, y) of the ring sample points: tool-polygon corner directions, evenly spread."""
        n_tool = self._n_tool_sides
        if n_tool % self.ring_azimuths:
            raise ValueError(f"ring_azimuths must divide the {n_tool} sides of the tool polygon.")
        ang = 2.0 * np.pi * np.arange(self.ring_azimuths) / self.ring_azimuths
        return np.column_stack([self._electrode_radius * np.cos(ang), self._electrode_radius * np.sin(ang)])

    def make_tool_array(
        self,
        geom: pg.core.Mesh,
        array: str = "normal",
        spacings=(0.4064, 1.6256),
        measuring_spacing: float = 0.1,
        mn_spacing: float = 0.8128,
        electrode_offset: Optional[float] = None,
        protocol_filename: Optional[str] = None,
    ) -> pg.DataContainerERT:
        """
        Logging-tool arrays with remote electrodes, moved along the borehole.

        * ``"normal"``: current electrode A and potential electrode M on the sonde,
          M above A at distance AM (``spacings``, e.g. 16" = 0.4064 m and
          64" = 1.6256 m); B and N are remote (pole-pole).  Depth reference: AM midpoint.
        * ``"lateral"``: A on the sonde, M and N above it, centred at distance AO
          (``spacings``, e.g. 18'8" = 5.69 m) and ``mn_spacing`` apart
          (32" = 0.8128 m); B is remote (pole-dipole).  Depth reference: O.

        Remote electrodes are true poles ("at infinity"); with a tool body the
        geometric factor is calibrated numerically in ``run``.  The data
        container gets two extra columns: ``depthref`` (m) and ``spacing`` (m).

        Args:
            geom (pg.core.Mesh): The PLC from ``make_basic_geometry``.
            array (str): ``"normal"`` or ``"lateral"``.
            spacings: AM (normal) or AO (lateral) values in m.
            measuring_spacing (float): Step of the tool along the hole (m).
            mn_spacing (float): MN distance of the lateral (m).
            electrode_offset (Optional[float]): Depth offset of the station grid;
                ``None`` = chosen so that no electrode sits on an interface.
            protocol_filename (Optional[str]): Save the protocol to this file.

        Returns:
            pg.DataContainerERT: The protocol.
        """
        array = array.lower()
        if array not in ("normal", "lateral"):
            raise ValueError("array must be 'normal' or 'lateral'.")
        spacings = [float(x) for x in np.atleast_1d(spacings)]
        if max(spacings) > self.long_spacing * (1 + 1e-9):
            warnings.warn(f"Spacing {max(spacings)} m exceeds Geometry.long_spacing="
                          f"{self.long_spacing}; set long_spacing to it so the refined zone is large enough.")
        L = self.borehole_length

        def configs(off):
            """(spacing, depth A, depth M, depth N or None, depth reference) per measurement."""
            out = []
            a_depths = off + measuring_spacing * np.arange(int(np.floor((L - off) / measuring_spacing + 1e-9)) + 1)
            for sp in spacings:
                for da in a_depths:
                    if array == "normal":
                        dm, dn, ref = da - sp, None, da - sp / 2.0
                    else:
                        dm, dn, ref = da - sp + mn_spacing / 2.0, da - sp - mn_spacing / 2.0, da - sp
                    top = dm if dn is None else dn
                    if top > 1e-3 and da < L - 1e-3:
                        out.append((sp, da, dm, dn, ref))
            return out

        all_depths = lambda off: sorted({round(d, 6) for c in configs(off) for d in c[1:4] if d is not None})
        electrode_offset = self._choose_offset(all_depths, measuring_spacing, electrode_offset)
        cfg = configs(electrode_offset)
        if not cfg:
            raise ValueError("No tool position fits in the borehole with these spacings.")
        depths = np.array(all_depths(electrode_offset))
        index = {round(d, 6): i for i, d in enumerate(depths)}

        self._refine_for_spacing(geom, min(min(spacings), mn_spacing if array == "lateral" else np.inf))

        data = pg.DataContainerERT()
        data.setSensorPositions(self._add_electrode_nodes(geom, depths))
        data.resize(len(cfg))
        data.set("a", [index[round(c[1], 6)] for c in cfg])
        data.set("b", [-1] * len(cfg))
        data.set("m", [index[round(c[2], 6)] for c in cfg])
        data.set("n", [-1 if c[3] is None else index[round(c[3], 6)] for c in cfg])
        data.set("valid", np.ones(len(cfg), dtype=int))
        data["depthref"] = np.array([c[4] for c in cfg])
        data["spacing"] = np.array([c[0] for c in cfg])

        if protocol_filename:
            data.save(protocol_filename)
        return data

    def make_mesh(self, geom: pg.core.Mesh, vtk_filename: str = "mesh.vtk", quality: float = 1.3) -> pg.core.Mesh:
        """
        Mesh the PLC with TetGen and attach the resistivity model.

        Args:
            geom (pg.core.Mesh): The geometry to mesh.
            vtk_filename (str, optional): The filename for exporting the mesh to VTK format. Defaults to 'mesh.vtk'.
            quality (float, optional): TetGen maximum radius-edge ratio (3D). 1.2-1.4 gives
                well-shaped tetrahedra; larger values allow slivers. Defaults to 1.3.
        """
        if quality > 2.0:
            warnings.warn(f"quality={quality} is a 2D min-angle value; in 3D it is the TetGen "
                          "radius-edge ratio and should be 1.2-1.4.")
        if not self._regions:
            raise ValueError("Call make_basic_geometry() before make_mesh().")

        mesh = pg.meshtools.createMesh(geom, quality=quality)

        # TetGen carries the facet markers over; enforce them on the hull anyway
        for bound in mesh.boundaries():
            if bound.outside():
                top = abs(bound.center()[2]) < 1e-6
                bound.setMarker(int(BoundaryMarkers.NEUMANN if top else BoundaryMarkers.ROBIN))
            else:
                bound.setMarker(int(BoundaryMarkers.INTERNAL))

        unknown = set(mesh.cellMarkers()) - set(self._regions)
        if unknown:
            raise RuntimeError(f"Mesh cells with unexpected region markers {sorted(unknown)}.")

        self.rhomap = self._rhomap()
        mesh["Resistivity"] = self.cell_resistivity(mesh, include_borehole=True)

        if vtk_filename:
            mesh.exportVTK(vtk_filename)
            mesh.save("mesh.bms")

        return mesh

    def _rhomap(self, include_borehole: bool = True) -> List[List[float]]:
        """[marker, resistivity] for every region, evaluated at the region centre."""
        markers = sorted(self._regions)
        r, d = np.array([self._regions[m] for m in markers]).T
        return [[m, float(rho)] for m, rho in zip(markers, self.resistivity_at(r, d, include_borehole))]

    def _interface_depths(self) -> np.ndarray:
        """Depths (> 0) of layer interfaces, zone limits and the borehole bottom."""
        d = {self.borehole_length}
        if self.layer_1d_geometry is not None:
            d.update(float(x) for x in self.layer_1d_geometry[:, :2].ravel())
        for z in self.zones:
            d.update((z.top, z.bottom))
        return np.array(sorted(x for x in d if x > 1e-9))

    def cell_resistivity(self, mesh: pg.core.Mesh, include_borehole: bool = True) -> np.ndarray:
        """Per-cell resistivity from the regions (``include_borehole=False`` -> no-hole reference)."""
        rho = dict(self._rhomap(include_borehole))
        return np.array([rho[m] for m in mesh.cellMarkers()])

    def plot_boundary_conditions(self, mesh: pg.core.Mesh, show_internal: bool = True):
        """
        Plot the boundary conditions (markers) of the mesh using PyVista.

        Args:
            mesh (pg.core.Mesh): The mesh to extract boundaries from.
            show_internal (bool): If False, hides internal boundaries (marker == 0).
        """
        points = []
        faces = []
        markers = []

        node_map = {}
        node_idx = 0

        for bound in mesh.boundaries():
            marker = bound.marker()
            if not show_internal and marker == 0:
                continue

            b_node_ids = []
            for i in range(bound.nodeCount()):
                n = bound.node(i)
                if n.id() not in node_map:
                    node_map[n.id()] = node_idx
                    pos = n.pos()
                    points.append([pos[0], pos[1], pos[2]])
                    node_idx += 1
                b_node_ids.append(node_map[n.id()])
            faces.append([bound.nodeCount()] + b_node_ids)
            markers.append(marker)

        surf = pv.PolyData(np.array(points), np.hstack(faces))
        surf.cell_data["Boundary Markers"] = np.array(markers)

        pl = pv.Plotter()
        pl.add_mesh(surf, scalars="Boundary Markers", cmap="Set1", show_edges=True, opacity=0.5)
        pl.add_axes()
        pl.add_title("Mesh Boundary Conditions")
        pl.show()

    def run(self, mesh: pg.core.Mesh, data: pg.DataContainerERT, rhomap: list = None,
            output_filename: str = "simulated_data.dat", noise_level: float = 0.0,
            noise_abs: float = 0.0, include_borehole: bool = True) -> pg.DataContainerERT:
        """
        Run the ERT forward model using the generated mesh and data protocol.

        Args:
            mesh (pg.core.Mesh): The generated mesh.
            data (pg.DataContainerERT): The measurement protocol.
            rhomap (Optional[List[List[float]]]): Custom [marker, resistivity] pairs overriding the model.
            output_filename (str, optional): The filename for exporting the simulated data. Defaults to 'simulated_data.dat'.
            noise_level (float, optional): Relative noise level to add to the data (e.g. 0.05 for 5%). Defaults to 0.0.
            noise_abs (float, optional): Absolute noise level in Ohms. Defaults to 0.0.
            include_borehole (bool, optional): False replaces the borehole by formation
                (reference for the borehole effect). Defaults to True.

        Returns:
            pg.DataContainerERT: The simulated data container.
        """
        if rhomap is not None:
            rhomap_dict = dict(rhomap)
            res = np.array([rhomap_dict[m] for m in mesh.cellMarkers()])
        else:
            res = self.cell_resistivity(mesh, include_borehole)

        fwd_mesh = mesh
        if self.p2:
            if self._p2_cache[0] is not mesh:
                self._p2_cache = (mesh, mesh.createP2())
            fwd_mesh = self._p2_cache[1]

        print(f"Simulating ERT data ({'P2' if self.p2 else 'P1'}, "
              f"{'with' if include_borehole else 'without'} borehole)...")
        if self.tool_radius is None:
            sim_data = self._simulate(fwd_mesh, data, res, noise_level, noise_abs)
        else:
            # Ring electrodes on the tool body; tool constant k such that a
            # homogeneous earth (with the tool) reads its true resistivity
            r = self._ring_resistances(fwd_mesh, data, res)
            k = self._tool_constant(mesh, fwd_mesh, data)
            rhoa = k * r
            if noise_level > 0 or noise_abs > 0:
                rhoa = rhoa * (1.0 + noise_level * np.random.randn(len(rhoa))) \
                    + noise_abs * k * np.random.randn(len(rhoa))
            sim_data = pg.DataContainerERT(data)
            sim_data["r"], sim_data["k"], sim_data["rhoa"] = r, k, rhoa
            sim_data["valid"] = np.ones(len(r))

        if output_filename:
            sim_data.save(output_filename)

        return sim_data

    @staticmethod
    def _simulate(fwd_mesh, data, res, noise_level=0.0, noise_abs=0.0):
        return ert.simulate(
            mesh=fwd_mesh,
            scheme=pg.DataContainerERT(data),
            res=pg.Vector(res),
            noiseLevel=noise_level,
            noiseAbs=noise_abs,
            sr=True,  # Singularity removal: analytic primary potential per source
            verbose=False,
        )

    def _ring_resistances(self, fwd_mesh, data, res) -> np.ndarray:
        """Transfer resistances (V/A) between ring electrodes on the tool body.

        Solves only for the current electrodes A (one FE solve each; PyGIMLi's
        ``simulate`` would also solve for every potential electrode) and reads
        the potentials at the measuring nodes.  The model is axisymmetric, so
        the ring-to-ring resistance is the average over the ring points of M
        (and N) for a point source A.
        """
        pos = np.array([list(data.sensorPosition(i)) for i in range(data.sensorCount())])
        a_idx = np.array(data("a"), dtype=int)
        if np.any(np.array(data("b"), dtype=int) >= 0):
            raise ValueError("Tool-body arrays need a remote B electrode (b = -1).")
        sources = sorted(set(a_idx))
        src_row = {s: i for i, s in enumerate(sources)}

        ds = pg.DataContainerERT()
        ds.setSensorPositions(pos[sources].tolist())
        ds.resize(len(sources))
        ds.set("a", np.arange(len(sources)))
        ds.set("m", (np.arange(len(sources)) + 1) % len(sources) if len(sources) > 1 else [-1])
        ds.set("b", [-1] * len(sources))
        ds.set("n", [-1] * len(sources))
        ds.set("valid", np.ones(len(sources)))
        fop = ert.ERTModelling(sr=True, verbose=False)
        fop.setData(ds)
        fop.setMesh(fwd_mesh, ignoreRegionManager=True)
        fop.response(pg.Vector(res))
        sol = np.array(fop.solution())          # (n_sources, n_nodes), unit current

        ring = self._ring_xy()
        node_cache = {}

        def ring_nodes(sensor):
            if sensor not in node_cache:
                z = pos[sensor, 2]
                node_cache[sensor] = [fwd_mesh.findNearestNode([x, y, z]) for x, y in ring]
            return node_cache[sensor]

        r = np.zeros(data.size())
        for j, (a, m, n) in enumerate(zip(a_idx, np.array(data("m"), dtype=int), np.array(data("n"), dtype=int))):
            u = sol[src_row[a]]
            r[j] = u[ring_nodes(m)].mean() - (u[ring_nodes(n)].mean() if n >= 0 else 0.0)
        return r

    def _tool_constant(self, mesh, fwd_mesh, data) -> np.ndarray:
        """Numerical geometric factor: 1 / R for a 1 Ohm.m homogeneous earth with the tool body."""
        key = (id(mesh), data.size(),
               tuple(np.array(data("a"))), tuple(np.array(data("m"))), tuple(np.array(data("n"))))
        cache = getattr(self, "_k_cache", None)
        if cache is not None and cache[0] == key:
            return cache[1]
        r, d = np.array([self._regions[m] for m in mesh.cellMarkers()]).T
        in_tool = (r < self.tool_radius) & (d >= 0.0) & (d < self.borehole_length)
        res_cal = np.where(in_tool, self.tool_resistivity, 1.0)
        print("Calibrating the tool constant (homogeneous 1 Ohm.m earth with tool body)...")
        k = 1.0 / self._ring_resistances(fwd_mesh, data, res_cal)
        self._k_cache = (key, k)
        return k

    def plot_model_and_data(self, data: pg.DataContainerERT):
        """
        Plots a 2D slice of the 1D resistivity model and borehole alongside a
        depth profile of the short and long Wenner measurements.

        Args:
            data (pg.DataContainerERT): The ERT data container to plot.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8), sharey=True)

        # --- Plot 1: 2D Slice of 1D model & Borehole ---
        width = max(20.0 * self.borehole_diameter, 2.0)
        bottom_plot = 1.1 * self.borehole_length

        # Determine color scaling bounds from layers and borehole zones
        res_values = [z.resistivity for z in self.zones]
        if self.layer_1d_geometry is not None:
            res_values.extend(self.layer_1d_geometry[:, 2])
        else:
            res_values.append(self.background_resistivity)

        min_res, max_res = min(res_values), max(res_values)
        if min_res == max_res:
            min_res, max_res = min_res * 0.5, max_res * 2.0

        norm = mcolors.LogNorm(vmin=min_res, vmax=max_res)
        cmap = cm.jet

        if self.layer_1d_geometry is not None:
            rows = [(float(r[0]), float(r[1]), float(r[2])) for r in self.layer_1d_geometry]
            rows[-1] = (rows[-1][0], max(rows[-1][1], bottom_plot), rows[-1][2])
        else:
            rows = [(0.0, bottom_plot, self.background_resistivity)]
        for top, bottom, res in rows:
            rect = patches.Rectangle((-width/2, -bottom), width, bottom - top,
                                     facecolor=cmap(norm(res)), edgecolor='k')
            ax1.add_patch(rect)
            # Annotate layer resistivity value on the left side (avoiding the central borehole)
            ax1.text(-width * 0.3, -(top + bottom) / 2.0, f"{res:.1f} $\\Omega\\cdot$m",
                     color='black', ha='center', va='center',
                     bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2'))

        # Draw borehole zones from the outside in
        for z in reversed(self.zones):
            bh_rect = patches.Rectangle((-z.r_outer, -z.bottom), 2 * z.r_outer, z.bottom - z.top,
                                        facecolor=cmap(norm(z.resistivity)), edgecolor='white', hatch='///')
            ax1.add_patch(bh_rect)

        ax1.set_xlim(-width/2, width/2)
        ax1.set_ylim(-self.borehole_length * 1.05, 0)
        ax1.set_xlabel("x (m)")
        ax1.set_ylabel("Depth (m)")
        ax1.set_title("1D Model & Borehole Slice")

        # Setup colorbar based on resistivity values
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax1, orientation='vertical', fraction=0.05, pad=0.05)
        cbar.set_label(r"Resistivity ($\Omega \cdot$m)")

        # --- Plot 2: Depth Profile of Wenner Measures ---
        # Calculate depth (z_center) and the 'a' spacing dynamically for each measurement quadrupole
        z_A = np.array([data.sensorPosition(int(i))[2] for i in data("a")])
        z_B = np.array([data.sensorPosition(int(i))[2] for i in data("b")])
        z_M = np.array([data.sensorPosition(int(i))[2] for i in data("m")])
        z_N = np.array([data.sensorPosition(int(i))[2] for i in data("n")])

        z_center = (z_A + z_B + z_M + z_N) / 4.0

        # In a generic collinear setup, sorting the coordinates easily identifies adjacent electrode spacing 'a'
        a_spacings = [abs(sorted([z_A[i], z_B[i], z_M[i], z_N[i]])[1] - sorted([z_A[i], z_B[i], z_M[i], z_N[i]])[0]) for i in range(data.size())]
        a_spacings = np.round(a_spacings, decimals=2)

        rho_a = np.array(data["rhoa"]) if "rhoa" in data.tokenList() else np.array(data["r"]) * np.array(data["k"])
        unique_a = np.unique(a_spacings)

        if len(unique_a) > 0:
            mask_short = (a_spacings == unique_a[0])
            ax2.plot(rho_a[mask_short], z_center[mask_short], 'o-', label=f"Short (a={unique_a[0]}m)", color='blue')

        if len(unique_a) > 1:
            mask_long = (a_spacings == unique_a[-1])
            ax2.plot(rho_a[mask_long], z_center[mask_long], 's-', label=f"Long (a={unique_a[-1]}m)", color='red')

        # Draw true 1D resistivity profile as a step-wise line
        true_z, true_res = [], []
        for top, bottom, res in rows:
            true_z.extend([-top, -bottom])
            true_res.extend([res, res])
        ax2.plot(true_res, true_z, '--', color='black', label="True 1D Model", linewidth=2)

        ax2.set_xscale('log')
        ax2.set_xlabel(r"Apparent Resistivity ($\Omega \cdot$m)")
        ax2.set_title("Apparent Resistivity vs Depth")
        ax2.legend()
        ax2.grid(True, which='both', linestyle='--', alpha=0.7)

        plt.tight_layout()
        plt.show()


# Example usage:
if __name__ == "__main__":
    # Create a dummy layer configuration: [top, bottom, resistivity]
    layers = np.array([
        [0.0, 5.0, 100.0],
        [5.0, 10.0, 50.0],
        [10.0, 15.0, 200.0]
    ])
    geometry = Geometry(borehole_length=15.0, borehole_diameter=0.2, layer_1d_geometry=layers,
                        backfill_resistivity=5.0, long_spacing=1.0)
    geom = geometry.make_basic_geometry()

    # Generate the electrode array and protocol prior to meshing
    data = geometry.make_array(
        geom=geom,
        abmn_order=(1, 4, 2, 3),
        short_spacing=0.25,
        long_spacing=geometry.long_spacing,
        measuring_spacing=0.25
    )
    print(f"ERT Data generated with {data.size()} measurements.")

    mesh = geometry.make_mesh(geom)
    print(f"Mesh created: {mesh}")

    # Run the forward simulation
    simulated_data = geometry.run(mesh, data)
    print(f"Forward simulation complete. Simulated data saved to 'simulated_data.dat' with {simulated_data.size()} measurements.")

    # Plot the 2D slice of the 1D model and the depth profiles of the simulated data
    geometry.plot_model_and_data(simulated_data)
