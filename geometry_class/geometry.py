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
        """Resistivity at (r, depth) for this model (depth positive downward)."""
        return resistivity_rz(r, depth, self.layer_1d_geometry, self.background_resistivity,
                              self.zones, include_borehole)

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

        radii = [z.r_outer for z in self.zones] + [self.near_radius, self.mid_radius, self.world_radius]
        poly_radii = [_polygon_radius(r, n) for r in radii]
        n_zones, last_j = len(self.zones), len(radii) - 1

        # The inner rings stop at the bottom of the refined column; below it only
        # the world ring remains (otherwise TetGen fills thin, deep columns with
        # tiny tetrahedra).  present[k] = rings existing at level k.
        present = [list(range(len(radii))) if d <= refined_bottom + 1e-9 else [last_j] for d in levels]

        plc = pg.Mesh(3, isGeometry=True)
        ang = 2.0 * np.pi * np.arange(n) / n
        centre, rings = [], []
        for k, d in enumerate(levels):
            centre.append(plc.createNode([0.0, 0.0, -d]))
            rings.append({j: [plc.createNode([poly_radii[j] * np.cos(a), poly_radii[j] * np.sin(a), -d])
                              for a in ang] for j in present[k]})

        last_k = len(levels) - 1
        for k in range(len(levels)):
            marker = int(BoundaryMarkers.NEUMANN if k == 0
                         else BoundaryMarkers.ROBIN if k == last_k else BoundaryMarkers.INTERNAL)
            js = present[k]
            for i in range(n):
                i1 = (i + 1) % n
                # Triangle fan around the axis node, so the electrode line meets the plane at a vertex
                plc.createTriangleFace(centre[k], rings[k][js[0]][i], rings[k][js[0]][i1], marker)
                for j0, j1 in zip(js[:-1], js[1:]):
                    plc.createQuadrangleFace(rings[k][j0][i], rings[k][j0][i1],
                                             rings[k][j1][i1], rings[k][j1][i], marker)
        for k in range(len(levels) - 1):
            for j in present[k + 1]:
                marker = int(BoundaryMarkers.ROBIN if j == last_j else BoundaryMarkers.INTERNAL)
                for i in range(n):
                    i1 = (i + 1) % n
                    plc.createQuadrangleFace(rings[k][j][i], rings[k][j][i1],
                                             rings[k + 1][j][i1], rings[k + 1][j][i], marker)

        # One region per (ring slot, slab); marker placed mid-slot on the polygon's bisector
        self._regions = {}
        marker = 0
        for k in range(len(levels) - 1):
            d_mid = 0.5 * (levels[k] + levels[k + 1])
            js = present[k + 1]
            for idx, j in enumerate(js):
                j_in = js[idx - 1] if idx > 0 else None
                r_in = radii[j_in] if j_in is not None else 0.0
                rp_in = poly_radii[j_in] if j_in is not None else 0.0
                r_mid = 0.5 * (r_in + radii[j])
                rp_mid = 0.5 * (rp_in + poly_radii[j]) * np.cos(np.pi / n)
                if j < n_zones:
                    area = self.borehole_area
                elif j == last_j or len(js) == 1:
                    area = self.world_area
                elif j == n_zones:
                    area = self.near_area
                else:
                    area = self.mid_area
                marker += 1
                self._regions[marker] = (r_mid, d_mid)
                plc.addRegionMarker([rp_mid * np.cos(np.pi / n), rp_mid * np.sin(np.pi / n), -d_mid],
                                    marker=marker, area=area)

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
        protocol_filename: str = "protocol.dat"
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

        Returns:
            pg.DataContainerERT: The generated data container with the protocol.
        """
        if not (long_spacing > short_spacing):
            raise ValueError("long_spacing must be greater than short_spacing")
        if long_spacing > self.long_spacing * (1 + 1e-9):
            warnings.warn(f"long_spacing={long_spacing} exceeds Geometry.long_spacing="
                          f"{self.long_spacing}; the refined mid zone may be too small.")

        num_electrodes = int(round(self.borehole_length / measuring_spacing)) + 1
        z_positions = -np.arange(num_electrodes) * measuring_spacing

        # Electrodes are PLC nodes on the borehole axis (reusing the plane-centre
        # nodes where they coincide) ...
        for z in z_positions:
            geom.createNodeWithCheck([0.0, 0.0, z]).setMarker(ELECTRODE_NODE_MARKER)

        # ... chained by edges through every axis node inside the borehole, which
        # forces TetGen to keep them and refines the mesh along the tool path
        axis = sorted((nd for nd in geom.nodes()
                       if abs(nd.pos()[0]) < 1e-9 and abs(nd.pos()[1]) < 1e-9
                       and -self.borehole_length - 1e-9 <= nd.pos()[2] <= 1e-9),
                      key=lambda nd: -nd.pos()[2])
        for n0, n1 in zip(axis[:-1], axis[1:]):
            geom.createEdge(n0, n1)

        # Create the ERT DataContainer
        data = pg.DataContainerERT()
        data.setSensorPositions([[0.0, 0.0, z] for z in z_positions])

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
        sim_data = ert.simulate(
            mesh=fwd_mesh,
            scheme=pg.DataContainerERT(data),
            res=pg.Vector(res),
            noiseLevel=noise_level,
            noiseAbs=noise_abs,
            sr=True,  # Singularity removal: analytic primary potential per source
            verbose=False,
        )

        if output_filename:
            sim_data.save(output_filename)

        return sim_data

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

        rho_a = np.array(data("rhoa")) if "rhoa" in data.tokenList() else np.array(data("r") * data("k"))
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
