# -*- coding: utf-8 -*-
"""
em_geometry.py
==============

Frequency-Domain Electromagnetic (FDEM) borehole simulation using SimPEG
on a full 3-D finite-element mesh.

The ``EMGeometry`` class shares the same layer/geometry inputs as the
``Geometry`` (ERT) class but drives a 3-D FDEM forward simulation:

  * **Mesh**      : ``discretize.CylindricalMesh`` — cylindrical (r, theta, z)
                    mesh that exploits the borehole's axial geometry.
                    For VMD (``coil_orientation='z'``) the problem is fully
                    axisymmetric and only ONE theta cell is needed, reducing
                    cell count from O(n^3) to O(n_r x n_z) — typically
                    < 30 000 cells vs. millions for TensorMesh.
                    For HMD (``coil_orientation='x'/'y'``) use ``n_theta_cells``
                    > 1 (default 1 with a warning).
  * **Solver**    : ``simpeg.electromagnetics.frequency_domain.Simulation3DElectricField``
                    — edge-based E-field FEM formulation of Maxwell's equations.
  * **Survey**    : ``fdem.sources.MagDipole`` placed at each tool depth;
                    ``fdem.receivers.PointMagneticFieldSecondary`` for each
                    coil spacing (real + imaginary components).

All sources that share the same operating frequency also share the same
factorised system matrix, so SimPEG performs one LU decomposition per
frequency and applies it to all depth-RHS vectors simultaneously.

Notes
-----
* The borehole axis is the z-axis (r=0).  Surface is at z=0; depth is
  negative.  This matches the SimPEG/discretize convention (+z upward).
* The resistivity model (layers, concentric borehole zones, air) comes from
  ``borehole_model.resistivity_rz`` and is shared with the ERT ``Geometry``
  class.  ``run(include_borehole=False)`` gives the no-borehole reference.
* Two z-directed coils on the borehole axis form a *coaxial* pair.
* SimPEG uses the e^{+i omega t} time convention.
* ``gridCC`` for a CylindricalMesh returns columns (r, theta, z); column 2
  is always z so ``_paint_conductivity`` works unchanged.
* Raw secondary H-field values (A/m) are returned.
  To convert to ppm: ``ppm = H_sec / H_primary_freespace * 1e6``.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from .borehole_model import make_zones, resistivity_rz
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
import matplotlib.cm as cm

# ── Optional heavy dependencies ────────────────────────────────────────────────
try:
    import discretize as _discretize
    _DISCRETIZE_AVAILABLE = True
except ImportError:  # pragma: no cover
    _DISCRETIZE_AVAILABLE = False
    warnings.warn(
        "discretize is not installed.  "
        "Install it with: pip install discretize\n"
        "EMGeometry.make_mesh_3d() and run() will raise ImportError at call time.",
        stacklevel=2,
    )

try:
    import simpeg.electromagnetics.frequency_domain as fdem
    from simpeg import maps as _maps
    _SIMPEG_AVAILABLE = True
except ImportError:  # pragma: no cover
    _SIMPEG_AVAILABLE = False
    warnings.warn(
        "simpeg is not installed.  "
        "Install it with: pip install simpeg\n"
        "EMGeometry.run() will raise ImportError at call time.",
        stacklevel=2,
    )
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class EMGeometry:
    """
    3-D frequency-domain EM borehole forward simulation using SimPEG.

    Parameters
    ----------
    borehole_length : float
        Total depth of the borehole (m).
    borehole_diameter : float
        Diameter of the borehole (m).
    layer_1d_geometry : np.ndarray, shape (N, 3), optional
        Layer table with columns ``[top_depth, bottom_depth, resistivity]``.
        Depths are positive-downward (m); resistivity in Ω·m.
    background_resistivity : float, optional
        Formation resistivity (Ω·m) when no layers are given.  The last layer
        extends to infinite depth.  Legacy: also used as the borehole fill
        when ``backfill_resistivity`` is not given.  Defaults to 5.0.
    backfill_resistivity : float, optional
        Resistivity of the borehole backfill (Ω·m).
    borehole_zones : list, optional
        Concentric zones ``(r_outer, resistivity[, top[, bottom]])`` or
        ``BoreholeZone`` objects; overrides ``backfill_resistivity``.
    air_resistivity : float, optional
        Resistivity of the air above the surface.  Defaults to 1e8 Ω·m.
    frequencies : list of float
        Transmitter frequencies (Hz), e.g. ``[400.0, 8000.0]``.
    coil_spacings : list of float
        Transmitter–receiver coil offsets (m), e.g. ``[0.5, 1.0, 2.0]``.
        Multiple spacings are simulated simultaneously (one receiver per
        spacing per source).
    coil_orientation : str
        Coil-axis orientation:

        * ``'z'`` — coil axes along the borehole: coaxial pair (standard
          induction logging tool).  Default.
        * ``'x'`` or ``'y'`` — coil axes across the borehole: coplanar pair.

    measuring_spacing : float, optional
        Depth step for sliding the tool (m).  Defaults to 0.1 m.
    receiver_above : bool, optional
        If ``False`` (default) the receiver is placed *below* the transmitter
        (deeper in borehole).  Set ``True`` to reverse.  Both coils always stay
        inside the borehole; the reported ``depth`` is the TX-RX midpoint.
    core_cell_size : float, optional
        Edge length (m) of the uniform fine cells in the radial (r) direction
        outside the borehole (inside the borehole see ``borehole_cells``).
        The z-direction cell size is ``min(core_cell_size, min(coil_spacings) / 20)``
        (convergence-tested for a 1 Ω·m backfill in 100 Ω·m formation).
        Should be ≤ skin-depth / 5 at the highest frequency in the most
        conductive layer.  Defaults to 0.05 m.
    n_padding : int, optional
        Number of geometric-expansion padding cells on each boundary side.
        Defaults to 10.
    pad_factor : float, optional
        Geometric expansion ratio per padding cell (each cell is
        ``pad_factor`` times wider than its inner neighbour).  Defaults to 1.4.
    domain_radius : float, optional
        Outer boundary radius (m) for both the radial far-field and the
        vertical far-field padding.  ``None`` (default) = max(100 m,
        3 skin depths in the most resistive formation at the lowest
        frequency).  A boundary closer than that mainly distorts the
        in-phase field.
    borehole_cells : int, optional
        Number of radial cells across the innermost borehole zone (cylindrical
        mesh).  Defaults to 10 (1 cm cells for a 0.2 m hole).
    n_theta_cells : int, optional
        Number of azimuthal cells in the CylindricalMesh.

        * ``1`` (default) — axisymmetric; exact for VMD (``coil_orientation='z'``)
          and approximately valid for HMD in horizontally layered media.
        * ``≥ 8`` — recommended for HMD (``coil_orientation='x'`` or ``'y'``);
          increases cell count by ``n_theta_cells`` but is still far smaller
          than a full TensorMesh.
    mesh_type : str, optional
        Which mesh to use for the 3-D FDEM solve:

        * ``'cylindrical'`` (default) — :class:`discretize.CylindricalMesh`.
          Axisymmetric; one LU factorisation per frequency covers every tool
          depth simultaneously.  Best for VMD surveys with many depth positions.
        * ``'tree'`` — :class:`discretize.TreeMesh` (OcTree).
          Each tool depth is simulated independently with a *local* mesh
          centred around that position, so the mesh is much smaller than a
          full-borehole model.  Requires one LU factorisation per depth per
          frequency but handles HMD accurately without extra theta cells.

    Attributes
    ----------
    The above constructor arguments are stored as instance attributes.
    """

    # ── Shared geometry (mirrors Geometry) ─────────────────────────────────────
    borehole_length: float
    borehole_diameter: float
    layer_1d_geometry: Optional[np.ndarray] = None
    background_resistivity: float = 5.0
    backfill_resistivity: Optional[float] = None
    borehole_zones: Optional[list] = None
    air_resistivity: float = 1e8

    # ── EM survey parameters ───────────────────────────────────────────────────
    frequencies: List[float] = field(default_factory=lambda: [1000.0])
    coil_spacings: List[float] = field(default_factory=lambda: [1.0])
    coil_orientation: str = "z"
    measuring_spacing: float = 0.1
    receiver_above: bool = False

    # ── 3-D mesh control ───────────────────────────────────────────────────────
    mesh_type: str = "cylindrical"     # 'cylindrical' or 'tree'
    core_cell_size: float = 0.05
    n_padding: int = 10
    pad_factor: float = 1.4
    domain_radius: Optional[float] = None
    n_theta_cells: int = 1
    borehole_cells: int = 10

    # ──────────────────────────────────────────────────────────────────────────
    def __post_init__(self) -> None:
        # ── Layer table validation ────────────────────────────────────────────
        if self.layer_1d_geometry is not None:
            self.layer_1d_geometry = np.asarray(self.layer_1d_geometry, dtype=float)
            if (
                self.layer_1d_geometry.ndim != 2
                or self.layer_1d_geometry.shape[1] < 3
            ):
                raise ValueError(
                    "layer_1d_geometry must be a 2-D array with columns "
                    "[top_depth, bottom_depth, resistivity]."
                )
            if self.layer_1d_geometry[0, 0] != 0.0:
                self.layer_1d_geometry[0, 0] = 0.0
                warnings.warn(
                    "Top of first layer was not 0 m — forced to 0.", stacklevel=2
                )
            if self.layer_1d_geometry[-1, 1] > self.borehole_length:
                self.borehole_length = float(self.layer_1d_geometry[-1, 1])
                warnings.warn(
                    f"Bottom of last layer exceeds borehole_length; "
                    f"borehole_length updated to {self.borehole_length} m.",
                    stacklevel=2,
                )

        # ── mesh_type validation ──────────────────────────────────────────────
        self.mesh_type = self.mesh_type.lower()
        if self.mesh_type not in ("cylindrical", "tree"):
            raise ValueError("mesh_type must be 'cylindrical' or 'tree'.")

        # ── Borehole zones ────────────────────────────────────────────────────
        if self.backfill_resistivity is None and self.borehole_zones is None:
            self.backfill_resistivity = self.background_resistivity
            warnings.warn(
                "backfill_resistivity not given: using background_resistivity "
                f"({self.background_resistivity} Ω·m) as the borehole fill.",
                stacklevel=2,
            )
        self.zones = make_zones(self.borehole_diameter, self.borehole_length,
                                self.backfill_resistivity, self.borehole_zones)

        # ── EM parameter validation ───────────────────────────────────────────
        self.coil_orientation = self.coil_orientation.lower()
        # A transverse dipole is not axisymmetric: one theta cell cannot represent it
        if (
            self.coil_orientation in ("x", "y")
            and self.n_theta_cells < 8
            and self.mesh_type == "cylindrical"
        ):
            raise ValueError(
                f"coil_orientation='{self.coil_orientation}' on a CylindricalMesh needs "
                f"n_theta_cells >= 8 (got {self.n_theta_cells}); or use mesh_type='tree'."
            )
        if self.coil_orientation not in ("x", "y", "z"):
            raise ValueError("coil_orientation must be 'x', 'y', or 'z'.")
        self.frequencies = list(self.frequencies)
        self.coil_spacings = list(self.coil_spacings)
        if not self.frequencies:
            raise ValueError("At least one frequency must be supplied.")
        if not self.coil_spacings:
            raise ValueError("At least one coil spacing must be supplied.")
        if self.measuring_spacing <= 0:
            raise ValueError("measuring_spacing must be positive.")
        if self.core_cell_size <= 0:
            raise ValueError("core_cell_size must be positive.")
        if max(self.coil_spacings) >= self.borehole_length:
            raise ValueError("Coil spacings must be shorter than the borehole.")

        # ── Domain size: >= 3 skin depths in the most resistive formation ─────
        if self.domain_radius is None:
            rho_max = max(self._formation_resistivities())
            skin = 503.0 * np.sqrt(rho_max / min(self.frequencies))
            self.domain_radius = float(max(100.0, 3.0 * skin))
        if self.mesh_type == "tree" and self.core_cell_size > self.zones[0].r_outer / 3.0:
            warnings.warn(
                f"TreeMesh: core_cell_size={self.core_cell_size} m does not resolve the "
                f"borehole (radius {self.zones[0].r_outer} m); use <= radius/3.",
                stacklevel=2,
            )

        # ── Skin-depth advisory ───────────────────────────────────────────────
        self._check_skin_depth()

    # ──────────────────────────────────────────────────────────────────────────
    def _formation_resistivities(self) -> List[float]:
        if self.layer_1d_geometry is None:
            return [float(self.background_resistivity)]
        return [float(r) for r in self.layer_1d_geometry[:, 2]]

    # ──────────────────────────────────────────────────────────────────────────
    def _check_skin_depth(self) -> None:
        """Issue a warning if ``core_cell_size`` is too coarse for the EM problem."""
        max_freq = max(self.frequencies)
        min_res = min(self._formation_resistivities() + [z.resistivity for z in self.zones])
        # Approximate skin depth: δ ≈ 503 √(ρ/f)  [m]
        skin_depth = 503.0 * np.sqrt(min_res / max_freq)
        recommended = skin_depth / 5.0
        if self.core_cell_size > recommended:
            warnings.warn(
                f"core_cell_size={self.core_cell_size} m may be too coarse. "
                f"Minimum skin depth at {max_freq:.0f} Hz in the most conductive "
                f"material ({min_res:.1f} Ω·m) is ≈ {skin_depth:.2f} m; "
                f"recommend core_cell_size ≤ {recommended:.3f} m for adequate "
                f"accuracy (currently set to skin_depth/5).",
                stacklevel=3,
            )

    # ──────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _n_pad_cells(start: float, end: float, cs: float, pfac: float) -> int:
        """
        Number of geometric-expansion cells needed to bridge from *start* to
        *end*, where the first cell has width *cs* and each subsequent cell
        is *pfac* times wider.

        Uses: start + cs*(pfac^n - 1)/(pfac - 1) = end  →  solve for n.
        Returns at least 5 cells.
        """
        if end <= start:
            return 5
        ratio = 1.0 + (end - start) * (pfac - 1.0) / cs
        return max(5, int(np.ceil(np.log(ratio) / np.log(pfac))))

    # ──────────────────────────────────────────────────────────────────────────
    def _radial_widths(self, pfac: float) -> np.ndarray:
        """Radial cell widths with faces on every borehole-zone radius."""
        cs_r = self.core_cell_size
        dr = self.zones[0].r_outer / self.borehole_cells
        widths: list = []
        r_in = 0.0
        for z in self.zones:
            n = max(2, int(np.ceil((z.r_outer - r_in) / dr - 1e-9)))
            widths += [(z.r_outer - r_in) / n] * n
            r_in = z.r_outer
        # Grow smoothly (x1.2) from the borehole cells to cs_r, uniform to r_fine
        w, r = widths[-1], r_in
        r_fine = max(2.0 * max(self.coil_spacings), self.borehole_diameter * 10.0)
        while r < r_fine:
            w = min(cs_r, w * 1.2)
            widths.append(w)
            r += w
        while r < self.domain_radius:
            w *= pfac
            widths.append(w)
            r += w
        return np.array(widths)

    # ──────────────────────────────────────────────────────────────────────────
    def make_mesh_3d(self) -> "_discretize.CylindricalMesh":
        """
        Build and return the 3-D ``CylindricalMesh`` for the FDEM simulation.

        Mesh layout  (r, theta, z)
        --------------------------
        * **r** — radial axis from the borehole axis outward.
          Inside the borehole, cell faces fall exactly on every zone radius
          (``borehole_cells`` cells across the innermost zone); outside, cells
          grow smoothly (x1.2) to ``core_cell_size``, stay uniform to
          ``max(2 × max(coil_spacings), 10 × borehole diameter)`` and then
          expand geometrically to ``domain_radius``.
        * **theta** — azimuthal axis.
          ``n_theta_cells = 1`` (default, VMD): one cell covering 2π —
          fully axisymmetric, exact for VMD.
          ``n_theta_cells > 1`` (HMD): uniformly distributed cells.
        * **z** — vertical axis, positive upward, surface at z=0.
          Fine cells span the borehole column; geometric padding above and
          below extends to ``domain_radius``.

        Returns
        -------
        discretize.CylindricalMesh
        """
        if not _DISCRETIZE_AVAILABLE:
            raise ImportError(
                "discretize is required.  Install with: pip install discretize"
            )

        pfac = self.pad_factor

        # ── Radial (r) — borehole zones resolved, then fine, then padding ───
        hr = self._radial_widths(pfac)

        # ── Azimuthal (theta) ────────────────────────────────────────────────
        n_th = max(1, self.n_theta_cells)
        if n_th == 1:
            h_theta = [2.0 * np.pi]       # single cell covers full circle
        else:
            dtheta = 2.0 * np.pi / n_th
            h_theta = [dtheta] * n_th     # uniform azimuthal cells

        # ── Vertical (z) cell size ───────────────────────────────────────────
        # A conductive borehole channels current along the hole; resolving the
        # resulting vertical field variation needs >= 20 cells per smallest coil
        # spacing (s/5 under-predicts the borehole effect by ~10 %).
        cs_z = min(self.core_cell_size, min(self.coil_spacings) / 20.0)

        n_core_z    = int(np.ceil(self.borehole_length / cs_z)) + 1
        n_above     = max(2, self.n_padding // 4)   # 2–3 cells above z=0 is enough
        npad_z_below = self._n_pad_cells(0.0, self.domain_radius, cs_z, pfac)
        npad_z_above = self._n_pad_cells(n_above * cs_z, self.domain_radius, cs_z, pfac)

        hz = [
            (cs_z, npad_z_below, -pfac),  # below borehole — expanding downward
            (cs_z, n_core_z),             # fine borehole column
            (cs_z, n_above),              # a few uniform cells above surface
            (cs_z, npad_z_above, pfac),   # above surface — expanding upward
        ]

        # ── Origin ──────────────────────────────────────────────────────────
        # r starts at 0 (axis); theta starts at 0.
        # z0: top of the fine borehole column aligns with z=0 (surface).
        # Discretize's (cs, n, -f) format generates widths [cs·f^n, cs·f^(n-1), …, cs·f]
        # (the minimum cell adjacent to the fine region has width cs·f, not cs).
        # Correct geometric sum = cs · f · (f^n − 1) / (f − 1).
        sum_pad_below = cs_z * pfac * (pfac**npad_z_below - 1.0) / (pfac - 1.0)
        z0 = -(sum_pad_below + n_core_z * cs_z)

        mesh = _discretize.CylindricalMesh([hr, h_theta, hz], origin=[0.0, 0.0, z0])
        return mesh

    # ──────────────────────────────────────────────────────────────────────────
    def _paint_conductivity(self, mesh, include_borehole: bool = True) -> np.ndarray:
        """
        Per-cell conductivity (S/m) from the shared borehole model.

        Cells are classified by their centre: air above z=0, borehole zones by
        radial distance from the axis, and the 1-D layers (last layer extending
        to infinite depth) everywhere else.  ``include_borehole=False`` gives
        the no-borehole reference model.

        Parameters
        ----------
        mesh : discretize.CylindricalMesh or discretize.TreeMesh
        include_borehole : bool, optional

        Returns
        -------
        np.ndarray, shape (mesh.nC,)
        """
        cc = mesh.gridCC
        if "Cyl" in type(mesh).__name__:
            r = cc[:, 0]                      # (r, theta, z)
        else:
            r = np.hypot(cc[:, 0], cc[:, 1])  # (x, y, z)
        rho = resistivity_rz(r, -cc[:, 2], self.layer_1d_geometry, self.background_resistivity,
                             self.zones, include_borehole, self.air_resistivity)
        return 1.0 / rho

    # ──────────────────────────────────────────────────────────────────────────
    def _build_survey(self, depths: np.ndarray):
        """
        Construct the FDEM survey with all sources and receivers.

        One ``MagDipole`` source per (frequency, transmitter depth); one
        real + one imaginary ``PointMagneticFluxDensitySecondary`` receiver
        per coil spacing whose receiver still lies inside the borehole.

        Parameters
        ----------
        depths : np.ndarray
            Transmitter depths (m, positive-downward).

        Returns
        -------
        survey : fdem.survey.Survey
        meta : list of (frequency, tx_depth, spacing)
            One entry per (real, imag) receiver pair, in ``dpred`` order.
        """
        source_list: list = []
        meta: list = []
        tol = 1e-9

        for freq in self.frequencies:
            for depth in depths:
                receivers: list = []
                for spacing in self.coil_spacings:
                    rx_depth = depth - spacing if self.receiver_above else depth + spacing
                    if not (-tol <= rx_depth <= self.borehole_length + tol):
                        continue  # the tool cannot leave the borehole
                    rx_loc = np.array([[0.0, 0.0, -float(rx_depth)]])
                    for component in ("real", "imag"):
                        receivers.append(
                            fdem.receivers.PointMagneticFluxDensitySecondary(
                                locations=rx_loc,
                                orientation=self.coil_orientation,
                                component=component,
                            )
                        )
                    meta.append((float(freq), float(depth), float(spacing)))
                if not receivers:
                    continue
                source_list.append(
                    fdem.sources.MagDipole(
                        receiver_list=receivers,
                        frequency=freq,
                        location=np.array([[0.0, 0.0, -float(depth)]]),
                        orientation=self.coil_orientation,
                    )
                )

        return fdem.survey.Survey(source_list), meta

    # ──────────────────────────────────────────────────────────────────────────
    def _tx_depths(self) -> np.ndarray:
        n_depths = int(np.floor(self.borehole_length / self.measuring_spacing + 1e-9)) + 1
        return np.arange(n_depths) * self.measuring_spacing

    # ──────────────────────────────────────────────────────────────────────────
    def _records(self, dpred_vec: np.ndarray, meta: list) -> list:
        """Turn ``dpred`` (B in T, real/imag pairs) into records with H in A/m."""
        MU0 = 4.0 * np.pi * 1e-7  # H/m
        records: list = []
        for i, (freq, tx_depth, spacing) in enumerate(meta):
            h_real = float(dpred_vec[2 * i]) / MU0
            h_imag = float(dpred_vec[2 * i + 1]) / MU0
            rx_depth = tx_depth - spacing if self.receiver_above else tx_depth + spacing
            records.append(
                {
                    "depth":     0.5 * (tx_depth + rx_depth),
                    "tx_depth":  tx_depth,
                    "spacing":   spacing,
                    "frequency": freq,
                    "real":      h_real,
                    "imag":      h_imag,
                    "amplitude": float(np.hypot(h_real, h_imag)),
                    "phase":     float(np.degrees(np.arctan2(h_imag, h_real))),
                    "apparent_resistivity": self._lin_apparent_resistivity(h_imag, spacing, freq),
                }
            )
        return records

    # ──────────────────────────────────────────────────────────────────────────
    def run(self, verbose: bool = True, include_borehole: bool = True) -> pd.DataFrame:
        """
        Run the 3-D FDEM forward simulation.

        Workflow
        --------
        1. Build the mesh via :meth:`make_mesh_3d` (or one TreeMesh per depth).
        2. Paint the model (layers, borehole zones, air) via
           :meth:`_paint_conductivity`.
        3. Build the FDEM survey via :meth:`_build_survey`.
        4. ``Simulation3DElectricField`` with ``sigmaMap = IdentityMap``;
           ``sim.dpred`` factorises once per frequency and solves all
           transmitter positions together.
        5. Parse ``dpred`` into a tidy ``pd.DataFrame``.

        Parameters
        ----------
        verbose : bool, optional
            Print progress messages.  Defaults to ``True``.
        include_borehole : bool, optional
            ``False`` replaces the borehole by formation (no-borehole
            reference for the borehole effect).  Defaults to ``True``.

        Returns
        -------
        pd.DataFrame
            One row per ``(position, spacing, frequency)`` with columns:

            * ``depth``     — TX-RX midpoint depth (m, positive-downward)
            * ``tx_depth``  — transmitter depth (m)
            * ``spacing``   — coil spacing (m)
            * ``frequency`` — frequency (Hz)
            * ``real``      — real part of H_secondary (A/m)  [= B_real/μ₀]
            * ``imag``      — imaginary part of H_secondary (A/m)  [= B_imag/μ₀]
            * ``amplitude`` — |H_secondary| (A/m)
            * ``phase``     — phase angle (°)
            * ``apparent_resistivity`` — LIN apparent resistivity (Ω·m)

        Raises
        ------
        ImportError
            If ``simpeg`` or ``discretize`` are not installed.
        """
        if not _SIMPEG_AVAILABLE:
            raise ImportError(
                "simpeg is required.  Install with: pip install simpeg"
            )
        if not _DISCRETIZE_AVAILABLE:
            raise ImportError(
                "discretize is required.  Install with: pip install discretize"
            )

        # ── Solver setup (used by both tree and cylindrical paths) ───────────
        try:
            from simpeg.utils import get_default_solver
            default_solver = get_default_solver()
        except Exception:
            default_solver = None

        depths = self._tx_depths()

        # ── 1. Mesh / dispatch ────────────────────────────────────────────────
        if self.mesh_type == "tree":
            if verbose:
                print(
                    f"[TreeMesh] {len(depths)} depth positions × "
                    f"{len(self.frequencies)} freq(s) × "
                    f"{len(self.coil_spacings)} spacing(s) — "
                    "one local simulation per depth."
                )
            return self._run_per_depth(depths, default_solver, verbose, include_borehole)

        if verbose:
            n_th = max(1, self.n_theta_cells)
            mtype = "CylindricalMesh (axisymmetric)" if n_th == 1 else f"CylindricalMesh ({n_th} theta cells)"
            print(f"Building 3-D {mtype} …")
        mesh = self.make_mesh_3d()
        if verbose:
            sr, st, sz = mesh.shape_cells
            print(f"  Mesh: {mesh.nC:,} cells  (n_r={sr}, n_theta={st}, n_z={sz}), "
                  f"domain radius {self.domain_radius:.0f} m")

        # ── 2. Conductivity model ─────────────────────────────────────────────
        sigma = self._paint_conductivity(mesh, include_borehole)
        if verbose:
            print(f"  Model {'with' if include_borehole else 'without'} borehole; "
                  f"σ range: {sigma.min():.4g} – {sigma.max():.4g} S/m")

        # ── 3. Survey ─────────────────────────────────────────────────────────
        survey, meta = self._build_survey(depths)
        if verbose:
            print(f"Building survey: {survey.nSrc} sources, {len(meta)} TX-RX pairs …")

        # ── 4. Simulation ─────────────────────────────────────────────────────
        sim_kwargs = dict(
            mesh=mesh,
            survey=survey,
            sigmaMap=_maps.IdentityMap(nP=mesh.nC),
        )
        if default_solver is not None:
            sim_kwargs["solver"] = default_solver

        sim = fdem.Simulation3DElectricField(**sim_kwargs)
        dpred_vec = sim.dpred(sigma)

        # ── 5. Parse dpred ────────────────────────────────────────────────────
        records = self._records(dpred_vec, meta)
        if verbose:
            print(f"  Forward simulation complete: {len(records)} data points.")
        return pd.DataFrame(records)

    # ──────────────────────────────────────────────────────────────────────────
    def _make_local_tree_mesh(self, depth: float) -> "_discretize.TreeMesh":
        """
        Build a small OcTree (``TreeMesh``) centred on one tool depth.

        The mesh covers the required lateral and vertical extent.  With
        ``h = [N * [cs]]`` (N equal cells of size *cs*), the discretize
        ``TreeMesh`` invariant gives::

            max_level = log2(N)
            finest cell size  = cs          (at insert level = max_level)
            coarsest cell size = cs * N     (at insert level = 0)

        N is chosen as the smallest power of 2 such that ``N * cs`` covers
        the required span on each axis.

        Parameters
        ----------
        depth : float
            Transmitter depth (m, positive-downward).

        Returns
        -------
        discretize.TreeMesh
        """
        if not _DISCRETIZE_AVAILABLE:
            raise ImportError("discretize is required.  pip install discretize")

        cs          = self.core_cell_size
        dom         = self.domain_radius
        max_spacing = max(self.coil_spacings)
        min_spacing = min(self.coil_spacings)

        # TX / RX z-coordinates (SimPEG: +z upward, z=0 surface)
        z_tx = -depth
        z_rx = z_tx - max_spacing if not self.receiver_above else z_tx + max_spacing

        # Vertical padding: boundary at least half the domain radius from TX/RX
        z_pad    = max(dom * 0.5, max_spacing * 4.0)
        z_top    = min(z_tx + z_pad, +dom)
        z_bottom = max(z_rx - z_pad, -dom)

        # Required spans to cover
        xy_span = 2.0 * dom
        z_span  = z_top - z_bottom

        # N per axis: smallest power-of-2 so that N * cs >= required_span.
        # With h = [N * [cs]], max_level = log2(N) and finest cell = cs.
        nx = max(8, int(2 ** np.ceil(np.log2(max(1.0, xy_span / cs)))))
        nz = max(8, int(2 ** np.ceil(np.log2(max(1.0, z_span  / cs)))))

        # Maximum refinement levels available on each axis
        max_level_xy = int(np.log2(nx))
        max_level_z  = int(np.log2(nz))
        max_level    = min(max_level_xy, max_level_z)

        # Centre the mesh on the TX-RX midpoint (z) and the borehole axis (xy)
        z_mid    = 0.5 * (z_tx + z_rx)
        z_origin = z_mid - nz * cs / 2.0

        x0 = [-nx * cs / 2.0, -nx * cs / 2.0, z_origin]

        mesh = _discretize.TreeMesh(
            [nx * [cs], nx * [cs], nz * [cs]],
            x0=x0,
        )

        # Refinement along the TX-RX segment (borehole axis: x=0, y=0)
        cs_z = max(cs, min_spacing / 5.0)
        n_bh = max(3, int(np.ceil(abs(z_tx - z_rx) / cs_z)) + 1)
        z_bh = np.linspace(z_tx, z_rx, n_bh)

        n_shells = min(max_level, 5)

        for shell_idx in range(n_shells):
            level_here = max_level - shell_idx   # finest = max_level, coarser outward
            if level_here < 1:
                break

            if shell_idx == 0:
                # Axis and every borehole-zone wall — finest refinement
                pts = [np.column_stack([np.zeros(n_bh), np.zeros(n_bh), z_bh])]
                for zone in self.zones:
                    for ang in np.linspace(0, 2 * np.pi, 16, endpoint=False):
                        pts.append(np.column_stack([
                            zone.r_outer * np.cos(ang) * np.ones(n_bh),
                            zone.r_outer * np.sin(ang) * np.ones(n_bh),
                            z_bh,
                        ]))
                pts = np.vstack(pts)
            else:
                # Concentric cylinder at radius r_shell
                r_shell = cs * (2 ** shell_idx)
                arcs = [
                    np.column_stack([
                        r_shell * np.cos(ang) * np.ones(n_bh),
                        r_shell * np.sin(ang) * np.ones(n_bh),
                        z_bh,
                    ])
                    for ang in np.linspace(0, 2 * np.pi, 8, endpoint=False)
                ]
                pts = np.vstack(arcs)

            mesh.insert_cells(
                pts,
                level_here * np.ones(len(pts), dtype=int),
                finalize=False,
            )

        mesh.finalize()
        return mesh

    # ──────────────────────────────────────────────────────────────────────────
    def _run_per_depth(
        self,
        depths: np.ndarray,
        default_solver,
        verbose: bool,
        include_borehole: bool = True,
    ) -> pd.DataFrame:
        """
        Run FDEM forward simulation once per tool depth using a local TreeMesh.

        Each depth position is independent:
        1. Build a small OcTree mesh centred on that depth.
        2. Paint conductivity from the 1-D layer model.
        3. Build a single-depth survey (all frequencies, all spacings).
        4. Solve the FEM system and extract H-field at receiver locations.
        5. Append parsed records and proceed to the next depth.

        Parameters
        ----------
        depths : np.ndarray
            Array of transmitter depths (m, positive-downward).
        default_solver : solver class or None
            SimPEG linear solver (from ``get_default_solver``).
        verbose : bool
            Print progress to stdout.
        include_borehole : bool
            ``False`` gives the no-borehole reference.

        Returns
        -------
        pd.DataFrame
        """
        n_tot  = len(depths)
        records: list = []

        for i_d, depth in enumerate(depths):
            # ── Single-depth survey (skip positions with no receiver in the hole)
            survey_d, meta_d = self._build_survey(np.array([depth]))
            if not meta_d:
                continue
            if verbose:
                print(
                    f"  [TreeMesh] depth {i_d + 1}/{n_tot}: "
                    f"{depth:.2f} m  ",
                    end="",
                    flush=True,
                )

            # ── Local mesh for this depth ─────────────────────────────────────
            mesh_d  = self._make_local_tree_mesh(depth)
            sigma_d = self._paint_conductivity(mesh_d, include_borehole)

            if verbose:
                print(f"({mesh_d.nC:,} cells)", flush=True)

            # ── Simulation ───────────────────────────────────────────────────
            sim_kwargs: dict = dict(
                mesh=mesh_d,
                survey=survey_d,
                sigmaMap=_maps.IdentityMap(nP=mesh_d.nC),
            )
            if default_solver is not None:
                sim_kwargs["solver"] = default_solver

            sim_d   = fdem.Simulation3DElectricField(**sim_kwargs)
            records += self._records(sim_d.dpred(sigma_d), meta_d)

        if verbose:
            print(f"  Parsed {len(records)} data points into DataFrame.")
        return pd.DataFrame(records)

    # ──────────────────────────────────────────────────────────────────────────
    def _lin_apparent_resistivity(self, h_imag: float, spacing: float, freq: float) -> float:
        """
        Low-Induction-Number (LIN) apparent resistivity for borehole EM.

        Valid when the skin depth >> coil spacing.
        Returns NaN if the formula is not applicable.

        Sign convention
        ---------------
        SimPEG uses the ``e^{+iomega*t}`` time convention, in which
        Im(H_sec) < 0 for a conductive earth; we use q = -Im(H_sec).

        Geometry (full space around the tool)
        -------------------------------------
        Expanding the exact full-space dipole fields to first order in
        omega*mu*sigma*s^2 gives Im(H_s/H_p) = -omega*mu*sigma*s^2/2 for both
        geometries, hence:

          coaxial  (orientation='z'):    k = 4*pi  (H_primary = 1 / (2*pi*s^3))
          coplanar (orientation='x/y'):  k = 8*pi  (|H_primary| = 1 / (4*pi*s^3))

          sigma_a = (k * s) / (mu0 * omega) * (-h_imag)
          rho_a   = 1 / sigma_a

        Parameters
        ----------
        h_imag : float
            Imaginary part of secondary H-field (A/m) in SimPEG convention
            (typically negative for a conductive earth).
        spacing : float
            Coil spacing (m).
        freq : float
            Operating frequency (Hz).

        Returns
        -------
        float
            Apparent resistivity (Ohm.m), or NaN if undefined.
        """
        # Negate: SimPEG e^{+iwt} gives Im(H_sec) < 0 for a conductive earth
        q = -h_imag
        if q <= 0.0:
            return float("nan")
        MU0   = 4.0 * np.pi * 1e-7
        omega = 2.0 * np.pi * freq
        # Full-space coefficients (coaxial / coplanar)
        k = 4.0 * np.pi if self.coil_orientation == "z" else 8.0 * np.pi
        sigma_a = (k * spacing) / (MU0 * omega) * q
        return float(1.0 / sigma_a)

    # ──────────────────────────────────────────────────────────────────────────
    def plot_mesh_slice(
        self,
        mesh: "_discretize.CylindricalMesh",
        sigma: Optional[np.ndarray] = None,
        figsize: Tuple[float, float] = (10, 6),
    ) -> plt.Figure:
        """
        Plot an R-Z cross-section of the CylindricalMesh.

        For a CylindricalMesh, ``gridCC[:, 0]`` = r (radial) and
        ``gridCC[:, 2]`` = z (vertical).  A single theta-slice is shown
        (all theta values collapse to the same r-z plane).

        Parameters
        ----------
        mesh : discretize.CylindricalMesh
            The mesh returned by :meth:`make_mesh_3d`.
        sigma : np.ndarray, optional
            Per-cell conductivity (S/m).  Cells are coloured by log10(sigma)
            when provided; uniform colour otherwise.
        figsize : tuple, optional
            Figure size.  Defaults to ``(10, 6)``.

        Returns
        -------
        matplotlib.figure.Figure
        """
        mesh_cls = type(mesh).__name__

        # ── TreeMesh branch ──────────────────────────────────────────────────
        if "TreeMesh" in mesh_cls:
            fig, ax = plt.subplots(figsize=figsize)
            if sigma is not None:
                v = np.log10(np.clip(sigma, 1e-12, None))
                try:
                    out = mesh.plot_slice(
                        v, v_type="CC", normal="Y", ax=ax,
                        pcolor_opts=dict(cmap="viridis"),
                    )
                    plt.colorbar(out[0], ax=ax, label="log₁₀(σ)  [S/m]")
                except Exception:
                    # Fallback: scatter at y≈0 cells
                    y_cc = mesh.gridCC[:, 1]
                    mask = np.abs(y_cc) < (max(self.coil_spacings) * 0.05 + 0.01)
                    x_pl = mesh.gridCC[mask, 0]
                    z_pl = mesh.gridCC[mask, 2]
                    sc   = ax.scatter(x_pl, z_pl, c=v[mask], cmap="viridis", s=6)
                    plt.colorbar(sc, ax=ax, label="log₁₀(σ)  [S/m]")
            else:
                mesh.plot_grid(ax=ax)
            ax.set_xlabel("x  (m)")
            ax.set_ylabel("z  (m, + upward)")
            ax.set_title(f"TreeMesh  |  y = 0 slice  ({mesh.nC:,} cells)")
            plt.tight_layout()
            return fig

        # ── CylindricalMesh branch (original) ────────────────────────────────
        fig, ax = plt.subplots(figsize=figsize)

        # For CylindricalMesh: gridCC columns are (r, theta, z)
        # Keep only one theta-slice (they all represent the same r-z plane)
        r_cc     = mesh.gridCC[:, 0]   # radial coordinate
        z_cc     = mesh.gridCC[:, 2]   # vertical coordinate
        theta_cc = mesh.gridCC[:, 1]

        # Use the smallest theta value (first slice)
        theta_unique = np.unique(theta_cc)
        mask = np.isclose(theta_cc, theta_unique[0], atol=1e-6)

        r_plot = r_cc[mask]
        z_plot = z_cc[mask]

        if sigma is not None:
            s_slice = sigma[mask]
            log_s   = np.log10(np.clip(s_slice, 1e-12, None))
            vmin, vmax = log_s.min(), log_s.max()
            if np.isclose(vmin, vmax):
                vmin, vmax = vmin - 1, vmax + 1
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
            cmap = cm.viridis
            sc = ax.scatter(
                r_plot, z_plot, c=log_s, cmap=cmap, norm=norm, s=3, marker="s"
            )
            cbar = fig.colorbar(sc, ax=ax, fraction=0.035, pad=0.04)
            cbar.set_label("log10(sigma)  [log S/m]")
        else:
            ax.scatter(r_plot, z_plot, c="steelblue", s=3, marker="s")

        # Reference lines
        ax.axvline(
            self.borehole_diameter / 2,
            color="white", lw=1.0, ls="--", alpha=0.8,
            label=f"Borehole wall (r={self.borehole_diameter/2:.3f} m)",
        )
        ax.axhline(0.0, color="orange", lw=0.8, ls="--", alpha=0.7, label="Surface (z=0)")
        ax.axhline(
            -self.borehole_length,
            color="red", lw=0.8, ls="--", alpha=0.7,
            label=f"Borehole bottom ({self.borehole_length} m)",
        )

        view_r = min(self.domain_radius * 0.3, 10.0 * max(self.coil_spacings))
        ax.set_xlim(0.0, view_r)
        ax.set_ylim(-self.borehole_length * 1.3, self.borehole_length * 0.2)
        ax.set_xlabel("r (m)")
        ax.set_ylabel("z (m, +up)")
        ax.set_title(
            "CylindricalMesh R-Z slice"
            + ("  (coloured by log sigma)" if sigma is not None else "")
        )
        ax.legend(fontsize=8)
        plt.tight_layout()
        return fig

    # ──────────────────────────────────────────────────────────────────────────
    def plot_log(
        self,
        result: pd.DataFrame,
        quantity: str = "apparent_resistivity",
        log_scale: Optional[bool] = None,
        figsize: Optional[Tuple[float, float]] = None,
    ) -> plt.Figure:
        """
        Plot the simulated EM log as depth profiles.

        One curve is drawn for every unique ``(spacing, frequency)`` combination.
        Depth increases downward on the y-axis.  Horizontal dashed lines mark
        the layer boundaries from ``layer_1d_geometry``.

        Parameters
        ----------
        result : pd.DataFrame
            Output of :meth:`run`.
        quantity : str, optional
            Column to plot.  Choices:

            * ``'apparent_resistivity'`` (default) — LIN apparent resistivity
              in Ohm.m; log x-axis applied automatically.
            * ``'amplitude'`` — |H_secondary| in A/m.
            * ``'phase'`` — phase angle in degrees.
            * ``'real'`` / ``'imag'`` — H-field components in A/m.

        log_scale : bool or None, optional
            Force logarithmic x-axis (``True``/``False``).
            If ``None`` (default), log scale is applied automatically for
            ``'apparent_resistivity'`` and ``'amplitude'``.
        figsize : tuple, optional
            Figure size.  Auto-sized if ``None``.

        Returns
        -------
        matplotlib.figure.Figure
        """
        _valid = ("apparent_resistivity", "amplitude", "phase", "real", "imag")
        if quantity not in _valid:
            raise ValueError(f"quantity must be one of {_valid}.")

        # Auto log-scale: on for resistivity and amplitude, off for phase/field
        if log_scale is None:
            log_scale = quantity in ("apparent_resistivity", "amplitude")

        combos  = result.groupby(["spacing", "frequency"], sort=True)
        n_combos = len(combos)

        if figsize is None:
            figsize = (5, max(6, self.borehole_length / 3))

        fig, ax = plt.subplots(figsize=figsize)
        tab_cmap = cm.tab10
        colors   = [tab_cmap(i / max(n_combos - 1, 1)) for i in range(n_combos)]

        for idx, ((spacing, freq), grp) in enumerate(combos):
            grp_s = grp.sort_values("depth")
            ax.plot(
                grp_s[quantity],
                -grp_s["depth"],   # negative → depth increases downward
                label=f"s={spacing} m, f={freq:.0f} Hz",
                color=colors[idx],
                linewidth=1.5,
            )

        # Layer boundary lines
        if self.layer_1d_geometry is not None:
            for row in self.layer_1d_geometry[1:]:   # skip surface boundary
                ax.axhline(
                    -row[0], color="gray", linestyle="--", linewidth=0.8, alpha=0.6
                )

        _units = {
            "apparent_resistivity": "Ohm.m",
            "amplitude": "A/m",
            "real":      "A/m (real)",
            "imag":      "A/m (imag)",
            "phase":     "degrees",
        }
        _labels = {
            "apparent_resistivity": "Apparent Resistivity",
            "amplitude": "Amplitude",
            "real":  "Real Component",
            "imag":  "Imaginary Component",
            "phase": "Phase",
        }
        xlabel = f"{_labels.get(quantity, quantity)} [{_units.get(quantity, '')}]"
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Depth (m)")
        ax.set_title(f"EM Borehole Log  |  {_labels.get(quantity, quantity)}")
        ax.set_ylim(-self.borehole_length * 1.02, 0.0)
        if log_scale:
            _col_vals = result[quantity].replace([np.inf, -np.inf], np.nan).dropna()
            if (_col_vals > 0).any():
                ax.set_xscale("log")
            else:
                # No positive finite values — log scale not possible; keep linear
                warnings.warn(
                    f"Cannot apply log scale for quantity={quantity!r}: "
                    "no finite positive values in the data. Using linear scale.",
                    stacklevel=2,
                )
        ax.invert_yaxis()
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        return fig

    # ──────────────────────────────────────────────────────────────────────────
    def plot_model_and_log(
        self,
        result: pd.DataFrame,
        quantity: str = "apparent_resistivity",
        log_scale: Optional[bool] = None,
    ) -> plt.Figure:
        """
        Side-by-side plot: 1-D resistivity model cross-section and EM log(s).

        Parameters
        ----------
        result : pd.DataFrame
            Output of :meth:`run`.
        quantity : str, optional
            Column to plot on the log panel.  Choices are the same as in
            :meth:`plot_log`.  Defaults to ``'apparent_resistivity'``.
        log_scale : bool or None, optional
            Force logarithmic x-axis.  If ``None`` (default), applied
            automatically for ``'apparent_resistivity'`` and ``'amplitude'``.

        Returns
        -------
        matplotlib.figure.Figure
        """
        # Auto log-scale
        if log_scale is None:
            log_scale = quantity in ("apparent_resistivity", "amplitude")
        bh_total = self.borehole_length
        fig, (ax_model, ax_log) = plt.subplots(
            1, 2,
            figsize=(12, max(6, bh_total / 3)),
            sharey=True,
        )

        # ── Left panel: 1-D resistivity cross-section ─────────────────────────
        width = max(self.borehole_diameter * 20, 5.0)

        res_values = [self.background_resistivity]
        if self.layer_1d_geometry is not None:
            res_values.extend(self.layer_1d_geometry[:, 2].tolist())
        min_res = min(res_values)
        max_res = max(res_values)
        if np.isclose(min_res, max_res):
            min_res, max_res = min_res * 0.5, max_res * 2.0
        norm  = mcolors.LogNorm(vmin=min_res, vmax=max_res)
        cmap  = cm.jet

        if self.layer_1d_geometry is not None:
            for i, row in enumerate(self.layer_1d_geometry):
                top, bottom, res = float(row[0]), float(row[1]), float(row[2])
                if i == len(self.layer_1d_geometry) - 1:
                    bottom = max(bottom, 1.1 * bh_total)
                rect = patches.Rectangle(
                    (-width / 2, -bottom), width, bottom - top,
                    facecolor=cmap(norm(res)), edgecolor="k", linewidth=0.5,
                )
                ax_model.add_patch(rect)
                y_mid = -(top + bottom) / 2.0
                ax_model.text(
                    0.0, y_mid, f"{res:.1f} Ω·m",
                    ha="center", va="center", fontsize=8,
                    bbox=dict(
                        facecolor="white", alpha=0.7,
                        edgecolor="none", boxstyle="round,pad=0.15"
                    ),
                )
        else:
            rect = patches.Rectangle(
                (-width / 2, -1.1 * bh_total), width, 1.1 * bh_total,
                facecolor=cmap(norm(self.background_resistivity)), edgecolor="k",
            )
            ax_model.add_patch(rect)

        # Background half-space label
        ax_model.text(
            0.0, -1.05 * bh_total,
            f"BG: {self.background_resistivity:.1f} Ω·m",
            ha="center", va="center", fontsize=7, color="white",
            bbox=dict(facecolor="gray", alpha=0.7, edgecolor="none", boxstyle="round,pad=0.15"),
        )

        # Borehole rectangle
        bh_rect = patches.Rectangle(
            (-self.borehole_diameter / 2, -bh_total),
            self.borehole_diameter, bh_total,
            facecolor="white", edgecolor="navy", hatch="///", linewidth=1.0,
        )
        ax_model.add_patch(bh_rect)

        ax_model.set_xlim(-width / 2, width / 2)
        ax_model.set_ylim(-bh_total * 1.08, 0.1)
        ax_model.set_xlabel("x (m)")
        ax_model.set_ylabel("Depth (m)")
        ax_model.set_title("1-D Resistivity Model")
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        fig.colorbar(
            sm, ax=ax_model, orientation="vertical",
            fraction=0.05, pad=0.04, label="Resistivity (Ω·m)"
        )

        # ── Right panel: EM log ───────────────────────────────────────────────
        combos   = result.groupby(["spacing", "frequency"], sort=True)
        n_combos = len(combos)
        tab_cmap = cm.tab10
        colors   = [tab_cmap(i / max(n_combos - 1, 1)) for i in range(n_combos)]

        for idx, ((spacing, freq), grp) in enumerate(combos):
            grp_s = grp.sort_values("depth")
            ax_log.plot(
                grp_s[quantity],
                -grp_s["depth"],
                label=f"s={spacing} m, f={freq:.0f} Hz",
                color=colors[idx],
                linewidth=1.5,
            )

        # Layer boundary lines on log panel
        if self.layer_1d_geometry is not None:
            for row in self.layer_1d_geometry[1:]:
                ax_log.axhline(
                    -row[0], color="gray", linestyle="--", linewidth=0.8, alpha=0.6
                )

        _units = {
            "apparent_resistivity": "Ohm.m",
            "amplitude": "A/m",
            "real":      "A/m (real)",
            "imag":      "A/m (imag)",
            "phase":     "degrees",
        }
        _labels = {
            "apparent_resistivity": "Apparent Resistivity",
            "amplitude": "Amplitude",
            "real":  "Real Component",
            "imag":  "Imaginary Component",
            "phase": "Phase",
        }
        xlabel = f"{_labels.get(quantity, quantity)} [{_units.get(quantity, '')}]"
        ax_log.set_xlabel(xlabel)
        ax_log.set_title(f"EM Log  |  {_labels.get(quantity, quantity)}")
        if log_scale:
            _col_vals = result[quantity].replace([np.inf, -np.inf], np.nan).dropna()
            if (_col_vals > 0).any():
                ax_log.set_xscale("log")
            else:
                warnings.warn(
                    f"Cannot apply log scale for quantity={quantity!r}: "
                    "no finite positive values in the data. Using linear scale.",
                    stacklevel=2,
                )
        ax_log.invert_yaxis()
        ax_log.legend(fontsize=8, loc="lower right")
        ax_log.grid(True, linestyle="--", alpha=0.5)

        plt.suptitle(
            f"3-D FDEM Borehole Simulation  |  orientation={self.coil_orientation.upper()}",
            fontsize=11, fontweight="bold",
        )
        plt.tight_layout()
        return fig


# ──────────────────────────────────────────────────────────────────────────────
# Quick self-test
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import numpy as np

    layers = np.array([
        [0.0,   5.0, 100.0],
        [5.0,  10.0,  10.0],
        [10.0, 15.0, 200.0],
    ])

    em = EMGeometry(
        borehole_length=15.0,
        borehole_diameter=0.2,
        layer_1d_geometry=layers,
        backfill_resistivity=1.0,
        frequencies=[400.0, 8000.0],
        coil_spacings=[0.5, 1.0],
        coil_orientation="z",
        measuring_spacing=0.5,    # coarse for quick test
        mesh_type="cylindrical",  # Options: "cylindrical" (default), "tree"
    )

    print("=== Mesh inspection ===")
    mesh = em.make_mesh_3d()
    print(f"Mesh:  {mesh.nC:,} cells  ({mesh.shape_cells})")

    print("\n=== Conductivity model ===")
    sigma = em._paint_conductivity(mesh)
    print(f"sigma: min={sigma.min():.4g}  max={sigma.max():.4g}  S/m")

    print("\n=== Running 3D FDEM simulation ===")
    result = em.run()
    print(result[["depth","spacing","frequency","apparent_resistivity","amplitude","phase"]].head(10))

    fig = em.plot_model_and_log(result)   # defaults to apparent_resistivity
    plt.show()
