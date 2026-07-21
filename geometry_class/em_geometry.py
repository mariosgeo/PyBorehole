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
* The 1-D layer model is "painted" onto the mesh by z-coordinate of each
  cell centre — no borehole-fluid correction is applied.
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
        Diameter of the borehole (m) — used only for visualisation.
    layer_1d_geometry : np.ndarray, shape (N, 3), optional
        Layer table with columns ``[top_depth, bottom_depth, resistivity]``.
        Depths are positive-downward (m); resistivity in Ω·m.
    background_resistivity : float, optional
        Resistivity (Ω·m) assigned to all cells not covered by an explicit
        layer (i.e. above the first layer, below the last layer, and the
        outer far-field).  Defaults to 5.0.
    frequencies : list of float
        Transmitter frequencies (Hz), e.g. ``[400.0, 8000.0]``.
    coil_spacings : list of float
        Transmitter–receiver coil offsets (m), e.g. ``[0.5, 1.0, 2.0]``.
        Multiple spacings are simulated simultaneously (one receiver per
        spacing per source).
    coil_orientation : str
        Coil-axis orientation:

        * ``'z'`` — Vertical Magnetic Dipole (VMD); coils are horizontal
          (HCP configuration).  Default.
        * ``'x'`` or ``'y'`` — Horizontal Magnetic Dipole (HMD); coils are
          vertical (VCP / HLP configuration).

    measuring_spacing : float, optional
        Depth step for sliding the tool (m).  Defaults to 0.1 m.
    receiver_above : bool, optional
        If ``False`` (default) the receiver is placed *below* the transmitter
        (deeper in borehole).  Set ``True`` to reverse.
    core_cell_size : float, optional
        Edge length (m) of the uniform fine cells in the mesh core region
        around the borehole.  Should be ≤ skin-depth / 5 at the highest
        frequency in the most conductive layer.  Defaults to 0.1 m.
    n_padding : int, optional
        Number of geometric-expansion padding cells on each boundary side.
        Defaults to 10.
    pad_factor : float, optional
        Geometric expansion ratio per padding cell (each cell is
        ``pad_factor`` times wider than its inner neighbour).  Defaults to 1.4.
    domain_radius : float, optional
        Outer boundary radius (m) for both the radial far-field and the
        vertical far-field padding.  Defaults to 100.0 m.  Should be at
        least 2–3 skin depths at the lowest frequency.
    n_theta_cells : int, optional
        Number of azimuthal cells in the CylindricalMesh.

        * ``1`` (default) — axisymmetric; exact for VMD (``coil_orientation='z'``)
          and approximately valid for HMD in horizontally layered media.
        * ``≥ 8`` — recommended for HMD (``coil_orientation='x'`` or ``'y'``);
          increases cell count by ``n_theta_cells`` but is still far smaller
          than a full TensorMesh.

    Attributes
    ----------
    The above constructor arguments are stored as instance attributes.
    """

    # ── Shared geometry (mirrors Geometry) ─────────────────────────────────────
    borehole_length: float
    borehole_diameter: float
    layer_1d_geometry: Optional[np.ndarray] = None
    background_resistivity: float = 5.0

    # ── EM survey parameters ───────────────────────────────────────────────────
    frequencies: List[float] = field(default_factory=lambda: [1000.0])
    coil_spacings: List[float] = field(default_factory=lambda: [1.0])
    coil_orientation: str = "z"
    measuring_spacing: float = 0.1
    receiver_above: bool = False

    # ── 3-D mesh control (CylindricalMesh) ────────────────────────────────────
    core_cell_size: float = 0.05
    n_padding: int = 10
    pad_factor: float = 1.4
    domain_radius: float = 100.0
    n_theta_cells: int = 1

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

        # ── HMD + single theta cell advisory ─────────────────────────────────
        if self.coil_orientation in ("x", "y") and self.n_theta_cells < 3:
            warnings.warn(
                f"HMD orientation (coil_orientation='{self.coil_orientation}') with "
                f"n_theta_cells={self.n_theta_cells} is not azimuthally symmetric. "
                "Results will be approximate. Consider n_theta_cells >= 8 for "
                "accurate HMD simulation.",
                stacklevel=2,
            )

        # ── EM parameter validation ───────────────────────────────────────────
        self.coil_orientation = self.coil_orientation.lower()
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

        # ── Skin-depth advisory ───────────────────────────────────────────────
        self._check_skin_depth()

    # ──────────────────────────────────────────────────────────────────────────
    def _check_skin_depth(self) -> None:
        """Issue a warning if ``core_cell_size`` is too coarse for the EM problem."""
        max_freq = max(self.frequencies)
        res_values = [self.background_resistivity]
        if self.layer_1d_geometry is not None:
            res_values.extend(self.layer_1d_geometry[:, 2].tolist())
        min_res = min(res_values)
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
    def make_mesh_3d(self) -> "_discretize.CylindricalMesh":
        """
        Build and return the 3-D ``CylindricalMesh`` for the FDEM simulation.

        Mesh layout  (r, theta, z)
        --------------------------
        * **r** — radial axis from the borehole axis outward.
          Fine cells (width = ``core_cell_size``) extend to
          ``3 × max(coil_spacings)``; geometric padding expands to
          ``domain_radius``.
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

        cs   = self.core_cell_size
        pfac = self.pad_factor

        # ── Radial (r) ────────────────────────────────────────────────────────
        # Fine cells: axis → 3× largest coil spacing (resolves the TX-RX geometry)
        r_fine = max(3.0 * max(self.coil_spacings), self.borehole_diameter * 10.0)
        n_fine_r = max(5, int(np.ceil(r_fine / cs)))
        # Geometric padding: r_fine → domain_radius
        npad_r = self._n_pad_cells(r_fine, self.domain_radius, cs, pfac)
        hr = [(cs, n_fine_r), (cs, npad_r, pfac)]

        # ── Azimuthal (theta) ─────────────────────────────────────────────────
        n_th = max(1, self.n_theta_cells)
        if n_th == 1:
            h_theta = [2.0 * np.pi]          # single cell covers full circle
        else:
            dtheta = 2.0 * np.pi / n_th
            h_theta = [dtheta] * n_th        # uniform azimuthal cells

        # ── Vertical (z) ──────────────────────────────────────────────────────
        # Fine core: covers borehole column (z = -borehole_length to 0)
        n_core_z = int(np.ceil(self.borehole_length / cs)) + 1
        n_above  = max(3, self.n_padding // 3)  # a few cells above z=0
        # Geometric padding below (borehole bottom → -domain_radius)
        npad_z_below = self._n_pad_cells(0.0, self.domain_radius, cs, pfac)
        # Geometric padding above (n_above cells → +domain_radius above surface)
        npad_z_above = self._n_pad_cells(n_above * cs, self.domain_radius, cs, pfac)

        hz = [
            (cs, npad_z_below, -pfac),  # below borehole — expanding downward
            (cs, n_core_z),             # fine borehole column
            (cs, n_above),              # a few uniform cells above surface
            (cs, npad_z_above, pfac),   # above surface — expanding upward
        ]

        # ── Origin ───────────────────────────────────────────────────────────
        # r always starts at 0 in CylindricalMesh (axis)
        # theta always starts at 0
        # z0: set so that TOP of the fine borehole column = 0 (surface)
        #   z0 + sum_pad_below + n_core_z·cs = 0
        sum_pad_below = cs * (pfac**npad_z_below - 1.0) / (pfac - 1.0)
        z0 = -(sum_pad_below + n_core_z * cs)

        mesh = _discretize.CylindricalMesh([hr, h_theta, hz], origin=[0.0, 0.0, z0])
        return mesh

    # ──────────────────────────────────────────────────────────────────────────
    def _paint_conductivity(
        self, mesh: "_discretize.TensorMesh"
    ) -> np.ndarray:
        """
        Assign per-cell conductivity (S/m) based on the 1-D layer model.

        Layer assignment is done by the z-coordinate of each cell centre
        (z positive upward, surface at z=0):

        * Cells with ``z_bottom < z_center ≤ z_top`` belong to that layer.
        * All remaining cells (air above surface, half-space below last layer,
          and the lateral far-field) receive ``background_resistivity``.

        Parameters
        ----------
        mesh : discretize.TensorMesh

        Returns
        -------
        np.ndarray, shape (mesh.nC,)
            Conductivity (S/m) at every cell centre.
        """
        z_cc = mesh.gridCC[:, 2]  # z-coordinates of all cell centres
        sigma = np.full(mesh.nC, 1.0 / self.background_resistivity)  # background

        if self.layer_1d_geometry is not None:
            for row in self.layer_1d_geometry:
                top_depth    = float(row[0])
                bottom_depth = float(row[1])
                resistivity  = float(row[2])

                z_top    = -top_depth     # e.g. top_depth=0  → z=0
                z_bottom = -bottom_depth  # e.g. bottom_depth=5 → z=−5

                in_layer = (z_cc > z_bottom) & (z_cc <= z_top)
                sigma[in_layer] = 1.0 / resistivity

        return sigma

    # ──────────────────────────────────────────────────────────────────────────
    def _build_survey(
        self, depths: np.ndarray
    ) -> "fdem.survey.Survey":
        """
        Construct the FDEM survey with all sources and receivers.

        **Construction order** (critical for correct ``dpred`` parsing):

        .. code-block::

            for freq in self.frequencies:          # outer loop
                for depth in depths:               # middle loop
                    source = MagDipole(...)
                        for spacing in self.coil_spacings:   # inner loop
                            receiver_real = PointMagneticFluxDensitySecondary(..., 'real')
                            receiver_imag = PointMagneticFluxDensitySecondary(..., 'imag')

        Note
        ----
        SimPEG 0.21+ uses ``PointMagneticFluxDensitySecondary`` (B-field, T).
        The ``run()`` method converts B → H by dividing by μ₀ so that the
        returned DataFrame columns remain in A/m.

        Parameters
        ----------
        depths : np.ndarray
            Tool depth positions (m, positive-downward).

        Returns
        -------
        fdem.survey.Survey
        """
        source_list: list = []

        for freq in self.frequencies:
            for depth in depths:
                tx_z   = -float(depth)
                tx_loc = np.array([[0.0, 0.0, tx_z]])

                receivers: list = []
                for spacing in self.coil_spacings:
                    if self.receiver_above:
                        rx_z = -(depth - spacing)
                    else:
                        rx_z = -(depth + spacing)

                    rx_loc = np.array([[0.0, 0.0, float(rx_z)]])

                    receivers.append(
                        fdem.receivers.PointMagneticFluxDensitySecondary(
                            locations=rx_loc,
                            orientation=self.coil_orientation,
                            component="real",
                        )
                    )
                    receivers.append(
                        fdem.receivers.PointMagneticFluxDensitySecondary(
                            locations=rx_loc,
                            orientation=self.coil_orientation,
                            component="imag",
                        )
                    )

                src = fdem.sources.MagDipole(
                    receiver_list=receivers,
                    frequency=freq,
                    location=tx_loc,
                    orientation=self.coil_orientation,
                )
                source_list.append(src)

        return fdem.survey.Survey(source_list)

    # ──────────────────────────────────────────────────────────────────────────
    def run(self, verbose: bool = True) -> pd.DataFrame:
        """
        Run the 3-D FDEM forward simulation.

        Workflow
        --------
        1. Build a 3-D ``TensorMesh`` via :meth:`make_mesh_3d`.
        2. Paint the 1-D layer conductivities onto the mesh via
           :meth:`_paint_conductivity`.
        3. Build the FDEM survey via :meth:`_build_survey`.
        4. Instantiate ``Simulation3DElectricField`` with
           ``sigmaMap = IdentityMap(nP=mesh.nC)``.
        5. Call ``sim.dpred(sigma)`` — SimPEG batches all sources that share
           the same frequency into a single factorised linear solve.
        6. Parse the flat ``dpred`` vector (same order as source construction)
           into a tidy ``pd.DataFrame``.

        Parameters
        ----------
        verbose : bool, optional
            Print progress messages.  Defaults to ``True``.

        Returns
        -------
        pd.DataFrame
            One row per ``(depth, spacing, frequency)`` combination with columns:

            * ``depth``     — tool depth (m, positive-downward)
            * ``spacing``   — coil spacing (m)
            * ``frequency`` — frequency (Hz)
            * ``real``      — real part of H_secondary (A/m)  [= B_real/μ₀]
            * ``imag``      — imaginary part of H_secondary (A/m)  [= B_imag/μ₀]
            * ``amplitude`` — |H_secondary| (A/m)
            * ``phase``     — phase angle (°)

        Note
        ----
        Internally uses ``PointMagneticFluxDensitySecondary`` (B-field, Tesla)
        and divides by μ₀ = 4π×10⁻⁷ H/m to yield H in A/m.

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

        # ── 1. Mesh ───────────────────────────────────────────────────────────
        if verbose:
            n_th = max(1, self.n_theta_cells)
            mesh_type = "CylindricalMesh (axisymmetric)" if n_th == 1 else f"CylindricalMesh ({n_th} theta cells)"
            print(f"Building 3-D {mesh_type} …")
        mesh = self.make_mesh_3d()
        if verbose:
            sr, st, sz = mesh.shape_cells
            print(f"  Mesh: {mesh.nC:,} cells  (n_r={sr}, n_theta={st}, n_z={sz})")

        # ── 2. Conductivity model ─────────────────────────────────────────────
        if verbose:
            print("Painting conductivity model onto mesh …")
        sigma = self._paint_conductivity(mesh)
        if verbose:
            print(
                f"  σ range: {sigma.min():.4g} – {sigma.max():.4g} S/m  "
                f"(ρ range: {1/sigma.max():.4g} – {1/sigma.min():.4g} Ω·m)"
            )

        # ── 3. Survey ─────────────────────────────────────────────────────────
        n_depths = int(np.floor(self.borehole_length / self.measuring_spacing)) + 1
        depths   = np.arange(n_depths) * self.measuring_spacing

        n_sources = len(self.frequencies) * n_depths
        if verbose:
            print(
                f"Building survey: "
                f"{len(self.frequencies)} freq(s) × {n_depths} depths "
                f"× {len(self.coil_spacings)} spacing(s) "
                f"= {n_sources} sources …"
            )
        survey = self._build_survey(depths)

        # ── 4. Simulation ─────────────────────────────────────────────────────
        if verbose:
            print("Running Simulation3DElectricField …")
            print(
                "  SimPEG performs one LU factorisation per frequency and solves "
                f"all {n_depths} depth RHS vectors simultaneously."
            )

        try:
            from simpeg.utils import get_default_solver
            default_solver = get_default_solver()
        except Exception:
            default_solver = None

        sim_kwargs = dict(
            mesh=mesh,
            survey=survey,
            sigmaMap=_maps.IdentityMap(nP=mesh.nC),
        )
        if default_solver is not None:
            sim_kwargs["solver"] = default_solver

        sim = fdem.Simulation3DElectricField(**sim_kwargs)

        dpred_vec = sim.dpred(sigma)

        if verbose:
            print("  Forward simulation complete.")

        # ── 5. Parse dpred ────────────────────────────────────────────────────
        # dpred order mirrors the source_list construction order exactly:
        #   outer: frequencies → middle: depths → inner: spacings (real, imag pairs)
        # dpred values are B-field (T); divide by μ₀ to convert to H-field (A/m)
        MU0 = 4.0 * np.pi * 1e-7  # H/m
        records: list = []
        idx = 0
        for freq in self.frequencies:
            for depth in depths:
                for spacing in self.coil_spacings:
                    b_real = float(dpred_vec[idx])
                    b_imag = float(dpred_vec[idx + 1])
                    idx += 2
                    # Convert B (T) → H (A/m):  H = B / μ₀
                    h_real = b_real / MU0
                    h_imag = b_imag / MU0
                    records.append(
                        {
                            "depth":     float(depth),
                            "spacing":   float(spacing),
                            "frequency": float(freq),
                            "real":      h_real,
                            "imag":      h_imag,
                            "amplitude": float(np.sqrt(h_real**2 + h_imag**2)),
                            "phase":     float(np.degrees(np.arctan2(h_imag, h_real))),
                        }
                    )

        if verbose:
            print(f"  Parsed {len(records)} data points into DataFrame.")

        return pd.DataFrame(records)

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
        fig, ax = plt.subplots(figsize=figsize)

        # For CylindricalMesh: gridCC columns are (r, theta, z)
        # Keep only one theta-slice (they all represent the same r-z plane)
        r_cc   = mesh.gridCC[:, 0]   # radial coordinate
        z_cc   = mesh.gridCC[:, 2]   # vertical coordinate
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
        quantity: str = "amplitude",
        log_scale: bool = False,
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
            Column to plot: ``'amplitude'``, ``'phase'``, ``'real'``, or
            ``'imag'``.  Defaults to ``'amplitude'``.
        log_scale : bool, optional
            Logarithmic x-axis.  Useful for amplitude.  Defaults to ``False``.
        figsize : tuple, optional
            Figure size.  Auto-sized if ``None``.

        Returns
        -------
        matplotlib.figure.Figure
        """
        _valid = ("amplitude", "phase", "real", "imag")
        if quantity not in _valid:
            raise ValueError(f"quantity must be one of {_valid}.")

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
            "amplitude": "A/m",
            "real":      "A/m (real)",
            "imag":      "A/m (imag)",
            "phase":     "degrees",
        }
        ax.set_xlabel(f"{quantity.capitalize()} [{_units[quantity]}]")
        ax.set_ylabel("Depth (m)")
        ax.set_title(f"EM Borehole Log — {quantity.capitalize()}")
        ax.set_ylim(-self.borehole_length * 1.02, 0.0)
        if log_scale:
            ax.set_xscale("log")
        ax.invert_yaxis()
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        return fig

    # ──────────────────────────────────────────────────────────────────────────
    def plot_model_and_log(
        self,
        result: pd.DataFrame,
        quantity: str = "amplitude",
        log_scale: bool = False,
    ) -> plt.Figure:
        """
        Side-by-side plot: 1-D resistivity model cross-section and EM log(s).

        Parameters
        ----------
        result : pd.DataFrame
            Output of :meth:`run`.
        quantity : str, optional
            Column to plot on the log panel.  See :meth:`plot_log`.
        log_scale : bool, optional
            Logarithmic x-axis for the log panel.  Defaults to ``False``.

        Returns
        -------
        matplotlib.figure.Figure
        """
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
            "amplitude": "A/m",
            "real":      "A/m (real)",
            "imag":      "A/m (imag)",
            "phase":     "degrees",
        }
        ax_log.set_xlabel(f"{quantity.capitalize()} [{_units.get(quantity, '')}]")
        ax_log.set_title(f"EM Log — {quantity.capitalize()}")
        if log_scale:
            ax_log.set_xscale("log")
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
        background_resistivity=50.0,
        frequencies=[400.0, 8000.0],
        coil_spacings=[0.5, 1.0],
        coil_orientation="z",
        measuring_spacing=0.5,    # coarse for quick test
        core_cell_size=0.05,       # coarse for quick test
        n_padding=8,
        domain_radius=50.0,
    )

    print("=== Mesh inspection ===")
    mesh = em.make_mesh_3d()
    print(f"Mesh:  {mesh.nC:,} cells  ({mesh.shape_cells})")

    print("\n=== Conductivity model ===")
    sigma = em._paint_conductivity(mesh)
    print(f"sigma: min={sigma.min():.4g}  max={sigma.max():.4g}  S/m")

    print("\n=== Running 3D FDEM simulation ===")
    result = em.run()
    print(result.head(10))

    fig = em.plot_model_and_log(result, quantity="amplitude")
    plt.show()
