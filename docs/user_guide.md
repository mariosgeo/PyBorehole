# PyBorehole user guide

This guide covers everything beyond the quick start in the [README](../README.md): the model, all parameters of the two simulator classes, their outputs, how to trade accuracy against run time, the physics, and common problems.

Most users only need [`run_combined.py`](../run_combined.py). Read on if you want to change the borehole build-up, use the classes in your own scripts, or understand what the numbers mean.

## Contents

1. [The model](#1-the-model)
2. [ERT simulator: `Geometry`](#2-ert-simulator-geometry)
3. [EM simulator: `EMGeometry`](#3-em-simulator-emgeometry)
4. [Measuring the borehole effect](#4-measuring-the-borehole-effect)
5. [Accuracy and run time](#5-accuracy-and-run-time)
6. [Physics and boundary conditions](#6-physics-and-boundary-conditions)
7. [Testing](#7-testing)
8. [Troubleshooting](#8-troubleshooting)
9. [Gamma-ray simulator: `GammaGeometry`](#9-gamma-ray-simulator-gammageometry)

---

## 1. The model

Both simulators read the same model, defined in [`geometry_class/borehole_model.py`](../geometry_class/borehole_model.py), so the ERT and EM logs are directly comparable.

**Coordinates.** The borehole axis is vertical through `x = y = 0`. Depths are positive downward from the surface (depth 0). Internally the meshes use `z = -depth`.

**Layers** (`layer_1d_geometry`). An array with one row per layer, `[top_depth, bottom_depth, resistivity]`:

```python
layers = np.array([
    [0.0,   5.0, 100.0],
    [5.0,  10.0,  10.0],
    [10.0, 15.0, 200.0],
])
```

- The first top depth is forced to 0 (with a warning).
- The last layer continues downward to the bottom of the model.
- Without layers, the formation is a half-space of resistivity `background_resistivity`.

**Borehole.** A vertical cylinder from the surface to `borehole_length`, with diameter `borehole_diameter`. Its content is either:

- one material, given by `backfill_resistivity`, or
- concentric zones, given by `borehole_zones`. Each zone is a tuple `(r_outer, resistivity[, top[, bottom]])` and fills the ring from the previous zone's radius (or the axis) out to `r_outer`, between depths `top` (default 0) and `bottom` (default: the borehole bottom). Outside its depth interval the ring belongs to the formation.

```python
# PVC-cased well: water inside the casing, gravel pack outside, bentonite seal at the top
zones = [
    (0.045, 20.0),             # water inside the casing
    (0.050, 1e6),              # 5 mm PVC wall
    (0.100, 30.0, 2.0, 15.0),  # gravel pack from 2 to 15 m
]
```

Rules and current limits:

- The outermost zone radius may not exceed the borehole radius.
- Every zone needs its own radius, so one ring cannot hold two materials at different depths. In this example the ring between 5 and 10 cm is formation from 0 to 2 m; a bentonite seal in that interval can't be added in the same run.
- Thin zones (such as a 5 mm casing wall) are resolved exactly but make the ERT mesh larger, because cells have to be smaller than the zone is thick.
- A closed PVC wall blocks the ERT current, as in reality: galvanic tools only see the formation through open hole or a slotted screen.

**Legacy behaviour.** If neither `backfill_resistivity` nor `borehole_zones` is given, `background_resistivity` is used as the borehole fill, with a warning. This keeps old scripts and `demo.ipynb` working.

**Air** (EM only) has resistivity `air_resistivity` (default 1e8 Ohm.m). ERT needs no air: the surface is a no-flux boundary.

---

## 2. ERT simulator: `Geometry`

[`geometry_class/geometry.py`](../geometry_class/geometry.py) uses PyGIMLi and TetGen.

### Workflow

```python
geo  = Geometry(borehole_length=15, borehole_diameter=0.2, layer_1d_geometry=layers,
                backfill_resistivity=10.0, long_spacing=1.0)
plc  = geo.make_basic_geometry()                  # 1. geometry (PLC)
data = geo.make_array(plc, (1, 4, 2, 3), 0.5, 1.0, 0.1)  # 2. electrodes + protocol
mesh = geo.make_mesh(plc)                         # 3. TetGen mesh
sim  = geo.run(mesh, data)                        # 4. forward model
geo.plot_model_and_data(sim)                      # 5. quick plot
```

The order matters: electrodes are added to the geometry before meshing, so every electrode becomes a mesh node.

### Parameters

| Parameter | Default | Meaning |
|---|---|---|
| `borehole_length` | required | Borehole depth (m). |
| `borehole_diameter` | required | Borehole diameter (m). |
| `layer_1d_geometry` | `None` | Layer table (see §1). |
| `background_resistivity` | 5.0 | Half-space resistivity when there are no layers; legacy backfill (see §1). |
| `backfill_resistivity` | `None` | Resistivity of a uniform backfill (Ohm.m). |
| `borehole_zones` | `None` | Concentric zones (see §1); overrides `backfill_resistivity`. |
| `long_spacing` | 1.0 | Largest electrode spacing you will use (m). Sizes the refined zone around the hole. |
| `n_segments` | 24 | Number of sides of the polygons that approximate circles. |
| `near_radius` | 10 × borehole radius | Radius of the finely meshed zone around the hole (m). |
| `mid_radius` | max(3 × `long_spacing`, 2 × `near_radius`) | Radius of the medium zone (m). |
| `world_radius` | 100.0 | Radius of the model (m). |
| `world_depth` | `borehole_length + world_radius` | Depth of the model bottom (m). |
| `borehole_area`, `near_area`, `mid_area` | from the zone sizes | Maximum tetrahedron volume (m³) in each zone. |
| `world_area` | 0 | Maximum volume in the far field; 0 = no limit. |
| `p2` | `True` | Quadratic elements. Keep `True` (see §5). |
| `tool_diameter` | `None` | Diameter of an insulating tool body (m), centred, over the whole borehole length. Electrodes then become rings on its surface and the geometric factor is calibrated (see [Realistic tools](#realistic-tools)). |
| `tool_resistivity` | 1e6 | Resistivity of the tool body (Ohm.m). |
| `ring_azimuths` | 4 | Points around the tool averaged to form a ring electrode; 1 = point electrodes (up to ~20% off for short spacings in resistive backfill). Must divide the 8 sides of the tool polygon. |

### Methods

| Method | What it does |
|---|---|
| `make_basic_geometry(vtk_filename="plc.vtk")` | Builds the geometry: concentric polygons (borehole zones → near zone → mid zone → world) cut by horizontal planes at every layer interface, zone top/bottom and the borehole bottom. Each piece is its own region. Pass `vtk_filename=None` to skip the VTK file. |
| `make_array(geom, abmn_order, short_spacing, long_spacing, measuring_spacing, protocol_filename="protocol.dat", electrode_offset=None)` | Puts electrodes every `measuring_spacing` along the axis, starting at depth `electrode_offset`, and builds the protocol for the two spacings. `abmn_order` gives the positions of A, B, M, N in the array, e.g. `(1, 4, 2, 3)` = Wenner A–M–N–B. By default the first electrode is at 0, unless the grid would put an electrode exactly on a layer interface, zone boundary or the borehole bottom: then the grid is shifted by half a step (with a warning), because a point electrode on a resistivity jump converges poorly. If `short_spacing` is smaller than the borehole cell size, the borehole cells are refined to match. Returns a `pg.DataContainerERT`. |
| `make_tool_array(geom, array="normal", spacings=(0.4064, 1.6256), measuring_spacing=0.1, mn_spacing=0.8128, electrode_offset=None, protocol_filename=None)` | Logging arrays with remote electrodes. `"normal"`: A and M on the sonde (M above A at distance AM), B and N remote. `"lateral"`: A on the sonde, M and N above it centred at distance AO and `mn_spacing` apart, B remote. Adds the columns `depthref` (AM midpoint for the normal, O for the lateral) and `spacing`. |
| `merge_protocols(*datas)` (module function) | Merges protocols built on the same geometry into one, so they are simulated in a single run; adds a `protocol` column with the index of the source protocol. |
| `make_mesh(geom, vtk_filename="mesh.vtk", quality=1.3)` | Runs TetGen and sets the boundary markers. `quality` is TetGen's radius–edge ratio; keep 1.2–1.4. |
| `run(mesh, data, rhomap=None, output_filename="simulated_data.dat", noise_level=0, noise_abs=0, include_borehole=True)` | Forward model. Returns a data container with `rhoa` (apparent resistivity), `r` (resistance) and `k` (geometric factor). `include_borehole=False` replaces the hole by formation. |
| `set_borehole(backfill_resistivity=None, borehole_zones=None)` | Changes the borehole resistivities on the existing mesh. The zone radii and depths must stay the same. |
| `cell_resistivity(mesh, include_borehole=True)` | Resistivity of every cell, e.g. for plotting. |
| `resistivity_at(r, depth, include_borehole=True)` | Model resistivity at any point. |
| `plot_model_and_data(data)` | Model section plus the two spacing logs. |
| `plot_boundary_conditions(mesh)` | 3D view of the boundary markers (PyVista). |

---

## 3. EM simulator: `EMGeometry`

[`geometry_class/em_geometry.py`](../geometry_class/em_geometry.py) uses SimPEG's frequency-domain E-field formulation.

### Workflow

```python
em  = EMGeometry(borehole_length=15, borehole_diameter=0.2, layer_1d_geometry=layers,
                 backfill_resistivity=10.0, frequencies=[400, 8000], coil_spacings=[0.5, 1.0])
log = em.run()
em.plot_model_and_log(log)
```

Everything happens inside `run()`: mesh, model, survey, solve. With the cylindrical mesh SimPEG factorises the system once per frequency and solves all tool positions together, so a full log takes seconds.

`run(depths=[...])` simulates only the listed transmitter depths instead of the full log. Use it with the 3D octree mesh, which solves each position separately:

```python
# Off-centre tool, 2 cm from the wall of a 0.2 m hole, at three depths
em3d = EMGeometry(borehole_length=15, borehole_diameter=0.2, layer_1d_geometry=layers,
                  backfill_resistivity=1.0, frequencies=[400], coil_spacings=[0.5, 1.0],
                  mesh_type="tree", tool_offset=0.08)
log3d = em3d.run(depths=[2.0, 7.0, 12.0])
```

### Parameters

| Parameter | Default | Meaning |
|---|---|---|
| `borehole_length`, `borehole_diameter`, `layer_1d_geometry`, `background_resistivity`, `backfill_resistivity`, `borehole_zones` | | As for `Geometry` (§1). |
| `air_resistivity` | 1e8 | Resistivity of the air (Ohm.m). |
| `frequencies` | `[1000.0]` | Frequencies (Hz). |
| `coil_spacings` | `[1.0]` | Transmitter–receiver distances (m). Must be shorter than the borehole. |
| `coil_orientation` | `"z"` | `"z"`: coil axes along the hole (coaxial, as in standard induction tools). `"x"`/`"y"`: across the hole (coplanar); needs `n_theta_cells >= 8` or `mesh_type="tree"`. |
| `measuring_spacing` | 0.1 | Tool step along the hole (m). |
| `receiver_above` | `False` | Receiver below (default) or above the transmitter. |
| `mesh_type` | `"cylindrical"` | `"cylindrical"`: axisymmetric; a full log in seconds; exact for a centred tool with coaxial coils. `"tree"`: full 3D octree, one mesh per tool position (~75k cells, ~10 s per position and frequency for a 0.2 m hole); needed for `tool_offset > 0` and recommended for `coil_orientation="x"`/`"y"`. Both are validated against the analytic borehole solution. |
| `tool_offset` | 0.0 | Distance of the tool axis from the borehole axis (m); must be inside the hole. Needs `mesh_type="tree"`. |
| `tree_cell_size` | `None` | Finest octree cell (m). Default min(borehole radius / 5, thinnest zone / 2, smallest spacing / 20), i.e. 2 cm for a 0.2 m hole. Halving it changes results by < 0.5% but costs ~10× more time. |
| `tree_padding` | 4 | Cells of each size before the octree coarsens by 2. |
| `core_cell_size` | 0.05 | Radial cell size outside the hole (m). The vertical cell size is `min(core_cell_size, smallest spacing / 20)`. |
| `borehole_cells` | 10 | Radial cells across the innermost borehole zone. |
| `domain_radius` | `None` | Model radius (m). `None` = max(100 m, 3 skin depths in the most resistive layer at the lowest frequency). |
| `pad_factor` | 1.4 | Growth factor of the padding cells. |
| `n_padding` | 10 | Sets the number of uniform cells above the surface (`n_padding // 4`). |
| `n_theta_cells` | 1 | Azimuthal cells of the cylindrical mesh; 1 is exact for coaxial coils. |
| `bucking_spacings` | `None` | One entry per coil spacing: the distance of a bucking receiver from the transmitter (m, smaller than the spacing) or `None` for a plain two-coil pair. See [Realistic tools](#realistic-tools). |

### Output of `run()`

A pandas DataFrame with one row per tool position, coil spacing and frequency:

| Column | Meaning |
|---|---|
| `depth` | Tool centre: midpoint between transmitter and receiver (m). |
| `tx_depth` | Transmitter depth (m). |
| `spacing` | Coil spacing (m). |
| `bucking` | Bucking receiver distance (m), NaN for a two-coil pair. Fields and apparent resistivities are those of the combined array. |
| `frequency` | Frequency (Hz). |
| `real`, `imag` | Secondary magnetic field H (A/m) for a unit-moment transmitter. |
| `amplitude`, `phase` | Magnitude (A/m) and phase (degrees) of the secondary field. |
| `apparent_resistivity` | Resistivity of the homogeneous full space that produces the same quadrature field (Ohm.m). Exact at any frequency. |
| `apparent_resistivity_lin` | Low-induction-number approximation of the same (Ohm.m); too high by up to ~13% in 1 Ohm.m ground at 8000 Hz with s = 1 m. |

Both coils always stay inside the borehole, so a log covers depths from s/2 to `borehole_length` − s/2.

Other methods: `make_mesh_3d()` returns the cylindrical mesh, `plot_mesh_slice(mesh, sigma)` shows it, and `plot_log(result, quantity)` / `plot_model_and_log(result, quantity)` plot any output column.

---

## 4. Measuring the borehole effect

The borehole effect is the ratio between the same tool with and without the hole:

```python
ratio_ert = np.array(geo.run(mesh, data)("rhoa")) / \
            np.array(geo.run(mesh, data, include_borehole=False)("rhoa"))

with_hole, no_hole = em.run(), em.run(include_borehole=False)
ratio_em = with_hole.apparent_resistivity / no_hole.apparent_resistivity
```

`include_borehole=False` fills the hole with formation and leaves everything else (mesh, tool positions, layers) unchanged, so the ratio isolates the hole. It removes the layer and shoulder effects that a comparison against the true resistivity would mix in.

For a backfill sweep, build the ERT mesh once and change only the resistivity with `geo.set_borehole(backfill_resistivity=...)`; `run_combined.py` does exactly this.

---

## 5. Accuracy and run time

### ERT

The default mesh reproduces the analytic borehole solution within 0.1% for Wenner spacings from a = R (borehole radius) to 10 R. In a layered model (100/10/200 Ohm.m, 1 Ohm.m backfill), a 1.7× finer mesh changes the logs by at most 0.04%, both with and without the hole. This relies on the electrode grid avoiding interfaces (see `make_array`): with an electrode exactly on an interface the no-hole reference was off by up to 5%. Its cost depends mostly on the borehole diameter, the electrode step and the borehole length:

| Default case (15 m, D = 0.2 m, 151 electrodes) | |
|---|---|
| Cells | ~375 000 |
| Meshing | ~25 s |
| One solve (P2) | ~1–2.5 min |
| Peak memory | ~12 GB |

To make it cheaper, in order of preference:

1. Increase `ERT_MEAS_SPACING` / `measuring_spacing` (fewer electrodes, fewer refined nodes).
2. Simulate a shorter borehole section.
3. Increase `near_area` / `borehole_area` and check the result against the analytic test (`pytest tests -m slow`).
4. `p2=False` halves the cost but leaves a bias of a few percent for strong borehole contrasts.

Settings that matter for accuracy, which you normally shouldn't change:

- `quality` in `make_mesh`: values above 2 allow badly shaped tetrahedra. The old default of 34.2 is a 2D value and gave errors of about 17%.
- `n_segments`: fewer than 12 sides misrepresents the hole; the polygons are scaled to the same area as the true circle.
- `p2`: linear elements cannot resolve the field around an electrode inside a high-contrast hole.

### EM

The defaults are convergence-tested: the borehole effect changes by less than 1.5% when the radial cells are refined 4× and by less than 0.4% for finer vertical cells. A full log runs in seconds. If you change the borehole diameter a lot, keep `borehole_cells` at 10 or more and check that `core_cell_size` stays well below the skin depth (the class warns if it doesn't).

---

## 6. Physics and boundary conditions

### ERT

- **Equation:** DC potential, ∇·(σ∇u) = −I δ(r − r_s), solved with finite elements (PyGIMLi).
- **Boundaries:** the surface is no-flux (marker −1); the outer side and the bottom use a mixed (Robin) condition (marker −2), which assumes the field decays like a point source there.
- **Singularity removal:** the potential near each electrode is split into an analytic part and a smooth numerical part, which is what makes small electrode spacings accurate.
- **Apparent resistivity:** ρa = k · ΔV / I with the analytic half-space geometric factor *k*, which includes image electrodes above the surface. A homogeneous earth therefore returns its true resistivity at any depth.

### EM

- **Equation:** frequency-domain Maxwell equations, E-field formulation (SimPEG), with a point magnetic dipole source of unit moment. SimPEG uses the e^{+iωt} time convention.
- **Boundaries:** tangential E = 0 at the outer boundary, placed at least 3 skin depths away so that it doesn't affect the result. The axis of the cylindrical mesh is handled exactly.
- **Measured quantity:** the secondary magnetic field (total minus free-space primary) at the receiver.
- **Apparent resistivity** (`apparent_resistivity`): the resistivity of the homogeneous full space whose quadrature field −Im H_s equals the simulated one, found by inverting the exact full-space dipole formula. The quadrature of a full space rises with conductivity up to an induction number s/δ of about 1–2 and then falls; only the rising branch is used, so the result is unique.
- **LIN apparent resistivity** (`apparent_resistivity_lin`): σa = k · s · (−Im H_s) / (μ₀ ω), with k = 4π for coaxial and 8π for coplanar coils. It assumes s ≪ δ ≈ 503 √(ρ/f) m; in a homogeneous earth it is too high by 0.3% (100 Ohm.m, 400 Hz, s = 1 m) up to 13% (1 Ohm.m, 8000 Hz, s = 1 m).
- **In-phase field:** at these induction numbers the in-phase secondary field is ~1e-8 of the primary, below the accuracy of any mesh, so `real`, `amplitude` and `phase` should not be interpreted.

### EM in 3D: the octree mesh

The octree mesh is built around one tool position: cubes of `tree_cell_size` fill a box around the borehole and the coils, and every `tree_padding` cells the cells double in size, out to `domain_radius`. The tool axis is placed on a grid line. Because cubes cut the round wall, each cell gets its volume-averaged conductivity (4 × 4 × 4 samples). Arithmetic averaging fits the induced currents, which run parallel to the wall and the layers. Without averaging, shifting the grid by half a cell changed the result by up to 4%.

Checks: centred tool in a borehole within 0.1–1.5% of the analytic solution; coplanar coils in a full space within 0.7%; an off-centre tool in homogeneous ground reads the true resistivity within 0.5%; a 2× finer mesh changes the off-centre result by 0.2%.

**Off-centre tool in conductive backfill.** Moving the tool towards the wall *reduces* the borehole effect. For 1 Ohm.m backfill in 100 Ohm.m, 0.2 m hole, 400 Hz:

| Tool offset | ρa, s = 0.5 m | ρa, s = 1 m |
|---|---|---|
| 0 (centred) | 19.0 | 49.1 |
| 3 cm | 20.8 | 51.5 |
| 6 cm | 28.7 | 60.3 |
| 8 cm (2 cm from the wall) | 45.6 | 73.0 |

Off-centre, the eddy-current loops around the tool axis cross the wall; charge builds up on it and blocks part of the current in the backfill. The geometric-factor (Doll) picture, which has no such charges, predicts almost no change, and that is indeed what the 3D solver gives for a weak contrast (50 vs 100 Ohm.m: 95.5 → 96.2 Ohm.m).

### Realistic tools

`run_realistic.py` uses the following options.

**Tool body.** `Geometry(tool_diameter=...)` puts an insulating cylinder in the middle of the hole. It is kept in the no-borehole reference, because it is part of the tool. The tool polygon has 8 sides (area-corrected): with 24 sides its millimetre-wide facets running the whole hole made TetGen generate ~10⁶ extra cells.

**Ring electrodes.** Real sonde electrodes are metal rings, equipotential all the way round. A single point electrode on the tool surface is a poor stand-in when the backfill is resistive: with 1000 Ohm.m backfill, the potential 0.2 m away on the far side of the tool was 21% lower than on the near side, because the current hardly spreads around the tool in a thin resistive annulus. For a centred tool the model is axisymmetric, so a ring-to-ring resistance equals the average, over points around M, of the resistance from one point at A. The code samples 4 points around the tool. Instead of PyGIMLi's `simulate` (which solves once per electrode, used or not), it solves only for the current electrodes and reads the potentials at the measuring points.

**Normal and lateral arrays.** In a *normal* (A–M on the sonde, B and N "at infinity") the measurement is the potential at M; the depth of investigation is about 2 × AM and the response is symmetric. A *lateral* (A on the sonde, M and N close together at distance AO, B at infinity) measures roughly the potential gradient at AO; it sees deeper and its response is asymmetric. Remote electrodes are modelled as ideal poles (index −1).

**Tool constant.** With a tool body, the analytic geometric factor no longer applies. Like a real tool, the code calibrates it: it simulates the same protocol in a homogeneous 1 Ohm.m earth with the tool and uses k = 1 / R, so a homogeneous earth reads its true resistivity at every depth.

**Focused (bucked) induction tool.** A bucking receiver at distance s_b, wound against the main receiver with weight −(s_b/s)³, cancels the direct field. The near-axis contribution of a conductive borehole scales the same way (∝ area / s³), so it cancels to first order as well, which is why focused tools see much less of the backfill. With 1 Ohm.m backfill in 100 Ohm.m, s = 0.5 m: two-coil 18.9 Ohm.m, focused (s_b = 0.25 m) 67.4 Ohm.m.

**Validation** (`geometry_class/analytic.py`): the ring-electrode results are compared with a three-region Fourier–Bessel solution (insulating mandrel, backfill, formation) for ring electrodes on the mandrel, using the same calibration: normal 16"/64" and lateral within 0.3% for 1 and 1000 Ohm.m backfill. The focused EM tool agrees with the analytic borehole solution within 0.2–1.3%.

### Why the two tools react differently

ERT measures currents that the electrodes inject directly into the borehole fill, so a conductive backfill channels current along the hole. Short arrays read close to the backfill; longer Wenner arrays can overshoot the formation. Induction measures eddy currents that loop around the axis, so a conductive annulus adds its own conductivity in proportion to its share of the tool's sensitivity, which is large for short coils. A resistive backfill blocks very little eddy current, so its effect on induction is small.

---

## 7. Testing

```bash
pytest tests/ -m "not slow"   # model + cylindrical EM tests, ~30 s
pytest tests/                 # all tests; the slow 3D ERT and octree EM tests take a few minutes each
```

[`tests/test_validation.py`](../tests/test_validation.py) checks:

- the shared model (zones, layers, air, the no-borehole switch),
- EM against the analytic full-space dipole field and against a refined mesh,
- that the EM tool stays inside the hole and that invalid coil settings are rejected,
- ERT against the analytic infinite-borehole Wenner response (`slow`),
- the ERT no-borehole reference and the boundary markers.

[`tests/test_gamma.py`](../tests/test_gamma.py) checks the gamma-ray simulator (§9, about 30 s; marked `gamma` and skipped without Geant4).

The reference solutions are in [`geometry_class/analytic.py`](../geometry_class/analytic.py) and can be used on their own, e.g. `borehole_wenner_rhoa(a, radius, rho_hole, rho_formation)`.

---

## 8. Troubleshooting

| Problem | Cause and fix |
|---|---|
| `FileNotFoundError: ...-1.node` while meshing | TetGen isn't found. Install it (`conda install -c conda-forge tetgen`) and run inside the activated environment. |
| ERT run is killed or the machine swaps | Not enough memory; see §5 to reduce the model size. |
| Warning `backfill_resistivity not given` | You passed only `background_resistivity`; it is used as the backfill. Pass `backfill_resistivity` explicitly. |
| Warning `quality=... is a 2D min-angle value` | Use `quality` between 1.2 and 1.4 in `make_mesh`. |
| `ValueError: coil_orientation='x' ... needs n_theta_cells >= 8` | Coplanar coils aren't axisymmetric; set `n_theta_cells=8` or more, or use `mesh_type="tree"`. |
| `ValueError: Outermost borehole zone ... exceeds the borehole radius` | A zone radius is larger than `borehole_diameter / 2`. |
| `ValueError: set_borehole can only change resistivities` | You changed zone radii or depths; build a new geometry and mesh. |
| `apparent_resistivity` is NaN | The quadrature has the wrong sign or is larger than any full space can produce (very high induction number: very conductive ground, high frequency, long spacing). Use a lower frequency or shorter spacing, or work with `imag` directly. |
| Warning `Electrodes would sit on layer/zone interfaces` | Informational: the electrode grid was shifted by half a step. Pass `electrode_offset=` to choose the positions yourself. |
| `FileNotFoundError: Geant4 data set G4EMLOW* not found` | The Geant4 data sets aren't installed. Download `G4EMLOW`, `G4ENSDFSTATE` and `PhotonEvaporation` and unpack them in `%USERPROFILE%\geant4-data` (or set `GEANT4_DATA_DIR`). |
| `G4Exception ... environment variable not set` from `gammalog` | Another Geant4 data set is needed; its name is in the message. Add it to the same folder; `geant4_env()` in `gamma_geometry.py` lists the variables it sets. |
| `gammalog.exe` fails to start (missing DLL) | The Geant4 `bin` folder must be on `PATH`; `GammaGeometry` adds `GEANT4_DIR\bin` itself. Run `pyborehole-build-gamma` again after a Geant4 upgrade. |

---

## 9. Gamma-ray simulator: `GammaGeometry`

`GammaGeometry` ([`geometry_class/gamma_geometry.py`](../geometry_class/gamma_geometry.py)) simulates a natural gamma-ray tool in the same layered earth and borehole, by Monte Carlo photon transport in Geant4. The transport is done by [`geometry_class/gamma/gammalog.cc`](../geometry_class/gamma/gammalog.cc), compiled once with `pyborehole-build-gamma` (see [INSTALL.md](INSTALL.md)).

### The model

- **Formation and fill.** Each layer and each borehole zone is a `GammaMedium`: a `material` (a water-saturated mineral mixture from `MATERIALS`: `water`, `sand`, `clay`, `gravel`, `limestone`, `sand_pack`, `bentonite`, `cement`; or a salt solution from `SOLUTIONS`: `seawater`, `kcl_brine`; add your own there) and its radioactivity `K` (%), `U` (ppm), `Th` (ppm) per bulk (wet) mass. `density` overrides the preset bulk density.
- **Sources.** K-40 (1.461 MeV) and the main gamma lines of the U-238 and Th-232 series in secular equilibrium (`GAMMA_LINES`). Activities: 1 % K = 313 Bq/kg, 1 ppm U = 12.35 Bq/kg, 1 ppm Th = 4.06 Bq/kg (IAEA-TECDOC-1363).
- **Tool.** A NaI(Tl) crystal, 38 × 102 mm by default, in a 50 mm steel housing with 3 mm walls, centred in the hole. The housing is the probe's own body, not a casing: the default borehole is open hole with backfill. Add a casing as a fill zone if your well has one. The energy deposited in the crystal is smeared to 7 % FWHM at 662 keV. Depth is the centre of the crystal.
- **Geometry.** The model is axisymmetric: coaxial (r, z) tubes for the layers, the borehole zones, the housing and the crystal, rebuilt at every station. Air is above the surface.
- **No-borehole reference.** `run(include_borehole=False)` fills the hole with the surrounding formation and keeps the tool, so the ratio of the two runs isolates the borehole (the same convention as ERT and EM).

### Workflow

```python
from geometry_class import GammaGeometry, GammaMedium

gr = GammaGeometry(
    layers=[(0, 5, dict(material="sand", K=1.2, U=1.5, Th=6.0)),
            (5, 10, dict(material="clay", K=2.5, U=3.5, Th=12.0))],
    borehole_length=10, borehole_diameter=0.2,
    backfill=GammaMedium("bentonite", K=0.3, U=2.0, Th=8.0),
)
log    = gr.run()                         # with the backfilled hole
ref    = gr.run(include_borehole=False)   # no borehole
effect = log.cps / ref.cps
gr.set_backfill(dict(material="water"))   # or set_backfill(zones=[(r, medium), ...])
log, spectra = gr.run(return_spectra=True)
```

Concentric zones work as for ERT and EM, e.g. water inside a PVC casing and a gravel pack outside:

```python
gr = GammaGeometry(..., zones=[(0.045, dict(material="water")),
                               (0.100, dict(material="sand_pack", K=0.3, U=0.5, Th=1.5))])
```

(The 5 mm PVC wall is thin and light enough to leave out for the gamma ray.)

### Parameters

| Parameter | Default | Meaning |
|---|---|---|
| `layers` | | List of `(top, bottom, medium)`; depths positive down; the last layer continues downwards. |
| `borehole_length`, `borehole_diameter` | | m |
| `backfill` / `zones` | water | One fill medium, or concentric zones `(outer_radius, medium)` from the inside out. |
| `measuring_spacing` | 0.1 | Station step (m). |
| `top_clearance`, `bottom_clearance` | 0.5 | The first and last stations are this far from the surface and the hole bottom (the housing extends 0.5 m above and 0.3 m below the crystal centre). |
| `crystal_diameter`, `crystal_length` | 0.038, 0.102 | NaI crystal (m). |
| `tool_diameter`, `housing_thickness` | 0.050, 0.003 | Steel housing (m). |
| `resolution` | 0.07 | FWHM / E at 662 keV. |
| `threshold` | 0.1 | Lower energy threshold of the gross count (MeV). |
| `events` | 150 000 | Simulated emissions per station and nuclide group. |
| `n_jobs` | all cores | Parallel Geant4 processes; stations are shared between them. |
| `seed` | 12345 | Random seed (each process gets its own stream). |
| `calibration_material` | `sand` | Material of the calibration formation. |
| `source_radius`, `sampling_length` | 1.0, 0.15 | Variance-reduction settings (below); there is normally no reason to change them. |

### Output of `run()`

| Column | Meaning |
|---|---|
| `depth` | Crystal centre (m). |
| `cps`, `cps_err` | Gross count rate above `threshold` (counts/s) and its 1σ Monte Carlo error. |
| `cps_K`, `cps_U`, `cps_Th` | Gross count rate from each nuclide group. |
| `K_win`, `U_win`, `Th_win` | Count rates in the standard spectral windows 1.37–1.57, 1.66–1.86 and 2.41–2.81 MeV. |
| `K_app`, `U_app`, `Th_app` | Apparent concentrations: count rate of each group divided by its count rate per unit concentration in the calibration formation. |
| `GR_api`, `GR_err` | Gross count rate in API units and its error. |
| `GR_true` | 16 K + 8 U + 4 Th of the layer at the station. |

### Calibration

`calibrate()` simulates an infinite formation of `calibration_material` with 1 % K, 1 ppm U and 1 ppm Th, without a borehole. The gross count rate is converted to API with one factor (counts/s per API), chosen so that a formation with the API-pit mix (4 % K, 13 ppm U, 24 ppm Th) reads 16 K + 8 U + 4 Th API, like a gross-count tool calibrated in the API pit. A formation with a different K/U/Th mix can read a few percent off its 16 K + 8 U + 4 Th value without a borehole, as with a real gross-count tool. The calibration is done once and kept across `set_backfill` calls.

### Monte Carlo method and accuracy

A photon only counts if it reaches the 4 × 10 cm crystal, which most photons emitted in the formation never do. Two weight-compensated (unbiased) techniques concentrate the effort on the photons that matter:

- **Source sampling.** Emission points within `source_radius` (1 m) of the crystal are drawn with density ∝ e^(−s/λ)/s at distance s (λ = `sampling_length`), and each photon carries the weight activity / sampling density. Beyond 1 m the formation contributes well under 1 % (more than ten attenuation lengths).
- **Weight window.** With the same importance function, photons are split in up to 10 copies as they move towards the crystal and play Russian roulette as they move away. Each copy is scored as its own pulse, and the error is computed per emitted photon, so the correlation between copies is included in `cps_err`.

Together they cut the noise for a given run time severalfold compared with plain sampling. With the default 150 000 events the gross count rate has about 2 % noise per station, and a full 15 m log at 0.1 m (141 stations) takes about a minute on 24 cores. The noise falls as 1/√`events`.

Validation (`tests/test_gamma.py`):

- In an infinite medium the count rate per unit mass concentration does not depend on bulk density, because the emission per volume and the attenuation length scale together. The simulation reproduces this within 1 % from 1.6 to 2.7 g/cm³.
- A 0.2 m (8") water-filled hole with a centred 50 mm (≈2") tool reduces the gross count rate to 0.76 of the no-hole value; the service-company correction charts give about 0.77 for this case.
- Without a borehole, mid-bed values match 16 K + 8 U + 4 Th within 2 % (sand, gravel) to 6 % (clay), the spread coming from the single gross-count calibration factor.

### What controls the borehole effect

About 80 % of the gamma-ray signal comes from within 30 cm of the crystal centre. The fill of a 0.2 m hole takes up a large part of that volume, so its effect is strong:

- **Density and composition** of the fill decide how much formation radiation it absorbs. Water absorbs least per centimetre; a dense cement or sand pack absorbs more.
- **Radioactivity** of the fill adds its own signal. Bentonite grout contains thorium and uranium and can raise the reading in a clean formation while lowering it in a shale.
- **Hole size and tool position**: bigger holes and centred tools have larger effects. Only a centred tool is modelled.
- **Water resistivity does not matter.** Salt changes resistivity a hundredfold but density by only a few percent. In a 0.2 m hole in sand, fresh water and seawater (35 g/kg NaCl) both give 0.767 ± 0.006 of the no-hole count rate. A **KCl** brine is different because potassium is radioactive: 5 % KCl (2.6 % K) gives 0.920, i.e. 20 % more than fresh water.
