# PyBorehole: borehole backfill effect on resistivity, induction and gamma-ray logs

[![PyPI](https://img.shields.io/pypi/v/pyborehole)](https://pypi.org/project/pyborehole/)
[![Python](https://img.shields.io/pypi/pyversions/pyborehole)](https://pypi.org/project/pyborehole/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](https://github.com/mariosgeo/PyBorehole/blob/main/LICENSE)

PyBorehole simulates what a borehole **resistivity tool (ERT, PyGIMLi)**, an **electromagnetic induction tool (FDEM, SimPEG)** and a **natural gamma-ray tool (Monte Carlo, Geant4)** measure in a backfilled borehole through a layered earth. All tools are simulated with and without the borehole, so you can see how much the backfill distorts each log.

You provide:

- the layered earth model (depths and resistivities; for the gamma ray also the material and K, U, Th content of each layer),
- the borehole diameter and length, and the backfill resistivity (or several concentric zones such as casing + gravel pack); for the gamma ray, the backfill material and its radioactivity,
- the electrode spacings (ERT) and the coil spacings and frequencies (EM).

You get apparent-resistivity and gamma-ray logs for every tool, the no-borehole reference logs, and the **borehole effect**: the ratio of the two.

![Logs for a 1 Ohm.m backfill](https://raw.githubusercontent.com/mariosgeo/PyBorehole/main/results/combined_log_backfill_1.png)

## Install

```bash
conda create -n pyborehole -c conda-forge python=3.12 tetgen
conda activate pyborehole
pip install "pyborehole[all]"
pyborehole-build-gamma        # optional: compiles the Geant4 gamma-ray program (needs Geant4)
```

Geant4 and the optional OpenMC check are installed separately: see the step-by-step **[installation guide](https://github.com/mariosgeo/PyBorehole/blob/main/docs/INSTALL.md)**. The Python package is imported as `geometry_class`.

---

## Start here: `run_combined.py`

**[`run_combined.py`](https://github.com/mariosgeo/PyBorehole/blob/main/run_combined.py) is the main file.** It runs the complete study: ERT and EM for a sweep of backfill resistivities, and the gamma ray for a sweep of backfill materials, each with and without the borehole, and saves all the figures. Usually you only edit its `SETTINGS` block and run it.

### 1. Install

**Full step-by-step instructions, including Geant4, are in [docs/INSTALL.md](https://github.com/mariosgeo/PyBorehole/blob/main/docs/INSTALL.md).** In short:

```bash
conda create -n pyborehole -c conda-forge python=3.12 tetgen
conda activate pyborehole
pip install "pyborehole[all]"
```

TetGen (the 3D mesh generator used for ERT) comes from conda-forge; everything else from pip. Extras: `ert` (PyGIMLi, PyVista), `em` (SimPEG, discretize), `test`, `all`. To run `run_combined.py` or change the code, clone the repository and use `pip install -e ".[all]"` instead.

Check that `tetgen` is found (`tetgen -h`). If it isn't, ERT meshing fails with `No such file or directory: '...-1.node'`.

**Gamma ray (optional).** The gamma-ray simulation is a small Geant4 program, [`gammalog.cc`](https://github.com/mariosgeo/PyBorehole/blob/main/geometry_class/gamma/gammalog.cc), that pip installs as source and you compile once:

| | Windows | Linux / macOS |
|---|---|---|
| Geant4 | installer from <https://geant4.web.cern.ch/download/> into `C:\Program Files\Geant4-11.4` | `conda install -c conda-forge geant4` (plus build tools, see the guide) |
| Data sets | G4EMLOW, G4ENSDFSTATE, PhotonEvaporation (360 MB) into `%USERPROFILE%\geant4-data` | included with conda-forge |
| Compiler | Visual Studio 2022, *Desktop development with C++* | conda-forge `cxx-compiler cmake make` |
| Compile | `pyborehole-build-gamma` | `pyborehole-build-gamma` |

Without Geant4, set `RUN_GAMMA = False` in `run_combined.py`.

**OpenMC check (optional).** An independent check of the gamma ray with [OpenMC](https://openmc.org) (in WSL on Windows) and the 3.4 GB ENDF/B-VIII.0 library; see [docs/INSTALL.md, Part D](https://github.com/mariosgeo/PyBorehole/blob/main/docs/INSTALL.md#part-d-openmc-check-optional). Without it, set `RUN_OPENMC = False`.

### 2. Edit the settings

Open `run_combined.py` and edit the block at the top:

```python
LAYERS = np.array([          # [top_depth, bottom_depth, resistivity]; last layer extends down
    [0.0,   5.0,  100.0],
    [5.0,  10.0,   10.0],
    [10.0, 15.0,  200.0],
])

BOREHOLE_LENGTH = 15.0
BOREHOLE_DIAM   = 0.20
BACKFILL_RES    = 10.0                       # always included in the sweep
BACKFILL_SWEEP  = [1.0, 10.0, 100.0, 1000.0]  # one combined_log_backfill_<rho>.png each

ERT_SHORT_SPACING = 0.5
ERT_LONG_SPACING  = 1.0
ERT_MEAS_SPACING  = 0.1

EM_COIL_SPACINGS  = [0.5, 1.0]
EM_FREQUENCIES    = [400.0, 8000.0]
EM_MEAS_SPACING   = 0.1
EM_MESH_TYPE      = "cylindrical"

OUTPUT_DIR        = "results"               # all figures are written here
```

| Setting | Meaning |
|---|---|
| `LAYERS` | One row per layer: top depth (m), bottom depth (m), resistivity (Ohm.m). Depths are positive downward from the surface. The first layer must start at 0; the last layer continues to the bottom of the model. |
| `BOREHOLE_LENGTH`, `BOREHOLE_DIAM` | Borehole depth and diameter (m). The borehole is vertical and starts at the surface. |
| `BACKFILL_SWEEP` | Backfill resistivities (Ohm.m) to simulate. Each value costs one ERT solve (about 1–2.5 min). |
| `BACKFILL_RES` | A backfill value that is always simulated, even if it isn't in the sweep. |
| `ERT_SHORT_SPACING`, `ERT_LONG_SPACING` | Wenner electrode spacings *a* (m). The array is A–M–N–B with spacing *a* between neighbours, so it is 3*a* long. |
| `ERT_MEAS_SPACING` | Electrode step along the hole (m). Spacings must be multiples of it. |
| `EM_COIL_SPACINGS` | Transmitter–receiver distances (m) of the coaxial induction coils. |
| `EM_FREQUENCIES` | EM frequencies (Hz). |
| `EM_MEAS_SPACING` | Step of the EM tool along the hole (m). |
| `EM_MESH_TYPE` | `"cylindrical"` (axisymmetric, a full log in seconds; use this for a centred tool) or `"tree"` (full 3D octree, ~10 s per tool position and frequency; for an off-centre tool, see the user guide). |
| `RUN_GAMMA` | `False` skips the gamma-ray simulation. |
| `GR_LAYERS` | One entry per `LAYERS` row: formation `material` (`sand`, `clay`, `gravel`, `limestone`, ...) and radioactivity `K` (%), `U` (ppm), `Th` (ppm) per bulk (wet) mass. |
| `GR_BACKFILL_SWEEP` | Backfill materials for the gamma ray: name → `material` and `K`, `U`, `Th`. The gamma ray depends on the density, composition and radioactivity of the backfill, not on its resistivity, so it has its own sweep (fresh water vs seawater shows this; KCl brine shows the effect of potassium in the hole). Each entry costs about one minute. |
| `GR_BACKFILL` | The entry of `GR_BACKFILL_SWEEP` shown next to the resistivity logs. |
| `GR_MEAS_SPACING` | Step of the gamma-ray tool along the hole (m). |
| `RUN_OPENMC` | `False` skips the OpenMC check. |
| `OM_DEPTHS` | Depths (m) of the OpenMC check; `None` = the middle of every layer. |
| `OM_PARTICLES` | Source photons per OpenMC depth (analog Monte Carlo, about 30 s each on 24 cores; error ~2–3%). |
| `GR_EVENTS` | Simulated gamma emissions per station and nuclide group. The Monte Carlo noise falls as 1/√`GR_EVENTS`; the default gives about 2% per station. |
| `OUTPUT_DIR` | Folder for the figures. |

### 3. Run

```bash
conda activate pyborehole
python run_combined.py
```

With the default settings this takes about **8–12 minutes** and needs about **12 GB of RAM**, almost all of it for the 3D ERT solves. The EM simulations take seconds.

What the script does:

1. **ERT:** builds one 3D mesh (borehole, layers and electrodes are built into it), then solves for the no-borehole reference and for every backfill value in the sweep, reusing the mesh.
2. **EM:** solves the same model on an axisymmetric cylindrical mesh, with and without the borehole, for every backfill value.
3. **Gamma ray:** Geant4 Monte Carlo of a NaI(Tl) tool in the same layers, with no borehole and for every backfill material, in parallel on all CPU cores (a few minutes in total). Then the same problem in OpenMC at the middle of every layer, as an independent check (about 8 minutes).
4. **Plots:** one log figure per backfill value, one summary figure of the borehole effect, and one gamma-ray figure.
5. **Tables:** print the borehole effect at the middle of every layer.

### 4. Read the results

All figures go to `results/`:

| File | What it shows |
|---|---|
| `combined_log_backfill_<rho>.png` | One per backfill value. Left: the model. Right: apparent resistivity versus depth for every tool. Solid = with the backfilled hole, dotted = the same tool without a hole, dashed black = true formation resistivity. |
| `backfill_effect.png` | One panel per tool, one curve per backfill value: ρa(with hole) / ρa(no hole). **1 = no effect**; below 1 the hole makes the tool read too low, above 1 too high. |
| `gamma_backfill_effect.png` | Gamma ray for every backfill material: the GR log (API), the borehole effect cps(with hole) / cps(no hole), and the apparent K, U and Th. Dotted = no borehole, dashed = true formation values. |
| `gamma_logs.csv` | The gamma-ray logs as a table, one row per station and backfill. |
| `gamma_openmc_check.csv` | Geant4 vs OpenMC count rates at the check depths. The OpenMC points are also drawn, with error bars, in `gamma_backfill_effect.png`. |

![Borehole effect](https://raw.githubusercontent.com/mariosgeo/PyBorehole/main/results/backfill_effect.png)

Depth on all plots is the **centre of the tool**: the midpoint of the Wenner array, or the midpoint between the transmitter and receiver coils.

The printed table looks like this (default model, D = 0.2 m):

```
 layer_rho  depth  backfill  ERT a=0.5  ERT a=1  EM s=0.5  EM s=1
     100.0    2.5       1.0      0.512    1.199     0.210   0.549
     100.0    2.5     100.0      1.000    1.000     1.000   1.000
     200.0   12.5       1.0      0.315    0.887     0.135   0.445
     ...
```

What these default results show:

- A **conductive backfill** (1 Ohm.m) has a large effect on the short induction coils: s = 0.5 m reads 5–7 times too low. Longer coils are much less affected.
- The **ERT Wenner** array reads too low at short spacing and can *overshoot* the true value at longer spacing (ratio > 1). This is real Wenner behaviour in a conductive hole and matches the analytic solution.
- A **resistive backfill** (1000 Ohm.m) changes ERT by 4–8% and EM by only 1–5%.
- When the backfill equals the formation, every ratio is exactly 1, as it should be.

![Gamma-ray borehole effect](https://raw.githubusercontent.com/mariosgeo/PyBorehole/main/results/gamma_backfill_effect.png)

The gamma-ray table (GR in API; ratio = gross count rate with / without the hole):

```
 depth  GR true  no hole  water  ratio  filter sand  ratio  bentonite  ratio  cement  ratio
   2.5     55.2     53.4   39.4   0.74         37.2   0.70       54.9   1.03    45.8   0.86
   7.5    116.0    109.1   87.4   0.80         70.4   0.64       91.4   0.84    77.0   0.71
  12.5     19.2     19.1   14.6   0.76         17.0   0.89       30.3   1.59    26.3   1.38
```

- A **water-filled** 0.2 m hole lowers the gamma ray by 20–26 % in every layer: water shields the formation but adds no radiation.
- A dense, low-activity **filter sand** shields more (down to 0.64 in the clay) but adds a little signal in the clean gravel.
- **Radioactive backfills** (bentonite, cement) compress the contrast: the clay reads too low, the clean gravel too high (bentonite: ×1.6), and the 100 API clay/gravel contrast drops to about 60 API. In the sand, whose activity is close to the bentonite's, the two effects cancel.

---

## Realistic tools: `run_realistic.py`

`run_combined.py` uses idealised tools: point electrodes on the borehole axis in a Wenner array, and a plain two-coil EM tool. **[`run_realistic.py`](https://github.com/mariosgeo/PyBorehole/blob/main/run_realistic.py)** repeats the backfill study with tools that behave like real logging sondes:

| | `run_combined.py` | `run_realistic.py` |
|---|---|---|
| ERT electrodes | points on the axis | rings on an insulating tool body (default 50 mm) |
| ERT arrays | Wenner (A, M, N, B all downhole) | **normal** (A–M on the sonde, B and N remote; 16" and 64") and **lateral** (A, M, N on the sonde, B remote) |
| ERT geometric factor | analytic half-space formula | calibrated like a real tool: the sonde reads the true resistivity in a homogeneous earth |
| EM tool | two-coil pair | two-coil pair and a **focused** tool with a bucking receiver |

Settings (top of the script):

| Setting | Meaning |
|---|---|
| `TOOL_DIAM` | Diameter of the insulating ERT tool body (m); `None` = no tool body. |
| `NORMAL_SPACINGS` | AM distances of the normal arrays (m). Default 16" (0.41 m) and 64" (1.63 m). |
| `LATERAL_SPACINGS`, `LATERAL_MN` | AO distance(s) (A to the midpoint of MN) and MN distance of the lateral (m). The defaults are an example; the classic oilfield lateral is AO = 5.69 m (18'8"), MN = 0.81 m. |
| `ERT_MEAS_SPACING` | Tool step along the hole (m). |
| `EM_COIL_SPACINGS`, `EM_BUCKING` | Main transmitter–receiver distance per EM tool, and the bucking receiver distance for each (`None` = plain two-coil). |
| `LAYERS`, `BOREHOLE_*`, `BACKFILL_SWEEP`, `EM_FREQUENCIES`, `OUTPUT_DIR` | As in `run_combined.py`. |
| `REUSE_SAVED_RESULTS` | `True` redraws all figures from `results/realistic_logs.csv` without simulating again (seconds instead of ~30 minutes). |

With the defaults it takes about **30 minutes and up to ~17 GB of RAM**: ring electrodes on a tool body need a finer ERT mesh (~1.2M cells). Setting `ERT_MEAS_SPACING = 0.2` roughly halves both.

Outputs in `results/`:

- `realistic_log_backfill_<rho>.png`: the model, a **to-scale close-up of the tools in the hole** at mid-depth (ERT sonde with its tool body and ring electrodes A, M, N; EM tool with transmitter, bucking and main receiver; the window is shaded on the model), and all tool logs for that backfill (dotted = no borehole).
- `realistic_backfill_effect.png`: with-hole / no-hole ratio per tool.
- `realistic_logs.csv`: every simulated value (tool, depth, apparent resistivity, backfill; empty backfill = no-borehole reference), plus a printed summary table.

![Realistic tools, 1 Ohm.m backfill](https://raw.githubusercontent.com/mariosgeo/PyBorehole/main/results/realistic_log_backfill_1.png)

Replace the example spacings and tool diameter with those of your sondes. The physics of these tools is described in [docs/user_guide.md](https://github.com/mariosgeo/PyBorehole/blob/main/docs/user_guide.md#realistic-tools).

---

## Using the classes directly

`run_combined.py` is built on three classes you can also use on their own:

```python
import numpy as np
from geometry_class import Geometry, EMGeometry

layers = np.array([[0, 5, 100], [5, 10, 10], [10, 15, 200]], dtype=float)

# ERT
geo  = Geometry(borehole_length=15, borehole_diameter=0.2, layer_1d_geometry=layers,
                backfill_resistivity=10.0, long_spacing=1.0)
plc  = geo.make_basic_geometry()
data = geo.make_array(plc, abmn_order=(1, 4, 2, 3), short_spacing=0.5,
                      long_spacing=1.0, measuring_spacing=0.1)
mesh = geo.make_mesh(plc)
with_hole = geo.run(mesh, data)                        # pygimli DataContainer, 'rhoa' column
no_hole   = geo.run(mesh, data, include_borehole=False)
geo.set_borehole(backfill_resistivity=1000.0)          # new backfill, same mesh

# EM
em = EMGeometry(borehole_length=15, borehole_diameter=0.2, layer_1d_geometry=layers,
                backfill_resistivity=10.0, frequencies=[400, 8000], coil_spacings=[0.5, 1.0])
log    = em.run()                                      # pandas DataFrame
log_nh = em.run(include_borehole=False)

# Gamma ray (Geant4)
from geometry_class import GammaGeometry
gr = GammaGeometry(layers=[(0, 5, dict(material="sand", K=1.2, U=1.5, Th=6.0)),
                           (5, 10, dict(material="clay", K=2.5, U=3.5, Th=12.0)),
                           (10, 15, dict(material="gravel", K=0.4, U=0.6, Th=2.0))],
                   borehole_length=15, borehole_diameter=0.2,
                   backfill=dict(material="bentonite", K=0.3, U=2.0, Th=8.0))
gr_log = gr.run()                                      # DataFrame: depth, cps, GR_api, K_app, ...
gr_nh  = gr.run(include_borehole=False)
gr.set_backfill(dict(material="water"))                # new backfill
```

Instead of a single backfill you can give concentric zones `(outer_radius, resistivity[, top, bottom])`, for example a water-filled PVC casing inside a gravel pack:

```python
zones = [(0.045, 20.0),            # r < 4.5 cm: water inside the casing
         (0.050, 1e6),             # 5 mm PVC wall (insulating)
         (0.100, 30.0, 2.0, 15.0)] # gravel pack, 30 Ohm.m, from 2 to 15 m depth
geo = Geometry(..., borehole_zones=zones)
em  = EMGeometry(..., borehole_zones=zones)
```

(A closed PVC wall blocks the ERT current, as in reality; galvanic tools only see the formation through open hole or a slotted screen.)

**See [docs/user_guide.md](https://github.com/mariosgeo/PyBorehole/blob/main/docs/user_guide.md)** for all parameters, outputs, accuracy and run-time settings, the physics, and troubleshooting.

---

## Validation

The simulators are checked against exact solutions (`pytest tests/`; add `-m "not slow"` to skip the 3D ERT and octree EM tests, which take a few minutes each):

| Check | Reference | Result |
|---|---|---|
| ERT Wenner in a 0.2 m borehole, backfill 1 and 1000 Ohm.m in 100 Ohm.m, a = 0.1–1 m | Fourier–Bessel solution for an infinite borehole | within 0.1% |
| ERT in a layered model (100/10/200 Ohm.m), with and without the hole | 1.7× finer mesh | within 0.04% |
| ERT without borehole | homogeneous half-space | exact |
| EM coaxial coils in a full space (1–100 Ohm.m), 400 and 8000 Hz | analytic dipole field | quadrature and apparent resistivity within ~1.5% |
| EM coaxial coils in a 0.2 m borehole, backfill 1–1000 Ohm.m | Fourier–Bessel solution for an infinite borehole | quadrature within 0.1–1.7% (cylindrical), 0.1–1.5% (3D octree) |
| EM coplanar coils in a full space, 3D octree | analytic dipole field | within 0.7% |
| ERT normal (16", 64") and lateral on the axis, 1 Ohm.m backfill | Fourier–Bessel solution | within 0.3% |
| ERT ring electrodes on a 50 mm insulating tool, normal and lateral, backfill 1 and 1000 Ohm.m | Fourier–Bessel solution with a mandrel | within 0.3% |
| EM focused tool (bucking receiver), backfill 1–1000 Ohm.m | Fourier–Bessel solution | within 0.2–1.3% |
| Gamma ray in an infinite medium, bulk density 1.6–2.7 g/cm³ | count rate independent of density (exact for a fixed mass concentration) | within 1% |
| Gamma ray, 0.2 m water-filled hole, centred 50 mm tool | service-company GR borehole-correction charts (8" hole, 1.0 g/cm³ fluid: ~0.77) | 0.76 |
| Gamma ray without borehole, thick beds (gross count calibrated to API) | true 16 K + 8 U + 4 Th | within 2% (sand, gravel) to 6% (clay: K/U/Th mix far from the API pit) |

The gamma-ray tests (`tests/test_gamma.py`, about 30 s) are skipped when Geant4 or the program is not available.

## Limitations

- The borehole is vertical and the layers are horizontal.
- The ERT tool is centred in the hole; its body is an insulating cylinder over the whole borehole length (the real sonde is a few metres long, with a thinner cable above). The EM tool can be off-centre (`tool_offset`, 3D octree mesh).
- Ring electrodes are modelled as the average over 4 points around the tool, which is exact for a centred tool (checked against the analytic solution); remote electrodes are ideal poles at infinity.
- Focused induction tools are represented by one bucking receiver per coil pair; commercial tools use their own (often proprietary) coil sets.
- The default EM solver is axisymmetric (exact for a centred coaxial tool in flat layers). The 3D octree solver handles off-centre tools and transverse coils but takes ~10 s per tool position and frequency, so it is meant for selected depths rather than full logs.
- EM `real`, `amplitude` and `phase` are not meaningful at these low induction numbers (the in-phase secondary field is ~1e-8 of the primary); use the quadrature (`imag`) or the apparent resistivity.
- The gamma-ray tool is centred, and its interior above and below the crystal is modelled as air (the photomultiplier and electronics are not modelled). Sources are the main gamma lines of K-40 and of the U and Th series in secular equilibrium. Electrons, positrons and fluorescence photons created outside the crystal are not tracked (their contribution is well under 1%). The gamma-ray logs carry Monte Carlo noise (`cps_err` / `GR_err`, about 2% per station by default).
- Gamma-ray materials are water-saturated mineral mixtures (`MATERIALS` in `gamma_geometry.py`). The backfill radioactivities in `run_combined.py` are typical values, not measurements; replace them with assays of your own grout or sand if you have them.

## Repository layout

```
run_combined.py               main script: ERT + EM + gamma-ray backfill study  <- start here
run_realistic.py              the same study with realistic tools (tool body, normal/lateral, focused EM)
geometry_class/
    geometry.py               ERT: 3D mesh, electrodes, PyGIMLi forward model
    em_geometry.py            EM: cylindrical / octree mesh, SimPEG forward model
    borehole_model.py         shared resistivity model (layers, borehole zones, air)
    gamma_geometry.py         gamma ray: materials, K/U/Th sources, runs gammalog, calibration to API
    gamma_openmc.py           the same gamma-ray problem in OpenMC (independent check, runs in WSL)
    analytic.py               exact reference solutions used by the tests
    plotting.py               drawing helpers (tools-in-the-hole close-up)
    gamma/
        gammalog.cc           Geant4 program: NaI tool in the layered model, Monte Carlo transport
        CMakeLists.txt        CMake project of the program
        build.bat, build.sh   compile it (called by pyborehole-build-gamma)
        openmc_worker.py      runs the OpenMC jobs (inside WSL on Windows)
tests/test_validation.py      validation against the analytic solutions
tests/test_gamma.py           gamma-ray validation (Geant4, OpenMC)
docs/user_guide.md            full documentation
docs/INSTALL.md               step-by-step installation (Python, Geant4, OpenMC)
docs/gamma_ray_geant4.docx    how the gamma-ray simulation works (Word)
results/                      figures written by run_combined.py
demo.ipynb                    older ERT-only notebook demo
pyproject.toml                package definition (pip install pyborehole)
```

## License

MIT, see [LICENSE](https://github.com/mariosgeo/PyBorehole/blob/main/LICENSE).
