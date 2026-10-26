# Installing PyBorehole

This guide installs everything PyBorehole needs, step by step: the Python package (ERT and EM), Geant4 for the gamma-ray simulation, and, optionally, OpenMC for the independent gamma-ray check. Every route below was tested on the systems named in it.

## Contents

- [Overview: what you need](#overview-what-you-need)
- [Part A. The Python package](#part-a-the-python-package)
- [Part B. Geant4 on Windows](#part-b-geant4-on-windows)
- [Part C. Geant4 on Linux and macOS](#part-c-geant4-on-linux-and-macos)
- [Part D. OpenMC check (optional)](#part-d-openmc-check-optional)
- [Part E. Environment variables](#part-e-environment-variables)
- [Part F. Troubleshooting](#part-f-troubleshooting)

---

## Overview: what you need

| Component | Needed for | How it is installed | Download | Time |
|---|---|---|---|---|
| PyBorehole + PyGIMLi, SimPEG, ... | everything | `pip install "pyborehole[all]"` ([Part A](#part-a-the-python-package)) | ~0.5 GB | 5 min |
| TetGen mesh generator | ERT | conda-forge ([Part A](#part-a-the-python-package)) | small | 1 min |
| Geant4 11.4 | gamma ray | Windows installer ([Part B](#part-b-geant4-on-windows)) or conda-forge ([Part C](#part-c-geant4-on-linux-and-macos)) | ~0.5 GB (Windows) / ~2.5 GB (conda-forge, with all data) | 10–20 min |
| Geant4 data sets: G4EMLOW, G4ENSDFSTATE, PhotonEvaporation | gamma ray | download and unpack (Windows); included with conda-forge | ~360 MB | 5 min |
| C++ compiler + CMake | compiling the gamma-ray program once | Visual Studio 2022 (Windows) or conda-forge compilers | ~6 GB (Visual Studio) | 20–30 min |
| OpenMC + ENDF/B-VIII.0 library | optional cross-check of the gamma ray | conda-forge (in WSL on Windows) ([Part D](#part-d-openmc-check-optional)) | 3.4 GB (13 GB unpacked) | 20–40 min |

The gamma ray and the OpenMC check are optional: without Geant4, set `RUN_GAMMA = False` in `run_combined.py`; without OpenMC, set `RUN_OPENMC = False`. ERT and EM work with Part A alone.

**Why is Geant4 not installed by pip?** Geant4 is a C++ toolkit. PyBorehole contains a small Geant4 program, [`geometry_class/gamma/gammalog.cc`](../geometry_class/gamma/gammalog.cc), that does the photon transport. pip installs its source with the package; the command `pyborehole-build-gamma` compiles it once against your Geant4, and the Python class `GammaGeometry` runs it. Python bindings for Geant4 exist (`geant4_pybind`), but per-step Python callbacks would make the variance-reduced transport one to two orders of magnitude slower.

---

## Part A. The Python package

### A1. Install conda

If you do not have conda yet, install [Miniforge](https://github.com/conda-forge/miniforge) (recommended) or Anaconda. Open the **Miniforge Prompt** (Windows) or a terminal (Linux, macOS). conda is used for TetGen (and on Linux for Geant4); everything else comes from pip.

### A2. Install PyBorehole

**Recommended (Windows, Linux, macOS).** Tested on Windows 11 (Python 3.12) and Ubuntu 24.04 (Python 3.11):

```bash
conda create -n pyborehole -c conda-forge python=3.12 tetgen
conda activate pyborehole
pip install "pyborehole[all]"
```

On Windows use **Python 3.12** (or newer): the newest PyGIMLi wheels for Windows are built for it. Until the package is on PyPI, install it from GitHub with `pip install "pyborehole[all] @ git+https://github.com/mariosgeo/PyBorehole"`.

The extras select what is installed:

| Command | Installs |
|---|---|
| `pip install pyborehole` | the core (numpy, scipy, pandas, matplotlib): the model, the gamma-ray simulator, the analytic solutions |
| `pip install "pyborehole[ert]"` | + PyGIMLi and PyVista (ERT) |
| `pip install "pyborehole[em]"` | + SimPEG and discretize (EM) |
| `pip install "pyborehole[all]"` | everything above + pytest |

**Alternative (Windows): PyGIMLi from its conda channel.** If you prefer PyGIMLi from the `gimli` channel, or already have such an environment, install PyBorehole without the `ert` extra so that pip leaves the conda PyGIMLi alone:

```bash
conda create -n pyborehole -c gimli -c conda-forge python=3.11 pygimli tetgen
conda activate pyborehole
pip install pyvista "pyborehole[em,test]"
```

(Older conda builds of PyGIMLi run on NumPy 1.26, but their package metadata asks for NumPy 2.1 or newer; with the `ert` extra pip would upgrade NumPy and the compiled PyGIMLi core would no longer load. On Linux, do not use the `gimli` channel: its PyGIMLi 1.5 build is compiled against older libraries than conda-forge now provides and fails to import with `libcholmod.so.3: cannot open shared object file`.)

**From source (to run `run_combined.py` or change the code).** Clone the repository and install it in editable mode, so that the scripts and tests use your working copy:

```bash
git clone https://github.com/mariosgeo/PyBorehole
cd PyBorehole
conda create -n pyborehole -c conda-forge python=3.12 tetgen
conda activate pyborehole
pip install -e ".[all]"
```

[`requirements.txt`](../requirements.txt) lists the same dependencies for those who prefer `pip install -r requirements.txt` in a checkout (it leaves PyGIMLi out on purpose; install it as above).

### A3. Check

```bash
tetgen -h                                     # prints the TetGen help
python -c "import geometry_class, pygimli, simpeg; print(geometry_class.__version__, pygimli.__version__, simpeg.__version__)"
```

In a source checkout, also run the quick tests (about 30 s):

```bash
python -m pytest tests/test_validation.py -m "not slow"
```

If `tetgen -h` fails, ERT meshing later fails with `No such file or directory: '...-1.node'`: make sure the environment is activated.

---

## Part B. Geant4 on Windows

Tested on Windows 11 with Geant4 11.4.3 and Visual Studio 2022 Community 17.14.

### B1. Install Visual Studio 2022 (C++ compiler, CMake, Ninja)

1. Download **Visual Studio 2022 Community** (free) from <https://visualstudio.microsoft.com/vs/community/>. The smaller **Build Tools for Visual Studio 2022** work as well.
2. In the installer, tick the workload **Desktop development with C++**. Its default components include everything needed: the MSVC v143 compiler, the Windows SDK and **C++ CMake tools for Windows** (which brings CMake and Ninja). Make sure that last component stays ticked.
3. Install (about 6 GB). No need to add anything to `PATH`: the build script finds Visual Studio itself.

### B2. Install Geant4

1. Go to <https://geant4.web.cern.ch/download/> and download the **Windows installer** of Geant4 11.4 for Visual Studio 2022 (64-bit), e.g. `Geant4-11.4.3-win64.exe`.
2. Run it and keep the default folder **`C:\Program Files\Geant4-11.4`**. If you choose another folder, set the environment variable `GEANT4_DIR` to it ([Part E](#part-e-environment-variables)).
3. Adding Geant4 to `PATH` is not needed: PyBorehole adds `GEANT4_DIR\bin` itself when it runs the program.

Check that the installation has the CMake configuration that the build needs:

```powershell
dir "C:\Program Files\Geant4-11.4\lib\cmake\Geant4\Geant4Config.cmake"
```

### B3. Install the Geant4 data sets

Geant4 reads cross sections and atomic data from separate data sets. The Windows installer does **not** necessarily install them: check whether `C:\Program Files\Geant4-11.4\share\Geant4\data` exists and contains `G4EMLOW8.8`. If it does, skip to B4. Otherwise download the three data sets that the gamma-ray simulation needs:

| Data set | Folder | Size | MD5 |
|---|---|---|---|
| Low-energy electromagnetic data | `G4EMLOW8.8` | 350 MB | `328330009df633f7e9b3a9f445745298` |
| Nuclide states | `G4ENSDFSTATE3.0` | 0.3 MB | `c500728534ce3e9fb2fefa0112eb3a74` |
| Photon evaporation | `PhotonEvaporation6.1.2` | ~10 MB | `d80ba9bdefcf9e23487a26adfa273304` |

These versions belong to Geant4 11.4. For another Geant4 version, the matching names and checksums are listed in `lib\cmake\Geant4\Geant4Config.cmake` (search for `G4EMLOW`), or on the download page.

In **PowerShell** (Windows 10 and 11 include `curl.exe` and `tar`):

```powershell
$data = "$env:USERPROFILE\geant4-data"
New-Item -ItemType Directory -Force $data | Out-Null
Set-Location $data
foreach ($f in "G4EMLOW.8.8", "G4ENSDFSTATE.3.0", "G4PhotonEvaporation.6.1.2") {
    curl.exe -L -o "$f.tar.gz" "https://cern.ch/geant4-data/datasets/$f.tar.gz"
    (Get-FileHash -Algorithm MD5 "$f.tar.gz").Hash      # compare with the table
    tar -xzf "$f.tar.gz"
    Remove-Item "$f.tar.gz"
}
Get-ChildItem $data       # G4EMLOW8.8, G4ENSDFSTATE3.0, PhotonEvaporation6.1.2
```

`%USERPROFILE%\geant4-data` (e.g. `C:\Users\<you>\geant4-data`) is where PyBorehole looks by default. To keep the data elsewhere, set `GEANT4_DATA_DIR` to that folder ([Part E](#part-e-environment-variables)). No other data set (neutron, hadronic) is needed.

### B4. Compile the gamma-ray program

With the `pyborehole` environment activated:

```bash
pyborehole-build-gamma
```

It finds Visual Studio (with `vswhere`), configures the program with CMake and Ninja against Geant4, compiles it, and checks the data sets. It ends with:

```
Built ...\geometry_class\gamma\build\gammalog.exe

OK: ...\geometry_class\gamma\build\gammalog.exe
Geant4 data sets found.
```

A line `'vswhere.exe' is not recognized ...` near the start is printed by Visual Studio's own setup script and is harmless. If Geant4 is not in `C:\Program Files\Geant4-11.4`, give its folder: `pyborehole-build-gamma --geant4 "D:\Geant4\11.4"`. The program is written next to its sources inside the installed package, or, if that folder is not writable, to `%LOCALAPPDATA%\pyborehole\gammalog-build`.

You compile once. Compile again after upgrading Geant4 or after editing `gammalog.cc`. If you skip this step, `GammaGeometry` compiles the program itself the first time it needs it. (In a source checkout without `pip install -e`, run `geometry_class\gamma\build.bat` instead.)

### B5. Check

A quick run (about 20 s):

```python
from geometry_class import GammaGeometry
gr = GammaGeometry(layers=[(0, 10, dict(material="sand", K=1.2, U=1.5, Th=6.0))],
                   borehole_length=10, borehole_diameter=0.2,
                   backfill=dict(material="water"), measuring_spacing=1.0)
print(gr.run()[["depth", "cps", "GR_api", "GR_true"]])
```

`GR_api` should be about 0.75 of `GR_true` (55 API) in the middle of the hole: that is the effect of the water-filled hole.

In a source checkout, run the gamma-ray tests:

```bash
python -m pytest tests/test_gamma.py -v
```

Expected result: **4 passed, 1 skipped** in about 35 s (the skipped test is the OpenMC comparison, which needs Part D; with OpenMC installed, 5 pass). The tests check the material presets, that the count rate in an infinite formation does not depend on its density, that a water-filled 0.2 m hole reduces the count rate to 0.70–0.82 of the no-hole value (charts: ~0.77), and that thick beds read their true API value.

---

## Part C. Geant4 on Linux and macOS

Tested on Ubuntu 24.04 (in WSL) with conda-forge Geant4 11.4.3 and the pip-installed package. macOS uses the same commands but was not tested.

### C1. Install Geant4 and a compiler into the environment

conda-forge provides Geant4 with all its data sets, and the compilers and CMake to build against it. Create the environment with them, then install PyBorehole with pip:

```bash
conda create -n pyborehole -c conda-forge python=3.11 tetgen \
    geant4 cmake make cxx-compiler \
    expat zlib xerces-c freetype hdf5 clhep qt6-main xorg-libx11 xorg-libxext \
    libgl-devel libegl-devel libopengl-devel libglx-devel
conda activate pyborehole
pip install "pyborehole[all]"
geant4-config --version      # 11.4.3
```

The third and fourth lines are the development files of libraries that the conda-forge Geant4 was built with (XML, compression, HDF5, Qt and OpenGL visualisation). Geant4's CMake configuration checks for all of them, even though the gamma-ray program does not use visualisation; without them CMake stops with `Could NOT find EXPAT`, `Could NOT find ZLIB` and so on. On macOS, drop `xorg-libx11`, `xorg-libxext` and the `lib*-devel` packages and add whatever CMake reports missing.

The download is large (about 2.5 GB) because conda-forge installs all Geant4 data sets. Activating the environment sets `G4LEDATA` and the other data variables; PyBorehole also finds the data under `<env>/share/Geant4/data` without them.

### C2. Compile the gamma-ray program

```bash
pyborehole-build-gamma
```

It ends with `OK: .../geometry_class/gamma/build/gammalog` and `Geant4 data sets found.`. Lines such as `Could NOT find WrapVulkanHeaders` come from Qt and are harmless. To build against another Geant4 installation (e.g. one compiled from source), give its prefix: `pyborehole-build-gamma --geant4 /opt/geant4`. (In a source checkout without `pip install -e`: `bash geometry_class/gamma/build.sh`.)

### C3. Check

The quick run of B5, or in a source checkout `python -m pytest tests/test_gamma.py -v` (4 passed, 1 skipped).

### C4. Alternative: Geant4 from source

If you compile Geant4 yourself (<https://geant4.web.cern.ch/docs/>), configure it with `-DGEANT4_INSTALL_DATA=ON` so that the data sets are downloaded into `<prefix>/share/Geant4/data`, where PyBorehole finds them. Then `pyborehole-build-gamma --geant4 <prefix>`.

---

## Part D. OpenMC check (optional)

`run_combined.py` can repeat the gamma-ray simulation with [OpenMC](https://openmc.org) at the middle of every layer. It is an independent check (a different code, different cross-section data, no variance reduction), not needed for normal use. OpenMC does not run natively on Windows; there it runs inside WSL. On Linux and macOS it runs directly.

### D1. Windows only: install WSL with Ubuntu

In an **administrator** PowerShell:

```powershell
wsl --install -d Ubuntu
```

Restart Windows when asked, open **Ubuntu** from the Start menu and choose a user name and password. Then, inside Ubuntu, install Miniforge:

```bash
curl -L -O https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh -b -p ~/miniforge3
```

All following commands of this part run **inside Ubuntu** (on Windows) or in your terminal (Linux, macOS).

### D2. Install OpenMC

```bash
~/miniforge3/bin/conda create -y -n openmc -c conda-forge openmc
~/miniforge3/envs/openmc/bin/python -c "import openmc; print(openmc.__version__)"   # 0.16 or newer
```

On Windows, PyBorehole calls `$HOME/miniforge3/envs/openmc/bin/python` inside WSL. If your OpenMC Python is elsewhere, set `OPENMC_PYTHON` (a Windows environment variable holding the Linux path). On Linux and macOS, either install `openmc` into the `pyborehole` environment, or point `OPENMC_PYTHON` to the Python of the `openmc` environment.

### D3. Install the nuclear data library

OpenMC needs its HDF5 data library, including neutron data even for a photon-only run (otherwise it stops with `Could not find nuclide Si28 in the nuclear data library`). Use the official ENDF/B-VIII.0 library, 3.4 GB to download and 13 GB unpacked:

```bash
mkdir -p ~/openmc-data && cd ~/openmc-data
curl -L -o endfb80.tar.xz https://anl.box.com/shared/static/uhbxlrx7hvxqw27psymfbhi7bx7s6u6a.xz
tar xJf endfb80.tar.xz && rm endfb80.tar.xz
ls ~/openmc-data/endfb-viii.0-hdf5/cross_sections.xml
```

PyBorehole finds the library under `~/openmc-data/` automatically. If you keep it elsewhere, set `OPENMC_CROSS_SECTIONS` to its `cross_sections.xml` (inside WSL on Windows, e.g. in `~/.bashrc`). The link is the one listed on <https://openmc.org/data/>; check there if it has moved.

### D4. Check

In a source checkout, back in the `pyborehole` environment (on Windows: in the normal Windows prompt):

```bash
python -m pytest tests/test_gamma.py -v      # now 5 passed, including test_openmc_matches_geant4 (~3 min)
```

---

## Part E. Environment variables

None of these is needed with the default locations above.

| Variable | Default | Meaning |
|---|---|---|
| `GEANT4_DIR` | `C:\Program Files\Geant4-11.4` (Windows); `geant4-config --prefix` (Linux, macOS) | Geant4 installation; its `bin` folder is put on `PATH` when the program runs |
| `GEANT4_DATA_DIR` | `%USERPROFILE%\geant4-data`, then `<GEANT4_DIR>\share\Geant4\data` | Folder holding the Geant4 data sets |
| `G4LEDATA`, `G4ENSDFSTATEDATA`, `G4LEVELGAMMADATA` | found in the folders above | Individual data sets (set by conda-forge activation; take precedence) |
| `GAMMALOG_EXE` | `geometry_class/gamma/build/gammalog[.exe]` inside the package, or the per-user build folder | The compiled gamma-ray program |
| `OPENMC_PYTHON` | `$HOME/miniforge3/envs/openmc/bin/python` in WSL (Windows); the current Python (Linux, macOS) | Python that has `openmc` |
| `OPENMC_WSL_DISTRO` | `Ubuntu` | WSL distribution (Windows only) |
| `OPENMC_CROSS_SECTIONS` | first library with neutron data under `~/openmc-data/` | OpenMC `cross_sections.xml` |

On Windows, set a variable permanently with `setx GEANT4_DATA_DIR "D:\geant4-data"` and open a new prompt; on Linux and macOS add `export GEANT4_DATA_DIR=...` to `~/.bashrc` (or set it in the conda environment with `conda env config vars set GEANT4_DATA_DIR=...`).

---

## Part F. Troubleshooting

| Message | Cause and fix |
|---|---|
| `FileNotFoundError: ...-1.node` while meshing (ERT) | TetGen is not found. Activate the environment; `conda install -c conda-forge tetgen`. |
| `ImportError: The ERT simulator needs PyGIMLi and PyVista` | Install the `ert` extra (`pip install "pyborehole[ert]"`) or PyGIMLi with conda (A2). |
| `libcholmod.so.3: cannot open shared object file` (Linux) | PyGIMLi from the `gimli` conda channel; use the pip route of A2 instead. |
| `FileNotFoundError: Geant4 data set *EMLOW* not found in [...]` | The data sets are missing or elsewhere: B3, or set `GEANT4_DATA_DIR`. |
| `G4Exception ... G4LEVELGAMMADATA environment variable not set` (or another `G4...DATA`) | A data set is missing: download the one named in the message into the same folder. |
| `Visual Studio with C++ tools not found` from the build | Install the *Desktop development with C++* workload (B1). |
| `Could not find a package configuration file provided by "Geant4"` | CMake does not find Geant4: `pyborehole-build-gamma --geant4 <folder>`, or set `GEANT4_DIR`. |
| `Could NOT find EXPAT` / `ZLIB` / `X11` / `OpenGL` ... (Linux) | Install the development packages listed in C1. |
| `gammalog.exe` exits at once or reports a missing DLL | `GEANT4_DIR` points to the wrong folder, or Geant4 was upgraded: run `pyborehole-build-gamma` again. |
| `gammalog failed:` with a Geant4 exception | The message shows the Geant4 error; most often a data set (above). |
| OpenMC: `Could not find nuclide Si28 in the nuclear data library` | A photon-only library was given; use the full ENDF/B-VIII.0 library (D3). |
| OpenMC: `No such file or directory: 'openmc'` | PyBorehole passes the executable explicitly, so check that `OPENMC_PYTHON` points to the Python of the OpenMC environment. |
| OpenMC worker fails on Windows with `wsl.exe` errors | WSL or the distribution is not installed (D1), or `OPENMC_WSL_DISTRO` names another distribution (`wsl -l -v` lists them). |
| Downloads fail with `CERTIFICATE_VERIFY_FAILED` in Python | Download with `curl` (as above) instead of from Python. |
