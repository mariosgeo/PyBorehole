# -*- coding: utf-8 -*-
"""
gamma_geometry.py
=================

Natural gamma-ray (GR) borehole log simulation with Geant4.

The ``GammaGeometry`` class takes the same layered earth and borehole as
``Geometry`` (ERT) and ``EMGeometry`` (EM). Instead of a resistivity, each
layer and each borehole fill zone has a material (bulk composition and
density) and a radioactivity: potassium K (%), uranium U (ppm) and thorium
Th (ppm), all per unit bulk (wet) mass.

The Monte Carlo transport runs in ``geometry_class/gamma/gammalog.cc``, a
small Geant4 application compiled once against the local Geant4 install, with
the ``pyborehole-build-gamma`` command (or ``build.bat`` / ``build.sh`` in the
same folder):

  * **Tool**      : NaI(Tl) crystal in a steel housing, centred in the hole.
  * **Model**     : axisymmetric stack of coaxial (r, z) tubes: layers,
                    borehole fill zones, housing, crystal; rebuilt for every
                    station.
  * **Source**    : K-40 and the U-238 and Th-232 decay series in secular
                    equilibrium (main gamma lines), emitted within
                    ``source_radius`` of the crystal centre.
  * **Variance**  : emission points are importance sampled around the
                    crystal, and a weight window splits photons moving towards
                    the crystal and plays Russian roulette with photons moving
                    away. All of it is weight-compensated, so the count rates
                    are unbiased.
  * **Physics**   : G4EmStandardPhysics (photoelectric, Compton, Rayleigh,
                    pair production) with the G4EMLOW data set.
  * **Detector**  : energy deposited in the crystal, Gaussian-smeared to the
                    NaI resolution; gross counts above a threshold and counts
                    in the standard K / U / Th windows.

Results are absolute count rates (counts/s) for the given concentrations.
``calibrate()`` simulates an infinite homogeneous formation with no borehole,
the analogue of a calibration pit. ``run()`` uses it to convert:

  * the gross count rate to API units, with one factor (counts/s per API) set
    so that a formation with the API-pit mix ``API_PIT`` reads
    ``16 K + 8 U + 4 Th`` API, as a gross-count tool is calibrated;
  * the count rate of each nuclide group to apparent K (%), U and Th (ppm).

Validation (``tests/test_gamma.py``): in an infinite medium the count rate
does not depend on bulk density (within 1 %), and a water-filled 0.2 m hole
around a centred 50 mm tool reduces the gross count rate to ~0.76 of the
no-hole value, as in the service-company borehole-correction charts.

``run(include_borehole=False)`` fills the borehole with the surrounding
formation, keeping the tool: the no-borehole reference.

Environment
-----------
GEANT4_DIR       Geant4 install (default ``C:\\Program Files\\Geant4-11.4``)
GEANT4_DATA_DIR  folder holding the G4EMLOW, G4ENSDFSTATE and
                 PhotonEvaporation data sets (default ``~/geant4-data``);
                 a variable such as G4LEDATA that is already set wins.
GAMMALOG_EXE     the compiled program (default: see ``gammalog_exe``)
"""

from __future__ import annotations

import argparse
import glob
import os
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

_GAMMA_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gamma")
_EXE_NAME = "gammalog.exe" if os.name == "nt" else "gammalog"

# ---------------------------------------------------------------------------
# Radionuclide data
# ---------------------------------------------------------------------------
# Activity of the chain parent per kg of bulk material, per unit concentration
# (IAEA-TECDOC-1363): 1 % K = 313 Bq/kg K-40, 1 ppm U = 12.35 Bq/kg U-238,
# 1 ppm Th = 4.06 Bq/kg Th-232.
SPECIFIC_ACTIVITY = {"K": 313.0, "U": 12.35, "Th": 4.06}

# Main gamma lines (MeV, photons per decay of the chain parent), U and Th
# series in secular equilibrium (Tl-208 lines include the 35.94 % branch).
GAMMA_LINES = {
    "K": [(1.4608, 0.1066)],
    "U": [  # Th-234, Pa-234m, Ra-226, Pb-214, Bi-214
        (0.0633, 0.037), (1.0010, 0.0084), (0.1862, 0.0364),
        (0.2420, 0.0727), (0.2952, 0.1842), (0.3519, 0.3560),
        (0.6093, 0.4549), (0.7684, 0.0489), (0.9341, 0.0310), (1.1203, 0.1491),
        (1.2381, 0.0583), (1.3777, 0.0399), (1.4080, 0.0239), (1.5092, 0.0213),
        (1.7296, 0.0288), (1.7645, 0.1531), (1.8474, 0.0203), (2.2042, 0.0491),
        (2.4477, 0.0155),
    ],
    "Th": [  # Ac-228, Ra-224, Pb-212, Bi-212, Tl-208
        (0.2093, 0.0389), (0.3383, 0.1127), (0.4630, 0.0440), (0.7950, 0.0425),
        (0.9112, 0.2580), (0.9648, 0.0499), (0.9690, 0.1580), (1.5882, 0.0322),
        (0.2410, 0.0410), (0.2386, 0.4360), (0.3001, 0.0330),
        (0.7273, 0.0667), (1.6205, 0.0147),
        (0.2774, 0.0237), (0.5108, 0.0812), (0.5832, 0.3055), (0.8606, 0.0448),
        (2.6145, 0.3585),
    ],
}
GROUPS = ("K", "U", "Th")

# Standard spectral windows (MeV), IAEA-TECDOC-1363
WINDOWS = {"K_win": (1.370, 1.570), "U_win": (1.660, 1.860), "Th_win": (2.410, 2.810)}

# API-equivalent gamma ray from concentrations (Ellis & Singer, 2007), and the
# K/U/Th mix of the high-activity zone of the API calibration pit
API_WEIGHTS = {"K": 16.0, "U": 8.0, "Th": 4.0}
API_PIT = {"K": 4.0, "U": 13.0, "Th": 24.0}

# ---------------------------------------------------------------------------
# Materials: water-saturated mineral mixtures
# ---------------------------------------------------------------------------
_Z = {"H": 1, "C": 6, "O": 8, "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "S": 16,
      "Cl": 17, "K": 19, "Ca": 20, "Fe": 26}

# Element mass fractions and grain density (g/cm3) of the solids
MINERALS = {
    "quartz":  ({"Si": 0.4674, "O": 0.5326}, 2.65),
    "clay":    ({"O": 0.50, "Si": 0.25, "Al": 0.13, "K": 0.04, "Fe": 0.04,
                 "Mg": 0.02, "Ca": 0.01, "H": 0.01}, 2.70),
    "calcite": ({"Ca": 0.4004, "C": 0.1200, "O": 0.4796}, 2.71),
    "cement":  ({"Ca": 0.44, "O": 0.39, "Si": 0.10, "Al": 0.03, "Fe": 0.02,
                 "S": 0.01, "H": 0.01}, 2.90),
}
_WATER = {"H": 0.1119, "O": 0.8881}

# name -> (solid mineral volume fractions, porosity); pores are water-filled
MATERIALS = {
    "water":     ({}, 1.00),
    "sand":      ({"quartz": 0.90, "clay": 0.10}, 0.30),
    "clay":      ({"clay": 0.70, "quartz": 0.30}, 0.40),
    "gravel":    ({"quartz": 1.00}, 0.25),
    "limestone": ({"calcite": 1.00}, 0.15),
    "sand_pack": ({"quartz": 1.00}, 0.35),   # filter sand
    "bentonite": ({"clay": 1.00}, 0.75),     # bentonite grout, ~1.4 g/cm3
    "cement":    ({"cement": 1.00}, 0.55),   # cement grout, ~1.85 g/cm3
}

_SALTS = {"NaCl": {"Na": 0.3934, "Cl": 0.6066}, "KCl": {"K": 0.5244, "Cl": 0.4756}}


def _solution(salt: str, mass_fraction: float):
    """Element mass fractions of a salt dissolved in water."""
    comp = {el: (1.0 - mass_fraction) * f for el, f in _WATER.items()}
    for el, f in _SALTS[salt].items():
        comp[el] = comp.get(el, 0.0) + mass_fraction * f
    return comp


# Salt solutions: name -> (density g/cm3 at 20 C, element mass fractions).
# Salinity changes resistivity a hundredfold but density only by a few percent,
# so these test how much (little) the water resistivity matters for gamma rays.
# KCl also makes the water radioactive: give it K = 52.44 % x the KCl fraction.
SOLUTIONS = {
    "seawater":  (1.025, _solution("NaCl", 0.035)),   # 35 g/kg, ~0.2 Ohm.m
    "kcl_brine": (1.030, _solution("KCl", 0.05)),     # 5 % KCl (mud filtrate), 2.62 % K
}


def material_composition(name: str):
    """Bulk density (g/cm3) and element mass fractions of a preset material."""
    if name in SOLUTIONS:
        rho, comp = SOLUTIONS[name]
        return rho, dict(comp)
    solids, phi = MATERIALS[name]
    tot = sum(solids.values())
    masses = {}   # g per cm3 of bulk
    rho = phi * 1.0
    for el, f in _WATER.items():
        masses[el] = masses.get(el, 0.0) + phi * f
    for mineral, vf in solids.items():
        comp, rho_g = MINERALS[mineral]
        m = (1.0 - phi) * vf / tot * rho_g
        rho += m
        for el, f in comp.items():
            masses[el] = masses.get(el, 0.0) + m * f / sum(comp.values())
    return rho, {el: m / rho for el, m in masses.items()}


@dataclass
class GammaMedium:
    """A radioactive medium: preset material name and K (%), U, Th (ppm)."""
    material: str = "sand"
    K: float = 0.0
    U: float = 0.0
    Th: float = 0.0
    density: Optional[float] = None   # g/cm3; default from the material preset

    @property
    def rho(self) -> float:
        return self.density if self.density is not None else material_composition(self.material)[0]

    def activity(self) -> List[float]:
        """Parent decays / s / m3 for each nuclide group."""
        return [self.rho * 1000.0 * getattr(self, g) * SPECIFIC_ACTIVITY[g] for g in GROUPS]

    @property
    def api(self) -> float:
        return sum(API_WEIGHTS[g] * getattr(self, g) for g in GROUPS)


def _medium(m: Union[GammaMedium, dict]) -> GammaMedium:
    return m if isinstance(m, GammaMedium) else GammaMedium(**m)


# ---------------------------------------------------------------------------
# Running the Geant4 program
# ---------------------------------------------------------------------------
def _build_dirs() -> List[str]:
    """Where the program is built: next to its sources if that folder is
    writable (a git checkout or a user environment), else a per-user folder."""
    user = os.path.join(os.environ.get("LOCALAPPDATA", os.path.expanduser("~/.cache")),
                        "pyborehole", "gammalog-build")
    pkg = os.path.join(_GAMMA_SRC, "build")
    return [pkg, user] if os.access(_GAMMA_SRC, os.W_OK) else [user]


def build_gammalog(geant4_dir: Optional[str] = None, build_dir: Optional[str] = None) -> str:
    """Compile the Geant4 gamma-ray program; returns the path of the executable.

    Needs Geant4 and a C++ compiler with CMake (Visual Studio 2022 on Windows,
    e.g. conda-forge ``cxx-compiler cmake make`` on Linux); see docs/INSTALL.md.
    """
    build_dir = build_dir or _build_dirs()[0]
    os.makedirs(build_dir, exist_ok=True)
    if os.name == "nt":
        cmd = ["cmd", "/c", os.path.join(_GAMMA_SRC, "build.bat"), geant4_dir or _geant4_dir(), build_dir]
    else:
        cmd = ["bash", os.path.join(_GAMMA_SRC, "build.sh"), geant4_dir or os.environ.get("GEANT4_DIR", ""),
               build_dir]
    subprocess.run(cmd, check=True)
    exe = os.path.join(build_dir, _EXE_NAME)
    if not os.path.isfile(exe):
        raise RuntimeError(f"the build finished but {exe} does not exist")
    return exe


def build_gammalog_main(argv: Optional[Sequence[str]] = None) -> None:
    """Command ``pyborehole-build-gamma``: compile the gamma-ray program."""
    ap = argparse.ArgumentParser(prog="pyborehole-build-gamma",
                                 description="Compile the Geant4 gamma-ray program of PyBorehole.")
    ap.add_argument("--geant4", help="Geant4 install folder (default: $GEANT4_DIR, geant4-config, "
                                     "or C:\\Program Files\\Geant4-11.4 on Windows)")
    ap.add_argument("--build-dir", help=f"build folder (default: {_build_dirs()[0]})")
    a = ap.parse_args(argv)
    exe = build_gammalog(a.geant4, a.build_dir)
    print(f"\nOK: {exe}")
    try:
        geant4_env()
        print("Geant4 data sets found.")
    except FileNotFoundError as err:
        print(f"WARNING: {err}")


def gammalog_exe() -> str:
    """Path of the gammalog program: $GAMMALOG_EXE, else an existing build
    (see ``_build_dirs``), else it is built now."""
    if "GAMMALOG_EXE" in os.environ:
        return os.environ["GAMMALOG_EXE"]
    for d in _build_dirs() + [os.path.join(_GAMMA_SRC, "build")]:
        exe = os.path.join(d, _EXE_NAME)
        if os.path.isfile(exe):
            return exe
    print("The Geant4 gamma-ray program is not built yet; building it now ...")
    return build_gammalog()


def _geant4_dir() -> str:
    """Geant4 install prefix: $GEANT4_DIR, else `geant4-config --prefix`, else the Windows default."""
    if "GEANT4_DIR" in os.environ:
        return os.environ["GEANT4_DIR"]
    try:
        out = subprocess.run(["geant4-config", "--prefix"], capture_output=True, text=True)
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except OSError:
        pass
    return r"C:\Program Files\Geant4-11.4" if os.name == "nt" else "/usr/local"


def geant4_env() -> dict:
    """Environment with the Geant4 libraries on PATH and the data sets located.

    Data sets are taken from G4LEDATA etc. if set, else searched in
    $GEANT4_DATA_DIR, ~/geant4-data and <Geant4 prefix>/share/Geant4/data.
    """
    env = dict(os.environ)
    g4 = _geant4_dir()
    env["PATH"] = os.path.join(g4, "bin") + os.pathsep + env.get("PATH", "")
    dirs = [os.environ.get("GEANT4_DATA_DIR", ""), os.path.join(os.path.expanduser("~"), "geant4-data"),
            os.path.join(g4, "share", "Geant4", "data")]
    # folder names: G4EMLOW8.8 (CERN tarballs) or EMLOW8.8 (conda-forge)
    for var, pattern in (("G4LEDATA", "*EMLOW*"), ("G4ENSDFSTATEDATA", "*ENSDFSTATE*"),
                         ("G4LEVELGAMMADATA", "PhotonEvaporation*")):
        if var in env:
            continue
        found = sorted(f for d in dirs if d for f in glob.glob(os.path.join(d, pattern)))
        if not found:
            raise FileNotFoundError(
                f"Geant4 data set {pattern} not found in {[d for d in dirs if d]}. "
                f"See docs/INSTALL.md, or set GEANT4_DATA_DIR or {var}.")
        env[var] = found[-1]
    return env


@dataclass
class GammaGeometry:
    """
    Natural gamma-ray log in a backfilled borehole through a layered earth.

    Parameters
    ----------
    layers : list of (top, bottom, GammaMedium or dict)
        Depths positive down (m). The last layer extends downwards.
    borehole_length, borehole_diameter : float
        Borehole depth and diameter (m).
    backfill : GammaMedium or dict
        Fill between the tool and the borehole wall.
    zones : list of (outer_radius, medium), optional
        Concentric fill zones (inside -> out), e.g. casing + gravel pack.
        Overrides ``backfill``; the last radius is the borehole radius.
    measuring_spacing : float
        Station step (m). Stations run from ``top_clearance`` to
        ``borehole_length - bottom_clearance``.
    crystal_diameter, crystal_length : NaI(Tl) crystal size (m).
    tool_diameter, housing_thickness : steel housing (m).
    events : int
        Simulated emissions per station and nuclide group. The Monte Carlo
        error of the gross count rate is returned (``cps_err``) and its median
        printed by ``run``; it falls as 1/sqrt(events).
    n_jobs : int
        Parallel Geant4 processes (default: all CPU cores).
    """
    layers: Sequence
    borehole_length: float
    borehole_diameter: float
    backfill: Union[GammaMedium, dict] = field(default_factory=lambda: GammaMedium("water"))
    zones: Optional[Sequence] = None
    measuring_spacing: float = 0.1
    top_clearance: float = 0.5
    bottom_clearance: float = 0.5
    crystal_diameter: float = 0.038
    crystal_length: float = 0.102
    tool_diameter: float = 0.050
    housing_thickness: float = 0.003
    housing_below: float = 0.3        # housing length below / above the crystal centre
    housing_above: float = 0.5
    resolution: float = 0.07          # FWHM/E at 662 keV
    threshold: float = 0.1            # MeV, gross-count threshold
    source_radius: float = 1.0        # emission sphere around the crystal centre (m)
    sampling_length: float = 0.15     # importance-sampling length (m)
    events: int = 150_000
    n_jobs: Optional[int] = None
    seed: int = 12345
    calibration_material: str = "sand"

    def __post_init__(self) -> None:
        self.layers = [(float(t), float(b), _medium(m)) for t, b, m in self.layers]
        self.set_backfill(self.backfill, self.zones)
        if self.n_jobs is None:
            self.n_jobs = os.cpu_count() or 1
        self._calibration: Optional[Dict[str, float]] = None

    def set_backfill(self, backfill=None, zones=None) -> None:
        """Change the borehole fill: one ``backfill`` medium or a list of ``zones``."""
        if zones is not None:
            self.zones = [(float(r), _medium(m)) for r, m in zones]
        else:
            self.backfill = _medium(backfill)
            self.zones = [(self.borehole_diameter / 2.0, self.backfill)]
        if self.tool_diameter / 2.0 >= self.zones[0][0]:
            raise ValueError("tool_diameter must be smaller than the innermost zone diameter")

    # ------------------------------------------------------------------
    def stations(self) -> np.ndarray:
        z0, z1 = self.top_clearance, self.borehole_length - self.bottom_clearance
        return np.round(np.arange(z0, z1 + 1e-9, self.measuring_spacing), 6)

    def _input(self, stations, layers, zones, hole_length, seed, out, spec) -> str:
        mats = {}

        def mat_id(name):
            return mats.setdefault(name, len(mats) + 3)

        lines = [f"seed {seed}", f"events {self.events}",
                 f"source {self.source_radius} {self.sampling_length} 0.5",
                 f"resolution {self.resolution}",
                 f"spectrum 300 3.0 {self.threshold}",
                 "material 0 nist G4_AIR", "material 1 nist G4_SODIUM_IODIDE",
                 "material 2 nist G4_STAINLESS-STEEL", "air 0"]
        for g in GROUPS:
            lines.append(f"group {g} {len(GAMMA_LINES[g])} " +
                         " ".join(f"{e} {y}" for e, y in GAMMA_LINES[g]))
        for name, (a, b) in WINDOWS.items():
            lines.append(f"window {name} {a} {b}")

        def region(m: GammaMedium):
            key = (m.material, m.rho)
            return f"{mat_id(key)} " + " ".join(f"{a:.6g}" for a in m.activity())

        for t, b, m in layers:
            lines.append(f"layer {t} {b} {region(m)}")
        for r, m in zones:
            lines.append(f"zone {r} {region(m)}")
        lines.append(f"hole {hole_length}")
        rc, rho = self.crystal_diameter / 2.0, self.tool_diameter / 2.0
        lines.append(f"tool {rc} {self.crystal_length} {rho - self.housing_thickness} {rho} "
                     f"{self.housing_below} {self.housing_above} 1 2 0")
        for (name, rho_bulk), i in mats.items():
            _, comp = material_composition(name)
            lines.append(f"material {i} mix {rho_bulk:.6g} {len(comp)} " +
                         " ".join(f"{_Z[el]} {f:.6g}" for el, f in comp.items()))
        lines.append("stations " + " ".join(f"{z:g}" for z in stations))
        lines.append(f"output {out} {spec}")
        return "\n".join(lines) + "\n"

    def _simulate(self, stations, layers, zones, hole_length):
        """Run gammalog over the stations in parallel; returns (counts, spectra)."""
        exe, env = gammalog_exe(), geant4_env()
        n = max(1, min(self.n_jobs, len(stations)))
        chunks = [stations[i::n] for i in range(n)]
        with tempfile.TemporaryDirectory(prefix="gammalog_") as tmp:
            def job(i):
                out, spec = (os.path.join(tmp, f"{k}_{i}.csv") for k in ("out", "spec"))
                inp = os.path.join(tmp, f"in_{i}.txt")
                with open(inp, "w") as fh:
                    fh.write(self._input(chunks[i], layers, zones, hole_length,
                                         self.seed + 7919 * i, out, spec))
                r = subprocess.run([exe, inp], env=env, cwd=tmp, capture_output=True, text=True)
                if r.returncode != 0:
                    raise RuntimeError(f"gammalog failed:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
                return pd.read_csv(out), pd.read_csv(spec)

            with ThreadPoolExecutor(n) as pool:
                res = list(pool.map(job, range(n)))
        counts = pd.concat([r[0] for r in res]).sort_values(["depth", "group"]).reset_index(drop=True)
        spectra = pd.concat([r[1] for r in res]).sort_values(["depth", "group"]).reset_index(drop=True)
        return counts, spectra

    # ------------------------------------------------------------------
    def calibrate(self, verbose: bool = True) -> Dict[str, float]:
        """
        Count rate (counts/s) per unit concentration (1 % K, 1 ppm U, 1 ppm Th)
        in an infinite homogeneous formation of ``calibration_material``,
        no borehole. Keys: '<group>' (gross), '<group>_<window>', and
        'cps_per_api' (gross count rate per API unit for the API_PIT mix).
        The calibration does not depend on the borehole, so it is kept across
        ``set_backfill`` calls.
        """
        if self._calibration is None:
            unit = GammaMedium(self.calibration_material, K=1.0, U=1.0, Th=1.0)
            # the same station once per process, averaged
            counts, _ = self._simulate([5.0] * self.n_jobs, [(0.0, 1e3, unit)], [], 0.0)
            cal = {}
            for g, d in counts.groupby("group"):
                cal[g] = float(d["cps"].mean())
                for w in WINDOWS:
                    cal[f"{g}_{w}"] = float(d[w].mean())
            pit_api = sum(API_WEIGHTS[g] * API_PIT[g] for g in GROUPS)
            cal["cps_per_api"] = sum(cal[g] * API_PIT[g] for g in GROUPS) / pit_api
            self._calibration = cal
            if verbose:
                print("  GR calibration (cps per 1 % K / 1 ppm U / 1 ppm Th): " +
                      ", ".join(f"{g} {cal[g]:.3g}" for g in GROUPS) +
                      f";  {cal['cps_per_api']:.3g} cps/API")
        return self._calibration

    def run(self, verbose: bool = True, include_borehole: bool = True,
            return_spectra: bool = False):
        """
        Simulate the log.

        Returns a DataFrame with one row per station:
          depth, cps (gross count rate), cps_err (1-sigma counting error of the
          simulation), cps_K / cps_U / cps_Th (gross count rate from each
          nuclide group), K_win / U_win / Th_win (window count rates),
          K_app / U_app / Th_app (apparent concentrations from the calibration),
          GR_api (gross count rate in API units), GR_err (its Monte Carlo
          error) and GR_true (16 K + 8 U + 4 Th of the layer at the station).
        With ``return_spectra`` a second DataFrame of pulse-height spectra
        (counts/s per 10 keV bin, summed over the groups) is also returned.
        """
        cal = self.calibrate(verbose)
        stations = list(self.stations())
        zones = self.zones if include_borehole else []
        if verbose:
            what = "with borehole" if include_borehole else "no borehole"
            print(f"  GR: {len(stations)} stations, {self.events} events/station/group, "
                  f"{self.n_jobs} processes ({what})")
        counts, spectra = self._simulate(stations, self.layers, zones,
                                         self.borehole_length if include_borehole else 0.0)

        rows = []
        for z, d in counts.groupby("depth"):
            d = d.set_index("group")
            row = {"depth": z,
                   "cps": float(d["cps"].sum()),
                   "cps_err": float(np.sqrt((d["cps_err"] ** 2).sum()))}
            for g in GROUPS:
                row[f"cps_{g}"] = float(d.loc[g, "cps"]) if g in d.index else 0.0
            for w in WINDOWS:
                row[w] = float(d[w].sum())
            for g in GROUPS:
                row[f"{g}_app"] = row[f"cps_{g}"] / cal[g] if cal[g] > 0 else np.nan
            row["GR_api"] = row["cps"] / cal["cps_per_api"]
            row["GR_err"] = row["cps_err"] / cal["cps_per_api"]
            row["GR_true"] = self._layer_at(z).api
            rows.append(row)
        log = pd.DataFrame(rows)
        if verbose:
            print(f"  GR: median counting error {np.median(log.cps_err / log.cps) * 100:.1f} %")
        if not return_spectra:
            return log

        bins = [c for c in spectra.columns if c.startswith("b")]
        spec = spectra.groupby("depth")[bins].sum()
        spec.columns = np.round((np.arange(len(bins)) + 0.5) * 3.0 / len(bins), 4)
        return log, spec

    def _layer_at(self, depth: float) -> GammaMedium:
        for t, b, m in self.layers:
            if depth < b:
                return m
        return self.layers[-1][2]
