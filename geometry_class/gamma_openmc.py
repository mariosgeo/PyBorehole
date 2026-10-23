# -*- coding: utf-8 -*-
"""
gamma_openmc.py
===============

Natural gamma-ray log simulation with OpenMC, as an independent check of the
Geant4 simulator (``gamma_geometry.py``).

``OpenMCGammaGeometry`` is a ``GammaGeometry``: same parameters, layers,
materials, gamma lines, tool and API calibration, so the two codes see
exactly the same problem. Only the transport differs:

  * **Code**      : OpenMC (photon transport, thick-target bremsstrahlung for
                    electrons), with the ENDF/B-VIII.0 HDF5 library.
  * **Model**     : the same coaxial (r, z) cells as in Geant4, rebuilt for
                    every station; the crystal is one cell.
  * **Source**    : every radioactive cell within ``source_radius`` of the
                    crystal (cylinder r <= R, |z| <= R) is a uniform source
                    with its emission rate as strength.
  * **Tally**     : pulse-height spectrum in the crystal; the NaI resolution
                    is applied to the spectrum afterwards.
  * **Sampling**  : analog, no variance reduction. This makes the run an
                    independent check of the Geant4 importance sampling and
                    weight window, at the price of many more particles: use a
                    coarser ``measuring_spacing`` (default 0.25 m).

OpenMC does not run natively on Windows. The jobs are run in WSL by
``gamma/openmc_worker.py``, one process per chunk of stations.

Setup (in WSL)
--------------
    conda create -n openmc -c conda-forge openmc
    # ENDF/B-VIII.0 HDF5 library (3.4 GB) from https://openmc.org/data/,
    # unpacked under ~/openmc-data/ (or set OPENMC_CROSS_SECTIONS)

Environment
-----------
OPENMC_WSL_DISTRO  WSL distribution (default ``Ubuntu``)
OPENMC_PYTHON      Python of the OpenMC env inside WSL
                   (default ``$HOME/miniforge3/envs/openmc/bin/python``)
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.special import ndtr

from .gamma_geometry import (API_PIT, API_WEIGHTS, GAMMA_LINES, GROUPS, WINDOWS,
                             GammaGeometry, GammaMedium, material_composition)

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_WORKER = os.path.join(_ROOT, "gamma", "openmc_worker.py")

_FIXED_MATERIALS = {
    "air": {"density": 0.00120479, "elements": {"C": 0.000124, "N": 0.755268, "O": 0.231781,
                                                "Ar": 0.012827}},
    "nai": {"density": 3.667, "elements": {"Na": 0.153373, "I": 0.846627}},
    "steel": {"density": 8.0, "elements": {"Fe": 0.7462, "Cr": 0.1690, "Ni": 0.0848}},
}


def _wsl_path(path: str) -> str:
    """Windows path -> /mnt/<drive>/... path inside WSL."""
    path = os.path.abspath(path).replace("\\", "/")
    return f"/mnt/{path[0].lower()}{path[2:]}"


@dataclass
class OpenMCGammaGeometry(GammaGeometry):
    """
    Same parameters as ``GammaGeometry``, plus:

    particles : int
        Source photons per station (all nuclide groups together). The
        counting error falls as 1/sqrt(particles); 10 million gives about
        2-3 % per station.
    measuring_spacing : float
        Default 0.25 m (analog transport is slower than the Geant4 run).
    depths : list of float, optional
        Simulate only these station depths (m) instead of a full log, e.g.
        the middle of each layer; about 30 s each on 24 cores with the
        default ``particles``.
    """
    measuring_spacing: float = 0.25
    depths: Optional[Sequence[float]] = None
    particles: int = 10_000_000
    batches: int = 10
    energy_bins: int = 1000

    def stations(self) -> np.ndarray:
        if self.depths is not None:
            return np.asarray(self.depths, dtype=float)
        return super().stations()

    # ------------------------------------------------------------------
    def _region_at(self, r, d, zc, layers, zones, hole_length):
        """Material key and radioactive medium (or None) at radius r, depth d."""
        rc, rho = self.crystal_diameter / 2.0, self.tool_diameter / 2.0
        if abs(d - zc) <= 0.5 * self.crystal_length and r < rc:
            return "nai", None
        if zc - self.housing_above <= d <= zc + self.housing_below:
            if r < rho - self.housing_thickness:
                return "air", None
            if r < rho:
                return "steel", None
        if d < 0:
            return "air", None
        if d < hole_length and zones and r < zones[-1][0]:
            for rout, m in zones:
                if r < rout:
                    return (m.material, m.rho), m
        for top, bot, m in layers:
            if d < bot:
                return (m.material, m.rho), m
        return (layers[-1][2].material, layers[-1][2].rho), layers[-1][2]

    def _job(self, zc, layers, zones, hole_length, groups, particles, seed, threads):
        """Self-contained OpenMC job (cm, local z = depth above the crystal centre)."""
        R = self.source_radius
        Rw = Hw = R + 0.5
        rc, rho = self.crystal_diameter / 2.0, self.tool_diameter / 2.0
        radii = [0.0, rc, rho - self.housing_thickness, rho] + [r for r, _ in zones] + [Rw, R]
        radii = sorted({round(r, 9) for r in radii if 0 <= r <= Rw})
        d0, d1 = zc - Hw, zc + Hw
        zs = [d0, d1, 0.0, hole_length, zc - 0.5 * self.crystal_length, zc + 0.5 * self.crystal_length,
              zc - self.housing_above, zc + self.housing_below, zc - R, zc + R]
        for t, b, _ in layers:
            zs += [t, b]
        zs = sorted({round(z, 9) for z in zs if d0 <= z <= d1})

        materials, cells, sources = {}, [], []
        for zlo_d, zhi_d in zip(zs[:-1], zs[1:]):
            dm = 0.5 * (zlo_d + zhi_d)
            # local coordinates (cm), z up
            zlo, zhi = 100.0 * (zc - zhi_d), 100.0 * (zc - zlo_d)
            for rin, rout in zip(radii[:-1], radii[1:]):
                key, medium = self._region_at(0.5 * (rin + rout), dm, zc, layers, zones, hole_length)
                if key == "nai":
                    continue                      # the crystal is one separate cell
                name = key if isinstance(key, str) else f"{key[0]}_{key[1]:.4f}"
                if name not in materials:
                    if isinstance(key, str):
                        materials[name] = _FIXED_MATERIALS[key]
                    else:
                        _, comp = material_composition(key[0])
                        materials[name] = {"density": key[1], "elements": comp}
                cells.append([100.0 * rin, 100.0 * rout, zlo, zhi, name])
                if medium is None or rout > R + 1e-9 or abs(zc - dm) > R:
                    continue
                act = dict(zip(GROUPS, medium.activity()))      # decays / s / m3
                vol = np.pi * (rout ** 2 - rin ** 2) * (zhi_d - zlo_d)   # m3
                for g in groups:
                    e, y = map(list, zip(*GAMMA_LINES[g]))
                    strength = act[g] * sum(y) * vol            # photons / s
                    if strength > 0:
                        sources.append([100.0 * rin, 100.0 * rout, zlo, zhi, strength,
                                        [1e6 * x for x in e], y])
        materials["nai"] = _FIXED_MATERIALS["nai"]
        h = 50.0 * self.crystal_length
        return {"depth": zc, "group": "+".join(groups), "materials": materials, "cells": cells,
                "crystal": [100.0 * rc, -h, h], "world": [100.0 * Rw, 100.0 * Hw],
                "sources": sources, "particles": int(particles), "batches": self.batches,
                "seed": int(seed), "threads": int(threads),
                "edges": np.linspace(0.0, 3e6, self.energy_bins + 1).tolist(),
                "threshold": 1e6 * self.threshold}

    def _spectrum_rates(self, res):
        """Counts/s above the threshold and in each window, NaI resolution applied."""
        edges = np.linspace(0.0, 3.0, self.energy_bins + 1)
        e = 0.5 * (edges[1:] + edges[:-1])
        rate = np.asarray(res["mean"])          # counts/s per bin (see openmc_worker.py)
        sigma = np.maximum(self.resolution / 2.3548 * np.sqrt(0.662 * e), 1e-9)
        out = {"cps": float((rate * (1.0 - ndtr((self.threshold - e) / sigma))).sum())}
        for w, (a, b) in WINDOWS.items():
            out[w] = float((rate * (ndtr((b - e) / sigma) - ndtr((a - e) / sigma))).sum())
        rel = res["gross_std"] / res["gross_mean"] if res["gross_mean"] > 0 else np.nan
        out["cps_err"] = out["cps"] * rel
        return out

    def _simulate(self, stations, layers, zones, hole_length, groups=GROUPS, particles=None):
        """Run OpenMC in WSL over the stations; one row per station."""
        particles = particles or self.particles
        n = max(1, min(self.n_jobs, len(stations)))
        threads = max(1, self.n_jobs // n)
        chunks = [list(range(i, len(stations), n)) for i in range(n)]
        distro = os.environ.get("OPENMC_WSL_DISTRO", "Ubuntu")
        python = os.environ.get("OPENMC_PYTHON", "$HOME/miniforge3/envs/openmc/bin/python")
        with tempfile.TemporaryDirectory(prefix="openmc_gr_") as tmp:
            def job(i):
                jobs = [self._job(stations[k], layers, zones, hole_length, groups, particles,
                                  self.seed + 7919 * k + 1, threads) for k in chunks[i]]
                jf, rf = os.path.join(tmp, f"jobs_{i}.json"), os.path.join(tmp, f"res_{i}.json")
                with open(jf, "w") as fh:
                    json.dump(jobs, fh)
                cmd = f'"{python}" "{_wsl_path(_WORKER)}" "{_wsl_path(jf)}" "{_wsl_path(rf)}"'
                r = subprocess.run(["wsl.exe", "-d", distro, "--", "bash", "-c", cmd],
                                   capture_output=True, text=True)
                if r.returncode != 0:
                    raise RuntimeError(f"OpenMC worker failed:\n{r.stdout[-2000:]}\n{r.stderr[-3000:]}")
                with open(rf) as fh:
                    return json.load(fh)

            with ThreadPoolExecutor(n) as pool:
                results = [r for chunk in pool.map(job, range(n)) for r in chunk]
        rows = [{"depth": r["depth"], "group": r["group"], **self._spectrum_rates(r)} for r in results]
        return pd.DataFrame(rows).sort_values("depth").reset_index(drop=True)

    # ------------------------------------------------------------------
    def calibrate(self, verbose: bool = True) -> Dict[str, float]:
        """As ``GammaGeometry.calibrate``: one run per nuclide group in an
        infinite formation of ``calibration_material``, 4x the particles."""
        if self._calibration is None:
            unit = GammaMedium(self.calibration_material, K=1.0, U=1.0, Th=1.0)
            cal = {}
            for g in GROUPS:
                d = self._simulate([5.0], [(0.0, 1e3, unit)], [], 0.0, groups=(g,),
                                   particles=4 * self.particles)
                cal[g] = float(d["cps"].iloc[0])
                for w in WINDOWS:
                    cal[f"{g}_{w}"] = float(d[w].iloc[0])
            pit_api = sum(API_WEIGHTS[g] * API_PIT[g] for g in GROUPS)
            cal["cps_per_api"] = sum(cal[g] * API_PIT[g] for g in GROUPS) / pit_api
            self._calibration = cal
            if verbose:
                print("  OpenMC GR calibration (cps per 1 % K / 1 ppm U / 1 ppm Th): " +
                      ", ".join(f"{g} {cal[g]:.3g}" for g in GROUPS) +
                      f";  {cal['cps_per_api']:.3g} cps/API")
        return self._calibration

    def run(self, verbose: bool = True, include_borehole: bool = True,
            calibration: Optional[Dict[str, float]] = None) -> pd.DataFrame:
        """
        Simulate the log: depth, cps, cps_err, K_win / U_win / Th_win,
        GR_api, GR_err, GR_true (as ``GammaGeometry.run``; the analog run does
        not separate the nuclide groups, so there are no K_app / U_app / Th_app).

        ``calibration``: use another calibration, e.g. the Geant4 one, so
        that both logs are converted to API with the same factor.
        """
        cal = calibration or self.calibrate(verbose)
        stations = list(self.stations())
        zones = self.zones if include_borehole else []
        if verbose:
            what = "with borehole" if include_borehole else "no borehole"
            print(f"  OpenMC GR: {len(stations)} stations, {self.particles:.3g} photons/station, "
                  f"{self.n_jobs} cores ({what})")
        log = self._simulate(stations, self.layers, zones,
                             self.borehole_length if include_borehole else 0.0).drop(columns="group")
        log["GR_api"] = log["cps"] / cal["cps_per_api"]
        log["GR_err"] = log["cps_err"] / cal["cps_per_api"]
        log["GR_true"] = [self._layer_at(z).api for z in log["depth"]]
        if verbose:
            print(f"  OpenMC GR: median counting error {np.median(log.cps_err / log.cps) * 100:.1f} %")
        return log
