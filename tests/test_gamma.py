"""
Validation of the Geant4 natural gamma-ray simulator.

    pytest tests/test_gamma.py        # about a minute on 24 cores

Skipped when the gammalog program is not built (pyborehole-build-gamma) or the
Geant4 data sets are missing.
"""
import os
import sys
import warnings

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

from geometry_class.gamma_geometry import (GammaGeometry, GammaMedium, geant4_env,  # noqa: E402
                                           material_composition, MATERIALS, SOLUTIONS,
                                           _build_dirs, _EXE_NAME)


def _ready():
    """Data sets found and the program already built (the tests never build it)."""
    try:
        geant4_env()
    except FileNotFoundError:
        return False
    if "GAMMALOG_EXE" in os.environ:
        return os.path.isfile(os.environ["GAMMALOG_EXE"])
    return any(os.path.isfile(os.path.join(d, _EXE_NAME)) for d in _build_dirs())


gamma = pytest.mark.skipif(not _ready(), reason="gammalog or Geant4 data not available")
UNIT = dict(K=1.0, U=1.0, Th=1.0)


def test_materials():
    for name in list(MATERIALS) + list(SOLUTIONS):
        rho, comp = material_composition(name)
        assert abs(sum(comp.values()) - 1.0) < 1e-9
        assert 0.99 < rho < 2.8
    assert abs(material_composition("water")[0] - 1.0) < 1e-12
    assert abs(material_composition("kcl_brine")[1]["K"] - 0.05 * 0.5244) < 1e-9


def _infinite(medium, hole=False, backfill="water", events=60_000):
    """Mean gross count rate per group over one station per CPU core."""
    g = GammaGeometry([(0.0, 20.0, medium)], 20.0, 0.2, backfill=GammaMedium(backfill),
                      events=events, calibration_material=medium.material)
    counts, _ = g._simulate([8.0] * g.n_jobs, g.layers, g.zones if hole else [], 20.0 if hole else 0.0)
    return counts.groupby("group")["cps"].mean(), g.n_jobs


@pytest.mark.gamma
@gamma
def test_infinite_medium_density_invariance():
    """In an infinite medium the count rate per unit mass concentration does not
    depend on bulk density (emission ~ rho, attenuation length ~ 1/rho)."""
    lo, _ = _infinite(GammaMedium("sand", density=1.6, **UNIT))
    hi, _ = _infinite(GammaMedium("sand", density=2.7, **UNIT))
    np.testing.assert_allclose(hi.values, lo.values, rtol=0.05)


@pytest.mark.gamma
@gamma
def test_water_filled_hole_correction():
    """8-inch water-filled hole, centred 50 mm tool: the gross count rate drops
    to ~0.77 of the no-hole value (service-company GR correction charts)."""
    m = GammaMedium("sand", **UNIT)
    no_hole, _ = _infinite(m)
    hole, _ = _infinite(m, hole=True)
    assert 0.70 < hole.sum() / no_hole.sum() < 0.82


def _openmc_ready():
    """OpenMC importable where the worker runs (WSL on Windows, this Python elsewhere)."""
    import subprocess
    if os.name == "nt":
        py = os.environ.get("OPENMC_PYTHON", "$HOME/miniforge3/envs/openmc/bin/python")
        cmd = ["wsl.exe", "-d", os.environ.get("OPENMC_WSL_DISTRO", "Ubuntu"), "--",
               "bash", "-c", f'"{py}" -c "import openmc"']
    else:
        cmd = [os.environ.get("OPENMC_PYTHON", sys.executable), "-c", "import openmc"]
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return r.returncode == 0


@pytest.mark.gamma
@pytest.mark.skipif(not (_ready() and _openmc_ready()), reason="Geant4 or OpenMC (WSL) not available")
def test_openmc_matches_geant4():
    """Infinite sand, 1 % K / 1 ppm U / 1 ppm Th: the analog OpenMC count rate
    agrees with the Geant4 one (weighted, importance sampled) within 3 sigma."""
    from geometry_class import OpenMCGammaGeometry
    m = GammaMedium("sand", **UNIT)
    om = OpenMCGammaGeometry([(0.0, 20.0, m)], 20.0, 0.2, particles=10_000_000)
    d = om._simulate([8.0], om.layers, [], 0.0).iloc[0]
    g4, _ = _infinite(m)
    assert abs(d.cps - g4.sum()) < 3 * d.cps_err


@pytest.mark.gamma
@gamma
def test_log_reads_true_api_in_thick_beds():
    """No borehole: mid-bed GR in API matches 16 K + 8 U + 4 Th within 6 %."""
    layers = [(0.0, 5.0, dict(material="sand", K=1.2, U=1.5, Th=6.0)),
              (5.0, 10.0, dict(material="clay", K=2.5, U=3.5, Th=12.0))]
    g = GammaGeometry(layers, 10.0, 0.2, measuring_spacing=1.0, top_clearance=1.5,
                      bottom_clearance=1.5, events=100_000)
    log = g.run(verbose=False, include_borehole=False)
    mid = log[(log.depth - 5.0).abs() > 1.0]   # at least 1 m from the bed boundary
    np.testing.assert_allclose(mid["GR_api"], mid["GR_true"], rtol=0.06)
