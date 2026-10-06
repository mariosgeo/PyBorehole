"""
Validation of the ERT and EM simulators against analytic solutions.

    pytest tests/                 # all tests (ERT tests take a few minutes)
    pytest tests/ -m "not slow"   # EM + model tests only (seconds)
"""
import os
import sys
import warnings

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

from geometry_class import EMGeometry, Geometry, merge_protocols  # noqa: E402
from geometry_class.analytic import (borehole_coaxial_hs, borehole_potential, borehole_wenner_rhoa,  # noqa: E402
                                     fullspace_hs, ring_on_mandrel_potential, tool_lateral_rhoa,
                                     tool_normal_rhoa)
from geometry_class.borehole_model import make_zones, resistivity_rz  # noqa: E402

LAYERS_HOMOG = np.array([[0.0, 20.0, 100.0]])


# ── Shared model ──────────────────────────────────────────────────────────────
def test_resistivity_model():
    layers = np.array([[0.0, 5.0, 100.0], [5.0, 10.0, 10.0]])
    zones = make_zones(0.2, 8.0, 1.0, [(0.05, 1e6), (0.1, 30.0, 2.0, 6.0)])
    r = np.array([0.02, 0.08, 0.08, 0.5, 0.5, 0.5])
    d = np.array([3.0, 3.0, 7.0, 3.0, 7.0, 50.0])
    rho = resistivity_rz(r, d, layers, 5.0, zones, air_resistivity=1e8)
    np.testing.assert_allclose(rho, [1e6, 30.0, 10.0, 100.0, 10.0, 10.0])
    rho0 = resistivity_rz(r, d, layers, 5.0, zones, include_borehole=False)
    np.testing.assert_allclose(rho0, [100.0, 100.0, 10.0, 100.0, 10.0, 10.0])
    assert resistivity_rz(0.5, -1.0, layers, 5.0, zones, air_resistivity=1e8)[0] == 1e8


# ── EM ────────────────────────────────────────────────────────────────────────
def _em(rho_bf, **kw):
    kw.setdefault("air_resistivity", 1e8)
    kw.setdefault("measuring_spacing", 10.0)
    kw.setdefault("frequencies", [400.0, 8000.0])
    return EMGeometry(borehole_length=20.0, borehole_diameter=0.2, layer_1d_geometry=LAYERS_HOMOG,
                      backfill_resistivity=rho_bf, coil_spacings=[0.5, 1.0], **kw)


def test_em_fullspace_matches_analytic():
    """Homogeneous full space (air = formation, no hole): quadrature within 2 %."""
    df = _em(100.0, air_resistivity=100.0).run(verbose=False, include_borehole=False)
    df = df[np.isclose(df.tx_depth, 10.0)]
    assert len(df) == 4
    for _, row in df.iterrows():
        hs, _ = fullspace_hs(100.0, row.frequency, row.spacing)
        assert abs(row.imag / hs.imag - 1) < 0.02
        assert abs(row.apparent_resistivity / 100.0 - 1) < 0.02


@pytest.mark.parametrize("rho_bf", [1.0, 1000.0])
def test_em_borehole_matches_analytic(rho_bf):
    """Coaxial coils in an infinite borehole: quadrature within 2 % of the Fourier-Bessel solution."""
    df = _em(rho_bf).run(verbose=False)
    df = df[np.isclose(df.tx_depth, 10.0)]
    assert len(df) == 4
    for _, row in df.iterrows():
        hs = borehole_coaxial_hs(rho_bf, 100.0, 0.1, row.frequency, row.spacing)
        assert abs(row.imag / hs.imag - 1) < 0.02


def test_em_borehole_effect_converged():
    """Conductive backfill: default mesh within 1.5 % of a 4x radially refined mesh."""
    pick = lambda df: df[np.isclose(df.tx_depth, 10.0)].apparent_resistivity.values
    coarse = pick(_em(1.0).run(verbose=False))
    fine = pick(_em(1.0, borehole_cells=40).run(verbose=False))
    np.testing.assert_allclose(coarse, fine, rtol=0.015)
    assert coarse[0] < 25.0  # strong effect for s=0.5 m: ~19 Ohm.m instead of 100


def test_em_exact_apparent_resistivity_beyond_lin():
    """1 Ohm.m full space at 8000 Hz, s = 1 m: LIN is ~13 % off, the exact inversion within 2 %."""
    em = EMGeometry(borehole_length=20.0, borehole_diameter=0.2, layer_1d_geometry=np.array([[0.0, 20.0, 1.0]]),
                    backfill_resistivity=1.0, air_resistivity=1.0, frequencies=[8000.0],
                    coil_spacings=[1.0], measuring_spacing=10.0)
    row = em.run(verbose=False).query("tx_depth == 10.0").iloc[0]
    assert abs(row.apparent_resistivity - 1.0) < 0.02
    assert row.apparent_resistivity_lin > 1.10


def test_em_tool_stays_in_borehole():
    df = _em(1.0, measuring_spacing=0.5).run(verbose=False)
    rx = df.tx_depth + df.spacing
    assert rx.max() <= 20.0 + 1e-9
    np.testing.assert_allclose(df.depth, df.tx_depth + df.spacing / 2)


def test_em_octree_matches_analytic():
    """3D octree path, centred tool, 1 Ohm.m backfill: within 1.5 % of the borehole solution."""
    em = _em(1.0, mesh_type="tree", frequencies=[400.0])
    df = em.run(verbose=False, depths=[10.0])
    assert len(df) == 2
    for _, row in df.iterrows():
        hs = borehole_coaxial_hs(1.0, 100.0, 0.1, row.frequency, row.spacing)
        assert abs(row.imag / hs.imag - 1) < 0.015


def test_em_octree_offset_homogeneous():
    """Off-centre tool in a homogeneous earth (backfill = formation) still reads 100 Ohm.m."""
    df = _em(100.0, mesh_type="tree", frequencies=[400.0], tool_offset=0.06,
             air_resistivity=100.0).run(verbose=False, depths=[10.0])
    np.testing.assert_allclose(df.apparent_resistivity, 100.0, rtol=0.015)


def test_em_offset_needs_octree():
    with pytest.raises(ValueError):
        _em(1.0, tool_offset=0.05)
    with pytest.raises(ValueError):
        _em(1.0, mesh_type="tree", tool_offset=0.1)  # on the wall


def test_em_transverse_needs_theta_cells():
    with pytest.raises(ValueError):
        _em(1.0, coil_orientation="x")


# ── ERT ───────────────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def ert_setup():
    geo = Geometry(borehole_length=15.0, borehole_diameter=0.2, layer_1d_geometry=np.array([[0.0, 15.0, 100.0]]),
                   backfill_resistivity=1.0, long_spacing=1.0)
    plc = geo.make_basic_geometry(vtk_filename=None)
    data = geo.make_array(plc, (1, 4, 2, 3), 0.5, 1.0, 0.1, protocol_filename=None)
    mesh = geo.make_mesh(plc, vtk_filename=None)
    return geo, mesh, data


def _ert_mid(sim, a_w):
    z = np.array([[sim.sensorPosition(int(i))[2] for i in sim(k)] for k in "abmn"])
    zs = np.sort(z, axis=0)
    sel = np.isclose(zs[1] - zs[0], a_w) & (z.mean(0) < -5.0) & (z.mean(0) > -10.0)
    return np.array(sim("rhoa"))[sel]


@pytest.mark.slow
@pytest.mark.parametrize("rho_bf", [1.0, 1000.0])
def test_ert_borehole_matches_analytic(ert_setup, rho_bf):
    """Wenner in an infinite borehole: 3D P2 result within 0.5 % of the Hankel solution."""
    geo, mesh, data = ert_setup
    geo.set_borehole(backfill_resistivity=rho_bf)
    sim = geo.run(mesh, data, output_filename=None)
    for a_w in (0.5, 1.0):
        exact = borehole_wenner_rhoa(a_w, 0.1, rho_bf, 100.0)
        assert abs(_ert_mid(sim, a_w).mean() / exact - 1) < 0.005


@pytest.mark.slow
def test_ert_no_borehole_reference(ert_setup):
    geo, mesh, data = ert_setup
    sim = geo.run(mesh, data, output_filename=None, include_borehole=False)
    np.testing.assert_allclose(np.array(sim("rhoa")), 100.0, rtol=1e-3)


def test_ert_boundary_markers():
    geo = Geometry(borehole_length=5.0, borehole_diameter=0.2, layer_1d_geometry=np.array([[0.0, 5.0, 100.0]]),
                   backfill_resistivity=1.0, long_spacing=1.0, n_segments=12,
                   borehole_area=1e-3, near_area=1e-1, mid_area=1.0)
    plc = geo.make_basic_geometry(vtk_filename=None)
    geo.make_array(plc, (1, 4, 2, 3), 0.5, 1.0, 0.5, protocol_filename=None)
    mesh = geo.make_mesh(plc, vtk_filename=None)
    for b in mesh.boundaries():
        if b.outside():
            assert b.marker() == (-1 if abs(b.center()[2]) < 1e-6 else -2)
        else:
            assert b.marker() == 0
    # Borehole cells carry the backfill resistivity and fill the (equal-area) hole volume
    vol = sum(c.size() for c in mesh.cells() if c.center()[2] > -5.0 and np.hypot(c.center()[0], c.center()[1]) < 0.1)
    assert abs(vol / (np.pi * 0.1**2 * 5.0) - 1) < 0.05


def test_ert_electrodes_avoid_interfaces():
    """Electrodes are shifted off layer interfaces; small spacings refine the borehole cells."""
    geo = Geometry(borehole_length=6.0, borehole_diameter=0.2, layer_1d_geometry=np.array(
        [[0.0, 2.0, 100.0], [2.0, 4.0, 10.0], [4.0, 6.0, 200.0]]), backfill_resistivity=1.0)
    plc = geo.make_basic_geometry(vtk_filename=None)
    area_before = geo.borehole_area
    data = geo.make_array(plc, (1, 4, 2, 3), 0.03, 0.06, 0.01, protocol_filename=None)
    depths = -np.array([data.sensorPosition(i)[2] for i in range(data.sensorCount())])
    assert np.isclose(depths[0], 0.005)
    assert np.min(np.abs(depths[:, None] - np.array([2.0, 4.0, 6.0])[None, :])) > 1e-3
    assert geo.borehole_area < area_before


# ── Realistic tools ──────────────────────────────────────────────────────────
def test_analytic_mandrel_limits():
    """Ring-on-mandrel solution: a vanishing tool gives the on-axis point solution; far away -> rho/(4 pi z)."""
    for z in (0.4064, 1.6256):
        np.testing.assert_allclose(ring_on_mandrel_potential(z, 1e-4, 0.1, 1.0, 100.0),
                                   borehole_potential(z, 0.1, 1.0, 100.0), rtol=1e-4)
    np.testing.assert_allclose(ring_on_mandrel_potential(5.0, 0.025, 0.1, 1.0, 1.0), 1 / (4 * np.pi * 5.0), rtol=1e-3)


def test_em_bucking_matches_analytic():
    """Focused (bucked) coaxial tool in a 1 Ohm.m backfilled hole: within 2 % of the analytic solution."""
    em = EMGeometry(borehole_length=20.0, borehole_diameter=0.2, layer_1d_geometry=LAYERS_HOMOG,
                    backfill_resistivity=1.0, frequencies=[400.0], coil_spacings=[0.5],
                    bucking_spacings=[0.25], measuring_spacing=10.0)
    row = em.run(verbose=False).query("tx_depth == 10.0").iloc[0]
    exact = borehole_coaxial_hs(1.0, 100.0, 0.1, 400.0, 0.5) \
        - 0.5**3 * borehole_coaxial_hs(1.0, 100.0, 0.1, 400.0, 0.25)
    assert abs(row.imag / exact.imag - 1) < 0.02
    assert row.apparent_resistivity > 60.0   # two-coil tool reads ~19 Ohm.m here


def test_merge_protocols():
    geo = Geometry(borehole_length=6.0, borehole_diameter=0.2, layer_1d_geometry=np.array([[0.0, 6.0, 100.0]]),
                   backfill_resistivity=1.0, long_spacing=2.0)
    plc = geo.make_basic_geometry(vtk_filename=None)
    d1 = geo.make_tool_array(plc, "normal", [0.4064], 0.5)
    d2 = geo.make_tool_array(plc, "lateral", [1.83], 0.5, mn_spacing=0.2)
    m = merge_protocols(d1, d2)
    assert m.size() == d1.size() + d2.size()
    assert m.sensorCount() < d1.sensorCount() + d2.sensorCount() + 1
    assert set(np.array(m["protocol"])) == {0.0, 1.0}
    assert np.all(np.array(m("b")) == -1)


@pytest.mark.slow
def test_ert_tool_rings_match_analytic():
    """Ring electrodes on a 5 cm insulating tool in 1000 Ohm.m backfill (the case where point
    electrodes are ~20 % off): normal and lateral within 0.5 % of the Fourier-Bessel solution."""
    geo = Geometry(borehole_length=10.0, borehole_diameter=0.2, layer_1d_geometry=np.array([[0.0, 10.0, 100.0]]),
                   backfill_resistivity=1000.0, long_spacing=2.0, tool_diameter=0.05)
    plc = geo.make_basic_geometry(vtk_filename=None)
    p = merge_protocols(geo.make_tool_array(plc, "normal", [0.4064], 0.5),
                        geo.make_tool_array(plc, "lateral", [1.83], 0.5, mn_spacing=0.2))
    mesh = geo.make_mesh(plc, vtk_filename=None)
    s = geo.run(mesh, p, output_filename=None)
    r, dep, prot = np.array(s["rhoa"]), np.array(s["depthref"]), np.array(s["protocol"])
    for pi in (0, 1):
        sel = (prot == pi) & (dep > 4.0) & (dep < 6.0)
        z = dep[sel].mean()
        exact = (tool_normal_rhoa(0.4064, 0.025, 0.1, 1000.0, 100.0, z) if pi == 0
                 else tool_lateral_rhoa(1.83, 0.2, 0.025, 0.1, 1000.0, 100.0, z))
        assert abs(r[sel].mean() / exact - 1) < 0.005
