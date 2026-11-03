"""Checks of the borehole NMR simulator (slim water-well probe)."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from geometry_class.nmr_geometry import GAMMA_H, NMRGeometry, NMRMedium  # noqa: E402
from geometry_class.petrophysics import BACKFILL_NMR, LITHOLOGIES  # noqa: E402


def medium(name):
    lit = LITHOLOGIES[name]
    return NMRMedium(lit["porosity"], lit["t2lm"], lit["t2_sigma"], lit["cbw"])


SAND, GRAVEL, CLAY = medium("Sand"), medium("Gravel"), medium("Clay")


def mid(log, diameter, z0, z1):
    """Mean of the log over stations well inside [z0, z1] for one shell."""
    return log[(np.isclose(log.diameter, diameter)) & (log.depth > z0 + 0.5) & (log.depth < z1 - 0.5)].mean()


def test_shell_radius_follows_the_magnet_field():
    g = NMRGeometry([(0, 10, SAND)], 10, 0.1, frequencies=[450e3], magnet_diameter=0.025, remanence=1.2)
    s = g.shells().iloc[0]
    b0 = 2 * np.pi * 450e3 / GAMMA_H
    r = np.sqrt(1.2 * 0.0125 ** 2 / 2 / b0)            # B0 = Br a^2 / (2 r^2)
    assert s.radius == pytest.approx(r)
    assert s.gradient == pytest.approx(2 * b0 / r)
    assert 0.17 < s.diameter < 0.21                     # a slim-probe depth of investigation


def test_noise_free_formation_is_recovered():
    """Thick sand and gravel, no borehole, no noise: water and mobile water within 3 %."""
    g = NMRGeometry([(0, 5, SAND), (5, 10, GRAVEL)], 10, 0.1, noise=0.0, frequencies=[310e3])
    log = g.run(include_borehole=False)
    for (z0, z1), m in (((0, 5), SAND), ((5, 10), GRAVEL)):
        r = mid(log, log.diameter.iloc[0], z0, z1)
        assert r.water == pytest.approx(m.water, rel=0.03)
        assert r.mobile == pytest.approx(m.fractions()["mobile"], rel=0.05)


def test_noisy_log_scatter_is_a_few_percent():
    g = NMRGeometry([(0, 10, SAND)], 10, 0.1, noise=0.02, frequencies=[310e3])
    log = g.run(include_borehole=False)
    s = log[(log.depth > 1) & (log.depth < 9)]
    assert abs(s.water.mean() - SAND.water) < 0.03
    assert s.water.std() < 0.04


def test_shell_inside_the_hole_reads_the_fill():
    """20 cm water-filled hole: the 19 cm shell reads ~100 % water, the 23 cm shell the formation."""
    g = NMRGeometry([(0, 10, SAND)], 10, 0.20, fill=BACKFILL_NMR["Water"], noise=0.0)
    hole, ref = g.run(include_borehole=True), g.run(include_borehole=False)
    inner, outer = sorted(hole.diameter.unique())
    assert inner < 0.20 < outer
    assert mid(hole, inner, 0, 10).water == pytest.approx(1.0, abs=0.05)
    assert mid(hole, outer, 0, 10).water == pytest.approx(mid(ref, outer, 0, 10).water, abs=0.01)
    assert mid(ref, inner, 0, 10).water == pytest.approx(SAND.water, rel=0.05)


def test_bound_water_of_grout_needs_signal_to_noise():
    """Bentonite grout: its water is bound (T2 ~2 ms, TE = 1.5 ms). Without noise the
    inversion extrapolates it back; with realistic noise it is underestimated. Either
    way it is bound, not mobile water."""
    grout = BACKFILL_NMR["Bentonite grout"]
    reads = {}
    for noise in (0.0, 0.02):
        g = NMRGeometry([(0, 10, SAND)], 10, 0.20, fill=grout, noise=noise)
        reads[noise] = mid(g.run(include_borehole=True), g.shells().diameter.min(), 0, 10)
    assert reads[0.0].water == pytest.approx(grout["water"], rel=0.08)
    assert reads[0.02].water < 0.9 * grout["water"]
    assert reads[0.0].mobile < 0.05 and reads[0.02].mobile < 0.05


def test_diffusion_shortens_long_t2_more_in_the_stronger_gradient():
    g = NMRGeometry([(0, 10, NMRMedium(1.0, 2500.0, 0.05))], 10, 0.1, noise=0.0)
    log = g.run(include_borehole=False)
    inner, outer = sorted(log.diameter.unique())         # inner shell: higher field gradient
    t_in, t_out = mid(log, inner, 0, 10).t2lm_ms, mid(log, outer, 0, 10).t2lm_ms
    assert t_in < t_out < 2500.0


def test_shell_inside_the_probe_is_rejected():
    with pytest.raises(ValueError):
        NMRGeometry([(0, 10, SAND)], 10, 0.2, frequencies=[5e6])


def test_database_t2_follows_pore_size():
    for name, lit in LITHOLOGIES.items():
        assert lit["t2lm"] > 0 and lit["t2_sigma"] > 0 and 0 <= lit["cbw"] <= 1, name
    t2 = {n: LITHOLOGIES[n]["t2lm"] for n in ("Gravel", "Sand", "Silt", "Clay")}
    assert t2["Gravel"] > t2["Sand"] > t2["Silt"] > t2["Clay"]
