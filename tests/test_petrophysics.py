"""Checks of the rock-physics relations and the built-in database (no simulators)."""
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from geometry_class.gamma_geometry import MATERIALS, SOLUTIONS  # noqa: E402
from geometry_class.petrophysics import (BACKFILLS, LITHOLOGIES, WATERS, archie_resistivity,  # noqa: E402
                                         backfill_resistivity, bulk_density, gamma_api, qv_from_cec,
                                         unit_properties, waxman_smits_b, waxman_smits_resistivity)


def test_archie_formula():
    # rho = a rho_w phi^-m Sw^-n
    assert archie_resistivity(10.0, 0.25, a=1.0, m=2.0, n=2.0, sw=1.0) == pytest.approx(160.0)
    assert archie_resistivity(10.0, 0.25, a=1.0, m=2.0, n=2.0, sw=0.5) == pytest.approx(640.0)
    assert archie_resistivity(10.0, 0.25, a=0.62, m=2.15, n=2.0) == pytest.approx(0.62 * 10 * 0.25 ** -2.15)


def test_waxman_smits_without_clay_is_archie():
    for sw in (1.0, 0.6):
        assert waxman_smits_resistivity(10.0, 0.25, a=0.8, m=1.8, n=2.2, sw=sw, cec=0.0) == \
            pytest.approx(archie_resistivity(10.0, 0.25, a=0.8, m=1.8, n=2.2, sw=sw))


def test_waxman_smits_formula():
    # sigma = phi^m Sw^n / a * (1/rho_w + B Qv / Sw)
    phi, m, n, sw, rho_w, cec, rg = 0.45, 2.0, 2.0, 0.8, 20.0, 10.0, 2.70
    qv = cec / 100 * rg * (1 - phi) / phi
    b = 4.6 * (1 - 0.6 * math.exp(-0.77 / rho_w))
    expected = 1.0 / (phi ** m * sw ** n * (1 / rho_w + b * qv / sw))
    assert qv_from_cec(cec, phi, rg) == pytest.approx(qv)
    assert waxman_smits_b(rho_w) == pytest.approx(b)
    assert waxman_smits_resistivity(rho_w, phi, 1.0, m, n, sw, cec, rg) == pytest.approx(expected)
    # an explicit B overrides the fit
    assert waxman_smits_resistivity(rho_w, phi, 1.0, m, n, sw, cec, rg, b=0.0) == \
        pytest.approx(archie_resistivity(rho_w, phi, 1.0, m, n, sw))


def test_b_limits():
    assert waxman_smits_b(0.01) == pytest.approx(4.6, rel=1e-6)           # brine
    assert waxman_smits_b(1e6) == pytest.approx(4.6 * 0.4, rel=1e-4)      # very fresh water
    assert waxman_smits_b(0.2) > waxman_smits_b(2.0) > waxman_smits_b(20.0)


def test_clay_in_fresh_water():
    clay = unit_properties("Clay", "Fresh")["resistivity"]
    lit = LITHOLOGIES["Clay"]
    plain = archie_resistivity(20.0, lit["porosity"], lit["a"], lit["m"], lit["n"])
    assert 5.0 < clay < 20.0 < plain     # Archie alone overestimates a clay several-fold


def test_shaly_units_are_less_sensitive_to_salinity():
    """Clay conduction does not scale with the water: fresh/salt is far below the water ratio."""
    water_ratio = WATERS["Fresh"]["rho_w"] / WATERS["Salt"]["rho_w"]
    clay = unit_properties("Clay", "Fresh")["resistivity"] / unit_properties("Clay", "Salt")["resistivity"]
    assert clay < 0.2 * water_ratio


def test_clay_term_grows_as_saturation_falls():
    """Waxman-Smits: the counter-ion term scales with Sw^(n-1), the water term with Sw^n."""
    full = waxman_smits_resistivity(20.0, 0.4, m=2.0, n=2.0, sw=1.0, cec=5.0)
    half = waxman_smits_resistivity(20.0, 0.4, m=2.0, n=2.0, sw=0.5, cec=5.0)
    clean_ratio = archie_resistivity(20.0, 0.4, sw=0.5) / archie_resistivity(20.0, 0.4, sw=1.0)
    assert 1.0 < half / full < clean_ratio


def test_bad_input_is_rejected():
    with pytest.raises(ValueError):
        archie_resistivity(-1.0, 0.3)
    with pytest.raises(ValueError):
        waxman_smits_resistivity(10.0, 0.0)
    with pytest.raises(ValueError):
        waxman_smits_resistivity(10.0, 0.3, cec=-1.0)


def test_database_is_consistent():
    for name, lit in LITHOLOGIES.items():
        assert 0 < lit["porosity"] < 1 and lit["m"] >= 1 and lit["cec"] >= 0, name
        assert lit["gamma_material"] in MATERIALS, name
    for name, bf in BACKFILLS.items():
        assert bf["gamma_material"] in MATERIALS or bf["gamma_material"] in SOLUTIONS, name
        for w in WATERS.values():
            assert backfill_resistivity(name, w["rho_w"]) > 0


def test_salt_water_lowers_resistivity_hundredfold_for_clean_sand():
    fresh = unit_properties("Sand", "Fresh")["resistivity"]
    salt = unit_properties("Sand", "Salt")["resistivity"]
    assert fresh / salt == pytest.approx(WATERS["Fresh"]["rho_w"] / WATERS["Salt"]["rho_w"])


def test_density_and_api():
    assert bulk_density(0.3, 2.65) == pytest.approx(0.7 * 2.65 + 0.3)
    assert gamma_api(1.0, 1.0, 1.0) == 28.0
