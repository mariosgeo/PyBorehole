"""Checks of the rock-physics relations and the built-in database (no simulators)."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from geometry_class.gamma_geometry import MATERIALS, SOLUTIONS  # noqa: E402
from geometry_class.petrophysics import (BACKFILLS, LITHOLOGIES, WATERS, archie_resistivity,  # noqa: E402
                                         backfill_resistivity, bulk_density, gamma_api, unit_properties)


def test_archie_formula():
    # rho = a rho_w phi^-m Sw^-n
    assert archie_resistivity(10.0, 0.25, a=1.0, m=2.0, n=2.0, sw=1.0) == pytest.approx(160.0)
    assert archie_resistivity(10.0, 0.25, a=1.0, m=2.0, n=2.0, sw=0.5) == pytest.approx(640.0)
    assert archie_resistivity(10.0, 0.25, a=0.62, m=2.15, n=2.0) == pytest.approx(0.62 * 10 * 0.25 ** -2.15)


def test_surface_conduction_in_parallel():
    pure = archie_resistivity(20.0, 0.45, m=2.0)
    with_clay = archie_resistivity(20.0, 0.45, m=2.0, sigma_s=0.08)
    assert with_clay == pytest.approx(1.0 / (1.0 / pure + 0.08))
    assert with_clay < 20.0 < pure     # Archie alone overestimates a clay several-fold


def test_archie_rejects_bad_input():
    with pytest.raises(ValueError):
        archie_resistivity(-1.0, 0.3)
    with pytest.raises(ValueError):
        archie_resistivity(10.0, 0.0)


def test_database_is_consistent():
    for name, lit in LITHOLOGIES.items():
        assert 0 < lit["porosity"] < 1 and lit["m"] >= 1 and lit["sigma_s"] >= 0, name
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
