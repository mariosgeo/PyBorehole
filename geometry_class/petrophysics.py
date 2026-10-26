# -*- coding: utf-8 -*-
"""
petrophysics.py
===============

A small built-in database of lithologies, pore waters and borehole fills, and
the rock-physics relations that turn them into the inputs of the simulators:
bulk resistivity (ERT, EM), and material, bulk density and K, U, Th (gamma
ray).

Bulk resistivity follows Archie's law, with an optional surface-conduction
term for clay-bearing units:

    sigma_bulk = Sw**n * phi**m / (a * rho_w)  +  sigma_s
    rho_bulk   = 1 / sigma_bulk

``sigma_s`` = 0 gives Archie's law exactly (clean sand, gravel, sandstone,
limestone). Archie's law assumes that current flows through the pore water
only; clay minerals also conduct along their surfaces, so plain Archie
overestimates the resistivity of clays several-fold in fresh water (a clay
with 45 % porosity and 20 Ohm.m water would be ~115 Ohm.m instead of the
~5-20 Ohm.m measured). ``sigma_s`` adds that conduction in parallel, the
simplest form of the Waxman-Smits idea.

The values are typical values for orientation, not measurements: Archie
parameters from common practice for unconsolidated (m ~ 1.3-1.6) and
consolidated (m ~ 2) sediments, K/U/Th from the compilation in
docs/gamma_ray_geant4.docx (IAEA-TECDOC-1363, Dickson & Scott 1997, Ellis &
Singer 2007).
"""
from __future__ import annotations

from typing import Dict, Optional

# ---------------------------------------------------------------------------
# Lithologies
# ---------------------------------------------------------------------------
# porosity phi (-), grain density (g/cm3), Archie a, m, n, surface conductivity
# sigma_s (S/m), K (%), U (ppm), Th (ppm), and the gamma-ray material preset
# (MATERIALS in gamma_geometry.py) that sets the element composition.
LITHOLOGIES: Dict[str, dict] = {
    "Gravel":      dict(porosity=0.25, grain_density=2.65, a=1.0, m=1.3, n=2.0, sigma_s=0.000,
                        K=0.4, U=0.6, Th=2.0, gamma_material="gravel"),
    "Sand":        dict(porosity=0.32, grain_density=2.65, a=1.0, m=1.3, n=2.0, sigma_s=0.000,
                        K=0.6, U=0.8, Th=2.5, gamma_material="sand"),
    "Silty sand":  dict(porosity=0.32, grain_density=2.66, a=1.0, m=1.5, n=2.0, sigma_s=0.003,
                        K=1.2, U=1.5, Th=6.0, gamma_material="sand"),
    "Silt":        dict(porosity=0.38, grain_density=2.68, a=1.0, m=1.6, n=2.0, sigma_s=0.010,
                        K=1.8, U=2.5, Th=8.0, gamma_material="sand"),
    "Clay":        dict(porosity=0.45, grain_density=2.70, a=1.0, m=2.0, n=2.0, sigma_s=0.080,
                        K=2.5, U=3.5, Th=12.0, gamma_material="clay"),
    "Glacial till": dict(porosity=0.20, grain_density=2.70, a=1.0, m=1.6, n=2.0, sigma_s=0.010,
                         K=1.8, U=2.0, Th=8.0, gamma_material="sand"),
    "Sandstone":   dict(porosity=0.18, grain_density=2.65, a=1.0, m=2.0, n=2.0, sigma_s=0.000,
                        K=0.3, U=0.6, Th=2.0, gamma_material="sand"),
    "Limestone":   dict(porosity=0.12, grain_density=2.71, a=1.0, m=2.0, n=2.0, sigma_s=0.000,
                        K=0.3, U=2.0, Th=1.5, gamma_material="limestone"),
}

# ---------------------------------------------------------------------------
# Pore waters
# ---------------------------------------------------------------------------
# resistivity rho_w (Ohm.m) at ~20 C, electrical conductivity for reference,
# density (g/cm3). Salinity barely changes the gamma ray (a few % in density).
WATERS: Dict[str, dict] = {
    "Fresh":    dict(rho_w=20.0, ec_uS_cm=500,   density=1.000, note="~0.3 g/L, drinking water"),
    "Brackish": dict(rho_w=2.0,  ec_uS_cm=5000,  density=1.003, note="~3 g/L"),
    "Salt":     dict(rho_w=0.2,  ec_uS_cm=50000, density=1.025, note="seawater, 35 g/L"),
}

# ---------------------------------------------------------------------------
# Borehole fills
# ---------------------------------------------------------------------------
# resistivity (Ohm.m) for ERT/EM; for the gamma ray the material preset and
# K, U, Th per bulk mass. "Filter sand" resistivity depends on its pore water
# and is computed with Archie's law from the chosen water.
BACKFILLS: Dict[str, dict] = {
    "Water":           dict(resistivity=None, archie=None, gamma_material="water",
                            K=0.0, U=0.0, Th=0.0, note="resistivity = that of the chosen water"),
    "Filter sand":     dict(resistivity=None, archie=dict(porosity=0.35, a=1.0, m=1.3, n=2.0, sigma_s=0.0),
                            gamma_material="sand_pack", K=0.3, U=0.5, Th=1.5,
                            note="water-saturated quartz sand; Archie with the chosen water"),
    "Bentonite grout": dict(resistivity=3.0, archie=None, gamma_material="bentonite",
                            K=0.3, U=2.0, Th=8.0, note="~1.4 g/cm3; conductive and thorium-rich"),
    "Cement grout":    dict(resistivity=20.0, archie=None, gamma_material="cement",
                            K=0.35, U=2.0, Th=4.0, note="~1.85 g/cm3; resistivity rises as it cures"),
}


def archie_resistivity(rho_w: float, porosity: float, a: float = 1.0, m: float = 2.0,
                       n: float = 2.0, sw: float = 1.0, sigma_s: float = 0.0) -> float:
    """Bulk resistivity (Ohm.m) from Archie's law plus surface conduction.

    sigma = Sw**n * phi**m / (a * rho_w) + sigma_s ;  rho = 1 / sigma
    """
    if rho_w <= 0 or not (0 < porosity <= 1) or not (0 < sw <= 1) or a <= 0:
        raise ValueError("need rho_w > 0, 0 < porosity <= 1, 0 < Sw <= 1 and a > 0")
    sigma = sw ** n * porosity ** m / (a * rho_w) + max(sigma_s, 0.0)
    return 1.0 / sigma


def bulk_density(porosity: float, grain_density: float, sw: float = 1.0,
                 water_density: float = 1.0) -> float:
    """Bulk density (g/cm3): grains plus the water-filled part of the pores (air neglected)."""
    return (1.0 - porosity) * grain_density + porosity * sw * water_density


def gamma_api(K: float, U: float, Th: float) -> float:
    """Standard API estimate 16 K + 8 U + 4 Th."""
    return 16.0 * K + 8.0 * U + 4.0 * Th


def backfill_resistivity(name: str, rho_w: float) -> float:
    """Default resistivity of a borehole fill (Ohm.m) for a given water resistivity."""
    bf = BACKFILLS[name]
    if bf["resistivity"] is not None:
        return bf["resistivity"]
    if bf["archie"] is not None:
        return archie_resistivity(rho_w, **bf["archie"])
    return rho_w


def unit_properties(lithology: str, water: str, sw: float = 1.0,
                    rho_w: Optional[float] = None) -> dict:
    """Database values of one unit, with the derived bulk resistivity and density."""
    lit, wat = LITHOLOGIES[lithology], WATERS[water]
    rho_w = wat["rho_w"] if rho_w is None else rho_w
    props = dict(lit, rho_w=rho_w, sw=sw)
    props["resistivity"] = archie_resistivity(rho_w, lit["porosity"], lit["a"], lit["m"], lit["n"],
                                              sw, lit["sigma_s"])
    props["density"] = bulk_density(lit["porosity"], lit["grain_density"], sw, wat["density"])
    props["api"] = gamma_api(lit["K"], lit["U"], lit["Th"])
    return props
