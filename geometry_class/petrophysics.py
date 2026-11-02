# -*- coding: utf-8 -*-
"""
petrophysics.py
===============

A small built-in database of lithologies, pore waters and borehole fills, and
the rock-physics relations that turn them into the inputs of the simulators:
bulk resistivity (ERT, EM), and material, bulk density and K, U, Th (gamma
ray).

Bulk resistivity follows the Waxman-Smits (1968) model for shaly sediments:

    sigma_bulk = (phi**m * Sw**n / a) * (sigma_w + B * Qv / Sw)
    rho_bulk   = 1 / sigma_bulk

    Qv = CEC/100 * rho_grain * (1 - phi) / phi        [meq/cm3 of pore volume]
    B  = 4.6 * (1 - 0.6 * exp(-0.77 / rho_w))         [(S/m) per (meq/cm3), 25 C]

with sigma_w = 1/rho_w the pore-water conductivity, CEC the cation-exchange
capacity of the solids (meq/100 g), Qv the exchange cations per unit pore
volume and B their equivalent conductance (Waxman & Smits 1968; the B fit is
theirs, at 25 C). Clay minerals conduct along their surfaces through these
counter-ions, in parallel with the pore water; Archie's law ignores it and
overestimates the resistivity of clays several-fold in fresh water (a clay
with 45 % porosity and 20 Ohm.m water would be ~100 Ohm.m instead of the
~5-20 Ohm.m measured). Unlike a constant surface conductivity, the clay term
grows with salinity through B and with decreasing saturation through 1/Sw.
CEC = 0 gives Archie's law exactly (clean sand, gravel, sandstone, limestone).

The values are typical values for orientation, not measurements: exponents
from common practice for unconsolidated (m ~ 1.3-1.6) and consolidated
(m ~ 2) sediments, CEC typical of the clay content (kaolinite 3-15, illite
10-40, smectite 80-150 meq/100 g as pure minerals), K/U/Th from the
compilation in docs/gamma_ray_geant4.docx (IAEA-TECDOC-1363, Dickson & Scott
1997, Ellis & Singer 2007).
"""
from __future__ import annotations

import math
from typing import Dict, Optional

# ---------------------------------------------------------------------------
# Lithologies
# ---------------------------------------------------------------------------
# porosity phi (-), grain density (g/cm3), Waxman-Smits a, m, n (= Archie's
# for clean rock), cation-exchange capacity cec (meq/100 g of solids), K (%),
# U (ppm), Th (ppm), and the gamma-ray material preset (MATERIALS in
# gamma_geometry.py) that sets the element composition.
LITHOLOGIES: Dict[str, dict] = {
    # -- gravels
    "Gravel":                      dict(porosity=0.25, grain_density=2.65, a=1.0, m=1.3, n=2.0, cec=0.0,
                                        K=0.4, U=0.6, Th=2.0, gamma_material="gravel"),
    "Gravel, sandy":               dict(porosity=0.22, grain_density=2.65, a=1.0, m=1.3, n=2.0, cec=0.0,
                                        K=0.5, U=0.7, Th=2.3, gamma_material="gravel"),
    "Gravel, clayey":              dict(porosity=0.20, grain_density=2.66, a=1.0, m=1.5, n=2.0, cec=1.5,
                                        K=1.0, U=1.2, Th=4.0, gamma_material="gravel"),
    # -- sands
    "Sand":                        dict(porosity=0.32, grain_density=2.65, a=1.0, m=1.3, n=2.0, cec=0.0,
                                        K=0.6, U=0.8, Th=2.5, gamma_material="sand"),
    "Sand, coarse":                dict(porosity=0.30, grain_density=2.65, a=1.0, m=1.3, n=2.0, cec=0.0,
                                        K=0.4, U=0.6, Th=2.0, gamma_material="sand"),
    "Sand, medium":                dict(porosity=0.34, grain_density=2.65, a=1.0, m=1.3, n=2.0, cec=0.0,
                                        K=0.5, U=0.7, Th=2.2, gamma_material="sand"),
    "Sand, fine":                  dict(porosity=0.38, grain_density=2.65, a=1.0, m=1.4, n=2.0, cec=0.2,
                                        K=0.7, U=1.0, Th=3.0, gamma_material="sand"),
    "Silty sand":                  dict(porosity=0.32, grain_density=2.66, a=1.0, m=1.5, n=2.0, cec=0.5,
                                        K=1.2, U=1.5, Th=6.0, gamma_material="sand"),
    "Sand, clayey":                dict(porosity=0.30, grain_density=2.66, a=1.0, m=1.6, n=2.0, cec=2.0,
                                        K=1.5, U=1.8, Th=7.0, gamma_material="sand"),
    "Sand, glauconitic":           dict(porosity=0.32, grain_density=2.70, a=1.0, m=1.6, n=2.0, cec=3.0,
                                        K=2.5, U=1.5, Th=4.0, gamma_material="sand"),
    "Sand, arkosic":               dict(porosity=0.30, grain_density=2.62, a=1.0, m=1.4, n=2.0, cec=0.3,
                                        K=2.5, U=1.5, Th=5.0, gamma_material="arkose"),
    "Sand, heavy-mineral":         dict(porosity=0.30, grain_density=2.90, a=1.0, m=1.3, n=2.0, cec=0.0,
                                        K=0.3, U=6.0, Th=50.0, gamma_material="sand"),
    # -- silts
    "Silt":                        dict(porosity=0.38, grain_density=2.68, a=1.0, m=1.6, n=2.0, cec=2.0,
                                        K=1.8, U=2.5, Th=8.0, gamma_material="sand"),
    "Silt, clayey":                dict(porosity=0.42, grain_density=2.69, a=1.0, m=1.8, n=2.0, cec=5.0,
                                        K=2.0, U=2.8, Th=9.5, gamma_material="silt"),
    "Loess":                       dict(porosity=0.45, grain_density=2.68, a=1.0, m=1.6, n=2.0, cec=4.0,
                                        K=1.8, U=2.5, Th=9.0, gamma_material="silt"),
    # -- clays
    "Clay":                        dict(porosity=0.45, grain_density=2.70, a=1.0, m=2.0, n=2.0, cec=10.0,
                                        K=2.5, U=3.5, Th=12.0, gamma_material="clay"),
    "Clay, kaolinitic":            dict(porosity=0.42, grain_density=2.62, a=1.0, m=2.0, n=2.0, cec=5.0,
                                        K=0.8, U=2.5, Th=12.0, gamma_material="clay"),
    "Clay, illitic":               dict(porosity=0.45, grain_density=2.75, a=1.0, m=2.0, n=2.0, cec=20.0,
                                        K=3.5, U=3.0, Th=13.0, gamma_material="clay"),
    "Clay, smectitic (bentonitic)":dict(porosity=0.55, grain_density=2.60, a=1.0, m=2.2, n=2.0, cec=60.0,
                                        K=0.8, U=3.5, Th=16.0, gamma_material="clay"),
    "Clay, marine":                dict(porosity=0.50, grain_density=2.70, a=1.0, m=2.0, n=2.0, cec=15.0,
                                        K=2.2, U=2.5, Th=10.0, gamma_material="clay"),
    "Clay, lacustrine":            dict(porosity=0.48, grain_density=2.70, a=1.0, m=2.0, n=2.0, cec=12.0,
                                        K=2.2, U=2.5, Th=10.0, gamma_material="clay"),
    "Clay, organic":               dict(porosity=0.60, grain_density=2.50, a=1.0, m=2.0, n=2.0, cec=25.0,
                                        K=1.5, U=4.0, Th=9.0, gamma_material="clay"),
    # -- tills and organic soils
    "Glacial till":                dict(porosity=0.20, grain_density=2.70, a=1.0, m=1.6, n=2.0, cec=2.0,
                                        K=1.8, U=2.0, Th=8.0, gamma_material="sand"),
    "Till, sandy":                 dict(porosity=0.22, grain_density=2.68, a=1.0, m=1.5, n=2.0, cec=1.0,
                                        K=1.5, U=1.8, Th=6.5, gamma_material="sand"),
    "Boulder clay":                dict(porosity=0.25, grain_density=2.70, a=1.0, m=1.8, n=2.0, cec=5.0,
                                        K=2.0, U=2.2, Th=9.0, gamma_material="clay"),
    "Peat":                        dict(porosity=0.85, grain_density=1.50, a=1.0, m=1.5, n=2.0, cec=15.0,
                                        K=0.1, U=1.0, Th=0.5, gamma_material="peat"),
    # -- sedimentary rocks
    "Sandstone":                   dict(porosity=0.18, grain_density=2.65, a=1.0, m=2.0, n=2.0, cec=0.0,
                                        K=0.3, U=0.6, Th=2.0, gamma_material="sand"),
    "Sandstone, shaly":            dict(porosity=0.15, grain_density=2.66, a=1.0, m=2.0, n=2.0, cec=3.0,
                                        K=1.2, U=1.5, Th=6.0, gamma_material="sand"),
    "Siltstone":                   dict(porosity=0.12, grain_density=2.68, a=1.0, m=2.0, n=2.0, cec=4.0,
                                        K=1.8, U=2.5, Th=8.0, gamma_material="silt"),
    "Mudstone":                    dict(porosity=0.15, grain_density=2.70, a=1.0, m=2.0, n=2.0, cec=10.0,
                                        K=2.0, U=3.0, Th=10.0, gamma_material="shale"),
    "Shale":                       dict(porosity=0.12, grain_density=2.72, a=1.0, m=2.0, n=2.0, cec=15.0,
                                        K=2.7, U=3.7, Th=12.0, gamma_material="shale"),
    "Shale, black (organic-rich)": dict(porosity=0.10, grain_density=2.60, a=1.0, m=2.0, n=2.0, cec=15.0,
                                        K=2.5, U=20.0, Th=10.0, gamma_material="shale"),
    "Marl":                        dict(porosity=0.25, grain_density=2.70, a=1.0, m=2.0, n=2.0, cec=8.0,
                                        K=1.2, U=2.5, Th=5.0, gamma_material="marl"),
    "Limestone":                   dict(porosity=0.12, grain_density=2.71, a=1.0, m=2.0, n=2.0, cec=0.0,
                                        K=0.3, U=2.0, Th=1.5, gamma_material="limestone"),
    "Chalk":                       dict(porosity=0.35, grain_density=2.71, a=1.0, m=2.0, n=2.0, cec=0.5,
                                        K=0.1, U=1.5, Th=0.8, gamma_material="chalk"),
    "Dolomite":                    dict(porosity=0.10, grain_density=2.84, a=1.0, m=2.0, n=2.0, cec=0.0,
                                        K=0.2, U=1.5, Th=0.8, gamma_material="dolomite"),
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
# and is computed with Waxman-Smits (clean sand: CEC = 0, i.e. Archie) from the
# chosen water.
BACKFILLS: Dict[str, dict] = {
    "Water":           dict(resistivity=None, ws=None, gamma_material="water",
                            K=0.0, U=0.0, Th=0.0, note="resistivity = that of the chosen water"),
    "Filter sand":     dict(resistivity=None, ws=dict(porosity=0.35, a=1.0, m=1.3, n=2.0, cec=0.0),
                            gamma_material="sand_pack", K=0.3, U=0.5, Th=1.5,
                            note="water-saturated quartz sand; Waxman-Smits (CEC 0 = Archie) with the chosen water"),
    "Bentonite grout": dict(resistivity=3.0, ws=None, gamma_material="bentonite",
                            K=0.3, U=2.0, Th=8.0, note="~1.4 g/cm3; conductive and thorium-rich"),
    "Cement grout":    dict(resistivity=20.0, ws=None, gamma_material="cement",
                            K=0.35, U=2.0, Th=4.0, note="~1.85 g/cm3; resistivity rises as it cures"),
}


def _check(rho_w: float, porosity: float, sw: float, a: float) -> None:
    if rho_w <= 0 or not (0 < porosity <= 1) or not (0 < sw <= 1) or a <= 0:
        raise ValueError("need rho_w > 0, 0 < porosity <= 1, 0 < Sw <= 1 and a > 0")


def archie_resistivity(rho_w: float, porosity: float, a: float = 1.0, m: float = 2.0,
                       n: float = 2.0, sw: float = 1.0, sigma_s: float = 0.0) -> float:
    """Bulk resistivity (Ohm.m) from Archie's law, optionally with a constant
    surface conductivity sigma_s (S/m) in parallel.

    sigma = Sw**n * phi**m / (a * rho_w) + sigma_s ;  rho = 1 / sigma

    Kept for clean rock and for compatibility; the database and the app use
    ``waxman_smits_resistivity``, which reduces to this with CEC = 0.
    """
    _check(rho_w, porosity, sw, a)
    sigma = sw ** n * porosity ** m / (a * rho_w) + max(sigma_s, 0.0)
    return 1.0 / sigma


def waxman_smits_b(rho_w: float) -> float:
    """Equivalent counter-ion conductance B ((S/m) per meq/cm3) at 25 C.

    B = 4.6 * (1 - 0.6 * exp(-0.77 / rho_w))  (Waxman & Smits 1968):
    about 4.6 in brines, falling to about 1.9 in fresh water.
    """
    if rho_w <= 0:
        raise ValueError("rho_w must be positive")
    return 4.6 * (1.0 - 0.6 * math.exp(-0.77 / rho_w))


def qv_from_cec(cec: float, porosity: float, grain_density: float = 2.65) -> float:
    """Exchange cations per unit pore volume Qv (meq/cm3) from the CEC (meq/100 g of solids).

    Qv = CEC/100 * rho_grain * (1 - phi) / phi
    """
    if cec < 0 or not (0 < porosity < 1) or grain_density <= 0:
        raise ValueError("need CEC >= 0, 0 < porosity < 1 and grain density > 0")
    return cec / 100.0 * grain_density * (1.0 - porosity) / porosity


def waxman_smits_resistivity(rho_w: float, porosity: float, a: float = 1.0, m: float = 2.0,
                             n: float = 2.0, sw: float = 1.0, cec: float = 0.0,
                             grain_density: float = 2.65, b: Optional[float] = None) -> float:
    """Bulk resistivity (Ohm.m) from the Waxman-Smits model.

    sigma = (phi**m * Sw**n / a) * (1/rho_w + B * Qv / Sw) ;  rho = 1 / sigma

    m and n are the Waxman-Smits exponents (m*, n*); Qv comes from ``cec``
    (``qv_from_cec``) and B from ``waxman_smits_b`` unless given. CEC = 0 is
    Archie's law exactly.
    """
    _check(rho_w, porosity, sw, a)
    if porosity >= 1:
        raise ValueError("porosity must be below 1 for Waxman-Smits")
    qv = qv_from_cec(cec, porosity, grain_density)
    b = waxman_smits_b(rho_w) if b is None else b
    sigma = porosity ** m * sw ** n / a * (1.0 / rho_w + b * qv / sw)
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
    if bf["ws"] is not None:
        return waxman_smits_resistivity(rho_w, **bf["ws"])
    return rho_w


def unit_properties(lithology: str, water: str, sw: float = 1.0,
                    rho_w: Optional[float] = None) -> dict:
    """Database values of one unit, with the derived bulk resistivity and density."""
    lit, wat = LITHOLOGIES[lithology], WATERS[water]
    rho_w = wat["rho_w"] if rho_w is None else rho_w
    props = dict(lit, rho_w=rho_w, sw=sw)
    props["qv"] = qv_from_cec(lit["cec"], lit["porosity"], lit["grain_density"])
    props["B"] = waxman_smits_b(rho_w)
    props["resistivity"] = waxman_smits_resistivity(rho_w, lit["porosity"], lit["a"], lit["m"], lit["n"],
                                                    sw, lit["cec"], lit["grain_density"])
    props["density"] = bulk_density(lit["porosity"], lit["grain_density"], sw, wat["density"])
    props["api"] = gamma_api(lit["K"], lit["U"], lit["Th"])
    return props
