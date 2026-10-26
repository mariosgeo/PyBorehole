# -*- coding: utf-8 -*-
"""
study.py
========

Run the ERT, EM and gamma-ray simulators for one layered model and return the
logs as tidy DataFrames, with and without the borehole. Used by the Streamlit
app (streamlit_app.py); the same calls as run_combined.py / run_realistic.py.

Every function returns a DataFrame with the columns

    tool    name of the curve (array / spacing / frequency)
    depth   tool reference depth (m)
    value   apparent resistivity (Ohm.m) or gamma ray (API)
    case    "borehole" or "no borehole"

plus, for the gamma ray, ``error`` (1-sigma Monte Carlo error, API).
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd


def _layers(units: pd.DataFrame, column: str = "resistivity") -> np.ndarray:
    """[top, bottom, resistivity] rows from a unit table (sorted by top)."""
    u = units.sort_values("top")
    return u[["top", "bottom", column]].to_numpy(dtype=float)


# ---------------------------------------------------------------------------
def run_ert(units: pd.DataFrame, borehole_length: float, borehole_diameter: float,
            backfill_resistivity: float, array: str = "wenner",
            spacings: Sequence[float] = (0.5, 1.0), measuring_spacing: float = 0.1,
            mn_spacing: float = 0.2, tool_diameter: Optional[float] = None,
            progress=None) -> pd.DataFrame:
    """3D ERT (PyGIMLi). ``array``: "wenner" (two spacings a), "normal" (AM
    spacings) or "lateral" (AO spacings, with ``mn_spacing``). One mesh, two
    solves (with and without the borehole)."""
    from . import Geometry, merge_protocols
    say = progress or (lambda msg: None)
    spacings = sorted(float(s) for s in spacings)
    long = max(spacings) + (mn_spacing / 2 if array == "lateral" else 0.0)
    geo = Geometry(borehole_length=borehole_length, borehole_diameter=borehole_diameter,
                   layer_1d_geometry=_layers(units), backfill_resistivity=backfill_resistivity,
                   long_spacing=long, tool_diameter=tool_diameter)
    plc = geo.make_basic_geometry(vtk_filename=None)

    if array == "wenner":
        if len(spacings) != 2:
            raise ValueError("The Wenner array needs exactly two spacings a (short and long).")
        proto = geo.make_array(plc, abmn_order=(1, 4, 2, 3), short_spacing=spacings[0],
                               long_spacing=spacings[1], measuring_spacing=measuring_spacing,
                               protocol_filename=None)

        def table(sim):
            z = np.array([[sim.sensorPosition(int(i))[2] for i in sim(k)] for k in "abmn"])
            zs = np.sort(z, axis=0)
            return pd.DataFrame({"tool": [f"ERT Wenner a={a:g} m" for a in np.round(zs[1] - zs[0], 3)],
                                 "depth": -z.mean(axis=0), "value": np.array(sim("rhoa"))})
    elif array in ("normal", "lateral"):
        proto = merge_protocols(geo.make_tool_array(plc, array, spacings, measuring_spacing,
                                                    mn_spacing=mn_spacing))
        label = "AM" if array == "normal" else "AO"

        def table(sim):
            return pd.DataFrame({"tool": [f"ERT {array} {label}={s:.2f} m" for s in np.array(sim["spacing"])],
                                 "depth": np.array(sim["depthref"]), "value": np.array(sim("rhoa"))})
    else:
        raise ValueError("array must be 'wenner', 'normal' or 'lateral'")

    say(f"ERT: meshing ({proto.size()} measurements)")
    mesh = geo.make_mesh(plc, vtk_filename=None)
    say(f"ERT: no-borehole reference ({mesh.cellCount():,} cells)")
    ref = table(geo.run(mesh, proto, output_filename=None, include_borehole=False))
    say("ERT: with the borehole")
    hole = table(geo.run(mesh, proto, output_filename=None))
    return pd.concat([hole.assign(case="borehole"), ref.assign(case="no borehole")], ignore_index=True)


# ---------------------------------------------------------------------------
def run_em(units: pd.DataFrame, borehole_length: float, borehole_diameter: float,
           backfill_resistivity: float, coil_spacings: Sequence[float] = (0.5, 1.0),
           frequencies: Sequence[float] = (400.0, 8000.0), measuring_spacing: float = 0.1,
           progress=None) -> pd.DataFrame:
    """Frequency-domain EM with coaxial coils (SimPEG, axisymmetric mesh)."""
    from . import EMGeometry
    say = progress or (lambda msg: None)

    def make():
        return EMGeometry(borehole_length=borehole_length, borehole_diameter=borehole_diameter,
                          layer_1d_geometry=_layers(units), backfill_resistivity=backfill_resistivity,
                          frequencies=list(frequencies), coil_spacings=list(coil_spacings),
                          coil_orientation="z", measuring_spacing=measuring_spacing,
                          mesh_type="cylindrical")

    def table(df):
        return pd.DataFrame({"tool": [f"EM s={r.spacing:g} m, {r.frequency:g} Hz" for r in df.itertuples()],
                             "depth": df["depth"].values, "value": df["apparent_resistivity"].values})

    say("EM: no-borehole reference")
    ref = table(make().run(verbose=False, include_borehole=False))
    say("EM: with the borehole")
    hole = table(make().run(verbose=False))
    return pd.concat([hole.assign(case="borehole"), ref.assign(case="no borehole")], ignore_index=True)


# ---------------------------------------------------------------------------
def run_gamma(units: pd.DataFrame, borehole_length: float, borehole_diameter: float,
              backfill: dict, measuring_spacing: float = 0.1, events: int = 150_000,
              crystal_diameter: float = 0.038, crystal_length: float = 0.102,
              tool_diameter: float = 0.050, progress=None) -> pd.DataFrame:
    """Natural gamma ray (Geant4). ``units`` needs columns gamma_material, K, U,
    Th and density; ``backfill`` is a dict (material, K, U, Th[, density])."""
    from .gamma_geometry import GammaGeometry, GammaMedium
    say = progress or (lambda msg: None)
    u = units.sort_values("top")
    layers = [(r.top, r.bottom, GammaMedium(r.gamma_material, K=r.K, U=r.U, Th=r.Th, density=r.density))
              for r in u.itertuples()]
    gr = GammaGeometry(layers=layers, borehole_length=borehole_length, borehole_diameter=borehole_diameter,
                       backfill=GammaMedium(**backfill), measuring_spacing=measuring_spacing, events=events,
                       crystal_diameter=crystal_diameter, crystal_length=crystal_length,
                       tool_diameter=tool_diameter)
    say("Gamma ray: calibration (infinite formation)")
    gr.calibrate(verbose=False)
    out = []
    for case, include in (("no borehole", False), ("borehole", True)):
        say(f"Gamma ray: {case}")
        log = gr.run(verbose=False, include_borehole=include)
        out.append(pd.DataFrame({"tool": "Gamma ray", "depth": log["depth"], "value": log["GR_api"],
                                 "error": log["GR_err"], "K": log["K_app"], "U": log["U_app"],
                                 "Th": log["Th_app"], "case": case}))
    return pd.concat(out, ignore_index=True)


def borehole_effect(results: pd.DataFrame) -> pd.DataFrame:
    """Ratio value(borehole) / value(no borehole) per tool and depth."""
    key = ["tool", "depth"]
    a = results[results["case"] == "borehole"].groupby(key)["value"].mean()
    b = results[results["case"] == "no borehole"].groupby(key)["value"].mean()
    r = (a / b).dropna().rename("ratio").reset_index()
    return r
