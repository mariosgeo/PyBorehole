"""
run_combined.py
===============
Runs both the ERT (Geometry / PyGIMLi) and EM (EMGeometry / SimPEG)
forward simulations over the same 1-D layered earth and borehole model,
with and without the borehole, for a sweep of backfill resistivities.

Outputs (in ``OUTPUT_DIR``, default ``results/``)
-------------------------------------------------
* ``combined_log_backfill_<rho>.png`` -- apparent-resistivity logs (ERT Wenner
  + EM coils) for every backfill resistivity in the sweep, with the
  no-borehole reference curves.
* ``backfill_effect.png`` -- borehole effect rho_a(with hole) / rho_a(no hole)
  versus depth, one panel per tool, one curve per backfill resistivity.

Usage
-----
    conda run -n pgy python run_combined.py

Adjust the SETTINGS block to change the model or acquisition geometry.
"""

import warnings
warnings.filterwarnings("ignore")

# Ensure tetgen.exe (bundled inside the conda environment) is on the PATH so
# PyGIMLi's make_mesh() can call it.  Safe to run even if already on PATH.
import os, sys
for _bin in (os.path.join(sys.prefix, "Library", "bin"), os.path.join(sys.prefix, "bin")):
    if os.path.isdir(_bin) and _bin not in os.environ.get("PATH", ""):
        os.environ["PATH"] = _bin + os.pathsep + os.environ.get("PATH", "")

# Force UTF-8 on Windows consoles (cp1252 cannot encode σ, Ω, em-dash, etc.)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from matplotlib.ticker import NullFormatter, ScalarFormatter
from matplotlib.lines import Line2D

from geometry_class import Geometry, EMGeometry

# ---------------------------------------------------------------------------
# SETTINGS  -- edit here
# ---------------------------------------------------------------------------
LAYERS = np.array([          # [top_depth, bottom_depth, resistivity]; last layer extends down
    [0.0,   5.0,  100.0],
    [5.0,  10.0,   10.0],
    [10.0, 15.0,  200.0],
])

BOREHOLE_LENGTH = 15.0
BOREHOLE_DIAM   = 0.20
BACKFILL_RES    = 10.0                       # always included in the sweep
BACKFILL_SWEEP  = [1.0, 10.0, 100.0, 1000.0]  # one combined_log_backfill_<rho>.png each

ERT_SHORT_SPACING = 0.5
ERT_LONG_SPACING  = 1.0
ERT_MEAS_SPACING  = 0.1

EM_COIL_SPACINGS  = [0.5, 1.0]
EM_FREQUENCIES    = [400.0, 8000.0]
EM_MEAS_SPACING   = 0.1
EM_MESH_TYPE      = "cylindrical"

OUTPUT_DIR        = "results"               # all figures are written here
# ---------------------------------------------------------------------------

sweep = sorted(set(BACKFILL_SWEEP) | {BACKFILL_RES})
os.makedirs(OUTPUT_DIR, exist_ok=True)


def ert_table(sim) -> pd.DataFrame:
    """Depth (array midpoint, positive down), Wenner spacing a and rho_a."""
    z = np.array([[sim.sensorPosition(int(i))[2] for i in sim(k)] for k in "abmn"])
    zs = np.sort(z, axis=0)
    return pd.DataFrame({
        "depth": -z.mean(axis=0),
        "a": np.round(zs[1] - zs[0], 3),
        "rhoa": np.array(sim("rhoa")),
    })


# ===========================================================================
# 1.  ERT simulation (one mesh, all backfill values)
# ===========================================================================
print("=" * 60)
print("Running ERT simulation (PyGIMLi / Geometry)...")
print("=" * 60)

geo = Geometry(
    borehole_length=BOREHOLE_LENGTH,
    borehole_diameter=BOREHOLE_DIAM,
    layer_1d_geometry=LAYERS.copy(),
    backfill_resistivity=BACKFILL_RES,
    long_spacing=ERT_LONG_SPACING,
)
geom_plc = geo.make_basic_geometry(vtk_filename=None)
ert_proto = geo.make_array(
    geom=geom_plc,
    abmn_order=(1, 4, 2, 3),
    short_spacing=ERT_SHORT_SPACING,
    long_spacing=ERT_LONG_SPACING,
    measuring_spacing=ERT_MEAS_SPACING,
    protocol_filename=None,
)
print(f"  ERT protocol: {ert_proto.size()} measurements")
ert_mesh = geo.make_mesh(geom_plc, vtk_filename=None)
print(f"  ERT mesh: {ert_mesh}")

ert_ref = ert_table(geo.run(ert_mesh, ert_proto, output_filename=None, include_borehole=False))
ert_res = {}
for rho_bf in sweep:
    geo.set_borehole(backfill_resistivity=rho_bf)
    ert_res[rho_bf] = ert_table(geo.run(ert_mesh, ert_proto, output_filename=None))
    print(f"  ERT done for backfill {rho_bf:g} Ohm.m")


# ===========================================================================
# 2.  EM simulation
# ===========================================================================
print()
print("=" * 60)
print("Running EM simulation (SimPEG / EMGeometry)...")
print("=" * 60)


def make_em(rho_bf):
    return EMGeometry(
        borehole_length=BOREHOLE_LENGTH,
        borehole_diameter=BOREHOLE_DIAM,
        layer_1d_geometry=LAYERS.copy(),
        backfill_resistivity=rho_bf,
        frequencies=EM_FREQUENCIES,
        coil_spacings=EM_COIL_SPACINGS,
        coil_orientation="z",
        measuring_spacing=EM_MEAS_SPACING,
        mesh_type=EM_MESH_TYPE,
    )


em_ref = make_em(BACKFILL_RES).run(verbose=True, include_borehole=False)
em_res = {rho_bf: make_em(rho_bf).run(verbose=False) for rho_bf in sweep}
print(f"  EM done for backfill {', '.join(f'{r:g}' for r in sweep)} Ohm.m")


# ===========================================================================
# 3.  Combined log plot, one figure per backfill value
# ===========================================================================
def plot_combined(rho_bf, filename):
    """Model + ERT/EM logs for one backfill, with the no-borehole reference."""
    fig, (ax_model, ax_log) = plt.subplots(
        1, 2, figsize=(13, 8), sharey=True,
        gridspec_kw={"width_ratios": [1, 2.5]},
    )
    fig.suptitle(
        "Borehole Survey -- Apparent Resistivity Log\n"
        f"ERT (Wenner) vs EM (coaxial coils), backfill {rho_bf:g} Ohm.m",
        fontsize=13, fontweight="bold",
    )

    all_res = LAYERS[:, 2].tolist() + [rho_bf]
    norm_model = mcolors.LogNorm(vmin=min(all_res), vmax=max(all_res) * (1.0001 if min(all_res) == max(all_res) else 1))
    cmap_model = cm.RdYlBu_r

    # Panel 1: model
    ax_model.set_title("1-D Resistivity Model", fontsize=11)
    width = 3.0
    for i, (top, bot, res) in enumerate(LAYERS):
        display_bot = bot if i < len(LAYERS) - 1 else BOREHOLE_LENGTH * 1.05
        rect = patches.Rectangle((-width/2, -display_bot), width, display_bot - top,
                                  facecolor=cmap_model(norm_model(res)), edgecolor="k", lw=0.8)
        ax_model.add_patch(rect)
        ax_model.text(0.6, -(top + display_bot)/2, f"{res:.0f} Ohm.m",
                      ha="center", va="center", fontsize=9,
                      bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", boxstyle="round,pad=0.2"))
    bh = patches.Rectangle((-BOREHOLE_DIAM/2, -BOREHOLE_LENGTH), BOREHOLE_DIAM, BOREHOLE_LENGTH,
                            facecolor=cmap_model(norm_model(rho_bf)), edgecolor="k", lw=1.2, hatch="////")
    ax_model.add_patch(bh)
    ax_model.set_xlim(-width/2, width/2)
    ax_model.set_ylim(-BOREHOLE_LENGTH * 1.08, 0.3)
    ax_model.set_xlabel("x (m)  [borehole not to scale]", fontsize=10)
    ax_model.set_ylabel("Depth (m)", fontsize=10)
    ax_model.tick_params(labelsize=9)
    sm = cm.ScalarMappable(cmap=cmap_model, norm=norm_model)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax_model, fraction=0.06, pad=0.04)
    cbar.set_label("True Resistivity (Ohm.m)", fontsize=9)

    # Panel 2: log -- solid = with borehole, thin dotted = no-borehole reference
    ax_log.set_title("Apparent Resistivity vs Depth (dotted: no borehole)", fontsize=11)
    ert_colors = ["#1f77b4", "#ff7f0e"]
    em_colors  = [cm.Set2(i / max(len(EM_FREQUENCIES) - 1, 1)) for i in range(len(EM_FREQUENCIES))]
    em_markers = dict(zip(EM_COIL_SPACINGS, ["o", "s", "^", "D"]))
    em_lstyles = dict(zip(EM_COIL_SPACINGS, ["-", "--", "-.", ":"]))
    legend_handles = []

    ert_main = ert_res[rho_bf]
    for idx, a in enumerate(np.unique(ert_main["a"])):
        color, marker = ert_colors[idx % 2], "o" if idx == 0 else "s"
        d = ert_main[ert_main["a"] == a].sort_values("depth")
        d0 = ert_ref[ert_ref["a"] == a].sort_values("depth")
        ax_log.plot(d["rhoa"], -d["depth"], marker=marker, ls="-", lw=1.8, color=color, ms=3)
        ax_log.plot(d0["rhoa"], -d0["depth"], ls=":", lw=1.2, color=color)
        legend_handles.append(Line2D([], [], marker=marker, color=color, lw=1.8, ms=5,
                                     label=f"ERT Wenner  a={a:.2g} m"))

    em_main = em_res[rho_bf]
    for fi, freq in enumerate(EM_FREQUENCIES):
        for spacing in EM_COIL_SPACINGS:
            sel = lambda df: df[(df["frequency"] == freq) & (df["spacing"] == spacing)].sort_values("depth")
            d, d0 = sel(em_main), sel(em_ref)
            if d.empty:
                continue
            color, mk, ls = em_colors[fi], em_markers.get(spacing, "x"), em_lstyles.get(spacing, "-")
            ax_log.plot(d["apparent_resistivity"], -d["depth"], marker=mk, ls=ls, lw=1.5, color=color,
                        ms=3, alpha=0.85, markevery=5)
            ax_log.plot(d0["apparent_resistivity"], -d0["depth"], ls=":", lw=1.2, color=color)
            legend_handles.append(Line2D([], [], marker=mk, color=color, lw=1.5, ms=5, ls=ls,
                                         label=f"EM  f={freq:.0f} Hz  s={spacing:.2g} m"))

    true_z, true_res = [], []
    for i, (top, bot, res) in enumerate(LAYERS):
        if i == len(LAYERS) - 1:
            bot = BOREHOLE_LENGTH * 1.05
        true_z.extend([-top, -bot])
        true_res.extend([res, res])
    ax_log.plot(true_res, true_z, "k--", lw=2.2)
    legend_handles.append(Line2D([], [], color="k", ls="--", lw=2.2, label="True 1-D model"))
    legend_handles.append(Line2D([], [], color="gray", ls=":", lw=1.2, label="No-borehole reference"))

    ax_log.set_xscale("log")
    ax_log.set_xlabel("Apparent Resistivity (Ohm.m)", fontsize=10)
    ax_log.tick_params(labelsize=9)
    ax_log.legend(handles=legend_handles, fontsize=8.5, loc="lower right", framealpha=0.9)
    ax_log.grid(True, which="both", ls="--", alpha=0.45)

    for _, bot, _ in LAYERS[:-1]:
        ax_log.axhline(-bot, color="gray", ls=":", lw=0.9)
        ax_model.axhline(-bot, color="gray", ls=":", lw=0.9)

    plt.tight_layout()
    fig.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {filename}")


print()
print("Plotting combined results...")
for rho_bf in sweep:
    plot_combined(rho_bf, os.path.join(OUTPUT_DIR, f"combined_log_backfill_{rho_bf:g}.png"))


# ===========================================================================
# 4.  Borehole-effect plot: rho_a(with hole) / rho_a(no hole) per backfill
# ===========================================================================
tools = [("ERT", a, None) for a in np.unique(ert_ref["a"])] + \
        [("EM", s, f) for s in EM_COIL_SPACINGS for f in EM_FREQUENCIES[:1]]
fig2, axes = plt.subplots(1, len(tools), figsize=(3.2 * len(tools), 7), sharey=True)
fig2.suptitle(f"Borehole effect: rho_a(with hole) / rho_a(no hole),  D = {BOREHOLE_DIAM:g} m",
              fontsize=12, fontweight="bold")
bf_colors = cm.viridis(np.linspace(0, 0.9, len(sweep)))

for ax, (kind, size, freq) in zip(np.atleast_1d(axes), tools):
    for c, rho_bf in zip(bf_colors, sweep):
        if kind == "ERT":
            d = ert_res[rho_bf][ert_res[rho_bf]["a"] == size].sort_values("depth")
            d0 = ert_ref[ert_ref["a"] == size].sort_values("depth")
            ratio, depth = d["rhoa"].values / d0["rhoa"].values, d["depth"].values
        else:
            sel = lambda df: df[(df["frequency"] == freq) & (df["spacing"] == size)].sort_values("depth")
            d, d0 = sel(em_res[rho_bf]), sel(em_ref)
            ratio = d["apparent_resistivity"].values / d0["apparent_resistivity"].values
            depth = d["depth"].values
        ax.plot(ratio, -depth, color=c, lw=1.6, label=f"{rho_bf:g} Ohm.m")
    ax.axvline(1.0, color="k", lw=0.8)
    for _, bot, _ in LAYERS[:-1]:
        ax.axhline(-bot, color="gray", ls=":", lw=0.9)
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_title(f"ERT Wenner a={size:g} m" if kind == "ERT" else f"EM s={size:g} m, {freq:.0f} Hz",
                 fontsize=10)
    ax.set_xlabel("ratio", fontsize=9)
    ax.grid(True, which="both", ls="--", alpha=0.45)
np.atleast_1d(axes)[0].set_ylabel("Depth (m)")
np.atleast_1d(axes)[0].legend(title="Backfill", fontsize=8, loc="lower left")
plt.tight_layout()
_effect_png = os.path.join(OUTPUT_DIR, "backfill_effect.png")
plt.savefig(_effect_png, dpi=150, bbox_inches="tight")
print(f"Saved: {_effect_png}")

# Summary table at mid-depth of every layer
print("\nBorehole effect (ratio with/without hole) at layer mid-depths:")
rows = []
for top, bot, res in LAYERS:
    zmid = 0.5 * (top + bot)
    for rho_bf in sweep:
        row = {"layer_rho": res, "depth": zmid, "backfill": rho_bf}
        for a in np.unique(ert_ref["a"]):
            d, d0 = ert_res[rho_bf], ert_ref
            i = (d["a"] == a) & np.isclose(d["depth"], zmid, atol=ERT_MEAS_SPACING / 2 + 1e-6)
            i0 = (d0["a"] == a) & np.isclose(d0["depth"], zmid, atol=ERT_MEAS_SPACING / 2 + 1e-6)
            row[f"ERT a={a:g}"] = d.loc[i, "rhoa"].mean() / d0.loc[i0, "rhoa"].mean()
        for s in EM_COIL_SPACINGS:
            f = EM_FREQUENCIES[0]
            sel = lambda df: df[(df["frequency"] == f) & (df["spacing"] == s)
                                & np.isclose(df["depth"], zmid, atol=EM_MEAS_SPACING / 2 + 1e-6)]
            row[f"EM s={s:g}"] = sel(em_res[rho_bf])["apparent_resistivity"].mean() / \
                sel(em_ref)["apparent_resistivity"].mean()
        rows.append(row)
print(pd.DataFrame(rows).round(3).to_string(index=False))

plt.show()
