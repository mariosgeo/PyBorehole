"""
run_combined.py
===============
Runs both the ERT (Geometry / PyGIMLi) and EM (EMGeometry / SimPEG)
forward simulations over the same 1-D layered earth model and plots
their apparent-resistivity depth profiles on the same axes.

Usage
-----
    conda run -n pg2 python run_combined.py

Adjust the SETTINGS block to change the model or acquisition geometry.
"""

import warnings
warnings.filterwarnings("ignore")

# Ensure tetgen.exe (bundled inside the conda environment) is on the PATH so
# PyGIMLi's make_mesh() can call it.  Safe to run even if already on PATH.
import os, sys
_conda_lib_bin = os.path.join(sys.prefix, "Library", "bin")
if os.path.isdir(_conda_lib_bin) and _conda_lib_bin not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _conda_lib_bin + os.pathsep + os.environ.get("PATH", "")

# Force UTF-8 on Windows consoles (cp1252 cannot encode σ, Ω, em-dash, etc.)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")





import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from matplotlib.lines import Line2D

from geometry_class import Geometry, EMGeometry

# ---------------------------------------------------------------------------
# SETTINGS  -- edit here
# ---------------------------------------------------------------------------
LAYERS = np.array([
    [0.0,   5.0,  100.0],
    [5.0,  10.0,   10.0],
    [10.0, 15.0,  200.0],
])

BACKGROUND_RES  = 50.0
BOREHOLE_LENGTH = 15.0
BOREHOLE_DIAM   = 0.20

ERT_SHORT_SPACING = 0.5
ERT_LONG_SPACING  = 1.0
ERT_MEAS_SPACING  = 0.5

EM_COIL_SPACINGS  = [0.5, 1.0]
EM_FREQUENCIES    = [400.0, 8000.0]
EM_MEAS_SPACING   = 0.5
EM_CORE_CELL_SIZE = 0.05
EM_N_PADDING      = 8
EM_DOMAIN_RADIUS  = 50.0
EM_MESH_TYPE      = "cylindrical"
# ---------------------------------------------------------------------------


# ===========================================================================
# 1.  ERT simulation
# ===========================================================================
print("=" * 60)
print("Running ERT simulation (PyGIMLi / Geometry)...")
print("=" * 60)

geo = Geometry(
    borehole_length=BOREHOLE_LENGTH,
    borehole_diameter=BOREHOLE_DIAM,
    layer_1d_geometry=LAYERS.copy(),
    background_resistivity=BACKGROUND_RES,
    long_spacing=ERT_LONG_SPACING,
)

geom_plc = geo.make_basic_geometry()
ert_data_proto = geo.make_array(
    geom=geom_plc,
    abmn_order=(1, 4, 2, 3),
    short_spacing=ERT_SHORT_SPACING,
    long_spacing=ERT_LONG_SPACING,
    measuring_spacing=ERT_MEAS_SPACING,
)
print(f"  ERT protocol: {ert_data_proto.size()} measurements")

ert_mesh = geo.make_mesh(geom_plc)
print(f"  ERT mesh: {ert_mesh}")

ert_sim = geo.run(ert_mesh, ert_data_proto, noise_level=0.0)
print(f"  ERT done: {ert_sim.size()} data points")

z_A = np.array([ert_sim.sensorPosition(int(i))[2] for i in ert_sim("a")])
z_B = np.array([ert_sim.sensorPosition(int(i))[2] for i in ert_sim("b")])
z_M = np.array([ert_sim.sensorPosition(int(i))[2] for i in ert_sim("m")])
z_N = np.array([ert_sim.sensorPosition(int(i))[2] for i in ert_sim("n")])
z_center_ert = (z_A + z_B + z_M + z_N) / 4.0

a_spacings_ert = np.round(
    np.array([abs(sorted([z_A[i], z_B[i], z_M[i], z_N[i]])[1] -
                  sorted([z_A[i], z_B[i], z_M[i], z_N[i]])[0])
              for i in range(ert_sim.size())]), 2)

rhoa_ert = (np.array(ert_sim("rhoa")) if "rhoa" in ert_sim.tokenList()
            else np.array(ert_sim("r")) * np.array(ert_sim("k")))


# ===========================================================================
# 2.  EM simulation
# ===========================================================================
print()
print("=" * 60)
print("Running EM simulation (SimPEG / EMGeometry)...")
print("=" * 60)

em = EMGeometry(
    borehole_length=BOREHOLE_LENGTH,
    borehole_diameter=BOREHOLE_DIAM,
    layer_1d_geometry=LAYERS.copy(),
    background_resistivity=BACKGROUND_RES,
    frequencies=EM_FREQUENCIES,
    coil_spacings=EM_COIL_SPACINGS,
    coil_orientation="z",
    measuring_spacing=EM_MEAS_SPACING,
    core_cell_size=EM_CORE_CELL_SIZE,
    n_padding=EM_N_PADDING,
    domain_radius=EM_DOMAIN_RADIUS,
    mesh_type=EM_MESH_TYPE,
)

em_df = em.run(verbose=True)
print(em_df[["depth", "spacing", "frequency", "apparent_resistivity"]].to_string(index=False))


# ===========================================================================
# 3.  Combined plot
# ===========================================================================
print()
print("Plotting combined results...")

fig, (ax_model, ax_log) = plt.subplots(
    1, 2, figsize=(13, 8), sharey=True,
    gridspec_kw={"width_ratios": [1, 2.5]},
)
fig.suptitle(
    "Borehole Survey -- Apparent Resistivity Log\nERT (Wenner) vs EM (Induction Coil)",
    fontsize=13, fontweight="bold",
)

all_res = [BACKGROUND_RES] + LAYERS[:, 2].tolist()
vmin, vmax = min(all_res), max(all_res)
if vmin == vmax:
    vmin, vmax = vmin * 0.5, vmax * 2.0
norm_model = mcolors.LogNorm(vmin=vmin, vmax=vmax)
cmap_model = cm.RdYlBu_r

# Panel 1: model
ax_model.set_title("1-D Resistivity Model", fontsize=11)
width = 3.0
for i, (top, bot, res) in enumerate(LAYERS):
    display_bot = bot if i < len(LAYERS) - 1 else BOREHOLE_LENGTH * 1.05
    rect = patches.Rectangle((-width/2, -display_bot), width, display_bot - top,
                              facecolor=cmap_model(norm_model(res)), edgecolor="k", lw=0.8)
    ax_model.add_patch(rect)
    ax_model.text(0, -(top + display_bot)/2, f"{res:.0f} Ohm.m",
                  ha="center", va="center", fontsize=9,
                  bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", boxstyle="round,pad=0.2"))
bh = patches.Rectangle((-BOREHOLE_DIAM/2, -BOREHOLE_LENGTH), BOREHOLE_DIAM, BOREHOLE_LENGTH,
                        facecolor="white", edgecolor="k", lw=1.2, hatch="////")
ax_model.add_patch(bh)
ax_model.set_xlim(-width/2, width/2)
ax_model.set_ylim(-BOREHOLE_LENGTH * 1.08, 0.3)
ax_model.set_xlabel("x (m)", fontsize=10)
ax_model.set_ylabel("Depth (m)", fontsize=10)
ax_model.tick_params(labelsize=9)
sm = cm.ScalarMappable(cmap=cmap_model, norm=norm_model)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax_model, fraction=0.06, pad=0.04)
cbar.set_label("True Resistivity (Ohm.m)", fontsize=9)

# Panel 2: log
ax_log.set_title("Apparent Resistivity vs Depth", fontsize=11)
ert_colors = {0: "#1f77b4", 1: "#ff7f0e"}
em_colors  = [cm.Set2(i / max(len(EM_FREQUENCIES) - 1, 1)) for i in range(len(EM_FREQUENCIES))]
em_markers = dict(zip(EM_COIL_SPACINGS, ["o", "s", "^", "D"]))
em_lstyles = dict(zip(EM_COIL_SPACINGS, ["-", "--", "-.", ":"]))
legend_handles = []

unique_a = np.unique(a_spacings_ert)
for idx, a in enumerate(unique_a):
    mask = (a_spacings_ert == a)
    z_unique = np.unique(z_center_ert[mask])
    rhoa_vals = np.array([rhoa_ert[mask][np.isclose(z_center_ert[mask], zz)].mean()
                          for zz in z_unique])
    color  = ert_colors.get(idx, f"C{idx}")
    marker = "o" if idx == 0 else "s"
    ax_log.plot(rhoa_vals, z_unique, marker=marker, linestyle="-",
                lw=1.8, color=color, ms=5)
    legend_handles.append(Line2D([], [], marker=marker, color=color, lw=1.8, ms=6,
                                  label=f"ERT Wenner  a={a:.2g} m"))

for fi, freq in enumerate(EM_FREQUENCIES):
    df_f = em_df[em_df["frequency"] == freq]
    for spacing in EM_COIL_SPACINGS:
        df_fs = df_f[df_f["spacing"] == spacing].sort_values("depth")
        if df_fs.empty:
            continue
        z_em   = -df_fs["depth"].values
        rho_em = df_fs["apparent_resistivity"].values
        color  = em_colors[fi]
        mk     = em_markers.get(spacing, "x")
        ls     = em_lstyles.get(spacing, "-")
        ax_log.plot(rho_em, z_em, marker=mk, ls=ls, lw=1.5, color=color, ms=4, alpha=0.85)
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

ax_log.set_xscale("log")
ax_log.set_xlabel("Apparent Resistivity (Ohm.m)", fontsize=10)
ax_log.tick_params(labelsize=9)
ax_log.legend(handles=legend_handles, fontsize=8.5, loc="lower right", framealpha=0.9)
ax_log.grid(True, which="both", ls="--", alpha=0.45)

for _, bot, _ in LAYERS[:-1]:
    ax_log.axhline(-bot, color="gray", ls=":", lw=0.9)
    ax_model.axhline(-bot, color="gray", ls=":", lw=0.9)

plt.tight_layout()
plt.savefig("combined_log.png", dpi=150, bbox_inches="tight")
print("Saved: combined_log.png")
plt.show()
