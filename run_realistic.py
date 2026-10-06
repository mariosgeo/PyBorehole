"""
run_realistic.py
================
Backfill study with realistic logging tools (compare with run_combined.py,
which uses idealised tools: point electrodes on the axis, a Wenner array
and plain two-coil EM).

* ERT sonde: an insulating tool body (mandrel) in the middle of the hole,
  electrodes on its surface, and the classic logging arrays:
    - normal  (A-M on the sonde, B and N remote): 16" and 64" by default,
    - lateral (A-M-N on the sonde, B remote).
  The tool constant (geometric factor) is calibrated like a real tool: the
  same sonde in a homogeneous earth reads its true resistivity.
* EM tool: coaxial coils, plain two-coil pair and a focused version with a
  bucking receiver that cancels the direct field (and, to first order, the
  borehole signal near the axis).

Outputs (in OUTPUT_DIR, default results/)
-----------------------------------------
* ``realistic_log_backfill_<rho>.png`` -- logs of all tools for every backfill
  resistivity, with the no-borehole reference curves.
* ``realistic_backfill_effect.png`` -- rho_a(with hole) / rho_a(no hole) per tool.

Usage
-----
    conda run -n pgy python run_realistic.py

Adjust the SETTINGS block to describe your tools. The example spacings are
the classic electric-log values; replace them with your sonde's.
"""

import warnings
warnings.filterwarnings("ignore")

import os, sys
for _bin in (os.path.join(sys.prefix, "Library", "bin"), os.path.join(sys.prefix, "bin")):
    if os.path.isdir(_bin) and _bin not in os.environ.get("PATH", ""):
        os.environ["PATH"] = _bin + os.pathsep + os.environ.get("PATH", "")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from matplotlib.ticker import NullFormatter, ScalarFormatter

from geometry_class import Geometry, EMGeometry, merge_protocols
from geometry_class.plotting import draw_tools_in_borehole

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
BACKFILL_SWEEP  = [1.0, 10.0, 100.0, 1000.0]

# ERT sonde
TOOL_DIAM        = 0.05               # insulating tool body diameter (m); None = no tool body
NORMAL_SPACINGS  = [0.4064, 1.6256]   # AM (m): 16" and 64" normals
LATERAL_SPACINGS = [1.83]             # AO (m), centre of MN to A; classic oilfield value is 5.69 (18'8")
LATERAL_MN       = 0.2                # MN (m); classic oilfield value is 0.81 (32")
ERT_MEAS_SPACING = 0.1                # tool step along the hole (m)

# EM tool (coaxial coils)
EM_FREQUENCIES   = [400.0, 8000.0]
EM_COIL_SPACINGS = [0.5, 0.5]         # main transmitter-receiver distance (m)
EM_BUCKING       = [None, 0.25]       # bucking receiver distance (m) per coil pair; None = two-coil
EM_MEAS_SPACING  = 0.1

OUTPUT_DIR       = "results"
REUSE_SAVED_RESULTS = False           # True: redraw the figures from results/realistic_logs.csv
# ---------------------------------------------------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)
sweep = sorted(BACKFILL_SWEEP)


def simulate():
    """Run all ERT and EM simulations; return the no-borehole and per-backfill tables."""
    # ===========================================================================
    # 1.  ERT: one mesh, normal + lateral protocols merged into one simulation
    # ===========================================================================
    print("=" * 60)
    print("ERT sonde (PyGIMLi): normal and lateral arrays"
          + (f", tool body {TOOL_DIAM * 1000:.0f} mm" if TOOL_DIAM else ""))
    print("=" * 60)

    geo = Geometry(
        borehole_length=BOREHOLE_LENGTH,
        borehole_diameter=BOREHOLE_DIAM,
        layer_1d_geometry=LAYERS.copy(),
        backfill_resistivity=sweep[0],
        long_spacing=max(NORMAL_SPACINGS + [a + LATERAL_MN / 2 for a in LATERAL_SPACINGS]),
        tool_diameter=TOOL_DIAM,
    )
    plc = geo.make_basic_geometry(vtk_filename=None)
    protocols, tool_names = [], []
    if NORMAL_SPACINGS:
        protocols.append(geo.make_tool_array(plc, "normal", NORMAL_SPACINGS, ERT_MEAS_SPACING))
        tool_names.append("normal")
    if LATERAL_SPACINGS:
        protocols.append(geo.make_tool_array(plc, "lateral", LATERAL_SPACINGS, ERT_MEAS_SPACING,
                                             mn_spacing=LATERAL_MN))
        tool_names.append("lateral")
    proto = merge_protocols(*protocols)
    mesh = geo.make_mesh(plc, vtk_filename=None)
    print(f"  {proto.size()} measurements, {proto.sensorCount()} electrode positions, {mesh.cellCount():,} cells")


    def ert_table(sim) -> pd.DataFrame:
        """One row per measurement: tool name, spacing, reference depth, rho_a."""
        names = [tool_names[int(p)] for p in np.array(sim["protocol"])]
        sp = np.array(sim["spacing"])
        return pd.DataFrame({
            "tool": [f'ERT {n} {"AM" if n == "normal" else "AO"}={s:.2f} m' for n, s in zip(names, sp)],
            "depth": np.array(sim["depthref"]),
            "rhoa": np.array(sim("rhoa")),
        })


    ert_ref = ert_table(geo.run(mesh, proto, output_filename=None, include_borehole=False))
    ert_res = {}
    for rho_bf in sweep:
        geo.set_borehole(backfill_resistivity=rho_bf)
        ert_res[rho_bf] = ert_table(geo.run(mesh, proto, output_filename=None))
        print(f"  ERT done for backfill {rho_bf:g} Ohm.m")


    # ===========================================================================
    # 2.  EM: two-coil and focused (bucked) coaxial tools
    # ===========================================================================
    print()
    print("=" * 60)
    print("EM tool (SimPEG): two-coil and focused coaxial arrays")
    print("=" * 60)


    def make_em(rho_bf):
        return EMGeometry(
            borehole_length=BOREHOLE_LENGTH,
            borehole_diameter=BOREHOLE_DIAM,
            layer_1d_geometry=LAYERS.copy(),
            backfill_resistivity=rho_bf,
            frequencies=EM_FREQUENCIES,
            coil_spacings=EM_COIL_SPACINGS,
            bucking_spacings=EM_BUCKING,
            measuring_spacing=EM_MEAS_SPACING,
        )


    def em_table(df: pd.DataFrame) -> pd.DataFrame:
        name = lambda r: (f"EM s={r.spacing:g} m, {r.frequency:.0f} Hz" if np.isnan(r.bucking)
                          else f"EM focused s={r.spacing:g}/{r.bucking:g} m, {r.frequency:.0f} Hz")
        return pd.DataFrame({"tool": [name(r) for r in df.itertuples()],
                             "depth": df["depth"].values,
                             "rhoa": df["apparent_resistivity"].values})


    em_ref = em_table(make_em(sweep[0]).run(verbose=False, include_borehole=False))
    em_res = {rho_bf: em_table(make_em(rho_bf).run(verbose=False)) for rho_bf in sweep}
    print(f"  EM done for backfill {', '.join(f'{r:g}' for r in sweep)} Ohm.m")

    ref = pd.concat([ert_ref, em_ref])
    res = {rho_bf: pd.concat([ert_res[rho_bf], em_res[rho_bf]]) for rho_bf in sweep}
    return ref, res


results_csv = os.path.join(OUTPUT_DIR, "realistic_logs.csv")
if REUSE_SAVED_RESULTS and os.path.exists(results_csv):
    saved = pd.read_csv(results_csv)
    ref = saved[saved["backfill"].isna()].drop(columns="backfill")
    sweep = sorted(saved["backfill"].dropna().unique())
    res = {rho_bf: saved[saved["backfill"] == rho_bf].drop(columns="backfill") for rho_bf in sweep}
    print(f"Loaded saved results from {results_csv}")
else:
    ref, res = simulate()
    pd.concat([ref.assign(backfill=np.nan)] + [res[b].assign(backfill=b) for b in sweep]).to_csv(
        results_csv, index=False)
    print(f"Saved: {results_csv}  (backfill empty = no-borehole reference)")
tools = list(dict.fromkeys(ref["tool"]))


# ===========================================================================
# 3.  Plots
# ===========================================================================
colors = dict(zip(tools, plt.cm.tab10(np.linspace(0, 0.9, len(tools)))))


def curve(df, tool):
    d = df[df["tool"] == tool].sort_values("depth")
    return d["depth"].values, d["rhoa"].values


# Close-up position: both tools centred on the middle of the hole
ERT_EXTENT = max(list(NORMAL_SPACINGS) + [a + LATERAL_MN / 2 for a in LATERAL_SPACINGS] + [0.0])
CLOSEUP_A_DEPTH = BOREHOLE_LENGTH / 2 + ERT_EXTENT / 2
CLOSEUP_TX_DEPTH = BOREHOLE_LENGTH / 2 - max(EM_COIL_SPACINGS) / 2


def plot_logs(rho_bf, filename):
    fig = plt.figure(figsize=(16, 8))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1.15, 2.5])
    ax_m = fig.add_subplot(gs[0])
    ax_t = fig.add_subplot(gs[1])
    ax = fig.add_subplot(gs[2], sharey=ax_m)
    fig.suptitle(f"Realistic tools -- backfill {rho_bf:g} Ohm.m"
                 + (f", ERT tool body {TOOL_DIAM * 1000:.0f} mm" if TOOL_DIAM else ""),
                 fontsize=13, fontweight="bold")
    vals = LAYERS[:, 2].tolist() + [rho_bf]
    norm = mcolors.LogNorm(vmin=min(vals), vmax=max(vals) * (1.0001 if min(vals) == max(vals) else 1))
    for i, (top, bot, r) in enumerate(LAYERS):
        bot = bot if i < len(LAYERS) - 1 else BOREHOLE_LENGTH * 1.05
        ax_m.add_patch(patches.Rectangle((-1.5, -bot), 3, bot - top, facecolor=cm.RdYlBu_r(norm(r)),
                                         edgecolor="k", lw=0.8))
        ax_m.text(0.6, -(top + bot) / 2, f"{r:.0f} Ohm.m", ha="center", va="center", fontsize=9,
                  bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))
    ax_m.add_patch(patches.Rectangle((-BOREHOLE_DIAM / 2, -BOREHOLE_LENGTH), BOREHOLE_DIAM, BOREHOLE_LENGTH,
                                     facecolor=cm.RdYlBu_r(norm(rho_bf)), edgecolor="k", hatch="////"))
    ax_m.set_xlim(-1.5, 1.5)
    ax_m.set_ylim(-BOREHOLE_LENGTH * 1.08, 0.3)
    ax_m.set_xlabel("x (m)  [borehole not to scale]")
    ax_m.set_ylabel("Depth (m)")
    ax_m.set_title("1-D Resistivity Model")

    # To-scale close-up of the tools in the hole at one depth, marked on the model
    mid = BOREHOLE_LENGTH / 2
    rho_mid = next((r for t, b, r in LAYERS if t <= mid < b), LAYERS[-1, 2])
    top, bottom = draw_tools_in_borehole(
        ax_t, CLOSEUP_A_DEPTH, BOREHOLE_DIAM, cm.RdYlBu_r(norm(rho_bf)), cm.RdYlBu_r(norm(rho_mid)),
        tool_diameter=TOOL_DIAM, normal_spacings=NORMAL_SPACINGS, lateral_spacings=LATERAL_SPACINGS,
        lateral_mn=LATERAL_MN, em_tx_depth=CLOSEUP_TX_DEPTH,
        em_spacings=list(dict.fromkeys(EM_COIL_SPACINGS)),
        em_bucking=[b for b in dict.fromkeys(EM_BUCKING) if b is not None])
    ax_m.axhspan(-bottom, -top, color="k", alpha=0.15, zorder=5)
    ax_m.text(-1.4, -(top + bottom) / 2, "close-up", fontsize=8, va="center", zorder=6,
              bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"))

    for tool in tools:
        z, r = curve(res[rho_bf], tool)
        z0, r0 = curve(ref, tool)
        ax.plot(r, -z, color=colors[tool], lw=1.6, label=tool)
        ax.plot(r0, -z0, color=colors[tool], lw=1.0, ls=":")
    tz, tr = [], []
    for i, (top, bot, r) in enumerate(LAYERS):
        bot = bot if i < len(LAYERS) - 1 else BOREHOLE_LENGTH * 1.05
        tz += [-top, -bot]
        tr += [r, r]
    ax.plot(tr, tz, "k--", lw=2, label="True 1-D model")
    ax.plot([], [], color="gray", ls=":", label="No-borehole reference")
    ax.set_xscale("log")
    ax.set_xlabel("Apparent resistivity (Ohm.m)")
    ax.set_title("Apparent resistivity vs depth (dotted: no borehole)")
    ax.grid(True, which="both", ls="--", alpha=0.45)
    ax.legend(fontsize=8, loc="lower right", framealpha=0.9)
    for _, bot, _ in LAYERS[:-1]:
        ax.axhline(-bot, color="gray", ls=":", lw=0.9)
        ax_m.axhline(-bot, color="gray", ls=":", lw=0.9)
    plt.tight_layout()
    fig.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {filename}")


print()
for rho_bf in sweep:
    plot_logs(rho_bf, os.path.join(OUTPUT_DIR, f"realistic_log_backfill_{rho_bf:g}.png"))

# Borehole effect: one panel per tool
fig, axes = plt.subplots(1, len(tools), figsize=(2.8 * len(tools), 7), sharey=True)
fig.suptitle(f"Realistic tools -- borehole effect rho_a(with hole) / rho_a(no hole), D = {BOREHOLE_DIAM:g} m",
             fontsize=12, fontweight="bold")
bf_colors = cm.viridis(np.linspace(0, 0.9, len(sweep)))
for ax, tool in zip(np.atleast_1d(axes), tools):
    z0, r0 = curve(ref, tool)
    for c, rho_bf in zip(bf_colors, sweep):
        z, r = curve(res[rho_bf], tool)
        ax.plot(r / r0, -z, color=c, lw=1.5, label=f"{rho_bf:g} Ohm.m")
    ax.axvline(1.0, color="k", lw=0.8)
    for _, bot, _ in LAYERS[:-1]:
        ax.axhline(-bot, color="gray", ls=":", lw=0.9)
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_title(tool.replace(", ", "\n"), fontsize=9)
    ax.set_xlabel("ratio")
    ax.grid(True, which="both", ls="--", alpha=0.45)
np.atleast_1d(axes)[0].set_ylabel("Depth (m)")
np.atleast_1d(axes)[0].legend(title="Backfill", fontsize=8, loc="lower left")
plt.tight_layout()
effect_png = os.path.join(OUTPUT_DIR, "realistic_backfill_effect.png")
fig.savefig(effect_png, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {effect_png}")

# Summary at the middle of each layer
print("\nBorehole effect (ratio with/without hole) at layer mid-depths:")
rows = []
for top, bot, rho_layer in LAYERS:
    zmid = 0.5 * (top + bot)
    for rho_bf in sweep:
        row = {"layer_rho": rho_layer, "depth": zmid, "backfill": rho_bf}
        for tool in tools:
            z, r = curve(res[rho_bf], tool)
            z0, r0 = curve(ref, tool)
            if len(z) and z.min() <= zmid <= z.max():
                row[tool] = np.interp(zmid, z, r) / np.interp(zmid, z0, r0)
            else:
                row[tool] = np.nan  # tool cannot reach this depth
        rows.append(row)
summary = pd.DataFrame(rows)
with pd.option_context("display.width", 250, "display.max_columns", 20):
    print(summary.round(3).to_string(index=False))
