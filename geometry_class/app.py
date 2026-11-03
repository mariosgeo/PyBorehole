"""
PyBorehole - interactive borehole log simulator (Streamlit app).

    pyborehole-app                      # after pip install "pyborehole[app]"
    streamlit run streamlit_app.py      # in a clone of the repository

Tabs:
  1. Geometry   - borehole and the depth intervals of the units (IDs U1, U2, ...)
  2. Properties - per unit: lithology and pore water from the built-in database,
                  Waxman-Smits parameters -> bulk resistivity, K/U/Th, density; the borehole fill
  3. Tools      - ERT array, EM coils and frequencies, gamma-ray probe, NMR probe
  4. Run        - choose ERT / EM / gamma ray / NMR, run, plot the logs and the borehole effect
"""
import io
import json
import os
import shutil
import sys
import threading
import warnings

warnings.filterwarnings("ignore")

# TetGen (ERT meshing) lives in the environment's bin folder; make it findable
for _bin in (os.path.join(sys.prefix, "Library", "bin"), os.path.join(sys.prefix, "bin")):
    if os.path.isdir(_bin) and _bin not in os.environ.get("PATH", ""):
        os.environ["PATH"] = _bin + os.pathsep + os.environ.get("PATH", "")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import streamlit as st

from geometry_class.petrophysics import (BACKFILL_NMR, BACKFILLS, LITHOLOGIES, WATERS, qv_from_cec, waxman_smits_b,
                                         waxman_smits_resistivity,
                                         backfill_resistivity, bulk_density, gamma_api)

st.set_page_config(page_title="PyBorehole", page_icon="🕳️", layout="wide")
ss = st.session_state

LITH_COLORS = {"Gravel": "#e8c170", "Sand": "#f2d38f", "Silty sand": "#d9c48f", "Silt": "#b8b38a",
               "Clay": "#8fa9c9", "Glacial till": "#a39e8f", "Sandstone": "#e0a96d", "Limestone": "#c9d6df"}
EXAMPLE_UNITS = pd.DataFrame({"top": [0.0, 5.0, 10.0], "bottom": [5.0, 10.0, 15.0]})
EXAMPLE_LITH = ["Silty sand", "Clay", "Gravel"]
NMR_KEYS = ("t2lm", "t2_sigma", "cbw")


# ===========================================================================
# State helpers
# ===========================================================================
def init_state():
    ss.setdefault("units_base", EXAMPLE_UNITS.copy())
    ss.setdefault("units_ver", 0)
    ss.setdefault("L", 15.0)
    ss.setdefault("D", 0.20)
    ss.setdefault("results", None)
    ss.setdefault("results_key", None)


def unit_ids(n):
    return [f"U{i + 1}" for i in range(n)]


def clean_units(df: pd.DataFrame):
    """Sorted unit table with IDs, and a list of problems (empty = valid)."""
    d = df.dropna(how="all").copy()
    problems = []
    if d.empty:
        return d, ["Add at least one unit."]
    if d[["top", "bottom"]].isna().any().any():
        problems.append("Every unit needs a top and a bottom depth.")
        d = d.dropna(subset=["top", "bottom"])
    d = d.astype({"top": float, "bottom": float}).sort_values("top").reset_index(drop=True)
    d.insert(0, "id", unit_ids(len(d)))
    if len(d) and abs(d["top"].iloc[0]) > 1e-9:
        problems.append("The first unit must start at 0 m (the surface).")
    for i, r in d.iterrows():
        if r.bottom <= r.top:
            problems.append(f"{r.id}: the bottom must be deeper than the top.")
        if i > 0 and abs(r.top - d["bottom"].iloc[i - 1]) > 1e-9:
            gap = "gap" if r.top > d["bottom"].iloc[i - 1] else "overlap"
            problems.append(f"{d['id'].iloc[i - 1]} / {r.id}: {gap} between {d['bottom'].iloc[i - 1]:g} "
                            f"and {r.top:g} m.")
    return d, problems


def load_unit_defaults(uid, lith=None, water=None):
    """Fill a unit's widgets from the database (keeps values the user did not ask to reset)."""
    lith = lith or ss.get(f"{uid}_lith", "Sand")
    water = water or ss.get(f"{uid}_water", "Fresh")
    L = LITHOLOGIES[lith]
    ss[f"{uid}_lith"], ss[f"{uid}_water"] = lith, water
    ss[f"{uid}_rhow"] = WATERS[water]["rho_w"]
    for k in ("porosity", "a", "m", "n", "cec", "grain_density", "K", "U", "Th") + NMR_KEYS:
        ss[f"{uid}_{k}"] = float(L[k])
    ss.setdefault(f"{uid}_sw", 1.0)
    ss.setdefault(f"{uid}_override", False)
    ss.setdefault(f"{uid}_rho_manual", 100.0)


def ensure_unit(uid, index):
    if f"{uid}_lith" not in ss:
        lith = EXAMPLE_LITH[index] if index < len(EXAMPLE_LITH) else "Sand"
        load_unit_defaults(uid, lith, "Fresh")
    for k in NMR_KEYS:                       # units set up before the NMR tool existed
        ss.setdefault(f"{uid}_{k}", float(LITHOLOGIES[ss[f"{uid}_lith"]][k]))


def on_lith_change(uid):
    load_unit_defaults(uid, lith=ss[f"{uid}_lith"], water=ss[f"{uid}_water"])


def on_water_change(uid):
    ss[f"{uid}_rhow"] = WATERS[ss[f"{uid}_water"]]["rho_w"]


def ensure_backfill():
    if "bf_type" not in ss:
        ss["bf_type"], ss["bf_water"] = "Water", "Fresh"
        on_backfill_change()
    for k, v in BACKFILLS[ss["bf_type"]]["nmr"].items():
        ss.setdefault(f"bf_nmr_{k}", float(v))


def on_backfill_change():
    bf = BACKFILLS[ss["bf_type"]]
    ss["bf_res"] = round(float(backfill_resistivity(ss["bf_type"], WATERS[ss["bf_water"]]["rho_w"])), 3)
    for k in ("K", "U", "Th"):
        ss[f"bf_{k}"] = float(bf[k])
    if ss["bf_type"] == "Water" and ss["bf_water"] == "Salt":
        ss["bf_K"] = 0.04
    for k, v in bf["nmr"].items():
        ss[f"bf_nmr_{k}"] = float(v)


def backfill_nmr():
    return {k: ss[f"bf_nmr_{k}"] for k in ("water", "t2lm", "t2_sigma", "cbw")}


def backfill_material():
    if ss["bf_type"] == "Water":
        return "seawater" if ss["bf_water"] == "Salt" else "water"
    return BACKFILLS[ss["bf_type"]]["gamma_material"]


def unit_table(units: pd.DataFrame) -> pd.DataFrame:
    """Units with all properties and derived values, from the widgets."""
    rows = []
    for _, u in units.iterrows():
        g = lambda k: ss[f"{u.id}_{k}"]
        rho_ws = waxman_smits_resistivity(g("rhow"), g("porosity"), g("a"), g("m"), g("n"), g("sw"),
                                          g("cec"), g("grain_density"))
        rho = g("rho_manual") if g("override") else rho_ws
        rows.append(dict(id=u.id, top=u.top, bottom=u.bottom, lithology=g("lith"), water=g("water"),
                         rho_w=g("rhow"), porosity=g("porosity"), sw=g("sw"), a=g("a"), m=g("m"), n=g("n"),
                         cec=g("cec"), qv=qv_from_cec(g("cec"), g("porosity"), g("grain_density")),
                         B=waxman_smits_b(g("rhow")), rho_ws=rho_ws, resistivity=rho,
                         density=bulk_density(g("porosity"), g("grain_density"), g("sw"),
                                              WATERS[g("water")]["density"]),
                         K=g("K"), U=g("U"), Th=g("Th"), api=gamma_api(g("K"), g("U"), g("Th")),
                         gamma_material=LITHOLOGIES[g("lith")]["gamma_material"],
                         water_content=g("porosity") * g("sw"), t2lm=g("t2lm"), t2_sigma=g("t2_sigma"),
                         cbw=g("cbw")))
    return pd.DataFrame(rows)


def parse_list(text, name, n_min=1):
    try:
        vals = [float(x) for x in str(text).replace(";", ",").split(",") if x.strip()]
    except ValueError:
        st.error(f"{name}: enter numbers separated by commas.")
        return None
    if len(vals) < n_min or any(v <= 0 for v in vals):
        st.error(f"{name}: enter at least {n_min} positive value(s).")
        return None
    return vals


def model_sketch(units: pd.DataFrame, L, D, ax=None, label="id"):
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(3.2, 6))
    zmax = max(units["bottom"].max(), L) * 1.05
    for i, u in units.iterrows():
        bot = zmax if i == len(units) - 1 else u.bottom
        lith = u.get("lithology", ss.get(f"{u.id}_lith", ""))
        col = LITH_COLORS.get(lith, plt.cm.Pastel1(i % 9))
        ax.add_patch(mpatches.Rectangle((-1, -bot), 2, bot - u.top, facecolor=col, edgecolor="k", lw=0.6))
        txt = u.id if label == "id" else f"{u.id}  {u.lithology}\n{u.resistivity:.3g} Ω·m, {u.api:.0f} API"
        ax.text(0.55, -(u.top + bot) / 2, txt, ha="center", va="center", fontsize=8,
                bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=1.5))
    ax.add_patch(mpatches.Rectangle((-0.12, -L), 0.24, L, facecolor="#b9d9b0", edgecolor="k", lw=0.8, hatch="///"))
    ax.set_xlim(-1, 1); ax.set_ylim(-zmax, 0.2)
    ax.set_xticks([]); ax.set_ylabel("Depth (m)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{-v + 0.0:g}"))
    ax.set_title(f"Model (hole {D * 100:.0f} cm, not to scale)", fontsize=9)
    if own:
        fig.tight_layout()
        return fig


init_state()
# Streamlit deletes the state of widgets that are not drawn in a run, e.g. the unit
# properties while the unit table is briefly invalid (a new row without a bottom
# depth yet). Re-assigning the keys makes them user state, which is kept.
for _k in list(ss.keys()):
    if _k.startswith(("U", "bf_")):
        ss[_k] = ss[_k]
st.title("PyBorehole")
st.caption("Borehole backfill effect on resistivity (ERT), induction (EM), natural gamma-ray and NMR logs")
tab1, tab2, tab3, tab4 = st.tabs(["1 · Geometry", "2 · Properties", "3 · Tools", "4 · Run"])

# ===========================================================================
# Tab 1: geometry
# ===========================================================================
with tab1:
    c1, c2 = st.columns([2, 1])
    with c1:
        st.subheader("Borehole")
        b1, b2 = st.columns(2)
        b1.number_input("Borehole depth (m)", min_value=1.0, max_value=200.0, step=1.0, key="L")
        b2.number_input("Borehole diameter (m)", min_value=0.05, max_value=1.0, step=0.01, format="%.3f", key="D")

        st.subheader("Units")
        st.write("One row per unit, depths in metres (positive down). Units must follow each other "
                 "without gaps; the last unit continues below its bottom. Add or delete rows in the table.")
        edited = st.data_editor(
            ss.units_base, key=f"units_editor_{ss.units_ver}", num_rows="dynamic", width="stretch",
            column_config={"top": st.column_config.NumberColumn("Top (m)", min_value=0.0, step=0.5, format="%.2f"),
                           "bottom": st.column_config.NumberColumn("Bottom (m)", min_value=0.0, step=0.5,
                                                                   format="%.2f")})
        if st.button("Reset to the example (3 units, 15 m)"):
            ss.units_base = EXAMPLE_UNITS.copy()
            ss.units_ver += 1
            st.rerun()
        units, problems = clean_units(edited)
        for p in problems:
            st.error(p)
        if not problems:
            st.success(f"{len(units)} units: " + ", ".join(
                f"**{u.id}** {u.top:g}–{u.bottom:g} m" for u in units.itertuples()))
    with c2:
        if not units.empty:
            st.pyplot(model_sketch(units, ss.L, ss.D), width="content")
units_ok = not problems and not units.empty

# ===========================================================================
# Tab 2: properties
# ===========================================================================
with tab2:
    if not units_ok:
        st.warning("Fix the unit geometry in tab 1 first.")
    else:
        st.write("Pick a lithology and a pore water for each unit: the database fills in the values, which you "
                 "can then change. Bulk resistivity follows the **Waxman-Smits** model for shaly sediments:  "
                 r"$\sigma = \dfrac{\phi^{\,m} S_w^{\,n}}{a}\left(\dfrac{1}{\rho_w} + \dfrac{B\,Q_v}{S_w}\right)$,  "
                 r"$\rho = 1/\sigma$, with $Q_v = \mathrm{CEC}/100\cdot\rho_g(1-\phi)/\phi$ and "
                 r"$B = 4.6\,(1 - 0.6\,e^{-0.77/\rho_w})$.  CEC = 0 is Archie's law (clean sand, gravel).")
        for i, uid in enumerate(units["id"]):
            ensure_unit(uid, i)
            u = units.iloc[i]
            # the title must not change when the lithology changes, or the panel collapses
            with st.expander(f"**{uid}** — {u.top:g} to {u.bottom:g} m", expanded=(i == 0)):
                c1, c2, c3, c4 = st.columns([1.2, 1.2, 1, 1.1])
                with c1:
                    st.markdown("**Lithology and water**")
                    st.selectbox("Lithology (database)", list(LITHOLOGIES), key=f"{uid}_lith",
                                 on_change=on_lith_change, args=(uid,))
                    st.selectbox("Pore water", list(WATERS), key=f"{uid}_water",
                                 on_change=on_water_change, args=(uid,),
                                 help="Fresh 20 Ω·m, brackish 2 Ω·m, salt (seawater) 0.2 Ω·m")
                    st.number_input("Water resistivity ρw (Ω·m)", min_value=0.01, step=0.1,
                                    format="%.3f", key=f"{uid}_rhow")
                    st.number_input("Water saturation Sw (-)", min_value=0.05, max_value=1.0, step=0.05,
                                    key=f"{uid}_sw", help="1 = below the water table")
                with c2:
                    st.markdown("**Waxman-Smits parameters**")
                    st.number_input("Porosity φ (-)", min_value=0.01, max_value=0.9, step=0.01,
                                    key=f"{uid}_porosity")
                    st.number_input("Tortuosity a (-)", min_value=0.1, max_value=5.0, step=0.1, key=f"{uid}_a")
                    st.number_input("Cementation m (-)", min_value=1.0, max_value=4.0, step=0.1, key=f"{uid}_m")
                    st.number_input("Saturation exponent n (-)", min_value=1.0, max_value=4.0, step=0.1,
                                    key=f"{uid}_n")
                    st.number_input("Cation-exchange capacity CEC (meq/100 g)", min_value=0.0, max_value=150.0,
                                    step=0.5, format="%.1f", key=f"{uid}_cec",
                                    help="Clay surface conduction. 0 = clean sand (Archie); silt ~2, clay ~10; "
                                         "pure minerals: kaolinite 3-15, illite 10-40, smectite 80-150")
                with c3:
                    st.markdown("**Gamma ray**")
                    st.number_input("K (%)", min_value=0.0, max_value=60.0, step=0.1, key=f"{uid}_K")
                    st.number_input("U (ppm)", min_value=0.0, max_value=1000.0, step=0.1, key=f"{uid}_U")
                    st.number_input("Th (ppm)", min_value=0.0, max_value=1000.0, step=0.5, key=f"{uid}_Th")
                    st.number_input("Grain density (g/cm³)", min_value=1.0, max_value=5.0, step=0.01,
                                    key=f"{uid}_grain_density")
                    st.markdown("**NMR**")
                    st.number_input("T2 log-mean (ms)", min_value=0.5, max_value=3000.0, step=10.0,
                                    format="%.1f", key=f"{uid}_t2lm",
                                    help="Main T2 mode; follows pore size: gravel ~500, sand ~300, silt ~25, "
                                         "clay a few ms")
                    st.number_input("T2 spread (decades)", min_value=0.05, max_value=1.5, step=0.05,
                                    key=f"{uid}_t2_sigma")
                    st.number_input("Clay-bound water fraction (-)", min_value=0.0, max_value=1.0, step=0.05,
                                    key=f"{uid}_cbw", help="Share of the pore water bound to clay (T2 ≈ 2 ms)")
                with c4:
                    t = unit_table(units.iloc[[i]]).iloc[0]
                    st.markdown("**Result**")
                    st.metric("Waxman-Smits bulk resistivity", f"{t.rho_ws:.3g} Ω·m")
                    st.caption(f"Qv = {t.qv:.3f} meq/cm³, B = {t.B:.2f} (S/m)/(meq/cm³)")
                    st.checkbox("Override resistivity", key=f"{uid}_override")
                    if ss[f"{uid}_override"]:
                        st.number_input("Resistivity used (Ω·m)", min_value=0.01, key=f"{uid}_rho_manual")
                    st.metric("Bulk density", f"{t.density:.2f} g/cm³")
                    st.metric("Gamma ray (16K+8U+4Th)", f"{t.api:.0f} API")
                    st.metric("Water content (NMR)", f"{t.water_content:.3f}")

        st.subheader("Borehole fill")
        ensure_backfill()
        f1, f2, f3 = st.columns([1.2, 1.2, 2])
        with f1:
            st.selectbox("Fill (database)", list(BACKFILLS), key="bf_type", on_change=on_backfill_change)
            st.selectbox("Water in the hole / pores of the fill", list(WATERS), key="bf_water",
                         on_change=on_backfill_change)
            st.caption(BACKFILLS[ss["bf_type"]]["note"])
        with f2:
            st.number_input("Fill resistivity (Ω·m)", min_value=0.01, step=0.5, format="%.3f", key="bf_res")
            st.caption(f"Gamma-ray material: `{backfill_material()}`")
        with f3:
            k1, k2, k3 = st.columns(3)
            k1.number_input("Fill K (%)", min_value=0.0, step=0.05, key="bf_K")
            k2.number_input("Fill U (ppm)", min_value=0.0, step=0.1, key="bf_U")
            k3.number_input("Fill Th (ppm)", min_value=0.0, step=0.5, key="bf_Th")
            st.caption(f"Fill gamma ray: {gamma_api(ss.bf_K, ss.bf_U, ss.bf_Th):.0f} API")
            n1, n2, n3, n4 = st.columns(4)
            n1.number_input("Fill water content (-)", min_value=0.0, max_value=1.0, step=0.05, key="bf_nmr_water")
            n2.number_input("Fill T2 log-mean (ms)", min_value=0.5, max_value=3000.0, step=10.0, format="%.1f",
                            key="bf_nmr_t2lm")
            n3.number_input("Fill T2 spread (decades)", min_value=0.05, max_value=1.5, step=0.05,
                            key="bf_nmr_t2_sigma")
            n4.number_input("Fill clay-bound fraction (-)", min_value=0.0, max_value=1.0, step=0.05,
                            key="bf_nmr_cbw")
            st.caption("NMR of the fill: open water relaxes slowly (T2 ≈ 2.5 s); the water in bentonite and "
                       "cement grout is bound (T2 ≈ 1–2 ms) and partly invisible to a slim probe.")

        st.subheader("Summary")
        tbl = unit_table(units)
        st.dataframe(tbl[["id", "top", "bottom", "lithology", "water", "rho_w", "porosity", "m", "cec", "qv",
                          "resistivity", "density", "K", "U", "Th", "api", "water_content", "t2lm", "cbw"]].round(3),
                     width="stretch", hide_index=True,
                     column_config={"rho_w": "ρw (Ω·m)", "cec": "CEC (meq/100 g)", "qv": "Qv (meq/cm³)",
                                    "resistivity": "ρ bulk (Ω·m)",
                                    "density": "ρ bulk (g/cm³)", "api": "GR (API)",
                                    "water_content": "water content (-)", "t2lm": "T2LM (ms)",
                                    "cbw": "clay-bound (-)"})
        with st.expander("Built-in database"):
            st.markdown("**Lithologies** (typical values; replace them with local data when available)")
            st.dataframe(pd.DataFrame(LITHOLOGIES).T, width="stretch")
            st.markdown("**Pore waters**")
            st.dataframe(pd.DataFrame(WATERS).T, width="stretch")
            st.markdown("**Borehole fills**")
            st.dataframe(pd.DataFrame(BACKFILLS).T.drop(columns=["ws", "nmr"]), width="stretch")
            st.markdown("**Borehole fills, NMR**")
            st.dataframe(pd.DataFrame(BACKFILL_NMR).T, width="stretch")

# ===========================================================================
# Tab 3: tools
# ===========================================================================
with tab3:
    t1, t2, t3, t4 = st.columns(4)
    with t1:
        st.subheader("ERT (DC resistivity)")
        arr = st.radio("Array", ["Wenner", "Normal", "Lateral"], key="ert_array", horizontal=True,
                       help="Wenner: A-M-N-B with spacing a. Normal: A-M on the sonde (16\", 64\"). "
                            "Lateral: A-M-N on the sonde, AO = distance from A to the MN midpoint.")
        defaults = {"Wenner": "0.5, 1.0", "Normal": "0.41, 1.63", "Lateral": "1.83"}
        ert_sp_text = st.text_input({"Wenner": "Spacings a (m), exactly two",
                                     "Normal": "AM spacings (m)", "Lateral": "AO spacings (m)"}[arr],
                                    value=defaults[arr], key=f"ert_sp_{arr}")
        ert_mn = st.number_input("MN (m)", min_value=0.05, value=0.2, step=0.05, key="ert_mn") \
            if arr == "Lateral" else 0.2
        ert_step = st.number_input("Measuring step (m)", min_value=0.05, max_value=1.0, value=0.1, step=0.05,
                                   key="ert_step")
        ert_tool = st.checkbox("Insulating tool body", value=(arr != "Wenner"), key=f"ert_tool_{arr}")
        ert_tool_d = st.number_input("Tool diameter (m)", min_value=0.01, max_value=0.5, value=0.05, step=0.005,
                                     format="%.3f", key="ert_tool_d") if ert_tool else None
        ert_sp = parse_list(ert_sp_text, "ERT spacings", 2 if arr == "Wenner" else 1)
        if ert_sp and arr == "Wenner" and len(ert_sp) != 2:
            st.error("Wenner: enter exactly two spacings.")
            ert_sp = None
        if ert_sp and any(abs(s / ert_step - round(s / ert_step)) > 1e-6 for s in ert_sp) and arr == "Wenner":
            st.warning("Wenner spacings should be multiples of the measuring step.")
    with t2:
        st.subheader("EM (induction, coaxial coils)")
        em_sp = parse_list(st.text_input("Coil spacings (m)", "0.5, 1.0", key="em_sp"), "EM coil spacings")
        em_f = parse_list(st.text_input("Frequencies (Hz)", "400, 8000", key="em_f"), "EM frequencies")
        em_step = st.number_input("Measuring step (m)", min_value=0.05, max_value=1.0, value=0.1, step=0.05,
                                  key="em_step")
    with t3:
        st.subheader("Gamma ray (NaI probe)")
        gr_step = st.number_input("Measuring step (m)", min_value=0.05, max_value=2.0, value=0.1, step=0.05,
                                  key="gr_step")
        gr_events = st.select_slider("Statistics", options=[40_000, 150_000, 400_000], value=150_000,
                                     key="gr_events", format_func=lambda v: {40_000: "fast (~4 % noise)",
                                                                             150_000: "standard (~2 %)",
                                                                             400_000: "precise (~1 %)"}[v])
        gr_cd = st.number_input("Crystal diameter (m)", min_value=0.01, max_value=0.1, value=0.038, step=0.001,
                                format="%.3f", key="gr_cd")
        gr_cl = st.number_input("Crystal length (m)", min_value=0.02, max_value=0.3, value=0.102, step=0.001,
                                format="%.3f", key="gr_cl")
        gr_td = st.number_input("Probe diameter (m)", min_value=0.02, max_value=0.2, value=0.050, step=0.001,
                                format="%.3f", key="gr_td")
        if units_ok and gr_td >= ss.D:
            st.error("The probe must be narrower than the borehole.")
    with t4:
        st.subheader("NMR (slim water-well probe)")
        nmr_f = parse_list(st.text_input("Frequencies (kHz), one shell each", "450, 310", key="nmr_f"),
                           "NMR frequencies")
        nmr_td = st.number_input("Probe diameter (m)", min_value=0.02, max_value=0.2, value=0.060, step=0.001,
                                 format="%.3f", key="nmr_td")
        nmr_md = st.number_input("Magnet diameter (m)", min_value=0.005, max_value=0.15, value=0.025, step=0.001,
                                 format="%.3f", key="nmr_md")
        nmr_br = st.number_input("Magnet remanence (T)", min_value=0.1, max_value=1.5, value=1.2, step=0.05,
                                 key="nmr_br", help="NdFeB about 1.2 T, SmCo about 1.0 T")
        nmr_ap = st.number_input("Sensitive length (m)", min_value=0.1, max_value=2.0, value=0.5, step=0.05,
                                 key="nmr_ap", help="Vertical extent of the sensitive region (antenna length)")
        nmr_te = st.number_input("Echo spacing TE (ms)", min_value=0.2, max_value=10.0, value=1.5, step=0.1,
                                 key="nmr_te")
        nmr_ne = st.number_input("Number of echoes", min_value=50, max_value=5000, value=1000, step=50,
                                 key="nmr_ne")
        nmr_noise = st.number_input("Noise per echo (% water content)", min_value=0.0, max_value=20.0, value=2.0,
                                    step=0.5, key="nmr_noise", help="After stacking; 0 = noise-free")
        nmr_step = st.number_input("Measuring step (m)", min_value=0.05, max_value=1.0, value=0.1, step=0.05,
                                   key="nmr_step")
        nmr_ok = bool(nmr_f)
        if nmr_f:
            from geometry_class.nmr_geometry import GAMMA_H
            c = nmr_br * (nmr_md / 2) ** 2 / 2
            sh = pd.DataFrame({"f (kHz)": nmr_f})
            b0 = 2 * np.pi * sh["f (kHz)"] * 1e3 / GAMMA_H
            sh["shell ⌀ (cm)"] = 200 * np.sqrt(c / b0)
            sh["gradient (T/m)"] = 2 * b0 / (sh["shell ⌀ (cm)"] / 200)
            st.dataframe(sh.round(2), hide_index=True, width="stretch")
            if (sh["shell ⌀ (cm)"] / 100 <= nmr_td).any():
                st.error("A shell lies inside the probe: lower that frequency or use a stronger magnet.")
                nmr_ok = False
            inside = sh[sh["shell ⌀ (cm)"] / 100 < ss.D]
            if len(inside):
                st.warning(f"The {', '.join(f'{v:.1f}' for v in inside['shell ⌀ (cm)'])} cm shell lies inside "
                           f"the {ss.D * 100:.0f} cm borehole: it measures the fill, not the formation.")

# ===========================================================================
# Tab 4: run
# ===========================================================================
@st.cache_resource
def memory_gb():
    """Memory this process may use (GB): the container limit if there is one
    (e.g. on Streamlit Community Cloud), else the machine's RAM; None if unknown."""
    sizes = []
    for f in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            v = open(f).read().strip()
            if v.isdigit() and int(v) < 1 << 60:
                sizes.append(int(v))
        except OSError:
            pass
    try:
        sizes.append(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
    except (AttributeError, ValueError, OSError):
        pass
    return min(sizes) / 1e9 if sizes else None


# The 3D ERT solve needs several GB; below this the app could be killed for
# every visitor (Streamlit Community Cloud gives about 2.7 GB).
ERT_MIN_GB = float(os.environ.get("PYBOREHOLE_ERT_MIN_GB", "6"))


@st.cache_resource
def ert_slots():
    """How many ERT runs may go at once on this server, shared by all visitors:
    $PYBOREHOLE_MAX_ERT, else as many as fit in memory (ERT_MIN_GB each), at least one."""
    if os.environ.get("PYBOREHOLE_MAX_ERT"):
        n = max(1, int(os.environ["PYBOREHOLE_MAX_ERT"]))
    else:
        mem = memory_gb()
        n = max(1, int(mem // ERT_MIN_GB)) if mem else 1
    return threading.BoundedSemaphore(n), n


def availability():
    """Which simulators can run here, with a reason when not."""
    out = {}
    try:
        import geometry_class
        geometry_class.Geometry  # noqa: B018
        import pygimli  # noqa: F401
        out["ERT"] = (True, "PyGIMLi found") if shutil.which("tetgen") else \
            (False, "TetGen not found: conda install -c conda-forge tetgen")
        mem = memory_gb()
        if out["ERT"][0] and mem is not None and mem < ERT_MIN_GB:
            out["ERT"] = (False, f"needs about {ERT_MIN_GB:g} GB of RAM, this server has {mem:.1f} GB: "
                                 "run the app locally for ERT")
    except Exception as e:  # noqa: BLE001
        out["ERT"] = (False, f"PyGIMLi not available ({e})")
    try:
        import simpeg  # noqa: F401
        import discretize  # noqa: F401
        out["EM"] = (True, "SimPEG found")
    except Exception as e:  # noqa: BLE001
        out["EM"] = (False, f"SimPEG not available ({e})")
    try:
        from geometry_class.gamma_geometry import _build_dirs, _EXE_NAME, geant4_env
        geant4_env()
        built = os.path.isfile(os.environ.get("GAMMALOG_EXE", "")) or \
            any(os.path.isfile(os.path.join(d, _EXE_NAME)) for d in _build_dirs())
        out["Gamma"] = (True, "Geant4 program and data found") if built else \
            (False, "Geant4 data found, but the program is not compiled: run pyborehole-build-gamma")
    except Exception as e:  # noqa: BLE001
        out["Gamma"] = (False, f"Geant4 not available ({e})")
    return out


def tool_styles(tools):
    """Colour per array / coil spacing, marker per EM frequency (the same in every plot)."""
    base = lambda t: t.split(", ")[0]
    bases = list(dict.fromkeys(base(t) for t in tools if t != "Gamma ray"))
    cols = dict(zip(bases, plt.cm.tab10(np.arange(len(bases)) % 10)))
    freqs = list(dict.fromkeys(t.split(", ")[1] for t in tools if t.startswith("EM") and ", " in t))
    marks = dict(zip(freqs, ["o", "^", "s", "D", "v", "P"]))
    return {t: ("#2ca02c", None) if t == "Gamma ray" else
            (cols[base(t)], marks.get(t.split(", ")[1]) if ", " in t else None) for t in tools}


def plot_logs(results, tbl, L, D):
    have_res = results["tool"].str.startswith(("ERT", "EM")).any()
    have_gr = (results["tool"] == "Gamma ray").any()
    have_nmr = results["tool"].str.startswith("NMR").any()
    ncols = 1 + have_res + have_gr + have_nmr
    fig, axes = plt.subplots(1, ncols, figsize=(3.2 + 4.5 * (ncols - 1), 8.5), sharey=True,
                             gridspec_kw={"width_ratios": [1] + [2.2] * (ncols - 1)})
    axes = np.atleast_1d(axes)
    model_sketch(tbl, L, D, ax=axes[0], label="full")
    zmax = max(tbl["bottom"].max(), L) * 1.05
    k = 1
    all_tools = list(dict.fromkeys(results["tool"]))
    tools = [t for t in all_tools if t.startswith(("ERT", "EM"))]
    style = tool_styles([t for t in all_tools if t != "Gamma ray"])
    step = lambda col: (np.repeat(tbl[col].values, 2),
                        -np.ravel([[t, b if i < len(tbl) - 1 else zmax]
                                   for i, (t, b) in enumerate(zip(tbl["top"], tbl["bottom"]))]))
    if have_res:
        ax = axes[k]; k += 1
        for t in tools:
            col, mk = style[t]
            for case, ls, lw in (("borehole", "-", 1.6), ("no borehole", ":", 1.2)):
                d = results[(results.tool == t) & (results.case == case)].sort_values("depth")
                ax.plot(d.value, -d.depth, ls=ls, lw=lw, color=col, label=t if case == "borehole" else None,
                        marker=mk if case == "borehole" else None, ms=4, markevery=max(len(d) // 12, 1))
        ax.plot(*step("resistivity"), "k--", lw=2, label="true (bulk)")
        ax.set_xscale("log"); ax.set_xlabel("Apparent resistivity (Ω·m)")
        ax.set_title("Resistivity logs (dotted: no borehole)", fontsize=10)
        ax.legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=2)
        ax.grid(True, which="both", ls="--", alpha=0.4)
    if have_nmr:
        ax = axes[k]; k += 1
        for t in [t for t in all_tools if t.startswith("NMR")]:
            col, _ = style[t]
            for case, ls, lw in (("borehole", "-", 1.6), ("no borehole", ":", 1.2)):
                d = results[(results.tool == t) & (results.case == case)].sort_values("depth")
                ax.plot(d.value, -d.depth, ls=ls, lw=lw, color=col, label=t if case == "borehole" else None)
        ax.plot(*step("water_content"), "k--", lw=2, label="true water content")
        ax.set_xlim(0, max(1.05, ax.get_xlim()[1])); ax.set_xlabel("NMR water content (-)")
        ax.set_title("NMR log (dotted: no borehole)", fontsize=10)
        ax.legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=1)
        ax.grid(True, ls="--", alpha=0.4)
    if have_gr:
        ax = axes[k]
        for case, ls, col in (("borehole", "-", "#2ca02c"), ("no borehole", ":", "#2ca02c")):
            d = results[(results.tool == "Gamma ray") & (results.case == case)].sort_values("depth")
            if case == "borehole":
                ax.fill_betweenx(-d.depth, d.value - d.error, d.value + d.error, color=col, alpha=0.2, lw=0)
            ax.plot(d.value, -d.depth, ls=ls, color=col, lw=1.6, label=f"Gamma ray ({case})")
        ax.plot(*step("api"), "k--", lw=2, label="true (16K+8U+4Th)")
        ax.set_xlim(left=0); ax.set_xlabel("Gamma ray (API)")
        ax.set_title("Gamma-ray log", fontsize=10)
        ax.legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=1)
        ax.grid(True, ls="--", alpha=0.4)
    for ax in axes:
        for b in tbl["bottom"].values[:-1]:
            ax.axhline(-b, color="gray", ls=":", lw=0.8)
    fig.tight_layout()
    return fig


def plot_t2(spectra, tbl):
    """T2 distribution against depth, per shell (columns) with and without the borehole (rows)."""
    shells = sorted({d for _, d in spectra})
    fig, axes = plt.subplots(2, len(shells), figsize=(4.6 * len(shells), 9), sharey=True, squeeze=False)
    # colour scale from the formation alone (no-borehole panels): open borehole water,
    # one strong T2 peak, would otherwise hide the formation; it saturates instead
    vmax = max((v[2].max() for (case, _), v in spectra.items() if case == "no borehole"), default=0.0) \
        or max(v[2].max() for v in spectra.values()) or 1.0
    for j, d in enumerate(shells):
        for i, case in enumerate(("borehole", "no borehole")):
            ax = axes[i, j]
            g, z, f = spectra[(case, d)]
            edges = np.sqrt(g[1:] * g[:-1])
            edges = np.concatenate([[g[0] ** 2 / edges[0]], edges, [g[-1] ** 2 / edges[-1]]])
            zs = np.concatenate([[z[0] - (z[1] - z[0]) / 2], (z[1:] + z[:-1]) / 2, [z[-1] + (z[-1] - z[-2]) / 2]]) \
                if len(z) > 1 else np.array([z[0] - 0.05, z[0] + 0.05])
            pc = ax.pcolormesh(edges * 1e3, -zs, f, cmap="viridis", vmin=0, vmax=vmax, shading="flat")
            for cut in (3, 33):
                ax.axvline(cut, color="w", ls="--", lw=0.8)
            for b in tbl["bottom"].values[:-1]:
                ax.axhline(-b, color="w", ls=":", lw=0.8)
            ax.set_xscale("log")
            ax.set_title(f"shell {d * 100:.0f} cm, {case}", fontsize=10)
            if i == 1:
                ax.set_xlabel("T2 (ms)")
            if j == 0:
                ax.set_ylabel("Depth (m)")
    axes[0, 0].yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{-v + 0.0:g}"))
    fig.colorbar(pc, ax=axes, shrink=0.6, extend="max", label="water content per T2 bin (-)")
    fig.suptitle("T2 distributions (dashed: 3 ms clay-bound and 33 ms mobile-water cut-offs)", fontsize=11)
    return fig


def nmr_table(results, tbl, L):
    """NMR quantities at the middle of each unit (nearest station), per shell and case."""
    rows = []
    nmr = results[results["tool"].str.startswith("NMR")]
    for u in tbl.itertuples():
        zmid = 0.5 * (u.top + min(u.bottom, L))
        for (t, case), d in nmr.groupby(["tool", "case"], sort=False):
            r = d.loc[(d.depth - zmid).abs().idxmin()]
            rows.append({"unit": f"{u.id} ({zmid:g} m)", "shell": t.replace("NMR water ", ""), "case": case,
                         "water": r.water, "true water": u.water_content, "mobile": r.mobile,
                         "bound": r.clay_bound + r.capillary, "T2LM (ms)": r.t2lm_ms,
                         "K SDR (m/day)": r.K_sdr_m_day})
    return pd.DataFrame(rows)


def plot_effect(results):
    from geometry_class.study import borehole_effect
    eff = borehole_effect(results)
    tools = list(dict.fromkeys(eff["tool"]))
    style = tool_styles(tools)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for t in tools:
        d = eff[eff.tool == t].sort_values("depth")
        col, mk = style[t]
        ax.plot(d.ratio, -d.depth, color=col, lw=1.2 if t == "Gamma ray" else 1.5, label=t,
                marker=mk, ms=4, markevery=max(len(d) // 12, 1))
    ax.axvline(1, color="k", lw=0.8)
    ax.set_xlabel("borehole effect: value(with hole) / value(no hole)")
    ax.set_ylabel("Depth (m)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{-v + 0.0:g}"))
    ax.set_title("1 = no effect;  < 1 the hole makes the tool read low", fontsize=10)
    ax.legend(fontsize=8); ax.grid(True, ls="--", alpha=0.4)
    fig.tight_layout()
    return fig, eff


with tab4:
    if not units_ok:
        st.warning("Fix the unit geometry in tab 1 first.")
        st.stop()
    for i, uid in enumerate(units["id"]):
        ensure_unit(uid, i)
    ensure_backfill()
    tbl = unit_table(units)
    avail = availability()

    st.subheader("What to simulate")
    r1, r2, r3, r4 = st.columns(4)
    run_ert = r1.checkbox("ERT", value=False, disabled=not avail["ERT"][0], key="run_ert")
    r1.caption(("✅ " if avail["ERT"][0] else "⚠️ ") + avail["ERT"][1])
    r1.caption("3D finite elements: a few minutes and several GB of RAM.")
    run_em = r2.checkbox("EM", value=avail["EM"][0], disabled=not avail["EM"][0], key="run_em")
    r2.caption(("✅ " if avail["EM"][0] else "⚠️ ") + avail["EM"][1])
    r2.caption("Axisymmetric: about half a minute.")
    run_gr = r3.checkbox("Gamma ray", value=avail["Gamma"][0], disabled=not avail["Gamma"][0], key="run_gr")
    r3.caption(("✅ " if avail["Gamma"][0] else "⚠️ ") + avail["Gamma"][1])
    r3.caption("Monte Carlo (Geant4) on all CPU cores: ~10 s to a few minutes.")
    run_nmr = r4.checkbox("NMR", value=True, key="run_nmr")
    r4.caption("✅ NumPy / SciPy")
    r4.caption("Slim water-well probe: a few seconds.")
    if not avail["Gamma"][0] and "not compiled" in avail["Gamma"][1]:
        if r3.button("Compile the gamma-ray program now"):
            from geometry_class.gamma_geometry import build_gammalog
            with st.spinner("Compiling (about a minute) ..."):
                build_gammalog()
            availability.clear()
            st.rerun()

    inputs = dict(units=tbl.round(6).to_dict("list"), L=ss.L, D=ss.D,
                  fill=[ss.bf_type, ss.bf_water, ss.bf_res, ss.bf_K, ss.bf_U, ss.bf_Th],
                  ert=[arr, ert_sp, ert_mn, ert_step, ert_tool_d], em=[em_sp, em_f, em_step],
                  gr=[gr_step, gr_events, gr_cd, gr_cl, gr_td],
                  nmr=[nmr_f, nmr_td, nmr_md, nmr_br, nmr_ap, nmr_te, nmr_ne, nmr_noise, nmr_step, backfill_nmr()])
    key = json.dumps(inputs, sort_keys=True, default=str)
    bad_tools = (run_ert and not ert_sp) or (run_em and not (em_sp and em_f)) or (run_gr and gr_td >= ss.D) \
        or (run_nmr and not nmr_ok)
    if st.button("▶ Run", type="primary", disabled=not (run_ert or run_em or run_gr or run_nmr) or bad_tools):
        from geometry_class.study import run_em as do_em, run_ert as do_ert, run_gamma as do_gr, run_nmr as do_nmr
        parts, spectra = [], None
        with st.status("Simulating ...", expanded=True) as status:
            try:
                if run_nmr:
                    log, spectra = do_nmr(tbl, ss.L, ss.D, backfill_nmr(), [f * 1e3 for f in nmr_f], nmr_step,
                                          progress=st.write, tool_diameter=nmr_td, magnet_diameter=nmr_md,
                                          remanence=nmr_br, aperture=nmr_ap, echo_spacing=nmr_te * 1e-3,
                                          n_echoes=int(nmr_ne), noise=nmr_noise / 100)
                    parts.append(log)
                if run_em:
                    parts.append(do_em(tbl, ss.L, ss.D, ss.bf_res, em_sp, em_f, em_step, progress=st.write))
                if run_gr:
                    parts.append(do_gr(tbl, ss.L, ss.D,
                                       dict(material=backfill_material(), K=ss.bf_K, U=ss.bf_U, Th=ss.bf_Th),
                                       gr_step, gr_events, gr_cd, gr_cl, gr_td, progress=st.write))
                if run_ert:
                    slots, n_slots = ert_slots()
                    if not slots.acquire(blocking=False):
                        st.write(f"Waiting for a free ERT slot: this server runs {n_slots} ERT "
                                 f"simulation{'s' if n_slots > 1 else ''} at a time ...")
                        slots.acquire()
                    try:
                        parts.append(do_ert(tbl, ss.L, ss.D, ss.bf_res, arr.lower(), ert_sp, ert_step, ert_mn,
                                            ert_tool_d, progress=st.write))
                    finally:
                        slots.release()
                status.update(label="Done", state="complete", expanded=False)
                ss.results, ss.results_key = pd.concat(parts, ignore_index=True), key
                ss.results_tbl, ss.results_t2 = tbl, spectra
            except Exception as e:  # noqa: BLE001
                status.update(label="Simulation failed", state="error")
                st.exception(e)

    if ss.results is not None:
        if ss.results_key != key:
            st.info("The inputs changed since this run: the plots show the previous results. Press Run to update.")
        res, rtbl = ss.results, ss.results_tbl
        st.subheader("Logs")
        st.pyplot(plot_logs(res, rtbl, ss.L, ss.D))
        st.subheader("Borehole effect")
        fig, eff = plot_effect(res)
        c1, c2 = st.columns([3, 2])
        c1.pyplot(fig)
        mids = []
        for u in rtbl.itertuples():
            zmid = 0.5 * (u.top + min(u.bottom, ss.L))
            for t in dict.fromkeys(eff.tool):
                d = eff[eff.tool == t]
                if len(d):
                    j = (d.depth - zmid).abs().idxmin()
                    mids.append(dict(unit=f"{u.id} ({zmid:g} m)", tool=t, ratio=round(d.ratio[j], 3)))
        c2.markdown("Ratio at the middle of each unit (nearest station)")
        c2.dataframe(pd.DataFrame(mids).pivot_table(index="unit", columns="tool", values="ratio", sort=False),
                     width="stretch")
        if ss.get("results_t2"):
            st.subheader("NMR: T2 distributions")
            st.pyplot(plot_t2(ss.results_t2, rtbl))
            st.markdown("NMR at the middle of each unit (nearest station). Water contents are volume fractions; "
                        "**mobile** water has T2 > 33 ms, **bound** water < 33 ms. K SDR = 4 φ⁴ T2LM² mD "
                        "(sandstone constant, not calibrated for your aquifer).")
            st.dataframe(nmr_table(res, rtbl, ss.L).round(3), hide_index=True, width="stretch")
        buf = io.StringIO()
        res.to_csv(buf, index=False)
        st.download_button("Download the logs (CSV)", buf.getvalue(), "pyborehole_logs.csv", "text/csv")
