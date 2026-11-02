---
title: PyBorehole
emoji: 🕳️
colorFrom: blue
colorTo: yellow
sdk: docker
app_port: 8501
pinned: false
license: mit
short_description: Borehole backfill effect on ERT, EM and gamma-ray logs
---

# PyBorehole

Interactive app for the borehole and backfill effect on resistivity (ERT),
induction (EM) and natural gamma-ray logs, with 3D simulations in PyGIMLi,
SimPEG and Geant4. All three simulators run here.

Developed jointly by Metinco, Deltares and the Aristotle University of
Thessaloniki. Code, documentation and local installation:
<https://github.com/mariosgeo/PyBorehole>.

Notes for visitors

- ERT is a 3D finite-element solve: a few minutes per run and a lot of memory,
  so only one ERT run goes at a time and further visitors wait in a queue.
- The free Space sleeps when nobody has used it for a while; the first visit
  then takes a minute or so to wake it up.
