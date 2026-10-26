# -*- coding: utf-8 -*-
"""Smoke test: both mesh_type values, local tree mesh, conductivity painting, and validation."""
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
from geometry_class import EMGeometry

layers = np.array([[0.,5.,100.],[5.,10.,10.],[10.,15.,200.]])

# ---- cylindrical (default) ----
em_cyl = EMGeometry(
    borehole_length=15., borehole_diameter=0.2,
    layer_1d_geometry=layers, background_resistivity=50.,
    frequencies=[400.], coil_spacings=[0.5, 1.0],
    coil_orientation='z', measuring_spacing=5.,
    core_cell_size=0.1, n_padding=6, domain_radius=50.,
    mesh_type='cylindrical',
)
print(f"cylindrical OK  mesh_type={em_cyl.mesh_type!r}")

# ---- tree ----
em_tree = EMGeometry(
    borehole_length=15., borehole_diameter=0.2,
    layer_1d_geometry=layers, background_resistivity=50.,
    frequencies=[400.], coil_spacings=[0.5, 1.0],
    coil_orientation='z', measuring_spacing=5.,
    core_cell_size=0.1, n_padding=6, domain_radius=50.,
    mesh_type='tree',
)
print(f"tree OK  mesh_type={em_tree.mesh_type!r}")

# build a local tree mesh for depth=5m
m = em_tree._make_local_tree_mesh(5.0)
print(f"Local TreeMesh at depth=5m: {m.nC:,} cells  shape={m.shape_cells}")

sigma = em_tree._paint_conductivity(m)
print(f"sigma: {sigma.min():.4g} - {sigma.max():.4g} S/m")

# bad mesh_type
try:
    EMGeometry(borehole_length=5., borehole_diameter=0.1,
               frequencies=[400.], coil_spacings=[1.],
               mesh_type='bad')
    print("ERROR: should have raised ValueError")
except ValueError as e:
    print(f"Validation OK: {e}")

# run() tree path (3 depths only, measuring_spacing=5m)
print("\nRunning tree simulation (3 depths) ...")
result = em_tree.run(verbose=True)
print(result[["depth","spacing","frequency","apparent_resistivity"]].to_string(index=False))

print("\nSMOKE TEST PASSED")
