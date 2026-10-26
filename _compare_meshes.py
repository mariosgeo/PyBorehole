"""Compare cylindrical vs tree mesh results for same EM geometry."""
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import warnings; warnings.filterwarnings('ignore')
import numpy as np
from geometry_class import EMGeometry

layers = np.array([[0.,5.,100.],[5.,10.,10.],[10.,15.,200.]])
common = dict(
    borehole_length=15., borehole_diameter=0.2,
    layer_1d_geometry=layers, background_resistivity=50.,
    frequencies=[400.], coil_spacings=[0.5, 1.0],
    coil_orientation='z', measuring_spacing=5.,
    core_cell_size=0.1, n_padding=6, domain_radius=50.,
)

print('=== CylindricalMesh ===')
em_cyl = EMGeometry(**common, mesh_type='cylindrical')
df_cyl = em_cyl.run(verbose=False)
print(df_cyl[['depth','spacing','apparent_resistivity']].to_string(index=False))

print('\n=== TreeMesh ===')
em_tree = EMGeometry(**common, mesh_type='tree')
df_tree = em_tree.run(verbose=True)
print(df_tree[['depth','spacing','apparent_resistivity']].to_string(index=False))

print('\n=== Ratio tree/cyl ===')
for _, row_c in df_cyl.iterrows():
    row_t = df_tree[(df_tree.depth==row_c.depth) & (df_tree.spacing==row_c.spacing)].iloc[0]
    ratio = row_t.apparent_resistivity / row_c.apparent_resistivity
    print(f'  depth={row_c.depth:.1f}m  s={row_c.spacing:.1f}m  '
          f'cyl={row_c.apparent_resistivity:.2f}  tree={row_t.apparent_resistivity:.2f}  ratio={ratio:.3f}')
