"""Diagnose both mesh types vs homogeneous analytical solution.

For a VMD in a homogeneous full-space:
  Im(H_z_sec) = omega*mu0*sigma / (4*pi*s)
  => rho_a should equal rho_true exactly (in LIN limit).
"""
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import warnings; warnings.filterwarnings('ignore')
import numpy as np
from geometry_class import EMGeometry

MU0 = 4*np.pi*1e-7
RHO_BG = 50.0   # Ohm.m  (homogeneous earth)
FREQ   = 400.0   # Hz
SPACINGS = [0.5, 1.0]
omega = 2*np.pi*FREQ

def analytical_rho_a(rho, s, freq):
    """LIN apparent resistivity in homogeneous full-space (VMD, k=4pi)."""
    sigma = 1.0/rho
    omega = 2*np.pi*freq
    h_imag = omega*MU0*sigma / (4*np.pi*s)   # positive sign (McNeill convention)
    k = 4*np.pi   # VMD full-space
    sigma_a = (k * s) / (MU0 * omega) * h_imag
    return 1.0/sigma_a   # should equal rho exactly

print("=== Analytical check ===")
for s in SPACINGS:
    ra = analytical_rho_a(RHO_BG, s, FREQ)
    print(f"  s={s:.1f}m  f={FREQ:.0f}Hz  rho_a = {ra:.4f} Ohm.m  (expected {RHO_BG})")

# ----------------------------------------------------------
# Run with NO LAYERS (homogeneous background only)
# ----------------------------------------------------------
no_layers = None   # disable layer painting

common = dict(
    borehole_length=10., borehole_diameter=0.2,
    layer_1d_geometry=no_layers,    # <-- pure homogeneous
    background_resistivity=RHO_BG,
    frequencies=[FREQ],
    coil_spacings=SPACINGS,
    coil_orientation='z',
    measuring_spacing=5.,
    core_cell_size=0.1, n_padding=6, domain_radius=50.,
)

print('\n=== Homogeneous CylindricalMesh ===')
em_cyl = EMGeometry(**common, mesh_type='cylindrical')
df_cyl = em_cyl.run(verbose=False)
print(df_cyl[['depth','spacing','apparent_resistivity']].to_string(index=False))

print('\n=== Homogeneous TreeMesh ===')
em_tree = EMGeometry(**common, mesh_type='tree')
df_tree = em_tree.run(verbose=False)
print(df_tree[['depth','spacing','apparent_resistivity']].to_string(index=False))

# ----------------------------------------------------------
# Also print raw B-field for layered model (first depth only)
# ----------------------------------------------------------
print('\n=== Layered model: raw dpred at depth=0 (debug) ===')
layers = np.array([[0.,5.,100.],[5.,10.,10.],[10.,15.,200.]])
import simpeg.electromagnetics.frequency_domain as fdem
import discretize as discretize_mod
import simpeg.maps as _maps

for label, mesh_type in [('cyl', 'cylindrical'), ('tree', 'tree')]:
    em = EMGeometry(
        borehole_length=15., borehole_diameter=0.2,
        layer_1d_geometry=layers, background_resistivity=50.,
        frequencies=[400.], coil_spacings=[0.5, 1.0],
        coil_orientation='z', measuring_spacing=5.,
        core_cell_size=0.1, n_padding=6, domain_radius=50.,
        mesh_type=mesh_type,
    )
    if mesh_type == 'cylindrical':
        mesh = em.make_mesh_3d()
    else:
        mesh = em._make_local_tree_mesh(0.0)
    sigma = em._paint_conductivity(mesh)
    print(f'\n  [{label}] mesh nC={mesh.nC}, sigma min={sigma.min():.4g} max={sigma.max():.4g} S/m')
    # Check z-coordinate column
    z_cc = mesh.gridCC[:, 2]
    print(f'  [{label}] gridCC[:,2] range: {z_cc.min():.2f} to {z_cc.max():.2f} m')
    # Check how many cells in each layer
    for top,bot,rho in layers:
        n = np.sum((z_cc > -bot) & (z_cc <= -top))
        print(f'  [{label}] layer {top:.0f}-{bot:.0f}m ({rho:.0f} Ohm.m): {n} cells')

    survey = em._build_survey(np.array([0.0]))
    sim = fdem.Simulation3DElectricField(
        mesh=mesh, survey=survey,
        sigmaMap=_maps.IdentityMap(nP=mesh.nC),
    )
    dpred = sim.dpred(sigma)
    print(f'  [{label}] dpred (depth=0): {dpred}')
