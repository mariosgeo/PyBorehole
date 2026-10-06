"""
analytic.py
===========

Closed-form / semi-analytic reference solutions used to validate the
numerical ERT and EM simulators.

* ``borehole_potential`` / ``borehole_wenner_rhoa`` -- DC potential on the
  axis of an infinitely long fluid- (or backfill-) filled borehole in a
  homogeneous full space, point source on the axis (classic Fourier-Bessel
  solution, e.g. Dakhnov 1962; Kunz & Moran 1958).
* ``fullspace_hs`` -- secondary magnetic field of a magnetic dipole in a
  homogeneous conducting full space, coaxial (z) or coplanar (x/y)
  receiver, in SimPEG's e^{+i omega t} convention.
"""

import numpy as np
from scipy.integrate import quad
from scipy.special import i0e, i1e, k0e, k1e

MU0 = 4e-7 * np.pi


def borehole_potential(d, radius, rho_hole, rho_formation, current=1.0):
    """Potential (V) on the borehole axis at distance ``d`` from a point source on the axis.

    Inside the hole  V = I rho_m / (4 pi) [ 1/d + int_0^inf A(l) cos(l d) dl ]  with
    A(l) = (2/pi) (1-s) K0 K1 / (I1 K0 + s I0 K1),  s = rho_m / rho_t,  Bessels at l*radius.
    """
    s = rho_hole / rho_formation

    def a_of_l(l):
        x = l * radius
        # Exponentially scaled Bessels: I*K products are scale-free, K0*K1 carries exp(-2x)
        num = (1.0 - s) * k1e(x) * k0e(x) * np.exp(-2.0 * x)
        den = i1e(x) * k0e(x) + s * i0e(x) * k1e(x)
        return (2.0 / np.pi) * num / den

    integral, _ = quad(lambda l: a_of_l(l) * np.cos(l * d), 0.0, 60.0 / radius, limit=2000)
    return current * rho_hole / (4.0 * np.pi) * (1.0 / d + integral)


def borehole_wenner_rhoa(a, radius, rho_hole, rho_formation):
    """Apparent resistivity of an on-axis Wenner array (A M N B, spacing ``a``), k = 4 pi a."""
    dv = 2.0 * borehole_potential(a, radius, rho_hole, rho_formation) \
        - 2.0 * borehole_potential(2.0 * a, radius, rho_hole, rho_formation)
    return 4.0 * np.pi * a * dv


def fullspace_hs(rho, freq, spacing, orientation="z"):
    """Secondary H (A/m) for a unit-moment dipole in a full space (SimPEG e^{+i w t} convention).

    Returns ``(H_secondary, H_primary)``.  ``orientation='z'`` is the coaxial
    geometry (both coils along the borehole axis); ``'x'``/``'y'`` is coplanar.
    """
    k = np.sqrt(-1j * 2.0 * np.pi * freq * MU0 / rho)   # k^2 = -i w mu sigma
    ks = k * spacing
    if orientation == "z":
        hp = 1.0 / (2.0 * np.pi * spacing**3)
        h = hp * (1.0 + 1j * ks) * np.exp(-1j * ks)
    else:
        hp = -1.0 / (4.0 * np.pi * spacing**3)
        h = hp * (1.0 + 1j * ks - ks**2) * np.exp(-1j * ks)
    return h - hp, hp
