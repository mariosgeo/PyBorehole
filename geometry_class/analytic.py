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


def borehole_coaxial_hs(rho_hole, rho_formation, radius, freq, spacing):
    """Secondary Hz (A/m) for coaxial coils on the axis of an infinite borehole.

    Unit-moment z-dipole on the axis of a borehole (radius ``radius``,
    resistivity ``rho_hole``) in a homogeneous formation, SimPEG's
    e^{+i w t} convention.  Returned field is relative to the free-space
    primary, like ``fullspace_hs``.

    Inside the hole E_phi = C int [-nu1 K1(nu1 r)/(2 pi^2) + A(l) I1(nu1 r)] cos(l z) dl
    (first term = full space of the hole fill); continuity of E_phi and H_z at
    r = radius gives
        A = nu1/(2 pi^2) (nu2 K0(nu2 a) K1(nu1 a) - nu1 K0(nu1 a) K1(nu2 a))
                        / (nu1 I0(nu1 a) K1(nu2 a) + nu2 K0(nu2 a) I1(nu1 a)),
    and on the axis  dHz(z) = -int A(l) nu1 cos(l z) dl,  nu_i = sqrt(l^2 + i w mu sigma_i).
    """
    from scipy.special import ive, kve

    omega = 2.0 * np.pi * freq
    a = radius

    def integrand(l):
        nu1 = np.sqrt(l**2 + 1j * omega * MU0 / rho_hole)
        nu2 = np.sqrt(l**2 + 1j * omega * MU0 / rho_formation)
        x1, x2 = nu1 * a, nu2 * a
        # Scaled Bessels: K(x) = kve e^{-x}, I(x) = ive e^{Re x}
        num = nu2 * kve(0, x2) * kve(1, x1) - nu1 * kve(0, x1) * kve(1, x2)
        den = nu1 * ive(0, x1) * kve(1, x2) + nu2 * kve(0, x2) * ive(1, x1)
        amp = nu1 / (2.0 * np.pi**2) * np.exp(-x1 - x1.real) * num / den
        return -amp * nu1 * np.cos(l * spacing)

    upper = 60.0 / a
    re, _ = quad(lambda l: integrand(l).real, 0.0, upper, limit=4000)
    im, _ = quad(lambda l: integrand(l).imag, 0.0, upper, limit=4000)
    hs_fill, _ = fullspace_hs(rho_hole, freq, spacing, "z")
    return hs_fill + (re + 1j * im)
