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


def borehole_normal_rhoa(am, radius, rho_hole, rho_formation, depth=None):
    """Normal (pole-pole A-M, on the axis) apparent resistivity in an infinite borehole.

    ``depth`` (of the AM midpoint) adds the surface image of A to first order,
    matching a geometric factor that includes the image, as PyGIMLi's does.
    """
    v = borehole_potential(am, radius, rho_hole, rho_formation)
    g = 1.0 / am
    if depth is not None:
        v += rho_formation / (4.0 * np.pi * 2.0 * depth)
        g += 1.0 / (2.0 * depth)
    return 4.0 * np.pi * v / g


def borehole_lateral_rhoa(ao, mn, radius, rho_hole, rho_formation, depth=None):
    """Lateral (pole-dipole A-M-N, on the axis) apparent resistivity in an infinite borehole."""
    am, an = ao - mn / 2.0, ao + mn / 2.0
    v = borehole_potential(am, radius, rho_hole, rho_formation) \
        - borehole_potential(an, radius, rho_hole, rho_formation)
    g = 1.0 / am - 1.0 / an
    if depth is not None:
        # images of A at depth zA: distances to M and N are ~2 zA -/+ ...; first-order difference
        za = depth + ao
        img = 1.0 / (2.0 * za - am) - 1.0 / (2.0 * za - an)
        v += rho_formation / (4.0 * np.pi) * img
        g += img
    return 4.0 * np.pi * v / g


def ring_on_mandrel_potential(z, tool_radius, radius, rho_hole, rho_formation, current=1.0):
    """Potential (V) on an insulating tool body at axial distance ``z`` from a ring electrode.

    Infinite borehole (radius ``radius``, resistivity ``rho_hole``) in a
    homogeneous formation, with a centred insulating mandrel of radius
    ``tool_radius``; the current electrode and the measuring point are rings
    on the mandrel surface.  With V(r,z) = (1/pi) int V~(r,l) cos(l z) dl:

        hole:      V~ = B I0(l r) + C K0(l r)
        formation: V~ = D K0(l r)
        r = a:  sigma_h dV~/dr = -I / (2 pi a)        (all current leaves the mandrel)
        r = R:  V~ and sigma dV~/dr continuous
    """
    from scipy.special import i0e, i1e, k0e, k1e

    a, R = tool_radius, radius
    sh, sf = 1.0 / rho_hole, 1.0 / rho_formation
    q = current / (2.0 * np.pi * a)

    def vt(l):
        # Scaled Bessels, written so that every term stays finite for large l:
        # B' = B e^{lR} and C' = C e^{-la} absorb the exponential growth.
        xa, xR = l * a, l * R
        ea = np.exp(-(xR - xa))  # e^{-l(R-a)} < 1
        # Unknowns (B', C', D') with D' = D e^{-lR}:
        # sh l [B' i1e(xa) ea - C' k1e(xa)] = -q
        # B' i0e(xR) + C' k0e(xR) ea = D' k0e(xR)
        # sh [B' i1e(xR) - C' k1e(xR) ea] = -sf D' k1e(xR)
        m = np.array([
            [sh * l * i1e(xa) * ea, -sh * l * k1e(xa), 0.0],
            [i0e(xR), k0e(xR) * ea, -k0e(xR)],
            [sh * i1e(xR), -sh * k1e(xR) * ea, sf * k1e(xR)],
        ])
        b_, c_, _ = np.linalg.solve(m, np.array([-q, 0.0, 0.0]))
        # V~(a) = B I0(xa) + C K0(xa) = B' i0e(xa) e^{-l(R-a)} + C' k0e(xa)
        return b_ * i0e(xa) * ea + c_ * k0e(xa)

    # V~ ~ -ln(l) at small l and ~ 1/l at large l: regular quadrature up to l1,
    # then an oscillatory Fourier integral (QAWF) to infinity
    l1 = 20.0 / R
    head, _ = quad(lambda l: vt(l) * np.cos(l * z), 0.0, l1, limit=2000)
    tail, _ = quad(vt, l1, np.inf, weight="cos", wvar=z, limlst=200)
    return (head + tail) / np.pi


def tool_normal_rhoa(am, tool_radius, radius, rho_hole, rho_formation, depth=None):
    """Normal (ring electrodes on an insulating tool body) apparent resistivity.

    The tool constant is calibrated in a homogeneous 1 Ohm.m earth with the
    same tool, as in ``Geometry.run``; ``depth`` (of the AM midpoint) adds the
    surface image of A to first order in both runs.
    """
    img = 0.0 if depth is None else 1.0 / (4.0 * np.pi * 2.0 * depth)
    r = ring_on_mandrel_potential(am, tool_radius, radius, rho_hole, rho_formation) + rho_formation * img
    r_cal = ring_on_mandrel_potential(am, tool_radius, radius, 1.0, 1.0) + img
    return r / r_cal


def tool_lateral_rhoa(ao, mn, tool_radius, radius, rho_hole, rho_formation, depth=None):
    """Lateral (ring electrodes on an insulating tool body) apparent resistivity, calibrated like ``tool_normal_rhoa``."""
    am, an = ao - mn / 2.0, ao + mn / 2.0
    img = 0.0
    if depth is not None:
        za = depth + ao
        img = (1.0 / (2.0 * za - am) - 1.0 / (2.0 * za - an)) / (4.0 * np.pi)
    v = lambda d, rh, rf: ring_on_mandrel_potential(d, tool_radius, radius, rh, rf)
    r = v(am, rho_hole, rho_formation) - v(an, rho_hole, rho_formation) + rho_formation * img
    r_cal = v(am, 1.0, 1.0) - v(an, 1.0, 1.0) + img
    return r / r_cal
