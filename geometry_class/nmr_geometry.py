# -*- coding: utf-8 -*-
"""
nmr_geometry.py
===============

Borehole nuclear magnetic resonance (NMR) log of a slim water-well probe
(Javelin-type) in a backfilled borehole through a layered earth.

The probe
---------
A long permanent magnet, magnetised across its axis, makes a static field
that falls off with the distance r from the axis (exact for a long magnet):

    B0(r) = Br a^2 / (2 r^2)        Br: remanence, a: magnet radius

The probe excites the hydrogen nuclei where the Larmor frequency matches its
operating frequency f, i.e. on a thin cylindrical shell of radius

    r_f = sqrt(Br a^2 / (2 B_f)),   B_f = 2 pi f / gamma

Several frequencies give several shells (depths of investigation). The shell
thickness follows from the excitation bandwidth, dr = bandwidth / (gamma/2pi
G), with the field gradient G = 2 B_f / r_f. The vertical extent of the
sensitive region is the length of the antenna (``aperture``).

The signal
----------
A CPMG echo train with echo spacing TE. Every material on the shell adds

    water content * sum_k p_k exp(-t / T2app_k),
    1 / T2app = 1 / T2 + D (gamma G TE)^2 / 12

with its T2 distribution p_k (database: log-normal main mode plus a
clay-bound part at 2 ms) and the diffusion of water (D) in the gradient.
Gaussian noise is added, and the echo train is inverted for the T2
distribution (non-negative least squares with Tikhonov regularisation).

From the distribution: total water content (the NMR porosity below the water
table), clay-bound (T2 < 3 ms), capillary-bound (3-33 ms) and mobile water
(> 33 ms), the T2 log-mean, and the SDR permeability estimate
k = c phi^4 T2LM^2 (Kenyon et al. 1988; c = 4 mD/ms^2 is the sandstone
value - calibrate it for your aquifer).

The borehole effect
-------------------
The shell is thin, so the effect is sharp: where the shell lies inside the
borehole, the probe reads the fill (an open water-filled hole: 100 % water
with long T2). Javelin-type probes therefore need holes narrower than their
shells. ``run(include_borehole=False)`` fills the hole with formation (the
no-borehole reference), as for ERT, EM and the gamma ray.

The default probe (60 mm, 25 mm NdFeB magnet, 450 and 310 kHz -> shells of
about 19 and 23 cm diameter, 0.5 m antenna, TE 1.5 ms) is typical of slim
water-well NMR probes, not a copy of a commercial design: set the values of
your probe.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd
from scipy.optimize import nnls

GAMMA_H = 2.675222e8      # gyromagnetic ratio of 1H (rad/s/T)
D_WATER = 2.0e-9          # self-diffusion of water near 20 C (m^2/s)
T2_CBW = 2e-3             # T2 of clay-bound water (s)
CUT_CBW, CUT_BVI = 3e-3, 33e-3   # T2 cut-offs: clay-bound / capillary-bound / mobile (s)
MD_TO_M_PER_DAY = 8.36e-4         # hydraulic conductivity of 1 mD for water at 20 C (m/day)


@dataclass
class NMRMedium:
    """Water content (volume fraction) and T2 distribution of a material.

    t2lm: log-mean T2 of the main mode (ms); t2_sigma: its spread (decades);
    cbw: fraction of the water that is clay-bound (T2 = 2 ms).
    """
    water: float
    t2lm: float = 300.0
    t2_sigma: float = 0.3
    cbw: float = 0.0

    def amplitudes(self, t2_grid: np.ndarray) -> np.ndarray:
        """Water content per T2 bin (sums to ``water``)."""
        lg = np.log10(t2_grid)
        main = np.exp(-0.5 * ((lg - np.log10(self.t2lm * 1e-3)) / max(self.t2_sigma, 1e-3)) ** 2)
        main /= main.sum()
        cb = np.exp(-0.5 * ((lg - np.log10(T2_CBW)) / 0.1) ** 2)
        cb /= cb.sum()
        return self.water * ((1.0 - self.cbw) * main + self.cbw * cb)

    def fractions(self) -> dict:
        """True clay-bound, capillary-bound and mobile water (volume fractions)."""
        g = np.logspace(-4, 1.5, 2000)
        a = self.amplitudes(g)
        return dict(water=self.water, clay_bound=a[g < CUT_CBW].sum(),
                    capillary=a[(g >= CUT_CBW) & (g < CUT_BVI)].sum(), mobile=a[g >= CUT_BVI].sum())


def _medium(m: Union[NMRMedium, dict]) -> NMRMedium:
    return m if isinstance(m, NMRMedium) else NMRMedium(**m)


@dataclass
class NMRGeometry:
    """
    Slim borehole NMR probe in a layered earth.

    Parameters
    ----------
    layers : list of (top, bottom, NMRMedium or dict)
        Depths positive down (m); the last layer extends downwards.
    borehole_length, borehole_diameter : float (m)
    fill : NMRMedium or dict
        Borehole fill (default: open water).
    frequencies : operating frequencies (Hz), one sensitive shell each.
    tool_diameter, magnet_diameter (m), remanence (T): the probe and its magnet.
    bandwidth : excitation bandwidth (Hz), sets the shell thickness.
    aperture : vertical length of the sensitive region (m).
    echo_spacing (s), n_echoes : CPMG train.
    noise : standard deviation of the noise per echo, in water-content units
        (after stacking); 0 = noise-free.
    measuring_spacing, top_clearance, bottom_clearance : stations (m).
    regularization : Tikhonov weight of the T2 inversion.
    sdr_c : SDR constant (mD/ms^2).
    """
    layers: Sequence
    borehole_length: float
    borehole_diameter: float
    fill: Union[NMRMedium, dict] = field(default_factory=lambda: NMRMedium(1.0, 2500.0, 0.05, 0.0))
    frequencies: Sequence[float] = (450e3, 310e3)
    tool_diameter: float = 0.060
    magnet_diameter: float = 0.025
    remanence: float = 1.2
    bandwidth: float = 10e3
    aperture: float = 0.5
    echo_spacing: float = 1.5e-3
    n_echoes: int = 1000
    noise: float = 0.02
    measuring_spacing: float = 0.1
    top_clearance: float = 0.25
    bottom_clearance: float = 0.25
    regularization: float = 10.0
    sdr_c: float = 4.0
    diffusion: float = D_WATER
    seed: int = 1
    t2_grid: np.ndarray = field(default_factory=lambda: np.logspace(np.log10(5e-4), np.log10(5.0), 80))

    def __post_init__(self) -> None:
        self.layers = [(float(t), float(b), _medium(m)) for t, b, m in self.layers]
        self.fill = _medium(self.fill)
        self.frequencies = [float(f) for f in self.frequencies]
        for s in self.shells().itertuples():
            if s.radius <= self.tool_diameter / 2.0:
                raise ValueError(f"The {s.frequency / 1e3:g} kHz shell (diameter {s.diameter * 100:.1f} cm) "
                                 "lies inside the probe: lower the frequency or use a stronger magnet.")

    # ------------------------------------------------------------------
    @property
    def _c(self) -> float:
        """Br a^2 / 2 (T m^2): B0(r) = C / r^2."""
        return self.remanence * (self.magnet_diameter / 2.0) ** 2 / 2.0

    def shells(self) -> pd.DataFrame:
        """Sensitive shell of every frequency: field, radius, diameter, gradient, thickness."""
        rows = []
        for f in self.frequencies:
            b0 = 2 * np.pi * f / GAMMA_H
            r = np.sqrt(self._c / b0)
            g = 2 * b0 / r
            rows.append(dict(frequency=f, field=b0, radius=r, diameter=2 * r, gradient=g,
                             thickness=self.bandwidth / (GAMMA_H / (2 * np.pi) * g)))
        return pd.DataFrame(rows)

    def stations(self) -> np.ndarray:
        z0, z1 = self.top_clearance, self.borehole_length - self.bottom_clearance
        return np.round(np.arange(z0, z1 + 1e-9, self.measuring_spacing), 6)

    def echo_times(self) -> np.ndarray:
        return self.echo_spacing * np.arange(1, self.n_echoes + 1)

    def _medium_at(self, r: float, depth: float, include_borehole: bool) -> Optional[NMRMedium]:
        if depth < 0:
            return None                                   # air
        if include_borehole and depth < self.borehole_length and r < self.borehole_diameter / 2.0:
            return self.fill
        for top, bot, m in self.layers:
            if depth < bot:
                return m
        return self.layers[-1][2]

    def _station_weights(self, depth: float, shell, include_borehole: bool) -> dict:
        """Share of the sensitive region in each material (by id), over the shell
        thickness (weighted by r) and the aperture (uniform)."""
        rs = shell.radius + shell.thickness * np.linspace(-0.5, 0.5, 5)
        zs = depth + self.aperture * (np.arange(50) + 0.5) / 50 - self.aperture / 2
        w = {}
        for r in rs:
            for z in zs:
                m = self._medium_at(r, z, include_borehole)
                key = id(m) if m is not None else None
                w[key] = w.get(key, 0.0) + r
        tot = sum(w.values())
        return {k: v / tot for k, v in w.items()}

    def _decay(self, medium: NMRMedium, gradient: float) -> np.ndarray:
        """Noise-free echo train of a unit volume of the medium on a shell."""
        t2 = self.t2_grid
        rate = 1.0 / t2 + self.diffusion * (GAMMA_H * gradient * self.echo_spacing) ** 2 / 12.0
        return np.exp(-np.outer(self.echo_times(), rate)) @ medium.amplitudes(t2)

    def _inverter(self):
        """Kernel and regularised NNLS solver for the T2 distribution."""
        k = np.exp(-np.outer(self.echo_times(), 1.0 / self.t2_grid))
        lam = self.regularization * max(self.noise, 1e-3) * np.sqrt(self.n_echoes) / len(self.t2_grid) ** 0.5
        a = np.vstack([k, lam * np.eye(len(self.t2_grid))])
        zero = np.zeros(len(self.t2_grid))
        return lambda d: nnls(a, np.concatenate([d, zero]), maxiter=50 * len(self.t2_grid))[0]

    def _summarise(self, f: np.ndarray) -> dict:
        g = self.t2_grid
        water = f.sum()
        t2lm = np.exp((f * np.log(g)).sum() / water) if water > 1e-6 else np.nan
        k_md = self.sdr_c * water ** 4 * (t2lm * 1e3) ** 2 if water > 1e-6 else 0.0
        return dict(water=water, clay_bound=f[g < CUT_CBW].sum(), capillary=f[(g >= CUT_CBW) & (g < CUT_BVI)].sum(),
                    mobile=f[g >= CUT_BVI].sum(), t2lm_ms=t2lm * 1e3, k_sdr_md=k_md,
                    K_sdr_m_day=k_md * MD_TO_M_PER_DAY)

    # ------------------------------------------------------------------
    def run(self, include_borehole: bool = True, return_spectra: bool = False):
        """
        Simulate the log, one row per station and shell: depth, frequency,
        shell diameter, water (total NMR water content), clay_bound, capillary,
        mobile, t2lm_ms, k_sdr_md, K_sdr_m_day, and the true values of the
        formation at the station (true_water, true_mobile).
        With ``return_spectra`` also a dict {shell diameter: (t2_grid, depths,
        distributions)}.
        """
        rng = np.random.default_rng(self.seed + (0 if include_borehole else 1))
        media = {id(m): m for m in [self.fill] + [m for _, _, m in self.layers]}
        invert = self._inverter()
        rows, spectra = [], {}
        stations = self.stations()
        for shell in self.shells().itertuples():
            decays = {k: self._decay(m, shell.gradient) for k, m in media.items()}
            dists = []
            for z in stations:
                w = self._station_weights(z, shell, include_borehole)
                signal = sum(wt * decays[k] for k, wt in w.items() if k is not None)
                if not np.ndim(signal):
                    signal = np.zeros(self.n_echoes)
                data = signal + (rng.normal(0.0, self.noise, self.n_echoes) if self.noise > 0 else 0.0)
                f = invert(data)
                dists.append(f)
                truth = self._medium_at(1e9, z, False).fractions()
                rows.append(dict(depth=z, frequency=shell.frequency, diameter=shell.diameter,
                                 **self._summarise(f), true_water=truth["water"], true_mobile=truth["mobile"]))
            spectra[round(shell.diameter, 4)] = (self.t2_grid, stations, np.array(dists))
        log = pd.DataFrame(rows)
        return (log, spectra) if return_spectra else log
