"""PyBorehole: borehole backfill effect on ERT, EM and natural gamma-ray logs.

The ERT class needs PyGIMLi (``pip install "pyborehole[ert]"`` or conda); the
rest of the package imports without it, so the EM and gamma-ray simulators can
be used on their own.
"""
try:
    from .geometry import Geometry, merge_protocols
except ImportError as _err:          # PyGIMLi (or PyVista) not installed
    _ert_error = _err

    def _missing(*args, **kwargs):
        raise ImportError(
            "The ERT simulator needs PyGIMLi and PyVista, which could not be imported "
            f"({_ert_error}). Install them, see docs/INSTALL.md.") from _ert_error

    Geometry = merge_protocols = _missing

from .em_geometry import EMGeometry
from .gamma_geometry import GammaGeometry, GammaMedium, build_gammalog
from .gamma_openmc import OpenMCGammaGeometry

__version__ = "0.3.1"
