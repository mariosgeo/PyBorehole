"""Entry point of the PyBorehole Streamlit app (also for Streamlit Community Cloud).

    streamlit run streamlit_app.py

The app itself is geometry_class/app.py; after ``pip install "pyborehole[app]"``
the command ``pyborehole-app`` starts it as well.
"""
import os
import runpy
import sys


def _ours(name):
    return name == "geometry_class" or name.startswith("geometry_class.")


def _source_mtime(module):
    path = getattr(module, "__file__", None)
    return os.path.getmtime(path) if path and os.path.exists(path) else None


# Streamlit keeps imported modules between reruns. When the repository is updated
# under a running app (Streamlit Community Cloud pulls every push), app.py would
# run against old, cached geometry_class modules and fail on new names. Each
# module remembers the mtime of its source when loaded; if any source changed
# (or a module was loaded without this check), all geometry_class modules are
# dropped and imported again.
if any(getattr(m, "_pyb_mtime", "unset") != _source_mtime(m)
       for n, m in list(sys.modules.items()) if _ours(n)):
    for name in [n for n in sys.modules if _ours(n)]:
        del sys.modules[name]

try:
    runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "geometry_class", "app.py"),
                   run_name="__main__")
finally:
    for name, module in list(sys.modules.items()):
        if _ours(name) and not hasattr(module, "_pyb_mtime"):
            try:
                module._pyb_mtime = _source_mtime(module)
            except Exception:
                pass
