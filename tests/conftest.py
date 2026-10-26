"""Shared test setup: make the TetGen executable of the active Python
environment findable, as run_combined.py does, so the ERT tests also work
when the environment's Python is called without activating it."""
import os
import sys

for _bin in (os.path.join(sys.prefix, "Library", "bin"), os.path.join(sys.prefix, "bin")):
    if os.path.isdir(_bin) and _bin not in os.environ.get("PATH", ""):
        os.environ["PATH"] = _bin + os.pathsep + os.environ.get("PATH", "")
