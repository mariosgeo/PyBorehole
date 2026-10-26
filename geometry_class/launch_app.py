"""Command ``pyborehole-app``: start the Streamlit app (geometry_class/app.py)."""
import os
import subprocess
import sys


def main(argv=None):
    try:
        import streamlit  # noqa: F401
    except ImportError:
        sys.exit('The app needs Streamlit: pip install "pyborehole[app]"')
    app = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app.py")
    args = sys.argv[1:] if argv is None else list(argv)
    sys.exit(subprocess.call([sys.executable, "-m", "streamlit", "run", app, *args]))
