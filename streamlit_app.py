"""Entry point of the PyBorehole Streamlit app (also for Streamlit Community Cloud).

    streamlit run streamlit_app.py

The app itself is geometry_class/app.py; after ``pip install "pyborehole[app]"``
the command ``pyborehole-app`` starts it as well.
"""
import os
import runpy

runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "geometry_class", "app.py"),
               run_name="__main__")
