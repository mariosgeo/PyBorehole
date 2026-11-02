# PyBorehole app with all three simulators (ERT, EM, gamma ray) in one image.
#
#   docker build -t pyborehole .
#   docker run -p 8501:8501 pyborehole        then open http://localhost:8501
#
# For a VPS, Google Cloud Run, Azure Container Apps, Render, Fly.io, ...
# ERT memory: the 3-unit example peaked above 7.6 GB in Linux (killed in an 8 GB
# container) and at 23 GB footprint on macOS, so give the container 16 GB or more.
# Hugging Face Spaces: see deploy/huggingface/ (same steps, source from GitHub).
FROM python:3.12-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    GEANT4_DIR=/opt/geant4 \
    GAMMALOG_EXE=/opt/gammalog/gammalog \
    MPLBACKEND=Agg

# 1. System packages, Python packages, Geant4 (cached unless requirements.txt changes)
COPY requirements.txt deploy/container_setup.sh /tmp/setup/
RUN bash /tmp/setup/container_setup.sh deps /tmp/setup

# 2. The code, and the Geant4 gamma-ray program compiled from it
COPY . /app
RUN bash /app/deploy/container_setup.sh build /app

# Run as an ordinary user (uid 1000, as Hugging Face Spaces expect)
RUN useradd -m -u 1000 user && chown -R user /app
USER user
ENV HOME=/home/user MPLCONFIGDIR=/home/user/.cache/matplotlib
WORKDIR /app

EXPOSE 8501
HEALTHCHECK CMD curl -fs http://localhost:8501/_stcore/health || exit 1
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0", \
     "--server.headless=true"]
