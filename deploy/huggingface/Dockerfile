# PyBorehole on Hugging Face Spaces (Docker Space).
#
# This file and README.md (next to it) are the only files in the Space; the code
# is taken from GitHub at build time, so the Space always runs the current
# mariosgeo/PyBorehole main branch. Steps as in the Dockerfile in the repository
# root (see deploy/container_setup.sh).
FROM python:3.12-slim-bookworm

ARG REPO=https://github.com/mariosgeo/PyBorehole
ARG REF=main
ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    GEANT4_DIR=/opt/geant4 \
    GAMMALOG_EXE=/opt/gammalog/gammalog \
    MPLBACKEND=Agg \
    PYBOREHOLE_MAX_ERT=1

# 1. System packages, Python packages, Geant4 (cached unless requirements.txt changes)
ADD ${REPO}/raw/${REF}/requirements.txt ${REPO}/raw/${REF}/deploy/container_setup.sh /tmp/setup/
RUN bash /tmp/setup/container_setup.sh deps /tmp/setup

# 2. The code: the commit id is downloaded first, so a new commit on GitHub
#    invalidates the cache and a rebuild of the Space takes the new code
ADD https://api.github.com/repos/mariosgeo/PyBorehole/commits/${REF} /tmp/commit.json
RUN git clone --depth 1 --branch ${REF} ${REPO} /app \
    && rm -rf /app/docs /app/.git \
    && bash /app/deploy/container_setup.sh build /app

RUN useradd -m -u 1000 user && chown -R user /app
USER user
ENV HOME=/home/user MPLCONFIGDIR=/home/user/.cache/matplotlib
WORKDIR /app

EXPOSE 8501
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0", \
     "--server.headless=true"]
