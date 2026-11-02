#!/usr/bin/env bash
# Install PyBorehole and all three simulators into a Debian-based Python image.
# Used by the Dockerfile in the repository root and by deploy/huggingface/Dockerfile.
#
#   bash deploy/container_setup.sh deps  <source dir>   system packages, Python packages, Geant4
#   bash deploy/container_setup.sh build <source dir>   compile the Geant4 gamma-ray program
#
# "deps" only needs requirements.txt from the source dir, so Docker can cache it
# separately from the code.
set -euo pipefail
stage="$1"
src="$(cd "$2" && pwd)"
G4=/opt/geant4                       # conda environment with Geant4, CMake and a C++ compiler
BUILD=/opt/gammalog                  # where the gamma-ray program is compiled

case "$stage" in
deps)
    # TetGen meshes the ERT model; git and curl for the image itself
    apt-get update
    apt-get install -y --no-install-recommends tetgen ca-certificates curl bzip2 git
    rm -rf /var/lib/apt/lists/*

    # Python packages: the same pip requirements as on Streamlit Community Cloud
    pip install --no-cache-dir -r "$src/requirements.txt"

    # Geant4 is not on pip: install it with micromamba from conda-forge into its
    # own environment (headless "noqt" build, plus the compiler for gammalog)
    curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C /usr/local bin/micromamba
    micromamba create -y -p "$G4" -c conda-forge \
        "geant4=11.4.3=noqt*" geant4-data-emlow geant4-data-ensdfstate geant4-data-photonevaporation \
        cmake make cxx-compiler \
        expat zlib freetype xorg-libx11 xorg-libxext libgl-devel libopengl-devel libglx-devel
    # (the last line: development files that Geant4Config.cmake requires when
    #  gammalog is compiled; Geant4 itself only ships the runtime libraries)
    micromamba clean -y --all
    ;;
build)
    # Compile gammalog against the conda Geant4 (activation sets CXX for the conda compiler)
    micromamba run -p "$G4" bash "$src/geometry_class/gamma/build.sh" "$G4" "$BUILD"
    test -x "$BUILD/gammalog"
    rm -rf "$BUILD/CMakeFiles"
    ;;
*)
    echo "usage: $0 deps|build <source dir>" >&2
    exit 2
    ;;
esac
