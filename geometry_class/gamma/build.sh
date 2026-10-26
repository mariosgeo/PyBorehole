#!/usr/bin/env bash
# Build gammalog with CMake and Geant4 (Linux / macOS).
#   bash build.sh [Geant4 install prefix] [build dir]
# Without a prefix, Geant4 is found through $GEANT4_DIR, geant4-config
# (e.g. an activated conda environment with geant4) or the CMake search path.
# The build dir defaults to the "build" folder next to this script.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
prefix="${1:-${GEANT4_DIR:-}}"
bdir="${2:-$here/build}"
if [ -z "$prefix" ] && command -v geant4-config >/dev/null 2>&1; then
    prefix="$(geant4-config --prefix)"
fi
args=(-S "$here" -B "$bdir" -DCMAKE_BUILD_TYPE=Release)
if [ -n "$prefix" ]; then
    args+=(-DCMAKE_PREFIX_PATH="$prefix")
fi
cmake "${args[@]}"
cmake --build "$bdir" --parallel
echo "Built $bdir/gammalog"
