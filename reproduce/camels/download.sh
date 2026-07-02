#!/usr/bin/env bash
# ABOUTME: Download CAMELS SB28 IllustrisTNG field maps and params file.
# ABOUTME: Fetches Mcdm, Mgas, Mstar maps and params_SB28 from the Flatiron CDN.
set -euo pipefail

BASE_URL="https://users.flatironinstitute.org/~camels/CMD/2D_maps/data/IllustrisTNG"
DATA_DIR="${1:-/anvil/scratch/x-ctirapongpra/degen_detector/data/camels_sb28}"

mkdir -p "${DATA_DIR}"

FILES=(
    "Maps_Mcdm_IllustrisTNG_SB28_z=0.00.npy"
    "Maps_Mgas_IllustrisTNG_SB28_z=0.00.npy"
    "Maps_Mstar_IllustrisTNG_SB28_z=0.00.npy"
    "params_SB28_IllustrisTNG.txt"
)

echo "Downloading CAMELS SB28 data to: ${DATA_DIR}"
echo "Base URL: ${BASE_URL}"
echo ""

for fname in "${FILES[@]}"; do
    dest="${DATA_DIR}/${fname}"
    if [ -f "${dest}" ]; then
        echo "SKIP (exists): ${fname}"
        continue
    fi
    echo "Downloading: ${fname} ..."
    curl -L --progress-bar -o "${dest}" "${BASE_URL}/${fname}"
    echo "Done: ${fname}"
done

echo ""
echo "Verifying shapes..."
PROJECT_DIR="/home/x-ctirapongpra/scratch/degen_detector"
for field in Mcdm Mgas Mstar; do
    fpath="${DATA_DIR}/Maps_${field}_IllustrisTNG_SB28_z=0.00.npy"
    if [ -f "${fpath}" ]; then
        shape=$(uv run --project "${PROJECT_DIR}" python -c "import numpy as np; a=np.load('${fpath}'); print(a.shape, a.dtype)" 2>/dev/null)
        echo "  ${field}: ${shape}"
    fi
done

params_file="${DATA_DIR}/params_SB28_IllustrisTNG.txt"
if [ -f "${params_file}" ]; then
    shape=$(uv run --project "${PROJECT_DIR}" python -c "import numpy as np; a=np.loadtxt('${params_file}'); print(a.shape)" 2>/dev/null)
    echo "  params:  ${shape}"
fi

echo ""
echo "Expected: maps (30720, 256, 256) float32; params (2048, 28)"
echo "Done."
