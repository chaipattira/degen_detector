#!/usr/bin/env bash
# download_obs_chains.sh
# Download observational MCMC chains for degen_detector experiments.
#
# Datasets:
#   1. DESI DR1 full-shape (all tracers, LCDM, velocileptors)
#      Source: https://data.desi.lbl.gov/public/dr1/vac/dr1/full-shape-cosmo-params/
#      Ref: DESI 2024 VII (arXiv:2411.12022)
#
#   2. Planck 2018 base_plikHM_TTTEEE_lowl_lowE_lensing
#      Source: Planck Legacy Archive (PLA), ESA
#      Ref: Planck 2018 cosmological parameters (arXiv:1807.06209)
#
#   3. DES Y3 3x2pt LCDM MagLim
#      Source: DES public data server (NCSA)
#      Ref: DES Y3 (Abbott et al. 2022, arXiv:2207.10900)
#
# Usage:
#   bash scripts/download_obs_chains.sh
#   bash scripts/download_obs_chains.sh --skip-desi
#   bash scripts/download_obs_chains.sh --skip-planck
#   bash scripts/download_obs_chains.sh --skip-des
#
# Files written:
#   data/desi_dr1_fs/chain.{1..4}.txt
#   data/base/plikHM_TTTEEE_lowl_lowE_lensing/base_plikHM_TTTEEE_lowl_lowE_lensing*.txt
#   data/base/plikHM_TTTEEE_lowl_lowE_lensing/base_plikHM_TTTEEE_lowl_lowE_lensing.paramnames
#   data/des_y3_3x2pt/chain_3x2pt_lcdm_SR_maglim.txt

set -euo pipefail

# ── Resolve repo root (script lives in scripts/) ──────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
DATA_DIR="${REPO_ROOT}/data"

# ── Flags ─────────────────────────────────────────────────────────────────────
SKIP_DESI=0
SKIP_PLANCK=0
SKIP_DES=0

for arg in "$@"; do
    case "$arg" in
        --skip-desi)    SKIP_DESI=1 ;;
        --skip-planck)  SKIP_PLANCK=1 ;;
        --skip-des)     SKIP_DES=1 ;;
        -h|--help)
            sed -n '/^# Usage:/,/^[^#]/p' "$0" | grep '^#' | sed 's/^# \{0,2\}//'
            exit 0
            ;;
        *) echo "Unknown option: $arg"; exit 1 ;;
    esac
done

# ── Downloader (wget preferred; fall back to curl) ────────────────────────────
_download() {
    local url="$1"
    local dest="$2"
    if [[ -f "$dest" && -s "$dest" ]]; then
        echo "  Already exists, skipping: $(basename "$dest")"
        return 0
    fi
    echo "  Downloading: $(basename "$dest")"
    if command -v wget &>/dev/null; then
        wget -q --show-progress --retry-connrefused --tries=3 -O "$dest" "$url"
    elif command -v curl &>/dev/null; then
        curl -fSL --retry 3 -o "$dest" "$url"
    else
        echo "ERROR: neither wget nor curl found." >&2
        exit 1
    fi
}

# ── 1. DESI DR1 full-shape chains ─────────────────────────────────────────────
if [[ "$SKIP_DESI" -eq 0 ]]; then
    echo ""
    echo "=== DESI DR1 full-shape chains ==="

    DESI_DIR="${DATA_DIR}/desi_dr1_fs"
    mkdir -p "$DESI_DIR"

    DESI_BASE_URL="https://data.desi.lbl.gov/public/dr1/vac/dr1/full-shape-cosmo-params"
    DESI_DATASET="desi-reptvelocileptors-fs-all_schoneberg2024-bbn_planck2018-ns10"
    DESI_URL="${DESI_BASE_URL}/v1.0/cobaya/base/${DESI_DATASET}"

    for i in 1 2 3 4; do
        _download "${DESI_URL}/chain.${i}.txt" "${DESI_DIR}/chain.${i}.txt"
    done

    # Also grab the cobaya metadata files
    for ext in margestats covmat input.yaml updated.yaml; do
        url="${DESI_URL}/chain.${ext}"
        dest="${DESI_DIR}/chain.${ext}"
        if [[ ! -f "$dest" ]]; then
            if command -v wget &>/dev/null; then
                wget -q --tries=2 -O "$dest" "$url" 2>/dev/null || rm -f "$dest"
            else
                curl -fSL --retry 2 -o "$dest" "$url" 2>/dev/null || rm -f "$dest"
            fi
            [[ -f "$dest" && -s "$dest" ]] && echo "  Downloaded: chain.${ext}" || true
        fi
    done

    echo "  DESI chains saved to: ${DESI_DIR}"
fi

# ── 2. Planck 2018 chains ─────────────────────────────────────────────────────
if [[ "$SKIP_PLANCK" -eq 0 ]]; then
    echo ""
    echo "=== Planck 2018 chains ==="

    # Planck 2018 chains are mirrored at IPAC (IRSA).
    # COM_CosmoParams_base-plikHM-TTTEEE-lowl-lowE_R3.00.zip  (62 MB)
    #   -> base/plikHM_TTTEEE_lowl_lowE/  (TTTEEE+lowl+lowE, no lensing)
    # COM_CosmoParams_base-plikHM_R3.01.zip  (471 MB)
    #   -> base/plikHM_TTTEEE_lowl_lowE_lensing/  (TTTEEE+lowl+lowE+lensing)
    #      plus many other base plikHM chain sets
    IPAC_BASE="https://irsa.ipac.caltech.edu/data/Planck/release_3/ancillary-data/cosmoparams"

    _download_planck_zip() {
        local zip_name="$1"   # e.g. COM_CosmoParams_base-plikHM_R3.01.zip
        local subdir="$2"     # subdir inside zip to extract, e.g. base/plikHM_TTTEEE_lowl_lowE_lensing
        local target_dir="$3" # destination in data/
        local tmp_zip="${DATA_DIR}/${zip_name}"

        mkdir -p "$target_dir"

        # Check if chain txt files already present
        local n_txt
        n_txt=$(find "$target_dir" -maxdepth 1 -name "*_[0-9].txt" 2>/dev/null | wc -l)
        if [[ "$n_txt" -gt 0 ]]; then
            echo "  Already have chain files in: $(basename "$target_dir"), skipping."
            return 0
        fi

        echo "  Downloading: ${zip_name}  (this may take a while)"
        local url="${IPAC_BASE}/${zip_name}"
        if command -v wget &>/dev/null; then
            wget -q --show-progress --retry-connrefused --tries=3 \
                -O "$tmp_zip" "$url"
        else
            curl -fSL --retry 3 -o "$tmp_zip" "$url"
        fi

        echo "  Extracting ${subdir}/ -> ${target_dir}"
        local tmp_extract="${DATA_DIR}/_planck_extract_tmp"
        rm -rf "$tmp_extract"
        mkdir -p "$tmp_extract"
        # Extract only the requested subdirectory
        unzip -q "$tmp_zip" "${subdir}/*" -d "$tmp_extract"

        # Move files from the extracted subdir into target_dir
        if [[ -d "${tmp_extract}/${subdir}" ]]; then
            find "${tmp_extract}/${subdir}" -maxdepth 1 -type f | while read -r f; do
                mv "$f" "${target_dir}/$(basename "$f")"
            done
        else
            # Fallback: move everything
            find "$tmp_extract" -type f | while read -r f; do
                mv "$f" "${target_dir}/$(basename "$f")"
            done
        fi
        rm -rf "$tmp_extract" "$tmp_zip"
        echo "  Planck chain extracted to: ${target_dir}"
    }

    # plikHM_TTTEEE_lowl_lowE  (no lensing; 62 MB zip)
    _download_planck_zip \
        "COM_CosmoParams_base-plikHM-TTTEEE-lowl-lowE_R3.00.zip" \
        "base/plikHM_TTTEEE_lowl_lowE" \
        "${DATA_DIR}/base/plikHM_TTTEEE_lowl_lowE"

    # plikHM_TTTEEE_lowl_lowE_lensing  (471 MB zip contains many chains incl. lensing)
    _download_planck_zip \
        "COM_CosmoParams_base-plikHM_R3.01.zip" \
        "base/plikHM_TTTEEE_lowl_lowE_lensing" \
        "${DATA_DIR}/base/plikHM_TTTEEE_lowl_lowE_lensing"

    echo "  Planck chains saved to: ${DATA_DIR}/base/"
fi

# ── 3. DES Y3 3x2pt LCDM MagLim chain ────────────────────────────────────────
if [[ "$SKIP_DES" -eq 0 ]]; then
    echo ""
    echo "=== DES Y3 3x2pt LCDM MagLim chain ==="

    DES_DIR="${DATA_DIR}/des_y3_3x2pt"
    mkdir -p "$DES_DIR"

    DES_URL="https://desdr-server.ncsa.illinois.edu/despublic/y3a2_files/chains/chain_3x2pt_lcdm_SR_maglim.txt"
    _download "$DES_URL" "${DES_DIR}/chain_3x2pt_lcdm_SR_maglim.txt"

    echo "  DES Y3 chain saved to: ${DES_DIR}"
fi

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "=== Download complete ==="
echo ""
echo "Expected files:"
[[ "$SKIP_DESI"   -eq 0 ]] && echo "  data/desi_dr1_fs/chain.{1..4}.txt"
[[ "$SKIP_PLANCK" -eq 0 ]] && echo "  data/base/plikHM_TTTEEE_lowl_lowE_lensing/base_plikHM_TTTEEE_lowl_lowE_lensing*.txt"
[[ "$SKIP_PLANCK" -eq 0 ]] && echo "  data/base/plikHM_TTTEEE_lowl_lowE/base_plikHM_TTTEEE_lowl_lowE*.txt"
[[ "$SKIP_DES"    -eq 0 ]] && echo "  data/des_y3_3x2pt/chain_3x2pt_lcdm_SR_maglim.txt"
echo ""
echo "Run experiments:"
[[ "$SKIP_DESI"   -eq 0 ]] && echo "  python scripts/run_desi_logmode.py"
[[ "$SKIP_PLANCK" -eq 0 ]] && echo "  python scripts/run_planck_logmode.py"
[[ "$SKIP_DES"    -eq 0 ]] && echo "  python scripts/run_des_y3_logmode.py"
