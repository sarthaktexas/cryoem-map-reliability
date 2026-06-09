#!/bin/bash
# FSC-Q (Xmipp + BlocRes) on an HPC cluster — SLURM template.
#
# EDIT the #SBATCH lines and the module/env block for your cluster, then submit:
#
#     sbatch scripts/fscq_cluster.sbatch 49450 33734 44471
#
# Positional args are EMDB IDs to process (default: the three thesis anchors).
# This runs in --compute-only mode: it writes outputs/emd_<ID>/fscq/fscq.mrc on the
# cluster (the heavy BlocRes step) and defers the Cα/V correlation to a local run.
#
# For PBS/Torque clusters, replace #SBATCH with #PBS directives and `sbatch` with
# `qsub`; the body below is scheduler-agnostic.

#SBATCH --job-name=fscq
#SBATCH --partition=cpu            # <-- your partition/queue
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=fscq_%j.log

set -euo pipefail

# --- Cluster software stack (EDIT) ---------------------------------------------
# Provide Xmipp (xmipp_volume_from_pdb, xmipp_image_convert) and Bsoft (blocres).
module load scipion 2>/dev/null || true     # e.g. `module load xmipp bsoft`
module load bsoft   2>/dev/null || true

# How to launch Xmipp programs. If xmipp_* are on PATH, leave XMIPP_RUN empty.
export XMIPP_RUN="${XMIPP_RUN:-scipion3 run}"
# Bsoft blocres path (autodetect, else set explicitly).
export BLOCRES_BIN="${BLOCRES_BIN:-$(command -v blocres || echo /usr/local/bsoft/bin/blocres)}"

# --- Python environment (EDIT) -------------------------------------------------
cd "${SLURM_SUBMIT_DIR:-$PWD}"
# Use the project venv if present; otherwise rely on the module python + deps.
if [ -d .venv ]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

EMD_IDS=("$@")
if [ "${#EMD_IDS[@]}" -eq 0 ]; then
    EMD_IDS=(49450 33734 44471)
fi

echo "[fscq] XMIPP_RUN='${XMIPP_RUN}'  BLOCRES_BIN='${BLOCRES_BIN}'"
echo "[fscq] processing: ${EMD_IDS[*]}"

for ID in "${EMD_IDS[@]}"; do
    echo "==== EMD-${ID} ===="
    python scripts/run_fscq_validation.py --emd-id "${ID}" --compute-only
done

echo "[fscq] done. Transfer outputs/emd_*/fscq/fscq.mrc back, then run locally:"
echo "       python scripts/run_fscq_validation.py --emd-id <ID> --cohort-summary"
