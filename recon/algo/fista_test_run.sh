#!/bin/bash
export SETUP=ctp528-ideal
export TARGET_DOSE=5
export N_PROJECTIONS=720
export PIXEL_W=0.25
export REGULARISER=TV
export ALPHA=1e-1
export ETA=0
export N_ITER=100
export X_INIT=fbp

CODE_DIR=${HOME}/pct-recon
export PYTHONPATH=${CODE_DIR}

export LOGURU_LEVEL=DEBUG
# NSLOTS is set inside a Grid Engine session (e.g. qrsh -pe smp 8); 16 otherwise.
export N_THREADS=${NSLOTS:-16}
export BLOSC_NTHREADS=1
export BLOSC_NOLOCK=1

export N_SIGMA=3

export WORKDIR=${HOME}/Scratch/data/recon/${SETUP}/${TARGET_DOSE}mGy/${N_PROJECTIONS}proj
# export WORKDIR=${HOME}/ACFS/data/recon/${SETUP}/${TARGET_DOSE}mGy/${N_PROJECTIONS}proj
# export ACFS_DIR=${HOME}/ACFS/data/recon/${SETUP}/${TARGET_DOSE}mGy/${N_PROJECTIONS}proj

export MATRIX_DIR=${WORKDIR}/processed-${N_SIGMA}sigma/matrix-${PIXEL_W}mm
export VECTOR_DIR=${WORKDIR}/processed-${N_SIGMA}sigma/b
export FBP_DIR=${WORKDIR}/processed-${N_SIGMA}sigma/fbp
export OUTPUT_DIR=${TMPDIR}/recon

export ITER_START=0
export ITER_END=${N_ITER}

export N_SLICES=3
export SLICE_THICKNESS=1.0

# tar -xvf ${MATRIX_DIR}.tar -C .
# tar -xvf ${VECTOR_DIR}.tar.gz -C .
# tar -xvf ${FBP_DIR}.tar -C .

# export MATRIX_DIR=${TMPDIR}/matrix-${PIXEL_W}mm
# export VECTOR_DIR=${TMPDIR}/b
# export FBP_DIR=${TMPDIR}/fbp

source ~/pct-recon/.venv/bin/activate
echo "Start time $(date "+%Y%m%d %H:%M:%S")"
/usr/bin/time --verbose python ${CODE_DIR}/recon/algo/fista.py
echo "End time $(date "+%Y%m%d %H:%M:%S")"
