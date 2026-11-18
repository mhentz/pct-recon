#!/bin/bash

SETUP=gammex-ideal
TARGET_DOSE=5
N_PROJECTIONS=45
PIXEL_W=0.5
REGULARISER=TV
ALPHA=1e-3
ETA=0
N_ITER=10
X_INIT=zero

CODE_DIR=${HOME}/pct-recon
cd ${CODE_DIR} || { echo "Error: cannot change to ${CODE_DIR}" >&2; exit 1; }

export PYTHONPATH=${CODE_DIR}

export LOGURU_LEVEL=DEBUG
# NSLOTS is set inside a Grid Engine session (e.g. qrsh -pe smp 8); 16 otherwise.
export N_THREADS=${NSLOTS:-16}
export BLOSC_NTHREADS=1
export BLOSC_NOLOCK=1

export TARGET_DOSE=${TARGET_DOSE}
export N_SIGMA=3

export WORKDIR=${HOME}/Scratch/data/recon/${SETUP}/${TARGET_DOSE}mGy/${N_PROJECTIONS}proj

export MATRIX_DIR=${WORKDIR}/processed-${N_SIGMA}sigma/matrix-${PIXEL_W}mm
export VECTOR_DIR=${WORKDIR}/processed-${N_SIGMA}sigma/b
export FBP_DIR=${WORKDIR}/processed-${N_SIGMA}sigma/fbp
export OUTPUT_DIR=${WORKDIR}/recon

export ITER_START=0
export ITER_END=${N_ITER}

export SETUP=${SETUP}
export N_PROJECTIONS=${N_PROJECTIONS}
export PIXEL_W=${PIXEL_W}

export N_SLICES=3
export SLICE_THICKNESS=1.0

export X_INIT=${X_INIT}

export REGULARISER=${REGULARISER}

if [ "$REGULARISER" == "dTV" ]; then
  export REF_DIR=${HOME}/pct-recon/phantoms/xray_refs
fi

export PROX_MAX_LOOPS=2000

if [[ -n $ALPHA ]]; then
  export ALPHA
else
  export ALPHA=0
fi

if [[ -n $ETA ]]; then
  export ETA
else
  export ETA=0
fi

echo "Start time $(date "+%Y%m%d %H:%M:%S")"
/usr/bin/time --verbose python recon/algo/fista.py
echo "End time $(date "+%Y%m%d %H:%M:%S")"
