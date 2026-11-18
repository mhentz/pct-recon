#!/bin/bash
SETUP=${1}
TARGET_DOSE=${2}
N_PROJECTIONS=${3}
PIXEL_W=${4}
IMAGE_W=${5}
N_ITERATIONS=${6}

__usage="Usage: $(basename $0) <setup> <target dose=5 | 1 | 0.1> <n_projections> <pixel_w> <image_w> <n_iterations>"

print_usage() {
  echo "$__usage"
}

if [[ -z ${SETUP} ]]; then
  echo "Error: no setup given."
  print_usage
  exit 1
fi

if [[ -z ${TARGET_DOSE} ]]; then
  echo "Error: no target dose given."
  print_usage
  exit 1
elif [[ ${TARGET_DOSE} != 5 && ${TARGET_DOSE} != 1 && ${TARGET_DOSE} != 0.2 ]]; then
  echo "Error: target dose must be 5, 1, or 0.2."
  print_usage
  exit 1
fi

if [[ -z ${N_PROJECTIONS} ]]; then
  echo "Error: number of projections not given."
  print_usage
  exit 1
fi

if [[ -z ${PIXEL_W} ]]; then
  echo "Error: no pixel width given."
  print_usage
  exit 1
fi

if [[ -z ${IMAGE_W} ]]; then
  echo "Error: no image width given."
  print_usage
  exit 1
fi

if [[ -z ${N_ITERATIONS} ]]; then
  echo "Error: no number of iterations given."
  print_usage
  exit 1
fi

PHANTOM=${SETUP%-*}
VARIATION=${SETUP#*-}
SHORTHAND=${PHANTOM:0:1}${VARIATION:0:1}
JOBNAME=lip.${SHORTHAND}.${TARGET_DOSE}.${N_PROJECTIONS}

N_SIGMA=3
MATRIX_DIR=${HOME}/Scratch/data/recon/${SETUP}/${TARGET_DOSE}mGy/${N_PROJECTIONS}proj/processed-${N_SIGMA}sigma/matrix-${PIXEL_W}mm

qsub -N $JOBNAME \
  -v N_PROJECTIONS=${N_PROJECTIONS} \
  -v IMAGE_W=${IMAGE_W} \
  -v MATRIX_DIR=${MATRIX_DIR} \
  -v N_ITERATIONS=${N_ITERATIONS} \
  ~/pct-recon/scripts/lipschitz.job
