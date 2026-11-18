#!/bin/bash
SETUP=${1}
N_PROJECTIONS=${2}
PIXEL_W=${3}
RATIO=${4}

__usage="Usage: $(basename $0) <setup> <n_projections> <pixel_w> <ratio>"

print_usage() {
  echo "$__usage"
}

if [[ -z ${SETUP} ]]; then
  echo "Error: no setup given."
  print_usage
  exit 1
fi

if [[ -z ${N_PROJECTIONS} ]]; then
  echo "Error: number of projections not given."
  print_usage
  exit 1
fi

if [[ -z ${PIXEL_W} ]]; then
  echo "Error: pixel width not given."
  print_usage
  exit 1
fi

if [[ -z ${RATIO} ]]; then
  echo "Error: subsampling ratio not given."
  print_usage
  exit 1
fi

PHANTOM=${SETUP%-*}
VARIATION=${SETUP#*-}

SHORTHAND=${PHANTOM:0:1}${VARIATION:0:1}

JOBNAME=sub.${SHORTHAND}.${RATIO}.${N_PROJECTIONS}
qsub -N ${JOBNAME} \
  -v SETUP=${SETUP} \
  -v N_PROJECTIONS=${N_PROJECTIONS} \
  -v PIXEL_W=${PIXEL_W} \
  -v RATIO=${RATIO} \
  ~/pct-recon/scripts/subsample.job
