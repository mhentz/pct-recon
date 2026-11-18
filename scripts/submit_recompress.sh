#!/bin/bash
SETUP=${1}
DOSE=${2}
N_PROJECTIONS=${3}
PIXEL_W=${4}

__usage="Usage: $(basename $0) <setup> <dose> <n_projections> <pixel_w>"

print_usage() {
  echo "$__usage"
}

if [[ -z ${SETUP} ]]; then
  echo "Error: no setup given."
  print_usage
  exit 1
fi

if [[ -z ${DOSE} ]]; then
  echo "Error: dose not given."
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

PHANTOM=${SETUP%-*}
VARIATION=${SETUP#*-}

SHORTHAND=${PHANTOM:0:1}${VARIATION:0:1}

JOBNAME=rec.${SHORTHAND}.${DOSE}.${N_PROJECTIONS}
qsub -N ${JOBNAME} \
  -v SETUP=${SETUP} \
  -v DOSE=${DOSE} \
  -v N_PROJECTIONS=${N_PROJECTIONS} \
  -v PIXEL_W=${PIXEL_W} \
  ~/pct-recon/scripts/recompress.job
