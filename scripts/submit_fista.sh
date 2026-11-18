#!/bin/bash
trap "echo; exit" INT

SETUP=${1}
TARGET_DOSE=${2}
N_PROJECTIONS=${3}
PIXEL_W=${4}
REGULARISER=${5}
ALPHA=${6}
ETA=${7}
N_ITER=${8}
X_INIT=${9}

__usage="Usage: $(basename $0) [setup] [target dose] [n_proj] [pixel_w] [reg] [alpha] [eta] [n_iter] <zero | fbp>"

print_usage() {
  echo "$__usage"
}

if [[ -z ${SETUP} ]]; then
  echo "Error: no setup given."
  print_usage
  exit 1
fi

if [[ -z ${N_ITER} ]]; then
  echo "Error: one or more arguments are missing."
  print_usage
  exit 1
fi

if [[ -z ${X_INIT}  ]];  then
  echo "Error: no initial image type given."
  print_usage
  exit 1
elif [[ ${X_INIT} != "zero" && ${X_INIT} != "fbp" ]]; then
  echo "Error: initial image type must be either 'zero' or 'fbp'."
  print_usage
  exit 1
fi

PHANTOM=${SETUP%-*}
VARIATION=${SETUP#*-}
SHORTHAND=${PHANTOM:0:1}${VARIATION:0:1}

JOBNAME=fista.${SHORTHAND}.${TARGET_DOSE}.${N_PROJECTIONS}.${REGULARISER,,}

if [[ ${REGULARISATION} != "None" ]]; then
  if [[ -n ${ETA} ]]; then
    JOBNAME=${JOBNAME}.${ALPHA}.${ETA}
  else
    JOBNAME=${JOBNAME}.${ALPHA}
  fi
fi

qsub -N ${JOBNAME} \
  -v SETUP=${SETUP} \
  -v TARGET_DOSE=${TARGET_DOSE} \
  -v N_PROJECTIONS=${N_PROJECTIONS} \
  -v PIXEL_W=${PIXEL_W} \
  -v REGULARISER=${REGULARISER} \
  -v ALPHA=${ALPHA} \
  -v ETA=${ETA} \
  -v N_ITER=${N_ITER} \
  -v X_INIT=${X_INIT} \
  ~/pct-recon/scripts/fista.job
