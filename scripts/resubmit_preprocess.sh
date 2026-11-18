#!/bin/bash
trap "echo; exit" INT

SETUP=${1}
N_PROJECTIONS=${2}
N_CHUNKS=${3}
BATCH_SIZE=${4}
START=${5}
STOP=${6}

__usage="Usage: $(basename $0) [setup] [n_projections] [n_chunks] [batch_size] [start_index] <stop_index>"

print_usage() {
  echo "$__usage"
}

if [[ -z ${SETUP} ]]; then
  echo "Error: no setup given."
  print_usage
  exit 1
fi

if [[ -z ${N_PROJECTIONS} ]]; then
  echo "Error: no number of projections given."
  print_usage
  exit 1
fi

if [[ -z ${N_CHUNKS} ]]; then
  echo "Error: number of chunks for simulation files not given."
  print_usage
  exit 1
fi

if [[ -z ${BATCH_SIZE} ]]; then
  echo "Error: no batch size given."
  print_usage
  exit 1
fi

if [[ -z ${START} ]]; then
  echo "Error: provide the start index of chunks for simulation files not given."
  print_usage
  exit 1
fi

if [[ -z ${STOP} ]]; then
  echo "Error: provide the stop index of chunks for simulation files not given."
  print_usage
  exit 1
fi

if [[ "${SETUP}" == "ctp528-ideal" ]]; then
  if [[ "${N_PROJECTIONS}" == "720" ]]; then
    ACTUAL_DOSE=5.01168
  elif [[ "${N_PROJECTIONS}" == "360" ]]; then
    ACTUAL_DOSE=5.01179
  elif [[ "${N_PROJECTIONS}" == "180" ]]; then
    ACTUAL_DOSE=5.01167
  elif [[ "${N_PROJECTIONS}" == "90" ]]; then
    ACTUAL_DOSE=5.0114
  elif [[ "${N_PROJECTIONS}" == "45" ]]; then
    ACTUAL_DOSE=5.01146
  else
    echo "Error: setup ${SETUP} only supports 720, 360, 180, 90, or 45 projections."
    exit 1
  fi
elif [[ "${SETUP}" == "ctp528-realistic" ]]; then
  if [[ "${N_PROJECTIONS}" == "720" ]]; then
    ACTUAL_DOSE=5.01829
  else
    echo "Error: setup ${SETUP} only supports 720 projections."
    exit 1
  fi
elif [[ "${SETUP}" == "gammex-ideal" ]]; then
  if [[ "${N_PROJECTIONS}" == "720" ]]; then
    ACTUAL_DOSE=5.00974
  elif [[ "${N_PROJECTIONS}" == "45" ]]; then
    ACTUAL_DOSE=5.00967
  elif [[ "${N_PROJECTIONS}" == "15" ]]; then
    ACTUAL_DOSE=5.00976
  else
    echo "Error: setup ${SETUP} only supports 720 and 45 projections."
    exit 1
  fi
elif [[ "${SETUP}" == "gammex-realistic" ]]; then
  if [[ "${N_PROJECTIONS}" == "720" ]]; then
    ACTUAL_DOSE=5.02040
  else
    echo "Error: setup ${SETUP} only supports 720 projections."
    exit 1
  fi
elif [[ "${SETUP}" == "spiral-ideal" ]]; then
  if [[ "${N_PROJECTIONS}" == "720" ]]; then
    ACTUAL_DOSE=5.0117
  elif [[ "${N_PROJECTIONS}" == "360" ]]; then
    ACTUAL_DOSE=5.01175
  elif [[ "${N_PROJECTIONS}" == "180" ]]; then
    ACTUAL_DOSE=5.01153
  elif [[ "${N_PROJECTIONS}" == "90" ]]; then
    ACTUAL_DOSE=5.01113
  elif [[ "${N_PROJECTIONS}" == "45" ]]; then
    ACTUAL_DOSE=5.01123
  elif [[ "${N_PROJECTIONS}" == "15" ]]; then
    ACTUAL_DOSE=5.00959
  else
    echo "Error: setup ${SETUP} only supports 720 projections."
    exit 1
  fi
elif [[ "${SETUP}" == "spiral-realistic" ]]; then
  if [[ "${N_PROJECTIONS}" == "720" ]]; then
    ACTUAL_DOSE=5.02113
  else
    echo "Error: setup ${SETUP} only supports 720 projections."
    exit 1
  fi
else
  echo "Error: unknown setup ${SETUP}."
  print_usage
  exit 1
fi

PHANTOM=${SETUP%-*}
VARIATION=${SETUP#*-}
SHORTHAND=${PHANTOM:0:1}${VARIATION:0:1}

if [[ "${VARIATION}" == "realistic" ]]; then
  DEGRADE_FLAG="--degrade"
fi

LAST=$(echo "${START} ${BATCH_SIZE}" | awk '{print $1+$2}')
JOBNAME=$(printf "pre.${SHORTHAND}.${N_PROJECTIONS}.%03d" ${START})

qsub -N ${JOBNAME} \
  -v SETUP=${SETUP} \
  -v N_PROJECTIONS=${N_PROJECTIONS} \
  -v ACTUAL_DOSE=${ACTUAL_DOSE} \
  -v START=${START} \
  -v STOP=${LAST} \
  -v N_CHUNKS=${N_CHUNKS} \
  -v DEGRADE_FLAG=${DEGRADE_FLAG} \
  ~/pct-recon/scripts/preprocess.job
