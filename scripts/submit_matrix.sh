#!/bin/bash
trap "echo; exit" INT
SIM=${1}
N_PROJECTIONS=${2}
PIXEL_W=${3}
IMAGE_W=${4}
N_INPUT_CHUNKS=${5}
N_OUTPUT_CHUNKS=${6}
BATCH_SIZE=${7}

__usage="Usage: $(basename $0) [ci|cr|gi|gr|si|sr] [n_projections] [pixel_w] [image_w] [n_input_chunks] [n_output_chunks] [batch_size]"

print_usage() {
  echo "$__usage"
}

if [ -z ${SIM} ]; then
  echo "Error: no sim chosen. Please choose one of the following: "
  echo "   - ci: CTP528 ideal"
  echo "   - cr: CTP528 realistic"
  echo "   - gi: Gammex ideal"
  echo "   - gr: Gammex realistic"
  echo "   - si: spiral ideal"
  echo "   - sr: spiral realistic"
  print_usage
  exit 1
fi

if [ -z ${N_PROJECTIONS} ]; then
  echo "Error: number of projections not chosen."
  print_usage
  exit 1
fi

if [ -z "${BATCH_SIZE}" ]; then
  echo "Error: batch size not chosen."
  print_usage
  exit 1
fi

case ${SIM} in
ci)
  SETUP=ctp528-ideal
  ;;
cr)
  SETUP=ctp528-realistic
  ;;
gi)
  SETUP=gammex-ideal
  ;;
gr)
  SETUP=gammex-realistic
  ;;
si)
  SETUP=spiral-ideal
  ;;
sr)
  SETUP=spiral-realistic
  ;;
*)
  echo "Error: sim not available. Please choose one of the following: "
  echo "   - ci: CTP528 ideal"
  echo "   - cr: CTP528 realistic"
  echo "   - gi: Gammex ideal"
  echo "   - gr: Gammex realistic"
  echo "   - si: spiral ideal"
  echo "   - sr: spiral realistic"
  print_usage
  exit 1
  ;;
esac

LAST_PROJECTION=359

for FIRST in $(seq 0 ${BATCH_SIZE} ${LAST_PROJECTION}); do
  LAST=$(echo "${FIRST} ${BATCH_SIZE}" | awk '{print $1+$2}')
  JOBNAME=$(printf "mat.${SIM}.${N_PROJECTIONS}.%03d" ${FIRST})

  qsub -N ${JOBNAME} \
    -v SETUP=${SETUP} \
    -v START=${FIRST} \
    -v STOP=${LAST} \
    -v N_PROJECTIONS=${N_PROJECTIONS} \
    -v PIXEL_W=${PIXEL_W} \
    -v N_INPUT_CHUNKS=${N_INPUT_CHUNKS} \
    -v IMAGE_W=${IMAGE_W} \
    -v N_OUTPUT_CHUNKS=${N_OUTPUT_CHUNKS} \
    ~/pct-recon/scripts/matrix.job
done
