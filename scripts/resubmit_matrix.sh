#!/bin/bash
trap "echo; exit" INT

SIM=${1}
N_PROJECTIONS=${2}
PIXEL_W=${3}
IMAGE_W=${4}
N_INPUT_CHUNKS=${5}
N_OUTPUT_CHUNKS=${6}
START=${7}
STOP=${8}
BATCH_SIZE=${9}

__usage="Usage: $(basename $0) <setup=ci|cr|gi|gr|si|sr> <n_projections> <pixel width> <image width> <n input chunks> <n output chunks> <start> <stop (exclusive)> <batch_size>"

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
  echo "Error: number of projections not given."
  print_usage
  exit 1
fi

if [ -z ${PIXEL_W} ]; then
  echo "Error: pixel width not given."
  print_usage
  exit 1
fi

if [ -z ${IMAGE_W} ]; then
  echo "Error: image width in pixels not given."
  print_usage
  exit 1
fi

if [ -z ${START} ]; then
  echo "Error: start angle not given."
  print_usage
  exit 1
fi

if [ -z ${STOP} ]; then
  echo "Error: stop angle not given."
  print_usage
  exit 1
fi

if [ -z ${BATCH_SIZE} ]; then
  echo "Error: batch size not given."
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

for FIRST in $(seq ${START} ${BATCH_SIZE} $(( STOP - 1 ))); do
  LAST=$(( FIRST + BATCH_SIZE ))
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
