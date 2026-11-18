#!/bin/bash
trap "echo; exit" SIGINT

__usage="Usage: $(basename $0) <setup> <dose> <n_projections> <pixel_w> <n_iter> <init=zero | fbp>

Args:
  <setup>             Setup to submit reconstructions for. Available options
                      include 'ctp528-ideal', 'gammex-ideal', 'spiral-ideal.
  <dose>              Dose level used in reconstructions. Available options
                      are 5.0, 1.0, 0.1.
  <n_projections>     Number of projections used in reconstructions. Available
                      options are 720 and 45.
  <pixel_w>           Pixel width in mm.
  <n_iter>            Number of iterations used in the reconstruction.
  <init>              Initialisation method. Available options are 'zero' and 'fbp'.

Options:
  -h, --help          Print this help message.
"

print_usage() {
  echo "$__usage"
}

SETUP=${1}
DOSE=${2}
N_PROJECTIONS=${3}
PIXEL_W=${4}
N_ITER=${5}
X_INIT=${6}

if [[ "$1" == "--help" || "$1" == "-h" ]]; then
  print_usage
  exit 0
fi

if [ $# -le 5 ]; then
  echo "Error: not enough arguments provided! Got $# but expected 6."
  print_usage
  exit 1
fi

if [[ ${N_PROJECTIONS} != "720" && ${N_PROJECTIONS} != "360" && ${N_PROJECTIONS} != "180" && ${N_PROJECTIONS} != "90" && ${N_PROJECTIONS} != "45" && ${N_PROJECTIONS} != "15" ]]; then
  echo "Error: invalid number of projections given! Expected 720, 360, 180, 90, 45, or 15."
  print_usage
  exit 1
fi


# The alpha values of the L-curves in the thesis; not every setup used all of them.
for ALPHA in {1e-5,2.5e-5,5e-5,1e-4,2.5e-4,5e-4,7.1e-4,1e-3,1.58e-3,2.5e-3,3.54e-3,5e-3,1e-2,2.24e-2,2.5e-2,5e-2,1e-1,5e-1}; do
  ~/pct-recon/scripts/submit_fista.sh ${SETUP} ${DOSE} ${N_PROJECTIONS} ${PIXEL_W} TV ${ALPHA} 0 ${N_ITER} ${X_INIT}
done
