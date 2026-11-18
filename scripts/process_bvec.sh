#!/bin/bash -e
trap "echo; exit" INT

setup=${1}
n_projections=${2}
n_input_chunks=${3}
n_output_chunks=${4}

if [ -z ${setup} ]; then
  echo "Error: no setup specified."
  echo "Usage: ./process_bvec.sh <setup> <n_projections> <n_input_chunks> <n_output_chunks>"
  exit 1
fi

if [ -z ${n_projections} ]; then
  echo "Error: one or more arguments are missing."
  echo "Usage: ./process_bvec.sh <setup> <n_projections> <n_input_chunks> <n_output_chunks>"
  exit 1
fi

if [ -z ${n_input_chunks} ]; then
  echo "Error: one or more arguments are missing."
  echo "Usage: ./process_bvec.sh <setup> <n_projections> <n_input_chunks> <n_output_chunks>"
  exit 1
fi

if [ -z ${n_output_chunks} ]; then
  echo "Error: one or more arguments are missing."
  echo "Usage: ./process_bvec.sh <setup> <n_projections> <n_input_chunks> <n_output_chunks>"
  exit 1
fi

input_dir=${HOME}/Scratch/data/recon/${setup}/5mGy/${n_projections}proj/processed-3sigma/protons
output_dir=${HOME}/Scratch/data/recon/${setup}/5mGy/${n_projections}proj/processed-3sigma/b

export PYTHONPATH=${HOME}/pct-recon
pushd $PYTHONPATH >/dev/null

python recon/preprocess/b_vec.py \
  ${n_projections} \
  ${n_input_chunks} \
  ${n_output_chunks} \
  ${input_dir} \
  ${output_dir}

popd >/dev/null
