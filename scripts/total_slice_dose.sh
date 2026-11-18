#!/bin/bash
trap "echo; exit" INT

DIR=${1}
if [ -z ${DIR} ]; then
  echo "Error: no path provided. Provide path to directory that contains scoring data."
  echo "Usage: ./total_slice_dose.txt path/to/scoring"
  exit
fi

tmpfile=$(mktemp /tmp/sliceDoseSummary.XXX)

trap 'rm "${tmpfile}"; echo; exit' INT

for f in ${DIR}/sliceDose*.csv; do
  cat $f | tail -n 1 | cut -d "," -f 4 >>${tmpfile}
done

awk '{ sum += $1 } END { print sum*1000 " mGy"}' ${tmpfile}
rm ${tmpfile}
