#!/bin/bash
#
# List the FISTA logs (fista.job output, e.g. fista.ci.5.0.720.dtv.1e-4.2e-1.o56750)
# that contain "END OF RECONSTRUCTION", i.e. whose run finished.
# The ls flags need GNU ls, as on Myriad.
#
# Usage: find_finished_fista.sh fista.ci.*

__usage="Usage: $(basename "$0") FILE [FILE ...]

List the FISTA logs among FILE ... that contain \"END OF RECONSTRUCTION\"."

if [[ "$1" == "--help" || "$1" == "-h" ]]; then
    echo "$__usage"
    exit 0
fi

files=("$@")

if [ "$#" -lt 1 ]; then
    echo "Error: No files provided."
    echo "Usage: $(basename $0) FILE [FILE ...]"
    exit 1
fi

if [ ! -f "$1" ]; then
    echo "Error: $1 is not a file or does not exist."
    echo "Usage: $(basename $0) FILE [FILE ...]"
    exit 1
fi

for f in "${files[@]}"; do
    if [[ "$f" != *fista.* ]]; then
        echo "$(basename $f) is not a fista log file"
        exit
    fi
done

finished=()
while IFS= read -r f; do
    finished+=("$f")
done < <(grep -l "END OF RECONSTRUCTION" "${files[@]}")

# ls with no arguments would list the current directory
if [ "${#finished[@]}" -eq 0 ]; then
    exit 0
fi

ls -lvhGtr --color=auto "${finished[@]}"
