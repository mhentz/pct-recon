#!/bin/bash
#
# List the FISTA logs (fista.job output, e.g. fista.ci.5.0.720.dtv.1e-4.2e-1.o56750)
# without "END OF RECONSTRUCTION" whose job is no longer in qstat, i.e. runs that
# stopped early (wall-clock limit, memory). The job ID is the last field of the name.
#
# Usage: find_unfinished_fista.sh fista.ci.*

__usage="Usage: $(basename "$0") FILE [FILE ...]

List the FISTA logs among FILE ... that do not contain \"END OF RECONSTRUCTION\"
and whose job is no longer queued or running."

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

qstat_output=$(qstat)

for f in "${files[@]}"; do
    job_id=$(echo "$f" | rev | cut -d "." -f 1 | rev | cut -c 2-)

    if [ -z "$(grep "END OF RECONSTRUCTION" "$f")" ] && \
        [ -z "$(echo $qstat_output | grep -w "$job_id")" ]; then
            echo "$f"
    fi
done
