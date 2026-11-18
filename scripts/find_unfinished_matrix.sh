#!/bin/bash

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
    if [[ "$f" != *mat.* ]]; then
        echo "$(basename $f) is not a matrix log file"
        exit
    fi
done

qstat_output=$(qstat)

for f in "${files[@]}"; do
    job_id=$(echo "$f" | rev | cut -d "." -f 1 | rev | cut -c 2-)

    if [ -z "$(grep "\- Done" "$f")" ] && \
        [ -z "$(echo $qstat_output | grep -w "$job_id")" ]; then
            echo "$f"
    fi
done
