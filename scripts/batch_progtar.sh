#!/bin/bash -e

for d in ~/Scratch/data/recon/*/*/*/*/b*; do
    pushd "$(dirname "$d")" >/dev/null

    src=$(basename "$d")
    archive_destination=$(echo $d | sed -e "s/Scratch/ACFS/g").tar
    ~/pct-recon/scripts/progtar.sh -f $archive_destination $src

    popd >/dev/null
done
