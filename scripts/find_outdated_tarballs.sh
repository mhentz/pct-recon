#!/bin/bash

for tarball in ~/ACFS/data/recon/*/*/*/*/matrix*; do
    tarball_mtime=$(stat -c %Y $tarball)

    scratch_dir=$(dirname "$(echo "$tarball" | sed 's/ACFS/Scratch/')")

    filename=$(basename $tarball)
    matrix_dir=${scratch_dir}/${filename%.tar}

    newest_scratch_file=${matrix_dir}/$(ls -rt ${matrix_dir} | tail -n 1)
    newest_scratch_file_mtime=$(stat -c %Y ${newest_scratch_file})

    if [ ! -d $scratch_dir ]; then
        echo "$(dirname $tarball) does not exist in Scratch."
        continue
    fi

    if [ $tarball_mtime -lt $newest_scratch_file_mtime ]; then
        echo "$tarball is older than files in Scratch"
        # echo "        tarball mtime $tarball_mtime < scratch mtime $newest_scratch_file_mtime"
    fi
done
