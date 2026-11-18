#!/bin/bash
set -euo pipefail

# Tar a directory (files in version-sort order) with a pv progress bar; -z gzips.
usage() {
  echo "Usage: $(basename "$0") [-z] -f ARCHIVE SRC_DIR"
  exit 1
}

compress=0
archive=""
while getopts "zf:" opt; do
  case $opt in
    z) compress=1 ;;
    f) archive="$OPTARG" ;;
    *) usage ;;
  esac
done
shift $((OPTIND-1))

if [[ -z "$archive" || $# -ne 1 ]]; then
  usage
fi
src="$1"

# total size, for the pv progress bar
TOTAL=$(find "$src" -type f -print0 \
        | du --files0-from=- -cb 2>/dev/null \
        | tail -n1 | awk '{print $1}')

echo "Creating $archive"

if [[ $compress -eq 1 ]]; then
  find "$src" -type f \
    | sort -V \
    | tar -cf - --files-from=- \
    | pv -s "$TOTAL" -pterb \
    | gzip -c > "$archive"
else
  find "$src" -type f \
    | sort -V \
    | tar -cf - --files-from=- \
    | pv -s "$TOTAL" -pterb \
    > "$archive"
fi
