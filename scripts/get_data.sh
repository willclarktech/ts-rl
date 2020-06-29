#!/bin/bash
set -o errexit -o nounset -o pipefail
command -v shellcheck > /dev/null && shellcheck "$0"

DATADIR="src/data"
mkdir -p "$DATADIR"

FILENAME="$DATADIR/cars.json"
TMP_FILENAME="$FILENAME.tmp"
wget -O "$TMP_FILENAME" https://storage.googleapis.com/tfjs-tutorials/carsData.json
jq '[.[] | {mpg: .Miles_per_Gallon, horsepower: .Horsepower} | select(.mpg != null and .horsepower != null)]' "$TMP_FILENAME" > "$FILENAME"
rm "$TMP_FILENAME"
