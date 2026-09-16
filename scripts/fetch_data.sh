#!/usr/bin/env bash
# Fetch MovieLens 10M (63MB download, 253MB extracted) and convert it to parquet.
#
# The raw .dat files are deleted afterwards: they are 5x the size of the parquet, the
# conversion is deterministic, and anyone can re-run this script. Keeping a quarter-gig
# of re-derivable text around is how a repo becomes un-clonable.
set -euo pipefail

RAW_DIR="data/raw"
URL="https://files.grouplens.org/datasets/movielens/ml-10m.zip"
KEEP_RAW="${KEEP_RAW:-0}"

mkdir -p "$RAW_DIR"

if [ ! -d "$RAW_DIR/ml-10M100K" ]; then
  echo "== downloading MovieLens 10M (63MB)"
  curl -fL --progress-bar -o "$RAW_DIR/ml-10m.zip" "$URL"
  echo "== extracting"
  unzip -q -o "$RAW_DIR/ml-10m.zip" -d "$RAW_DIR"
  rm -f "$RAW_DIR/ml-10m.zip"
fi

echo "== converting to parquet"
PYTHONPATH=src python -m activation.ingest

if [ "$KEEP_RAW" != "1" ]; then
  echo "== removing raw .dat files (set KEEP_RAW=1 to keep them)"
  rm -rf "$RAW_DIR/ml-10M100K"
fi

echo "== done"
