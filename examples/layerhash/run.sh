#!/bin/sh
# run.sh MODEL.safetensors NGEN OUTBASE: build the binaries and run each build
# in MODES (default: cr tree treefma stock) on prompts.tsv, all at once, or
# one at a time with SERIAL=1; print their chain hashes. OUTBASE gets one
# directory per build, the logs, and host.txt.
set -e
[ $# -eq 3 ] || { echo "usage: run.sh MODEL.safetensors NGEN OUTBASE"; exit 2; }
M=$1 N=$2 O=$3
cd "$(dirname "$0")"
make -s
MODES=${MODES:-cr tree treefma stock}
for m in $MODES; do mkdir -p "$O/$m"; done
# SERIAL=1: one at a time, on a machine with few cores to spare
for m in $MODES; do
  if [ "${SERIAL:-0}" = 1 ]; then
    ./layerhash-$m "$M" prompts.tsv "$N" "$O/$m" > "$O/$m.chain" 2> "$O/$m.log" || true
  else
    ./layerhash-$m "$M" prompts.tsv "$N" "$O/$m" > "$O/$m.chain" 2> "$O/$m.log" &
  fi
done
wait
{ uname -a; ${CC:-cc} --version 2>&1 | head -1; } > "$O/host.txt"
for m in $MODES; do
  [ -s "$O/$m.chain" ] || { echo "VOID: $m produced no chain hash (see $O/$m.log)"; exit 1; }
done
for m in $MODES; do cat "$O/$m.chain"; done
