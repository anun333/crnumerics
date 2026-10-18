#!/bin/sh
# run.sh MODEL.safetensors NGEN OUTBASE: build the three binaries, run each on
# prompts.tsv (the three at once, or one at a time with SERIAL=1), and print
# their chain hashes. OUTBASE gets
# one directory per build, the logs, and host.txt.
set -e
[ $# -eq 3 ] || { echo "usage: run.sh MODEL.safetensors NGEN OUTBASE"; exit 2; }
M=$1 N=$2 O=$3
cd "$(dirname "$0")"
make -s
for m in cr tree stock; do mkdir -p "$O/$m"; done
# SERIAL=1: one at a time, on a machine with few cores to spare
for m in cr tree stock; do
  if [ "${SERIAL:-0}" = 1 ]; then
    ./layerhash-$m "$M" prompts.tsv "$N" "$O/$m" > "$O/$m.chain" 2> "$O/$m.log" || true
  else
    ./layerhash-$m "$M" prompts.tsv "$N" "$O/$m" > "$O/$m.chain" 2> "$O/$m.log" &
  fi
done
wait
{ uname -a; ${CC:-cc} --version 2>&1 | head -1; } > "$O/host.txt"
for m in cr tree stock; do
  [ -s "$O/$m.chain" ] || { echo "VOID: $m produced no chain hash (see $O/$m.log)"; exit 1; }
done
cat "$O"/cr.chain "$O"/tree.chain "$O"/stock.chain
