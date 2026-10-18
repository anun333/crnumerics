#!/bin/bash
# build.sh: the GPU builds of layerhash (layerhash.cu), on a machine with
# CUDA, from this checkout: layerhash-gpu-tree and layerhash-gpu-treefma.
# crsum.c and crnn.c become device code through devify-tree.py, CORE-MATH's
# binary64 exp, log1p, erfc and rsqrt through devify.sh (only "__device__"
# added, plus the two edits below), into gen/. NVCC=... to choose nvcc.
set -euo pipefail
cd "$(dirname "$0")"
TOP=../../..
NVCC=${NVCC:-$(command -v nvcc || echo /usr/local/cuda/bin/nvcc)}
mkdir -p gen
for f in exp log1p erfc rsqrt; do ./devify.sh $TOP/core-math/$f.c > gen/$f.cuh; done
python3 devify-tree.py $TOP gen/dev crsum/crsum.h crsum/crsum.c nn/crnn.h nn/crnn.c nn/crnn-fast.h nn/crnn-exceptions.h
# C++: crnn's xmalloc returns the converting pointer; crsum's bins per thread
sed -i 's/^__device__ static void \*xmalloc(size_t n)/__device__ static crdev_ptr xmalloc(size_t n)/' gen/dev/nn/crnn.c
sed -i 's|^static _Thread_local uint64_t crsum_tbin\[4096\];$|/* crsum_tbin: per GPU thread, crdev_tbin() in layerhash.cu */|' gen/dev/crsum/crsum.c
grep -q '^__device__ static crdev_ptr xmalloc' gen/dev/nn/crnn.c || { echo "VOID: crnn.c's xmalloc not replaced"; exit 2; }
grep -q 'crsum_tbin: per GPU thread' gen/dev/crsum/crsum.c && ! grep -q '_Thread_local' gen/dev/crsum/crsum.c \
  || { echo "VOID: crsum.c's thread-local bins not replaced"; exit 2; }
# the host's RoPE tables: CORE-MATH's double sin, cos and pow, as the CPU builds
F="-O2 -ffp-contract=off -frounding-math"
for f in sin cos pow/pow; do ${CC:-cc} $F -c -o gen/$(basename $f).o $TOP/core-math/$f.c; done
NV="$NVCC -O2 -arch=native -fmad=false -diag-suppress 1041,177,550,20037 -I gen -I .."
$NV -DBUILD_FLAGS='"nvcc -O2 -arch=native -fmad=false"' -o layerhash-gpu-tree layerhash.cu gen/sin.o gen/cos.o gen/pow.o -lm
$NV -DTREEFMA -DBUILD_FLAGS='"nvcc -O2 -arch=native -fmad=false -DTREEFMA"' -o layerhash-gpu-treefma layerhash.cu gen/sin.o gen/cos.o gen/pow.o -lm
echo "built layerhash-gpu-tree layerhash-gpu-treefma"
