#!/bin/bash
# devify.sh SRC.c > OUT.cuh: a copy of a CORE-MATH source as CUDA device
# code. Adds "__device__" to every function
# definition (a line at column 0: an optional static/inline/__attribute__,
# a type, a name, a parameter list, and "{" or nothing; declarations end
# with ";"; a definition whose parameters continue on the next line ends
# with ","; a one-line definition ends with "}"), and to every file-scope
# constant (a line at column 0 starting
# "static const" or "const", which device code can't read otherwise), and to
# every static forward declaration of a function (log1p.c's
# as_log1p_refine: nvcc took the definition's __device__ for it, clang's HIP
# does not). Nothing else changes.
sed -E \
  -e 's/^((static |inline |__attribute__\(\([a-z_]+\)\) )*[A-Za-z_][A-Za-z_0-9]*( ?\*)? *[A-Za-z_][A-Za-z_0-9]* ?\(.*\) *\{?)$/__device__ \1/' \
  -e 's/^((static |inline )*[A-Za-z_][A-Za-z_0-9]*( ?\*)? *[A-Za-z_][A-Za-z_0-9]* ?\([^)]*,)$/__device__ \1/' \
  -e 's/^((static |inline )*[A-Za-z_][A-Za-z_0-9]*( ?\*)? *[A-Za-z_][A-Za-z_0-9]* ?\([^;{]*\) *\{.*\})$/__device__ \1/' \
  -e 's/^((static )?const [A-Za-z_].*[=\[].*)$/__device__ \1/' \
  -e 's/^(static [^=;{]*\([^=;{]*\);)$/__device__ \1/' "$1"
