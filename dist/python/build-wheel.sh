#!/bin/sh
# build-wheel.sh: a wheel of crdistr (crdist for Python) that carries libcrdist.so.0 (libcrdist.0.dylib on macOS),
# so that pip install needs nothing else. Builds the library at crnumerics' root (make build/libcrdist.so), copies it
# and the licenses (crnumerics' and CORE-MATH's, whose notices the MIT license asks a binary copy to keep) into
# src/crdistr for the build, then removes them. With auditwheel on the path, the wheel is then checked and retagged
# manylinux. Writes dist/. A wheel for PyPI would be built inside quay.io/pypa/manylinux_2_28_x86_64 (or _aarch64),
# as ival's is. 2026-10-13, from ival/python/build-wheel.sh.
set -eu
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
# On macOS the library is libcrdist.dylib, carried as libcrdist.0.dylib; MACOSX_DEPLOYMENT_TARGET (11.0 if unset,
# the first macOS on Apple Silicon) sets both the library's minimum and the wheel's tag, and delocate, when on the
# path, checks that the wheel needs nothing outside macOS (as ival's build-wheel.sh).
case "$(uname -s)" in
  Darwin) lib=libcrdist.dylib; dst=libcrdist.0.dylib; export MACOSX_DEPLOYMENT_TARGET="${MACOSX_DEPLOYMENT_TARGET:-11.0}"; mk="CC=clang" ;;
  *) lib=libcrdist.so; dst=libcrdist.so.0; mk= ;;
esac
make -C "$root" -s $mk "build/$lib"
cp "$root/build/$lib" "$here/src/crdistr/$dst"
cp "$root/LICENSE" "$here/src/crdistr/LICENSE"
cp "$root/core-math/LICENSE" "$here/src/crdistr/LICENSE-CORE-MATH"
trap 'rm -f "$here/src/crdistr/$dst" "$here/src/crdistr/LICENSE" "$here/src/crdistr/LICENSE-CORE-MATH"; rm -rf "$here/build" "$here/src/"*.egg-info' EXIT
cd "$here"
python3 -m pip wheel --no-deps --no-build-isolation -w dist . > /dev/null
w=$(ls -t dist/*.whl | head -1)
if command -v auditwheel > /dev/null; then
  auditwheel repair -w dist "$w" && rm -f "$w"
fi
if command -v delocate-wheel > /dev/null; then
  delocate-listdeps --all "$w"
  delocate-wheel -v "$w"
fi
ls -l dist/
