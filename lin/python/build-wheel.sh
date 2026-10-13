#!/bin/sh
# build-wheel.sh: a wheel of crlin (verified linear systems for Python) that carries libcrlin.so.0 (libcrlin.0.dylib on macOS),
# so that pip install needs nothing else. Builds the library at crnumerics' root (make build/libcrlin.so), copies it
# and crnumerics' license (whose notice the MIT license asks a binary copy to keep; crlin carries no CORE-MATH code) into
# src/crlin for the build, then removes them. With auditwheel on the path, the wheel is then checked and retagged
# manylinux. Writes dist/. A wheel for PyPI would be built inside quay.io/pypa/manylinux_2_28_x86_64 (or _aarch64),
# as ival's is. 2026-10-13, from dist/python/build-wheel.sh.
set -eu
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
# On macOS the library is libcrlin.dylib, carried as libcrlin.0.dylib; MACOSX_DEPLOYMENT_TARGET (11.0 if unset,
# the first macOS on Apple Silicon) sets both the library's minimum and the wheel's tag, and delocate, when on the
# path, checks that the wheel needs nothing outside macOS (as ival's build-wheel.sh).
case "$(uname -s)" in
  Darwin) lib=libcrlin.dylib; dst=libcrlin.0.dylib; export MACOSX_DEPLOYMENT_TARGET="${MACOSX_DEPLOYMENT_TARGET:-11.0}"; mk="CC=clang" ;;
  *) lib=libcrlin.so; dst=libcrlin.so.0; mk= ;;
esac
make -C "$root" -s $mk "build/$lib"
cp "$root/build/$lib" "$here/src/crlin/$dst"
cp "$root/LICENSE" "$here/src/crlin/LICENSE"
trap 'rm -f "$here/src/crlin/$dst" "$here/src/crlin/LICENSE"; rm -rf "$here/build" "$here/src/"*.egg-info' EXIT
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
