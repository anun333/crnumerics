#!/bin/sh
# build-wheel.sh: a wheel of ival that carries libival.so.0, so that pip install needs nothing else. Builds the
# library at crnumerics' root (make build/libival.so), copies it and the licenses (crnumerics' and CORE-MATH's,
# whose notices the MIT license asks a binary copy to keep) into src/ival for the build, then removes them.
# With auditwheel on the path, the wheel is then checked and retagged manylinux. Writes dist/.
# The tag follows the glibc it was built on: on Ubuntu 24.04 manylinux_2_38 (fmod is versioned 2.38, dlopen 2.34).
# A wheel for PyPI is built inside quay.io/pypa/manylinux_2_28_x86_64 (or _aarch64) instead, for glibc 2.28 and up.
set -eu
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
# On macOS the library is libival.dylib, carried as libival.0.dylib; MACOSX_DEPLOYMENT_TARGET (11.0 if unset, the
# first macOS on Apple Silicon) sets both the library's minimum and the wheel's tag, and delocate, when on the path,
# checks that the wheel needs nothing outside macOS.
case "$(uname -s)" in
  Darwin) lib=libival.dylib; dst=libival.0.dylib; export MACOSX_DEPLOYMENT_TARGET="${MACOSX_DEPLOYMENT_TARGET:-11.0}" ;;
  *) lib=libival.so; dst=libival.so.0 ;;
esac
make -C "$root" -s "build/$lib"
cp "$root/build/$lib" "$here/src/ival/$dst"
cp "$root/LICENSE" "$here/src/ival/LICENSE"
cp "$root/core-math/LICENSE" "$here/src/ival/LICENSE-CORE-MATH"
trap 'rm -f "$here/src/ival/$dst" "$here/src/ival/LICENSE" "$here/src/ival/LICENSE-CORE-MATH"; rm -rf "$here/build" "$here/src/ival.egg-info"' EXIT
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
