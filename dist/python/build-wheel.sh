#!/bin/sh
# build-wheel.sh: a wheel of crdist that carries libcrdist.so.0, so that pip install needs nothing else (Linux;
# crdist is not built on macOS yet). Builds the library at crnumerics' root (make build/libcrdist.so), copies it
# and the licenses (crnumerics' and CORE-MATH's, whose notices the MIT license asks a binary copy to keep) into
# src/crdist for the build, then removes them. With auditwheel on the path, the wheel is then checked and retagged
# manylinux. Writes dist/. A wheel for PyPI would be built inside quay.io/pypa/manylinux_2_28_x86_64 (or _aarch64),
# as ival's is. 2026-10-13, from ival/python/build-wheel.sh.
set -eu
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
make -C "$root" -s build/libcrdist.so
cp "$root/build/libcrdist.so" "$here/src/crdist/libcrdist.so.0"
cp "$root/LICENSE" "$here/src/crdist/LICENSE"
cp "$root/core-math/LICENSE" "$here/src/crdist/LICENSE-CORE-MATH"
trap 'rm -f "$here/src/crdist/libcrdist.so.0" "$here/src/crdist/LICENSE" "$here/src/crdist/LICENSE-CORE-MATH"; rm -rf "$here/build" "$here/src/crdist.egg-info"' EXIT
cd "$here"
python3 -m pip wheel --no-deps --no-build-isolation -w dist . > /dev/null
w=$(ls -t dist/*.whl | head -1)
if command -v auditwheel > /dev/null; then
  auditwheel repair -w dist "$w" && rm -f "$w"
fi
ls -l dist/
