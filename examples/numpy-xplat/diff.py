#!/usr/bin/env python3
"""diff.py REF_DIR DIR...: numpy_xplat.py's saved arrays (DIR/arrays/TEST.npy), each host's against the
reference host's, element by element: how many differ, and by how many units in the last place (floats) or how much
(integers). 2026-10-14."""
import os, sys
import numpy as np

def ulps(a, b):
    """|a - b| in ulps, through the ordered integer view of IEEE floats (finite values; NaN and inf counted apart)"""
    it = np.int64 if a.dtype == np.float64 else np.int32
    def key(x):
        i = x.view(it).astype(np.int64)
        return np.where(i < 0, np.int64(np.iinfo(it).min) - i, i)
    return np.abs(key(a) - key(b))

ref, others = sys.argv[1], sys.argv[2:]
names = sorted(f[:-4] for f in os.listdir(os.path.join(ref, "arrays")))
print("%-30s %s" % ("test", "   ".join("%-26s" % os.path.basename(o.rstrip("/")).replace("arr-", "") for o in others)))
for n in names:
    a = np.load(os.path.join(ref, "arrays", n + ".npy")).ravel()
    cells = []
    for o in others:
        b = np.load(os.path.join(o, "arrays", n + ".npy")).ravel()
        d = a != b
        if a.dtype.kind == "f":
            d = d & ~(np.isnan(a) & np.isnan(b))
        k = int(d.sum())
        if not k:
            cells.append("same")
        elif a.dtype.kind == "f":
            fin = d & np.isfinite(a) & np.isfinite(b)
            u = ulps(a[fin], b[fin]) if fin.any() else np.zeros(1, dtype=np.int64)
            cells.append("%d (%.3g%%), max %d ulp" % (k, 100.0 * k / a.size, int(u.max())))
        else:
            cells.append("%d (%.3g%%), max %d" % (k, 100.0 * k / a.size, int(np.abs(a[d].astype(np.int64) - b[d]).max())))
    print("%-30s %s" % (n, "   ".join("%-26s" % c for c in cells)))
