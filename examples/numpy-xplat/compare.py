#!/usr/bin/env python3
"""compare.py DIR: the reports numpy_xplat.py wrote, one HOST.json each, compared test by test. Prints each
host's platform, then per test the hosts grouped by identical result (equivalence classes), and a summary per group of
tests. 2026-10-14."""
import json, os, sys, collections
d = sys.argv[1]
reps = {f[:-5]: json.load(open(os.path.join(d, f))) for f in sorted(os.listdir(d)) if f.endswith(".json")}
hosts = list(reps)
for h, r in reps.items():
    feats = r["simd"].get("found") if isinstance(r.get("simd"), dict) else None
    lib = (r["libc"][0] + " " + r["libc"][1] if r["libc"][0] else "macOS " + r["mac"] if r.get("mac") else
           "Windows " + " ".join(r["windows"][:2]) if r.get("windows") and r["windows"][0] else r["platform"])
    print("%-14s %-7s %s numpy %s python %s %s%s" % (h, r["machine"], lib,
          r["numpy"], r["python"], "SIMD found " + ",".join(feats) if feats else "", ""))
tests = sorted(set().union(*(r["tests"] for r in reps.values())))
summary = collections.defaultdict(lambda: [0, 0])
print()
for t in tests:
    cls = collections.OrderedDict()
    absent = [h for h in hosts if t not in reps[h]["tests"]]   # not run there (crdistr has no wheel): not a difference
    for h in hosts:
        if t in reps[h]["tests"]:
            e = reps[h]["tests"][t]
            # integer draws compare by value (older reports have no "values": their int64 bytes are the values)
            cls.setdefault(e.get("values", e["sha256"]), []).append(h)
    grp = t.split(".")[0]
    summary[grp][0] += 1
    if len(cls) > 1:
        summary[grp][1] += 1
        print("%-34s %s%s" % (t, " | ".join("{" + ",".join(v) + "}" for v in cls.values()), "   (absent: " + ",".join(absent) + ")" if absent else ""))
    dts = collections.defaultdict(list)
    for h in hosts:
        if t in reps[h]["tests"]:
            dts[reps[h]["tests"][t]["dtype"]].append(h)
    if len(dts) > 1:
        print("%-34s dtype %s" % (t, " | ".join(k + " {" + ",".join(v) + "}" for k, v in dts.items())))
print()
for g, (n, dif) in summary.items():
    print("%-8s %3d tests, %3d differ between hosts" % (g, n, dif))
