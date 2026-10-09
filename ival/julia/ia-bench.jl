# ia-bench.jl: the Julia side of make ival-julia-compare. IntervalArithmetic.jl's default rounding (:correct: CRlibm
# where it has the function, MPFR otherwise) on ia-bench.c's intervals: ns per interval for exp, exp2 and tanh (the
# least of 7 passes after a warm-up), and whether every bound equals ival's. Both claim the tightest enclosure, so
# they must agree bit for bit; the verdict says whether they do.
using IntervalArithmetic
const N = 4096
v = reinterpret(Float64, read("ivals.bin"))
length(v) == 8N || (println("VERDICT: VOID (ivals.bin holds $(length(v)) numbers, not $(8N))"); exit(2))
lo, hi = v[1:N], v[N+1:2N]
xs = [interval(lo[i], hi[i]) for i in 1:N]
differ = 0
for (k, (name, f)) in enumerate((("exp", exp), ("exp2", exp2), ("tanh", tanh)))
    yl, yh = v[2N*k+1:2N*k+N], v[2N*k+N+1:2N*k+2N]
    ys = f.(xs)
    best = minimum(begin t = time_ns(); f.(xs); (time_ns() - t) / N end for _ in 1:7)
    same = count(i -> inf(ys[i]) == yl[i] && sup(ys[i]) == yh[i], 1:N)
    global differ += N - same
    println("IntervalArithmetic.jl $(rpad(name, 5)) $(lpad(round(best, digits=1), 7)) ns, $same of $N intervals the same as ival's")
end
println(differ == 0 ? "VERDICT: IDENTICAL (IntervalArithmetic $(pkgversion(IntervalArithmetic)), Julia $VERSION: every bound the same as ival's)" :
                      "VERDICT: DIFFERS ($differ intervals)")
exit(differ == 0 ? 0 : 1)
