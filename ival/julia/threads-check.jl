# threads-check.jl LIBIVAL (run with julia -t 4 or more): ival from several Julia threads at once, through ccall,
# against IntervalArithmetic.jl's :correct, while a sentinel task on another thread keeps checking that its own
# rounding mode stays round-to-nearest. ival changes the mode inside each call, and the mode is per thread (MXCSR,
# FPCR), so no other thread should ever see it change. Prints the threads each task ran on, how many calls and
# sentinel checks overlapped in time, mismatches against :correct, and modes other than nearest seen by the sentinel.
using IntervalArithmetic, Random, Libdl
Threads.nthreads() >= 3 || (println("VERDICT: VOID (run with julia -t 4: needs a sentinel and at least two workers)"); exit(2))
const LIB = Libdl.dlopen(ARGS[1])
const N, CHUNKS = 4096, 8
fegetround() = ccall(:fegetround, Cint, ())
const NEAREST = fegetround()
probe(a, b) = a[] + b[]

rng = MersenneTwister(784)
lohi(a, b, w) = (lo = a .+ (b - a) .* rand(rng, N); (lo, lo .+ w .* rand(rng, N)))
FUNCS = (("exp", :ival_exp, -700.0, 700.0, 1e-3), ("exp2", :ival_exp2, -1000.0, 1000.0, 1e-3),
         ("exp10", :ival_exp10, -300.0, 300.0, 1e-4), ("tanh", :ival_tanh, -20.0, 20.0, 1e-2),
         ("cbrt", :ival_cbrt, -1e6, 1e6, 1.0), ("asinh", :ival_asinh, -1e6, 1e6, 1.0),
         ("acosh", :ival_acosh, 1.0, 1e6, 1.0), ("atanh", :ival_atanh, -0.999, 0.99, 1e-3))
# inputs and :correct's bounds, computed once on this thread
cases = [begin
             lo, hi = lohi(a, b, w)
             ys = getfield(Base, Symbol(name)).(interval.(lo, hi))
             (Libdl.dlsym(LIB, sym), lo, hi, inf.(ys), sup.(ys))
         end for (name, sym, a, b, w) in FUNCS for _ in 1:CHUNKS]

const running = Threads.Atomic{Bool}(true)
function sentinel()
    a, b = Ref(1.0), Ref(0x1p-60)
    checks, other, tids = 0, 0, Set{Int}()
    t0 = time_ns()
    while running[]
        push!(tids, Threads.threadid())
        checks += 1
        (fegetround() != NEAREST || probe(a, b) != 1.0) && (other += 1)
    end
    (checks, other, tids, t0, time_ns())
end
function worker(k, nworkers, rounds)
    ylo, yhi = Vector{Float64}(undef, N), Vector{Float64}(undef, N)
    calls, bad, tids = 0, 0, Set{Int}()
    t0 = time_ns()
    for _ in 1:rounds, c in k:nworkers:length(cases)
        fp, lo, hi, rlo, rhi = cases[c]
        push!(tids, Threads.threadid())
        ccall(fp, Cvoid, (Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Csize_t), lo, hi, ylo, yhi, N)
        calls += 1
        bad += count(i -> ylo[i] !== rlo[i] || yhi[i] !== rhi[i], 1:N)
    end
    (calls, bad, tids, t0, time_ns())
end

nworkers = Threads.nthreads() - 1
s = Threads.@spawn sentinel()
ws = [Threads.@spawn worker(k, nworkers, 400) for k in 1:nworkers]
wr = fetch.(ws)
running[] = false
sr = fetch(s)

calls = sum(r[1] for r in wr); bad = sum(r[2] for r in wr)
wtids = union((r[3] for r in wr)...)
overlap = min(sr[5], maximum(r[5] for r in wr)) - max(sr[4], minimum(r[4] for r in wr))
println("workers: $nworkers tasks on threads $(sort(collect(wtids))), $calls ival calls of $N intervals, $bad bounds differing from :correct")
println("sentinel: thread(s) $(sort(collect(sr[3]))), $(sr[1]) checks, $(sr[2]) saw a mode other than nearest; overlap with the workers $(round(overlap / 1e9, digits=2)) s")
shared = !isempty(intersect(sr[3], wtids))
ok = bad == 0 && sr[2] == 0 && calls > 0 && sr[1] > 0 && overlap > 0
verdict = !ok ? "DIFFERS" : shared ? "VOID (the sentinel shared a thread with a worker; run with more threads)" :
          "IDENTICAL from $nworkers threads at once, and the sentinel thread never left round-to-nearest"
println("VERDICT: $verdict (IntervalArithmetic $(pkgversion(IntervalArithmetic)), Julia $VERSION)")
exit(ok && !shared ? 0 : 1)
