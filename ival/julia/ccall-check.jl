# ccall-check.jl LIBIVAL: ival called from Julia through ccall, against IntervalArithmetic.jl's default rounding
# (:correct) on the same intervals, bound for bound, for exp, exp2, exp10, tanh, cbrt, asinh, acosh and atanh. Also
# whether Julia still runs in round-to-nearest after each call: libc's fegetround, and a Julia addition (1 + 2^-60)
# that gives 1.0 only to nearest. ival sets the rounding mode inside each call and restores it before returning, so
# no code Julia compiles runs in another mode. The checks are tested first (VOID if a check could not fail).
using IntervalArithmetic, Random, Libdl
const LIB = Libdl.dlopen(ARGS[1])
const N = 4096
fegetround() = ccall(:fegetround, Cint, ())
const NEAREST = fegetround()
probe(a, b) = a[] + b[]
const A, B = Ref(1.0), Ref(0x1p-60)
nearest_now() = fegetround() == NEAREST && probe(A, B) == 1.0

# the checks must be able to fail: another mode is seen, and a bound one ulp off is not the same
nearest_now() || (println("VERDICT: VOID (not in round-to-nearest at the start)"); exit(2))
tested = 0
for r in (RoundDown, RoundUp, RoundToZero)   # the platform's FE_ constants, through Julia's own table
    m = Base.Rounding.to_fenv(r)
    ccall(:fesetround, Cint, (Cint,), m) == 0 || continue
    seen = fegetround() != NEAREST && (r != RoundUp || probe(A, B) != 1.0)   # upward also moves 1 + 2^-60
    ccall(:fesetround, Cint, (Cint,), NEAREST)
    seen || (println("VERDICT: VOID ($r not detected)"); exit(2))
    global tested += 1
end
tested > 0 || (println("VERDICT: VOID (no other rounding mode could be set to test the check)"); exit(2))
nearest_now() || (println("VERDICT: VOID (round-to-nearest not restored after the control)"); exit(2))
let y = exp(interval(1.0, 1.001))
    prevfloat(inf(y)) === inf(y) && (println("VERDICT: VOID (the bound comparison cannot fail)"); exit(2))
end

function check(name, sym, lo, hi)
    ylo, yhi = similar(lo), similar(hi)
    fp = Libdl.dlsym(LIB, sym)
    ok = nearest_now()
    ccall(fp, Cvoid, (Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Csize_t), lo, hi, ylo, yhi, N)
    ok &= nearest_now()
    ys = getfield(Base, Symbol(name)).(interval.(lo, hi))
    same = count(i -> inf(ys[i]) === ylo[i] && sup(ys[i]) === yhi[i], 1:N)
    println(rpad(name, 6), " $same of $N intervals the same as :correct; round-to-nearest before and after the call: $ok")
    return same == N && ok
end
rng = MersenneTwister(784)
lohi(a, b, w) = (lo = a .+ (b - a) .* rand(rng, N); (lo, lo .+ w .* rand(rng, N)))
ok = true
for (name, sym, a, b, w) in (("exp", :ival_exp, -700.0, 700.0, 1e-3), ("exp2", :ival_exp2, -1000.0, 1000.0, 1e-3),
                             ("exp10", :ival_exp10, -300.0, 300.0, 1e-4), ("tanh", :ival_tanh, -20.0, 20.0, 1e-2),
                             ("cbrt", :ival_cbrt, -1e6, 1e6, 1.0), ("asinh", :ival_asinh, -1e6, 1e6, 1.0),
                             ("acosh", :ival_acosh, 1.0, 1e6, 1.0), ("atanh", :ival_atanh, -0.999, 0.99, 1e-3))
    global ok &= check(name, sym, lohi(a, b, w)...)
end
println(ok ? "VERDICT: IDENTICAL, and Julia stays in round-to-nearest (IntervalArithmetic $(pkgversion(IntervalArithmetic)), Julia $VERSION)" :
             "VERDICT: DIFFERS")
exit(ok ? 0 : 1)
