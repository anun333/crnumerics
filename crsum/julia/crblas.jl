# crsum/julia/crblas.jl: Julia's own matrix products, correctly rounded, by
# pointing Julia's BLAS switchboard (libblastrampoline) at crblas.c, with
# the multiplying still done by Julia's OpenBLAS. Checked here against an
# exact BigFloat reference, on data built to make floating-point GEMMs
# round differently (wide exponents, rows that cancel).
#   make build/libcrblas.so && julia crsum/julia/crblas.jl
# Ends in a VERDICT line; exit 0 only if every check holds.
using LinearAlgebra, Random

lib = abspath(get(ARGS, 1, joinpath(@__DIR__, "..", "..", "build", "libcrblas.so")))
isfile(lib) || error("no $lib: make build/libcrblas.so first")
openblas = BLAS.get_config().loaded_libs[1].libname   # Julia's own BLAS, before the switch

# the exact product, rounded once to nearest: every product of two doubles
# and every partial sum here fits in 2400 bits (exponents stay within 2^±60)
exact(A, B, β = 0.0, C = zeros(size(A, 1), size(B, 2))) = setprecision(BigFloat, 2400) do
    Float64.(big.(A) * big.(B) .+ big(β) .* big.(C))
end
ndiff(X, Y) = count(i -> reinterpret(UInt64, X[i]) != reinterpret(UInt64, Y[i]), eachindex(X, Y))

rng = Xoshiro(20261001)
function hard(m, n)   # wide exponents, and each odd column cancelling the even one before it
    M = randn(rng, m, n) .* exp2.(rand(rng, -20:20, m, n))
    for j in 2:2:n; M[:, j] .= -M[:, j-1] .* (1 .+ 2.0^-30 .* randn(rng, m)); end
    M
end
A = hard(150, 96); B = permutedims(hard(110, 96)); C0 = randn(rng, 150, 110)
R = exact(A, B)

ok = true
check(what, cond) = (println(rpad(what, 72), cond ? "yes" : "NO"); global ok &= cond)

println("Julia $(VERSION), BLAS before: ", BLAS.get_config())
BLAS.set_num_threads(1); P1 = A * B
BLAS.set_num_threads(8); P8 = A * B
println("control: OpenBLAS's A*B differs from the exact product on $(ndiff(P1, R)) of $(length(R)) elements (1 thread), $(ndiff(P8, R)) (8 threads)")
check("control: OpenBLAS differs from the exact product (must)", ndiff(P1, R) > 0)

BLAS.lbt_forward(lib; clear = false, suffix_hint = "64_")
println("BLAS after:  ", BLAS.get_config())
@assert ccall((:crblas_set_inner, lib), Cint, (Cstring,), openblas) == 0
n0 = ccall((:crblas_gemm_calls, lib), Clong, ())
Q8 = A * B
check("A*B now goes through crblas (its call count moved)", ccall((:crblas_gemm_calls, lib), Clong, ()) > n0)
check("A*B, inner OpenBLAS on 8 threads: 0 differ from the exact product", ndiff(Q8, R) == 0)
BLAS.set_num_threads(1); Q1 = A * B
check("A*B, inner OpenBLAS on 1 thread: the same bits", ndiff(Q1, Q8) == 0)
ccall((:crblas_set_inner, lib), Cint, (Cstring,), C_NULL); Qi = A * B
check("A*B, crsum's internal GEMM: the same bits", ndiff(Qi, Q8) == 0)
ccall((:crblas_set_inner, lib), Cint, (Cstring,), openblas); BLAS.set_num_threads(8)

At = permutedims(A); Bt = permutedims(B)
check("A'*B' (both transposed): 0 differ", ndiff(At' * Bt', R) == 0)
check("A*B' and A'*B, mixed: 0 differ", ndiff(A * Bt', R) == 0 && ndiff(At' * B, R) == 0)
C = copy(C0); mul!(C, A, B, 1.0, 1.0)
check("mul!(C, A, B, 1, 1) = AB + C rounded once: 0 differ", ndiff(C, exact(A, B, 1.0, C0)) == 0)
C = copy(C0); mul!(C, A, B, 2.0, -0.5)
check("mul!(C, A, B, 2, -0.5) = (2A)B - C/2 rounded once: 0 differ", ndiff(C, exact(2 .* A, B, -0.5, C0)) == 0)
C = copy(C0); mul!(C, A, B, 0.1, 0.0)
check("mul!(C, A, B, 0.1, 0) = (0.1 A rounded) B, rounded once: 0 differ", ndiff(C, exact(0.1 .* A, B)) == 0)

# the dot products crblas exports for libblastrampoline's probes, called
# directly (Julia's own dot goes through CBLAS, cblas_sdot64_ and the like,
# which crblas leaves to OpenBLAS)
x = Float32.(hard(1, 4096)[:]); y = Float32.(randn(rng, 4096))
zx = complex.(hard(1, 2048)[:], hard(1, 2048)[:]); zy = complex.(randn(rng, 2048), randn(rng, 2048)); cx = ComplexF32.(zx); cy = ComplexF32.(zy)
ex(T, u, v) = setprecision(BigFloat, 2400) do; T(sum(conj.(big.(u)) .* big.(v))); end
I = (Ref{Int64}, Ptr{Cvoid}, Ref{Int64}, Ptr{Cvoid}, Ref{Int64})
sd = ccall((:sdot_64_, lib), Float32, (Ref{Int64}, Ptr{Float32}, Ref{Int64}, Ptr{Float32}, Ref{Int64}), length(x), x, 1, y, 1)
zd = ccall((:zdotc_64_, lib), ComplexF64, (Ref{Int64}, Ptr{ComplexF64}, Ref{Int64}, Ptr{ComplexF64}, Ref{Int64}), length(zx), zx, 1, zy, 1)
cd = ccall((:cdotc_64_, lib), ComplexF32, (Ref{Int64}, Ptr{ComplexF32}, Ref{Int64}, Ptr{ComplexF32}, Ref{Int64}), length(cx), cx, 1, cy, 1)
check("sdot, zdotc, cdotc (_64_, called directly): exact, rounded once",
      sd === ex(Float32, x, y) && zd === ex(ComplexF64, zx, zy) && cd === ex(ComplexF32, cx, cy))
println("(Julia's dot(x, y) stays OpenBLAS's: ", dot(x, y) === sd ? "the same as crblas here" : "differs from crblas's sdot here", ")")

println(ok ? "VERDICT: IDENTICAL: Julia's products through crblas are the exact products rounded once, on any thread count or inner GEMM" :
             "VERDICT: DIFFER")
exit(ok ? 0 : 1)
