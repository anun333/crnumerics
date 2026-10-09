// compare.cpp: ival against other interval libraries, on the same intervals: time per operation, and each result
// against ival's tightest one (checked against MPFR by ival's own checks): whether it holds it (a result that does not
// is not an enclosure: it misses true values), and how many binary64 values wider it is at its ends.
//
// Libraries, each where it is found (make ival-compare, ival/README.md): ival's tight and accurate modes (the latter
// through crmvec when IVAL_CRMVEC names it); MPFI (MPFR intervals at 53 bits); Boost.Interval (double, its
// rounded_transc_std policy: the C library's functions in the rounding mode set down or up); filib++ (its
// native_switched rounding, normal mode); libieeep1788 (MPFR, the 1788 reference implementation).
// Operations: add, mul, div, exp, log, sin, atan, on 4096 narrow intervals (relative width up to 1e-6) at a time.
// Timing: the least of 5 passes after a warm-up, each at least 20 ms; every result is checked in the same run.
#include <cmath>
#include <cfenv>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <functional>
#include <string>
#include <vector>
extern "C" {
#include "ival.h"
}
#ifdef HAVE_MPFI
#include <mpfi.h>
#endif
#ifdef HAVE_BOOST
#include <boost/numeric/interval.hpp>
#endif
#ifdef HAVE_FILIB
#include <interval/interval.hpp>
#endif
#ifdef HAVE_P1788
#include "p1788/p1788.hpp"
#endif

static double now() { timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }
static uint64_t rs = 0x243f6a8885a308d3ULL;
static double unit() { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return std::ldexp((double)(rs >> 11), -53); }   // C++14: no hexadecimal literals

enum { N = 4096, NOPS = 7 };
static const char *OPS[NOPS] = { "add", "mul", "div", "exp", "log", "sin", "atan" };
static double al[N], ah[N], bl[N], bh[N], tl[N][NOPS], th[N][NOPS], zl[N], zh[N];

// the steps from a to b (a <= b) over the binary64 values: how many ulps wider
static double ulps(double a, double b)
{
  if (a == b) return 0;
  if (std::isinf(a) || std::isinf(b)) return 1e300;
  int64_t x, y;
  std::memcpy(&x, &a, 8); std::memcpy(&y, &b, 8);
  if (x < 0) x = INT64_MIN - x;
  if (y < 0) y = INT64_MIN - y;
  return (double)(y - x);
}
struct row { double ns; long tight, wider, bad; double maxw; };
static row measure(const std::function<void()> &f, int op)
{
  size_t reps = 1;
  for (;;) { double t = now(); for (size_t r = 0; r < reps; r++) f(); if (now() - t > 0.02) break; reps *= 2; }
  double m = 1e30;
  for (int p = 0; p < 5; p++) {
    double t = now();
    for (size_t r = 0; r < reps; r++) f();
    t = (now() - t) / ((double)reps * N) * 1e9;
    if (t < m) m = t;
  }
  row R = { m, 0, 0, 0, 0 };
  for (int k = 0; k < N; k++) {
    double L = tl[k][op], U = th[k][op];
    if (!(zl[k] <= L && zh[k] >= U)) { R.bad++; continue; }   // not holding the tightest: not an enclosure
    double w = ulps(zl[k], L) + ulps(U, zh[k]);
    if (w == 0) R.tight++; else R.wider++;
    if (w > R.maxw) R.maxw = w;
  }
  return R;
}
static void report(const char *lib, const row *r, const bool *have)
{
  std::printf("%-26s", lib);
  for (int op = 0; op < NOPS; op++) {
    if (!have[op]) { std::printf(" %21s", "-"); continue; }
    char w[32];
    if (r[op].bad) std::snprintf(w, sizeof w, "%ld NOT ENCL.", r[op].bad);
    else if (!r[op].wider) std::snprintf(w, sizeof w, "tight");
    else std::snprintf(w, sizeof w, "%.0f%% +%.0f ulp", 100.0 * r[op].wider / N, r[op].maxw);
    std::printf(" %7.1f %13s", r[op].ns, w);
  }
  std::printf("\n");
}

int main()
{
  for (int k = 0; k < N; k++) {
    double x = (unit() - 0.5) * 20, y = 0.5 + unit() * 10;
    al[k] = x; ah[k] = x + std::fabs(x) * unit() * 1e-6;
    bl[k] = y; bh[k] = y + y * unit() * 1e-6;
  }
  // the inputs for exp, sin, atan: al, ah; for log: bl, bh (positive)
  // ival's tight results, the reference
  double L[N], U[N];
  ival_add(al, ah, bl, bh, L, U, N); for (int k = 0; k < N; k++) { tl[k][0] = L[k]; th[k][0] = U[k]; }
  ival_mul(al, ah, bl, bh, L, U, N); for (int k = 0; k < N; k++) { tl[k][1] = L[k]; th[k][1] = U[k]; }
  ival_div(al, ah, bl, bh, L, U, N); for (int k = 0; k < N; k++) { tl[k][2] = L[k]; th[k][2] = U[k]; }
  ival_exp(al, ah, L, U, N); for (int k = 0; k < N; k++) { tl[k][3] = L[k]; th[k][3] = U[k]; }
  ival_log(bl, bh, L, U, N); for (int k = 0; k < N; k++) { tl[k][4] = L[k]; th[k][4] = U[k]; }
  ival_sin(al, ah, L, U, N); for (int k = 0; k < N; k++) { tl[k][5] = L[k]; th[k][5] = U[k]; }
  ival_atan(al, ah, L, U, N); for (int k = 0; k < N; k++) { tl[k][6] = L[k]; th[k][6] = U[k]; }
  bool all[NOPS] = { true, true, true, true, true, true, true };
  std::printf("ns per operation, n = %d narrow intervals; then the results against ival's tight ones: tight,\n"
              "the share wider and the most ulps wider, or how many do not hold the tight result (NOT ENCL.)\n", N);
  std::printf("%-26s", "");
  for (int op = 0; op < NOPS; op++) std::printf(" %21s", OPS[op]);
  std::printf("\n");
  row r[NOPS];
  {  // ival, tight
    std::function<void()> f[NOPS] = {
      [] { ival_add(al, ah, bl, bh, zl, zh, N); }, [] { ival_mul(al, ah, bl, bh, zl, zh, N); },
      [] { ival_div(al, ah, bl, bh, zl, zh, N); }, [] { ival_exp(al, ah, zl, zh, N); }, [] { ival_log(bl, bh, zl, zh, N); },
      [] { ival_sin(al, ah, zl, zh, N); }, [] { ival_atan(al, ah, zl, zh, N); } };
    for (int op = 0; op < NOPS; op++) r[op] = measure(f[op], op);
    report("ival tight", r, all);
  }
  {  // ival, accurate
    std::function<void()> f[NOPS] = {
      [] { ival_add(al, ah, bl, bh, zl, zh, N); }, [] { ival_mul(al, ah, bl, bh, zl, zh, N); },
      [] { ival_div(al, ah, bl, bh, zl, zh, N); }, [] { ival_acc_exp(al, ah, zl, zh, N); },
      [] { ival_acc_log(bl, bh, zl, zh, N); }, [] { ival_acc_sin(al, ah, zl, zh, N); }, [] { ival_acc_atan(al, ah, zl, zh, N); } };
    for (int op = 0; op < NOPS; op++) r[op] = measure(f[op], op);
    const char *lib = std::getenv("IVAL_CRMVEC");
    report(lib && *lib ? "ival accurate (crmvec)" : "ival accurate (no crmvec)", r, all);
  }
#ifdef HAVE_MPFI
  {
    std::vector<mpfi_t> A(N), B(N);
    mpfi_t Z;
    mpfi_init2(Z, 53);
    for (int k = 0; k < N; k++) {
      mpfi_init2(A[k], 53); mpfi_init2(B[k], 53);
      mpfi_interv_d(A[k], al[k], ah[k]); mpfi_interv_d(B[k], bl[k], bh[k]);
    }
    auto out = [&](int k) {
      zl[k] = mpfr_get_d(&Z->left, MPFR_RNDD); zh[k] = mpfr_get_d(&Z->right, MPFR_RNDU);
    };
    std::function<void()> f[NOPS] = {
      [&] { for (int k = 0; k < N; k++) { mpfi_add(Z, A[k], B[k]); out(k); } },
      [&] { for (int k = 0; k < N; k++) { mpfi_mul(Z, A[k], B[k]); out(k); } },
      [&] { for (int k = 0; k < N; k++) { mpfi_div(Z, A[k], B[k]); out(k); } },
      [&] { for (int k = 0; k < N; k++) { mpfi_exp(Z, A[k]); out(k); } },
      [&] { for (int k = 0; k < N; k++) { mpfi_log(Z, B[k]); out(k); } },
      [&] { for (int k = 0; k < N; k++) { mpfi_sin(Z, A[k]); out(k); } },
      [&] { for (int k = 0; k < N; k++) { mpfi_atan(Z, A[k]); out(k); } } };
    for (int op = 0; op < NOPS; op++) r[op] = measure(f[op], op);
    report("MPFI (53 bits)", r, all);
  }
#endif
#ifdef HAVE_BOOST
  {
    using namespace boost::numeric;
    using namespace interval_lib;
    typedef interval<double, policies<save_state<rounded_transc_std<double>>, checking_base<double>>> I;
    std::vector<I> A(N), B(N);
    for (int k = 0; k < N; k++) { A[k] = I(al[k], ah[k]); B[k] = I(bl[k], bh[k]); }
    auto out = [&](int k, const I &z) { zl[k] = z.lower(); zh[k] = z.upper(); };
    std::function<void()> f[NOPS] = {
      [&] { for (int k = 0; k < N; k++) out(k, A[k] + B[k]); }, [&] { for (int k = 0; k < N; k++) out(k, A[k] * B[k]); },
      [&] { for (int k = 0; k < N; k++) out(k, A[k] / B[k]); }, [&] { for (int k = 0; k < N; k++) out(k, exp(A[k])); },
      [&] { for (int k = 0; k < N; k++) out(k, log(B[k])); }, [&] { for (int k = 0; k < N; k++) out(k, sin(A[k])); },
      [&] { for (int k = 0; k < N; k++) out(k, atan(A[k])); } };
    for (int op = 0; op < NOPS; op++) r[op] = measure(f[op], op);
    report("Boost.Interval", r, all);
  }
#endif
#ifdef HAVE_FILIB
  {
    typedef filib::interval<double, filib::native_switched, filib::i_mode_normal> I;
    filib::fp_traits<double, filib::native_switched>::setup();
    std::vector<I> A, B;
    for (int k = 0; k < N; k++) { A.push_back(I(al[k], ah[k])); B.push_back(I(bl[k], bh[k])); }
    auto out = [&](int k, const I &z) { zl[k] = z.inf(); zh[k] = z.sup(); };
    std::function<void()> f[NOPS] = {
      [&] { for (int k = 0; k < N; k++) out(k, A[k] + B[k]); }, [&] { for (int k = 0; k < N; k++) out(k, A[k] * B[k]); },
      [&] { for (int k = 0; k < N; k++) out(k, A[k] / B[k]); }, [&] { for (int k = 0; k < N; k++) out(k, filib::exp(A[k])); },
      [&] { for (int k = 0; k < N; k++) out(k, filib::log(B[k])); }, [&] { for (int k = 0; k < N; k++) out(k, filib::sin(A[k])); },
      [&] { for (int k = 0; k < N; k++) out(k, filib::atan(A[k])); } };
    for (int op = 0; op < NOPS; op++) r[op] = measure(f[op], op);
    report("filib++", r, all);
    std::fesetround(FE_TONEAREST);
  }
#endif
#ifdef HAVE_P1788
  {
    typedef p1788::infsup::interval<double, p1788::flavor::infsup::setbased::mpfr_bin_ieee754_flavor> I;
    std::vector<I> A, B;
    for (int k = 0; k < N; k++) { A.push_back(I(al[k], ah[k])); B.push_back(I(bl[k], bh[k])); }
    auto out = [&](int k, const I &z) { zl[k] = inf(z); zh[k] = sup(z); };
    std::function<void()> f[NOPS] = {
      [&] { for (int k = 0; k < N; k++) out(k, A[k] + B[k]); }, [&] { for (int k = 0; k < N; k++) out(k, A[k] * B[k]); },
      [&] { for (int k = 0; k < N; k++) out(k, A[k] / B[k]); }, [&] { for (int k = 0; k < N; k++) out(k, exp(A[k])); },
      [&] { for (int k = 0; k < N; k++) out(k, log(B[k])); }, [&] { for (int k = 0; k < N; k++) out(k, sin(A[k])); },
      [&] { for (int k = 0; k < N; k++) out(k, atan(A[k])); } };
    for (int op = 0; op < NOPS; op++) r[op] = measure(f[op], op);
    report("libieeep1788", r, all);
  }
#endif
  return 0;
}
