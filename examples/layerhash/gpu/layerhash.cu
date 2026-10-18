/* layerhash.cu: ../layerhash.c's tree contract (or
   treefma with -DTREEFMA), on an NVIDIA GPU. The same model, prompts and
   output files as layerhash.c, so its hashes can be compared with the CPU
   builds' directly.

   On the device:
   - every matrix-vector product: 16 threads per row, thread j summing the
     elements i = j mod 16 in increasing i, each step one __fmul_rn and one
     __fadd_rn (or one __fmaf_rn); then lanes j and j + w combined for w = 8,
     4, 2, 1: the CPU build's order exactly;
   - attention's dot products: the same order, one thread per head;
   - RMS norm, softmax: crnn's own code (crnn.c on crsum.c's exact sums),
     compiled as device code by devify-tree.py, one thread per vector; SiLU:
     crnn_siluf, one thread per element;
   - RoPE: the CPU build's tables (CORE-MATH's double pow, sin and cos, on the
     host), the rotation in binary64 with explicit __dmul_rn/__dsub_rn;
   - additions and products elementwise: __fadd_rn, __fmul_rn.
   Built with nvcc -fmad=false besides. The host only loads, hashes the
   residual stream it copies back after each layer, and takes the argmax.

   usage: layerhash-gpu-tree MODEL.safetensors PROMPTS.tsv NGEN OUTDIR
   (build.sh builds it, and layerhash-gpu-treefma). */
#include "cm64.cuh"
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <stdint.h>
#include <stddef.h>
#include <fenv.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <time.h>
/* crsum.c and crnn.c compiled as C++ device code: malloc and calloc return
   a pointer that converts to any pointer type, as C's void * does; crsum's
   thread-local bins become one block of 4096 per GPU thread; the x86 and Arm
   kernels are hidden (their architecture macros undefined); the
   floating-point environment calls do nothing, a GPU rounding to nearest as
   crnn sets the CPU to. */
struct crdev_ptr {
  void *p;
  __device__ crdev_ptr(void *q) : p(q) {}
  template <class T> __device__ operator T *() const { return (T *)p; }
};
__device__ crdev_ptr crdev_malloc(size_t n) { return crdev_ptr(::malloc(n)); }
__device__ crdev_ptr crdev_calloc(size_t n, size_t s) { void *p = ::malloc(n * s); if (p) memset(p, 0, n * s); return crdev_ptr(p); }
__device__ int crdev_strcmp(const char *a, const char *b) { while (*a && *a == *b) { a++; b++; } return (unsigned char)*a - (unsigned char)*b; }
__device__ uint64_t *crdev_tbin_base;
__device__ uint64_t *crdev_tbin(void) { return crdev_tbin_base + ((size_t)blockIdx.x * blockDim.x + threadIdx.x) * 4096; }
#define crsum_tbin (crdev_tbin())
namespace dnc {
__device__ double cr_exp(double x) { return d_exp::cr_exp_dev(x); }
__device__ double cr_log1p(double x) { return d_log1p::cr_log1p_dev(x); }
__device__ double cr_erfc(double x) { return d_erfc::cr_erfc_dev(x); }
__device__ double cr_rsqrt(double x) { return d_rsqrt::cr_rsqrt_dev(x); }
}
#pragma push_macro("__x86_64__")
#pragma push_macro("__aarch64__")
#undef __x86_64__
#undef __aarch64__
#define calloc crdev_calloc
#define malloc crdev_malloc
#define fegetenv(e) ((void)0)
#define fesetenv(e) ((void)0)
#define fprintf(f, ...) printf(__VA_ARGS__)
#define abort() __trap()
#define getenv(x) ((char *)0)
#define strcmp crdev_strcmp
namespace dnc {
#include "dev/crsum/crsum.c"
#include "dev/nn/crnn.c"
}
#undef calloc
#undef malloc
#undef fegetenv
#undef fesetenv
#undef fprintf
#undef abort
#undef getenv
#undef strcmp
#pragma pop_macro("__x86_64__")
#pragma pop_macro("__aarch64__")

#include "sha256.h"
extern "C" { /* host CORE-MATH, one declarator each (nvcc's host pass splits a list badly) */
double cr_sin(double);
double cr_cos(double);
double cr_pow(double, double);
}

enum { D = 576, NL = 30, NH = 9, NKV = 3, HD = 64, FF = 1536, V = 49152, MAXT = 256 };
#ifdef TREEFMA
#define MODE_NAME "treefma"
#define LANE(acc, x, y) __fmaf_rn(x, y, acc)
#else
#define MODE_NAME "tree"
#define LANE(acc, x, y) __fadd_rn(acc, __fmul_rn(x, y))
#endif

#define CK(e) do { cudaError_t err_ = (e); if (err_ != cudaSuccess) { fprintf(stderr, "CUDA: %s at %s:%d\n", cudaGetErrorString(err_), __FILE__, __LINE__); exit(1); } } while (0)

/* ------------------------------------------------------------- the device */

/* the CPU build's dotf, as one thread: b[i * bstride] */
__device__ float tree_dot_seq(const float *a, const float *b, int n, int bstride)
{
  float s[16];
  for (int j = 0; j < 16; j++) s[j] = 0.0f;
  int i = 0;
  for (; i + 16 <= n; i += 16)
    for (int j = 0; j < 16; j++) s[j] = LANE(s[j], a[i + j], b[(size_t)(i + j) * bstride]);
  for (int j = 0; j < n - i; j++) s[j] = LANE(s[j], a[i + j], b[(size_t)(i + j) * bstride]);
  for (int w = 8; w; w /= 2)
    for (int j = 0; j < w; j++) s[j] = __fadd_rn(s[j], s[j + w]);
  return s[0];
}

/* y = W x: block (16, 8), eight rows a block, sixteen lanes a row */
__global__ void k_matvec(float *y, const float *W, const float *x, int rows, int cols)
{
  __shared__ float s[8][16];
  int j = threadIdx.x, ry = threadIdx.y, r = blockIdx.x * 8 + ry;
  float acc = 0.0f;
  if (r < rows) {
    const float *w = W + (size_t)r * cols;
    for (int i = j; i < cols; i += 16) acc = LANE(acc, w[i], x[i]);
  }
  s[ry][j] = acc;
  __syncthreads();
  for (int ww = 8; ww; ww /= 2) {
    if (j < ww) s[ry][j] = __fadd_rn(s[ry][j], s[ry][j + ww]);
    __syncthreads();
  }
  if (j == 0 && r < rows) y[r] = s[ry][0];
}

__global__ void k_rmsnorm(float *y, const float *x, const float *g) { dnc::crnn_rmsnormf(y, x, D, g, 1e-5f); }

__global__ void k_silu_mul(float *g, const float *u, int n)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) { dnc::crnn_siluf(g + i, g + i, 1); g[i] = __fmul_rn(g[i], u[i]); }
}

__global__ void k_add(float *x, const float *o, int n)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) x[i] = __fadd_rn(x[i], o[i]);
}

/* RoPE on q (NH heads) and k (NKV heads), the halves form; c, s: this
   position's 32 cosines and sines */
__global__ void k_rope(float *q, float *k, const float *c, const float *s)
{
  int t = threadIdx.x, h = t / 32, j = t % 32;
  if (h >= NH + NKV) return;
  float *x = h < NH ? q + h * HD : k + (h - NH) * HD;
  double a = x[j], b = x[j + 32], cc = c[j], ss = s[j];
  x[j] = __double2float_rn(__dsub_rn(__dmul_rn(a, cc), __dmul_rn(b, ss)));
  x[j + 32] = __double2float_rn(__dadd_rn(__dmul_rn(b, cc), __dmul_rn(a, ss)));
}

/* attention for one position, one thread per head; kc, vc: this layer's
   cache, MAXT rows of NKV * HD; sc, pr: NH rows of MAXT scratch */
__global__ void k_attn(float *att, const float *q, const float *kc, const float *vc, int pos, float *sc, float *pr)
{
  int hh = threadIdx.x;
  if (hh >= NH) return;
  int g = hh / (NH / NKV);
  float *S = sc + hh * MAXT, *P = pr + hh * MAXT;
  for (int s = 0; s <= pos; s++) S[s] = __fmul_rn(tree_dot_seq(q + hh * HD, kc + (size_t)s * NKV * HD + g * HD, HD, 1), 0.125f);
  dnc::crnn_softmaxf(P, S, (size_t)pos + 1);
  for (int d = 0; d < HD; d++) att[hh * HD + d] = tree_dot_seq(P, vc + g * HD + d, pos + 1, NKV * HD);
}

/* --------------------------------------------------------------- the host */

static void die(const char *m) { fprintf(stderr, "layerhash-gpu: %s\n", m); exit(1); }
static char *hdr;
static const uint8_t *blob;
static size_t blob_n;

static float *tensor(const char *name, size_t count)
{
  char key[192];
  snprintf(key, sizeof key, "\"%s\":", name);
  char *p = strstr(hdr, key);
  if (!p) { fprintf(stderr, "no tensor %s\n", name); exit(1); }
  char *end = strchr(p, '}'), *dt = strstr(p, "\"dtype\":\""), *off = strstr(p, "\"data_offsets\":[");
  if (!end || !dt || !off || dt > end || off > end) die("unexpected safetensors header");
  if (strncmp(dt + 9, "BF16\"", 5)) die("expected BF16 tensors");
  unsigned long long a, b;
  if (sscanf(off + 16, "%llu,%llu", &a, &b) != 2 || b < a || b > blob_n || b - a != 2 * (unsigned long long)count) die("wrong tensor size");
  float *t = (float *)malloc(count * sizeof *t);
  if (!t) die("out of memory");
  const uint8_t *s = blob + a;
  for (size_t i = 0; i < count; i++) { uint32_t u = (uint32_t)(s[2 * i] | s[2 * i + 1] << 8) << 16; memcpy(t + i, &u, 4); }
  return t;
}

static float *todev(const float *h, size_t n)
{
  float *d;
  CK(cudaMalloc(&d, n * sizeof *d));
  CK(cudaMemcpy(d, h, n * sizeof *d, cudaMemcpyHostToDevice));
  return d;
}

struct layer { float *in_norm, *wq, *wk, *wv, *wo, *post_norm, *wg, *wu, *wd; };
static layer L[NL];
static float *emb_h, *emb_d, *final_norm_d, *rope_c_d, *rope_s_d;
static float *kc_d, *vc_d;            /* NL x MAXT x NKV*HD */
static float *x_d, *h_d, *q_d, *k_d, *v_d, *att_d, *o_d, *gt_d, *up_d, *sc_d, *pr_d, *logits_d;
static FILE *layers_out;
static sha256_ctx chain;

static void record(int pid, int pos, int layer, const float *x)
{
  uint8_t d[32]; char hex[65]; sha256_ctx c;
  sha256_init(&c); sha256_update(&c, x, D * sizeof *x); sha256_final(&c, d);
  sha256_update(&chain, d, 32); sha256_tohex(d, hex);
  fprintf(layers_out, "%d\t%d\t%d\t%s\n", pid, pos, layer, hex);
}

static void matvec(float *y, const float *W, const float *x, int rows, int cols)
{
  k_matvec<<<(rows + 7) / 8, dim3(16, 8)>>>(y, W, x, rows, cols);
}

static void step(int pid, int tok, int pos, float *logits, float *resid)
{
  static float xh[D];
  memcpy(xh, emb_h + (size_t)tok * D, sizeof xh);
  CK(cudaMemcpy(x_d, xh, sizeof xh, cudaMemcpyHostToDevice));
  record(pid, pos, 0, xh);
  if (resid) memcpy(resid, xh, sizeof xh);
  for (int l = 0; l < NL; l++) {
    const layer *w = &L[l];
    float *kc = kc_d + (size_t)l * MAXT * NKV * HD, *vc = vc_d + (size_t)l * MAXT * NKV * HD;
    k_rmsnorm<<<1, 1>>>(h_d, x_d, w->in_norm);
    matvec(q_d, w->wq, h_d, D, D);
    matvec(k_d, w->wk, h_d, NKV * HD, D);
    matvec(v_d, w->wv, h_d, NKV * HD, D);
    k_rope<<<1, (NH + NKV) * 32>>>(q_d, k_d, rope_c_d + pos * 32, rope_s_d + pos * 32);
    CK(cudaMemcpy(kc + (size_t)pos * NKV * HD, k_d, NKV * HD * sizeof(float), cudaMemcpyDeviceToDevice));
    CK(cudaMemcpy(vc + (size_t)pos * NKV * HD, v_d, NKV * HD * sizeof(float), cudaMemcpyDeviceToDevice));
    k_attn<<<1, 32>>>(att_d, q_d, kc, vc, pos, sc_d, pr_d);
    matvec(o_d, w->wo, att_d, D, D);
    k_add<<<(D + 127) / 128, 128>>>(x_d, o_d, D);
    k_rmsnorm<<<1, 1>>>(h_d, x_d, w->post_norm);
    matvec(gt_d, w->wg, h_d, FF, D);
    matvec(up_d, w->wu, h_d, FF, D);
    k_silu_mul<<<(FF + 127) / 128, 128>>>(gt_d, up_d, FF);
    matvec(o_d, w->wd, gt_d, D, FF);
    k_add<<<(D + 127) / 128, 128>>>(x_d, o_d, D);
    CK(cudaMemcpy(xh, x_d, sizeof xh, cudaMemcpyDeviceToHost));
    record(pid, pos, l + 1, xh);
    if (resid) memcpy(resid + (size_t)(l + 1) * D, xh, sizeof xh);
  }
  if (logits) {
    k_rmsnorm<<<1, 1>>>(h_d, x_d, final_norm_d);
    matvec(logits_d, emb_d, h_d, V, D);
    CK(cudaMemcpy(logits, logits_d, V * sizeof(float), cudaMemcpyDeviceToHost));
  }
  CK(cudaGetLastError());
}

int main(int argc, char **argv)
{
  if (argc != 5) die("usage: layerhash-gpu-tree MODEL.safetensors PROMPTS.tsv NGEN OUTDIR");
  if (!sha256_selftest()) die("SHA-256 self-test failed");
  int ngen = atoi(argv[3]);
  const char *dir = argv[4];
  char path[1024], mhex[65], phex[65];

  FILE *f = fopen(argv[1], "rb");
  if (!f) die("cannot open the model");
  fseek(f, 0, SEEK_END); long n = ftell(f); rewind(f);
  uint8_t *all = (uint8_t *)malloc((size_t)n);
  if (!all || fread(all, 1, (size_t)n, f) != (size_t)n) die("cannot read the model");
  fclose(f);
  sha256_hex(all, (size_t)n, mhex);
  uint64_t hl = 0;
  for (int i = 7; i >= 0; i--) hl = hl << 8 | all[i];
  hdr = (char *)malloc(hl + 1); memcpy(hdr, all + 8, hl); hdr[hl] = 0;
  blob = all + 8 + hl; blob_n = (size_t)n - 8 - hl;

  CK(cudaDeviceSetLimit(cudaLimitMallocHeapSize, (size_t)1 << 30));
  CK(cudaDeviceSetLimit(cudaLimitStackSize, 64 * 1024));
  emb_h = tensor("model.embed_tokens.weight", (size_t)V * D);
  emb_d = todev(emb_h, (size_t)V * D);
  float *t = tensor("model.norm.weight", D); final_norm_d = todev(t, D); free(t);
  for (int l = 0; l < NL; l++) {
    char nm[128];
#define T(field, suffix, count) snprintf(nm, sizeof nm, "model.layers.%d.%s", l, suffix), t = tensor(nm, count), L[l].field = todev(t, count), free(t)
    T(in_norm, "input_layernorm.weight", D);
    T(wq, "self_attn.q_proj.weight", (size_t)D * D);
    T(wk, "self_attn.k_proj.weight", (size_t)NKV * HD * D);
    T(wv, "self_attn.v_proj.weight", (size_t)NKV * HD * D);
    T(wo, "self_attn.o_proj.weight", (size_t)D * D);
    T(post_norm, "post_attention_layernorm.weight", D);
    T(wg, "mlp.gate_proj.weight", (size_t)FF * D);
    T(wu, "mlp.up_proj.weight", (size_t)FF * D);
    T(wd, "mlp.down_proj.weight", (size_t)D * FF);
#undef T
  }
  free(all); free(hdr);

  static float rc[MAXT][32], rs[MAXT][32];   /* as layerhash.c's rope_tables, cr builds */
  for (int p = 0; p < MAXT; p++)
    for (int i = 0; i < 32; i++) {
      double a = (double)p * cr_pow(100000.0, -(double)i / 32);
      rc[p][i] = (float)cr_cos(a), rs[p][i] = (float)cr_sin(a);
    }
  rope_c_d = todev(&rc[0][0], MAXT * 32); rope_s_d = todev(&rs[0][0], MAXT * 32);
  CK(cudaMalloc(&kc_d, (size_t)NL * MAXT * NKV * HD * sizeof(float)));
  CK(cudaMalloc(&vc_d, (size_t)NL * MAXT * NKV * HD * sizeof(float)));
  float **bufs[] = {&x_d, &h_d, &q_d, &att_d, &o_d};
  for (float **b : bufs) CK(cudaMalloc(b, D * sizeof(float)));
  CK(cudaMalloc(&k_d, NKV * HD * sizeof(float))); CK(cudaMalloc(&v_d, NKV * HD * sizeof(float)));
  CK(cudaMalloc(&gt_d, FF * sizeof(float))); CK(cudaMalloc(&up_d, FF * sizeof(float)));
  CK(cudaMalloc(&sc_d, NH * MAXT * sizeof(float))); CK(cudaMalloc(&pr_d, NH * MAXT * sizeof(float)));
  CK(cudaMalloc(&logits_d, V * sizeof(float)));
  uint64_t *tb; size_t tbn = (size_t)32 * 4096;
  CK(cudaMalloc(&tb, tbn * 8)); CK(cudaMemset(tb, 0, tbn * 8));
  CK(cudaMemcpyToSymbol(crdev_tbin_base, &tb, sizeof tb));

  FILE *pf = fopen(argv[2], "rb");
  if (!pf) die("cannot open the prompts");
  static char ptext[1 << 16];
  size_t pn = fread(ptext, 1, sizeof ptext - 1, pf); fclose(pf); ptext[pn] = 0;
  sha256_hex(ptext, pn, phex);
  snprintf(path, sizeof path, "%s/layers.tsv", dir);
  if (!(layers_out = fopen(path, "w"))) die("cannot write OUTDIR/layers.tsv");
  snprintf(path, sizeof path, "%s/tokens.tsv", dir);
  FILE *tok_out = fopen(path, "w");
  if (!tok_out) die("cannot write OUTDIR/tokens.tsv");
  fprintf(layers_out, "prompt\tpos\tlayer\tsha256\n");
  fprintf(tok_out, "prompt\tstep\ttoken\tlogits_sha256\n");

  static float logits[V], resid[MAXT][(NL + 1) * D];
  int nprompts = 0, ntok = 0;
  struct timespec t0, t1;
  clock_gettime(CLOCK_MONOTONIC, &t0);
  sha256_init(&chain);
  for (char *line = strtok(ptext, "\n"); line; line = strtok(NULL, "\n")) {
    int pid, ids[MAXT], m = 0, used;
    if (sscanf(line, "%d%n", &pid, &used) != 1) continue;
    char *s = line + used;
    while (*s == '\t' || *s == ' ') s++;
    while (m < MAXT && sscanf(s, "%d%n", &ids[m], &used) == 1) m++, s += used;
    if (m == 0 || m + ngen > MAXT) die("a prompt is empty or too long");
    for (int tpos = 0; tpos < m; tpos++) step(pid, ids[tpos], tpos, tpos == m - 1 ? logits : NULL, nprompts == 0 ? resid[tpos] : NULL);
    snprintf(path, sizeof path, "%s/logits-%d.f32", dir, pid);
    FILE *lf = fopen(path, "wb");
    if (!lf || fwrite(logits, sizeof logits, 1, lf) != 1) die("cannot write logits");
    fclose(lf);
    if (nprompts == 0) {
      snprintf(path, sizeof path, "%s/resid-0.f32", dir);
      FILE *rf = fopen(path, "wb");
      if (!rf || fwrite(resid, sizeof resid[0], (size_t)m, rf) != (size_t)m) die("cannot write resid");
      fclose(rf);
    }
    for (int g = 0; g < ngen; g++) {
      char lhex[65];
      sha256_hex(logits, sizeof logits, lhex);
      int b = 0;
      for (int i = 1; i < V; i++) if (logits[i] > logits[b]) b = i;
      fprintf(tok_out, "%d\t%d\t%d\t%s\n", pid, g, b, lhex);
      if (g + 1 < ngen) step(pid, b, m + g, logits, NULL);
    }
    ntok += m + ngen - 1;
    nprompts++;
    fprintf(stderr, "prompt %d: %d tokens + %d generated\n", pid, m, ngen);
  }
  clock_gettime(CLOCK_MONOTONIC, &t1);
  fclose(layers_out); fclose(tok_out);
  uint8_t d[32]; char chex[65];
  sha256_final(&chain, d); sha256_tohex(d, chex);
  cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
  snprintf(path, sizeof path, "%s/meta.txt", dir);
  FILE *mf = fopen(path, "w");
  if (!mf) die("cannot write meta.txt");
  fprintf(mf, "mode\t%s\nchain_sha256\t%s\nmodel_sha256\t%s\nprompts_sha256\t%s\nprompts\t%d\nngen\t%d\ntoken_steps\t%d\n"
              "wall_seconds\t%.1f\ndevice\t%s\ncompiler\tnvcc %d.%d\nflags\t%s\n",
          MODE_NAME, chex, mhex, phex, nprompts, ngen, ntok,
          (t1.tv_sec - t0.tv_sec) + 1e-9 * (t1.tv_nsec - t0.tv_nsec), prop.name, __CUDACC_VER_MAJOR__, __CUDACC_VER_MINOR__,
          BUILD_FLAGS);
  fclose(mf);
  printf("%s\t%s\n", MODE_NAME, chex);
  return 0;
}
