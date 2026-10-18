/* gpt2sae: GPT-2 small and its sparse autoencoders' features (layers 2, 6,
   10), as readout.py computes them, but with nothing left to the machine:
   every dot product in layerhash's fixed order (element i to partial sum
   i mod 16, each step one product and one sum, the sixteen combined
   pairwise; -DTREEFMA: each step one fmaf), LayerNorm and softmax from crnn,
   GELU (GPT-2's tanh form) from CORE-MATH's correctly rounded tanh, the
   residual stream's mean (for TransformerLens's centring) from crsum's
   correctly rounded sum. Compiled with -ffp-contract=off.

   usage: gpt2sae DIR OUTDIR [THREADS]
   DIR is fetch.sh's; tokens.txt is read from this file's directory (given
   by -DHERE). OUTDIR gets raw arrays that pack.py turns into readout.py's
   .npz format, so compare.py compares the two. Sequences are spread over
   THREADS threads; each is computed alone, so the thread count changes
   nothing but the time. */
#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "sha256.h"
#include "crnn.h"
#include "crsum.h"
double cr_tanh (double);

#ifdef TREEFMA
#define MODE_NAME "treefma"
#define LANE(acc, x, y) fmaf (x, y, acc)
#else
#define MODE_NAME "tree"
#define LANE(acc, x, y) ((acc) + (x) * (y))
#endif
#ifndef HERE
#define HERE "."
#endif
#ifndef BUILD_FLAGS
#define BUILD_FLAGS "?"
#endif

enum { D = 768, NL = 12, NH = 12, HD = 64, FF = 3072, NV = 50257, CTX = 128, NSEQ = 200, NTOK = NSEQ * CTX,
       DS = 24576, NSAE = 3 };
static const int SAE_LAYER[NSAE] = {2, 6, 10};

static void die (const char *m)
{
  fprintf (stderr, "gpt2sae: %s\n", m);
  exit (1);
}

/* ------------------------------------------------------------- numerics */

static float dotf (const float *a, const float *b, size_t n)
{
  float s[16] = {0};
  size_t i = 0;
  for (; i + 16 <= n; i += 16)
    for (int j = 0; j < 16; j++) s[j] = LANE (s[j], a[i + j], b[i + j]);
  for (size_t j = 0; j < n - i; j++) s[j] = LANE (s[j], a[i + j], b[i + j]);
  for (int w = 8; w; w /= 2)
    for (int j = 0; j < w; j++) s[j] += s[j + w];
  return s[0];
}

/* y = W x + b, W row-major (rows of cols) */
static void affine (float *y, const float *W, const float *b, const float *x, int rows, int cols)
{
  for (int r = 0; r < rows; r++) y[r] = dotf (W + (size_t) r * cols, x, (size_t) cols) + b[r];
}

static float gelu_new (float x)
{
  double xd = x, c = 0.7978845608028654; /* sqrt(2/pi), as transformers' NewGELU computes it in binary64 */
  double t = cr_tanh (c * (xd + 0.044715 * (xd * xd * xd)));
  return (float) (0.5 * xd * (1.0 + t));
}

/* ------------------------------------------------------------- weights */

static uint8_t *file_all (const char *path, size_t *n)
{
  FILE *f = fopen (path, "rb");
  if (!f) { fprintf (stderr, "gpt2sae: cannot open %s\n", path); exit (1); }
  fseek (f, 0, SEEK_END);
  long m = ftell (f);
  rewind (f);
  uint8_t *a = malloc ((size_t) m);
  if (!a || fread (a, 1, (size_t) m, f) != (size_t) m) die ("read failed");
  fclose (f);
  *n = (size_t) m;
  return a;
}

/* a binary32 tensor from a safetensors file in memory */
static const float *tensor (const uint8_t *all, size_t n, const char *name, size_t count)
{
  uint64_t hl = 0;
  for (int i = 7; i >= 0; i--) hl = hl << 8 | all[i];
  if (hl > n - 8) die ("bad safetensors header");
  char key[160];
  snprintf (key, sizeof key, "\"%s\":", name);
  const char *h = (const char *) all + 8, *p = NULL;
  for (size_t i = 0; i + strlen (key) <= hl; i++)
    if (!memcmp (h + i, key, strlen (key))) { p = h + i; break; }
  if (!p) { fprintf (stderr, "gpt2sae: no tensor %s\n", name); exit (1); }
  const char *end = memchr (p, '}', hl - (size_t) (p - h));
  const char *dt = strstr (p, "\"dtype\":\""), *off = strstr (p, "\"data_offsets\":[");
  if (!end || !dt || !off || dt > end || off > end || strncmp (dt + 9, "F32\"", 4)) die ("expected F32 tensors");
  unsigned long long a, b;
  if (sscanf (off + 16, "%llu,%llu", &a, &b) != 2 || b - a != 4 * (unsigned long long) count || 8 + hl + b > n)
    { fprintf (stderr, "gpt2sae: %s: wrong size\n", name); exit (1); }
  float *t = malloc (count * sizeof *t);
  if (!t) die ("out of memory");
  memcpy (t, all + 8 + hl + a, count * sizeof *t);
  return t;
}

/* a Conv1D weight [in][out], transposed to [out][in] */
static const float *conv (const uint8_t *all, size_t n, const char *name, int in, int out)
{
  const float *w = tensor (all, n, name, (size_t) in * out);
  float *t = malloc ((size_t) in * out * sizeof *t);
  if (!t) die ("out of memory");
  for (int i = 0; i < in; i++)
    for (int o = 0; o < out; o++) t[(size_t) o * in + i] = w[(size_t) i * out + o];
  free ((void *) w);
  return t;
}

typedef struct { const float *ln1g, *ln1b, *attnW, *attnb, *projW, *projb, *ln2g, *ln2b, *fcW, *fcb, *mprojW, *mprojb; } layer_t;
static layer_t L[NL];
static const float *wte, *wpe;
static const float *encT[NSAE], *benc[NSAE], *bdec[NSAE]; /* W_enc transposed: [DS][D] */
static int tokens[NTOK];

/* results, by token */
static float *resid[NSAE];           /* centred residual stream, NTOK x D */
static int32_t *cnt[NSAE], *top[NSAE];
static int32_t **idxs[NSAE];
static float **vals[NSAE];

/* ------------------------------------------------------------- forward */

typedef struct { float *kc, *vc; } cache_t;

static void sequence (int s, cache_t *c)
{
  float x[D], h[D], qkv[3 * D], att[D], o[D], fc[FF], sc[CTX], pr[CTX], col[CTX], xc[D], pre[DS];
  for (int t = 0; t < CTX; t++) {
    int tok = tokens[s * CTX + t], tk = s * CTX + t;
    for (int i = 0; i < D; i++) x[i] = wte[(size_t) tok * D + i] + wpe[(size_t) t * D + i];
    for (int l = 0; l < NL; l++) {
      for (int k = 0; k < NSAE; k++)
        if (SAE_LAYER[k] == l) {
          float mean = crsumf (x, D, CRSUM_NEAREST) / D;
          for (int i = 0; i < D; i++) xc[i] = x[i] - mean;
          memcpy (resid[k] + (size_t) tk * D, xc, sizeof xc);
          float in[D];
          for (int i = 0; i < D; i++) in[i] = xc[i] - bdec[k][i];
          int n = 0;
          for (int j = 0; j < DS; j++) {
            float v = dotf (encT[k] + (size_t) j * D, in, D) + benc[k][j];
            pre[j] = v > 0 ? v : 0;
            n += pre[j] > 0;
          }
          int32_t *ii = malloc ((n ? n : 1) * sizeof *ii);
          float *vv = malloc ((n ? n : 1) * sizeof *vv);
          int m = 0;
          for (int j = 0; j < DS; j++)
            if (pre[j] > 0) ii[m] = j, vv[m] = pre[j], m++;
          cnt[k][tk] = n, idxs[k][tk] = ii, vals[k][tk] = vv;
          /* top 10: by value, ties by index (as readout.py) */
          int32_t *tp = top[k] + (size_t) tk * 10;
          for (int q = 0; q < 10; q++) tp[q] = -1;
          for (int q = 0; q < 10 && q < m; q++) {
            int best = -1;
            for (int r = 0; r < m; r++) {
              int used = 0;
              for (int u = 0; u < q; u++) used |= tp[u] == ii[r];
              if (used) continue;
              if (best < 0 || vv[r] > vv[best] || (vv[r] == vv[best] && ii[r] < ii[best])) best = r;
            }
            tp[q] = ii[best];
          }
        }
      const layer_t *w = &L[l];
      float *kc = c->kc + (size_t) l * CTX * D, *vc = c->vc + (size_t) l * CTX * D;
      crnn_layernormf (h, x, D, w->ln1g, w->ln1b, 1e-5f);
      affine (qkv, w->attnW, w->attnb, h, 3 * D, D);
      memcpy (kc + (size_t) t * D, qkv + D, D * sizeof (float));
      memcpy (vc + (size_t) t * D, qkv + 2 * D, D * sizeof (float));
      for (int hh = 0; hh < NH; hh++) {
        for (int u = 0; u <= t; u++) sc[u] = dotf (qkv + hh * HD, kc + (size_t) u * D + hh * HD, HD) * 0.125f;
        crnn_softmaxf (pr, sc, (size_t) t + 1);
        for (int d = 0; d < HD; d++) {
          for (int u = 0; u <= t; u++) col[u] = vc[(size_t) u * D + hh * HD + d];
          att[hh * HD + d] = dotf (pr, col, (size_t) t + 1);
        }
      }
      affine (o, w->projW, w->projb, att, D, D);
      for (int i = 0; i < D; i++) x[i] += o[i];
      crnn_layernormf (h, x, D, w->ln2g, w->ln2b, 1e-5f);
      affine (fc, w->fcW, w->fcb, h, FF, D);
      for (int i = 0; i < FF; i++) fc[i] = gelu_new (fc[i]);
      affine (o, w->mprojW, w->mprojb, fc, D, FF);
      for (int i = 0; i < D; i++) x[i] += o[i];
    }
  }
}

static int nthreads;
static void *worker (void *arg)
{
  int id = (int) (intptr_t) arg;
  cache_t c = {malloc ((size_t) NL * CTX * D * sizeof (float)), malloc ((size_t) NL * CTX * D * sizeof (float))};
  if (!c.kc || !c.vc) die ("out of memory");
  for (int s = id; s < NSEQ; s += nthreads) sequence (s, &c);
  free (c.kc), free (c.vc);
  return NULL;
}

static void wr (const char *dir, const char *name, const void *p, size_t bytes)
{
  char path[1024];
  snprintf (path, sizeof path, "%s/%s", dir, name);
  FILE *f = fopen (path, "wb");
  if (!f || fwrite (p, 1, bytes, f) != bytes) die ("write failed");
  fclose (f);
}

int main (int argc, char **argv)
{
  if (argc < 3) die ("usage: gpt2sae DIR OUTDIR [THREADS]");
  if (!sha256_selftest ()) die ("SHA-256 self-test failed");
  nthreads = argc > 3 ? atoi (argv[3]) : 1;
  if (nthreads < 1) nthreads = 1;
  char path[1024];
  size_t n;
  snprintf (path, sizeof path, "%s/gpt2/model.safetensors", argv[1]);
  uint8_t *g = file_all (path, &n);
  wte = tensor (g, n, "wte.weight", (size_t) NV * D);
  wpe = tensor (g, n, "wpe.weight", (size_t) 1024 * D);
  for (int l = 0; l < NL; l++) {
    char nm[128];
#define NM(s) (snprintf (nm, sizeof nm, "h.%d.%s", l, s), nm)
    L[l].ln1g = tensor (g, n, NM ("ln_1.weight"), D);
    L[l].ln1b = tensor (g, n, NM ("ln_1.bias"), D);
    L[l].attnW = conv (g, n, NM ("attn.c_attn.weight"), D, 3 * D);
    L[l].attnb = tensor (g, n, NM ("attn.c_attn.bias"), 3 * D);
    L[l].projW = conv (g, n, NM ("attn.c_proj.weight"), D, D);
    L[l].projb = tensor (g, n, NM ("attn.c_proj.bias"), D);
    L[l].ln2g = tensor (g, n, NM ("ln_2.weight"), D);
    L[l].ln2b = tensor (g, n, NM ("ln_2.bias"), D);
    L[l].fcW = conv (g, n, NM ("mlp.c_fc.weight"), D, FF);
    L[l].fcb = tensor (g, n, NM ("mlp.c_fc.bias"), FF);
    L[l].mprojW = conv (g, n, NM ("mlp.c_proj.weight"), FF, D);
    L[l].mprojb = tensor (g, n, NM ("mlp.c_proj.bias"), D);
#undef NM
  }
  free (g);
  for (int k = 0; k < NSAE; k++) {
    snprintf (path, sizeof path, "%s/sae/l%d.safetensors", argv[1], SAE_LAYER[k]);
    uint8_t *a = file_all (path, &n);
    encT[k] = conv (a, n, "W_enc", D, DS);
    benc[k] = tensor (a, n, "b_enc", DS);
    bdec[k] = tensor (a, n, "b_dec", D);
    free (a);
    resid[k] = malloc ((size_t) NTOK * D * sizeof (float));
    cnt[k] = malloc (NTOK * sizeof (int32_t));
    top[k] = malloc ((size_t) NTOK * 10 * sizeof (int32_t));
    idxs[k] = malloc (NTOK * sizeof (int32_t *));
    vals[k] = malloc (NTOK * sizeof (float *));
    if (!resid[k] || !cnt[k] || !top[k] || !idxs[k] || !vals[k]) die ("out of memory");
  }
  FILE *tf = fopen (HERE "/tokens.txt", "r");
  if (!tf) die ("cannot open tokens.txt");
  for (int i = 0; i < NTOK; i++)
    if (fscanf (tf, "%d", &tokens[i]) != 1) die ("tokens.txt is short");
  fclose (tf);

  struct timespec t0, t1;
  clock_gettime (CLOCK_MONOTONIC, &t0);
  pthread_t th[64];
  if (nthreads > 64) nthreads = 64;
  for (int i = 0; i < nthreads; i++) pthread_create (&th[i], NULL, worker, (void *) (intptr_t) i);
  for (int i = 0; i < nthreads; i++) pthread_join (th[i], NULL);
  clock_gettime (CLOCK_MONOTONIC, &t1);
  double secs = (double) (t1.tv_sec - t0.tv_sec) + 1e-9 * (double) (t1.tv_nsec - t0.tv_nsec);

  const char *out = argv[2];
  FILE *mf;
  snprintf (path, sizeof path, "%s/meta.txt", out);
  if (!(mf = fopen (path, "w"))) die ("cannot write meta.txt");
  for (int k = 0; k < NSAE; k++) {
    int L_ = SAE_LAYER[k];
    char hex[65], nm[64];
    sha256_hex (resid[k], (size_t) NTOK * D * sizeof (float), hex);
    fprintf (mf, "rsha%d\t%s\n", L_, hex);
    size_t tot = 0;
    for (int t = 0; t < NTOK; t++) tot += (size_t) cnt[k][t];
    int32_t *ii = malloc ((tot ? tot : 1) * sizeof *ii);
    float *vv = malloc ((tot ? tot : 1) * sizeof *vv);
    size_t m = 0;
    for (int t = 0; t < NTOK; t++) {
      memcpy (ii + m, idxs[k][t], (size_t) cnt[k][t] * sizeof *ii);
      memcpy (vv + m, vals[k][t], (size_t) cnt[k][t] * sizeof *vv);
      m += (size_t) cnt[k][t];
    }
    snprintf (nm, sizeof nm, "cnt%d.i32", L_), wr (out, nm, cnt[k], NTOK * sizeof (int32_t));
    snprintf (nm, sizeof nm, "idx%d.i32", L_), wr (out, nm, ii, tot * sizeof *ii);
    snprintf (nm, sizeof nm, "val%d.f32", L_), wr (out, nm, vv, tot * sizeof *vv);
    snprintf (nm, sizeof nm, "top%d.i32", L_), wr (out, nm, top[k], (size_t) NTOK * 10 * sizeof (int32_t));
    if (L_ == 6) wr (out, "resid6.f32", resid[k], (size_t) 10 * CTX * D * sizeof (float));
  }
  fprintf (mf, "mode\t%s\nthreads\t%d\nseconds\t%.1f\ncompiler\t%s\nflags\t%s\n", MODE_NAME, nthreads, secs,
#ifdef __VERSION__
           __VERSION__,
#else
           "?",
#endif
           BUILD_FLAGS);
  fclose (mf);
  printf ("%s: %.1f s, %d threads\n", MODE_NAME, secs, nthreads);
  return 0;
}
