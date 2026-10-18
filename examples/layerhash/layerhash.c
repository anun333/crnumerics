/* layerhash: SmolLM2-135M (a Llama-shaped model) in plain C, one token at a
   time, with a SHA-256 of the residual stream after every layer, so two runs
   on two machines can be compared layer by layer and the first layer where
   they part is named.

   One file, three builds (Makefile):
     cr     every dot product correctly rounded (crsum); softmax, RMS norm and
            SiLU from crnn (correctly rounded, or a fixed sequence of
            correctly rounded steps on exact sums); RoPE's angles from
            CORE-MATH's double pow, sin and cos
     tree   as cr, but each dot product in a fixed order: 16 interleaved
            partial sums, combined pairwise
     stock  what plain C inference usually is: a running sum, the system's
            libm, and -O3 -ffast-math
   cr and tree are compiled with -ffp-contract=off, so the arithmetic is the
   arithmetic written, and nothing in them depends on the machine. stock's
   results may.

   usage: layerhash MODEL.safetensors PROMPTS.tsv NGEN OUTDIR
   MODEL is HuggingFaceTB/SmolLM2-135M's model.safetensors (bfloat16 weights,
   widened to binary32 exactly). PROMPTS.tsv: one prompt a line, its id, a
   tab, its token ids separated by spaces (prompts.py writes it). Each prompt
   is run, then NGEN tokens are generated greedily. OUTDIR gets layers.tsv
   (a hash per prompt, position and layer), tokens.tsv (the tokens chosen,
   with a hash of the logits each came from), logits-<id>.f32 (the logits
   after the prompt), resid-0.f32 (prompt 0's residual stream, for checking
   against a reference) and meta.txt. Every hash is of the values' bytes,
   little-endian on every machine this ran on. */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "sha256.h"
#ifndef MODE_STOCK
#include "crnn.h"
#include "crsum.h"
double cr_sin (double), cr_cos (double), cr_pow (double, double);
#endif

#if defined(MODE_CR)
#define MODE_NAME "cr"
#elif defined(MODE_TREE)
#define MODE_NAME "tree"
#elif defined(MODE_STOCK)
#define MODE_NAME "stock"
#else
#error "build with -DMODE_CR, -DMODE_TREE or -DMODE_STOCK"
#endif
#ifndef BUILD_FLAGS
#define BUILD_FLAGS "?"
#endif

enum { D = 576, NL = 30, NH = 9, NKV = 3, HD = 64, FF = 1536, V = 49152, MAXT = 256, MAXP = 64 };
#define ROPE_THETA 100000.0
#define RMS_EPS 1e-5f

typedef struct {
  float *in_norm, *wq, *wk, *wv, *wo, *post_norm, *wg, *wu, *wd;
} layer_t;
static float *emb, *final_norm;
static layer_t L[NL];
static float kc[NL][MAXT][NKV * HD], vc[NL][MAXT][NKV * HD];
static float rope_c[MAXT][HD / 2], rope_s[MAXT][HD / 2];

static void die (const char *m)
{
  fprintf (stderr, "layerhash: %s\n", m);
  exit (1);
}

/* ---------------------------------------------------------------- weights */

static char *hdr;
static const uint8_t *blob;
static size_t blob_n;

/* one bfloat16 tensor from the safetensors header, widened to binary32 */
static float *tensor (const char *name, size_t count)
{
  char key[192];
  snprintf (key, sizeof key, "\"%s\":", name);
  char *p = strstr (hdr, key);
  if (!p) {
    fprintf (stderr, "layerhash: no tensor %s\n", name);
    exit (1);
  }
  char *end = strchr (p, '}'), *dt = strstr (p, "\"dtype\":\""), *off = strstr (p, "\"data_offsets\":[");
  if (!end || !dt || !off || dt > end || off > end) die ("unexpected safetensors header");
  if (strncmp (dt + 9, "BF16\"", 5)) die ("expected BF16 tensors");
  unsigned long long a, b;
  if (sscanf (off + 16, "%llu,%llu", &a, &b) != 2 || b < a || b > blob_n || b - a != 2 * (unsigned long long) count) {
    fprintf (stderr, "layerhash: %s: wrong size\n", name);
    exit (1);
  }
  float *t = malloc (count * sizeof *t);
  if (!t) die ("out of memory");
  const uint8_t *s = blob + a;
  for (size_t i = 0; i < count; i++) {
    uint32_t u = (uint32_t) (s[2 * i] | s[2 * i + 1] << 8) << 16;
    memcpy (t + i, &u, 4);
  }
  return t;
}

static void load (const char *path, char hex[65])
{
  FILE *f = fopen (path, "rb");
  if (!f) die ("cannot open the model");
  if (fseek (f, 0, SEEK_END)) die ("cannot seek the model");
  long n = ftell (f);
  rewind (f);
  if (n < 8) die ("model too short");
  uint8_t *all = malloc ((size_t) n);
  if (!all || fread (all, 1, (size_t) n, f) != (size_t) n) die ("cannot read the model");
  fclose (f);
  sha256_hex (all, (size_t) n, hex);
  uint64_t hl = 0;
  for (int i = 7; i >= 0; i--) hl = hl << 8 | all[i];
  if (hl > (uint64_t) n - 8) die ("bad safetensors header length");
  hdr = malloc (hl + 1);
  memcpy (hdr, all + 8, hl);
  hdr[hl] = 0;
  blob = all + 8 + hl;
  blob_n = (size_t) n - 8 - hl;
  emb = tensor ("model.embed_tokens.weight", (size_t) V * D);
  final_norm = tensor ("model.norm.weight", D);
  for (int l = 0; l < NL; l++) {
    char nm[128];
#define T(field, suffix, count) \
  snprintf (nm, sizeof nm, "model.layers.%d.%s", l, suffix), L[l].field = tensor (nm, count)
    T (in_norm, "input_layernorm.weight", D);
    T (wq, "self_attn.q_proj.weight", (size_t) D * D);
    T (wk, "self_attn.k_proj.weight", (size_t) NKV * HD * D);
    T (wv, "self_attn.v_proj.weight", (size_t) NKV * HD * D);
    T (wo, "self_attn.o_proj.weight", (size_t) D * D);
    T (post_norm, "post_attention_layernorm.weight", D);
    T (wg, "mlp.gate_proj.weight", (size_t) FF * D);
    T (wu, "mlp.up_proj.weight", (size_t) FF * D);
    T (wd, "mlp.down_proj.weight", (size_t) D * FF);
#undef T
  }
  free (all);
  free (hdr);
}

/* --------------------------------------------------------------- numerics */

static float dotf (const float *a, const float *b, size_t n)
{
#if defined(MODE_CR)
  return crdotf (a, b, n, CRSUM_NEAREST);
#elif defined(MODE_TREE)
  float s[16] = {0};
  for (size_t i = 0; i < n; i++) s[i % 16] += a[i] * b[i];
  for (int w = 8; w; w /= 2)
    for (int j = 0; j < w; j++) s[j] += s[j + w];
  return s[0];
#else
  float s = 0;
  for (size_t i = 0; i < n; i++) s += a[i] * b[i];
  return s;
#endif
}

/* y = W x, W row-major with rows rows of cols */
static void matvec (float *y, const float *W, const float *x, size_t rows, size_t cols)
{
#if defined(MODE_CR)
  if (crgemvf (0, rows, cols, W, cols, x, 0.0f, y, CRSUM_NEAREST)) die ("crgemvf failed");
#else
  for (size_t i = 0; i < rows; i++) y[i] = dotf (W + i * cols, x, cols);
#endif
}

static void rmsnorm (float *y, const float *x, const float *g)
{
#ifdef MODE_STOCK
  float ss = 0;
  for (int i = 0; i < D; i++) ss += x[i] * x[i];
  float r = 1.0f / sqrtf (ss / D + RMS_EPS);
  for (int i = 0; i < D; i++) y[i] = x[i] * r * g[i];
#else
  crnn_rmsnormf (y, x, D, g, RMS_EPS);
#endif
}

static void softmax (float *p, const float *s, int n)
{
#ifdef MODE_STOCK
  float m = s[0], z = 0;
  for (int i = 1; i < n; i++) m = s[i] > m ? s[i] : m;
  for (int i = 0; i < n; i++) z += p[i] = expf (s[i] - m);
  for (int i = 0; i < n; i++) p[i] /= z;
#else
  crnn_softmaxf (p, s, (size_t) n);
#endif
}

static void silu (float *x, int n)
{
#ifdef MODE_STOCK
  for (int i = 0; i < n; i++) x[i] = x[i] / (1.0f + expf (-x[i]));
#else
  crnn_siluf (x, x, (size_t) n);
#endif
}

/* RoPE, the halves form (HF's rotate_half): pairs (j, j + 32) of each head */
static void rope_tables (void)
{
  for (int p = 0; p < MAXT; p++)
    for (int i = 0; i < HD / 2; i++) {
#ifdef MODE_STOCK
      float a = (float) p * powf ((float) ROPE_THETA, -(float) i / (HD / 2));
      rope_c[p][i] = cosf (a), rope_s[p][i] = sinf (a);
#else
      double a = (double) p * cr_pow (ROPE_THETA, -(double) i / (HD / 2));
      rope_c[p][i] = (float) cr_cos (a), rope_s[p][i] = (float) cr_sin (a);
#endif
    }
}

static void rope (float *v, int heads, int pos)
{
  for (int h = 0; h < heads; h++) {
    float *x = v + h * HD;
    for (int j = 0; j < HD / 2; j++) {
      float c = rope_c[pos][j], s = rope_s[pos][j], a = x[j], b = x[j + HD / 2];
#ifdef MODE_STOCK
      x[j] = a * c - b * s, x[j + HD / 2] = b * c + a * s;
#else /* products of binary32 values are exact in binary64: one rounding, then the narrowing */
      x[j] = (float) ((double) a * c - (double) b * s), x[j + HD / 2] = (float) ((double) b * c + (double) a * s);
#endif
    }
  }
}

/* ------------------------------------------------------------------ model */

static FILE *layers_out;
static sha256_ctx chain;

static void record (int pid, int pos, int layer, const float *x)
{
  uint8_t d[32];
  char hex[65];
  sha256_ctx c;
  sha256_init (&c);
  sha256_update (&c, x, D * sizeof *x);
  sha256_final (&c, d);
  sha256_update (&chain, d, 32);
  sha256_tohex (d, hex);
  fprintf (layers_out, "%d\t%d\t%d\t%s\n", pid, pos, layer, hex);
}

/* one token at position pos; x is the residual stream; logits if wanted.
   resid, if not NULL, gets the residual stream after the embedding and after
   each layer (NL + 1 rows of D). */
static void step (int pid, int tok, int pos, float *x, float *logits, float *resid)
{
  static float h[D], q[D], k[NKV * HD], v[NKV * HD], att[D], o[D], gt[FF], up[FF], sc[MAXT], pr[MAXT], col[MAXT];
  memcpy (x, emb + (size_t) tok * D, D * sizeof *x);
  record (pid, pos, 0, x);
  if (resid) memcpy (resid, x, D * sizeof *x);
  for (int l = 0; l < NL; l++) {
    const layer_t *w = &L[l];
    rmsnorm (h, x, w->in_norm);
    matvec (q, w->wq, h, D, D);
    matvec (k, w->wk, h, NKV * HD, D);
    matvec (v, w->wv, h, NKV * HD, D);
    rope (q, NH, pos);
    rope (k, NKV, pos);
    memcpy (kc[l][pos], k, sizeof k);
    memcpy (vc[l][pos], v, sizeof v);
    for (int hh = 0; hh < NH; hh++) {
      int g = hh / (NH / NKV);
      for (int s = 0; s <= pos; s++) sc[s] = dotf (q + hh * HD, kc[l][s] + g * HD, HD) * 0.125f;
      softmax (pr, sc, pos + 1);
      for (int d = 0; d < HD; d++) {
        for (int s = 0; s <= pos; s++) col[s] = vc[l][s][g * HD + d];
        att[hh * HD + d] = dotf (pr, col, (size_t) pos + 1);
      }
    }
    matvec (o, w->wo, att, D, D);
    for (int i = 0; i < D; i++) x[i] += o[i];
    rmsnorm (h, x, w->post_norm);
    matvec (gt, w->wg, h, FF, D);
    matvec (up, w->wu, h, FF, D);
    silu (gt, FF);
    for (int i = 0; i < FF; i++) gt[i] *= up[i];
    matvec (o, w->wd, gt, D, FF);
    for (int i = 0; i < D; i++) x[i] += o[i];
    record (pid, pos, l + 1, x);
    if (resid) memcpy (resid + (size_t) (l + 1) * D, x, D * sizeof *x);
  }
  if (logits) {
    rmsnorm (h, x, final_norm);
    matvec (logits, emb, h, V, D); /* tied embeddings */
  }
}

static int argmax (const float *v)
{
  int b = 0;
  for (int i = 1; i < V; i++)
    if (v[i] > v[b]) b = i;
  return b;
}

/* ------------------------------------------------------------------- main */

int main (int argc, char **argv)
{
  if (argc != 5) die ("usage: layerhash MODEL.safetensors PROMPTS.tsv NGEN OUTDIR");
  if (!sha256_selftest ()) die ("SHA-256 self-test failed");
  int ngen = atoi (argv[3]);
  const char *dir = argv[4];
  char path[1024], mhex[65], phex[65];
  load (argv[1], mhex);
  rope_tables ();

  FILE *pf = fopen (argv[2], "rb");
  if (!pf) die ("cannot open the prompts");
  static char ptext[1 << 16];
  size_t pn = fread (ptext, 1, sizeof ptext - 1, pf);
  fclose (pf);
  ptext[pn] = 0;
  sha256_hex (ptext, pn, phex);

  snprintf (path, sizeof path, "%s/layers.tsv", dir);
  if (!(layers_out = fopen (path, "w"))) die ("cannot write OUTDIR/layers.tsv");
  snprintf (path, sizeof path, "%s/tokens.tsv", dir);
  FILE *tok_out = fopen (path, "w");
  if (!tok_out) die ("cannot write OUTDIR/tokens.tsv");
  fprintf (layers_out, "prompt\tpos\tlayer\tsha256\n");
  fprintf (tok_out, "prompt\tstep\ttoken\tlogits_sha256\n");

  static float x[D], logits[V], resid[MAXT][(NL + 1) * D];
  int nprompts = 0, ntok_total = 0;
  double t0 = (double) clock () / CLOCKS_PER_SEC;
  sha256_init (&chain);
  for (char *line = strtok (ptext, "\n"); line; line = strtok (NULL, "\n")) {
    int pid, ids[MAXT], n = 0, used;
    if (sscanf (line, "%d%n", &pid, &used) != 1) continue;
    char *s = line + used;
    while (*s == '\t' || *s == ' ') s++;
    while (n < MAXT && sscanf (s, "%d%n", &ids[n], &used) == 1) n++, s += used;
    if (n == 0 || n + ngen > MAXT) die ("a prompt is empty or too long");
    for (int t = 0; t < n; t++)
      step (pid, ids[t], t, x, t == n - 1 ? logits : NULL, nprompts == 0 ? resid[t] : NULL);
    snprintf (path, sizeof path, "%s/logits-%d.f32", dir, pid);
    FILE *lf = fopen (path, "wb");
    if (!lf || fwrite (logits, sizeof logits, 1, lf) != 1) die ("cannot write logits");
    fclose (lf);
    if (nprompts == 0) {
      snprintf (path, sizeof path, "%s/resid-0.f32", dir);
      FILE *rf = fopen (path, "wb");
      if (!rf || fwrite (resid, sizeof resid[0], (size_t) n, rf) != (size_t) n) die ("cannot write resid");
      fclose (rf);
    }
    for (int g = 0; g < ngen; g++) {
      char lhex[65];
      sha256_hex (logits, sizeof logits, lhex);
      int next = argmax (logits);
      fprintf (tok_out, "%d\t%d\t%d\t%s\n", pid, g, next, lhex);
      if (g + 1 < ngen) step (pid, next, n + g, x, logits, NULL);
    }
    ntok_total += n + ngen - 1;
    nprompts++;
    fprintf (stderr, "prompt %d: %d tokens + %d generated, %.1f s so far\n", pid, n, ngen,
             (double) clock () / CLOCKS_PER_SEC - t0);
  }
  fclose (layers_out);
  fclose (tok_out);

  uint8_t d[32];
  char chex[65];
  sha256_final (&chain, d);
  sha256_tohex (d, chex);
  double secs = (double) clock () / CLOCKS_PER_SEC - t0;
  snprintf (path, sizeof path, "%s/meta.txt", dir);
  FILE *mf = fopen (path, "w");
  if (!mf) die ("cannot write meta.txt");
  fprintf (mf, "mode\t%s\nchain_sha256\t%s\nmodel_sha256\t%s\nprompts_sha256\t%s\nprompts\t%d\nngen\t%d\n"
               "token_steps\t%d\ncpu_seconds\t%.1f\ncompiler\t%s\nflags\t%s\n",
           MODE_NAME, chex, mhex, phex, nprompts, ngen, ntok_total, secs,
#ifdef __VERSION__
           __VERSION__,
#else
           "?",
#endif
           BUILD_FLAGS);
  fclose (mf);
  printf ("%s\t%s\n", MODE_NAME, chex);
  return 0;
}
