/* rerun: generate text with a language model, and a receipt that anyone can
   check by running it again, on any machine.

   The model runs on layerhash's fixed-order contract (../layerhash): every
   dot product in one order (element i to partial sum i mod 16, each step a
   product and a sum, the sixteen combined pairwise), no fusion the compiler
   chooses (-ffp-contract=off), RMS norm, softmax and SiLU from crnn, RoPE's
   angles from CORE-MATH's double pow, sin and cos, the weights widened to
   binary32 exactly. Sampling takes its uniforms from crdist's counter-based
   stream (Philox4x32-10) keyed by the seed, and works in a fixed order too.
   So a given model, prompt, seed and settings give the same tokens, and the
   same residual stream after every layer, on every machine; threads split
   a matrix's rows and change nothing but the time.

   The receipt (JSON) names the model files by sha256, the contract, the
   settings, the prompt's and the output's token ids, a SHA-256 of the
   residual stream after every layer at every position (the first 16 hex
   digits of each), and a chain hash over all of them. `rerun verify`
   computes it all again and says VERIFIED, or names the first position and
   layer that differ.

   Models: Llama-family (LlamaForCausalLM: SmolLM2 and the like, without
   rope_scaling) and OLMo 2 (Olmo2ForCausalLM), from a directory of
   config.json and model.safetensors or model.safetensors.index.json shards
   (bfloat16, float16 or float32). Token ids in, token ids out: rerun.py
   wraps the tokenizer.

   usage: rerun gen MODEL_DIR PROMPT_IDS MAX_NEW RECEIPT.json [options]
            PROMPT_IDS: a file of token ids (whitespace separated)
            --temp T (0: greedy, the default)  --top-p P (1)  --seed S (0)
            --threads N (1)  --layers-tsv FILE (every layer's full hash)
            --logits FILE (the logits after the prompt, binary32)
            --probe PROBE.json (its reading at every position, in the receipt)
            --steer PROBE.json ALPHA (add ALPHA times its unit direction after its layer)
          rerun verify RECEIPT.json [MODEL_DIR] [--threads N] [--probe FILE] [--steer FILE]
          rerun dump MODEL_DIR SEQS OUT --layers A,B [--from I] [--to J] [--threads N]
                     [--probe FILE] [--steer FILE ALPHA]   (see dump () below) */
#define _GNU_SOURCE
#include <fcntl.h>
#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#include "sha256.h"
#include "crnn.h"
#include "crsum.h"
#include "crdist.h"
double cr_sin (double), cr_cos (double), cr_pow (double, double), cr_log (double);

#define CONTRACT "tree16/v0"
#define LANE(acc, x, y) ((acc) + (x) * (y))
#ifndef BUILD_FLAGS
#define BUILD_FLAGS "?"
#endif

static void die (const char *m)
{
  fprintf (stderr, "rerun: %s\n", m);
  exit (1);
}

static void *xmalloc (size_t n)
{
  void *p = malloc (n ? n : 1);
  if (!p) die ("out of memory");
  return p;
}

/* ------------------------------------------------------------ numerics */

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

static int nthreads = 1;
typedef struct { float *y; const float *W, *x; size_t r0, r1, cols; } mv_job;
static void *mv_run (void *a)
{
  mv_job *j = a;
  for (size_t r = j->r0; r < j->r1; r++) j->y[r] = dotf (j->W + r * j->cols, j->x, j->cols);
  return NULL;
}

/* y = W x, W row-major; rows split over the threads, each row's sum the same */
static void matvec (float *y, const float *W, const float *x, size_t rows, size_t cols)
{
  int t = nthreads;
  if (t > 1 && rows * cols < ((size_t) 1 << 20)) t = 1;
  if (t <= 1) {
    mv_job j = {y, W, x, 0, rows, cols};
    mv_run (&j);
    return;
  }
  pthread_t th[64];
  mv_job jobs[64];
  if (t > 64) t = 64;
  for (int i = 0; i < t; i++) {
    jobs[i] = (mv_job) {y, W, x, rows * i / t, rows * (i + 1) / t, cols};
    pthread_create (&th[i], NULL, mv_run, &jobs[i]);
  }
  for (int i = 0; i < t; i++) pthread_join (th[i], NULL);
}

/* ------------------------------------------------------------ config */

static char *read_text (const char *path, size_t *n)
{
  FILE *f = fopen (path, "rb");
  if (!f) return NULL;
  fseek (f, 0, SEEK_END);
  long m = ftell (f);
  rewind (f);
  char *s = xmalloc ((size_t) m + 1);
  if (fread (s, 1, (size_t) m, f) != (size_t) m) die ("read failed");
  fclose (f);
  s[m] = 0;
  if (n) *n = (size_t) m;
  return s;
}

/* the value after "key": in a flat JSON object, or NULL */
static const char *jval (const char *js, const char *key)
{
  char k[128];
  snprintf (k, sizeof k, "\"%s\"", key);
  const char *p = js;
  while ((p = strstr (p, k))) {
    const char *q = p + strlen (k);
    while (*q == ' ' || *q == '\t' || *q == '\n' || *q == '\r') q++;
    if (*q == ':') {
      q++;
      while (*q == ' ' || *q == '\t' || *q == '\n' || *q == '\r') q++;
      return q;
    }
    p = q;
  }
  return NULL;
}
static double jnum (const char *js, const char *key, double dflt)
{
  const char *v = jval (js, key);
  return v && (*v == '-' || (*v >= '0' && *v <= '9')) ? strtod (v, NULL) : dflt;
}
static int jbool (const char *js, const char *key, int dflt)
{
  const char *v = jval (js, key);
  return v ? !strncmp (v, "true", 4) : dflt;
}
static void jstr (const char *js, const char *key, char *out, size_t n)
{
  const char *v = jval (js, key);
  out[0] = 0;
  if (!v || *v != '"') return;
  size_t i = 0;
  for (v++; *v && *v != '"' && i + 1 < n; v++) out[i++] = *v;
  out[i] = 0;
}

/* ------------------------------------------------------------ weights */

typedef struct { char name[512]; uint8_t *map; size_t size; const char *hdr; size_t hl; char sha[65]; } shard_t;
static shard_t shards[16];
static int nshards;

static void open_shard (const char *dir, const char *name)
{
  if (nshards == 16) die ("too many shards");
  shard_t *s = &shards[nshards++];
  snprintf (s->name, sizeof s->name, "%s", name);
  char path[1024];
  snprintf (path, sizeof path, "%s/%s", dir, name);
  int fd = open (path, O_RDONLY);
  struct stat st;
  if (fd < 0 || fstat (fd, &st)) { fprintf (stderr, "rerun: cannot open %s\n", path); exit (1); }
  s->size = (size_t) st.st_size;
  s->map = mmap (NULL, s->size, PROT_READ, MAP_PRIVATE, fd, 0);
  close (fd);
  if (s->map == MAP_FAILED) die ("mmap failed");
  sha256_hex (s->map, s->size, s->sha);
  uint64_t hl = 0;
  for (int i = 7; i >= 0; i--) hl = hl << 8 | s->map[i];
  if (hl > s->size - 8) die ("bad safetensors header");
  s->hdr = (const char *) s->map + 8, s->hl = (size_t) hl;
}

/* a tensor, widened to binary32, or NULL when absent (if optional) */
static float *tensor (const char *name, size_t count, int optional)
{
  char key[512];
  snprintf (key, sizeof key, "\"%s\":", name);
  size_t kl = strlen (key);
  for (int k = 0; k < nshards; k++) {
    shard_t *s = &shards[k];
    const char *p = NULL;
    for (size_t i = 0; i + kl <= s->hl; i++)
      if (s->hdr[i] == '"' && !memcmp (s->hdr + i, key, kl)) { p = s->hdr + i; break; }
    if (!p) continue;
    const char *end = memchr (p, '}', s->hl - (size_t) (p - s->hdr));
    const char *dt = strstr (p, "\"dtype\":\""), *off = strstr (p, "\"data_offsets\":[");
    if (!end || !dt || !off || dt > end || off > end) die ("unexpected safetensors header");
    int bytes = !strncmp (dt + 9, "F32\"", 4) ? 4 : (!strncmp (dt + 9, "BF16\"", 5) || !strncmp (dt + 9, "F16\"", 4)) ? 2 : 0;
    int bf = !strncmp (dt + 9, "BF16\"", 5);
    if (!bytes) die ("unsupported dtype (F32, BF16, F16)");
    unsigned long long a, b;
    if (sscanf (off + 16, "%llu,%llu", &a, &b) != 2 || b - a != (unsigned long long) bytes * count || 8 + s->hl + b > s->size) {
      fprintf (stderr, "rerun: %s: wrong size\n", name);
      exit (1);
    }
    const uint8_t *src = s->map + 8 + s->hl + a;
    float *t = xmalloc (count * sizeof *t);
    if (bytes == 4) memcpy (t, src, count * 4);
    else if (bf)
      for (size_t i = 0; i < count; i++) { uint32_t u = (uint32_t) (src[2 * i] | src[2 * i + 1] << 8) << 16; memcpy (t + i, &u, 4); }
    else
      for (size_t i = 0; i < count; i++) { _Float16 h; memcpy (&h, src + 2 * i, 2); t[i] = (float) h; }
    return t;
  }
  if (optional) return NULL;
  fprintf (stderr, "rerun: no tensor %s\n", name);
  exit (1);
}

/* ------------------------------------------------------------ the model */

enum { LLAMA, OLMO2 };
typedef struct { float *in_norm, *wq, *wk, *wv, *wo, *q_norm, *k_norm, *post_attn, *post_ff, *wg, *wu, *wd; } layer_t;
static struct {
  int arch, D, NL, NH, NKV, HD, FF, V, maxpos, eos;
  double theta, eps;
  int tied;
  char cfg_sha[65], arch_name[64];
  float *emb, *final_norm, *lm_head;
  layer_t *L;
} M;

static void load_model (const char *dir)
{
  char path[1024];
  snprintf (path, sizeof path, "%s/config.json", dir);
  size_t cn;
  char *cfg = read_text (path, &cn);
  if (!cfg) die ("no config.json");
  sha256_hex (cfg, cn, M.cfg_sha);
  char mt[64];
  jstr (cfg, "model_type", mt, sizeof mt);
  if (!strcmp (mt, "llama")) M.arch = LLAMA;
  else if (!strcmp (mt, "olmo2")) M.arch = OLMO2;
  else { fprintf (stderr, "rerun: model_type %s is not supported (llama, olmo2)\n", mt); exit (1); }
  snprintf (M.arch_name, sizeof M.arch_name, "%s", mt);
  const char *rs = jval (cfg, "rope_scaling");
  if (rs && strncmp (rs, "null", 4)) die ("rope_scaling is not supported");
  if (jbool (cfg, "attention_bias", 0) || jbool (cfg, "mlp_bias", 0)) die ("biases are not supported");
  M.D = (int) jnum (cfg, "hidden_size", 0), M.NL = (int) jnum (cfg, "num_hidden_layers", 0);
  M.NH = (int) jnum (cfg, "num_attention_heads", 0), M.NKV = (int) jnum (cfg, "num_key_value_heads", M.NH);
  M.HD = (int) jnum (cfg, "head_dim", M.NH ? M.D / M.NH : 0), M.FF = (int) jnum (cfg, "intermediate_size", 0);
  M.V = (int) jnum (cfg, "vocab_size", 0), M.maxpos = (int) jnum (cfg, "max_position_embeddings", 2048);
  M.theta = jnum (cfg, "rope_theta", 10000), M.eps = jnum (cfg, "rms_norm_eps", 1e-6);
  M.eos = (int) jnum (cfg, "eos_token_id", -1), M.tied = jbool (cfg, "tie_word_embeddings", 0);
  if (!M.D || !M.NL || !M.NH || !M.HD || !M.FF || !M.V || M.HD % 2 || M.NH % M.NKV) die ("incomplete config");
  free (cfg);

  snprintf (path, sizeof path, "%s/model.safetensors.index.json", dir);
  char *idx = read_text (path, NULL);
  if (idx) {
    /* every distinct shard named in the weight map, in order of first appearance */
    for (const char *p = strstr (idx, "\"weight_map\""); p && (p = strstr (p, ".safetensors\""));) {
      const char *q = p;
      while (q > idx && *(q - 1) != '"') q--;
      char nm[512];
      size_t l = (size_t) (p + 12 - q);
      if (l >= sizeof nm) die ("bad index");
      memcpy (nm, q, l), nm[l] = 0;
      int seen = 0;
      for (int k = 0; k < nshards; k++) seen |= !strcmp (shards[k].name, nm);
      if (!seen) open_shard (dir, nm);
      p += 13;
    }
    free (idx);
  } else
    open_shard (dir, "model.safetensors");

  size_t D = (size_t) M.D, Q = (size_t) M.NH * M.HD, KV = (size_t) M.NKV * M.HD, FF = (size_t) M.FF;
  M.emb = tensor ("model.embed_tokens.weight", (size_t) M.V * D, 0);
  M.final_norm = tensor ("model.norm.weight", D, 0);
  M.lm_head = M.tied ? NULL : tensor ("lm_head.weight", (size_t) M.V * D, 1);
  if (!M.lm_head) M.lm_head = M.emb;
  M.L = xmalloc ((size_t) M.NL * sizeof *M.L);
  for (int l = 0; l < M.NL; l++) {
    layer_t *w = &M.L[l];
    char nm[256];
#define T(field, suffix, count, opt) snprintf (nm, sizeof nm, "model.layers.%d.%s", l, suffix), w->field = tensor (nm, count, opt)
    T (wq, "self_attn.q_proj.weight", Q * D, 0);
    T (wk, "self_attn.k_proj.weight", KV * D, 0);
    T (wv, "self_attn.v_proj.weight", KV * D, 0);
    T (wo, "self_attn.o_proj.weight", D * Q, 0);
    T (wg, "mlp.gate_proj.weight", FF * D, 0);
    T (wu, "mlp.up_proj.weight", FF * D, 0);
    T (wd, "mlp.down_proj.weight", D * FF, 0);
    T (post_attn, "post_attention_layernorm.weight", D, 0);
    if (M.arch == LLAMA) {
      T (in_norm, "input_layernorm.weight", D, 0);
    } else {
      T (q_norm, "self_attn.q_norm.weight", Q, 0);
      T (k_norm, "self_attn.k_norm.weight", KV, 0);
      T (post_ff, "post_feedforward_layernorm.weight", D, 0);
    }
#undef T
  }
}

/* ------------------------------------------------------------ the forward pass */

/* probes and steering (below) */
typedef struct { int on, layer; float bias, *w; char name[128], sha[65]; } probe_t;
static probe_t probe, steerdir;
static double steer_alpha;
static float *steer_add, *probe_vals, *dump_x;
static int ndump, dump_layer[8];

static float *kc, *vc, *rope_c, *rope_s; /* caches: NL x maxctx x KV; tables: maxctx x HD/2 */
static int maxctx;
static uint8_t *hashes;   /* positions x (NL + 1) x 32 */
static int npos;

static void setup (int ctx)
{
  maxctx = ctx;
  size_t KV = (size_t) M.NKV * M.HD;
  kc = xmalloc ((size_t) M.NL * ctx * KV * sizeof (float));
  vc = xmalloc ((size_t) M.NL * ctx * KV * sizeof (float));
  int h2 = M.HD / 2;
  rope_c = xmalloc ((size_t) ctx * h2 * sizeof (float));
  rope_s = xmalloc ((size_t) ctx * h2 * sizeof (float));
  for (int p = 0; p < ctx; p++)
    for (int i = 0; i < h2; i++) {
      double a = (double) p * cr_pow (M.theta, -(double) i / h2);
      rope_c[(size_t) p * h2 + i] = (float) cr_cos (a), rope_s[(size_t) p * h2 + i] = (float) cr_sin (a);
    }
  hashes = xmalloc ((size_t) ctx * (M.NL + 1) * 32);
  if (probe.on) probe_vals = xmalloc ((size_t) ctx * 4);
}

static void rope (float *v, int heads, int pos)
{
  int h2 = M.HD / 2;
  const float *c = rope_c + (size_t) pos * h2, *s = rope_s + (size_t) pos * h2;
  for (int h = 0; h < heads; h++) {
    float *x = v + (size_t) h * M.HD;
    for (int j = 0; j < h2; j++) {
      float a = x[j], b = x[j + h2];
      x[j] = (float) ((double) a * c[j] - (double) b * s[j]);
      x[j + h2] = (float) ((double) b * c[j] + (double) a * s[j]);
    }
  }
}

static void record (int pos, int layer, const float *x)
{
  sha256_ctx c;
  sha256_init (&c);
  sha256_update (&c, x, (size_t) M.D * sizeof *x);
  sha256_final (&c, hashes + ((size_t) pos * (M.NL + 1) + layer) * 32);
}

/* ------------------------------------------------------------ probes and steering */

/* a linear probe on the residual stream after layer L (1..NL): v = w . x +
   bias, the dot product in the contract's order. JSON: {"name": ...,
   "layer": L, "bias": b, "w": [D numbers]}; its sha256 goes in the receipt.
   Steering adds alpha times the unit vector along a probe's w to the
   residual stream after its layer, at every position (the norm summed in
   binary64 in index order); the probe reads the stream before that. */

static void load_probe (probe_t *p, const char *path)
{
  size_t n;
  char *s = read_text (path, &n);
  if (!s) die ("cannot read the probe");
  sha256_hex (s, n, p->sha);
  jstr (s, "name", p->name, sizeof p->name);
  p->layer = (int) jnum (s, "layer", -1);
  p->bias = (float) jnum (s, "bias", 0);
  if (p->layer < 1 || p->layer > M.NL) die ("probe: layer must be 1 to the number of layers");
  const char *w = jval (s, "w");
  if (!w || *w != '[') die ("probe: no w");
  p->w = xmalloc ((size_t) M.D * 4);
  int i = 0;
  for (w++; *w && *w != ']' && i < M.D;) {
    char *e;
    double v = strtod (w, &e);
    if (e == w) { w++; continue; }
    p->w[i++] = (float) v, w = e;
  }
  if (i != M.D) die ("probe: w has the wrong length");
  p->on = 1;
  free (s);
}

static void setup_steer (double alpha)
{
  double ss = 0;
  for (int i = 0; i < M.D; i++) ss += (double) steerdir.w[i] * steerdir.w[i];
  double nrm = sqrt (ss);
  if (!(nrm > 0)) die ("steer: zero vector");
  steer_alpha = alpha;
  steer_add = xmalloc ((size_t) M.D * 4);
  for (int i = 0; i < M.D; i++) steer_add[i] = (float) (alpha * ((double) steerdir.w[i] / nrm));
}

/* the entropy of softmax(logits), in nats: p from crnn, -sum p log p in
   binary64 in index order, log from CORE-MATH */
static double entropy (const float *logits, float *p)
{
  crnn_softmaxf (p, logits, (size_t) M.V);
  double h = 0;
  for (int i = 0; i < M.V; i++)
    if (p[i] > 0) h -= (double) p[i] * cr_log ((double) p[i]);
  return h;
}

/* one token at position pos; logits if wanted */
static void step (int tok, int pos, float *logits)
{
  size_t D = (size_t) M.D, Q = (size_t) M.NH * M.HD, KV = (size_t) M.NKV * M.HD, FF = (size_t) M.FF;
  static float *x, *h, *q, *k, *v, *att, *o, *gt, *up, *sc, *pr, *col;
  if (!x) {
    x = xmalloc (D * 4), h = xmalloc (D * 4), q = xmalloc (Q * 4), k = xmalloc (KV * 4), v = xmalloc (KV * 4);
    att = xmalloc (Q * 4), o = xmalloc (D * 4), gt = xmalloc (FF * 4), up = xmalloc (FF * 4);
    sc = xmalloc ((size_t) maxctx * 4), pr = xmalloc ((size_t) maxctx * 4), col = xmalloc ((size_t) maxctx * 4);
  }
  if (tok < 0 || tok >= M.V) die ("token id out of range");
  if (pos >= maxctx) die ("context full");
  memcpy (x, M.emb + (size_t) tok * D, D * 4);
  record (pos, 0, x);
  float scale = (float) (1.0 / sqrt ((double) M.HD)); /* exact for a head size that is a power of 4 */
  for (int l = 0; l < M.NL; l++) {
    const layer_t *w = &M.L[l];
    float *K = kc + (size_t) l * maxctx * KV, *Vv = vc + (size_t) l * maxctx * KV;
    const float *in = x;
    if (M.arch == LLAMA) crnn_rmsnormf (h, x, D, w->in_norm, (float) M.eps), in = h;
    matvec (q, w->wq, in, Q, D);
    matvec (k, w->wk, in, KV, D);
    matvec (v, w->wv, in, KV, D);
    if (M.arch == OLMO2) {
      crnn_rmsnormf (q, q, Q, w->q_norm, (float) M.eps);
      crnn_rmsnormf (k, k, KV, w->k_norm, (float) M.eps);
    }
    rope (q, M.NH, pos);
    rope (k, M.NKV, pos);
    memcpy (K + (size_t) pos * KV, k, KV * 4);
    memcpy (Vv + (size_t) pos * KV, v, KV * 4);
    int rep = M.NH / M.NKV;
    for (int hh = 0; hh < M.NH; hh++) {
      int g = hh / rep;
      for (int s = 0; s <= pos; s++) sc[s] = dotf (q + (size_t) hh * M.HD, K + (size_t) s * KV + (size_t) g * M.HD, (size_t) M.HD) * scale;
      crnn_softmaxf (pr, sc, (size_t) pos + 1);
      for (int d = 0; d < M.HD; d++) {
        for (int s = 0; s <= pos; s++) col[s] = Vv[(size_t) s * KV + (size_t) g * M.HD + d];
        att[(size_t) hh * M.HD + d] = dotf (pr, col, (size_t) pos + 1);
      }
    }
    matvec (o, w->wo, att, D, Q);
    if (M.arch == OLMO2) crnn_rmsnormf (o, o, D, w->post_attn, (float) M.eps);
    for (size_t i = 0; i < D; i++) x[i] += o[i];
    in = x;
    if (M.arch == LLAMA) crnn_rmsnormf (h, x, D, w->post_attn, (float) M.eps), in = h;
    matvec (gt, w->wg, in, FF, D);
    matvec (up, w->wu, in, FF, D);
    crnn_siluf (gt, gt, FF);
    for (size_t i = 0; i < FF; i++) gt[i] *= up[i];
    matvec (o, w->wd, gt, D, FF);
    if (M.arch == OLMO2) crnn_rmsnormf (o, o, D, w->post_ff, (float) M.eps);
    for (size_t i = 0; i < D; i++) x[i] += o[i];
    record (pos, l + 1, x);
    if (probe.on && probe.layer == l + 1) probe_vals[pos] = dotf (probe.w, x, D) + probe.bias;
    for (int k = 0; k < ndump; k++)
      if (dump_layer[k] == l + 1) memcpy (dump_x + (size_t) k * D, x, D * 4);
    if (steer_add && steerdir.layer == l + 1)
      for (size_t i = 0; i < D; i++) x[i] += steer_add[i];
  }
  if (logits) {
    crnn_rmsnormf (h, x, D, M.final_norm, (float) M.eps);
    matvec (logits, M.lm_head, h, (size_t) M.V, D);
  }
}

/* ------------------------------------------------------------ blocks of tokens */

/* fn (i, ctx) for i in [0, n), in contiguous chunks over the threads */
typedef struct { void (*fn) (int, void *); void *ctx; int a, b; } pf_job;
static void *pf_run (void *p)
{
  pf_job *j = p;
  for (int i = j->a; i < j->b; i++) j->fn (i, j->ctx);
  return NULL;
}
static void parfor (int n, void (*fn) (int, void *), void *ctx)
{
  int t = nthreads < n ? nthreads : n;
  if (t > 64) t = 64;
  if (t <= 1) {
    for (int i = 0; i < n; i++) fn (i, ctx);
    return;
  }
  pthread_t th[64];
  pf_job jobs[64];
  for (int i = 0; i < t; i++) {
    jobs[i] = (pf_job) {fn, ctx, (int) ((long) n * i / t), (int) ((long) n * (i + 1) / t)};
    pthread_create (&th[i], NULL, pf_run, &jobs[i]);
  }
  for (int i = 0; i < t; i++) pthread_join (th[i], NULL);
}

/* Y (N x rows) = X (N x cols) times W (rows x cols) transposed: every
   element the contract's dot product of a row of W and a row of X, so the
   same bits as matvec one token at a time; the rows in chunks over the
   threads, each row of W used for all N tokens while it is in cache */
typedef struct { float *Y; const float *W, *X; int N, chunks; size_t rows, cols; } mm_ctx;
static void mm_chunk (int k, void *p)
{
  mm_ctx *c = p;
  size_t r0 = c->rows * (size_t) k / (size_t) c->chunks, r1 = c->rows * (size_t) (k + 1) / (size_t) c->chunks;
  for (size_t r = r0; r < r1; r++) {
    const float *w = c->W + r * c->cols;
    for (int n = 0; n < c->N; n++) c->Y[(size_t) n * c->rows + r] = dotf (w, c->X + (size_t) n * c->cols, c->cols);
  }
}
static void matmul (float *Y, const float *W, const float *X, int N, size_t rows, size_t cols)
{
  mm_ctx c = {Y, W, X, N, rows < 512 ? (int) rows : 512, rows, cols};
  parfor (c.chunks, mm_chunk, &c);
}

/* attention for a block, one head per call: as step () does it, position by position */
typedef struct { const float *Q, *K, *V; float *ATT, *scratch; int T, pos0; float scale; } at_ctx;
static void at_head (int hh, void *p)
{
  at_ctx *c = p;
  size_t Qd = (size_t) M.NH * M.HD, KV = (size_t) M.NKV * M.HD;
  int g = hh / (M.NH / M.NKV);
  float *sc = c->scratch + (size_t) hh * 3 * maxctx, *pr = sc + maxctx, *col = pr + maxctx;
  for (int t = 0; t < c->T; t++) {
    int pos = c->pos0 + t;
    const float *q = c->Q + (size_t) t * Qd + (size_t) hh * M.HD;
    for (int s = 0; s <= pos; s++) sc[s] = dotf (q, c->K + (size_t) s * KV + (size_t) g * M.HD, (size_t) M.HD) * c->scale;
    crnn_softmaxf (pr, sc, (size_t) pos + 1);
    for (int d = 0; d < M.HD; d++) {
      for (int s = 0; s <= pos; s++) col[s] = c->V[(size_t) s * KV + (size_t) g * M.HD + d];
      c->ATT[(size_t) t * Qd + (size_t) hh * M.HD + d] = dotf (pr, col, (size_t) pos + 1);
    }
  }
}

static float *dump_xb; /* the block's dump: T x ndump x D */

/* T tokens at positions pos0 .. pos0 + T - 1, all at once: the same bits, in
   every hash, probe value, dump and logit, as step () on each in turn.
   logits: T x V (all), or V (the last position only, when last_only) */
static void forward_block (const int *toks, int T, int pos0, float *logits, int last_only)
{
  size_t D = (size_t) M.D, Q = (size_t) M.NH * M.HD, KV = (size_t) M.NKV * M.HD, FF = (size_t) M.FF;
  static float *X, *H, *QB, *KB, *VB, *ATT, *O, *GT, *UP, *scratch;
  static int cap;
  if (T > cap) {
    free (X), free (H), free (QB), free (KB), free (VB), free (ATT), free (O), free (GT), free (UP), free (scratch);
    cap = T;
    X = xmalloc ((size_t) T * D * 4), H = xmalloc ((size_t) T * D * 4), QB = xmalloc ((size_t) T * Q * 4);
    KB = xmalloc ((size_t) T * KV * 4), VB = xmalloc ((size_t) T * KV * 4), ATT = xmalloc ((size_t) T * Q * 4);
    O = xmalloc ((size_t) T * D * 4), GT = xmalloc ((size_t) T * FF * 4), UP = xmalloc ((size_t) T * FF * 4);
    scratch = xmalloc ((size_t) M.NH * 3 * maxctx * 4);
  }
  if (pos0 + T > maxctx) die ("context full");
  for (int t = 0; t < T; t++) {
    if (toks[t] < 0 || toks[t] >= M.V) die ("token id out of range");
    memcpy (X + t * D, M.emb + (size_t) toks[t] * D, D * 4);
    record (pos0 + t, 0, X + t * D);
  }
  float scale = (float) (1.0 / sqrt ((double) M.HD));
  for (int l = 0; l < M.NL; l++) {
    const layer_t *w = &M.L[l];
    float *K = kc + (size_t) l * maxctx * KV, *Vv = vc + (size_t) l * maxctx * KV;
    const float *in = X;
    if (M.arch == LLAMA) {
      for (int t = 0; t < T; t++) crnn_rmsnormf (H + t * D, X + t * D, D, w->in_norm, (float) M.eps);
      in = H;
    }
    matmul (QB, w->wq, in, T, Q, D);
    matmul (KB, w->wk, in, T, KV, D);
    matmul (VB, w->wv, in, T, KV, D);
    for (int t = 0; t < T; t++) {
      if (M.arch == OLMO2) {
        crnn_rmsnormf (QB + t * Q, QB + t * Q, Q, w->q_norm, (float) M.eps);
        crnn_rmsnormf (KB + t * KV, KB + t * KV, KV, w->k_norm, (float) M.eps);
      }
      rope (QB + t * Q, M.NH, pos0 + t);
      rope (KB + t * KV, M.NKV, pos0 + t);
      memcpy (K + (size_t) (pos0 + t) * KV, KB + t * KV, KV * 4);
      memcpy (Vv + (size_t) (pos0 + t) * KV, VB + t * KV, KV * 4);
    }
    at_ctx ac = {QB, K, Vv, ATT, scratch, T, pos0, scale};
    parfor (M.NH, at_head, &ac);
    matmul (O, w->wo, ATT, T, D, Q);
    for (int t = 0; t < T; t++) {
      if (M.arch == OLMO2) crnn_rmsnormf (O + t * D, O + t * D, D, w->post_attn, (float) M.eps);
      for (size_t i = 0; i < D; i++) X[t * D + i] += O[t * D + i];
    }
    in = X;
    if (M.arch == LLAMA) {
      for (int t = 0; t < T; t++) crnn_rmsnormf (H + t * D, X + t * D, D, w->post_attn, (float) M.eps);
      in = H;
    }
    matmul (GT, w->wg, in, T, FF, D);
    matmul (UP, w->wu, in, T, FF, D);
    for (int t = 0; t < T; t++) {
      crnn_siluf (GT + t * FF, GT + t * FF, FF);
      for (size_t i = 0; i < FF; i++) GT[t * FF + i] *= UP[t * FF + i];
    }
    matmul (O, w->wd, GT, T, D, FF);
    for (int t = 0; t < T; t++) {
      float *x = X + t * D;
      if (M.arch == OLMO2) crnn_rmsnormf (O + t * D, O + t * D, D, w->post_ff, (float) M.eps);
      for (size_t i = 0; i < D; i++) x[i] += O[t * D + i];
      record (pos0 + t, l + 1, x);
      if (probe.on && probe.layer == l + 1) probe_vals[pos0 + t] = dotf (probe.w, x, D) + probe.bias;
      for (int k = 0; k < ndump; k++)
        if (dump_layer[k] == l + 1) memcpy (dump_xb + ((size_t) t * ndump + k) * D, x, D * 4);
      if (steer_add && steerdir.layer == l + 1)
        for (size_t i = 0; i < D; i++) x[i] += steer_add[i];
    }
  }
  if (logits) {
    int t0 = last_only ? T - 1 : 0;
    for (int t = t0; t < T; t++) crnn_rmsnormf (H + t * D, X + t * D, D, M.final_norm, (float) M.eps);
    matmul (logits, M.lm_head, H + (size_t) t0 * D, T - t0, (size_t) M.V, D);
  }
}

/* ------------------------------------------------------------ sampling */

static const float *cmp_p;
static int cmp_desc (const void *a, const void *b)
{
  int i = *(const int *) a, j = *(const int *) b;
  if (cmp_p[i] != cmp_p[j]) return cmp_p[i] > cmp_p[j] ? -1 : 1;
  return i < j ? -1 : i > j;
}

/* the next token: greedy (temp 0: the largest logit, ties to the lower id),
   or from softmax(logits / temp) restricted to the smallest set of most
   probable tokens (ties to the lower id) whose probability reaches top_p,
   with one binary64 uniform from the stream; every sum in a fixed order */
static int sample (const float *logits, double temp, double top_p, crdist_stream *st)
{
  int V = M.V;
  if (temp <= 0) {
    int b = 0;
    for (int i = 1; i < V; i++)
      if (logits[i] > logits[b]) b = i;
    return b;
  }
  static float *z, *p;
  static int *ord;
  if (!z) z = xmalloc ((size_t) V * 4), p = xmalloc ((size_t) V * 4), ord = xmalloc ((size_t) V * sizeof (int));
  float t = (float) temp;
  for (int i = 0; i < V; i++) z[i] = logits[i] / t;
  crnn_softmaxf (p, z, (size_t) V);
  for (int i = 0; i < V; i++) ord[i] = i;
  cmp_p = p;
  qsort (ord, (size_t) V, sizeof *ord, cmp_desc);
  double mass = 0;
  int n = 0;
  while (n < V && (n == 0 || mass < top_p)) mass += p[ord[n++]];
  double u;
  crdist_uniform (&u, 1, st);
  double target = u * mass, c = 0;
  for (int i = 0; i < n; i++) {
    c += p[ord[i]];
    if (target < c) return ord[i];
  }
  return ord[n - 1];
}

/* ------------------------------------------------------------ generation and receipts */

typedef struct { int n, *v; } ids_t;

static ids_t read_ids (const char *path)
{
  char *s = read_text (path, NULL);
  if (!s) die ("cannot read the prompt ids");
  ids_t r = {0, xmalloc (strlen (s) * sizeof (int) / 2 + 4)};
  for (char *p = s;;) {
    char *e;
    long v = strtol (p, &e, 10);
    if (e == p) {
      if (!*p) break;
      p++;
      continue;
    }
    r.v[r.n++] = (int) v, p = e;
  }
  free (s);
  return r;
}

typedef struct { double temp, top_p; uint64_t seed; int max_new; } settings_t;

static const char *logits_out; /* --logits: the logits after the prompt, binary32 */
static ids_t generate (ids_t prompt, settings_t S, double *secs)
{
  if (prompt.n == 0) die ("empty prompt");
  setup (prompt.n + S.max_new);
  float *logits = xmalloc ((size_t) M.V * 4);
  crdist_stream st = {S.seed, 0, 0};
  ids_t out = {0, xmalloc ((size_t) (S.max_new + 1) * sizeof (int))};
  struct timespec t0, t1;
  clock_gettime (CLOCK_MONOTONIC, &t0);
  forward_block (prompt.v, prompt.n, 0, logits, 1); /* the prompt as one block: step ()'s bits */
  npos = prompt.n;
  if (logits_out) {
    FILE *f = fopen (logits_out, "wb");
    if (!f || fwrite (logits, 4, (size_t) M.V, f) != (size_t) M.V) die ("cannot write the logits");
    fclose (f);
  }
  for (int g = 0; g < S.max_new; g++) {
    int t = sample (logits, S.temp, S.top_p, &st);
    out.v[out.n++] = t;
    if (t == M.eos || g + 1 == S.max_new) break;
    step (t, npos++, logits);
  }
  clock_gettime (CLOCK_MONOTONIC, &t1);
  *secs = (double) (t1.tv_sec - t0.tv_sec) + 1e-9 * (double) (t1.tv_nsec - t0.tv_nsec);
  free (logits);
  return out;
}

static void chain_hex (char out[65])
{
  sha256_ctx c;
  uint8_t d[32];
  sha256_init (&c);
  sha256_update (&c, hashes, (size_t) npos * (M.NL + 1) * 32);
  sha256_final (&c, d);
  sha256_tohex (d, out);
}

static void write_receipt (const char *path, ids_t prompt, ids_t out, settings_t S, double secs)
{
  FILE *f = fopen (path, "w");
  if (!f) die ("cannot write the receipt");
  char chain[65];
  chain_hex (chain);
  fprintf (f, "{\n  \"format\": \"rerun-receipt/0\",\n  \"contract\": \"%s\",\n", CONTRACT);
  fprintf (f, "  \"model\": {\"arch\": \"%s\", \"config_sha256\": \"%s\", \"files\": [", M.arch_name, M.cfg_sha);
  for (int k = 0; k < nshards; k++) fprintf (f, "%s{\"name\": \"%s\", \"sha256\": \"%s\"}", k ? ", " : "", shards[k].name, shards[k].sha);
  fprintf (f, "]},\n  \"settings\": {\"temp\": %.17g, \"top_p\": %.17g, \"seed\": %llu, \"max_new\": %d},\n", S.temp, S.top_p,
           (unsigned long long) S.seed, S.max_new);
  if (steer_add)
    fprintf (f, "  \"steer\": {\"file_sha256\": \"%s\", \"layer\": %d, \"alpha\": %.17g},\n", steerdir.sha, steerdir.layer, steer_alpha);
  if (probe.on) {
    fprintf (f, "  \"probe\": {\"name\": \"%s\", \"file_sha256\": \"%s\", \"layer\": %d, \"values\": [", probe.name, probe.sha, probe.layer);
    for (int p = 0; p < npos; p++) fprintf (f, "%s%.9g", p ? ", " : "", (double) probe_vals[p]);
    fprintf (f, "]},\n");
  }
  fprintf (f, "  \"prompt_ids\": [");
  for (int i = 0; i < prompt.n; i++) fprintf (f, "%s%d", i ? ", " : "", prompt.v[i]);
  fprintf (f, "],\n  \"output_ids\": [");
  for (int i = 0; i < out.n; i++) fprintf (f, "%s%d", i ? ", " : "", out.v[i]);
  fprintf (f, "],\n  \"chain_sha256\": \"%s\",\n  \"layers\": [\n", chain);
  for (int p = 0; p < npos; p++) {
    fprintf (f, "    \"");
    for (int l = 0; l <= M.NL; l++) {
      char hex[65];
      sha256_tohex (hashes + ((size_t) p * (M.NL + 1) + l) * 32, hex);
      fprintf (f, "%s%.16s", l ? " " : "", hex);
    }
    fprintf (f, "\"%s\n", p + 1 < npos ? "," : "");
  }
  fprintf (f, "  ],\n  \"made_with\": {\"seconds\": %.1f, \"threads\": %d, \"compiler\": \"%s\", \"flags\": \"%s\"}\n}\n", secs,
           nthreads,
#ifdef __VERSION__
           __VERSION__,
#else
           "?",
#endif
           BUILD_FLAGS);
  fclose (f);
}

static ids_t jarray (const char *js, const char *key)
{
  const char *v = jval (js, key);
  if (!v || *v != '[') die ("receipt: missing an array");
  ids_t r = {0, xmalloc (strlen (v) * sizeof (int) / 2 + 4)};
  for (v++; *v && *v != ']';) {
    char *e;
    long x = strtol (v, &e, 10);
    if (e == v) { v++; continue; }
    r.v[r.n++] = (int) x, v = e;
  }
  return r;
}

static int verify (const char *receipt, const char *dir, const char *probe_path, const char *steer_path)
{
  char *js = read_text (receipt, NULL);
  if (!js) die ("cannot read the receipt");
  char contract[64], chain[65];
  jstr (js, "contract", contract, sizeof contract);
  if (strcmp (contract, CONTRACT)) { printf ("NOT VERIFIED: contract %s, this build is %s\n", contract, CONTRACT); return 1; }
  load_model (dir);
  /* the model files: each shard's name in the receipt, and the sha256 after it */
  for (int k = 0; k < nshards; k++) {
    const char *p = js, *found = NULL;
    while ((p = jval (p, "name"))) {
      char nm[512];
      size_t i = 0;
      if (*p == '"')
        for (const char *q = p + 1; *q && *q != '"' && i + 1 < sizeof nm; q++) nm[i++] = *q;
      nm[i] = 0;
      if (!strcmp (nm, shards[k].name)) { found = p; break; }
    }
    char sha[80] = "";
    if (found) jstr (found, "sha256", sha, sizeof sha);
    if (strcmp (sha, shards[k].sha)) { printf ("NOT VERIFIED: model file %s is not the receipt's (sha256 %s)\n", shards[k].name, shards[k].sha); return 1; }
  }
  char cs[80];
  jstr (js, "config_sha256", cs, sizeof cs);
  if (strcmp (cs, M.cfg_sha)) { printf ("NOT VERIFIED: config.json is not the receipt's\n"); return 1; }
  /* a probe and steering, if the receipt has them: the same files */
  const char *pv = jval (js, "probe"), *sv = jval (js, "steer");
  char fsha[80];
  if (pv) {
    if (!probe_path) { printf ("NOT VERIFIED: the receipt has probe readouts; give the probe with --probe FILE\n"); return 1; }
    load_probe (&probe, probe_path);
    jstr (pv, "file_sha256", fsha, sizeof fsha);
    if (strcmp (fsha, probe.sha)) { printf ("NOT VERIFIED: the probe file is not the receipt's\n"); return 1; }
  }
  if (sv) {
    if (!steer_path) { printf ("NOT VERIFIED: the receipt was steered; give the direction with --steer FILE\n"); return 1; }
    load_probe (&steerdir, steer_path);
    jstr (sv, "file_sha256", fsha, sizeof fsha);
    if (strcmp (fsha, steerdir.sha)) { printf ("NOT VERIFIED: the steering file is not the receipt's\n"); return 1; }
    setup_steer (jnum (sv, "alpha", 0));
  }
  settings_t S;
  const char *st = jval (js, "settings");
  if (!st) die ("receipt: no settings");
  S.temp = jnum (st, "temp", 0), S.top_p = jnum (st, "top_p", 1), S.seed = (uint64_t) strtoull (jval (st, "seed"), NULL, 10);
  S.max_new = (int) jnum (st, "max_new", 0);
  ids_t prompt = jarray (js, "prompt_ids"), want = jarray (js, "output_ids");
  double secs;
  ids_t got = generate (prompt, S, &secs);
  /* the layers, position by position */
  const char *lp = jval (js, "layers");
  if (!lp || *lp != '[') die ("receipt: no layers");
  lp++;
  for (int p = 0; p < npos; p++) {
    lp = strchr (lp, '"');
    if (!lp) { printf ("NOT VERIFIED: the receipt has %d positions, the run %d\n", p, npos); return 1; }
    lp++;
    for (int l = 0; l <= M.NL; l++) {
      char hex[65];
      sha256_tohex (hashes + ((size_t) p * (M.NL + 1) + l) * 32, hex);
      if (strncmp (lp, hex, 16)) {
        printf ("NOT VERIFIED: position %d (%s token), layer %d is the first to differ\n", p, p < prompt.n ? "prompt" : "generated", l);
        return 1;
      }
      lp += 16 + (l < M.NL);
    }
    lp = strchr (lp, '"') + 1;
  }
  int same = got.n == want.n;
  for (int i = 0; same && i < got.n; i++) same = got.v[i] == want.v[i];
  if (!same) { printf ("NOT VERIFIED: the layers agree but the generated tokens differ\n"); return 1; }
  char mine[65];
  chain_hex (mine);
  jstr (js, "chain_sha256", chain, sizeof chain);
  if (strcmp (mine, chain)) { printf ("NOT VERIFIED: chain hash differs\n"); return 1; }
  if (pv) {
    const char *v = jval (pv, "values");
    if (!v || *v != '[') die ("receipt: no probe values");
    v++;
    for (int p = 0; p < npos; p++) {
      char *e;
      float x = strtof (v, &e);
      if (e == v) { printf ("NOT VERIFIED: the receipt has %d probe values, the run %d\n", p, npos); return 1; }
      if (memcmp (&x, &probe_vals[p], 4)) {
        printf ("NOT VERIFIED: the probe's value at position %d differs (%.9g, here %.9g)\n", p, (double) x, (double) probe_vals[p]);
        return 1;
      }
      v = e;
      while (*v == ',' || *v == ' ' || *v == '\n' || *v == '\r' || *v == '\t') v++;
    }
  }
  printf ("VERIFIED: %d positions, %d layers each, %d generated tokens%s%s, chain %.16s (%.1f s)\n", npos, M.NL + 1, got.n,
          pv ? ", the probe's values" : "", sv ? ", steered" : "", mine, secs);
  return 0;
}

/* rerun dump MODEL_DIR SEQS OUT --layers A[,B...] [--from I] [--to J] [--threads N] [--probe FILE]
   [--steer FILE ALPHA]: teacher-forced, every sequence (a line of token ids)
   from I to J (exclusive): at every position the residual stream after the
   given layers (OUT.L<layer>.f32, D values a position), the entropy of the
   next-token distribution (OUT.ent.f64), the probe's value (OUT.probe.f32);
   OUT.meta.txt says what was run. For training probes and testing them. */
typedef struct { const float *logits; double *ents; float *scratch; } ent_ctx;
static void ent_one (int t, void *p)
{
  ent_ctx *c = p;
  c->ents[t] = entropy (c->logits + (size_t) t * M.V, c->scratch + (size_t) t * M.V);
}

static int dump (int argc, char **argv)
{
  const char *dir = argv[2], *seqs = argv[3], *pre = argv[4], *probe_path = NULL, *steer_path = NULL;
  int from = 0, to = 1 << 30;
  double alpha = 0;
  char layers[128] = "";
  for (int i = 5; i < argc; i++) {
    if (!strcmp (argv[i], "--layers") && i + 1 < argc) snprintf (layers, sizeof layers, "%s", argv[++i]);
    else if (!strcmp (argv[i], "--from") && i + 1 < argc) from = atoi (argv[++i]);
    else if (!strcmp (argv[i], "--to") && i + 1 < argc) to = atoi (argv[++i]);
    else if (!strcmp (argv[i], "--threads") && i + 1 < argc) nthreads = atoi (argv[++i]);
    else if (!strcmp (argv[i], "--probe") && i + 1 < argc) probe_path = argv[++i];
    else if (!strcmp (argv[i], "--steer") && i + 2 < argc) steer_path = argv[++i], alpha = strtod (argv[++i], NULL);
    else die ("dump: unknown option");
  }
  load_model (dir);
  for (char *t = strtok (layers, ","); t && ndump < 8; t = strtok (NULL, ",")) {
    dump_layer[ndump] = atoi (t);
    if (dump_layer[ndump] < 1 || dump_layer[ndump] > M.NL) die ("dump: a layer must be 1 to the number of layers");
    ndump++;
  }
  if (probe_path) load_probe (&probe, probe_path);
  if (steer_path) load_probe (&steerdir, steer_path), setup_steer (alpha);
  char *text = read_text (seqs, NULL);
  if (!text) die ("cannot read the sequences");
  /* the sequences, one a line */
  int nseq = 0, maxlen = 0, cap = 1024;
  ids_t *sq = xmalloc ((size_t) cap * sizeof *sq);
  for (char *line = strtok (text, "\n"); line; line = strtok (NULL, "\n")) {
    ids_t r = {0, xmalloc (strlen (line) * sizeof (int) / 2 + 4)};
    for (char *p = line;;) {
      char *e;
      long v = strtol (p, &e, 10);
      if (e == p) { if (!*p) break; p++; continue; }
      r.v[r.n++] = (int) v, p = e;
    }
    if (!r.n) continue;
    if (nseq == cap) cap *= 2, sq = realloc (sq, (size_t) cap * sizeof *sq);
    sq[nseq++] = r;
    if (r.n > maxlen) maxlen = r.n;
  }
  if (to > nseq) to = nseq;
  if (from < 0 || from >= to) die ("dump: empty range");
  setup (maxlen);
  dump_xb = xmalloc ((size_t) maxlen * (ndump ? ndump : 1) * M.D * 4);
  float *logits = xmalloc ((size_t) maxlen * M.V * 4), *escratch = xmalloc ((size_t) maxlen * M.V * 4);
  double *ents = xmalloc ((size_t) maxlen * sizeof (double));
  char path[1024];
  FILE *fl[8], *fe, *fp = NULL;
  for (int k = 0; k < ndump; k++) {
    snprintf (path, sizeof path, "%s.L%d.f32", pre, dump_layer[k]);
    if (!(fl[k] = fopen (path, "wb"))) die ("cannot write the dump");
  }
  snprintf (path, sizeof path, "%s.ent.f64", pre);
  if (!(fe = fopen (path, "wb"))) die ("cannot write the dump");
  if (probe.on) {
    snprintf (path, sizeof path, "%s.probe.f32", pre);
    if (!(fp = fopen (path, "wb"))) die ("cannot write the dump");
  }
  long ntok = 0;
  struct timespec t0, t1;
  clock_gettime (CLOCK_MONOTONIC, &t0);
  for (int q = from; q < to; q++) {
    int n = sq[q].n;
    forward_block (sq[q].v, n, 0, logits, 0);
    ent_ctx ec = {logits, ents, escratch};
    parfor (n, ent_one, &ec);
    fwrite (ents, 8, (size_t) n, fe);
    for (int t = 0; t < n; t++)
      for (int k = 0; k < ndump; k++) fwrite (dump_xb + ((size_t) t * ndump + k) * M.D, 4, (size_t) M.D, fl[k]);
    if (fp) fwrite (probe_vals, 4, (size_t) n, fp);
    ntok += n;
  }
  clock_gettime (CLOCK_MONOTONIC, &t1);
  for (int k = 0; k < ndump; k++) fclose (fl[k]);
  fclose (fe);
  if (fp) fclose (fp);
  snprintf (path, sizeof path, "%s.meta.txt", pre);
  FILE *mf = fopen (path, "w");
  if (!mf) die ("cannot write the dump");
  fprintf (mf, "contract\t%s\nconfig_sha256\t%s\nsequences\t%d %d\ntokens\t%ld\nD\t%d\nlayers\t%s\n", CONTRACT, M.cfg_sha, from, to,
           ntok, M.D, argv[5] ? layers : "");
  for (int k = 0; k < nshards; k++) fprintf (mf, "file\t%s\t%s\n", shards[k].name, shards[k].sha);
  if (probe.on) fprintf (mf, "probe\t%s\t%d\t%s\n", probe.name, probe.layer, probe.sha);
  if (steer_add) fprintf (mf, "steer\t%s\t%d\t%.17g\n", steerdir.sha, steerdir.layer, steer_alpha);
  fprintf (mf, "seconds\t%.1f\n", (double) (t1.tv_sec - t0.tv_sec) + 1e-9 * (double) (t1.tv_nsec - t0.tv_nsec));
  fclose (mf);
  printf ("dumped %ld tokens of sequences %d to %d\n", ntok, from, to);
  return 0;
}

/* rerun batch MODEL_DIR PROMPTS OUT --max N [--layers A,B] [--stop ID,ID] [--threads N]: for
   every prompt (a line of token ids), greedy, one model load: OUT.ids.txt
   (the generated ids, a line a prompt), OUT.L<layer>.f32 (the residual
   stream after the layer at the prompt's last position), OUT.ent.f64 (the
   entropy of the first generated token's distribution); the same bits as
   gen. For probes on questions (what the model is about to answer). */
static int batch (int argc, char **argv)
{
  const char *dir = argv[2], *prompts = argv[3], *pre = argv[4];
  int maxn = 8, stops[16], nstop = 0, share = 1, nsamp = 0;
  double stemp = 1.0, stop_p = 1.0;
  uint64_t sseed = 0;
  char layers[128] = "", stopsarg[128] = "";
  for (int i = 5; i < argc; i++) {
    if (!strcmp (argv[i], "--max") && i + 1 < argc) maxn = atoi (argv[++i]);
    else if (!strcmp (argv[i], "--layers") && i + 1 < argc) snprintf (layers, sizeof layers, "%s", argv[++i]);
    else if (!strcmp (argv[i], "--threads") && i + 1 < argc) nthreads = atoi (argv[++i]);
    else if (!strcmp (argv[i], "--stop") && i + 1 < argc) snprintf (stopsarg, sizeof stopsarg, "%s", argv[++i]);
    else if (!strcmp (argv[i], "--no-share")) share = 0;
    else if (!strcmp (argv[i], "--samples") && i + 1 < argc) nsamp = atoi (argv[++i]);
    else if (!strcmp (argv[i], "--temp") && i + 1 < argc) stemp = strtod (argv[++i], NULL);
    else if (!strcmp (argv[i], "--top-p") && i + 1 < argc) stop_p = strtod (argv[++i], NULL);
    else if (!strcmp (argv[i], "--seed") && i + 1 < argc) sseed = strtoull (argv[++i], NULL, 10);
    else die ("batch: unknown option");
  }
  load_model (dir);
  for (char *t = strtok (stopsarg, ","); t && nstop < 16; t = strtok (NULL, ",")) stops[nstop++] = atoi (t);
  for (char *t = strtok (layers, ","); t && ndump < 8; t = strtok (NULL, ",")) {
    dump_layer[ndump] = atoi (t);
    if (dump_layer[ndump] < 1 || dump_layer[ndump] > M.NL) die ("batch: a layer must be 1 to the number of layers");
    ndump++;
  }
  char *text = read_text (prompts, NULL);
  if (!text) die ("cannot read the prompts");
  int np = 0, maxlen = 0, cap = 1024;
  ids_t *pq = xmalloc ((size_t) cap * sizeof *pq);
  for (char *line = strtok (text, "\n"); line; line = strtok (NULL, "\n")) {
    ids_t r = {0, xmalloc (strlen (line) * sizeof (int) / 2 + 4)};
    for (char *p = line;;) {
      char *e;
      long v = strtol (p, &e, 10);
      if (e == p) { if (!*p) break; p++; continue; }
      r.v[r.n++] = (int) v, p = e;
    }
    if (!r.n) continue;
    if (np == cap) cap *= 2, pq = realloc (pq, (size_t) cap * sizeof *pq);
    pq[np++] = r;
    if (r.n > maxlen) maxlen = r.n;
  }
  setup (maxlen + maxn);
  dump_xb = xmalloc ((size_t) (maxlen + maxn) * (ndump ? ndump : 1) * M.D * 4);
  dump_x = xmalloc ((size_t) (ndump ? ndump : 1) * M.D * 4);
  float *logits = xmalloc ((size_t) M.V * 4), *scr = xmalloc ((size_t) M.V * 4);
  char path[1024];
  FILE *fl[8], *fe, *fo;
  for (int k = 0; k < ndump; k++) {
    snprintf (path, sizeof path, "%s.L%d.f32", pre, dump_layer[k]);
    if (!(fl[k] = fopen (path, "wb"))) die ("cannot write the batch");
  }
  snprintf (path, sizeof path, "%s.ent.f64", pre);
  if (!(fe = fopen (path, "wb"))) die ("cannot write the batch");
  snprintf (path, sizeof path, "%s.ids.txt", pre);
  if (!(fo = fopen (path, "w"))) die ("cannot write the batch");
  settings_t S = {0, 1, 0, maxn};
  /* the prefix all prompts share (leaving each at least one token of its
     own): computed once; each prompt then runs from there on the cache,
     which holds the same bits a fresh run would (--no-share: from 0) */
  int P = 0;
  if (share && np > 1) {
    int minn = pq[0].n;
    P = pq[0].n;
    for (int q = 1; q < np; q++) {
      int l = 0;
      while (l < P && l < pq[q].n && pq[q].v[l] == pq[0].v[l]) l++;
      P = l;
      if (pq[q].n < minn) minn = pq[q].n;
    }
    if (P > minn - 1) P = minn - 1;
  }
  FILE *fs = NULL;
  if (nsamp > 0) {
    snprintf (path, sizeof path, "%s.samples.txt", pre);
    if (!(fs = fopen (path, "w"))) die ("cannot write the batch");
  }
  float *plog = nsamp > 0 ? xmalloc ((size_t) M.V * 4) : NULL;
  struct timespec t0, t1;
  clock_gettime (CLOCK_MONOTONIC, &t0);
  if (P > 0) forward_block (pq[0].v, P, 0, NULL, 0);
  for (int q = 0; q < np; q++) {
    forward_block (pq[q].v + P, pq[q].n - P, P, logits, 1);
    if (plog) memcpy (plog, logits, (size_t) M.V * 4);
    for (int k = 0; k < ndump; k++) fwrite (dump_xb + ((size_t) (pq[q].n - 1 - P) * ndump + k) * M.D, 4, (size_t) M.D, fl[k]);
    double h = entropy (logits, scr);
    fwrite (&h, 8, 1, fe);
    int pos = pq[q].n;
    for (int g = 0; g < maxn; g++) {
      int t = sample (logits, S.temp, S.top_p, NULL);
      fprintf (fo, "%s%d", g ? " " : "", t);
      int stop = t == M.eos || g + 1 == maxn;
      for (int k = 0; k < nstop; k++) stop |= t == stops[k];
      if (stop) break;
      step (t, pos++, logits);
    }
    fprintf (fo, "\n");
    /* samples: each from the prompt's state (positions before the prompt's
       end untouched), on its own stream: seed, substream q * K + k */
    for (int k = 0; k < nsamp; k++) {
      crdist_stream st = {sseed, (uint64_t) q * (uint64_t) nsamp + (uint64_t) k, 0};
      memcpy (logits, plog, (size_t) M.V * 4);
      int pos2 = pq[q].n;
      fprintf (fs, "%s", k ? " | " : "");
      for (int g = 0; g < maxn; g++) {
        int t = sample (logits, stemp, stop_p, &st);
        fprintf (fs, "%s%d", g ? " " : "", t);
        int stop = t == M.eos || g + 1 == maxn;
        for (int j = 0; j < nstop; j++) stop |= t == stops[j];
        if (stop) break;
        step (t, pos2++, logits);
      }
    }
    if (fs) fprintf (fs, "\n");
    if ((q + 1) % 100 == 0) fprintf (stderr, "%d of %d\n", q + 1, np);
  }
  clock_gettime (CLOCK_MONOTONIC, &t1);
  for (int k = 0; k < ndump; k++) fclose (fl[k]);
  fclose (fe), fclose (fo);
  if (fs) fclose (fs);
  snprintf (path, sizeof path, "%s.meta.txt", pre);
  FILE *mf = fopen (path, "w");
  if (!mf) die ("cannot write the batch");
  fprintf (mf, "contract\t%s\nconfig_sha256\t%s\nprompts\t%d\nmax\t%d\nD\t%d\nthreads\t%d\nshared_prefix\t%d\n", CONTRACT, M.cfg_sha, np,
           maxn, M.D, nthreads, P);
  if (nsamp) fprintf (mf, "samples\t%d\ttemp %.17g\ttop_p %.17g\tseed %llu\n", nsamp, stemp, stop_p, (unsigned long long) sseed);
  for (int k = 0; k < nshards; k++) fprintf (mf, "file\t%s\t%s\n", shards[k].name, shards[k].sha);
  fprintf (mf, "seconds\t%.1f\n", (double) (t1.tv_sec - t0.tv_sec) + 1e-9 * (double) (t1.tv_nsec - t0.tv_nsec));
  fclose (mf);
  printf ("batch: %d prompts\n", np);
  return 0;
}

int main (int argc, char **argv)
{
  if (!sha256_selftest ()) die ("SHA-256 self-test failed");
  if (argc >= 3 && !strcmp (argv[1], "verify")) {
    const char *dir = argc >= 4 && argv[3][0] != '-' ? argv[3] : ".", *pp = NULL, *sp = NULL;
    for (int i = 3; i + 1 < argc; i++) {
      if (!strcmp (argv[i], "--threads")) nthreads = atoi (argv[i + 1]);
      if (!strcmp (argv[i], "--probe")) pp = argv[i + 1];
      if (!strcmp (argv[i], "--steer")) sp = argv[i + 1];
    }
    return verify (argv[2], dir, pp, sp);
  }
  if (argc >= 6 && !strcmp (argv[1], "dump")) return dump (argc, argv);
  if (argc >= 5 && !strcmp (argv[1], "batch")) return batch (argc, argv);
  if (argc < 6 || strcmp (argv[1], "gen"))
    die ("usage: rerun gen MODEL_DIR PROMPT_IDS MAX_NEW RECEIPT.json [--temp T] [--top-p P] [--seed S] [--threads N] "
         "[--layers-tsv FILE] | rerun verify RECEIPT.json [MODEL_DIR] [--threads N]");
  settings_t S = {0, 1, 0, atoi (argv[4])};
  const char *tsv = NULL, *probe_path = NULL, *steer_path = NULL;
  double alpha = 0;
  for (int i = 6; i + 1 < argc; i += 2) {
    if (!strcmp (argv[i], "--steer") && i + 2 < argc) { steer_path = argv[i + 1], alpha = strtod (argv[i + 2], NULL), i++; continue; }
    if (!strcmp (argv[i], "--temp")) S.temp = strtod (argv[i + 1], NULL);
    else if (!strcmp (argv[i], "--top-p")) S.top_p = strtod (argv[i + 1], NULL);
    else if (!strcmp (argv[i], "--seed")) S.seed = strtoull (argv[i + 1], NULL, 10);
    else if (!strcmp (argv[i], "--threads")) nthreads = atoi (argv[i + 1]);
    else if (!strcmp (argv[i], "--layers-tsv")) tsv = argv[i + 1];
    else if (!strcmp (argv[i], "--logits")) logits_out = argv[i + 1];
    else if (!strcmp (argv[i], "--probe")) probe_path = argv[i + 1];
    else die ("unknown option");
  }
  if (S.max_new < 1) die ("MAX_NEW must be at least 1");
  load_model (argv[2]);
  if (probe_path) load_probe (&probe, probe_path);
  if (steer_path) load_probe (&steerdir, steer_path), setup_steer (alpha);
  ids_t prompt = read_ids (argv[3]);
  double secs;
  ids_t out = generate (prompt, S, &secs);
  write_receipt (argv[5], prompt, out, S, secs);
  if (tsv) {
    FILE *f = fopen (tsv, "w");
    if (!f) die ("cannot write the layers file");
    fprintf (f, "prompt\tpos\tlayer\tsha256\n");
    for (int p = 0; p < npos; p++)
      for (int l = 0; l <= M.NL; l++) {
        char hex[65];
        sha256_tohex (hashes + ((size_t) p * (M.NL + 1) + l) * 32, hex);
        fprintf (f, "0\t%d\t%d\t%s\n", p, l, hex);
      }
    fclose (f);
  }
  printf ("generated %d tokens in %.1f s; receipt %s\n", out.n, secs, argv[5]);
  for (int i = 0; i < out.n; i++) printf ("%s%d", i ? " " : "", out.v[i]);
  printf ("\n");
  return 0;
}
