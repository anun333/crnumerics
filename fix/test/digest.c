/* digest: a hash of crfix's results over a broad domain, to compare builds and machines (2026-10-13). The same
   number from a 64-bit host with unsigned __int128 and from a 32-bit target without it, or with no FPU, shows the
   two sets of arithmetic give the same bits. FNV-1a over:
     the nine one-argument functions at every int16 input, fin and fout in {0, 15, 31}, every mode, int32 out;
     div over every int8 pair at fa, fb in {0, 7}, fout in {0, 15}, every mode;
     affine int8 tables for ten configurations of each approximated function, every mode;
     softmax of twenty int8 vectors, every mode.
   No floating-point operation: the scales below are constants, passed as bits. Prints the digest and a count.
   digest control: the same with one result made one larger, which must change the digest. */
#include <stdint.h>
#include <stdio.h>
#include "crfix.h"

static uint64_t h = 0xcbf29ce484222325ULL;
static unsigned long long n;
static int control;
static void put(int32_t v)
{
  if (control && n == 1234567) v++;
  for (int i = 0; i < 4; i++) { h ^= (uint8_t)(v >> (8 * i)); h *= 0x100000001b3ULL; }
  n++;
}

int main(int argc, char **argv)
{
  control = argc > 1 && argv[1][0] == 'c';
  static const int f3[] = {0, 15, 31};
  for (int fn = 0; fn < CRFIX_NFN; fn++)
    for (int a = 0; a < 3; a++)
      for (int b = 0; b < 3; b++)
        for (int mode = 0; mode < 4; mode++)
          for (int32_t x = -32768; x <= 32767; x++) put(crfix_eval(fn, x, f3[a], 32, f3[b], mode));
  for (int fa = 0; fa <= 7; fa += 7)
    for (int fb = 0; fb <= 7; fb += 7)
      for (int fout = 0; fout <= 15; fout += 15)
        for (int mode = 0; mode < 4; mode++)
          for (int32_t a = -128; a < 128; a++)
            for (int32_t b = -128; b < 128; b++) put(crfix_div(a, fa, b, fb, 16, fout, mode));
  /* affine: input scales 2^-k (1 + j/8), zero points -k; outputs TFLite's for sigmoid and tanh, else 2^-5 */
  static const double sc[10] = {0x1p-1, 0x1.2p-2, 0x1.4p-3, 0x1.6p-4, 0x1.8p-5, 0x1.ap-3, 0x1.cp-6, 0x1.ep-2, 0x1.1p-7, 0x1.3p-4};
  int32_t t[256];
  for (int fn = 0; fn <= CRFIX_TANH; fn++)
    for (int c = 0; c < 10; c++)
      for (int mode = 0; mode < 4; mode++) {
        crfix_affine in = {sc[c], -c, 8};
        crfix_affine out = fn == CRFIX_SIGMOID ? (crfix_affine){0x1p-8, -128, 8}
                         : fn == CRFIX_TANH ? (crfix_affine){0x1p-7, 0, 8} : (crfix_affine){0x1p-5, 3, 16};
        crfix_table_affine(fn, &in, &out, mode, t);
        for (int i = 0; i < 256; i++) put(t[i]);
      }
  uint64_t g = 7;
  crfix_affine so = {0x1p-8, -128, 8};
  for (int v = 0; v < 20; v++) {
    int32_t q[40], y[40];
    size_t len = 1 + (size_t)(v * 7 % 40);
    for (size_t i = 0; i < len; i++) { g = g * 6364136223846793005ULL + 1442695040888963407ULL; q[i] = (int32_t)(g >> 56) - 128; }
    for (int mode = 0; mode < 4; mode++) {
      crfix_softmax(len, q, sc[v % 10], &so, mode, y);
      for (size_t i = 0; i < len; i++) put(y[i]);
    }
  }
  printf("crfix %s digest %016llx over %llu results\n", crfix_version(), (unsigned long long)h, n);
  return 0;
}
