// tb.cpp: philox4x32_10.v against libcrdist's stream, bit for bit, under Verilator (2026-10-15).
//
// Every block the RTL computes is compared with crdist_words() for the same stream position: Random123's three known
// answers, the four stream positions crdist-check uses, and N random (seed, substream, j < 2^62) with the counter's
// high words often at their extremes. Also the latency (done exactly 10 clocks after start), back-to-back starts, and a
// reset in the middle of a block. Built with -DPHILOX_PLANTED_BUG, the RTL is wrong by one Weyl constant and this
// check must fail: that is its control (make philox-check runs both).
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include "Vphilox4x32_10.h"
#include "verilated.h"
#include "crdist.h"

static Vphilox4x32_10 *m;
static uint64_t cycles;

static void tick()
{
  m->clk = 0; m->eval();
  m->clk = 1; m->eval();
  cycles++;
}

// one block from the RTL; returns the clocks from start to done (or -1 on a timeout)
static int block(const uint32_t c[4], const uint32_t k[2], uint32_t w[4])
{
  for (int i = 0; i < 4; i++) m->ctr[i] = c[i];
  m->key = (uint64_t)k[1] << 32 | k[0];
  m->start = 1; tick(); m->start = 0;
  for (int n = 1; n <= 40; n++) {
    tick();
    if (m->done) {
      for (int i = 0; i < 4; i++) w[i] = m->out[i];
      return n;
    }
  }
  return -1;
}

// crdist's stream: words 4j .. 4j + 3 of (seed, substream)
static void ref(uint64_t seed, uint64_t sub, uint64_t j, uint32_t w[4])
{
  crdist_stream s = {seed, sub, 4 * j};
  crdist_words(w, 4, &s);
}

static uint64_t rng = 0x243f6a8885a308d3ULL;
static uint64_t next64()
{
  rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
  return rng;
}

int main(int argc, char **argv)
{
  long n = argc > 1 ? atol(argv[1]) : 1000000;
  Verilated::commandArgs(argc, argv);
  m = new Vphilox4x32_10;
  m->rst = 1; tick(); tick(); m->rst = 0; tick();
  long bad = 0, latbad = 0, total = 0;

  // Random123's known answers for Philox4x32-10: counter {c0..c3}, key {k0, k1}, the block
  static const uint32_t KAT[3][10] = {
    {0, 0, 0, 0, 0, 0, 0x6627e8d5, 0xe169c58d, 0xbc57ac4c, 0x9b00dbd8},
    {0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0x408f276d, 0x41c83b0e, 0xa20bc7c6, 0x6d5451fd},
    {0x243f6a88, 0x85a308d3, 0x13198a2e, 0x03707344, 0xa4093822, 0x299f31d0, 0xd16cfe09, 0x94fdcceb, 0x5001e420, 0x24126ea1}};
  int kat = 0;
  for (int t = 0; t < 3; t++) {
    uint32_t w[4];
    int lat = block(KAT[t], KAT[t] + 4, w);
    kat += lat == 10 && w[0] == KAT[t][6] && w[1] == KAT[t][7] && w[2] == KAT[t][8] && w[3] == KAT[t][9];
  }

  // the stream, at crdist-check's four positions and then at random ones
  static const uint64_t ST[4][3] = {{0, 0, 0}, {42, 7, 1}, {0xfedcba9876543210ULL, 0x8000000000000005ULL, 0x123456789abULL},
                                    {1, 0xffffffffffffffffULL, 0x3ffffffffffffff0ULL}};
  for (long i = 0; i < n + 4; i++) {
    uint64_t seed, sub, j;
    if (i < 4) { seed = ST[i][0]; sub = ST[i][1]; j = ST[i][2] / 4; }
    else {
      // crdist's index is a 64-bit word count, so its stream reaches blocks j < 2^62 (the known answers cover the
      // counter's top values)
      seed = next64(); sub = next64(); j = next64() >> 2;
      switch (i % 8) {   // the counter's and key's extremes, where carries and wraps live
        case 1: j = (~0ULL >> 2) - (i & 3); break;
        case 2: sub = ~0ULL; break;
        case 3: seed = 0; break;
        case 4: j &= 0xffffffffULL; break;
        default: break;
      }
    }
    uint32_t c[4] = {(uint32_t)j, (uint32_t)(j >> 32), (uint32_t)sub, (uint32_t)(sub >> 32)};
    uint32_t k[2] = {(uint32_t)seed, (uint32_t)(seed >> 32)};
    uint32_t w[4], r[4];
    int lat = block(c, k, w);
    ref(seed, sub, j, r);
    for (int q = 0; q < 4; q++) bad += w[q] != r[q];
    latbad += lat != 10;
    total++;
  }

  // a reset in the middle of a block leaves it idle, and the next block is still right
  {
    uint32_t c[4] = {1, 2, 3, 4}, k[2] = {5, 6}, w[4], r[4];
    for (int i = 0; i < 4; i++) m->ctr[i] = c[i];
    m->key = (uint64_t)k[1] << 32 | k[0];
    m->start = 1; tick(); m->start = 0; tick(); tick();
    m->rst = 1; tick(); m->rst = 0; tick();
    int idle = !m->busy && !m->done;
    int lat = block(c, k, w);
    crdist_stream s = {(uint64_t)k[1] << 32 | k[0], (uint64_t)c[3] << 32 | c[2], 4 * ((uint64_t)c[1] << 32 | c[0])};
    crdist_words(r, 4, &s);
    int same = lat == 10;
    for (int q = 0; q < 4; q++) same &= w[q] == r[q];
    bad += !(idle && same);
  }

  printf("philox4x32_10: Random123's known answers %d of 3; %ld blocks (%ld words) against crdist_words: %ld words differ; "
         "latency other than 10 clocks: %ld; %" PRIu64 " clocks\n", kat, total, 4 * total, bad, latbad, cycles);
  int ok = kat == 3 && bad == 0 && latbad == 0;
  printf("VERDICT: %s\n", ok ? "IDENTICAL the RTL gives crdist's stream on every block checked, in 10 clocks each"
                             : "DIFFERS the RTL is not crdist's stream");
  delete m;
  return ok ? 0 : 1;
}
