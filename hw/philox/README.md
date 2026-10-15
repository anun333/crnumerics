# hw/philox: crdist's random stream in hardware

`philox4x32_10.v` is Philox4x32-10 (Salmon, Moraes, Dror and Shaw, SC 2011),
the counter-based generator under crdist's samplers, as plain Verilog 2005:
one round per clock, a block of four 32-bit words ten clocks after `start`.
Given the counter {substream, j} and the key {seed}, it produces words 4j to
4j + 3 of crdist's stream (seed, substream), the same bits `crdist_words`
gives in software. Every crdist sampler is a fixed sequence of operations
on those words, so the same seed gives the same draws in hardware and in
software.

```
make philox-check    # Verilator, Icarus Verilog and Yosys
```

checks it:
- **Verilator:** Random123's three known answers, and 1,000,004 blocks
  (4,000,016 words) at random stream positions against `crdist_words()`
  from libcrdist, every word equal, ten clocks each.
- **Icarus Verilog:** the known answers again, in a second simulator.
- **The control:** both again with `PHILOX_PLANTED_BUG` (one key constant
  off by one), which must fail, and does: 0 of 3 known answers.
- **Yosys:** 6,358 cells, two-input gates and 326 flip-flops. The two
  multipliers are by constants, so they shrink. On SkyWater 130 nm that is
  roughly a 4×2 block of Tiny Tapeout tiles, by typical cell sizes; that is
  an estimate, not a placed layout.

The interface is a start pulse with `ctr` and `key`, then `done` with the
block on `out`. A bus wrapper (Wishbone or AXI-Lite) and a smaller variant
with a serial multiplier are the next steps. Why it exists: an open
hardware catalog checked on 2026-10-15 found one other Philox in RTL, under
a custom license.
