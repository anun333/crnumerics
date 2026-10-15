#!/bin/sh
# check.sh (make philox-check): philox4x32_10.v against crdist's stream (2026-10-15). Needs Verilator, Icarus Verilog
# and Yosys, and build/libcrdist.a (the make target builds it).
#   1. Verilator: the RTL against crdist_words() on Random123's known answers and N blocks (default 1,000,000).
#   2. Icarus Verilog: the known answers again, in a second simulator.
#   3. The control: both again with PHILOX_PLANTED_BUG (one Weyl constant off by one), which must fail.
#   4. Yosys: the cell count of the core, for its size.
set -u
B=${B:-build/philox}; N=${N:-1000000}; ROOT=$(pwd)
mkdir -p "$B"
for t in verilator iverilog vvp yosys; do command -v $t > /dev/null || { echo "philox-check: $t not found"; exit 2; }; done
fail=0
build() {   # $1: obj dir, $2: extra define
  verilator --cc --exe --build -O2 -Wno-fatal $2 --Mdir "$B/$1" -o philox-tb -CFLAGS "-O2 -I$ROOT/dist" \
    -LDFLAGS "$ROOT/build/libcrdist.a -lm" "$ROOT/hw/philox/philox4x32_10.v" "$ROOT/hw/philox/test/tb.cpp" > "$B/$1.log" 2>&1 ||
    { echo "philox-check: the Verilator build ($1) failed, see $B/$1.log"; exit 2; }
}
build obj ""
build obj-bug -DPHILOX_PLANTED_BUG
"$B/obj/philox-tb" "$N" | tee "$B/verilator.txt" | head -1
grep -q "VERDICT: IDENTICAL" "$B/verilator.txt" || fail=1
"$B/obj-bug/philox-tb" 20000 > "$B/verilator-bug.txt"
ctl1=$(grep -c "VERDICT: DIFFERS" "$B/verilator-bug.txt")
iverilog -g2005 -o "$B/kat.vvp" hw/philox/philox4x32_10.v hw/philox/test/tb_kat.v &&
  iverilog -g2005 -DPHILOX_PLANTED_BUG -o "$B/kat-bug.vvp" hw/philox/philox4x32_10.v hw/philox/test/tb_kat.v ||
  { echo "philox-check: the Icarus build failed"; exit 2; }
kat=$(vvp -n "$B/kat.vvp" | sed -n 's/^known answers \([0-9]\) of 3$/\1/p')
katbug=$(vvp -n "$B/kat-bug.vvp" | sed -n 's/^known answers \([0-9]\) of 3$/\1/p')
echo "Icarus Verilog: known answers ${kat:-?} of 3; with the planted bug ${katbug:-?} of 3 (must be 0)"
[ "$kat" = 3 ] || fail=1
ctl2=0; [ "$katbug" = 0 ] && ctl2=1
echo "control: the planted bug fails under Verilator: $([ "$ctl1" = 1 ] && echo yes || echo NO); under Icarus: $([ "$ctl2" = 1 ] && echo yes || echo NO)"
yosys -q -p "read_verilog hw/philox/philox4x32_10.v; synth -top philox4x32_10 -flatten; abc -g AND,NAND,OR,NOR,XOR,XNOR,ANDNOT,ORNOT,MUX; opt_clean; tee -q -o $B/yosys-stat.txt stat" > "$B/yosys.log" 2>&1 &&
  echo "Yosys: $(sed -n 's/^ *Number of cells: *\([0-9]*\)/\1/p' "$B/yosys-stat.txt" | head -1) cells (two-input gates and flip-flops)"
if [ $fail = 0 ] && [ "$ctl1" = 1 ] && [ "$ctl2" = 1 ]; then
  echo "VERDICT: IDENTICAL the Philox4x32-10 RTL gives crdist's stream bit for bit in two simulators, and the control differs"
else
  echo "VERDICT: DIFFERS"; exit 1
fi
