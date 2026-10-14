#!/bin/sh
# cross-check.sh: crfix on 32-bit ARM without an FPU, against the host (2026-10-14; make crfix-cross-check).
#   1  the library built for Cortex-M0 (-mfloat-abi=soft) calls no floating-point routine and no libm function
#      (control: a file that adds two doubles must show __aeabi_dadd), and has no instruction beyond ARMv6-M
#      (control: the same code built for Cortex-M3 must show some, such as it, cbz and udiv)
#   2  fix/test/digest.c's hash of 23 million results: the host's native build, its portable build, 32-bit ARM
#      Linux with soft float under qemu-arm, and the Cortex-M0 build bare-metal under qemu-system-arm (MPS2-AN385,
#      a Cortex-M3 without FPU, which runs ARMv6-M code) must all give the same digest
#   3  control: the digest with one result changed must differ
# Needs arm-none-eabi-gcc with newlib, arm-linux-gnueabi-gcc, qemu-arm and qemu-system-arm; without them, VOID.
set -u
B=${B:-build/crfix-cross}
mkdir -p "$B"
SRC="fix/test/digest.c fix/crfix.c fix/crfix-mp.c"
for t in arm-none-eabi-gcc arm-none-eabi-nm arm-none-eabi-objdump arm-linux-gnueabi-gcc qemu-arm qemu-system-arm; do
  command -v $t > /dev/null || { echo "VERDICT: VOID: $t is not installed"; exit 2; }
done
bad=0
# 1: no floating point
M0="-mcpu=cortex-m0 -mthumb -mfloat-abi=soft -O2"
FP='__aeabi_[df]|__aeabi_[iu]2[df]|__aeabi_[df]2|sqrt|exp|log|ldexp|frexp|__.*sf[23]|__.*df[23]'
arm-none-eabi-gcc $M0 -I fix -c -o "$B/crfix.o" fix/crfix.c && arm-none-eabi-gcc $M0 -I fix -c -o "$B/crfix-mp.o" fix/crfix-mp.c || exit 1
fpcalls=$(arm-none-eabi-nm -u "$B/crfix.o" "$B/crfix-mp.o" | grep -cE "$FP")
printf 'double f(double a, double b) { return a + b; }\n' > "$B/ctl.c"
arm-none-eabi-gcc $M0 -c -o "$B/ctl.o" "$B/ctl.c"
ctl=$(arm-none-eabi-nm -u "$B/ctl.o" | grep -cE "$FP")
V7='	(v[a-z]+|ldrd|strd|sdiv|udiv|cbn?z|it[te]*)[. 	]'
v7=$(arm-none-eabi-objdump -d "$B/crfix.o" "$B/crfix-mp.o" | grep -cE "$V7")
arm-none-eabi-gcc -mcpu=cortex-m3 -mthumb -mfloat-abi=soft -O2 -I fix -c -o "$B/crfix-m3.o" fix/crfix.c
v7ctl=$(arm-none-eabi-objdump -d "$B/crfix-m3.o" | grep -cE "$V7")
echo "Cortex-M0 build: $fpcalls floating-point or libm calls (must be 0; control: $ctl, must be > 0), $v7 instructions beyond ARMv6-M (must be 0; control, built for M3: $v7ctl, must be > 0)"
[ "$fpcalls" = 0 ] && [ "$v7" = 0 ] || bad=1
[ "$ctl" -gt 0 ] && [ "$v7ctl" -gt 0 ] || { echo "VERDICT: VOID: a control was not caught"; exit 2; }
# 2: the digests
cc -O2 -I fix -o "$B/native" $SRC && cc -O2 -DCRFIX_PORTABLE -I fix -o "$B/portable" $SRC || exit 1
arm-linux-gnueabi-gcc -O2 -static -march=armv6 -mfloat-abi=soft -I fix -o "$B/armel" $SRC || exit 1
arm-none-eabi-gcc $M0 --specs=rdimon.specs -T fix/test/m-profile/mps2-an385.ld -I fix -o "$B/m0.elf" \
  fix/test/m-profile/vectors.c $SRC 2> "$B/m0-link.log" || { cat "$B/m0-link.log"; exit 1; }
d() { echo "$1" | sed -n 's/.*digest \([0-9a-f]*\) .*/\1/p'; }
dn=$(d "$("$B/native")"); dp=$(d "$("$B/portable")"); da=$(d "$(qemu-arm "$B/armel")")
dm=$(d "$(timeout 3000 qemu-system-arm -M mps2-an385 -nographic -semihosting-config enable=on,target=native -kernel "$B/m0.elf" 2>&1)")
echo "digests: host native $dn, host portable $dp, 32-bit ARM Linux soft float $da, Cortex-M0 bare metal $dm"
[ -n "$dn" ] && [ "$dn" = "$dp" ] && [ "$dn" = "$da" ] && [ "$dn" = "$dm" ] || bad=1
# 3: the control
dc=$(d "$("$B/native" control)")
echo "control: one result changed gives $dc (must differ)"
[ -n "$dc" ] && [ "$dc" != "$dn" ] || { echo "VERDICT: VOID: the digest control did not differ"; exit 2; }
if [ $bad = 0 ]; then echo "VERDICT: IDENTICAL crfix has no floating point on Cortex-M0 and gives the host's bits on 32-bit ARM, soft float and bare metal"
else echo "VERDICT: DIFFERS: floating point in the M0 build, or a digest that is not the host's"; exit 1; fi
