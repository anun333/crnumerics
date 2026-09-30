# numerics: the groundwork for reproducible numerics libraries (README.md).
#   make         the kit and its self-test
#   make check   every check here; each ends in a verdict line, and one
#                that is not IDENTICAL fails the target
# gcc 13 or later (_Float16, __bf16), MPFR 4.2 or later, OpenMP; for
# repro-scan, Python 3 and binutils (and, for its tests' other ISAs,
# clang and the aarch64 and riscv64 cross compilers, or they are skipped).
CC      ?= gcc
CFLAGS  ?= -O2
# as in crmvec: no contraction, and no constant folding that assumes
# round-to-nearest, so that C code rounds as written in every mode
FP      := -ffp-contract=off -frounding-math
ROOT    := core-math
B       := build
KIT     := kit/fmt.c kit/run.c kit/fns.c kit/mx.c
# CORE-MATH's correctly rounded functions (vendored in core-math/), the
# answers the kit's self-test checks it against
CMSRC   := $(wildcard $(ROOT)/f16/*.c) $(wildcard $(ROOT)/bf16/*.c) \
           $(addprefix $(ROOT)/,expf.c logf.c sinf.c atan2f.c pow/pow.c atan2/atan2.c hypot.c \
             acos.c acosh.c acospi.c asin.c asinh.c asinpi.c atan.c atanh.c atanpi.c cbrt.c cos.c cosh.c \
             cospi.c erf.c erfc.c exp.c exp10.c exp2.c expm1.c lgamma.c log/log.c log10/log10.c log1p.c \
             log2.c rsqrt.c sin.c sinh.c sinpi.c tan.c tanh.c tanpi.c tgamma.c)
# lowp (lowp/lowp.h): its own copy of the CORE-MATH functions it calls, made
# local to it, so that it links beside CORE-MATH or crmvec without a clash
LOWPCM  := $(addprefix $(ROOT)/,pow/pow.c atan2/atan2.c atan2pi/atan2pi.c hypot.c \
             acos.c acosh.c acospi.c asin.c asinh.c asinpi.c atan.c atanh.c atanpi.c cbrt.c cos.c cosh.c \
             cospi.c erf.c erfc.c exp.c exp10.c exp2.c expm1.c lgamma.c log/log.c log10/log10.c log1p.c \
             log2.c rsqrt.c sin.c sinh.c sinpi.c tan.c tanh.c tanpi.c tgamma.c)
LOWPH   := lowp/lowp.h lowp/lowp-list.h lowp/lowp-mx-list.h lowp/lowp-tables.h
VERDICTS := '^VERDICT: IDENTICAL'

all: $(B)/selftest $(B)/liblowp.a $(B)/liblowp.so $(B)/lowp-check $(B)/mx-check $(B)/libival.a $(B)/libival.so \
     $(B)/ival-check

$(B)/libkit.a: $(KIT) kit/kit.h
	mkdir -p $(B)/kit
	for f in $(KIT); do $(CC) $(CFLAGS) $(FP) -fopenmp -Wall -Wextra -c -o $(B)/kit/$$(basename $$f .c).o $$f || exit 1; done
	rm -f $@ && ar rcs $@ $(B)/kit/*.o

$(B)/libcm.a: $(CMSRC) Makefile
	rm -rf $(B)/cm && mkdir -p $(B)/cm
	for f in $(CMSRC); do $(CC) $(CFLAGS) $(FP) -c -o $(B)/cm/$$(echo $$f | sed 's|^\.\./||; s|/|-|g').o $$f || exit 1; done
	rm -f $@ && ar rcs $@ $(B)/cm/*.o

$(B)/selftest: kit/test/selftest.c $(B)/libkit.a $(B)/libcm.a $(ROOT)/crmvec-f16-list.h
	$(CC) $(CFLAGS) $(FP) -fopenmp -Wall -Wextra -I kit -I $(ROOT) -o $@ kit/test/selftest.c $(B)/libkit.a $(B)/libcm.a -lmpfr -lgmp -lm

check: all $(B)/gen-tables
	@set -e; v() { echo "$$1" | tee -a $(B)/check.log | tail -1; echo "$$1" | tail -1 | grep -qE $(VERDICTS) || { echo "FAILED: $$2"; exit 1; }; }; \
	: > $(B)/check.log; \
	v "$$($(B)/selftest)" "kit selftest"; \
	v "$$(tools/test/run-tests $(B)/tools-test)" "repro-scan tests"; \
	$(B)/gen-tables > $(B)/lowp-tables.h; \
	if cmp -s $(B)/lowp-tables.h lowp/lowp-tables.h; then r="VERDICT: IDENTICAL: lowp-tables.h is what gen-tables makes"; \
	else r="VERDICT: DIFFERS: lowp-tables.h is not what gen-tables makes (make lowp-tables)"; fi; v "$$r" "lowp tables"; \
	v "$$($(B)/lowp-check)" "lowp check"; \
	v "$$($(B)/mx-check)" "lowp MX check"; \
	v "$$($(B)/ival-check)" "ival check"; \
	echo "make check: every verdict passed (details in $(B)/check.log)"

$(B)/lowp/lowp-all.o: lowp/lowp.c lowp/mx.c $(LOWPH) $(LOWPCM) Makefile
	rm -rf $(B)/lowp && mkdir -p $(B)/lowp
	$(CC) $(CFLAGS) $(FP) -fPIC -Wall -Wextra -c -o $(B)/lowp/lowp.o lowp/lowp.c
	$(CC) $(CFLAGS) $(FP) -fPIC -Wall -Wextra -c -o $(B)/lowp/mx.o lowp/mx.c
	for f in $(LOWPCM); do $(CC) $(CFLAGS) $(FP) -fPIC -fvisibility=hidden -c -o $(B)/lowp/cm-$$(basename $$f .c).o $$f || exit 1; done
	$(CC) -r -nostdlib -o $@ $(B)/lowp/lowp.o $(B)/lowp/mx.o $(B)/lowp/cm-*.o
	objcopy --localize-hidden $@

$(B)/liblowp.a: $(B)/lowp/lowp-all.o
	rm -f $@ && ar rcs $@ $<

$(B)/liblowp.so: $(B)/lowp/lowp-all.o
	$(CC) -shared -Wl,-soname,liblowp.so -Wl,-z,defs -o $@ $< -lm

$(B)/gen-tables: lowp/gen-tables.c lowp/lowp-list.h $(B)/libkit.a
	$(CC) $(CFLAGS) $(FP) -fopenmp -Wall -Wextra -I kit -I lowp -o $@ lowp/gen-tables.c $(B)/libkit.a -lmpfr -lgmp -lm

# regenerates the committed tables (make check says when they are stale)
lowp-tables: $(B)/gen-tables
	$(B)/gen-tables > lowp/lowp-tables.h

$(B)/lowp-check: lowp/test/check.c $(LOWPH) $(B)/liblowp.a $(B)/libkit.a $(B)/libcm.a
	$(CC) $(CFLAGS) $(FP) -fopenmp -Wall -Wextra -I kit -I lowp -o $@ lowp/test/check.c $(B)/liblowp.a $(B)/libkit.a $(B)/libcm.a -lmpfr -lgmp -lm

$(B)/mx-check: lowp/test/mx-check.c $(LOWPH) $(B)/liblowp.a $(B)/libkit.a
	$(CC) $(CFLAGS) $(FP) -fopenmp -Wall -Wextra -I kit -I lowp -o $@ lowp/test/mx-check.c $(B)/liblowp.a $(B)/libkit.a -lmpfr -lgmp -lm

# ival (ival/ival.h): interval functions, with their own local copy of
# CORE-MATH's binary64 functions
IVALCM  := $(filter-out $(addprefix $(ROOT)/,pow/pow.c atan2pi/atan2pi.c lgamma.c tgamma.c),$(LOWPCM))
IVALH   := ival/ival.h ival/ival-list.h
$(B)/ival/ival-all.o: ival/ival.c $(IVALH) $(IVALCM) Makefile
	rm -rf $(B)/ival && mkdir -p $(B)/ival
	$(CC) $(CFLAGS) $(FP) -fPIC -Wall -Wextra -c -o $(B)/ival/ival.o ival/ival.c
	for f in $(IVALCM); do $(CC) $(CFLAGS) $(FP) -fPIC -fvisibility=hidden -c -o $(B)/ival/cm-$$(basename $$f .c).o $$f || exit 1; done
	$(CC) -r -nostdlib -o $@ $(B)/ival/ival.o $(B)/ival/cm-*.o
	objcopy --localize-hidden $@

$(B)/libival.a: $(B)/ival/ival-all.o
	rm -f $@ && ar rcs $@ $<

$(B)/libival.so: $(B)/ival/ival-all.o
	$(CC) -shared -Wl,-soname,libival.so -Wl,-z,defs -o $@ $< -lm

$(B)/ival-check: ival/test/check.c $(IVALH) $(B)/libival.a $(B)/libkit.a
	$(CC) $(CFLAGS) $(FP) -fopenmp -Wall -Wextra -I kit -I ival -o $@ ival/test/check.c $(B)/libival.a $(B)/libkit.a -lmpfr -lgmp -lm

clean:
	rm -rf $(B)

.PHONY: all check clean lowp-tables
