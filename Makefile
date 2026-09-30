# numerics: the groundwork for reproducible numerics libraries (README.md).
#   make         the kit and its self-test
#   make check   every check here; each ends in a verdict line, and one
#                that is not IDENTICAL fails the target
# gcc 13 or later (_Float16, __bf16), MPFR 4.2 or later, OpenMP.
CC      ?= gcc
CFLAGS  ?= -O2
# as in crmvec: no contraction, and no constant folding that assumes
# round-to-nearest, so that C code rounds as written in every mode
FP      := -ffp-contract=off -frounding-math
ROOT    := ..
B       := build
KIT     := kit/fmt.c kit/run.c
# CORE-MATH's correctly rounded functions (vendored at the top), the
# answers the kit's self-test checks it against
CMSRC   := $(wildcard $(ROOT)/f16/*.c) $(wildcard $(ROOT)/bf16/*.c) \
           $(addprefix $(ROOT)/,expf.c logf.c sinf.c atan2f.c exp.c log/log.c sin.c tgamma.c erfc.c \
             pow/pow.c atan2/atan2.c hypot.c)
VERDICTS := '^VERDICT: IDENTICAL'

all: $(B)/selftest

$(B)/libkit.a: $(KIT) kit/kit.h
	mkdir -p $(B)/kit
	for f in $(KIT); do $(CC) $(CFLAGS) $(FP) -fopenmp -Wall -Wextra -c -o $(B)/kit/$$(basename $$f .c).o $$f || exit 1; done
	rm -f $@ && ar rcs $@ $(B)/kit/*.o

$(B)/libcm.a: $(CMSRC)
	rm -rf $(B)/cm && mkdir -p $(B)/cm
	for f in $(CMSRC); do $(CC) $(CFLAGS) $(FP) -c -o $(B)/cm/$$(echo $$f | sed 's|^\.\./||; s|/|-|g').o $$f || exit 1; done
	rm -f $@ && ar rcs $@ $(B)/cm/*.o

$(B)/selftest: kit/test/selftest.c $(B)/libkit.a $(B)/libcm.a $(ROOT)/crmvec-f16-list.h
	$(CC) $(CFLAGS) $(FP) -fopenmp -Wall -Wextra -I kit -I $(ROOT) -o $@ kit/test/selftest.c $(B)/libkit.a $(B)/libcm.a -lmpfr -lgmp -lm

check: all
	@set -e; v() { echo "$$1" | tee -a $(B)/check.log | tail -1; echo "$$1" | tail -1 | grep -qE $(VERDICTS) || { echo "FAILED: $$2"; exit 1; }; }; \
	: > $(B)/check.log; \
	v "$$($(B)/selftest)" "kit selftest"; \
	echo "make check: every verdict passed (details in $(B)/check.log)"

clean:
	rm -rf $(B)

.PHONY: all check clean
