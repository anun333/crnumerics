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
KIT     := kit/fmt.c kit/run.c kit/fns.c kit/mx.c kit/sum.c
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
     $(B)/ival-check $(B)/ival-arith-check $(B)/ival-1788-check $(B)/ival-rev-check $(B)/ival-text-check $(B)/ival-acc-check $(B)/ival-thread-check $(B)/ival-install-check $(B)/libcrsum.a $(B)/libcrsum.so $(B)/crsum-check $(B)/crsum-check-settle \
     $(B)/libcrnn.a $(B)/libcrnn.so $(B)/crnn-check $(B)/libcrblas.so

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
	v "$$($(B)/ival-arith-check)" "ival arithmetic check"; \
	v "$$($(B)/ival-1788-check)" "ival 1788 operations check"; \
	v "$$($(B)/ival-rev-check)" "ival reverse operations check"; \
	v "$$($(B)/ival-text-check)" "ival constructors check"; \
	v "$$(env -u IVAL_CRMVEC $(B)/ival-acc-check | tail -1)" "ival accurate mode check (without crmvec)"; \
	v "$$($(B)/ival-thread-check)" "ival from eight threads"; \
	v "$$(LD_LIBRARY_PATH=$(B)/stage/usr/lib $(B)/ival-install-check $(B)/stage/usr/lib/libival.so)" "ival installed, used through pkg-config"; \
	v "$$($(B)/crsum-check)" "crsum check"; \
	v "$$($(B)/crsum-check-settle)" "crsum check, carries settled every 3 terms"; \
	v "$$($(B)/crnn-check)" "crnn check"; \
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
IVALCM  := $(filter-out $(addprefix $(ROOT)/,atan2pi/atan2pi.c lgamma.c),$(LOWPCM))
IVALH   := ival/ival.h ival/ival-list.h ival/tgamma-table.h ival/ival-eft.h
$(B)/ival/ival-all.o: ival/ival.c ival/ival-arith.c ival/ival-1788.c ival/ival-rev.c ival/ival-text.c $(IVALH) $(IVALCM) Makefile
	rm -rf $(B)/ival && mkdir -p $(B)/ival
	$(CC) $(CFLAGS) $(FP) -fPIC -Wall -Wextra -c -o $(B)/ival/ival.o ival/ival.c
	$(CC) $(CFLAGS) $(FP) -fPIC -Wall -Wextra -c -o $(B)/ival/arith.o ival/ival-arith.c
	$(CC) $(CFLAGS) $(FP) -fPIC -Wall -Wextra -c -o $(B)/ival/i1788.o ival/ival-1788.c
	$(CC) $(CFLAGS) $(FP) -fPIC -Wall -Wextra -c -o $(B)/ival/rev.o ival/ival-rev.c
	$(CC) $(CFLAGS) $(FP) -fPIC -Wall -Wextra -c -o $(B)/ival/text.o ival/ival-text.c
	for f in $(IVALCM); do $(CC) $(CFLAGS) $(FP) -fPIC -fvisibility=hidden -c -o $(B)/ival/cm-$$(basename $$f .c).o $$f || exit 1; done
	$(CC) -r -nostdlib -o $@ $(B)/ival/ival.o $(B)/ival/arith.o $(B)/ival/i1788.o $(B)/ival/rev.o $(B)/ival/text.o $(B)/ival/cm-*.o
	objcopy --localize-hidden $@

$(B)/libival.a: $(B)/ival/ival-all.o
	rm -f $@ && ar rcs $@ $<

# the version is IVAL_VERSION in ival.h; the soname changes with its first number
IVAL_VERSION := $(shell sed -n 's/^\#define IVAL_VERSION "\(.*\)"/\1/p' ival/ival.h)
IVAL_MAJOR   := $(firstword $(subst ., ,$(IVAL_VERSION)))
$(B)/libival.so: $(B)/ival/ival-all.o
	$(CC) -shared -Wl,-soname,libival.so.$(IVAL_MAJOR) -Wl,-z,defs -o $@ $< -ldl -lm

# make install (ival only so far): libival.so.<version> with its links, libival.a, ival.h and ival-list.h, and
# ival.pc for pkg-config. PREFIX, LIBDIR (lib64 or a multiarch one), INCLUDEDIR and DESTDIR as usual
PREFIX       ?= /usr/local
LIBDIR       ?= $(PREFIX)/lib
INCLUDEDIR   ?= $(PREFIX)/include
PKGCONFIGDIR ?= $(LIBDIR)/pkgconfig
install: install-ival
uninstall: uninstall-ival
install-ival: $(B)/libival.so $(B)/libival.a
	@test -n "$(IVAL_VERSION)" || { echo "install: no IVAL_VERSION in ival/ival.h"; exit 1; }
	install -d $(DESTDIR)$(LIBDIR) $(DESTDIR)$(INCLUDEDIR) $(DESTDIR)$(PKGCONFIGDIR)
	install -m 755 $(B)/libival.so $(DESTDIR)$(LIBDIR)/libival.so.$(IVAL_VERSION)
	ln -sf libival.so.$(IVAL_VERSION) $(DESTDIR)$(LIBDIR)/libival.so.$(IVAL_MAJOR)
	ln -sf libival.so.$(IVAL_MAJOR) $(DESTDIR)$(LIBDIR)/libival.so
	install -m 644 $(B)/libival.a $(DESTDIR)$(LIBDIR)/libival.a
	install -m 644 ival/ival.h ival/ival-list.h $(DESTDIR)$(INCLUDEDIR)/
	sed -e 's|@PREFIX@|$(PREFIX)|' -e 's|@LIBDIR@|$(LIBDIR)|' -e 's|@INCLUDEDIR@|$(INCLUDEDIR)|' \
	  -e 's|@VERSION@|$(IVAL_VERSION)|' ival/ival.pc.in > $(DESTDIR)$(PKGCONFIGDIR)/ival.pc
uninstall-ival:
	rm -f $(DESTDIR)$(LIBDIR)/libival.so.$(IVAL_VERSION) $(DESTDIR)$(LIBDIR)/libival.so.$(IVAL_MAJOR) \
	  $(DESTDIR)$(LIBDIR)/libival.so $(DESTDIR)$(LIBDIR)/libival.a $(DESTDIR)$(INCLUDEDIR)/ival.h \
	  $(DESTDIR)$(INCLUDEDIR)/ival-list.h $(DESTDIR)$(PKGCONFIGDIR)/ival.pc
# in make check: install into $(B)/stage, then build and run programs against it the way a user would, through
# pkg-config: shared, static, and C++; the library exports ival_ names only, under the versioned soname
$(B)/ival-install-check: ival/test/install-check.c ival/ival.pc.in $(B)/libival.so $(B)/libival.a
	rm -rf $(B)/stage
	$(MAKE) --no-print-directory install-ival DESTDIR=$(CURDIR)/$(B)/stage PREFIX=/usr > /dev/null
	PKG_CONFIG_PATH= PKG_CONFIG_LIBDIR=$(B)/stage/usr/lib/pkgconfig PKG_CONFIG_SYSROOT_DIR=$(CURDIR)/$(B)/stage \
	  sh -c '$(CC) -O2 -o $@ ival/test/install-check.c $$(pkg-config --cflags --libs ival) && \
	  $(CC) -O2 -o $@-static ival/test/install-check.c $$(pkg-config --cflags ival) \
	    -Wl,-Bstatic $$(pkg-config --libs-only-L ival) -lival -Wl,-Bdynamic $$(pkg-config --static --libs-only-l ival | sed "s/-lival//") && \
	  printf "#include <ival.h>\nint main(void) { return ival_version()[0] == 0; }\n" | \
	    $(CXX) -x c++ -o $@-cxx - $$(pkg-config --cflags --libs ival)'
ival-install-check: $(B)/ival-install-check
	@LD_LIBRARY_PATH=$(B)/stage/usr/lib $(B)/ival-install-check $(B)/stage/usr/lib/libival.so

$(B)/ival-check: ival/test/check.c $(IVALH) $(B)/libival.a $(B)/libkit.a
	$(CC) $(CFLAGS) $(FP) -fopenmp -Wall -Wextra -I kit -I ival -o $@ ival/test/check.c $(B)/libival.a $(B)/libkit.a -lmpfr -lgmp -ldl -lm

$(B)/ival-arith-check: ival/test/arith-check.c $(IVALH) $(B)/libival.a
	$(CC) $(CFLAGS) $(FP) -Wall -Wextra -I ival -o $@ ival/test/arith-check.c $(B)/libival.a -lmpfr -lgmp -ldl -lm
# ival's checks under AddressSanitizer and UndefinedBehaviorSanitizer, any report fatal (B=build-asan): make
# ival-sanitize. The checks' own arrays are left to the end of the process, so leaks are not reported
IVAL_CHECKS := ival-check ival-arith-check ival-1788-check ival-rev-check ival-text-check ival-acc-check ival-thread-check
ival-sanitize:
	$(MAKE) B=build-asan CFLAGS="-O1 -g -fsanitize=address,undefined -fno-sanitize-recover=all" $(addprefix build-asan/,$(IVAL_CHECKS))
	@for c in $(IVAL_CHECKS); do r=$$(ASAN_OPTIONS=detect_leaks=0 build-asan/$$c 2>&1 | tail -1); echo "$$c: $$r"; \
	  echo "$$r" | grep -q '^VERDICT: IDENTICAL' || exit 1; done
# ival from eight threads at once: in make check; make ival-thread-tsan runs it under ThreadSanitizer (B=build-tsan)
$(B)/ival-thread-check: ival/test/thread-check.c $(IVALH) $(B)/libival.a
	$(CC) $(CFLAGS) $(FP) -Wall -Wextra -pthread -I ival -o $@ ival/test/thread-check.c $(B)/libival.a -ldl -lm
# (setarch -R: ThreadSanitizer cannot map its shadow memory under the address randomisation of recent kernels)
NORAND := $(shell setarch $$(uname -m) -R true 2>/dev/null && echo setarch $$(uname -m) -R)
ival-thread-tsan:
	$(MAKE) B=build-tsan CFLAGS="-O1 -g -fsanitize=thread" build-tsan/ival-thread-check
	TSAN_OPTIONS=halt_on_error=1 $(NORAND) build-tsan/ival-thread-check
	test -z "$(CRMVEC)" || TSAN_OPTIONS=halt_on_error=1 IVAL_CRMVEC=$(CRMVEC) $(NORAND) build-tsan/ival-thread-check
# the accurate mode: in make check without crmvec (it must equal the tight mode); make ival-acc-check
# CRMVEC=/path/to/crmvec's libmvec.so.1 runs it through crmvec's vector functions
$(B)/ival-acc-check: ival/test/acc-check.c $(IVALH) $(B)/libival.a
	$(CC) $(CFLAGS) $(FP) -Wall -Wextra -I ival -o $@ ival/test/acc-check.c $(B)/libival.a -ldl -lm
ival-acc-check: $(B)/ival-acc-check
	@test -n "$(CRMVEC)" || { echo "ival-acc-check: name crmvec's library: CRMVEC=/path/to/libmvec.so.1"; exit 2; }
	IVAL_CRMVEC=$(CRMVEC) $(B)/ival-acc-check
$(B)/ival-text-check: ival/test/text-check.c $(IVALH) $(B)/libival.a
	$(CC) $(CFLAGS) $(FP) -Wall -Wextra -I ival -o $@ ival/test/text-check.c $(B)/libival.a -lmpfr -lgmp -ldl -lm
$(B)/ival-rev-check: ival/test/rev-check.c $(IVALH) $(B)/libival.a
	$(CC) $(CFLAGS) $(FP) -Wall -Wextra -I ival -o $@ ival/test/rev-check.c $(B)/libival.a -lmpfr -lgmp -ldl -lm
$(B)/ival-1788-check: ival/test/1788-check.c $(IVALH) $(B)/libival.a
	$(CC) $(CFLAGS) $(FP) -Wall -Wextra -I ival -o $@ ival/test/1788-check.c $(B)/libival.a -lmpfr -lgmp -ldl -lm
# ival against other interval libraries (ival/test/compare.cpp), each where it is given: make ival-compare
# COMPARE="-DHAVE_MPFI -DHAVE_BOOST -DHAVE_FILIB -DHAVE_P1788" COMPARE_INC="-I..." COMPARE_LIBS="... -lmpfi -lmpfr
# -lgmp" (ival/README.md has the command used); IVAL_CRMVEC for the accurate mode through crmvec. C++14: filib++ has
# dynamic exception specifications
CXX ?= g++
ival-compare: ival/test/compare.cpp $(IVALH) $(B)/libival.a
	$(CXX) -O2 -std=c++14 -frounding-math -w $(COMPARE) -I ival $(COMPARE_INC) -o $(B)/ival-compare ival/test/compare.cpp \
	  $(B)/libival.a $(COMPARE_LIBS) -ldl -lm
	$(B)/ival-compare
# the functions' cost: make $(B)/ival-fn-bench
$(B)/ival-fn-bench: ival/test/fn-bench.c $(IVALH) $(B)/libival.a
	$(CC) $(CFLAGS) -Wall -Wextra -I ival -o $@ ival/test/fn-bench.c $(B)/libival.a -ldl -lm
# the arithmetic's cost against the alternatives: make $(B)/ival-arith-bench
$(B)/ival-arith-bench: ival/test/arith-bench.c $(IVALH) $(B)/libival.a
	$(CC) $(CFLAGS) $(FP) -Wall -Wextra -I ival -o $@ ival/test/arith-bench.c $(B)/libival.a -ldl -lm

# crsum (crsum/crsum.h): correctly rounded sums and dot products. The check
# runs twice: as built, and with the carries settled every 3 terms (the
# default, 2^29, is never reached by a test)
CRSUMH  := crsum/crsum.h
$(B)/crsum/crsum.o: crsum/crsum.c $(CRSUMH)
	mkdir -p $(B)/crsum
	$(CC) $(CFLAGS) $(FP) -fPIC -Wall -Wextra -c -o $@ crsum/crsum.c
$(B)/libcrsum.a: $(B)/crsum/crsum.o
	rm -f $@ && ar rcs $@ $<
$(B)/libcrsum.so: $(B)/crsum/crsum.o
	$(CC) -shared -Wl,-soname,libcrsum.so -Wl,-z,defs -o $@ $< -lm
# crblas (crsum/crblas.c): dgemm_ and dgemm_64_ through crgemm_oz, for a
# BLAS switchboard such as Julia's libblastrampoline (crsum/julia/crblas.jl);
# crsum inside it hidden, so it loads beside anything
$(B)/libcrblas.so: crsum/crblas.c crsum/crsum.c $(CRSUMH)
	mkdir -p $(B)/crblas
	$(CC) $(CFLAGS) $(FP) -fPIC -fvisibility=hidden -Wall -Wextra -c -o $(B)/crblas/crsum.o crsum/crsum.c
	$(CC) $(CFLAGS) $(FP) -fPIC -fvisibility=hidden -Wall -Wextra -c -o $(B)/crblas/crblas.o crsum/crblas.c
	$(CC) -shared -Wl,-soname,libcrblas.so -Wl,-z,defs -o $@ $(B)/crblas/crblas.o $(B)/crblas/crsum.o -ldl -lm
julia-check: $(B)/libcrblas.so
	julia --startup-file=no crsum/julia/crblas.jl $(B)/libcrblas.so
$(B)/crsum-check: crsum/test/check.c $(CRSUMH) $(B)/libcrsum.a $(B)/libkit.a
	$(CC) $(CFLAGS) $(FP) -fopenmp -Wall -Wextra -I kit -I crsum -o $@ crsum/test/check.c $(B)/libcrsum.a $(B)/libkit.a -lmpfr -lgmp -lm -ldl
$(B)/crsum-check-settle: crsum/test/check.c crsum/crsum.c $(CRSUMH) $(B)/libkit.a
	$(CC) $(CFLAGS) $(FP) -fopenmp -Wall -Wextra -DCRSUM_SETTLE_EVERY=3 -I kit -I crsum -o $@ crsum/test/check.c crsum/crsum.c $(B)/libkit.a -lmpfr -lgmp -lm -ldl

# crnn (nn/crnn.h): neural-network primitives, correctly rounded or
# specified bit for bit, with its own local copies of crsum and of the
# CORE-MATH functions it calls
CRNNCM  := $(addprefix $(ROOT)/,exp.c log1p.c erfc.c rsqrt.c)
CRNNH   := nn/crnn.h nn/crnn-fast.h nn/crnn-exceptions.h nn/crnn-round16.h nn/crnn-exceptions16.h $(CRSUMH)
$(B)/crnn/crnn-all.o: nn/crnn.c nn/crnn16.c crsum/crsum.c $(CRNNH) $(CRNNCM) Makefile
	rm -rf $(B)/crnn && mkdir -p $(B)/crnn
	$(CC) $(CFLAGS) $(FP) -fPIC -Wall -Wextra -c -o $(B)/crnn/crnn.o nn/crnn.c
	$(CC) $(CFLAGS) $(FP) -fPIC -Wall -Wextra -c -o $(B)/crnn/crnn16.o nn/crnn16.c
	$(CC) $(CFLAGS) $(FP) -fPIC -fvisibility=hidden -c -o $(B)/crnn/crsum.o crsum/crsum.c
	for f in $(CRNNCM); do $(CC) $(CFLAGS) $(FP) -fPIC -fvisibility=hidden -c -o $(B)/crnn/cm-$$(basename $$f .c).o $$f || exit 1; done
	$(CC) -r -nostdlib -o $@ $(B)/crnn/crnn.o $(B)/crnn/crnn16.o $(B)/crnn/crsum.o $(B)/crnn/cm-*.o
	objcopy --localize-hidden $@
$(B)/libcrnn.a: $(B)/crnn/crnn-all.o
	rm -f $@ && ar rcs $@ $<
$(B)/libcrnn.so: $(B)/crnn/crnn-all.o
	$(CC) -shared -Wl,-soname,libcrnn.so -Wl,-z,defs -o $@ $< -ldl -lm
$(B)/crnn-check: nn/test/check.c nn/crnn-ref.c nn/crnn-ref.h $(CRNNH) $(B)/libcrnn.a $(B)/libkit.a $(B)/libcm.a
	$(CC) $(CFLAGS) $(FP) -fopenmp -Wall -Wextra -I kit -I nn -o $@ nn/test/check.c nn/crnn-ref.c $(B)/libcrnn.a $(B)/libkit.a $(B)/libcm.a -lmpfr -lgmp -ldl -lm
$(B)/gen-exceptions: nn/gen-exceptions.c nn/crnn-ref.c nn/crnn-ref.h nn/crnn-fast.h $(B)/libkit.a $(B)/libcm.a
	$(CC) $(CFLAGS) $(FP) -fopenmp -Wall -Wextra -I kit -I nn -o $@ nn/gen-exceptions.c nn/crnn-ref.c $(B)/libkit.a $(B)/libcm.a -lmpfr -lgmp -lm
$(B)/gen-exceptions16: nn/gen-exceptions16.c nn/crnn-ref.c nn/crnn-ref.h nn/crnn-fast.h nn/crnn-round16.h $(B)/libkit.a $(B)/libcm.a
	$(CC) $(CFLAGS) $(FP) -fopenmp -Wall -Wextra -I kit -I nn -o $@ nn/gen-exceptions16.c nn/crnn-ref.c $(B)/libkit.a $(B)/libcm.a -lmpfr -lgmp -lm
$(B)/crnn-bench: nn/test/bench.c nn/crnn.h $(B)/libcrnn.a
	$(CC) $(CFLAGS) -Wall -Wextra -I nn -o $@ nn/test/bench.c $(B)/libcrnn.a -ldl -lm
# crnn's vector path through crmvec (nn/crnn.c): make crnn-vsame
# CRMVEC=/path/to/crmvec's libmvec.so.1 hashes every input's result on both
# paths, which must agree
$(B)/crnn-vsame: nn/test/vsame.c nn/crnn.h $(B)/libcrnn.a
	$(CC) $(CFLAGS) $(FP) -fopenmp -Wall -Wextra -I nn -o $@ nn/test/vsame.c $(B)/libcrnn.a -ldl -lm
crnn-vsame: $(B)/crnn-vsame
	@test -n "$(CRMVEC)" || { echo "crnn-vsame: name crmvec's library: CRMVEC=/path/to/libmvec.so.1"; exit 2; }
	@a=$$($(B)/crnn-vsame); b=$$(CRNN_CRMVEC=$(CRMVEC) $(B)/crnn-vsame); echo "$$a"; echo "$$b"; \
	echo "$$b" | head -1 | grep -q crmvec || { echo "VOID: the vector path did not load ($(CRMVEC))"; exit 2; }; \
	if [ "$$(echo "$$a" | tail -n +2)" = "$$(echo "$$b" | tail -n +2)" ]; then \
	  echo "VERDICT: IDENTICAL: the vector path gives the scalar path's bits on all 2^32 inputs of each function"; \
	else echo "VERDICT: DIFFERS"; exit 1; fi
# IEEE 1788's test suite, ITF1788, on ival (bare intervals): make itf1788-check
# ITF1788=/path/to/a clone of ITF1788 (on GitHub; b6ee1e2 checked). The
# converter reads its itl files there; none is copied into this repository
ITF1788_ITL ?= $(ITF1788)/itl
itf1788-check: ival/test/itf1788.py $(B)/libival.a $(B)/libcrsum.a
	@test -d "$(ITF1788_ITL)" || { echo "itf1788-check: name a clone of ITF1788 (on GitHub): ITF1788=/path, or its itl folder: ITF1788_ITL=/path"; exit 2; }
	python3 ival/test/itf1788.py $(ITF1788_ITL) > $(B)/itf1788-check.c
	$(CC) $(CFLAGS) $(FP) -Wall -I ival -I crsum -o $(B)/itf1788-check $(B)/itf1788-check.c $(B)/libival.a $(B)/libcrsum.a -ldl -lm
	$(B)/itf1788-check
# regenerates the committed table: every 2^32 input of five functions
# (minutes on a few cores)
crnn-exceptions: $(B)/gen-exceptions
	$(B)/gen-exceptions > $(B)/crnn-exceptions.h && mv $(B)/crnn-exceptions.h nn/crnn-exceptions.h
# every 2^32 input of each one-argument function against MPFR (hours of CPU)
crnn-exceptions16: $(B)/gen-exceptions16
	$(B)/gen-exceptions16 > $(B)/crnn-exceptions16.h && mv $(B)/crnn-exceptions16.h nn/crnn-exceptions16.h
crnn-check-all: $(B)/crnn-check
	$(B)/crnn-check all

clean:
	rm -rf $(B)

.PHONY: all check clean install uninstall install-ival uninstall-ival ival-install-check lowp-tables crnn-exceptions crnn-exceptions16 crnn-check-all julia-check crnn-vsame itf1788-check ival-acc-check ival-compare ival-thread-tsan ival-sanitize
