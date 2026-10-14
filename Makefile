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
     $(B)/ival-check $(B)/ival-arith-check $(B)/ival-1788-check $(B)/ival-rev-check $(B)/ival-text-check $(B)/ival-acc-check $(B)/ival-thread-check $(B)/ival-env-check $(B)/ival-mode-change-check $(B)/ival-install-check $(B)/libcrsum.a $(B)/libcrsum.so $(B)/crsum-check $(B)/crsum-check-settle \
     $(B)/libcrnn.a $(B)/libcrnn.so $(B)/crnn-check $(B)/libcrblas.so \
     $(B)/libcrdist.a $(B)/libcrdist.so $(B)/crdist-check $(B)/libcrlin.a $(B)/libcrlin.so $(B)/crlin-check \
     $(B)/libcrfix.a $(B)/libcrfix.so $(B)/crfix-check $(B)/crfix-check-portable $(B)/crfix-gen-tables

$(B)/libkit.a: $(KIT) kit/kit.h
	mkdir -p $(B)/kit
	for f in $(KIT); do $(CC) $(CFLAGS) $(FP) $(OPENMP) -Wall -Wextra -c -o $(B)/kit/$$(basename $$f .c).o $$f || exit 1; done
	rm -f $@ && ar rcs $@ $(B)/kit/*.o

# the bfloat16 functions get gcc13-bf16-fix.h: GCC 13 with AArch64's fp16 miscompiles __bf16 constants
$(B)/libcm.a: $(CMSRC) $(ROOT)/gcc13-bf16-fix.h Makefile
	rm -rf $(B)/cm && mkdir -p $(B)/cm
	for f in $(CMSRC); do x=; case $$f in $(ROOT)/bf16/*) x="-include $(ROOT)/gcc13-bf16-fix.h";; esac; \
	  $(CC) $(CFLAGS) $(FP) $$x -c -o $(B)/cm/$$(echo $$f | sed 's|^\.\./||; s|/|-|g').o $$f || exit 1; done
	rm -f $@ && ar rcs $@ $(B)/cm/*.o

$(B)/selftest: kit/test/selftest.c $(B)/libkit.a $(B)/libcm.a $(ROOT)/crmvec-f16-list.h
	$(CC) $(CFLAGS) $(FP) -fopenmp -Wall -Wextra -I kit -I $(ROOT) -o $@ kit/test/selftest.c $(B)/libkit.a $(B)/libcm.a -lmpfr -lgmp -lm

check: all $(B)/gen-tables $(B)/crfix-gen-tables
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
	v "$$($(B)/ival-env-check)" "ival leaves the caller's floating-point state"; \
	v "$$($(B)/ival-mode-change-check)" "ival-scalar.h across changes of rounding mode"; \
	v "$$(LD_LIBRARY_PATH=$(B)/stage/usr/lib $(B)/ival-install-check $(B)/stage/usr/lib/libival.so)" "ival installed, used through pkg-config"; \
	v "$$($(B)/crsum-check)" "crsum check"; \
	v "$$($(B)/crsum-check-settle)" "crsum check, carries settled every 3 terms"; \
	v "$$($(B)/crnn-check)" "crnn check"; \
	v "$$($(B)/crdist-check)" "crdist check"; \
	v "$$($(B)/crlin-check)" "crlin check"; \
	v "$$($(MAKE) --no-print-directory crlin-install-check | tail -1)" "crlin installed, used through pkg-config"; \
	v "$$($(MAKE) --no-print-directory crdist-install-check | tail -1)" "crdist installed, used through pkg-config"; \
	$(B)/crfix-gen-tables > $(B)/crfix-tables.h; \
	if cmp -s $(B)/crfix-tables.h fix/crfix-tables.h; then r="VERDICT: IDENTICAL: crfix-tables.h is what crfix-gen-tables makes"; \
	else r="VERDICT: DIFFERS: crfix-tables.h is not what crfix-gen-tables makes (make crfix-tables)"; fi; v "$$r" "crfix tables"; \
	v "$$($(B)/crfix-check)" "crfix check"; \
	v "$$($(B)/crfix-check-portable)" "crfix check, the portable 128-bit code"; \
	v "$$($(MAKE) --no-print-directory crfix-install-check | tail -1)" "crfix installed, used through pkg-config"; \
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
IVALH   := ival/ival.h ival/ival-list.h ival/tgamma-table.h ival/ival-eft.h ival/ival-powdd.inc ival/ival-scalar.h
# on x86-64 ival's CORE-MATH objects get their fenv calls renamed to MXCSR versions (ival/ival-fenv.c)
IVAL_TARGET := $(shell $(CC) -dumpmachine)
# macOS (Apple clang): its ld -r turns hidden symbols into local ones without objcopy; the shared library is a dylib;
# no OpenMP, so the one check that uses it runs on one thread
IVAL_DARWIN := $(findstring darwin,$(IVAL_TARGET))
IVAL_SHLIB  := $(B)/libival.$(if $(IVAL_DARWIN),dylib,so)
OPENMP      := $(if $(IVAL_DARWIN),,-fopenmp)
$(B)/ival/ival-all.o: ival/ival.c ival/ival-arith.c ival/ival-1788.c ival/ival-rev.c ival/ival-text.c ival/ival-fenv.c ival/ival-fenv.syms $(IVALH) $(IVALCM) Makefile
	rm -rf $(B)/ival && mkdir -p $(B)/ival
	$(CC) $(CFLAGS) $(FP) -fPIC -Wall -Wextra -c -o $(B)/ival/ival.o ival/ival.c
	$(CC) $(CFLAGS) $(FP) -fPIC -Wall -Wextra -c -o $(B)/ival/arith.o ival/ival-arith.c
	$(CC) $(CFLAGS) $(FP) -fPIC -Wall -Wextra -c -o $(B)/ival/i1788.o ival/ival-1788.c
	$(CC) $(CFLAGS) $(FP) -fPIC -Wall -Wextra -c -o $(B)/ival/rev.o ival/ival-rev.c
	$(CC) $(CFLAGS) $(FP) -fPIC -Wall -Wextra -c -o $(B)/ival/text.o ival/ival-text.c
	$(CC) $(CFLAGS) $(FP) -fPIC -Wall -Wextra -c -o $(B)/ival/fenv.o ival/ival-fenv.c
	for f in $(IVALCM); do $(CC) $(CFLAGS) $(FP) -fPIC -fvisibility=hidden -c -o $(B)/ival/cm-$$(basename $$f .c).o $$f || exit 1; done
	$(if $(IVAL_DARWIN),,$(if $(findstring x86_64,$(IVAL_TARGET)),for o in $(B)/ival/cm-*.o; do objcopy --redefine-syms=ival/ival-fenv.syms $$o || exit 1; done))
	$(CC) -r -nostdlib -o $@ $(B)/ival/ival.o $(B)/ival/arith.o $(B)/ival/i1788.o $(B)/ival/rev.o $(B)/ival/text.o $(B)/ival/fenv.o $(B)/ival/cm-*.o
	$(if $(IVAL_DARWIN),,objcopy --localize-hidden $@)

$(B)/libival.a: $(B)/ival/ival-all.o
	rm -f $@ && ar rcs $@ $<

# the version is IVAL_VERSION in ival.h; the soname changes with its first number
IVAL_VERSION := $(shell sed -n 's/^\#define IVAL_VERSION "\(.*\)"/\1/p' ival/ival.h)
IVAL_MAJOR   := $(firstword $(subst ., ,$(IVAL_VERSION)))
$(B)/libival.so: $(B)/ival/ival-all.o
	$(CC) -shared -Wl,-soname,libival.so.$(IVAL_MAJOR) -Wl,-z,defs -o $@ $< -ldl -lm
# macOS: the same library as a dylib, its install name versioned as the soname is
$(B)/libival.dylib: $(B)/ival/ival-all.o
	$(CC) -dynamiclib -install_name @rpath/libival.$(IVAL_MAJOR).dylib -o $@ $< -lm   # undefined symbols are errors by default

# make install (ival only so far): libival.so.<version> with its links, libival.a, ival.h, ival-list.h and ival-scalar.h, and
# ival.pc for pkg-config. PREFIX, LIBDIR (lib64 or a multiarch one), INCLUDEDIR and DESTDIR as usual
PREFIX       ?= /usr/local
LIBDIR       ?= $(PREFIX)/lib
INCLUDEDIR   ?= $(PREFIX)/include
PKGCONFIGDIR ?= $(LIBDIR)/pkgconfig
install: install-ival $(if $(IVAL_DARWIN),,install-crdist install-crlin install-crfix)
uninstall: uninstall-ival $(if $(IVAL_DARWIN),,uninstall-crdist uninstall-crlin uninstall-crfix)
# on macOS libival.<version>.dylib with the links libival.<first number>.dylib and libival.dylib, relinked here from
# the same object so that its install name is LIBDIR's (as Homebrew's libraries name themselves), not changed after
install-ival: $(IVAL_SHLIB) $(B)/libival.a
	@test -n "$(IVAL_VERSION)" || { echo "install: no IVAL_VERSION in ival/ival.h"; exit 1; }
	install -d $(DESTDIR)$(LIBDIR) $(DESTDIR)$(INCLUDEDIR) $(DESTDIR)$(PKGCONFIGDIR)
ifneq ($(IVAL_DARWIN),)
	$(CC) -dynamiclib -install_name $(LIBDIR)/libival.$(IVAL_MAJOR).dylib -o $(DESTDIR)$(LIBDIR)/libival.$(IVAL_VERSION).dylib $(B)/ival/ival-all.o -lm
	ln -sf libival.$(IVAL_VERSION).dylib $(DESTDIR)$(LIBDIR)/libival.$(IVAL_MAJOR).dylib
	ln -sf libival.$(IVAL_MAJOR).dylib $(DESTDIR)$(LIBDIR)/libival.dylib
else
	install -m 755 $(B)/libival.so $(DESTDIR)$(LIBDIR)/libival.so.$(IVAL_VERSION)
	ln -sf libival.so.$(IVAL_VERSION) $(DESTDIR)$(LIBDIR)/libival.so.$(IVAL_MAJOR)
	ln -sf libival.so.$(IVAL_MAJOR) $(DESTDIR)$(LIBDIR)/libival.so
endif
	install -m 644 $(B)/libival.a $(DESTDIR)$(LIBDIR)/libival.a
	install -m 644 ival/ival.h ival/ival-list.h ival/ival-scalar.h $(DESTDIR)$(INCLUDEDIR)/
	sed -e 's|@PREFIX@|$(PREFIX)|' -e 's|@LIBDIR@|$(LIBDIR)|' -e 's|@INCLUDEDIR@|$(INCLUDEDIR)|' \
	  -e 's|@VERSION@|$(IVAL_VERSION)|' ival/ival.pc.in > $(DESTDIR)$(PKGCONFIGDIR)/ival.pc
uninstall-ival:
	rm -f $(DESTDIR)$(LIBDIR)/libival.so.$(IVAL_VERSION) $(DESTDIR)$(LIBDIR)/libival.so.$(IVAL_MAJOR) \
	  $(DESTDIR)$(LIBDIR)/libival.$(IVAL_VERSION).dylib $(DESTDIR)$(LIBDIR)/libival.$(IVAL_MAJOR).dylib $(DESTDIR)$(LIBDIR)/libival.dylib \
	  $(DESTDIR)$(LIBDIR)/libival.so $(DESTDIR)$(LIBDIR)/libival.a $(DESTDIR)$(INCLUDEDIR)/ival.h \
	  $(DESTDIR)$(INCLUDEDIR)/ival-list.h $(DESTDIR)$(INCLUDEDIR)/ival-scalar.h $(DESTDIR)$(PKGCONFIGDIR)/ival.pc
# in make check: install into $(B)/stage, then build and run programs against it the way a user would, through
# pkg-config: shared, static, and C++; the library exports ival_ names only, under the versioned soname
# (macOS: the static build names libival.a itself, ld64 having no -Bstatic; the installed library is found by
# DYLD_LIBRARY_PATH, its install name being /usr/lib's)
IVAL_STATIC_LINK := $(if $(IVAL_DARWIN),$$(pkg-config --libs-only-L ival | sed "s/^ *-L//; s/ *$$//")/libival.a,-Wl,-Bstatic $$(pkg-config --libs-only-L ival) -lival -Wl,-Bdynamic)
IVAL_LIBPATH_VAR := $(if $(IVAL_DARWIN),DYLD_LIBRARY_PATH,LD_LIBRARY_PATH)
IVAL_INSTALLED   := $(B)/stage/usr/lib/$(if $(IVAL_DARWIN),libival.$(IVAL_MAJOR).dylib,libival.so)
$(B)/ival-install-check: ival/test/install-check.c ival/ival.pc.in $(IVAL_SHLIB) $(B)/libival.a
	rm -rf $(B)/stage
	$(MAKE) --no-print-directory install-ival DESTDIR=$(CURDIR)/$(B)/stage PREFIX=/usr > /dev/null
	PKG_CONFIG_PATH= PKG_CONFIG_LIBDIR=$(B)/stage/usr/lib/pkgconfig PKG_CONFIG_SYSROOT_DIR=$(CURDIR)/$(B)/stage \
	  sh -c '$(CC) -O2 -o $@ ival/test/install-check.c $$(pkg-config --cflags --libs ival) && \
	  $(CC) -O2 -o $@-static ival/test/install-check.c $$(pkg-config --cflags ival) \
	    $(IVAL_STATIC_LINK) $$(pkg-config --static --libs-only-l ival | sed "s/-lival//") && \
	  printf "#include <ival-scalar.h>\nint main(void) { double l, h; ival1_add(0.1, 0.1, 0.2, 0.2, &l, &h); return ival_version()[0] == 0 || !(l < h); }\n" | \
	    $(CXX) -x c++ -o $@-cxx - $$(pkg-config --cflags --libs ival)'
ival-install-check: $(B)/ival-install-check
	@$(IVAL_LIBPATH_VAR)=$(B)/stage/usr/lib $(B)/ival-install-check $(IVAL_INSTALLED)

$(B)/ival-check: ival/test/check.c $(IVALH) $(B)/libival.a $(B)/libkit.a
	$(CC) $(CFLAGS) $(FP) $(OPENMP) -Wall -Wextra -I kit -I ival -o $@ ival/test/check.c $(B)/libival.a $(B)/libkit.a -lmpfr -lgmp -ldl -lm

$(B)/ival-arith-check: ival/test/arith-check.c $(IVALH) $(B)/libival.a
	$(CC) $(CFLAGS) $(FP) -Wall -Wextra -I ival -o $@ ival/test/arith-check.c $(B)/libival.a -lmpfr -lgmp -ldl -lm
# ival's checks under AddressSanitizer and UndefinedBehaviorSanitizer, any report fatal (B=build-asan): make
# ival-sanitize. The checks' own arrays are left to the end of the process, so leaks are not reported
IVAL_CHECKS := ival-check ival-arith-check ival-1788-check ival-rev-check ival-text-check ival-acc-check ival-thread-check ival-env-check
ival-sanitize:
	$(MAKE) B=build-asan CFLAGS="-O1 -g -fsanitize=address,undefined -fno-sanitize-recover=all" $(addprefix build-asan/,$(IVAL_CHECKS))
	@for c in $(IVAL_CHECKS); do r=$$(ASAN_OPTIONS=detect_leaks=0 build-asan/$$c 2>&1 | tail -1); echo "$$c: $$r"; \
	  echo "$$r" | grep -q '^VERDICT: IDENTICAL' || exit 1; done
# ival from eight threads at once: in make check; make ival-thread-tsan runs it under ThreadSanitizer (B=build-tsan)
$(B)/ival-env-check: ival/test/env-check.c $(IVALH) $(B)/libival.a
	$(CC) $(CFLAGS) $(FP) -Wall -Wextra -I ival -o $@ ival/test/env-check.c $(B)/libival.a -ldl -lm
# built as callers build, without $(FP): the compiler may then treat floating-point operations as pure
$(B)/ival-mode-change-check: ival/test/mode-change-check.c $(IVALH) $(B)/libival.a
	$(CC) $(CFLAGS) -Wall -Wextra -I ival -o $@ ival/test/mode-change-check.c $(B)/libival.a -ldl -lm
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
# ival against IntervalArithmetic.jl's default (:correct) rounding on the same intervals (ival/julia/): times, and
# whether every bound agrees. Needs julia with IntervalArithmetic installed
ival-julia-compare: ival/julia/ia-bench.c $(IVALH) $(B)/libival.a
	$(CC) -O2 -I ival -o $(B)/ia-bench ival/julia/ia-bench.c $(B)/libival.a -ldl -lm
	cd $(B) && ./ia-bench && julia --startup-file=no $(CURDIR)/ival/julia/ia-bench.jl
# ival called from Julia through ccall (ival/julia/ccall-check.jl): every bound the same as IntervalArithmetic.jl's
# :correct, and Julia still in round-to-nearest after each call. Needs julia with IntervalArithmetic installed
ival-julia-ccall: $(IVAL_SHLIB)
	julia --startup-file=no ival/julia/ccall-check.jl $(abspath $(IVAL_SHLIB))
# the same from several Julia threads at once (ival/julia/threads-check.jl), while a sentinel thread checks that its
# own rounding mode stays round-to-nearest. Needs julia with IntervalArithmetic installed
ival-julia-threads: $(IVAL_SHLIB)
	julia --startup-file=no -t 4 ival/julia/threads-check.jl $(abspath $(IVAL_SHLIB))
# the Python binding (ival/python) against the library just built. Needs numpy, pytest and mpmath
ival-python-check: $(IVAL_SHLIB)
	cd ival/python && IVAL_LIBRARY=$(abspath $(IVAL_SHLIB)) PYTHONPATH=src python3 -m pytest -q tests
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
# crlin (lin/crlin.h): verified linear systems (2026-10-13; ROADMAP item 11), on crsum's exact products, with its own
# copy of crsum made local to it (the library exports crlin_ names only)
CRLINH := lin/crlin.h $(CRSUMH)
CRLIN_VERSION := $(shell sed -n 's/^\#define CRLIN_VERSION "\(.*\)"/\1/p' lin/crlin.h)
CRLIN_MAJOR   := $(firstword $(subst ., ,$(CRLIN_VERSION)))
$(B)/crlin/crlin-all.o: lin/crlin.c lin/band.c lin/crlin-internal.h crsum/crsum.c $(CRLINH) Makefile
	rm -rf $(B)/crlin && mkdir -p $(B)/crlin
	$(CC) $(CFLAGS) $(FP) $(OPENMP) -fPIC -Wall -Wextra -I crsum -c -o $(B)/crlin/crlin.o lin/crlin.c
	$(CC) $(CFLAGS) $(FP) $(OPENMP) -fPIC -Wall -Wextra -I crsum -c -o $(B)/crlin/band.o lin/band.c
	$(CC) $(CFLAGS) $(FP) -fPIC -fvisibility=hidden -c -o $(B)/crlin/crsum.o crsum/crsum.c
	$(CC) -r -nostdlib -o $@ $(B)/crlin/crlin.o $(B)/crlin/band.o $(B)/crlin/crsum.o
	$(if $(IVAL_DARWIN),,objcopy --localize-hidden $@)
$(B)/libcrlin.a: $(B)/crlin/crlin-all.o
	rm -f $@ && ar rcs $@ $<
$(B)/libcrlin.so: $(B)/crlin/crlin-all.o
	$(CC) -shared -Wl,-soname,libcrlin.so.$(CRLIN_MAJOR) -Wl,-z,defs -o $@ $< $(OPENMP) -lm
$(B)/libcrlin.dylib: $(B)/crlin/crlin-all.o
	$(CC) -dynamiclib -install_name @rpath/libcrlin.$(CRLIN_MAJOR).dylib -o $@ $< -lm
# make install-crlin (part of make install, Linux): libcrlin.so.0 and its links, libcrlin.a, crlin.h, crlin.pc
install-crlin: $(B)/libcrlin.so $(B)/libcrlin.a
	@test -n "$(CRLIN_VERSION)" || { echo "install: no CRLIN_VERSION in lin/crlin.h"; exit 1; }
	install -d $(DESTDIR)$(LIBDIR) $(DESTDIR)$(INCLUDEDIR) $(DESTDIR)$(PKGCONFIGDIR)
	install -m 755 $(B)/libcrlin.so $(DESTDIR)$(LIBDIR)/libcrlin.so.$(CRLIN_VERSION)
	ln -sf libcrlin.so.$(CRLIN_VERSION) $(DESTDIR)$(LIBDIR)/libcrlin.so.$(CRLIN_MAJOR)
	ln -sf libcrlin.so.$(CRLIN_MAJOR) $(DESTDIR)$(LIBDIR)/libcrlin.so
	install -m 644 $(B)/libcrlin.a $(DESTDIR)$(LIBDIR)/libcrlin.a
	install -m 644 lin/crlin.h $(DESTDIR)$(INCLUDEDIR)/
	sed -e 's|@PREFIX@|$(PREFIX)|' -e 's|@LIBDIR@|$(LIBDIR)|' -e 's|@INCLUDEDIR@|$(INCLUDEDIR)|' \
	  -e 's|@VERSION@|$(CRLIN_VERSION)|' -e 's|@OPENMP@|$(OPENMP)|' lin/crlin.pc.in > $(DESTDIR)$(PKGCONFIGDIR)/crlin.pc
uninstall-crlin:
	rm -f $(DESTDIR)$(LIBDIR)/libcrlin.so.$(CRLIN_VERSION) $(DESTDIR)$(LIBDIR)/libcrlin.so.$(CRLIN_MAJOR) \
	  $(DESTDIR)$(LIBDIR)/libcrlin.so $(DESTDIR)$(LIBDIR)/libcrlin.a $(DESTDIR)$(INCLUDEDIR)/crlin.h $(DESTDIR)$(PKGCONFIGDIR)/crlin.pc
$(B)/crlin-install-check: lin/test/install-check.c lin/crlin.pc.in $(B)/libcrlin.so $(B)/libcrlin.a
	rm -rf $(B)/stage-crlin
	$(MAKE) --no-print-directory install-crlin DESTDIR=$(CURDIR)/$(B)/stage-crlin PREFIX=/usr > /dev/null
	PKG_CONFIG_PATH= PKG_CONFIG_LIBDIR=$(B)/stage-crlin/usr/lib/pkgconfig PKG_CONFIG_SYSROOT_DIR=$(CURDIR)/$(B)/stage-crlin \
	  sh -c '$(CC) -O2 -o $@ lin/test/install-check.c $$(pkg-config --cflags --libs crlin) && \
	  $(CC) -O2 -o $@-static lin/test/install-check.c $$(pkg-config --cflags crlin) \
	    -Wl,-Bstatic $$(pkg-config --libs-only-L crlin) -lcrlin -Wl,-Bdynamic $$(pkg-config --static --libs-only-l crlin | sed "s/-lcrlin//") $$(pkg-config --static --libs-only-other crlin) && \
	  printf "#include <crlin.h>\nint main(void) { double A = 2, b = 1, l, h; return crlin_solve(1, &A, 1, &b, 0, 0, &l, &h, CRLIN_EXACT, 0, 0) != 0 || !(l <= 0.5 && 0.5 <= h); }\n" | \
	    $(CXX) -x c++ -o $@-cxx - $$(pkg-config --cflags --libs crlin)'
crlin-install-check: $(B)/crlin-install-check
	@LD_LIBRARY_PATH=$(B)/stage-crlin/usr/lib $(B)/crlin-install-check $(B)/stage-crlin/usr/lib/libcrlin.so
CRLIN_SHLIB := $(B)/libcrlin.$(if $(IVAL_DARWIN),dylib,so)
crlin-python-check: $(CRLIN_SHLIB)
	cd lin/python && CRLIN_LIBRARY=$(abspath $(CRLIN_SHLIB)) PYTHONPATH=src python3 -m pytest -q tests
# crfix (fix/crfix.h): correctly rounded fixed-point functions in integer arithmetic only (2026-10-14; ROADMAP item 3).
# The library exports crfix_ names only (the enclosure internals hidden); its check links crfix.c itself, for them.
CRFIXH := fix/crfix.h fix/crfix-internal.h fix/crfix-tables.h fix/crfix-exceptions.h
CRFIX_VERSION := $(shell sed -n 's/^\#define CRFIX_VERSION "\(.*\)"/\1/p' fix/crfix.h)
CRFIX_MAJOR   := $(firstword $(subst ., ,$(CRFIX_VERSION)))
$(B)/crfix/crfix.o: fix/crfix.c $(CRFIXH) Makefile
	mkdir -p $(B)/crfix
	$(CC) $(CFLAGS) -fPIC -Wall -Wextra -c -o $@ fix/crfix.c
$(B)/libcrfix.a: $(B)/crfix/crfix.o
	rm -f $@ && ar rcs $@ $<
$(B)/libcrfix.so: $(B)/crfix/crfix.o
	$(CC) -shared -Wl,-soname,libcrfix.so.$(CRFIX_MAJOR) -Wl,-z,defs -o $@ $<
$(B)/libcrfix.dylib: $(B)/crfix/crfix.o
	$(CC) -dynamiclib -install_name @rpath/libcrfix.$(CRFIX_MAJOR).dylib -o $@ $<
CRFIX_CHECK_SRC := fix/test/check.c fix/crfix.c fix/crfix-ref.c
$(B)/crfix-check: $(CRFIX_CHECK_SRC) fix/crfix-ref.h $(CRFIXH)
	@mkdir -p $(B)
	$(CC) $(CFLAGS) $(OPENMP) -Wall -Wextra -I fix -o $@ $(CRFIX_CHECK_SRC) -lmpfr -lgmp
$(B)/crfix-check-portable: $(CRFIX_CHECK_SRC) fix/crfix-ref.h $(CRFIXH)
	@mkdir -p $(B)
	$(CC) $(CFLAGS) $(OPENMP) -DCRFIX_PORTABLE -Wall -Wextra -I fix -o $@ $(CRFIX_CHECK_SRC) -lmpfr -lgmp
# make crfix-check-all: every 16-bit input, both builds (minutes on many cores)
crfix-check-all: $(B)/crfix-check $(B)/crfix-check-portable
	$(B)/crfix-check all && $(B)/crfix-check-portable all
$(B)/crfix-gen-tables: fix/gen-tables.c
	@mkdir -p $(B)
	$(CC) $(CFLAGS) -Wall -Wextra -o $@ $< -lmpfr -lgmp
crfix-tables: $(B)/crfix-gen-tables
	$< > fix/crfix-tables.h
$(B)/crfix-gen-exceptions: fix/gen-exceptions.c fix/crfix.c fix/crfix-ref.c fix/crfix-ref.h fix/crfix.h fix/crfix-internal.h fix/crfix-tables.h
	@mkdir -p $(B)
	$(CC) $(CFLAGS) $(OPENMP) -Wall -Wextra -I fix -o $@ fix/gen-exceptions.c fix/crfix.c fix/crfix-ref.c -lmpfr -lgmp
# make crfix-exceptions: the table, from every 16-bit input (a few minutes of CPU; the generator does not read the
# table it replaces)
crfix-exceptions: $(B)/crfix-gen-exceptions
	$< > $(B)/crfix-exceptions.h && mv $(B)/crfix-exceptions.h fix/crfix-exceptions.h
# make install-crfix (part of make install, Linux): libcrfix.so.0 and its links, libcrfix.a, crfix.h, crfix.pc
install-crfix: $(B)/libcrfix.so $(B)/libcrfix.a
	@test -n "$(CRFIX_VERSION)" || { echo "install: no CRFIX_VERSION in fix/crfix.h"; exit 1; }
	install -d $(DESTDIR)$(LIBDIR) $(DESTDIR)$(INCLUDEDIR) $(DESTDIR)$(PKGCONFIGDIR)
	install -m 755 $(B)/libcrfix.so $(DESTDIR)$(LIBDIR)/libcrfix.so.$(CRFIX_VERSION)
	ln -sf libcrfix.so.$(CRFIX_VERSION) $(DESTDIR)$(LIBDIR)/libcrfix.so.$(CRFIX_MAJOR)
	ln -sf libcrfix.so.$(CRFIX_MAJOR) $(DESTDIR)$(LIBDIR)/libcrfix.so
	install -m 644 $(B)/libcrfix.a $(DESTDIR)$(LIBDIR)/libcrfix.a
	install -m 644 fix/crfix.h $(DESTDIR)$(INCLUDEDIR)/
	sed -e 's|@PREFIX@|$(PREFIX)|' -e 's|@LIBDIR@|$(LIBDIR)|' -e 's|@INCLUDEDIR@|$(INCLUDEDIR)|' \
	  -e 's|@VERSION@|$(CRFIX_VERSION)|' fix/crfix.pc.in > $(DESTDIR)$(PKGCONFIGDIR)/crfix.pc
uninstall-crfix:
	rm -f $(DESTDIR)$(LIBDIR)/libcrfix.so.$(CRFIX_VERSION) $(DESTDIR)$(LIBDIR)/libcrfix.so.$(CRFIX_MAJOR) \
	  $(DESTDIR)$(LIBDIR)/libcrfix.so $(DESTDIR)$(LIBDIR)/libcrfix.a $(DESTDIR)$(INCLUDEDIR)/crfix.h $(DESTDIR)$(PKGCONFIGDIR)/crfix.pc
$(B)/crfix-install-check: fix/test/install-check.c fix/crfix.pc.in $(B)/libcrfix.so $(B)/libcrfix.a
	rm -rf $(B)/stage-crfix
	$(MAKE) --no-print-directory install-crfix DESTDIR=$(CURDIR)/$(B)/stage-crfix PREFIX=/usr > /dev/null
	PKG_CONFIG_PATH= PKG_CONFIG_LIBDIR=$(B)/stage-crfix/usr/lib/pkgconfig PKG_CONFIG_SYSROOT_DIR=$(CURDIR)/$(B)/stage-crfix \
	  sh -c '$(CC) -O2 -o $@ fix/test/install-check.c $$(pkg-config --cflags --libs crfix) && \
	  $(CC) -O2 -o $@-static fix/test/install-check.c $$(pkg-config --cflags crfix) \
	    -Wl,-Bstatic $$(pkg-config --libs-only-L crfix) -lcrfix -Wl,-Bdynamic && \
	  printf "#include <crfix.h>\nint main(void) { return crfix_exp(0, 0, 16, 8, CRFIX_NEAREST) != 256; }\n" | \
	    $(CXX) -x c++ -o $@-cxx - $$(pkg-config --cflags --libs crfix)'
crfix-install-check: $(B)/crfix-install-check
	@LD_LIBRARY_PATH=$(B)/stage-crfix/usr/lib $(B)/crfix-install-check $(B)/stage-crfix/usr/lib/libcrfix.so
# make crlin-cublas: lin/examples/cublas.c, crlin with cuBLAS's DGEMM and cuSOLVER's inverse (not in all: it needs
# CUDA, at CUDA_HOME)
CUDA_HOME ?= /usr/local/cuda
$(B)/crlin-cublas: lin/examples/cublas.c lin/crlin.h $(B)/libcrlin.a
	$(CC) $(CFLAGS) -Wall -Wextra -I lin -I $(CUDA_HOME)/include -o $@ lin/examples/cublas.c $(B)/libcrlin.a $(OPENMP) \
	  -L $(CUDA_HOME)/lib64 -Wl,-rpath,$(CUDA_HOME)/lib64 -lcusolver -lcublas -lcudart -lm
crlin-cublas: $(B)/crlin-cublas
$(B)/crlin-check: lin/test/check.c lin/crlin.h $(B)/libcrlin.a
	$(CC) $(CFLAGS) $(FP) $(OPENMP) -Wall -Wextra -I lin -o $@ lin/test/check.c $(B)/libcrlin.a -lgmp -lm

# crdist (dist/crdist.h): erfinv, erfcinv and probit (2026-10-12), ndtr,
# log_ndtr, normpdf and logit (2026-10-13), correctly rounded, with its own
# copies of the CORE-MATH functions it calls, made local to it as crnn's are
CRDISTCM := $(addprefix $(ROOT)/,log/log.c log1p.c exp.c erfc.c)
CRDISTH  := dist/crdist.h dist/crdist-fast.h dist/crdist-poly.h dist/crdist-exceptions.h dist/crdist-round16.h dist/crdist-exceptions16.h
$(B)/crdist/crdist-all.o: dist/crdist.c dist/crdist16.c dist/crdist-sample.c $(CRDISTH) $(CRDISTCM) Makefile
	rm -rf $(B)/crdist && mkdir -p $(B)/crdist
	$(CC) $(CFLAGS) $(FP) -fPIC -Wall -Wextra -c -o $(B)/crdist/crdist.o dist/crdist.c
	$(CC) $(CFLAGS) $(FP) -fPIC -Wall -Wextra -c -o $(B)/crdist/crdist16.o dist/crdist16.c
	$(CC) $(CFLAGS) $(FP) -fPIC -Wall -Wextra -c -o $(B)/crdist/crdist-sample.o dist/crdist-sample.c
	for f in $(CRDISTCM); do $(CC) $(CFLAGS) $(FP) -fPIC -fvisibility=hidden -c -o $(B)/crdist/cm-$$(basename $$f .c).o $$f || exit 1; done
	$(CC) -r -nostdlib -o $@ $(B)/crdist/crdist.o $(B)/crdist/crdist16.o $(B)/crdist/crdist-sample.o $(B)/crdist/cm-*.o
	$(if $(IVAL_DARWIN),,objcopy --localize-hidden $@)
$(B)/libcrdist.a: $(B)/crdist/crdist-all.o
	rm -f $@ && ar rcs $@ $<
CRDIST_VERSION := $(shell sed -n 's/^\#define CRDIST_VERSION "\(.*\)"/\1/p' dist/crdist.h)
CRDIST_MAJOR   := $(firstword $(subst ., ,$(CRDIST_VERSION)))
$(B)/libcrdist.so: $(B)/crdist/crdist-all.o
	$(CC) -shared -Wl,-soname,libcrdist.so.$(CRDIST_MAJOR) -Wl,-z,defs -o $@ $< -lm
# macOS (2026-10-13): a dylib, its install name versioned as the soname is, as ival's
$(B)/libcrdist.dylib: $(B)/crdist/crdist-all.o
	$(CC) -dynamiclib -install_name @rpath/libcrdist.$(CRDIST_MAJOR).dylib -o $@ $< -lm
CRDIST_SHLIB := $(B)/libcrdist.$(if $(IVAL_DARWIN),dylib,so)
# the CORE-MATH functions crdist calls, as an archive for its check and generators (the library's own copies are
# local to it); not libcm.a, whose binary16 files need functions macOS's libm lacks (exp10f)
$(B)/libcm-crdist.a: $(CRDISTCM) Makefile
	rm -rf $(B)/cm-crdist && mkdir -p $(B)/cm-crdist
	for f in $(CRDISTCM); do $(CC) $(CFLAGS) $(FP) -c -o $(B)/cm-crdist/$$(basename $$f .c).o $$f || exit 1; done
	rm -f $@ && ar rcs $@ $(B)/cm-crdist/*.o
# make install-crdist (part of make install, Linux only for now): the shared library under its versioned soname,
# libcrdist.a, crdist.h and crdist.pc
install-crdist: $(B)/libcrdist.so $(B)/libcrdist.a
	@test -n "$(CRDIST_VERSION)" || { echo "install: no CRDIST_VERSION in dist/crdist.h"; exit 1; }
	install -d $(DESTDIR)$(LIBDIR) $(DESTDIR)$(INCLUDEDIR) $(DESTDIR)$(PKGCONFIGDIR)
	install -m 755 $(B)/libcrdist.so $(DESTDIR)$(LIBDIR)/libcrdist.so.$(CRDIST_VERSION)
	ln -sf libcrdist.so.$(CRDIST_VERSION) $(DESTDIR)$(LIBDIR)/libcrdist.so.$(CRDIST_MAJOR)
	ln -sf libcrdist.so.$(CRDIST_MAJOR) $(DESTDIR)$(LIBDIR)/libcrdist.so
	install -m 644 $(B)/libcrdist.a $(DESTDIR)$(LIBDIR)/libcrdist.a
	install -m 644 dist/crdist.h $(DESTDIR)$(INCLUDEDIR)/
	sed -e 's|@PREFIX@|$(PREFIX)|' -e 's|@LIBDIR@|$(LIBDIR)|' -e 's|@INCLUDEDIR@|$(INCLUDEDIR)|' \
	  -e 's|@VERSION@|$(CRDIST_VERSION)|' dist/crdist.pc.in > $(DESTDIR)$(PKGCONFIGDIR)/crdist.pc
uninstall-crdist:
	rm -f $(DESTDIR)$(LIBDIR)/libcrdist.so.$(CRDIST_VERSION) $(DESTDIR)$(LIBDIR)/libcrdist.so.$(CRDIST_MAJOR) \
	  $(DESTDIR)$(LIBDIR)/libcrdist.so $(DESTDIR)$(LIBDIR)/libcrdist.a $(DESTDIR)$(INCLUDEDIR)/crdist.h \
	  $(DESTDIR)$(PKGCONFIGDIR)/crdist.pc
# in make check, as ival's: install into $(B)/stage-crdist, then build and run against it through pkg-config
$(B)/crdist-install-check: dist/test/install-check.c dist/crdist.pc.in $(B)/libcrdist.so $(B)/libcrdist.a
	rm -rf $(B)/stage-crdist
	$(MAKE) --no-print-directory install-crdist DESTDIR=$(CURDIR)/$(B)/stage-crdist PREFIX=/usr > /dev/null
	PKG_CONFIG_PATH= PKG_CONFIG_LIBDIR=$(B)/stage-crdist/usr/lib/pkgconfig PKG_CONFIG_SYSROOT_DIR=$(CURDIR)/$(B)/stage-crdist \
	  sh -c '$(CC) -O2 -o $@ dist/test/install-check.c $$(pkg-config --cflags --libs crdist) && \
	  $(CC) -O2 -o $@-static dist/test/install-check.c $$(pkg-config --cflags crdist) \
	    -Wl,-Bstatic $$(pkg-config --libs-only-L crdist) -lcrdist -Wl,-Bdynamic $$(pkg-config --static --libs-only-l crdist | sed "s/-lcrdist//") && \
	  printf "#include <crdist.h>\nint main(void) { float x = 0.5f, y; crdist_erfinvf(&y, &x, 1); return crdist_version()[0] == 0 || !(y > 0.47f); }\n" | \
	    $(CXX) -x c++ -o $@-cxx - $$(pkg-config --cflags --libs crdist)'
# the Python binding (dist/python): its tests against mpmath, on this build's library
crdist-python-check: $(CRDIST_SHLIB)
	cd dist/python && CRDIST_LIBRARY=$(abspath $(CRDIST_SHLIB)) PYTHONPATH=src python3 -m pytest -q tests
crdist-install-check: $(B)/crdist-install-check
	@LD_LIBRARY_PATH=$(B)/stage-crdist/usr/lib $(B)/crdist-install-check $(B)/stage-crdist/usr/lib/libcrdist.so
$(B)/crdist-check: dist/test/check.c dist/crdist-ref.c dist/crdist-ref.h $(CRDISTH) $(B)/libcrdist.a $(B)/libkit.a $(B)/libcm-crdist.a
	$(CC) $(CFLAGS) $(FP) $(OPENMP) -Wall -Wextra -I kit -I dist -o $@ dist/test/check.c dist/crdist-ref.c $(B)/libcrdist.a $(B)/libkit.a $(B)/libcm-crdist.a -lmpfr -lgmp -lm
$(B)/crdist-gen-exceptions: dist/gen-exceptions.c dist/crdist-ref.c dist/crdist-ref.h dist/crdist-fast.h dist/crdist-poly.h $(B)/libkit.a $(B)/libcm-crdist.a
	$(CC) $(CFLAGS) $(FP) $(OPENMP) -Wall -Wextra -I kit -I dist -o $@ dist/gen-exceptions.c dist/crdist-ref.c $(B)/libkit.a $(B)/libcm-crdist.a -lmpfr -lgmp -lm
crdist-exceptions: $(B)/crdist-gen-exceptions
	$(B)/crdist-gen-exceptions > $(B)/crdist-exceptions.h && mv $(B)/crdist-exceptions.h dist/crdist-exceptions.h
$(B)/crdist-gen-exceptions16: dist/gen-exceptions16.c dist/crdist-ref.c dist/crdist-ref.h dist/crdist-fast.h dist/crdist-poly.h dist/crdist-round16.h $(B)/libkit.a $(B)/libcm-crdist.a
	$(CC) $(CFLAGS) $(FP) $(OPENMP) -Wall -Wextra -I kit -I dist -o $@ dist/gen-exceptions16.c dist/crdist-ref.c $(B)/libkit.a $(B)/libcm-crdist.a -lmpfr -lgmp -lm
crdist-exceptions16: $(B)/crdist-gen-exceptions16
	$(B)/crdist-gen-exceptions16 > $(B)/crdist-exceptions16.h && mv $(B)/crdist-exceptions16.h dist/crdist-exceptions16.h
crdist-check-all: $(B)/crdist-check
	$(B)/crdist-check all
# the polynomials, from mpmath (Python 3); the table must be made again after
crdist-poly:
	cd dist && python3 fit.py > ../$(B)/crdist-poly.h && mv ../$(B)/crdist-poly.h crdist-poly.h

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

.PHONY: crfix-check-all crfix-tables crfix-exceptions install-crfix uninstall-crfix crfix-install-check crlin-cublas all check clean install uninstall install-ival uninstall-ival ival-install-check ival-python-check lowp-tables crnn-exceptions crnn-exceptions16 crnn-check-all crdist-exceptions crdist-exceptions16 crdist-check-all crdist-poly install-crdist uninstall-crdist crdist-install-check crdist-python-check crlin-python-check install-crlin uninstall-crlin crlin-install-check julia-check ival-julia-compare ival-julia-ccall ival-julia-threads crnn-vsame itf1788-check ival-acc-check ival-compare ival-thread-tsan ival-sanitize
