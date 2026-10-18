# rerun (rerun.c): crnn, crsum, crdist, crfix and the CORE-MATH functions they and
# RoPE need, from their sources with the repository's floating-point flags;
# rerun.c on layerhash's contract (no contraction; round-to-nearest only, so
# without -frounding-math).
#   make
CC     ?= cc
TOP    := ../..
CM     := $(TOP)/core-math
FP     := -ffp-contract=off -frounding-math
NATIVE ?= $(shell $(CC) -march=native -x c -c -o /dev/null /dev/null 2>/dev/null && echo -march=native || \
            ($(CC) -mcpu=native -x c -c -o /dev/null /dev/null 2>/dev/null && echo -mcpu=native))
MAINF  := -O3 $(NATIVE) -ffp-contract=off
LIBSRC := $(TOP)/nn/crnn.c $(TOP)/nn/crnn16.c $(TOP)/crsum/crsum.c \
          $(TOP)/dist/crdist.c $(TOP)/dist/crdist16.c $(TOP)/dist/crdist-sample.c \
          $(TOP)/fix/crfix.c $(TOP)/fix/crfix-mp.c \
          $(addprefix $(CM)/,exp.c log1p.c erfc.c rsqrt.c sin.c cos.c pow/pow.c log/log.c)
LIBS   := $(shell echo 'int main(void){return 0;}' | $(CC) -x c - -o /dev/null -ldl 2>/dev/null && echo -ldl) -lpthread -lm
INC    := -I ../layerhash -I $(TOP)/nn -I $(TOP)/crsum -I $(TOP)/dist -I $(TOP)/fix

all: rerun

librr.a: $(LIBSRC) Makefile
	rm -rf rr-obj && mkdir -p rr-obj
	for f in $(LIBSRC); do $(CC) -O2 $(FP) -c -o rr-obj/$$(basename $$f .c).o $$f || exit 1; done
	rm -f $@ && ar rcs $@ rr-obj/*.o

rerun: rerun.c ../layerhash/sha256.h librr.a
	$(CC) $(MAINF) -DBUILD_FLAGS='"$(MAINF)"' $(INC) -o $@ rerun.c librr.a $(LIBS)

clean:
	rm -rf rerun librr.a rr-obj

.PHONY: all clean
