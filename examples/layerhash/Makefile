# layerhash: four builds of one file (layerhash.c). cr, tree and treefma
# take crnn, crsum and the CORE-MATH functions they need from their sources,
# compiled here with the repository's floating-point flags; stock takes
# nothing but the system's libm.
#   make            the four binaries
#   make NATIVE=    stock, tree and treefma without -march=native
CC     ?= cc
TOP    := ../..
CM     := $(TOP)/core-math
FP     := -ffp-contract=off -frounding-math
OPT    := -O2
# stock: the flags plain C inference is usually built with
NATIVE ?= $(shell $(CC) -march=native -x c -c -o /dev/null /dev/null 2>/dev/null && echo -march=native || \
            ($(CC) -mcpu=native -x c -c -o /dev/null /dev/null 2>/dev/null && echo -mcpu=native))
STOCKF := -O3 -ffast-math $(NATIVE)
# tree and treefma: the same optimization as stock, without -ffast-math (the
# order is in the source) and without contraction. layerhash.c runs in
# round-to-nearest only, so it does without -frounding-math, which stops GCC
# from vectorizing fmaf; the library sources keep the repository's flags.
TREEF  := -O3 $(NATIVE) -ffp-contract=off
CRSRC  := $(TOP)/nn/crnn.c $(TOP)/nn/crnn16.c $(TOP)/crsum/crsum.c \
          $(addprefix $(CM)/,exp.c log1p.c erfc.c rsqrt.c sin.c cos.c pow/pow.c)
LIBS   := $(shell echo 'int main(void){return 0;}' | $(CC) -x c - -o /dev/null -ldl 2>/dev/null && echo -ldl) -lm
INC    := -I $(TOP)/nn -I $(TOP)/crsum

all: layerhash-cr layerhash-tree layerhash-treefma layerhash-stock

# crnn, crsum and the CORE-MATH functions, with the repository's flags
libcrlh.a: $(CRSRC)
	rm -rf lh-obj && mkdir -p lh-obj
	for f in $(CRSRC); do $(CC) $(OPT) $(FP) -c -o lh-obj/$$(basename $$f .c).o $$f || exit 1; done
	rm -f $@ && ar rcs $@ lh-obj/*.o

layerhash-cr: layerhash.c sha256.h libcrlh.a
	$(CC) $(OPT) $(FP) -DMODE_CR -DBUILD_FLAGS='"$(OPT) $(FP)"' $(INC) -o $@ layerhash.c libcrlh.a $(LIBS)
layerhash-tree: layerhash.c sha256.h libcrlh.a
	$(CC) $(TREEF) -DMODE_TREE -DBUILD_FLAGS='"$(TREEF)"' $(INC) -o $@ layerhash.c libcrlh.a $(LIBS)
layerhash-treefma: layerhash.c sha256.h libcrlh.a
	$(CC) $(TREEF) -DMODE_TREEFMA -DBUILD_FLAGS='"$(TREEF)"' $(INC) -o $@ layerhash.c libcrlh.a $(LIBS)
layerhash-stock: layerhash.c sha256.h
	$(CC) $(STOCKF) -DMODE_STOCK -DBUILD_FLAGS='"$(STOCKF)"' -o $@ layerhash.c -lm

clean:
	rm -rf layerhash-cr layerhash-tree layerhash-treefma layerhash-stock libcrlh.a lh-obj

.PHONY: all clean
