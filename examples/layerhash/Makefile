# layerhash: three builds of one file (layerhash.c). cr and tree take crnn,
# crsum and the CORE-MATH functions they need straight from their sources,
# compiled here with the repository's floating-point flags; stock takes
# nothing but the system's libm.
#   make            the three binaries
#   make NATIVE=    stock without -march=native
CC     ?= cc
TOP    := ../..
CM     := $(TOP)/core-math
FP     := -ffp-contract=off -frounding-math
OPT    := -O2
# stock: the flags plain C inference is usually built with
NATIVE ?= $(shell $(CC) -march=native -x c -c -o /dev/null /dev/null 2>/dev/null && echo -march=native || \
            ($(CC) -mcpu=native -x c -c -o /dev/null /dev/null 2>/dev/null && echo -mcpu=native))
STOCKF := -O3 -ffast-math $(NATIVE)
CRSRC  := $(TOP)/nn/crnn.c $(TOP)/nn/crnn16.c $(TOP)/crsum/crsum.c \
          $(addprefix $(CM)/,exp.c log1p.c erfc.c rsqrt.c sin.c cos.c pow/pow.c)
LIBS   := $(shell echo 'int main(void){return 0;}' | $(CC) -x c - -o /dev/null -ldl 2>/dev/null && echo -ldl) -lm

all: layerhash-cr layerhash-tree layerhash-stock

layerhash-cr: layerhash.c sha256.h $(CRSRC)
	$(CC) $(OPT) $(FP) -DMODE_CR -DBUILD_FLAGS='"$(OPT) $(FP)"' -I $(TOP)/nn -I $(TOP)/crsum -o $@ layerhash.c $(CRSRC) $(LIBS)
layerhash-tree: layerhash.c sha256.h $(CRSRC)
	$(CC) $(OPT) $(FP) -DMODE_TREE -DBUILD_FLAGS='"$(OPT) $(FP)"' -I $(TOP)/nn -I $(TOP)/crsum -o $@ layerhash.c $(CRSRC) $(LIBS)
layerhash-stock: layerhash.c sha256.h
	$(CC) $(STOCKF) -DMODE_STOCK -DBUILD_FLAGS='"$(STOCKF)"' -o $@ layerhash.c -lm

clean:
	rm -f layerhash-cr layerhash-tree layerhash-stock

.PHONY: all clean
