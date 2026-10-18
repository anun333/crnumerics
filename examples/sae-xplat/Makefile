# gpt2sae (gpt2sae.c): GPT-2 small and its autoencoders' features on
# layerhash's fixed-order contract. crnn, crsum and the CORE-MATH functions
# with the repository's floating-point flags; gpt2sae.c without
# -frounding-math (round-to-nearest only), as layerhash's tree builds.
#   make            gpt2sae-tree and gpt2sae-treefma
CC     ?= cc
TOP    := ../..
CM     := $(TOP)/core-math
FP     := -ffp-contract=off -frounding-math
NATIVE ?= $(shell $(CC) -march=native -x c -c -o /dev/null /dev/null 2>/dev/null && echo -march=native || \
            ($(CC) -mcpu=native -x c -c -o /dev/null /dev/null 2>/dev/null && echo -mcpu=native))
TREEF  := -O3 $(NATIVE) -ffp-contract=off
CRSRC  := $(TOP)/nn/crnn.c $(TOP)/nn/crnn16.c $(TOP)/crsum/crsum.c \
          $(addprefix $(CM)/,exp.c log1p.c erfc.c rsqrt.c tanh.c)
LIBS   := $(shell echo 'int main(void){return 0;}' | $(CC) -x c - -o /dev/null -ldl 2>/dev/null && echo -ldl) -lpthread -lm
INC    := -I ../layerhash -I $(TOP)/nn -I $(TOP)/crsum

all: gpt2sae-tree gpt2sae-treefma

libcrsx.a: $(CRSRC)
	rm -rf sx-obj && mkdir -p sx-obj
	for f in $(CRSRC); do $(CC) -O2 $(FP) -c -o sx-obj/$$(basename $$f .c).o $$f || exit 1; done
	rm -f $@ && ar rcs $@ sx-obj/*.o

gpt2sae-tree: gpt2sae.c ../layerhash/sha256.h libcrsx.a
	$(CC) $(TREEF) -DHERE='"$(CURDIR)"' -DBUILD_FLAGS='"$(TREEF)"' $(INC) -o $@ gpt2sae.c libcrsx.a $(LIBS)
gpt2sae-treefma: gpt2sae.c ../layerhash/sha256.h libcrsx.a
	$(CC) $(TREEF) -DTREEFMA -DHERE='"$(CURDIR)"' -DBUILD_FLAGS='"$(TREEF)"' $(INC) -o $@ gpt2sae.c libcrsx.a $(LIBS)

clean:
	rm -rf gpt2sae-tree gpt2sae-treefma libcrsx.a sx-obj

.PHONY: all clean
