/* vectors.c: the two words an M-profile core reads at reset, for fix/test/cross-check.sh's bare-metal run: the
   initial stack pointer, and newlib's _start (rdimon: output through semihosting) */
extern void _start(void);
extern unsigned long __stack_top;
__attribute__((section(".vectors"), used)) const void *const vectors[2] = {&__stack_top, (const void *)_start};
