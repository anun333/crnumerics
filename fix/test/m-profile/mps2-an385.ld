/* mps2-an385.ld: qemu-system-arm -M mps2-an385 (Cortex-M3, no FPU): code and data in the 4 MB SSRAM at 0, which
   qemu lets us write, bss, heap and stack in the 4 MB at 0x20000000; for fix/test/cross-check.sh */
ENTRY(_start)
MEMORY { ROM (rwx) : ORIGIN = 0x00000000, LENGTH = 4M
         RAM (rwx) : ORIGIN = 0x20000000, LENGTH = 4M }
SECTIONS {
  .text : { KEEP(*(.vectors)) *(.text*) KEEP(*(.init)) KEEP(*(.fini)) *(.rodata*) . = ALIGN(4); } > ROM
  .ARM.exidx : { *(.ARM.exidx*) } > ROM
  .preinit_array : { PROVIDE_HIDDEN(__preinit_array_start = .); KEEP(*(.preinit_array*)) PROVIDE_HIDDEN(__preinit_array_end = .); } > ROM
  .init_array : { PROVIDE_HIDDEN(__init_array_start = .); KEEP(*(SORT(.init_array.*))) KEEP(*(.init_array*)) PROVIDE_HIDDEN(__init_array_end = .); } > ROM
  .fini_array : { PROVIDE_HIDDEN(__fini_array_start = .); KEEP(*(.fini_array*)) PROVIDE_HIDDEN(__fini_array_end = .); } > ROM
  .data : { *(.data*) . = ALIGN(4); } > ROM
  .bss : { __bss_start__ = .; *(.bss*) *(COMMON) . = ALIGN(8); __bss_end__ = .; } > RAM
  end = .; _end = .; __end__ = .;
  __stack_top = ORIGIN(RAM) + LENGTH(RAM);
  __HeapLimit = __stack_top - 0x10000;
}
