/* install-check.c: crdist as a user gets it (2026-10-13). make check installs crdist into build/stage-crdist
   (make install-crdist DESTDIR=...) and builds this program three ways through pkg-config: against the shared
   library, against libcrdist.a with Libs.private, and a C++ program including crdist.h. Run as install-check LIB
   (the installed libcrdist.so), it checks:
     - the library's version is the header's (crdist_version against CRDIST_VERSION);
     - four results, correctly rounded (crdist-check proves them on every input; here they show that the
       installed library is the one that was checked): erfinv(1/2) = 0x1.e861fcp-2, probit(0.975f) = 0x1.f5c03ap+0,
       erfcinv(2^-149) = 0x1.40a27cp+3, and bfloat16's erfinv(1/2) = 0x3ef4;
     - LIB's soname is libcrdist.so.<first number of the version>, and it exports crdist_ names only (30);
     - the static build runs and does not load libcrdist; the C++ build runs.
   Linux only (readelf, nm -D): crdist is not built on macOS yet.
   Each check must be able to fail: the static and C++ builds missing count as failures, not as skips. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "crdist.h"

static int bad;
static char first[300];
static void fail(const char *what) { if (!bad++) snprintf(first, sizeof first, "%s", what); }

static int results(void)
{
  float x[3] = {0.5f, 0.975f, 0x1p-149f}, a, b, c;
  uint16_t h = 0x3f00, r;
  crdist_erfinvf(&a, &x[0], 1);
  crdist_probitf(&b, &x[1], 1);
  crdist_erfcinvf(&c, &x[2], 1);
  crdist_erfinv_bf16(&r, &h, 1);
  return strcmp(crdist_version(), CRDIST_VERSION) == 0 && a == 0x1.e861fcp-2f && b == 0x1.f5c03ap+0f &&
         c == 0x1.40a27cp+3f && r == 0x3ef4;
}
/* the first line of a command's output that contains key, or "" */
static void grab(const char *cmd, const char *key, char *out, size_t n)
{
  char line[512];
  FILE *f = popen(cmd, "r");
  out[0] = 0;
  if (!f) return;
  while (fgets(line, sizeof line, f)) if (strstr(line, key) && !out[0]) snprintf(out, n, "%s", line);
  pclose(f);
}
int main(int argc, char **argv)
{
  if (!results()) fail("version or results");
  if (argc < 2) return bad;   /* the static build, run by the shared one */
  char cmd[1024], line[512], want[64], soname[64] = "";
  snprintf(want, sizeof want, "[libcrdist.so.%.*s]", (int)strcspn(CRDIST_VERSION, "."), CRDIST_VERSION);
  snprintf(cmd, sizeof cmd, "readelf -d '%s'", argv[1]);
  grab(cmd, "SONAME", line, sizeof line);
  if (!strstr(line, want)) fail("soname");
  else snprintf(soname, sizeof soname, "%s", want);
  snprintf(cmd, sizeof cmd, "nm -D --defined-only '%s'", argv[1]);
  FILE *f = popen(cmd, "r");
  int n = 0, other = 0;
  while (f && fgets(line, sizeof line, f)) {
    char name[256];
    if (sscanf(line, "%*s %*s %255s", name) == 1) { n++; other += strncmp(name, "crdist_", 7) != 0; }
  }
  if (f) pclose(f);
  if (n != 30 || other) fail("exports");
  snprintf(cmd, sizeof cmd, "'%s-static'", argv[0]);
  if (system(cmd) != 0) fail("static build");
  snprintf(cmd, sizeof cmd, "readelf -d '%s-static'", argv[0]);
  grab(cmd, "libcrdist", line, sizeof line);
  if (line[0]) fail("static build loads libcrdist");
  snprintf(cmd, sizeof cmd, "'%s-cxx'", argv[0]);
  if (system(cmd) != 0) fail("C++ build");
  if (!bad)
    printf("VERDICT: IDENTICAL (crdist %s installed and used through pkg-config: shared, static and C++ builds give the "
           "correctly rounded results; soname %s, %d exports, all crdist_)\n", crdist_version(), soname, n);
  else
    printf("VERDICT: DIFFERS: crdist installed: %s (%d problem%s)\n", first, bad, bad > 1 ? "s" : "");
  return bad ? 1 : 0;
}
