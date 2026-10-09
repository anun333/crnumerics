/* install-check.c: ival as a user gets it. make check installs ival into build/stage (make install DESTDIR=...)
   and builds this program three ways through pkg-config: against the shared library, against libival.a with
   Libs.private, and a C++ program including ival.h. Run as install-check LIB (the installed libival.so), it checks:
     - the library's version is the header's (ival_version against IVAL_VERSION);
     - two results: exp of [0, 1] is [1, e rounded up], and [0.1, 0.1] + [0.2, 0.2] is that sum rounded both ways;
     - LIB's soname is libival.so.<first number of the version>, and it exports ival_ names only (at least 100);
     - the static build runs and does not load libival.so; the C++ build runs.
   Each check must be able to fail: the static and C++ builds missing count as failures, not as skips. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ival.h"

static int bad;
static char first[300];
static void fail(const char *what) { if (!bad++) snprintf(first, sizeof first, "%s", what); }

/* the results and the version: all the static and C++ builds are asked */
static int results(void)
{
  double lo = 0, hi = 1, yl, yh, a = 0.1, b = 0.2, sl, sh;
  ival_exp(&lo, &hi, &yl, &yh, 1);
  ival_add(&a, &a, &b, &b, &sl, &sh, 1);
  return strcmp(ival_version(), IVAL_VERSION) == 0 && yl == 1 && yh == 0x1.5bf0a8b14576ap+1 &&
         sl == 0x1.3333333333333p-2 && sh == 0x1.3333333333334p-2;
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
  char cmd[1024], line[512], want[64];
  /* the soname */
  snprintf(want, sizeof want, "[libival.so.%.*s]", (int)strcspn(IVAL_VERSION, "."), IVAL_VERSION);
  snprintf(cmd, sizeof cmd, "readelf -d '%s'", argv[1]);
  grab(cmd, "SONAME", line, sizeof line);
  if (!strstr(line, want)) fail("soname");
  /* the exports */
  snprintf(cmd, sizeof cmd, "nm -D --defined-only '%s'", argv[1]);
  FILE *f = popen(cmd, "r");
  int n = 0, other = 0;
  while (f && fgets(line, sizeof line, f)) {
    char name[256];
    if (sscanf(line, "%*s %*s %255s", name) == 1) { n++; other += strncmp(name, "ival_", 5) != 0; }
  }
  if (f) pclose(f);
  if (n < 100 || other) fail("exports");
  /* the static build: runs, and libival.so is not among its needs */
  snprintf(cmd, sizeof cmd, "'%s-static'", argv[0]);
  if (system(cmd) != 0) fail("static build");
  snprintf(cmd, sizeof cmd, "readelf -d '%s-static'", argv[0]);
  grab(cmd, "libival", line, sizeof line);
  if (line[0]) fail("static build loads libival.so");
  /* the C++ build */
  snprintf(cmd, sizeof cmd, "'%s-cxx'", argv[0]);
  if (system(cmd) != 0) fail("C++ build");
  if (!bad)
    printf("VERDICT: IDENTICAL (ival %s installed and used through pkg-config: shared, static and C++ builds give the "
           "tight results; soname %s, %d exports, all ival_)\n", ival_version(), want, n);
  else
    printf("VERDICT: DIFFERS (%d failed; first: %s)\n", bad, first);
  return bad != 0;
}
