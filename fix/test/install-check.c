/* install-check.c: crfix as a user gets it (2026-10-14, from lin/test/install-check.c). make check installs crfix into build/stage-crfix
   (make install-crfix DESTDIR=...) and builds this program three ways through pkg-config: against the shared
   library, against libcrfix.a with Libs.private, and a C++ program including crfix.h. Run as install-check LIB
   (the installed libcrfix.so), it checks:
     - the library's version is the header's (crfix_version against CRFIX_VERSION);
     - results: e^0 = 1 in Q1.14, sigmoid(0) = 1/2 in Q0.7, sqrt(2) in Q1.14 = 23170.475...: 23170 to nearest, 23171
       up, log2(8) = 3, and an int8 table;
     - LIB's soname is libcrfix.so.<first number of the version>, and it exports crfix_ names only (at least 12: the
       enclosure and rounding internals hidden);
     - the static build runs and does not load libcrfix; the C++ build runs.
   Linux only (readelf, nm -D): make install installs crfix on Linux only.
   Each check must be able to fail: the static and C++ builds missing count as failures, not as skips. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "crfix.h"

static int bad;
static char first[300];
static void fail(const char *what) { if (!bad++) snprintf(first, sizeof first, "%s", what); }

static int results(void)
{
  int32_t t[256];
  return strcmp(crfix_version(), CRFIX_VERSION) == 0 && crfix_exp(0, 4, 16, 14, CRFIX_NEAREST) == 16384 &&
         crfix_sigmoid(0, 3, 8, 7, CRFIX_NEAREST) == 64 && crfix_sqrt(2, 0, 16, 14, CRFIX_NEAREST) == 23170 &&
         crfix_sqrt(2, 0, 16, 14, CRFIX_UP) == 23171 && crfix_log2(8, 0, 8, 0, CRFIX_NEAREST) == 3 &&
         crfix_table(CRFIX_TANH, 5, 8, 7, CRFIX_NEAREST, t) == 0 && t[128] == 0 && t[255] == crfix_tanh(127, 5, 8, 7, 0);
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
  snprintf(want, sizeof want, "[libcrfix.so.%.*s]", (int)strcspn(CRFIX_VERSION, "."), CRFIX_VERSION);
  snprintf(cmd, sizeof cmd, "readelf -d '%s'", argv[1]);
  grab(cmd, "SONAME", line, sizeof line);
  if (!strstr(line, want)) fail("soname");
  else snprintf(soname, sizeof soname, "%s", want);
  snprintf(cmd, sizeof cmd, "nm -D --defined-only '%s'", argv[1]);
  FILE *f = popen(cmd, "r");
  int n = 0, other = 0;
  while (f && fgets(line, sizeof line, f)) {
    char name[256];
    if (sscanf(line, "%*s %*s %255s", name) == 1) { n++; other += strncmp(name, "crfix_", 6) != 0; }
  }
  if (f) pclose(f);
  if (n < 12 || other) fail("exports");
  snprintf(cmd, sizeof cmd, "'%s-static'", argv[0]);
  if (system(cmd) != 0) fail("static build");
  snprintf(cmd, sizeof cmd, "readelf -d '%s-static'", argv[0]);
  grab(cmd, "libcrfix", line, sizeof line);
  if (line[0]) fail("static build loads libcrfix");
  snprintf(cmd, sizeof cmd, "'%s-cxx'", argv[0]);
  if (system(cmd) != 0) fail("C++ build");
  if (!bad)
    printf("VERDICT: IDENTICAL (crfix %s installed and used through pkg-config: shared, static and C++ builds give "
           "the specification's results; soname %s, %d exports, all crfix_)\n", crfix_version(), soname, n);
  else
    printf("VERDICT: DIFFERS: crfix installed: %s (%d problem%s)\n", first, bad, bad > 1 ? "s" : "");
  return bad ? 1 : 0;
}
