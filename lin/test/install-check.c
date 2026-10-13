/* install-check.c: crlin as a user gets it (2026-10-13). make check installs crlin into build/stage-crlin
   (make install-crlin DESTDIR=...) and builds this program three ways through pkg-config: against the shared
   library, against libcrlin.a with Libs.private, and a C++ program including crlin.h. Run as install-check LIB
   (the installed libcrlin.so), it checks:
     - the library's version is the header's (crlin_version against CRLIN_VERSION);
     - a result: A = diag(2, 4), b = (1, 1) verified, and its enclosure holds (1/2, 1/4), which are binary64 values;
     - LIB's soname is libcrlin.so.<first number of the version>, and it exports crlin_ names only (at least 3: no
       crsum symbol, the library's copy of crsum being local to it);
     - the static build runs and does not load libcrlin; the C++ build runs.
   Linux only (readelf, nm -D): make install installs crlin on Linux only.
   Each check must be able to fail: the static and C++ builds missing count as failures, not as skips. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "crlin.h"

static int bad;
static char first[300];
static void fail(const char *what) { if (!bad++) snprintf(first, sizeof first, "%s", what); }

static int results(void)
{
  double A[4] = {2, 0, 0, 4}, b[2] = {1, 1}, lo[2], hi[2];
  int r = crlin_solve(2, A, 2, b, NULL, 0, lo, hi, CRLIN_EXACT, NULL, NULL);
  return strcmp(crlin_version(), CRLIN_VERSION) == 0 && r == CRLIN_VERIFIED && lo[0] <= 0.5 && 0.5 <= hi[0] &&
         lo[1] <= 0.25 && 0.25 <= hi[1];
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
  snprintf(want, sizeof want, "[libcrlin.so.%.*s]", (int)strcspn(CRLIN_VERSION, "."), CRLIN_VERSION);
  snprintf(cmd, sizeof cmd, "readelf -d '%s'", argv[1]);
  grab(cmd, "SONAME", line, sizeof line);
  if (!strstr(line, want)) fail("soname");
  else snprintf(soname, sizeof soname, "%s", want);
  snprintf(cmd, sizeof cmd, "nm -D --defined-only '%s'", argv[1]);
  FILE *f = popen(cmd, "r");
  int n = 0, other = 0;
  while (f && fgets(line, sizeof line, f)) {
    char name[256];
    if (sscanf(line, "%*s %*s %255s", name) == 1) { n++; other += strncmp(name, "crlin_", 6) != 0; }
  }
  if (f) pclose(f);
  if (n < 3 || other) fail("exports");
  snprintf(cmd, sizeof cmd, "'%s-static'", argv[0]);
  if (system(cmd) != 0) fail("static build");
  snprintf(cmd, sizeof cmd, "readelf -d '%s-static'", argv[0]);
  grab(cmd, "libcrlin", line, sizeof line);
  if (line[0]) fail("static build loads libcrlin");
  snprintf(cmd, sizeof cmd, "'%s-cxx'", argv[0]);
  if (system(cmd) != 0) fail("C++ build");
  if (!bad)
    printf("VERDICT: IDENTICAL (crlin %s installed and used through pkg-config: shared, static and C++ builds verify "
           "and enclose; soname %s, %d exports, all crlin_)\n", crlin_version(), soname, n);
  else
    printf("VERDICT: DIFFERS: crlin installed: %s (%d problem%s)\n", first, bad, bad > 1 ? "s" : "");
  return bad ? 1 : 0;
}
