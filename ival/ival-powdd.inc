/* ival-powdd.inc: the body of pown_dd (ival-eft.h), included twice: as written, and compiled for the FMA instruction */
IVAL_POWDD_ATTR static int IVAL_POWDD_NAME(double x, int p, int up, double *r)
{
  if (p < 2 || p > 64 || !(fabs(x) < INFINITY) || x == 0) return 0;
  int neg = x < 0 && (p & 1);
#if IVAL_PLANT_ARITH != 45   /* 45: a negative odd power rounded the way of its magnitude, not the other */
  if (neg) up = !up;   /* x^p = -|x|^p: down is minus |x|^p rounded up */
#endif
  double ax = fabs(x), h = ax, l = 0;
  int exact = 1;
  for (int k = 1; k < p; k++) {
    double ph = h * ax, pt = __builtin_fma(h, ax, -ph);
    if (l != 0) exact = 0;   /* l * ax + pt may round */
    double q = __builtin_fma(l, ax, pt);
    h = ph + q;              /* FastTwoSum: |ph| >= |q| */
    l = q - (h - ph);
  }
  if (!(h >= 0x1p-900 && h <= 0x1p1000)) return 0;
  double v;
  if (exact) v = up ? (l > 0 ? succ(h) : h) : (l < 0 ? pred(h) : h);
  else {
    double e = h * (p * 0x1p-100);
    if (l > e) v = up ? succ(h) : h;
    else if (l < -e) v = up ? h : pred(h);
    else return 0;
  }
  *r = neg ? -v : v;
  return 1;
}
