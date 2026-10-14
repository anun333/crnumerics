# crfix: correctly rounded fixed-point functions for Python

`crfix.sigmoid(x, fin, fout, bits)`: for every integer x, the exact sigmoid of x·2^−fin rounded once to an integer
with fout fractional bits, in the rounding mode you ask for, then saturated to `bits`. The same goes for exp, exp2,
log, log2, tanh, sqrt, rsqrt, the reciprocal, division, SiLU and GELU (the exact one, v·Φ(v)). A container is 8, 16 or
32 bits, or `"u8"` and `"u16"`, unsigned (TFLite's uint8). Every result is the one an exact computation followed by
one rounding would give, so a lookup table or a fixed-point kernel built from it has no error beyond its format.

```python
import numpy as np, crfix
x = np.arange(-128, 128, dtype=np.int8)
y = crfix.sigmoid(x, fin=4, fout=7, bits=8)              # sigmoid(x / 16) as round(sigmoid * 128), int8
t = crfix.table("tanh", fin=5, fout=7, bits=8)            # an int8 kernel's lookup table: t[x + 128]
q = crfix.div(a, 8, b, 8, fout=12, bits=16, mode="down")  # (a / 256) / (b / 256), rounded down
```

Layer normalization over the last axis, exact, with γ, β and ε in fixed point:

```python
y = crfix.layernorm(x, fin=3, fout=5, eps=1, fe=10, gamma=g, fg=7, beta=b, fb=4, bits=8)
```

TFLite's affine formats, where a value is scale·(q − zero_point), for any scales:

```python
inp, out = crfix.Affine(0.0478, 3), crfix.Affine(1 / 256, -128)
t = crfix.table_affine("sigmoid", inp, out)               # a LOGISTIC kernel's table, every entry correctly rounded
p = crfix.softmax(logits, c=beta * 0.0478, out=out)       # the exact softmax of c·logits, rounded into out
```

A lookup table as C source, CSV or a list, from the command line (`--f32` rounds decimal scales to float32 first,
as a TFLite model stores them; hex floats such as `0x1p-8` are taken exactly):

```
python -m crfix table tanh --fin 5 --fout 7 --bits 8 --format c
python -m crfix table-affine sigmoid --in 0.0478,3 --out 0x1p-8,-128 --f32 --format c --name logistic
```

- **Modes:** `"nearest"` (ties to even), `"up"`, `"down"` and `"zero"`. In an affine format, "zero" rounds the value
  itself toward 0, before the zero point is added.
- **Domain edges:** log of a value ≤ 0 gives the most negative result. sqrt of a negative value gives 0. rsqrt of a
  value ≤ 0, and the reciprocal of 0, give the largest. a/0 goes by a's sign, and 0/0 gives 0.
- **Integer arithmetic only:** the library has no floating point, and gives the same bits on x86-64, ARM64, 32-bit
  ARM and a Cortex-M0.
- **How it is proven:** every 16-bit input, in every format, mode and container, is checked against MPFR. Results
  that the fast 64-bit enclosures cannot decide go to interval arithmetic that is correct by construction. This
  package's own tests use a second, independent reference: Python's `decimal` and exact fractions.

`pip install crfix` brings a wheel with the library, `libcrfix.so.0` (or `libcrfix.0.dylib` on macOS);
`CRFIX_LIBRARY` names another. The methods and their checks are in crnumerics'
[`fix/README.md`](https://github.com/anun333/crnumerics/blob/main/fix/README.md).
