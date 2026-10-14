"""python -m crfix: an int8 kernel's lookup table, correctly rounded, as C, CSV or a list.

    python -m crfix table tanh --fin 5 --fout 7 --bits 8
    python -m crfix table-affine sigmoid --in 0.0478,3 --out 0x1p-8,-128 --f32 --format c --name logistic

A scale is a decimal or hex float (0x1p-8); --f32 rounds decimal scales to float32 first, as a TFLite model stores
them. The table's entry for x (or q) = -128 .. 127 is t[x + 128].
"""
import argparse
import sys

import numpy as np

import crfix


def _scale(s, f32):
    v = float.fromhex(s) if s.lower().lstrip("+-").startswith("0x") else float(s)
    return float(np.float32(v)) if f32 else v


def _affine(text, f32, default_bits):
    parts = text.split(",")
    if not 1 <= len(parts) <= 3:
        raise argparse.ArgumentTypeError("a format is SCALE[,ZERO_POINT[,BITS]]")
    return crfix.Affine(_scale(parts[0], f32), int(parts[1]) if len(parts) > 1 else 0,
                        int(parts[2]) if len(parts) > 2 else default_bits)


def render(t, fmt, name, bits, about):
    if fmt == "list":
        return " ".join(str(int(v)) for v in t) + "\n"
    if fmt == "csv":
        return "x,value\n" + "".join("%d,%d\n" % (x, int(v)) for x, v in zip(range(-128, 128), t))
    rows = [", ".join("%d" % int(v) for v in t[i:i + 16]) for i in range(0, 256, 16)]
    return ("/* %s; crfix %s; t[x + 128] for x = -128 .. 127 */\nstatic const int%d_t %s[256] = {\n  %s\n};\n"
            % (about, crfix.version(), bits, name, ",\n  ".join(rows)))


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m crfix", description="a correctly rounded lookup table for an int8 kernel")
    sub = p.add_subparsers(dest="cmd", required=True)
    for cmd in ("table", "table-affine"):
        s = sub.add_parser(cmd)
        s.add_argument("fn", choices=crfix.FUNCTIONS if cmd == "table" else crfix.FUNCTIONS[:6])
        if cmd == "table":
            s.add_argument("--fin", type=int, required=True, help="the input's fractional bits")
            s.add_argument("--fout", type=int, required=True, help="the output's fractional bits")
            s.add_argument("--bits", type=int, default=8, choices=(8, 16, 32), help="the output's container")
        else:
            s.add_argument("--in", dest="inp", required=True, help="the input's SCALE[,ZERO_POINT] (8 bits)")
            s.add_argument("--out", required=True, help="the output's SCALE[,ZERO_POINT[,BITS]]")
            s.add_argument("--f32", action="store_true", help="round decimal scales to float32 first")
        s.add_argument("--mode", default="nearest", choices=tuple(crfix.MODES))
        s.add_argument("--format", default="list", choices=("list", "csv", "c"))
        s.add_argument("--name", default=None, help="the C array's name")
    a = p.parse_args(argv)
    try:
        if a.cmd == "table":
            t, bits = crfix.table(a.fn, a.fin, a.fout, a.bits, a.mode), a.bits
            about = "%s from Q.%d to int%d with %d fractional bits, rounded %s" % (a.fn, a.fin, bits, a.fout, a.mode)
        else:
            inp, out = _affine(a.inp, a.f32, 8), _affine(a.out, a.f32, 8)
            t, bits = crfix.table_affine(a.fn, inp, out, a.mode), out.bits
            about = "%s, input scale %s zero point %d, output scale %s zero point %d int%d, rounded %s" % (
                a.fn, inp.scale.hex(), inp.zero_point, out.scale.hex(), out.zero_point, bits, a.mode)
    except (ValueError, argparse.ArgumentTypeError) as e:
        p.error(str(e))
    sys.stdout.write(render(t, a.format, a.name or "crfix_%s_table" % a.fn, bits, about))
    return 0


if __name__ == "__main__":
    sys.exit(main())
