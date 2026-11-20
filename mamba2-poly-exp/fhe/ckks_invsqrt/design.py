"""CKKS-only 1/sqrt(v) per RMSNorm instance: pick a Chebyshev degree and Newton count.

The CKKS-native alternative to fhe/seglut_invsqrt (scheme switching + table): for each
of the 49 norms, a Chebyshev series p(v) ~ 1/sqrt(v) fit on that layer's calibrated
range, then k Newton steps y <- y (3 - v y^2) / 2. No comparisons, no table, no
scheme switch; every operation is a CKKS add or multiply.

The fit minimises RELATIVE error (weight sqrt(v)), since Newton squares the relative
error and RMSNorm's output error is relative. Chebyshev convergence on [a, R a] goes
like rho^-d with rho = (sqrt R + 1) / (sqrt R - 1), so the degree a layer needs grows
like sqrt(R): this is entirely a question of each layer's range.

Depth, OpenFHE 1.2.1 EvalChebyshevSeries (Paterson-Stockmeyer, its own table) plus 3
levels per Newton step (y^2, * (-v/2), * y), the count fhe/seglut_invsqrt/bench.cpp uses.

The danger the table route has a guard for: outside [a, b] a polynomial extrapolates,
so `tail` reports what comes out at v = a / t and b * t.

Run:  python fhe/ckks_invsqrt/design.py eval/prenorm_tfhe/cal8k_m05_g.json
"""

from __future__ import annotations

import argparse
import json
import math

import numpy as np
from numpy.polynomial import chebyshev as C

# OpenFHE EvalChebyshevSeries depth by degree upper bound
PS_DEPTH = [(5, 3), (13, 4), (27, 5), (59, 6), (119, 7), (247, 8), (495, 9), (1007, 10),
            (2031, 11)]
DEGREES = [d for d, _ in PS_DEPTH]
NEWTON_DEPTH = 3


def cheb_depth(d):
    for top, depth in PS_DEPTH:
        if d <= top:
            return depth
    raise ValueError(d)


def widen_log(lo, hi, margin):
    """seglut.make_boundaries' log margin: the log-range grows by (1 + margin)."""
    llo, lhi = math.log(lo), math.log(hi)
    mid, half = 0.5 * (llo + lhi), 0.5 * (lhi - llo) * (1.0 + margin)
    return math.exp(mid - half), math.exp(mid + half)


def fit(a, b, d, n_pts=None):
    """Chebyshev coefficients on [a, b] (OpenFHE's convention: series in t = (2v-a-b)/(b-a))."""
    n = n_pts or max(8 * (d + 1), 512)
    t = np.cos(np.pi * (np.arange(n) + 0.5) / n)            # Chebyshev nodes
    v = 0.5 * (b - a) * t + 0.5 * (b + a)
    w = np.sqrt(v)                                          # relative error weight
    return C.chebfit(t, 1.0 / np.sqrt(v), d, w=w)


def evaluate(coef, a, b, v, newton):
    t = (2.0 * v - a - b) / (b - a)
    y = C.chebval(t, coef)
    for _ in range(newton):
        y = y * (3.0 - v * y * y) / 2.0
    return y


def rel_bits(y, v):
    rel = y * np.sqrt(v) - 1.0
    if not np.all(np.isfinite(rel)):
        return -np.inf, -np.inf
    return float(-np.log2(np.abs(rel).max() + 1e-300)), float(-np.log2(np.sqrt(np.mean(rel ** 2)) + 1e-300))


def sweep(a, b, newton_max=3, tail=2.0):
    """All (degree, newton) on the grid, with worst/rms rel bits on [a, b] and the tails."""
    v = np.exp(np.linspace(math.log(a), math.log(b), 4001))  # log-uniform: every decade counts
    vt = np.array([a / tail, b * tail])
    rows = []
    for d in DEGREES:
        coef = fit(a, b, d)
        for k in range(newton_max + 1):
            worst, rms = rel_bits(evaluate(coef, a, b, v, k), v)
            tw = rel_bits(evaluate(coef, a, b, vt, k), vt)[0]
            rows.append(dict(degree=d, newton=k, depth=cheb_depth(d) + NEWTON_DEPTH * k,
                             worst_bits=worst, rms_bits=rms, tail_bits=tw))
    return rows


def cheapest(rows, target):
    ok = [r for r in rows if r["worst_bits"] >= target]
    return min(ok, key=lambda r: (r["depth"], r["degree"])) if ok else None


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("ranges", help="a seglut_paired.py result JSON (uses norm_v_ranges)")
    ap.add_argument("--margin", type=float, default=0.05, help="log margin, as the tables use")
    ap.add_argument("--targets", type=float, nargs="+", default=[8.0, 12.0, 15.0],
                    help="worst-case relative bits to reach")
    ap.add_argument("--tail", type=float, default=2.0, help="report v = a/tail and b*tail")
    ap.add_argument("--out", default=None, help="write the per-norm choices as JSON")
    ap.add_argument("--emit", default=None,
                    help="write bench.cpp's case file: one line per (norm, target), "
                         "'name a b newton degree c_0 .. c_d' in numpy's Chebyshev convention")
    args = ap.parse_args()

    ranges = json.load(open(args.ranges))["norm_v_ranges"]
    out = {"margin": args.margin, "targets": args.targets, "tail": args.tail, "norms": {}}
    print(f"margin {args.margin} (log), targets {args.targets} worst-case rel bits, "
          f"tail at a/{args.tail:g} and b*{args.tail:g}\n")
    hdr = "  ".join(f"{'>=' + format(t, 'g') + ' bits':>22}" for t in args.targets)
    print(f"{'norm':<14}{'R = b/a':>10}  {hdr}")
    print(f"{'':<14}{'':>10}  " + "  ".join(f"{'deg+N  depth  tail':>22}" for _ in args.targets))
    for site, layers in ranges.items():
        for layer, (lo, hi) in sorted(layers.items(), key=lambda kv: int(kv[0])):
            a, b = widen_log(lo, hi, args.margin)
            rows = sweep(a, b, tail=args.tail)
            picks = [cheapest(rows, t) for t in args.targets]
            name = f"{site}.{layer}"
            out["norms"][name] = {"lo": lo, "hi": hi, "a": a, "b": b, "R": b / a,
                                  "picks": {str(t): p for t, p in zip(args.targets, picks)}}
            cells = []
            for p in picks:
                if p is None:
                    cells.append(f"{'none <= 2031':>22}")
                else:
                    tb = "div" if not math.isfinite(p["tail_bits"]) else f"{p['tail_bits']:.1f}"
                    cells.append(f"{str(p['degree']) + '+' + str(p['newton']):>10}  {p['depth']:>5}  {tb:>5}")
            print(f"{name:<14}{b / a:>10.0f}  " + "  ".join(cells))

    print("\nsummary (depth of the 1/sqrt alone; RMSNorm adds x^2 and x*r, 2 levels):")
    for t in args.targets:
        ds = [n["picks"][str(t)]["depth"] for n in out["norms"].values() if n["picks"][str(t)]]
        miss = [k for k, n in out["norms"].items() if not n["picks"][str(t)]]
        msg = (f"  >= {t:g} bits: {len(ds)}/{len(out['norms'])} reachable, depth median "
               f"{int(np.median(ds)) if ds else '-'} max {max(ds) if ds else '-'}")
        print(msg + (f"; unreachable: {', '.join(miss)}" if miss else ""))
    if args.out:
        json.dump(out, open(args.out, "w"), indent=1)
        print(f"\nwrote {args.out}")
    if args.emit:
        n = 0
        with open(args.emit, "w") as f:
            for name, nm in out["norms"].items():
                for t, p in nm["picks"].items():
                    if p is None:
                        continue
                    coef = fit(nm["a"], nm["b"], p["degree"])
                    f.write(f"{name}@{float(t):g} {nm['a']!r} {nm['b']!r} {p['newton']} {p['degree']} "
                            + " ".join(repr(float(c)) for c in coef) + "\n")
                    n += 1
        print(f"wrote {n} cases to {args.emit}")


if __name__ == "__main__":
    main()
