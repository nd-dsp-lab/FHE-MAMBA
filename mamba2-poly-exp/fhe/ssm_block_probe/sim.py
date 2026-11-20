"""Plaintext simulator of the packed HE-native SSM block that bench.cpp runs under CKKS.

Every ciphertext is modelled as a numpy vector of `slots` reals, and every homomorphic
operation as the matching vector operation: EvalRotate(ct, k) is np.roll(v, -k), a
plaintext multiply is an elementwise product, and so on. The script runs the packed
algorithm exactly as bench.cpp does (same layout, same padding, same log-depth scan) and
checks it against a naive sequential reference that knows nothing about packing.

This is the local test of the packing logic, since OpenFHE is only on tjws-03. It also
counts operations and multiplicative depth, which is the cost model the probe measures.

    .venv/bin/python fhe/ssm_block_probe/sim.py                  # toy shape
    .venv/bin/python fhe/ssm_block_probe/sim.py --probe-shape    # the tjws-03 shape
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field

import numpy as np
from numpy.polynomial import chebyshev as C


# ----------------------------------------------------------------------------- model

@dataclass
class Shape:
    slots: int = 4096
    D: int = 64          # d_model
    L: int = 64          # sequence length
    P: int = 32          # headdim
    N: int = 8           # d_state
    K: int = 4           # conv width
    expand: int = 2
    selective_heads: int = -1   # -1: all heads selective
    seed: int = 0

    @property
    def Di(self): return self.expand * self.D
    @property
    def H(self): return self.Di // self.P
    @property
    def Bk(self): return 2 * self.L                 # block: L pad slots, then L real slots
    @property
    def S(self): return self.slots // self.Bk       # sequences (blocks) per ciphertext
    @property
    def lgL(self): return int(np.log2(self.L))


ACT = (0.0, 0.5, 0.25)   # act(x) = 0.5 x + 0.25 x^2; p(0) = 0 keeps pad slots at zero
DECAY_DEG = 4


def cheb_lobatto(f, a, b, d):
    """Chebyshev interpolant of f on [a, b] at the d+1 Lobatto nodes (both endpoints
    exact), in OpenFHE's EvalChebyshevSeries convention: value = c0/2 + sum_j c_j T_j."""
    k = np.arange(d + 1)
    x = np.cos(np.pi * k / d)
    fk = f(0.5 * (b - a) * x + 0.5 * (a + b))
    w = np.ones(d + 1); w[0] = w[-1] = 0.5
    c = np.array([2.0 / d * np.sum(w * fk * np.cos(np.pi * j * k / d)) for j in range(d + 1)])
    c[-1] *= 0.5
    return c


def cheb_eval(c, a, b, x):
    c = np.array(c, dtype=float); c[0] *= 0.5
    return C.chebval((2 * x - a - b) / (b - a), c)


@dataclass
class Weights:
    Wx: np.ndarray; Wz: np.ndarray; WB: np.ndarray; WC: np.ndarray
    Wv: np.ndarray; bv: np.ndarray          # v = Wv u + bv; w = v^2 = |A_h| * Delta
    convx: np.ndarray; convB: np.ndarray; convC: np.ndarray       # (ch, K) taps, tap k reads t-k
    cbx: np.ndarray; cbB: np.ndarray; cbC: np.ndarray             # conv biases
    absA: np.ndarray; wmax: np.ndarray; decay: list               # per head
    a_pub: np.ndarray; selective: np.ndarray
    Dskip: np.ndarray; Wout: np.ndarray


def make_weights(s: Shape) -> Weights:
    r = np.random.default_rng(s.seed)
    g = lambda *sh: r.standard_normal(sh)
    D, Di, H, N, K = s.D, s.Di, s.H, s.N, s.K
    wmax = r.uniform(1.0, 6.0, H)
    absA = r.uniform(0.5, 4.0, H)
    sv = np.sqrt(wmax) / 2.0                       # keeps v^2 mostly inside [0, wmax]
    nsel = H if s.selective_heads < 0 else s.selective_heads
    return Weights(
        Wx=g(Di, D) / np.sqrt(D), Wz=g(Di, D) / np.sqrt(D),
        WB=g(N, D) / np.sqrt(D), WC=g(N, D) / np.sqrt(D),
        Wv=g(H, D) / np.sqrt(D) * sv[:, None], bv=0.1 * g(H) * sv,
        convx=0.5 * g(Di, K) / np.sqrt(K), convB=0.5 * g(N, K) / np.sqrt(K), convC=0.5 * g(N, K) / np.sqrt(K),
        cbx=0.1 * g(Di), cbB=0.1 * g(N), cbC=0.1 * g(N),
        absA=absA, wmax=wmax, decay=[cheb_lobatto(lambda w: np.exp(-w), 0.0, m, DECAY_DEG) for m in wmax],
        a_pub=np.exp(-wmax / 4.0), selective=np.arange(H) < nsel,
        Dskip=r.uniform(0.5, 1.5, H), Wout=g(D, Di) / np.sqrt(Di),
    )


def act(x):
    return ACT[0] + ACT[1] * x + ACT[2] * x * x


def reference(s: Shape, W: Weights, u: np.ndarray) -> np.ndarray:
    """Sequential forward pass, no packing. u: (D, L) -> (D, L)."""
    def conv(x, taps, bias):
        y = np.zeros_like(x)
        for k in range(s.K):
            y[:, k:] += taps[:, k:k + 1] * x[:, : s.L - k]
        return y + bias[:, None]

    xa = act(conv(W.Wx @ u, W.convx, W.cbx))
    Ba = act(conv(W.WB @ u, W.convB, W.cbB))
    Ca = act(conv(W.WC @ u, W.convC, W.cbC))
    za = act(W.Wz @ u)
    v = W.Wv @ u + W.bv[:, None]
    a = np.empty((s.H, s.L)); Dl = np.empty((s.H, s.L))
    for h in range(s.H):
        if W.selective[h]:
            w = v[h] ** 2
            a[h] = cheb_eval(W.decay[h], 0.0, W.wmax[h], w)
            Dl[h] = w / W.absA[h]
        else:
            a[h] = W.a_pub[h]; Dl[h] = 1.0
    head = np.arange(s.Di) // s.P
    y = np.zeros((s.Di, s.L))
    state = np.zeros((s.Di, s.N))
    for t in range(s.L):
        state = a[head, t][:, None] * state + (Dl[head, t] * xa[:, t])[:, None] * Ba[:, t][None, :]
        y[:, t] = state @ Ca[:, t] + W.Dskip[head] * xa[:, t]
    return W.Wout @ (y * za)


# ------------------------------------------------------------------- packed simulator

@dataclass
class Ct:
    v: np.ndarray
    level: int = 0       # multiplicative levels consumed


@dataclass
class Counter:
    ct_ct: int = 0; ct_pt: int = 0; ct_scalar: int = 0; rot: int = 0; add: int = 0
    rot_keys: set = field(default_factory=set)


class Packed:
    """The block's packed algorithm. Every method mirrors one function in bench.cpp."""

    def __init__(self, s: Shape, W: Weights):
        assert s.L & (s.L - 1) == 0, "L must be a power of two"
        assert s.S >= 1, "slots must hold at least one 2L block"
        assert s.P % s.S == 0, "each x ciphertext must lie inside one head (P % S == 0)"
        self.s, self.W, self.c = s, W, Counter()
        off = np.arange(s.slots) % s.Bk
        self.real = (off >= s.L) & (np.arange(s.slots) < s.S * s.Bk)

    # --- primitives (one per OpenFHE call)
    def rot(self, x, k):
        self.c.rot += 1; self.c.rot_keys.add(k)
        return Ct(np.roll(x.v, -k), x.level)

    def mul(self, x, y):
        self.c.ct_ct += 1
        return Ct(x.v * y.v, max(x.level, y.level) + 1)

    def mul_pt(self, x, p):
        self.c.ct_pt += 1
        return Ct(x.v * p, x.level + 1)

    def mul_scalar(self, x, a):
        self.c.ct_scalar += 1
        return Ct(x.v * a, x.level + 1)

    def add(self, x, y):
        self.c.add += 1
        return Ct(x.v + y.v, max(x.level, y.level))

    def add_pt(self, x, p):
        return Ct(x.v + p, x.level)

    # --- layout
    def block_vec(self, per_block_real):
        """Plaintext vector: per_block_real[b] (length L) on block b's real slots, 0 on pads."""
        s, out = self.s, np.zeros(self.s.slots)
        for b, vals in enumerate(per_block_real):
            out[b * s.Bk + s.L: (b + 1) * s.Bk] = vals
        return out

    def pack(self, seqs):
        s = self.s
        n = -(-len(seqs) // s.S)
        return [Ct(self.block_vec([seqs[j * s.S + b] if j * s.S + b < len(seqs) else np.zeros(s.L)
                                   for b in range(s.S)])) for j in range(n)]

    def unpack(self, cts, m):
        s = self.s
        return np.stack([cts[i // s.S].v[(i % s.S) * s.Bk + s.L: (i % s.S + 1) * s.Bk] for i in range(m)])

    # --- layers
    def block_rotations(self, X):
        """rot(X_i, r * Bk) for r = 0..S-1: computed once per input (hoisted in bench.cpp)
        and shared by every linear layer that reads X."""
        return [[x if r == 0 else self.rot(x, r * self.s.Bk) for r in range(self.s.S)] for x in X]

    def linear(self, rots, rows, bias=None):
        """rows: (n_out_cts * S, Min) weight rows, one per output block. Diagonal method over
        blocks: Y_j = sum_i sum_r diag(j,i,r) * rot(X_i, r * Bk). Pads of every diagonal are
        zero, so outputs have clean pads whatever the inputs hold."""
        s = self.s
        nout = rows.shape[0] // s.S
        Min = rows.shape[1]
        X = [r[0] for r in rots]
        Y = []
        for j in range(nout):
            acc = None
            for i in range(len(X)):
                for r in range(s.S):
                    coef = np.zeros(s.S)
                    for b in range(s.S):
                        col = i * s.S + (b + r) % s.S
                        if col < Min:
                            coef[b] = rows[j * s.S + b, col]
                    if not coef.any():
                        continue
                    p = self.block_vec([np.full(s.L, c) for c in coef])
                    t = self.mul_pt(rots[i][r], p)
                    acc = t if acc is None else self.add(acc, t)
            if bias is not None:
                acc = self.add_pt(acc, self.block_vec([np.full(s.L, bias[j * s.S + b]) for b in range(s.S)]))
            Y.append(acc)
        return Y

    def conv(self, X, taps, bias):
        """Causal depthwise conv. taps: (n_cts * S, K) aligned with X's blocks. Tap k reads t-k,
        i.e. a right shift: EvalRotate(ct, -k). Real slot t < k reads clean pad zeros; tap
        plaintexts are zero on pads, so output pads stay zero."""
        s = self.s
        Y = []
        for j, x in enumerate(X):
            acc = None
            for k in range(s.K):
                xs = x if k == 0 else self.rot(x, -k)
                p = self.block_vec([np.full(s.L, taps[j * s.S + b, k]) for b in range(s.S)])
                t = self.mul_pt(xs, p)
                acc = t if acc is None else self.add(acc, t)
            Y.append(self.add_pt(acc, self.block_vec([np.full(s.L, bias[j * s.S + b]) for b in range(s.S)])))
        return Y

    def act(self, x):
        """EvalPoly({0, c1, c2}): x^2 (1 level), then the constants (1 level)."""
        x2 = self.mul(x, x)
        return self.add(self.mul_scalar(x, ACT[1]), self.mul_scalar(x2, ACT[2]))

    def decay(self, w, h):
        """EvalChebyshevSeries of degree 4 on [0, wmax_h]: 3 levels in OpenFHE."""
        val = cheb_eval(self.W.decay[h], 0.0, self.W.wmax[h], w.v)
        return Ct(val, w.level + 3)

    def replicate(self, x):
        """Copy block content to every block: x has one nonzero block; log2 S rotate-and-adds
        by Bk * 2^i (no level). Needs slots == S * Bk so the rotation is cyclic over blocks."""
        s = self.s
        for i in range(int(np.log2(s.S))):
            x = self.add(x, self.rot(x, (1 << i) * s.Bk))
        return x

    def one_block(self, b, vals):
        return self.block_vec([vals if i == b else np.zeros(self.s.L) for i in range(self.s.S)])

    def forward(self, u, naive_inproj=False):
        s, W = self.s, self.W
        U = self.block_rotations(self.pack(u))
        nX = s.Di // s.S
        rep = lambda row: np.tile(row, (s.S, 1))
        sel = [h for h in range(s.H) if W.selective[h]]
        # in_proj: x and z per channel
        X = self.linear(U, W.Wx)
        Z = self.linear(U, W.Wz)
        if naive_inproj:
            # B, C, v each computed directly in replicated form: one full output ciphertext per row
            Brep = [self.linear(U, rep(W.WB[n]))[0] for n in range(s.N)]
            Crep = [self.linear(U, rep(W.WC[n]))[0] for n in range(s.N)]
            V = {h: self.linear(U, rep(W.Wv[h]), np.full(s.S, W.bv[h]))[0] for h in sel}
            Ba = [self.act(self.conv([Brep[n]], np.tile(W.convB[n], (s.S, 1)), np.full(s.S, W.cbB[n]))[0]) for n in range(s.N)]
            Ca = [self.act(self.conv([Crep[n]], np.tile(W.convC[n], (s.S, 1)), np.full(s.S, W.cbC[n]))[0]) for n in range(s.N)]
        else:
            # B, C, v once each, packed one per block in a small stream; then a block-selecting
            # conv (the selection is free: conv taps are a plaintext multiply anyway) and a
            # replication. v has no conv, so its selection is one plaintext multiply.
            rows = np.vstack([W.WB, W.WC] + [W.Wv[h][None, :] for h in sel])
            nsmall = -(-rows.shape[0] // s.S)
            rows = np.vstack([rows, np.zeros((nsmall * s.S - rows.shape[0], s.D))])
            bias = np.zeros(nsmall * s.S)
            for i, h in enumerate(sel):
                bias[2 * s.N + i] = W.bv[h]
            Sm = self.linear(U, rows, bias)
            SmR = [[c if k == 0 else self.rot(c, -k) for k in range(s.K)] for c in Sm]

            def conv_select(m, taps, cb):
                j, b = divmod(m, s.S)
                acc = None
                for k in range(s.K):
                    t = self.mul_pt(SmR[j][k], self.one_block(b, np.full(s.L, taps[k])))
                    acc = t if acc is None else self.add(acc, t)
                return self.replicate(self.add_pt(acc, self.one_block(b, np.full(s.L, cb))))

            Ba = [self.act(conv_select(n, W.convB[n], W.cbB[n])) for n in range(s.N)]
            Ca = [self.act(conv_select(s.N + n, W.convC[n], W.cbC[n])) for n in range(s.N)]
            V = {}
            for i, h in enumerate(sel):
                j, b = divmod(2 * s.N + i, s.S)
                V[h] = self.replicate(self.mul_pt(Sm[j], self.one_block(b, np.ones(s.L))))
        # conv + act
        Xa = [self.act(c) for c in self.conv(X, W.convx, W.cbx)]
        Za = [self.act(c) for c in Z]
        # decay and Delta (selective heads only)
        Acts, Dlt = {}, {}
        for h in sel:
            w = self.mul(V[h], V[h])
            Acts[h] = self.decay(w, h)
            Dlt[h] = self.mul_scalar(w, 1.0 / W.absA[h])
        head_of = lambda g: (g * s.S) // s.P
        # b_{n,g} = x * Delta * B
        XD = [self.mul(Xa[g], Dlt[head_of(g)]) if W.selective[head_of(g)] else Xa[g] for g in range(nX)]
        Bs = {(n, g): self.mul(XD[g], Ba[n]) for n in range(s.N) for g in range(nX)}
        # log-depth scan with 2^k right shifts; pads (length L) absorb wrap-around reads
        for k in range(s.lgL):
            sh = 1 << k
            for (n, g), b in Bs.items():
                h = head_of(g)
                if W.selective[h]:
                    Bs[(n, g)] = self.add(b, self.mul(Acts[h], self.rot(b, -sh)))
                else:
                    Bs[(n, g)] = self.add(b, self.mul_scalar(self.rot(b, -sh), W.a_pub[h] ** sh))
            for h in sel:
                Acts[h] = self.mul(Acts[h], self.rot(Acts[h], -sh))
        # readout, skip, gate
        G = []
        for g in range(nX):
            h = head_of(g)
            y = None
            for n in range(s.N):
                t = self.mul(Ca[n], Bs[(n, g)])
                y = t if y is None else self.add(y, t)
            y = self.add(y, self.mul_scalar(Xa[g], W.Dskip[h]))
            G.append(self.mul(y, Za[g]))
        return self.linear(self.block_rotations(G), W.Wout)


def run(s: Shape, verbose=True, naive_inproj=False):
    W = make_weights(s)
    r = np.random.default_rng(s.seed + 1)
    u = r.uniform(-1, 1, (s.D, s.L))
    ref = reference(s, W, u)
    pk = Packed(s, W)
    out = pk.forward(u, naive_inproj)
    got = pk.unpack(out, s.D)
    err = np.max(np.abs(got - ref)) / np.max(np.abs(ref))
    pads_clean = all(np.allclose(c.v[~pk.real], 0) for c in out)
    depth = max(c.level for c in out)
    if verbose:
        c = pk.c
        print(f"shape: slots {s.slots}, D {s.D}, L {s.L}, Di {s.Di}, heads {s.H} "
              f"(selective {int(W.selective.sum())}), P {s.P}, d_state {s.N}, S {s.S} blocks/ct")
        print(f"max rel error vs sequential reference: {err:.2e}   output pads clean: {pads_clean}")
        print(f"multiplicative depth: {depth}  (formula 9 + log2 L = {9 + s.lgL})")
        print(f"ops: ct-ct {c.ct_ct}, ct-pt {c.ct_pt}, ct-scalar {c.ct_scalar}, rotations {c.rot}, "
              f"adds {c.add}, rotation keys {len(c.rot_keys)}")
        print(f"output range: [{ref.min():.3f}, {ref.max():.3f}]")
    return err, pads_clean, depth


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe-shape", action="store_true", help="slots 32768, D 128, L 512, P 64, N 16")
    ap.add_argument("--selective-heads", type=int, default=-1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--naive-inproj", action="store_true", help="the first version: B, C, v replicated by in_proj")
    a = ap.parse_args()
    s = Shape(32768, 128, 512, 64, 16) if a.probe_shape else Shape()
    s.selective_heads, s.seed = a.selective_heads, a.seed
    assert s.S & (s.S - 1) == 0
    err, clean, _ = run(s, naive_inproj=a.naive_inproj)
    raise SystemExit(0 if err < 1e-9 and clean else 1)
