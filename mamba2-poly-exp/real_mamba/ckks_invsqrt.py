"""1/sqrt(v) as fhe/ckks_invsqrt/bench.cpp computes it, simulated in PLAINTEXT.

The CKKS-only route: per norm instance, a Chebyshev series p(v) ~ 1/sqrt(v) on the
calibrated range widened by `margin` (log), then `newton` steps y <- y (3 - v y^2) / 2.
Degree and Newton count are fhe/ckks_invsqrt/design.py's cheapest pick for
`target_bits` worst-case relative bits, the same selection that wrote cases.txt
and was benchmarked in OpenFHE. No comparisons, no table, no scheme switch.

What this shows that the benchmark cannot: the model's own inputs. Inside [a, b]
the error is the fit's; OUTSIDE it the series extrapolates, as it would under
encryption, and nothing clamps it. `n_oob` counts those inputs; `n_neg` counts
outputs <= 0, which RMSNorm turns into sign flips.

    1. t = (2v - a - b) / (b - a)                         the affine map EvalChebyshevSeries applies
    2. y = sum_k c_k T_k(t)                               Clenshaw, float64
    3. y <- y (3 - v y^2) / 2, `newton` times              CKKS, 3 levels per step
    4. y (1 + e_ckks), e_ckks ~ N(0, 2^-ckks_bits)          CKKS's noise floor

`range_guard` clamps v to [a, b] before step 1, and Newton sees the clamped v too,
as SchemeSwitchInvSqrt's guard does: an off-range input gets 1/sqrt of the nearest
end instead of an extrapolated value. This is the IDEAL clamp. In CKKS it is two
comparisons (sign polynomials), whose levels and error are not modelled here.
`n_guarded` counts the clamped elements; `n_oob` still counts them on the raw input.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "fhe" / "ckks_invsqrt"))
import design  # noqa: E402  (fhe/ckks_invsqrt/design.py)


class ChebInvSqrt(nn.Module):
    """One Chebyshev 1/sqrt for one norm instance; input v > 0, any shape."""

    def __init__(self, lo, hi, margin=0.05, target_bits=12.0, ckks_bits=None,
                 range_guard=False, seed=0, name=None):
        super().__init__()
        a, b = design.widen_log(float(lo), float(hi), margin)
        pick = design.cheapest(design.sweep(a, b), target_bits)
        if pick is None:
            raise ValueError(f"{name}: no degree <= 2031 reaches {target_bits} bits on "
                             f"[{a:.4g}, {b:.4g}] (R = {b / a:.3g})")
        self.a, self.b = a, b
        self.degree, self.newton = pick["degree"], pick["newton"]
        self.depth, self.design_bits = pick["depth"], pick["worst_bits"]
        self.ckks_bits = ckks_bits
        self.range_guard = bool(range_guard)
        self.seed = seed
        self._name = name or "cheb_invsqrt"
        self.register_buffer("coef", torch.as_tensor(design.fit(a, b, self.degree),
                                                     dtype=torch.float64))
        self._gen = None
        self.track = False
        self.reset_counters()

    @property
    def name(self):
        return self._name

    def reset_counters(self):
        self.n_seen = self.n_oob = self.n_neg = self.n_gt1 = self.n_nonfinite = 0
        self.n_guarded = 0

    def reseed(self, seed=None):
        self.seed = self.seed if seed is None else seed
        self._gen = None

    def _randn(self, like, std):
        if self._gen is None or self._gen.device != like.device:
            self._gen = torch.Generator(device=like.device)
            self._gen.manual_seed(int(self.seed))
        return torch.randn(like.shape, generator=self._gen, device=like.device,
                           dtype=like.dtype) * std

    def forward(self, v):
        dtype = v.dtype
        vd = v.double()
        bad = ~torch.isfinite(vd)
        if bad.any():
            vd = torch.where(bad, torch.full_like(vd, self.a), vd)
        oob = (vd < self.a) | (vd > self.b)
        if self.range_guard:
            vd = vd.clamp(self.a, self.b)
        t = (2.0 * vd - self.a - self.b) / (self.b - self.a)
        # Clenshaw: b_k = c_k + 2 t b_{k+1} - b_{k+2};  p = c_0 + t b_1 - b_2
        b1 = torch.zeros_like(t)
        b2 = torch.zeros_like(t)
        for c in self.coef.flip(0)[:-1]:
            b1, b2 = c + 2.0 * t * b1 - b2, b1
        y = self.coef[0] + t * b1 - b2
        for _ in range(self.newton):
            y = y * (3.0 - vd * y * y) / 2.0
        if self.ckks_bits:
            y = y * (1.0 + self._randn(y, 2.0 ** -float(self.ckks_bits)))
        y = torch.where(bad, float("nan"), y)
        if self.track:
            self.n_seen += vd.numel()
            self.n_nonfinite += int(bad.sum())
            self.n_oob += int(oob.sum())
            self.n_guarded += int(oob.sum()) if self.range_guard else 0
            self.n_neg += int((y <= 0).sum())
        return y.to(dtype)

    @torch.no_grad()
    def precision(self, n=4001, seed=0):
        """Same contract as SchemeSwitchInvSqrt.precision; log-uniform on [a, b] as design.py samples."""
        saved = self._gen, self.track, self.seed
        self._gen, self.track, self.seed = None, False, seed
        try:
            xs = torch.exp(torch.linspace(math.log(self.a), math.log(self.b), n,
                                          dtype=torch.float64)).to(self.coef.device)
            y = self.forward(xs)
        finally:
            self._gen, self.track, self.seed = saved
        exact = xs.rsqrt()
        err, rel = (y - exact).abs(), (y - exact) / exact
        return {"bits": np.array([float(-torch.log2(err.square().mean().sqrt()))]),
                "rel_bits": np.array([float(-torch.log2(rel.square().mean().sqrt()))]),
                "rel_bits_worst": np.array([float(-torch.log2(rel.abs().max()))]),
                "max_abs": np.array([float(err.max())]),
                "max_rel": np.array([float(rel.abs().max())])}

    def extra_repr(self):
        return (f"[{self.a:.4g}, {self.b:.4g}], degree={self.degree}, newton={self.newton}, "
                f"depth={self.depth}, design worst bits={self.design_bits:.1f}, "
                f"range guard {'on' if self.range_guard else 'off'}")
