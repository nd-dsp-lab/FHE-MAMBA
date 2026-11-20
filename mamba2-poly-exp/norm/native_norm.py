"""FHE-native RMSNorm replacements beyond Path A's constant divisor.

    Path P -- polynomial rstd.   r = q(v / s) / sqrt(s),  q a learned degree-k polynomial
    Path D -- DyT.               out = gamma * tanh(alpha * x)   (Zhu et al., CVPR 2025)

Path A (norm/const_norm.py) drops the data dependence entirely: r = 1/c. These two
keep some of it at a fixed, small CKKS cost, and like Path A they change the
function, so the network is distilled to live with them (norm/distill_norm.py).

PATH P. v = mean(x^2) is still computed (1 level), then a low-degree polynomial in
v replaces 1/sqrt(v). s is a FIXED plaintext prescale (the layer's median v), and
1/sqrt(s) and gamma fold into the next weight, so CKKS sees q(v') with
v' = v / s folded into q's coefficients: depth ceil(log2(k + 1)) for q, plus 1
for x * r. The polynomial need not match 1/sqrt -- it is initialised to the
relative-error least-squares fit on the layer's [p1, p99] and then trained -- but
a low degree extrapolates gently, where the degree-119 fits of
fhe/ckks_invsqrt/design.py go to 2^85 just past their range.

PATH D. No mean, no reciprocal: out = gamma * tanh(alpha * x), one learned
scalar alpha per instance. Initialised to alpha = 1 / (k sqrt(median v)) and
gamma <- k gamma, so it starts as Path A's x / c and tanh bends only values past
k times the typical RMS. k = 1, DyT's literal "normalise then squash", sent a CPU
smoke test's pre-norms alone to ppl 9e17: Mamba's large-magnitude channels all
saturate. k = 8 by default. tanh is EXACT here, the
best a polynomial tanh could do; under CKKS it is an odd polynomial on a bounded
range, ~5-6 levels for degree 31, elementwise and fully SIMD. So this measures
whether DyT's quality survives conversion, not its FHE cost.

Both keep the exact SiLU at the gated site, as Path A does: this isolates the norm.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from norm.const_norm import round_to_bits


def _rstd_exact(xf, eps):
    return 1.0 / torch.sqrt(xf.square().mean(dim=-1, keepdim=True) + eps)


def fit_rstd_poly(p1, p99, s, degree):
    """Power-basis q with q(v/s) ~ sqrt(s / v) on v in [p1, p99], relative error."""
    u = np.exp(np.linspace(math.log(p1 / s), math.log(p99 / s), 512))
    w = np.sqrt(u)                                   # weight so the residual is relative
    V = np.vander(u, degree + 1, increasing=True)
    a, *_ = np.linalg.lstsq(V * w[:, None], np.ones_like(u), rcond=None)
    return a


class _NativeBase(nn.Module):
    """Exact / record plumbing shared with Path A, so exact_mode() and the aux loss work."""

    def __init__(self, weight, eps, gated, train_weight, precision_bits):
        super().__init__()
        self.weight = nn.Parameter(weight.detach().clone())
        self.weight.requires_grad_(train_weight)
        self.register_buffer("weight_orig", weight.detach().clone())
        self.eps, self.gated = float(eps), gated
        self.precision_bits = precision_bits
        self.exact = False
        self.record = False
        self.last_out = None

    def _approx(self, xf):
        raise NotImplementedError

    def forward(self, x, z=None):
        dtype = x.dtype
        xf = x.float()
        if self.gated:
            xf = xf * F.silu(z.float())
        if self.exact:
            out = xf * _rstd_exact(xf, self.eps) * self.weight_orig.float()
        else:
            out = self._approx(xf) * self.weight.float()
            if self.precision_bits:
                out = round_to_bits(out, self.precision_bits)
        if self.record:
            self.last_out = out
        return out.to(dtype)


class PolyRstdNorm(_NativeBase):
    """Path P: r = q(v / s) / sqrt(s), q learned, degree `degree`."""

    def __init__(self, weight, s, p1, p99, degree=2, eps=1e-5, gated=False,
                 train_weight=True, precision_bits=None, init="fit"):
        super().__init__(weight, eps, gated, train_weight, precision_bits)
        self.degree = int(degree)
        self.register_buffer("s", torch.tensor(float(s)))
        if init == "const":
            # q = 1: r = 1/sqrt(s), Path A's starting constant (c = sqrt(median v));
            # the higher coefficients start at 0 and are learned as a correction
            a = np.zeros(self.degree + 1)
            a[0] = 1.0
        else:
            a = fit_rstd_poly(p1, p99, s, self.degree)
        self.poly_a = nn.Parameter(torch.tensor(a, dtype=torch.float32))

    @property
    def ct_ct_depth(self) -> int:
        # q of degree k (coefficients absorb 1/s), then x * r
        return math.ceil(math.log2(self.degree + 1)) + 1

    def _approx(self, xf):
        u = (xf.square().mean(dim=-1, keepdim=True) + self.eps) / self.s
        r = torch.zeros_like(u)
        for a in reversed(self.poly_a):              # Horner; CKKS would use PS
            r = r * u + a
        self.last_rstd = r
        return xf * (r / torch.sqrt(self.s))

    def extra_repr(self):
        a = ", ".join(f"{float(x):.3g}" for x in self.poly_a)
        return f"degree={self.degree}, s={float(self.s):.4g}, a=[{a}], ct_ct_depth={self.ct_ct_depth}"


class DyTNorm(_NativeBase):
    """Path D: gamma * tanh(alpha * x), alpha a learned scalar."""

    def __init__(self, weight, alpha_init, k=8.0, eps=1e-5, gated=False, train_weight=True,
                 precision_bits=None):
        super().__init__(weight, eps, gated, train_weight, precision_bits)
        self.log_alpha = nn.Parameter(torch.tensor(math.log(alpha_init / k), dtype=torch.float32))
        with torch.no_grad():
            self.weight.mul_(k)                      # weight_orig (the teacher) stays as is

    @property
    def alpha(self):
        return self.log_alpha.exp()

    @property
    def ct_ct_depth(self) -> int:
        return 6        # alpha * x, then a degree-31 odd polynomial tanh (not modelled here)

    def _approx(self, xf):
        return torch.tanh(self.alpha * xf)

    def extra_repr(self):
        return f"alpha={float(self.alpha):.4g}, tanh exact, ct_ct_depth~{self.ct_ct_depth}"


def install_native(model, stats, stage="A3", kind="P", degree=2, dyt_k=8.0,
                   precision_bits=None, train_weight=True, verbose=True, poly_init="fit"):
    """Swap in Path P or Path D for the sites in `stage` (A1 / A2 / A3, as Path A)."""
    from norm.install_norms import STAGES, NormHandle, _unfuse_add_norm, find_norm_sites

    if stage not in STAGES:
        raise ValueError(f"stage must be one of {sorted(STAGES)}")
    sites = find_norm_sites(model)
    h = NormHandle(model=model, stage=stage)
    _unfuse_add_norm(model, h)
    holder, missing = {}, []
    for site in STAGES[stage]:
        for layer, (parent, attr, old) in sorted(sites[site].items()):
            st = (stats.get(site) or {}).get(str(layer))
            if st is None:
                missing.append(f"{site}[{layer}]")
                continue
            gated = site == "norm_gated"
            eps = getattr(old, "eps", 1e-5)
            if kind == "P":
                new = PolyRstdNorm(old.weight.data, st["median"], st["p1"], st["p99"],
                                   degree=degree, eps=eps, gated=gated,
                                   train_weight=train_weight, precision_bits=precision_bits,
                                   init=poly_init)
            elif kind == "D":
                new = DyTNorm(old.weight.data, 1.0 / st["sqrt_median"], k=dyt_k, eps=eps,
                              gated=gated,
                              train_weight=train_weight, precision_bits=precision_bits)
            else:
                raise ValueError(f"kind must be P or D, not {kind!r}")
            if gated:
                h.gated_overrides[layer] = new
            else:
                setattr(parent, attr, new)
                h.originals.append((parent, attr, old))
            holder[f"{site}_{layer}"] = new
    if missing:
        # Path A once fell back to c = 1.0 in silence for half the sites; never again
        raise RuntimeError(f"no norm statistics for {len(missing)} instances: "
                           f"{', '.join(missing[:6])}; run norm/collect_norm_stats.py")
    h.replacements = nn.ModuleDict(holder)
    ref = next(model.parameters())
    model.norm_replacements = h.replacements.to(device=ref.device)
    if verbose:
        m = next(iter(holder.values()))
        print(f"[norm] stage {stage}: replaced {len(holder)} instances with "
              f"{type(m).__name__}; ct-ct depth {m.ct_ct_depth} each")
    return h


# =============================================================================
# Hybrid: Path A on the pre-norms, the CKKS Chebyshev 1/sqrt on gated + norm_f
# =============================================================================

class _ChebSTE(torch.autograd.Function):
    """Forward: the Chebyshev series exactly as CKKS evaluates it (clamped).
    Backward: d/dv of the true 1/sqrt at the clamped v, zero where the clamp bit.

    The series is >= 12 bits accurate on [a, b], so its slope is the true slope to
    the same order; differentiating Clenshaw through a degree-2031 series instead
    (norm_gated.0) would cost ~2k autograd nodes per call for no gain."""

    @staticmethod
    def forward(ctx, v, cheb):
        with torch.no_grad():
            y = cheb(v)
        inside = (v >= cheb.a) & (v <= cheb.b)
        ctx.save_for_backward(v.clamp(cheb.a, cheb.b), inside)
        return y

    @staticmethod
    def backward(ctx, g):
        vc, inside = ctx.saved_tensors
        return g * (-0.5 * vc.pow(-1.5)) * inside, None


class CkksChebNorm(_NativeBase):
    """RMSNorm with r = ChebInvSqrt(v): the CKKS-only 1/sqrt with the ideal range clamp.

    [a, b], degree and depth are the ones job_ckks_invsqrt.sh's ckks_t12_m05_g used
    (calibrated on the exact model, train split, L = 8192, log margin 0.05): the
    configuration that scored +0.0005 ppl at L = 8192 with all 49 norms. They stay
    FIXED while the backbone trains, as an FHE circuit's would; the clamp catches
    whatever the training moves out of range, and `cheb.n_guarded` counts it."""

    def __init__(self, weight, a=None, b=None, target_bits=12.0, eps=1e-5, gated=False,
                 train_weight=True, precision_bits=None, name=None):
        super().__init__(weight, eps, gated, train_weight, precision_bits)
        self.target_bits, self._name = target_bits, name
        self.calibrating = False
        self.register_buffer("v_min", torch.tensor(float("inf")))
        self.register_buffer("v_max", torch.tensor(float("-inf")))
        self.cheb = None
        # range-aware training (job_norm_hybrid_range.sh). oob_exact: while TRAINING,
        # an input outside [a, b] gets the exact 1/sqrt instead of the clamp, so one
        # stray token cannot send the run to NaN (job 1756344, step 37); eval always
        # clamps, as FHE would. range_inner > 0 records a hinge penalty pushing log v
        # into [a, b] shrunk by that fraction of its log width on each side.
        self.oob_exact, self.range_inner = False, None
        self.last_range_pen, self.last_oob = None, None
        # eval_oob_exact: the same fallback at EVAL. An oracle, not FHE: it shows the
        # PRIMARY out-of-range rate, with no clamp perturbing later blocks.
        # diag: record how far v leaves [a, b] on each side (diag_reset / diag_stats).
        self.eval_oob_exact, self.diag = False, False
        self.diag_reset()
        # calibration histogram of log10 v, 0.05-decade bins on [-10, 10]
        self.register_buffer("v_hist", torch.zeros(self.HIST_BINS), persistent=False)
        if a is None:
            self.exact = True               # until calibrate_hybrid() builds the series
        else:
            self.build(a, b, margin=0.0)

    HIST_BINS, HIST_LO, HIST_HI = 400, -10.0, 10.0

    def v_quantile(self, q):
        """Lower edge of the histogram bin holding calibration quantile q (conservative)."""
        h = self.v_hist.double()
        cdf = h.cumsum(0) / h.sum().clamp_min(1)
        i = int(torch.searchsorted(cdf, torch.tensor(float(q), dtype=torch.float64,
                                                     device=cdf.device)))
        w = (self.HIST_HI - self.HIST_LO) / self.HIST_BINS
        return max(10.0 ** (self.HIST_LO + w * min(i, self.HIST_BINS - 1)), float(self.v_min))

    def build(self, lo, hi, margin, hi_factor=None, lo_factor=1.0, lo_quantile=None):
        """Fit the series on [lo, hi] widened by `margin` (log, symmetric), as seglut's ckks
        impl does; or, with hi_factor, ASYMMETRICALLY: b = hi * hi_factor and
        a = (calibration quantile lo_quantile, else lo) / lo_factor.

        Why asymmetric: a clamp above b leaves r = 1/sqrt(b) while x keeps growing, so
        the output's rms is sqrt(v / b) > 1 and the next blocks see inflated inputs (job
        1768266: inf at L = 2048 with 1.8e-3 clamp hits). Below a the output's rms is
        sqrt(v / a) < 1: it shrinks, which does not compound. So spend the range where
        overflow is fatal, and raise a, which is cheap, to keep b / a fittable."""
        from real_mamba.ckks_invsqrt import ChebInvSqrt
        if hi_factor is not None:
            base = self.v_quantile(lo_quantile) if lo_quantile else lo
            lo, hi, margin = base / lo_factor, hi * hi_factor, 0.0
        self.cheb = ChebInvSqrt(lo, hi, margin=margin, target_bits=self.target_bits,
                                range_guard=True, name=self._name).to(self.v_min.device)
        self.exact = False

    def diag_reset(self):
        self.d_n = self.d_hi = self.d_lo = 0
        self.d_max_over_b, self.d_min_over_a = 0.0, float("inf")

    def diag_stats(self):
        n = max(self.d_n, 1)
        return {"n": self.d_n, "hi_frac": self.d_hi / n, "lo_frac": self.d_lo / n,
                "max_v_over_b": self.d_max_over_b, "min_v_over_a": self.d_min_over_a}

    def forward(self, x, z=None):
        if self.calibrating:
            with torch.no_grad():
                xf = x.float() * (F.silu(z.float()) if self.gated else 1.0)
                v = xf.square().mean(dim=-1) + self.eps
                self.v_min = torch.minimum(self.v_min, v.min())
                self.v_max = torch.maximum(self.v_max, v.max())
                self.v_hist += torch.histc(v.log10().clamp(self.HIST_LO, self.HIST_HI - 1e-6),
                                           self.HIST_BINS, self.HIST_LO, self.HIST_HI)
        return super().forward(x, z)

    @property
    def ct_ct_depth(self) -> int:
        # v = mean(x^2), the series (+ Newton), x * r; the clamp's comparisons are extra
        return 1 + self.cheb.depth + 1

    def _approx(self, xf):
        v = xf.square().mean(dim=-1, keepdim=True) + self.eps
        r = _ChebSTE.apply(v, self.cheb)
        a, b = self.cheb.a, self.cheb.b
        if self.diag:
            with torch.no_grad():
                self.d_n += v.numel()
                self.d_hi += int((v > b).sum())
                self.d_lo += int((v < a).sum())
                self.d_max_over_b = max(self.d_max_over_b, float(v.max()) / b)
                self.d_min_over_a = min(self.d_min_over_a, float(v.min()) / a)
        fallback = self.oob_exact if self.training else self.eval_oob_exact
        if fallback or (self.training and self.range_inner is not None):
            out_of = (v < a) | (v > b)
            self.last_oob = out_of.detach()
            if fallback:
                r = torch.where(out_of, v.rsqrt(), r)
            if self.training and self.range_inner is not None:
                la, lb = math.log(a), math.log(b)
                w = self.range_inner * (lb - la)
                lv = v.log()
                h = F.relu(lv - (lb - w)) + F.relu((la + w) - lv)
                # summed over positions, per sequence: the scale of the KD term, so a
                # rare far-out token is not averaged away
                self.last_range_pen = h.square().sum() / v.shape[0]
        self.last_rstd = r
        return xf * r

    def extra_repr(self):
        return f"{self.cheb.extra_repr()}, ct_ct_depth={self.ct_ct_depth} + clamp"


def install_hybrid(model, stats, ranges=None, target_bits=12.0, precision_bits=None,
                   train_weight=True, verbose=True):
    """All 49 norms replaced: ConstDivisorNorm on the 24 pre-norms (0 levels),
    CkksChebNorm on the 24 gated norms and norm_f.

    `ranges` is a seglut_paired.py JSON's `ckks_choices`: {'norm_gated.3': {a, b,
    degree, ...}}. The degree is re-derived from (a, b, target_bits) and checked.
    ranges=None leaves the CKKS sites exact until calibrate_hybrid() is run.

    Ranges calibrated on the EXACT model do not survive constant pre-norms: the
    mixer's input scale moves, gated-norm v leaves [a, b], and a clamped 1/sqrt
    stops normalising above b (r = 1/sqrt(b) while x keeps growing), so the 24
    blocks compound to inf again. Calibrate on the model the circuit will run."""
    from norm.const_norm import ConstDivisorNorm
    from norm.install_norms import STAGES, NormHandle, _unfuse_add_norm, find_norm_sites

    sites = find_norm_sites(model)
    h = NormHandle(model=model, stage="H")
    _unfuse_add_norm(model, h)
    holder, missing = {}, []
    for site in STAGES["A3"]:
        for layer, (parent, attr, old) in sorted(sites[site].items()):
            eps = getattr(old, "eps", 1e-5)
            if site == "norm_pre":
                st = (stats.get(site) or {}).get(str(layer))
                if st is None:
                    missing.append(f"stats {site}[{layer}]")
                    continue
                new = ConstDivisorNorm(old.weight.data, st["sqrt_median"], eps=eps,
                                       train_weight=train_weight,
                                       precision_bits=precision_bits)
            elif ranges is None:
                new = CkksChebNorm(old.weight.data, None, None, target_bits, eps=eps,
                                   gated=site == "norm_gated", train_weight=train_weight,
                                   precision_bits=precision_bits, name=f"{site}.{layer}")
            else:
                rg = ranges.get(f"{site}.{layer}")
                if rg is None:
                    missing.append(f"range {site}.{layer}")
                    continue
                new = CkksChebNorm(old.weight.data, rg["a"], rg["b"], target_bits,
                                   eps=eps, gated=site == "norm_gated",
                                   train_weight=train_weight,
                                   precision_bits=precision_bits, name=f"{site}.{layer}")
                if "degree" in rg and new.cheb.degree != rg["degree"]:
                    raise RuntimeError(f"{site}.{layer}: degree {new.cheb.degree} != "
                                       f"{rg['degree']} in the ranges file")
            if site == "norm_gated":
                h.gated_overrides[layer] = new
            else:
                setattr(parent, attr, new)
                h.originals.append((parent, attr, old))
            holder[f"{site}_{layer}"] = new
    if missing:
        raise RuntimeError(f"hybrid: {len(missing)} instances missing: {', '.join(missing[:6])}")
    h.replacements = nn.ModuleDict(holder)
    ref = next(model.parameters())
    model.norm_replacements = h.replacements.to(device=ref.device)
    if verbose and ranges is None:
        print(f"[norm] hybrid: 24 pre-norms -> constant divisor; gated/f exact until "
              f"calibrated")
    elif verbose:
        _report_hybrid(holder, target_bits)
    return h


def _report_hybrid(holder, target_bits):
    ck = [m for m in holder.values() if isinstance(m, CkksChebNorm)]
    d = [m.ct_ct_depth for m in ck]
    print(f"[norm] hybrid: {len(holder) - len(ck)} pre-norms -> constant divisor "
          f"(0 levels); {len(ck)} gated/f -> CKKS Chebyshev, clamped, "
          f">= {target_bits:g} bits, degree {min(m.cheb.degree for m in ck)}-"
          f"{max(m.cheb.degree for m in ck)}, depth {min(d)}-{max(d)} + clamp")


@torch.no_grad()
def calibrate_hybrid(handle, model, blocks, device, margin=0.05, target_bits=12.0,
                     hi_factor=None, lo_factor=1.0, lo_quantile=None):
    """Record each CKKS site's v range on `blocks` with the pre-norms as installed
    (constant) and the CKKS sites exact, then fit each series on that range
    (symmetric log margin, or asymmetric with hi_factor: see CkksChebNorm.build).
    Returns {site.layer: {a, b, degree, newton, depth}} in ckks_choices' format."""
    ck = {k: m for k, m in handle.replacements.items() if isinstance(m, CkksChebNorm)}
    was = model.training
    model.eval()
    for m in ck.values():
        m.exact, m.calibrating = True, True
        m.v_hist.zero_()
    for i in range(blocks.shape[0]):
        model(blocks[i:i + 1].to(device))
    out = {}
    for k, m in ck.items():
        m.calibrating = False
        if not torch.isfinite(m.v_max):
            raise RuntimeError(f"{k}: no finite calibration samples")
        m.build(float(m.v_min), float(m.v_max), margin, hi_factor=hi_factor,
                lo_factor=lo_factor, lo_quantile=lo_quantile)
        site, layer = k.rsplit("_", 1)
        out[f"{site}.{layer}"] = {"a": m.cheb.a, "b": m.cheb.b, "v_min": float(m.v_min),
                                  "v_max": float(m.v_max), "degree": m.cheb.degree,
                                  "newton": m.cheb.newton, "depth": m.cheb.depth,
                                  "design_worst_bits": m.cheb.design_bits,
                                  "v_q": {str(q): m.v_quantile(q)
                                          for q in (1e-4, 1e-3, 1e-2, 0.1, 0.5)}}
    model.train(was)
    _report_hybrid(dict(handle.replacements.items()), target_bits)
    return out
