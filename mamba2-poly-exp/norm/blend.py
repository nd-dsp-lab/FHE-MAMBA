"""Ramp a norm replacement in during distillation, instead of switching it on cold.

Replacing all 49 norms at step 0 sent Paths A and P to NaN before the first update
(jobs 1713635/1713636, stage A3). Here every installed replacement computes

    out = beta * replacement(x) + (1 - beta) * exact_rmsnorm(x)

with beta per site family ramped 0 -> 1 over a window of training steps, gated norms
first, then the pre-norms, then norm_f. At beta = 1 the forward is the replacement
alone, which is the only thing evaluated and the only thing FHE would run; the blend
exists during training only.

While recording, each site also keeps `last_exact`: the exact norm of the SAME input
the replacement saw. That is the local target for the auxiliary loss. The shared-
weight teacher's norm outputs stop being a valid target once the backbone trains.
"""

from __future__ import annotations

import re

# fraction-of-training windows, one per family: gated first, f last
DEFAULT_RAMP = "norm_gated:0.0-0.2,norm_pre:0.2-0.4,norm_f:0.4-0.5"


def parse_ramp(spec):
    """'fam:a-b,...' -> {fam: (a, b)}; 'none' -> {} (every beta stays 1)."""
    if not spec or spec == "none":
        return {}
    out = {}
    for part in spec.split(","):
        fam, win = part.split(":")
        a, b = (float(x) for x in win.split("-"))
        if not 0.0 <= a <= b <= 1.0:
            raise ValueError(f"bad ramp window {part!r}")
        out[fam.strip()] = (a, b)
    return out


def _family(key):
    m = re.match(r"(norm_pre|norm_gated|norm_f)_", key)
    return m.group(1) if m else None


def enable_blend(handle):
    """Wrap every replacement's forward; beta starts at 1 (pure replacement)."""
    for key, m in handle.replacements.items():
        if getattr(m, "_blend_wrapped", False):
            continue
        orig = m.forward
        m.blend, m.family, m.last_exact = 1.0, _family(key), None

        def fwd(*a, _m=m, _orig=orig, **k):
            if _m.exact:
                return _orig(*a, **k)
            need_exact = _m.blend < 1.0 or _m.record
            if need_exact:
                rec = _m.record
                _m.exact, _m.record = True, False
                e = _orig(*a, **k)
                _m.exact, _m.record = False, rec
                if rec:
                    _m.last_exact = e.detach()
            if _m.blend <= 0.0:
                if _m.record:
                    _m.last_out = e
                return e
            s = _orig(*a, **k)
            if _m.blend >= 1.0:
                return s
            out = (_m.blend * s.float() + (1.0 - _m.blend) * e.float()).to(s.dtype)
            if _m.record:
                _m.last_out = out
            return out

        m.forward = fwd
        m._blend_wrapped = True


def set_blend(handle, frac, ramp, control=False):
    """beta for every site at training progress `frac` in [0, 1]. control: beta = 0."""
    for m in handle.replacements.values():
        if control:
            m.blend = 0.0
            continue
        a, b = ramp.get(m.family, (0.0, 0.0))
        m.blend = 1.0 if frac >= b else 0.0 if frac < a else (frac - a) / max(b - a, 1e-9)


def set_all(handle, beta):
    for m in handle.replacements.values():
        m.blend = beta


def local_targets(handle):
    """{site: exact norm of the replacement's own input}, from the last recorded forward."""
    return {k: m.last_exact for k, m in handle.replacements.items()
            if getattr(m, "last_exact", None) is not None}
