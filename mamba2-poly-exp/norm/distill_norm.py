#!/usr/bin/env python
"""PHASE 2/3 -- distil the norm replacement back to teacher quality.

    python norm/distill_norm.py --stage A3 --token-budget 100000

The teacher is the untouched pretrained model. The student is the SAME weights
with the norm operator replaced. They share every parameter except the norm, so
only one model is in memory and the two provably differ in nothing else --
verified: teacher mode reproduces the untouched model bit-for-bit.

    L = lambda_lm * CE(student, labels)
      + lambda_kd * T^2 * KL(student/T || teacher/T)
      + lambda_aux * mean_over_sites  MSE(student_norm_out, teacher_norm_out)
                                      / mean(teacher_norm_out^2)

The auxiliary term is optional and is reported both with and without, as the
brief asks. It is a *local* signal at exactly the operator being replaced, which
the logit-level KL only sees after 24 layers of mixing. It is normalised by the
teacher's own scale so one lambda is meaningful across all 49 sites, whose
magnitudes differ by orders of magnitude.

Evaluated at BOTH sequence lengths every stage. A method that passes 512 and
fails 2048 is a drift result and is reported as such, never averaged.

OPTIMISED ROUND (--unfreeze / --ramp / --control). The first native round trained
only the replacements, on 1M tokens, cold, and was +20 ppl or worse everywhere.
  --unfreeze adjacent   also train each mixer's in_proj and out_proj, the linears the
                        pre-norm and gated norm feed (all: the whole backbone). The
                        teacher is then a SEPARATE frozen copy, since a shared-weight
                        teacher would drift with the student.
  --ramp                blend each family in (norm/blend.py), gated, then pre, then f,
                        instead of swapping all 49 at step 0. The aux target becomes
                        the exact norm of each replacement's own input.
  --control             the same run with every beta held at 0 (exact norms): what the
                        extra training alone does to perplexity. Read the replacement
                        against this, not only against the teacher.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from baby_mamba.transition import ExactExp
from norm.install_norms import (
    STAGES,
    exact_mode,
    install_path_a,
    install_path_b,
    norm_kwargs_for_mixer,
    norm_outputs,
    recording,
)
from real_mamba.data import get_blocks
from real_mamba.eval_lm import peak_memory_gb, reset_peak_memory
from real_mamba.model import DEFAULT_MODEL, load_model, recommended_dtype
from real_mamba.patch import mamba2_reference_forward, patch_transition
from real_mamba.train_utils import RunLogger, save_json, set_seed, token_budget_batches


def bind_forwards(handle, mixers, chunk_size):
    """Install the reference forward with this stage's gated-norm override."""
    for mixer in mixers:
        kw = norm_kwargs_for_mixer(handle, mixer.layer_idx)

        def bound(u, _m=mixer, _kw=kw, inference_params=None, **extra):
            return mamba2_reference_forward(_m, u, _m._transition,
                                             chunk_size=chunk_size, **_kw, **extra)
        mixer.forward = bound


@torch.no_grad()
def evaluate_ppl(model, blocks, device, vocab_size, batch_size=1):
    tot, ntok = 0.0, 0
    for i in range(0, blocks.shape[0], batch_size):
        ids = blocks[i:i + batch_size].to(device)
        lg = model(ids).logits[:, :-1].float()[..., :vocab_size]
        tot += float(F.cross_entropy(lg.reshape(-1, lg.shape[-1]),
                                     ids[:, 1:].reshape(-1), reduction="sum"))
        ntok += ids[:, 1:].numel()
    m = tot / ntok
    return (math.exp(m) if m < 700 else float("inf")), m


def aux_loss(student_outs, teacher_outs):
    """Scale-free MSE at the replaced sites."""
    terms = []
    for k, t in teacher_outs.items():
        s = student_outs.get(k)
        if s is None:
            continue
        tf = t.float()
        terms.append(F.mse_loss(s.float(), tf) / (tf.square().mean() + 1e-8))
    if not terms:
        return None
    return torch.stack(terms).mean()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--backend", default="auto", choices=("auto", "official", "local"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="float32",
                    choices=("auto", "float32", "bfloat16", "float16"))
    ap.add_argument("--path", default="A", choices=("A", "B", "P", "D", "H"),
                    help="A = learned constant divisor (0 levels). "
                         "B = prescaled Newton inverse sqrt (2/5/8 levels for t=1/2/3). "
                         "P = learned low-degree polynomial in v (norm/native_norm.py). "
                         "D = DyT, gamma * tanh(alpha * x) (norm/native_norm.py). "
                         "H = hybrid, all 49: A on the pre-norms, the clamped CKKS "
                         "Chebyshev 1/sqrt on gated + norm_f (--stage ignored).")
    ap.add_argument("--ckks-ranges", type=Path, default=None,
                    help="Path H: seglut_paired.py JSON whose ckks_choices give each "
                         "norm's [a, b] (calibrated on the EXACT model). Default: "
                         "calibrate on this model instead (--calib-*)")
    ap.add_argument("--calib-data", default=None, help="Path H; default --train-data")
    ap.add_argument("--calib-seq-len", type=int, default=8192)
    ap.add_argument("--calib-blocks", type=int, default=6)
    ap.add_argument("--calib-margin", type=float, default=0.05)
    ap.add_argument("--calib-hi-factor", type=float, default=None,
                    help="Path H: ASYMMETRIC range, b = calibrated max * this (overflow "
                         "above b compounds; below a it only shrinks). Replaces --calib-margin")
    ap.add_argument("--calib-lo-factor", type=float, default=1.0,
                    help="with --calib-hi-factor: a = (calibrated min or quantile) / this")
    ap.add_argument("--calib-lo-quantile", type=float, default=None,
                    help="with --calib-hi-factor: a from this calibration quantile of v, not "
                         "the min (raises a to keep b / a fittable)")
    ap.add_argument("--eval-oob", default="clamp", choices=("clamp", "exact"),
                    help="Path H, final eval: 'exact' gives out-of-range v the exact 1/sqrt "
                         "(an ORACLE, not FHE) to measure the primary out-of-range rate")
    ap.add_argument("--range-inner", type=float, default=None,
                    help="Path H: with --lambda-range > 0, penalise log v outside each "
                         "CKKS range shrunk by this fraction of its log width per side")
    ap.add_argument("--train-oob", default="clamp", choices=("clamp", "exact"),
                    help="Path H, training only: out-of-range v gets the clamp (as FHE) "
                         "or the exact 1/sqrt (cannot cascade to NaN). Eval always clamps.")
    ap.add_argument("--init-from", type=Path, default=None,
                    help="a trained_state.pt (--save-weights) to start from, e.g. A_A1's")
    ap.add_argument("--cheb-target", type=float, default=12.0,
                    help="Path H: worst-case relative bits of the Chebyshev 1/sqrt")
    ap.add_argument("--save-weights", action="store_true",
                    help="write every trained tensor to <outdir>/trained_state.pt")
    ap.add_argument("--poly-degree", type=int, default=2, help="Path P only")
    ap.add_argument("--poly-init", default="fit", choices=("fit", "const"),
                    help="Path P: 'fit' = LSQ of 1/sqrt on [p1, p99]; 'const' = start "
                         "at q = 1, i.e. Path A's constant, and learn the correction")
    ap.add_argument("--dyt-k", type=float, default=8.0,
                    help="Path D only: tanh starts bending at k x the typical RMS")
    ap.add_argument("--stage", default="A3", choices=sorted(STAGES))
    ap.add_argument("--t-steps", type=int, default=2,
                    help="Path B only: Newton iterations. MEASURED convergence basin is "
                         "v/s in [0.25, 2.0]; outside it MORE steps diverge faster.")
    ap.add_argument("--lambda-range", type=float, default=0.0,
                    help="Path B only: weight on mean(log(v/s)^2), pulling the prescaled "
                         "argument into the convergence basin.")
    ap.add_argument("--norm-stats", type=Path, default=Path("norm/norm_stats.json"))
    ap.add_argument("--stats-length", default="512",
                    help="which sequence length's Phase 1 medians initialise c")
    ap.add_argument("--train-data", default="wikitext2")
    ap.add_argument("--train-split", default="train")
    ap.add_argument("--eval-data", default="wikitext2")
    ap.add_argument("--eval-split", default="validation")
    ap.add_argument("--train-seq-len", type=int, default=512)
    ap.add_argument("--eval-lengths", type=int, nargs="+", default=[512, 2048])
    ap.add_argument("--eval-blocks", type=int, default=40)
    ap.add_argument("--chunk-size", type=int, default=128)
    ap.add_argument("--token-budget", type=int, default=100_000)
    ap.add_argument("--micro-batch", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--c-lr", type=float, default=1e-2,
                    help="separate lr for the log_c scalars, which start far from optimal")
    ap.add_argument("--lambda-lm", type=float, default=1.0)
    ap.add_argument("--lambda-kd", type=float, default=1.0)
    ap.add_argument("--lambda-aux", type=float, default=0.0)
    ap.add_argument("--temperature", type=float, default=2.0)
    ap.add_argument("--warmup-frac", type=float, default=0.05)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--precision-bits", type=int, default=None,
                    help="GUARDRAIL: re-evaluate with norm-path activations rounded "
                         "to this many significant bits (20 ~ post-bootstrap CKKS)")
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", type=Path, default=None)
    ap.add_argument("--unfreeze", default="none", choices=("none", "adjacent", "all"))
    ap.add_argument("--backbone-lr", type=float, default=2e-5)
    ap.add_argument("--ramp", default="none",
                    help="family:start-end fractions, e.g. norm/blend.py DEFAULT_RAMP; "
                         "'default' for that, 'none' to switch everything on at step 0")
    ap.add_argument("--control", action="store_true",
                    help="hold every replacement at beta = 0 (exact): the training-only control")
    args = ap.parse_args(argv)

    if not args.norm_stats.exists():
        raise SystemExit(f"need {args.norm_stats}; run norm/collect_norm_stats.py first")
    tag = (f"{args.stage}_kd{args.lambda_kd:g}_T{args.temperature:g}"
           f"_aux{args.lambda_aux:g}_{args.token_budget}tok")
    args.outdir = args.outdir or Path("runs/norm_pathA") / tag
    args.outdir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    dtype = (recommended_dtype(args.device) if args.dtype == "auto"
             else getattr(torch, args.dtype))
    model, cfg, backend = load_model(args.model, args.backend, args.device, dtype)
    stats_all = json.loads(args.norm_stats.read_text())
    stats = stats_all["by_length"][args.stats_length]

    ph = patch_transition(model, ExactExp(), chunk_size=args.chunk_size)
    if args.path == "A":
        handle = install_path_a(model, stats, stage=args.stage, precision_bits=None)
    elif args.path == "H":
        from norm.native_norm import install_hybrid
        ranges = (json.loads(args.ckks_ranges.read_text())["ckks_choices"]
                  if args.ckks_ranges else None)
        handle = install_hybrid(model, stats, ranges, target_bits=args.cheb_target)
    elif args.path in ("P", "D"):
        from norm.native_norm import install_native
        handle = install_native(model, stats, stage=args.stage, kind=args.path,
                                degree=args.poly_degree, dyt_k=args.dyt_k,
                                poly_init=args.poly_init, precision_bits=None)
    else:
        handle = install_path_b(model, stats, stage=args.stage.replace("A", "B"),
                                t_steps=args.t_steps, precision_bits=None)
    bind_forwards(handle, ph.mixers, args.chunk_size)
    if args.init_from:
        state = torch.load(args.init_from, map_location="cpu")
        params = dict(model.named_parameters())
        absent = [n for n in state if n not in params]
        if absent:
            raise SystemExit(f"--init-from: {len(absent)} tensors have no home: {absent[:4]}")
        with torch.no_grad():
            for n, q in state.items():
                params[n].copy_(q)
        print(f"[init] loaded {len(state)} trained tensors from {args.init_from}")
    calib_choices = None
    if args.path == "H" and not args.ckks_ranges:
        from norm.native_norm import calibrate_hybrid
        cb, _, _ = get_blocks(args.calib_data or args.train_data, "train", args.calib_seq_len,
                              max_tokens=args.calib_blocks * args.calib_seq_len, verbose=False)
        shape = (f"margin {args.calib_margin}" if args.calib_hi_factor is None else
                 f"asymmetric: b = max x {args.calib_hi_factor:g}, a = "
                 f"{'q' + format(args.calib_lo_quantile, 'g') if args.calib_lo_quantile else 'min'}"
                 f" / {args.calib_lo_factor:g}")
        print(f"[calib] CKKS ranges on {min(len(cb), args.calib_blocks)} x "
              f"{args.calib_seq_len} train tokens, this model, {shape}")
        calib_choices = calibrate_hybrid(handle, model, cb[: args.calib_blocks], args.device,
                                         margin=args.calib_margin, target_bits=args.cheb_target,
                                         hi_factor=args.calib_hi_factor,
                                         lo_factor=args.calib_lo_factor,
                                         lo_quantile=args.calib_lo_quantile)
        print(f"  {'site':14s} {'v_min':>9s} {'v q1e-3':>9s} {'v_max':>9s} {'a':>9s} "
              f"{'b':>9s} {'b/a':>8s} {'deg':>5s} {'depth':>5s}")
        for k, c in calib_choices.items():
            print(f"  {k:14s} {c['v_min']:9.3g} {c['v_q']['0.001']:9.3g} {c['v_max']:9.3g} "
                  f"{c['a']:9.3g} {c['b']:9.3g} {c['b'] / c['a']:8.2g} {c['degree']:5d} "
                  f"{c['depth'] + 2:5d}")
    h_sites =[m for m in handle.replacements.values() if getattr(m, "cheb", None) is not None]
    for m in h_sites:
        m.oob_exact = args.train_oob == "exact"
        m.eval_oob_exact = args.eval_oob == "exact"
        if args.lambda_range > 0:
            m.range_inner = args.range_inner or 0.0

    from norm.blend import DEFAULT_RAMP, enable_blend, local_targets, parse_ramp, set_all, set_blend
    ramp = parse_ramp(DEFAULT_RAMP if args.ramp == "default" else args.ramp)
    blending = bool(ramp) or args.control
    if blending:
        enable_blend(handle)
    teacher = None
    if args.unfreeze != "none":
        teacher, _, _ = load_model(args.model, args.backend, args.device, dtype)
        # the official fused mixer needs causal_conv1d, absent on CRC (job 1720400);
        # the reference forward is what the student runs anyway
        patch_transition(teacher, ExactExp(), chunk_size=args.chunk_size)
        teacher.eval()
        for q in teacher.parameters():
            q.requires_grad_(False)

    eval_sets = {}
    for L in args.eval_lengths:
        b, _, _ = get_blocks(args.eval_data, args.eval_split, L, verbose=False)
        eval_sets[L] = b[: args.eval_blocks]
    train_blocks, _, tinfo = get_blocks(args.train_data, args.train_split,
                                        args.train_seq_len, verbose=False)

    # ---------------- baselines ------------------------------------------
    print(f"\n--- teacher (untouched) and student-before, stage {args.stage} ---")
    base, start = {}, {}
    model.eval()
    for L, blocks in eval_sets.items():
        if teacher is not None:
            base[L] = evaluate_ppl(teacher, blocks, args.device, cfg.vocab_size)[0]
        else:
            with exact_mode(handle):
                base[L] = evaluate_ppl(model, blocks, args.device, cfg.vocab_size)[0]
        # eval-only (--token-budget 0): "before" IS "after"; skip the duplicate pass
        start[L] = (float("nan") if args.token_budget == 0 else
                    evaluate_ppl(model, blocks, args.device, cfg.vocab_size)[0])
        print(f"  L={L:>5}  teacher {base[L]:9.4f}   student before {start[L]:9.4f} "
              f"({start[L] - base[L]:+.4f})")

    # ---------------- optimiser ------------------------------------------
    # Train ONLY the installed replacements: their log_c (or log_s) scalars and
    # their gammas. Everything else is frozen.
    #
    # This used to read `other = [p for p in model.parameters() if
    # p.requires_grad ...]`, evaluated while the pretrained model still had
    # requires_grad=True everywhere -- so `other` became the whole network and
    # the freeze/unfreeze below was a no-op. Job 1464468 reported
    # "24 log_c scalars + 128,989,632 other" and fine-tuned all 129M parameters
    # at the KD learning rate; perplexity went 240 -> 5833. The header comment
    # in cluster/job_norm_pathA.sh says the trainable set is ~56k, which is what
    # the design intended and what this now produces.
    repl = getattr(model, "norm_replacements", None)
    if repl is None:
        raise RuntimeError("no norm_replacements on the model -- nothing was installed")
    named = list(repl.named_parameters())
    # the per-instance scalars (and Path P's few coefficients) start far from optimal
    c_params = [q for n, q in named
                if n.endswith(("log_c", "log_s", "log_alpha", "poly_a"))]
    c_ids = {id(q) for q in c_params}
    other = [q for _, q in named if id(q) not in c_ids]
    for q in model.parameters():
        q.requires_grad_(False)
    for q in c_params + other:
        q.requires_grad_(True)
    repl_ids = {id(q) for q in c_params + other}
    if args.unfreeze == "all":
        backbone = [q for q in model.parameters() if id(q) not in repl_ids]
    elif args.unfreeze == "adjacent":
        backbone = [q for mx in ph.mixers for lin in (mx.in_proj, mx.out_proj)
                    for q in lin.parameters()]
    else:
        backbone = []
    for q in backbone:
        q.requires_grad_(True)
    groups = [{"params": c_params, "lr": args.c_lr, "weight_decay": 0.0},
              {"params": other, "lr": args.lr, "weight_decay": 0.0}]
    if backbone:
        groups.append({"params": backbone, "lr": args.backbone_lr, "weight_decay": 0.0})
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.95))
    train_params = c_params + other + backbone
    n_train = sum(q.numel() for q in train_params)
    n_all = sum(q.numel() for q in model.parameters())
    print(f"[train] {len(c_params)} scalars + {sum(q.numel() for q in other):,} "
          f"gamma + {sum(q.numel() for q in backbone):,} backbone ({args.unfreeze}) "
          f"= {n_train:,} trainable of {n_all:,}")
    # A run that quietly fine-tunes the backbone is not a test of the operator:
    # any recovery could not be attributed to the replacement. Refuse loudly.
    if args.unfreeze == "none" and n_train > 0.01 * n_all:
        raise RuntimeError(
            f"{n_train:,} of {n_all:,} parameters are trainable ({100*n_train/n_all:.1f}%). "
            "Path A/B train only the installed replacements; this would be a "
            "full fine-tune wearing the experiment's name.")

    tps = args.micro_batch * args.grad_accum * args.train_seq_len
    total_steps = max(1, args.token_budget // tps)
    warm = max(1, int(args.warmup_frac * total_steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else
        0.5 * (1 + math.cos(math.pi * min((s - warm) / max(1, total_steps - warm), 1.0))))
    logger = RunLogger(args.outdir)
    reset_peak_memory(args.device)
    T = args.temperature

    print(f"\n--- distilling: {args.token_budget:,} tokens, {total_steps} steps, "
          f"lm={args.lambda_lm} kd={args.lambda_kd} T={T} aux={args.lambda_aux} ---")
    model.train()
    t0, step, micro = time.time(), 0, 0
    acc = {"loss": 0.0, "lm": 0.0, "kd": 0.0, "aux": 0.0, "rng": 0.0}
    opt.zero_grad(set_to_none=True)
    use_aux = args.lambda_aux > 0

    for batch, seen in token_budget_batches(train_blocks, args.token_budget,
                                            args.micro_batch, seed=args.seed):
        ids = batch.to(args.device)
        if blending:
            set_blend(handle, step / total_steps, ramp, control=args.control)
        if teacher is not None:
            with torch.no_grad():
                tl = teacher(ids).logits[:, :-1].float()[..., : cfg.vocab_size]
            t_outs = {}
        else:
            with torch.no_grad(), exact_mode(handle, record=use_aux and not blending):
                tl = model(ids).logits[:, :-1].float()[..., : cfg.vocab_size]
                t_outs = ({k: v.detach() for k, v in norm_outputs(handle).items()}
                          if use_aux and not blending else {})
        need_rec = use_aux or (args.path == "B" and args.lambda_range > 0)
        ctx = recording(handle) if need_rec else torch.enable_grad()
        with ctx:
            sl = model(ids).logits[:, :-1].float()[..., : cfg.vocab_size]
            s_outs = norm_outputs(handle) if use_aux else {}
            if use_aux and blending:
                t_outs = local_targets(handle)

        tgt = ids[:, 1:]
        lm = F.cross_entropy(sl.reshape(-1, sl.shape[-1]), tgt.reshape(-1))
        kd = F.kl_div(F.log_softmax(sl / T, -1), F.log_softmax(tl / T, -1),
                      reduction="batchmean", log_target=True) * (T * T)
        ax = aux_loss(s_outs, t_outs) if use_aux else None
        loss = args.lambda_lm * lm + args.lambda_kd * kd
        if ax is not None:
            loss = loss + args.lambda_aux * ax
        rng = None
        if h_sites and args.lambda_range > 0:
            pens = [m.last_range_pen for m in h_sites if m.last_range_pen is not None]
            if pens:
                rng = torch.stack(pens).sum()
                loss = loss + args.lambda_range * rng
            oob = sum(int(m.last_oob.sum()) for m in h_sites if m.last_oob is not None)
            acc["oob"] = acc.get("oob", 0) + oob
            acc["seen"] = acc.get("seen", 0) + sum(
                m.last_oob.numel() for m in h_sites if m.last_oob is not None)
        if args.path == "B" and args.lambda_range > 0:
            from norm.newton_norm import range_penalty
            rp = range_penalty(list(handle.replacements.values()))
            if rp is not None:
                loss = loss + args.lambda_range * rp
        lv = float(loss.detach())
        if not math.isfinite(lv):
            ck = {k: m.cheb for k, m in handle.replacements.items()
                  if getattr(m, "cheb", None) is not None}
            if ck:
                # which CKKS site saw what: re-run the batch counting clamp hits
                for c in ck.values():
                    c.reset_counters(); c.track = True
                with torch.no_grad():
                    model(ids)
                for k, c in ck.items():
                    c.track = False
                    if c.n_guarded or c.n_nonfinite:
                        print(f"    {k:14s} clamped {c.n_guarded}/{c.n_seen}  nonfinite "
                              f"{c.n_nonfinite}  range [{c.a:.3g}, {c.b:.3g}]")
            raise SystemExit(f"\n[DIVERGED] loss={lv} at step {step}. Lower --lr/--c-lr. "
                             f"Report it; do not clamp it.")
        # frozen backbone before any site has ramped in: nothing trainable is in the graph
        if loss.requires_grad:
            (loss / args.grad_accum).backward()
        for k, v in (("loss", loss), ("lm", lm), ("kd", kd),
                     ("aux", ax if ax is not None else torch.zeros(())),
                     ("rng", rng if rng is not None else torch.zeros(()))):
            acc[k] += float(v.detach())
        micro += 1
        if micro % args.grad_accum == 0:
            gn = torch.nn.utils.clip_grad_norm_(train_params, args.grad_clip)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True); step += 1
            if step % args.log_every == 0 or step == 1:
                d = args.grad_accum * (args.log_every if step > 1 else 1)
                logger.log(step=step, tokens_seen=seen, loss=acc["loss"] / d,
                           lm_loss=acc["lm"] / d, kd_loss=acc["kd"] / d,
                           transition_loss=acc["aux"] / d, lr=sched.get_last_lr()[0],
                           grad_norm=float(gn), peak_gpu_gb=peak_memory_gb(args.device))
                print(f"    step {step:5d}/{total_steps} tok {seen:>9,}  "
                      f"loss {acc['loss'] / d:.4f}  lm {acc['lm'] / d:.4f}  "
                      f"kd {acc['kd'] / d:.4f}  aux {acc['aux'] / d:.5f}  "
                      + (f"range {acc['rng'] / d:.4f}  out-of-range "
                         f"{acc.get('oob', 0) / max(acc.get('seen', 0), 1):.2e}  "
                         if h_sites and args.lambda_range > 0 else "")
                      + f"({time.time() - t0:.0f}s)")
                acc = {k: 0.0 for k in acc}
    train_s = time.time() - t0
    logger.close()

    # ---------------- final evaluation, both lengths, then 20-bit ---------
    print("\n--- after distillation ---")
    model.eval()
    if blending:
        set_all(handle, 0.0 if args.control else 1.0)
    end, end20, guarded, oob_diag = {}, {}, {}, {}
    chebs = {k: m.cheb for k, m in handle.replacements.items()
             if getattr(m, "cheb", None) is not None}
    if chebs and args.eval_oob == "exact":
        print("  (eval: out-of-range v gets the EXACT 1/sqrt, an oracle, not FHE; the "
              "rates below are primary, with no clamp perturbing later blocks)")
    for L, blocks in eval_sets.items():
        for c in chebs.values():
            c.reset_counters(); c.track = True
        for m in h_sites:
            m.diag_reset(); m.diag = True
        end[L] = evaluate_ppl(model, blocks, args.device, cfg.vocab_size)[0]
        for c in chebs.values():
            c.track = False
        for m in h_sites:
            m.diag = False
        if chebs:
            guarded[L] = {k: c.n_guarded / max(c.n_seen, 1) for k, c in chebs.items()}
            worst = max(guarded[L], key=guarded[L].get)
            tot = sum(c.n_guarded for c in chebs.values()) / max(sum(c.n_seen for c in chebs.values()), 1)
            print(f"  L={L:>5}  {'out of range' if args.eval_oob == 'exact' else 'clamp hit'} "
                  f"on {tot:.2e} of CKKS-norm inputs; worst {worst} {guarded[L][worst]:.2e}")
            ds = {k: m.diag_stats() for k, m in handle.replacements.items()
                  if getattr(m, "cheb", None) is not None}
            oob_diag[L] = ds
            hi = sum(d["hi_frac"] * d["n"] for d in ds.values()) / max(sum(d["n"] for d in ds.values()), 1)
            lo = sum(d["lo_frac"] * d["n"] for d in ds.values()) / max(sum(d["n"] for d in ds.values()), 1)
            wk = max(ds, key=lambda k: ds[k]["max_v_over_b"])
            print(f"  L={L:>5}  above b {hi:.2e}, below a {lo:.2e}; highest v/b anywhere "
                  f"{ds[wk]['max_v_over_b']:.3g} ({wk}); sites above b:")
            for k, d in ds.items():
                if d["hi_frac"] > 0:
                    print(f"      {k:14s} above b {d['hi_frac']:.2e}  max v/b {d['max_v_over_b']:9.3g}"
                          f"  below a {d['lo_frac']:.2e}")
        print(f"  L={L:>5}  teacher {base[L]:9.4f}  before {start[L]:9.4f}  "
              f"after {end[L]:9.4f}  (delta vs teacher {end[L] - base[L]:+.4f})")
    if args.precision_bits:
        for m in handle.replacements.values():
            m.precision_bits = args.precision_bits
        for L, blocks in eval_sets.items():
            end20[L] = evaluate_ppl(model, blocks, args.device, cfg.vocab_size)[0]
            print(f"  L={L:>5}  at {args.precision_bits}-bit norm precision: "
                  f"{end20[L]:9.4f}  (delta vs fp32 student {end20[L] - end[L]:+.4f})")
        for m in handle.replacements.values():
            m.precision_bits = None

    cs = {k: (float(m.c.detach()) if hasattr(m, "log_c") else
              float(m.alpha.detach()) if hasattr(m, "alpha") else
              [float(a) for a in m.poly_a.detach()] if hasattr(m, "poly_a") else None)
          for k, m in handle.replacements.items()}
    metrics = {
        "stage": args.stage, "teacher_ppl": base, "student_before_ppl": start,
        "student_after_ppl": end, "student_after_ppl_20bit": end20,
        "delta_vs_teacher": {str(L): end[L] - base[L] for L in end},
        "path": args.path,
        "poly_degree": args.poly_degree if args.path == "P" else None,
        "levels_per_instance": (0 if args.path == "A" else
                                {k: getattr(m, "ct_ct_depth", 0)
                                 for k, m in handle.replacements.items()}
                                if args.path == "H" else
                                next(iter(handle.replacements.values())).ct_ct_depth
                                if handle.replacements else None),
        "t_steps": args.t_steps if args.path == "B" else None,
        "lambda_range": args.lambda_range if args.path == "B" else None,
        "n_instances_replaced": len(handle.replacements),
        "n_trainable": n_train, "training_seconds": train_s,
        "training_hours": train_s / 3600, "optimiser_steps": step,
        "peak_gpu_memory_gb": peak_memory_gb(args.device),
        "learned_c": cs,
        "lambda_lm": args.lambda_lm, "lambda_kd": args.lambda_kd,
        "lambda_aux": args.lambda_aux, "temperature": T,
        "unfreeze": args.unfreeze, "backbone_lr": args.backbone_lr if backbone else None,
        "ramp": ramp, "control": args.control, "token_budget": args.token_budget,
        "poly_init": args.poly_init if args.path == "P" else None,
        "clamp_fraction": guarded,
        "ckks_ranges": str(args.ckks_ranges) if args.path == "H" else None,
        "ckks_choices": calib_choices, "init_from": str(args.init_from) if args.init_from else None,
        "cheb_target": args.cheb_target if args.path == "H" else None,
        "range_inner": args.range_inner, "train_oob": args.train_oob,
        "eval_oob": args.eval_oob, "oob_diag": oob_diag,
        "calib_hi_factor": args.calib_hi_factor, "calib_lo_factor": args.calib_lo_factor,
        "calib_lo_quantile": args.calib_lo_quantile,
        "lambda_range_h": args.lambda_range if args.path == "H" else None,
    }
    save_json(args.outdir / "metrics.json", metrics)
    save_json(args.outdir / "config.json",
              {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
              | {"backend": backend, "train_info": tinfo})
    if args.save_weights and step > 0:    # nothing trained, nothing new to save
        keep = {id(q) for q in train_params}
        torch.save({n: q.detach().cpu() for n, q in model.named_parameters() if id(q) in keep},
                   args.outdir / "trained_state.pt")
        print(f"saved {len(keep)} trained tensors to {args.outdir}/trained_state.pt")
    handle.restore(); ph.restore()
    print(f"\nwrote {args.outdir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
