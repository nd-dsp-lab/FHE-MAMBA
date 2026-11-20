# One HE-native SSM block under CKKS: the latency probe

The go/no-go measurement for pretraining an HE-native Mamba. It asks one question before any
pretraining: **how long does one block take under encryption, and how deep is it?**

Shape: d_model 128, L 512, d_inner 256, 4 heads of 64, d_state 16, conv width 4. CKKS
128-bit, OpenFHE 1.2.1, CPU, non-interactive (no client round-trips). Weights are random
with the right shapes and ranges; latency and depth don't depend on their values.

## The block

Every operator is an add or a multiply by construction. Nothing is approximated after training.

| Mamba-2 | here | depth |
|---|---|---|
| in_proj | linear, packed diagonal method | 1 |
| conv1d + SiLU | conv + `act(x) = 0.5x + 0.25x²` | 1 + 2 |
| Δ = softplus(·), z = AΔ | `w = v²` (`= |A_h|·Δ`, A folded into the weights) | 1 |
| a = exp(z) | degree-4 Chebyshev of `exp(-w)` on `[0, wmax_h]`, per head | 3 |
| selective scan | log-depth prefix scan, shifts of 2^k | log2 L |
| gated RMSNorm | dropped (folded or trained away) | 0 |
| out_proj | linear | 1 |

`--selective-heads s` makes heads `s..H-1` use a public (plaintext) decay instead, which is
the ρ knob from the idea: 0 is the Public-Decay HSSM end, all heads is full selectivity.

**Layout.** Each ciphertext holds S = slots / 2L sequences. Each sequence gets 2L slots: L pad
slots, then its L real positions. A scan shift reads into its own pad rather than the
neighbouring sequence. Every plaintext weight is zero on pads, so pads stay zero through the
whole block.

## Files

- `sim.py` runs the exact packed algorithm on plaintext vectors (rotation = `np.roll`) and
  checks it against a sequential reference. It counts operations and depth.
- `bench.cpp` is the same algorithm in OpenFHE, mirroring `sim.py` call for call. It has its
  own sequential reference and compares the decrypted output against it.

## Checked locally (sim.py, 2026-11-13)

| shape | selective heads | max rel. error vs reference | depth | ct×ct | ct×pt | rotations |
|---|---|---|---|---|---|---|
| toy: slots 4096, D 64, L 64 | 4 of 4 | 1.1e-15 | 15 | 316 | 2128 | 462 |
| probe: slots 32768, D 128, L 512 | 4 of 4 | 6.8e-16 | 18 | 1512 | 7840 | 1680 |
| probe | 2 of 4 | 9.2e-16 | 18 | 912 | 7584 | 1662 |
| probe | 0 of 4 | 6.8e-16 | 17 | 312 | 7328 | 1644 |

Two negative controls fail as they should:

- A pad shorter than L gives 6e-3 error.
- A scan that shifts the wrong way gives 100% error.

The probe needs 41 rotation keys.

**bench.cpp has not been compiled yet**, since OpenFHE is only on tjws-03. Its op counts
should match the table above exactly. The depth it reports may differ by one, because
FLEXIBLEAUTO rescales lazily.

## Running on tjws-03

```bash
mkdir -p logs && cmake -B build && cmake --build build -j       # links the system OpenFHE 1.2.1, installs nothing
./build/bench --toy                            # insecure N = 2^13, L 64: compile/API/correctness check, seconds
./build/bench | tee logs/probe_sel4.log        # the probe
./build/bench --selective-heads 0 | tee logs/probe_sel0.log
./build/bench --selective-heads 2 | tee logs/probe_sel2.log
./build/bench --boot | tee logs/boot.log       # one bootstrap with the block's depth left after it
```

- Before every run, the bench checks how plaintexts are encoded for a degree-2 ciphertext. If
  that check fails, rerun with `--pt-level0`.
- If evaluation fails with a level error, raise `--depth`. The default is 10 + log2 L.
- Before a large run, check host RAM and load with `free -g` and `uptime`. The machine is
  shared, and the `--boot` keys are large.

## What to read off

- **Block latency per stage.** The scan and in_proj should dominate. in_proj encodes about
  7,000 plaintexts on the fly, because precomputed plaintexts at N = 2^16 wouldn't fit in RAM.
- **Whether the block fits at N = 2^16 without bootstrapping.** Expected: yes, at depth 19.
- **`--boot`.** This times one bootstrap, and gives the ring it needs, when the block's full
  depth must remain afterwards. A stacked model bootstraps D / S = 4 ciphertexts per block.
  If this forces N = 2^17, that is a finding in itself.
- **The cost of selectivity.** Compare `--selective-heads 4` against `--selective-heads 0` at
  the same shape. This is the first point on the depth/accuracy curve.

## Results on tjws-03 (2026-11-15, OpenFHE 1.2.1, 20 threads)

These are 128-bit runs at ring 2^16 with D 128, L 512. Logs are in `logs/`. In every run the
op counts match `sim.py` exactly. Every run used one level fewer than the simulator predicts,
because FLEXIBLEAUTO rescales lazily.

| selective heads | block (s) | in_proj | scan | levels used / 19 | error vs reference | peak RSS |
|---|---|---|---|---|---|---|
| 4 of 4 | 572 | 329 | 163 | 17 | 9.1e-10 rel. (30.0 bits) | 11.8 GB |
| 2 of 4 | 540 | 314 | 141 | 17 | 6.5e-10 (30.5 bits) | 11.8 GB |
| 0 of 4 | 485 | 303 | 101 | 16 | 7.9e-10 (30.2 bits) | 11.7 GB |

The toy run (insecure ring 2^13, L 64) took 13.4 s with 3.4e-11 relative error.

Bootstrap (`--boot`), keeping the block's 19 levels after it:
- Total depth 41 forces **ring 2^17**.
- Setup and keygen take 52 s, with a 54 GB peak.
- **One bootstrap takes 51 s, with worst error 1.4e-3 (9.5 bits).**
- A stacked block needs 4 bootstraps, about 200 s.

What the results show:
- One block fits at ring 2^16 without bootstrapping.
- in_proj is 57% of block time, not the scan (fixed below). Most of it is encoding about 7,800 plaintexts
  on the fly, and 36 of the 52 output ciphertexts are B, C or v replicated across all 32 blocks.
- Selectivity is cheap at this size: +18% time and +1 level. Even the public-decay scan costs
  100 s, because it is dominated by rotations.
- Stacking blocks is the real problem. A full post-bootstrap budget pushes the ring to 2^17,
  and one bootstrap gives only 9.5 bits.

## Faster in_proj (2026-11-15, logs `probe_rep_*.log`)

The first version computed B, C and v directly in replicated form, one full output ciphertext
per row: 36 of in_proj's 52 outputs. Now each is computed once, into a small stream with one row
per block, and then copied to every block with log2 S rotate-and-adds:
- For B and C, the block selection rides on the conv taps, which are a plaintext multiply anyway.
- For v, the selection is one plaintext multiply.

`--naive-inproj` restores the first version. `sim.py` verifies both, and both match their op
counts on tjws-03.

| version | ring | sequences / ct | block (s) | in_proj | conv+act | scan | levels / 19 | bits | peak RSS |
|---|---|---|---|---|---|---|---|---|---|
| first | 2^16 | 32 | 572 | 329 | 31 | 163 | 17 | 30.0 | 11.8 GB |
| **replicated** | 2^16 | 32 | **378** | 127 | 47 | 156 | 18 | 30.7 | 11.2 GB |
| replicated | 2^17 | 64 | 459 | 138 | 91 | 174 | 18 | 29.0 | 23.2 GB |

- in_proj is 2.6× faster and the block is 34% faster. **The scan is now the largest stage.**
- One more level is used, so only one is spare at depth 19.
- Ring 2^17 halves the op count, but each op costs about 2.3× as much, so the block takes 1.2×
  as long. That run used the block-only Q (log2 Q 1011). In a stack, the ciphertexts after a
  bootstrap carry about the same number of towers, but the key-switching keys are sized for the
  full Q, so expect it to be somewhat slower.

**Estimated cost per block in a bootstrapped stack** at D 128, L 512: about 460 s for the block at
2^17, plus 2 output ciphertexts × about 95 s of two-iteration bootstrapping. That comes to
**about 11 minutes per block per input**, at about 20 bits after each bootstrap. This is an
estimate; no stacked run has been made yet.

## Bootstrap sweep (2026-11-15, `boot_sweep.sh`, log in `logs/boot_sweep.log`)

All runs are 128-bit, with 59-bit scaling and 32,768 slots. "Levels kept" is how many levels are
left after the bootstrap. Iteration 2 is OpenFHE's iterative bootstrap, which costs one extra
level.

| levels kept | budget | iterations | ring | one bootstrap | worst bits | rms bits | peak RSS |
|---|---|---|---|---|---|---|---|
| 19 | {4,4} | 1 | 2^17 | 49 s | 9.6 | 11.9 | — |
| 19 | {4,4} | 2 | 2^17 | 102 s | 20.4 | 22.9 | 56 GB |
| 12 | {3,3} | 2 | 2^17 | 93 s | 20.6 | 22.9 | 48 GB |
| 10 | {3,3} | 2 | 2^17 | 90 s | 20.6 | 22.9 | 45 GB |
| 10 | {4,4} | 2 | 2^17 | 148 s* | 20.6 | 22.9 | 44 GB |
| 10 | {3,3} | 1 | 2^17 | 43 s | 9.4 | 11.9 | 44 GB |
| 8 | {3,3} | 2 | 2^17 | 84 s | 20.6 | 22.9 | 42 GB |
| 8 | {2,2} | 2 | 2^17 | 182 s | 20.5 | 22.9 | 105 GB |
| 6 | {3,3} | 2 | 2^17 | 69 s | 20.3 | 22.9 | 39 GB |

\* Toy runs were competing for the CPU during this configuration.

What the sweep shows:
- **The precision problem is solved by iterating.** One iteration gives about 9.5 bits; two give
  about 20.5 bits worst-case and 23 bits rms, for twice the time and one more level.
- **Bootstrapping forces ring 2^17.** Even 6 levels kept needs depth 27, about 1,654 bits of Q
  plus the key-switching modulus, which is over the 128-bit limit at 2^16. A stacked model
  therefore runs at 2^17, where 64 sequences fit per ciphertext.
- **Budget {3,3} is the choice.** {4,4} is no better, and {2,2} is slower and needs 105 GB.
- **The configuration to use:** keep the block's full depth, budget {3,3}, two iterations. That
  is about 95 s and about 20.5 bits per bootstrap.

## Known simplifications

- Public-decay heads still use the log-depth scan, with scalar multiplies. A depth-1 causal
  convolution with a plaintext kernel would be cheaper for those heads, so the HSSM end is
  overcharged here.
- No outer parallelism. OpenFHE parallelises within each operation; parallelising over output
  ciphertexts may be faster on 20 cores.
- RMSNorm is assumed folded away. If the trained model needs it, add the cost from
  `fhe/ckks_invsqrt`.
