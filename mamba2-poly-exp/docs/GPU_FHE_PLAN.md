# Running FHE on GPUs for FHEMAMBA

How the team gets from "OpenFHE on tjws-03, GPUs on CRC" to an encrypted Mamba
model whose expensive parts run on a GPU. Written 2026-11-01. Host facts were
checked on tjws-03 and the CRC login node that day. Facts marked *unverified*
need a session on the GPU node itself.

## The end goal and what it implies

A client encrypts its input, the server runs it through the modified Mamba
architecture without decrypting, and the client decrypts the answer. Split by
role:

| Role | Work | Library | Hardware |
|---|---|---|---|
| Client | keygen, encode, encrypt, decrypt | OpenFHE (CPU) | any machine |
| Server, CKKS part | linear layers, polynomial activations (exp, 1/sqrt fits), rotations, **bootstrapping** | GPU CKKS library (FIDESlib) | GPU |
| Server, lookup part | seglut gates (CKKS → FHEW → CKKS scheme switching) | OpenFHE (CPU) | many cores, lots of RAM |
| Model development | polynomial fitting, distillation, evaluation, FHE-noise simulation | PyTorch / mamba_ssm | GPU (CRC) |

Two consequences shape everything below:

1. **OpenFHE itself has no GPU backend.** GPU acceleration means adding a
   separate GPU CKKS library. Wherever FHE runs on a GPU, that library has to be
   built there.
2. **Seglut stays on the CPU.** None of the GPU CKKS libraries implement
   OpenFHE's CKKS ↔ FHEW scheme switching, so a block with seglut gates is
   split: GPU for the CKKS part, CPU for the lookup, with a hand-off at each
   switch.

## The constraint that changes the picture

The GPU library that works with OpenFHE is **FIDESlib**
([github](https://github.com/CAPS-UMU/FIDESlib),
[paper](https://arxiv.org/abs/2507.04775)). OpenFHE does the client side, and
FIDESlib does server-side CKKS evaluation, including bootstrapping, bit-exact
with OpenFHE. Its requirements:

- **A patched OpenFHE 1.6.1**, which its build installs for you. The stock
  **1.2.1** on tjws-03 cannot be used with it.
- CUDA 12 or 13, GPUs sm_70 to sm_120, CMake ≥ 3.25.2, GCC ≥ 11.

So "OpenFHE is already installed on tjws-03" does not help the GPU path. Every
machine that does GPU FHE needs its own build of FIDESlib plus the patched
OpenFHE 1.6.1, wherever that machine is. Use the same patched 1.6.1 for the CPU
parts (seglut) too, so CPU and GPU code share one OpenFHE and can exchange
ciphertexts and keys.

The alternative, **HEonGPU** ([github](https://github.com/Alisah-Ozcan/HEonGPU)),
is standalone: CKKS with four bootstrapping variants, CUDA ≥ 12.8, CMake ≥ 4.0,
sm_75 listed as supported. It does not work with OpenFHE, so the circuit would
have to be rewritten in its API, and its numbers would not be directly
comparable to our OpenFHE results. Keep it as a fallback.

## Hardware we have

| | tjws-03 (checked 2026-11-01) | CRC `qa-rtx6k-019` |
|---|---|---|
| GPU | 1 × Quadro RTX 5000, **16 GB**, sm_75 | 4 × Quadro RTX 6000, **24 GB**, sm_75 |
| Driver | 570.158 (supports CUDA ≤ 12.8) | *unverified*. PyTorch cu121 runs there, so CUDA 12.1 works. |
| CUDA toolkit | system `nvcc` is **11.5** (too old). A CUDA 12 toolkit has to be installed in your home directory. | modules `cuda/12.1` (default) and `cuda/13.2.1`. Nothing to install. |
| Build tools | CMake 4.3.2, GCC 11.4 | modules `cmake/3.31.12`, `cmake/4.1.0`, `gcc/15.2.0` |
| CPU / RAM | 20 cores, 125 GB | our jobs request `-pe smp 1`, so 1 core |
| OpenFHE | 1.2.1 system-wide in `/usr/local` (shared, **do not overwrite**) | none, and no FHE modules of any kind |
| Disk | `/` 93% used, 127 GB free | `/groups` 5 TB, 11% used. `/groups/tjung` is writable by the `tjung-1` group. |
| Access | interactive, password ssh, shared with undergrads | SGE batch / `qrsh`, shared with DP-GRPO jobs |

Both GPUs are Turing (sm_75). Published GPU-FHE speedups are measured on A100-,
4090- and H100-class cards, so expect smaller gains here. Measure before
committing.

**GPU memory is the binding limit for bootstrapping.** Bootstrapping keys at
N = 2^16 should fit in 24 GB and may fit in 16 GB with a trimmed rotation set.
N = 2^17, whose keys already take tens of GB of host RAM, almost certainly will
not fit on either card. Phase 0 measures this rather than assuming it.

### If a workstation with 2 × H200 arrives

H200: 141 GB HBM3e, about 4.8 TB/s, Hopper sm_90, supported by both FIDESlib
and HEonGPU. That is 6× the memory and about 7× the bandwidth of the RTX 6000,
and GPU FHE is limited by exactly those two things. N = 2^17 keys fit (the
seglut run peaked at 24 GB host RSS), a whole model's keys can stay on the card,
and published GPU-FHE speedups were measured on this class of card. Install
OpenFHE on the same box and the tjws-03/CRC split goes away. Phase 2 becomes
"build on the H200 box".

It does **not** speed up seglut. The 128-bit run spends 5.9 s per value, about
13 s of single-thread FHEW work, entirely on the CPU. Spec that machine for the
CPU side too: many cores and enough RAM for several N = 2^17 key sets.

## Recommended path

### Phase 0: prove it on tjws-03 (about 1 week, one researcher)

tjws-03 is the lowest-friction place to start. It is interactive, it has the
GPU, the cores and the RAM, and building in your home directory touches nothing
shared.

1. **Install a CUDA 12.8 toolkit into your home directory**, toolkit only with
   no driver, using NVIDIA's runfile with `--toolkit --installpath=$HOME/opt/cuda-12.8`.
   The 570 driver already supports it.
2. **Build FIDESlib and its patched OpenFHE 1.6.1 into a private prefix**, e.g.
   `$HOME/opt/fides-<commit>/`. Set `CMAKE_INSTALL_PREFIX` explicitly.
   **Never let it install into `/usr/local`**: that would overwrite the shared
   OpenFHE 1.2.1 that other people's code links against. Build with
   `-DFIDESLIB_ARCH=75-real`.
3. **Benchmark bootstrapping on the GPU against the CPU.** Use one CKKS
   bootstrapping at N = 2^16, same parameters, OpenFHE 1.6.1 on 20 CPU threads
   versus FIDESlib on the RTX 5000. Record wall time, peak GPU memory and bits
   of precision.
4. **Port `fhe/seglut_invsqrt/bench.cpp` to the patched 1.6.1.** Check that its
   precision matches the 1.2.1 logs in `fhe/seglut_invsqrt/logs/`. The CPU
   numbers must not change just because the OpenFHE version did.
5. **Write the whole build down as a script** (like `cluster/make_env.sh`), with
   pinned commits for FIDESlib and the OpenFHE patch, so the next machine gets an
   identical build.

**Gate.** Go on if bootstrapping is clearly faster on the GPU and fits in memory
at the parameters the Mamba block needs. If it doesn't fit in 16 GB, that's an
argument for Phase 2 (24 GB cards on CRC), not for giving up.

### Phase 1: build the encrypted block (tjws-03 GPU plus CRC for plaintext)

- **Encrypted block on tjws-03.** CKKS parts on FIDESlib, seglut on CPU
  OpenFHE 1.6.1.
- **Plaintext model work stays on CRC, as now.** That covers fitting,
  distillation, evaluation, and training with **simulated** CKKS noise. Inject
  the error sizes measured on tjws-03 into PyTorch, so models see FHE-realistic
  errors at GPU speed without real FHE inside the training loop.
- **What moves between hosts is small and one-way:** coefficients
  (`coefficients/`, KB), per-head intervals, gate statistics, model weights.
  Route it through the laptop, since both hosts only accept passwords.

### Phase 2: scale to CRC GPUs when tjws-03 becomes the bottleneck

Signs: more than one researcher waiting on one 16 GB card, or parameters that
need 24 GB.

- Use the same build script from Phase 0, built **once** into a single versioned
  prefix in group space, e.g. `/groups/tjung/software/fides-<commit>-openfhe-1.6.1-cuda12.8/`.
  Everyone uses that build. Each researcher should not compile their own.
- On CRC this is only the library build. CUDA, CMake and GCC come from
  existing modules (`module load cuda/12.1 cmake/3.31.12`), so nothing else
  gets installed.
- **This needs sign-off first.** The standing rule (2026-10-06) is to not build
  or install things on CRC. Either whoever set that rule agrees to this one
  scoped exception, or ask CRC support to provide it as a module.
- Build it in a `qrsh` session on the GPU node, **never from a batch job**. A
  batch-job install once filled `$HOME` and killed the job, see
  `cluster/README.md`.
- GPU FHE jobs on CRC need more than `-pe smp 1` whenever a seglut (CPU) stage
  runs in the same job. Request cores to match, or keep seglut on tjws-03.

### Phase 3: end-to-end demo

Client (laptop or tjws-03): generate keys, encrypt the prompt, send the public
evaluation keys and ciphertext once. Server (GPU host): run all layers, with
GPU CKKS plus bootstrapping and CPU seglut at each gate. Client: decrypt and
compare with plaintext model output.

## Rules for researchers

- **Don't move keys between hosts for experiments.** Bootstrapping and rotation
  keys are GBs. Generate keys on the machine that evaluates, and ship code plus
  parameters instead. Only the Phase 3 demo needs the real client → server key
  hand-off.
- **Keep one OpenFHE per pipeline:** the patched 1.6.1 for anything that touches
  the GPU. Don't mix 1.2.1 and 1.6.1 objects.
- **Check the tjws-03 GPU before a long run** (`nvidia-smi`). It is shared with
  undergraduates. For big CPU runs, also check free RAM.
- **Record versions in every result log:** FIDESlib commit, OpenFHE version and
  patch, CUDA version, GPU name, ring dimension N and the full parameter set.
  Results from different GPUs or OpenFHE versions are not comparable without it.
- **CRC job settings stay as in `cluster/README.md`:** queue `gpu@@jung_gpu`,
  `-l gpu=1`.

## Open decisions

1. **CRC install (Phase 2):** allow one shared, versioned FIDESlib prefix under
   `/groups/tjung`, ask CRC support for a module, or stay on tjws-03 only?
2. **tjws-03 GPU sharing:** how to schedule one card between FHEMAMBA
   researchers and undergraduates.
3. **Ring dimension:** what N and depth the encrypted block needs. This decides
   whether 16 GB or 24 GB is enough.
4. **Still to verify on CRC:** the driver version on `qa-rtx6k-019`. It decides
   whether `cuda/13.2.1` works there, which HEonGPU needs (it requires ≥ 12.8);
   FIDESlib runs on `cuda/12.1`. Check with `nvidia-smi` inside a `qrsh` session.
