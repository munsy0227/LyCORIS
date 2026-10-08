# LyCORIS LoKR continuation instructions

## Primary objective

Continue toward a rigorously correct LoKR implementation. The user's primary
training configuration is full-matrix LoKR with DoRA. Do not narrow the goal to
only making existing tests pass; preserve mathematical correctness, checkpoint
compatibility, lifecycle safety, and bounded memory use.

Continue from the current branch HEAD. Commit `1109c9e` (`Harden full-matrix
DoRA LoKr`) is the original hardening baseline, except for the later checkpoint
compatibility rollback documented below. Do not redo the completed audit unless
current code contradicts this handoff.

## Communication and execution rules

- Respond to the user in Korean.
- Verify uncertain facts from current source or authoritative documentation.
- Before editing, inspect the current worktree and preserve unrelated changes.
- Use `rg`/`rg --files` for searches and `apply_patch` for source edits.
- Keep CPU thread counts at 1 for tests. This machine previously experienced an
  operating-system kill under memory pressure.
- Do not run the Kohya/SDXL integration unless explicitly needed. It is gated by
  `LYCORIS_RUN_KOHYA_INTEGRATION=1` because it loads a large checkpoint.
- Do not claim CUDA or MPS correctness without testing on matching hardware.

## Completed LoKR invariants

The following behavior is intentional and must remain covered by regression
tests:

- When both Kronecker factors are full matrices, LoKR uses `scale = 1.0` and
  ignores alpha/rank and rs-LoRA scaling, including noncanonical external
  checkpoint alpha values.
- DoRA computes a normalized adapted direction and applies the runtime
  multiplier to the complete DoRA residual relative to the base weight.
- The direction norm is detached for backward, and fp16/bf16 DoRA accumulation
  is promoted to float32. The trainable magnitude remains an FP32 master across
  full low-precision `.to()` casts without replacing an optimizer-visible
  Parameter; explicit float64 inputs remain float64.
- The standard normalized DoRA expression is used for every slice. An initially
  exact-zero base-norm slice remains a no-op and has zero initial DoRA gradient;
  no additive fallback state is stored in the checkpoint.
- Linear and grouped Conv1d/2d/3d are supported. Input-axis grouped DoRA
  magnitudes are per group and local input channel.
- Full-matrix, low-rank, Tucker, 1x1 Tucker, flattened functional Conv factors,
  mixed parameter dtypes, meta reconstruction, and `assign=True` dtype/device
  marker updates are handled explicitly.
- Automatically promoted full first and second factors set `full_matrix=True`
  and use unit scaling even when the user supplied only a very large rank.
- `use_scalar=True` PyTorch state carries versioned raw-factor/scalar resume
  data. Portable `save_weights()` exports strip it and retain the historical
  folded representation; direct checkpoint reconstruction preserves the exact
  scalar Parameter when resume data is present. Portable exports contain only
  standard LoKr/DoRA inference keys, and low-precision exports keep the DoRA
  magnitude in its master dtype.
- LoKr max-norm regularization limits the portable scalar-folded LoKr update,
  not the complete nonlinear DoRA residual relative to the base weight.
- Kohya optimizer grouping keeps all LoKr factors in the base-LR group because
  its full, Kronecker, and Tucker forms have no validated equivalent of LoRA's
  B role. LR-zero adapters are frozen with stale gradients cleared, and the mask
  is stable before or after adapter registration. Registered adapters removed
  from an active list are still frozen, and the first successful TE/U-Net apply
  selection is immutable; identical active calls are idempotent.
- Anima's `full` preset matches the official diffusion-only LoRA target range:
  all 16 Linear modules in each of 28 Blocks, including the six AdaLN
  modulation projections (448 adapters total). Norm, embedder, final-layer, and
  LLM-adapter modules remain excluded by default. Use sd-scripts'
  `network_train_unet_only=true` to omit text-encoder adapters entirely; a zero
  text-encoder LR only freezes them. Kohya `include_patterns` and
  `exclude_patterns` are forwarded so explicit scope overrides still work.
- Reversible LoKR merge uses one CPU base-weight ledger, detects target
  Parameter replacement, normal external mutation, and raw `.data` mutation by
  recomposition, and preserves adapter application order.
- The ledger also stores chunked SHA-256 fingerprints for the exact target bytes
  and every merge-relevant adapter value/configuration. It retains no second
  persistent base-sized tensor, and target Parameter identity uses a weak
  reference so replacement does not pin the old device storage.
- Factor/config mutation fails closed for normal undo, partial undo, and
  finalize. `resolve_merge_conflict(strategy="restore_base")` safely restores
  the complete target ledger only when the target is untouched;
  `strategy="adopt_current"` performs no target write and terminally adopts the
  exact current target. Exact partial conflict recovery is intentionally absent.
- Network finalize and on-the-fly restore preflight every target before making
  mutations. Nested LoKR on-the-fly frames are validated from top to bottom.
- Non-reversible multi-DoRA network merge follows wrapper application order.
- Ledger and on-the-fly validation run under `torch.no_grad()` to avoid retaining
  a full-matrix autograd graph.
- Destructive merge is rejected while wrappers, parametrizations, incompatible
  precise snapshots, or on-the-fly state are active.
- Permanent/on-the-fly merge into quantized weights is rejected unless explicit
  requantization exists. Runtime Quanto qint4/qint8 and bitsandbytes weight-only
  paths dequantize for forward computation.
- FullModule is exclusive on a target, has transactional apply rollback, handles
  a biasless base plus checkpoint `diff_b`, preserves hidden Parameter identity,
  and moves hidden Parameters/gradients with `_apply()`.

## Highest-priority remaining LoKR issue

No unresolved correctness defect is currently known in the checkpoint-
compatibility rollback. Drive the next correctness change from concrete failing
evidence rather than reintroducing nonstandard portable tensor keys.

The clearest remaining bounded-memory opportunity is checkpoint reconstruction:
factor tensors are cloned to avoid aliasing the caller's state dict, so loading
temporarily retains both checkpoint and adapter storage. Any zero-copy loader
must expose explicit ownership transfer and must not let later caller mutation
silently alter the adapter.

## Secondary limitations

- MPS remains unverified on matching hardware.
- Generic non-LoKR on-the-fly adapters still rely primarily on Tensor `_version`
  and may miss direct `.data` writes. LoKR itself uses value recomposition.
- Earlier broad project runs had unrelated failures: 10 LoRA/LoHa wrapper
  failures, 12 non-LoKR functional-test argument errors, and 72 repository-wide
  Ruff findings. Keep these separate from LoKR regressions unless intentionally
  expanding scope.

## Key files

- `lycoris/modules/lokr.py`: module math, DoRA, state reconstruction, merge and
  on-the-fly lifecycle.
- `lycoris/functional/lokr.py`: factor generation/rebuild and functional path.
- `lycoris/functional/general.py`: Tucker and DoRA helpers.
- `lycoris/modules/base.py`: shared wrapper/merge/on-the-fly behavior.
- `lycoris/wrapper.py`: network-level ordering, preflight, rollback, finalize.
- `lycoris/utils/quant.py`: Quanto and bitsandbytes dequantization.
- `test/lokr.py`: focused regression suite.
- `test/test_lokr.py`: standard unittest discovery entry point.
- `test/test_kohya_optimizer.py`: Kohya optimizer grouping, freezing, and
  `apply_to()` lifecycle regressions.
- `test/test_lokr_cuda.py`: CUDA full-matrix DoRA reference math/gradients,
  checkpoint/exact undo, zero-base initialization, and tiny mock Anima smoke.

## Last verified test evidence

On 2026-10-08, the synchronized tree was tested on an RTX 4070 under Linux.
See `.ai/worklogs/2026-10-08-gpu-validation.md` for the environment, discovered
kernel defects, fixes, and commands. The user's Windows use is out of scope.

- CUDA access requires execution outside the default sandbox's device mask.
  The earlier `nvidia-smi` failure was a sandbox access limitation, not a
  demonstrated driver malfunction.
- PyTorch 2.14.1+cu130 / Triton 3.8.0 / TileLang 0.1.15 were used. TileLang
  was tested with `TILELANG_EXECUTION_BACKEND=nvrtc`; its other execution
  backends were not verified.
- The final Triton selection ran 624 tests: 621 passed, three optional
  quantization tests skipped. It includes 54 kernel tests, the 567-test LoKR
  selection, and three dedicated CUDA tests. TileLang's 53 kernel tests,
  567-test LoKR selection (three optional skips), and three CUDA tests passed.
- Dedicated CUDA tests compare full-matrix DoRA against an independent
  Kronecker/normalization expression in 24 nonzero combinations: Linear,
  grouped Conv1d/2d/3d, both norm axes, FP32/FP16/BF16. They check gradients,
  runtime multipliers, native checkpoint reconstruction, and exact merge undo.
  Initial zero-base no-op and zero gradients also passed in all three dtypes.
- A tiny mock Anima applies all 448 diffusion adapters in BF16 with
  `network_dim=100000`, factor=4, DoRA, scalar, and regex dimensions. One
  selected AdaLN projection retains the exact initial no-op, gets a finite
  nonzero scalar gradient, and changes output after one SGD step.
- The three GPU-driven fixes are distinct: Triton LoRA branch-local variable
  names avoid incompatible SSA shapes; TileLang OFT tensor argument names
  avoid NVRTC wrapper locals; GradPack pads view starts to 256-byte FP32
  boundaries to prevent misaligned vector stores for small LoKR factors.
  Packing still uses one scratch allocation and one homogeneous cast, with
  at most 252 padding bytes between gradients.
- Seventeen Kohya optimizer/scope and compile-compatibility CPU tests passed.
  With automatic tuning enabled, five LoKR tests passed on each fused backend;
  ten tuning winners per backend were persisted in the temporary cache.
  Real Anima training, performance/long-duration behavior, MPS, optional
  quantization runtimes, and full Kohya/Flux integrations remain unverified.

On 2026-10-08, upstream main `4a6a333` was merged into the fork. See
`.ai/worklogs/2026-10-08-upstream-sync.md` for the conflict decisions and
verification commands. The pre-sync state is retained on
`codex/pre-upstream-sync-20261008` (`fc54285`).

- The 180-test focused/kernel/compile/precision selection passed on CPU, with
  29 skips (26 CUDA/backend cases and three optional quantization cases) and
  one expected precision-drift failure. All 890 module/functional cases passed.
- All three preset exclusion tests passed; 59 package modules imported cleanly,
  with 42 optional backend modules skipped. CPU smoke, package build, installed
  wheel DoRA backward/exact merge undo, Ruff, Black API formatting checks, Python
  compilation, and diff whitespace checks passed.
- The same 10 non-LoKR wrapper failure IDs were reproduced before and after
  synchronization. They remain separate from this sync. CUDA, MPS, Windows,
  real Anima, and the full Kohya/Flux integrations were not tested in this run.
- LoKr now uses upstream kernel dispatch for compatible rebuild and linear
  bypass paths, while retaining the fork's DoRA, scalar/checkpoint, grouped
  convolution, and merge lifecycle invariants. LoKr DoRA deliberately keeps its
  detached-norm residual implementation instead of the upstream shared fused
  DoRA epilogue. Float64 operands fall back from the fused kernel tiers.

On 2026-07-16, after the checkpoint-compatibility rollback:

- All 135 focused LoKR tests and 13 Kohya optimizer/scope tests passed under
  Python 3.12.13 and PyTorch 2.13.0. Three optional Quanto/bitsandbytes tests
  were skipped because those packages were not installed in the temporary
  environment.
- The focused suite verifies that native and portable saves omit the removed
  auxiliary tensor keys, DoRA magnitude keeps its master dtype, exact-zero base
  slices follow the standard zero-initial-gradient expression, and max-norm
  limits the scalar-folded portable LoKr update across checkpoint round trips.
- Relevant Ruff checks, Ruff format checks, Python compilation, and
  `git diff --check` passed.
- The broader `test.module` run exercised 864 parameterized cases but still
  reported 48 BOFT bypass shape errors outside the changed LoKR paths.

The 2026-07-15 evidence below predates the rollback and remains historical
context for unaffected behavior:

- The focused LoKR and Kohya optimizer/scope suites passed under Python 3.12.13,
  PyTorch 2.13.0+cu130,
  optimum-quanto 0.2.7, and bitsandbytes 0.49.2, with no skips.
- The Anima scope regression constructs all 28 diffusion Blocks and verifies
  the exact official 448-module set, diffusion-only application, and pattern
  overrides.
- An RTX 4070 BF16 smoke test applied all 448 mock Anima adapters with the
  user's `network_dim=100000`, `factor=4`, DoRA, scalar, and regex-dimension
  settings. It preserved the exact initial no-op, produced a finite scalar
  gradient, and changed the selected AdaLN output after one optimizer step.
- On an NVIDIA GeForce RTX 4070, 384 LoKR module combinations and 16 functional
  LoKR cases passed across CPU/CUDA float32, CUDA float16, and CUDA bfloat16.
- All 32 LoKR wrapper device/dtype/config combinations passed. The wrapper test
  now restores the reversible LoKR merge before applying a reconstructed DoRA
  checkpoint, so it compares both adapters on the same base instead of applying
  the base-dependent normalization twice.
- The former CUDA zero-base learning result depended on the removed additive
  fallback and is superseded. The replacement zero-initial-gradient behavior is
  covered on CPU; matching CUDA dtype coverage remains to be rerun.
- Twenty-four nonzero full-matrix DoRA CUDA combinations covering Linear,
  grouped Conv1d/2d/3d, both magnitude axes, and all three CUDA dtypes produced
  exact merged weights, exact undo, and bitwise-equal checkpoint reconstruction.
- The new `restore_base` and `adopt_current` conflict outcomes passed directly on
  CUDA for float32, float16, and bfloat16.
- Relevant Ruff checks and format checks, `compileall`, and `git diff --check`
  passed on the final source.

Use a low-memory invocation pattern such as:

```bash
env OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    PYTHONPATH=. \
    python -m unittest -q test.lokr test.test_kohya_optimizer
```

The previous temporary virtual environments and caches under `/tmp` were
deleted after validation. Recreate only the smallest environment required for
the next test, and clean it afterward.
