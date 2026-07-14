# LyCORIS LoKR continuation instructions

## Primary objective

Continue toward a rigorously correct LoKR implementation. The user's primary
training configuration is full-matrix LoKR with DoRA. Do not narrow the goal to
only making existing tests pass; preserve mathematical correctness, checkpoint
compatibility, lifecycle safety, and bounded memory use.

Start from commit `1109c9e` (`Harden full-matrix DoRA LoKr`). Do not redo the
completed audit unless current code contradicts this handoff.

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

The following behavior is intentional and covered by regression tests:

- When both Kronecker factors are full matrices, LoKR uses `scale = 1.0` and
  ignores alpha/rank and rs-LoRA scaling, including noncanonical external
  checkpoint alpha values.
- DoRA computes a normalized adapted direction and applies the runtime
  multiplier to the complete DoRA residual relative to the base weight.
- The direction norm is detached for backward, and fp16/bf16 DoRA accumulation
  is promoted to float32. Explicit float64 inputs remain float64.
- Linear and grouped Conv1d/2d/3d are supported. Input-axis grouped DoRA
  magnitudes are per group and local input channel.
- Full-matrix, low-rank, Tucker, 1x1 Tucker, flattened functional Conv factors,
  mixed parameter dtypes, meta reconstruction, and `assign=True` dtype/device
  marker updates are handled explicitly.
- Reversible LoKR merge uses one CPU base-weight ledger, detects target
  Parameter replacement, normal external mutation, and raw `.data` mutation by
  recomposition, and preserves adapter application order.
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

Reversible merge recovery after factor mutation is still incomplete.

Current ledger state in `lycoris/modules/lokr.py` stores a CPU clone of the base
weight, adapter entries/order, precise mode, target Parameter identity, and the
target Tensor version. Validation recomposes the expected target using the
*current* adapter factors.

Consequences:

1. After `adapter.merge_to(m)`, changing any active factor makes both
   `adapter.merge_to(-m)` and `adapter.finalize_merge()` fail even if the target
   weight itself is untouched.
2. Factor mutation and a raw target `.data` write can be ambiguous because the
   target Tensor version may not change.
3. Partial undo with multiple adapters cannot be recomposed correctly from
   mutated factors unless the original factor state or equivalent composition
   evidence was retained.

Required properties for the next design:

- Never overwrite a genuine external target-weight update silently.
- Retain detection of Parameter replacement and raw `.data` writes.
- Avoid another persistent base-sized tensor; full-matrix memory is a primary
  constraint.
- Define explicit semantics for full undo, partial undo, and finalize after
  factor mutation. If exact partial undo is information-theoretically
  impossible without retained state, expose a narrow, explicit recovery API
  instead of weakening normal validation.
- Add tests for one and multiple adapters, factor `copy_`/optimizer mutation,
  target `no_grad` mutation, target `.data` mutation, Parameter replacement,
  finalize, complete undo, and partial undo.

Do not implement a `force=True` path that blindly overwrites the target. A safe
explicit recovery operation must make the risk and chosen outcome unambiguous.

## Secondary limitations

- CUDA and MPS were unavailable in the audit environment. Run the existing
  device/dtype matrix on real hardware before declaring those paths complete.
- Checkpoint reconstruction clones factor tensors to avoid aliasing the caller's
  state dict. This temporarily keeps checkpoint and adapter factor storage at
  the same time. Any zero-copy loader must make ownership transfer explicit.
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

## Last verified test evidence

On 2026-07-15, before the temporary environments were deleted:

- 102 focused LoKR tests passed under Python 3.12, PyTorch 2.13.0+cpu,
  optimum-quanto 0.2.7, and bitsandbytes 0.49.2, with no skips.
- 96 LoKR module combinations, 4 functional LoKR cases, 8 wrapper LoKR cases,
  and 32 affected FullModule combinations passed.
- Relevant Ruff checks, Ruff format checks, `compileall`, and
  `git diff --check` passed.

Use a low-memory invocation pattern such as:

```bash
env OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    PYTHONPATH=. \
    python -m unittest discover -s test -p 'test_lokr.py'
```

The previous temporary virtual environments and caches under `/tmp` were
deleted after validation. Recreate only the smallest environment required for
the next test, and clean it afterward.
