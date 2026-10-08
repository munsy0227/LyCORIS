# API Reference [WIP]

## Module

### Class: `LycorisBaseModule`:

* classmethod

  * `parametrize`
  * `algo_check`
  * `extract_state_dict`
  * `make_module_from_state_dict`
* property

  * `dtype`
  * `device`
  * `org_weight`
* methods

  * `apply_to`
  * `restore`
  * `merge_to`
  * `finalize_merge`
  * `resolve_merge_conflict` (LoKr only)
  * `onfly_merge`
  * `onfly_restore`
  * `get_diff_weight`
  * `get_merged_weight`
  * `apply_max_norm`
  * `bypass_forward_diff`
  * `bypass_forward`
  * `parametrize_forward`
  * `forward`

#### Subclasses

* `LoConModule`
* `LohaModule`
* `LokrModule`
* `DyLoraModule`
* `GLoRAModule`
* `NormModule`
* `FullModule`
* `DiagOFTModule`

### Functions

* `get_module`: determine the algorithm and extract corresponding weights from state dict.
* `make_module`: based on given algorithm and weights to construct modules.

## Functional

For each modules, we have 3 basic methods:

* `weight_gen`: Generate weights for corresponding algorithm
* `diff_weight`: calculate $\Delta W$
* `bypass_forward_diff`: calculate $\Delta W X$

There are some other utilities:

* `factorization`: $fact(p, factor) = (m, n)$
  * where $m \times n = p$, $m < n$, $m<=factor$ and $m, n \in \mathbb{N}$
  * This method have been used in LoKr and Diag-OFT.
* `power2factorization`: $p2fact(p, factor) = (m, n)$
  * where $m \times n = p$, $m < n$, $m<=factor$, $m=2k$, $n=2^p$ and $m, n, p, k \in \mathbb{N}$
  * This method have been used in BOFT.
* `tucker_weight` and `tucker_weight_from_conv`: Reconstruct tucker decomposed weight from tensors or conv modules.

### Usage

For all the functional API, you can directly use any kind of them with following example:

```python
from lycoris.functional import xxx
weights = xxx.weight_gen(org_weight, rank=4)

def forward_with_diff_weight(x, org_weight, weights):
    return org_forward(x, org_weight + xxx.diff_weight(*weights))

def forward_with_diff_activation(x, org_weight, weights):
    org_out = org_forward(x, org_weight)
    return org_out + xxx.bypass_forward_diff(x, org_out, *weights)
```

Although different algorithms have different extra arguments for `diff_weight`
and `bypass_forward_diff`, the overall logic is the same.

### Backends

Every functional entry point keeps this signature and picks a backend for the
call underneath it — a fused Triton/TileLang kernel, a `torch.compile`d
version of the same op, or the eager body. Nothing about the call changes; see
[kernels/README.md](../kernels/README.md), and
[kernels/backends.md](../kernels/backends.md) for how to pin one.

`lycoris.functional.general` also exposes the two ops that are shared between
algorithms rather than owned by one:

* `weight_decompose`: the DoRA epilogue, `W · (m·(d/‖W‖ − 1) + 1)`, used by
  dora, doha and dokr alike.
* `add_scaled`: `W_org + γ·ΔW`, used by the `full` and `norm` modules.

## Others

### wrapper

* `LycorisNetwork`: the wrapper class to patch any pytorch modules to apply LyCORIS algorithms.
* `create_lycoris`: see example
* `create_lycoris_from_weights`: see example

For LoKr modules created with `use_scalar=True`, `network.state_dict()` preserves
the exact LoKr adapter parameterization needed for training resume. Optimizer,
scheduler, RNG, and other trainer state must still be checkpointed separately.
In addition to the historical scalar-folded first factor, the adapter state
contains versioned `_lycoris_lokr_training_*` entries for the raw first factor
and trainable scalar. This preserves the initial `scalar=0` state and
optimizer-compatible parameterization exactly. Factory reconstruction detects
these entries and restores `use_scalar=True` automatically. When loading the
state directly into a preconstructed module, that module must also use
`use_scalar=True`; a mismatch is rejected instead of silently changing the
optimizer parameter set.

`LycorisNetwork.save_weights()` and `LycorisNetworkKohya.save_weights()` remove
those resume-only entries and write the portable historical representation with
only standard LoKr factor, alpha, and optional DoRA magnitude keys. Requested
low save precision applies to LoKr factors, while `dora_scale` keeps its master
dtype so a low-precision save does not change the initialized DoRA magnitude.
When saving `state_dict()` directly for inference or interchange, call
`LokrModule.strip_training_state_keys(state_dict)` first. Conversely, do not
strip an Accelerate/PyTorch training checkpoint that must resume exactly. With
`load_state_dict(assign=True)`, create the optimizer after loading, as required
by the normal PyTorch Parameter-replacement contract.

For LoKr, `apply_max_norm()` limits the scalar-folded LoKr update represented by
the portable factor tensors. With DoRA, this portable update is not the same as
the complete nonlinear residual $W_{\mathrm{DoRA}}-W_0$, so the method does not
promise that the norm of that complete residual is bounded by the same value.

`LycorisNetwork.apply_to()` can be invoked multiple times with different wrapper
instances. Multiple LoKr wrappers may share a target. An additive, non-DoRA
LoKr may also share a target with non-DoRA LoCon, LoHa, or T-LoRA. FullModule
is exclusive, and base- or order-dependent cross-algorithm combinations are
rejected. Calling `restore()` removes only that wrapper's forward contribution.

LoKr uses a reversible merge ledger by default so non-additive DoRA composition
can be undone exactly with the opposite multiplier. This keeps one CPU copy of
the original target weight. Use `merge_to(..., reversible=False)` when the merge
will never be undone, or call `finalize_merge()` after a reversible merge, to
keep the current weight and release that ledger. A finalized or non-reversible
DoRA merge cannot be recovered by applying a negative multiplier, and the
committed adapter cannot be applied or merged again. Restore all forward
wrappers, remove active parametrizations, and restore on-the-fly changes before
a destructive merge. A reversible network merge rejects mixed LoKr and other
algorithms on one target. Do not train or otherwise mutate LoKr factors while a
reversible merge ledger is active; normal partial undo and finalize operations
fail closed if a factor or the target weight changes. Resolve such a conflict
for the complete target ledger with exactly one explicit outcome:

```python
adapter.resolve_merge_conflict(strategy="restore_base")
adapter.resolve_merge_conflict(strategy="adopt_current")
```

`restore_base` is available only when the target `Parameter` identity and its
exact merged bytes are unchanged; it restores the original base and leaves the
updated factors reusable. `adopt_current` never writes the target, keeps any
current or externally replaced target exactly as-is, and makes every adapter in
that target ledger terminal. Conflict recovery is intentionally target-wide:
an old partial composition cannot be reconstructed from changed factors without
retaining another potentially weight-sized recipe.

`onfly_restore()` must run in reverse order when multiple adapters temporarily
modify the same target. `LycorisNetwork.onfly_restore()` performs this reversal
automatically. The stack is shared by adapters on the target; mixing LoKr and a
different algorithm in that stack is rejected. Network operations preflight
all targets and roll back completed on-the-fly operations after a later
failure. Both permanent and on-the-fly LoKr paths reject untracked target-weight
changes instead of overwriting them.

See `example/stacked_wrapper_demo.py` for a script that showcases stacking and selective removal in practice.

### kohya

* the specialized wrapper for kohya-ss/sd-scripts.

Optimizer preparation freezes adapters omitted by an effective zero learning
rate and keeps that mask through `prepare_grad_etc()`, including calls made
before adapters are registered as network children. Supported LoRA+ higher-LR
roles are `lora_up` and `hada_w2_a`. LoKr parameters remain in the base-LR group:
its full, Kronecker, and Tucker representations do not have one validated
equivalent of LoRA's two-factor B matrix. Optimizers used with a non-unit LoRA+
ratio must support different nonzero learning rates across parameter groups.
Rebuild the optimizer after changing the effective adapter topology or
learning-rate groups.

The text-encoder/U-Net selection passed to Kohya `apply_to()` is immutable after
the first successful call. Repeating the same selection while active is a no-op,
and the same selection can be applied again after `restore()`. Changing the
selection later is rejected before any wrapper or module-list mutation; create a
new network for a different topology.
