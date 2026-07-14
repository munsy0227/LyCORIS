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

## Others

### wrapper

* `LycorisNetwork`: the wrapper class to patch any pytorch modules to apply LyCORIS algorithms.
* `create_lycoris`: see example
* `create_lycoris_from_weights`: see example

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
reversible merge ledger is active; undo or finalize the ledger before resuming
factor updates.

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
