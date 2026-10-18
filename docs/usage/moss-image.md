# MOSS-Image adapters

LyCORIS supports the local `MOSSImageTransformer` and official sd-scripts
`MossImage` through `create_lycoris` and
the Kohya-compatible `create_network` API. This provides adapter construction,
training integration, save/load, and merge operations. It does not add a MOSS
dataset trainer to sd-scripts or a ComfyUI loader.

## Choose a scope

| Preset | Transformer Linear modules | MOSS-Image-Base count |
| --- | --- | ---: |
| `moss` | Each DiT block's `proj.q`, `k`, `v`, `o`, `ffn_up_gate`, `ffn_down` | 252 |
| `moss-full` | The above plus text projection/text blocks, timestep embedding, video embedding, shared modulation, head modulation, output projection | 269 |

The example uses `moss`. The counts describe the supplied 42-block Base
architecture; smaller models have fewer adapters. Both presets select anchored
original module paths and exclude normalization layers, the VAE, and the
separate Qwen3.5 text encoder. Pass the **Transformer**, not the entire pipeline,
to the generic API.

The Transformer's `text_embedder` is its trainable text adapter: a projection and
two text-only Transformer blocks. It is distinct from the external Qwen3.5
`text_encoder`. `moss-full` includes this adapter while excluding Qwen itself.

These scopes are choices provided by LyCORIS. OpenMOSS's
[post-training guide](https://github.com/OpenMOSS/MOSS-Image/blob/main/USERGUIDE.md#lora-training-recommendations)
recommends keeping the text encoder frozen but does not specify LoRA target
layers. Neither preset is claimed to reproduce an official LoRA configuration.

## Local loading and training integration

The local sd-scripts fork integrates official `dev` through `6a3290182c`, including
the stage 3 network trainer at `cb5c24fc72`. Its `moss_train_network.py` is now
a compatibility subclass of `moss_image_train_network.py`, using the official DiT, text-only Qwen3.5, FLUX.2 VAE,
and caching strategies while keeping TXT/JSON/layout conditioning and both
LyCORIS presets. See the fork's `docs/moss_train_network.md` for folder, shard,
and floating-point ComfyUI file inputs. The official `MossImage` keeps the same
adapter paths and returns a Tensor directly; the original pipeline model below
returns an object with `.sample`. The following original-pipeline example remains
an alternative for applications that already use that model.

The official generic default matches `moss` at 252 targets, but its optional
`train_text_embedder=true` has 263 targets and differs from `moss-full` at 269.
Keep the LyCORIS module and presets for full-matrix LoKR+DoRA and existing
native checkpoint restoration. The old command retains discrete uniform
timestep sampling; the official entry defaults to sigmoid sampling. Qwen is
frozen by default, and cached prompt tensors are moved to the CPU before the
DiT is loaded. Qwen encoding uses BF16 weights with FP32 rotary buffers.

Use a separate environment with LyCORIS, PyTorch, and the dependencies required
by the model's local `pipeline_moss_image.py`. The published MOSS requirements
include `torch>=2.9`, `diffusers>=0.38`, `transformers>=5.3`,
`accelerate>=1.10`, and `safetensors>=0.6`. Loading Qwen's weights is unnecessary
when a caller already supplies cached text features. MOSS dependencies are not
added to LyCORIS's core installation.

The following loads only the Transformer from a complete local model folder.
The custom Python module is executed locally; use the source shipped with the
model you intend to run. No model files are downloaded by this example.

```python
import importlib.util
from pathlib import Path
import sys

import torch
from lycoris import create_lycoris, create_lycoris_from_weights

model_dir = Path("/path/to/MOSS-Image-Base")
spec = importlib.util.spec_from_file_location(
    "local_moss_image", model_dir / "pipeline_moss_image.py"
)
moss = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = moss
spec.loader.exec_module(moss)

device = torch.device("cuda")
dtype = torch.bfloat16
transformer = moss.MOSSImageTransformer.from_pretrained(
    model_dir / "transformer", local_files_only=True, torch_dtype=dtype
).to(device)
transformer.requires_grad_(False)

network = create_lycoris(
    transformer,
    preset="moss",             # or "moss-full"
    algo="lokr",
    linear_dim=16,
    linear_alpha=7,
    factor=4,
    full_matrix=True,
    weight_decompose=True,     # DoRA
)
network.apply_to()
network.to(device=device, dtype=dtype)
optimizer = torch.optim.AdamW(network.parameters(), lr=1e-4)


def train_step(noisy_latents, timesteps, text_features, text_mask, target):
    # The caller prepares the noise schedule, packed VAE latents, text features,
    # and flow-matching target. Qwen and the VAE stay frozen outside this network.
    # Base model shapes: [B, 128, H, W] and [B, S, 4096].
    transformer.train()
    network.train()
    optimizer.zero_grad(set_to_none=True)
    prediction = transformer(
        hidden_states=noisy_latents,
        timestep=timesteps,
        encoder_hidden_states=text_features,
        encoder_attention_mask=text_mask,
    ).sample
    loss = (prediction.float() - target.float()).square().mean()
    loss.backward()
    optimizer.step()
    return loss.detach()


# Call train_step(...) with batches prepared by your training application.
# The LR above illustrates optimizer wiring; it is not an OpenMOSS training recipe.
network.save_weights("moss-lokr.safetensors", dtype=dtype, metadata=None)
```

Full-matrix LoKR uses unit scaling, independent of rank/alpha and rs-LoRA.
DoRA keeps its trainable magnitude in FP32 even when the network is cast to
BF16; the portable export retains the magnitude's master dtype.

For standard LoRA use `algo="lora"` and omit `full_matrix`, `factor`, and
`weight_decompose`. The target scopes remain the same.

## Restore and merge

Use the matching generic save/load API pair. The following removes the trained
wrappers and reapplies the saved adapter to the unchanged base Transformer,
without loading a second copy of the 5B model:

```python
network.restore()
del optimizer, network

network, _ = create_lycoris_from_weights(
    1.0, "moss-lokr.safetensors", transformer
)
network.apply_to()
network.to(device=device, dtype=dtype).eval()
transformer.eval()
network.set_multiplier(0.5)
# transformer(...) now applies the adapter at strength 0.5.

# Remove wrappers before merging to avoid applying the adapter twice.
network.restore()
network.merge_to(0.5)
# transformer(...) now uses the merged weights.
network.merge_to(-0.5)  # LoKR's reversible ledger restores the original weights.
```

Safetensors use the existing LyCORIS inference keys. A portable adapter is not a
complete optimizer/training-resume checkpoint. For exact continuation, also save
the network's native `state_dict()`, optimizer state, and your trainer state.
Generic custom-prefix presets can be passed again as `preset=...` to
`create_lycoris_from_weights`; factory calls no longer persist a prefix globally.

## Kohya-compatible factory

As an alternative to the generic factory, start with an unwrapped, frozen
Transformer and create a diffusion-only network:

```python
from lycoris.kohya import create_network, create_network_from_weights

network = create_network(
    1.0, 16, 7, None, None, transformer,
    preset="moss", algo="lokr", factor=4,
    full_matrix=True, weight_decompose=True,
)
network.apply_to(None, transformer, apply_text_encoder=False, apply_unet=True)
network.to(device=device, dtype=dtype)
optimizer = torch.optim.AdamW(network.parameters(), lr=1e-4)
```

The `unet` slot holds the MOSS Transformer. Both MOSS presets create zero Qwen
text-encoder adapters even if a text encoder is supplied. This API uses the
existing `lora_unet_...` prefix; use `create_network_from_weights` for its exports.
The local sd-scripts fork adds `moss_train_network.py` for this factory, including
unchanged Anima-style TXT sidecars, MOSS caption JSON and layout JSON. Its
`docs/moss_train_network.md` contains commands and a native ComfyUI workflow.
The matching local ComfyUI patch accepts both `lora_unet_` and generic `lycoris_`
names, including separate Q/K/V adapters on fused QKV weights, and corrects the
shared output-axis DoRA direction norm. These integrations are separate from
LyCORIS installation; install the local forks to use them.

## Validation

`test/test_moss_sd_scripts.py` additionally checks both factories with the
official sd-scripts model, the 252/269 production scopes, and LoRA/full-matrix
LoKR/LoKR+DoRA backward on CPU FP32 and CUDA BF16 when available. Add the local
sd-scripts checkout to `PYTHONPATH` to run this optional integration.

The standard tests use a small structural model with all 269 production module
paths. Optional integration tests read the real checkpoint headers and import
the local official source, then construct a two-block, 64-dimensional model.
They do not load the 5B checkpoint or the Qwen/VAE weights.

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  PYTHONPATH=. python -m pytest test/test_moss.py -q

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  LYCORIS_MOSS_MODEL_PATH="/path/to/MOSS-Image-Base" \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  PYTHONPATH=. python -m pytest test/test_moss.py -q
```

Integration covers both presets and APIs, LoKR+DoRA and LoRA, optimizer updates,
native/portable checkpoint reconstruction, multipliers, and merge/undo on CPU
FP32 and CUDA BF16 when available. LoKR undo is checked bitwise; additive LoRA
undo is checked numerically. These checks do not establish full-size training
memory requirements, convergence, image quality, or external loader compatibility.
