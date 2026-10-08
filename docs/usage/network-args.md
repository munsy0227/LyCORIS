# Network Arguments

Arguments to put in `network_args` for kohya sd scripts

### Algo

- Set with `algo=ALGO_NAME`
- Check [List of Implemented Algorithms](../algorithms/README.md) for algorithms to use

### Preset

- Set with `preset=PRESET/CONFIG_FILE`
- Pre-implemented: `full` (default), `attn-mlp`, `attn-only` etc.
- Valid for all but (IA)^3
- Use `preset=xxx.toml` to choose config file (for LyCORIS module settings)
- More info in [Preset](presets.md)

### Anima official diffusion scope

- With an Anima model and the `full` preset, LyCORIS targets the same diffusion
  Block Linear-module range found in the official Anima style LoRA: the six
  `adaln_modulation_*` projections, four self-attention projections, four
  cross-attention projections, and two MLP projections in every Block.
- Norm, embedder, final-layer, and LLM-adapter modules remain excluded by
  default. Set `train_llm_adapter=True` only when that additional adapter is
  intentionally part of the training scope.
- The official style LoRA is diffusion-only. In sd-scripts, set
  `network_train_unet_only=true` to reproduce that boundary. A zero
  `text_encoder_lr` freezes text-encoder adapters but does not prevent them from
  being registered and saved.
- LoKr uses Kronecker factors, optional DoRA state, and an optional scalar, so
  matching the base Linear-module range does not imply matching LoRA tensor
  names, shapes, rank, or tensor count.
- `include_patterns` and `exclude_patterns` can override or narrow the default
  scope and are forwarded by `create_network()`. `network_reg_dims` only
  changes the dimension of modules already selected; it does not select the
  training scope.

### Dimension

- Dimension of the linear layers is set with the _script argument_ `network_dim`
- Dimension of the convolutional layers is set with `conv_dim=INT`
- Valid for all but (IA)^3 and native fine-tuning
- For LoKr, setting dimension to sufficiently large value (>10240/2) prevents the second block from being further decomposed

### Alpha

- Alpha of the linear layers is set with the _script argument_ `network_alpha`
- Alpha of the convolutional layers is set with `conv_alpha=FLOAT`
- Valid for all but (IA)^3 and native fine-tuning, ignored by full dimension LoKr as well
- Merge ratio is alpha/dimension, check Appendix B.1 of our [paper](https://arxiv.org/abs/2309.14859) for relation between alpha and learning rate / initialization

### Dropouts

- Set with `dropout=FLOAT`, `rank_dropout=FLOAT`, `module_dropout=FLOAT`
- Set the dropout rate, the types of dropout that are valid could vary from method to method
- Set `rank_dropout_scale=True` to divide retained LoKr rows by their keep
  probability during training

### Factor

- Set with `factor=INT`
- Valid for LoKr
- Use `-1` to select the most balanced factorization automatically
- Other values must be positive integers

### Full Matrix LoKr

- Enabled with `full_matrix=True`
- Valid for LoKr
- Keeps both Kronecker factors as full matrices instead of applying an
  additional low-rank decomposition
- Uses unit LoKr scaling, so alpha is ignored

### Decompose both

- Enabled with `decompose_both=True`
- Valid for LoKr
- Perform LoRA decomposition of both matrices resulting from LoKr decomposition (by default only the larger matrix is decomposed)

### Rank-stabilized scaling

- Enabled with `rs_lora=True`
- Valid for LoKr and other rank-scaled adapters
- Uses alpha divided by the square root of rank instead of alpha divided by rank

### Unbalanced LoKr factorization

- Enabled with `unbalanced_factorization=True`
- Valid for LoKr
- Swaps the two output-side factors while retaining the exact target shape

### Block Size

- Set with `block_size=INT`
- Valid for DyLoRA
- Set the "unit" of DyLoRA (i.e. how many rows / columns to update each time)

### Tucker Decomposition

- Enabled with `use_tucker=True`
- Valid for all but (IA)^3 and native fine-tuning
- It was given the wrong name `use_cp=` in older version

### Scalar

- Enabled with `use_scalar=True`
- Valid for LoRA, LoHa, and LoKr.
- Train an additional scalar in front of the weight difference
- Use a different weight initialization strategy

### Weight Decompose

* Enabled with `dora_wd=True`
* Valid for LoRA, LoHa, and LoKr
* Enable the DoRA method for these algorithms.
* Will force `bypass_mode=False`
* LoKr DoRA supports `full_matrix=True` and grouped Conv1d/2d/3d layers.
* Set `wd_on_output=False` to learn magnitudes along the input axis instead of
  the output axis.
* For grouped convolutions, input-axis magnitudes are learned independently for
  every group and local input channel.
* Weight-only bitsandbytes 4-bit/8-bit and Quanto qint4/qint8 layers are
  supported at runtime. Permanent and on-the-fly merging into a quantized
  weight are rejected because they require backend-specific requantization.

### Bypass Mode

* Enabled with `bypass_mode=True`
* Valid for LoRA, LoHa, LoKr
* Use $Y = WX + \Delta WX$  instead of $Y=(W+\Delta W)X$
* Designed for bnb 8bit/4bit linear layer. (QLyCORIS)

### Normalization Layers

- Enabled with `train_norm=True`
- Valid for all but (IA)^3

### Rescaled OFT

- Enabled with `rescaled=True`
- Valid for Diag-OFT

### Constrained OFT

- Enabled with `constraint=FLOAT`
- Valid for Diag-OFT

### Singular Vector Type (T-LoRA)

- Set with `sig_type=STRING`
- Valid for T-LoRA
- Options: `principal` (default), `last`, `middle`
- Controls which singular vectors from SVD are used for initialization:
  - `principal`: Top-k singular vectors (largest singular values)
  - `last`: Bottom-k singular vectors (smallest singular values)
  - `middle`: Middle-k singular vectors

### Data-Dependent Initialization (T-LoRA)

- Enabled with `use_data_init=True` (default)
- Valid for T-LoRA
- When True, performs SVD on the original layer weights
- When False, performs SVD on a random matrix (data-independent)

### Timestep Mask Group (T-LoRA)

- Set with `mask_group_id=INT`
- Valid for T-LoRA
- Default: 0
- For multi-network scenarios, allows different networks to use different timestep masks
