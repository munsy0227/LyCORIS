import os
import ast
import fnmatch
import re
import logging

from typing import Any

import torch

from .utils import precalculate_safetensors_hashes
from .wrapper import (
    LycorisNetwork,
    deprecated_arg_dict,
    merge_module_options,
    network_module_dict,
    normalize_module_options,
)
from .modules.glora import GLoRAModule
from .modules.norms import NormModule
from .modules import make_module, get_module

from .config import PRESET
from .utils.preset import read_preset
from .utils import str_bool
from .logging import logger


ANIMA_DEFAULT_EXCLUDE_PATTERNS = (r".*(_modulation|_norm|_embedder|final_layer).*",)
ANIMA_REQUIRED_MODULE_CLASSES = {"Block", "PatchEmbed", "TimestepEmbedding"}


def normalize_patterns(patterns):
    if patterns is None:
        return []

    if isinstance(patterns, str):
        try:
            patterns = ast.literal_eval(patterns)
        except (SyntaxError, ValueError):
            patterns = [patterns]

    if not isinstance(patterns, (list, tuple)):
        patterns = [patterns]

    return list(patterns)


def is_anima_unet(unet):
    if unet is None:
        return False

    if unet.__class__.__name__ == "Anima":
        return True

    module_classes = {module.__class__.__name__ for module in unet.modules()}
    return ANIMA_REQUIRED_MODULE_CLASSES.issubset(module_classes)


def compile_patterns(patterns):
    re_patterns = []
    for pattern in patterns:
        try:
            re_patterns.append(re.compile(pattern))
        except re.error as e:
            logger.error(f"Invalid pattern '{pattern}': {e}")
    return re_patterns


def matches_any_pattern(patterns, name):
    return any(pattern.fullmatch(name) for pattern in patterns)


def with_anima_default_excludes(patterns):
    patterns = normalize_patterns(patterns)
    for pattern in ANIMA_DEFAULT_EXCLUDE_PATTERNS:
        if pattern not in patterns:
            patterns.append(pattern)
    return patterns


def create_network(
    multiplier,
    network_dim,
    network_alpha,
    vae,
    text_encoder,
    unet,
    warn_on_unmatched=True,
    **kwargs,
):
    for key, value in list(kwargs.items()):
        if key in deprecated_arg_dict:
            logger.warning(
                f"{key} is deprecated. Please use {deprecated_arg_dict[key]} instead.",
                stacklevel=2,
            )
            kwargs[deprecated_arg_dict[key]] = value
    if network_dim is None:
        network_dim = 4  # default
    conv_dim = int(kwargs.get("conv_dim", network_dim) or network_dim)
    conv_alpha = float(kwargs.get("conv_alpha", network_alpha) or network_alpha)
    dropout = float(kwargs.get("dropout", 0.0) or 0.0)
    rank_dropout = float(kwargs.get("rank_dropout", 0.0) or 0.0)
    module_dropout = float(kwargs.get("module_dropout", 0.0) or 0.0)
    algo = (kwargs.get("algo", "lora") or "lora").lower()
    use_tucker = str_bool(
        not kwargs.get("disable_conv_cp", True)
        or kwargs.get("use_conv_cp", False)
        or kwargs.get("use_cp", False)
        or kwargs.get("use_tucker", False)
    )
    use_scalar = str_bool(kwargs.get("use_scalar", False))
    block_size = int(kwargs.get("block_size", None) or 4)
    train_norm = str_bool(kwargs.get("train_norm", False))
    constraint = float(kwargs.get("constraint", None) or 0)
    rescaled = str_bool(kwargs.get("rescaled", False))
    weight_decompose = str_bool(
        kwargs.get("dora_wd", kwargs.get("weight_decompose", False))
    )
    wd_on_output = str_bool(kwargs.get("wd_on_output", kwargs.get("wd_on_out", True)))
    full_matrix = str_bool(kwargs.get("full_matrix", False))
    bypass_mode = str_bool(kwargs.get("bypass_mode", False))
    rs_lora = str_bool(kwargs.get("rs_lora", False))
    rank_dropout_scale = str_bool(kwargs.get("rank_dropout_scale", False))
    decompose_both = str_bool(kwargs.get("decompose_both", False))
    unbalanced_factorization = str_bool(kwargs.get("unbalanced_factorization", False))
    train_t5xxl = str_bool(kwargs.get("train_t5xxl", False))
    train_llm_adapter = str_bool(kwargs.get("train_llm_adapter", False))
    # lora_plus
    loraplus_lr_ratio = (
        float(kwargs.get("loraplus_lr_ratio", None))
        if kwargs.get("loraplus_lr_ratio", None) is not None
        else None
    )
    loraplus_unet_lr_ratio = (
        float(kwargs.get("loraplus_unet_lr_ratio", None))
        if kwargs.get("loraplus_unet_lr_ratio", None) is not None
        else None
    )
    loraplus_text_encoder_lr_ratio = (
        float(kwargs.get("loraplus_text_encoder_lr_ratio", None))
        if kwargs.get("loraplus_text_encoder_lr_ratio", None) is not None
        else None
    )

    if unbalanced_factorization:
        logger.info("Unbalanced factorization for LoKr is enabled")

    if bypass_mode:
        logger.info("Bypass mode is enabled")

    if weight_decompose:
        logger.info("Weight decomposition is enabled")

    if full_matrix:
        logger.info("Full matrix mode for LoKr is enabled")

    preset_str = kwargs.get("preset", "full")
    if preset_str not in PRESET:
        preset = read_preset(preset_str)
    else:
        preset = PRESET[preset_str]
    assert preset is not None
    LycorisNetworkKohya.apply_preset(preset)

    logger.info(f"Using rank adaptation algo: {algo}")

    if algo == "ia3" and preset_str != "ia3":
        logger.warning("It is recommended to use preset ia3 for IA^3 algorithm")

    is_anima_model = is_anima_unet(unet)
    if is_anima_model:
        kwargs["exclude_patterns"] = with_anima_default_excludes(
            kwargs.get("exclude_patterns", None)
        )

    # regex-specific learning rates / dimensions
    def parse_kv_pairs(kv_pair_str: str, is_int: bool) -> dict[str, float]:
        """
        Parse a string of key-value pairs separated by commas.
        """
        pairs = {}
        for pair in kv_pair_str.split(","):
            pair = pair.strip()
            if not pair:
                continue
            if "=" not in pair:
                logger.warning(f"Invalid format: {pair}, expected 'key=value'")
                continue
            key, value = pair.split("=", 1)
            key = key.strip()
            value = value.strip()
            try:
                pairs[key] = int(value) if is_int else float(value)
            except ValueError:
                logger.warning(f"Invalid value for {key}: {value}")
        return pairs

    network_reg_lrs = kwargs.get("network_reg_lrs", None)
    if network_reg_lrs is not None:
        reg_lrs = parse_kv_pairs(network_reg_lrs, is_int=False)
    else:
        reg_lrs = None

    network_reg_dims = kwargs.get("network_reg_dims", None)
    if network_reg_dims is not None:
        reg_dims = parse_kv_pairs(network_reg_dims, is_int=True)
    else:
        reg_dims = None

    network = LycorisNetworkKohya(
        text_encoder,
        unet,
        multiplier=multiplier,
        lora_dim=network_dim,
        conv_lora_dim=conv_dim,
        alpha=network_alpha,
        conv_alpha=conv_alpha,
        dropout=dropout,
        rank_dropout=rank_dropout,
        module_dropout=module_dropout,
        use_tucker=use_tucker,
        use_scalar=use_scalar,
        network_module=algo,
        train_norm=train_norm,
        decompose_both=decompose_both,
        factor=kwargs.get("factor", -1),
        rank_dropout_scale=rank_dropout_scale,
        block_size=block_size,
        constraint=constraint,
        rescaled=rescaled,
        weight_decompose=weight_decompose,
        wd_on_out=wd_on_output,
        full_matrix=full_matrix,
        bypass_mode=bypass_mode,
        rs_lora=rs_lora,
        unbalanced_factorization=unbalanced_factorization,
        train_t5xxl=train_t5xxl,
        warn_on_unmatched=warn_on_unmatched,
        train_llm_adapter=train_llm_adapter,
        reg_dims=reg_dims,
        reg_lrs=reg_lrs,
        is_anima_model=is_anima_model,
    )
    if (
        loraplus_lr_ratio is not None
        or loraplus_unet_lr_ratio is not None
        or loraplus_text_encoder_lr_ratio is not None
    ):
        network.set_loraplus_lr_ratio(
            loraplus_lr_ratio, loraplus_unet_lr_ratio, loraplus_text_encoder_lr_ratio
        )

    return network


def create_network_from_weights(
    multiplier,
    file,
    vae,
    text_encoder,
    unet,
    weights_sd=None,
    for_inference=False,
    **kwargs,
):
    if weights_sd is None:
        if os.path.splitext(file)[1] == ".safetensors":
            from safetensors.torch import load_file

            weights_sd = load_file(file)
        else:
            weights_sd = torch.load(file, map_location="cpu")

    # get dim/alpha mapping
    unet_loras = {}
    te_loras = {}
    for key, value in weights_sd.items():
        if "." not in key:
            continue

        lora_name = key.split(".")[0]
        if lora_name.startswith(LycorisNetworkKohya.LORA_PREFIX_UNET):
            unet_loras[lora_name] = None
        elif lora_name.startswith(LycorisNetworkKohya.LORA_PREFIX_TEXT_ENCODER):
            te_loras[lora_name] = None

    for name, modules in unet.named_modules():
        lora_name = f"{LycorisNetworkKohya.LORA_PREFIX_UNET}_{name}".replace(".", "_")
        if lora_name in unet_loras:
            unet_loras[lora_name] = modules

    if text_encoder:
        if isinstance(text_encoder, list):
            text_encoders = text_encoder
            use_index = True
        else:
            text_encoders = [text_encoder]
            use_index = False

        for idx, te in enumerate(text_encoders):
            if use_index:
                prefix = f"{LycorisNetworkKohya.LORA_PREFIX_TEXT_ENCODER}{idx + 1}"
            else:
                prefix = LycorisNetworkKohya.LORA_PREFIX_TEXT_ENCODER
            for name, modules in te.named_modules():
                lora_name = f"{prefix}_{name}".replace(".", "_")
                if lora_name in te_loras:
                    te_loras[lora_name] = modules

    original_level = logger.level
    logger.setLevel(logging.ERROR)
    network = LycorisNetworkKohya(text_encoder, unet)
    network.unet_loras = []
    network.text_encoder_loras = []
    logger.setLevel(original_level)

    logger.info("Loading UNet Modules from state dict...")
    for lora_name, orig_modules in unet_loras.items():
        if orig_modules is None:
            continue
        lyco_type, params = get_module(weights_sd, lora_name)
        module = make_module(lyco_type, params, lora_name, orig_modules)
        if module is not None:
            network.unet_loras.append(module)
    logger.info(f"{len(network.unet_loras)} Modules Loaded")

    logger.info("Loading TE Modules from state dict...")

    if text_encoder:
        for lora_name, orig_modules in te_loras.items():
            if orig_modules is None:
                continue
            lyco_type, params = get_module(weights_sd, lora_name)
            module = make_module(lyco_type, params, lora_name, orig_modules)
            if module is not None:
                network.text_encoder_loras.append(module)
        logger.info(f"{len(network.text_encoder_loras)} Modules Loaded")

    for lora in network.unet_loras + network.text_encoder_loras:
        lora.multiplier = multiplier

    return network, weights_sd


class LycorisNetworkKohya(LycorisNetwork):
    """
    LoRA + LoCon
    """

    # Ignore proj_in or proj_out, their channels is only a few.
    ENABLE_CONV = True
    UNET_TARGET_REPLACE_MODULE = [
        "Transformer2DModel",
        "ResnetBlock2D",
        "Downsample2D",
        "Upsample2D",
        "HunYuanDiTBlock",
        "DoubleStreamBlock",
        "SingleStreamBlock",
        "SingleDiTBlock",
        "MMDoubleStreamBlock",  # HunYuanVideo
        "MMSingleStreamBlock",  # HunYuanVideo
        "WanAttentionBlock",  # Wan
        "HunyuanVideoTransformerBlock",  # FramePack
        "HunyuanVideoSingleTransformerBlock",  # FramePack
        "JointTransformerBlock",  # lumina-image-2
        "FinalLayer",  # lumina-image-2, Anima
        "QwenImageTransformerBlock",  # Qwen
        "ZImageTransformerBlock",
        "Block",  # Anima
        "PatchEmbed",  # Anima
        "TimestepEmbedding",  # Anima
    ]
    UNET_TARGET_REPLACE_NAME = [
        "conv_in",
        "conv_out",
        "time_embedding.linear_1",
        "time_embedding.linear_2",
    ]
    TEXT_ENCODER_TARGET_REPLACE_MODULE = [
        "CLIPAttention",
        "CLIPSdpaAttention",
        "CLIPMLP",
        "MT5Block",
        "BertLayer",
        "Gemma2Attention",
        "Gemma2FlashAttention2",
        "Gemma2SdpaAttention",
        "Gemma2MLP",
        "Qwen3Attention",  # Anima / Qwen3
        "Qwen3FlashAttention2",  # Anima / Qwen3
        "Qwen3SdpaAttention",  # Anima / Qwen3
        "Qwen3MLP",  # Anima / Qwen3
    ]
    TEXT_ENCODER_TARGET_REPLACE_NAME = []
    LORA_PREFIX_UNET = "lora_unet"
    LORA_PREFIX_TEXT_ENCODER = "lora_te"
    MODULE_ALGO_MAP = {}
    NAME_ALGO_MAP = {}
    USE_FNMATCH = False

    @classmethod
    def apply_preset(cls, preset):
        if "enable_conv" in preset:
            cls.ENABLE_CONV = preset["enable_conv"]
        if "unet_target_module" in preset:
            cls.UNET_TARGET_REPLACE_MODULE = preset["unet_target_module"]
        if "unet_target_name" in preset:
            cls.UNET_TARGET_REPLACE_NAME = preset["unet_target_name"]
        if "text_encoder_target_module" in preset:
            cls.TEXT_ENCODER_TARGET_REPLACE_MODULE = preset[
                "text_encoder_target_module"
            ]
        if "text_encoder_target_name" in preset:
            cls.TEXT_ENCODER_TARGET_REPLACE_NAME = preset["text_encoder_target_name"]
        if "module_algo_map" in preset:
            cls.MODULE_ALGO_MAP = preset["module_algo_map"]
        if "name_algo_map" in preset:
            cls.NAME_ALGO_MAP = preset["name_algo_map"]
        if "use_fnmatch" in preset:
            cls.USE_FNMATCH = preset["use_fnmatch"]
        return cls

    def __init__(
        self,
        text_encoder,
        unet,
        multiplier=1.0,
        lora_dim=4,
        conv_lora_dim=4,
        alpha=1,
        conv_alpha=1,
        use_tucker=False,
        dropout=0,
        rank_dropout=0,
        module_dropout=0,
        network_module: str = "locon",
        norm_modules=NormModule,
        train_norm=False,
        train_t5xxl=False,
        warn_on_unmatched=True,
        train_llm_adapter=False,
        reg_dims=None,
        reg_lrs=None,
        is_anima_model=False,
        **kwargs,
    ) -> None:
        torch.nn.Module.__init__(self)
        root_kwargs = normalize_module_options(kwargs)
        dropout = float(dropout)
        rank_dropout = float(rank_dropout)
        module_dropout = float(module_dropout)
        train_norm = str_bool(train_norm) if isinstance(train_norm, str) else train_norm
        self.multiplier = multiplier
        self.lora_dim = lora_dim
        self.train_t5xxl = train_t5xxl
        self.train_llm_adapter = (
            str_bool(train_llm_adapter)
            if isinstance(train_llm_adapter, str)
            else train_llm_adapter
        )
        self.reg_dims = reg_dims
        self.reg_lrs = reg_lrs
        self.is_anima_model = is_anima_model

        # 初始化LoRA+相关属性
        self.loraplus_lr_ratio = None
        self.loraplus_unet_lr_ratio = None
        self.loraplus_text_encoder_lr_ratio = None

        if not self.ENABLE_CONV:
            conv_lora_dim = 0

        self.conv_lora_dim = int(conv_lora_dim)
        if self.conv_lora_dim and self.conv_lora_dim != self.lora_dim:
            logger.info("Apply different lora dim for conv layer")
            logger.info(f"Conv Dim: {conv_lora_dim}, Linear Dim: {lora_dim}")
        elif self.conv_lora_dim == 0:
            logger.info("Disable conv layer")

        self.alpha = alpha
        self.conv_alpha = float(conv_alpha)
        if self.conv_lora_dim and self.alpha != self.conv_alpha:
            logger.info("Apply different alpha value for conv layer")
            logger.info(f"Conv alpha: {conv_alpha}, Linear alpha: {alpha}")

        if 1 >= dropout >= 0:
            logger.info(f"Use Dropout value: {dropout}")
        self.dropout = dropout
        self.rank_dropout = rank_dropout
        self.module_dropout = module_dropout

        self.use_tucker = (
            str_bool(use_tucker) if isinstance(use_tucker, str) else use_tucker
        )

        if self.is_anima_model:
            self.exclude_patterns = with_anima_default_excludes(
                kwargs.get("exclude_patterns", None)
            )
        else:
            self.exclude_patterns = normalize_patterns(
                kwargs.get("exclude_patterns", None)
            )
        self.include_patterns = normalize_patterns(kwargs.get("include_patterns", None))
        self.exclude_re_patterns = compile_patterns(self.exclude_patterns)
        self.include_re_patterns = compile_patterns(self.include_patterns)

        def create_single_module(
            lora_name: str,
            module: torch.nn.Module,
            algo_name,
            dim=None,
            alpha=None,
            use_tucker=None,
            original_name=None,
            **kwargs,
        ):
            local_options = dict(kwargs)
            if use_tucker is not None:
                local_options["use_tucker"] = use_tucker
            kwargs = merge_module_options(root_kwargs, local_options)
            use_tucker = kwargs.pop("use_tucker", self.use_tucker)
            configured_dim = kwargs.pop("dim", None)
            configured_alpha = kwargs.pop("alpha", None)
            dim = dim if dim is not None else configured_dim
            alpha = alpha if alpha is not None else configured_alpha
            kwargs.pop("algo", None)
            adapter_dropout = float(kwargs.pop("dropout", self.dropout))
            adapter_rank_dropout = float(kwargs.pop("rank_dropout", self.rank_dropout))
            adapter_module_dropout = float(
                kwargs.pop("module_dropout", self.module_dropout)
            )

            if train_norm and "Norm" in module.__class__.__name__:
                return norm_modules(
                    lora_name,
                    module,
                    self.multiplier,
                    adapter_rank_dropout,
                    adapter_module_dropout,
                    **kwargs,
                )

            if self.reg_dims is not None and original_name is not None:
                for reg, d in self.reg_dims.items():
                    if re.fullmatch(reg, original_name):
                        dim = d
                        alpha = self.alpha
                        logger.info(
                            f"Module {original_name} matched regex '{reg}' -> dim: {dim}"
                        )
                        break

            if dim is not None and dim == 0:
                return None

            lora = None
            if isinstance(module, torch.nn.Linear) and lora_dim > 0:
                dim = dim or lora_dim
                alpha = alpha or self.alpha
            elif isinstance(
                module, (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d)
            ):
                k_size, *_ = module.kernel_size
                if k_size == 1 and lora_dim > 0:
                    dim = dim or lora_dim
                    alpha = alpha or self.alpha
                elif conv_lora_dim > 0 or dim:
                    dim = dim or conv_lora_dim
                    alpha = alpha or self.conv_alpha
                else:
                    return None
            else:
                return None
            lora = network_module_dict[algo_name](
                lora_name,
                module,
                self.multiplier,
                dim,
                alpha,
                adapter_dropout,
                adapter_rank_dropout,
                adapter_module_dropout,
                use_tucker,
                **kwargs,
            )
            if lora is not None:
                lora.original_name = original_name
            return lora

        def create_modules_(
            prefix: str,
            root_module: torch.nn.Module,
            algo,
            configs={},
            original_prefix=None,
        ):
            loras = {}
            lora_names = []
            for name, module in root_module.named_modules():
                if original_prefix and name:
                    full_original_name = f"{original_prefix}.{name}"
                else:
                    full_original_name = original_prefix or name

                is_excluded = matches_any_pattern(
                    self.exclude_re_patterns, full_original_name
                )
                is_included = matches_any_pattern(
                    self.include_re_patterns, full_original_name
                )
                if is_excluded and not is_included:
                    continue

                module_name = module.__class__.__name__
                if module_name in self.MODULE_ALGO_MAP and module is not root_module:
                    next_config = self.MODULE_ALGO_MAP[module_name]
                    next_algo = next_config.get("algo", algo)
                    new_loras, new_lora_names = create_modules_(
                        f"{prefix}_{name}",
                        module,
                        next_algo,
                        next_config,
                        original_prefix=full_original_name,
                    )
                    for lora_name, lora in zip(new_lora_names, new_loras):
                        if lora_name not in loras:
                            loras[lora_name] = lora
                            lora_names.append(lora_name)
                    continue
                if name:
                    lora_name = prefix + "." + name
                else:
                    lora_name = prefix
                lora_name = lora_name.replace(".", "_")
                if lora_name in loras:
                    continue

                lora = create_single_module(
                    lora_name,
                    module,
                    algo,
                    original_name=full_original_name,
                    **configs,
                )
                if lora is not None:
                    loras[lora_name] = lora
                    lora_names.append(lora_name)
            return [loras[lora_name] for lora_name in lora_names], lora_names

        # create module instances
        def create_modules(
            prefix,
            root_module: torch.nn.Module,
            target_replace_modules,
            target_replace_names=[],
        ) -> tuple:
            logger.info("Create LyCORIS Module")
            loras = []
            next_config = {}
            # Track which targets were matched
            matched_modules = set()
            matched_names = set()
            for name, module in root_module.named_modules():
                module_name = module.__class__.__name__
                if module_name in target_replace_modules and not any(
                    self.match_fn(t, name) for t in target_replace_names
                ):
                    matched_modules.add(module_name)
                    if module_name in self.MODULE_ALGO_MAP:
                        next_config = self.MODULE_ALGO_MAP[module_name]
                        algo = next_config.get("algo", network_module)
                    else:
                        algo = network_module
                    loras.extend(
                        create_modules_(
                            f"{prefix}_{name}",
                            module,
                            algo,
                            next_config,
                            original_prefix=name,
                        )[0]
                    )
                    next_config = {}
                elif name in target_replace_names or any(
                    self.match_fn(t, name) for t in target_replace_names
                ):
                    is_excluded = matches_any_pattern(self.exclude_re_patterns, name)
                    is_included = matches_any_pattern(self.include_re_patterns, name)
                    if is_excluded and not is_included:
                        continue

                    # Track which pattern matched and the module class
                    matched_modules.add(module_name)
                    if name in target_replace_names:
                        matched_names.add(name)
                    for t in target_replace_names:
                        if self.match_fn(t, name):
                            matched_names.add(t)
                    conf_from_name = self.find_conf_for_name(name)
                    if conf_from_name is not None:
                        next_config = conf_from_name
                        algo = next_config.get("algo", network_module)
                    elif module_name in self.MODULE_ALGO_MAP:
                        next_config = self.MODULE_ALGO_MAP[module_name]
                        algo = next_config.get("algo", network_module)
                    else:
                        algo = network_module
                    lora_name = prefix + "." + name
                    lora_name = lora_name.replace(".", "_")
                    lora = create_single_module(
                        lora_name, module, algo, original_name=name, **next_config
                    )
                    next_config = {}
                    if lora is not None:
                        loras.append(lora)
            return loras, matched_modules, matched_names

        if network_module == GLoRAModule:
            logger.info("GLoRA enabled, only train transformer")
            # only train transformer (for GLoRA)
            LycorisNetworkKohya.UNET_TARGET_REPLACE_MODULE = [
                "Transformer2DModel",
                "Attention",
            ]
            LycorisNetworkKohya.UNET_TARGET_REPLACE_NAME = []

        self.text_encoder_loras = []
        te_matched_modules = set()
        te_matched_names = set()
        if text_encoder:
            if isinstance(text_encoder, list):
                text_encoders = text_encoder
                use_index = True
            else:
                text_encoders = [text_encoder]
                use_index = False

            for i, te in enumerate(text_encoders):
                loras, matched_mods, matched_nms = create_modules(
                    LycorisNetworkKohya.LORA_PREFIX_TEXT_ENCODER
                    + (f"{i + 1}" if use_index else ""),
                    te,
                    LycorisNetworkKohya.TEXT_ENCODER_TARGET_REPLACE_MODULE,
                    LycorisNetworkKohya.TEXT_ENCODER_TARGET_REPLACE_NAME,
                )
                self.text_encoder_loras.extend(loras)
                te_matched_modules.update(matched_mods)
                te_matched_names.update(matched_nms)
            logger.info(
                f"create LyCORIS for Text Encoder: {len(self.text_encoder_loras)} modules."
            )

        unet_target_modules = list(LycorisNetworkKohya.UNET_TARGET_REPLACE_MODULE)
        if self.train_llm_adapter:
            unet_target_modules.append("LLMAdapterTransformerBlock")
            logger.info("Enable training for LLM Adapter (Anima)")

        self.unet_loras, unet_matched_modules, unet_matched_names = create_modules(
            LycorisNetworkKohya.LORA_PREFIX_UNET,
            unet,
            unet_target_modules,
            LycorisNetworkKohya.UNET_TARGET_REPLACE_NAME,
        )
        logger.info(f"create LyCORIS for U-Net: {len(self.unet_loras)} modules.")

        # Warn about unmatched targets if enabled. Anima presets intentionally
        # include optional module variants, so detailed unmatched lists are noisy.
        if warn_on_unmatched:
            if not self.is_anima_model:
                # Check text encoder targets
                if text_encoder:
                    te_unmatched_modules = (
                        set(LycorisNetworkKohya.TEXT_ENCODER_TARGET_REPLACE_MODULE)
                        - te_matched_modules
                    )
                    te_unmatched_names = (
                        set(LycorisNetworkKohya.TEXT_ENCODER_TARGET_REPLACE_NAME)
                        - te_matched_names
                    )
                    if te_unmatched_modules:
                        logger.warning(
                            "Text Encoder: No modules matched the following target "
                            f"module classes: {sorted(te_unmatched_modules)}"
                        )
                    if te_unmatched_names:
                        logger.warning(
                            "Text Encoder: No modules matched the following target "
                            f"names/patterns: {sorted(te_unmatched_names)}"
                        )

                # Check unet targets
                unet_unmatched_modules = set(unet_target_modules) - unet_matched_modules
                unet_unmatched_names = (
                    set(LycorisNetworkKohya.UNET_TARGET_REPLACE_NAME)
                    - unet_matched_names
                )
                if unet_unmatched_modules:
                    logger.warning(
                        "UNet: No modules matched the following target module "
                        f"classes: {sorted(unet_unmatched_modules)}"
                    )
                if unet_unmatched_names:
                    logger.warning(
                        "UNet: No modules matched the following target "
                        f"names/patterns: {sorted(unet_unmatched_names)}"
                    )

            # Warn if no modules created at all
            total_modules = len(self.text_encoder_loras) + len(self.unet_loras)
            if total_modules == 0:
                logger.warning(
                    "No LyCORIS modules were created. "
                    "This may indicate a mismatch between your LyCORIS config "
                    "and the model architecture. "
                    "Please verify your preset/target settings match the model you are using."
                )

        algo_table = {}
        for lora in self.text_encoder_loras + self.unet_loras:
            algo_table[lora.__class__.__name__] = (
                algo_table.get(lora.__class__.__name__, 0) + 1
            )
        logger.info(f"module type table: {algo_table}")

        self.weights_sd = None

        self.loras = self.text_encoder_loras + self.unet_loras
        # assertion
        names = set()
        for lora in self.loras:
            assert lora.lora_name not in names, (
                f"duplicated lora name: {lora.lora_name}"
            )
            names.add(lora.lora_name)

    def match_fn(self, pattern: str, name: str) -> bool:
        if self.USE_FNMATCH:
            return fnmatch.fnmatch(name, pattern)
        return re.match(pattern, name)

    def find_conf_for_name(
        self,
        name: str,
    ) -> dict[str, Any]:
        if name in self.NAME_ALGO_MAP.keys():
            return self.NAME_ALGO_MAP[name]

        for key, value in self.NAME_ALGO_MAP.items():
            if self.match_fn(key, name):
                return value

        return None

    def load_weights(self, file):
        if os.path.splitext(file)[1] == ".safetensors":
            from safetensors.torch import load_file

            self.weights_sd = load_file(file)
        else:
            self.weights_sd = torch.load(file, map_location="cpu")
        missing, unexpected = self.load_state_dict(self.weights_sd, strict=False)
        state = {}
        if missing:
            state["missing keys"] = missing
        if unexpected:
            state["unexpected keys"] = unexpected
        return state

    def apply_to(self, text_encoder, unet, apply_text_encoder=None, apply_unet=None):
        assert apply_text_encoder is not None and apply_unet is not None, (
            "internal error: flag not set"
        )

        if apply_text_encoder:
            logger.info("enable LyCORIS for text encoder")
        else:
            self.text_encoder_loras = []

        if apply_unet:
            logger.info("enable LyCORIS for U-Net")
        else:
            self.unet_loras = []

        self.loras = self.text_encoder_loras + self.unet_loras

        for lora in self.loras:
            lora.apply_to()
            self.add_module(lora.lora_name, lora)

        if self.weights_sd:
            # if some weights are not in state dict, it is ok because initial LoRA does nothing (lora_up is initialized by zeros)
            info = self.load_state_dict(self.weights_sd, False)
            logger.info(f"weights are loaded: {info}")

    # TODO refactor to common function with apply_to
    def merge_to(self, text_encoder, unet, weights_sd, dtype, device):
        apply_text_encoder = apply_unet = False
        for key in weights_sd.keys():
            if key.startswith(LycorisNetworkKohya.LORA_PREFIX_TEXT_ENCODER):
                apply_text_encoder = True
            elif key.startswith(LycorisNetworkKohya.LORA_PREFIX_UNET):
                apply_unet = True

        if apply_text_encoder:
            logger.info("enable LoRA for text encoder")
        else:
            self.text_encoder_loras = []

        if apply_unet:
            logger.info("enable LoRA for U-Net")
        else:
            self.unet_loras = []

        self.loras = self.text_encoder_loras + self.unet_loras
        # This path writes a final checkpoint and never unmerges in memory.
        super().merge_to(1, reversible=False)

    def apply_max_norm_regularization(self, max_norm_value, device):
        key_scaled = 0
        norms = []
        for module in self.unet_loras + self.text_encoder_loras:
            scaled, norm = module.apply_max_norm(max_norm_value, device)
            if scaled is None:
                continue
            norms.append(norm)
            key_scaled += scaled

        if key_scaled == 0:
            return 0, 0, 0

        return key_scaled, sum(norms) / len(norms), max(norms)

    def set_loraplus_lr_ratio(
        self, loraplus_lr_ratio, loraplus_unet_lr_ratio, loraplus_text_encoder_lr_ratio
    ):
        self.loraplus_lr_ratio = loraplus_lr_ratio
        self.loraplus_unet_lr_ratio = loraplus_unet_lr_ratio
        self.loraplus_text_encoder_lr_ratio = loraplus_text_encoder_lr_ratio

        logger.info(
            f"LoRA+ UNet LR Ratio: {self.loraplus_unet_lr_ratio or self.loraplus_lr_ratio}"
        )
        logger.info(
            "LoRA+ Text Encoder LR Ratio: "
            f"{self.loraplus_text_encoder_lr_ratio or self.loraplus_lr_ratio}"
        )

    def prepare_optimizer_params(
        self, text_encoder_lr=None, unet_lr: float = 1e-4, learning_rate=None
    ):
        self.requires_grad_(True)

        all_params = []
        lr_descriptions = []

        def assemble_params(loras, lr, ratio):
            param_groups = {"lora": {}, "plus": {}}
            reg_groups = {}
            reg_lrs_list = (
                list(self.reg_lrs.items()) if self.reg_lrs is not None else []
            )

            for lora in loras:
                matched_reg_lr = None
                if hasattr(lora, "original_name") and lora.original_name:
                    for i, (regex_str, reg_lr) in enumerate(reg_lrs_list):
                        if re.fullmatch(regex_str, lora.original_name):
                            matched_reg_lr = (i, reg_lr)
                            logger.info(
                                f"Module {lora.original_name} matched regex "
                                f"'{regex_str}' -> LR {reg_lr}"
                            )
                            break

                for name, param in lora.named_parameters():
                    if matched_reg_lr is not None:
                        reg_idx, reg_lr = matched_reg_lr
                        group_key = f"reg_lr_{reg_idx}"
                        if group_key not in reg_groups:
                            reg_groups[group_key] = {
                                "lora": {},
                                "plus": {},
                                "lr": reg_lr,
                            }
                        if ratio is not None and "lora_up" in name:
                            reg_groups[group_key]["plus"][
                                f"{lora.lora_name}.{name}"
                            ] = param
                        else:
                            reg_groups[group_key]["lora"][
                                f"{lora.lora_name}.{name}"
                            ] = param
                        continue

                    if ratio is not None and "lora_up" in name:
                        param_groups["plus"][f"{lora.lora_name}.{name}"] = param
                    else:
                        param_groups["lora"][f"{lora.lora_name}.{name}"] = param

            params = []
            descriptions = []

            for group_key, group in reg_groups.items():
                reg_lr = group["lr"]
                for key in ("lora", "plus"):
                    param_data = {"params": group[key].values()}
                    if len(param_data["params"]) == 0:
                        continue
                    if key == "plus":
                        param_data["lr"] = (
                            reg_lr * ratio if ratio is not None else reg_lr
                        )
                    else:
                        param_data["lr"] = reg_lr

                    if (
                        param_data.get("lr", None) == 0
                        or param_data.get("lr", None) is None
                    ):
                        continue

                    params.append(param_data)
                    desc = f"reg_lr_{group_key.split('_')[-1]}"
                    descriptions.append(desc + (" plus" if key == "plus" else ""))

            for key in param_groups.keys():
                param_data = {"params": param_groups[key].values()}

                if len(param_data["params"]) == 0:
                    continue

                if lr is not None:
                    if key == "plus":
                        param_data["lr"] = lr * ratio
                    else:
                        param_data["lr"] = lr

                if (
                    param_data.get("lr", None) == 0
                    or param_data.get("lr", None) is None
                ):
                    logger.info("NO LR skipping!")
                    continue

                params.append(param_data)
                descriptions.append("plus" if key == "plus" else "")

            return params, descriptions

        if self.text_encoder_loras:
            params, descriptions = assemble_params(
                self.text_encoder_loras,
                text_encoder_lr if text_encoder_lr is not None else learning_rate,
                self.loraplus_text_encoder_lr_ratio or self.loraplus_lr_ratio,
            )
            all_params.extend(params)
            lr_descriptions.extend(
                ["textencoder" + (" " + d if d else "") for d in descriptions]
            )

        if self.unet_loras:
            params, descriptions = assemble_params(
                self.unet_loras,
                unet_lr if unet_lr is not None else learning_rate,
                self.loraplus_unet_lr_ratio or self.loraplus_lr_ratio,
            )
            all_params.extend(params)
            lr_descriptions.extend(
                ["unet" + (" " + d if d else "") for d in descriptions]
            )

        return all_params, lr_descriptions

    def enable_gradient_checkpointing(self):
        # not supported
        pass

    def prepare_grad_etc(self, *args):
        self.requires_grad_(True)

    def on_epoch_start(self, *args):
        self.train()

    def on_step_start(self, *args):
        pass

    def get_trainable_params(self):
        return self.parameters()

    def save_weights(self, file, dtype, metadata):
        if metadata is not None and len(metadata) == 0:
            metadata = None

        state_dict = self.state_dict()

        if dtype is not None:
            for key in list(state_dict.keys()):
                v = state_dict[key]
                v = v.detach().clone().to("cpu").to(dtype)
                state_dict[key] = v

        if os.path.splitext(file)[1] == ".safetensors":
            from safetensors.torch import save_file

            # Precalculate model hashes to save time on indexing
            if metadata is None:
                metadata = {}
            model_hash = precalculate_safetensors_hashes(state_dict)
            metadata["sshs_model_hash"] = model_hash

            save_file(state_dict, file, metadata)
        else:
            torch.save(state_dict, file)
