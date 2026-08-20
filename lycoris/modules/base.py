import math
from collections import OrderedDict
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrize as parametrize

from ..utils.quant import (
    QuantLinears,
    dequantize_module_weight,
    log_bypass,
    log_fp8_bypass,
    log_suspect,
)

try:
    from peft.tuners.tuners_utils import BaseTunerLayer
except Exception:  # pragma: no cover - PEFT is optional
    BaseTunerLayer = None


def is_weight_only_fp8_linear(module: nn.Module) -> bool:
    return (
        module.__class__.__name__ == "Fp8Linear"
        and hasattr(module, "in_features")
        and hasattr(module, "out_features")
        and hasattr(module, "weight")
        and hasattr(module, "weight_scale")
    )


def is_linear_like_module(module: nn.Module) -> bool:
    return isinstance(module, nn.Linear) or is_weight_only_fp8_linear(module)


_FP8_BYPASS_ALGOS = frozenset({"lora", "locon", "loha", "lokr", "glora"})


def is_supported_linear_module(
    module: nn.Module, algo_name: str, *, weight_decompose: bool = False
) -> bool:
    if is_weight_only_fp8_linear(module):
        return algo_name in _FP8_BYPASS_ALGOS and (
            algo_name == "lokr" or not weight_decompose
        )
    return isinstance(module, nn.Linear)


def dequantize_weight_only_fp8(module: nn.Module) -> torch.Tensor:
    weight = module.weight.to(torch.float32)
    scale = module.weight_scale.to(device=weight.device, dtype=torch.float32)
    if scale.ndim == 1:
        scale = scale.unsqueeze(1)
    return weight * scale


def _is_additive_lokr_stack_adapter(adapter: nn.Module) -> bool:
    """Return whether ``adapter`` composes with additive LoKr exactly.

    Only adapters whose residual is independent of the target's current base
    weight can be reordered with an additive LoKr wrapper.  Unknown adapters
    are deliberately treated as base-dependent.
    """

    return getattr(adapter, "name", None) in {"locon", "loha", "tlora"} and not getattr(
        adapter, "wd", False
    )


def _ensure_target_unwrapped_for_merge(
    module: nn.Module,
    adapter: nn.Module | None = None,
) -> None:
    if adapter is not None and getattr(
        adapter,
        "_lycoris_is_parametrization",
        False,
    ):
        raise RuntimeError(
            "A parametrization adapter cannot be merged destructively while "
            "the parametrization is active. Remove the parametrization first."
        )
    if getattr(module, "_lycoris_wrappers", []):
        raise RuntimeError(
            "Restore all adapters from the target before merging them into "
            "its weight; otherwise the forward path would apply them twice."
        )


class ModuleCustomSD(nn.Module):
    def __init__(self):
        super().__init__()
        self._register_load_state_dict_pre_hook(self.load_weight_prehook)
        self.register_load_state_dict_post_hook(self.load_weight_hook)

    def load_weight_prehook(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        pass

    def load_weight_hook(self, module, incompatible_keys):
        pass

    def custom_state_dict(self):
        return None

    def state_dict(self, *args, destination=None, prefix="", keep_vars=False):
        # TODO: Remove `args` and the parsing logic when BC allows.
        if len(args) > 0:
            if destination is None:
                destination = args[0]
            if len(args) > 1 and prefix == "":
                prefix = args[1]
            if len(args) > 2 and keep_vars is False:
                keep_vars = args[2]
            # DeprecationWarning is ignored by default

        if destination is None:
            destination = OrderedDict()
            destination._metadata = OrderedDict()

        local_metadata = dict(version=self._version)
        if hasattr(destination, "_metadata"):
            destination._metadata[prefix[:-1]] = local_metadata

        if (custom_sd := self.custom_state_dict()) is not None:
            for k, v in custom_sd.items():
                if isinstance(v, torch.Tensor) and not keep_vars:
                    v = v.detach()
                destination[f"{prefix}{k}"] = v
            return destination
        else:
            return super().state_dict(
                *args, destination=destination, prefix=prefix, keep_vars=keep_vars
            )


@dataclass
class _MergeContext:
    precise: bool
    target_device: torch.device
    target_dtype: torch.dtype
    compute_dtype: torch.dtype
    param_device: torch.device | None
    param_dtype: torch.dtype | None
    module: nn.Module
    weight_param: torch.Tensor
    bias_param: torch.Tensor | None


class LycorisBaseModule(ModuleCustomSD):
    name: str
    dtype_tensor: torch.Tensor
    support_module = {}
    weight_list = []
    weight_list_det = []

    def __init__(
        self,
        lora_name,
        org_module: nn.Module,
        multiplier=1.0,
        dropout=0.0,
        rank_dropout=0.0,
        module_dropout=0.0,
        rank_dropout_scale=False,
        bypass_mode=None,
        **kwargs,
    ):
        """if alpha == 0 or None, alpha is rank (no scaling)."""
        super().__init__()
        self.lora_name = lora_name
        self.not_supported = False

        # Keep the optional PEFT wrapper as a plain reference.  Registering it
        # as a child module would make its base-model parameters appear in the
        # adapter optimizer/state dict and would let adapter ``to()`` calls move
        # the entire wrapped layer.
        object.__setattr__(self, "peft_wrapper", None)
        if BaseTunerLayer is not None and isinstance(org_module, BaseTunerLayer):
            object.__setattr__(self, "peft_wrapper", org_module)
            base_layer = getattr(org_module, "base_layer", None)
            if base_layer is None and hasattr(org_module, "get_base_layer"):
                base_layer = org_module.get_base_layer()
            if base_layer is not None:
                org_module = base_layer

        self.module = type(org_module)
        if is_linear_like_module(org_module):
            self.module_type = "linear"
            self.shape = (org_module.out_features, org_module.in_features)
            self.op = F.linear
            self.dim = org_module.out_features
            self.kw_dict = {}
        elif isinstance(org_module, nn.Conv1d):
            self.module_type = "conv1d"
            self.shape = tuple(org_module.weight.shape)
            self.op = F.conv1d
            self.dim = org_module.out_channels
            self.kw_dict = {
                "stride": org_module.stride,
                "padding": org_module.padding,
                "dilation": org_module.dilation,
                "groups": org_module.groups,
            }
        elif isinstance(org_module, nn.Conv2d):
            self.module_type = "conv2d"
            self.shape = tuple(org_module.weight.shape)
            self.op = F.conv2d
            self.dim = org_module.out_channels
            self.kw_dict = {
                "stride": org_module.stride,
                "padding": org_module.padding,
                "dilation": org_module.dilation,
                "groups": org_module.groups,
            }
        elif isinstance(org_module, nn.Conv3d):
            self.module_type = "conv3d"
            self.shape = tuple(org_module.weight.shape)
            self.op = F.conv3d
            self.dim = org_module.out_channels
            self.kw_dict = {
                "stride": org_module.stride,
                "padding": org_module.padding,
                "dilation": org_module.dilation,
                "groups": org_module.groups,
            }
        elif isinstance(org_module, nn.LayerNorm):
            self.module_type = "layernorm"
            self.shape = tuple(org_module.normalized_shape)
            self.op = F.layer_norm
            self.dim = org_module.normalized_shape[0]
            self.kw_dict = {
                "normalized_shape": org_module.normalized_shape,
                "eps": org_module.eps,
            }
        elif isinstance(org_module, nn.GroupNorm):
            self.module_type = "groupnorm"
            self.shape = (org_module.num_channels,)
            self.op = F.group_norm
            self.group_num = org_module.num_groups
            self.dim = org_module.num_channels
            self.kw_dict = {"num_groups": org_module.num_groups, "eps": org_module.eps}
        else:
            self.not_supported = True
            self.module_type = "unknown"

        self.register_buffer("dtype_tensor", torch.tensor(0.0), persistent=False)

        self.is_quant = False
        if is_weight_only_fp8_linear(org_module):
            if not bypass_mode:
                log_fp8_bypass()
            self.is_quant = True
            bypass_mode = True
        elif isinstance(org_module, QuantLinears):
            if not bypass_mode:
                log_bypass()
            self.is_quant = True
            bypass_mode = True
        if (
            is_linear_like_module(org_module)
            and org_module.__class__.__name__ != "Linear"
        ):
            if bypass_mode is None:
                log_suspect()
                bypass_mode = True
            if bypass_mode is True:
                self.is_quant = True
        self.bypass_mode = bypass_mode
        self.dropout = dropout
        self.rank_dropout = rank_dropout
        self.rank_dropout_scale = rank_dropout_scale
        self.module_dropout = module_dropout

        ## Dropout things
        # Since LoKr/LoHa/OFT/BOFT are hard to follow the rank_dropout definition from kohya
        # We redefine the dropout procedure here.
        # g(x) = WX + drop(Brank_drop(AX)) for LoCon(lora), bypass
        # g(x) = WX + drop(ΔWX) for any algo except LoCon(lora), bypass
        # g(x) = (W + Brank_drop(A))X for LoCon(lora), rebuid
        # g(x) = (W + rank_drop(ΔW))X for any algo except LoCon(lora), rebuild
        self.drop = nn.Identity() if dropout == 0 else nn.Dropout(dropout)
        self.rank_drop = (
            nn.Identity() if rank_dropout == 0 else nn.Dropout(rank_dropout)
        )

        self.multiplier = multiplier
        self.org_forward = org_module.forward
        self.org_module = [org_module]

    @classmethod
    def parametrize(cls, org_module, attr, *args, **kwargs):
        from .full import FullModule

        if cls is FullModule:
            raise RuntimeError("FullModule cannot be used for parametrize.")
        target_param = getattr(org_module, attr)
        kwargs["bypass_mode"] = False
        if target_param.dim() == 2:
            proxy_module = nn.Linear(
                target_param.shape[1], target_param.shape[0], bias=False
            )
            proxy_module.weight = target_param
        elif target_param.dim() in (3, 4, 5):
            module_type = [
                None,
                None,
                None,
                nn.Conv1d,
                nn.Conv2d,
                nn.Conv3d,
                None,
                None,
            ][target_param.dim()]
            groups = 1
            in_channels = target_param.shape[1]
            if attr == "weight" and isinstance(
                org_module,
                (nn.Conv1d, nn.Conv2d, nn.Conv3d),
            ):
                groups = org_module.groups
                in_channels = target_param.shape[1] * groups
                if in_channels != org_module.in_channels:
                    raise ValueError(
                        "Convolution metadata does not match the parameterized "
                        f"weight: in_channels={org_module.in_channels}, "
                        f"weight={tuple(target_param.shape)}, groups={groups}."
                    )
            proxy_module = module_type(
                in_channels=in_channels,
                out_channels=target_param.shape[0],
                kernel_size=tuple(target_param.shape[2:]),
                groups=groups,
                bias=False,
                device=target_param.device,
                dtype=target_param.dtype,
            )
            proxy_module.weight = target_param
        else:
            raise ValueError(
                "Only matrix and Conv1d/2d/3d-shaped parameters can be "
                f"parameterized, got shape {tuple(target_param.shape)}."
            )
        module_obj = cls("", proxy_module, *args, **kwargs)
        module_obj._lycoris_is_parametrization = True
        module_obj.forward = module_obj.parametrize_forward
        module_obj.to(target_param)
        parametrize.register_parametrization(org_module, attr, module_obj)
        return module_obj

    @classmethod
    def algo_check(cls, state_dict, lora_name):
        return any(f"{lora_name}.{k}" in state_dict for k in cls.weight_list_det)

    @classmethod
    def extract_state_dict(cls, state_dict, lora_name):
        return [state_dict.get(f"{lora_name}.{k}", None) for k in cls.weight_list]

    @classmethod
    def make_module_from_state_dict(cls, lora_name, orig_module, *weights):
        raise NotImplementedError

    @property
    def dtype(self):
        return self.dtype_tensor.dtype

    @property
    def device(self):
        return self.dtype_tensor.device

    @property
    def org_weight(self):
        return self.org_module[0].weight

    @org_weight.setter
    def org_weight(self, value):
        with torch.no_grad():
            self.org_module[0].weight.copy_(value)

    def _current_weight(self):
        if not hasattr(self.org_module[0], "weight"):
            return self.org_weight.detach()
        if is_weight_only_fp8_linear(self.org_module[0]):
            return dequantize_weight_only_fp8(self.org_module[0]).detach()
        if self.is_quant:
            return dequantize_module_weight(self.org_module[0]).detach()
        return self.org_module[0].weight.detach()

    def _current_bias(self):
        if not hasattr(self.org_module[0], "bias"):
            org_bias = getattr(self, "org_bias", None)
            return None if org_bias is None else org_bias[0].detach()
        bias = self.org_module[0].bias
        return None if bias is None else bias.detach()

    def _weight_forward(self, x, weight, bias=None):
        kwargs = self.kw_dict
        if self.module_type.startswith("conv"):
            module = self.org_module[0]
            if module.padding_mode != "zeros":
                x = F.pad(
                    x,
                    module._reversed_padding_repeated_twice,
                    mode=module.padding_mode,
                )
                kwargs = {**self.kw_dict, "padding": 0}
        return self.op(x, weight, bias, **kwargs)

    def apply_to(self, **kwargs):
        if self.not_supported:
            return

        module = self.org_module[0]
        if getattr(module, "_lycoris_onfly_stack", []):
            raise RuntimeError(
                "Restore the target's on-the-fly merge stack before applying "
                "a forward adapter."
            )
        if not hasattr(module, "_lycoris_original_forward"):
            module._lycoris_original_forward = module.forward

        wrappers = list(getattr(module, "_lycoris_wrappers", []))
        if self in wrappers:
            return
        if any(getattr(wrapper, "name", None) == "full" for wrapper in wrappers):
            raise RuntimeError(
                "FullModule cannot be stacked with another adapter on the same target."
            )
        lokr_wrappers = [
            wrapper for wrapper in wrappers if getattr(wrapper, "name", None) == "kron"
        ]
        if (
            getattr(self, "name", None) != "kron"
            and lokr_wrappers
            and (
                not _is_additive_lokr_stack_adapter(self)
                or any(getattr(wrapper, "wd", False) for wrapper in lokr_wrappers)
            )
        ):
            raise RuntimeError(
                "Stacking LoKr with this adapter on the same target is not "
                "supported because the composition is base- or order-dependent."
            )

        self.org_forward = module.forward
        wrappers.append(self)

        module._lycoris_wrappers = wrappers
        module.forward = self.forward

    def restore(self):
        if self.not_supported:
            return
        module = self.org_module[0]
        wrappers = list(getattr(module, "_lycoris_wrappers", []))

        if not wrappers:
            module.forward = getattr(
                module, "_lycoris_original_forward", self.org_forward
            )
            return

        try:
            idx = wrappers.index(self)
        except ValueError:
            module.forward = (
                wrappers[-1].forward
                if wrappers
                else getattr(module, "_lycoris_original_forward", self.org_forward)
            )
            return

        wrappers.pop(idx)

        if idx < len(wrappers):
            wrappers[idx].org_forward = self.org_forward

        if wrappers:
            module._lycoris_wrappers = wrappers
            module.forward = wrappers[-1].forward
        else:
            module.forward = getattr(
                module, "_lycoris_original_forward", self.org_forward
            )
            module.__dict__.pop("_lycoris_wrappers", None)
            module.__dict__.pop("_lycoris_original_forward", None)

    @torch.no_grad()
    def merge_to(
        self,
        multiplier=1.0,
        *,
        precise: bool = False,
        reversible: bool = True,
    ):
        if self.not_supported:
            return

        module = self.org_module[0]
        if is_weight_only_fp8_linear(module):
            raise RuntimeError(
                "Merging LyCORIS modules into weight-only FP8 Linear is not supported."
            )
        _ensure_target_unwrapped_for_merge(module, self)
        if getattr(module, "_lycoris_onfly_stack", []):
            raise RuntimeError(
                "Cannot permanently merge an adapter while an on-the-fly "
                "merge is active on the target."
            )
        if getattr(module, "_lycoris_lokr_merge_entries", {}):
            raise RuntimeError(
                "Cannot merge a different adapter while a reversible LoKr "
                "merge is active on the target; undo or finalize LoKr first."
            )

        ctx = self._prepare_merge_context(precise)

        if precise:
            weight_prec, bias_prec = self._compute_precise_result(ctx, multiplier)
            self._apply_precise_weights(ctx, weight_prec, bias_prec)
        else:
            weight, bias = self.get_merged_weight(
                multiplier,
                ctx.weight_param.shape,
                ctx.target_device,
            )
            self._apply_merged_weights(ctx, weight, bias)

        self._restore_merge_context(ctx)
        if not reversible:
            self.finalize_merge()

    @torch.no_grad()
    def finalize_merge(self):
        """Keep merged values and discard optional precise merge snapshots."""
        module = self.org_module[0]
        removed = False
        for name in (
            "_lycoris_precise_weight_base",
            "_lycoris_precise_weight_current",
            "_lycoris_precise_bias_base",
            "_lycoris_precise_bias_current",
        ):
            if name in module.__dict__:
                module.__dict__.pop(name)
                removed = True
        return removed

    @torch.no_grad()
    def onfly_merge(self, multiplier=1.0):
        if self.not_supported:
            return
        if is_weight_only_fp8_linear(self.org_module[0]):
            raise RuntimeError(
                "Merging LyCORIS modules into weight-only FP8 Linear is not supported."
            )
        multiplier = float(multiplier)
        if not math.isfinite(multiplier):
            raise ValueError(
                f"On-the-fly merge multiplier must be finite, got {multiplier}."
            )
        if hasattr(self, "_lycoris_onfly_active"):
            raise RuntimeError("onfly_merge() called twice without onfly_restore().")

        module = self.org_module[0]
        _ensure_target_unwrapped_for_merge(module, self)
        if getattr(module, "_lycoris_lokr_merge_entries", {}):
            raise RuntimeError(
                "Cannot use an on-the-fly merge while a reversible LoKr "
                "merge is active on the target."
            )
        stack = list(getattr(module, "_lycoris_onfly_stack", []))
        if any(
            getattr(frame.get("adapter"), "name", None) == "kron" for frame in stack
        ):
            raise RuntimeError(
                "Mixing LoKr and a different adapter in one target's "
                "on-the-fly merge stack is not supported."
            )
        if stack:
            active_frame = stack[-1]
            if module.weight is not active_frame["weight_param"]:
                raise RuntimeError(
                    "The target weight Parameter was replaced during an "
                    "on-the-fly merge; refusing to extend the stack."
                )
            if module.weight._version != active_frame["weight_version"]:
                raise RuntimeError(
                    "The target weight changed during an on-the-fly merge; "
                    "refusing to extend the stack."
                )
            if module.bias is not active_frame["bias_param"]:
                raise RuntimeError(
                    "The target bias Parameter was replaced during an "
                    "on-the-fly merge; refusing to extend the stack."
                )
            if (
                module.bias is not None
                and module.bias._version != active_frame["bias_version"]
            ):
                raise RuntimeError(
                    "The target bias changed during an on-the-fly merge; "
                    "refusing to extend the stack."
                )

        parameters = tuple(self.parameters())
        first_parameter = parameters[0] if parameters else None
        self_device = first_parameter.device if first_parameter is not None else None
        self_dtype = first_parameter.dtype if first_parameter is not None else None
        original_weight = None
        original_bias = module.bias
        original_bias_exists = original_bias is not None
        original_bias_value = None
        try:
            if multiplier != 0:
                original_weight = self.org_weight.detach().cpu().clone()
                original_bias_value = (
                    original_bias.detach().cpu().clone()
                    if original_bias is not None
                    else None
                )
                self.to(self.org_weight)
                weight, bias = self.get_merged_weight(
                    multiplier,
                    self.org_weight.shape,
                    self.org_weight.device,
                )
                self.org_weight = weight
                if bias is not None:
                    if original_bias is not None:
                        original_bias.copy_(bias.to(original_bias))
                    else:
                        module.bias = nn.Parameter(bias.to(self.org_weight))

            self.cached_org_weight = original_weight
            self.cached_org_bias_exists = original_bias_exists
            self.cached_org_bias = original_bias_value
            self._lycoris_onfly_active = True
            stack.append(
                {
                    "adapter": self,
                    "kind": "generic",
                    "weight_param": module.weight,
                    "weight_version": module.weight._version,
                    "bias_param": module.bias,
                    "bias_version": (
                        module.bias._version if module.bias is not None else None
                    ),
                    "multiplier": multiplier,
                }
            )
            module._lycoris_onfly_stack = stack
        except Exception:
            if original_weight is not None:
                self.org_weight = original_weight.to(self.org_weight)
            if original_bias_exists:
                if module.bias is original_bias:
                    original_bias.copy_(original_bias_value.to(original_bias))
                else:
                    module.bias = original_bias
            elif module.bias is not None:
                module.bias = None
            for name in (
                "cached_org_weight",
                "cached_org_bias",
                "cached_org_bias_exists",
                "_lycoris_onfly_active",
            ):
                self.__dict__.pop(name, None)
            raise
        finally:
            if self_device is not None and self_dtype is not None:
                self.to(device=self_device, dtype=self_dtype)

    @torch.no_grad()
    def onfly_restore(self):
        if self.not_supported:
            return
        if not hasattr(self, "_lycoris_onfly_active"):
            raise RuntimeError("onfly_restore() called without onfly_merge().")
        module = self.org_module[0]
        stack = list(getattr(module, "_lycoris_onfly_stack", []))
        if not stack or stack[-1].get("adapter") is not self:
            raise RuntimeError(
                "On-the-fly adapters must be restored in reverse merge order."
            )
        frame = stack[-1]
        if module.weight is not frame["weight_param"]:
            raise RuntimeError(
                "The target weight Parameter was replaced during an "
                "on-the-fly merge; refusing to overwrite it."
            )
        if module.weight._version != frame["weight_version"]:
            raise RuntimeError(
                "The target weight changed during an on-the-fly merge; "
                "refusing to overwrite the external update."
            )
        if module.bias is not frame["bias_param"]:
            raise RuntimeError(
                "The target bias Parameter was replaced during an "
                "on-the-fly merge; refusing to overwrite it."
            )
        if module.bias is not None and module.bias._version != frame["bias_version"]:
            raise RuntimeError(
                "The target bias changed during an on-the-fly merge; "
                "refusing to overwrite the external update."
            )

        if frame["multiplier"] != 0:
            self.org_weight = self.cached_org_weight.to(self.org_weight)
            if self.cached_org_bias_exists:
                module.bias.copy_(self.cached_org_bias.to(module.bias))
            else:
                module.bias = None

        stack.pop()
        for name in (
            "cached_org_weight",
            "cached_org_bias",
            "cached_org_bias_exists",
            "_lycoris_onfly_active",
        ):
            self.__dict__.pop(name, None)
        if stack:
            stack[-1]["weight_version"] = module.weight._version
            stack[-1]["bias_param"] = module.bias
            stack[-1]["bias_version"] = (
                module.bias._version if module.bias is not None else None
            )
            module._lycoris_onfly_stack = stack
        else:
            module.__dict__.pop("_lycoris_onfly_stack", None)

    def get_diff_weight(self, multiplier=1.0, shape=None, device=None):
        raise NotImplementedError

    def get_merged_weight(self, multiplier=1.0, shape=None, device=None):
        raise NotImplementedError

    @torch.no_grad()
    def apply_max_norm(self, max_norm, device=None):
        return None, None

    def bypass_forward_diff(self, x, scale=1):
        raise NotImplementedError

    def bypass_forward(self, x, scale=1):
        raise NotImplementedError

    def parametrize_forward(self, x: torch.Tensor, *args, **kwargs):
        return self.get_merged_weight(
            multiplier=self.multiplier, shape=x.shape, device=x.device
        )[0].to(x.dtype)

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def _prepare_merge_context(self, precise: bool) -> _MergeContext:
        module = self.org_module[0]
        weight_param = module.weight
        bias_param = module.bias

        params = tuple(self.parameters())
        first_param = params[0] if params else None
        param_device = first_param.device if first_param is not None else None
        param_dtype = first_param.dtype if first_param is not None else None

        target_device = weight_param.device
        target_dtype = weight_param.dtype
        compute_dtype = torch.float64 if precise else target_dtype

        if first_param is not None:
            self.to(device=target_device, dtype=compute_dtype)
        else:
            self.to(target_device)
            if precise:
                self.to(dtype=compute_dtype)

        if precise:
            self._ensure_precise_snapshot(module, weight_param, bias_param)
            self._load_precise_snapshot(
                module,
                weight_param,
                bias_param,
                target_device,
                compute_dtype,
            )

        return _MergeContext(
            precise=precise,
            target_device=target_device,
            target_dtype=target_dtype,
            compute_dtype=compute_dtype,
            param_device=param_device,
            param_dtype=param_dtype,
            module=module,
            weight_param=weight_param,
            bias_param=bias_param,
        )

    def _apply_merged_weights(
        self,
        ctx: _MergeContext,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> None:
        merged_weight = weight.to(ctx.target_dtype)
        ctx.weight_param.copy_(merged_weight)

        if bias is not None:
            merged_bias = bias.to(ctx.target_dtype)
            if ctx.bias_param is not None:
                ctx.bias_param.copy_(merged_bias)
            else:
                ctx.module.bias = nn.Parameter(merged_bias)
        elif ctx.bias_param is None:
            ctx.module.bias = None

        if ctx.precise:
            ctx.module._lycoris_precise_weight_current = weight.to(torch.float64).cpu()
            if ctx.bias_param is not None:
                if bias is not None:
                    ctx.module._lycoris_precise_bias_current = bias.to(
                        torch.float64
                    ).cpu()
                else:
                    ctx.module._lycoris_precise_bias_current = (
                        ctx.module._lycoris_precise_bias_base
                    )

    def _restore_merge_context(self, ctx: _MergeContext) -> None:
        if ctx.param_device is not None and ctx.param_dtype is not None:
            self.to(device=ctx.param_device, dtype=ctx.param_dtype)
        elif ctx.param_device is not None:
            self.to(ctx.param_device)
        elif ctx.param_dtype is not None:
            self.to(dtype=ctx.param_dtype)

    def _compute_precise_result(
        self, ctx: _MergeContext, multiplier: float
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        base_weight = ctx.module._lycoris_precise_weight_current
        diff_weight, diff_bias = self.get_diff_weight(
            multiplier=1.0, device=ctx.target_device
        )
        diff_weight_prec = diff_weight.to(torch.float64).cpu()
        new_weight = base_weight + diff_weight_prec * multiplier

        new_bias = None
        if diff_bias is not None:
            diff_bias_prec = diff_bias.to(torch.float64).cpu()
            base_bias = ctx.module._lycoris_precise_bias_current
            if base_bias is None:
                base_bias = torch.zeros_like(diff_bias_prec)
            new_bias = base_bias + diff_bias_prec * multiplier
        else:
            new_bias = ctx.module._lycoris_precise_bias_current

        ctx.module._lycoris_precise_weight_current = new_weight.clone()
        if diff_bias is not None:
            ctx.module._lycoris_precise_bias_current = (
                new_bias.clone() if new_bias is not None else None
            )

        return new_weight, new_bias

    def _apply_precise_weights(
        self,
        ctx: _MergeContext,
        weight_prec: torch.Tensor,
        bias_prec: torch.Tensor | None,
    ) -> None:
        ctx.weight_param.copy_(weight_prec.to(ctx.target_device, ctx.target_dtype))

        if bias_prec is not None:
            if ctx.bias_param is not None:
                ctx.bias_param.copy_(bias_prec.to(ctx.target_device, ctx.target_dtype))
            else:
                ctx.module.bias = nn.Parameter(
                    bias_prec.to(ctx.target_device, ctx.target_dtype)
                )
        elif ctx.bias_param is None:
            ctx.module.bias = None

    @staticmethod
    def _ensure_precise_snapshot(
        module: nn.Module,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> None:
        if not hasattr(module, "_lycoris_precise_weight_base"):
            base = weight.detach().cpu().double()
            module._lycoris_precise_weight_base = base
            module._lycoris_precise_weight_current = base.clone()
        if not hasattr(module, "_lycoris_precise_weight_current"):
            module._lycoris_precise_weight_current = (
                module._lycoris_precise_weight_base.clone()
            )

        if not hasattr(module, "_lycoris_precise_bias_base"):
            if bias is not None:
                base_bias = bias.detach().cpu().double()
            else:
                base_bias = None
            module._lycoris_precise_bias_base = base_bias
            module._lycoris_precise_bias_current = (
                base_bias.clone() if base_bias is not None else None
            )
        if not hasattr(module, "_lycoris_precise_bias_current"):
            module._lycoris_precise_bias_current = (
                module._lycoris_precise_bias_base.clone()
                if module._lycoris_precise_bias_base is not None
                else None
            )

    @staticmethod
    def _load_precise_snapshot(
        module: nn.Module,
        weight_param: torch.Tensor,
        bias_param: torch.Tensor | None,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        weight_param.copy_(
            module._lycoris_precise_weight_current.to(device=device, dtype=dtype)
        )
        if bias_param is not None:
            bias_snapshot = module._lycoris_precise_bias_current
            if bias_snapshot is None and module._lycoris_precise_bias_base is not None:
                bias_snapshot = module._lycoris_precise_bias_base
                module._lycoris_precise_bias_current = bias_snapshot
            if bias_snapshot is not None:
                bias_param.copy_(bias_snapshot.to(device=device, dtype=dtype))
