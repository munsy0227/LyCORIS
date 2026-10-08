from functools import cache

import torch
import torch.nn as nn

from .base import LycorisBaseModule
from ..functional.general import add_scaled
from ..logging import logger


@cache
def log_bypass_override():
    return logger.warning(
        "Automatic Bypass-Mode detected in algo=full, "
        "override with bypass_mode=False since algo=full not support bypass mode. "
        "If you are using quantized model which require bypass mode, please don't use algo=full. "
    )


class FullModule(LycorisBaseModule):
    name = "full"
    support_module = {
        "linear",
        "conv1d",
        "conv2d",
        "conv3d",
    }
    weight_list = ["diff", "diff_b"]
    weight_list_det = ["diff"]

    def __init__(
        self,
        lora_name,
        org_module: nn.Module,
        multiplier=1.0,
        lora_dim=4,
        alpha=1,
        dropout=0.0,
        rank_dropout=0.0,
        module_dropout=0.0,
        use_tucker=False,
        use_scalar=False,
        rank_dropout_scale=False,
        bypass_mode=None,
        **kwargs,
    ):
        org_bypass = bypass_mode
        super().__init__(
            lora_name,
            org_module,
            multiplier,
            dropout,
            rank_dropout,
            module_dropout,
            rank_dropout_scale,
            bypass_mode,
        )
        if bypass_mode and org_bypass is None:
            self.bypass_mode = False
            log_bypass_override()

        if self.module_type not in self.support_module:
            raise ValueError(f"{self.module_type} is not supported in Full algo.")

        if self.is_quant:
            raise ValueError(
                "Quant Linear is not supported and meaningless in Full algo."
            )

        if self.bypass_mode:
            raise ValueError("bypass mode is not supported in Full algo.")

        self.weight = nn.Parameter(torch.zeros_like(org_module.weight))
        if org_module.bias is not None:
            self.bias = nn.Parameter(torch.zeros_like(org_module.bias))
        else:
            self.bias = None
        self.is_diff = True
        self._org_weight = [self.org_module[0].weight.data.cpu().clone()]
        if self.org_module[0].bias is not None:
            self.org_bias = [self.org_module[0].bias.data.cpu().clone()]
        else:
            self.org_bias = None

    @classmethod
    @torch.no_grad()
    def make_module_from_state_dict(cls, lora_name, orig_module, diff, diff_b):
        module = cls(
            lora_name,
            orig_module,
            1,
        )
        module.weight.copy_(diff)
        if diff_b is not None:
            if orig_module.bias is not None:
                module.bias.copy_(diff_b)
            else:
                module.bias = nn.Parameter(diff_b.detach().clone())
        module.is_diff = True
        return module

    @property
    def org_weight(self):
        module = self.org_module[0]
        if self.is_diff and hasattr(module, "weight"):
            return module.weight
        return self._org_weight[0]

    @org_weight.setter
    def org_weight(self, value):
        with torch.no_grad():
            self.org_module[0].weight.copy_(value)

    def _apply(self, fn, recurse=True):
        module = super()._apply(fn, recurse)
        for name in (
            "_full_target_weight_param",
            "_full_target_bias_param",
        ):
            parameter = self.__dict__.get(name)
            if not isinstance(parameter, nn.Parameter):
                continue
            with torch.no_grad():
                transformed = fn(parameter)
            parameter.data = transformed.data
            if parameter.grad is not None:
                with torch.no_grad():
                    parameter.grad = fn(parameter.grad)
        return module

    def apply_to(self, **kwargs):
        module = self.org_module[0]
        wrappers = list(getattr(module, "_lycoris_wrappers", []))
        if self in wrappers:
            return
        if wrappers:
            raise RuntimeError(
                "FullModule cannot be stacked with another adapter on the same target."
            )
        if getattr(module, "_lycoris_lokr_merge_entries", {}):
            raise RuntimeError(
                "FullModule cannot be applied while a reversible LoKr merge "
                "is active on the target."
            )
        if any(
            name in module.__dict__
            for name in (
                "_lycoris_precise_weight_base",
                "_lycoris_precise_weight_current",
                "_lycoris_precise_bias_base",
                "_lycoris_precise_bias_current",
            )
        ):
            raise RuntimeError(
                "Finalize the target's precise merge before applying FullModule."
            )

        if tuple(module.weight.shape) != tuple(self.weight.shape):
            raise RuntimeError(
                "The target weight shape changed after FullModule was created."
            )
        if (module.bias is not None) != (self.org_bias is not None):
            raise RuntimeError(
                "The target bias structure changed after FullModule was created."
            )

        target_weight = module.weight
        target_bias = module.bias
        weight_snapshot = target_weight.detach().cpu().clone()
        bias_snapshot = (
            target_bias.detach().cpu().clone() if target_bias is not None else None
        )
        adapter_weight_snapshot = self.weight.detach().cpu().clone()
        adapter_bias_snapshot = (
            self.bias.detach().cpu().clone() if self.bias is not None else None
        )
        weight_base = target_weight.detach().to(self.weight)
        bias_base = (
            target_bias.detach().to(self.bias) if target_bias is not None else None
        )
        previous_weight_snapshot = self._org_weight
        previous_bias_snapshot = self.org_bias

        super().apply_to(**kwargs)
        weight_removed = False
        bias_removed = False
        try:
            object.__setattr__(self, "_full_target_weight_param", target_weight)
            object.__setattr__(self, "_full_target_bias_param", target_bias)
            with torch.no_grad():
                self.weight.add_(weight_base)
            self._org_weight = [weight_snapshot]
            if target_bias is not None:
                with torch.no_grad():
                    self.bias.add_(bias_base)
                self.org_bias = [bias_snapshot]
            else:
                self.org_bias = None

            delattr(module, "weight")
            weight_removed = True
            if target_bias is not None:
                delattr(module, "bias")
                bias_removed = True
            self.is_diff = False
        except Exception:
            if weight_removed:
                module.weight = target_weight
            if bias_removed:
                module.bias = target_bias
            with torch.no_grad():
                self.weight.copy_(adapter_weight_snapshot.to(self.weight))
                if self.bias is not None:
                    self.bias.copy_(adapter_bias_snapshot.to(self.bias))
            self._org_weight = previous_weight_snapshot
            self.org_bias = previous_bias_snapshot
            self.is_diff = True
            self.__dict__.pop("_full_target_weight_param", None)
            self.__dict__.pop("_full_target_bias_param", None)
            super().restore()
            raise

    def restore(self):
        module = self.org_module[0]
        if self not in getattr(module, "_lycoris_wrappers", []):
            return
        if hasattr(module, "weight"):
            raise RuntimeError(
                "The target weight was replaced while FullModule was active; "
                "refusing to overwrite the external Parameter."
            )
        if self.org_bias is not None and hasattr(module, "bias"):
            raise RuntimeError(
                "The target bias was replaced while FullModule was active; "
                "refusing to overwrite the external Parameter."
            )
        with torch.no_grad():
            self.weight.sub_(self._org_weight[0].to(self.weight))
            if self.bias is not None and self.org_bias is not None:
                self.bias.sub_(self.org_bias[0].to(self.bias))
        self.is_diff = True

        weight_param = self.__dict__.pop("_full_target_weight_param")
        bias_param = self.__dict__.pop("_full_target_bias_param")
        module.weight = weight_param
        if self.org_bias is not None:
            module.bias = bias_param
        super().restore()

    def custom_state_dict(self):
        diff = (
            self.weight.detach().cpu()
            if self.is_diff
            else self.weight.detach().cpu() - self._org_weight[0]
        )
        sd = {"diff": diff}
        if self.bias is not None:
            if self.is_diff or self.org_bias is None:
                diff_bias = self.bias.detach().cpu()
            else:
                diff_bias = self.bias.detach().cpu() - self.org_bias[0]
            sd["diff_b"] = diff_bias
        return sd

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
        diff_weight = state_dict.pop(f"{prefix}diff")
        state_dict[f"{prefix}weight"] = diff_weight + self.weight.data.to(diff_weight)
        if f"{prefix}diff_b" in state_dict:
            diff_bias = state_dict.pop(f"{prefix}diff_b")
            state_dict[f"{prefix}bias"] = diff_bias + self.bias.data.to(diff_bias)

    def make_weight(self, scale=1, device=None):
        use_rank_dropout = bool(self.rank_dropout and self.training)
        drop = (
            (torch.rand(self.dim, device=device) > self.rank_dropout).to(
                self.weight.dtype
            )
            if use_rank_dropout
            else None
        )
        if use_rank_dropout or scale != 1 or self.is_diff:
            diff_w, diff_b = self.get_diff_weight(scale, device=device)
            weight_drop = (
                drop.view(-1, *[1] * (diff_w.dim() - 1)) if drop is not None else 1
            )
            weight = add_scaled(self.org_weight.to(diff_w), diff_w, weight_drop)
            if self.is_diff and hasattr(self.org_module[0], "bias"):
                base_bias = self.org_module[0].bias
            else:
                base_bias = self.org_bias[0] if self.org_bias is not None else None
            if base_bias is not None:
                bias = base_bias.to(diff_b) + diff_b * (drop if drop is not None else 1)
            elif diff_b is not None:
                bias = diff_b * (drop if drop is not None else 1)
            else:
                bias = None
        else:
            weight = self.weight
            bias = self.bias
        return weight, bias

    def get_diff_weight(self, multiplier=1, shape=None, device=None):
        if self.is_diff:
            diff_b = None
            if self.bias is not None:
                diff_b = self.bias * multiplier
            diff = self.weight.to(device) * multiplier
            if shape is not None:
                diff = diff.view(shape)
            if diff_b is not None:
                diff_b = diff_b.to(device)
            return diff, diff_b
        org_weight = self.org_weight.to(device, dtype=self.weight.dtype)
        diff = self.weight.to(device) - org_weight
        diff_b = None
        if shape:
            diff = diff.view(shape)
        if self.bias is not None:
            org_bias = self.org_bias[0] if self.org_bias is not None else None
            diff_b = self.bias.to(device)
            if org_bias is not None:
                diff_b = diff_b - org_bias.to(device, dtype=self.bias.dtype)
        if device is not None:
            diff = diff.to(device)
            if self.bias is not None:
                diff_b = diff_b.to(device)
        diff = diff * multiplier
        if diff_b is not None:
            diff_b = diff_b * multiplier
        return diff, diff_b

    def get_merged_weight(self, multiplier=1, shape=None, device=None):
        weight, bias = self.make_weight(multiplier, device)
        if shape is not None:
            weight = weight.view(shape)
            if bias is not None:
                bias = bias.view(shape[0])
        return weight, bias

    def forward(self, x: torch.Tensor, *args, **kwargs):
        if (
            self.module_dropout
            and self.training
            and torch.rand(1) < self.module_dropout
        ):
            bias = self.org_bias[0] if self.org_bias is not None else None
            return self._weight_forward(
                x,
                self._org_weight[0].to(x),
                None if bias is None else bias.to(x),
            )

        weight, bias = self.make_weight(self.multiplier, x.device)
        return self._weight_forward(
            x,
            weight.to(x),
            None if bias is None else bias.to(x),
        )
