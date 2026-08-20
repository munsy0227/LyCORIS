import hashlib
import math
import operator
import weakref
from contextvars import ContextVar
from functools import cache

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import (
    LycorisBaseModule,
    _ensure_target_unwrapped_for_merge,
    _is_additive_lokr_stack_adapter,
    is_weight_only_fp8_linear,
)
from ..functional import factorization, rebuild_tucker
from ..functional.lokr import make_kron
from ..logging import logger


_lokr_forward_weights = ContextVar("lokr_forward_weights", default=None)
_MERGE_FINGERPRINT_CHUNK_BYTES = 1024 * 1024
_TRAINING_STATE_PREFIX = "_lycoris_lokr_training_"
_TRAINING_STATE_VERSION = 1
_TRAINING_STATE_VERSION_KEY = f"{_TRAINING_STATE_PREFIX}version"
_TRAINING_STATE_SCALAR_KEY = f"{_TRAINING_STATE_PREFIX}scalar"
_TRAINING_STATE_W1_KEY = f"{_TRAINING_STATE_PREFIX}unfolded_w1"
_TRAINING_STATE_W1_A_KEY = f"{_TRAINING_STATE_PREFIX}unfolded_w1_a"


@cache
def logging_force_full_matrix(lora_dim, dim, factor):
    logger.warning(
        f"lora_dim {lora_dim} is too large for"
        f" dim={dim} and {factor=}"
        ", using full matrix mode."
    )


@cache
def logging_disable_bypass_for_dora():
    logger.warning("LoKr DoRA requires rebuilt-weight mode; setting bypass_mode=False.")


class LokrModule(LycorisBaseModule):
    name = "kron"
    support_module = {
        "linear",
        "conv1d",
        "conv2d",
        "conv3d",
    }
    weight_list = [
        "lokr_w1",
        "lokr_w1_a",
        "lokr_w1_b",
        "lokr_w2",
        "lokr_w2_a",
        "lokr_w2_b",
        "lokr_t1",
        "lokr_t2",
        "alpha",
        "dora_scale",
        _TRAINING_STATE_VERSION_KEY,
        _TRAINING_STATE_SCALAR_KEY,
        _TRAINING_STATE_W1_KEY,
        _TRAINING_STATE_W1_A_KEY,
    ]
    weight_list_det = ["lokr_w1", "lokr_w1_a"]

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
        decompose_both=False,
        factor: int = -1,  # factorization factor
        rank_dropout_scale=False,
        weight_decompose=False,
        wd_on_out=True,
        full_matrix=False,
        bypass_mode=None,
        rs_lora=False,
        unbalanced_factorization=False,
        _dora_scale=None,
        **kwargs,
    ):
        try:
            lora_dim = operator.index(lora_dim)
        except TypeError as error:
            raise TypeError(
                f"lora_dim must be an integer, got {lora_dim!r}."
            ) from error
        if lora_dim <= 0:
            raise ValueError(f"lora_dim must be positive, got {lora_dim}.")
        if not 0 <= rank_dropout <= 1:
            raise ValueError(
                f"rank_dropout must be between 0 and 1, got {rank_dropout}."
            )
        if not 0 <= module_dropout <= 1:
            raise ValueError(
                f"module_dropout must be between 0 and 1, got {module_dropout}."
            )
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
        if self.module_type not in self.support_module:
            raise ValueError(f"{self.module_type} is not supported in LoKr algo.")

        try:
            factor = operator.index(factor)
        except TypeError as error:
            if isinstance(factor, str):
                try:
                    factor = int(factor)
                except ValueError:
                    raise TypeError(
                        f"factor must be an integer, got {factor!r}."
                    ) from error
            else:
                raise TypeError(
                    f"factor must be an integer, got {factor!r}."
                ) from error
        if factor == 0 or factor < -1:
            raise ValueError(f"factor must be -1 or a positive integer, got {factor}.")
        self.lora_dim = lora_dim
        self.tucker = False
        self.use_w1 = False
        self.use_w2 = False
        self.full_matrix = full_matrix
        self.rs_lora = rs_lora

        if self.module_type.startswith("conv"):
            in_dim = org_module.in_channels // org_module.groups
            k_size = org_module.kernel_size
            out_dim = org_module.out_channels
            self.shape = (out_dim, in_dim, *k_size)

            in_m, in_n = factorization(in_dim, factor)
            out_l, out_k = factorization(out_dim, factor)
            if unbalanced_factorization:
                out_l, out_k = out_k, out_l
            shape = ((out_l, out_k), (in_m, in_n), *k_size)  # ((a, b), (c, d), *k_size)
            self.tucker = use_tucker and any(i != 1 for i in k_size)
            if (
                decompose_both
                and lora_dim < max(shape[0][0], shape[1][0]) / 2
                and not self.full_matrix
            ):
                self.lokr_w1_a = nn.Parameter(torch.empty(shape[0][0], lora_dim))
                self.lokr_w1_b = nn.Parameter(torch.empty(lora_dim, shape[1][0]))
            else:
                self.use_w1 = True
                self.lokr_w1 = nn.Parameter(
                    torch.empty(shape[0][0], shape[1][0])
                )  # a*c, 1-mode

            if lora_dim >= max(shape[0][1], shape[1][1]) / 2 or self.full_matrix:
                if not self.full_matrix:
                    logging_force_full_matrix(lora_dim, max(in_dim, out_dim), factor)
                self.use_w2 = True
                self.lokr_w2 = nn.Parameter(
                    torch.empty(shape[0][1], shape[1][1], *k_size)
                )
            elif self.tucker:
                self.lokr_t2 = nn.Parameter(torch.empty(lora_dim, lora_dim, *shape[2:]))
                self.lokr_w2_a = nn.Parameter(
                    torch.empty(lora_dim, shape[0][1])
                )  # b, 1-mode
                self.lokr_w2_b = nn.Parameter(
                    torch.empty(lora_dim, shape[1][1])
                )  # d, 2-mode
            else:  # Conv2d not tucker
                # bigger part. weight and LoRA. [b, dim] x [dim, d*k1*k2]
                self.lokr_w2_a = nn.Parameter(torch.empty(shape[0][1], lora_dim))
                self.lokr_w2_b = nn.Parameter(
                    torch.empty(
                        lora_dim,
                        shape[1][1] * math.prod(shape[2:]),
                    )
                )
                # w1 ⊗ (w2_a x w2_b) = (a, b)⊗((c, dim)x(dim, d*k1*k2)) = (a, b)⊗(c, d*k1*k2) = (ac, bd*k1*k2)
        else:  # Linear
            in_dim = org_module.in_features
            out_dim = org_module.out_features
            self.shape = (out_dim, in_dim)

            in_m, in_n = factorization(in_dim, factor)
            out_l, out_k = factorization(out_dim, factor)
            if unbalanced_factorization:
                out_l, out_k = out_k, out_l
            shape = (
                (out_l, out_k),
                (in_m, in_n),
            )  # ((a, b), (c, d)), out_dim = a*c, in_dim = b*d
            # smaller part. weight scale
            if (
                decompose_both
                and lora_dim < max(shape[0][0], shape[1][0]) / 2
                and not self.full_matrix
            ):
                self.lokr_w1_a = nn.Parameter(torch.empty(shape[0][0], lora_dim))
                self.lokr_w1_b = nn.Parameter(torch.empty(lora_dim, shape[1][0]))
            else:
                self.use_w1 = True
                self.lokr_w1 = nn.Parameter(
                    torch.empty(shape[0][0], shape[1][0])
                )  # a*c, 1-mode
            if lora_dim < max(shape[0][1], shape[1][1]) / 2 and not self.full_matrix:
                # bigger part. weight and LoRA. [b, dim] x [dim, d]
                self.lokr_w2_a = nn.Parameter(torch.empty(shape[0][1], lora_dim))
                self.lokr_w2_b = nn.Parameter(torch.empty(lora_dim, shape[1][1]))
                # w1 ⊗ (w2_a x w2_b) = (a, b)⊗((c, dim)x(dim, d)) = (a, b)⊗(c, d) = (ac, bd)
            else:
                if not self.full_matrix:
                    logging_force_full_matrix(lora_dim, max(in_dim, out_dim), factor)
                self.use_w2 = True
                self.lokr_w2 = nn.Parameter(torch.empty(shape[0][1], shape[1][1]))

        # ``full_matrix`` describes the representation that was actually
        # selected, not only whether it was explicitly requested.  A large
        # lora_dim can promote both factors to full matrices automatically.
        self.full_matrix = self.use_w1 and self.use_w2

        self.wd = weight_decompose
        self.wd_on_out = wd_on_out
        if self.wd:
            if self.bypass_mode and not is_weight_only_fp8_linear(org_module):
                logging_disable_bypass_for_dora()
                self.bypass_mode = False

            self.dora_norm_dims = len(self.shape) - 1
            org_weight = self._current_weight()
            if org_weight.is_meta and _dora_scale is None:
                raise RuntimeError(
                    "LoKr DoRA requires a materialized base weight to initialize "
                    "its magnitude."
                )
            if tuple(org_weight.shape) != self.shape:
                raise ValueError(
                    "Dequantized base weight shape does not match the target "
                    f"module: expected {self.shape}, got {tuple(org_weight.shape)}."
                )
            initial_magnitude = (
                None
                if org_weight.is_meta
                else self._initial_dora_magnitude(org_module, org_weight)
            )
            if _dora_scale is not None:
                self._validate_dora_scale_shape(org_module, _dora_scale)
                magnitude_dtype = self._dora_accumulator_dtype(_dora_scale.dtype)
                magnitude = _dora_scale.detach().to(dtype=magnitude_dtype).clone()
            else:
                adapter_device = next(self.parameters()).device
                magnitude = initial_magnitude.to(adapter_device)
            self.dora_scale = nn.Parameter(magnitude)

        self.dropout = dropout
        self.rank_dropout = rank_dropout
        self.rank_dropout_scale = rank_dropout_scale
        self.module_dropout = module_dropout

        if isinstance(alpha, torch.Tensor):
            if alpha.numel() != 1:
                raise ValueError("alpha must be a scalar tensor.")
            alpha = alpha.detach().float().item()
        alpha = lora_dim if alpha is None or alpha == 0 else alpha
        try:
            alpha = float(alpha)
        except (TypeError, ValueError) as error:
            raise TypeError(f"alpha must be a finite number, got {alpha!r}.") from error
        if not math.isfinite(alpha):
            raise ValueError(f"alpha must be finite, got {alpha}.")
        uses_full_weight_matrices = self.full_matrix
        if uses_full_weight_matrices:
            # use scale = 1
            alpha = lora_dim

        r_factor = lora_dim
        if self.rs_lora:
            r_factor = math.sqrt(r_factor)

        self.scale = 1.0 if uses_full_weight_matrices else alpha / r_factor

        stored_alpha = (
            alpha if uses_full_weight_matrices else alpha * (lora_dim / r_factor)
        )
        self.register_buffer("alpha", torch.tensor(stored_alpha))

        if use_scalar:
            self.scalar = nn.Parameter(torch.tensor(0.0))
        else:
            self.register_buffer("scalar", torch.tensor(1.0), persistent=False)

        if self.use_w2:
            if use_scalar:
                torch.nn.init.kaiming_uniform_(self.lokr_w2, a=math.sqrt(5))
            else:
                torch.nn.init.constant_(self.lokr_w2, 0)
        else:
            if self.tucker:
                torch.nn.init.kaiming_uniform_(self.lokr_t2, a=math.sqrt(5))
            torch.nn.init.kaiming_uniform_(self.lokr_w2_a, a=math.sqrt(5))
            if use_scalar:
                torch.nn.init.kaiming_uniform_(self.lokr_w2_b, a=math.sqrt(5))
            else:
                torch.nn.init.constant_(self.lokr_w2_b, 0)

        if self.use_w1:
            torch.nn.init.kaiming_uniform_(self.lokr_w1, a=math.sqrt(5))
        else:
            torch.nn.init.kaiming_uniform_(self.lokr_w1_a, a=math.sqrt(5))
            torch.nn.init.kaiming_uniform_(self.lokr_w1_b, a=math.sqrt(5))

    def _apply(self, fn, recurse=True):
        # DoRA norms are accumulated in float32 for fp16/bf16 targets.  Keep the
        # trainable magnitude in that same master precision when the rest of a
        # network is cast for full low-precision training.  Wrap the conversion
        # itself instead of restoring data afterwards: cross-device meta moves
        # cannot accept set_data(), and to_empty() must retain its empty-storage
        # semantics.  PyTorch's normal Module._apply lifecycle still preserves
        # the Parameter object for ordinary .to() calls, as required when an
        # optimizer was constructed before the cast.
        dora_parameter = getattr(self, "dora_scale", None)
        if not isinstance(dora_parameter, nn.Parameter) or dora_parameter.is_meta:
            return super()._apply(fn, recurse)
        dora_gradient = dora_parameter.grad

        def preserve_dora_master(tensor):
            transformed = fn(tensor)
            if tensor is not dora_parameter and tensor is not dora_gradient:
                return transformed
            if transformed.is_meta or transformed.dtype not in {
                torch.float16,
                torch.bfloat16,
            }:
                return transformed
            master_dtype = self._dora_accumulator_dtype(tensor.dtype)
            return tensor.to(device=transformed.device, dtype=master_dtype)

        module = super()._apply(preserve_dora_master, recurse)
        transformed_parameter = self.dora_scale
        if (
            transformed_parameter is not dora_parameter
            and not transformed_parameter.is_meta
        ):
            # ``overwrite_module_params_on_conversion`` may ask PyTorch to
            # replace Parameters during .to().  Reattach the original object so
            # an optimizer created before the cast cannot retain a stale DoRA
            # magnitude reference.  Using the already transformed tensor also
            # preserves to_empty() semantics.
            dora_parameter.data = transformed_parameter.detach()
            dora_parameter.grad = transformed_parameter.grad
            self._parameters["dora_scale"] = dora_parameter
        return module

    @classmethod
    def _dora_scale_shapes(cls, orig_module):
        weight_shape = cls._target_weight_shape(orig_module)
        output_shape = (weight_shape[0], *[1] * (len(weight_shape) - 1))
        input_shape = (
            1,
            weight_shape[1],
            *[1] * (len(weight_shape) - 2),
        )
        grouped_input_shape = None
        groups = getattr(orig_module, "groups", 1)
        if groups > 1 and len(weight_shape) > 2:
            grouped_input_shape = (
                groups,
                1,
                weight_shape[1],
                *[1] * (len(weight_shape) - 2),
            )
        return output_shape, input_shape, grouped_input_shape

    @staticmethod
    def _dora_accumulator_dtype(dtype):
        if dtype in {torch.float16, torch.bfloat16}:
            return torch.float32
        return dtype

    @classmethod
    def _validate_dora_scale_shape(cls, orig_module, dora_scale):
        valid_shapes = {
            shape for shape in cls._dora_scale_shapes(orig_module) if shape is not None
        }
        magnitude_shape = tuple(dora_scale.shape)
        if magnitude_shape not in valid_shapes:
            raise ValueError(
                "Invalid dora_scale shape for target weight: "
                f"weight={cls._target_weight_shape(orig_module)}, "
                f"dora_scale={magnitude_shape}, expected one of "
                f"{sorted(valid_shapes)}."
            )

    def _initial_dora_magnitude(self, orig_module, org_weight):
        compute_dtype = self._dora_accumulator_dtype(org_weight.dtype)
        direction = org_weight.to(dtype=compute_dtype)
        groups = getattr(orig_module, "groups", 1)
        if self.wd_on_out:
            norm_dims = tuple(range(1, direction.dim()))
            return torch.linalg.vector_norm(
                direction,
                dim=norm_dims,
                keepdim=True,
            )
        if groups > 1 and self.module_type.startswith("conv"):
            out_per_group = direction.shape[0] // groups
            grouped = direction.reshape(
                groups,
                out_per_group,
                direction.shape[1],
                *direction.shape[2:],
            )
            norm_dims = (1, *range(3, grouped.dim()))
            return torch.linalg.vector_norm(
                grouped,
                dim=norm_dims,
                keepdim=True,
            )
        norm_dims = (0, *range(2, direction.dim()))
        return torch.linalg.vector_norm(
            direction,
            dim=norm_dims,
            keepdim=True,
        )

    @classmethod
    def _infer_factorization_config(
        cls,
        out_dim,
        in_dim,
        w1_shape,
        w2_shape,
    ):
        candidates = {-1}
        for dimension in (out_dim, in_dim):
            for candidate in range(1, math.isqrt(dimension) + 1):
                if dimension % candidate == 0:
                    candidates.add(candidate)
                    candidates.add(dimension // candidate)

        expected_in = (w1_shape[1], w2_shape[1])
        expected_out = (w1_shape[0], w2_shape[0])
        ordered_candidates = [-1, *sorted(candidates - {-1})]
        for factor in ordered_candidates:
            if factorization(in_dim, factor) != expected_in:
                continue
            output_factors = factorization(out_dim, factor)
            if output_factors == expected_out:
                return factor, False
            if output_factors[::-1] == expected_out:
                return factor, True

        raise ValueError(
            "Cannot infer LoKr factorization from checkpoint shapes: "
            f"weight=({out_dim}, {in_dim}), w1={w1_shape}, w2={w2_shape}."
        )

    @staticmethod
    def _target_weight_shape(orig_module):
        if isinstance(orig_module, nn.Linear):
            return (orig_module.out_features, orig_module.in_features)
        if isinstance(orig_module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            return (
                orig_module.out_channels,
                orig_module.in_channels // orig_module.groups,
                *orig_module.kernel_size,
            )
        return tuple(orig_module.weight.shape)

    @staticmethod
    def _infer_wd_on_out(orig_module, dora_scale):
        if dora_scale is None:
            return True

        weight_shape = LokrModule._target_weight_shape(orig_module)
        output_shape, input_shape, grouped_input_shape = LokrModule._dora_scale_shapes(
            orig_module
        )
        magnitude_shape = tuple(dora_scale.shape)
        if magnitude_shape == output_shape:
            return True
        if magnitude_shape in {input_shape, grouped_input_shape}:
            return False
        raise ValueError(
            "Cannot infer wd_on_out from dora_scale shape: "
            f"weight={weight_shape}, dora_scale={magnitude_shape}."
        )

    @classmethod
    @torch.no_grad()
    def make_module_from_state_dict(
        cls,
        lora_name,
        orig_module,
        w1,
        w1a,
        w1b,
        w2,
        w2a,
        w2b,
        _,
        t2,
        alpha,
        dora_scale,
        training_version=None,
        training_scalar=None,
        training_w1=None,
        training_w1_a=None,
    ):
        if w1 is None:
            if w1a is None or w1b is None:
                raise ValueError(
                    "A LoKr checkpoint must contain lokr_w1 or both "
                    "lokr_w1_a and lokr_w1_b."
                )
        elif w1a is not None or w1b is not None:
            raise ValueError(
                "A LoKr checkpoint cannot mix lokr_w1 with low-rank w1 factors."
            )

        training_values = (
            training_version,
            training_scalar,
            training_w1,
            training_w1_a,
        )
        has_training_state = any(value is not None for value in training_values)
        if has_training_state:
            if training_version is None or training_scalar is None:
                raise ValueError(
                    "Incomplete LoKr training state: version and scalar are required."
                )
            if (
                not isinstance(training_version, torch.Tensor)
                or training_version.dim() != 0
                or training_version.is_meta
                or training_version.dtype != torch.int64
                or training_version.item() != _TRAINING_STATE_VERSION
            ):
                raise ValueError(
                    "Unsupported LoKr training state version; expected "
                    f"a materialized int64 scalar equal to "
                    f"{_TRAINING_STATE_VERSION}."
                )
            if (
                not isinstance(training_scalar, torch.Tensor)
                or training_scalar.dim() != 0
                or training_scalar.is_meta
                or not training_scalar.is_floating_point()
            ):
                raise ValueError(
                    "LoKr training scalar must be a materialized floating-point "
                    "scalar tensor."
                )
            if w1 is not None:
                if training_w1 is None or training_w1_a is not None:
                    raise ValueError(
                        "Full-factor LoKr training state requires only unfolded_w1."
                    )
                if (
                    not isinstance(training_w1, torch.Tensor)
                    or training_w1.is_meta
                    or tuple(training_w1.shape) != tuple(w1.shape)
                ):
                    raise ValueError(
                        "LoKr training unfolded_w1 must match the portable first "
                        "factor exactly."
                    )
                w1 = training_w1
            else:
                if training_w1_a is None or training_w1 is not None:
                    raise ValueError(
                        "Low-rank LoKr training state requires only unfolded_w1_a."
                    )
                if (
                    not isinstance(training_w1_a, torch.Tensor)
                    or training_w1_a.is_meta
                    or tuple(training_w1_a.shape) != tuple(w1a.shape)
                ):
                    raise ValueError(
                        "LoKr training unfolded_w1_a must match the portable first "
                        "factor exactly."
                    )
                w1a = training_w1_a
        if w2 is None:
            if w2a is None or w2b is None:
                raise ValueError(
                    "A LoKr checkpoint must contain lokr_w2 or both "
                    "lokr_w2_a and lokr_w2_b."
                )
        elif w2a is not None or w2b is not None or t2 is not None:
            raise ValueError(
                "A LoKr checkpoint cannot mix lokr_w2 with low-rank or "
                "Tucker w2 factors."
            )
        if alpha is None or (isinstance(alpha, torch.Tensor) and alpha.numel() != 1):
            raise ValueError("A LoKr checkpoint must contain a scalar alpha.")
        if w1 is not None:
            if w1.dim() != 2:
                raise ValueError(f"lokr_w1 must be 2D, got {tuple(w1.shape)}.")
        else:
            if w1a.dim() != 2 or w1b.dim() != 2:
                raise ValueError("lokr_w1_a and lokr_w1_b must both be 2D.")
            if w1a.size(1) != w1b.size(0):
                raise ValueError(
                    "LoKr w1 factor ranks do not match: "
                    f"w1_a={tuple(w1a.shape)}, w1_b={tuple(w1b.shape)}."
                )

        target_shape = cls._target_weight_shape(orig_module)
        kernel_shape = tuple(target_shape[2:])
        if w2 is not None:
            expected_dims = len(target_shape)
            if w2.dim() != expected_dims or tuple(w2.shape[2:]) != kernel_shape:
                raise ValueError(
                    "lokr_w2 does not match the target kernel dimensions: "
                    f"w2={tuple(w2.shape)}, kernel={kernel_shape}."
                )
        elif t2 is not None:
            if not kernel_shape:
                raise ValueError("Tucker LoKr factors require a convolutional target.")
            if w2a.dim() != 2 or w2b.dim() != 2:
                raise ValueError("Tucker lokr_w2_a and lokr_w2_b must both be 2D.")
            if t2.dim() != len(target_shape) or tuple(t2.shape[2:]) != kernel_shape:
                raise ValueError(
                    "lokr_t2 does not match the target kernel dimensions: "
                    f"t2={tuple(t2.shape)}, kernel={kernel_shape}."
                )
            if (
                t2.size(0) != w2a.size(0)
                or t2.size(1) != w2b.size(0)
                or t2.size(0) != t2.size(1)
            ):
                raise ValueError(
                    "LoKr Tucker ranks do not match: "
                    f"t2={tuple(t2.shape)}, w2_a={tuple(w2a.shape)}, "
                    f"w2_b={tuple(w2b.shape)}."
                )
        else:
            valid_w2b_dims = {2}
            if kernel_shape:
                valid_w2b_dims.add(len(target_shape))
            if w2a.dim() != 2 or w2b.dim() not in valid_w2b_dims:
                raise ValueError(
                    "lokr_w2_a must be 2D and lokr_w2_b must be either "
                    "flattened or match the target convolution dimensions."
                )
            if w2b.dim() > 2 and tuple(w2b.shape[2:]) != kernel_shape:
                raise ValueError(
                    "lokr_w2_b does not match the target kernel dimensions: "
                    f"w2b={tuple(w2b.shape)}, kernel={kernel_shape}."
                )
            if w2a.size(1) != w2b.size(0):
                raise ValueError(
                    "LoKr w2 factor ranks do not match: "
                    f"w2_a={tuple(w2a.shape)}, w2_b={tuple(w2b.shape)}."
                )
            if w2b.dim() > 2:
                w2b = w2b.reshape(w2b.size(0), -1)

        full_matrix = False
        if w1a is not None:
            lora_dim = w1a.size(1)
        elif w2a is not None:
            lora_dim = w2a.size(0) if t2 is not None else w2a.size(1)
        else:
            full_matrix = True
            alpha_value = float(alpha)
            lora_dim = (
                int(alpha_value) if alpha_value > 0 and alpha_value.is_integer() else 1
            )

        if w1 is None:
            w1_shape = (w1a.size(0), w1b.size(1))
        else:
            w1_shape = tuple(w1.shape[:2])

        if w2 is not None:
            w2_shape = tuple(w2.shape[:2])
        elif t2 is not None:
            w2_shape = (w2a.size(1), w2b.size(1))
        else:
            target_shape = cls._target_weight_shape(orig_module)
            kernel_elements = math.prod(target_shape[2:])
            flattened_input = math.prod(w2b.shape[1:])
            if flattened_input % kernel_elements != 0:
                raise ValueError(
                    "Invalid convolutional LoKr checkpoint shape: "
                    f"w2b={tuple(w2b.shape)}, kernel={target_shape[2:]}."
                )
            w2_shape = (w2a.size(0), flattened_input // kernel_elements)

        out_dim, in_dim = target_shape[:2]
        if w1_shape[0] * w2_shape[0] != out_dim or w1_shape[1] * w2_shape[1] != in_dim:
            raise ValueError(
                "LoKr checkpoint factors do not match the target weight: "
                f"target=({out_dim}, {in_dim}), w1={w1_shape}, w2={w2_shape}."
            )
        factor, unbalanced_factorization = cls._infer_factorization_config(
            out_dim,
            in_dim,
            w1_shape,
            w2_shape,
        )
        wd_on_out = cls._infer_wd_on_out(orig_module, dora_scale)

        if w1a is not None and w2a is not None:
            w2_rank = w2a.size(0) if t2 is not None else w2a.size(1)
            if w1a.size(1) != w2_rank:
                raise ValueError(
                    "LoKr checkpoint uses inconsistent ranks for w1 and w2: "
                    f"w1_rank={w1a.size(1)}, w2_rank={w2_rank}."
                )

        # Reconstruct placeholders on meta so a full-matrix checkpoint never
        # has two materialized copies of its base-sized factors at once.
        with torch.device("meta"):
            module = cls(
                lora_name,
                orig_module,
                1,
                lora_dim,
                float(alpha),
                use_tucker=t2 is not None,
                decompose_both=w1 is None,
                factor=factor,
                weight_decompose=dora_scale is not None,
                wd_on_out=wd_on_out,
                full_matrix=full_matrix,
                unbalanced_factorization=unbalanced_factorization,
                use_scalar=has_training_state,
                _dora_scale=dora_scale,
            )

        # The checkpoint representation is authoritative. Constructor rank
        # heuristics can legitimately choose a different representation for
        # the same dimensions, so remove every placeholder and restore only
        # the factors that are actually present in the checkpoint.
        for factor_name in (
            "lokr_w1",
            "lokr_w1_a",
            "lokr_w1_b",
            "lokr_w2",
            "lokr_w2_a",
            "lokr_w2_b",
            "lokr_t2",
        ):
            if hasattr(module, factor_name):
                delattr(module, factor_name)
        module.use_w1 = w1 is not None
        module.use_w2 = w2 is not None
        module.tucker = t2 is not None
        module.full_matrix = module.use_w1 and module.use_w2
        module.lora_dim = lora_dim
        module.scale = 1.0 if module.full_matrix else float(alpha) / lora_dim

        def restore_parameter(name, value):
            setattr(
                module,
                name,
                nn.Parameter(value.detach().clone()),
            )

        if w1 is not None:
            restore_parameter("lokr_w1", w1)
        else:
            restore_parameter("lokr_w1_a", w1a)
            restore_parameter("lokr_w1_b", w1b)
        if w2 is not None:
            restore_parameter("lokr_w2", w2)
        else:
            restore_parameter("lokr_w2_a", w2a)
            restore_parameter("lokr_w2_b", w2b)
        if t2 is not None:
            restore_parameter("lokr_t2", t2)
        if dora_scale is not None:
            dora_dtype = cls._dora_accumulator_dtype(dora_scale.dtype)
            restore_parameter("dora_scale", dora_scale.to(dtype=dora_dtype))

        reference = next(
            (
                tensor
                for tensor in (w1, w1a, w1b, w2, w2a, w2b, t2, dora_scale)
                if tensor is not None and not tensor.is_meta
            ),
            None,
        )
        if reference is None:
            raise ValueError(
                "A LoKr checkpoint must contain at least one materialized factor."
            )
        if has_training_state:
            module.scalar = nn.Parameter(training_scalar.detach().clone())
        else:
            module.scalar = reference.new_ones(())
        module.dtype_tensor = reference.new_zeros(())
        if isinstance(alpha, torch.Tensor):
            module.alpha = alpha.detach().clone()
        else:
            module.alpha = reference.new_tensor(alpha)

        # Optimizer state uses positional parameter IDs.  Reconstructing the
        # authoritative checkpoint representation above removes and re-adds
        # factors, which would otherwise place them after the already-created
        # DoRA magnitude and scalar.  Restore the same registration order as a
        # normally constructed module so a training-state reconstruction can
        # load its optimizer state without assigning slots to different
        # parameter shapes.
        parameter_order = []
        if module.use_w1:
            parameter_order.append("lokr_w1")
        else:
            parameter_order.extend(("lokr_w1_a", "lokr_w1_b"))
        if module.use_w2:
            parameter_order.append("lokr_w2")
        elif module.tucker:
            parameter_order.extend(("lokr_t2", "lokr_w2_a", "lokr_w2_b"))
        else:
            parameter_order.extend(("lokr_w2_a", "lokr_w2_b"))
        if module.wd:
            parameter_order.append("dora_scale")
        if isinstance(module.scalar, nn.Parameter):
            parameter_order.append("scalar")

        ordered_parameters = type(module._parameters)()
        for name in parameter_order:
            parameter = module._parameters.get(name)
            if parameter is not None:
                ordered_parameters[name] = parameter
        for name, parameter in module._parameters.items():
            if name not in ordered_parameters:
                ordered_parameters[name] = parameter
        module._parameters = ordered_parameters
        return module

    @staticmethod
    def is_training_state_key(key):
        """Return whether a network state key is LoKr resume-only data."""
        return key.rsplit(".", 1)[-1].startswith(_TRAINING_STATE_PREFIX)

    @classmethod
    def strip_training_state_keys(cls, state_dict):
        """Remove resume-only entries in place before a portable export."""
        for key in tuple(state_dict):
            if cls.is_training_state_key(key):
                state_dict.pop(key)
        return state_dict

    @classmethod
    def export_master_dtype_keys(cls, state_dict):
        """Return LoKr auxiliary keys that must keep their structural dtype."""
        lokr_prefixes = {
            key.rpartition(".")[0]
            for key in state_dict
            if key.rpartition(".")[2] in cls.weight_list_det
        }
        return {
            key
            for key in state_dict
            if key.rpartition(".")[0] in lokr_prefixes
            and key.rpartition(".")[2] == "dora_scale"
        }

    @torch.no_grad()
    def _consume_training_state(self, state_dict, prefix, local_metadata):
        version_key = f"{prefix}{_TRAINING_STATE_VERSION_KEY}"
        scalar_key = f"{prefix}{_TRAINING_STATE_SCALAR_KEY}"
        raw_w1_key = f"{prefix}{_TRAINING_STATE_W1_KEY}"
        raw_w1_a_key = f"{prefix}{_TRAINING_STATE_W1_A_KEY}"
        auxiliary_keys = (version_key, scalar_key, raw_w1_key, raw_w1_a_key)
        present_keys = {key for key in auxiliary_keys if key in state_dict}

        if not present_keys:
            return False
        if version_key not in present_keys:
            raise RuntimeError(
                "LoKr training state is missing its format version marker."
            )

        version = state_dict[version_key]
        if (
            not isinstance(version, torch.Tensor)
            or version.dim() != 0
            or version.is_meta
            or version.dtype != torch.int64
            or version.item() != _TRAINING_STATE_VERSION
        ):
            raise RuntimeError(
                "Unsupported LoKr training state version; expected "
                f"a materialized int64 scalar equal to "
                f"{_TRAINING_STATE_VERSION}."
            )

        if not isinstance(self.scalar, nn.Parameter):
            raise RuntimeError(
                "LoKr training state contains a trainable scalar, but the target "
                "module was created with use_scalar=False. Load a portable "
                "checkpoint for inference or recreate the training module with "
                "use_scalar=True."
            )

        expected_raw_key = raw_w1_key if self.use_w1 else raw_w1_a_key
        unexpected_raw_key = raw_w1_a_key if self.use_w1 else raw_w1_key
        missing_keys = [
            key for key in (scalar_key, expected_raw_key) if key not in present_keys
        ]
        if missing_keys or unexpected_raw_key in present_keys:
            raise RuntimeError(
                "Incomplete or incompatible LoKr training state for the first "
                "Kronecker factor."
            )

        scalar = state_dict[scalar_key]
        raw_w1 = state_dict[expected_raw_key]
        if (
            not isinstance(scalar, torch.Tensor)
            or scalar.dim() != 0
            or scalar.is_meta
            or not scalar.is_floating_point()
        ):
            raise RuntimeError(
                "LoKr training scalar must be a materialized floating-point "
                "scalar tensor."
            )
        if not isinstance(raw_w1, torch.Tensor) or raw_w1.dim() != 2:
            raise RuntimeError(
                "LoKr training state must contain an unfolded 2D first factor."
            )

        for key in auxiliary_keys:
            state_dict.pop(key, None)
        factor_name = "lokr_w1" if self.use_w1 else "lokr_w1_a"
        state_dict[f"{prefix}{factor_name}"] = raw_w1
        state_dict[f"{prefix}scalar"] = scalar
        return True

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
        has_training_state = self._consume_training_state(
            state_dict,
            prefix,
            local_metadata,
        )
        factor_keys = (
            "lokr_w1",
            "lokr_w1_a",
            "lokr_w1_b",
            "lokr_w2",
            "lokr_w2_a",
            "lokr_w2_b",
            "lokr_t2",
            "dora_scale",
        )
        reference = next(
            (
                value
                for key in factor_keys
                if isinstance((value := state_dict.get(f"{prefix}{key}")), torch.Tensor)
                and not value.is_meta
            ),
            None,
        )
        if self.wd:
            dora_scale_key = f"{prefix}dora_scale"
            checkpoint_magnitude = state_dict.get(dora_scale_key)
            if isinstance(checkpoint_magnitude, torch.Tensor):
                self._validate_dora_scale_shape(
                    self.org_module[0],
                    checkpoint_magnitude,
                )
                state_dict[dora_scale_key] = checkpoint_magnitude.to(
                    dtype=self._dora_accumulator_dtype(checkpoint_magnitude.dtype)
                )

        if not has_training_state:
            # Portable LoKr checkpoints fold the scalar into the first factor.
            # Existing checkpoints intentionally have no scalar key.
            scalar_key = f"{prefix}scalar"
            if isinstance(self.scalar, nn.Parameter):
                state_dict[scalar_key] = (
                    reference.new_ones(())
                    if reference is not None
                    else torch.ones_like(self.scalar)
                )
            else:
                state_dict.pop(scalar_key, None)
                if self.scalar.is_meta and reference is not None:
                    self.scalar = reference.new_ones(())
                else:
                    self.scalar.fill_(1.0)
        assign = local_metadata.get("assign_to_params_buffers", False)
        if (self.dtype_tensor.is_meta or assign) and reference is not None:
            self.dtype_tensor = reference.new_zeros(())

    def _factor_tensors(self):
        tensors = []
        for name in (
            "lokr_w1",
            "lokr_w1_a",
            "lokr_w1_b",
            "lokr_w2",
            "lokr_w2_a",
            "lokr_w2_b",
            "lokr_t2",
        ):
            tensor = getattr(self, name, None)
            if tensor is not None:
                tensors.append(tensor)
        return tensors

    def _weight_compute_dtype(self, base_dtype=None, override=None):
        if override is not None:
            return override
        dtype = base_dtype
        for tensor in self._factor_tensors():
            dtype = (
                tensor.dtype
                if dtype is None
                else torch.promote_types(
                    dtype,
                    tensor.dtype,
                )
            )
        if isinstance(self.scalar, nn.Parameter):
            dtype = torch.promote_types(dtype, self.scalar.dtype)
        if self.wd:
            dtype = torch.promote_types(dtype, self.dora_scale.dtype)
            dtype = self._dora_accumulator_dtype(dtype)
        return dtype

    @staticmethod
    def _factor_for_compute(tensor, device, dtype):
        return tensor.to(device=device, dtype=dtype)

    def get_weight(self, shape, *, device=None, dtype=None):
        factors = self._factor_tensors()
        if not factors:
            raise RuntimeError("LoKr has no factor tensors to rebuild.")
        if device is None:
            device = factors[0].device
        dtype = self._weight_compute_dtype(override=dtype)

        device_type = torch.device(device).type
        with torch.autocast(device_type=device_type, enabled=False):
            if self.use_w1:
                w1 = self._factor_for_compute(self.lokr_w1, device, dtype)
            else:
                w1a = self._factor_for_compute(self.lokr_w1_a, device, dtype)
                w1b = self._factor_for_compute(self.lokr_w1_b, device, dtype)
                w1 = w1a @ w1b

            if self.use_w2:
                w2 = self._factor_for_compute(self.lokr_w2, device, dtype)
            else:
                w2a = self._factor_for_compute(self.lokr_w2_a, device, dtype)
                w2b = self._factor_for_compute(self.lokr_w2_b, device, dtype)
                if self.tucker:
                    t2 = self._factor_for_compute(self.lokr_t2, device, dtype)
                    w2 = rebuild_tucker(t2, w2a, w2b)
                else:
                    w2 = w2a @ w2b

            weight = make_kron(
                w1,
                w2,
                self.scale,
            )
        if shape is not None:
            weight = weight.view(shape)
        if self.training and self.rank_dropout:
            drop = (
                torch.rand(weight.size(0), device=weight.device) > self.rank_dropout
            ).to(dtype)
            drop = drop.view(-1, *[1] * len(weight.shape[1:]))
            if self.rank_dropout_scale:
                keep_probability = 1 - self.rank_dropout
                if keep_probability > 0:
                    drop /= keep_probability
            weight *= drop
        return weight

    def _get_effective_diff_weight(self, shape, base_weight, compute_dtype=None):
        diff = self.get_weight(
            shape,
            device=base_weight.device,
            dtype=compute_dtype,
        )
        scalar = self.scalar.to(device=diff.device, dtype=diff.dtype)
        return diff * scalar

    def _calculate_merged_weight(
        self,
        base_weight,
        multiplier=1,
        shape=None,
        compute_dtype=None,
    ):
        target_shape = tuple(base_weight.shape) if shape is None else shape
        compute_dtype = self._weight_compute_dtype(
            base_weight.dtype,
            override=compute_dtype,
        )
        base_weight = base_weight.to(dtype=compute_dtype)
        diff = self._get_effective_diff_weight(
            target_shape,
            base_weight,
            compute_dtype,
        )

        if self.wd:
            return self.apply_weight_decompose(
                base_weight + diff,
                multiplier,
                base_weight=base_weight,
            )
        return base_weight + diff * multiplier

    def get_diff_weight(self, multiplier=1, shape=None, device=None):
        base_weight = self._current_weight()
        if device is not None:
            base_weight = base_weight.to(device)

        merged = self._calculate_merged_weight(base_weight, multiplier, shape)
        return merged - base_weight.to(merged), None

    def get_merged_weight(self, multiplier=1, shape=None, device=None):
        base_weight = self._current_weight()
        if device is not None:
            base_weight = base_weight.to(device)
        return self._calculate_merged_weight(base_weight, multiplier, shape), None

    def apply_to(self, **kwargs):
        module = self.org_module[0]
        if getattr(self, "_lokr_merge_committed", False):
            raise RuntimeError(
                "This LoKr adapter was committed into the target weight and "
                "cannot be applied again."
            )
        entries = getattr(module, "_lycoris_lokr_merge_entries", {})
        if self in entries:
            raise RuntimeError(
                "Undo or finalize this LoKr adapter's reversible merge before "
                "applying it as a forward wrapper."
            )
        wrappers = list(getattr(module, "_lycoris_wrappers", []))
        different_wrappers = [
            wrapper for wrapper in wrappers if not isinstance(wrapper, LokrModule)
        ]
        if (
            self not in wrappers
            and different_wrappers
            and (
                self.wd
                or any(
                    not _is_additive_lokr_stack_adapter(wrapper)
                    for wrapper in different_wrappers
                )
            )
        ):
            raise RuntimeError(
                "Stacking LoKr with this adapter on the same target is not "
                "supported because the composition is base- or order-dependent."
            )
        already_applied = self in wrappers
        super().apply_to(**kwargs)
        if not already_applied and self in getattr(module, "_lycoris_wrappers", []):
            order = getattr(module, "_lycoris_lokr_next_order", 0)
            self._lokr_application_order = order
            module._lycoris_lokr_next_order = order + 1

    @staticmethod
    def _tensor_values_equal(actual, expected):
        expected = expected.to(device=actual.device, dtype=actual.dtype)
        return bool(
            torch.allclose(
                actual.detach(),
                expected,
                rtol=0.0,
                atol=0.0,
                equal_nan=True,
            )
        )

    @staticmethod
    def _tensor_fingerprint(tensor):
        """Return an exact-value digest without a persistent tensor-sized copy."""
        if tensor.layout != torch.strided:
            raise RuntimeError(
                "LoKr merge conflict detection requires strided target and "
                "factor tensors."
            )

        digest = hashlib.sha256()
        metadata = (
            str(tensor.dtype),
            tuple(tensor.shape),
            tuple(tensor.stride()),
            str(tensor.layout),
        )
        digest.update(repr(metadata).encode("utf-8"))

        tensor = tensor.detach()
        if tensor.numel() == 0:
            return digest.digest()

        max_elements = max(
            1,
            _MERGE_FINGERPRINT_CHUNK_BYTES // tensor.element_size(),
        )

        def update(chunk):
            chunk_bytes = (
                chunk.resolve_conj()
                .resolve_neg()
                .contiguous()
                .view(torch.uint8)
                .reshape(-1)
                .to(device="cpu")
            )
            digest.update(chunk_bytes.numpy().tobytes())

        def visit(value):
            if value.numel() <= max_elements:
                update(value)
                return

            split_dim = next(
                index for index, size in enumerate(value.shape) if size > 1
            )
            elements_per_index = value.numel() // value.shape[split_dim]
            step = max(1, max_elements // elements_per_index)
            for start in range(0, value.shape[split_dim], step):
                length = min(step, value.shape[split_dim] - start)
                visit(value.narrow(split_dim, start, length))

        if tensor.is_contiguous():
            flat = tensor.reshape(-1)
            for start in range(0, flat.numel(), max_elements):
                update(flat.narrow(0, start, min(max_elements, flat.numel() - start)))
        else:
            visit(tensor)
        return digest.digest()

    def _merge_state_fingerprint(self):
        """Commit to every adapter value that can affect merge composition."""
        digest = hashlib.sha256()
        module = self.org_module[0]
        config = (
            self.module_type,
            tuple(self.shape),
            self.use_w1,
            self.use_w2,
            self.tucker,
            self.wd,
            self.wd_on_out,
            self.scale,
            getattr(module, "groups", None),
            getattr(self, "_lokr_application_order", None),
        )
        digest.update(repr(config).encode("utf-8"))

        for name in (
            "lokr_w1",
            "lokr_w1_a",
            "lokr_w1_b",
            "lokr_w2",
            "lokr_w2_a",
            "lokr_w2_b",
            "lokr_t2",
            "scalar",
            "dora_scale",
        ):
            digest.update(name.encode("utf-8"))
            tensor = getattr(self, name, None)
            if tensor is None:
                digest.update(b"\x00")
                continue
            if not isinstance(tensor, torch.Tensor):
                raise RuntimeError(
                    f"LoKr merge state {name} was replaced with a non-Tensor value."
                )
            digest.update(b"\x01")
            digest.update(self._tensor_fingerprint(tensor))
        return digest.digest()

    @staticmethod
    def _clear_merge_ledger(module):
        for name in (
            "_lycoris_lokr_merge_base",
            "_lycoris_lokr_merge_entries",
            "_lycoris_lokr_merge_order",
            "_lycoris_lokr_merge_precise",
            "_lycoris_lokr_merge_weight_param",
            "_lycoris_lokr_merge_weight_param_ref",
            "_lycoris_lokr_merge_weight_version",
            "_lycoris_lokr_merge_weight_fingerprint",
            "_lycoris_lokr_merge_adapter_fingerprints",
        ):
            module.__dict__.pop(name, None)

    @classmethod
    def _merge_target_conflict(cls, module, weight_param):
        stored_param_ref = module.__dict__.get("_lycoris_lokr_merge_weight_param_ref")
        stored_param = (
            stored_param_ref()
            if isinstance(stored_param_ref, weakref.ReferenceType)
            else module.__dict__.get("_lycoris_lokr_merge_weight_param")
        )
        if stored_param is not weight_param:
            return "parameter"

        stored_fingerprint = module.__dict__.get(
            "_lycoris_lokr_merge_weight_fingerprint"
        )
        if stored_fingerprint is None:
            return "missing_fingerprint"
        if cls._tensor_fingerprint(weight_param) == stored_fingerprint:
            return None
        if weight_param._version != module._lycoris_lokr_merge_weight_version:
            return "tracked_write"
        return "untracked_write"

    @staticmethod
    def _raise_merge_target_conflict(conflict):
        if conflict == "parameter":
            raise RuntimeError(
                "The target weight Parameter was replaced outside the active "
                "LoKr merge ledger; refusing to overwrite it."
            )
        if conflict == "tracked_write":
            raise RuntimeError(
                "The target weight changed outside the active LoKr merge "
                "ledger; refusing to overwrite the external update."
            )
        if conflict == "untracked_write":
            raise RuntimeError(
                "The target weight changed through an untracked data write "
                "outside the active LoKr merge ledger; refusing to overwrite it."
            )
        raise RuntimeError(
            "The active LoKr merge ledger has no target fingerprint; refusing "
            "to overwrite the target."
        )

    @staticmethod
    def _changed_merge_adapters(module):
        entries = module._lycoris_lokr_merge_entries
        fingerprints = module.__dict__.get(
            "_lycoris_lokr_merge_adapter_fingerprints",
            {},
        )
        return [
            adapter
            for adapter in entries
            if fingerprints.get(adapter) != adapter._merge_state_fingerprint()
        ]

    @staticmethod
    def _raise_merge_factor_conflict():
        raise RuntimeError(
            "An active LoKr factor changed after the reversible merge. Normal "
            "merge, partial undo, and finalize operations cannot safely infer "
            "the earlier composition. Resolve the complete target ledger with "
            "resolve_merge_conflict(strategy='restore_base') or "
            "resolve_merge_conflict(strategy='adopt_current')."
        )

    @staticmethod
    @torch.no_grad()
    def _compose_merge_ledger(
        merge_base,
        entries,
        merge_order,
        precise,
        weight_param,
    ):
        compute_dtype = torch.float64 if precise else None
        compute_device = torch.device("cpu") if precise else weight_param.device
        merged_weight = merge_base.to(
            device=compute_device,
            dtype=compute_dtype or weight_param.dtype,
        )
        adapter_states = []
        try:
            for adapter in merge_order:
                if adapter not in entries:
                    continue
                adapter_states.append((adapter, adapter.training))
                adapter.eval()
                merged_weight = adapter._calculate_merged_weight(
                    merged_weight,
                    entries[adapter],
                    tuple(weight_param.shape),
                    compute_dtype=compute_dtype,
                )
        finally:
            for adapter, training in reversed(adapter_states):
                adapter.train(training)
        return merged_weight

    @torch.no_grad()
    def _validate_merge_ledger(self, module, weight_param):
        target_conflict = self._merge_target_conflict(module, weight_param)
        if target_conflict is not None:
            self._raise_merge_target_conflict(target_conflict)
        if self._changed_merge_adapters(module):
            self._raise_merge_factor_conflict()

        expected = self._compose_merge_ledger(
            module._lycoris_lokr_merge_base,
            module._lycoris_lokr_merge_entries,
            module._lycoris_lokr_merge_order,
            module._lycoris_lokr_merge_precise,
            weight_param,
        )
        if not self._tensor_values_equal(weight_param, expected):
            self._raise_merge_factor_conflict()
        module._lycoris_lokr_merge_weight_version = weight_param._version

    @torch.no_grad()
    def resolve_merge_conflict(self, *, strategy: str) -> bool:
        """Resolve an invalid reversible ledger without a partial rebase."""
        valid_strategies = ("restore_base", "adopt_current")
        if strategy not in valid_strategies:
            raise ValueError(
                "LoKr merge conflict strategy must be 'restore_base' or "
                f"'adopt_current', got {strategy!r}."
            )

        module = self.org_module[0]
        entries = getattr(module, "_lycoris_lokr_merge_entries", None)
        if not entries:
            return False
        if self not in entries:
            raise RuntimeError(
                "This LoKr adapter is not part of the target's active reversible "
                "merge ledger."
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
                "Cannot resolve overlapping LoKr and different-adapter precise "
                "merge states."
            )

        weight_param = module.weight
        target_conflict = self._merge_target_conflict(module, weight_param)
        factors_changed = bool(self._changed_merge_adapters(module))
        composition_changed = False
        if target_conflict is None and not factors_changed:
            expected = self._compose_merge_ledger(
                module._lycoris_lokr_merge_base,
                entries,
                module._lycoris_lokr_merge_order,
                module._lycoris_lokr_merge_precise,
                weight_param,
            )
            composition_changed = not self._tensor_values_equal(
                weight_param,
                expected,
            )

        if not target_conflict and not factors_changed and not composition_changed:
            module._lycoris_lokr_merge_weight_version = weight_param._version
            raise RuntimeError(
                "The reversible LoKr merge ledger is valid; use merge_to() for "
                "normal undo or finalize_merge() to commit it."
            )

        if strategy == "restore_base":
            if target_conflict is not None:
                raise RuntimeError(
                    "Cannot restore the LoKr merge base because the target weight "
                    "was changed or replaced outside the ledger. Use "
                    "strategy='adopt_current' to keep that exact current target."
                )
            weight_param.copy_(module._lycoris_lokr_merge_base)
        else:
            for adapter in entries:
                adapter._lokr_merge_committed = True

        self._clear_merge_ledger(module)
        return True

    @torch.no_grad()
    def finalize_merge(self):
        """Keep the current merged weight and release the reversible ledger."""
        module = self.org_module[0]
        if not hasattr(module, "_lycoris_lokr_merge_entries"):
            return False
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
                "Cannot finalize an overlapping LoKr and different-adapter "
                "precise merge; fully undo LoKr first."
            )
        self._validate_merge_ledger(module, module.weight)
        for adapter in module._lycoris_lokr_merge_entries:
            adapter._lokr_merge_committed = True
        self._clear_merge_ledger(module)
        return True

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
        if getattr(self, "_lokr_merge_committed", False):
            raise RuntimeError(
                "This LoKr adapter was already committed into the target weight."
            )

        multiplier = float(multiplier)
        if not math.isfinite(multiplier):
            raise ValueError(f"Merge multiplier must be finite, got {multiplier}.")
        if multiplier == 0:
            return
        if is_weight_only_fp8_linear(self.org_module[0]):
            raise RuntimeError(
                "Merging LyCORIS modules into weight-only FP8 Linear is not supported."
            )
        if self.is_quant:
            raise NotImplementedError(
                "Merging LoKr into a quantized base weight requires explicit "
                "requantization support."
            )

        module = self.org_module[0]
        _ensure_target_unwrapped_for_merge(module, self)
        if getattr(module, "_lycoris_onfly_stack", []):
            raise RuntimeError(
                "Cannot permanently merge LoKr while an on-the-fly merge is active."
            )
        weight_param = module.weight
        has_merge_state = hasattr(module, "_lycoris_lokr_merge_entries")
        has_precise_state = any(
            name in module.__dict__
            for name in (
                "_lycoris_precise_weight_base",
                "_lycoris_precise_weight_current",
                "_lycoris_precise_bias_base",
                "_lycoris_precise_bias_current",
            )
        )
        if has_precise_state and not has_merge_state:
            raise RuntimeError(
                "Cannot merge LoKr while a different adapter has an active "
                "precise-merge snapshot; finalize that merge first."
            )
        if has_merge_state:
            self._validate_merge_ledger(module, weight_param)

        if not reversible:
            if has_merge_state:
                raise RuntimeError(
                    "Finalize or fully undo the active reversible LoKr merge "
                    "before using a non-reversible merge."
                )
            was_training = self.training
            self.eval()
            try:
                merge_base = (
                    weight_param.detach().cpu() if precise else weight_param.detach()
                )
                merged_weight = self._calculate_merged_weight(
                    merge_base,
                    multiplier,
                    tuple(weight_param.shape),
                    compute_dtype=torch.float64 if precise else None,
                )
            finally:
                self.train(was_training)
            weight_param.copy_(merged_weight.to(weight_param))
            self._lokr_merge_committed = True
            return

        merge_base = (
            module._lycoris_lokr_merge_base
            if has_merge_state
            else weight_param.detach().cpu().clone()
        )
        entries = dict(module._lycoris_lokr_merge_entries) if has_merge_state else {}
        merge_order = list(module._lycoris_lokr_merge_order) if has_merge_state else []
        if self not in merge_order:
            merge_order.append(self)
            if all(
                hasattr(adapter, "_lokr_application_order") for adapter in merge_order
            ):
                merge_order.sort(key=lambda adapter: adapter._lokr_application_order)
        use_precise_merge = (
            module._lycoris_lokr_merge_precise if has_merge_state else False
        ) or precise
        merged_multiplier = entries.get(self, 0.0) + multiplier
        if math.isclose(merged_multiplier, 0.0, abs_tol=1e-12):
            entries.pop(self, None)
            merge_order = [adapter for adapter in merge_order if adapter is not self]
        else:
            entries[self] = merged_multiplier
        if all(hasattr(adapter, "_lokr_application_order") for adapter in merge_order):
            merge_order.sort(key=lambda adapter: adapter._lokr_application_order)

        if not entries:
            weight_param.copy_(merge_base)
            if has_merge_state:
                self._clear_merge_ledger(module)
            return

        adapter_fingerprints = {
            adapter: adapter._merge_state_fingerprint() for adapter in entries
        }
        if not has_merge_state:
            # Probe the target before mutating it so fingerprinting cannot leave
            # a newly merged weight without its recovery metadata.
            self._tensor_fingerprint(weight_param)
        merged_weight = self._compose_merge_ledger(
            merge_base,
            entries,
            merge_order,
            use_precise_merge,
            weight_param,
        )
        weight_param.copy_(
            merged_weight.to(device=weight_param.device, dtype=weight_param.dtype)
        )
        module._lycoris_lokr_merge_base = merge_base
        module._lycoris_lokr_merge_entries = entries
        module._lycoris_lokr_merge_order = merge_order
        module._lycoris_lokr_merge_precise = use_precise_merge
        module.__dict__["_lycoris_lokr_merge_weight_param_ref"] = weakref.ref(
            weight_param
        )
        module._lycoris_lokr_merge_weight_version = weight_param._version
        module._lycoris_lokr_merge_weight_fingerprint = self._tensor_fingerprint(
            weight_param
        )
        module._lycoris_lokr_merge_adapter_fingerprints = adapter_fingerprints

    @classmethod
    @torch.no_grad()
    def _validate_onfly_stack(cls, module, stack):
        weight_param = module.weight
        observed_weight = weight_param
        for frame in reversed(stack):
            if frame["multiplier"] == 0:
                continue
            if frame["weight_param"] is not weight_param:
                raise RuntimeError(
                    "The target weight Parameter was replaced during a LoKr "
                    "on-the-fly merge; refusing to overwrite it."
                )
            adapter = frame["adapter"]
            original = adapter.cached_org_weight.to(
                device=weight_param.device,
                dtype=weight_param.dtype,
            )
            was_training = adapter.training
            adapter.eval()
            try:
                expected = adapter._calculate_merged_weight(
                    original,
                    frame["multiplier"],
                    tuple(weight_param.shape),
                )
            finally:
                adapter.train(was_training)
            if not cls._tensor_values_equal(observed_weight, expected):
                raise RuntimeError(
                    "The target weight or active LoKr factors changed during an "
                    "on-the-fly merge; refusing to overwrite the update."
                )
            observed_weight = original

    @torch.no_grad()
    def onfly_merge(self, multiplier=1.0):
        if getattr(self, "_lokr_merge_committed", False):
            raise RuntimeError(
                "This LoKr adapter was committed into the target weight and "
                "cannot be merged again."
            )
        multiplier = float(multiplier)
        if not math.isfinite(multiplier):
            raise ValueError(
                f"On-the-fly merge multiplier must be finite, got {multiplier}."
            )
        if multiplier != 0 and is_weight_only_fp8_linear(self.org_module[0]):
            raise RuntimeError(
                "Merging LyCORIS modules into weight-only FP8 Linear is not supported."
            )
        if multiplier != 0 and self.is_quant:
            raise NotImplementedError(
                "On-the-fly LoKr merging into a quantized base weight requires "
                "requantization support."
            )
        if hasattr(self, "_lokr_onfly_multiplier"):
            raise RuntimeError("onfly_merge() called twice without onfly_restore().")

        module = self.org_module[0]
        _ensure_target_unwrapped_for_merge(module, self)
        if getattr(module, "_lycoris_lokr_merge_entries", {}):
            raise RuntimeError(
                "Cannot use on-the-fly LoKr merge while a permanent merge is active."
            )
        stack = list(getattr(module, "_lycoris_onfly_stack", []))
        if any(not isinstance(frame.get("adapter"), LokrModule) for frame in stack):
            raise RuntimeError(
                "Mixing LoKr and a different adapter in one target's "
                "on-the-fly merge stack is not supported."
            )
        self._validate_onfly_stack(module, stack)
        original_weight = (
            module.weight.detach().cpu().clone() if multiplier != 0 else None
        )
        was_training = self.training
        self.eval()
        try:
            if multiplier != 0:
                merged_weight = self._calculate_merged_weight(
                    module.weight.detach(),
                    multiplier,
                    module.weight.shape,
                )
                module.weight.copy_(merged_weight.to(module.weight))
            self.cached_org_weight = original_weight
            self._lokr_onfly_multiplier = multiplier
            stack.append(
                {
                    "adapter": self,
                    "kind": "lokr",
                    "weight_param": module.weight,
                    "multiplier": multiplier,
                }
            )
            module._lycoris_onfly_stack = stack
        except Exception:
            if original_weight is not None:
                module.weight.copy_(original_weight.to(module.weight))
            for name in ("cached_org_weight", "_lokr_onfly_multiplier"):
                self.__dict__.pop(name, None)
            raise
        finally:
            self.train(was_training)

    @torch.no_grad()
    def onfly_restore(self):
        if not hasattr(self, "_lokr_onfly_multiplier"):
            raise RuntimeError("onfly_restore() called without onfly_merge().")
        module = self.org_module[0]
        stack = list(getattr(module, "_lycoris_onfly_stack", []))
        if not stack or stack[-1]["adapter"] is not self:
            raise RuntimeError(
                "LoKr on-the-fly adapters must be restored in reverse merge order."
            )
        if self._lokr_onfly_multiplier != 0:
            self._validate_onfly_stack(module, stack)
            module.weight.copy_(self.cached_org_weight.to(module.weight))
        stack.pop()
        self.__dict__.pop("cached_org_weight", None)
        self.__dict__.pop("_lokr_onfly_multiplier", None)
        if stack:
            module._lycoris_onfly_stack = stack
        else:
            module.__dict__.pop("_lycoris_onfly_stack", None)

    def apply_weight_decompose(self, weight, multiplier=1, base_weight=None):
        if base_weight is None:
            base_weight = self._current_weight()
        compute_dtype = torch.promote_types(weight.dtype, self.dora_scale.dtype)
        compute_dtype = self._dora_accumulator_dtype(compute_dtype)
        direction = weight.to(dtype=compute_dtype)
        magnitude = self.dora_scale.to(
            device=direction.device,
            dtype=compute_dtype,
        )
        magnitude_shape = tuple(magnitude.shape)
        output_shape = (direction.shape[0], *[1] * (direction.dim() - 1))
        input_shape = (1, direction.shape[1], *[1] * (direction.dim() - 2))

        if magnitude_shape == output_shape:
            norm_dims = tuple(range(1, direction.dim()))
            direction_norm = torch.linalg.vector_norm(
                direction,
                dim=norm_dims,
                keepdim=True,
            )
            scaled_direction = direction
        elif magnitude_shape == input_shape:
            norm_dims = (0, *range(2, direction.dim()))
            direction_norm = torch.linalg.vector_norm(
                direction,
                dim=norm_dims,
                keepdim=True,
            )
            scaled_direction = direction
        elif len(magnitude_shape) == direction.dim() + 1:
            groups = magnitude_shape[0]
            if (
                magnitude_shape[1] != 1
                or magnitude_shape[2] != direction.shape[1]
                or direction.shape[0] % groups != 0
                or any(size != 1 for size in magnitude_shape[3:])
            ):
                raise ValueError(
                    "Invalid grouped-input DoRA magnitude shape: "
                    f"weight={tuple(direction.shape)}, dora_scale={magnitude_shape}."
                )
            grouped_shape = (
                groups,
                direction.shape[0] // groups,
                direction.shape[1],
                *direction.shape[2:],
            )
            scaled_direction = direction.reshape(grouped_shape)
            norm_dims = (1, *range(3, scaled_direction.dim()))
            direction_norm = torch.linalg.vector_norm(
                scaled_direction,
                dim=norm_dims,
                keepdim=True,
            )
        else:
            raise ValueError(
                "Cannot infer the DoRA norm axis from dora_scale: "
                f"weight={tuple(direction.shape)}, dora_scale={magnitude_shape}."
            )

        # DoRA treats the direction norm as a constant during backpropagation.
        direction_norm = direction_norm.clamp_min(
            torch.finfo(direction.dtype).tiny
        ).detach()
        dora_weight = (scaled_direction * (magnitude / direction_norm)).reshape_as(
            direction
        )

        base_weight = base_weight.to(dora_weight)
        # The runtime multiplier scales the complete DoRA adapter residual.
        return base_weight + (dora_weight - base_weight) * multiplier

    def custom_state_dict(self):
        destination = {}
        destination["alpha"] = self.alpha
        if self.wd:
            destination["dora_scale"] = self.dora_scale
        if self.use_w1:
            destination["lokr_w1"] = self.lokr_w1 * self.scalar
        else:
            destination["lokr_w1_a"] = self.lokr_w1_a * self.scalar
            destination["lokr_w1_b"] = self.lokr_w1_b

        # A portable checkpoint must keep the historical folded first factor,
        # but folding loses the scalar (and loses the complete factor when the
        # scalar is zero).  Standard PyTorch/Accelerate state saves therefore
        # carry a versioned, resume-only copy of the original parameterization.
        # ``strip_training_state_keys`` removes these entries for a minimal
        # portable inference export.
        if isinstance(self.scalar, nn.Parameter):
            destination[_TRAINING_STATE_VERSION_KEY] = self.scalar.new_tensor(
                _TRAINING_STATE_VERSION,
                dtype=torch.int64,
            )
            destination[_TRAINING_STATE_SCALAR_KEY] = self.scalar
            if self.use_w1:
                destination[_TRAINING_STATE_W1_KEY] = self.lokr_w1
            else:
                destination[_TRAINING_STATE_W1_A_KEY] = self.lokr_w1_a

        if self.use_w2:
            destination["lokr_w2"] = self.lokr_w2
        else:
            destination["lokr_w2_a"] = self.lokr_w2_a
            destination["lokr_w2_b"] = self.lokr_w2_b
            if self.tucker:
                destination["lokr_t2"] = self.lokr_t2
        return destination

    @torch.no_grad()
    def apply_max_norm(self, max_norm, device=None):
        max_norm = float(max_norm)
        if not math.isfinite(max_norm) or max_norm <= 0:
            raise ValueError(f"max_norm must be positive, got {max_norm}.")
        module = self.org_module[0]
        entries = getattr(module, "_lycoris_lokr_merge_entries", {})
        if entries or getattr(module, "_lycoris_onfly_stack", []):
            raise RuntimeError(
                "Cannot apply max norm while the target has merged LoKr adapters."
            )

        base_weight = self._current_weight()
        if device is not None:
            base_weight = base_weight.to(device)
        was_training = self.training
        original_scalar = self.scalar.detach().clone()

        def stored_scalar_candidate(target_cpu):
            target_cpu = target_cpu.detach().to(device="cpu", dtype=torch.float64)
            candidate_cpu = target_cpu.to(dtype=self.scalar.dtype)
            if torch.abs(candidate_cpu.to(torch.float64)) > torch.abs(target_cpu):
                candidate_cpu = torch.nextafter(
                    candidate_cpu,
                    torch.zeros_like(candidate_cpu),
                )
            return candidate_cpu.to(self.scalar.device)

        self.eval()
        try:
            update = self._get_effective_diff_weight(
                self.shape,
                base_weight,
            )
            orig_norm = update.norm()
            if not torch.isfinite(orig_norm):
                raise RuntimeError("Cannot normalize a non-finite LoKr update.")
            if orig_norm <= max_norm:
                return False, orig_norm

            ratio = max_norm / float(orig_norm)
            target_scalar = self.scalar.detach().cpu().to(torch.float64) * ratio
            candidate = stored_scalar_candidate(target_scalar)
            self.scalar.copy_(candidate)

            # Recompute the real stored-dtype result.  If norm rounding still
            # overshoots, each iteration makes strict progress toward zero.
            for _ in range(32):
                update = self._get_effective_diff_weight(
                    self.shape,
                    base_weight,
                )
                bounded_norm = update.norm()
                if bounded_norm <= max_norm:
                    return True, bounded_norm
                if not torch.isfinite(bounded_norm):
                    raise RuntimeError("Max-norm scaling produced a non-finite norm.")

                correction = max_norm / float(bounded_norm)
                current = self.scalar.detach().clone()
                target_scalar = current.cpu().to(torch.float64) * correction
                candidate = stored_scalar_candidate(target_scalar)
                if torch.abs(candidate) >= torch.abs(current):
                    candidate_cpu = candidate.detach().cpu()
                    candidate = torch.nextafter(
                        candidate_cpu,
                        torch.zeros_like(candidate_cpu),
                    ).to(self.scalar.device)
                self.scalar.copy_(candidate)

            # Pathological rounding must still satisfy the public postcondition.
            self.scalar.zero_()
            update = self._get_effective_diff_weight(
                self.shape,
                base_weight,
            )
            bounded_norm = update.norm()
            if not torch.isfinite(bounded_norm) or bounded_norm > max_norm:
                raise RuntimeError("Unable to enforce the requested LoKr max norm.")
            return True, bounded_norm
        except Exception:
            self.scalar.copy_(original_scalar)
            raise
        finally:
            self.train(was_training)

    def bypass_forward_diff(self, h, scale=1):
        is_conv = self.module_type.startswith("conv")
        compute_dtype = self._weight_compute_dtype(h.dtype)
        h = h.to(dtype=compute_dtype)
        rebuild_for_rank_dropout = self.training and self.rank_dropout
        if is_conv or rebuild_for_rank_dropout:
            module = self.org_module[0]
            rebuild_for_conv = is_conv and (
                module.groups != 1 or module.padding_mode != "zeros"
            )
            if rebuild_for_conv or rebuild_for_rank_dropout:
                diff_weight = (
                    self.get_weight(
                        self.shape,
                        device=h.device,
                        dtype=compute_dtype,
                    )
                    * self.scalar.to(h)
                    * scale
                )
                return self.drop(self._weight_forward(h, diff_weight, None))

        if self.use_w2:
            ba = self.lokr_w2.to(h)
        else:
            a = self.lokr_w2_b.to(h)
            w2_up = self.lokr_w2_a.to(h)

            if self.tucker:
                t = self.lokr_t2.to(h)
                a = a.view(*a.shape, *[1] * (len(t.shape) - 2))
                w2_up = w2_up.transpose(0, 1).contiguous()
                w2_up = w2_up.view(
                    *w2_up.shape,
                    *[1] * (len(t.shape) - 2),
                )
            elif is_conv:
                a = a.view(a.shape[0], -1, *self.shape[2:])
                w2_up = w2_up.view(
                    *w2_up.shape,
                    *[1] * (len(self.shape) - 2),
                )

        if self.use_w1:
            c = self.lokr_w1.to(h)
        else:
            c = self.lokr_w1_a.to(h) @ self.lokr_w1_b.to(h)
        uq = c.size(1)

        if is_conv:
            # (b, uq), vq, ...
            batch_size, _, *rest = h.shape
            h_in_group = h.reshape(batch_size * uq, -1, *rest)
        else:
            # b, ..., uq, vq
            h_in_group = h.reshape(*h.shape[:-1], uq, -1)

        if self.use_w2:
            hb = self.op(h_in_group, ba, **self.kw_dict)
        else:
            if is_conv:
                if self.tucker:
                    ha = self.op(h_in_group, a)
                    ht = self.op(ha, t, **self.kw_dict)
                    hb = self.op(ht, w2_up)
                else:
                    ha = self.op(h_in_group, a, **self.kw_dict)
                    hb = self.op(ha, w2_up)
            else:
                ha = self.op(h_in_group, a, **self.kw_dict)
                hb = self.op(ha, w2_up)

        if is_conv:
            # (b, uq), vp, ..., f
            # -> b, uq, vp, ..., f
            # -> b, f, vp, ..., uq
            hb = hb.view(batch_size, -1, *hb.shape[1:])
            h_cross_group = hb.transpose(1, -1)
        else:
            # b, ..., uq, vq
            # -> b, ..., vq, uq
            h_cross_group = hb.transpose(-1, -2)

        hc = F.linear(h_cross_group, c)
        if is_conv:
            # b, f, vp, ..., up
            # -> b, up, vp, ... ,f
            # -> b, c, ..., f
            hc = hc.transpose(1, -1)
            h = hc.reshape(batch_size, -1, *hc.shape[3:])
        else:
            # b, ..., vp, up
            # -> b, ..., up, vp
            # -> b, ..., c
            hc = hc.transpose(-1, -2)
            h = hc.reshape(*hc.shape[:-2], -1)

        return self.drop(h * scale * self.scale * self.scalar.to(h))

    def bypass_forward(self, x, scale=1, *args, **kwargs):
        base = self.org_forward(x, *args, **kwargs)
        delta = self.bypass_forward_diff(x, scale=scale)
        return base + delta.to(base)

    def forward(self, x: torch.Tensor, *args, **kwargs):
        forward_weights = _lokr_forward_weights.get()
        context_token = None
        if forward_weights is None:
            forward_weights = {}
            context_token = _lokr_forward_weights.set(forward_weights)

        module_key = id(self.org_module[0])
        if module_key not in forward_weights:
            forward_weights[module_key] = None

        def current_forward_weight():
            weight = forward_weights[module_key]
            if weight is None:
                weight = self._current_weight().to(x.device)
                forward_weights[module_key] = weight
            return weight

        try:
            if self.module_dropout and self.training:
                if torch.rand(1) < self.module_dropout:
                    return self.org_forward(x, *args, **kwargs)

            fp8_weight_decompose = self.wd and is_weight_only_fp8_linear(
                self.org_module[0]
            )
            if self.bypass_mode and not fp8_weight_decompose:
                if context_token is None:
                    base = self.org_forward(x, *args, **kwargs)
                    base_weight = current_forward_weight().to(x.device)
                    new_weight = self._calculate_merged_weight(
                        base_weight,
                        self.multiplier,
                        self.shape,
                    )
                    forward_weights[module_key] = new_weight
                    delta_weight = (new_weight - base_weight.to(new_weight)).to(
                        dtype=x.dtype
                    )
                    delta = self._weight_forward(x, delta_weight, None)
                    return base + self.drop(delta).to(dtype=base.dtype)
                return self.bypass_forward(x, self.multiplier, *args, **kwargs)

            base = self.org_forward(x, *args, **kwargs)
            base_weight = current_forward_weight().to(x.device)
            new_weight = self._calculate_merged_weight(
                base_weight,
                self.multiplier,
                self.shape,
            )
            forward_weights[module_key] = new_weight

            delta_weight = (new_weight - base_weight.to(new_weight)).to(dtype=x.dtype)
            delta = self._weight_forward(x, delta_weight, None)
            return base + self.drop(delta).to(dtype=base.dtype)
        finally:
            if context_token is not None:
                _lokr_forward_weights.reset(context_token)


if __name__ == "__main__":
    base = nn.Conv2d(128, 128, 3, 1, 1)
    net = LokrModule(
        "",
        base,
        multiplier=1,
        lora_dim=4,
        alpha=1,
        weight_decompose=False,
        use_tucker=False,
        use_scalar=False,
        decompose_both=True,
    )
    net.apply_to()
    sd = net.state_dict()
    for key in sd:
        if key != "alpha":
            sd[key] = torch.randn_like(sd[key])
    net.load_state_dict(sd)

    test_input = torch.randn(1, 128, 16, 16)
    test_output = net(test_input)
    print(test_output.shape)

    net2 = LokrModule(
        "",
        base,
        multiplier=1,
        lora_dim=4,
        alpha=1,
        weight_decompose=False,
        use_tucker=False,
        use_scalar=False,
        bypass_mode=True,
        decompose_both=True,
    )
    net2.apply_to()
    net2.load_state_dict(sd)
    print(net2)

    test_output2 = net(test_input)
    print(F.mse_loss(test_output, test_output2))
