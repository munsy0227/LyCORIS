from functools import cache

import torch

from ..logging import logger

SUPPORT_QUANT = False
try:
    from bitsandbytes.nn import LinearNF4, Linear8bitLt, LinearFP4

    SUPPORT_QUANT = True
except Exception:
    import torch.nn as nn

    class LinearNF4(nn.Linear):
        pass

    class Linear8bitLt(nn.Linear):
        pass

    class LinearFP4(nn.Linear):
        pass


try:
    from quanto.nn import QLinear, QConv2d, QLayerNorm

    SUPPORT_QUANT = True
except Exception:
    import torch.nn as nn

    class QLinear(nn.Linear):
        pass

    class QConv2d(nn.Conv2d):
        pass

    class QLayerNorm(nn.LayerNorm):
        pass


try:
    from optimum.quanto.nn import (
        QLinear as QLinearOpt,
        QConv2d as QConv2dOpt,
        QLayerNorm as QLayerNormOpt,
    )

    SUPPORT_QUANT = True
except Exception:
    import torch.nn as nn

    class QLinearOpt(nn.Linear):
        pass

    class QConv2dOpt(nn.Conv2d):
        pass

    class QLayerNormOpt(nn.LayerNorm):
        pass


QuantLinears = (
    Linear8bitLt,
    LinearFP4,
    LinearNF4,
    QLinear,
    QConv2d,
    QLayerNorm,
    QLinearOpt,
    QConv2dOpt,
    QLayerNormOpt,
)


def dequantize_module_weight(module):
    """Return a floating-point view of a supported quantized module weight."""
    if hasattr(module, "W_q") and callable(getattr(module, "dequantize", None)):
        return module.dequantize()

    weight = module.weight
    qweight = getattr(module, "qweight", None)
    qweight_dequantize = getattr(qweight, "dequantize", None)
    if qweight is not None and callable(qweight_dequantize):
        return qweight_dequantize()

    class_name = weight.__class__.__name__

    if class_name == "Params4bit":
        if weight.quant_state is None:
            if weight.data.is_floating_point():
                return weight.data
            raise RuntimeError(
                "Cannot dequantize an initialized Params4bit weight without "
                "quantization state."
            )
        import bitsandbytes as bnb

        return bnb.functional.dequantize_4bit(
            weight.data,
            weight.quant_state,
        )

    if class_name == "Int8Params":
        import bitsandbytes as bnb

        state = getattr(module, "state", None)
        if state is None:
            raise ValueError(
                "Cannot dequantize a bitsandbytes Int8Params weight without "
                "the owning module state."
            )
        if state.SCB is None:
            state.SCB = weight.SCB
        if state.SCB is None:
            if weight.data.is_floating_point():
                return weight.data
            raise RuntimeError(
                "Cannot dequantize an initialized Int8Params weight without row scales."
            )
        scale = state.SCB.to(weight.data.device)
        if hasattr(bnb.functional, "int8_vectorwise_dequant"):
            return bnb.functional.int8_vectorwise_dequant(
                weight.data,
                scale,
            )
        return weight.data * scale.view(-1, 1) / 127

    dequantize = getattr(weight, "dequantize", None)
    if callable(dequantize) and weight.__class__ not in {
        torch.Tensor,
        torch.nn.Parameter,
    }:
        return dequantize()

    return weight


@cache
def log_bypass():
    return logger.warning(
        "Using bnb/quanto/optimum-quanto with LyCORIS will enable force-bypass mode."
    )


@cache
def log_fp8_bypass():
    return logger.warning(
        "Using weight-only FP8 Linear with LyCORIS will enable force-bypass mode."
    )


@cache
def log_suspect():
    return logger.warning(
        "Non-native Linear detected but bypass_mode is not set. "
        "Automatically using force-bypass mode to avoid possible issues. "
        "Please set bypass_mode=False explicitly if there are no quantized layers."
    )
