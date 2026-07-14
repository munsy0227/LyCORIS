import copy
import unittest
from unittest import mock

import torch
import torch.nn as nn
import torch.nn.functional as F

from lycoris import create_lycoris, create_lycoris_from_weights
from lycoris.config import PRESET
from lycoris.config_sdk import PresetConfig
from lycoris.functional import apply_dora_scale
from lycoris.functional import lokr as functional_lokr
from lycoris.wrapper import LycorisNetwork
from lycoris.kohya import LycorisNetworkKohya
from lycoris.modules import (
    DiagOFTModule,
    FullModule,
    GLoRAModule,
    IA3Module,
    LoConModule,
    LokrModule,
)

try:
    from optimum.quanto import freeze, qint4, qint8, quantize

    QUANTO_AVAILABLE = True
except ImportError:
    QUANTO_AVAILABLE = False

try:
    import bitsandbytes as bnb

    BITSANDBYTES_AVAILABLE = True
except Exception:
    BITSANDBYTES_AVAILABLE = False


class LokrConsistencyTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        generic_names = (
            "ENABLE_CONV",
            "TARGET_REPLACE_MODULE",
            "TARGET_REPLACE_NAME",
            "LORA_PREFIX",
            "MODULE_ALGO_MAP",
            "NAME_ALGO_MAP",
            "USE_FNMATCH",
            "TARGET_EXCLUDE_NAME",
        )
        kohya_names = (
            "ENABLE_CONV",
            "UNET_TARGET_REPLACE_MODULE",
            "UNET_TARGET_REPLACE_NAME",
            "TEXT_ENCODER_TARGET_REPLACE_MODULE",
            "TEXT_ENCODER_TARGET_REPLACE_NAME",
            "MODULE_ALGO_MAP",
            "NAME_ALGO_MAP",
            "USE_FNMATCH",
        )
        self._generic_preset_state = {
            name: copy.deepcopy(getattr(LycorisNetwork, name)) for name in generic_names
        }
        self._kohya_preset_state = {
            name: copy.deepcopy(getattr(LycorisNetworkKohya, name))
            for name in kohya_names
        }

    def tearDown(self):
        for name, value in self._generic_preset_state.items():
            setattr(LycorisNetwork, name, value)
        for name, value in self._kohya_preset_state.items():
            setattr(LycorisNetworkKohya, name, value)

    @staticmethod
    def _make_full_matrix_dora(base, multiplier=0.35, **kwargs):
        module = LokrModule(
            "test",
            base,
            multiplier=multiplier,
            lora_dim=4,
            alpha=1,
            full_matrix=True,
            weight_decompose=True,
            **kwargs,
        )
        with torch.no_grad():
            module.lokr_w1.normal_(mean=0.0, std=0.2)
            module.lokr_w2.normal_(mean=0.0, std=0.2)
            magnitude_scale = torch.linspace(
                0.7,
                1.3,
                module.dora_scale.numel(),
                dtype=module.dora_scale.dtype,
            ).reshape_as(module.dora_scale)
            module.dora_scale.mul_(magnitude_scale)
            if isinstance(module.scalar, nn.Parameter):
                module.scalar.fill_(0.6)
        module.eval()
        return module

    @staticmethod
    def _randomize_module(module):
        with torch.no_grad():
            for name, parameter in module.named_parameters():
                if name == "dora_scale":
                    scale = torch.linspace(
                        0.8,
                        1.2,
                        parameter.numel(),
                        dtype=parameter.dtype,
                        device=parameter.device,
                    ).reshape_as(parameter)
                    parameter.mul_(scale)
                elif name == "scalar":
                    parameter.fill_(0.6)
                else:
                    parameter.normal_(mean=0.0, std=0.2)

    def test_full_matrix_dora_multiplier_endpoints_and_formula(self):
        base = nn.Linear(8, 8)
        module = self._make_full_matrix_dora(base)
        base_weight = base.weight.detach()

        direction = base_weight + module.get_weight(base_weight.shape) * module.scalar
        direction_norm = (
            direction.reshape(direction.shape[0], -1)
            .norm(dim=1)
            .reshape_as(module.dora_scale)
            .clamp_min(torch.finfo(direction.dtype).eps)
        )
        expected_dora = direction * (module.dora_scale / direction_norm)

        merged_zero, _ = module.get_merged_weight(0.0, base_weight.shape)
        merged_full, _ = module.get_merged_weight(1.0, base_weight.shape)
        merged_partial, _ = module.get_merged_weight(0.35, base_weight.shape)
        diff_partial, _ = module.get_diff_weight(0.35, base_weight.shape)

        torch.testing.assert_close(merged_zero, base_weight, rtol=0.0, atol=0.0)
        torch.testing.assert_close(merged_full, expected_dora)
        torch.testing.assert_close(
            merged_partial,
            base_weight + 0.35 * (expected_dora - base_weight),
        )
        torch.testing.assert_close(diff_partial, merged_partial - base_weight)

    def test_functional_dora_matches_lokr_for_both_norm_axes(self):
        for wd_on_out in (True, False):
            with self.subTest(wd_on_out=wd_on_out):
                base = nn.Linear(12, 8)
                module = self._make_full_matrix_dora(
                    base,
                    wd_on_out=wd_on_out,
                )
                base_weight = base.weight.detach()
                diff_weight = module.get_weight(base_weight.shape) * module.scalar
                expected = module.apply_weight_decompose(
                    base_weight + diff_weight,
                    multiplier=0.4,
                    base_weight=base_weight,
                )

                actual = apply_dora_scale(
                    base_weight,
                    diff_weight,
                    module.dora_scale,
                    0.4,
                )

                torch.testing.assert_close(actual, expected)

    def test_full_matrix_dora_forward_matches_regular_and_precise_merge(self):
        template_base = nn.Linear(8, 8)
        template_module = self._make_full_matrix_dora(
            template_base,
            use_scalar=True,
        )
        base_state = {
            key: value.detach().clone()
            for key, value in template_base.state_dict().items()
        }
        adapter_state = {
            key: value.detach().clone()
            for key, value in template_module.state_dict().items()
        }
        test_input = torch.randn(3, 8)

        for precise in (False, True):
            with self.subTest(precise=precise):
                base = nn.Linear(8, 8)
                base.load_state_dict(base_state)
                module = self._make_full_matrix_dora(base, use_scalar=True)
                module.load_state_dict(adapter_state)
                module.apply_to()

                forward_output = base(test_input)
                module.restore()
                module.merge_to(module.multiplier, precise=precise)
                merged_output = base(test_input)

                torch.testing.assert_close(
                    merged_output,
                    forward_output,
                    rtol=1e-5,
                    atol=1e-6,
                )

    def test_full_matrix_dora_training_gradients_are_finite(self):
        base = nn.Linear(8, 8)
        module = self._make_full_matrix_dora(base, use_scalar=True)
        module.apply_to()

        output = base(torch.randn(3, 8))
        output.square().mean().backward()

        for name, parameter in module.named_parameters():
            with self.subTest(parameter=name):
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())
                self.assertGreater(parameter.grad.abs().sum().item(), 0.0)

    def test_full_matrix_dora_bfloat16_forward_matches_merge(self):
        base = nn.Linear(8, 8).to(dtype=torch.bfloat16)
        module = self._make_full_matrix_dora(base).to(dtype=torch.bfloat16)
        test_input = torch.randn(3, 8, dtype=torch.bfloat16)
        module.apply_to()

        forward_output = base(test_input)
        module.restore()
        module.merge_to(module.multiplier)
        merged_output = base(test_input)

        torch.testing.assert_close(
            merged_output,
            forward_output,
            rtol=5e-3,
            atol=5e-3,
        )

    def test_full_matrix_dora_state_dict_reconstruction_preserves_weight(self):
        base = nn.Linear(8, 8)
        module = self._make_full_matrix_dora(base, use_scalar=True)
        state_dict = module.state_dict()
        weights = tuple(state_dict.get(name) for name in module.weight_list)
        expected_weight, _ = module.get_merged_weight(1.0, base.weight.shape)

        rebuilt_base = nn.Linear(8, 8)
        rebuilt_base.load_state_dict(base.state_dict())
        with torch.no_grad():
            rebuilt = LokrModule.make_module_from_state_dict(
                "rebuilt",
                rebuilt_base,
                *weights,
            )
        rebuilt_weight, _ = rebuilt.get_merged_weight(
            1.0,
            rebuilt_base.weight.shape,
        )

        self.assertTrue(rebuilt.full_matrix)
        self.assertTrue(rebuilt.wd)
        self.assertEqual(rebuilt.lora_dim, module.lora_dim)
        torch.testing.assert_close(rebuilt.alpha, module.alpha)
        torch.testing.assert_close(rebuilt_weight, expected_weight)

    def test_full_matrix_checkpoint_ignores_noncanonical_alpha_scale(self):
        base = nn.Linear(8, 8)
        source = self._make_full_matrix_dora(base, use_scalar=True)
        state_dict = source.state_dict()
        weights = [state_dict.get(name) for name in source.weight_list]
        canonical = LokrModule.make_module_from_state_dict(
            "canonical",
            base,
            *weights,
        )
        weights[8] = torch.tensor(0.5)

        rebuilt = LokrModule.make_module_from_state_dict(
            "rebuilt",
            base,
            *weights,
        )

        self.assertTrue(rebuilt.full_matrix)
        self.assertEqual(rebuilt.scale, 1.0)
        self.assertEqual(rebuilt.alpha.item(), 0.5)
        torch.testing.assert_close(
            rebuilt.get_weight(base.weight.shape),
            canonical.get_weight(base.weight.shape),
        )

    def test_full_matrix_dora_conv_forward_matches_merge(self):
        configs = (
            {"groups": 1, "padding_mode": "zeros"},
            {"groups": 2, "padding_mode": "zeros"},
            {"groups": 4, "padding_mode": "zeros"},
            {"groups": 1, "padding_mode": "reflect"},
            {"groups": 2, "padding_mode": "circular"},
        )

        for config in configs:
            with self.subTest(**config):
                base = nn.Conv2d(
                    4,
                    4,
                    kernel_size=3,
                    padding=1,
                    **config,
                )
                module = self._make_full_matrix_dora(base, multiplier=0.65)
                test_input = torch.randn(2, 4, 5, 5)
                module.apply_to()

                forward_output = base(test_input)
                module.restore()
                module.merge_to(module.multiplier)
                merged_output = base(test_input)

                torch.testing.assert_close(merged_output, forward_output)

    def test_full_matrix_dora_state_restores_unbalanced_and_input_norm(self):
        base = nn.Linear(12, 8)
        module = self._make_full_matrix_dora(
            base,
            factor=2,
            unbalanced_factorization=True,
            wd_on_out=False,
            use_scalar=True,
        )
        state_dict = module.state_dict()
        weights = tuple(state_dict.get(name) for name in module.weight_list)
        expected_weight, _ = module.get_merged_weight(1.0, base.weight.shape)

        rebuilt_base = nn.Linear(12, 8)
        rebuilt_base.load_state_dict(base.state_dict())
        with torch.no_grad():
            rebuilt = LokrModule.make_module_from_state_dict(
                "rebuilt",
                rebuilt_base,
                *weights,
            )
        rebuilt_weight, _ = rebuilt.get_merged_weight(
            1.0,
            rebuilt_base.weight.shape,
        )

        self.assertFalse(rebuilt.wd_on_out)
        self.assertEqual(rebuilt.lokr_w1.shape, module.lokr_w1.shape)
        self.assertEqual(rebuilt.lokr_w2.shape, module.lokr_w2.shape)
        torch.testing.assert_close(rebuilt_weight, expected_weight)

    def test_rank_dropout_scale_all_dropped_is_finite(self):
        base = nn.Linear(8, 8)
        module = LokrModule(
            "test",
            base,
            lora_dim=4,
            full_matrix=True,
            rank_dropout=1.0,
            rank_dropout_scale=True,
        )
        with torch.no_grad():
            module.lokr_w1.normal_()
            module.lokr_w2.normal_()
        module.train()

        weight = module.get_weight(base.weight.shape)

        self.assertTrue(torch.isfinite(weight).all())
        self.assertEqual(torch.count_nonzero(weight).item(), 0)

    def test_bypass_rank_dropout_all_dropped_returns_base(self):
        base = nn.Linear(8, 8)
        module = LokrModule(
            "test",
            base,
            lora_dim=1,
            rank_dropout=1.0,
            rank_dropout_scale=True,
            bypass_mode=True,
        )
        self._randomize_module(module)
        test_input = torch.randn(2, 8)
        expected = F.linear(test_input, base.weight, base.bias)
        module.train()
        module.apply_to()

        actual = base(test_input)

        torch.testing.assert_close(actual, expected)
        self.assertTrue(torch.isfinite(actual).all())

    def test_invalid_rank_dropout_is_rejected(self):
        base = nn.Linear(8, 8)

        with self.assertRaisesRegex(ValueError, "rank_dropout"):
            LokrModule("test", base, rank_dropout=1.1)

    def test_invalid_rank_and_factor_are_rejected(self):
        base = nn.Linear(8, 8)

        with self.assertRaisesRegex(TypeError, "lora_dim"):
            LokrModule("test", base, lora_dim=1.5)
        with self.assertRaisesRegex(ValueError, "factor"):
            LokrModule("test", base, factor=0)
        with self.assertRaisesRegex(TypeError, "rank"):
            functional_lokr.weight_gen(base.weight, rank=1.5)

        module = LokrModule("test", base, factor="4")
        self.assertEqual(module.shape, tuple(base.weight.shape))

    def test_training_dropout_drops_rebuilt_dora_residual(self):
        base = nn.Linear(8, 8)
        module = self._make_full_matrix_dora(base, dropout=1.0)
        test_input = torch.randn(2, 8)
        expected = F.linear(test_input, base.weight, base.bias)
        module.train()
        module.apply_to()

        actual = base(test_input)

        torch.testing.assert_close(actual, expected)

    def test_quantized_dora_uses_runtime_dequantized_weight(self):
        class FakeQuantParameter(nn.Parameter):
            def dequantize(self):
                return self.as_subclass(torch.Tensor) * 2

        class QuantLikeLinear(nn.Linear):
            def __init__(self, in_features, out_features):
                super().__init__(in_features, out_features)
                packed = self.weight.detach() / 2
                self.weight = FakeQuantParameter(
                    packed,
                    requires_grad=False,
                )

            def forward(self, input_tensor):
                return F.linear(
                    input_tensor,
                    self.weight.dequantize(),
                    self.bias,
                )

        base = QuantLikeLinear(8, 8)
        module = LokrModule(
            "test",
            base,
            multiplier=0.4,
            full_matrix=True,
            weight_decompose=True,
            bypass_mode=True,
        )
        self._randomize_module(module)
        module.eval()
        test_input = torch.randn(2, 8)
        expected_weight, _ = module.get_merged_weight(
            module.multiplier,
            (base.out_features, base.in_features),
        )
        expected = F.linear(test_input, expected_weight, base.bias)

        module.apply_to()
        actual = base(test_input)

        self.assertTrue(module.is_quant)
        self.assertFalse(module.bypass_mode)
        self.assertFalse(torch.equal(module._current_weight(), base.weight.detach()))
        torch.testing.assert_close(actual, expected)
        actual.square().mean().backward()
        for parameter in module.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
        with self.assertRaisesRegex(NotImplementedError, "requantization"):
            module.merge_to(module.multiplier)
        with self.assertRaisesRegex(NotImplementedError, "requantization"):
            module.onfly_merge(module.multiplier)

    def test_quantized_additive_bypass_does_not_dequantize_base(self):
        class QuantLikeLinear(nn.Linear):
            pass

        base = QuantLikeLinear(8, 8)
        module = LokrModule(
            "test",
            base,
            multiplier=0.4,
            lora_dim=1,
            bypass_mode=True,
        )
        self._randomize_module(module)
        module.eval()
        test_input = torch.randn(2, 8)
        expected = module.bypass_forward(test_input, module.multiplier)

        def fail_if_dequantized():
            raise AssertionError("single additive bypass must remain lazy")

        module._current_weight = fail_if_dequantized
        module.apply_to()
        actual = base(test_input)

        torch.testing.assert_close(actual, expected)

    @unittest.skipUnless(QUANTO_AVAILABLE, "optimum-quanto is not installed")
    def test_quanto_qdora_forward_and_backward(self):
        for quant_type, dimension in ((qint8, 8), (qint4, 16)):
            for frozen in (False, True):
                with self.subTest(quant_type=quant_type, frozen=frozen):
                    self._check_quanto_qdora(quant_type, dimension, frozen)

    def _check_quanto_qdora(self, quant_type, dimension, frozen):
        model = nn.Sequential(nn.Linear(dimension, dimension))
        quantize(model, weights=quant_type)
        if frozen:
            freeze(model)
        base = model[0]
        module = LokrModule(
            "test",
            base,
            multiplier=0.4,
            full_matrix=True,
            weight_decompose=True,
        )
        self._randomize_module(module)
        test_input = torch.randn(2, dimension)
        expected_weight, _ = module.get_merged_weight(
            module.multiplier,
            (base.out_features, base.in_features),
        )
        expected = F.linear(test_input, expected_weight, base.bias)

        module.apply_to()
        actual = base(test_input)

        self.assertTrue(module.is_quant)
        torch.testing.assert_close(
            actual,
            expected,
            rtol=1e-5,
            atol=1e-6,
        )
        actual.square().mean().backward()
        for parameter in module.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
        module.restore()
        state_dict = module.state_dict()
        weights = tuple(
            state_dict.get(weight_name) for weight_name in module.weight_list
        )
        rebuilt = LokrModule.make_module_from_state_dict(
            "rebuilt",
            base,
            *weights,
        )
        rebuilt_weight, _ = rebuilt.get_merged_weight(
            module.multiplier,
            (base.out_features, base.in_features),
        )
        torch.testing.assert_close(rebuilt_weight, expected_weight)

    @unittest.skipUnless(QUANTO_AVAILABLE, "optimum-quanto is not installed")
    def test_quanto_conv2d_qdora_forward_and_backward(self):
        model = nn.Sequential(
            nn.Conv2d(
                4,
                4,
                kernel_size=3,
                padding=1,
                groups=2,
                padding_mode="reflect",
            )
        )
        quantize(model, weights=qint8)
        freeze(model)
        base = model[0]
        module = LokrModule(
            "test",
            base,
            multiplier=0.4,
            full_matrix=True,
            weight_decompose=True,
        )
        self._randomize_module(module)
        test_input = torch.randn(2, 4, 5, 5)
        expected_weight, _ = module.get_merged_weight(
            module.multiplier,
            module.shape,
        )
        padded_input = F.pad(
            test_input,
            base._reversed_padding_repeated_twice,
            mode=base.padding_mode,
        )
        expected = F.conv2d(
            padded_input,
            expected_weight,
            base.bias,
            stride=base.stride,
            padding=0,
            dilation=base.dilation,
            groups=base.groups,
        )

        module.apply_to()
        actual = base(test_input)

        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
        actual.square().mean().backward()
        for parameter in module.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())

    @unittest.skipUnless(
        BITSANDBYTES_AVAILABLE,
        "bitsandbytes is not installed",
    )
    def test_bitsandbytes_qdora_forward_and_backward(self):
        constructors = (
            ("LinearNF4", lambda: bnb.nn.LinearNF4(16, 16)),
            (
                "Linear8bitLt",
                lambda: bnb.nn.Linear8bitLt(
                    16,
                    16,
                    has_fp16_weights=False,
                ),
            ),
        )
        for layer_name, constructor in constructors:
            with self.subTest(layer=layer_name):
                reference = nn.Linear(16, 16)
                base = constructor()
                base.load_state_dict(reference.state_dict())
                base = base.to("cpu")
                module = LokrModule(
                    "test",
                    base,
                    multiplier=0.4,
                    full_matrix=True,
                    weight_decompose=True,
                )
                self._randomize_module(module)
                test_input = torch.randn(2, 16)
                expected_weight, _ = module.get_merged_weight(
                    module.multiplier,
                    (base.out_features, base.in_features),
                )
                expected = F.linear(test_input, expected_weight, base.bias)

                module.apply_to()
                actual = base(test_input)

                self.assertTrue(module.is_quant)
                torch.testing.assert_close(
                    actual,
                    expected,
                    rtol=1e-5,
                    atol=1e-5,
                )
                actual.square().mean().backward()
                for parameter in module.parameters():
                    self.assertIsNotNone(parameter.grad)
                    self.assertTrue(torch.isfinite(parameter.grad).all())
                module.restore()
                state_dict = module.state_dict()
                weights = tuple(
                    state_dict.get(weight_name) for weight_name in module.weight_list
                )
                with torch.no_grad():
                    rebuilt = LokrModule.make_module_from_state_dict(
                        "rebuilt",
                        base,
                        *weights,
                    )
                rebuilt_weight, _ = rebuilt.get_merged_weight(
                    module.multiplier,
                    (base.out_features, base.in_features),
                )
                torch.testing.assert_close(rebuilt_weight, expected_weight)

    def test_cpu_onfly_restore_is_exact(self):
        base = nn.Linear(8, 8)
        module = self._make_full_matrix_dora(base)
        original_weight = base.weight.detach().clone()

        module.onfly_merge(1.0)
        self.assertFalse(torch.equal(base.weight, original_weight))
        module.onfly_restore()

        self.assertTrue(torch.equal(base.weight, original_weight))
        with self.assertRaisesRegex(RuntimeError, "without onfly_merge"):
            module.onfly_restore()

    def test_stacked_onfly_merge_restores_in_reverse_order(self):
        base = nn.Linear(8, 8)
        additive = LokrModule(
            "additive",
            base,
            multiplier=0.4,
            full_matrix=True,
        )
        self._randomize_module(additive)
        dora = self._make_full_matrix_dora(base, multiplier=0.7)
        original_weight = base.weight.detach().clone()
        expected_additive = additive._calculate_merged_weight(
            original_weight,
            additive.multiplier,
            original_weight.shape,
        )
        expected_both = dora._calculate_merged_weight(
            expected_additive,
            dora.multiplier,
            original_weight.shape,
        )

        additive.onfly_merge(additive.multiplier)
        dora.onfly_merge(dora.multiplier)

        torch.testing.assert_close(base.weight, expected_both)
        dora.onfly_restore()
        torch.testing.assert_close(base.weight, expected_additive)
        additive.onfly_restore()
        self.assertTrue(torch.equal(base.weight, original_weight))

    def test_onfly_merge_ignores_training_rank_dropout(self):
        base = nn.Linear(8, 8)
        module = LokrModule(
            "test",
            base,
            full_matrix=True,
            rank_dropout=1.0,
        )
        self._randomize_module(module)
        original_weight = base.weight.detach().clone()
        module.eval()
        expected, _ = module.get_merged_weight(1.0, original_weight.shape)
        module.train()

        module.onfly_merge(1.0)

        self.assertTrue(module.training)
        torch.testing.assert_close(base.weight, expected)
        module.onfly_restore()

    def test_max_norm_bounds_actual_dora_residual_and_persists(self):
        base = nn.Linear(8, 8)
        module = self._make_full_matrix_dora(base, use_scalar=True)
        base_weight = base.weight.detach()
        before, _ = module.get_merged_weight(1.0, base_weight.shape)
        before_norm = (before - base_weight).norm()
        max_norm = before_norm.item() * 0.4

        scaled, reported_norm = module.apply_max_norm(max_norm)
        after, _ = module.get_merged_weight(1.0, base_weight.shape)
        after_norm = (after - base_weight).norm()

        self.assertTrue(scaled)
        torch.testing.assert_close(
            reported_norm,
            torch.tensor(max_norm, dtype=reported_norm.dtype),
        )
        torch.testing.assert_close(
            after_norm,
            torch.tensor(max_norm, dtype=after_norm.dtype),
        )

        state_dict = module.state_dict()
        weights = tuple(state_dict.get(name) for name in module.weight_list)
        rebuilt_base = nn.Linear(8, 8)
        rebuilt_base.load_state_dict(base.state_dict())
        with torch.no_grad():
            rebuilt = LokrModule.make_module_from_state_dict(
                "rebuilt",
                rebuilt_base,
                *weights,
            )
        rebuilt_weight, _ = rebuilt.get_merged_weight(1.0, base_weight.shape)
        torch.testing.assert_close(rebuilt_weight, after)

        with self.assertRaisesRegex(ValueError, "max_norm"):
            module.apply_max_norm(float("nan"))

        module.merge_to(0.1)
        with self.assertRaisesRegex(RuntimeError, "merged LoKr"):
            module.apply_max_norm(max_norm)
        module.merge_to(-0.1)

    def test_legacy_state_without_residual_scale_still_loads(self):
        base = nn.Linear(8, 8)
        module = self._make_full_matrix_dora(base)
        legacy_state = module.state_dict()
        legacy_state.pop("lokr_residual_scale")

        rebuilt = self._make_full_matrix_dora(base)
        rebuilt.lokr_residual_scale.fill_(0.25)
        result = rebuilt.load_state_dict(legacy_state)

        self.assertEqual(result.missing_keys, [])
        self.assertEqual(result.unexpected_keys, [])
        self.assertEqual(rebuilt.lokr_residual_scale.item(), 1.0)

    def test_corrupt_checkpoint_factor_set_is_rejected(self):
        base = nn.Linear(8, 8)
        module = LokrModule("test", base, lora_dim=1)
        state_dict = module.state_dict()
        weights = [state_dict.get(name) for name in module.weight_list]
        weights[module.weight_list.index("lokr_w2_b")] = None

        with self.assertRaisesRegex(ValueError, "lokr_w2"):
            LokrModule.make_module_from_state_dict(
                "rebuilt",
                base,
                *weights,
            )

    def test_state_dict_detaches_folded_weights_by_default(self):
        base = nn.Linear(8, 8)
        module = self._make_full_matrix_dora(base, use_scalar=True)

        state_dict = module.state_dict()
        state_dict_with_vars = module.state_dict(keep_vars=True)

        self.assertFalse(state_dict["lokr_w1"].requires_grad)
        self.assertFalse(state_dict["lokr_w2"].requires_grad)
        self.assertTrue(state_dict_with_vars["lokr_w1"].requires_grad)
        self.assertTrue(state_dict_with_vars["lokr_w2"].requires_grad)

    def test_wrapper_full_matrix_dora_checkpoint_round_trip(self):
        base = nn.Sequential(nn.Linear(8, 8))
        rebuilt_base = nn.Sequential(nn.Linear(8, 8))
        rebuilt_base.load_state_dict(base.state_dict())
        test_input = torch.randn(3, 8)
        try:
            network = create_lycoris(
                base,
                multiplier=0.4,
                linear_dim=4,
                linear_alpha=1,
                algo="lokr",
                dora_wd=True,
                full_matrix=True,
            )
            self.assertEqual(len(network.loras), 1)
            module = network.loras[0]
            self._randomize_module(module)
            module.eval()
            network.apply_to()
            expected = base(test_input)
            state_dict = network.state_dict()
            network.restore()

            rebuilt_network, _ = create_lycoris_from_weights(
                0.4,
                None,
                rebuilt_base,
                state_dict,
            )
            rebuilt_network.apply_to()
            actual = rebuilt_base(test_input)

            self.assertTrue(rebuilt_network.loras[0].full_matrix)
            self.assertTrue(rebuilt_network.loras[0].wd)
            self.assertIn(
                "lycoris_0.lokr_residual_scale",
                state_dict,
            )
            torch.testing.assert_close(actual, expected)
        finally:
            LycorisNetwork.apply_preset(PRESET["full"])

    def test_dora_merge_round_trip_is_exact_and_releases_state(self):
        for precise in (False, True):
            with self.subTest(precise=precise):
                base = nn.Linear(8, 8)
                module = self._make_full_matrix_dora(base)
                original_weight = base.weight.detach().clone()

                module.merge_to(0.7, precise=precise)
                module.merge_to(-0.7, precise=precise)

                self.assertTrue(torch.equal(base.weight, original_weight))
                self.assertFalse(hasattr(base, "_lycoris_lokr_merge_base"))
                self.assertFalse(hasattr(base, "_lycoris_lokr_merge_entries"))
                self.assertFalse(hasattr(base, "_lycoris_precise_weight_base"))
                self.assertFalse(hasattr(base, "_lycoris_precise_weight_current"))

    def test_stacked_dora_forward_matches_merge_and_unmerges_exactly(self):
        for precise in (False, True):
            with self.subTest(precise=precise):
                base = nn.Linear(8, 8)
                first = self._make_full_matrix_dora(base, multiplier=0.4)
                second = self._make_full_matrix_dora(base, multiplier=0.7)
                test_input = torch.randn(3, 8)
                original_weight = base.weight.detach().clone()

                first.apply_to()
                second.apply_to()
                forward_output = base(test_input)
                second.restore()
                first.restore()

                first.merge_to(first.multiplier, precise=precise)
                second.merge_to(second.multiplier, precise=precise)
                merged_output = base(test_input)
                torch.testing.assert_close(merged_output, forward_output)

                first.merge_to(-first.multiplier, precise=precise)
                second.merge_to(-second.multiplier, precise=precise)
                self.assertTrue(torch.equal(base.weight, original_weight))

    def test_stacked_grouped_conv_dora_forward_matches_merge(self):
        base = nn.Conv2d(
            4,
            4,
            kernel_size=3,
            padding=1,
            groups=2,
            padding_mode="reflect",
        )
        first = self._make_full_matrix_dora(base, multiplier=0.4)
        second = self._make_full_matrix_dora(base, multiplier=0.7)
        test_input = torch.randn(2, 4, 5, 5)

        first.apply_to()
        second.apply_to()
        forward_output = base(test_input)
        second.restore()
        first.restore()
        first.merge_to(first.multiplier)
        second.merge_to(second.multiplier)
        merged_output = base(test_input)

        torch.testing.assert_close(merged_output, forward_output)
        second.merge_to(-second.multiplier)
        first.merge_to(-first.multiplier)

    def test_stacked_additive_and_dora_lokr_match_merge(self):
        base = nn.Linear(8, 8)
        additive = LokrModule(
            "additive",
            base,
            multiplier=0.4,
            lora_dim=4,
            full_matrix=True,
        )
        with torch.no_grad():
            additive.lokr_w1.normal_(mean=0.0, std=0.2)
            additive.lokr_w2.normal_(mean=0.0, std=0.2)
        additive.eval()
        dora = self._make_full_matrix_dora(base, multiplier=0.7)
        test_input = torch.randn(3, 8)
        original_weight = base.weight.detach().clone()

        additive.apply_to()
        dora.apply_to()
        forward_output = base(test_input)
        dora.restore()
        additive.restore()

        additive.merge_to(additive.multiplier)
        dora.merge_to(dora.multiplier)
        merged_output = base(test_input)
        torch.testing.assert_close(merged_output, forward_output)

        additive.merge_to(-additive.multiplier)
        dora.merge_to(-dora.multiplier)
        self.assertTrue(torch.equal(base.weight, original_weight))

    def test_lokr_configuration_matrix_forward_merge_round_trip(self):
        cases = (
            (
                "linear_low_rank",
                nn.Linear(16, 16),
                torch.randn(2, 16),
                {"lora_dim": 1, "alpha": 0.5, "factor": 4},
            ),
            (
                "linear_decompose_both",
                nn.Linear(16, 16),
                torch.randn(2, 16),
                {
                    "lora_dim": 1,
                    "factor": 4,
                    "decompose_both": True,
                    "use_scalar": True,
                    "rs_lora": True,
                },
            ),
            (
                "linear_w1_low_rank_w2_full",
                nn.Linear(8, 32),
                torch.randn(2, 8),
                {
                    "lora_dim": 2,
                    "factor": -1,
                    "decompose_both": True,
                    "unbalanced_factorization": True,
                },
            ),
            (
                "conv1d_tucker",
                nn.Conv1d(8, 8, kernel_size=3, padding=1),
                torch.randn(2, 8, 5),
                {"lora_dim": 1, "factor": 2, "use_tucker": True},
            ),
            (
                "conv2d_tucker_dora",
                nn.Conv2d(8, 8, kernel_size=3, padding=1),
                torch.randn(2, 8, 5, 5),
                {
                    "lora_dim": 1,
                    "factor": 2,
                    "use_tucker": True,
                    "weight_decompose": True,
                },
            ),
            (
                "conv2d_grouped_bypass",
                nn.Conv2d(8, 8, kernel_size=3, padding=1, groups=2),
                torch.randn(2, 8, 5, 5),
                {"lora_dim": 1, "factor": 2, "bypass_mode": True},
            ),
            (
                "conv3d_full_dora",
                nn.Conv3d(4, 4, kernel_size=3, padding=1),
                torch.randn(1, 4, 4, 4, 4),
                {
                    "lora_dim": 2,
                    "full_matrix": True,
                    "weight_decompose": True,
                    "wd_on_out": False,
                },
            ),
        )

        for name, base, test_input, kwargs in cases:
            with self.subTest(case=name):
                module = LokrModule(
                    name,
                    base,
                    multiplier=0.37,
                    **kwargs,
                )
                self._randomize_module(module)
                module.eval()
                original_weight = base.weight.detach().clone()

                module.apply_to()
                forward_output = base(test_input)
                module.restore()
                module.merge_to(module.multiplier)
                merged_output = base(test_input)

                torch.testing.assert_close(merged_output, forward_output)
                module.merge_to(-module.multiplier)
                self.assertTrue(torch.equal(base.weight, original_weight))

                state_dict = module.state_dict()
                weights = tuple(
                    state_dict.get(weight_name) for weight_name in module.weight_list
                )
                with torch.no_grad():
                    rebuilt = LokrModule.make_module_from_state_dict(
                        f"{name}_rebuilt",
                        base,
                        *weights,
                    )
                expected_weight, _ = module.get_merged_weight(
                    module.multiplier,
                    base.weight.shape,
                )
                rebuilt_weight, _ = rebuilt.get_merged_weight(
                    module.multiplier,
                    base.weight.shape,
                )
                torch.testing.assert_close(rebuilt_weight, expected_weight)

    def test_nonsquare_parameterization_preserves_weight_shape(self):
        base = nn.Linear(4, 6, bias=False)
        original_weight = base.weight.detach().clone()
        module = LokrModule.parametrize(
            base,
            "weight",
            multiplier=0.5,
            lora_dim=1,
            factor=2,
            full_matrix=True,
            weight_decompose=True,
        )
        self._randomize_module(module)
        parametrized_weight = base.weight

        expected_weight, _ = module.get_merged_weight(
            module.multiplier,
            original_weight.shape,
        )

        self.assertEqual(parametrized_weight.shape, original_weight.shape)
        torch.testing.assert_close(parametrized_weight, expected_weight)

    def test_full_matrix_dora_detaches_direction_norm(self):
        base = nn.Linear(8, 8)
        module = self._make_full_matrix_dora(base)
        direction = torch.randn_like(base.weight, requires_grad=True)
        zero_base = torch.zeros_like(direction)

        output = module.apply_weight_decompose(
            direction,
            multiplier=1.0,
            base_weight=zero_base,
        )
        output.sum().backward()

        direction_norm = (
            direction.detach()
            .reshape(direction.shape[0], -1)
            .norm(dim=1)
            .reshape_as(module.dora_scale)
            .clamp_min(torch.finfo(direction.dtype).eps)
        )
        expected_gradient = (module.dora_scale.detach() / direction_norm).expand_as(
            direction
        )
        torch.testing.assert_close(direction.grad, expected_gradient)

    def test_full_matrix_uses_unit_scale_with_rs_lora(self):
        base = nn.Linear(8, 8)
        module = self._make_full_matrix_dora(base, rs_lora=True)

        self.assertEqual(module.scale, 1.0)

    def test_dora_forces_rebuilt_weight_mode(self):
        base = nn.Linear(8, 8)
        module = self._make_full_matrix_dora(base, bypass_mode=True)

        self.assertFalse(module.bypass_mode)

    def test_non_dora_bypass_matches_merged_weight(self):
        base = nn.Linear(16, 16)
        module = LokrModule(
            "test",
            base,
            multiplier=0.4,
            lora_dim=1,
            alpha=0.5,
            factor=4,
            use_scalar=True,
            bypass_mode=True,
        )
        with torch.no_grad():
            module.scalar.fill_(0.6)
        module.eval()
        test_input = torch.randn(2, 16)
        original_weight = base.weight.detach().clone()
        original_bias = base.bias.detach().clone()

        module.apply_to()
        bypass_output = base(test_input)
        merged_weight, _ = module.get_merged_weight(
            module.multiplier,
            original_weight.shape,
        )
        expected_output = F.linear(test_input, merged_weight, original_bias)

        torch.testing.assert_close(bypass_output, expected_output)
        module.restore()
        module.merge_to(module.multiplier)
        torch.testing.assert_close(base(test_input), bypass_output)

    def test_apply_to_is_idempotent(self):
        base = nn.Linear(8, 8)
        module = self._make_full_matrix_dora(base)
        test_input = torch.randn(2, 8)

        module.apply_to()
        expected = base(test_input)
        module.apply_to()
        actual = base(test_input)

        self.assertEqual(base._lycoris_wrappers, [module])
        torch.testing.assert_close(actual, expected)

    def test_dora_magnitude_initialization_matches_weight_norm(self):
        for wd_on_out in (True, False):
            with self.subTest(wd_on_out=wd_on_out):
                base = nn.Conv2d(6, 8, kernel_size=3, groups=2)
                module = LokrModule(
                    "test",
                    base,
                    lora_dim=2,
                    full_matrix=True,
                    weight_decompose=True,
                    wd_on_out=wd_on_out,
                )
                if wd_on_out:
                    norm_dims = (1, 2, 3)
                    expected = torch.linalg.vector_norm(
                        base.weight.detach(),
                        dim=norm_dims,
                        keepdim=True,
                        dtype=torch.float32,
                    )
                else:
                    grouped = base.weight.detach().reshape(2, 4, 3, 3, 3)
                    expected = torch.linalg.vector_norm(
                        grouped,
                        dim=(1, 3, 4),
                        keepdim=True,
                        dtype=torch.float32,
                    )

                torch.testing.assert_close(module.dora_scale, expected)

    def test_nonfinite_alpha_is_rejected(self):
        base = nn.Linear(8, 8)

        with self.assertRaisesRegex(ValueError, "alpha"):
            LokrModule("test", base, alpha=float("nan"))

    def test_dora_rejects_unmaterialized_meta_weight(self):
        base = nn.Linear(8, 8, device="meta")

        with self.assertRaisesRegex(RuntimeError, "materialized"):
            LokrModule(
                "test",
                base,
                full_matrix=True,
                weight_decompose=True,
            )

    def test_nonsquare_tucker_bypass_matches_merged_weight(self):
        base = nn.Conv2d(12, 18, kernel_size=3, padding=1)
        module = LokrModule(
            "test",
            base,
            multiplier=0.4,
            lora_dim=1,
            alpha=0.7,
            factor=3,
            use_tucker=True,
            bypass_mode=True,
        )
        self._randomize_module(module)
        module.eval()
        test_input = torch.randn(2, 12, 5, 5)
        original_weight = base.weight.detach().clone()
        original_bias = base.bias.detach().clone()

        module.apply_to()
        bypass_output = base(test_input)
        merged_weight, _ = module.get_merged_weight(
            module.multiplier,
            original_weight.shape,
        )
        expected_output = F.conv2d(
            test_input,
            merged_weight,
            original_bias,
            padding=1,
        )

        self.assertEqual(module.lokr_w2_a.shape, (1, 6))
        torch.testing.assert_close(bypass_output, expected_output)

    def test_nonsquare_non_tucker_bypass_matches_merged_weight(self):
        base = nn.Conv2d(12, 18, kernel_size=3, padding=1)
        module = LokrModule(
            "test",
            base,
            multiplier=0.4,
            lora_dim=1,
            alpha=0.7,
            factor=3,
            use_tucker=False,
            bypass_mode=True,
        )
        self._randomize_module(module)
        module.eval()
        test_input = torch.randn(2, 12, 5, 5)
        merged_weight, _ = module.get_merged_weight(
            module.multiplier,
            base.weight.shape,
        )
        expected = F.conv2d(
            test_input,
            merged_weight,
            base.bias,
            padding=1,
        )

        module.apply_to()
        actual = base(test_input)

        self.assertEqual(module.lokr_w2_a.shape, (6, 1))
        torch.testing.assert_close(actual, expected)

    def test_functional_full_matrix_is_unscaled_and_finite(self):
        weight = torch.randn(12, 8)
        test_input = torch.randn(3, 8)
        params = functional_lokr.weight_gen(
            weight,
            rank=1,
            factor=2,
            decompose_both=True,
            full_matrix=True,
        )
        with torch.no_grad():
            for parameter in params:
                if parameter is not None:
                    parameter.normal_(mean=0.0, std=0.2)

        diff_weight = functional_lokr.diff_weight(*params, gamma=0.0)
        bypass_output = functional_lokr.bypass_forward_diff(
            test_input,
            *params,
            gamma=0.0,
        )

        self.assertIsNotNone(params[0])
        self.assertIsNotNone(params[3])
        self.assertTrue(torch.isfinite(diff_weight).all())
        torch.testing.assert_close(
            bypass_output,
            F.linear(test_input, diff_weight),
        )

    def test_functional_weight_generation_preserves_dtype_and_device(self):
        weight = torch.randn(12, 8, dtype=torch.float64)

        params = functional_lokr.weight_gen(weight, rank=1, factor=2)

        for parameter in params:
            if parameter is not None:
                self.assertEqual(parameter.dtype, weight.dtype)
                self.assertEqual(parameter.device, weight.device)
        with self.assertRaisesRegex(TypeError, "floating-point"):
            functional_lokr.weight_gen(
                torch.ones(8, 8, dtype=torch.int8),
                rank=1,
            )

    def test_functional_nonsquare_tucker_matches_rebuilt_weight(self):
        weight = torch.randn(18, 12, 3, 3)
        test_input = torch.randn(2, 12, 5, 5)
        params = functional_lokr.weight_gen(
            weight,
            rank=1,
            factor=3,
            tucker=True,
        )
        with torch.no_grad():
            for parameter in params:
                if parameter is not None:
                    parameter.normal_(mean=0.0, std=0.2)

        diff_weight = functional_lokr.diff_weight(*params, gamma=0.7)
        bypass_output = functional_lokr.bypass_forward_diff(
            test_input,
            torch.empty(0),
            *params,
            gamma=0.7,
            extra_args={"padding": 1},
        )

        self.assertEqual(params[4].shape, (1, 6))
        torch.testing.assert_close(
            bypass_output,
            F.conv2d(test_input, diff_weight, padding=1),
        )

    def test_functional_nonsquare_non_tucker_matches_rebuilt_weight(self):
        weight = torch.randn(18, 12, 3, 3)
        test_input = torch.randn(2, 12, 5, 5)
        params = functional_lokr.weight_gen(
            weight,
            rank=1,
            factor=3,
            tucker=False,
        )
        with torch.no_grad():
            for parameter in params:
                if parameter is not None:
                    parameter.normal_(mean=0.0, std=0.2)

        diff_weight = functional_lokr.diff_weight(*params, gamma=0.7)
        bypass_output = functional_lokr.bypass_forward_diff(
            test_input,
            torch.empty(0),
            *params,
            gamma=0.7,
            extra_args={"padding": 1},
        )

        self.assertEqual(params[4].shape, (6, 1))
        torch.testing.assert_close(
            bypass_output,
            F.conv2d(test_input, diff_weight, padding=1),
        )

    def test_functional_grouped_conv_and_padding_mode_fall_back_safely(self):
        weight = torch.randn(8, 4, 3, 3)
        test_input = torch.randn(2, 8, 5, 5)
        params = functional_lokr.weight_gen(
            weight,
            rank=1,
            factor=2,
            tucker=True,
        )
        with torch.no_grad():
            for parameter in params:
                if parameter is not None:
                    parameter.normal_(mean=0.0, std=0.2)

        diff_weight = functional_lokr.diff_weight(*params, gamma=0.7)
        bypass_output = functional_lokr.bypass_forward_diff(
            test_input,
            torch.empty(0),
            *params,
            gamma=0.7,
            extra_args={
                "groups": 2,
                "padding": 1,
                "padding_mode": "reflect",
            },
        )
        padded_input = F.pad(test_input, (1, 1, 1, 1), mode="reflect")

        torch.testing.assert_close(
            bypass_output,
            F.conv2d(padded_input, diff_weight, groups=2),
        )

    def test_conv_parameterization_preserves_nonsquare_weight_shapes(self):
        layers = (
            nn.Conv1d(4, 6, kernel_size=3, bias=False),
            nn.Conv2d(4, 6, kernel_size=(2, 3), bias=False),
            nn.Conv3d(4, 6, kernel_size=(2, 3, 1), bias=False),
        )

        for layer in layers:
            with self.subTest(layer=layer.__class__.__name__):
                original_weight = layer.weight.detach().clone()
                module = LokrModule.parametrize(
                    layer,
                    "weight",
                    multiplier=0.4,
                    lora_dim=2,
                    full_matrix=True,
                    weight_decompose=True,
                )
                self._randomize_module(module)
                expected, _ = module.get_merged_weight(
                    module.multiplier,
                    original_weight.shape,
                )

                self.assertEqual(layer.weight.shape, original_weight.shape)
                self.assertEqual(module.shape, original_weight.shape)
                torch.testing.assert_close(layer.weight, expected)

    def test_small_low_precision_dora_direction_keeps_initial_noop(self):
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                base = nn.Linear(4, 4, bias=False, dtype=dtype)
                with torch.no_grad():
                    base.weight.fill_(1e-4)
                module = LokrModule(
                    "test",
                    base,
                    lora_dim=2,
                    full_matrix=True,
                    weight_decompose=True,
                )

                merged, _ = module.get_merged_weight(1.0, base.weight.shape)
                functional = apply_dora_scale(
                    base.weight.detach(),
                    torch.zeros_like(base.weight),
                    module.dora_scale,
                    1.0,
                )

                torch.testing.assert_close(
                    merged,
                    base.weight.detach().to(merged),
                    rtol=0.0,
                    atol=1e-7,
                )
                torch.testing.assert_close(
                    functional,
                    base.weight.detach().to(functional),
                    rtol=0.0,
                    atol=1e-7,
                )

    def test_float64_dora_initialization_preserves_precision(self):
        base = nn.Linear(7, 5, bias=False, dtype=torch.float64)
        module = LokrModule(
            "test",
            base,
            lora_dim=2,
            full_matrix=True,
            weight_decompose=True,
        )
        expected_magnitude = torch.linalg.vector_norm(
            base.weight.detach(),
            dim=1,
            keepdim=True,
        )
        merged, _ = module.get_merged_weight(1.0, base.weight.shape)

        self.assertEqual(module.dora_scale.dtype, torch.float64)
        torch.testing.assert_close(module.dora_scale, expected_magnitude)
        torch.testing.assert_close(
            merged,
            base.weight.detach(),
            rtol=1e-14,
            atol=1e-14,
        )

    def test_legacy_residual_scale_load_is_scoped_per_adapter(self):
        first = self._make_full_matrix_dora(nn.Linear(8, 8))
        second = self._make_full_matrix_dora(nn.Linear(8, 8))
        adapters = nn.ModuleList((first, second))
        with torch.no_grad():
            first.lokr_residual_scale.fill_(0.25)
            second.lokr_residual_scale.fill_(0.5)
        state_dict = {
            key: value.detach().clone() for key, value in adapters.state_dict().items()
        }
        state_dict.pop("1.lokr_residual_scale")
        with torch.no_grad():
            first.lokr_residual_scale.fill_(0.9)
            second.lokr_residual_scale.fill_(0.9)

        adapters.load_state_dict(state_dict, strict=True)

        self.assertEqual(first.lokr_residual_scale.item(), 0.25)
        self.assertEqual(second.lokr_residual_scale.item(), 1.0)

    def test_dora_checkpoint_reconstruction_skips_base_materialization(self):
        source = self._make_full_matrix_dora(nn.Linear(8, 8))
        state_dict = source.state_dict()
        weights = tuple(state_dict.get(name) for name in source.weight_list)
        meta_base = nn.Linear(8, 8, device="meta")

        rebuilt = LokrModule.make_module_from_state_dict(
            "rebuilt",
            meta_base,
            *weights,
        )

        self.assertTrue(meta_base.weight.is_meta)
        torch.testing.assert_close(rebuilt.dora_scale, source.dora_scale)

    def test_checkpoint_reconstruction_materializes_helpers_in_meta_context(self):
        source = self._make_full_matrix_dora(nn.Linear(8, 8))
        state_dict = source.state_dict()
        state_dict.pop("lokr_residual_scale")
        weights = tuple(state_dict.get(name) for name in source.weight_list)
        meta_base = nn.Linear(8, 8, device="meta")

        with torch.device("meta"):
            rebuilt = LokrModule.make_module_from_state_dict(
                "rebuilt",
                meta_base,
                *weights,
            )

        self.assertFalse(rebuilt.lokr_w1.is_meta)
        self.assertFalse(rebuilt.lokr_w2.is_meta)
        self.assertFalse(rebuilt.scalar.is_meta)
        self.assertFalse(rebuilt.lokr_residual_scale.is_meta)
        self.assertFalse(rebuilt.dtype_tensor.is_meta)
        self.assertEqual(rebuilt.lokr_residual_scale.item(), 1.0)

    def test_assign_load_materializes_legacy_helpers_from_meta(self):
        source = LokrModule(
            "source",
            nn.Linear(8, 8),
            lora_dim=4,
            full_matrix=True,
        )
        state_dict = source.state_dict()
        state_dict.pop("lokr_residual_scale")
        with torch.device("meta"):
            target = LokrModule(
                "target",
                nn.Linear(8, 8, device="meta"),
                lora_dim=4,
                full_matrix=True,
            )

        target.load_state_dict(state_dict, strict=True, assign=True)

        self.assertFalse(target.lokr_w1.is_meta)
        self.assertFalse(target.lokr_w2.is_meta)
        self.assertFalse(target.scalar.is_meta)
        self.assertFalse(target.lokr_residual_scale.is_meta)
        self.assertFalse(target.dtype_tensor.is_meta)

    def test_assign_load_updates_dtype_marker_from_checkpoint_factors(self):
        source = LokrModule(
            "source",
            nn.Linear(8, 8),
            lora_dim=4,
            full_matrix=True,
        ).to(dtype=torch.bfloat16)
        target = LokrModule(
            "target",
            nn.Linear(8, 8),
            lora_dim=4,
            full_matrix=True,
        )

        target.load_state_dict(source.state_dict(), strict=True, assign=True)

        self.assertEqual(target.lokr_w1.dtype, torch.bfloat16)
        self.assertEqual(target.lokr_w2.dtype, torch.bfloat16)
        self.assertEqual(target.dtype, torch.bfloat16)

    def test_checkpoint_reconstruction_rejects_incompatible_inner_ranks(self):
        source = LokrModule(
            "source",
            nn.Linear(64, 64),
            lora_dim=2,
            factor=8,
            decompose_both=True,
        )
        state_dict = source.state_dict()
        weights = [state_dict.get(name) for name in source.weight_list]
        w1b_index = source.weight_list.index("lokr_w1_b")
        weights[w1b_index] = torch.randn(3, source.lokr_w1_b.size(1))

        with self.assertRaisesRegex(ValueError, "w1 factor ranks"):
            LokrModule.make_module_from_state_dict(
                "rebuilt",
                nn.Linear(64, 64),
                *weights,
            )

    def test_checkpoint_reconstruction_preserves_factor_representation(self):
        w1a = torch.randn(2, 2)
        w1b = torch.randn(2, 2)
        w2a = torch.randn(2, 2)
        w2b = torch.randn(2, 2)
        alpha = torch.tensor(1.0)

        rebuilt = LokrModule.make_module_from_state_dict(
            "rebuilt",
            nn.Linear(4, 4),
            None,
            w1a,
            w1b,
            None,
            w2a,
            w2b,
            None,
            None,
            alpha,
            None,
        )

        self.assertFalse(rebuilt.use_w1)
        self.assertFalse(rebuilt.use_w2)
        self.assertFalse(any(parameter.is_meta for parameter in rebuilt.parameters()))
        expected = torch.kron(w1a @ w1b, w2a @ w2b) * 0.5
        torch.testing.assert_close(rebuilt.get_weight((4, 4)), expected)

    def test_checkpoint_reconstruction_supports_1x1_tucker(self):
        w1 = torch.randn(2, 2)
        w2a = torch.randn(1, 2)
        w2b = torch.randn(1, 2)
        t2 = torch.randn(1, 1, 1, 1)

        rebuilt = LokrModule.make_module_from_state_dict(
            "rebuilt",
            nn.Conv2d(4, 4, kernel_size=1),
            w1,
            None,
            None,
            None,
            w2a,
            w2b,
            None,
            t2,
            torch.tensor(1.0),
            None,
        )

        self.assertTrue(rebuilt.use_w1)
        self.assertFalse(rebuilt.use_w2)
        self.assertTrue(rebuilt.tucker)
        self.assertEqual(rebuilt.get_weight((4, 4, 1, 1)).shape, (4, 4, 1, 1))

    def test_checkpoint_reconstruction_accepts_functional_conv_w2b(self):
        w1 = torch.randn(2, 2)
        w2a = torch.randn(2, 1)
        w2b = torch.randn(1, 2, 3, 3)

        rebuilt = LokrModule.make_module_from_state_dict(
            "rebuilt",
            nn.Conv2d(4, 4, kernel_size=3),
            w1,
            None,
            None,
            None,
            w2a,
            w2b,
            None,
            None,
            torch.tensor(1.0),
            None,
        )

        self.assertEqual(rebuilt.lokr_w2_b.shape, (1, 18))
        self.assertEqual(rebuilt.get_weight((4, 4, 3, 3)).shape, (4, 4, 3, 3))

    def test_checkpoint_reconstruction_preserves_mixed_parameter_dtypes(self):
        base = nn.Linear(8, 8, dtype=torch.bfloat16)
        module = self._make_full_matrix_dora(base).to(dtype=torch.bfloat16)
        module.dora_scale = nn.Parameter(module.dora_scale.detach().float())
        state_dict = module.state_dict()
        weights = tuple(state_dict.get(name) for name in module.weight_list)

        rebuilt = LokrModule.make_module_from_state_dict(
            "rebuilt",
            nn.Linear(8, 8, dtype=torch.bfloat16),
            *weights,
        )

        self.assertEqual(rebuilt.lokr_w1.dtype, torch.bfloat16)
        self.assertEqual(rebuilt.lokr_w2.dtype, torch.bfloat16)
        self.assertEqual(rebuilt.dora_scale.dtype, torch.float32)

    def test_low_rank_rebuild_preserves_compute_dtype_under_autocast(self):
        module = LokrModule(
            "test",
            nn.Linear(64, 64),
            lora_dim=1,
            factor=8,
            decompose_both=True,
        )
        self._randomize_module(module)
        expected = module.get_weight((64, 64), dtype=torch.float32)

        with torch.autocast("cpu", dtype=torch.bfloat16):
            actual = module.get_weight((64, 64), dtype=torch.float32)

        self.assertEqual(actual.dtype, torch.float32)
        torch.testing.assert_close(actual, expected)

    def test_merge_preserves_mixed_parameter_and_buffer_dtypes(self):
        base = nn.Linear(8, 8, dtype=torch.bfloat16)
        module = self._make_full_matrix_dora(base).to(dtype=torch.bfloat16)
        module.dora_scale = nn.Parameter(module.dora_scale.detach().float())
        parameter_dtypes = {
            name: parameter.dtype for name, parameter in module.named_parameters()
        }
        buffer_dtypes = {name: buffer.dtype for name, buffer in module.named_buffers()}

        module.merge_to(0.4, precise=True)
        module.merge_to(-0.4, precise=True)

        self.assertEqual(
            parameter_dtypes,
            {name: parameter.dtype for name, parameter in module.named_parameters()},
        )
        self.assertEqual(
            buffer_dtypes,
            {name: buffer.dtype for name, buffer in module.named_buffers()},
        )

    def test_onfly_merge_preserves_mixed_parameter_and_buffer_dtypes(self):
        base = nn.Linear(8, 8, dtype=torch.bfloat16)
        module = self._make_full_matrix_dora(base).to(dtype=torch.bfloat16)
        module.dora_scale = nn.Parameter(module.dora_scale.detach().float())
        parameter_dtypes = {
            name: parameter.dtype for name, parameter in module.named_parameters()
        }
        buffer_dtypes = {name: buffer.dtype for name, buffer in module.named_buffers()}

        module.onfly_merge(0.4)
        module.onfly_restore()

        self.assertEqual(
            parameter_dtypes,
            {name: parameter.dtype for name, parameter in module.named_parameters()},
        )
        self.assertEqual(
            buffer_dtypes,
            {name: buffer.dtype for name, buffer in module.named_buffers()},
        )

    def test_merge_ledger_rejects_external_weight_mutation(self):
        base = nn.Linear(8, 8)
        module = self._make_full_matrix_dora(base)
        module.merge_to(0.4)
        with torch.no_grad():
            base.weight.add_(1.0)

        with self.assertRaisesRegex(RuntimeError, "outside"):
            module.merge_to(-0.4)

    def test_merge_ledger_rejects_raw_data_mutation_and_parameter_replacement(self):
        for mutation in ("data", "parameter"):
            with self.subTest(mutation=mutation):
                base = nn.Linear(8, 8)
                module = self._make_full_matrix_dora(base)
                module.merge_to(0.4)
                if mutation == "data":
                    base.weight.data.add_(1.0)
                    message = "untracked data write"
                else:
                    base.weight = nn.Parameter(base.weight.detach().clone())
                    message = "Parameter was replaced"

                with self.assertRaisesRegex(RuntimeError, message):
                    module.merge_to(-0.4)

    def test_finalize_and_nonreversible_merge_release_full_weight_ledger(self):
        base = nn.Linear(8, 8)
        module = self._make_full_matrix_dora(base)
        expected, _ = module.get_merged_weight(0.4, base.weight.shape)

        module.merge_to(0.4)
        self.assertTrue(hasattr(base, "_lycoris_lokr_merge_base"))
        self.assertTrue(module.finalize_merge())
        self.assertFalse(hasattr(base, "_lycoris_lokr_merge_base"))
        self.assertFalse(hasattr(base, "_lycoris_lokr_merge_entries"))
        torch.testing.assert_close(base.weight, expected.to(base.weight))

        second_base = nn.Linear(8, 8)
        second = self._make_full_matrix_dora(second_base)
        second_expected, _ = second.get_merged_weight(
            0.4,
            second_base.weight.shape,
        )
        second.merge_to(0.4, reversible=False)
        self.assertFalse(hasattr(second_base, "_lycoris_lokr_merge_base"))
        torch.testing.assert_close(
            second_base.weight,
            second_expected.to(second_base.weight),
        )

    def test_stacked_dora_merge_uses_stable_application_order(self):
        base = nn.Linear(8, 8)
        first = self._make_full_matrix_dora(base, multiplier=0.4)
        second = self._make_full_matrix_dora(base, multiplier=0.4)
        test_input = torch.randn(3, 8)
        original_weight = base.weight.detach().clone()

        first.apply_to()
        second.apply_to()
        expected = base(test_input)
        second.restore()
        first.restore()

        second.merge_to(second.multiplier)
        first.merge_to(first.multiplier)
        torch.testing.assert_close(base(test_input), expected)

        first.merge_to(-first.multiplier)
        first.merge_to(first.multiplier)
        torch.testing.assert_close(base(test_input), expected)

        first.merge_to(-first.multiplier)
        second.merge_to(-second.multiplier)
        self.assertTrue(torch.equal(base.weight, original_weight))

    def test_onfly_merge_validates_multiplier_and_restore_order(self):
        base = nn.Linear(8, 8)
        first = self._make_full_matrix_dora(base, multiplier=0.4)
        second = self._make_full_matrix_dora(base, multiplier=0.7)
        original_weight = base.weight.detach().clone()

        with self.assertRaisesRegex(ValueError, "finite"):
            first.onfly_merge(float("nan"))
        self.assertTrue(torch.equal(base.weight, original_weight))

        first.onfly_merge(0.0)
        self.assertIsNone(first.cached_org_weight)
        first.onfly_restore()
        self.assertTrue(torch.equal(base.weight, original_weight))

        first.onfly_merge(first.multiplier)
        second.onfly_merge(second.multiplier)
        with self.assertRaisesRegex(RuntimeError, "reverse merge order"):
            first.onfly_restore()
        second.onfly_restore()
        first.onfly_restore()
        self.assertTrue(torch.equal(base.weight, original_weight))

    def test_onfly_restore_rejects_external_update_and_never_touches_bias(self):
        base = nn.Linear(8, 8, dtype=torch.bfloat16)
        base.bias = nn.Parameter(torch.randn(8, dtype=torch.float32))
        original_bias = base.bias.detach().clone()
        module = self._make_full_matrix_dora(base).to(dtype=torch.bfloat16)
        module.onfly_merge(0.4)
        self.assertTrue(torch.equal(base.bias, original_bias))
        base.weight.data.add_(1)

        with self.assertRaisesRegex(RuntimeError, "changed during"):
            module.onfly_restore()
        self.assertTrue(torch.equal(base.bias, original_bias))

    def test_network_onfly_restore_uses_reverse_merge_order(self):
        base = nn.Linear(8, 8)
        original_weight = base.weight.detach().clone()
        first = self._make_full_matrix_dora(base, multiplier=0.4)
        second = self._make_full_matrix_dora(base, multiplier=0.7)
        network = LycorisNetwork(nn.Sequential(), init_only=True)
        network.loras = [first, second]

        network.onfly_merge()
        network.onfly_restore()

        self.assertTrue(torch.equal(base.weight, original_weight))

    def test_additive_locon_and_lokr_stack_exactly_in_both_orders(self):
        base = nn.Linear(8, 8)
        original_weight = base.weight.detach().clone()
        original_bias = base.bias.detach().clone()
        lokr = LokrModule(
            "lokr",
            base,
            multiplier=0.4,
            full_matrix=True,
        )
        locon = LoConModule(
            "locon",
            base,
            multiplier=0.7,
            lora_dim=2,
            alpha=2,
        )
        self._randomize_module(lokr)
        self._randomize_module(locon)
        lokr.eval()
        locon.eval()
        test_input = torch.randn(3, 8)

        lokr_weight = lokr._calculate_merged_weight(
            original_weight,
            lokr.multiplier,
            original_weight.shape,
        )
        locon_diff, _ = locon.get_diff_weight(
            locon.multiplier,
            original_weight.shape,
        )
        expected = F.linear(
            test_input,
            lokr_weight + locon_diff,
            original_bias,
        )

        for first, second in ((locon, lokr), (lokr, locon)):
            with self.subTest(order=(first.name, second.name)):
                first.apply_to()
                second.apply_to()
                torch.testing.assert_close(base(test_input), expected)
                second.restore()
                first.restore()

    def test_base_dependent_adapters_cannot_stack_with_lokr(self):
        adapter_types = (
            DiagOFTModule,
            GLoRAModule,
            IA3Module,
            FullModule,
        )

        def make_other(adapter_type, base):
            if adapter_type in (DiagOFTModule, GLoRAModule):
                return adapter_type("other", base, lora_dim=2)
            return adapter_type("other", base)

        for adapter_type in adapter_types:
            for lokr_first in (False, True):
                with self.subTest(
                    adapter=adapter_type.__name__,
                    lokr_first=lokr_first,
                ):
                    base = nn.Linear(8, 8)
                    lokr = LokrModule(
                        "lokr",
                        base,
                        full_matrix=True,
                    )
                    other = make_other(adapter_type, base)
                    first, second = (lokr, other) if lokr_first else (other, lokr)
                    first.apply_to()
                    with self.assertRaises(RuntimeError):
                        second.apply_to()
                    first.restore()

    def test_merge_requires_inactive_forward_and_parametrization(self):
        base = nn.Linear(8, 8)
        module = self._make_full_matrix_dora(base)
        module.apply_to()
        with self.assertRaisesRegex(RuntimeError, "forward path|Restore"):
            module.merge_to(0.4)
        with self.assertRaisesRegex(RuntimeError, "forward path|Restore"):
            module.onfly_merge(0.4)
        module.restore()

        layer = nn.Linear(8, 8)
        parametrized = LokrModule.parametrize(
            layer,
            "weight",
            full_matrix=True,
            weight_decompose=True,
        )
        with self.assertRaisesRegex(RuntimeError, "parametrization"):
            parametrized.merge_to(0.4)
        with self.assertRaisesRegex(RuntimeError, "parametrization"):
            parametrized.onfly_merge(0.4)

    def test_active_and_committed_lokr_cannot_be_applied_twice(self):
        base = nn.Linear(8, 8)
        module = self._make_full_matrix_dora(base)
        module.merge_to(0.4)
        with self.assertRaisesRegex(RuntimeError, "reversible merge"):
            module.apply_to()
        module.merge_to(-0.4)
        module.apply_to()
        module.restore()

        module.merge_to(0.4)
        module.finalize_merge()
        with self.assertRaisesRegex(RuntimeError, "committed"):
            module.apply_to()
        with self.assertRaisesRegex(RuntimeError, "committed"):
            module.merge_to(0.4)

        second_base = nn.Linear(8, 8)
        second = self._make_full_matrix_dora(second_base)
        second.merge_to(0.4, reversible=False)
        with self.assertRaisesRegex(RuntimeError, "committed"):
            second.apply_to()

    def test_apply_rejects_active_onfly_merge(self):
        base = nn.Linear(8, 8)
        module = self._make_full_matrix_dora(base)
        module.onfly_merge(0.4)
        with self.assertRaisesRegex(RuntimeError, "on-the-fly"):
            module.apply_to()
        module.onfly_restore()

    def test_cross_algorithm_merge_states_fail_before_overwrite(self):
        base = nn.Linear(8, 8)
        ia3 = IA3Module("ia3", base)
        lokr = self._make_full_matrix_dora(base)
        with torch.no_grad():
            ia3.weight.normal_(mean=0.0, std=0.2)
        ia3.merge_to(0.3, precise=True)
        weight_after_ia3 = base.weight.detach().clone()

        with self.assertRaisesRegex(RuntimeError, "precise-merge snapshot"):
            lokr.merge_to(0.4)
        self.assertTrue(torch.equal(base.weight, weight_after_ia3))

        ia3.finalize_merge()
        lokr.merge_to(0.4)
        with self.assertRaisesRegex(RuntimeError, "reversible LoKr"):
            ia3.merge_to(0.3)
        lokr.merge_to(-0.4)

    def test_cross_algorithm_onfly_stack_and_generic_lifo_are_safe(self):
        base = nn.Linear(8, 8)
        lokr = self._make_full_matrix_dora(base)
        locon = LoConModule("locon", base, lora_dim=2)
        self._randomize_module(locon)
        original_weight = base.weight.detach().clone()

        lokr.onfly_merge(0.4)
        with self.assertRaisesRegex(RuntimeError, "Mixing LoKr"):
            locon.onfly_merge(0.3)
        lokr.onfly_restore()

        locon.onfly_merge(0.3)
        with self.assertRaisesRegex(RuntimeError, "Mixing LoKr"):
            lokr.onfly_merge(0.4)
        locon.onfly_restore()
        self.assertTrue(torch.equal(base.weight, original_weight))

        first = LoConModule("first", base, lora_dim=2)
        second = LoConModule("second", base, lora_dim=2)
        self._randomize_module(first)
        self._randomize_module(second)
        first.onfly_merge(0.2)
        second.onfly_merge(0.3)
        with self.assertRaisesRegex(RuntimeError, "reverse merge order"):
            first.onfly_restore()
        second.onfly_restore()
        first.onfly_restore()
        self.assertTrue(torch.equal(base.weight, original_weight))

    def test_network_finalize_visits_every_adapter_on_one_target(self):
        base = nn.Linear(8, 8)
        locon = LoConModule("locon", base, lora_dim=2)
        lokr = self._make_full_matrix_dora(base)
        lokr.merge_to(0.4)
        network = LycorisNetwork(nn.Sequential(), init_only=True)
        network.loras = [locon, lokr]

        network.finalize_merge()

        self.assertFalse(hasattr(base, "_lycoris_lokr_merge_entries"))
        self.assertTrue(lokr._lokr_merge_committed)

    def test_network_finalize_preflights_every_lokr_ledger(self):
        first_base = nn.Linear(8, 8)
        second_base = nn.Linear(8, 8)
        ia3 = IA3Module("ia3", first_base)
        lokr = self._make_full_matrix_dora(second_base)
        with torch.no_grad():
            ia3.weight.normal_(mean=0.0, std=0.2)
        ia3.merge_to(0.3, precise=True)
        lokr.merge_to(0.4)
        with torch.no_grad():
            second_base.weight.add_(1.0)

        network = LycorisNetwork(nn.Sequential(), init_only=True)
        network.loras = [ia3, lokr]

        with self.assertRaisesRegex(RuntimeError, "changed outside"):
            network.finalize_merge()

        self.assertTrue(
            hasattr(first_base, "_lycoris_precise_weight_current"),
            "A failed finalize preflight must not partially commit earlier targets.",
        )

    def test_network_rejects_mixed_reversible_merge_before_mutation(self):
        base = nn.Linear(8, 8)
        original_weight = base.weight.detach().clone()
        locon = LoConModule("locon", base, lora_dim=2)
        lokr = self._make_full_matrix_dora(base)
        network = LycorisNetwork(nn.Sequential(), init_only=True)
        network.loras = [locon, lokr]

        with self.assertRaisesRegex(RuntimeError, "cannot mix LoKr"):
            network.merge_to(reversible=True)
        self.assertTrue(torch.equal(base.weight, original_weight))

    def test_network_nonreversible_merge_preflights_every_target(self):
        first_base = nn.Linear(8, 8)
        second_base = nn.Linear(8, 8)
        first_weight = first_base.weight.detach().clone()
        second_weight = second_base.weight.detach().clone()
        first = self._make_full_matrix_dora(first_base)
        second = self._make_full_matrix_dora(second_base)
        ia3 = IA3Module("ia3", second_base)
        with torch.no_grad():
            ia3.weight.normal_(mean=0.0, std=0.2)
        ia3.merge_to(0.3, precise=True)
        second_weight = second_base.weight.detach().clone()
        network = LycorisNetwork(nn.Sequential(), init_only=True)
        network.loras = [first, second]

        with self.assertRaisesRegex(RuntimeError, "precise-merge snapshot"):
            network.merge_to(reversible=False)

        self.assertTrue(torch.equal(first_base.weight, first_weight))
        self.assertTrue(torch.equal(second_base.weight, second_weight))
        self.assertFalse(getattr(first, "_lokr_merge_committed", False))

    def test_network_nonreversible_merge_uses_lokr_application_order(self):
        base = nn.Linear(8, 8)
        first = self._make_full_matrix_dora(base, multiplier=0.4)
        second = self._make_full_matrix_dora(base, multiplier=0.4)
        test_input = torch.randn(3, 8)

        first.apply_to()
        second.apply_to()
        expected = base(test_input)
        second.restore()
        first.restore()

        network = LycorisNetwork(nn.Sequential(), init_only=True)
        network.loras = [second, first]
        network.merge_to(0.4, reversible=False)

        torch.testing.assert_close(base(test_input), expected)

    def test_network_onfly_restore_preflights_every_target(self):
        first_base = nn.Linear(8, 8)
        second_base = nn.Linear(8, 8)
        first_original = first_base.weight.detach().clone()
        second_original = second_base.weight.detach().clone()
        first = self._make_full_matrix_dora(first_base)
        second = self._make_full_matrix_dora(second_base)
        network = LycorisNetwork(nn.Sequential(), init_only=True)
        network.loras = [first, second]
        network.onfly_merge(0.4)
        first_merged = first_base.weight.detach().clone()
        second_merged = second_base.weight.detach().clone()
        with torch.no_grad():
            first_base.weight.add_(1.0)

        with self.assertRaisesRegex(RuntimeError, "changed"):
            network.onfly_restore()

        self.assertTrue(torch.equal(second_base.weight, second_merged))
        self.assertTrue(hasattr(first, "_lokr_onfly_multiplier"))
        self.assertTrue(hasattr(second, "_lokr_onfly_multiplier"))

        with torch.no_grad():
            first_base.weight.copy_(first_merged)
        network.onfly_restore()
        self.assertTrue(torch.equal(first_base.weight, first_original))
        self.assertTrue(torch.equal(second_base.weight, second_original))

    def test_network_onfly_restore_preflights_lower_lokr_frames(self):
        base = nn.Linear(8, 8)
        first = self._make_full_matrix_dora(base)
        second = self._make_full_matrix_dora(base)
        network = LycorisNetwork(nn.Sequential(), init_only=True)
        network.loras = [first, second]
        network.onfly_merge(0.4)
        merged = base.weight.detach().clone()
        first_factor = first.lokr_w1.detach().clone()
        with torch.no_grad():
            first.lokr_w1.add_(1.0)

        with self.assertRaisesRegex(RuntimeError, "active LoKr factors changed"):
            network.onfly_restore()

        self.assertTrue(torch.equal(base.weight, merged))
        self.assertTrue(hasattr(first, "_lokr_onfly_multiplier"))
        self.assertTrue(hasattr(second, "_lokr_onfly_multiplier"))

        with torch.no_grad():
            first.lokr_w1.copy_(first_factor)
        network.onfly_restore()

    def test_full_lifecycle_preserves_a_later_committed_lokr(self):
        base = nn.Linear(8, 8)
        original_weight_param = base.weight
        original_bias_param = base.bias
        original_weight = base.weight.detach().clone()
        original_bias = base.bias.detach().clone()
        full = FullModule("full", base)
        self._randomize_module(full)
        full_diff = full.weight.detach().clone()
        full_bias_diff = full.bias.detach().clone()
        test_input = torch.randn(3, 8)

        full.apply_to()
        torch.testing.assert_close(
            base(test_input),
            F.linear(
                test_input,
                original_weight + full_diff,
                original_bias + full_bias_diff,
            ),
        )
        full.restore()
        self.assertTrue(full.is_diff)
        self.assertIs(base.weight, original_weight_param)
        self.assertIs(base.bias, original_bias_param)
        torch.testing.assert_close(full.weight, full_diff)
        torch.testing.assert_close(full.bias, full_bias_diff)

        lokr = self._make_full_matrix_dora(base)
        lokr.merge_to(0.4, reversible=False)
        committed_weight = base.weight.detach().clone()
        committed_bias = base.bias.detach().clone()

        full.onfly_merge(0.5)
        torch.testing.assert_close(
            base.weight,
            committed_weight + full_diff * 0.5,
        )
        full.onfly_restore()
        self.assertTrue(torch.equal(base.weight, committed_weight))
        self.assertTrue(torch.equal(base.bias, committed_bias))

        full.merge_to(0.5, reversible=False)
        torch.testing.assert_close(
            base.weight,
            committed_weight + full_diff * 0.5,
        )

    def test_full_apply_rejects_changed_bias_structure_without_mutation(self):
        base = nn.Linear(8, 8, bias=False)
        full = FullModule("full", base)
        weight_param = base.weight
        weight_snapshot = base.weight.detach().clone()
        full_weight = full.weight.detach().clone()
        base.bias = nn.Parameter(torch.randn(8))

        with self.assertRaisesRegex(RuntimeError, "bias structure"):
            full.apply_to()

        self.assertIs(base.weight, weight_param)
        self.assertTrue(torch.equal(base.weight, weight_snapshot))
        self.assertTrue(torch.equal(full.weight, full_weight))
        self.assertTrue(full.is_diff)
        self.assertFalse(getattr(base, "_lycoris_wrappers", []))

    def test_full_checkpoint_can_add_bias_to_a_biasless_target(self):
        base = nn.Linear(8, 8, bias=False)
        original_weight = base.weight.detach().clone()
        diff = torch.randn_like(base.weight)
        diff_bias = torch.randn(8)
        full = FullModule.make_module_from_state_dict(
            "full",
            base,
            diff,
            diff_bias,
        )
        test_input = torch.randn(3, 8)

        full.apply_to()
        torch.testing.assert_close(
            base(test_input),
            F.linear(test_input, original_weight + diff, diff_bias),
        )
        full.restore()

        self.assertIsNone(base.bias)
        self.assertTrue(torch.equal(base.weight, original_weight))
        torch.testing.assert_close(full.weight, diff)
        self.assertTrue(torch.equal(full.bias, diff_bias))
        self.assertTrue(full.is_diff)

    def test_full_apply_rolls_back_if_target_parameter_removal_fails(self):
        base = nn.Linear(8, 8)
        full = FullModule("full", base)
        self._randomize_module(full)
        weight_param = base.weight
        bias_param = base.bias
        weight_snapshot = base.weight.detach().clone()
        bias_snapshot = base.bias.detach().clone()
        full_weight = full.weight.detach().clone()
        full_bias = full.bias.detach().clone()

        original_delattr = nn.Linear.__delattr__

        def fail_bias_delete(instance, name):
            if instance is base and name == "bias":
                raise RuntimeError("forced bias deletion failure")
            original_delattr(instance, name)

        with mock.patch.object(nn.Linear, "__delattr__", fail_bias_delete):
            with self.assertRaisesRegex(RuntimeError, "forced bias deletion"):
                full.apply_to()

        self.assertIs(base.weight, weight_param)
        self.assertIs(base.bias, bias_param)
        self.assertTrue(torch.equal(base.weight, weight_snapshot))
        self.assertTrue(torch.equal(base.bias, bias_snapshot))
        self.assertTrue(torch.equal(full.weight, full_weight))
        self.assertTrue(torch.equal(full.bias, full_bias))
        self.assertTrue(full.is_diff)
        self.assertFalse(getattr(base, "_lycoris_wrappers", []))

    def test_full_active_dtype_move_preserves_hidden_parameter_identity(self):
        base = nn.Linear(8, 8)
        weight_param = base.weight
        bias_param = base.bias
        full = FullModule("full", base)
        self._randomize_module(full)
        full.apply_to()

        base.to(dtype=torch.bfloat16)
        full.to(dtype=torch.bfloat16)
        test_input = torch.randn(3, 8, dtype=torch.bfloat16)
        self.assertEqual(base(test_input).dtype, torch.bfloat16)
        full.restore()

        self.assertIs(base.weight, weight_param)
        self.assertIs(base.bias, bias_param)
        self.assertEqual(base.weight.dtype, torch.bfloat16)
        self.assertEqual(base.bias.dtype, torch.bfloat16)
        self.assertEqual(base(test_input).dtype, torch.bfloat16)

    def test_grouped_conv_input_axis_dora_uses_per_group_magnitudes(self):
        base = nn.Conv2d(4, 6, kernel_size=3, groups=2, bias=False)
        module = self._make_full_matrix_dora(
            base,
            wd_on_out=False,
            multiplier=1.0,
        )
        base_weight = base.weight.detach()
        diff = module.get_weight(base_weight.shape) * module.scalar
        direction = base_weight + diff
        grouped = direction.reshape(2, 3, 2, 3, 3)
        direction_norm = torch.linalg.vector_norm(
            grouped,
            dim=(1, 3, 4),
            keepdim=True,
        ).clamp_min(torch.finfo(grouped.dtype).tiny)
        expected = (grouped * (module.dora_scale / direction_norm)).reshape_as(
            direction
        )
        actual, _ = module.get_merged_weight(1.0, base_weight.shape)

        self.assertEqual(module.dora_scale.shape, (2, 1, 2, 1, 1))
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(
            apply_dora_scale(
                base_weight,
                diff,
                module.dora_scale,
                1.0,
            ),
            expected,
        )

        state_dict = module.state_dict()
        weights = tuple(state_dict.get(name) for name in module.weight_list)
        rebuilt_base = nn.Conv2d(4, 6, kernel_size=3, groups=2, bias=False)
        rebuilt_base.load_state_dict(base.state_dict())
        rebuilt = LokrModule.make_module_from_state_dict(
            "rebuilt",
            rebuilt_base,
            *weights,
        )
        rebuilt_weight, _ = rebuilt.get_merged_weight(1.0, rebuilt_base.weight.shape)
        self.assertFalse(rebuilt.wd_on_out)
        self.assertEqual(rebuilt.dora_scale.shape, module.dora_scale.shape)
        torch.testing.assert_close(rebuilt_weight, actual)

    def test_grouped_conv_parameterization_preserves_grouped_dora_axis(self):
        layers = (
            nn.Conv1d(4, 6, kernel_size=3, groups=2, bias=False),
            nn.Conv2d(4, 6, kernel_size=(3, 2), groups=2, bias=False),
            nn.Conv3d(4, 6, kernel_size=(2, 3, 2), groups=2, bias=False),
        )
        for layer in layers:
            with self.subTest(layer=layer.__class__.__name__):
                original_shape = tuple(layer.weight.shape)
                module = LokrModule.parametrize(
                    layer,
                    "weight",
                    lora_dim=1,
                    factor=2,
                    full_matrix=True,
                    weight_decompose=True,
                    wd_on_out=False,
                )
                expected_scale_shape = (
                    2,
                    1,
                    2,
                    *([1] * (layer.weight.dim() - 2)),
                )

                self.assertEqual(module.org_module[0].groups, 2)
                self.assertEqual(tuple(layer.weight.shape), original_shape)
                self.assertEqual(module.dora_scale.shape, expected_scale_shape)

    def test_bfloat16_max_norm_never_rounds_above_limit(self):
        base = nn.Linear(8, 8, dtype=torch.bfloat16)
        module = self._make_full_matrix_dora(base).to(dtype=torch.bfloat16)
        self._randomize_module(module)
        limit = 0.125

        scaled, reported_norm = module.apply_max_norm(limit)
        merged, _ = module.get_merged_weight(1.0, base.weight.shape)
        actual_norm = (merged - base.weight.detach().to(merged)).norm().item()

        self.assertTrue(scaled)
        self.assertLessEqual(reported_norm.item(), limit)
        self.assertLessEqual(actual_norm, limit)

    def test_max_norm_is_deterministic_in_training_and_repeated_clips(self):
        base = nn.Linear(8, 8)
        module = self._make_full_matrix_dora(base, rank_dropout=0.5)
        module.eval()
        before, _ = module.get_merged_weight(1.0, base.weight.shape)
        before_norm = (before - base.weight.detach().to(before)).norm().item()
        module.train()

        first_limit = before_norm * 0.7
        scaled, first_norm = module.apply_max_norm(first_limit, device="cpu")
        self.assertTrue(scaled)
        self.assertTrue(module.training)
        self.assertLessEqual(first_norm.item(), first_limit)

        second_limit = first_norm.item() * 0.5
        scaled, second_norm = module.apply_max_norm(second_limit, device="cpu")
        self.assertTrue(scaled)
        self.assertTrue(module.training)
        self.assertLessEqual(second_norm.item(), second_limit)

        module.eval()
        merged, _ = module.get_merged_weight(1.0, base.weight.shape)
        actual_norm = (merged - base.weight.detach().to(merged)).norm()
        torch.testing.assert_close(actual_norm, second_norm)

    def test_bypass_accepts_mixed_factor_dtypes(self):
        base = nn.Linear(8, 8)
        module = LokrModule(
            "mixed",
            base,
            multiplier=0.4,
            lora_dim=2,
            full_matrix=True,
            bypass_mode=True,
        )
        self._randomize_module(module)
        module.lokr_w1 = nn.Parameter(module.lokr_w1.detach().double())
        test_input = torch.randn(3, 8)
        base_weight = base.weight.detach().clone()
        base_bias = base.bias.detach().clone()
        merged, _ = module.get_merged_weight(module.multiplier, base.weight.shape)
        expected = F.linear(test_input, base_weight, base_bias) + F.linear(
            test_input,
            (merged - base_weight.to(merged)).to(test_input),
        )

        module.apply_to()
        actual = base(test_input)

        torch.testing.assert_close(actual, expected)

    def test_functional_nonzero_padding_supports_same_and_valid(self):
        for padding in ("same", "valid"):
            with self.subTest(padding=padding):
                layer = nn.Conv2d(
                    4,
                    8,
                    kernel_size=(2, 3),
                    padding=padding,
                    padding_mode="reflect",
                    bias=False,
                )
                params = functional_lokr.weight_gen(
                    layer.weight,
                    rank=1,
                    factor=2,
                )
                with torch.no_grad():
                    for parameter in params:
                        if parameter is not None:
                            parameter.normal_(mean=0.0, std=0.2)
                diff_weight = functional_lokr.diff_weight(*params, gamma=0.7)
                test_input = torch.randn(2, 4, 6, 7)
                actual = functional_lokr.bypass_forward_diff(
                    test_input,
                    torch.empty(0),
                    *params,
                    gamma=0.7,
                    extra_args={
                        "padding": padding,
                        "padding_mode": "reflect",
                        "stride": layer.stride,
                        "dilation": layer.dilation,
                    },
                )
                padded = F.pad(
                    test_input,
                    layer._reversed_padding_repeated_twice,
                    mode=layer.padding_mode,
                )
                expected = F.conv2d(
                    padded,
                    diff_weight,
                    stride=layer.stride,
                    dilation=layer.dilation,
                )

                torch.testing.assert_close(actual, expected)

    def test_wrapper_forwards_all_lokr_scaling_options(self):
        base = nn.Sequential(nn.Linear(8, 8))
        try:
            network = create_lycoris(
                base,
                linear_dim=1,
                algo="lokr",
                dora_wd="True",
                wd_on_output="False",
                rs_lora="True",
                rank_dropout_scale="True",
                decompose_both="False",
            )
            module = network.loras[0]

            self.assertTrue(module.wd)
            self.assertFalse(module.wd_on_out)
            self.assertTrue(module.rs_lora)
            self.assertTrue(module.rank_dropout_scale)
            self.assertTrue(module.use_w1)
        finally:
            LycorisNetwork.apply_preset(PRESET["full"])

    def test_lokr_preset_override_normalizes_dora_axis_alias(self):
        base = nn.Sequential(nn.Linear(8, 8))
        preset = {
            "enable_conv": False,
            "target_module": [],
            "target_name": ["0"],
            "module_algo_map": {},
            "name_algo_map": {
                "0": {
                    "algo": "lokr",
                    "dim": 4,
                    "full_matrix": True,
                    "weight_decompose": True,
                    "wd_on_output": False,
                }
            },
            "use_fnmatch": False,
            "exclude_name": [],
        }
        try:
            LycorisNetwork.apply_preset(preset)
            network = LycorisNetwork(
                base,
                network_module="lora",
                lora_dim=4,
                conv_lora_dim=0,
            )
            module = network.loras[0]

            self.assertIsInstance(module, LokrModule)
            self.assertTrue(module.wd)
            self.assertFalse(module.wd_on_out)
            PresetConfig.from_dict(
                {"name_algo_map": preset["name_algo_map"]},
                strict=True,
            )
        finally:
            LycorisNetwork.apply_preset(PRESET["full"])

    def test_direct_wrapper_normalizes_root_aliases_and_local_dropouts(self):
        LycorisNetwork.apply_preset(
            {
                "enable_conv": False,
                "target_module": [],
                "target_name": ["0"],
                "module_algo_map": {},
                "name_algo_map": {
                    "0": {
                        "algo": "lokr",
                        "dim": 1,
                        "dropout": 0.1,
                        "rank_dropout": 0.2,
                        "module_dropout": 0.3,
                        "full_matrix": "False",
                        "weight_decompose": "False",
                        "decompose_both": "False",
                    }
                },
                "use_fnmatch": False,
                "exclude_name": [],
            }
        )
        network = LycorisNetwork(
            nn.Sequential(nn.Linear(64, 64)),
            network_module="lokr",
            lora_dim=1,
            conv_lora_dim=0,
            dora_wd="True",
            wd_on_output="False",
            rs_lora="False",
        )
        module = network.loras[0]

        # A local canonical value must override a root alias.
        self.assertFalse(module.wd)
        self.assertFalse(module.full_matrix)
        self.assertFalse(module.rs_lora)
        self.assertTrue(module.use_w1)
        self.assertEqual(module.dropout, 0.1)
        self.assertEqual(module.rank_dropout, 0.2)
        self.assertEqual(module.module_dropout, 0.3)

    def test_direct_kohya_wrapper_normalizes_root_dora_aliases(self):
        LycorisNetworkKohya.apply_preset(
            {
                "enable_conv": False,
                "unet_target_module": [],
                "unet_target_name": ["0"],
                "text_encoder_target_module": [],
                "text_encoder_target_name": [],
                "module_algo_map": {},
                "name_algo_map": {},
                "use_fnmatch": False,
            }
        )
        network = LycorisNetworkKohya(
            None,
            nn.Sequential(nn.Linear(8, 8)),
            network_module="lokr",
            lora_dim=4,
            conv_lora_dim=0,
            dora_wd="True",
            wd_on_output="False",
            full_matrix="True",
        )
        module = network.unet_loras[0]

        self.assertTrue(module.wd)
        self.assertFalse(module.wd_on_out)
        self.assertTrue(module.full_matrix)

    def test_strict_inherited_preset_rejects_unknown_option(self):
        with self.assertRaisesRegex(ValueError, "full_matrx"):
            PresetConfig.from_dict(
                {"name_algo_map": {"0": {"full_matrx": True}}},
                strict=True,
            )

    def test_mixed_algorithm_dora_stack_is_rejected(self):
        base = nn.Linear(8, 8)
        lokr = self._make_full_matrix_dora(base)
        locon = LoConModule("locon", base, lora_dim=2)
        lokr.apply_to()

        with self.assertRaisesRegex(RuntimeError, "order-dependent"):
            locon.apply_to()

        reverse_base = nn.Linear(8, 8)
        reverse_locon = LoConModule("locon", reverse_base, lora_dim=2)
        reverse_lokr = self._make_full_matrix_dora(reverse_base)
        reverse_locon.apply_to()

        with self.assertRaisesRegex(RuntimeError, "order-dependent"):
            reverse_lokr.apply_to()


if __name__ == "__main__":
    unittest.main()
