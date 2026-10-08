import unittest
from itertools import product

import torch
from torch import nn
from lycoris.modules.lokr import LokrModule


def reference(module, multiplier):
    base = module.org_module[0].weight.detach().float()
    w1 = module.lokr_w1.float()
    w2 = module.lokr_w2.float()
    w1 = w1.reshape(*w1.shape, *([1] * (w2.dim() - 2)))
    direction = (
        base + torch.kron(w1.contiguous(), w2.contiguous()) * module.scalar.float()
    )
    magnitude = module.dora_scale
    if module.wd_on_out:
        viewed = direction
        dims = tuple(range(1, direction.dim()))
    elif direction.dim() == 2:
        viewed = direction
        dims = (0,)
    else:
        groups = module.org_module[0].groups
        viewed = direction.reshape(
            groups, direction.shape[0] // groups, *direction.shape[1:]
        )
        dims = (1, *range(3, viewed.dim()))
    norm = torch.linalg.vector_norm(viewed, dim=dims, keepdim=True)
    norm = norm.clamp_min(torch.finfo(viewed.dtype).tiny).detach()
    adapted = (viewed * (magnitude / norm)).reshape_as(direction)
    return base + (adapted - base) * multiplier


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class CudaDora(unittest.TestCase):
    def test_full_matrix_dora_math_grad_checkpoint_and_undo(self):
        cases = list(
            product(
                ("linear", "conv1d", "conv2d", "conv3d"),
                (True, False),
                (torch.float32, torch.float16, torch.bfloat16),
            )
        )
        for kind, axis, dtype in cases:
            with self.subTest(kind=kind, output_axis=axis, dtype=dtype):
                torch.manual_seed(41)
                if kind == "linear":
                    base = nn.Linear(48, 64, bias=False)
                else:
                    cls = {
                        "conv1d": nn.Conv1d,
                        "conv2d": nn.Conv2d,
                        "conv3d": nn.Conv3d,
                    }[kind]
                    base = cls(8, 12, 3, padding=1, groups=2, bias=False)
                base = base.to(device="cuda", dtype=dtype)
                original = base.weight.detach().clone()
                kwargs = dict(
                    lora_dim=100000,
                    alpha=7,
                    factor=4,
                    full_matrix=True,
                    weight_decompose=True,
                    use_scalar=True,
                    wd_on_out=axis,
                )
                net = LokrModule("gpu", base, **kwargs).to(device="cuda", dtype=dtype)
                net.eval()
                self.assertEqual(net.scale, 1.0)
                self.assertEqual(net.wd_on_out, axis)
                self.assertEqual(net.dora_scale.dtype, torch.float32)
                with torch.no_grad():
                    net.lokr_w1.normal_(0, 0.2)
                    net.lokr_w2.normal_(0, 0.02)
                    net.scalar.fill_(0.4)
                for multiplier in (0.0, 0.3, 1.0):
                    actual = net.get_merged_weight(multiplier)[0]
                    expected = reference(net, multiplier)
                    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
                actual = net.get_merged_weight(0.3)[0]
                expected = reference(net, 0.3)
                grad = torch.randn_like(actual)
                leaves = (net.lokr_w1, net.lokr_w2, net.scalar, net.dora_scale)
                actual_grads = torch.autograd.grad(actual, leaves, grad)
                expected_grads = torch.autograd.grad(expected, leaves, grad)
                tol = {
                    torch.float32: (1e-4, 1e-5),
                    torch.float16: (2e-2, 1e-4),
                    torch.bfloat16: (5e-2, 1e-3),
                }[dtype]
                for got, wanted in zip(actual_grads, expected_grads):
                    self.assertTrue(torch.isfinite(got).all())
                    torch.testing.assert_close(got, wanted, rtol=tol[0], atol=tol[1])
                rebuilt = LokrModule("gpu", base, **kwargs).to(
                    device="cuda", dtype=dtype
                )
                rebuilt.eval()
                rebuilt.load_state_dict(net.state_dict())
                self.assertTrue(
                    torch.equal(
                        net.get_merged_weight()[0], rebuilt.get_merged_weight()[0]
                    )
                )
                expected_merged = net.get_merged_weight()[0].detach().to(dtype)
                net.merge_to(1.0)
                torch.testing.assert_close(base.weight, expected_merged, rtol=0, atol=0)
                net.merge_to(-1.0)
                self.assertTrue(torch.equal(base.weight, original))

    def test_zero_base_initial_gradient(self):
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                base = nn.Linear(16, 16, bias=False).to(device="cuda", dtype=dtype)
                with torch.no_grad():
                    base.weight[0].zero_()
                net = LokrModule(
                    "zero",
                    base,
                    lora_dim=100000,
                    factor=4,
                    weight_decompose=True,
                    use_scalar=True,
                ).to(device="cuda", dtype=dtype)
                merged = net.get_merged_weight()[0]
                self.assertTrue(torch.equal(merged, base.weight.float()))
                grads = torch.autograd.grad(
                    merged[0].sum(), (net.scalar, net.dora_scale)
                )
                self.assertTrue(torch.equal(grads[0], torch.zeros_like(grads[0])))
                self.assertEqual(grads[1][0].abs().sum().item(), 0)

    def test_mock_anima_448_bf16_optimizer_step(self):
        # Uses tiny fixture layers with the real 28-block scope; no model
        # checkpoint is needed and this does not assert real training quality.
        from lycoris.kohya import create_network
        from test.test_kohya_optimizer import (
            Anima,
            AnimaOfficialScopeTests,
            _AnimaTextEncoder,
        )

        fixture = AnimaOfficialScopeTests()
        fixture.setUp()
        network = None
        try:
            unet = Anima(28).to(device="cuda", dtype=torch.bfloat16)
            text_encoder = _AnimaTextEncoder()
            unet.requires_grad_(False)
            network = create_network(
                1.0,
                100000,
                1.0,
                None,
                text_encoder,
                unet,
                algo="lokr",
                preset="full",
                factor=4,
                dora_wd=True,
                use_scalar=True,
                train_llm_adapter=False,
                warn_on_unmatched=False,
                network_reg_dims=(
                    r".*self\_attn.*=100000,"
                    r".*cross\_attn.*=100000,"
                    r".*mlp.*=100000"
                ),
            )
            self.assertEqual(len(network.unet_loras), 448)
            network.apply_to(text_encoder, unet, False, True)
            network.to(device="cuda", dtype=torch.bfloat16)
            self.assertEqual(len(network.loras), 448)
            self.assertTrue(all(lora.full_matrix for lora in network.loras))
            self.assertTrue(
                all(lora.dora_scale.dtype == torch.float32 for lora in network.loras)
            )
            groups, _ = network.prepare_optimizer_params(0.0, 0.1, 0.1)
            optimizer = torch.optim.SGD(groups)
            layer = unet.blocks[0].adaln_modulation_self_attn[1]
            adapter = next(
                lora for lora in network.loras if lora.org_module[0] is layer
            )
            inputs = torch.randn(4, 4, device="cuda", dtype=torch.bfloat16)
            before = torch.nn.functional.linear(inputs, layer.weight).detach()
            actual = layer(inputs)
            self.assertTrue(torch.equal(before, actual))
            actual.float().square().mean().backward()
            self.assertIsNotNone(adapter.scalar.grad)
            self.assertTrue(torch.isfinite(adapter.scalar.grad).all())
            self.assertGreater(adapter.scalar.grad.abs().sum().item(), 0)
            optimizer.step()
            after = layer(inputs)
            self.assertTrue(torch.isfinite(after).all())
            self.assertFalse(torch.equal(before, after))
        finally:
            if network is not None:
                network.restore()
            fixture.tearDown()
