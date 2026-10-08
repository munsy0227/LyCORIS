import unittest

import torch
import torch.nn.functional as F

from lycoris.functional.lokr import (
    _apply_factor_cap,
    bypass_forward_diff,
    diff_weight,
    kron_bypass,
    weight_gen,
)
from lycoris.kernels.select import choose, compiled


class TorchCompileCompatibility(unittest.TestCase):
    def setUp(self):
        torch._dynamo.reset()
        compiled.cache_clear()
        _apply_factor_cap.cache_clear()

    def test_outer_compile_owns_backend_selection(self):
        def fn(x):
            if choose((x,), backend="compile") != "torch":
                raise AssertionError("nested backend selected")
            return x + 1

        x = torch.randn(2)
        actual = torch.compile(fn, backend="eager", fullgraph=True)(x)
        torch.testing.assert_close(actual, x + 1)

    def test_outer_compile_preserves_full_matrix_zero_alpha(self):
        x = torch.randn(2, 6, dtype=torch.float64, requires_grad=True)
        w1 = torch.randn(2, 2, dtype=torch.float64, requires_grad=True)
        w2 = torch.randn(3, 3, dtype=torch.float64, requires_grad=True)

        def fn(x, w1, w2):
            weights = (w1, None, None, w2, None, None, None)
            rebuilt = diff_weight(*weights, gamma=0.0)
            bypass = bypass_forward_diff(x, None, *weights, gamma=0.0)
            return F.linear(x, rebuilt) + bypass

        expected = 2 * F.linear(x, torch.kron(w1, w2))
        actual = torch.compile(fn, backend="eager", fullgraph=True)(x, w1, w2)
        self.assertEqual(actual.dtype, torch.float64)
        torch.testing.assert_close(actual, expected)
        grad = torch.randn_like(actual)
        for actual_grad, expected_grad in zip(
            torch.autograd.grad(actual, (x, w1, w2), grad),
            torch.autograd.grad(expected, (x, w1, w2), grad),
        ):
            torch.testing.assert_close(actual_grad, expected_grad)

    def test_outer_compile_preserves_grouped_tucker_conv(self):
        x = torch.randn(2, 12, 5, 5, dtype=torch.float64, requires_grad=True)
        weights = weight_gen(torch.empty(18, 6, 3, 3, dtype=torch.float64), 1, factor=3)
        for weight in weights:
            if weight is not None:
                weight.normal_().requires_grad_(True)

        def fn(x, *weights):
            return bypass_forward_diff(
                x,
                None,
                *weights,
                gamma=0.7,
                extra_args={"groups": 2, "padding": 1},
            )

        w1, _, _, _, w2a, w2b, t = weights
        w2 = torch.einsum("ijhw,ip,jr->prhw", t, w2a, w2b)
        oracle = torch.kron(w1[:, :, None, None].contiguous(), w2.contiguous()) * 0.7
        expected = F.conv2d(x, oracle, groups=2, padding=1)
        actual = torch.compile(fn, backend="eager", fullgraph=True)(x, *weights)
        self.assertEqual(actual.dtype, torch.float64)
        torch.testing.assert_close(actual, expected)
        leaves = (x, *(w for w in weights if w is not None))
        grad = torch.randn_like(actual)
        for actual_grad, expected_grad in zip(
            torch.autograd.grad(actual, leaves, grad),
            torch.autograd.grad(expected, leaves, grad),
        ):
            torch.testing.assert_close(actual_grad, expected_grad)

    def test_outer_compile_does_not_start_per_op_compile(self):
        x = torch.randn(2, 16, requires_grad=True)
        w1 = torch.randn(2, 2, requires_grad=True)
        w2 = torch.randn(4, 8, requires_grad=True)

        def fn(x, w1, w2):
            return kron_bypass(
                x,
                w1,
                None,
                None,
                w2,
                None,
                None,
                scale=0.25,
                backend="compile",
            )

        expected = fn(x, w1, w2)
        compiled.cache_clear()
        actual = torch.compile(fn, backend="eager", fullgraph=True)(x, w1, w2)

        torch.testing.assert_close(actual, expected)
        self.assertEqual(compiled.cache_info().misses, 0)

        grad = torch.randn_like(actual)
        expected_grads = torch.autograd.grad(expected, (x, w1, w2), grad)
        actual_grads = torch.autograd.grad(actual, (x, w1, w2), grad)
        for actual_grad, expected_grad in zip(actual_grads, expected_grads):
            torch.testing.assert_close(actual_grad, expected_grad)


if __name__ == "__main__":
    unittest.main()
