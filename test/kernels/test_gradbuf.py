"""Aligned scratch gradients with independent low-precision custom-op outputs."""

import unittest

import torch

from lycoris.kernels.gradbuf import GradPack


class GradPackTests(unittest.TestCase):
    def test_scratch_and_tuning_twin_keep_aligned_nonoverlapping_views(self):
        pack = GradPack("cpu", (3, 3), (2, 5), (7,))
        for candidate in (pack, pack.like()):
            for value, view in enumerate(candidate, 1):
                self.assertEqual(view.storage_offset() * view.element_size() % 256, 0)
                view.fill_(value)
            for value, view in enumerate(candidate, 1):
                self.assertTrue(torch.equal(view, torch.full_like(view, value)))

    def test_cast_outputs_have_independent_storage_and_keep_values(self):
        pack = GradPack("cpu", (3, 3), (2, 5))
        for value, view in enumerate(pack, 1):
            view.fill_(value)
        for dtypes in (
            (torch.float16,),
            (torch.bfloat16,),
            (torch.float16, torch.float32),
        ):
            with self.subTest(dtypes=dtypes):
                outputs = pack.to(*dtypes)
                self.assertNotEqual(
                    outputs[0].untyped_storage().data_ptr(),
                    outputs[1].untyped_storage().data_ptr(),
                )
                for value, output in enumerate(outputs, 1):
                    expected_dtype = (
                        dtypes[0] if len(dtypes) == 1 else dtypes[value - 1]
                    )
                    self.assertEqual(output.dtype, expected_dtype)
                    self.assertTrue(torch.equal(output, torch.full_like(output, value)))
