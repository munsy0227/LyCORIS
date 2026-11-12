"""Both LyCORIS factories against the official sd-scripts MOSS implementation."""

import unittest

import torch
from torch import nn

from lycoris import create_lycoris
from lycoris.kohya import create_network


class OfficialSdScriptsMossTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            from library.moss_image_models import MossImage
        except ImportError as exc:
            raise unittest.SkipTest(
                "Requires sd-scripts with official MOSS support on PYTHONPATH"
            ) from exc
        cls.model_type = MossImage

    def test_production_scope_counts(self):
        from lycoris.config import MOSS_BLOCK_TARGETS, MOSS_FULL_TARGETS
        import re

        with torch.device("meta"):
            model = self.model_type(
                dim=32, ffn_dim=48, num_heads=4, num_layers=42, text_dim=16, freq_dim=16
            )
        for patterns, count in [(MOSS_BLOCK_TARGETS, 252), (MOSS_FULL_TARGETS, 269)]:
            targets = [
                name
                for name, module in model.named_modules()
                if isinstance(module, nn.Linear)
                and any(re.fullmatch(pattern, name) for pattern in patterns)
            ]
            self.assertEqual(len(targets), count)

    def test_both_factories_train_the_official_model(self):
        devices = [("cpu", torch.float32)]
        if torch.cuda.is_available():
            devices.append(("cuda", torch.bfloat16))
        for device, dtype in devices:
            for factory in ["generic", "kohya"]:
                for preset, count in [("moss", 12), ("moss-full", 29)]:
                    for algo, dora in [
                        ("lora", False),
                        ("lokr", False),
                        ("lokr", True),
                    ]:
                        with self.subTest(
                            device=device,
                            factory=factory,
                            preset=preset,
                            algo=algo,
                            dora=dora,
                        ):
                            torch.manual_seed(97)
                            model = self.model_type(
                                dim=32,
                                ffn_dim=48,
                                num_heads=4,
                                num_layers=2,
                                text_dim=16,
                                freq_dim=16,
                                latent_channels=128,
                            ).to(device=device, dtype=dtype)
                            model.requires_grad_(False)
                            x = torch.randn(1, 128, 2, 2, device=device, dtype=dtype)
                            text = torch.randn(1, 3, 16, device=device, dtype=dtype)
                            t = torch.tensor([300.0], device=device)
                            before = model(x, t, text).detach()
                            kwargs = dict(preset=preset, algo=algo)
                            if algo == "lokr":
                                kwargs.update(
                                    full_matrix=True, factor=4, weight_decompose=dora
                                )
                            if factory == "generic":
                                network = create_lycoris(
                                    model, linear_dim=4, linear_alpha=4, **kwargs
                                )
                                network.apply_to()
                            else:
                                network = create_network(
                                    1.0, 4, 4, None, None, model, **kwargs
                                )
                                network.apply_to(
                                    None,
                                    model,
                                    apply_text_encoder=False,
                                    apply_unet=True,
                                )
                            network.to(device=device, dtype=dtype)
                            adapters = (
                                network.loras
                                if factory == "generic"
                                else network.unet_loras
                            )
                            self.assertEqual(len(adapters), count)
                            torch.testing.assert_close(
                                model(x, t, text), before, rtol=0, atol=0
                            )
                            optimizer = torch.optim.AdamW(network.parameters(), lr=0.01)
                            for _ in range(2):
                                optimizer.zero_grad(set_to_none=True)
                                model(x, t, text).float().square().mean().backward()
                                grads = [
                                    p.grad
                                    for p in network.parameters()
                                    if p.grad is not None
                                ]
                                self.assertTrue(
                                    grads
                                    and all(torch.isfinite(g).all() for g in grads)
                                )
                                self.assertTrue(
                                    any(torch.count_nonzero(g) for g in grads)
                                )
                                optimizer.step()
                            self.assertFalse(torch.equal(before, model(x, t, text)))
                            self.assertTrue(
                                all(p.grad is None for p in model.parameters())
                            )
                            network.restore()


if __name__ == "__main__":
    unittest.main()
