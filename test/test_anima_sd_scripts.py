"""Opt-in integration with a local sd-scripts checkout and a small real Anima.

Run with LYCORIS_SD_SCRIPTS_PATH=/path/to/sd-scripts. No pretrained checkpoint
is loaded. BF16 portable export drift is reported separately from exact native
resume checks; it is not hidden by widening the comparison tolerance.
"""

import json
import logging
import os
from pathlib import Path
import sys
import tempfile
import unittest
from contextlib import nullcontext

import torch


@unittest.skipUnless(
    os.environ.get("LYCORIS_SD_SCRIPTS_PATH"), "local sd-scripts checkout required"
)
class RealAnimaLoKr(unittest.TestCase):
    def test_precision_checkpoint_and_optimizer(self):
        sd_root = Path(os.environ["LYCORIS_SD_SCRIPTS_PATH"]).resolve()
        sys.path.insert(0, str(sd_root))
        try:
            from library.anima_models import Anima
        finally:
            sys.path.remove(str(sd_root))
        source = Path(sys.modules[Anima.__module__].__file__).resolve()
        self.assertEqual(source, sd_root / "library/anima_models.py")
        from lycoris.kohya import (
            LycorisNetworkKohya,
            create_network,
            create_network_from_weights,
        )
        from lycoris.logging import logger
        from lycoris.modules.lokr import LokrModule
        from safetensors.torch import load_file

        for name in (
            "ENABLE_CONV",
            "UNET_TARGET_REPLACE_MODULE",
            "UNET_TARGET_REPLACE_NAME",
            "TEXT_ENCODER_TARGET_REPLACE_MODULE",
            "TEXT_ENCODER_TARGET_REPLACE_NAME",
            "MODULE_ALGO_MAP",
            "NAME_ALGO_MAP",
            "USE_FNMATCH",
            "TARGET_EXCLUDE_NAME",
        ):
            self.addCleanup(
                setattr, LycorisNetworkKohya, name, getattr(LycorisNetworkKohya, name)
            )

        cases = [
            ("cpu", torch.float32, torch.float32),
            ("cpu", torch.bfloat16, torch.float32),
            ("cpu", torch.bfloat16, torch.bfloat16),
        ]
        if torch.cuda.is_available():
            cases += [
                ("cuda", torch.float32, torch.float32),
            ]
            if torch.cuda.is_bf16_supported():
                cases += [
                    ("cuda", torch.bfloat16, torch.float32),
                    ("cuda", torch.bfloat16, torch.bfloat16),
                ]
        original_level = logger.level
        logger.setLevel(logging.ERROR)
        try:
            for device, base_dtype, network_dtype in cases:
                for checkpointing in (False, True):
                    with self.subTest(
                        device=device,
                        base_dtype=base_dtype,
                        network_dtype=network_dtype,
                        checkpointing=checkpointing,
                    ):
                        torch.manual_seed(41)
                        model = Anima(
                            max_img_h=4,
                            max_img_w=4,
                            max_frames=1,
                            in_channels=16,
                            out_channels=16,
                            patch_spatial=2,
                            patch_temporal=1,
                            concat_padding_mask=False,
                            model_channels=64,
                            num_blocks=28,
                            num_heads=4,
                            mlp_ratio=2,
                            crossattn_emb_channels=32,
                            pos_emb_cls="rope3d",
                            use_adaln_lora=True,
                            adaln_lora_dim=8,
                            rope_enable_fps_modulation=False,
                            use_llm_adapter=False,
                            attn_mode="torch",
                        )
                        with torch.no_grad():
                            for module in model.modules():
                                if (
                                    isinstance(module, torch.nn.Linear)
                                    and torch.count_nonzero(module.weight) == 0
                                ):
                                    module.weight.normal_(0, 0.02)
                        model.requires_grad_(False).to(
                            device=device, dtype=base_dtype
                        ).train()
                        if checkpointing:
                            model.enable_gradient_checkpointing()
                        base_state = {
                            k: v.detach().cpu().clone()
                            for k, v in model.state_dict().items()
                        }
                        x = torch.randn(
                            1,
                            16,
                            1,
                            4,
                            4,
                            device=device,
                            dtype=base_dtype,
                            requires_grad=True,
                        )
                        timestep = torch.tensor(
                            [0.4], device=device, dtype=torch.float32
                        )
                        context = torch.randn(1, 8, 32, device=device, dtype=base_dtype)

                        def forward():
                            cast = (
                                torch.autocast(device, dtype=base_dtype)
                                if base_dtype != torch.float32
                                else nullcontext()
                            )
                            with cast:
                                return model(x, timestep, context=context)

                        with torch.no_grad():
                            baseline = forward()
                        network = create_network(
                            1.0,
                            100000,
                            1.0,
                            None,
                            None,
                            model,
                            algo="lokr",
                            preset="full",
                            factor=4,
                            dora_wd=True,
                            use_scalar=True,
                            train_llm_adapter=False,
                            warn_on_unmatched=False,
                        )
                        self.assertEqual(len(network.unet_loras), 448)
                        self.assertFalse(network.text_encoder_loras)
                        self.assertTrue(
                            all(
                                m.original_name.startswith("blocks.")
                                for m in network.unet_loras
                            )
                        )
                        self.assertTrue(
                            all(
                                m.full_matrix and m.scale == 1.0
                                for m in network.unet_loras
                            )
                        )
                        active = None
                        try:
                            network.apply_to(None, model, False, True)
                            active = network
                            network.to(device=device, dtype=network_dtype)
                            self.assertTrue(
                                all(
                                    m.dora_scale.dtype == torch.float32
                                    for m in network.unet_loras
                                )
                            )
                            with torch.no_grad():
                                self.assertTrue(torch.equal(forward(), baseline))
                            groups, _ = network.prepare_optimizer_params(
                                0.0, 0.03, 0.03
                            )
                            optimizer = torch.optim.SGD(groups)
                            optimizer.zero_grad(set_to_none=True)
                            loss = forward().float().square().mean()
                            loss.backward()
                            grads = [m.scalar.grad for m in network.unet_loras]
                            self.assertTrue(
                                all(
                                    g is not None and torch.isfinite(g).all()
                                    for g in grads
                                )
                            )
                            nonzero = sum(bool(torch.count_nonzero(g)) for g in grads)
                            self.assertGreater(nonzero, 0)
                            self.assertTrue(
                                all(
                                    torch.isfinite(p.grad).all()
                                    for p in network.parameters()
                                    if p.grad is not None
                                )
                            )
                            self.assertTrue(
                                all(p.grad is None for p in model.parameters())
                            )
                            optimizer.step()
                            with torch.no_grad():
                                trained = forward().detach()
                            self.assertTrue(torch.isfinite(trained).all())
                            self.assertFalse(torch.equal(trained, baseline))
                            with tempfile.TemporaryDirectory() as directory:
                                native = Path(directory) / "native.pt"
                                portable = Path(directory) / "portable.safetensors"
                                torch.save(network.state_dict(), native)
                                network.save_weights(str(portable), base_dtype, {})
                                portable_state = load_file(str(portable))
                                self.assertFalse(
                                    any(
                                        LokrModule.is_training_state_key(k)
                                        for k in portable_state
                                    )
                                )
                                network.restore()
                                active = None
                                self.assertTrue(
                                    all(
                                        torch.equal(v, model.state_dict()[k].cpu())
                                        for k, v in base_state.items()
                                    )
                                )
                                for kind, state in (
                                    (
                                        "native",
                                        torch.load(
                                            native,
                                            map_location="cpu",
                                            weights_only=True,
                                        ),
                                    ),
                                    ("portable", portable_state),
                                ):
                                    restored, _ = create_network_from_weights(
                                        1.0,
                                        None,
                                        None,
                                        None,
                                        model,
                                        weights_sd=state,
                                    )
                                    self.assertEqual(len(restored.unet_loras), 448)
                                    restored.apply_to(None, model, False, True)
                                    active = restored
                                    # Native preserves each tensor's resume dtype; portable uses its saved factor dtype.
                                    restored.to(device=device)
                                    with torch.no_grad():
                                        reloaded = forward().detach()
                                    self.assertTrue(torch.isfinite(reloaded).all())
                                    if kind == "native":
                                        torch.testing.assert_close(
                                            reloaded, trained, rtol=0, atol=0
                                        )
                                    else:
                                        difference = (
                                            reloaded.float() - trained.float()
                                        ).abs()
                                        tolerance = (
                                            3e-2
                                            if base_dtype == torch.bfloat16
                                            else 2e-5
                                        )
                                        within_reference = torch.allclose(
                                            reloaded,
                                            trained,
                                            rtol=tolerance,
                                            atol=tolerance,
                                        )
                                        if base_dtype == torch.float32:
                                            self.assertTrue(within_reference)
                                        print(
                                            json.dumps(
                                                {
                                                    "device": device,
                                                    "base_dtype": str(base_dtype),
                                                    "network_dtype": str(network_dtype),
                                                    "checkpointing": checkpointing,
                                                    "adapters": 448,
                                                    "nonzero_scalar_gradients": nonzero,
                                                    "initial_noop": True,
                                                    "base_preserved": True,
                                                    "native_exact": True,
                                                    "portable_max_abs": difference.max().item(),
                                                    "portable_mean_abs": difference.mean().item(),
                                                    "portable_reference_atol_rtol": tolerance,
                                                    "portable_within_reference": within_reference,
                                                }
                                            ),
                                            flush=True,
                                        )
                                    restored.restore()
                                    active = None
                        finally:
                            if active is not None:
                                active.restore()
        finally:
            logger.setLevel(original_level)
