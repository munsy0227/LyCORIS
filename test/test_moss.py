"""MOSS scopes without weights, plus opt-in tests of the local official source.

LYCORIS_MOSS_MODEL_PATH must point to a complete local MOSS-Image model folder.
The integration tests instantiate a tiny Transformer; they never load 5B weights.
"""

import copy
import importlib.util
import json
import logging
import os
from pathlib import Path
import re
import struct
import sys
import tempfile
import unittest

import torch
from torch import nn

from lycoris.config import PRESET, list_builtin_presets
from lycoris.kohya import (
    LycorisNetworkKohya,
    create_network,
    create_network_from_weights,
)
from lycoris.logging import logger
from lycoris.wrapper import (
    LycorisNetwork,
    create_lycoris,
    create_lycoris_from_weights,
)

PROJECTIONS = ("q", "k", "v", "o", "ffn_up_gate", "ffn_down")
MODEL_PATH = os.environ.get("LYCORIS_MOSS_MODEL_PATH")
DEFAULTS = {
    cls: {
        attr: copy.deepcopy(getattr(cls, attr)) for attr in cls._PRESET_FIELDS.values()
    }
    for cls in (LycorisNetwork, LycorisNetworkKohya)
}


class ProjectionBlock(nn.Module):
    # Deliberately does not have MOSS's private _Block class name.
    def __init__(self):
        super().__init__()
        self.proj = nn.ModuleDict({k: nn.Linear(8, 8) for k in PROJECTIONS})
        self.norm = nn.LayerNorm(8)


class TextBlock(nn.Module):
    def __init__(self):
        super().__init__()
        for name in ("qkv", "proj", "ffn_gate", "ffn_up", "ffn_down"):
            setattr(self, name, nn.Linear(8, 8))


class MossStructure(nn.Module):
    """Small storage with the production model's 269 Linear module paths."""

    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([ProjectionBlock() for _ in range(42)])
        self.text_embedder = nn.Module()
        self.text_embedder.proj = nn.Linear(8, 8)
        self.text_embedder.blocks = nn.ModuleList([TextBlock(), TextBlock()])
        self.time_embedding = nn.Sequential(nn.Linear(8, 8), nn.SiLU(), nn.Linear(8, 8))
        for name in (
            "video_embedder",
            "shared_block_modulation",
            "final_linear_video",
            "video_head_modulation",
        ):
            setattr(self, name, nn.Linear(8, 8))


class Qwen3Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(8, 8)


def make_network(api, model, preset, algo="lokr", text_encoder=None):
    options = dict(
        preset=preset,
        algo=algo,
        warn_on_unmatched=False,
        full_matrix=algo == "lokr",
        weight_decompose=algo == "lokr",
        use_scalar=algo == "lokr",
        rs_lora=algo == "lokr",
        factor=4,
    )
    if api == "generic":
        return create_lycoris(model, linear_dim=16, linear_alpha=7, **options)
    return create_network(1.0, 16, 7, None, text_encoder, model, **options)


def adapters(api, network):
    return network.loras if api == "generic" else network.unet_loras


def apply_network(api, network, model):
    if api == "generic":
        network.apply_to()
    else:
        network.apply_to(None, model, apply_text_encoder=False, apply_unet=True)


class PresetTestCase(unittest.TestCase):
    def setUp(self):
        self.saved = {
            cls: {attr: getattr(cls, attr) for attr in values}
            for cls, values in DEFAULTS.items()
        }
        for cls, values in DEFAULTS.items():
            for attr, value in values.items():
                setattr(cls, attr, copy.deepcopy(value))
        self.log_level = logger.level
        logger.setLevel(logging.ERROR)

    def tearDown(self):
        for cls, values in self.saved.items():
            for attr, value in values.items():
                setattr(cls, attr, value)
        logger.setLevel(self.log_level)


class MossPresetTests(PresetTestCase):
    def test_both_apis_select_exact_paths_and_exclude_text_encoder(self):
        self.assertTrue({"moss", "moss-full"} <= list_builtin_presets().keys())
        for api in ("generic", "kohya"):
            for preset, count in (("moss", 252), ("moss-full", 269)):
                with self.subTest(api=api, preset=preset):
                    model = MossStructure()
                    expected = {
                        id(module)
                        for name, module in model.named_modules()
                        if isinstance(module, nn.Linear)
                        and (preset == "moss-full" or name.startswith("layers."))
                    }
                    network = make_network(
                        api, model, preset, text_encoder=Qwen3Attention()
                    )
                    actual = adapters(api, network)
                    self.assertEqual(len(actual), count)
                    self.assertEqual({id(a.org_module[0]) for a in actual}, expected)
                    self.assertEqual(len({a.lora_name for a in actual}), count)
                    if api == "kohya":
                        self.assertEqual(network.text_encoder_loras, [])

    def test_patterns_do_not_match_similar_or_nested_names(self):
        model = MossStructure()
        model.layers[0].proj["q_extra"] = nn.Linear(8, 8)
        model.layers[0].proj["q"].extra = nn.Linear(8, 8)
        model.text_encoder = MossStructure()
        model.vae = nn.Linear(8, 8)
        model.time_embedding.add_module("10", nn.Linear(8, 8))
        for api in ("generic", "kohya"):
            for preset, count in (("moss", 252), ("moss-full", 269)):
                with self.subTest(api=api, preset=preset):
                    self.assertEqual(
                        len(adapters(api, make_network(api, model, preset))), count
                    )

    def test_success_failure_and_later_factories_do_not_leak_presets(self):
        for api, cls in (("generic", LycorisNetwork), ("kohya", LycorisNetworkKohya)):
            with self.subTest(api=api):
                before = copy.deepcopy(
                    {attr: getattr(cls, attr) for attr in DEFAULTS[cls]}
                )
                generic_before = make_network(api, MossStructure(), "full")
                first = make_network(api, MossStructure(), "moss")
                second = make_network(api, MossStructure(), "moss-full")
                with self.assertRaises(KeyError):
                    make_network(api, MossStructure(), "moss", algo="invalid")
                generic_after = make_network(api, MossStructure(), "full")
                self.assertEqual(
                    len(adapters(api, generic_before)),
                    len(adapters(api, generic_after)),
                )
                self.assertEqual(before, {attr: getattr(cls, attr) for attr in before})
                self.assertEqual(len(adapters(api, first)), 252)
                self.assertEqual(len(adapters(api, second)), 269)
                fields = (
                    "TARGET_REPLACE_NAME"
                    if api == "generic"
                    else "UNET_TARGET_REPLACE_NAME"
                )
                getattr(first, fields).append("unrelated")
                self.assertNotIn("unrelated", getattr(second, fields))
                self.assertNotIn("unrelated", PRESET["moss"]["target_name"])

    def test_instance_prefix_round_trip_without_global_state(self):
        # Existing custom-prefix presets must remain usable with the now-local config.
        with tempfile.TemporaryDirectory() as directory:
            preset = Path(directory) / "custom.toml"
            preset.write_text('target_module = ["Linear"]\nlora_prefix = "custom"\n')
            model = nn.Sequential(nn.Linear(8, 8))
            network = create_lycoris(model, preset=str(preset))
            network.apply_to()
            path = str(Path(directory) / "adapter.safetensors")
            network.save_weights(path, dtype=None, metadata=None)
            rebuilt, _ = create_lycoris_from_weights(
                1, path, nn.Sequential(nn.Linear(8, 8)), preset=str(preset)
            )
            self.assertEqual(len(rebuilt.loras), 1)
            self.assertEqual(rebuilt.loras[0].lora_name, "custom_0")
            self.assertEqual(
                LycorisNetwork.LORA_PREFIX, DEFAULTS[LycorisNetwork]["LORA_PREFIX"]
            )


@unittest.skipUnless(
    MODEL_PATH, "Set LYCORIS_MOSS_MODEL_PATH for local MOSS integration"
)
class LocalMossTests(PresetTestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(MODEL_PATH)
        spec = importlib.util.spec_from_file_location(
            "lycoris_test_moss_source", root / "pipeline_moss_image.py"
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        cls.transformer_class = module.MOSSImageTransformer

    def test_production_weight_headers_match_the_scopes(self):
        root = Path(MODEL_PATH) / "transformer"
        header = {}
        for path in sorted(root.glob("*.safetensors")):
            with path.open("rb") as stream:
                size = struct.unpack("<Q", stream.read(8))[0]
                header.update(json.loads(stream.read(size)))
        linear_paths = {
            key.removesuffix(".weight")
            for key, value in header.items()
            if key.endswith(".weight") and len(value["shape"]) == 2
        }
        self.assertEqual(len(linear_paths), 269)
        for preset, count in (("moss", 252), ("moss-full", 269)):
            expected = {
                p
                for p in linear_paths
                if preset == "moss-full" or p.startswith("layers.")
            }
            patterns = PRESET[preset]["target_name"]
            selected = {
                p for p in linear_paths if any(re.match(t, p) for t in patterns)
            }
            self.assertEqual(selected, expected)
            self.assertEqual(len(selected), count)

    def test_tiny_official_transformer_cpu(self):
        self._exercise(torch.device("cpu"), torch.float32)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_tiny_official_transformer_cuda_bf16(self):
        self.assertTrue(torch.cuda.is_bf16_supported())
        self._exercise(torch.device("cuda"), torch.bfloat16)

    def _exercise(self, device, dtype):
        # Fixed tolerances also apply to BF16 portable saves and merged inference.
        rtol, atol = (1e-5, 1e-6) if dtype == torch.float32 else (0.03, 0.03)
        for api in ("generic", "kohya"):
            for preset in ("moss", "moss-full"):
                for algo in ("lokr", "lora"):
                    with self.subTest(api=api, preset=preset, algo=algo, dtype=dtype):
                        torch.manual_seed(42)
                        model = (
                            self.transformer_class(
                                dim=64,
                                ffn_dim=128,
                                num_heads=4,
                                num_layers=2,
                                text_dim=32,
                                freq_dim=32,
                                latent_channels=8,
                            )
                            .to(device=device, dtype=dtype)
                            .requires_grad_(False)
                        )
                        base_state = copy.deepcopy(model.state_dict())
                        base_parameters = {
                            n: p.detach().clone() for n, p in model.named_parameters()
                        }
                        inputs = dict(
                            hidden_states=torch.randn(
                                2, 8, 2, 2, device=device, dtype=dtype
                            ),
                            timestep=torch.tensor([0.3, 0.7], device=device),
                            encoder_hidden_states=torch.randn(
                                2, 4, 32, device=device, dtype=dtype
                            ),
                            encoder_attention_mask=torch.tensor(
                                [[1, 1, 1, 1], [1, 1, 0, 0]], device=device
                            ),
                        )
                        with torch.no_grad():
                            baseline = model(**inputs).sample
                        network = make_network(api, model, preset, algo)
                        self.assertEqual(
                            len(adapters(api, network)), 12 if preset == "moss" else 29
                        )
                        apply_network(api, network, model)
                        network.to(device=device, dtype=dtype)
                        for adapter in adapters(api, network):
                            if algo == "lokr":
                                self.assertEqual(adapter.scale, 1.0)
                                self.assertEqual(
                                    adapter.dora_scale.dtype, torch.float32
                                )
                        torch.testing.assert_close(
                            model(**inputs).sample, baseline, rtol=rtol, atol=atol
                        )
                        optimizer = torch.optim.AdamW(network.parameters(), lr=1e-3)
                        target = torch.randn_like(baseline).float()
                        for _ in range(2):
                            optimizer.zero_grad(set_to_none=True)
                            loss = (
                                (model(**inputs).sample.float() - target)
                                .square()
                                .mean()
                            )
                            self.assertTrue(torch.isfinite(loss))
                            loss.backward()
                            grads = [
                                p.grad
                                for p in network.parameters()
                                if p.grad is not None
                            ]
                            self.assertTrue(grads)
                            self.assertTrue(all(torch.isfinite(g).all() for g in grads))
                            self.assertTrue(any(torch.count_nonzero(g) for g in grads))
                            optimizer.step()
                        for name, parameter in model.named_parameters():
                            self.assertIsNone(parameter.grad)
                            self.assertTrue(
                                torch.equal(parameter, base_parameters[name])
                            )
                        network.eval()
                        with torch.no_grad():
                            learned = model(**inputs).sample
                        self.assertFalse(torch.equal(learned, baseline))
                        native_state = copy.deepcopy(network.state_dict())
                        with tempfile.TemporaryDirectory() as directory:
                            path = str(Path(directory) / "adapter.safetensors")
                            network.save_weights(path, dtype=dtype, metadata=None)
                            if algo == "lokr":
                                from safetensors.torch import load_file

                                magnitudes = [
                                    v
                                    for k, v in load_file(path).items()
                                    if k.endswith("dora_scale")
                                ]
                                self.assertEqual(
                                    len(magnitudes), len(adapters(api, network))
                                )
                                self.assertTrue(
                                    all(v.dtype == torch.float32 for v in magnitudes)
                                )
                            network.restore()
                            for portable in (False, True):
                                fresh = self.transformer_class.from_config(model.config)
                                fresh.load_state_dict(base_state)
                                fresh = fresh.to(
                                    device=device, dtype=dtype
                                ).requires_grad_(False)
                                if portable:
                                    if api == "generic":
                                        rebuilt, _ = create_lycoris_from_weights(
                                            1, path, fresh
                                        )
                                    else:
                                        rebuilt, _ = create_network_from_weights(
                                            1, path, None, None, fresh
                                        )
                                else:
                                    rebuilt = make_network(api, fresh, preset, algo)
                                apply_network(api, rebuilt, fresh)
                                rebuilt.to(device=device, dtype=dtype).eval()
                                if not portable:
                                    rebuilt.load_state_dict(native_state)
                                with torch.no_grad():
                                    restored = fresh(**inputs).sample
                                torch.testing.assert_close(
                                    restored, learned, rtol=rtol, atol=atol
                                )
                                rebuilt.restore()
                        apply_network(api, network, model)
                        network.set_multiplier(0)
                        torch.testing.assert_close(
                            model(**inputs).sample, baseline, rtol=rtol, atol=atol
                        )
                        network.set_multiplier(0.5)
                        with torch.no_grad():
                            weighted = model(**inputs).sample
                        network.restore()
                        LycorisNetwork.merge_to(network, 0.5)
                        torch.testing.assert_close(
                            model(**inputs).sample, weighted, rtol=rtol, atol=atol
                        )
                        LycorisNetwork.merge_to(network, -0.5)
                        for name, parameter in model.named_parameters():
                            if algo == "lokr":
                                self.assertTrue(
                                    torch.equal(parameter, base_parameters[name])
                                )
                            else:
                                # LoRA uses additive merge/undo, without LoKR's exact ledger.
                                torch.testing.assert_close(
                                    parameter,
                                    base_parameters[name],
                                    rtol=rtol,
                                    atol=atol,
                                )


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
