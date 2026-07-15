import unittest

import torch
from torch import nn

from lycoris.kohya import LycorisNetworkKohya, create_network
from lycoris.modules import LoConModule, LohaModule, LokrModule


class _AnimaAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(4, 4, bias=False)
        self.k_proj = nn.Linear(4, 4, bias=False)
        self.v_proj = nn.Linear(4, 4, bias=False)
        self.output_proj = nn.Linear(4, 4, bias=False)


class _AnimaMlp(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(4, 8, bias=False)
        self.layer2 = nn.Linear(8, 4, bias=False)


def _anima_modulation():
    return nn.Sequential(
        nn.SiLU(),
        nn.Linear(4, 4, bias=False),
        nn.Linear(4, 12, bias=False),
    )


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_norm_self_attn = nn.LayerNorm(4, elementwise_affine=False)
        self.adaln_modulation_cross_attn = _anima_modulation()
        self.adaln_modulation_mlp = _anima_modulation()
        self.adaln_modulation_self_attn = _anima_modulation()
        self.cross_attn = _AnimaAttention()
        self.layer_norm_cross_attn = nn.LayerNorm(4, elementwise_affine=False)
        self.layer_norm_mlp = nn.LayerNorm(4, elementwise_affine=False)
        self.mlp = _AnimaMlp()
        self.self_attn = _AnimaAttention()


class PatchEmbed(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(4, 4, bias=False)


class TimestepEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_1 = nn.Linear(4, 4, bias=False)
        self.linear_2 = nn.Linear(4, 4, bias=False)


class FinalLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4, bias=False)


class LLMAdapterTransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(4, 4, bias=False)


class _LLMAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([LLMAdapterTransformerBlock()])


class Anima(nn.Module):
    def __init__(self, block_count):
        super().__init__()
        self.blocks = nn.ModuleList(Block() for _ in range(block_count))
        self.x_embedder = PatchEmbed()
        self.t_embedder = TimestepEmbedding()
        self.final_layer = FinalLayer()
        self.llm_adapter = _LLMAdapter()


class Qwen3Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(4, 4, bias=False)
        self.k_proj = nn.Linear(4, 4, bias=False)
        self.v_proj = nn.Linear(4, 4, bias=False)
        self.o_proj = nn.Linear(4, 4, bias=False)


class _AnimaTextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = Qwen3Attention()


class AnimaOfficialScopeTests(unittest.TestCase):
    _PRESET_STATE_FIELDS = (
        "ENABLE_CONV",
        "UNET_TARGET_REPLACE_MODULE",
        "UNET_TARGET_REPLACE_NAME",
        "TEXT_ENCODER_TARGET_REPLACE_MODULE",
        "TEXT_ENCODER_TARGET_REPLACE_NAME",
        "MODULE_ALGO_MAP",
        "NAME_ALGO_MAP",
        "USE_FNMATCH",
    )
    OFFICIAL_BLOCK_SUFFIXES = (
        "adaln_modulation_cross_attn.1",
        "adaln_modulation_cross_attn.2",
        "adaln_modulation_mlp.1",
        "adaln_modulation_mlp.2",
        "adaln_modulation_self_attn.1",
        "adaln_modulation_self_attn.2",
        "cross_attn.q_proj",
        "cross_attn.k_proj",
        "cross_attn.v_proj",
        "cross_attn.output_proj",
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.output_proj",
        "mlp.layer1",
        "mlp.layer2",
    )

    def setUp(self):
        self._preset_state = {
            field: getattr(LycorisNetworkKohya, field)
            for field in self._PRESET_STATE_FIELDS
        }
        LycorisNetworkKohya.MODULE_ALGO_MAP = {}
        LycorisNetworkKohya.NAME_ALGO_MAP = {}
        LycorisNetworkKohya.USE_FNMATCH = False

    @staticmethod
    def _create_network(block_count=28, **kwargs):
        text_encoder = _AnimaTextEncoder()
        unet = Anima(block_count)
        network = create_network(
            1.0,
            4,
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
            **kwargs,
        )
        return network, text_encoder, unet

    def tearDown(self):
        for field, value in self._preset_state.items():
            setattr(LycorisNetworkKohya, field, value)

    def test_anima_full_preset_matches_official_diffusion_module_scope(self):
        network, text_encoder, unet = self._create_network(train_norm=True)
        expected_names = {
            f"blocks.{block_index}.{suffix}"
            for block_index in range(28)
            for suffix in self.OFFICIAL_BLOCK_SUFFIXES
        }
        actual_names = {lora.original_name for lora in network.unet_loras}

        self.assertEqual(len(network.unet_loras), 448)
        self.assertEqual(actual_names, expected_names)
        self.assertTrue(
            all(isinstance(lora, LokrModule) for lora in network.unet_loras)
        )
        self.assertTrue(
            all(
                isinstance(lora.org_module[0], nn.Linear) for lora in network.unet_loras
            )
        )

        # sd-scripts' network_train_unet_only=true maps to this selection.
        self.assertGreater(len(network.text_encoder_loras), 0)
        try:
            network.apply_to(text_encoder, unet, False, True)
            self.assertEqual(network.text_encoder_loras, [])
            self.assertEqual(len(network.loras), 448)
            self.assertFalse(
                any(key.startswith("lora_te") for key in network.state_dict())
            )
        finally:
            network.restore()

    def test_anima_patterns_are_forwarded_and_can_override_default_scope(self):
        network, _, _ = self._create_network(
            block_count=1,
            exclude_patterns=[r".*self_attn.*"],
            include_patterns=[r".*final_layer.*"],
        )
        expected_names = {
            f"blocks.0.{suffix}"
            for suffix in self.OFFICIAL_BLOCK_SUFFIXES
            if "self_attn" not in suffix
        }
        expected_names.add("final_layer.linear")

        self.assertEqual(
            {lora.original_name for lora in network.unet_loras}, expected_names
        )

    def test_user_dimension_patterns_cover_every_official_block_target(self):
        network, _, _ = self._create_network(
            block_count=1,
            network_reg_dims=(
                r".*self\_attn.*=100000,"
                r".*cross\_attn.*=100000,"
                r".*mlp.*=100000"
            ),
        )

        self.assertEqual(len(network.unet_loras), 16)
        self.assertTrue(all(lora.lora_dim == 100000 for lora in network.unet_loras))
        self.assertTrue(all(lora.full_matrix for lora in network.unet_loras))


class KohyaOptimizerParamTests(unittest.TestCase):
    @staticmethod
    def _linear():
        layer = nn.Linear(8, 8, bias=False)
        layer.requires_grad_(False)
        return layer

    @classmethod
    def _lokr(cls, name):
        return LokrModule(name, cls._linear(), lora_dim=4, full_matrix=True)

    @classmethod
    def _network(
        cls,
        text_encoder_loras=(),
        unet_loras=(),
        reg_lrs=None,
        register=True,
    ):
        network = LycorisNetworkKohya.__new__(LycorisNetworkKohya)
        nn.Module.__init__(network)
        network.text_encoder_loras = list(text_encoder_loras)
        network.unet_loras = list(unet_loras)
        network.loras = network.text_encoder_loras + network.unet_loras
        network.reg_lrs = reg_lrs
        network.loraplus_lr_ratio = None
        network.loraplus_unet_lr_ratio = None
        network.loraplus_text_encoder_lr_ratio = None

        if register:
            for adapter in network.loras:
                network.add_module(adapter.lora_name, adapter)

        return network

    @staticmethod
    def _group_names(network, groups, descriptions):
        names_by_id = {id(param): name for name, param in network.named_parameters()}
        return {
            description: {names_by_id[id(param)] for param in group["params"]}
            for group, description in zip(groups, descriptions)
        }

    def test_loraplus_groups_match_supported_adapter_parameter_roles(self):
        lokr = self._lokr("unet_lokr")
        low_rank_layer = nn.Linear(64, 64, bias=False)
        low_rank_layer.requires_grad_(False)
        low_rank_lokr = LokrModule(
            "unet_lokr_low_rank",
            low_rank_layer,
            lora_dim=1,
            factor=4,
            decompose_both=True,
        )
        locon = LoConModule("unet_locon", self._linear(), lora_dim=2)
        loha = LohaModule("unet_loha", self._linear(), lora_dim=2)
        network = self._network(
            unet_loras=(lokr, low_rank_lokr, locon, loha),
        )
        network.set_loraplus_lr_ratio(2.0, None, None)

        groups, descriptions = network.prepare_optimizer_params(
            text_encoder_lr=0.0,
            unet_lr=1.0,
            learning_rate=1.0,
        )

        self.assertEqual(descriptions, ["unet", "unet plus"])
        self.assertEqual([group["lr"] for group in groups], [1.0, 2.0])
        names = self._group_names(network, groups, descriptions)
        self.assertEqual(
            names["unet plus"],
            {
                "unet_locon.lora_up.weight",
                "unet_loha.hada_w2_a",
            },
        )
        self.assertTrue(
            {
                "unet_lokr.lokr_w1",
                "unet_lokr.lokr_w2",
                "unet_lokr_low_rank.lokr_w1_a",
                "unet_lokr_low_rank.lokr_w1_b",
                "unet_lokr_low_rank.lokr_w2_a",
                "unet_lokr_low_rank.lokr_w2_b",
            }.issubset(names["unet"])
        )
        self.assertIn("unet_locon.lora_down.weight", names["unet"])
        self.assertIn("unet_loha.hada_w2_b", names["unet"])

    def test_zero_text_encoder_lr_stays_frozen_after_prepare_grad(self):
        text_lokr = self._lokr("te_lokr")
        unet_lokr = self._lokr("unet_lokr")
        network = self._network(
            text_encoder_loras=(text_lokr,), unet_loras=(unet_lokr,)
        )
        network.set_loraplus_lr_ratio(2.0, None, None)

        network.prepare_optimizer_params(1.0, 1.0, 1.0)
        for param in text_lokr.parameters():
            param.grad = torch.ones_like(param)

        groups, _ = network.prepare_optimizer_params(0.0, 1.0, 1.0)
        optimizer_param_ids = {
            id(param) for group in groups for param in group["params"]
        }
        self.assertTrue(
            all(
                id(param) not in optimizer_param_ids for param in text_lokr.parameters()
            )
        )
        self.assertTrue(
            all(not param.requires_grad for param in text_lokr.parameters())
        )
        self.assertTrue(all(param.grad is None for param in text_lokr.parameters()))
        self.assertTrue(all(param.requires_grad for param in unet_lokr.parameters()))

        network.requires_grad_(True)
        network.prepare_grad_etc(None, None)
        self.assertTrue(
            all(not param.requires_grad for param in text_lokr.parameters())
        )
        self.assertTrue(all(param.requires_grad for param in unet_lokr.parameters()))

        inputs = torch.randn(2, 8)
        for _ in range(2):
            (text_lokr(inputs).sum() + unet_lokr(inputs).sum()).backward()
        self.assertTrue(all(param.grad is None for param in text_lokr.parameters()))
        self.assertTrue(any(param.grad is not None for param in unet_lokr.parameters()))

    def test_none_unet_lr_freezes_and_a_later_nonzero_lr_reenables(self):
        unet_lokr = self._lokr("unet_lokr")
        network = self._network(unet_loras=(unet_lokr,))

        groups, descriptions = network.prepare_optimizer_params(
            text_encoder_lr=None,
            unet_lr=None,
            learning_rate=None,
        )
        self.assertEqual(groups, [])
        self.assertEqual(descriptions, [])
        self.assertTrue(
            all(not param.requires_grad for param in unet_lokr.parameters())
        )

        groups, descriptions = network.prepare_optimizer_params(
            text_encoder_lr=None,
            unet_lr=0.5,
            learning_rate=None,
        )
        self.assertEqual(descriptions, ["unet"])
        self.assertEqual(groups[0]["lr"], 0.5)
        self.assertTrue(all(param.requires_grad for param in unet_lokr.parameters()))

    def test_zero_regex_lr_also_freezes_matching_adapter(self):
        lokr = self._lokr("unet_lokr")
        lokr.original_name = "blocks.0.self_attn"
        network = self._network(unet_loras=(lokr,), reg_lrs={r".*self_attn": 0.0})
        network.set_loraplus_lr_ratio(2.0, None, None)

        groups, descriptions = network.prepare_optimizer_params(
            text_encoder_lr=None,
            unet_lr=1.0,
            learning_rate=1.0,
        )

        self.assertEqual(groups, [])
        self.assertEqual(descriptions, [])
        self.assertTrue(all(not param.requires_grad for param in lokr.parameters()))

    def test_optimizer_mask_is_stable_when_prepared_before_registration(self):
        text_lokr = self._lokr("te_lokr")
        unet_lokr = self._lokr("unet_lokr")
        network = self._network(
            text_encoder_loras=(text_lokr,),
            unet_loras=(unet_lokr,),
            register=False,
        )

        groups, descriptions = network.prepare_optimizer_params(
            text_encoder_lr=0.0,
            unet_lr=1.0,
            learning_rate=1.0,
        )
        for adapter in network.loras:
            network.add_module(adapter.lora_name, adapter)
        network.prepare_grad_etc()

        self.assertEqual(descriptions, ["unet"])
        optimizer_ids = {id(param) for group in groups for param in group["params"]}
        self.assertTrue(
            all(id(param) in optimizer_ids for param in unet_lokr.parameters())
        )
        self.assertTrue(all(param.requires_grad for param in unet_lokr.parameters()))
        self.assertTrue(
            all(not param.requires_grad for param in text_lokr.parameters())
        )
        self.assertEqual(
            {id(param) for param in network.get_trainable_params()},
            optimizer_ids,
        )

    def test_registered_adapter_removed_from_active_lists_is_frozen(self):
        text_lokr = self._lokr("te_lokr")
        unet_lokr = self._lokr("unet_lokr")
        network = self._network(
            text_encoder_loras=(text_lokr,), unet_loras=(unet_lokr,)
        )
        network.prepare_optimizer_params(1.0, 1.0, 1.0)
        for param in text_lokr.parameters():
            param.grad = torch.ones_like(param)

        network.text_encoder_loras = []
        network.loras = list(network.unet_loras)
        groups, _ = network.prepare_optimizer_params(0.0, 1.0, 1.0)

        optimizer_ids = {id(param) for group in groups for param in group["params"]}
        self.assertTrue(
            all(id(param) not in optimizer_ids for param in text_lokr.parameters())
        )
        self.assertTrue(
            all(not param.requires_grad for param in text_lokr.parameters())
        )
        self.assertTrue(all(param.grad is None for param in text_lokr.parameters()))
        self.assertTrue(
            all(
                id(param)
                not in {id(trainable) for trainable in network.get_trainable_params()}
                for param in text_lokr.parameters()
            )
        )

    def test_late_added_adapter_is_hidden_until_optimizer_is_reprepared(self):
        unet_lokr = self._lokr("unet_lokr")
        network = self._network(unet_loras=(unet_lokr,))
        groups, _ = network.prepare_optimizer_params(None, 1.0, 1.0)
        optimizer_ids = {id(param) for group in groups for param in group["params"]}

        late_lokr = self._lokr("late_lokr")
        network.unet_loras.append(late_lokr)
        network.loras.append(late_lokr)

        self.assertTrue(all(param.requires_grad for param in late_lokr.parameters()))
        self.assertTrue(
            all(
                id(param)
                not in {id(trainable) for trainable in network.get_trainable_params()}
                for param in late_lokr.parameters()
            )
        )
        self.assertEqual(
            {id(param) for param in network.get_trainable_params()}, optimizer_ids
        )

        network.prepare_grad_etc()
        self.assertTrue(
            all(not param.requires_grad for param in late_lokr.parameters())
        )

    def test_apply_to_rejects_changed_selection_without_mutation(self):
        text_lokr = self._lokr("te_lokr")
        unet_lokr = self._lokr("unet_lokr")
        network = self._network(
            text_encoder_loras=(text_lokr,),
            unet_loras=(unet_lokr,),
            register=False,
        )
        network.weights_sd = None
        network.apply_to(None, None, True, True)

        original_loras = tuple(network.loras)
        original_children = tuple(network._modules.items())
        with self.assertRaisesRegex(RuntimeError, "cannot be changed"):
            network.apply_to(None, None, False, True)

        self.assertEqual(tuple(network.text_encoder_loras), (text_lokr,))
        self.assertEqual(tuple(network.unet_loras), (unet_lokr,))
        self.assertEqual(tuple(network.loras), original_loras)
        self.assertEqual(tuple(network._modules.items()), original_children)
        network.restore()

    def test_apply_to_with_unchanged_selection_is_idempotent(self):
        text_lokr = self._lokr("te_lokr")
        unet_lokr = self._lokr("unet_lokr")
        network = self._network(
            text_encoder_loras=(text_lokr,),
            unet_loras=(unet_lokr,),
            register=False,
        )
        network.weights_sd = None
        network.apply_to(None, None, True, True)

        text_wrappers = tuple(text_lokr.org_module[0]._lycoris_wrappers)
        unet_wrappers = tuple(unet_lokr.org_module[0]._lycoris_wrappers)
        children = tuple(network._modules.items())
        network.apply_to(None, None, True, True)

        self.assertEqual(
            tuple(text_lokr.org_module[0]._lycoris_wrappers), text_wrappers
        )
        self.assertEqual(
            tuple(unet_lokr.org_module[0]._lycoris_wrappers), unet_wrappers
        )
        self.assertEqual(tuple(network._modules.items()), children)
        network.restore()

    def test_apply_to_can_reapply_same_selection_after_restore(self):
        unet_lokr = self._lokr("unet_lokr")
        network = self._network(unet_loras=(unet_lokr,), register=False)
        network.weights_sd = None
        network.apply_to(None, None, False, True)
        network.restore()
        self.assertNotIn(
            unet_lokr, getattr(unet_lokr.org_module[0], "_lycoris_wrappers", [])
        )

        network.apply_to(None, None, False, True)
        self.assertIn(
            unet_lokr, getattr(unet_lokr.org_module[0], "_lycoris_wrappers", [])
        )
        network.restore()


if __name__ == "__main__":
    unittest.main()
