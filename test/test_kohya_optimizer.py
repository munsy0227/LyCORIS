import unittest

import torch
from torch import nn

from lycoris.kohya import LycorisNetworkKohya
from lycoris.modules import LoConModule, LohaModule, LokrModule


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
