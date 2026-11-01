"""Marlin API compatibility checks without importing CUDA or model packages."""

from contextlib import nullcontext
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch, sentinel

from vllm_compat import gpt_oss_repack


class MarlinCompatibilityTests(unittest.TestCase):
    def check_api(self, original, args, kwargs):
        marlin = SimpleNamespace(_repack_marlin_experts=original)
        module = SimpleNamespace(marlin_utils_fp4=marlin)
        with patch.dict(sys.modules, {"vllm.model_executor.layers.quantization.utils": module}):
            with patch.object(gpt_oss_repack, "repack_experts_in_place", return_value=sentinel.packed) as repack:
                with gpt_oss_repack.bounded_marlin_repacking():
                    result = marlin._repack_marlin_experts(sentinel.weight, *args, **kwargs)
                    self.assertIs(result, sentinel.packed)
                    repack.assert_called_once_with(sentinel.weight, *args, repack=original, **kwargs)
                self.assertIs(marlin._repack_marlin_experts, original)

    def test_current_four_argument_api(self):
        def original(weight, size_n, size_k, is_a_8bit):
            pass

        self.check_api(original, (256, 128, False), {})
        self.check_api(original, (), dict(size_n=256, size_k=128, is_a_8bit=True))

    def test_legacy_five_argument_api(self):
        def original(weight, size_n, size_k, perm, is_a_8bit):
            pass

        self.check_api(original, (256, 128, sentinel.perm, False), {})
        self.check_api(original, (), dict(size_n=256, size_k=128, perm=sentinel.perm, is_a_8bit=True))

    def test_repacking_forwards_arguments_one_expert_at_a_time(self):
        for arguments in ((256, 128, False), (256, 128, sentinel.perm, False)):
            with self.subTest(arguments=arguments):
                weight = MagicMock(ndim=3, shape=(2, 16, 16), requires_grad=False, device="cuda")
                weight.element_size.return_value = 1
                weight.__getitem__.return_value.numel.return_value = 256
                converted = MagicMock(shape=(1, 4, 16), device="cuda")
                converted.numel.return_value = 64
                converted.element_size.return_value = 4
                packing = MagicMock(return_value=converted)
                with patch.dict(sys.modules, {"torch": SimpleNamespace(no_grad=nullcontext)}):
                    result = gpt_oss_repack.repack_experts_in_place(weight, *arguments, repack=packing)
                self.assertEqual(packing.call_count, 2)
                for call in packing.call_args_list:
                    self.assertEqual(call.args[1:], arguments)
                self.assertEqual(weight.__getitem__.call_args_list[0].args, (slice(0, 1),))
                self.assertIs(result, weight.view.return_value.view.return_value)
                self.assertEqual(result.__getitem__.return_value.copy_.call_count, 2)

    def test_restores_original_helper_after_load_failure(self):
        def original(weight, size_n, size_k, is_a_8bit):
            pass

        marlin = SimpleNamespace(_repack_marlin_experts=original)
        module = SimpleNamespace(marlin_utils_fp4=marlin)
        with patch.dict(sys.modules, {"vllm.model_executor.layers.quantization.utils": module}):
            with self.assertRaisesRegex(RuntimeError, "load failed"):
                with gpt_oss_repack.bounded_marlin_repacking():
                    raise RuntimeError("load failed")
        self.assertIs(marlin._repack_marlin_experts, original)

    def test_unknown_required_parameters_fail_before_patching(self):
        def original(weight, size_n, size_k, is_a_8bit, unknown):
            pass

        marlin = SimpleNamespace(_repack_marlin_experts=original)
        module = SimpleNamespace(marlin_utils_fp4=marlin)
        with patch.dict(sys.modules, {"vllm.model_executor.layers.quantization.utils": module}):
            with self.assertRaisesRegex(RuntimeError, "Unsupported vLLM"):
                with gpt_oss_repack.bounded_marlin_repacking():
                    self.fail("unsupported helper was installed")
        self.assertIs(marlin._repack_marlin_experts, original)


if __name__ == "__main__":
    unittest.main()
