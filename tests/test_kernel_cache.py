"""Kernel preflight reuses installed version pins and fails before GPU loading."""

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from vllm_compat.kernel_cache import cache_attention_artifacts, prepare_kernel_cache


class KernelCacheTests(unittest.TestCase):
    def modules(self, bad=False):
        fetch = Mock(side_effect=lambda name, digest: (b"abc ../escape\n" if bad else
                     b"a" * 64 + b"  include/meta.h\n" + b"b" * 64 + b"  kernel.cubin\n")
                     if name.endswith("checksums.txt") else b"verified")
        loader = SimpleNamespace(download_file=Mock(), get_artifact=fetch)
        artifacts = SimpleNamespace(ArtifactPath=SimpleNamespace(TRTLLM_GEN_FMHA="installed-pin/fmha/"),
                                    CheckSumHash=SimpleNamespace(TRTLLM_GEN_FMHA="pinned-manifest-hash"))
        return fetch, {
            "flashinfer": SimpleNamespace(artifacts=artifacts),
            "flashinfer.jit": SimpleNamespace(cubin_loader=loader),
        }

    def test_fetches_only_installed_attention_pin_with_hash_verification(self):
        fetch, modules = self.modules()
        with patch.dict("sys.modules", modules):
            cache_attention_artifacts()
        self.assertEqual(fetch.call_args_list[0].args, ("installed-pin/fmha/checksums.txt", "pinned-manifest-hash"))
        self.assertEqual({call.args for call in fetch.call_args_list[1:]}, {
            ("installed-pin/fmha/include/meta.h", "a" * 64),
            ("installed-pin/fmha/kernel.cubin", "b" * 64)})

    def test_rejects_unsafe_manifest(self):
        _, modules = self.modules(bad=True)
        with patch.dict("sys.modules", modules), self.assertRaisesRegex(ValueError, "manifest"):
            cache_attention_artifacts()

    def test_preflight_is_isolated_and_detects_b200_without_yaml_dependency(self):
        import subprocess
        with tempfile.TemporaryDirectory() as folder, patch("runtime.gpu_runtime.query_gpu_memory", return_value={0: SimpleNamespace(name="NVIDIA B200")}):
            with patch("vllm_compat.kernel_cache.subprocess.run") as run:
                prepare_kernel_cache([0], Path(folder) / "dependencies.log")
                self.assertEqual(run.call_args.kwargs["env"]["CUDA_VISIBLE_DEVICES"], "")
                self.assertEqual(run.call_args.args[0][1:], ["-m", "vllm_compat.kernel_cache"])
                run.side_effect = subprocess.TimeoutExpired("cache", 900)
                with self.assertRaisesRegex(RuntimeError, "before model loading"):
                    prepare_kernel_cache([0], Path(folder) / "dependencies.log")
            with patch("runtime.gpu_runtime.query_gpu_memory", return_value={0: SimpleNamespace(name="RTX6000")}), patch("vllm_compat.kernel_cache.subprocess.run") as run:
                prepare_kernel_cache([0], Path(folder) / "dependencies.log")
                run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
