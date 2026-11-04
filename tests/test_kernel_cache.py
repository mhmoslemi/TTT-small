"""Regression contracts for the lazy worker-side download fix."""

from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from vllm_compat.kernel_cache import _configure_downloads


class KernelCacheTests(unittest.TestCase):
    def test_configuration_does_not_download_and_preserves_verification(self):
        download = Mock(return_value=True)
        module = SimpleNamespace(download_file=download)
        _configure_downloads(module)
        guarded = module.download_file
        _configure_downloads(module)
        self.assertIs(module.download_file, guarded)
        download.assert_not_called()
        module.download_file("remote", "local", expected_sha256="pinned-hash")
        download.assert_called_once_with("remote", "local", retries=4, timeout=60,
                                         lock_timeout=600, expected_sha256="pinned-hash")

    def test_trainer_has_no_added_kernel_preflight(self):
        root = Path(__file__).resolve().parents[1]
        training = (root / "train_multy_CVaR.py").read_text()
        self.assertNotIn("prepare_kernel_cache", training)
        self.assertNotIn("cache_attention_artifacts", training)


if __name__ == "__main__":
    unittest.main()
