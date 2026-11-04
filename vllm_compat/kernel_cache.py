"""Download compatibility installed lazily inside vLLM workers only."""

import functools
import importlib.abc
import importlib.machinery
import sys


def _configure_downloads(module):
    download = module.download_file
    if getattr(download, "_ttt_download_guard", False):
        return
    # Let other engines wait for a cold-cache download instead of failing
    # at FlashInfer's default 10s HTTP / 30s cache-lock deadlines.
    module.download_file = functools.partial(download, retries=4, timeout=60, lock_timeout=600)
    module.download_file._ttt_download_guard = True


class _FlashInferDownloadGuard(importlib.abc.MetaPathFinder):
    _ttt_download_guard = True

    def find_spec(self, fullname, path=None, target=None):
        if fullname != "flashinfer.jit.cubin_loader":
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is not None and spec.loader is not None:
            execute = spec.loader.exec_module

            def exec_module(module):
                execute(module)
                _configure_downloads(module)

            spec.loader.exec_module = exec_module
        return spec


def install_download_guard():
    """No torch/FlashInfer import, downloads, subprocesses or waiting here."""
    module = sys.modules.get("flashinfer.jit.cubin_loader")
    if module is not None:
        _configure_downloads(module)
    elif not any(getattr(finder, "_ttt_download_guard", False) for finder in sys.meta_path):
        sys.meta_path.insert(0, _FlashInferDownloadGuard())
