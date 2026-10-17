import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import host_memory
from gen_workers import PhasedVLLMGenerationPool


class _FakeCuda:
    def __init__(self, owned, active=0):
        self.owned = int(owned)
        self.active = int(active)
        self.synchronized = []

    def host_memory_stats(self):
        return {
            "allocated_bytes.current": self.owned,
            "active_bytes.current": self.active,
        }

    def synchronize(self, device):
        self.synchronized.append(int(device))


class _FakeTorchC:
    def __init__(self, cuda):
        self.cuda = cuda
        self.calls = 0

    def _host_emptyCache(self):
        self.calls += 1
        self.cuda.owned = self.cuda.active


class HostMemoryTests(unittest.TestCase):
    def test_linux_meminfo_parser_uses_available_memory(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "meminfo"
            path.write_text(
                "MemTotal:       1000 kB\n"
                "MemFree:         100 kB\n"
                "MemAvailable:    700 kB\n",
                encoding="utf-8",
            )
            self.assertEqual(
                host_memory.host_memory_info(path),
                {"total": 1000 * 1024, "available": 700 * 1024},
            )

    def test_large_inactive_pinned_cache_is_released(self):
        cuda = _FakeCuda(owned=20 * host_memory.GIB, active=2 * host_memory.GIB)
        torch_c = _FakeTorchC(cuda)
        fake_torch = types.SimpleNamespace(cuda=cuda, _C=torch_c)
        healthy = {"total": 100 * host_memory.GIB,
                   "available": 80 * host_memory.GIB}

        with (patch("host_memory.host_memory_info", return_value=healthy),
              patch("host_memory.process_rss_bytes", return_value=123),
              patch("host_memory.gc.collect"),
              patch("host_memory._malloc_trim", return_value=True)):
            result = host_memory.reclaim_unused_host_memory(
                fake_torch, cuda_devices=(3,), force=False)

        self.assertTrue(result["attempted"])
        self.assertEqual(result["pinned_cached_before"], 18 * host_memory.GIB)
        self.assertEqual(result["pinned_cached_after"], 0)
        self.assertEqual(cuda.synchronized, [3])
        self.assertEqual(torch_c.calls, 1)

    def test_healthy_small_cache_keeps_fast_path(self):
        cuda = _FakeCuda(owned=2 * host_memory.GIB)
        torch_c = _FakeTorchC(cuda)
        fake_torch = types.SimpleNamespace(cuda=cuda, _C=torch_c)
        healthy = {"total": 100 * host_memory.GIB,
                   "available": 80 * host_memory.GIB}

        with (patch("host_memory.host_memory_info", return_value=healthy),
              patch("host_memory.process_rss_bytes", return_value=123)):
            result = host_memory.reclaim_unused_host_memory(
                fake_torch, cuda_devices=(0,), force=False)

        self.assertFalse(result["attempted"])
        self.assertEqual(cuda.synchronized, [])
        self.assertEqual(torch_c.calls, 0)

    def test_sleeping_level_one_pool_can_be_discarded_under_pressure(self):
        class Pool:
            def __init__(self):
                self.shutdown_calls = 0

            def shutdown(self):
                self.shutdown_calls += 1

        wrapper = PhasedVLLMGenerationPool.__new__(
            PhasedVLLMGenerationPool)
        pool = Pool()
        wrapper._awake = False
        wrapper._pool = pool
        wrapper._persistent = True
        wrapper._sleep_level = 1
        wrapper._pool_kwargs = {"num_workers": 8}
        wrapper.num_workers = 8

        self.assertTrue(wrapper.discard_sleeping_host_residency())
        self.assertIsNone(wrapper._pool)
        self.assertFalse(wrapper._persistent)
        self.assertEqual(pool.shutdown_calls, 1)


if __name__ == "__main__":
    unittest.main()
