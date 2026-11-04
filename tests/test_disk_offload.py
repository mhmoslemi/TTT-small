"""Exact byte transfers and bounded lifecycle, without torch or CUDA."""

import ctypes
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import runtime.disk_offload as disk
import runtime.host_memory as host
from generation.gen_workers import GenerationPool


class FakeTensor:
    def __init__(self, owner, address, count):
        self.owner, self.address, self.count = owner, address, count

    def data_ptr(self):
        return self.address


class DiskOffloadTests(unittest.TestCase):
    def make_replica(self, directory, cpu_backup=False):
        payload = bytes(range(256)) * 40
        gpu = ctypes.create_string_buffer(payload, len(payload))
        pointer = ctypes.addressof(gpu)
        data = SimpleNamespace(tag="weights", handle=(0, len(payload), pointer),
                               cpu_backup_tensor=None, is_asleep=False)
        copies, normal = [], []

        def memcpy(dst, src, count):
            copies.append(count)
            ctypes.memmove(dst, src, count)

        cumem = SimpleNamespace(libcudart=SimpleNamespace(cudaMemcpy=memcpy),
                                unmap_and_release=lambda handle: ctypes.memset(handle[2], 0, handle[1]))
        allocator = SimpleNamespace(default_tag="default", pointer_to_data={pointer: data})

        def wake(tags=None):
            normal.append("wake")
            if tags is None or data.tag in tags:
                if data.cpu_backup_tensor is not None:
                    cumem.libcudart.cudaMemcpy(pointer, data.cpu_backup_tensor.data_ptr(), len(payload))
                    data.cpu_backup_tensor = None
                data.is_asleep = False

        allocator.sleep = lambda tags=None: normal.append("sleep")
        allocator.wake_up = wake

        def frombuffer(mapping, **kwargs):
            address = ctypes.addressof(ctypes.c_char.from_buffer(mapping)) + kwargs["offset"]
            return FakeTensor(mapping, address, kwargs["count"])

        torch = SimpleNamespace(uint8="uint8", frombuffer=frombuffer,
                                cuda=SimpleNamespace(synchronize=lambda: None, empty_cache=lambda: None))
        if cpu_backup:
            backup = ctypes.create_string_buffer(payload, len(payload))
            data.cpu_backup_tensor = FakeTensor(backup, ctypes.addressof(backup), len(payload))
            data.is_asleep = True
            ctypes.memset(pointer, 0, len(payload))
        storage = disk.VLLMDiskOffload(allocator, cumem, torch, directory)
        return storage, allocator, data, gpu, payload, copies, normal

    def test_direct_disk_sleep_exact_restore_and_repeat_without_accumulation(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(disk, "TRANSFER_BYTES", 4096):
            storage, allocator, data, gpu, payload, copies, normal = self.make_replica(folder)
            for _ in range(2):
                storage.next_sleep_on_disk = True
                allocator.sleep(("weights",))
                backup = storage.buffer
                self.assertEqual(bytes(gpu), b"\0" * len(payload))
                self.assertTrue(data.is_asleep)
                self.assertEqual(list(Path(folder).iterdir()), [])  # already unlinked
                allocator.wake_up()
                self.assertEqual(bytes(gpu), payload)
                self.assertTrue(backup.file.closed)
                self.assertIsNone(storage.buffer)
                self.assertIsNone(data.cpu_backup_tensor)
            self.assertEqual(max(copies), 4096)
            self.assertEqual(normal, ["wake", "wake"])

    def test_spill_existing_backup_once_and_keep_it_until_selected_wake(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(disk, "TRANSFER_BYTES", 4096):
            storage, allocator, data, gpu, payload, _, _ = self.make_replica(folder, cpu_backup=True)
            self.assertEqual(storage.spill_sleeping(), len(payload))
            backup = storage.buffer
            self.assertEqual(storage.spill_sleeping(), 0)
            self.assertIs(storage.buffer, backup)
            allocator.wake_up(tags=["kv_cache"])
            self.assertIs(storage.buffer, backup)
            allocator.wake_up(tags=["weights"])
            self.assertEqual(bytes(gpu), payload)
            self.assertTrue(backup.file.closed)

    def test_ram_fast_path_creates_no_file(self):
        with tempfile.TemporaryDirectory() as folder:
            storage, allocator, _, _, _, _, normal = self.make_replica(folder)
            allocator.sleep(("weights",))
            allocator.wake_up()
            self.assertIsNone(storage.buffer)
            self.assertEqual(normal, ["sleep", "wake"])
            self.assertEqual(list(Path(folder).iterdir()), [])

    def test_failed_restore_retains_exact_backup(self):
        with tempfile.TemporaryDirectory() as folder:
            storage, allocator, data, gpu, payload, _, _ = self.make_replica(folder, cpu_backup=True)
            storage.spill_sleeping()
            backup = storage.buffer
            original = storage.cumem.libcudart.cudaMemcpy
            with patch.object(storage.cumem.libcudart, "cudaMemcpy", side_effect=RuntimeError("copy failed")):
                with self.assertRaisesRegex(RuntimeError, "copy failed"):
                    allocator.wake_up()
            self.assertIs(storage.cumem.libcudart.cudaMemcpy, original)
            self.assertFalse(backup.file.closed)
            self.assertIsNotNone(data.cpu_backup_tensor)
            allocator.wake_up()
            self.assertEqual(bytes(gpu), payload)

    def test_no_space_and_tmpfs_are_rejected_before_backup(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(disk.shutil, "disk_usage", return_value=SimpleNamespace(total=10**12, free=0)):
                with self.assertRaises(OSError):
                    disk.DiskOffloadBuffer(folder, 8192)
            with patch.object(Path, "exists", return_value=True), patch.object(Path, "read_text", return_value="none / tmpfs rw 0 0\n"):
                with self.assertRaisesRegex(RuntimeError, "tmpfs"):
                    disk.DiskOffloadBuffer(folder, 8192)
            self.assertEqual(list(Path(folder).iterdir()), [])

    def test_low_ram_sleeps_on_disk_without_removing_engines(self):
        import multiprocessing
        with patch.object(host, "host_memory_info", return_value={"total": 200 * host.GIB, "available": 30 * host.GIB}):
            gate = host.HostMemoryAdmission(multiprocessing.get_context("spawn"))
            pool = object.__new__(GenerationPool)
            pool._persistent_worker_ids, pool._transient_worker_ids = (0, 1), ()
            pool._gpu_groups = [[0], [1]]
            pool.num_workers, pool.sleep_level = 2, 1
            pool._stop_workers = lambda ranks: self.assertEqual(ranks, ())
            events = []

            def control(command, ranks):
                events.append((command, tuple(ranks)))
                if command == "host_memory":
                    return {rank: {"backup_bytes": 70 * host.GIB} for rank in ranks}
                self.assertEqual(command, "sleep_disk")

            pool._vllm_control = control
            restored = []
            pool.sleep(host_admission=gate, after_sleep=restored.extend)
            self.assertEqual(pool._persistent_worker_ids, (0, 1))
            self.assertEqual(restored, [0, 1])
            self.assertEqual(gate.reserved_bytes, 0)


if __name__ == "__main__":
    unittest.main()
