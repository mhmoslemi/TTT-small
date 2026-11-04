"""Bounded, unlinked disk storage for inactive model replicas."""

import ctypes
import errno
import mmap
import os
from pathlib import Path
import shutil
import tempfile


TRANSFER_BYTES = 64 * 1024**2


class DiskOffloadBuffer:
    """One file per sleeping rank; closing it immediately returns its space."""

    def __init__(self, directory, size):
        directory = Path(directory).resolve()
        directory.mkdir(parents=True, exist_ok=True)
        # /tmp is often tmpfs: spilling there would still consume host RAM.
        mounts = Path("/proc/mounts")
        if mounts.exists():
            matching = []
            for line in mounts.read_text().splitlines():
                fields = line.split()
                if len(fields) >= 3:
                    mount = Path(fields[1].replace("\\040", " "))
                    if directory == mount or mount in directory.parents:
                        matching.append((len(str(mount)), fields[2]))
            if matching and max(matching)[1] in {"tmpfs", "ramfs"}:
                raise RuntimeError(f"model offload requires disk, not tmpfs: {directory}")
        usage = shutil.disk_usage(directory)
        reserve = min(16 * 1024**3, max(1024**3, usage.total // 100))
        if size <= 0 or usage.free < size + reserve:
            raise OSError(errno.ENOSPC, f"insufficient disk space for {size} offload bytes in {directory}")
        self.size = int(size)
        self.mapping = None
        # TemporaryFile unlinks on creation on Linux: even SIGKILL cannot
        # strand hundreds of GB of named per-step model backups.
        self.file = tempfile.TemporaryFile(prefix="model-offload-", dir=directory)
        try:
            if hasattr(os, "posix_fallocate"):
                try:
                    os.posix_fallocate(self.file.fileno(), 0, self.size)
                except OSError as exc:
                    if exc.errno not in {errno.EOPNOTSUPP, errno.ENOSYS, errno.EINVAL}:
                        raise
                    self.file.truncate(self.size)
            else:
                self.file.truncate(self.size)
            self.mapping = mmap.mmap(self.file.fileno(), self.size)
            self.address = ctypes.addressof(ctypes.c_char.from_buffer(self.mapping))
        except BaseException:
            self.file.close()
            raise

    def contains(self, address, size):
        return self.address <= address and address + size <= self.address + self.size

    def copy(self, destination, source, size, copier=ctypes.memmove):
        """Stream transfers and evict mapped pages instead of growing RSS."""
        writing = self.contains(destination, size)
        reading = self.contains(source, size)
        if not writing and not reading:
            return copier(destination, source, size)
        for start in range(0, size, TRANSFER_BYTES):
            count = min(TRANSFER_BYTES, size - start)
            copier(destination + start, source + start, count)
            offset = (destination if writing else source) + start - self.address
            aligned = offset // mmap.PAGESIZE * mmap.PAGESIZE
            end = min(self.size, ((offset + count + mmap.PAGESIZE - 1) // mmap.PAGESIZE) * mmap.PAGESIZE)
            if writing:
                self.mapping.flush(aligned, end - aligned)
            if hasattr(self.mapping, "madvise"):
                self.mapping.madvise(mmap.MADV_DONTNEED, aligned, end - aligned)
            if hasattr(os, "posix_fadvise"):
                os.posix_fadvise(self.file.fileno(), aligned, end - aligned, os.POSIX_FADV_DONTNEED)

    def close(self):
        if self.mapping is not None:
            self.mapping.close()
            self.mapping = None
        self.file.close()


class VLLMDiskOffload:
    """Preserve vLLM's loaded/converted bytes, not a second checkpoint reload."""

    def __init__(self, allocator, cumem, torch, directory):
        self.allocator, self.cumem, self.torch = allocator, cumem, torch
        self.directory = directory
        self.buffer = None
        self.next_sleep_on_disk = False
        self.normal_sleep, self.normal_wake = allocator.sleep, allocator.wake_up
        allocator.sleep, allocator.wake_up = self.sleep, self.wake_up

    def _backup(self, items):
        if self.buffer is not None:
            raise RuntimeError("a replica already has an outstanding disk backup")
        size = sum(int(data.handle[1]) for _, data in items)
        if not size:
            return
        self.buffer = DiskOffloadBuffer(self.directory, size)
        offset = 0
        for pointer, data in items:
            count = int(data.handle[1])
            tensor = self.torch.frombuffer(self.buffer.mapping, dtype=self.torch.uint8,
                                          count=count, offset=offset)
            old_backup = data.cpu_backup_tensor
            source = old_backup.data_ptr() if old_backup is not None else pointer
            copier = ctypes.memmove if old_backup is not None else self.cumem.libcudart.cudaMemcpy
            self.buffer.copy(tensor.data_ptr(), source, count, copier)
            data.cpu_backup_tensor = tensor
            if old_backup is not None and hasattr(self.cumem, "release_cpu_backup"):
                self.cumem.release_cpu_backup(old_backup)
            del old_backup, tensor
            offset += count

    def spill_sleeping(self):
        if self.buffer is not None:
            return 0
        items = [(ptr, data) for ptr, data in self.allocator.pointer_to_data.items()
                 if data.cpu_backup_tensor is not None]
        self._backup(items)
        return sum(int(data.handle[1]) for _, data in items)

    def sleep(self, offload_tags=None):
        if not self.next_sleep_on_disk:
            return self.normal_sleep(offload_tags)
        tags = ((self.allocator.default_tag,) if offload_tags is None else
                (offload_tags,) if isinstance(offload_tags, str) else offload_tags)
        self.torch.cuda.synchronize()
        items = list(self.allocator.pointer_to_data.items())
        self._backup([(ptr, data) for ptr, data in items
                      if data.tag in tags and not getattr(data, "is_asleep", False)])
        for _, data in items:
            if not getattr(data, "is_asleep", False):
                self.cumem.unmap_and_release(data.handle)
                if hasattr(data, "is_asleep"):
                    data.is_asleep = True
        self.next_sleep_on_disk = False
        self.torch.cuda.empty_cache()

    def wake_up(self, tags=None):
        if self.buffer is None:
            return self.normal_wake(tags)
        original_copy = self.cumem.libcudart.cudaMemcpy
        try:
            self.cumem.libcudart.cudaMemcpy = lambda dst, src, count: self.buffer.copy(dst, src, count, original_copy)
            result = self.normal_wake(tags)
        finally:
            self.cumem.libcudart.cudaMemcpy = original_copy
        if not any(data.cpu_backup_tensor is not None for data in self.allocator.pointer_to_data.values()):
            self.buffer.close()
            self.buffer = None
        return result
