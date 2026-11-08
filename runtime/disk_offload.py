"""Bounded disk parking for ordinary frozen trainer tensors."""

import os
from pathlib import Path
import shutil
import tempfile

import torch


CHUNK_BYTES = 64 * 1024**2


class DiskWeightOffload:
    """Keep Parameter objects alive while their frozen storage lives on disk."""

    def __init__(self, model, directory):
        tensors = list(dict.fromkeys(
            tensor for tensor in (*model.parameters(), *model.buffers())
            if not tensor.requires_grad))
        # Quantized tensor subclasses carry additional storage/placement state.
        if any(type(tensor) not in (torch.Tensor, torch.nn.Parameter)
               or tensor.layout != torch.strided or not tensor.is_contiguous()
               or tensor.device.type == "meta" or tensor.grad is not None
               for tensor in tensors):
            raise ValueError("disk trainer offload requires contiguous, ordinary frozen tensors")
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        required = sum(t.numel() * t.element_size() for t in tensors)
        if shutil.disk_usage(directory).free < required + CHUNK_BYTES:
            raise OSError(f"insufficient disk space for {required / 1024**3:.1f} GiB of trainer weights in {directory}")
        print(f"[offload] streaming {required / 1024**3:.1f} GiB of frozen trainer weights to {directory}", flush=True)
        self.file = tempfile.TemporaryFile(dir=directory, prefix="trainer-")
        self.entries = []
        try:
            for tensor in tensors:
                offset = self.file.tell()
                shape, device = tensor.shape, tensor.device
                flat = tensor.detach().view(-1)
                width = max(1, CHUNK_BYTES // tensor.element_size())
                for start in range(0, flat.numel(), width):
                    chunk = flat[start:start + width].cpu().view(torch.uint8).numpy()
                    self.file.write(memoryview(chunk))
                    self._release_file_cache()
                    del chunk
                del flat
                self.entries.append((tensor, shape, device, offset))
                tensor.data = torch.empty(0, dtype=tensor.dtype)
        except BaseException:
            self.restore()
            raise

    def _release_file_cache(self):
        self.file.flush()
        os.fsync(self.file.fileno())
        if hasattr(os, "posix_fadvise"):
            os.posix_fadvise(self.file.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)

    def restore(self):
        if not self.entries:
            self.file.close()
            return
        print("[offload] reloading frozen trainer weights from disk", flush=True)
        buffer = bytearray(CHUNK_BYTES)
        while self.entries:
            tensor, shape, device, offset = self.entries[-1]
            restored = torch.empty(shape, dtype=tensor.dtype, device=device)
            flat = restored.view(-1)
            width = max(1, CHUNK_BYTES // tensor.element_size())
            self.file.seek(offset)
            for start in range(0, flat.numel(), width):
                count = min(width, flat.numel() - start)
                size = count * tensor.element_size()
                if self.file.readinto(memoryview(buffer)[:size]) != size:
                    raise OSError("truncated trainer offload file; refusing incomplete weights")
                flat[start:start + count].copy_(torch.frombuffer(
                    buffer, dtype=tensor.dtype, count=count))
                self._release_file_cache()
            tensor.data = restored
            self.entries.pop()
            del flat, restored
        self.file.close()
