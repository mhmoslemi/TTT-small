"""Bounded disk parking for ordinary frozen trainer tensors."""

import os
from pathlib import Path
import shutil
import tempfile

import torch


CHUNK_BYTES = 64 * 1024**2


class FrozenWeightSnapshot:
    """B200: write immutable parameters once, reuse the file at every handoff."""

    def __init__(self, model, directory):
        self.model = model
        self.entries, self.parked, self.runtime = [], set(), []
        self.runtime_devices = {name: tensor.device for name, tensor in
                                (*model.named_parameters(), *model.named_buffers())}
        seen = set()
        tensors = []
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                continue
            scales = getattr(parameter, "mxfp4_scales", None)
            if type(parameter) not in (torch.Tensor, torch.nn.Parameter) and not (
                    parameter.dtype == torch.uint8 and isinstance(scales, torch.Tensor)):
                raise ValueError("B200 disk residency supports ordinary or packed MXFP4 parameters")
            tensors.append((name, parameter))
            if scales is not None:
                tensors.append((name + ".mxfp4_scales", scales))
        for name, tensor in tensors:
            if id(tensor) in seen:
                continue
            if not tensor.is_contiguous() or tensor.device.type == "meta" or tensor.grad is not None:
                raise ValueError(f"cannot snapshot frozen tensor {name}")
            seen.add(id(tensor))
            self.entries.append((name, tensor, tuple(tensor.shape), tensor.device, 0))
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        required = sum(t.numel() * t.element_size() for _, t, *_ in self.entries)
        if shutil.disk_usage(directory).free < required + CHUNK_BYTES:
            raise OSError(f"B200 frozen snapshot needs {required / 1024**3:.1f} GiB free in {directory}")
        self.file = tempfile.TemporaryFile(dir=directory, prefix="b200-frozen-", buffering=0)
        try:
            for index, (name, tensor, shape, device, _) in enumerate(self.entries):
                offset = self.file.tell()
                self.entries[index] = (name, tensor, shape, device, offset)
                flat = tensor.detach().view(-1)
                width = max(1, CHUNK_BYTES // tensor.element_size())
                for start in range(0, flat.numel(), width):
                    chunk = flat[start:start + width].cpu().view(torch.uint8).numpy()
                    if self.file.write(memoryview(chunk)) != chunk.nbytes:
                        raise OSError("incomplete frozen snapshot write")
                    self._drop_cache(offset + start * tensor.element_size(), chunk.nbytes, written=True)
                    del chunk
            print(f"[offload] B200 reusable frozen snapshot: {required / 1024**3:.1f} GiB; "
                  "base weights will not be rewritten each step", flush=True)
        except BaseException:
            self.file.close()
            raise

    def _drop_cache(self, offset, size, *, written=False):
        if written:
            os.fsync(self.file.fileno())
        if hasattr(os, "posix_fadvise"):
            os.posix_fadvise(self.file.fileno(), offset, size, os.POSIX_FADV_DONTNEED)

    @torch.no_grad()
    def park(self):
        for index, (name, tensor, shape, _, _) in enumerate(self.entries):
            if index in self.parked:
                continue
            if tuple(tensor.shape) != shape or tensor.requires_grad:
                raise RuntimeError(f"frozen parameter contract changed: {name}")
        for index, (_, tensor, _, _, _) in enumerate(self.entries):
            if index in self.parked:
                continue
            tensor.data = torch.empty(0, dtype=tensor.dtype)
            self.parked.add(index)
        # Buffers can change during training; preserve their current values.
        frozen_ids = {id(row[1]) for row in self.entries}
        self.runtime = [(tensor, self.runtime_devices.setdefault(name, tensor.device)) for name, tensor in
                        (*self.model.named_parameters(), *self.model.named_buffers())
                        if id(tensor) not in frozen_ids]
        for tensor, _ in self.runtime:
            tensor.data = tensor.data.cpu()
            if tensor.grad is not None:
                tensor.grad = tensor.grad.cpu()

    @torch.no_grad()
    def restore(self):
        if not self.parked and not self.runtime:
            return
        buffer = bytearray(CHUNK_BYTES)
        for index in sorted(self.parked):
            name, tensor, shape, device, offset = self.entries[index]
            restored = torch.empty(shape, dtype=tensor.dtype, device=device)
            flat = restored.view(-1)
            width = max(1, CHUNK_BYTES // tensor.element_size())
            self.file.seek(offset)
            for start in range(0, flat.numel(), width):
                count = min(width, flat.numel() - start)
                size = count * tensor.element_size()
                if self.file.readinto(memoryview(buffer)[:size]) != size:
                    raise OSError(f"truncated B200 frozen snapshot: {name}")
                flat[start:start + count].copy_(torch.frombuffer(buffer, dtype=tensor.dtype, count=count))
                self._drop_cache(offset + start * tensor.element_size(), size)
            tensor.data = restored
            self.parked.remove(index)
            del flat, restored
        for tensor, device in self.runtime:
            tensor.data = tensor.data.to(device)
            if tensor.grad is not None:
                tensor.grad = tensor.grad.to(device)
        self.runtime = []


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
