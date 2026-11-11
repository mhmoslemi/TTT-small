"""Copy plain trainer tensors into real CPU pages as RAM becomes available.

An anonymous mapping supplies each destination's contiguous virtual address
range. Its physical pages are populated only inside admitted copy chunks.
Completed storages replace the original GPU storage without replacing any
Parameter objects, including those owned by the optimizer.
"""

import mmap
import time

from runtime.host_memory import GIB, RAM_WAIT_TIMEOUT_S


COPY_CHUNK_BYTES = 512 * 1024**2
MIN_COPY_BYTES = 64 * 1024**2


def _model_tensors(model):
    for parameter in model.parameters():
        yield parameter
        if parameter.grad is not None:
            yield parameter.grad
    yield from model.buffers()


def can_incrementally_offload(model, cfg, disk_directory=None):
    """Custom packed/quantized Parameters retain their backend's move path."""
    import torch

    if disk_directory or getattr(cfg, "hardware_profile", "") not in ("B200", "B200_full_ram"):
        return False
    return all(type(tensor) in (torch.Tensor, torch.nn.Parameter)
               and tensor.layout == torch.strided and not tensor.is_quantized
               and not tensor.is_conj() and not tensor.is_neg()
               and tensor.device.type in ("cpu", "cuda")
               for tensor in _model_tensors(model))


def _copy_tensors_to_cpu(tensors, admission, *, stage, deadline):
    """Preserve object identity, shared storage, offsets, dtypes, and strides."""
    import torch

    groups, seen = {}, set()
    for tensor in tensors:
        if id(tensor) in seen:
            continue
        seen.add(id(tensor))
        storage = tensor.untyped_storage()
        if not storage.nbytes():
            tensor.data = tensor.detach().cpu()
            continue
        key = (tensor.device, storage.data_ptr(), storage.nbytes())
        groups.setdefault(key, []).append(
            (tensor, tensor.storage_offset(), tuple(tensor.shape), tuple(tensor.stride())))
    total = sum(key[2] for key in groups)
    copied, next_report = 0, 0
    if total:
        print(f"[ram-offload] {stage}: copying {total / GIB:.1f} GiB incrementally; "
              "up to 512 MiB per reservation", flush=True)
    with torch.no_grad():
        for (device, _, size), aliases in groups.items():
            source = torch.empty(0, dtype=torch.uint8, device=device).set_(
                aliases[0][0].untyped_storage(), 0, (size,), (1,))
            # No full-size allocation is touched or zero-filled up front.
            mapping = mmap.mmap(-1, size, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
            destination = None
            try:
                destination = torch.frombuffer(mapping, dtype=torch.uint8)
                offset = 0
                while offset < size:
                    requested = min(COPY_CHUNK_BYTES, size - offset)
                    label = f"{stage} ({copied / GIB:.1f}/{total / GIB:.1f} GiB copied)"
                    with admission.reservation(
                            requested, minimum_bytes=min(MIN_COPY_BYTES, requested),
                            priority=True, stage=label, deadline=deadline) as amount:
                        # Blocking copy: the destination pages are real resident
                        # data before the temporary admission charge is released.
                        destination[offset:offset + amount].copy_(
                            source[offset:offset + amount], non_blocking=False)
                    offset += amount
                    copied += amount
                cpu_storage = destination.untyped_storage()
                views = [torch.empty(0, dtype=tensor.dtype, device="cpu").set_(
                    cpu_storage, start, shape, stride)
                    for tensor, start, shape, stride in aliases]
            except BaseException:
                # No alias was published. Drop only this partial destination;
                # its intact GPU storage and earlier completed tensors survive.
                destination = None
                mapping = None
                raise
            for (tensor, _, _, _), view in zip(aliases, views):
                tensor.data = view
            # frombuffer's storage owns the mapping until its last tensor dies.
            del views, cpu_storage, destination, mapping, source
            if copied >= next_report or copied == total:
                print(f"[ram-offload] {stage}: {copied / GIB:.1f}/{total / GIB:.1f} GiB copied", flush=True)
                next_report = copied + 8 * GIB


def offload_model(model, admission, *, deadline=None):
    if deadline is None:
        deadline = time.monotonic() + RAM_WAIT_TIMEOUT_S
    devices = sorted({str(t.device) for t in _model_tensors(model) if t.device.type != "cpu"})
    tensors = (t for t in _model_tensors(model) if t.device.type != "cpu")
    _copy_tensors_to_cpu(tensors, admission,
                         stage=f"trainer {','.join(devices)} CPU offload", deadline=deadline)


def offload_optimizer(optimizer, admission, *, deadline=None):
    import torch

    def tensors(value):
        if torch.is_tensor(value):
            if value.device.type != "cpu":
                yield value
        elif isinstance(value, dict):
            for child in value.values():
                yield from tensors(child)
        elif isinstance(value, (tuple, list)):
            for child in value:
                yield from tensors(child)

    if deadline is None:
        deadline = time.monotonic() + RAM_WAIT_TIMEOUT_S
    _copy_tensors_to_cpu(tensors(optimizer.state), admission,
                         stage="optimizer CPU offload", deadline=deadline)
