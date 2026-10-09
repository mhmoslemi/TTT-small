"""Host-RAM accounting and safe reclamation for long-lived GPU workers.

PyTorch's CUDA host allocator caches freed pinned blocks.  That is normally a
useful optimization, but exact saved-activation CPU offload can leave tens of
GiB cached in every persistent trainer process.  ``torch.cuda.empty_cache``
does not touch this host-side cache.  These helpers release only *unused*
pinned blocks and return ordinary glibc arenas to Linux; live tensors and vLLM
sleep backups are never modified.
"""

from __future__ import annotations

import ctypes
import gc
import os
from pathlib import Path


GIB = 1024 ** 3
DEFAULT_PINNED_CACHE_LIMIT_BYTES = 16 * GIB
DEFAULT_LOW_AVAILABLE_FRACTION = 0.18


def host_memory_info(meminfo_path="/proc/meminfo"):
    """Return Linux total/available memory in bytes, or ``None`` off Linux."""
    try:
        values = {}
        for line in Path(meminfo_path).read_text(encoding="utf-8").splitlines():
            key, raw = line.split(":", 1)
            fields = raw.strip().split()
            if fields:
                values[key] = int(fields[0]) * 1024
        total = int(values["MemTotal"])
        available = int(values.get("MemAvailable", values.get("MemFree", 0)))
        if total > 0 and available >= 0:
            return {"total": total, "available": available}
    except (OSError, KeyError, TypeError, ValueError):
        pass

    try:
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        total = page_size * int(os.sysconf("SC_PHYS_PAGES"))
        available = page_size * int(os.sysconf("SC_AVPHYS_PAGES"))
        if total > 0 and available >= 0:
            return {"total": total, "available": available}
    except (AttributeError, OSError, TypeError, ValueError):
        return None
    return None


def process_rss_bytes(statm_path="/proc/self/statm"):
    """Return this process's resident bytes when Linux exposes ``statm``."""
    try:
        fields = Path(statm_path).read_text(encoding="utf-8").split()
        return int(fields[1]) * int(os.sysconf("SC_PAGE_SIZE"))
    except (OSError, IndexError, TypeError, ValueError):
        return None


def _host_allocator_stats(torch_module):
    cuda = getattr(torch_module, "cuda", None)
    getter = getattr(cuda, "host_memory_stats", None)
    if not callable(getter):
        memory = getattr(cuda, "memory", None)
        getter = getattr(memory, "host_memory_stats", None)
    if not callable(getter):
        return {"owned": 0, "active": 0, "cached": 0}
    try:
        stats = getter()
        owned = int(stats.get("allocated_bytes.current", 0) or 0)
        active = int(stats.get("active_bytes.current", 0) or 0)
        return {
            "owned": owned,
            "active": active,
            "cached": max(0, owned - active),
        }
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return {"owned": 0, "active": 0, "cached": 0}


def host_cache_reclaim_supported(torch_module):
    accelerator = getattr(torch_module, "accelerator", None)
    if callable(getattr(accelerator, "empty_host_cache", None)):
        return True
    torch_c = getattr(torch_module, "_C", None)
    return callable(getattr(torch_c, "_host_emptyCache", None))


def _empty_host_cache(torch_module):
    accelerator = getattr(torch_module, "accelerator", None)
    public_empty = getattr(accelerator, "empty_host_cache", None)
    if callable(public_empty):
        try:
            public_empty()
            return "torch.accelerator.empty_host_cache"
        except (AttributeError, RuntimeError, TypeError):
            pass
    torch_c = getattr(torch_module, "_C", None)
    private_empty = getattr(torch_c, "_host_emptyCache", None)
    if callable(private_empty):
        try:
            private_empty()
            return "torch._C._host_emptyCache"
        except (AttributeError, RuntimeError, TypeError):
            pass
    return None


def _malloc_trim():
    """Return free glibc arenas to Linux; harmlessly no-op elsewhere."""
    try:
        libc = ctypes.CDLL(None)
        trim = getattr(libc, "malloc_trim")
        trim.argtypes = [ctypes.c_size_t]
        trim.restype = ctypes.c_int
        return bool(trim(0))
    except (AttributeError, OSError, TypeError, ValueError):
        return False


def reclaim_unused_host_memory(
        torch_module, *, cuda_devices=(), force=False,
        pinned_cache_limit_bytes=DEFAULT_PINNED_CACHE_LIMIT_BYTES,
        low_available_fraction=DEFAULT_LOW_AVAILABLE_FRACTION):
    """Release inactive pinned blocks and unused libc arenas when warranted.

    Non-forced calls preserve the fast pinned-memory cache while the node is
    healthy.  They reclaim once this process has cached a large activation
    footprint or Linux reports low available RAM.  Forced calls are intended
    for phase boundaries immediately before complete model replicas move from
    GPU to CPU.
    """
    memory_before = host_memory_info()
    allocator_before = _host_allocator_stats(torch_module)
    low_memory = bool(
        memory_before is not None
        and memory_before["available"]
        < float(low_available_fraction) * memory_before["total"]
    )
    oversized_cache = (
        allocator_before["cached"] >= int(pinned_cache_limit_bytes))
    should_reclaim = bool(force or low_memory or oversized_cache)
    result = {
        "attempted": should_reclaim,
        "method": None,
        "low_memory": low_memory,
        "pinned_owned_before": allocator_before["owned"],
        "pinned_active_before": allocator_before["active"],
        "pinned_cached_before": allocator_before["cached"],
        "pinned_cached_after": allocator_before["cached"],
        "rss_before": process_rss_bytes(),
        "rss_after": None,
        "available_before": (
            None if memory_before is None else memory_before["available"]),
        "available_after": (
            None if memory_before is None else memory_before["available"]),
    }
    if not should_reclaim:
        result["rss_after"] = result["rss_before"]
        return result

    gc.collect()
    cuda = getattr(torch_module, "cuda", None)
    synchronize = getattr(cuda, "synchronize", None)
    if callable(synchronize):
        for device in sorted({int(device) for device in cuda_devices
                              if device is not None}):
            try:
                synchronize(device)
            except (RuntimeError, TypeError, ValueError):
                # Emptying the host cache below is itself conservative: blocks
                # with outstanding events remain active and are not released.
                pass
    result["method"] = _empty_host_cache(torch_module)
    gc.collect()
    _malloc_trim()

    allocator_after = _host_allocator_stats(torch_module)
    memory_after = host_memory_info()
    result["pinned_cached_after"] = allocator_after["cached"]
    result["rss_after"] = process_rss_bytes()
    result["available_after"] = (
        None if memory_after is None else memory_after["available"])
    return result


def estimated_module_bytes(model):
    """Estimate bytes needed for one full CPU copy of a model replica."""
    total = 0
    seen = set()
    tensors = list(model.parameters()) + list(model.buffers())
    for tensor in tensors:
        try:
            storage = tensor.untyped_storage()
            identity = (int(storage.data_ptr()), int(storage.nbytes()))
            size = int(storage.nbytes())
        except (AttributeError, RuntimeError, TypeError, ValueError):
            identity = id(tensor)
            try:
                size = int(tensor.numel()) * int(tensor.element_size())
            except (AttributeError, TypeError, ValueError):
                continue
        if identity in seen:
            continue
        seen.add(identity)
        total += max(0, size)
    return total


def estimated_optimizer_bytes(optimizer):
    """Estimate tensor storage moved to CPU with an optimizer."""
    total = 0
    seen = set()
    for state in getattr(optimizer, "state", {}).values():
        for value in state.values():
            if not hasattr(value, "numel"):
                continue
            try:
                storage = value.untyped_storage()
                identity = (int(storage.data_ptr()), int(storage.nbytes()))
                size = int(storage.nbytes())
            except (AttributeError, RuntimeError, TypeError, ValueError):
                identity = id(value)
                try:
                    size = int(value.numel()) * int(value.element_size())
                except (AttributeError, TypeError, ValueError):
                    continue
            if identity in seen:
                continue
            seen.add(identity)
            total += max(0, size)
    return total
