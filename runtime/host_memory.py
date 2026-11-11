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
import errno
import gc
import math
import os
from pathlib import Path
import threading
import time
from contextlib import contextmanager


GIB = 1024 ** 3
RAM_RETRY_INTERVAL_S = 60.0
RAM_POLL_INTERVAL_S = 1.0
RAM_WAIT_TIMEOUT_S = 2 * 60 * 60.0
DEFAULT_PINNED_CACHE_LIMIT_BYTES = 16 * GIB
DEFAULT_LOW_AVAILABLE_FRACTION = 0.18
DEFAULT_ACTIVATION_OFFLOAD_SLOT_BYTES = 384 * GIB
DEFAULT_ACTIVATION_OFFLOAD_RESERVE_FRACTION = 0.35
DEFAULT_ACTIVATION_OFFLOAD_FREE_RESERVE_BYTES = 128 * GIB
DEFAULT_ACTIVATION_OFFLOAD_MIN_REQUEST_BYTES = 64 * GIB
DEFAULT_ACTIVATION_OFFLOAD_UNIT_BYTES = 8 * GIB
DEFAULT_ACTIVATION_OFFLOAD_CALIBRATION_SAMPLES = 2
DEFAULT_ACTIVATION_OFFLOAD_CALIBRATION_MARGIN = 1.25


def validate_memory_handoff_runtime(training_file):
    """Reject mixed checkouts before loading models; identify the actual files."""
    import hashlib
    import inspect
    import generation.gen_workers as gen_workers
    from vllm_compat import sleep_memory

    root = Path(training_file).resolve().parent
    components = {
        "train_multy_CVaR.py": Path(training_file).resolve(),
        'runtime/host_memory.py': Path(__file__).resolve(),
        'generation/gen_workers.py': Path(gen_workers.__file__).resolve(),
        "vllm_compat/sleep_memory.py": Path(sleep_memory.__file__).resolve(),
    }
    for name, loaded_path in components.items():
        if loaded_path != root / name:
            raise RuntimeError(
                f"mixed training checkout: {name} loaded from {loaded_path}; "
                f"expected {root / name}. Launch a fresh process from the "
                "updated checkout.")
    sleep_parameters = inspect.signature(gen_workers.GenerationPool.sleep).parameters
    if (not {"after_sleep", "host_admission"}.issubset(sleep_parameters)
            or not callable(getattr(gen_workers.PhasedVLLMGenerationPool,
                                    "release_for_training", None))
            or not callable(getattr(sleep_memory.SleepMemoryExtension,
                                    "ttt_host_memory", None))):
        raise RuntimeError(
            "incomplete memory-handoff update: update train_multy_CVaR.py, "
            "generation/gen_workers.py, runtime/host_memory.py and "
            "vllm_compat/sleep_memory.py "
            "together, then launch a fresh process")
    return {
        "checkout": str(root),
        "files": {name: hashlib.sha256(path.read_bytes()).hexdigest()
                  for name, path in components.items()},
        "memory_handoff": "pageable backups; bounded transfers; incremental trainer restore",
    }


def _cgroup_memory_bounds(root="/sys/fs/cgroup", membership="/proc/self/cgroup"):
    """Account for container/service limits, including a tighter ancestor."""
    root = Path(root)
    candidates = {root, root / "memory"}
    try:
        for line in Path(membership).read_text().splitlines():
            _, controllers, relative = line.split(":", 2)
            if not controllers or "memory" in controllers.split(","):
                base = root / "memory" if controllers else root
                path = base / relative.lstrip("/")
                if ".." not in path.parts:
                    candidates.update(p for p in (path, *path.parents) if p == base or base in p.parents)
    except (OSError, ValueError):
        pass
    bounds = []
    for path in candidates:
        for limit_name, usage_name in (("memory.max", "memory.current"),
                                       ("memory.limit_in_bytes", "memory.usage_in_bytes")):
            try:
                limit = int((path / limit_name).read_text().strip())
                used = int((path / usage_name).read_text().strip())
                if limit > 0 and used >= 0:
                    bounds.append((limit, max(0, limit - used)))
            except (OSError, ValueError):
                pass  # Missing controller or the v2 unlimited value "max".
    return bounds


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
            if str(meminfo_path) == "/proc/meminfo":
                for limit, free in _cgroup_memory_bounds():
                    total, available = min(total, limit), min(available, free)
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


def automatic_activation_offload_budget_bytes(
        *, memory_info=None,
        reserve_fraction=DEFAULT_ACTIVATION_OFFLOAD_RESERVE_FRACTION,
        free_reserve_bytes=DEFAULT_ACTIVATION_OFFLOAD_FREE_RESERVE_BYTES):
    """Return the RAM budget available to concurrent activation offloads."""
    reserve_fraction = min(0.95, max(0.0, float(reserve_fraction)))
    free_reserve_bytes = max(0, int(free_reserve_bytes))
    info = host_memory_info() if memory_info is None else memory_info
    if not info:
        return 0
    total = max(1, int(info.get("total", 0)))
    available = max(0, int(info.get("available", total)))
    return min(
        max(0, int(total * (1.0 - reserve_fraction))),
        max(0, available - free_reserve_bytes),
    )


def host_memory_reserve_bytes(info, hardware_profile="RTX6000"):
    """Leave OS/runtime headroom without assuming a particular model or node."""
    total = max(1, int(info["total"]))
    override = os.environ.get("TTT_HOST_MEMORY_RESERVE_GIB")
    if override is not None:
        reserve = float(override)
        if not math.isfinite(reserve) or reserve <= 0:
            raise ValueError("TTT_HOST_MEMORY_RESERVE_GIB must be finite and positive")
        return min(total // 4, int(reserve * GIB))
    if hardware_profile == "B200":
        # This job may own two GPUs on a busy multi-TiB host, not all its RAM.
        return min(total // 4, 16 * GIB)
    if hardware_profile == "B200_full_ram":
        # Use the same user-requested headroom as trainer handoff, including
        # startup/evaluation gates; full machine RAM is not exclusive to us.
        return min(total // 4, 2 * GIB)
    return min(total // 4, max(GIB, min(128 * GIB, total // 10)))


class HostMemoryUnavailable(MemoryError):
    """A transfer cannot fit and no admitted work can release RAM for it."""


class HostMemoryTimeout(HostMemoryUnavailable):
    """A bounded RAM wait expired; stop without a traceback."""


class HostMemoryAdmission:
    """Share live RAM admission across evaluator, transfer and trainer processes.

    Outstanding reservations are charged in addition to current RSS. This is
    deliberately conservative: a newly admitted process may not have touched
    its pages yet. Reading MemAvailable alone lets concurrent allocations all
    spend the same free RAM. No GPU-resident forward acquires this gate.
    """

    def __init__(self, multiprocessing_context, *, hardware_profile="RTX6000"):
        self.hardware_profile = hardware_profile
        self._reserved = multiprocessing_context.Value("q", 0, lock=False)
        self._priority_waiters = multiprocessing_context.Value("q", 0, lock=False)
        self._ram_wait_failed = multiprocessing_context.Value("b", False, lock=False)
        self._condition = multiprocessing_context.Condition(
            multiprocessing_context.RLock())

    @property
    def reserved_bytes(self):
        with self._condition:
            return int(self._reserved.value)

    def available_bytes(self):
        with self._condition:
            info = host_memory_info()
            if info is None:
                return None
            return max(0, int(info["available"])
                       - host_memory_reserve_bytes(info, self.hardware_profile)
                       - int(self._reserved.value))

    def stop_waiting(self):
        with self._condition:
            self._ram_wait_failed.value = True
            self._condition.notify_all()

    @contextmanager
    def prioritize_transfers(self):
        """Keep new evaluators from taking RAM between incremental chunks."""
        with self._condition:
            self._priority_waiters.value += 1
        try:
            yield
        finally:
            with self._condition:
                self._priority_waiters.value -= 1
                self._condition.notify_all()

    @contextmanager
    def reservation(self, size_bytes, *, wait_for_external=True, stage="RAM allocation",
                    priority=False, reserve_bytes=None, minimum_bytes=None, deadline=None):
        size_bytes = max(0, int(size_bytes))
        minimum_bytes = size_bytes if minimum_bytes is None else int(minimum_bytes)
        if not 0 <= minimum_bytes <= size_bytes:
            raise ValueError("minimum reservation must be between zero and the requested bytes")
        bounded_wait = self.hardware_profile in ("B200", "B200_full_ram")
        if bounded_wait and deadline is None:
            deadline = time.monotonic() + RAM_WAIT_TIMEOUT_S
        next_report = 0.0
        waited = False
        with self._condition:
            if priority:
                self._priority_waiters.value += 1
            try:
                while True:
                    if self._ram_wait_failed.value:
                        raise HostMemoryTimeout("a shared RAM wait exceeded 2 hours; stopping this run")
                    info = host_memory_info()
                    if info is None and not bounded_wait:
                        break
                    reserve = (max(0, int(reserve_bytes)) if reserve_bytes is not None else
                               host_memory_reserve_bytes(info, self.hardware_profile) if info else 0)
                    available = (int(info["available"]) - reserve - int(self._reserved.value)
                                 if info else None)
                    if not bounded_wait and size_bytes > int(info["total"]) - reserve:
                        raise HostMemoryUnavailable("host-memory request exceeds node capacity")
                    # Let active evaluations finish, but do not let new ones
                    # repeatedly spend the space needed by a model handoff.
                    has_priority = priority or not self._priority_waiters.value
                    if has_priority and available is not None and available >= minimum_bytes:
                        size_bytes = min(size_bytes, available)
                        break
                    waited = True
                    if bounded_wait:
                        now = time.monotonic()
                        if now >= deadline:
                            self._ram_wait_failed.value = True
                            self._condition.notify_all()
                            raise HostMemoryTimeout(f"{stage}: RAM remained unavailable for 2 hours")
                        if now >= next_report:
                            free = "unknown" if available is None else f"{max(0, available) / GIB:.1f} GiB"
                            needed = (f"{size_bytes / GIB:.1f}" if minimum_bytes == size_bytes else
                                      f"{minimum_bytes / GIB:.3f}–{size_bytes / GIB:.3f}")
                            print(f"[ram-wait] {stage}: need {needed} GiB, "
                                  f"admissible={free}; monitoring every {RAM_POLL_INTERVAL_S:g}s, "
                                  f"{(deadline - now) / 60:.0f} min remaining", flush=True)
                            next_report = now + RAM_RETRY_INTERVAL_S
                        self._condition.wait(timeout=min(RAM_POLL_INTERVAL_S, deadline - now))
                        continue
                    if not wait_for_external and not self._reserved.value:
                        raise HostMemoryUnavailable("host RAM cannot retain another model backup")
                    self._condition.wait(timeout=0.25)
                # Check and claim under the same lock, before any allocation.
                self._reserved.value += size_bytes
            finally:
                if priority:
                    self._priority_waiters.value -= 1
                    self._condition.notify_all()
        try:
            if waited:
                print(f"[ram-ready] {stage}: reserved {size_bytes / GIB:.1f} GiB; continuing", flush=True)
            yield size_bytes
        finally:
            with self._condition:
                self._reserved.value -= size_bytes
                self._condition.notify_all()


def wait_for_host_memory(size_bytes, *, hardware_profile="RTX6000", stage):
    """Preflight for cooperative CPU parking on shared B200 hosts."""
    if hardware_profile not in ("B200", "B200_full_ram"):
        return
    import multiprocessing
    admission = HostMemoryAdmission(multiprocessing.get_context("spawn"), hardware_profile=hardware_profile)
    with admission.reservation(size_bytes, stage=stage):
        pass


def retry_host_allocation(allocation, *, hardware_profile="RTX6000", stage,
                          required_bytes=0):
    """Retry CPU allocation failures only; never replay a training update."""
    if hardware_profile not in ("B200", "B200_full_ram"):
        return allocation()
    from contextlib import nullcontext
    import multiprocessing
    admission = (HostMemoryAdmission(multiprocessing.get_context("spawn"), hardware_profile=hardware_profile)
                 if required_bytes else None)
    deadline = None
    while True:
        if deadline is not None and time.monotonic() >= deadline:
            raise HostMemoryTimeout(f"{stage}: CPU allocation failed for 2 hours")
        try:
            with (admission.reservation(required_bytes, stage=stage, priority=True)
                  if admission is not None else nullcontext()):
                return allocation()
        except (MemoryError, OSError, RuntimeError) as error:
            message = str(error).lower()
            if isinstance(error, HostMemoryUnavailable) or not (
                    isinstance(error, MemoryError)
                    or isinstance(error, OSError) and error.errno == errno.ENOMEM
                    or "cpuallocator" in message and any(text in message for text in
                        ("not enough memory", "can't allocate memory", "cannot allocate memory"))):
                raise
        # Leave the exception scope before collecting partially loaded tensors.
        gc.collect()
        now = time.monotonic()
        if deadline is None:
            deadline = now + RAM_WAIT_TIMEOUT_S
        print(f"[ram-wait] {stage}: CPU allocation failed; retry in 60s, "
              f"{max(0, deadline - now) / 60:.0f} min remaining", flush=True)
        time.sleep(min(RAM_RETRY_INTERVAL_S, max(0, deadline - now)))


def host_transfer_wave(worker_ids, backup_bytes, *, memory_info=None,
                       available_bytes=None):
    """Bound temporary CPU duplicates; generation parallelism is unaffected."""
    info = host_memory_info() if memory_info is None else memory_info
    # Large concurrent cudaHostAlloc/copies used to stall every engine. Limit
    # only phase-boundary transfers, and restore matching trainers between waves.
    budget = 128 * GIB if info is None else max(
        GIB, min(256 * GIB, int(info["total"]) // 10))
    if available_bytes is not None:
        budget = min(budget, max(0, int(available_bytes)))
    wave, used = [], 0
    for rank in worker_ids:
        size = max(0, int(backup_bytes[rank]))
        if wave and used + size > budget:
            break
        wave.append(rank)
        used += size
    return tuple(wave)


def automatic_activation_offload_slots(
        worker_count, *, memory_info=None,
        bytes_per_slot=DEFAULT_ACTIVATION_OFFLOAD_SLOT_BYTES,
        reserve_fraction=DEFAULT_ACTIVATION_OFFLOAD_RESERVE_FRACTION,
        free_reserve_bytes=DEFAULT_ACTIVATION_OFFLOAD_FREE_RESERVE_BYTES):
    """Return conservative full-context offload concurrency.

    Ordinary GPU-resident forwards remain fully parallel.  This limit applies
    only after an exact long sequence has exhausted all GPU-headroom stages
    and falls back to ``save_on_cpu``. This compatibility helper reports how
    many worst-case, full-context graphs fit. Runtime admission below is
    token-weighted, so shorter trajectories can use more ranks concurrently.
    """
    worker_count = max(1, int(worker_count))
    bytes_per_slot = max(1, int(bytes_per_slot))
    budget = automatic_activation_offload_budget_bytes(
        memory_info=memory_info,
        reserve_fraction=reserve_fraction,
        free_reserve_bytes=free_reserve_bytes,
    )
    return min(worker_count, max(1, budget // bytes_per_slot))


class _ActivationOffloadReservation:
    """One atomic weighted reservation from a shared admission gate."""

    def __init__(self, gate, work):
        self._gate = gate
        self._work = max(1, int(work))
        self._units = None
        self._acquired = False

    def __enter__(self):
        self._units = self._gate._acquire(self._work)
        self._host_context = self._gate.host_admission.reservation(
            self._units * self._gate.unit_bytes)
        try:
            self._host_context.__enter__()
        except BaseException:
            self._gate._release(self._units)
            raise
        self._acquired = True
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self._acquired:
            self._host_context.__exit__(exc_type, exc_value, traceback)
            self._gate._release(self._units)
            self._acquired = False
        return False


class TokenWeightedActivationOffloadGate:
    """Atomically share a host-RAM budget according to trajectory length.

    Admission begins with the existing conservative 384-GiB charge.  After
    two longest-first jobs complete, workers share the measured high-water
    activation bytes/token and reserve that estimate with a safety margin.
    This opens additional ranks for shorter trajectories without guessing at
    their footprint or weakening the node-wide RAM reserve. Reservations are
    atomic, so workers cannot partially acquire the budget and deadlock.
    """

    def __init__(
            self, multiprocessing_context, capacity_bytes, reference_work,
            *, full_context_bytes=DEFAULT_ACTIVATION_OFFLOAD_SLOT_BYTES,
            minimum_request_bytes=DEFAULT_ACTIVATION_OFFLOAD_MIN_REQUEST_BYTES,
            unit_bytes=DEFAULT_ACTIVATION_OFFLOAD_UNIT_BYTES,
            calibration_samples=DEFAULT_ACTIVATION_OFFLOAD_CALIBRATION_SAMPLES,
            calibration_margin=DEFAULT_ACTIVATION_OFFLOAD_CALIBRATION_MARGIN):
        self.capacity_bytes = max(0, int(capacity_bytes))
        self.reference_work = max(1, int(reference_work))
        self.full_context_bytes = max(1, int(full_context_bytes))
        self.minimum_request_bytes = max(1, int(minimum_request_bytes))
        self.unit_bytes = max(1, int(unit_bytes))
        self.calibration_samples = max(1, int(calibration_samples))
        self.calibration_margin = max(1.0, float(calibration_margin))
        self.capacity_units = self.capacity_bytes // self.unit_bytes
        self.minimum_request_units = int(math.ceil(
            self.minimum_request_bytes / self.unit_bytes))
        self.enabled = bool(
            self.capacity_units >= self.minimum_request_units)
        self._available_units = multiprocessing_context.Value(
            "q", int(self.capacity_units), lock=False)
        self._observations = multiprocessing_context.Value(
            "q", 0, lock=False)
        self._peak_bytes_per_work = multiprocessing_context.Value(
            "d", 0.0, lock=False)
        self._condition = multiprocessing_context.Condition(
            multiprocessing_context.RLock())
        self.host_admission = HostMemoryAdmission(multiprocessing_context)

    @property
    def full_context_concurrency(self):
        units = self.request_units(self.reference_work)
        if units is None:
            return 0
        return max(1, self.capacity_units // units)

    def estimated_request_bytes(self, work):
        work = max(1, int(work))
        # Begin with the original worst-case 384-GiB charge. Once two real
        # offloads have completed, every process sees the shared high-water
        # bytes/token measurement and can safely admit more shorter jobs. The
        # estimate retains a safety margin rather than assuming the original
        # full-context estimate is an upper bound on every model/runtime.
        if int(self._observations.value) < self.calibration_samples:
            return self.full_context_bytes
        scaled = int(math.ceil(
            float(self._peak_bytes_per_work.value)
            * self.calibration_margin * min(work, self.reference_work)))
        return max(self.minimum_request_bytes, scaled)

    def request_units(self, work):
        if not self.enabled:
            return None
        units = int(math.ceil(
            self.estimated_request_bytes(work) / self.unit_bytes))
        # Never reserve less than the estimated allocation. An oversized
        # request must use the existing model-sharded fallback instead.
        return max(1, units) if units <= self.capacity_units else None

    def can_admit(self, work):
        units = self.request_units(work)
        if units is None:
            return False
        info = host_memory_info()
        if info is None:
            return True
        # Other admitted work can finish and return its RAM. If even that
        # cannot make this request fit, don't wait forever on resident models.
        return units * self.unit_bytes <= (
            int(info["available"]) + self.host_admission.reserved_bytes
            - host_memory_reserve_bytes(info, self.host_admission.hardware_profile))

    def reservation(self, work):
        units = self.request_units(work)
        if units is None:
            raise RuntimeError(
                "activation-offload request exceeds the safe host-RAM budget")
        return _ActivationOffloadReservation(self, work)

    def observe(self, work, footprint_bytes):
        """Publish one completed offload's retained host-memory high water."""
        work = max(1, int(work))
        footprint_bytes = max(0, int(footprint_bytes))
        if footprint_bytes < GIB:
            return
        rate = float(footprint_bytes) / work
        with self._condition:
            self._peak_bytes_per_work.value = max(
                float(self._peak_bytes_per_work.value), rate)
            self._observations.value += 1
            self._condition.notify_all()

    def _acquire(self, work):
        work = max(1, int(work))
        with self._condition:
            while True:
                if self.host_admission._ram_wait_failed.value:
                    raise HostMemoryTimeout("a shared RAM wait exceeded 2 hours; stopping this run")
                units = self.request_units(work)
                if units is None:
                    raise RuntimeError(
                        "activation-offload request exceeds the safe "
                        "host-RAM budget")
                if int(self._available_units.value) >= units:
                    self._available_units.value -= units
                    return units
                self._condition.wait(timeout=RAM_POLL_INTERVAL_S)

    def _release(self, units):
        units = int(units)
        with self._condition:
            next_value = int(self._available_units.value) + units
            if next_value > self.capacity_units:
                raise RuntimeError(
                    "activation-offload admission released excess capacity")
            self._available_units.value = next_value
            self._condition.notify_all()


def process_rss_bytes(statm_path="/proc/self/statm"):
    """Return this process's resident bytes when Linux exposes ``statm``."""
    try:
        fields = Path(statm_path).read_text(encoding="utf-8").split()
        return int(fields[1]) * int(os.sysconf("SC_PAGE_SIZE"))
    except (OSError, IndexError, TypeError, ValueError):
        return None


class ProcessRSSPeakSampler:
    """Measure a process's incremental RSS peak across one offloaded graph.

    Saved tensors can be freed before the backward call returns, so comparing
    RSS only at the two endpoints can miss the true live peak.  Sampling
    ``/proc/self/statm`` from a tiny daemon thread captures that peak without
    synchronizing CUDA or retaining any tensor.  The sampler is used only by
    the exceptional CPU-activation-offload path.
    """

    def __init__(self, interval_seconds=0.05):
        self.interval_seconds = max(0.01, float(interval_seconds))
        self.baseline_bytes = None
        self.peak_bytes = None
        self._stop = threading.Event()
        self._thread = None

    def _sample(self):
        rss = process_rss_bytes()
        if rss is None:
            return
        rss = int(rss)
        if self.baseline_bytes is None:
            self.baseline_bytes = rss
        if self.peak_bytes is None or rss > self.peak_bytes:
            self.peak_bytes = rss

    def _run(self):
        while not self._stop.wait(self.interval_seconds):
            self._sample()

    def __enter__(self):
        self._sample()
        if self.baseline_bytes is not None:
            self._thread = threading.Thread(
                target=self._run,
                name="ttt-activation-rss-sampler",
                daemon=True,
            )
            self._thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._sample()
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, 2 * self.interval_seconds))
        self._sample()
        return False

    @property
    def footprint_bytes(self):
        if self.baseline_bytes is None or self.peak_bytes is None:
            return 0
        return max(0, int(self.peak_bytes) - int(self.baseline_bytes))


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


def estimated_module_bytes(model, *, frozen_to_disk=False):
    """Estimate bytes needed for one full CPU copy of a model replica."""
    total = 0
    seen = set()
    tensors = list(model.parameters()) + list(model.buffers())
    if frozen_to_disk:
        tensors = [tensor for tensor in tensors if tensor.requires_grad]
        tensors += [parameter.grad for parameter in model.parameters()
                    if parameter.grad is not None]
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
