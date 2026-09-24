"""
Persistent multi-GPU rollout generation pool.

HF runs one persistent worker process per GPU. vLLM runs one persistent engine
per configured GPU group; every engine can use tensor and pipeline parallelism
instead of loading a complete copy of a large model on every GPU. Each worker:
  - loads either a plain Transformers model or a vLLM engine
  - applies the current LoRA adapter saved by the trainer
  - generates its share of rollouts in batches
  - reports results PER JOB so the main process can (a) drive a rollout progress
    bar and (b) start evaluating each rollout's program on CPU threads WHILE the
    GPUs keep generating the rest.

The main process owns the differentiable training model. Each step it saves the
small LoRA adapter to a new directory, then asks the pool to generate with those
weights. In vLLM mode this is deliberately a trainer/rollout split: vLLM is the
fast inference engine, while HF+PEFT (or optionally Unsloth) performs backward.

LOGPROBS. vLLM rollout calls can return the sampled token's logprob with each
token. The trainer retains only those scalar values on the CPU, not full-vocab
distributions. While CPU reward sandboxes continue running, the same awake vLLM
engines score each completed prompt/response once without LoRA to obtain the
fixed reference logprobs. HF training then performs only the differentiable
current-policy pass. Non-training generation calls keep the old compact
(text, token_ids) result format.

SEEDING. HF workers reseed random / numpy / torch from (seed, step, rank)
before every task. vLLM workers use stable per-request seeds derived from
(seed, step, rank, group). Keying on the step makes step t reproducible on its
own, and the memory maker's offset calls cannot shift the rollout stream.
Determinism holds for a fixed num_gpus, group_size AND gen_micro_batch, since
distribute_jobs splits the group across workers and changing the split (or the
micro-batch chunking) changes which sequence each worker draws.

OOM RECOVERY. The HF worker halves its per-call sequence count and retries when
generate() OOMs. vLLM owns scheduling and KV-cache admission itself; an engine
failure is sent back to the main process and aborts the step instead of hanging
or silently converting an infrastructure problem into low rewards.

No async. Plain torch.multiprocessing with persistent workers and queues.

Protocol (per step):
  main -> worker[w].task_queue:   (step, adapter_path, jobs, gen_kwargs)
       where jobs = [(group_idx, prompt_text, num_samples), ...]
  worker[w] -> result_queue:      (rank, group_idx, [result, ...])
       where result is (text, token_ids), optionally with sampled logprobs as
       a third item when the caller requests them.
       one message PER JOB, so the pool can stream results as they land.
"""

import importlib.metadata
import os
from array import array
from contextlib import nullcontext
import queue
import re
import sys
import threading
import time
import traceback
import multiprocessing as mp

# Level-1 vLLM sleep releases tagged weights and KV blocks, but the sleeping
# process still owns CUDA contexts, NCCL state, compiled graphs, and allocator
# metadata. Two alternating, RAM-resident pools therefore cannot each claim
# 90% of the same card. Keep enough unclaimed memory for the inactive pool.
# Long prompt-logprob scoring briefly materializes a vocabulary projection for
# an entire chunk.  Qwen3-Coder's 150k-token vocabulary needs several GiB on
# top of the steady model/KV allocation, while the sleeping strategy engine
# and offloaded trainer still retain CUDA context state.  Twelve GiB left less
# than one projection's headroom in real 96-GiB runs; keep sixteen unclaimed.
VLLM_CORESIDENT_SLEEP_RESERVE_GIB = 16.0
# Prompt-logprob scoring materializes a vocabulary projection for every prefill
# chunk.  On 150k-vocabulary coder models that transient is substantially
# larger than ordinary generation activations, and CUDA-graph memory can make
# vLLM exceed its nominal utilization target by several GiB.  Preserve enough
# real headroom for the projection instead of letting an otherwise healthy
# engine die after rollouts have already completed.
VLLM_TOKEN_SCORING_RESERVE_GIB = 24.0
VLLM_TOKEN_SCORING_MAX_BATCHED_TOKENS = 4096
VLLM_STARTUP_FREE_MARGIN_GIB = 1.0

# tqdm ships with transformers/huggingface_hub, so it's almost always present.
# Fall back to a coarse print bar if it isn't, so nothing depends on it.
try:
    from tqdm import tqdm
    _HAS_TQDM = True
except Exception:
    tqdm = None
    _HAS_TQDM = False


def _positive_env_timeout(name, default):
    """Read a bounded-operation timeout without accepting infinite waits."""
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return float(default)
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive number") from exc
    if value <= 0:
        raise ValueError(f"{name} must be a positive number")
    return value


class _PrintBar:
    """Minimal stand-in for tqdm: prints progress at ~10% increments."""

    def __init__(self, total, desc="progress"):
        self.total = max(int(total), 1)
        self.n = 0
        self.desc = desc
        self._last_decile = -1

    def update(self, k=1):
        self.n += k
        decile = int(10 * self.n / self.total)
        if decile != self._last_decile:
            self._last_decile = decile
            pct = int(100 * self.n / self.total)
            print(f"[pool] {self.desc} {self.n}/{self.total} ({pct}%)", flush=True)

    def close(self):
        if self.n < self.total:
            print(f"[pool] {self.desc} {self.n}/{self.total}", flush=True)


def make_progress_bar(total, desc="progress"):
    total = int(max(total, 1))
    if _HAS_TQDM:
        return tqdm(total=total, desc=desc, unit="it",
                    leave=False, dynamic_ncols=True)
    return _PrintBar(total, desc=desc)


def distribute_jobs(prompts_by_group, group_size, num_workers,
                    counts_by_group=None):
    """
    prompts_by_group: list of prompt strings, one per group (index = group_idx)

    Returns worker_jobs: list (len num_workers) of lists of
        (group_idx, prompt_text, count)
    so that across workers each group gets exactly group_size samples and
    large groups use every worker. For groups smaller than the worker count,
    the remainder rotates by group index so small jobs do not all land on rank
    zero and leave the other GPUs idle.
    """
    worker_jobs = [[] for _ in range(num_workers)]
    counts = ([int(group_size)] * len(prompts_by_group)
              if counts_by_group is None else [int(x) for x in counts_by_group])
    if len(counts) != len(prompts_by_group):
        raise ValueError("counts_by_group must align with prompts_by_group")
    for g, (prompt, count) in enumerate(zip(prompts_by_group, counts)):
        if count < 0:
            raise ValueError("generation counts must be non-negative")
        base = count // num_workers
        rem = count % num_workers
        for w in range(num_workers):
            worker_count = base + (
                1 if ((w - g) % num_workers) < rem else 0)
            if worker_count > 0:
                worker_jobs[w].append((g, prompt, worker_count))
    return worker_jobs


def worker_seed(seed, step, rank):
    """
    Deterministic seed for (run, step, worker). Kept as a module-level function
    so the trainer's single-GPU path can key its own reseed the same way.
    """
    return (int(seed) * 1_000_003 + int(step) * 1009 + int(rank) * 7 + 13) % (2 ** 31 - 1)


def _chosen_token_logprobs(token_ids, position_logprobs):
    """Extract only each observed token's scalar logprob from vLLM output."""
    if position_logprobs is None or len(position_logprobs) != len(token_ids):
        return None
    values = []
    for token_id, candidates in zip(token_ids, position_logprobs):
        if candidates is None:
            return None
        try:
            entry = candidates.get(int(token_id))
        except AttributeError:
            try:
                entry = candidates[int(token_id)]
            except (KeyError, IndexError, TypeError):
                entry = None
        if entry is None:
            return None
        value = getattr(entry, "logprob", entry)
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            return None
    return values


def _iter_hf_job_batches(gen_model, tokenizer, jobs, device,
                         max_seq_length, gen_kwargs, cap_state=None,
                         log_prefix="hf"):
    """Generate assigned HF rollouts in cross-prompt micro-batches.

    Each expanded batch contains requests from as many prompt jobs as fit, so
    one GPU does not finish an entire parent before starting the next.  The
    mutable cap_state makes an OOM-reduced batch ceiling sticky across steps.
    """
    import torch

    if cap_state is None:
        cap_state = {"value": 0}
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id or eos_id
    mb = int(gen_kwargs.get("micro_batch", 0) or 0)

    pending = []
    for group_idx, prompt, count in jobs:
        count = int(count)
        prompt_len = len(tokenizer.encode(prompt))
        if prompt_len >= int(max_seq_length):
            print(f"[{log_prefix}] group {group_idx} prompt is {prompt_len} "
                  f"tokens, at/over max_seq_length={max_seq_length}; dropping "
                  f"{count} rollout(s)", flush=True)
            yield group_idx, [("", []) for _ in range(count)]
            continue
        pending.extend((group_idx, prompt, prompt_len) for _ in range(count))

    # Similar-length requests minimize left-padding and keep more of the token
    # budget useful while still mixing prompt jobs in every micro-batch.
    pending.sort(key=lambda item: item[2])
    while pending:
        limit = len(pending)
        if mb > 0:
            limit = min(limit, mb)
        learned_cap = int(cap_state.get("value", 0) or 0)
        if learned_cap > 0:
            limit = min(limit, learned_cap)
        n = max(1, limit)
        batch_items = pending[:n]
        prompts = [item[1] for item in batch_items]
        enc = tokenizer(prompts, padding=True, return_tensors="pt").to(device)
        input_width = int(enc.input_ids.shape[1])
        max_new_tokens = min(
            int(gen_kwargs["max_new_tokens"]),
            int(max_seq_length) - input_width,
        )
        if max_new_tokens < 1:
            by_group = {}
            for group_idx, _prompt, _prompt_len in batch_items:
                by_group.setdefault(group_idx, []).append(("", []))
            del pending[:n]
            del enc
            for group_idx, results in by_group.items():
                yield group_idx, results
            continue

        try:
            with torch.inference_mode():
                out = gen_model.generate(
                    **enc,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=gen_kwargs["temperature"],
                    top_p=gen_kwargs["top_p"],
                    pad_token_id=pad_id,
                )
        except torch.cuda.OutOfMemoryError:
            del enc
            torch.cuda.empty_cache()
            if n == 1:
                group_idx, _prompt, prompt_len = pending.pop(0)
                print(f"[{log_prefix}] OOM at one sequence (prompt "
                      f"{prompt_len} tok); dropping one rollout", flush=True)
                yield group_idx, [("", [])]
                continue
            cap_state["value"] = max(1, n // 2)
            print(f"[{log_prefix}] OOM at cross-prompt batch n={n}; halving "
                  f"the sticky per-call ceiling to {cap_state['value']} and "
                  "retrying", flush=True)
            continue

        by_group = {}
        rows = int(out.shape[0])
        for row, (group_idx, _prompt, _prompt_len) in enumerate(batch_items):
            if row >= rows:
                item = ("", [])
            else:
                gen_ids = out[row, input_width:].tolist()
                if eos_id is not None and eos_id in gen_ids:
                    gen_ids = gen_ids[:gen_ids.index(eos_id) + 1]
                item = (tokenizer.decode(gen_ids, skip_special_tokens=True),
                        gen_ids)
            by_group.setdefault(group_idx, []).append(item)
        del pending[:n]
        del out, enc
        for group_idx, results in by_group.items():
            yield group_idx, results


def _hf_worker_loop(rank, gpu_id, model_name, max_seq_length, load_in_4bit,
                    task_queue, result_queue, ready_queue, seed=None, **_unused):
    """
    Persistent worker. Loads the model once, then serves generation tasks
    until it receives None.
    """
    # Pin to our GPU. Do this before heavy CUDA work.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    import random
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    from peft.utils import set_peft_model_state_dict
    from safetensors.torch import load_file

    # With CUDA_VISIBLE_DEVICES set, our GPU is cuda:0 inside this process
    device = "cuda:0"

    print(f"[worker {rank}] loading {model_name} on physical GPU {gpu_id} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # left-pad for decoder-only batched generation

    model_kwargs = dict(dtype=torch.bfloat16, trust_remote_code=True)
    if load_in_4bit:
        try:
            from transformers import BitsAndBytesConfig
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4",
            )
        except ImportError:
            pass

    base = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    if not load_in_4bit:
        base = base.to(device)
    base.eval()

    peft_model = None          # created on first adapter load
    current_adapter = None     # path of the adapter currently loaded

    def ensure_adapter(adapter_path):
        nonlocal peft_model, current_adapter
        if adapter_path is None:
            return peft_model if peft_model is not None else base
        if peft_model is None:
            peft_model = PeftModel.from_pretrained(base, adapter_path, is_trainable=False)
            peft_model.eval()
            current_adapter = adapter_path
            return peft_model
        if adapter_path != current_adapter:
            # Reload just the LoRA weights into the existing wrapper
            sd_path = os.path.join(adapter_path, "adapter_model.safetensors")
            try:
                weights = load_file(sd_path)
                set_peft_model_state_dict(peft_model, weights)
            except Exception as e:
                # Fallback: rewrap from scratch (re-reads only the tiny adapter)
                print(f"[worker {rank}] adapter reload fallback ({e})", flush=True)
                peft_model = PeftModel.from_pretrained(base, adapter_path, is_trainable=False)
                peft_model.eval()
            current_adapter = adapter_path
        return peft_model

    # Persistent per-worker ceiling on sequences per generate() call. 0 means
    # "no learned limit yet"; the effective cap is then the configured
    # micro-batch (or the whole job). It ONLY ever shrinks: once this worker
    # OOMs at some size it never tries that size again, this step or any later
    # one. Prompts only grow as the search finds longer programs, so a limit
    # learned at step 2 is the right limit for step 3+. This is what stops a
    # long-prompt step from killing the worker and hanging the whole run.
    gen_cap = 0

    # Signal that we finished loading
    ready_queue.put(("ready", rank, ""))

    while True:
        task = task_queue.get()
        if task is None:
            break
        step, adapter_path, jobs, gen_kwargs = task

        if seed is not None:
            # Keyed on (seed, step, rank), not advanced sequentially, so step t
            # is reproducible on its own and does not depend on how many
            # generations happened before it. No-op when seed is None (the
            # non-deterministic default).
            import random
            import numpy as _np
            s = (int(seed) * 1_000_003 + int(step) * 1009
                 + rank * 7 + 13) % (2**31 - 1)
            random.seed(s)
            _np.random.seed(s % (2**32 - 1))
            torch.manual_seed(s)
            torch.cuda.manual_seed_all(s)

        gen_model = ensure_adapter(adapter_path)

        adapter_context = (
            gen_model.disable_adapter()
            if adapter_path is None and peft_model is not None
            else nullcontext()
        )
        cap_state = {"value": gen_cap}
        with adapter_context:
            for group_idx, job_results in _iter_hf_job_batches(
                    gen_model, tokenizer, jobs, device, max_seq_length,
                    gen_kwargs, cap_state=cap_state,
                    log_prefix=f"worker {rank}"):
                # Report every completed micro-batch so evaluation can overlap
                # the remaining GPU work.
                result_queue.put((rank, group_idx, job_results))
        gen_cap = int(cap_state["value"])

    print(f"[worker {rank}] shutting down", flush=True)


def _resolve_vllm_quantization(model_name, load_in_4bit, quantization):
    """Resolve only the explicit rollout-engine quantization setting.

    ``load_in_4bit`` belongs to the differentiable training copy.  Propagating
    it into vLLM silently changes the separately configured rollout model and,
    for BitsAndBytes, selects a loader whose weights vLLM cannot reload after
    level-2 sleep.
    """
    del model_name, load_in_4bit
    requested = str(quantization or "").strip()
    if not requested or requested.lower() in ("auto", "none"):
        return None
    return requested


def _vllm_engine_kwargs(model_name, max_seq_length, load_in_4bit,
                        lora_rank, gpu_memory_utilization,
                        enforce_eager=False, enable_prefix_caching=True,
                        max_num_seqs=0, seed=None, quantization=None,
                        tensor_parallel_size=1, pipeline_parallel_size=1,
                        max_num_batched_tokens=0,
                        enable_expert_parallel=False,
                        enable_sleep_mode=False,
                        disable_custom_all_reduce=False):
    """Build vLLM constructor arguments without importing vLLM.

    Kept separate both for unit testing and so the parent process never imports
    vLLM/CUDA before workers are pinned to their GPUs.
    """
    supported_lora_ranks = (1, 8, 16, 32, 64, 128, 256, 320, 512)
    requested_rank = int(lora_rank)
    if requested_rank < 1:
        raise ValueError("LoRA rank must be positive")
    max_lora_rank = next(
        (rank for rank in supported_lora_ranks if rank >= requested_rank), None)
    if max_lora_rank is None:
        raise ValueError(
            f"vLLM LoRA rank {requested_rank} exceeds supported maximum "
            f"{supported_lora_ranks[-1]}")

    kwargs = {
        "model": model_name,
        "dtype": "bfloat16",
        "trust_remote_code": True,
        "max_model_len": int(max_seq_length),
        "enable_lora": True,
        "max_lora_rank": max_lora_rank,
        "max_loras": 1,
        "max_cpu_loras": 2,
        "gpu_memory_utilization": float(gpu_memory_utilization),
        "enforce_eager": bool(enforce_eager),
        "enable_prefix_caching": bool(enable_prefix_caching),
        "disable_log_stats": True,
        "tensor_parallel_size": int(tensor_parallel_size),
        "pipeline_parallel_size": int(pipeline_parallel_size),
    }
    if "gpt-oss-120b" in str(model_name).lower():
        # vLLM's default safetensors strategy prefetches the complete
        # checkpoint into the OS page cache when the Hugging Face cache is on
        # NFS/Lustre.  That adds roughly another 61 GiB of host-memory pressure
        # while this model's QLoRA trainer is already offloaded to CPU, and the
        # scheduler can SIGKILL the parent even though the TP=4 engine fits on
        # the GPUs.  Lazy mmap loading keeps those cache pages reclaimable and
        # disables the eager network-filesystem prefetch without changing the
        # weights or quantization used by vLLM.
        kwargs["safetensors_load_strategy"] = "lazy"
    if int(tensor_parallel_size) > 1:
        # LoRA work is otherwise repeated on every TP rank. Sharding it also
        # avoids a large adapter-side memory spike on wide models.
        kwargs["fully_sharded_loras"] = True
    if int(tensor_parallel_size) * int(pipeline_parallel_size) > 1:
        kwargs["distributed_executor_backend"] = "mp"
    if bool(enable_expert_parallel):
        kwargs["enable_expert_parallel"] = True
    if bool(enable_sleep_mode):
        kwargs["enable_sleep_mode"] = True
    if bool(disable_custom_all_reduce):
        # vLLM's CUDA custom all-reduce uses process-global IPC resources.
        # Multiple independent TP engines on disjoint, remapped GPU subsets
        # can collide during CUDA-graph warm-up (custom_all_reduce.cuh:164).
        # NCCL/PyNCCL computes the same collective without that shared state.
        kwargs["disable_custom_all_reduce"] = True
    if int(max_num_seqs or 0) > 0:
        kwargs["max_num_seqs"] = int(max_num_seqs)
    if int(max_num_batched_tokens or 0) > 0:
        kwargs["max_num_batched_tokens"] = int(max_num_batched_tokens)
        if int(max_num_batched_tokens) < int(max_seq_length):
            kwargs["enable_chunked_prefill"] = True
    if seed is not None:
        kwargs["seed"] = int(seed)
    # Rollout quantization is independent of the training copy's QLoRA format.
    quantization = _resolve_vllm_quantization(
        model_name, load_in_4bit, quantization)
    if quantization:
        kwargs["quantization"] = quantization
    return kwargs


def _vllm_job_seed(seed, step, rank, group_idx, sample_offset=0):
    """Stable per-request seed; None preserves vLLM's stochastic default."""
    if seed is None:
        return None
    return (
        worker_seed(seed, step, rank)
        + int(group_idx) * 104_729
        + int(sample_offset) * 130_363
    ) % (2 ** 31 - 1)


def _python_development_header_path():
    """Return an installed Python.h path, or the expected path when absent."""
    import sysconfig

    candidates = []
    paths = sysconfig.get_paths()
    for key in ("include", "platinclude"):
        if paths.get(key):
            candidates.append(os.path.join(paths[key], "Python.h"))
    for key in ("INCLUDEPY", "CONFINCLUDEPY"):
        include_dir = sysconfig.get_config_var(key)
        if include_dir:
            candidates.append(os.path.join(include_dir, "Python.h"))

    # Preserve order while avoiding repeated checks; sysconfig commonly maps
    # all four entries to the same interpreter include directory.
    candidates = list(dict.fromkeys(candidates))
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate, True
    expected = (candidates[0] if candidates else
                f"Python.h for Python {sys.version_info.major}."
                f"{sys.version_info.minor}")
    return expected, False


def _resolve_vllm_enforce_eager(requested, header_available=None):
    """Avoid vLLM's TorchInductor startup path without Python dev headers."""
    if bool(requested):
        return True, False
    if header_available is None:
        _header_path, header_available = _python_development_header_path()
    fallback = not bool(header_available)
    return fallback, fallback


def _prepare_vllm_sleep_allocator_env():
    """Remove PyTorch's expandable allocator only inside sleep-mode workers.

    vLLM's CuMemAllocator backs sleep/wake and explicitly rejects expandable
    segments. The trainer remains in its parent process and keeps the setting.
    """
    for name in ("PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_ALLOC_CONF"):
        raw = os.environ.get(name)
        if not raw:
            continue
        entries = [entry.strip() for entry in raw.split(",") if entry.strip()]
        kept = [entry for entry in entries
                if entry.lower() != "expandable_segments:true"]
        if len(kept) == len(entries):
            continue
        if kept:
            os.environ[name] = ",".join(kept)
        else:
            os.environ.pop(name, None)
        print(f"[vllm] removed expandable_segments from {name}; sleep mode "
              "requires vLLM's CuMemAllocator", flush=True)


def _flashinfer_comm_guard_required(package_version, python_version=None):
    """Return whether FlashInfer's comm package has the annotation bug."""
    python_version = (sys.version_info[:2] if python_version is None
                      else tuple(python_version[:2]))
    if python_version not in ((3, 10), (3, 11)):
        return False
    match = re.match(r"^(\d+)\.(\d+)\.(\d+)", str(package_version or ""))
    if not match:
        return False
    return tuple(int(part) for part in match.groups()) < (0, 6, 17)


def _prepare_flashinfer_comm_compat():
    """Disable only broken FlashInfer communication in vLLM subprocesses.

    FlashInfer before 0.6.17 evaluates ``array.array[int]`` while importing its
    communication package. Python 3.10/3.11 reject that annotation, and vLLM
    0.28 imports the optional module even when FlashInfer all-reduce is off.
    Blocking that package makes vLLM use its regular NCCL/custom all-reduce.
    A scoped ``sitecustomize`` path propagates the block into vLLM's freshly
    spawned engine and tensor-parallel workers.
    """
    try:
        package_version = importlib.metadata.version("flashinfer-python")
    except importlib.metadata.PackageNotFoundError:
        return False
    if not _flashinfer_comm_guard_required(package_version):
        return False

    marker = "TTT_VLLM_DISABLE_BROKEN_FLASHINFER_COMM"
    os.environ[marker] = "1"
    os.environ["VLLM_ALLREDUCE_USE_FLASHINFER"] = "0"
    compat_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "vllm_compat")
    python_path = [
        entry for entry in os.environ.get("PYTHONPATH", "").split(os.pathsep)
        if entry
    ]
    if compat_dir not in python_path:
        os.environ["PYTHONPATH"] = os.pathsep.join([compat_dir, *python_path])

    # Covers this worker and forked children. sitecustomize covers children
    # created with multiprocessing's fresh-interpreter spawn method.
    for module_name in tuple(sys.modules):
        if (module_name == "flashinfer.comm"
                or module_name.startswith("flashinfer.comm.")):
            sys.modules.pop(module_name, None)
    sys.modules["flashinfer.comm"] = None
    print(f"[vllm] FlashInfer {package_version} communication disabled for "
          f"Python {sys.version_info.major}.{sys.version_info.minor}; using "
          "vLLM NCCL/custom all-reduce", flush=True)
    return True


def _redirect_vllm_output(log_path, rank, gpu_group):
    """Send this worker and all engine-core descendants to one append log."""
    if not log_path:
        return None
    log_path = os.path.abspath(os.fspath(log_path))
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception:
        pass
    handle = open(log_path, "a", buffering=1, encoding="utf-8")
    os.dup2(handle.fileno(), 1)
    os.dup2(handle.fileno(), 2)
    print(f"\n=== vLLM worker {rank} pid={os.getpid()} "
          f"physical_gpus={list(gpu_group)} ===", flush=True)
    return handle


def _vllm_worker_loop(rank, gpu_id, model_name, max_seq_length, load_in_4bit,
                      task_queue, result_queue, ready_queue, seed=None,
                      lora_rank=32, gpu_memory_utilization=0.9,
                      enforce_eager=False, enable_prefix_caching=True,
                      gen_micro_batch=0, quantization=None,
                      tensor_parallel_size=1, pipeline_parallel_size=1,
                      max_num_batched_tokens=0,
                      enable_expert_parallel=False, enable_sleep_mode=False,
                      sleep_level=1, control_queue=None, vllm_log_path=None,
                      disable_custom_all_reduce=False):
    """Persistent vLLM engine spanning one physical GPU group."""
    gpu_group = ([int(x) for x in gpu_id]
                 if isinstance(gpu_id, (list, tuple)) else [int(gpu_id)])
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(x) for x in gpu_group)
    # Redirect before importing vLLM/torch. Engine-core subprocesses inherit
    # stdout/stderr and therefore append to the same run-local file.
    _worker_log_handle = _redirect_vllm_output(
        vllm_log_path, rank, gpu_group)
    # Must happen before importing vLLM (and therefore torch). Spawned engine
    # core children inherit this worker-specific environment.
    if enable_sleep_mode:
        _prepare_vllm_sleep_allocator_env()
    # FlashInfer sampling defaults on in recent vLLM releases and may try to
    # compile at runtime with nvcc. Keep the dependency-free native sampler as
    # the default while preserving an explicit operator override.
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    _prepare_flashinfer_comm_compat()
    if seed is not None:
        # vLLM documents this setting for deterministic V1 offline inference.
        os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

    try:
        from vllm import LLM, SamplingParams
        from vllm.lora.request import LoRARequest
        sleep_api_available = bool(
            hasattr(LLM, "sleep") and hasattr(LLM, "wake_up"))

        engine_seed = (worker_seed(seed, 0, rank) if seed is not None
                       else int.from_bytes(os.urandom(4), "little") % (2 ** 31 - 1))
        engine_kwargs = _vllm_engine_kwargs(
            model_name=model_name,
            max_seq_length=max_seq_length,
            load_in_4bit=load_in_4bit,
            lora_rank=lora_rank,
            gpu_memory_utilization=gpu_memory_utilization,
            enforce_eager=enforce_eager,
            enable_prefix_caching=enable_prefix_caching,
            max_num_seqs=gen_micro_batch,
            seed=engine_seed,
            quantization=quantization,
            tensor_parallel_size=tensor_parallel_size,
            pipeline_parallel_size=pipeline_parallel_size,
            max_num_batched_tokens=max_num_batched_tokens,
            enable_expert_parallel=enable_expert_parallel,
            # Older vLLM builds do not accept this constructor option. They
            # start normally and report no sleep capability so the parent can
            # select its transient compatibility path.
            enable_sleep_mode=(enable_sleep_mode and sleep_api_available),
            disable_custom_all_reduce=disable_custom_all_reduce,
        )
        print(f"[vllm worker {rank}] loading {model_name} on physical GPU "
              f"group {gpu_group} (TP={tensor_parallel_size}, "
              f"PP={pipeline_parallel_size}, collectives="
              f"{'NCCL' if disable_custom_all_reduce else 'auto'}) ...",
              flush=True)
        llm = LLM(**engine_kwargs)
    except Exception:
        detail = traceback.format_exc()
        if vllm_log_path:
            print(f"[vllm worker {rank}] startup failed:\n{detail}",
                  file=sys.stderr, flush=True)
            detail = f"details: {os.path.abspath(os.fspath(vllm_log_path))}"
        ready_queue.put(("error", rank, detail))
        return

    # Paths are versioned (adapter_step000, adapter_step001, ...). Giving each
    # path a unique positive id prevents vLLM from serving a stale cached LoRA.
    adapter_ids = {}
    next_adapter_id = 1

    def _lora_request(adapter_path):
        nonlocal next_adapter_id
        if adapter_path is None:
            return None
        adapter_key = os.path.realpath(os.fspath(adapter_path))
        if adapter_key not in adapter_ids:
            adapter_ids[adapter_key] = next_adapter_id
            next_adapter_id += 1
        adapter_id = adapter_ids[adapter_key]
        return LoRARequest(
            f"ttt_adapter_{adapter_id}", adapter_id, adapter_key)

    sleep_level = int(sleep_level)
    basic_sleep = bool(hasattr(llm, "sleep") and hasattr(llm, "wake_up"))
    # Level 2 avoids keeping another full copy of every replica's weights in
    # host RAM. Waking it requires reloading the discarded base weights.
    deep_sleep = bool(
        basic_sleep and sleep_level == 2
        and hasattr(llm, "collective_rpc"))
    can_sleep = bool(basic_sleep and (sleep_level == 1 or deep_sleep))
    ready_queue.put((
        "ready", rank, f"sleep:{sleep_level}" if can_sleep else ""))

    while True:
        task = task_queue.get()
        if task is None:
            break
        if (isinstance(task, tuple) and len(task) >= 2
                and task[0] == "__control__"):
            command = task[1]
            try:
                if not can_sleep:
                    raise RuntimeError(
                        "installed vLLM does not expose LLM.sleep/wake_up")
                if command == "sleep":
                    print(f"[vllm worker {rank}] entering sleep level "
                          f"{sleep_level}", flush=True)
                    llm.sleep(level=sleep_level)
                elif command == "wake_up":
                    print(f"[vllm worker {rank}] waking from sleep level "
                          f"{sleep_level}", flush=True)
                    if sleep_level == 2:
                        llm.wake_up(tags=["weights"])
                        llm.collective_rpc("reload_weights")
                        llm.wake_up(tags=["kv_cache"])
                    else:
                        llm.wake_up()
                else:
                    raise ValueError(f"unknown vLLM control command {command!r}")
                print(f"[vllm worker {rank}] {command} complete", flush=True)
                if control_queue is not None:
                    control_queue.put(("ok", rank, command, ""))
            except Exception:
                detail = traceback.format_exc()
                if vllm_log_path:
                    print(f"[vllm worker {rank}] {command} failed:\n{detail}",
                          file=sys.stderr, flush=True)
                    detail = ("details: "
                              f"{os.path.abspath(os.fspath(vllm_log_path))}")
                if control_queue is not None:
                    control_queue.put(
                        ("error", rank, command, detail))
            continue
        if (isinstance(task, tuple) and len(task) in (2, 3)
                and task[0] == "__score__"):
            score_jobs = task[1]
            score_adapter_path = task[2] if len(task) == 3 else None
            if not score_jobs:
                continue
            try:
                prompts = []
                score_params = []
                exact_limit = []
                for _request_idx, prompt_ids, response_ids in score_jobs:
                    tokens = list(prompt_ids) + list(response_ids)
                    at_limit = len(tokens) == int(max_seq_length)
                    # Offline vLLM requires at least one generated token. For
                    # an exact-limit sequence, leave the observed final token
                    # out of the prompt and request that specific token's
                    # unmodified base-model logprob at the generated position.
                    # Requesting the token explicitly is exact and avoids the
                    # full-vocabulary `logprobs=-1` payload, which vLLM rejects
                    # when its configured max_logprobs is smaller than the
                    # vocabulary. This scores all observed tokens without
                    # exceeding max_model_len.
                    prompt_tokens = tokens[:-1] if at_limit else tokens
                    prompts.append({"prompt_token_ids": prompt_tokens})
                    kwargs = dict(
                        max_tokens=1,
                        temperature=0.0,
                        prompt_logprobs=0,
                        detokenize=False,
                    )
                    if at_limit:
                        kwargs["logprob_token_ids"] = [
                            int(response_ids[-1])
                        ]
                    score_params.append(SamplingParams(**kwargs))
                    exact_limit.append(at_limit)
                outputs = llm.generate(
                    prompts,
                    sampling_params=score_params,
                    lora_request=_lora_request(score_adapter_path),
                    use_tqdm=False,
                )
                scored = []
                for job_pos, (request_idx, prompt_ids, response_ids) in enumerate(
                        score_jobs):
                    request_output = (outputs[job_pos]
                                      if job_pos < len(outputs) else None)
                    values = None
                    if request_output is not None:
                        prompt_logprobs = getattr(
                            request_output, "prompt_logprobs", None)
                        start = len(prompt_ids)
                        prompt_response_count = (
                            len(response_ids) - 1
                            if exact_limit[job_pos] else len(response_ids))
                        end = start + prompt_response_count
                        if (prompt_logprobs is not None
                                and len(prompt_logprobs) >= end):
                            values = _chosen_token_logprobs(
                                response_ids[:prompt_response_count],
                                prompt_logprobs[start:end])
                        if values is not None and exact_limit[job_pos]:
                            candidates = None
                            generated = getattr(
                                request_output, "outputs", None) or []
                            if generated:
                                generated_logprobs = getattr(
                                    generated[0], "logprobs", None)
                                if generated_logprobs:
                                    candidates = generated_logprobs[:1]
                            final_value = _chosen_token_logprobs(
                                response_ids[-1:], candidates)
                            values = (
                                values + final_value
                                if final_value is not None else None)
                    scored.append((
                        request_idx,
                        array("f", values) if values is not None else None,
                    ))
                result_queue.put((rank, "__score__", scored))
            except Exception:
                detail = traceback.format_exc()
                print(f"[vllm worker {rank}] reference scoring failed:\n"
                      f"{detail}", file=sys.stderr, flush=True)
                if vllm_log_path:
                    detail = ("details: "
                              f"{os.path.abspath(os.fspath(vllm_log_path))}")
                result_queue.put((rank, None, {"error": detail}))
            continue
        step, adapter_path, jobs, gen_kwargs = task
        if not jobs:
            continue

        try:
            lora_request = _lora_request(adapter_path)

            # Dispatch in scheduler-sized waves.  A single blocking generate()
            # over the worker's entire assignment withheld every completed
            # rollout until the slowest sequence in that assignment finished,
            # so CPU verification could not overlap generation in practice.
            # Each wave still mixes prompts and fills max_num_seqs, but its
            # results are returned immediately before the next wave begins.
            tokenizer = llm.get_tokenizer()
            runnable_jobs = []
            for group_idx, prompt, count in jobs:
                prompt_len = len(tokenizer.encode(prompt))
                max_tokens = min(
                    int(gen_kwargs["max_new_tokens"]),
                    int(max_seq_length) - int(prompt_len),
                )
                if max_tokens < 1:
                    print(f"[vllm worker {rank}] group {group_idx} prompt is "
                          f"{prompt_len} tokens, at/over max_model_len="
                          f"{max_seq_length}; dropping {count} rollout(s)",
                          flush=True)
                    result_queue.put(
                        (rank, group_idx, [("", []) for _ in range(int(count))]))
                    continue
                runnable_jobs.append((
                    int(group_idx), prompt, int(count), int(max_tokens)))

            if not runnable_jobs:
                continue

            return_logprobs = bool(gen_kwargs.get("return_logprobs", False))
            remaining = [job[2] for job in runnable_jobs]
            emitted = [0 for _job in runnable_jobs]
            total_remaining = sum(remaining)
            wave_limit = int(gen_kwargs.get("micro_batch", 0) or 0)
            if wave_limit <= 0:
                wave_limit = total_remaining
            cursor = 0

            while total_remaining > 0:
                wave_size = min(wave_limit, total_remaining)
                allocation = [0 for _job in runnable_jobs]
                probe = cursor
                for _slot in range(wave_size):
                    while remaining[probe] <= allocation[probe]:
                        probe = (probe + 1) % len(runnable_jobs)
                    allocation[probe] += 1
                    probe = (probe + 1) % len(runnable_jobs)
                cursor = probe

                wave_indices = [
                    index for index, count in enumerate(allocation) if count]
                prompts = [runnable_jobs[index][1]
                           for index in wave_indices]
                sampling = []
                for index in wave_indices:
                    group_idx, _prompt, _count, max_tokens = (
                        runnable_jobs[index])
                    sampling_kwargs = dict(
                        n=int(allocation[index]),
                        max_tokens=int(max_tokens),
                        temperature=float(gen_kwargs["temperature"]),
                        top_p=float(gen_kwargs["top_p"]),
                        seed=_vllm_job_seed(
                            seed, step, rank, group_idx,
                            sample_offset=emitted[index]),
                        skip_special_tokens=True,
                    )
                    if return_logprobs:
                        # vLLM always includes the sampled token in logprobs;
                        # zero asks for no additional top-k entries.
                        sampling_kwargs["logprobs"] = 0
                    sampling.append(SamplingParams(**sampling_kwargs))

                outputs = llm.generate(
                    prompts,
                    sampling_params=sampling,
                    lora_request=lora_request,
                    use_tqdm=False,
                )

                for wave_pos, index in enumerate(wave_indices):
                    group_idx, _prompt, _count, _max_tokens = (
                        runnable_jobs[index])
                    requested = int(allocation[index])
                    request_output = (
                        outputs[wave_pos] if wave_pos < len(outputs) else None)
                    job_results = []
                    if request_output is not None:
                        for candidate in request_output.outputs[:requested]:
                            token_ids = list(candidate.token_ids)
                            if return_logprobs:
                                values = _chosen_token_logprobs(
                                    token_ids,
                                    getattr(candidate, "logprobs", None))
                                job_results.append(
                                    (candidate.text, token_ids,
                                     array("f", values)
                                     if values is not None else None))
                            else:
                                job_results.append(
                                    (candidate.text, token_ids))
                    # Preserve the queue contract even if an engine/version
                    # returns fewer samples than requested; downstream treats
                    # placeholders as invalid instead of blocking forever.
                    if len(job_results) < requested:
                        missing = requested - len(job_results)
                        placeholder = (("", [], None) if return_logprobs
                                       else ("", []))
                        job_results.extend(
                            [placeholder for _ in range(missing)])
                    result_queue.put((rank, group_idx, job_results))
                    remaining[index] -= requested
                    emitted[index] += requested
                    total_remaining -= requested
        except Exception:
            detail = traceback.format_exc()
            print(f"[vllm worker {rank}] generation failed:\n{detail}",
                  file=sys.stderr, flush=True)
            if vllm_log_path:
                detail = ("details: "
                          f"{os.path.abspath(os.fspath(vllm_log_path))}")
            result_queue.put((rank, None, {"error": detail}))

    print(f"[vllm worker {rank}] shutting down", flush=True)


def _assert_vllm_memory_available(gpu_ids, utilization, model_name):
    """Fail before spawning vLLM if stale allocations exceed its budget."""
    try:
        from gpu_runtime import query_gpu_memory
        memory = query_gpu_memory()
    except Exception:
        return
    if not memory:
        return
    deficits = []
    for gpu_id in gpu_ids:
        item = memory.get(int(gpu_id))
        if item is None:
            continue
        required = float(item.total_gib) * float(utilization)
        if float(item.free_gib) + 0.25 < required:
            deficits.append(
                f"GPU {gpu_id}: free={item.free_gib:.1f} GiB, "
                f"required={required:.1f} GiB")
    if deficits:
        raise RuntimeError(
            f"refusing to load vLLM model {model_name}: trainer/model "
            "offload did not release the configured GPU budget ("
            + "; ".join(deficits) + ")")


def _vllm_utilization_with_reserve(utilization, total_gib, reserve_gib):
    """Cap an allocator fraction while leaving an absolute per-GPU reserve."""
    utilization = float(utilization)
    total_gib = float(total_gib)
    reserve_gib = float(reserve_gib)
    if total_gib <= 0.0:
        return utilization
    cap = (total_gib - reserve_gib) / total_gib
    if cap <= 0.0:
        raise RuntimeError(
            f"cannot reserve {reserve_gib:.1f} GiB on a {total_gib:.1f}-GiB "
            "GPU for co-resident vLLM sleep state")
    return min(utilization, cap)


def _cap_vllm_utilization_for_gpus(utilization, gpu_ids, *, reserve_gib,
                                   reserve_from_free=False):
    """Apply one conservative allocator fraction across a TP/PP GPU group."""
    try:
        from gpu_runtime import query_gpu_memory
        memory = query_gpu_memory()
    except Exception:
        return float(utilization)
    caps = []
    for gpu_id in gpu_ids:
        item = memory.get(int(gpu_id))
        if item is None:
            continue
        available = (float(item.free_gib) if reserve_from_free
                     else float(item.total_gib))
        caps.append(_vllm_utilization_with_reserve(
            utilization,
            float(item.total_gib),
            float(item.total_gib) - available + float(reserve_gib),
        ))
    return min(caps, default=float(utilization))


class GenerationPool:
    """
    Manages persistent generation processes. HF uses one engine per GPU. vLLM
    partitions gpu_ids into TP*PP groups and loads one sharded engine per group.

    backend="hf" preserves the original Transformers workers. backend="vllm"
    uses vLLM for rollout inference while keeping exactly the same queue API.
    """

    def __init__(self, model_name, num_workers, gpu_ids=None,
                 max_seq_length=4096, load_in_4bit=False, seed=None,
                 gen_micro_batch=0, backend="hf", lora_rank=32,
                 vllm_gpu_memory_utilization=0.9,
                 vllm_enforce_eager=False,
                 vllm_enable_prefix_caching=True,
                 vllm_quantization=None,
                 vllm_tensor_parallel_size=0,
                 vllm_pipeline_parallel_size=1,
                 vllm_max_num_batched_tokens=0,
                 vllm_enable_expert_parallel=False,
                 vllm_enable_sleep_mode=False,
                 vllm_sleep_level=1,
                 vllm_persistent_workers=None,
                 vllm_co_resident_sleep=False,
                 vllm_token_scoring=False,
                 vllm_staged_loading=False,
                 vllm_log_path=None):
        self.model_name = model_name
        requested_num_gpus = int(num_workers)
        self.gpu_ids = gpu_ids or list(range(num_workers))
        self.seed = seed
        self.max_seq_length = int(max_seq_length)
        self.gen_micro_batch = int(gen_micro_batch or 0)
        self.backend = str(backend).lower()
        requested_vllm_utilization = float(vllm_gpu_memory_utilization)
        self.vllm_co_resident_sleep = bool(vllm_co_resident_sleep)
        self.vllm_token_scoring = bool(vllm_token_scoring)
        self.vllm_gpu_memory_utilization = requested_vllm_utilization
        self.sleep_requested = bool(vllm_enable_sleep_mode)
        self.sleep_level = int(vllm_sleep_level)
        if self.sleep_level not in (1, 2):
            raise ValueError("vllm_sleep_level must be 1 or 2")
        self.vllm_staged_loading = bool(vllm_staged_loading)
        self.sleep_supported = False
        self.vllm_log_path = (os.path.abspath(os.fspath(vllm_log_path))
                              if vllm_log_path else None)
        if self.backend not in ("hf", "vllm"):
            raise ValueError(
                f"unknown generation backend {backend!r}; expected hf|vllm")

        effective_enforce_eager = bool(vllm_enforce_eager)
        if self.backend == "vllm":
            effective_enforce_eager, eager_fallback = (
                _resolve_vllm_enforce_eager(effective_enforce_eager))
            if eager_fallback:
                header_path, _available = _python_development_header_path()
                python_dev_package = (
                    f"python{sys.version_info.major}.{sys.version_info.minor}-dev")
                print(
                    f"[vllm] {header_path} is missing; enabling eager mode "
                    "because vLLM TorchInductor cannot compile without "
                    f"Python.h. Install {python_dev_package} to re-enable "
                    "compiled execution.",
                    flush=True,
                )
        if len(self.gpu_ids) != requested_num_gpus:
            raise ValueError("gpu_ids must contain exactly num_workers entries")
        if len(set(self.gpu_ids)) != len(self.gpu_ids):
            raise ValueError("gpu_ids must not contain duplicates")

        reserve_gib = 0.0
        reserve_reasons = []
        if self.vllm_co_resident_sleep:
            reserve_gib = max(
                reserve_gib, VLLM_CORESIDENT_SLEEP_RESERVE_GIB)
            reserve_reasons.append("shared sleep residency")
        if self.vllm_token_scoring:
            reserve_gib = max(
                reserve_gib, VLLM_TOKEN_SCORING_RESERVE_GIB)
            reserve_reasons.append("exact token scoring")
        if self.backend == "vllm" and reserve_gib > 0.0:
            self.vllm_gpu_memory_utilization = (
                _cap_vllm_utilization_for_gpus(
                    requested_vllm_utilization,
                    self.gpu_ids,
                    reserve_gib=reserve_gib,
                ))
            if (self.vllm_gpu_memory_utilization
                    < requested_vllm_utilization - 1e-9):
                print(
                    f"[pool] {' + '.join(reserve_reasons)}: reserving at "
                    f"least {reserve_gib:.1f} GiB/GPU; "
                    "vLLM utilization "
                    f"{requested_vllm_utilization:.3f} -> "
                    f"{self.vllm_gpu_memory_utilization:.3f}",
                    flush=True,
                )

        effective_max_batched_tokens = int(
            vllm_max_num_batched_tokens or 0)
        if (self.backend == "vllm" and self.vllm_token_scoring
                and (effective_max_batched_tokens <= 0
                     or effective_max_batched_tokens
                     > VLLM_TOKEN_SCORING_MAX_BATCHED_TOKENS)):
            configured = ("automatic" if effective_max_batched_tokens <= 0
                          else str(effective_max_batched_tokens))
            effective_max_batched_tokens = (
                VLLM_TOKEN_SCORING_MAX_BATCHED_TOKENS)
            print(
                "[pool] exact token scoring: max batched tokens "
                f"{configured} -> {effective_max_batched_tokens} to bound "
                "the temporary vocabulary projection",
                flush=True,
            )

        if self.backend == "vllm":
            pp = int(vllm_pipeline_parallel_size or 1)
            tp = int(vllm_tensor_parallel_size or 0)
            if pp < 1 or tp < 0:
                raise ValueError("vLLM TP must be >= 0 and PP must be >= 1")
            if tp == 0:
                if requested_num_gpus % pp:
                    raise ValueError(
                        "generation GPU count must be divisible by vLLM PP")
                tp = requested_num_gpus // pp
            world_size = tp * pp
            if world_size < 1 or requested_num_gpus % world_size:
                raise ValueError(
                    f"{requested_num_gpus} generation GPUs cannot be split into "
                    f"vLLM groups of TP={tp} * PP={pp} ({world_size} GPUs)")
            gpu_groups = [
                self.gpu_ids[i:i + world_size]
                for i in range(0, requested_num_gpus, world_size)
            ]
        else:
            tp, pp = 1, 1
            gpu_groups = [[gpu_id] for gpu_id in self.gpu_ids]

        # Jobs are distributed over independent engines, not over TP ranks.
        self.num_workers = len(gpu_groups)
        configured_persistent_workers = (
            None if vllm_persistent_workers is None
            else int(vllm_persistent_workers))
        if (configured_persistent_workers is not None
                and not 0 <= configured_persistent_workers <= self.num_workers):
            raise ValueError(
                "vllm_persistent_workers must be between zero and the "
                f"number of vLLM engines ({self.num_workers})")
        if not self.sleep_requested:
            persistent_workers = 0
        elif configured_persistent_workers is None:
            persistent_workers = self.num_workers
        else:
            persistent_workers = configured_persistent_workers
        if (self.backend != "vllm"
                and configured_persistent_workers not in (None, 0)):
            raise ValueError(
                "vllm_persistent_workers is valid only for backend='vllm'")
        self.persistent_workers = persistent_workers
        self._persistent_worker_ids = tuple(range(persistent_workers))
        self._transient_worker_ids = tuple(
            range(persistent_workers, self.num_workers))

        ctx = mp.get_context("spawn")
        self._ctx = ctx
        self._gpu_groups = gpu_groups
        self.task_queues = [None for _ in range(self.num_workers)]
        self.result_queue = ctx.Queue()
        self._ready_queue = ctx.Queue()
        self.control_queue = ctx.Queue()

        if self.backend == "vllm":
            _assert_vllm_memory_available(
                self.gpu_ids, self.vllm_gpu_memory_utilization, model_name)

        self.procs = [None for _ in range(self.num_workers)]
        self._worker_target = (_vllm_worker_loop
                               if self.backend == "vllm" else _hf_worker_loop)
        self._worker_options = {
            "lora_rank": int(lora_rank),
            "gpu_memory_utilization": float(
                self.vllm_gpu_memory_utilization),
            "enforce_eager": effective_enforce_eager,
            "enable_prefix_caching": bool(vllm_enable_prefix_caching),
            "gen_micro_batch": self.gen_micro_batch,
            "quantization": vllm_quantization,
            "tensor_parallel_size": tp,
            "pipeline_parallel_size": pp,
            "max_num_batched_tokens": effective_max_batched_tokens,
            "enable_expert_parallel": bool(vllm_enable_expert_parallel),
            # Separate TP engines remap different physical GPU pairs to the
            # same local CUDA ordinals. vLLM 0.28's custom all-reduce IPC can
            # then fail while the second engine profiles CUDA graphs. Use the
            # exact NCCL collective for this layout. Single TP engines and
            # ordinary one-GPU replicas retain vLLM's faster automatic path.
            "disable_custom_all_reduce": bool(
                self.backend == "vllm"
                and self.num_workers > 1
                and tp > 1),
            "sleep_level": self.sleep_level,
            "control_queue": self.control_queue,
            "vllm_log_path": self.vllm_log_path,
        }
        self._load_in_4bit = bool(load_in_4bit)
        self._startup_timeout = _positive_env_timeout(
            "TTT_VLLM_STARTUP_TIMEOUT_S", 900.0)
        self._sleep_capable_workers = set()
        self._ready_worker_ids = set()

        # Large first loads are optionally serialized by independent engine.
        # This prevents concurrent checkpoint materialization/page-cache spikes
        # from exhausting host RAM before steady-state GPU residency is reached.
        staged = bool(
            self.backend == "vllm" and self.vllm_staged_loading
            and self.num_workers > 1)
        mode = " one at a time" if staged else ""
        print(f"[pool] loading {self.num_workers} {self.backend} engine(s){mode} "
              f"on {requested_num_gpus} GPU(s) ...", flush=True)
        self._staged_loading = staged
        self._start_workers(range(self.num_workers), initial=True)
        self.sleep_supported = bool(
            self.backend == "vllm" and self.sleep_requested
            and self.persistent_workers > 0
            and set(self._persistent_worker_ids).issubset(
                self._sleep_capable_workers))

    def _start_worker(self, rank):
        rank = int(rank)
        current = self.procs[rank]
        if current is not None and current.is_alive():
            raise RuntimeError(f"generation worker {rank} is already alive")
        task_queue = self._ctx.Queue()
        self.task_queues[rank] = task_queue
        worker_options = dict(self._worker_options)
        startup_utilization = _cap_vllm_utilization_for_gpus(
            worker_options["gpu_memory_utilization"],
            self._gpu_groups[rank],
            reserve_gib=VLLM_STARTUP_FREE_MARGIN_GIB,
            reserve_from_free=True,
        )
        if (startup_utilization
                < worker_options["gpu_memory_utilization"] - 1e-9):
            print(
                f"[pool] worker {rank}: current free memory lowers vLLM "
                f"utilization {worker_options['gpu_memory_utilization']:.3f} "
                f"-> {startup_utilization:.3f}",
                flush=True,
            )
            worker_options["gpu_memory_utilization"] = startup_utilization
        worker_options["enable_sleep_mode"] = bool(
            self.sleep_requested
            and rank in self._persistent_worker_ids)
        process = self._ctx.Process(
            target=self._worker_target,
            args=(
                rank,
                (self._gpu_groups[rank] if self.backend == "vllm"
                 else self._gpu_groups[rank][0]),
                self.model_name,
                self.max_seq_length,
                self._load_in_4bit,
                task_queue,
                self.result_queue,
                self._ready_queue,
                self.seed,
            ),
            kwargs=worker_options,
            # vLLM may manage child processes depending on its version and
            # engine settings; Python daemonic processes cannot do that.
            daemon=(self.backend != "vllm"),
        )
        self.procs[rank] = process
        process.start()

    def _wait_workers(self, worker_ids):
        pending = {int(rank) for rank in worker_ids}
        deadline = time.monotonic() + self._startup_timeout
        while pending:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self.shutdown()
                raise TimeoutError(
                    f"vLLM startup exceeded {self._startup_timeout:.0f}s "
                    f"while waiting for workers {sorted(pending)}; details: "
                    f"{self.vllm_log_path or 'console'}")
            try:
                status, rank, detail = self._ready_queue.get(
                    timeout=min(1.0, remaining))
            except queue.Empty:
                dead = [
                    (rank, None if self.procs[rank] is None
                     else self.procs[rank].exitcode)
                    for rank in pending
                    if (self.procs[rank] is None
                        or self.procs[rank].exitcode is not None)
                ]
                if dead:
                    self.shutdown()
                    raise RuntimeError(
                        "generation worker(s) exited during startup: "
                        f"{dead}")
                continue
            rank = int(rank)
            if rank not in pending:
                self.shutdown()
                raise RuntimeError(
                    f"unexpected startup acknowledgement from worker {rank}")
            if status == "error":
                self.shutdown()
                raise RuntimeError(
                    f"{self.backend} generation worker {rank} failed to "
                    f"start:\n{detail}")
            if str(detail).startswith("sleep:"):
                self._sleep_capable_workers.add(rank)
            self._ready_worker_ids.add(rank)
            pending.remove(rank)
            print(f"[pool] {len(self._ready_worker_ids)}/"
                  f"{self.num_workers} workers ready", flush=True)

    def _start_workers(self, worker_ids, *, initial=False):
        worker_ids = tuple(int(rank) for rank in worker_ids)
        if not worker_ids:
            return
        if not initial:
            print(f"[pool] reloading {len(worker_ids)} transient "
                  f"{self.backend} engine(s)"
                  + (" one at a time" if self._staged_loading else "")
                  + " ...", flush=True)
        if self._staged_loading:
            for rank in worker_ids:
                self._start_worker(rank)
                self._wait_workers((rank,))
        else:
            for rank in worker_ids:
                self._start_worker(rank)
            self._wait_workers(worker_ids)

    def _stop_workers(self, worker_ids):
        worker_ids = tuple(int(rank) for rank in worker_ids)
        for rank in worker_ids:
            process = self.procs[rank]
            task_queue = self.task_queues[rank]
            if process is not None and process.is_alive() and task_queue is not None:
                try:
                    task_queue.put(None)
                except Exception:
                    pass
        for rank in worker_ids:
            process = self.procs[rank]
            task_queue = self.task_queues[rank]
            if process is not None:
                process.join(timeout=10)
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=10)
                if process.is_alive():
                    process.kill()
                    process.join(timeout=5)
            self.procs[rank] = None
            self._ready_worker_ids.discard(rank)
            self._sleep_capable_workers.discard(rank)
            if task_queue is not None:
                try:
                    task_queue.close()
                    task_queue.join_thread()
                except (AttributeError, OSError, ValueError):
                    pass
            self.task_queues[rank] = None

    def _vllm_control(self, command, worker_ids=None):
        if self.backend != "vllm":
            raise RuntimeError("sleep/wake controls are only valid for vLLM")
        if not self.sleep_supported:
            raise RuntimeError("the installed vLLM engine lacks sleep mode")
        worker_ids = tuple(
            self._persistent_worker_ids
            if worker_ids is None else (int(rank) for rank in worker_ids))
        for rank in worker_ids:
            process = self.procs[rank]
            task_queue = self.task_queues[rank]
            if (process is None or not process.is_alive()
                    or task_queue is None):
                raise RuntimeError(
                    f"vLLM worker {rank} is unavailable during {command}")
            task_queue.put(("__control__", command))
        completed_workers = set()
        control_timeout = _positive_env_timeout(
            "TTT_VLLM_CONTROL_TIMEOUT_S", 600.0)
        control_deadline = time.monotonic() + control_timeout
        while len(completed_workers) < len(worker_ids):
            remaining = control_deadline - time.monotonic()
            if remaining <= 0:
                self.shutdown()
                raise TimeoutError(
                    f"vLLM {command} exceeded {control_timeout:.0f}s while "
                    f"waiting for {len(completed_workers)}/"
                    f"{len(worker_ids)} engines; "
                    f"details: {self.vllm_log_path or 'console'}")
            try:
                status, rank, got_command, detail = self.control_queue.get(
                    timeout=min(1.0, remaining))
            except queue.Empty:
                dead = [
                    (rank, None if self.procs[rank] is None
                     else self.procs[rank].exitcode)
                    for rank in worker_ids
                    if (self.procs[rank] is None
                        or self.procs[rank].exitcode is not None)
                ]
                if dead:
                    raise RuntimeError(
                        f"vLLM worker(s) exited during {command}: {dead}")
                continue
            if status == "error":
                raise RuntimeError(
                    f"vLLM worker {rank} failed to {got_command}:\n{detail}")
            if got_command != command:
                raise RuntimeError(
                    f"unexpected vLLM control acknowledgement {got_command!r}")
            if int(rank) not in worker_ids:
                raise RuntimeError(
                    f"unexpected vLLM control worker {rank} during {command}")
            if int(rank) in completed_workers:
                raise RuntimeError(
                    f"duplicate vLLM control acknowledgement from worker "
                    f"{rank} during {command}")
            completed_workers.add(int(rank))

    def sleep(self):
        # Stop transient engines first. Creating the sole persistent worker's
        # host-RAM backup afterwards avoids a temporary peak containing both
        # that backup and all transient engine processes.
        self._stop_workers(self._transient_worker_ids)
        self._vllm_control("sleep", self._persistent_worker_ids)

    def wake_up(self):
        if not self.vllm_co_resident_sleep:
            _assert_vllm_memory_available(
                self.gpu_ids, self.vllm_gpu_memory_utilization,
                self.model_name)
        # For co-resident pools, a full-budget preflight is mathematically
        # wrong: this pool's untagged CUDA/NCCL allocations are already part
        # of the used-memory reading, and wake_up only restores its released
        # tagged allocations. The fixed outside-vLLM reserve above protects
        # that delta while the other pool remains asleep.
        # Wake the RAM-resident engine before loading transient checkpoints so
        # its offloaded weights leave host memory as early as vLLM permits.
        self._vllm_control("wake_up", self._persistent_worker_ids)
        self._start_workers(self._transient_worker_ids)

    def iter_group_jobs(self, prompts_by_group, group_size, adapter_path,
                        max_new_tokens, temperature, top_p, step_idx=0,
                        show_progress=True, counts_by_group=None,
                        return_logprobs=False, progress_desc="rollouts"):
        """
        Stream generation results as each (worker, group) job completes.

        Yields (group_idx, [(text, token_ids), ...]) per job. The caller can
        dispatch each rollout for reward evaluation immediately, overlapping
        CPU eval with ongoing GPU generation.

        step_idx is passed through to the workers and keys their reseed, so
        pass the real step here. The memory maker passes step_idx + 1_000_000
        so its calls draw from a separate slot and leave the rollout stream
        untouched.

        Drives a "rollouts" progress bar over total rollouts (set
        show_progress=False to suppress). Stops after exactly total_expected
        rollouts, so every per-job message is drained and none leak into the
        next step's queue. A worker that hits an unrecoverable OOM emits empty
        placeholders for its dropped rollouts, so the count still reaches
        total_expected instead of hanging here.
        """
        worker_jobs = distribute_jobs(
            prompts_by_group, group_size, self.num_workers,
            counts_by_group=counts_by_group)
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "micro_batch": self.gen_micro_batch,
            "return_logprobs": bool(return_logprobs),
        }
        total_expected = sum(count for wj in worker_jobs for (_, _, count) in wj)

        # Dispatch one task per worker (some may have empty job lists)
        for r in range(self.num_workers):
            self.task_queues[r].put((step_idx, adapter_path, worker_jobs[r], gen_kwargs))

        collected = 0
        bar = (make_progress_bar(total_expected, desc=progress_desc)
               if show_progress else None)
        try:
            while collected < total_expected:
                try:
                    rank, group_idx, job_results = self.result_queue.get(timeout=1.0)
                except queue.Empty:
                    dead = [
                        (idx, None if proc is None else proc.exitcode)
                        for idx, proc in enumerate(self.procs)
                        if proc is None or proc.exitcode is not None]
                    if dead:
                        raise RuntimeError(
                            f"generation worker(s) exited during inference: {dead}")
                    continue
                if group_idx is None and isinstance(job_results, dict):
                    detail = job_results.get("error", "unknown worker failure")
                    raise RuntimeError(
                        f"generation worker {rank} failed during inference:\n"
                        f"{detail}")
                collected += len(job_results)
                if bar is not None:
                    bar.update(len(job_results))
                yield group_idx, job_results
        finally:
            if bar is not None:
                bar.close()

    def score_token_logprobs(self, prompt_response_pairs, show_progress=True,
                             adapter_path=None, require_complete=False,
                             max_retries=None):
        """Score observed response tokens with the base model or one LoRA.

        Calls are length-balanced over the already-awake engines. Omitting
        ``adapter_path`` scores the fixed base model. A failed engine is
        discarded, reloaded, and assigned only the still-missing trajectories.
        Results align with prompt_response_pairs; None marks an optional input
        that must fall back to HF. ``require_complete`` instead makes any
        irrecoverable omission fatal, as required for SPO-RS policy KL.
        """
        if self.backend != "vllm":
            raise RuntimeError("token scoring is only available on vLLM pools")
        pairs = list(prompt_response_pairs)
        scores = [None] * len(pairs)
        scorable = []
        skipped = []
        for request_idx, (prompt_ids, response_ids) in enumerate(pairs):
            total = len(prompt_ids) + len(response_ids)
            # Exact-limit inputs are handled by the worker by moving their last
            # observed token into the one generated scoring position.
            if not response_ids or total > self.max_seq_length:
                skipped.append(request_idx)
                continue
            scorable.append(request_idx)

        if require_complete and skipped:
            raise ValueError(
                "exact vLLM token scoring cannot score "
                f"{len(skipped)}/{len(pairs)} empty or overlength responses")
        if not scorable:
            return scores

        score_label = ("vllm policy logprobs" if adapter_path is not None
                       else "vllm reference logprobs")
        bar = (make_progress_bar(len(pairs), desc=score_label)
               if show_progress else None)
        if bar is not None and skipped:
            bar.update(len(skipped))

        def _balanced_jobs(request_indices):
            worker_jobs = [[] for _ in range(self.num_workers)]
            worker_loads = [0 for _ in range(self.num_workers)]
            ordered = sorted(
                request_indices,
                key=lambda idx: len(pairs[idx][0]) + len(pairs[idx][1]),
                reverse=True,
            )
            for request_idx in ordered:
                prompt_ids, response_ids = pairs[request_idx]
                worker = min(
                    range(self.num_workers),
                    key=lambda idx: worker_loads[idx],
                )
                worker_jobs[worker].append((
                    request_idx, list(prompt_ids), list(response_ids)))
                worker_loads[worker] += len(prompt_ids) + len(response_ids)
            return worker_jobs

        def _dispatch(worker_jobs):
            waiting = {
                worker for worker, jobs in enumerate(worker_jobs) if jobs
            }
            failed_workers = set()
            for worker in waiting:
                self.task_queues[worker].put(
                    ("__score__", worker_jobs[worker], adapter_path))

            while waiting:
                try:
                    rank, group_idx, job_results = self.result_queue.get(
                        timeout=1.0)
                except queue.Empty:
                    dead = {
                        idx for idx in waiting
                        if (self.procs[idx] is None
                            or self.procs[idx].exitcode is not None)
                    }
                    if dead:
                        failed_workers.update(dead)
                        waiting.difference_update(dead)
                    continue

                rank = int(rank)
                if rank not in waiting:
                    raise RuntimeError(
                        "unexpected or duplicate vLLM token-scoring result "
                        f"from worker {rank}")
                waiting.remove(rank)
                if group_idx is None and isinstance(job_results, dict):
                    detail = job_results.get(
                        "error", "unknown worker failure")
                    print(
                        f"[warn] vLLM worker {rank} failed during exact "
                        f"token scoring ({detail})",
                        flush=True,
                    )
                    failed_workers.add(rank)
                    continue
                if group_idx != "__score__":
                    raise RuntimeError(
                        "unexpected generation result during token scoring")

                newly_scored = 0
                for request_idx, values in job_results:
                    request_idx = int(request_idx)
                    expected = len(pairs[request_idx][1])
                    if values is None or len(values) != expected:
                        continue
                    if scores[request_idx] is None:
                        newly_scored += 1
                    scores[request_idx] = values
                if bar is not None and newly_scored:
                    bar.update(newly_scored)
            return failed_workers

        try:
            pending = list(scorable)
            retries = max(0, int(
                2 if max_retries is None and require_complete
                else (max_retries or 0)))
            attempt = 0
            while pending:
                failed_workers = _dispatch(_balanced_jobs(pending))
                pending = [
                    request_idx for request_idx in scorable
                    if scores[request_idx] is None
                ]
                if not pending or attempt >= retries:
                    break
                attempt += 1
                if failed_workers:
                    failed = sorted(failed_workers)
                    print(
                        f"[pool] exact token scoring retry {attempt}/"
                        f"{retries}: restarting failed vLLM worker(s) "
                        f"{failed} and rescoring {len(pending)} missing "
                        "trajectories",
                        flush=True,
                    )
                    self._stop_workers(failed)
                    self._start_workers(failed)
                else:
                    print(
                        f"[pool] exact token scoring retry {attempt}/"
                        f"{retries}: rescoring {len(pending)} incomplete "
                        "trajectories",
                        flush=True,
                    )

            if require_complete and pending:
                raise RuntimeError(
                    "exact vLLM token scoring remained incomplete for "
                    f"{len(pending)}/{len(pairs)} trajectories after "
                    f"{attempt} retry attempt(s)")
        finally:
            if bar is not None:
                bar.close()
        return scores

    def generate_groups(self, prompts_by_group, group_size, adapter_path,
                        max_new_tokens, temperature, top_p, step_idx=0,
                        counts_by_group=None, return_logprobs=False):
        """
        Backward-compatible blocking variant. Returns:
          dict group_idx -> list of (text, token_ids, behavior_logprobs)
        The last item is None unless return_logprobs=True on a vLLM pool.
        Prefer iter_group_jobs() when you want to overlap reward evaluation
        with generation.
        """
        num_groups = len(prompts_by_group)
        by_group = {g: [] for g in range(num_groups)}
        for group_idx, job_results in self.iter_group_jobs(
                prompts_by_group, group_size, adapter_path,
                max_new_tokens, temperature, top_p, step_idx=step_idx,
                counts_by_group=counts_by_group,
                return_logprobs=return_logprobs):
            for item in job_results:
                text, token_ids = item[:2]
                behavior_logprobs = item[2] if len(item) > 2 else None
                by_group[group_idx].append(
                    (text, token_ids, behavior_logprobs))
        return by_group

    def shutdown(self):
        self._stop_workers(range(self.num_workers))


class HybridHFGenerationPool:
    """Use the live trainer plus persistent HF workers at the same time.

    The trainer GPU receives one fair share of every prompt.  Remaining shares
    are dispatched to workers on the other physical GPUs before local
    generation starts, so all cards generate concurrently without placing a
    duplicate base model beside the trainer.
    """

    sequential = False

    def __init__(self, remote_pool, local_iter):
        if remote_pool.backend != "hf":
            raise ValueError("HybridHFGenerationPool requires an HF worker pool")
        self._remote = remote_pool
        self._local_iter = local_iter
        self.num_workers = 1 + int(remote_pool.num_workers)

    def iter_group_jobs(self, prompts_by_group, group_size, adapter_path,
                        max_new_tokens, temperature, top_p, step_idx=0,
                        show_progress=True, counts_by_group=None,
                        progress_desc="rollouts"):
        counts = ([int(group_size)] * len(prompts_by_group)
                  if counts_by_group is None
                  else [int(value) for value in counts_by_group])
        if len(counts) != len(prompts_by_group):
            raise ValueError("counts_by_group must align with prompts_by_group")

        # This is the same fair split as distribute_jobs with local rank zero;
        # the remote pool then divides the remainder over its ranks.
        local_counts = []
        for group_idx, count in enumerate(counts):
            base, remainder = divmod(count, self.num_workers)
            local_counts.append(
                base + (1 if ((-group_idx) % self.num_workers) < remainder
                        else 0))
        remote_counts = [count - local for count, local
                         in zip(counts, local_counts)]
        events = queue.Queue()

        def run_remote():
            try:
                for item in self._remote.iter_group_jobs(
                        prompts_by_group, group_size, adapter_path,
                        max_new_tokens, temperature, top_p,
                        step_idx=step_idx, show_progress=False,
                        counts_by_group=remote_counts):
                    events.put(("result", item))
            except BaseException as exc:
                events.put(("error", exc))
            finally:
                events.put(("done", None))

        remote_thread = threading.Thread(
            target=run_remote, name="hf-rollout-workers", daemon=True)
        remote_thread.start()
        total_expected = sum(counts)
        bar = (make_progress_bar(total_expected, desc=progress_desc)
               if show_progress else None)
        remote_done = False

        def drain_remote(block=False):
            nonlocal remote_done
            while not remote_done:
                try:
                    kind, payload = events.get(
                        timeout=1.0 if block else None,
                        block=block,
                    )
                except queue.Empty:
                    return
                if kind == "done":
                    remote_done = True
                    return
                if kind == "error":
                    raise payload
                group_idx, results = payload
                if bar is not None:
                    bar.update(len(results))
                yield group_idx, results
                block = False

        try:
            local_kwargs = {
                "prompts_by_group": prompts_by_group,
                "counts_by_group": local_counts,
                "adapter_path": adapter_path,
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "step_idx": step_idx,
            }
            for group_idx, results in self._local_iter(**local_kwargs):
                yield from drain_remote(block=False)
                if bar is not None:
                    bar.update(len(results))
                yield group_idx, results
            while not remote_done:
                yield from drain_remote(block=True)
        finally:
            remote_thread.join()
            if bar is not None:
                bar.close()

    def generate_groups(self, prompts_by_group, group_size, adapter_path,
                        max_new_tokens, temperature, top_p, step_idx=0,
                        counts_by_group=None):
        by_group = {idx: [] for idx in range(len(prompts_by_group))}
        for group_idx, results in self.iter_group_jobs(
                prompts_by_group, group_size, adapter_path,
                max_new_tokens, temperature, top_p, step_idx=step_idx,
                counts_by_group=counts_by_group):
            by_group[group_idx].extend(
                (text, token_ids, None) for text, token_ids in results)
        return by_group

    def shutdown(self):
        self._remote.shutdown()


class PhasedVLLMGenerationPool:
    """Alternate shared-card vLLM residency with differentiable updates.

    The configured vLLM sleep level is acknowledged by every worker before
    the next phase begins. If sleep mode is unavailable, generation safely
    falls back to a transient engine. The trainer is never resident on the
    generation cards at the same time as an awake vLLM engine.
    """

    sequential = True

    def __init__(self, before_start, after_stop, **pool_kwargs):
        self._before_start = before_start
        self._after_stop = after_stop
        self._pool_kwargs = dict(pool_kwargs)
        requested_sleep_level = int(
            self._pool_kwargs.pop("vllm_sleep_level", 1))
        if requested_sleep_level not in (1, 2):
            raise ValueError("vllm_sleep_level must be 1 or 2")
        model_name = str(self._pool_kwargs.get("model_name", "")).lower()
        effective_quantization = _resolve_vllm_quantization(
            model_name,
            bool(self._pool_kwargs.get("load_in_4bit", False)),
            self._pool_kwargs.get("vllm_quantization"),
        )
        uses_bitsandbytes = bool(
            effective_quantization
            and effective_quantization.lower() == "bitsandbytes")
        # The lightweight debug models are small enough to retain level-1 CPU
        # weight backups.  This avoids rereading either checkpoint every step;
        # for GPT-OSS it also bypasses the broken native-MXFP4 level-2 reload.
        # BitsAndBytes gets the same treatment because vLLM explicitly does
        # not implement reload_weights for that loader.
        level_one_model = None
        if uses_bitsandbytes:
            level_one_model = "BitsAndBytes"
        elif "gpt-oss-20b" in model_name:
            level_one_model = "GPT-OSS-20B"
        elif "qwen2.5-coder-7b" in model_name:
            level_one_model = "Qwen2.5-Coder-7B"
        self._level_one_override_model = (
            level_one_model if requested_sleep_level == 2 else None)
        self._sleep_level = (
            1 if self._level_one_override_model is not None
            else requested_sleep_level)
        # Native GPT-OSS MXFP4 parameters are transformed while vLLM first
        # loads the model.  vLLM's level-2 wake path then tries to reload the
        # original parameter names and fails (for example, w2_bias is no
        # longer present).  A fresh process is exact and also releases the
        # residual CUDA allocations that level-2 sleep keeps alive.
        self._force_transient = bool(
            self._sleep_level == 2 and "gpt-oss" in model_name)
        self._pool = None
        self._persistent = False
        self._awake = False
        self._sleep_mode_announced = False
        self.num_workers = 1

    @property
    def active(self):
        return self._awake

    def _ensure_started(self):
        if self._awake:
            return self._pool
        self._before_start()
        try:
            if self._pool is None:
                # Start lazily at the first real rollout. The old eager probe
                # loaded every engine, immediately slept it, restored training,
                # then reloaded the same weights for step 0. More importantly,
                # a broken initial sleep could hang before the run even began.
                if self._force_transient:
                    self._pool = GenerationPool(**self._pool_kwargs)
                else:
                    self._pool = GenerationPool(
                        **self._pool_kwargs,
                        vllm_enable_sleep_mode=True,
                        vllm_sleep_level=self._sleep_level,
                    )
                self.num_workers = self._pool.num_workers
                self._persistent = bool(
                    not self._force_transient
                    and self._pool.sleep_supported)
                if not self._sleep_mode_announced:
                    if self._force_transient:
                        print(
                            "[pool] GPT-OSS level-2 weight reload is unsafe; "
                            "using a fresh transient vLLM pool for each phase",
                            flush=True,
                        )
                    elif self._persistent:
                        persistent_workers = int(getattr(
                            self._pool, "persistent_workers",
                            self.num_workers))
                        if persistent_workers < self.num_workers:
                            detail = (
                                f"{persistent_workers}/{self.num_workers} "
                                "engine(s) backed up in host RAM; remaining "
                                "engines reload each phase")
                        else:
                            detail = (
                                "weights discarded; no host-RAM backups"
                                if self._sleep_level == 2 else
                                "weights backed up in host RAM")
                        prefix = (
                            f"{self._level_one_override_model} persistent "
                            "vLLM sleep level 1"
                            if self._level_one_override_model is not None else
                            f"vLLM sleep level {self._sleep_level}")
                        print(f"[pool] {prefix} ready; {detail}", flush=True)
                    else:
                        print("[pool] installed vLLM lacks safe deep sleep; "
                              "using transient engines for phase sharing",
                              flush=True)
                    self._sleep_mode_announced = True
            elif self._persistent:
                try:
                    self._pool.wake_up()
                except Exception as exc:
                    # Some vLLM loaders advertise level-2 sleep but do not
                    # implement reload_weights. Recover in the same phase
                    # instead of losing the completed training step.
                    failed_pool, self._pool = self._pool, None
                    failed_pool.shutdown()
                    self._sleep_level = 1
                    self._force_transient = False
                    self._level_one_override_model = "reload-fallback"
                    print(
                        "[pool] level-2 vLLM wake failed "
                        f"({type(exc).__name__}); restarting once with "
                        "persistent level-1 sleep",
                        flush=True,
                    )
                    self._pool = GenerationPool(
                        **self._pool_kwargs,
                        vllm_enable_sleep_mode=True,
                        vllm_sleep_level=1,
                    )
                    self.num_workers = self._pool.num_workers
                    self._persistent = self._pool.sleep_supported
            else:
                self._pool = GenerationPool(**self._pool_kwargs)
                self.num_workers = self._pool.num_workers
            self._awake = True
            return self._pool
        except Exception:
            self._after_stop()
            raise

    def iter_group_jobs(self, *args, **kwargs):
        yield from self._ensure_started().iter_group_jobs(*args, **kwargs)

    def generate_groups(self, *args, **kwargs):
        return self._ensure_started().generate_groups(*args, **kwargs)

    def score_token_logprobs(self, *args, **kwargs):
        return self._ensure_started().score_token_logprobs(*args, **kwargs)

    def release(self):
        if not self._awake:
            return
        try:
            if self._persistent:
                try:
                    self._pool.sleep()
                except Exception as exc:
                    # A vLLM EngineCore can die after an inference OOM.  It
                    # cannot acknowledge sleep, but shutting down the complete
                    # pool still releases its remaining processes and makes a
                    # clean pool reload on the next phase safe.  Cleanup must
                    # not turn an already handled optional-score failure into
                    # a fatal training-step failure.
                    failed_pool, self._pool = self._pool, None
                    self._persistent = False
                    try:
                        failed_pool.shutdown()
                    except Exception as shutdown_error:
                        print(
                            "[pool] failed vLLM engine also raised during "
                            f"forced shutdown ({type(shutdown_error).__name__}); "
                            "continuing after process cleanup request",
                            flush=True,
                        )
                    print(
                        "[pool] vLLM sleep failed after an engine failure "
                        f"({type(exc).__name__}); discarded the pool and will "
                        "reload it next phase",
                        flush=True,
                    )
            else:
                pool, self._pool = self._pool, None
                pool.shutdown()
        finally:
            self._awake = False
            self._after_stop()

    def shutdown(self):
        if self._awake:
            self.release()
        if self._pool is not None:
            self._pool.shutdown()
            self._pool = None


class OnDemandGenerationPool:
    """A transient pool for a generation group that includes the trainer GPU.

    The callbacks move the differentiable model and optimizer off CUDA before
    vLLM starts, then restore them after rollout generation.  This keeps the
    two runtimes sequential instead of asking both allocators to coexist on one
    H100.  The public surface intentionally matches GenerationPool.
    """

    sequential = True

    def __init__(self, before_start, after_stop, **pool_kwargs):
        self._before_start = before_start
        self._after_stop = after_stop
        self._pool_kwargs = dict(pool_kwargs)
        self._pool = None
        self.num_workers = 1

    @property
    def active(self):
        return self._pool is not None

    def _ensure_started(self):
        if self._pool is not None:
            return self._pool
        self._before_start()
        try:
            self._pool = GenerationPool(**self._pool_kwargs)
        except Exception:
            self._after_stop()
            raise
        self.num_workers = self._pool.num_workers
        return self._pool

    def iter_group_jobs(self, *args, **kwargs):
        pool = self._ensure_started()
        yield from pool.iter_group_jobs(*args, **kwargs)

    def generate_groups(self, *args, **kwargs):
        return self._ensure_started().generate_groups(*args, **kwargs)

    def score_token_logprobs(self, *args, **kwargs):
        return self._ensure_started().score_token_logprobs(*args, **kwargs)

    def release(self):
        if self._pool is None:
            return
        pool, self._pool = self._pool, None
        try:
            pool.shutdown()
        finally:
            self._after_stop()
            self.num_workers = 1

    def shutdown(self):
        self.release()
