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

IMPORTANCE SAMPLING (fix 6) IS DISABLED FOR SPEED.
We do NOT compute per-token behavior logprobs; generation returns tokens only.
generate_groups() fills the trainer's logprob slot with None (read as on-policy,
IS ratio = 1). iter_group_jobs() yields plain (text, token_ids) pairs.

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
  worker[w] -> result_queue:      (rank, group_idx, [(text, token_ids), ...])
       one message PER JOB, so the pool can stream results as they land.
"""

import os
import queue
import sys
import threading
import traceback
import multiprocessing as mp

# tqdm ships with transformers/huggingface_hub, so it's almost always present.
# Fall back to a coarse print bar if it isn't, so nothing depends on it.
try:
    from tqdm import tqdm
    _HAS_TQDM = True
except Exception:
    tqdm = None
    _HAS_TQDM = False


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
            # No adapter yet (step 0 before any training) -> use base as-is
            return base
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

        cap_state = {"value": gen_cap}
        for group_idx, job_results in _iter_hf_job_batches(
                gen_model, tokenizer, jobs, device, max_seq_length, gen_kwargs,
                cap_state=cap_state, log_prefix=f"worker {rank}"):
            # Report every completed micro-batch so evaluation can overlap the
            # remaining GPU work.
            result_queue.put((rank, group_idx, job_results))
        gen_cap = int(cap_state["value"])

    print(f"[worker {rank}] shutting down", flush=True)


def _vllm_engine_kwargs(model_name, max_seq_length, load_in_4bit,
                        lora_rank, gpu_memory_utilization,
                        enforce_eager=False, enable_prefix_caching=True,
                        max_num_seqs=0, seed=None, quantization=None,
                        tensor_parallel_size=1, pipeline_parallel_size=1,
                        max_num_batched_tokens=0,
                        enable_expert_parallel=False,
                        enable_sleep_mode=False):
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
    if int(max_num_seqs or 0) > 0:
        kwargs["max_num_seqs"] = int(max_num_seqs)
    if int(max_num_batched_tokens or 0) > 0:
        kwargs["max_num_batched_tokens"] = int(max_num_batched_tokens)
        if int(max_num_batched_tokens) < int(max_seq_length):
            kwargs["enable_chunked_prefill"] = True
    if seed is not None:
        kwargs["seed"] = int(seed)
    # `load_in_4bit` belongs to the differentiable training copy. Deliberately
    # do not translate it to vLLM BitsAndBytes: a QLoRA trainer adapter is valid
    # on a BF16/FP8/MXFP4 inference base, and coupling the two modes was a common
    # source of vLLM startup crashes. Pre-quantized checkpoints are auto-detected.
    # Users who really want an explicit vLLM mode set vllm_quantization in YAML.
    del load_in_4bit
    quantization = str(quantization or "").strip()
    if quantization and quantization.lower() not in ("auto", "none"):
        kwargs["quantization"] = quantization
    return kwargs


def _vllm_job_seed(seed, step, rank, group_idx):
    """Stable per-request seed; None preserves vLLM's stochastic default."""
    if seed is None:
        return None
    return (worker_seed(seed, step, rank) + int(group_idx) * 104_729) % (2 ** 31 - 1)


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
                      control_queue=None, vllm_log_path=None):
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
        )
        print(f"[vllm worker {rank}] loading {model_name} on physical GPU "
              f"group {gpu_group} (TP={tensor_parallel_size}, "
              f"PP={pipeline_parallel_size}) ...", flush=True)
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
    can_sleep = bool(hasattr(llm, "sleep") and hasattr(llm, "wake_up"))
    ready_queue.put(("ready", rank, "sleep" if can_sleep else ""))

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
                    llm.sleep(level=1)
                elif command == "wake_up":
                    llm.wake_up()
                else:
                    raise ValueError(f"unknown vLLM control command {command!r}")
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
        step, adapter_path, jobs, gen_kwargs = task
        if not jobs:
            continue

        try:
            lora_request = None
            if adapter_path is not None:
                adapter_key = os.path.realpath(os.fspath(adapter_path))
                if adapter_key not in adapter_ids:
                    adapter_ids[adapter_key] = next_adapter_id
                    next_adapter_id += 1
                adapter_id = adapter_ids[adapter_key]
                lora_request = LoRARequest(
                    f"ttt_adapter_{adapter_id}", adapter_id, adapter_key)

            # One call gives vLLM the whole worker workload so its scheduler can
            # batch different prompts and all n samples under the memory limit.
            tokenizer = llm.get_tokenizer()
            runnable_jobs = []
            max_tokens_by_job = []
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
                runnable_jobs.append((group_idx, prompt, count))
                max_tokens_by_job.append(max_tokens)

            if not runnable_jobs:
                continue

            prompts = [prompt for (_group_idx, prompt, _count) in runnable_jobs]
            sampling = [
                SamplingParams(
                    n=int(count),
                    max_tokens=int(max_tokens),
                    temperature=float(gen_kwargs["temperature"]),
                    top_p=float(gen_kwargs["top_p"]),
                    seed=_vllm_job_seed(seed, step, rank, group_idx),
                    skip_special_tokens=True,
                )
                for (group_idx, _prompt, count), max_tokens
                in zip(runnable_jobs, max_tokens_by_job)
            ]
            outputs = llm.generate(
                prompts,
                sampling_params=sampling,
                lora_request=lora_request,
                use_tqdm=False,
            )

            for job_pos, (group_idx, _prompt, count) in enumerate(runnable_jobs):
                request_output = (outputs[job_pos]
                                  if job_pos < len(outputs) else None)
                job_results = ([
                    (candidate.text, list(candidate.token_ids))
                    for candidate in request_output.outputs
                ][:int(count)] if request_output is not None else [])
                # Preserve the queue contract even if an engine/version returns
                # fewer samples than requested; downstream treats these as
                # invalid rollouts instead of blocking forever.
                if len(job_results) < int(count):
                    job_results.extend(
                        [("", []) for _ in range(int(count) - len(job_results))])
                result_queue.put((rank, group_idx, job_results))
        except Exception:
            detail = traceback.format_exc()
            print(f"[vllm worker {rank}] generation failed:\n{detail}",
                  file=sys.stderr, flush=True)
            if vllm_log_path:
                detail = ("details: "
                          f"{os.path.abspath(os.fspath(vllm_log_path))}")
            result_queue.put((rank, None, {"error": detail}))

    print(f"[vllm worker {rank}] shutting down", flush=True)


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
                 vllm_log_path=None):
        self.model_name = model_name
        requested_num_gpus = int(num_workers)
        self.gpu_ids = gpu_ids or list(range(num_workers))
        self.seed = seed
        self.gen_micro_batch = int(gen_micro_batch or 0)
        self.backend = str(backend).lower()
        self.sleep_requested = bool(vllm_enable_sleep_mode)
        self.sleep_supported = False
        self.vllm_log_path = (os.path.abspath(os.fspath(vllm_log_path))
                              if vllm_log_path else None)
        if self.backend not in ("hf", "vllm"):
            raise ValueError(
                f"unknown generation backend {backend!r}; expected hf|vllm")
        if len(self.gpu_ids) != requested_num_gpus:
            raise ValueError("gpu_ids must contain exactly num_workers entries")
        if len(set(self.gpu_ids)) != len(self.gpu_ids):
            raise ValueError("gpu_ids must not contain duplicates")

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

        ctx = mp.get_context("spawn")
        self.task_queues = [ctx.Queue() for _ in range(self.num_workers)]
        self.result_queue = ctx.Queue()
        ready_queue = ctx.Queue()
        self.control_queue = ctx.Queue()

        self.procs = []
        worker_target = (_vllm_worker_loop
                         if self.backend == "vllm" else _hf_worker_loop)
        worker_options = {
            "lora_rank": int(lora_rank),
            "gpu_memory_utilization": float(vllm_gpu_memory_utilization),
            "enforce_eager": bool(vllm_enforce_eager),
            "enable_prefix_caching": bool(vllm_enable_prefix_caching),
            "gen_micro_batch": self.gen_micro_batch,
            "quantization": vllm_quantization,
            "tensor_parallel_size": tp,
            "pipeline_parallel_size": pp,
            "max_num_batched_tokens": int(vllm_max_num_batched_tokens or 0),
            "enable_expert_parallel": bool(vllm_enable_expert_parallel),
            "enable_sleep_mode": self.sleep_requested,
            "control_queue": self.control_queue,
            "vllm_log_path": self.vllm_log_path,
        }
        for r in range(self.num_workers):
            p = ctx.Process(
                target=worker_target,
                args=(r, (gpu_groups[r] if self.backend == "vllm"
                          else gpu_groups[r][0]), model_name, max_seq_length,
                      load_in_4bit, self.task_queues[r], self.result_queue,
                      ready_queue, self.seed),
                kwargs=worker_options,
                # vLLM may manage child processes depending on its version and
                # engine settings; Python daemonic processes cannot do that.
                daemon=(self.backend != "vllm"),
            )
            p.start()
            self.procs.append(p)

        # Wait for all workers to finish loading
        print(f"[pool] waiting for {self.num_workers} {self.backend} engine(s) "
              f"on {requested_num_gpus} GPU(s) to load ...", flush=True)
        loaded = 0
        sleep_capable = 0
        while loaded < self.num_workers:
            try:
                status, rank, detail = ready_queue.get(timeout=1.0)
            except queue.Empty:
                dead = [(idx, proc.exitcode) for idx, proc in enumerate(self.procs)
                        if proc.exitcode is not None]
                if dead:
                    self.shutdown()
                    raise RuntimeError(
                        f"generation worker(s) exited during startup: {dead}")
                continue
            if status == "error":
                self.shutdown()
                raise RuntimeError(
                    f"{self.backend} generation worker {rank} failed to start:\n"
                    f"{detail}")
            if detail == "sleep":
                sleep_capable += 1
            loaded += 1
            print(f"[pool] {loaded}/{self.num_workers} workers ready", flush=True)
        self.sleep_supported = bool(
            self.backend == "vllm" and self.sleep_requested
            and sleep_capable == self.num_workers)

    def _vllm_control(self, command):
        if self.backend != "vllm":
            raise RuntimeError("sleep/wake controls are only valid for vLLM")
        if not self.sleep_supported:
            raise RuntimeError("the installed vLLM engine lacks sleep mode")
        for task_queue in self.task_queues:
            task_queue.put(("__control__", command))
        completed = 0
        while completed < self.num_workers:
            try:
                status, rank, got_command, detail = self.control_queue.get(
                    timeout=1.0)
            except queue.Empty:
                dead = [(idx, proc.exitcode)
                        for idx, proc in enumerate(self.procs)
                        if proc.exitcode is not None]
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
            completed += 1

    def sleep(self):
        self._vllm_control("sleep")

    def wake_up(self):
        self._vllm_control("wake_up")

    def iter_group_jobs(self, prompts_by_group, group_size, adapter_path,
                        max_new_tokens, temperature, top_p, step_idx=0,
                        show_progress=True, counts_by_group=None):
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
        }
        total_expected = sum(count for wj in worker_jobs for (_, _, count) in wj)

        # Dispatch one task per worker (some may have empty job lists)
        for r in range(self.num_workers):
            self.task_queues[r].put((step_idx, adapter_path, worker_jobs[r], gen_kwargs))

        collected = 0
        bar = make_progress_bar(total_expected, desc="rollouts") if show_progress else None
        try:
            while collected < total_expected:
                try:
                    rank, group_idx, job_results = self.result_queue.get(timeout=1.0)
                except queue.Empty:
                    dead = [(idx, proc.exitcode)
                            for idx, proc in enumerate(self.procs)
                            if proc.exitcode is not None]
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

    def generate_groups(self, prompts_by_group, group_size, adapter_path,
                        max_new_tokens, temperature, top_p, step_idx=0,
                        counts_by_group=None):
        """
        Backward-compatible blocking variant. Returns:
          dict group_idx -> list of (text, token_ids, None)
        (behavior_logprobs is None; IS disabled). Prefer iter_group_jobs() when
        you want to overlap reward evaluation with generation.
        """
        num_groups = len(prompts_by_group)
        by_group = {g: [] for g in range(num_groups)}
        for group_idx, job_results in self.iter_group_jobs(
                prompts_by_group, group_size, adapter_path,
                max_new_tokens, temperature, top_p, step_idx=step_idx,
                counts_by_group=counts_by_group):
            for (text, token_ids) in job_results:
                by_group[group_idx].append((text, token_ids, None))
        return by_group

    def shutdown(self):
        for r in range(self.num_workers):
            try:
                self.task_queues[r].put(None)
            except Exception:
                pass
        for p in self.procs:
            p.join(timeout=10)
            if p.is_alive():
                p.terminate()
                p.join(timeout=10)
            if p.is_alive():
                p.kill()
                p.join(timeout=5)


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
                        show_progress=True, counts_by_group=None):
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
        bar = (make_progress_bar(total_expected, desc="rollouts")
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
    """Keep shared-card vLLM alive but asleep during differentiable updates.

    Level-1 sleep offloads vLLM weights and discards its KV cache.  If sleep
    mode is unavailable, generation safely falls back to a transient engine;
    in either mode, the trainer is never resident on the generation cards at
    the same time as an awake vLLM engine.
    """

    sequential = True

    def __init__(self, before_start, after_stop, **pool_kwargs):
        self._before_start = before_start
        self._after_stop = after_stop
        self._pool_kwargs = dict(pool_kwargs)
        self._pool = None
        self._persistent = False
        self._awake = False
        self.num_workers = 1

        self._before_start()
        candidate = None
        try:
            candidate = GenerationPool(
                **self._pool_kwargs, vllm_enable_sleep_mode=True)
            self.num_workers = candidate.num_workers
            if candidate.sleep_supported:
                candidate.sleep()
                self._pool = candidate
                self._persistent = True
                print("[pool] vLLM sleep mode ready; engine will persist "
                      "between rollout phases", flush=True)
            else:
                candidate.shutdown()
                print("[pool] installed vLLM lacks sleep mode; using a "
                      "transient engine for safe phase sharing", flush=True)
        except Exception:
            # Startup errors such as an invalid checkpoint or an engine OOM
            # must remain fatal; they are not a sleep-mode compatibility issue.
            if candidate is not None:
                candidate.shutdown()
            self._after_stop()
            raise
        self._after_stop()

    @property
    def active(self):
        return self._awake

    def _ensure_started(self):
        if self._awake:
            return self._pool
        self._before_start()
        try:
            if self._persistent:
                self._pool.wake_up()
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

    def release(self):
        if not self._awake:
            return
        try:
            if self._persistent:
                try:
                    self._pool.sleep()
                except Exception:
                    # Do not restore the trainer next to an engine whose GPU
                    # allocations may still be live.
                    self._pool.shutdown()
                    self._pool = None
                    self._persistent = False
                    raise
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
