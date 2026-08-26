"""
Persistent multi-GPU rollout generation pool.

We run one persistent worker process per GPU. Each worker:
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
    every worker participates in every group (max GPU utilization).
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
            worker_count = base + (1 if w < rem else 0)
            if worker_count > 0:
                worker_jobs[w].append((g, prompt, worker_count))
    return worker_jobs


def worker_seed(seed, step, rank):
    """
    Deterministic seed for (run, step, worker). Kept as a module-level function
    so the trainer's single-GPU path can key its own reseed the same way.
    """
    return (int(seed) * 1_000_003 + int(step) * 1009 + int(rank) * 7 + 13) % (2 ** 31 - 1)


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

    model_kwargs = dict(torch_dtype=torch.bfloat16, trust_remote_code=True)
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

    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id or eos_id

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

        # Micro-batch cap: hold at most `mb` sequences in flight per generate()
        # call, looping until this job's `count` is done. This bounds KV memory
        # by `mb` regardless of how big group_size is. mb <= 0 (or >= count)
        # means one shot. NOTE: mb bounds the number of sequences, NOT the
        # per-sequence prefill cost, which is O(prompt_len^2) in eager attention
        # and grows as parents get longer. The OOM-halving loop below is what
        # keeps that spike from killing the worker.
        mb = int(gen_kwargs.get("micro_batch", 0) or 0)

        for (group_idx, prompt, count) in jobs:
            enc = tokenizer([prompt], return_tensors="pt").to(device)
            input_len = enc.input_ids.shape[1]

            remaining = count
            while remaining > 0:
                # Effective per-call ceiling: the smaller of the configured
                # micro-batch and any limit learned from a past OOM. Both unset
                # means the whole remaining job in one call.
                limit = count
                if mb > 0:
                    limit = mb
                if gen_cap > 0:
                    limit = min(limit, gen_cap)
                n = min(limit, remaining)

                try:
                    with torch.inference_mode():
                        out = gen_model.generate(
                            **enc,
                            num_return_sequences=n,
                            max_new_tokens=gen_kwargs["max_new_tokens"],
                            do_sample=True,
                            temperature=gen_kwargs["temperature"],
                            top_p=gen_kwargs["top_p"],
                            pad_token_id=pad_id,
                        )
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    if n <= 1:
                        # Cannot fit even one sequence for this prompt. Emit
                        # empty placeholders for the rest of the job so the main
                        # process's rollout counter still reaches total_expected
                        # and the step does not hang on result_queue.get().
                        # These land as empty (invalid) rollouts and are skipped
                        # in training (len(token_ids) == 0). They are lost, not
                        # fatal.
                        print(f"[worker {rank}] OOM at n=1 on group {group_idx} "
                              f"(prompt {input_len} tok); dropping {remaining} "
                              f"rollout(s) this step", flush=True)
                        result_queue.put(
                            (rank, group_idx, [("", []) for _ in range(remaining)]))
                        remaining = 0
                        break
                    gen_cap = max(1, n // 2)
                    print(f"[worker {rank}] OOM on group {group_idx} at n={n} "
                          f"(prompt {input_len} tok); halving per-call to "
                          f"{gen_cap} and retrying (sticky for the run)",
                          flush=True)
                    continue

                # out shape: (n, input_len + gen_len)
                job_results = []
                for i in range(out.shape[0]):
                    gen_ids = out[i, input_len:].tolist()
                    if eos_id is not None and eos_id in gen_ids:
                        gen_ids = gen_ids[: gen_ids.index(eos_id) + 1]
                    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
                    job_results.append((text, gen_ids))

                # Report each chunk immediately so the main process can advance
                # the progress bar AND start evaluating these rollouts while the
                # GPUs keep generating the rest. The consumer counts rollouts,
                # not messages, so several messages per group are fine.
                result_queue.put((rank, group_idx, job_results))
                remaining -= n
                del out

    print(f"[worker {rank}] shutting down", flush=True)


def _vllm_engine_kwargs(model_name, max_seq_length, load_in_4bit,
                        lora_rank, gpu_memory_utilization,
                        enforce_eager=False, enable_prefix_caching=True,
                        max_num_seqs=0, seed=None):
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
    }
    if int(max_num_seqs or 0) > 0:
        kwargs["max_num_seqs"] = int(max_num_seqs)
    if seed is not None:
        kwargs["seed"] = int(seed)
    if load_in_4bit:
        # Current vLLM uses the out-of-tree vllm-bnb-plugin for this mode.
        # Pre-quantized checkpoints are inferred automatically, but explicitly
        # requesting 4-bit here means in-flight BitsAndBytes quantization.
        kwargs["quantization"] = "bitsandbytes"
    return kwargs


def _vllm_job_seed(seed, step, rank, group_idx):
    """Stable per-request seed; None preserves vLLM's stochastic default."""
    if seed is None:
        return None
    return (worker_seed(seed, step, rank) + int(group_idx) * 104_729) % (2 ** 31 - 1)


def _vllm_worker_loop(rank, gpu_id, model_name, max_seq_length, load_in_4bit,
                      task_queue, result_queue, ready_queue, seed=None,
                      lora_rank=32, gpu_memory_utilization=0.9,
                      enforce_eager=False, enable_prefix_caching=True,
                      gen_micro_batch=0):
    """Persistent vLLM engine, one process and one engine per generation GPU."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    if seed is not None:
        # vLLM documents this setting for deterministic V1 offline inference.
        os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

    try:
        from vllm import LLM, SamplingParams
        from vllm.lora.request import LoRARequest

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
        )
        print(f"[vllm worker {rank}] loading {model_name} on physical GPU "
              f"{gpu_id} ...", flush=True)
        llm = LLM(**engine_kwargs)
    except Exception:
        ready_queue.put(("error", rank, traceback.format_exc()))
        return

    # Paths are versioned (adapter_step000, adapter_step001, ...). Giving each
    # path a unique positive id prevents vLLM from serving a stale cached LoRA.
    adapter_ids = {}
    next_adapter_id = 1
    ready_queue.put(("ready", rank, ""))

    while True:
        task = task_queue.get()
        if task is None:
            break
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
            prompts = [prompt for (_group_idx, prompt, _count) in jobs]
            sampling = [
                SamplingParams(
                    n=int(count),
                    max_tokens=int(gen_kwargs["max_new_tokens"]),
                    temperature=float(gen_kwargs["temperature"]),
                    top_p=float(gen_kwargs["top_p"]),
                    seed=_vllm_job_seed(seed, step, rank, group_idx),
                    skip_special_tokens=True,
                )
                for (group_idx, _prompt, count) in jobs
            ]
            outputs = llm.generate(
                prompts,
                sampling_params=sampling,
                lora_request=lora_request,
                use_tqdm=False,
            )

            for job_pos, (group_idx, _prompt, count) in enumerate(jobs):
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
                  flush=True)
            result_queue.put((rank, None, {"error": detail}))

    print(f"[vllm worker {rank}] shutting down", flush=True)


class GenerationPool:
    """
    Manages persistent generation processes (one engine per GPU).

    backend="hf" preserves the original Transformers workers. backend="vllm"
    uses vLLM for rollout inference while keeping exactly the same queue API.
    """

    def __init__(self, model_name, num_workers, gpu_ids=None,
                 max_seq_length=4096, load_in_4bit=False, seed=None,
                 gen_micro_batch=0, backend="hf", lora_rank=32,
                 vllm_gpu_memory_utilization=0.9,
                 vllm_enforce_eager=False,
                 vllm_enable_prefix_caching=True):
        self.model_name = model_name
        self.num_workers = int(num_workers)
        self.gpu_ids = gpu_ids or list(range(num_workers))
        self.seed = seed
        self.gen_micro_batch = int(gen_micro_batch or 0)
        self.backend = str(backend).lower()
        if self.backend not in ("hf", "vllm"):
            raise ValueError(
                f"unknown generation backend {backend!r}; expected hf|vllm")
        if len(self.gpu_ids) != self.num_workers:
            raise ValueError("gpu_ids must contain exactly num_workers entries")
        if len(set(self.gpu_ids)) != len(self.gpu_ids):
            raise ValueError("gpu_ids must not contain duplicates")

        ctx = mp.get_context("spawn")
        self.task_queues = [ctx.Queue() for _ in range(self.num_workers)]
        self.result_queue = ctx.Queue()
        ready_queue = ctx.Queue()

        self.procs = []
        worker_target = (_vllm_worker_loop
                         if self.backend == "vllm" else _hf_worker_loop)
        worker_options = {
            "lora_rank": int(lora_rank),
            "gpu_memory_utilization": float(vllm_gpu_memory_utilization),
            "enforce_eager": bool(vllm_enforce_eager),
            "enable_prefix_caching": bool(vllm_enable_prefix_caching),
            "gen_micro_batch": self.gen_micro_batch,
        }
        for r in range(self.num_workers):
            p = ctx.Process(
                target=worker_target,
                args=(r, self.gpu_ids[r], model_name, max_seq_length,
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
        print(f"[pool] waiting for {self.num_workers} {self.backend} worker(s) "
              f"to load ...", flush=True)
        loaded = 0
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
            loaded += 1
            print(f"[pool] {loaded}/{self.num_workers} workers ready", flush=True)

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
