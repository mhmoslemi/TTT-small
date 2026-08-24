"""
Multi-GPU generation pool (Approach 2b).

We run one persistent worker process per GPU. Each worker:
  - loads a PLAIN transformers copy of the base model on its GPU (no Unsloth)
  - wraps it with the current LoRA adapter (loaded from a file on disk)
  - generates its share of rollouts with batched model.generate()
  - reloads the adapter from disk at the start of each step (weight sync)
  - reports results PER JOB so the main process can (a) drive a rollout progress
    bar and (b) start evaluating each rollout's program on CPU threads WHILE the
    GPUs keep generating the rest.

The MAIN process (in train_multy.py) keeps the Unsloth model for TRAINING only.
Each step it saves the LoRA adapter to a directory, then asks the pool to
generate using that adapter. This keeps Unsloth where it helps (the backward
pass) and uses boring-but-reliable HF for generation across all GPUs.

IMPORTANCE SAMPLING (fix 6) IS DISABLED FOR SPEED.
We do NOT compute per-token behavior logprobs; generation returns tokens only.
generate_groups() fills the trainer's logprob slot with None (read as on-policy,
IS ratio = 1). iter_group_jobs() yields plain (text, token_ids) pairs.

SEEDING. Each worker reseeds random / numpy / torch from (seed, step, rank)
before every task, rather than seeding once at startup and letting the stream
advance. Keying on the step is what makes step t reproducible on its own: it no
longer depends on how many generations happened before it, so a memory run and
a no-memory run draw the same samples at step 0, and the memory maker's own
calls (which use step_idx + 1_000_000) cannot shift the rollout stream.
Determinism holds for a fixed num_gpus, group_size AND gen_micro_batch, since
distribute_jobs splits the group across workers and changing the split (or the
micro-batch chunking) changes which sequence each worker draws.

OOM RECOVERY. The worker halves its per-call sequence count and retries when
generate() OOMs, and keeps the reduced size for the rest of the run. This
mirrors the single-GPU path in train_multy.py. Without it a step whose parents
have grown longer than the seed states (longer prompt -> larger eager prefill
attention) killed the worker outright, and the main process then hung forever
on result_queue.get().

No async. Plain torch.multiprocessing with persistent workers and queues.

Protocol (per step):
  main -> worker[w].task_queue:   (step, adapter_path, jobs, gen_kwargs)
       where jobs = [(group_idx, prompt_text, num_samples), ...]
  worker[w] -> result_queue:      (rank, group_idx, [(text, token_ids), ...])
       one message PER JOB, so the pool can stream results as they land.
"""

import os
import time
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


def _worker_loop(rank, gpu_id, model_name, max_seq_length, load_in_4bit,
                 task_queue, result_queue, ready_queue, seed=None):
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
    ready_queue.put(rank)

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


class GenerationPool:
    """
    Manages num_workers persistent generation processes (one per GPU).
    """

    def __init__(self, model_name, num_workers, gpu_ids=None,
                 max_seq_length=4096, load_in_4bit=False, seed=None,
                 gen_micro_batch=0):
        self.model_name = model_name
        self.num_workers = num_workers
        self.gpu_ids = gpu_ids or list(range(num_workers))
        self.seed = seed
        self.gen_micro_batch = int(gen_micro_batch or 0)
        assert len(self.gpu_ids) == num_workers

        ctx = mp.get_context("spawn")
        self.task_queues = [ctx.Queue() for _ in range(num_workers)]
        self.result_queue = ctx.Queue()
        ready_queue = ctx.Queue()

        self.procs = []
        for r in range(num_workers):
            p = ctx.Process(
                target=_worker_loop,
                args=(r, self.gpu_ids[r], model_name, max_seq_length,
                      load_in_4bit, self.task_queues[r], self.result_queue,
                      ready_queue, self.seed),
                daemon=True,
            )
            p.start()
            self.procs.append(p)

        # Wait for all workers to finish loading
        print(f"[pool] waiting for {num_workers} workers to load ...", flush=True)
        loaded = 0
        while loaded < num_workers:
            ready_queue.get()
            loaded += 1
            print(f"[pool] {loaded}/{num_workers} workers ready", flush=True)

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
                rank, group_idx, job_results = self.result_queue.get()
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
