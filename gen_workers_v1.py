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

The MAIN process (in train.py) keeps the Unsloth model for TRAINING only.
Each step it saves the LoRA adapter to a directory, then asks the pool to
generate using that adapter. This keeps Unsloth where it helps (the backward
pass) and uses boring-but-reliable HF for generation across all GPUs.

IMPORTANCE SAMPLING (fix 6) IS DISABLED FOR SPEED.
We do NOT compute per-token behavior logprobs; generation returns tokens only.
generate_groups() fills the trainer's logprob slot with None (read as on-policy,
IS ratio = 1). iter_group_jobs() yields plain (text, token_ids) pairs.

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


def distribute_jobs(prompts_by_group, group_size, num_workers):
    """
    prompts_by_group: list of prompt strings, one per group (index = group_idx)

    Returns worker_jobs: list (len num_workers) of lists of
        (group_idx, prompt_text, count)
    so that across workers each group gets exactly group_size samples and
    every worker participates in every group (max GPU utilization).
    """
    worker_jobs = [[] for _ in range(num_workers)]
    for g, prompt in enumerate(prompts_by_group):
        base = group_size // num_workers
        rem = group_size % num_workers
        for w in range(num_workers):
            count = base + (1 if w < rem else 0)
            if count > 0:
                worker_jobs[w].append((g, prompt, count))
    return worker_jobs


def _worker_loop(rank, gpu_id, model_name, max_seq_length, load_in_4bit,
                 task_queue, result_queue, ready_queue):
    """
    Persistent worker. Loads the model once, then serves generation tasks
    until it receives None.
    """
    # Pin to our GPU. Do this before heavy CUDA work.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

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

    # Signal that we finished loading
    ready_queue.put(rank)

    while True:
        task = task_queue.get()
        if task is None:
            break
        step, adapter_path, jobs, gen_kwargs = task
        gen_model = ensure_adapter(adapter_path)

        for (group_idx, prompt, count) in jobs:
            enc = tokenizer([prompt], return_tensors="pt").to(device)
            input_len = enc.input_ids.shape[1]
            with torch.inference_mode():
                out = gen_model.generate(
                    **enc,
                    num_return_sequences=count,
                    max_new_tokens=gen_kwargs["max_new_tokens"],
                    do_sample=True,
                    temperature=gen_kwargs["temperature"],
                    top_p=gen_kwargs["top_p"],
                    pad_token_id=pad_id,
                )
            # out shape: (count, input_len + gen_len)
            job_results = []
            for i in range(out.shape[0]):
                gen_ids = out[i, input_len:].tolist()
                if eos_id is not None and eos_id in gen_ids:
                    gen_ids = gen_ids[: gen_ids.index(eos_id) + 1]
                text = tokenizer.decode(gen_ids, skip_special_tokens=True)
                job_results.append((text, gen_ids))

            # Report this job immediately so the main process can advance the
            # progress bar AND start evaluating these rollouts while the GPUs
            # keep generating the remaining jobs.
            result_queue.put((rank, group_idx, job_results))

    print(f"[worker {rank}] shutting down", flush=True)


class GenerationPool:
    """
    Manages num_workers persistent generation processes (one per GPU).
    """

    def __init__(self, model_name, num_workers, gpu_ids=None,
                 max_seq_length=4096, load_in_4bit=False):
        self.model_name = model_name
        self.num_workers = num_workers
        self.gpu_ids = gpu_ids or list(range(num_workers))
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
                      ready_queue),
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
                        max_new_tokens, temperature, top_p, show_progress=True):
        """
        Stream generation results as each (worker, group) job completes.

        Yields (group_idx, [(text, token_ids), ...]) per job. The caller can
        dispatch each rollout for reward evaluation immediately, overlapping
        CPU eval with ongoing GPU generation.

        Drives a "rollouts" progress bar over total rollouts (set
        show_progress=False to suppress). Stops after exactly total_expected
        rollouts, so every per-job message is drained and none leak into the
        next step's queue.
        """
        worker_jobs = distribute_jobs(prompts_by_group, group_size, self.num_workers)
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }
        total_expected = sum(count for wj in worker_jobs for (_, _, count) in wj)

        # Dispatch one task per worker (some may have empty job lists)
        for r in range(self.num_workers):
            self.task_queues[r].put((0, adapter_path, worker_jobs[r], gen_kwargs))

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
                         max_new_tokens, temperature, top_p):
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
                max_new_tokens, temperature, top_p):
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

# """
# Multi-GPU generation pool (Approach 2b).

# We run one persistent worker process per GPU. Each worker:
#   - loads a PLAIN transformers copy of the base model on its GPU (no Unsloth)
#   - wraps it with the current LoRA adapter (loaded from a file on disk)
#   - generates its share of rollouts with batched model.generate()
#   - reloads the adapter from disk at the start of each step (weight sync)
#   - reports results PER JOB so the main process can drive a rollout progress bar

# The MAIN process (in train.py) keeps the Unsloth model for TRAINING only.
# Each step it saves the LoRA adapter to a directory, then asks the pool to
# generate using that adapter. This keeps Unsloth where it helps (the backward
# pass) and uses boring-but-reliable HF for generation across all GPUs.

# IMPORTANCE SAMPLING (fix 6) IS DISABLED HERE FOR SPEED.
# We do NOT compute per-token behavior logprobs. Generation returns tokens only.
# The pool fills the trainer's logprob slot with None, which the training loop
# reads as "on-policy" (IS ratio = 1, no correction). To re-enable IS later,
# compute logprobs in the worker (one teacher-forced forward over prompt+response
# per sequence) and return them in place of None.

# No async. Plain torch.multiprocessing with persistent workers and queues.

# Protocol (per step):
#   main -> worker[w].task_queue:   (step, adapter_path, jobs, gen_kwargs)
#        where jobs = [(group_idx, prompt_text, num_samples), ...]
#   worker[w] -> result_queue:      (rank, group_idx, [(text, token_ids), ...])
#        one message PER JOB (not one per worker), so the pool can advance a
#        rollout progress bar as each (worker, group) chunk lands.
#   pool returns to caller:         dict group_idx -> [(text, token_ids, None), ...]
# """

# import os
# import time
# import multiprocessing as mp

# # tqdm ships with transformers/huggingface_hub, so it's almost always present.
# # Fall back to a coarse print bar if it isn't, so generation never depends on it.
# try:
#     from tqdm import tqdm
#     _HAS_TQDM = True
# except Exception:
#     tqdm = None
#     _HAS_TQDM = False


# class _PrintBar:
#     """Minimal stand-in for tqdm: prints progress at ~10% increments."""

#     def __init__(self, total, desc="rollouts"):
#         self.total = max(int(total), 1)
#         self.n = 0
#         self.desc = desc
#         self._last_decile = -1

#     def update(self, k=1):
#         self.n += k
#         decile = int(10 * self.n / self.total)
#         if decile != self._last_decile:
#             self._last_decile = decile
#             pct = int(100 * self.n / self.total)
#             print(f"[pool] {self.desc} {self.n}/{self.total} ({pct}%)", flush=True)

#     def close(self):
#         if self.n < self.total:
#             print(f"[pool] {self.desc} {self.n}/{self.total}", flush=True)


# def _make_bar(total, desc="rollouts"):
#     total = int(max(total, 1))
#     if _HAS_TQDM:
#         return tqdm(total=total, desc=desc, unit="roll",
#                     leave=False, dynamic_ncols=True)
#     return _PrintBar(total, desc=desc)


# def distribute_jobs(prompts_by_group, group_size, num_workers):
#     """
#     prompts_by_group: list of prompt strings, one per group (index = group_idx)

#     Returns worker_jobs: list (len num_workers) of lists of
#         (group_idx, prompt_text, count)
#     so that across workers each group gets exactly group_size samples and
#     every worker participates in every group (max GPU utilization).
#     """
#     worker_jobs = [[] for _ in range(num_workers)]
#     for g, prompt in enumerate(prompts_by_group):
#         base = group_size // num_workers
#         rem = group_size % num_workers
#         for w in range(num_workers):
#             count = base + (1 if w < rem else 0)
#             if count > 0:
#                 worker_jobs[w].append((g, prompt, count))
#     return worker_jobs


# def _worker_loop(rank, gpu_id, model_name, max_seq_length, load_in_4bit,
#                  task_queue, result_queue, ready_queue):
#     """
#     Persistent worker. Loads the model once, then serves generation tasks
#     until it receives None.
#     """
#     # Pin to our GPU. Do this before heavy CUDA work.
#     os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

#     import torch
#     from transformers import AutoModelForCausalLM, AutoTokenizer
#     from peft import PeftModel
#     from peft.utils import set_peft_model_state_dict
#     from safetensors.torch import load_file

#     # With CUDA_VISIBLE_DEVICES set, our GPU is cuda:0 inside this process
#     device = "cuda:0"

#     print(f"[worker {rank}] loading {model_name} on physical GPU {gpu_id} ...", flush=True)
#     tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
#     if tokenizer.pad_token_id is None:
#         tokenizer.pad_token = tokenizer.eos_token
#     tokenizer.padding_side = "left"  # left-pad for decoder-only batched generation

#     model_kwargs = dict(torch_dtype=torch.bfloat16, trust_remote_code=True)
#     if load_in_4bit:
#         try:
#             from transformers import BitsAndBytesConfig
#             model_kwargs["quantization_config"] = BitsAndBytesConfig(
#                 load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
#                 bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4",
#             )
#         except ImportError:
#             pass

#     base = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
#     if not load_in_4bit:
#         base = base.to(device)
#     base.eval()

#     peft_model = None          # created on first adapter load
#     current_adapter = None     # path of the adapter currently loaded

#     def ensure_adapter(adapter_path):
#         nonlocal peft_model, current_adapter
#         if adapter_path is None:
#             # No adapter yet (step 0 before any training) -> use base as-is
#             return base
#         if peft_model is None:
#             peft_model = PeftModel.from_pretrained(base, adapter_path, is_trainable=False)
#             peft_model.eval()
#             current_adapter = adapter_path
#             return peft_model
#         if adapter_path != current_adapter:
#             # Reload just the LoRA weights into the existing wrapper
#             sd_path = os.path.join(adapter_path, "adapter_model.safetensors")
#             try:
#                 weights = load_file(sd_path)
#                 set_peft_model_state_dict(peft_model, weights)
#             except Exception as e:
#                 # Fallback: rewrap from scratch (re-reads only the tiny adapter)
#                 print(f"[worker {rank}] adapter reload fallback ({e})", flush=True)
#                 peft_model = PeftModel.from_pretrained(base, adapter_path, is_trainable=False)
#                 peft_model.eval()
#             current_adapter = adapter_path
#         return peft_model

#     eos_id = tokenizer.eos_token_id
#     pad_id = tokenizer.pad_token_id or eos_id

#     # Signal that we finished loading
#     ready_queue.put(rank)

#     while True:
#         task = task_queue.get()
#         if task is None:
#             break
#         step, adapter_path, jobs, gen_kwargs = task
#         gen_model = ensure_adapter(adapter_path)

#         for (group_idx, prompt, count) in jobs:
#             enc = tokenizer([prompt], return_tensors="pt").to(device)
#             input_len = enc.input_ids.shape[1]
#             with torch.inference_mode():
#                 out = gen_model.generate(
#                     **enc,
#                     num_return_sequences=count,
#                     max_new_tokens=gen_kwargs["max_new_tokens"],
#                     do_sample=True,
#                     temperature=gen_kwargs["temperature"],
#                     top_p=gen_kwargs["top_p"],
#                     pad_token_id=pad_id,
#                 )
#             # out shape: (count, input_len + gen_len)
#             job_results = []
#             for i in range(out.shape[0]):
#                 gen_ids = out[i, input_len:].tolist()
#                 if eos_id is not None and eos_id in gen_ids:
#                     gen_ids = gen_ids[: gen_ids.index(eos_id) + 1]
#                 text = tokenizer.decode(gen_ids, skip_special_tokens=True)
#                 job_results.append((text, gen_ids))

#             # Report this job immediately so the main process can advance the
#             # rollout progress bar instead of waiting for the whole worker.
#             result_queue.put((rank, group_idx, job_results))

#     print(f"[worker {rank}] shutting down", flush=True)


# class GenerationPool:
#     """
#     Manages num_workers persistent generation processes (one per GPU).
#     """

#     def __init__(self, model_name, num_workers, gpu_ids=None,
#                  max_seq_length=4096, load_in_4bit=False):
#         self.model_name = model_name
#         self.num_workers = num_workers
#         self.gpu_ids = gpu_ids or list(range(num_workers))
#         assert len(self.gpu_ids) == num_workers

#         ctx = mp.get_context("spawn")
#         self.task_queues = [ctx.Queue() for _ in range(num_workers)]
#         self.result_queue = ctx.Queue()
#         ready_queue = ctx.Queue()

#         self.procs = []
#         for r in range(num_workers):
#             p = ctx.Process(
#                 target=_worker_loop,
#                 args=(r, self.gpu_ids[r], model_name, max_seq_length,
#                       load_in_4bit, self.task_queues[r], self.result_queue,
#                       ready_queue),
#                 daemon=True,
#             )
#             p.start()
#             self.procs.append(p)

#         # Wait for all workers to finish loading
#         print(f"[pool] waiting for {num_workers} workers to load ...", flush=True)
#         loaded = 0
#         while loaded < num_workers:
#             ready_queue.get()
#             loaded += 1
#             print(f"[pool] {loaded}/{num_workers} workers ready", flush=True)

#     def generate_groups(self, prompts_by_group, group_size, adapter_path,
#                          max_new_tokens, temperature, top_p):
#         """
#         prompts_by_group: list of prompts, one per group.

#         Returns: dict group_idx -> list of (text, token_ids, behavior_logprobs),
#         each list of length group_size (order within a group is not meaningful).

#         behavior_logprobs is always None here (IS / fix 6 disabled for speed);
#         the slot exists only so the training loop's 3-tuple unpacking and its
#         None -> on-policy (ratio=1) branch keep working unchanged.

#         Shows a tqdm progress bar over total rollouts as each (worker, group)
#         job completes.
#         """
#         num_groups = len(prompts_by_group)
#         worker_jobs = distribute_jobs(prompts_by_group, group_size, self.num_workers)
#         gen_kwargs = {
#             "max_new_tokens": max_new_tokens,
#             "temperature": temperature,
#             "top_p": top_p,
#         }

#         # Total rollouts expected this step == sum of all per-job counts.
#         # We collect exactly this many rollouts, so the per-job messages are
#         # fully drained and none are left in the queue for the next step.
#         total_expected = sum(count for wj in worker_jobs for (_, _, count) in wj)

#         # Dispatch one task per worker (some may have empty job lists)
#         for r in range(self.num_workers):
#             self.task_queues[r].put((0, adapter_path, worker_jobs[r], gen_kwargs))

#         # Collect results as each job lands, advancing the progress bar.
#         by_group = {g: [] for g in range(num_groups)}
#         collected = 0
#         bar = _make_bar(total_expected, desc="rollouts")
#         try:
#             while collected < total_expected:
#                 rank, group_idx, job_results = self.result_queue.get()
#                 for (text, token_ids) in job_results:
#                     # None -> trainer treats this rollout as on-policy (IS ratio 1).
#                     by_group[group_idx].append((text, token_ids, None))
#                 collected += len(job_results)
#                 bar.update(len(job_results))
#         finally:
#             bar.close()

#         return by_group

#     def shutdown(self):
#         for r in range(self.num_workers):
#             try:
#                 self.task_queues[r].put(None)
#             except Exception:
#                 pass
#         for p in self.procs:
#             p.join(timeout=10)
#             if p.is_alive():
#                 p.terminate()

# # """
# # Multi-GPU generation pool (Approach 2b).

# # We run one persistent worker process per GPU. Each worker:
# #   - loads a PLAIN transformers copy of the base model on its GPU (no Unsloth)
# #   - wraps it with the current LoRA adapter (loaded from a file on disk)
# #   - generates its share of rollouts with batched model.generate()
# #   - reloads the adapter from disk at the start of each step (weight sync)
# #   - returns per-token behavior logprobs alongside the tokens, so the trainer
# #     can apply an importance-sampling correction for the generation/scoring
# #     backend mismatch (plain HF here vs Unsloth in the main process)

# # The MAIN process (in train.py) keeps the Unsloth model for TRAINING only.
# # Each step it saves the LoRA adapter to a directory, then asks the pool to
# # generate using that adapter. This keeps Unsloth where it helps (the backward
# # pass) and uses boring-but-reliable HF for generation across all GPUs.

# # No async. Plain torch.multiprocessing with persistent workers and queues.

# # Protocol (per step):
# #   main -> worker[w].task_queue:   (step, adapter_path, jobs, gen_kwargs)
# #        where jobs = [(group_idx, prompt_text, num_samples), ...]
# #   worker[w] -> result_queue:      (rank, [(group_idx, text, token_ids,
# #                                            behavior_logprobs), ...])
# #        where behavior_logprobs is a per-token list aligned 1:1 with token_ids
# # """

# # import os
# # import time
# # import multiprocessing as mp


# # def distribute_jobs(prompts_by_group, group_size, num_workers):
# #     """
# #     prompts_by_group: list of prompt strings, one per group (index = group_idx)

# #     Returns worker_jobs: list (len num_workers) of lists of
# #         (group_idx, prompt_text, count)
# #     so that across workers each group gets exactly group_size samples and
# #     every worker participates in every group (max GPU utilization).
# #     """
# #     worker_jobs = [[] for _ in range(num_workers)]
# #     for g, prompt in enumerate(prompts_by_group):
# #         base = group_size // num_workers
# #         rem = group_size % num_workers
# #         for w in range(num_workers):
# #             count = base + (1 if w < rem else 0)
# #             if count > 0:
# #                 worker_jobs[w].append((g, prompt, count))
# #     return worker_jobs


# # def _worker_loop(rank, gpu_id, model_name, max_seq_length, load_in_4bit,
# #                  task_queue, result_queue, ready_queue):
# #     """
# #     Persistent worker. Loads the model once, then serves generation tasks
# #     until it receives None.
# #     """
# #     # Pin to our GPU. Do this before heavy CUDA work.
# #     os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

# #     import torch
# #     from transformers import AutoModelForCausalLM, AutoTokenizer
# #     from peft import PeftModel
# #     from peft.utils import set_peft_model_state_dict
# #     from safetensors.torch import load_file

# #     # With CUDA_VISIBLE_DEVICES set, our GPU is cuda:0 inside this process
# #     device = "cuda:0"

# #     print(f"[worker {rank}] loading {model_name} on physical GPU {gpu_id} ...", flush=True)
# #     tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
# #     if tokenizer.pad_token_id is None:
# #         tokenizer.pad_token = tokenizer.eos_token
# #     tokenizer.padding_side = "left"  # left-pad for decoder-only batched generation

# #     model_kwargs = dict(torch_dtype=torch.bfloat16, trust_remote_code=True)
# #     if load_in_4bit:
# #         try:
# #             from transformers import BitsAndBytesConfig
# #             model_kwargs["quantization_config"] = BitsAndBytesConfig(
# #                 load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
# #                 bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4",
# #             )
# #         except ImportError:
# #             pass

# #     base = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
# #     if not load_in_4bit:
# #         base = base.to(device)
# #     base.eval()

# #     peft_model = None          # created on first adapter load
# #     current_adapter = None     # path of the adapter currently loaded

# #     def ensure_adapter(adapter_path):
# #         nonlocal peft_model, current_adapter
# #         if adapter_path is None:
# #             # No adapter yet (step 0 before any training) -> use base as-is
# #             return base
# #         if peft_model is None:
# #             peft_model = PeftModel.from_pretrained(base, adapter_path, is_trainable=False)
# #             peft_model.eval()
# #             current_adapter = adapter_path
# #             return peft_model
# #         if adapter_path != current_adapter:
# #             # Reload just the LoRA weights into the existing wrapper
# #             sd_path = os.path.join(adapter_path, "adapter_model.safetensors")
# #             try:
# #                 weights = load_file(sd_path)
# #                 set_peft_model_state_dict(peft_model, weights)
# #             except Exception as e:
# #                 # Fallback: rewrap from scratch (re-reads only the tiny adapter)
# #                 print(f"[worker {rank}] adapter reload fallback ({e})", flush=True)
# #                 peft_model = PeftModel.from_pretrained(base, adapter_path, is_trainable=False)
# #                 peft_model.eval()
# #             current_adapter = adapter_path
# #         return peft_model

# #     eos_id = tokenizer.eos_token_id
# #     pad_id = tokenizer.pad_token_id or eos_id

# #     # Signal that we finished loading
# #     ready_queue.put(rank)

# #     while True:
# #         task = task_queue.get()
# #         if task is None:
# #             break
# #         step, adapter_path, jobs, gen_kwargs = task
# #         gen_model = ensure_adapter(adapter_path)

# #         results = []
# #         for (group_idx, prompt, count) in jobs:
# #             enc = tokenizer([prompt], return_tensors="pt").to(device)
# #             input_len = enc.input_ids.shape[1]
# #             with torch.inference_mode():
# #                 out = gen_model.generate(
# #                     **enc,
# #                     num_return_sequences=count,
# #                     max_new_tokens=gen_kwargs["max_new_tokens"],
# #                     do_sample=True,
# #                     temperature=gen_kwargs["temperature"],
# #                     top_p=gen_kwargs["top_p"],
# #                     pad_token_id=pad_id,
# #                     return_dict_in_generate=True,
# #                     output_scores=True,
# #                 )
# #             seqs = out.sequences                    # (count, input_len + gen_len)
# #             gen_token_ids = seqs[:, input_len:]     # (count, gen_len)
# #             # out.scores: tuple (len gen_len) of (count, vocab) tensors, the
# #             # per-step logits AFTER temperature/top_p warping, i.e. the actual
# #             # sampling distribution. At temperature=1.0 and top_p=1.0 (the
# #             # defaults) warping is the identity, so these equal the raw model
# #             # logprobs and the IS ratio in training is a clean
# #             # pi_learner / pi_sampler.
# #             #
# #             # Gather PER STEP, never stack the full (count, gen_len, vocab)
# #             # tensor: for a ~150k vocab and gen_len ~4k that stack is tens of GB
# #             # and would OOM the worker. Per-step peak is one (count, vocab) slice.
# #             n_seq = seqs.shape[0]
# #             gen_len = min(gen_token_ids.shape[1], len(out.scores))
# #             tok_logprobs = torch.empty(
# #                 (n_seq, gen_len), dtype=torch.float32, device=device
# #             )
# #             for t in range(gen_len):
# #                 step_lp = out.scores[t].float().log_softmax(dim=-1)   # (count, vocab)
# #                 tok_logprobs[:, t] = step_lp.gather(
# #                     1, gen_token_ids[:, t : t + 1]
# #                 ).squeeze(1)
# #                 del step_lp

# #             for i in range(n_seq):
# #                 gen_ids = gen_token_ids[i].tolist()[:gen_len]
# #                 lps = tok_logprobs[i].tolist()[:gen_len]
# #                 if eos_id is not None and eos_id in gen_ids:
# #                     cut = gen_ids.index(eos_id) + 1
# #                     gen_ids = gen_ids[:cut]
# #                     lps = lps[:cut]
# #                 text = tokenizer.decode(gen_ids, skip_special_tokens=True)
# #                 results.append((group_idx, text, gen_ids, lps))

# #         result_queue.put((rank, results))

# #     print(f"[worker {rank}] shutting down", flush=True)


# # class GenerationPool:
# #     """
# #     Manages num_workers persistent generation processes (one per GPU).
# #     """

# #     def __init__(self, model_name, num_workers, gpu_ids=None,
# #                  max_seq_length=4096, load_in_4bit=False):
# #         self.model_name = model_name
# #         self.num_workers = num_workers
# #         self.gpu_ids = gpu_ids or list(range(num_workers))
# #         assert len(self.gpu_ids) == num_workers

# #         ctx = mp.get_context("spawn")
# #         self.task_queues = [ctx.Queue() for _ in range(num_workers)]
# #         self.result_queue = ctx.Queue()
# #         ready_queue = ctx.Queue()

# #         self.procs = []
# #         for r in range(num_workers):
# #             p = ctx.Process(
# #                 target=_worker_loop,
# #                 args=(r, self.gpu_ids[r], model_name, max_seq_length,
# #                       load_in_4bit, self.task_queues[r], self.result_queue,
# #                       ready_queue),
# #                 daemon=True,
# #             )
# #             p.start()
# #             self.procs.append(p)

# #         # Wait for all workers to finish loading
# #         print(f"[pool] waiting for {num_workers} workers to load ...", flush=True)
# #         loaded = 0
# #         while loaded < num_workers:
# #             ready_queue.get()
# #             loaded += 1
# #             print(f"[pool] {loaded}/{num_workers} workers ready", flush=True)

# #     def generate_groups(self, prompts_by_group, group_size, adapter_path,
# #                          max_new_tokens, temperature, top_p):
# #         """
# #         prompts_by_group: list of prompts, one per group.

# #         Returns: dict group_idx -> list of (text, token_ids, behavior_logprobs),
# #         each list of length group_size (order within a group is not meaningful).
# #         behavior_logprobs is a per-token list aligned 1:1 with token_ids.
# #         """
# #         num_groups = len(prompts_by_group)
# #         worker_jobs = distribute_jobs(prompts_by_group, group_size, self.num_workers)
# #         gen_kwargs = {
# #             "max_new_tokens": max_new_tokens,
# #             "temperature": temperature,
# #             "top_p": top_p,
# #         }

# #         # Dispatch one task per worker (some may have empty job lists)
# #         active = 0
# #         for r in range(self.num_workers):
# #             self.task_queues[r].put((0, adapter_path, worker_jobs[r], gen_kwargs))
# #             active += 1

# #         # Collect results from all workers
# #         by_group = {g: [] for g in range(num_groups)}
# #         for _ in range(active):
# #             rank, results = self.result_queue.get()
# #             for (group_idx, text, token_ids, behavior_logprobs) in results:
# #                 by_group[group_idx].append((text, token_ids, behavior_logprobs))

# #         return by_group

# #     def shutdown(self):
# #         for r in range(self.num_workers):
# #             try:
# #                 self.task_queues[r].put(None)
# #             except Exception:
# #                 pass
# #         for p in self.procs:
# #             p.join(timeout=10)
# #             if p.is_alive():
# #                 p.terminate()