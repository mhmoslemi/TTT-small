"""
Rollout generation.

Two interchangeable generators behind one interface:

    InProcessGenerator   generation.num_gpus = 1. The training model generates
                         directly. Simplest, and the only option when the model
                         fits once.

    PoolGenerator        generation.num_gpus > 1. One persistent worker process
                         per GPU, each holding a plain transformers copy of the
                         base model. Workers reload the LoRA adapter from disk at
                         the start of every step, which is how the freshly
                         updated policy reaches them; the main process keeps its
                         (possibly Unsloth) model for training only.

The batch is unbalanced by construction -- a virtual target asks for 1 sample and
a leaf target for k -- so jobs are split by individual sample count rather than
by target, otherwise one worker would take every leaf expansion.

The pool follows the structure of the reference implementation's gen_workers.py.
It needs real GPUs to exercise; on a single-GPU or CPU box use num_gpus = 1.
"""

import os
from typing import List, Optional, Sequence, Tuple

# A job: (group_id, rendered_prompt_text, num_samples)
Job = Tuple[int, str, int]


def split_jobs(jobs: Sequence[Job], num_workers: int) -> List[List[Job]]:
    """
    Spread jobs over workers by SAMPLE count, splitting a large job if needed,
    so an 8-sample leaf expansion does not land whole on one worker while
    another gets a single virtual sample.
    """
    buckets: List[List[Job]] = [[] for _ in range(num_workers)]
    loads = [0] * num_workers
    for group_id, prompt, count in sorted(jobs, key=lambda j: -j[2]):
        remaining = count
        while remaining > 0:
            w = loads.index(min(loads))
            # Give this worker an even share, at least one sample.
            take = max(1, min(remaining, -(-count // num_workers)))
            buckets[w].append((group_id, prompt, take))
            loads[w] += take
            remaining -= take
    return buckets


class InProcessGenerator:
    name = "in_process"

    def __init__(self, backbone, gen_cfg, progress: bool = True):
        self.backbone = backbone
        self.cfg = gen_cfg
        self.progress = progress

    def generate(self, jobs: Sequence[Job], adapter_path: Optional[str] = None
                 ) -> dict:
        """
        Returns {group_id: [(text, token_ids), ...]}.

        The whole step's B_t rollouts go through as few generate() calls as the
        KV cache allows, rather than one call per target. Decoding re-reads the
        full weight matrix on every step whatever the batch size, so a target
        generating alone at batch 4 wastes most of the card's bandwidth; folding
        n targets into one call costs almost nothing extra per step.

        Targets ask for different counts (leaf -> k, virtual -> 1), and
        num_return_sequences can only apply a single count to a whole batch, so
        a prompt is simply repeated `count` times instead.
        """
        from runio.progress import make_bar

        flat: List[Tuple[int, str]] = []
        for group_id, messages, count in jobs:
            text = self.backbone.render(messages)
            flat.extend((group_id, text) for _ in range(count))
        if not flat:
            return {}

        # Sort by length so each padded batch is as uniform as possible; padding
        # is decoded at full cost, so mixing a short prompt with a long one
        # wastes the difference on every step.
        flat.sort(key=lambda item: len(item[1]))

        cap = int(getattr(self.cfg, "batch_size", 0) or 0) or len(flat)
        chunks = [flat[i:i + cap] for i in range(0, len(flat), cap)]

        bar = make_bar(len(chunks) * int(self.cfg.max_new_tokens),
                       f"generate {len(flat)} rollouts "
                       f"in {len(chunks)} batch(es) of <={cap}",
                       unit="step", enabled=self.progress)

        out: dict = {}
        done = 0
        try:
            for chunk in chunks:
                samples = self._generate_chunk([t for _, t in chunk], bar)
                for (group_id, _), sample in zip(chunk, samples):
                    out.setdefault(group_id, []).append(sample)
                done += len(chunk)
                bar.set_postfix_str(f"{done}/{len(flat)} rollouts")
        finally:
            bar.close()
        return out

    def _generate_chunk(self, texts: List[str], bar) -> List[Tuple[str, List[int]]]:
        """
        One generate() call, halving the batch and retrying on OOM.

        generation.batch_size is a hint rather than a guarantee: KV cache use
        depends on prompt length, which grows as parent programs accumulate, so
        a batch that fit at step 0 can fail at step 20. Backing off beats losing
        a run that has been going for hours.
        """
        try:
            return self.backbone.sample_batch(
                texts,
                max_new_tokens=self.cfg.max_new_tokens,
                temperature=self.cfg.temperature,
                top_p=self.cfg.top_p,
                on_step=lambda: bar.update(1),
                think_budget=getattr(self.cfg, "think_budget", 0),
                think_close_tag=getattr(self.cfg, "think_close_tag", "</think>"),
                think_force_text=getattr(self.cfg, "think_force_text",
                                         "\n</think>\n\n"),
                stop_on_code=getattr(self.cfg, "stop_on_code_block", True),
                stop_check_every=getattr(self.cfg, "stop_check_every", 16),
            )
        except Exception as e:
            if not _is_oom(e) or len(texts) == 1:
                raise
            half = len(texts) // 2
            print(f"\n[generation] OOM at batch {len(texts)}; retrying as "
                  f"{half} + {len(texts) - half}", flush=True)
            _free_cuda_cache()
            return (self._generate_chunk(texts[:half], bar)
                    + self._generate_chunk(texts[half:], bar))

    def shutdown(self):
        pass


def _is_oom(exc: BaseException) -> bool:
    try:
        import torch
        if isinstance(exc, torch.cuda.OutOfMemoryError):
            return True
    except Exception:
        pass
    return "out of memory" in str(exc).lower()


def _free_cuda_cache() -> None:
    try:
        import gc

        import torch
        gc.collect()
        torch.cuda.empty_cache()
    except Exception:
        pass


# ======================================================================
# Multi-GPU pool
# ======================================================================
def _worker_loop(rank, gpu_id, model_name, max_seq_length, load_in_4bit,
                 task_queue, result_queue, ready_queue):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from llm.backend import as_tokenizer

    tokenizer = as_tokenizer(
        AutoTokenizer.from_pretrained(model_name, trust_remote_code=True))
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs = dict(torch_dtype=torch.bfloat16, trust_remote_code=True)
    if load_in_4bit:
        try:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
        except ImportError:
            pass

    base = AutoModelForCausalLM.from_pretrained(model_name, **kwargs).to("cuda:0")
    base.eval()
    model = base
    loaded_adapter = None
    ready_queue.put(rank)

    while True:
        task = task_queue.get()
        if task is None:
            break
        adapter_path, jobs, gen_kwargs = task

        # Weight sync: pick up the adapter the trainer just wrote.
        if adapter_path and adapter_path != loaded_adapter:
            try:
                from peft import PeftModel
                model = PeftModel.from_pretrained(base, adapter_path)
                model.eval()
                loaded_adapter = adapter_path
            except Exception as e:
                print(f"[worker {rank}] adapter load failed: {e!r}", flush=True)
                model = base

        for group_id, prompt_text, count in jobs:
            try:
                tokenizer.padding_side = "left"
                enc = tokenizer(text=[prompt_text], return_tensors="pt",
                                truncation=True, max_length=max_seq_length,
                                add_special_tokens=False).to(model.device)
                input_len = enc["input_ids"].shape[1]
                with torch.no_grad():
                    out = model.generate(
                        **enc, num_return_sequences=count,
                        pad_token_id=tokenizer.pad_token_id, **gen_kwargs)
                batch = []
                for row in range(out.shape[0]):
                    new = out[row, input_len:]
                    ids = [int(t) for t in new if int(t) != tokenizer.pad_token_id]
                    batch.append((tokenizer.decode(new, skip_special_tokens=True), ids))
            except Exception as e:
                print(f"[worker {rank}] generation failed: {e!r}", flush=True)
                batch = [("", []) for _ in range(count)]
            result_queue.put((rank, group_id, batch))


class PoolGenerator:
    name = "pool"

    def __init__(self, model_cfg, gen_cfg, render_fn=None, progress: bool = True):
        import torch.multiprocessing as mp

        self.cfg = gen_cfg
        self.progress = progress
        # Jobs carry chat messages; workers need flat text. Rendering happens
        # here so every worker sees byte-identical prompts from one tokenizer.
        self.render_fn = render_fn or (lambda m: m if isinstance(m, str) else str(m))
        self.num_workers = int(gen_cfg.num_gpus)
        if gen_cfg.gpu_ids:
            self.gpu_ids = [int(x) for x in str(gen_cfg.gpu_ids).split(",") if x.strip()]
        else:
            self.gpu_ids = list(range(self.num_workers))
        if len(self.gpu_ids) != self.num_workers:
            raise ValueError(
                f"generation.num_gpus={self.num_workers} but gpu_ids has "
                f"{len(self.gpu_ids)} entries: {self.gpu_ids}")

        ctx = mp.get_context("spawn")
        self.task_queues = [ctx.Queue() for _ in range(self.num_workers)]
        self.result_queue = ctx.Queue()
        ready = ctx.Queue()

        self.procs = []
        for rank in range(self.num_workers):
            proc = ctx.Process(
                target=_worker_loop,
                args=(rank, self.gpu_ids[rank], model_cfg.name,
                      model_cfg.max_seq_length, model_cfg.load_in_4bit,
                      self.task_queues[rank], self.result_queue, ready),
                daemon=True)
            proc.start()
            self.procs.append(proc)

        print(f"[pool] waiting for {self.num_workers} workers on GPUs {self.gpu_ids} ...",
              flush=True)
        for i in range(self.num_workers):
            ready.get()
            print(f"[pool] {i + 1}/{self.num_workers} ready", flush=True)

    def generate(self, jobs: Sequence[Job], adapter_path: Optional[str] = None
                 ) -> dict:
        gen_kwargs = {
            "max_new_tokens": int(self.cfg.max_new_tokens),
            "do_sample": self.cfg.temperature > 0,
            "temperature": float(self.cfg.temperature),
            "top_p": float(self.cfg.top_p),
        }
        rendered = [(gid, self.render_fn(payload), count)
                    for gid, payload, count in jobs]
        buckets = split_jobs(rendered, self.num_workers)
        expected = sum(count for bucket in buckets for (_, _, count) in bucket)

        for rank in range(self.num_workers):
            self.task_queues[rank].put(
                (adapter_path, buckets[rank], gen_kwargs))

        # Workers are separate processes, so the finest granularity available
        # here is a completed job, not a token.
        from runio.progress import make_bar
        bar = make_bar(expected, f"generate on {self.num_workers} GPUs",
                       unit="rollout", enabled=self.progress)

        out: dict = {}
        collected = 0
        try:
            while collected < expected:
                _, group_id, batch = self.result_queue.get()
                out.setdefault(group_id, []).extend(batch)
                collected += len(batch)
                bar.update(len(batch))
        finally:
            bar.close()
        return out

    def shutdown(self):
        for queue in self.task_queues:
            try:
                queue.put(None)
            except Exception:
                pass
        for proc in self.procs:
            proc.join(timeout=10)
            if proc.is_alive():
                proc.terminate()


def build_generator(cfg, backbone):
    """cfg is the full Config."""
    if int(cfg.generation.num_gpus) > 1:
        return PoolGenerator(cfg.model, cfg.generation,
                             render_fn=backbone.render,
                             progress=cfg.run.progress)
    return InProcessGenerator(backbone, cfg.generation, progress=cfg.run.progress)
