"""
Local judge worker process for the Multi-Agent Elo re-ranker.

Why a separate PROCESS (not a thread in the trainer):
  * The training process loads the policy under Unsloth, which monkey-patches
    transformers GLOBALLY on import. A plain-HF judge in the same process would
    inherit those patches. gen_workers.py are separate processes for exactly
    this reason; the judge follows the same pattern.
  * A separate process has its own GIL, so batched judge generation never
    competes with the trainer's Python for the interpreter lock.
  * Pin the judge to its own GPU (judge_gpu) so it doesn't contend for compute
    with training or rollout generation. Co-residing on an already-used GPU is
    fine if there's spare VRAM (use judge_load_in_4bit to shrink the judge).

No HTTP server. Plain torch.multiprocessing with a persistent worker + queues.
The worker does BATCHED generation: the re-ranker sends a list of prompt
message-lists, the worker returns a list of completion strings (1:1, in order).

Protocol:
  client -> task_queue:   list[ list[ {"role","content"} ] ]   (a batch of prompts)
  worker -> result_queue: list[str]                            (completions, same order)
  client -> task_queue:   None                                 (shutdown)
"""

import os
import multiprocessing as mp


def _worker_loop(gpu_id, model_name, max_seq_length, max_new_tokens,
                 temperature, top_p, load_in_4bit,
                 task_queue, result_queue, ready_queue):
    # Pin to our GPU BEFORE importing torch / touching any CUDA state.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda:0"  # the only visible device inside this process

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"  # decoder-only batched generation

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

        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        if not load_in_4bit:
            model = model.to(device)
        model.eval()
    except Exception as e:
        # Tell the parent we failed to load; the parent disables the re-ranker.
        ready_queue.put(("error", repr(e)))
        return

    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id or eos_id

    def render(messages):
        # enable_thinking=False keeps outputs short so the verdict fits inside
        # max_new_tokens (Qwen3 etc. otherwise emit long <think> blocks that can
        # eat the whole budget before the "better idea:" line is produced).
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )

    ready_queue.put(("ok", None))

    while True:
        task = task_queue.get()
        if task is None:
            break
        batch_messages = task                      # list of message-lists
        prompts = [render(m) for m in batch_messages]
        try:
            enc = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
            input_len = enc.input_ids.shape[1]     # left-padded => same for all rows
            with torch.inference_mode():
                out = model.generate(
                    **enc,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    pad_token_id=pad_id,
                )
            texts = []
            for i in range(out.shape[0]):
                gen_ids = out[i, input_len:].tolist()
                if eos_id is not None and eos_id in gen_ids:
                    gen_ids = gen_ids[: gen_ids.index(eos_id) + 1]
                texts.append(tokenizer.decode(gen_ids, skip_special_tokens=True))
            result_queue.put(texts)
        except Exception:
            # Return blanks so the re-ranker treats these matches as no-ops.
            result_queue.put(["" for _ in batch_messages])


class LocalJudgeClient:
    """In-process handle to the judge worker process. Implements the judge
    interface used by MultiAgentReRanker: complete_batch / complete / close.
    The constructor BLOCKS until the model is loaded (bounded by load_timeout_s)
    so a misconfigured GPU surfaces as an exception instead of a silent hang."""

    def __init__(self, cfg, load_timeout_s: float = 600.0):
        self.cfg = cfg
        ctx = mp.get_context("spawn")
        self.task_queue = ctx.Queue()
        self.result_queue = ctx.Queue()
        ready_queue = ctx.Queue()

        self.proc = ctx.Process(
            target=_worker_loop,
            args=(cfg.judge_gpu, cfg.model, cfg.max_seq_length,
                  cfg.max_tokens, cfg.temperature, cfg.top_p,
                  cfg.judge_load_in_4bit,
                  self.task_queue, self.result_queue, ready_queue),
            daemon=True,
        )
        self.proc.start()
        print(f"[reranker] loading local judge {cfg.model} on GPU "
              f"{cfg.judge_gpu} ...", flush=True)
        try:
            status, err = ready_queue.get(timeout=load_timeout_s)
        except Exception:
            self.close()
            raise RuntimeError(f"local judge failed to load within {load_timeout_s}s")
        if status != "ok":
            self.close()
            raise RuntimeError(f"local judge load error: {err}")
        print(f"[reranker] local judge ready on GPU {cfg.judge_gpu}", flush=True)

    def complete_batch(self, list_of_messages):
        """Send a batch of prompts, get a list of completions back (in order)."""
        if not list_of_messages:
            return []
        self.task_queue.put(list(list_of_messages))
        return self.result_queue.get()

    def complete(self, messages):
        return self.complete_batch([messages])[0]

    def close(self):
        try:
            self.task_queue.put(None)
        except Exception:
            pass
        if self.proc is not None:
            self.proc.join(timeout=10)
            if self.proc.is_alive():
                self.proc.terminate()
            self.proc = None