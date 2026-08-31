"""
The completion handle the memory maker and the lookup selector share.

Sec. 2.2: "the same LLM backbone as the generator, but with dedicated extraction
prompts." Same weights, same LoRA adapter, same generation path.

complete_many exists for the lookup: 8 parents means 8 selection prompts, and
sending them as one pool round with group_size=1 costs one round trip instead of
eight. Both classes offer it; the in-process one just loops.

step_idx is offset before it reaches the seeded generator so the module's own
calls draw from a different slot than the rollouts. Extraction and lookup use
different offsets, so neither can shift the other or the rollout stream.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

EXTRACT_STEP_OFFSET = 1_000_000
LOOKUP_STEP_OFFSET = 2_000_000
CURATE_STEP_OFFSET = 3_000_000


def render_chat(tokenizer, messages: List[Dict]) -> str:
    """Same call and fallback the trainer uses for the rollout prompt."""
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False)
    except TypeError:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)


class InProcessMemoryLLM:
    """HF/Unsloth fallback: generate cautiously on the training model."""

    def __init__(self, cfg, backend, model, tokenizer, seed=None,
                 max_seq_length=None):
        self.cfg = cfg
        self.backend = backend
        self.model = model
        self.tokenizer = tokenizer
        self.seed = seed
        model_limit = getattr(getattr(model, "config", None),
                              "max_position_embeddings", 0)
        self.max_seq_length = int(max_seq_length or model_limit or 32768)

    def complete(self, messages: List[Dict], adapter_path=None, step_idx: int = 0,
                 max_new_tokens: Optional[int] = None,
                 temperature: Optional[float] = None) -> str:
        import torch

        prompt_text = render_chat(self.tokenizer, messages)
        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.model.device)
        input_len = inputs.input_ids.shape[1]
        eos_id = self.tokenizer.eos_token_id
        pad_id = self.tokenizer.pad_token_id or eos_id

        requested_tokens = int(max_new_tokens or self.cfg.max_new_tokens)
        context_room = self.max_seq_length - int(input_len)
        attempt_tokens = min(requested_tokens, context_room)
        if attempt_tokens < 1:
            print("[memory] prompt filled the model context; skipping completion")
            return ""

        call_seed = None
        if self.seed is not None:
            # Reseeding is what keeps the single-GPU path reproducible: without
            # it these calls advance the global RNG and every later step
            # diverges from the no-memory run.
            call_seed = (int(self.seed) * 1_000_003
                         + int(step_idx) * 1009 + 13) % (2 ** 31 - 1)

        self.backend.set_inference_mode()
        try:
            while True:
                if call_seed is not None:
                    # Retry the same stochastic call rather than advancing the
                    # memory stream merely because a larger allocation failed.
                    torch.manual_seed(call_seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(call_seed)
                try:
                    with torch.inference_mode():
                        out = self.model.generate(
                            **inputs,
                            max_new_tokens=attempt_tokens,
                            do_sample=True,
                            temperature=float(
                                temperature if temperature is not None
                                else self.cfg.temperature),
                            top_p=float(self.cfg.top_p),
                            pad_token_id=pad_id,
                            num_return_sequences=1)
                    break
                except Exception as exc:
                    oom_type = getattr(torch, "OutOfMemoryError", None)
                    is_oom = (
                        (oom_type is not None and isinstance(exc, oom_type))
                        or "out of memory" in str(exc).lower()
                    )
                    next_tokens = max(128, attempt_tokens // 2)
                    if not is_oom or next_tokens >= attempt_tokens:
                        raise
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    print(f"[memory] in-process generation OOM at "
                          f"{attempt_tokens} new tokens; retrying with "
                          f"{next_tokens}")
                    attempt_tokens = next_tokens
        finally:
            # The caller may be mid-train_step; leaving inference mode on would
            # change the update that follows.
            self.backend.set_training_mode()

        gen_ids = out[0, input_len:].tolist()
        if eos_id is not None and eos_id in gen_ids:
            gen_ids = gen_ids[: gen_ids.index(eos_id) + 1]
        return self.tokenizer.decode(gen_ids, skip_special_tokens=True)

    def complete_many(self, messages_list: Sequence[List[Dict]], adapter_path=None,
                      step_idx: int = 0, max_new_tokens: Optional[int] = None,
                      temperature: Optional[float] = None) -> List[str]:
        return [self.complete(m, adapter_path, step_idx + i, max_new_tokens,
                              temperature)
                for i, m in enumerate(messages_list)]


class PoolMemoryLLM:
    """Multi-GPU: run on the idle workers holding the current adapter."""

    def __init__(self, cfg, gen_pool, tokenizer):
        self.cfg = cfg
        self.pool = gen_pool
        self.tokenizer = tokenizer

    def complete_many(self, messages_list: Sequence[List[Dict]], adapter_path=None,
                      step_idx: int = 0, max_new_tokens: Optional[int] = None,
                      temperature: Optional[float] = None) -> List[str]:
        prompts = [render_chat(self.tokenizer, m) for m in messages_list]
        if not prompts:
            return []
        out = [""] * len(prompts)
        sequential = bool(getattr(self.pool, "sequential", False))
        was_active = bool(getattr(self.pool, "active", False))
        release_after = sequential and not was_active
        try:
            for group_idx, job_results in self.pool.iter_group_jobs(
                    prompts_by_group=prompts,
                    group_size=1,
                    adapter_path=adapter_path,
                    max_new_tokens=int(max_new_tokens or self.cfg.max_new_tokens),
                    temperature=float(temperature if temperature is not None
                                      else self.cfg.temperature),
                    top_p=float(self.cfg.top_p),
                    step_idx=int(step_idx),
                    show_progress=False):
                for (text, _ids) in job_results:
                    if 0 <= group_idx < len(out):
                        out[group_idx] = text
        finally:
            # Shared-card vLLM wakes by offloading the trainer. Memory calls are
            # self-contained phases, so restore the trainer even if generation
            # or parsing raises. If the pool was already awake, its owner keeps
            # responsibility for ending the surrounding rollout phase.
            if release_after:
                self.pool.release()
        return out

    def complete(self, messages: List[Dict], adapter_path=None, step_idx: int = 0,
                 max_new_tokens: Optional[int] = None,
                 temperature: Optional[float] = None) -> str:
        got = self.complete_many([messages], adapter_path, step_idx,
                                 max_new_tokens, temperature)
        return got[0] if got else ""


def make_memory_llm(cfg, backend=None, model=None, tokenizer=None,
                    gen_pool=None, seed=None, max_seq_length=None,
                    verbose: bool = True):
    if gen_pool is not None and bool(getattr(cfg, "use_gen_pool", True)):
        if verbose:
            print("[memory] memory calls run on the generation pool")
        return PoolMemoryLLM(cfg, gen_pool, tokenizer)
    if verbose:
        print("[memory] memory calls run in-process on the training model")
    return InProcessMemoryLLM(
        cfg, backend, model, tokenizer, seed=seed,
        max_seq_length=max_seq_length)
