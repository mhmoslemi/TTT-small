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
    """Single-GPU fallback: generate on the training model itself."""

    def __init__(self, cfg, backend, model, tokenizer, seed=None):
        self.cfg = cfg
        self.backend = backend
        self.model = model
        self.tokenizer = tokenizer
        self.seed = seed

    def complete(self, messages: List[Dict], adapter_path=None, step_idx: int = 0,
                 max_new_tokens: Optional[int] = None,
                 temperature: Optional[float] = None) -> str:
        import torch

        prompt_text = render_chat(self.tokenizer, messages)
        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.model.device)
        input_len = inputs.input_ids.shape[1]
        eos_id = self.tokenizer.eos_token_id
        pad_id = self.tokenizer.pad_token_id or eos_id

        if self.seed is not None:
            # Reseeding is what keeps the single-GPU path reproducible: without
            # it these calls advance the global RNG and every later step
            # diverges from the no-memory run.
            s = (int(self.seed) * 1_000_003 + int(step_idx) * 1009 + 13) % (2 ** 31 - 1)
            torch.manual_seed(s)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(s)

        self.backend.set_inference_mode()
        try:
            with torch.inference_mode():
                out = self.model.generate(
                    **inputs,
                    max_new_tokens=int(max_new_tokens or self.cfg.max_new_tokens),
                    do_sample=True,
                    temperature=float(temperature if temperature is not None
                                      else self.cfg.temperature),
                    top_p=float(self.cfg.top_p),
                    pad_token_id=pad_id,
                    num_return_sequences=1)
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
        return out

    def complete(self, messages: List[Dict], adapter_path=None, step_idx: int = 0,
                 max_new_tokens: Optional[int] = None,
                 temperature: Optional[float] = None) -> str:
        got = self.complete_many([messages], adapter_path, step_idx,
                                 max_new_tokens, temperature)
        return got[0] if got else ""


def make_memory_llm(cfg, backend=None, model=None, tokenizer=None,
                    gen_pool=None, seed=None, verbose: bool = True):
    if gen_pool is not None and bool(getattr(cfg, "use_gen_pool", True)):
        if verbose:
            print("[memory] memory calls run on the generation pool")
        return PoolMemoryLLM(cfg, gen_pool, tokenizer)
    if verbose:
        print("[memory] memory calls run in-process on the training model")
    return InProcessMemoryLLM(cfg, backend, model, tokenizer, seed=seed)
