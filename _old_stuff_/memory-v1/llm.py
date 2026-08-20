"""
The completion handle the memory maker uses.

Sec. 2.2: "The memory module uses the same LLM backbone as the generator, but
with dedicated extraction prompts."

  PoolMemoryLLM       when a multi-GPU GenerationPool exists. The two calls per
                      step run on an idle worker holding the same LoRA adapter
                      the rollouts were sampled with, so the main process keeps
                      its VRAM headroom for the backward pass that follows.

  InProcessMemoryLLM  single-GPU fallback: generate on the training model
                      directly, with inference mode on and training mode
                      restored afterwards.

Both take step_idx and offset it by MEMORY_STEP_OFFSET before handing it to the
seeded generator. Without the offset, the memory call at step t would draw from
the same seed slot as the rollouts at step t; with it, the two streams stay
independent, and the rollouts at step t are identical whether or not the memory
module ran.
"""

from __future__ import annotations

from typing import Dict, List, Optional

MEMORY_STEP_OFFSET = 1_000_000


def render_chat(tokenizer, messages: List[Dict]) -> str:
    """Same call the trainer makes, with the same enable_thinking fallback."""
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )


class InProcessMemoryLLM:
    def __init__(self, cfg, backend, model, tokenizer, seed=None):
        self.cfg = cfg
        self.backend = backend
        self.model = model
        self.tokenizer = tokenizer
        self.seed = seed

    def complete(self, messages: List[Dict], adapter_path=None,
                 step_idx: int = 0) -> str:
        import torch

        prompt_text = render_chat(self.tokenizer, messages)
        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.model.device)
        input_len = inputs.input_ids.shape[1]
        eos_id = self.tokenizer.eos_token_id
        pad_id = self.tokenizer.pad_token_id or eos_id

        if self.seed is not None:
            # Reseeding here is what keeps the single-GPU path reproducible:
            # otherwise these two calls advance the global RNG and every
            # subsequent step diverges from the no-memory run.
            s = (int(self.seed) * 1_000_003
                 + (MEMORY_STEP_OFFSET + int(step_idx)) * 1009 + 13) % (2**31 - 1)
            torch.manual_seed(s)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(s)

        self.backend.set_inference_mode()
        try:
            with torch.inference_mode():
                out = self.model.generate(
                    **inputs,
                    max_new_tokens=int(self.cfg.max_new_tokens),
                    do_sample=True,
                    temperature=float(self.cfg.temperature),
                    top_p=float(self.cfg.top_p),
                    pad_token_id=pad_id,
                    num_return_sequences=1,
                )
        finally:
            # The caller is mid-train_step; leaving the model in inference mode
            # would change the update that follows.
            self.backend.set_training_mode()

        gen_ids = out[0, input_len:].tolist()
        if eos_id is not None and eos_id in gen_ids:
            gen_ids = gen_ids[: gen_ids.index(eos_id) + 1]
        return self.tokenizer.decode(gen_ids, skip_special_tokens=True)


class PoolMemoryLLM:
    def __init__(self, cfg, gen_pool, tokenizer):
        self.cfg = cfg
        self.pool = gen_pool
        self.tokenizer = tokenizer

    def complete(self, messages: List[Dict], adapter_path=None,
                 step_idx: int = 0) -> str:
        prompt_text = render_chat(self.tokenizer, messages)
        texts = []
        for _group_idx, job_results in self.pool.iter_group_jobs(
                prompts_by_group=[prompt_text],
                group_size=1,
                adapter_path=adapter_path,
                max_new_tokens=int(self.cfg.max_new_tokens),
                temperature=float(self.cfg.temperature),
                top_p=float(self.cfg.top_p),
                step_idx=MEMORY_STEP_OFFSET + int(step_idx),
                show_progress=False):
            for (text, _ids) in job_results:
                texts.append(text)
        return texts[0] if texts else ""


def make_memory_llm(cfg, backend=None, model=None, tokenizer=None,
                    gen_pool=None, seed=None, verbose: bool = True):
    if gen_pool is not None and bool(getattr(cfg, "use_gen_pool", True)):
        if verbose:
            print("[memory] memory maker runs on the generation pool")
        return PoolMemoryLLM(cfg, gen_pool, tokenizer)
    if verbose:
        print("[memory] memory maker runs in-process on the training model")
    return InProcessMemoryLLM(cfg, backend, model, tokenizer, seed=seed)
