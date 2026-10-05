"""
Memory module (Sec. 2.2 of the EVOLVE paper): lesson extraction, an additive
bank with importance, top-m retrieval, and budgeted injection into the
generation prompt.

Master switch: the config key `memory` (CLI --memory / --no-memory). When it is
off, MemoryConfig.from_dict ignores every memory_* key and setup_memory returns
(disabled_cfg, None, None), so every caller guard is `if memory is not None`.

Wiring in a trainer:

    from memory import MemoryConfig, setup_memory, RolloutRecord
    from memory import build_injection, inject_block, parent_query_text

    # early, before the model loads, so the context top-up is applied
    mem_cfg = MemoryConfig.from_dict(merged)
    if mem_cfg.enabled and mem_cfg.grant_context:
        cfg.max_seq_length += mem_cfg.token_budget

    mem_cfg, memory, extractor = setup_memory(
        merged, problem, cfg, mem_cfg=mem_cfg, backend=backend, model=model,
        tokenizer=tokenizer, gen_pool=gen_pool, exp_dir=exp_dir, seed=cfg.seed)

    # per group, before generation
    if memory is not None:
        lessons = memory.retrieve(query_text)
        block, n_tok, kept = build_injection(lessons, tokenizer,
                                             mem_cfg.token_budget)
        messages = inject_block(messages, block, mem_cfg.inject_mode)

    # once per step, after every rollout is scored
    if memory is not None:
        extractor.update(records, step, adapter_path=adapter_path)
        memory.save()
"""

from memory.bank import MemoryBank
from memory.config import MemoryConfig
from memory.embedding import Embedder
from memory.extractor import LessonExtractor, build_meta_description
from memory.llm import make_memory_llm
from memory.prompts import (ExtractionResult, build_injection, count_tokens,
                            inject_block, parent_query_text, parse_extraction,
                            parse_lessons, render_memory_block)
from memory.types import FAILURE, SUCCESS, Lesson, RolloutRecord

__all__ = [
    "MemoryBank", "MemoryConfig", "Embedder", "LessonExtractor",
    "ExtractionResult", "build_meta_description", "make_memory_llm",
    "build_injection", "inject_block", "count_tokens", "parent_query_text",
    "parse_extraction", "parse_lessons", "render_memory_block",
    "Lesson", "RolloutRecord", "SUCCESS", "FAILURE", "setup_memory",
]


def setup_memory(merged: dict, problem, cfg, mem_cfg=None, backend=None,
                 model=None, tokenizer=None, gen_pool=None, exp_dir=None,
                 seed=None, resume_from=None, verbose: bool = True):
    """
    Build the whole module, or nothing at all.

    Returns (mem_cfg, memory, extractor). When the master flag is off, memory
    and extractor are None and mem_cfg.enabled is False.

    Pass `mem_cfg` when the trainer already built it early to apply the
    max_seq_length top-up; it is not rebuilt in that case.

    `resume_from` points at a memory.json from an earlier run to carry a bank
    across runs. Leave it None for the paper's setting, where M starts empty.
    """
    if mem_cfg is None:
        mem_cfg = MemoryConfig.from_dict(merged, verbose=verbose)
    if not mem_cfg.enabled:
        if verbose:
            print("[init] memory OFF")
        return mem_cfg, None, None

    embedder = Embedder(mem_cfg.embed_backend, mem_cfg.embed_model,
                        mem_cfg.embed_dim, mem_cfg.embed_device, verbose=verbose)
    bank = MemoryBank(mem_cfg, embedder)

    if resume_from:
        n = bank.load(resume_from)
        if verbose:
            print(f"[memory] resumed {n} lesson(s) from {resume_from}")
    if mem_cfg.persist and exp_dir is not None:
        from pathlib import Path
        bank.attach(Path(exp_dir) / "memory.json")

    llm = make_memory_llm(mem_cfg, backend=backend, model=model,
                          tokenizer=tokenizer, gen_pool=gen_pool, seed=seed,
                          verbose=verbose)
    extractor = LessonExtractor(
        mem_cfg, llm,
        build_meta_description(problem, cfg),
        bank=bank,
        fail_score=float(getattr(problem, "fail_score", 0.0)),
    )

    if verbose:
        print(f"[init] {mem_cfg.describe()}")
        if len(bank) == 0:
            print("[memory] bank is empty; step 0 prompts are identical to a "
                  "no-memory run")
    return mem_cfg, bank, extractor
