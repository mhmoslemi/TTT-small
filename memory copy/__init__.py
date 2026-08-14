"""
Memory module (Sec. 2.2 of the EVOLVE paper): lesson extraction, an additive
bank, top-m retrieval, and injection into the generation prompt.

Master switch: the single config key `memory` (CLI --memory / --no-memory).
When it is off, MemoryConfig.from_dict ignores every memory_* key and
setup_memory returns (disabled_cfg, None, None), so the caller's guards are
all `if memory is not None`.

Typical wiring in a trainer:

    from memory import setup_memory, RolloutRecord, inject_memories

    mem_cfg, memory, extractor = setup_memory(merged, problem, cfg,
                                              backend=backend, model=model,
                                              tokenizer=tokenizer,
                                              gen_pool=gen_pool,
                                              exp_dir=exp_dir)

    # per group, before generation
    if memory is not None:
        lessons = memory.retrieve(query_text)
        messages = inject_memories(messages, lessons, mem_cfg.inject_mode)

    # once per step, after every rollout is scored
    if memory is not None:
        memory.add_many(extractor.extract(records, step, adapter_path))
        memory.save()
"""

from memory.bank import MemoryBank
from memory.config import MemoryConfig
from memory.embedding import Embedder
from memory.extractor import LessonExtractor, build_meta_description
from memory.llm import make_memory_llm
from memory.prompts import (inject_memories, parent_query_text,
                            render_memory_block)
from memory.types import FAILURE, SUCCESS, Lesson, RolloutRecord

__all__ = [
    "MemoryBank", "MemoryConfig", "Embedder", "LessonExtractor",
    "build_meta_description", "make_memory_llm", "inject_memories",
    "parent_query_text", "render_memory_block", "Lesson", "RolloutRecord",
    "SUCCESS", "FAILURE", "setup_memory",
]


def setup_memory(merged: dict, problem, cfg, backend=None, model=None,
                 tokenizer=None, gen_pool=None, exp_dir=None,
                 resume_from=None, verbose: bool = True):
    """
    Build the whole module from the merged config dict, or nothing at all.

    Returns (mem_cfg, memory, extractor). When the master flag is off, memory
    and extractor are None and mem_cfg.enabled is False.

    `resume_from` points at a memory.json from an earlier run to carry a bank
    across runs. Leave it None for the paper's setting, where M starts empty.
    """
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
                          tokenizer=tokenizer, gen_pool=gen_pool, verbose=verbose)
    extractor = LessonExtractor(
        mem_cfg, llm,
        build_meta_description(problem, cfg),
        fail_score=float(getattr(problem, "fail_score", 0.0)),
    )

    if verbose:
        print(f"[init] {mem_cfg.describe()}")
    return mem_cfg, bank, extractor
