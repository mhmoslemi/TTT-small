"""
Memory module (Sec. 2.2), revised after the four-run experiment.

Two things changed and one component is gone.

  Retrieval is not RAG any more. There are no embeddings anywhere in this
  package. The model is shown a one-line index of the ENTIRE bank and names the
  ids it wants; the bodies of those lessons go into its generation prompt. See
  lookup.py for why: cosine retrieval left 87-89% of lessons unused and sent
  88-96% of retrieval mass to code, which is the least transferable content in
  the bank.

  Extraction now forbids constructions structurally. Lessons declare a `scope`,
  global-scope lessons may not contain code, and hygiene.py rejects coordinate
  and lattice construction at any scope. Extraction also sees each child's
  PARENT and score, so on a plateau it has a delta to talk about instead of
  restating the construction.

Master switch: the config key `memory` (CLI --memory / --no-memory).

Wiring in a trainer:

    from memory import MemoryConfig, setup_memory, RolloutRecord
    from memory import build_injection, inject_block

    mem_cfg = MemoryConfig.from_dict(merged)          # before the model loads
    if mem_cfg.enabled and mem_cfg.grant_context:
        cfg.max_seq_length += mem_cfg.token_budget

    mem_cfg, memory, extractor, lookup = setup_memory(
        merged, problem, cfg, mem_cfg=mem_cfg, backend=backend, model=model,
        tokenizer=tokenizer, gen_pool=gen_pool, exp_dir=exp_dir, seed=cfg.seed)

    # once per step, for all parents at once
    chosen = lookup.select_batch(parent_ctxs, step_idx, adapter_path)
    block, n_tok, kept = build_injection(chosen[g], tokenizer, mem_cfg.token_budget)
    messages = inject_block(messages, block, mem_cfg.inject_mode)

    # once per step, after every rollout is scored
    extractor.update(records, step, adapter_path=adapter_path)
    memory.save()
"""

from memory.bank import MemoryBank
from memory.bandit import (MemoryArm, allocate_memory_arms,
                           credit_memory_arms, expected_subsample_max)
from memory.curator import MemoryCurator
from memory.config import MemoryConfig
from memory.extractor import LessonExtractor, build_meta_description
from memory.hygiene import HygieneStats, violation
from memory.llm import make_memory_llm
from memory.lookup import MemoryLookup
from memory.prompts import (ExtractionResult, LookupResult, build_injection,
                            count_tokens, inject_block, parent_block,
                            parse_extraction, parse_lookup,
                            render_memory_block)
from memory.types import (FAILURE, GLOBAL, LOCAL, SUCCESS, Lesson,
                          RolloutRecord)

__all__ = [
    "MemoryBank", "MemoryConfig", "MemoryLookup", "MemoryCurator",
    "LessonExtractor",
    "ExtractionResult", "LookupResult", "HygieneStats", "violation",
    "build_meta_description", "make_memory_llm", "build_injection",
    "inject_block", "count_tokens", "parent_block", "parse_extraction",
    "parse_lookup", "render_memory_block", "Lesson", "RolloutRecord",
    "SUCCESS", "FAILURE", "LOCAL", "GLOBAL", "setup_memory",
    "MemoryArm", "allocate_memory_arms", "credit_memory_arms",
    "expected_subsample_max",
]


def setup_memory(merged: dict, problem, cfg, mem_cfg=None, backend=None,
                 model=None, tokenizer=None, gen_pool=None, exp_dir=None,
                 seed=None, resume_from=None, verbose: bool = True):
    """
    Build the module, or nothing at all.

    Returns (mem_cfg, memory, extractor, lookup, curator). When the master flag
    is off, the last four are None.

    Pass `mem_cfg` when the trainer already built it early for the
    max_seq_length top-up; it is not rebuilt in that case.
    """
    if mem_cfg is None:
        mem_cfg = MemoryConfig.from_dict(merged, verbose=verbose)
    if not mem_cfg.enabled:
        if verbose:
            print("[init] memory OFF")
        return mem_cfg, None, None, None, None

    bank = MemoryBank(mem_cfg)
    if resume_from:
        n = bank.load(resume_from)
        if verbose:
            print(f"[memory] resumed {n} lesson(s) from {resume_from}")
    if mem_cfg.persist and exp_dir is not None:
        from pathlib import Path
        bank.attach(Path(exp_dir) / "memory.json")

    llm = make_memory_llm(mem_cfg, backend=backend, model=model,
                          tokenizer=tokenizer, gen_pool=gen_pool, seed=seed,
                          max_seq_length=getattr(cfg, "max_seq_length", None),
                          verbose=verbose)
    meta = build_meta_description(problem, cfg)
    from memory.hygiene import resolve_profile
    profile = resolve_profile(mem_cfg, getattr(problem, "name", ""))
    extractor = LessonExtractor(
        mem_cfg, llm, meta, bank=bank,
        fail_score=float(getattr(problem, "fail_score", 0.0)),
        hygiene_profile=profile)
    lookup = MemoryLookup(mem_cfg, bank, llm, meta)
    curator = MemoryCurator(mem_cfg, bank, llm, meta, extractor=extractor)

    if verbose:
        print(f"[init] {mem_cfg.describe()}  hygiene={profile}")
        if len(bank) == 0:
            print("[memory] bank is empty; step 0 makes no lookup call and its "
                  "prompts are identical to a no-memory run")
    return mem_cfg, bank, extractor, lookup, curator
