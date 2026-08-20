"""
Memory configuration (Sec. 2.2, as revised).

`enabled` is the master switch, read from the single key `memory`. When it is
False, from_dict returns the disabled default without consulting any other
memory_* key.

Two things are gone from the previous version, because the module no longer
uses embeddings at all: embed_backend / embed_model / embed_dim / embed_device,
and top_m / min_similarity / importance_weight / dedup_threshold. Retrieval is
now the model reading a catalog of the whole bank and naming what it wants, and
deduplication is lexical. Any of those keys left in a YAML will be reported as
unknown rather than silently ignored.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Dict

PREFIX = "memory_"
_MISSING = object()

# Keys from the embedding-based version. Named explicitly so an old YAML gets a
# pointed message instead of a generic "unknown key".
RETIRED = {
    "memory_embed_backend", "memory_embed_model", "memory_embed_dim",
    "memory_embed_device", "memory_top_m", "memory_min_similarity",
    "memory_importance_weight", "memory_dedup_threshold",
    "memory_retrieval_scope",
}


@dataclass
class MemoryConfig:
    enabled: bool = False

    # --- lookup (replaces Eq. 7 retrieval) ---
    # select : show the model a catalog of the bank, it names the ids it wants.
    #          One extra LLM call per step (all parents batched together).
    # all    : inject the entire bank into every prompt. No extra call, but the
    #          rollout prompt carries the whole bank.
    # none   : never inject. For measuring extraction in isolation.
    lookup_mode: str = "select"
    lookup_max_select: int = 5      # ceiling on what the model may ask for
    lookup_max_new_tokens: int = 256
    lookup_temperature: float = 0.3  # near-greedy; this is a selection, not prose
    lookup_fallback: str = "none"    # none | recent | importance

    # --- catalog shown to the selector and to the extractor ---
    catalog_max_lessons: int = 0     # 0 = the whole bank, which is the point
    catalog_chars: int = 200         # per-entry summary budget

    # --- injection ---
    inject_mode: str = "append"      # append | system
    token_budget: int = 1200         # cap on the injected block
    grant_context: bool = True       # add that budget to max_seq_length

    # --- extraction (Sec. 2.2) ---
    # Which side of the batch produces lessons.
    #   both     : one call over S_t and one over F_t (the paper's 2L per step)
    #   failure  : only F_t. One call per step instead of two, and the bank holds
    #              only failure modes and their preventative measures.
    #   success  : only S_t.
    # This is about EXTRACTION, not injection: whatever ends up in the bank is
    # what the lookup can choose from.
    # How the batch is shown to the maker.
    #   contrast : ONE call over successes and failures together, asked why some
    #              worked and others did not (ReasoningBank self-contrast). A
    #              contrast exists even when the successes are indistinguishable
    #              from each other, which is the plateau case.
    #   split    : the paper's separate prompt+ / prompt- calls.
    extract_mode: str = "contrast"   # contrast | split
    extract_from: str = "both"       # both | failure | success  (split mode only)
    lessons_per_call: int = 3        # L, a ceiling
    require_full_lessons: bool = False
    max_examples_per_call: int = 8
    max_chars_per_example: int = 1500
    feedback_chars: int = 800
    reinforce_delta: float = 0.15
    max_new_tokens: int = 1024
    temperature: float = 0.7
    top_p: float = 0.95
    use_gen_pool: bool = True

    # --- extraction hygiene: the Table 3 problem ---
    # A lesson may describe an OPERATION. It may not hand over a CONSTRUCTION.
    forbid_constructions: bool = True
    hygiene_profile: str = "auto"    # auto | geometry | kernel | generic
    max_code_lines: int = 4          # longer code blocks are rejected outright
    global_scope_allows_code: bool = False

    # --- curation (Dynamic Cheatsheet) ---
    # Every N steps, hand the model the whole bank and take back the bank it
    # wants to keep. Anything not carried forward is dropped, which is what
    # forces a retention decision instead of an accumulation.
    curate_every: int = 0            # 0 = never
    curate_min_bank: int = 20        # do not curate a bank smaller than this
    curate_max_items: int = 60
    curate_max_new_tokens: int = 4096
    curate_min_keep_frac: float = 0.25   # reject a rewrite that drops more than this

    # --- bank ---
    max_lessons: int = 500
    dedup_jaccard: float = 0.6       # token-set overlap that counts as the same lesson
    persist: bool = True

    @classmethod
    def from_dict(cls, d: Dict[str, Any], verbose: bool = True) -> "MemoryConfig":
        d = dict(d or {})
        enabled = bool(d.get("memory", False))

        if not enabled:
            defaults = cls()
            ignored = sorted(
                k for k, v in d.items()
                if k.startswith(PREFIX) and v is not None
                and getattr(defaults, k[len(PREFIX):], _MISSING) != v
            )
            if verbose and ignored:
                print(f"[memory] disabled (--memory not set); ignoring "
                      f"{len(ignored)} memory_* key(s): {', '.join(ignored)}")
            return cls(enabled=False)

        kwargs: Dict[str, Any] = {"enabled": True}
        known = {f.name: f for f in fields(cls)}
        unknown, retired = [], []
        for key, value in d.items():
            if not key.startswith(PREFIX) or value is None:
                continue
            if key in RETIRED:
                retired.append(key)
                continue
            name = key[len(PREFIX):]
            if name not in known or name == "enabled":
                unknown.append(key)
                continue
            # `from __future__ import annotations` makes field.type a string,
            # so both forms are checked.
            target = known[name].type
            try:
                if target is bool or target == "bool":
                    value = _as_bool(value)
                elif target is int or target == "int":
                    value = int(value)
                elif target is float or target == "float":
                    value = float(value)
                else:
                    value = str(value)
            except (TypeError, ValueError):
                raise ValueError(f"[memory] bad value for {key}: {value!r}")
            kwargs[name] = value

        if verbose and retired:
            print(f"[memory] these keys no longer exist (the module does not "
                  f"embed anything now): {', '.join(sorted(retired))}")
        if verbose and unknown:
            print(f"[memory] unknown memory_* key(s) ignored: "
                  f"{', '.join(sorted(unknown))}")

        cfg = cls(**kwargs)
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if self.lessons_per_call < 1:
            raise ValueError("memory_lessons_per_call must be >= 1")
        if self.hygiene_profile not in ("auto", "geometry", "kernel", "generic"):
            raise ValueError(
                "memory_hygiene_profile must be auto|geometry|kernel|generic")
        if self.extract_mode not in ("contrast", "split"):
            raise ValueError("memory_extract_mode must be contrast|split")
        if self.extract_from not in ("both", "failure", "success"):
            raise ValueError("memory_extract_from must be both|failure|success")
        if self.lookup_mode not in ("select", "all", "none"):
            raise ValueError("memory_lookup_mode must be select|all|none")
        if self.lookup_fallback not in ("none", "recent", "importance"):
            raise ValueError("memory_lookup_fallback must be none|recent|importance")
        if self.inject_mode not in ("append", "system"):
            raise ValueError("memory_inject_mode must be append|system")
        if self.lookup_max_select < 0:
            raise ValueError("memory_lookup_max_select must be >= 0")

    def describe(self) -> str:
        if not self.enabled:
            return "memory OFF"
        src = ("" if self.extract_from == "both"
               else f"-{self.extract_from}-only")
        cur = "" if self.curate_every <= 0 else f"  curate/{self.curate_every}"
        return (f"memory ON  extract={self.extract_mode}{src}{cur}  "
                f"lookup={self.lookup_mode}"
                f"(<={self.lookup_max_select})  L<={self.lessons_per_call}  "
                f"cap={self.max_lessons}  budget={self.token_budget}tok  "
                f"constructions={'forbidden' if self.forbid_constructions else 'allowed'}  "
                f"inject={self.inject_mode}")


def _as_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    return str(v).strip().lower() in ("1", "true", "yes", "y", "on")
