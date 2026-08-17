"""
Memory configuration (Sec. 2.2).

`enabled` is the master switch, read from the single key `memory` in the merged
config dict. When it is False, from_dict returns the disabled default without
consulting any other memory_* key.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Dict

PREFIX = "memory_"
_MISSING = object()


@dataclass
class MemoryConfig:
    enabled: bool = False

    # --- retrieval (Eq. 7) ---
    top_m: int = 5                       # m
    retrieval_scope: str = "both"        # both | success | failure
    min_similarity: float = 0.0
    importance_weight: float = 0.15      # how far importance can move the rank

    # --- token budget ---
    # Hard cap on the tokens the injected block may occupy. The trainer adds
    # this same number to max_seq_length at startup, so the memory block is
    # granted context ON TOP of the no-memory setting rather than eating into
    # the space the response would otherwise have had. max_new_tokens is never
    # touched by the memory module.
    token_budget: int = 1200
    grant_context: bool = True           # do the max_seq_length top-up

    # --- extraction (Sec. 2.2) ---
    lessons_per_call: int = 3            # L, the ceiling per call
    require_full_lessons: bool = False   # True = always demand exactly L
    max_examples_per_call: int = 8
    max_chars_per_example: int = 1500
    feedback_chars: int = 800
    catalog_max_lessons: int = 60        # existing lessons shown to the maker
    catalog_chars: int = 160             # per catalog line
    reinforce_delta: float = 0.5         # importance bump on a confirmation

    # --- memory maker generation ---
    max_new_tokens: int = 1024
    temperature: float = 0.7
    top_p: float = 0.95
    use_gen_pool: bool = True

    # --- bank ---
    max_lessons: int = 500
    dedup_threshold: float = 0.95
    persist: bool = True

    # --- embedding e(.) ---
    embed_backend: str = "auto"          # auto | hash | sentence_transformers
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embed_dim: int = 2048
    embed_device: str = "cpu"

    # --- prompt injection ---
    inject_mode: str = "append"          # append | system

    @classmethod
    def from_dict(cls, d: Dict[str, Any], verbose: bool = True) -> "MemoryConfig":
        d = dict(d or {})
        enabled = bool(d.get("memory", False))

        if not enabled:
            # Only report keys the user actually moved off the default. The
            # trainer's Config dataclass carries every memory_* field, so the
            # merged dict always contains all of them and an unfiltered list
            # would print on every single run.
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
        unknown = []
        for key, value in d.items():
            if not key.startswith(PREFIX) or value is None:
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

        if verbose and unknown:
            print(f"[memory] unknown memory_* key(s) ignored: "
                  f"{', '.join(sorted(unknown))}")

        cfg = cls(**kwargs)
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if self.top_m < 0:
            raise ValueError("memory_top_m must be >= 0")
        if self.lessons_per_call < 1:
            raise ValueError("memory_lessons_per_call must be >= 1")
        if self.retrieval_scope not in ("both", "success", "failure"):
            raise ValueError("memory_retrieval_scope must be both|success|failure")
        if self.inject_mode not in ("append", "system"):
            raise ValueError("memory_inject_mode must be append|system")
        if self.embed_backend not in ("auto", "hash", "sentence_transformers"):
            raise ValueError(
                "memory_embed_backend must be auto|hash|sentence_transformers")

    def describe(self) -> str:
        if not self.enabled:
            return "memory OFF"
        return (f"memory ON  m={self.top_m}  L<={self.lessons_per_call}  "
                f"cap={self.max_lessons}  budget={self.token_budget}tok  "
                f"embed={self.embed_backend}  inject={self.inject_mode}")


def _as_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    return str(v).strip().lower() in ("1", "true", "yes", "y", "on")
