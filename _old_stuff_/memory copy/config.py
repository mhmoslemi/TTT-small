"""
Memory configuration (Sec. 2.2).

`enabled` is the master switch. It is read from the single key `memory` in the
merged config dict, and when it is False every other memory_* key is ignored:
from_dict returns the disabled default object without consulting them, and
prints once if any were set, so a run that silently ignored half its flags is
visible in the log rather than only in the results.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Dict

# Every tunable, as it appears in the merged config dict / CLI (--memory-top-m
# maps to memory_top_m). The master switch itself is the bare key "memory".
PREFIX = "memory_"


@dataclass
class MemoryConfig:
    enabled: bool = False

    # --- retrieval (Eq. 7) ---
    top_m: int = 5                       # m
    retrieval_scope: str = "both"        # both | success | failure
    min_similarity: float = 0.0          # drop retrievals below this cosine

    # --- extraction (Sec. 2.2) ---
    lessons_per_call: int = 3            # L, so 2L new lessons per step
    max_examples_per_call: int = 8       # rollouts shown to the memory maker
    max_chars_per_example: int = 1500
    feedback_chars: int = 800

    # --- memory maker generation ---
    max_new_tokens: int = 1024
    temperature: float = 0.7
    top_p: float = 0.95
    use_gen_pool: bool = True            # run the 2 calls on the worker pool

    # --- bank ---
    max_lessons: int = 500               # cap; oldest dropped first
    dedup_threshold: float = 0.95        # cosine above this = same lesson
    persist: bool = True                 # write memory.json into the run dir

    # --- embedding e(.) ---
    embed_backend: str = "auto"          # auto | hash | sentence_transformers
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embed_dim: int = 2048                # hash backend only
    embed_device: str = "cpu"

    # --- prompt injection ---
    inject_mode: str = "append"          # append | system

    @classmethod
    def from_dict(cls, d: Dict[str, Any], verbose: bool = True) -> "MemoryConfig":
        d = dict(d or {})
        enabled = bool(d.get("memory", False))

        if not enabled:
            ignored = sorted(k for k in d if k.startswith(PREFIX)
                             and d[k] is not None)
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
        return (f"memory ON  m={self.top_m}  L={self.lessons_per_call}  "
                f"cap={self.max_lessons}  embed={self.embed_backend}  "
                f"inject={self.inject_mode}")


def _as_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    return str(v).strip().lower() in ("1", "true", "yes", "y", "on")
