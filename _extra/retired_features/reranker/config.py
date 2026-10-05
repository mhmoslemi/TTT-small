"""Configuration for the Multi-Agent Elo re-ranker.

Reads `reranker_*` keys out of the merged config dict (Config defaults < YAML 
CLI) produced by train_multy.load_config(). Everything has a default, so the
feature stays OFF unless `reranker_enabled: true` is present in the YAML.
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class RerankerConfig:
    enabled: bool = False

    # --- judge model / backend ---
    backend: str = "openai"          # "openai" (OpenAI-compatible) | "gemini"
    model: str = "Qwen/Qwen3-8B"     # for a local server, whatever name it serves
    base_url: str = ""               # e.g. "http://localhost:8000/v1" for vLLM
    api_key: str = ""                # explicit key (prefer api_key_env instead)
    api_key_env: str = "RERANKER_API_KEY"
    temperature: float = 0.7
    max_tokens: int = 2048
    request_timeout_s: float = 120.0

    # --- local judge (backend == "local") ---
    judge_gpu: int = 5               # GPU for the judge process; pick one NOT
                                     # saturated by training/generation
    judge_batch_size: int = 16       # matches per batched generate() call
    judge_load_in_4bit: bool = False # shrink the judge to co-reside on a used GPU
    top_p: float = 1.0
    max_seq_length: int = 8192       # judge context (judge prompts are short)


    # --- tournament ---
    top_k: int = 20                  # how many top buffer states to re-rank (hyperparam)
    debate: bool = True              # True = simulated multi-turn debate prompt; False = single-turn
    tournament_mode: str = "round_robin"   # "round_robin" | "random"
    num_random_matches: int = 60     # used only when tournament_mode == "random"
    both_orders: bool = False        # round_robin: also play each pair in the swapped order
    judge_concurrency: int = 4       # parallel in-flight judge calls (I/O bound)
    max_code_chars: int = 4000       # truncate the code shown to the judge

    # --- elo -> prior ---
    elo_init: float = 1200.0         # initial rating (matches the co-scientist Ranking agent)
    elo_k: float = 24.0              # Elo K-factor
    elo_softmax_temp: float = 100.0  # Elo points per factor-e of prior weight
    prior_weight: float = 1.0        # alpha: 1.0 = Elo fully replaces rank within the elite set,
                                     # 0.0 = leave rank prior unchanged

    # --- scheduling ---
    poll_interval_s: float = 5.0     # sleep between tournament cycles
    min_states_to_rank: int = 2

    # --- problem context (optional override) ---
    goal: str = ""                   # if empty, built from metric_name/direction

    @classmethod
    def from_dict(cls, d: dict) -> "RerankerConfig":
        d = d or {}

        def g(key, default):
            return d.get(f"reranker_{key}", default)

        return cls(
            enabled=bool(g("enabled", cls.enabled)),
            backend=str(g("backend", cls.backend)),
            model=str(g("model", cls.model)),
            base_url=str(g("base_url", cls.base_url)),
            api_key=str(g("api_key", cls.api_key)),
            api_key_env=str(g("api_key_env", cls.api_key_env)),
            temperature=float(g("temperature", cls.temperature)),
            max_tokens=int(g("max_tokens", cls.max_tokens)),
            request_timeout_s=float(g("request_timeout_s", cls.request_timeout_s)),
            top_k=int(g("top_k", cls.top_k)),
            debate=bool(g("debate", cls.debate)),
            tournament_mode=str(g("tournament_mode", cls.tournament_mode)),
            num_random_matches=int(g("num_random_matches", cls.num_random_matches)),
            both_orders=bool(g("both_orders", cls.both_orders)),
            judge_concurrency=int(g("judge_concurrency", cls.judge_concurrency)),
            max_code_chars=int(g("max_code_chars", cls.max_code_chars)),
            elo_init=float(g("elo_init", cls.elo_init)),
            elo_k=float(g("elo_k", cls.elo_k)),
            elo_softmax_temp=float(g("elo_softmax_temp", cls.elo_softmax_temp)),
            prior_weight=float(g("prior_weight", cls.prior_weight)),
            poll_interval_s=float(g("poll_interval_s", cls.poll_interval_s)),
            min_states_to_rank=int(g("min_states_to_rank", cls.min_states_to_rank)),
            goal=str(g("goal", cls.goal)),

            judge_gpu=int(g("judge_gpu", cls.judge_gpu)),
            judge_batch_size=int(g("judge_batch_size", cls.judge_batch_size)),
            judge_load_in_4bit=bool(g("judge_load_in_4bit", cls.judge_load_in_4bit)),
            top_p=float(g("top_p", cls.top_p)),
            max_seq_length=int(g("max_seq_length", cls.max_seq_length)),

        )