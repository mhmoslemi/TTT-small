"""LLM judge backends for the Multi-Agent Elo re-ranker.

Backends:
  * OpenAICompatibleJudge - talks to any OpenAI-compatible /v1/chat/completions
    endpoint. This is the recommended way to use a LOCAL model (e.g. Qwen) as
    the judge: serve it with vLLM / Ollama / TGI in a SEPARATE process and point
    base_url at it. That keeps the tournament purely I/O-bound (the HTTP call
    releases the GIL), so it never blocks the training loop and never competes
    with the trainer for in-process GPU memory.
  * GeminiJudge - best-effort Google Gemini backend via the google-genai SDK.

Judge interface is tiny: complete(messages) -> str.
"""
from __future__ import annotations
import os
from typing import List, Optional


# class BaseJudge:
#     def complete(self, messages: List[dict]) -> str:
#         raise NotImplementedError


class BaseJudge:
    concurrency = 1  # set by subclasses for threaded HTTP batching

    def complete(self, messages: List[dict]) -> str:
        raise NotImplementedError

    def complete_batch(self, list_of_messages):
        """Default: run complete() concurrently across threads (good for HTTP
        backends; the GIL is released during the network call). Order preserved.
        The local judge OVERRIDES this with a real batched generate()."""
        from concurrent.futures import ThreadPoolExecutor
        n = max(1, int(getattr(self, "concurrency", 1)))
        if n == 1 or len(list_of_messages) <= 1:
            return [self.complete(m) for m in list_of_messages]
        out = [""] * len(list_of_messages)
        with ThreadPoolExecutor(max_workers=n) as pool:
            futs = {pool.submit(self.complete, m): i
                    for i, m in enumerate(list_of_messages)}
            for fut in futs:
                i = futs[fut]
                try:
                    out[i] = fut.result()
                except Exception:
                    out[i] = ""
        return out

    def close(self):
        pass




def _resolve_api_key(cfg) -> str:
    # Explicit key wins; otherwise read from the named env var. Local OpenAI-
    # compatible servers usually accept any non-empty string (commonly "EMPTY").
    if getattr(cfg, "api_key", ""):
        return cfg.api_key
    env_name = getattr(cfg, "api_key_env", "") or ""
    if env_name and os.environ.get(env_name):
        return os.environ[env_name]
    return "EMPTY"


class OpenAICompatibleJudge(BaseJudge):
    def __init__(self, cfg):
        from openai import OpenAI  # raises if the package is missing
        self.model = cfg.model
        self.temperature = float(cfg.temperature)
        self.max_tokens = int(cfg.max_tokens)
        self.timeout = float(cfg.request_timeout_s)
        base_url = cfg.base_url or os.environ.get("OPENAI_BASE_URL") or None
        # The client is safe to share across the tournament's worker threads.
        self.client = OpenAI(api_key=_resolve_api_key(cfg), base_url=base_url)

        self.concurrency = int(cfg.judge_concurrency)

    def complete(self, messages: List[dict]) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            timeout=self.timeout,
        )
        return resp.choices[0].message.content or ""


class GeminiJudge(BaseJudge):
    def __init__(self, cfg):
        from google import genai  # google-genai SDK
        self.model = cfg.model
        self.temperature = float(cfg.temperature)
        self.max_tokens = int(cfg.max_tokens)
        key = _resolve_api_key(cfg)
        if key == "EMPTY":
            key = os.environ.get("GOOGLE_API_KEY", "")
        self.client = genai.Client(api_key=key)

        self.concurrency = int(cfg.judge_concurrency)

    def complete(self, messages: List[dict]) -> str:
        # Flatten chat messages into a single prompt (system first).
        sys_txt = "\n".join(m["content"] for m in messages if m["role"] == "system")
        usr_txt = "\n".join(m["content"] for m in messages if m["role"] != "system")
        prompt = (sys_txt + "\n\n" + usr_txt).strip()
        resp = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config={"temperature": self.temperature,
                    "max_output_tokens": self.max_tokens},
        )
        return getattr(resp, "text", "") or ""


def make_judge(cfg) -> Optional[BaseJudge]:
    """Construct the configured judge, or return None (with a log line) if its
    backend/package is unavailable, so the caller can fall back to the rank
    prior without crashing."""
    backend = (getattr(cfg, "backend", "") or "openai").lower()
    try:
        if backend in ("local", "transformers", "hf", "qwen"):
            from reranker.judge_worker import LocalJudgeClient
            return LocalJudgeClient(cfg)         # separate process, own GPU
        if backend in ("openai", "openai_compatible", "vllm", "ollama"):
            return OpenAICompatibleJudge(cfg)
        if backend in ("gemini", "google"):
            return GeminiJudge(cfg)
        print(f"[reranker] unknown judge backend '{backend}'; disabling re-ranker")
        return None
    except Exception as e:
        print(f"[reranker] could not initialize judge backend '{backend}' "
              f"({e!r}); disabling re-ranker")
        return None