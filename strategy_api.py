"""OpenAI-compatible remote generation for the frozen strategist only."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from typing import Any


def _message_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        parts = []
        for item in value:
            if isinstance(item, dict):
                text = item.get("text", "")
            else:
                text = getattr(item, "text", "")
            if text:
                parts.append(str(text))
        return "".join(parts)
    return str(value)


class StrategyAPIGenerationPool:
    """Match the generation-pool surface without loading local model weights."""

    sequential = False
    active = True
    num_workers = 1

    def __init__(self, *, model_name: str, base_url: str,
                 api_key_env: str, concurrency: int, timeout_s: float,
                 max_retries: int, thinking: bool,
                 reasoning_effort: str):
        api_key_env = str(api_key_env).strip()
        api_key = os.environ.get(api_key_env, "")
        if not api_key:
            raise RuntimeError(
                f"strategy API mode requires ${api_key_env} to be set")

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "strategy API mode requires the 'openai' Python package") from exc

        self.model_name = str(model_name).strip()
        self.base_url = str(base_url).strip().rstrip("/")
        self.concurrency = max(1, int(concurrency))
        self.thinking = bool(thinking)
        self.reasoning_effort = str(reasoning_effort).strip().lower()
        self._deepseek = "deepseek" in self.base_url.lower()
        self._client = OpenAI(
            api_key=api_key,
            base_url=self.base_url,
            timeout=float(timeout_s),
            max_retries=int(max_retries),
        )
        self._executor = ThreadPoolExecutor(
            max_workers=self.concurrency,
            thread_name_prefix="strategy-api",
        )
        self._closed = False

    @staticmethod
    def _messages(prompt):
        if isinstance(prompt, (list, tuple)):
            messages = []
            for message in prompt:
                if not isinstance(message, dict):
                    raise TypeError(
                        "strategy API prompts must contain message mappings")
                role = str(message.get("role", "user"))
                content = message.get("content", "")
                messages.append({"role": role, "content": content})
            return messages
        return [{"role": "user", "content": str(prompt)}]

    def _generate_one(self, prompt, *, max_new_tokens, temperature, top_p):
        request = {
            "model": self.model_name,
            "messages": self._messages(prompt),
            "max_tokens": int(max_new_tokens),
            "temperature": float(temperature),
            "top_p": float(top_p),
        }
        if self._deepseek:
            effort = self.reasoning_effort if self.thinking else "none"
            request["extra_body"] = {
                # Keep provider-only fields in extra_body so this also works
                # with OpenAI SDK releases whose typed Chat Completions
                # signature predates reasoning_effort.
                "reasoning_effort": effort,
                "thinking": {
                    "type": "enabled" if self.thinking else "disabled",
                },
            }

        response = self._client.chat.completions.create(**request)
        if not response.choices:
            raise RuntimeError("strategy API returned no completion choices")
        choice = response.choices[0]
        message = choice.message
        final_text = _message_text(getattr(message, "content", None))
        reasoning = _message_text(
            getattr(message, "reasoning_content", None))

        # Keep the complete provider response for the raw debug artifact while
        # giving the existing extractor an unambiguous final-channel boundary.
        raw_response = final_text
        if reasoning:
            raw_response = (
                "analysis\n" + reasoning.rstrip()
                + "\nassistantfinal\n" + final_text.lstrip()
            )

        usage = getattr(response, "usage", None)
        usage_counts = {
            "prompt": int(getattr(usage, "prompt_tokens", 0) or 0),
            "completion": int(getattr(usage, "completion_tokens", 0) or 0),
            "total": int(getattr(usage, "total_tokens", 0) or 0),
        }
        return raw_response, usage_counts

    def iter_group_jobs(self, prompts_by_group, group_size, adapter_path,
                        max_new_tokens, temperature, top_p, step_idx=0,
                        show_progress=True, counts_by_group=None,
                        return_logprobs=False, progress_desc="strategies"):
        del step_idx, show_progress
        if self._closed:
            raise RuntimeError("strategy API pool is already closed")
        if adapter_path is not None:
            raise ValueError("strategy API generation cannot use a LoRA adapter")
        if return_logprobs:
            raise ValueError("strategy API generation does not return logprobs")

        prompts = list(prompts_by_group)
        counts = (list(counts_by_group) if counts_by_group is not None
                  else [int(group_size)] * len(prompts))
        if len(counts) != len(prompts) or any(int(count) != 1 for count in counts):
            raise ValueError(
                "strategy API generation requires exactly one response per prompt")

        print(
            f"[strategy-api] {progress_desc}: requesting {len(prompts)} "
            f"response(s) from {self.model_name} with concurrency="
            f"{min(self.concurrency, max(1, len(prompts)))}",
            flush=True,
        )
        futures = {
            self._executor.submit(
                self._generate_one,
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            ): index
            for index, prompt in enumerate(prompts)
        }
        usage = {"prompt": 0, "completion": 0, "total": 0}
        try:
            for future in as_completed(futures):
                index = futures[future]
                try:
                    raw_response, request_usage = future.result()
                except Exception as exc:
                    for pending in futures:
                        pending.cancel()
                    raise RuntimeError(
                        f"strategy API request {index} failed: {exc}") from exc
                for key in usage:
                    usage[key] += int(request_usage[key])
                yield index, [(raw_response, [])]
        finally:
            for future in futures:
                future.cancel()

        print(
            f"[strategy-api] {progress_desc}: completed {len(prompts)} "
            f"response(s); tokens prompt={usage['prompt']:,} "
            f"completion={usage['completion']:,} total={usage['total']:,}",
            flush=True,
        )

    def release(self):
        # There are no local model weights or GPU resources to release.
        return None

    def shutdown(self):
        if self._closed:
            return
        self._closed = True
        self._executor.shutdown(wait=True, cancel_futures=True)
        close = getattr(self._client, "close", None)
        if callable(close):
            close()
