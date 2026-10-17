"""OpenAI-compatible remote generation for the frozen strategist only."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from pathlib import Path
from typing import Any

from strategy_prompts_openai import STRATEGY_PLAN_FORMAT


OPENAI_STRATEGY_MODEL = "gpt-6-luna"
OPENAI_REASONING_EFFORTS = frozenset({"none", "low", "medium", "high", "xhigh", "max"})
OPENAI_MAX_OUTPUT_TOKENS = 128_000
OPENAI_KEY_FILE = Path(__file__).resolve().with_name(".openai_api_key")


def strategy_api_key(api_key_env):
    """Resolve a credential without putting it into saved configuration."""
    key = os.environ.get(api_key_env, "").strip()
    if not key and api_key_env == "OPENAI_API_KEY" and OPENAI_KEY_FILE.is_file():
        key = OPENAI_KEY_FILE.read_text().strip()
    if not key:
        suffix = " or the ignored .openai_api_key file" if api_key_env == "OPENAI_API_KEY" else ""
        raise ValueError(f"strategy API mode requires ${api_key_env}{suffix}")
    if any(char.isspace() for char in key):
        raise ValueError("strategy API key must be a single nonempty token")
    return key


def run_strategy_rounds(pool, *, num_parents, num_strategies, prompt_builder,
                        result_handler, max_retries, generation_options,
                        stop_event=None):
    """One independent request per parent; finish a depth before advancing.

    result_handler admits only valid plans into its parent's history and
    returns True when that slot needs a format retry. Successful slots are
    never requested twice, and pending retries cannot leak into the next round.
    """
    for strategy_index in range(num_strategies):
        pending = list(range(num_parents))
        for attempt in range(max_retries + 1):
            if stop_event is not None and stop_event.is_set():
                return
            prompts = [prompt_builder(parent, strategy_index, attempt)
                       for parent in pending]
            completed, retry_parents = set(), []
            for index, results in pool.iter_group_jobs(
                    prompts_by_group=prompts, group_size=1,
                    counts_by_group=[1] * len(prompts), adapter_path=None,
                    progress_desc=f"strategy {strategy_index + 1}/{num_strategies}"
                                  + (f" retry {attempt}" if attempt else ""),
                    **generation_options):
                if stop_event is not None and stop_event.is_set():
                    return
                if not isinstance(index, int) or not 0 <= index < len(pending):
                    raise RuntimeError("strategy API returned an invalid parent index")
                parent = pending[index]
                if parent in completed or len(results) != 1:
                    raise RuntimeError("strategy API must return one result per parent")
                if result_handler(parent, strategy_index, results[0][0], attempt):
                    retry_parents.append(parent)
                completed.add(parent)
            if completed != set(pending):
                raise RuntimeError("strategy API omitted a parent response")
            pending = sorted(retry_parents)
            if not pending:
                break
        if pending:
            raise RuntimeError("strategy retries exhausted without an accepted plan or fallback")


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
                 reasoning_effort: str, api_strategy: str = ""):
        api_key_env = str(api_key_env).strip()
        api_key = strategy_api_key(api_key_env)

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
        self._openai = bool(str(api_strategy).strip())
        if self._openai:
            if api_strategy != OPENAI_STRATEGY_MODEL or self.model_name != api_strategy:
                raise ValueError(f"api_strategy currently supports {OPENAI_STRATEGY_MODEL}")
            if self.base_url != "https://api.openai.com/v1":
                raise ValueError("api_strategy uses the official OpenAI API endpoint")
            if self.reasoning_effort not in OPENAI_REASONING_EFFORTS:
                raise ValueError("unsupported OpenAI strategy reasoning effort")
        self._deepseek = "deepseek" in self.base_url.lower()
        self._client = OpenAI(
            api_key=api_key,
            base_url=self.base_url,
            timeout=float(timeout_s),
            max_retries=int(max_retries),
        )
        if self._openai and not hasattr(self._client, "responses"):
            self._client.close()
            raise RuntimeError(
                "OpenAI strategy mode requires a newer SDK; "
                "install 'openai>=2.8.1' from requirements.txt")
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
                if not isinstance(content, str):
                    raise TypeError("strategy API accepts text messages only")
                messages.append({"role": role, "content": content})
            return messages
        return [{"role": "user", "content": str(prompt)}]

    def _generate_one(self, prompt, *, max_new_tokens, temperature, top_p,
                      top_k=None, min_p=None):
        if self._openai:
            return self._generate_openai(prompt, max_new_tokens=max_new_tokens)
        request = {
            "model": self.model_name,
            "messages": self._messages(prompt),
            "max_tokens": int(max_new_tokens),
            "temperature": float(temperature),
            "top_p": float(top_p),
        }
        extra_body = {}
        if top_k is not None:
            extra_body["top_k"] = int(top_k)
        if min_p is not None:
            extra_body["min_p"] = float(min_p)
        if self._deepseek:
            effort = self.reasoning_effort if self.thinking else "none"
            extra_body.update({
                # Keep provider-only fields in extra_body so this also works
                # with OpenAI SDK releases whose typed Chat Completions
                # signature predates reasoning_effort.
                "reasoning_effort": effort,
                "thinking": {
                    "type": "enabled" if self.thinking else "disabled",
                },
            })
        if extra_body:
            request["extra_body"] = extra_body

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

    def _generate_openai(self, prompt, *, max_new_tokens):
        # Reasoning GPT-6 requests must not forward local vLLM sampling fields
        # (temperature/top_p/top_k/min_p) or the DeepSeek thinking extension.
        response = self._client.responses.create(
            model=self.model_name,
            input=self._messages(prompt),
            reasoning={"effort": self.reasoning_effort if self.thinking else "none"},
            max_output_tokens=min(int(max_new_tokens), OPENAI_MAX_OUTPUT_TOKENS),
            tools=[],
            tool_choice="none",
            parallel_tool_calls=False,
            text={"format": STRATEGY_PLAN_FORMAT},
            store=False,
        )
        if getattr(response, "status", None) != "completed":
            raise RuntimeError("OpenAI strategy response was incomplete; no partial plan accepted")
        parts = []
        for item in response.output:
            if item.type == "reasoning":
                continue
            if item.type != "message":
                raise RuntimeError("OpenAI strategy returned an unexpected non-text output")
            if getattr(item, "phase", None) not in (None, "final_answer"):
                continue
            for content in item.content:
                if content.type == "refusal":
                    raise RuntimeError("OpenAI declined to provide a strategy plan")
                if content.type != "output_text":
                    raise RuntimeError("OpenAI strategy returned an unexpected content type")
                parts.append(content.text)
        final_text = "".join(parts)
        if not final_text.strip():
            raise RuntimeError("OpenAI strategy returned no final plan")
        usage = response.usage
        details = getattr(usage, "output_tokens_details", None)
        return final_text, {
            "prompt": int(getattr(usage, "input_tokens", 0) or 0),
            "completion": int(getattr(usage, "output_tokens", 0) or 0),
            "total": int(getattr(usage, "total_tokens", 0) or 0),
            "reasoning": int(getattr(details, "reasoning_tokens", 0) or 0),
        }

    def iter_group_jobs(self, prompts_by_group, group_size, adapter_path,
                        max_new_tokens, temperature, top_p, step_idx=0,
                        show_progress=True, counts_by_group=None,
                        return_logprobs=False, progress_desc="strategies",
                        top_k=None, min_p=None):
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
                top_k=top_k,
                min_p=min_p,
            ): index
            for index, prompt in enumerate(prompts)
        }
        usage = {"prompt": 0, "completion": 0, "total": 0, "reasoning": 0}
        try:
            for future in as_completed(futures):
                index = futures[future]
                try:
                    raw_response, request_usage = future.result()
                except Exception as exc:
                    for pending in futures:
                        pending.cancel()
                    # Provider error bodies may echo a credential or prompt.
                    status = getattr(exc, "status_code", None)
                    detail = f", HTTP {status}" if status is not None else ""
                    raise RuntimeError(
                        f"strategy API request {index} failed "
                        f"({type(exc).__name__}{detail})") from None
                for key in usage:
                    usage[key] += int(request_usage.get(key, 0))
                yield index, [(raw_response, [])]
        finally:
            for future in futures:
                future.cancel()

        print(
            f"[strategy-api] {progress_desc}: completed {len(prompts)} "
            f"response(s); tokens prompt={usage['prompt']:,} "
            f"completion={usage['completion']:,} "
            f"reasoning={usage['reasoning']:,} total={usage['total']:,}",
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
