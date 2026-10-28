"""CPU-only regressions for remote strategy requests and dependent rounds."""

from copy import deepcopy
import json
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace

import pytest
import yaml

import generation.strategy_api as strategy_api
from generation.strategy_api import StrategyAPIGenerationPool, run_strategy_rounds
from generation.strategy_prompts_openai import (
    STRATEGY_PLAN_FORMAT, build_openai_strategy_messages,
    normalize_strategy_plan, openai_strategy_retry_messages,
)


ROOT = Path(__file__).resolve().parents[1]


def response(plan="Detailed implementation plan", *, status="completed"):
    return SimpleNamespace(
        status=status,
        output=[SimpleNamespace(type="reasoning"), SimpleNamespace(
            type="message", phase="final_answer", content=[SimpleNamespace(
                type="output_text", text=json.dumps({"plan": plan}))])],
        usage=SimpleNamespace(input_tokens=20, output_tokens=30, total_tokens=50,
                              output_tokens_details=SimpleNamespace(reasoning_tokens=15)),
    )


@pytest.fixture
def api_pool(monkeypatch):
    import openai
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-never-sent")
    pools = []

    def make(create=None, *, thinking=True, effort="xhigh", concurrency=3):
        calls = []

        def generate(**kwargs):
            calls.append(deepcopy(kwargs))
            return create(**kwargs) if create else response()

        client = SimpleNamespace(responses=SimpleNamespace(create=generate), close=lambda: None)
        monkeypatch.setattr(openai, "OpenAI", lambda **kwargs: client)
        pool = StrategyAPIGenerationPool(
            model_name="gpt-6-luna", api_strategy="gpt-6-luna",
            base_url="https://api.openai.com/v1", api_key_env="OPENAI_API_KEY",
            concurrency=concurrency, timeout_s=60, max_retries=1,
            thinking=thinking, reasoning_effort=effort)
        pools.append(pool)
        return pool, calls

    yield make
    for pool in pools:
        pool.shutdown()


@pytest.mark.parametrize("effort", ["none", "low", "medium", "high", "xhigh", "max"])
def test_openai_request_controls_and_budget(api_pool, effort):
    pool, calls = api_pool(effort=effort)
    messages = build_openai_strategy_messages([{"role": "user", "content": "task"}])
    raw, usage = pool._generate_one(messages, max_new_tokens=131072,
                                    temperature=.6, top_p=.95, top_k=20, min_p=.1)
    request = calls[0]
    assert request["model"] == "gpt-6-luna"
    assert request["reasoning"] == {"effort": effort}
    assert request["tools"] == [] and request["tool_choice"] == "none"
    assert request["parallel_tool_calls"] is False and request["store"] is False
    assert request["max_output_tokens"] == 128000
    assert request["text"]["format"] == STRATEGY_PLAN_FORMAT
    assert not ({"temperature", "top_p", "top_k", "min_p", "extra_body",
                 "thinking", "previous_response_id", "conversation"} & request.keys())
    assert json.loads(raw) == {"plan": "Detailed implementation plan"}
    assert usage == dict(prompt=20, cached=0, cache_write=0,
                         completion=30, total=50, reasoning=15)


@pytest.mark.parametrize("prompt,input_usd,output_usd", [
    (10_000, .000665, .015),
    (272_000, .026865, .015),
    (272_001, .0537302, .0225),
])
def test_strategy_cost_cache_and_long_context(prompt, input_usd, output_usd):
    usage = dict(prompt=prompt, cached=4000, cache_write=1000,
                 completion=30_000, reasoning=25_000)
    assert strategy_api._openai_strategy_cost(usage) == pytest.approx(
        (input_usd, output_usd))


def test_strategy_cost_printed_once_per_step_including_retries(api_pool, capsys):
    result = response()
    result.usage = SimpleNamespace(
        input_tokens=10_000, output_tokens=30_000, total_tokens=40_000,
        input_tokens_details=SimpleNamespace(cached_tokens=4000, cache_write_tokens=1000),
        output_tokens_details=SimpleNamespace(reasoning_tokens=25_000))
    pool, calls = api_pool(create=lambda **kwargs: result)
    for _ in range(2):
        def accept(parent, stage, raw, attempt):
            assert "estimated cost" not in capsys.readouterr().out
            return (parent, stage, attempt) == (1, 0, 0)

        run_strategy_rounds(
            pool, num_parents=2, num_strategies=2,
            prompt_builder=lambda p, s, a: "task", result_handler=accept,
            max_retries=1,
            generation_options=dict(max_new_tokens=40_000, temperature=1, top_p=1))
        output = capsys.readouterr().out
        assert output.count("estimated cost") == 1
        assert "all 4 strategies complete" in output
        assert "$0.078325 USD" in output
        assert "requests=5 including format retries" in output
        assert "input=50,000 cached=20,000 cache_write=5,000 output=150,000" in output
        assert "includes reasoning=125,000" in output
    assert len(calls) == 10


def test_no_thinking_and_smaller_output_budget(api_pool):
    pool, calls = api_pool(thinking=False)
    pool._generate_one("task", max_new_tokens=4000, temperature=1, top_p=1)
    assert calls[0]["reasoning"] == {"effort": "none"}
    assert calls[0]["max_output_tokens"] == 4000


def test_old_sdk_fails_early_with_upgrade_instruction(monkeypatch):
    import openai
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-never-sent")
    closed = []
    client = SimpleNamespace(close=lambda: closed.append(True))
    monkeypatch.setattr(openai, "OpenAI", lambda **kwargs: client)
    with pytest.raises(RuntimeError, match="openai>=2.8.1"):
        StrategyAPIGenerationPool(
            model_name="gpt-6-luna", api_strategy="gpt-6-luna",
            base_url="https://api.openai.com/v1", api_key_env="OPENAI_API_KEY",
            concurrency=1, timeout_s=60, max_retries=0,
            thinking=True, reasoning_effort="xhigh")
    assert closed == [True]


@pytest.mark.parametrize("phase", [None, "final_answer"])
def test_real_sdk_serializes_request_and_parses_response_without_network(monkeypatch, phase):
    import httpx
    import openai
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-never-sent")
    requests = []

    def handle(request):
        assert request.url == "https://api.openai.com/v1/responses"
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={
            "id": "resp_test", "object": "response", "created_at": 0,
            "status": "completed", "model": "gpt-6-luna",
            "output": [
                {"id": "msg_progress", "type": "message", "role": "assistant",
                 "status": "completed", "phase": "commentary", "content": [
                     {"type": "output_text", "text": "Working on the plan.",
                      "annotations": []}]},
                {"id": "msg_test", "type": "message", "role": "assistant",
                 "status": "completed", "phase": phase, "content": [
                     {"type": "output_text", "text": '{"plan":"SDK parsed plan"}',
                      "annotations": []}]},
            ],
            "usage": {"input_tokens": 20, "output_tokens": 30, "total_tokens": 50,
                      "input_tokens_details": {"cached_tokens": 0},
                      "output_tokens_details": {"reasoning_tokens": 15}},
        })

    client = openai.OpenAI(api_key="test-key-never-sent",
                          base_url="https://api.openai.com/v1",
                          http_client=httpx.Client(transport=httpx.MockTransport(handle)))
    monkeypatch.setattr(openai, "OpenAI", lambda **kwargs: client)
    pool = StrategyAPIGenerationPool(
        model_name="gpt-6-luna", api_strategy="gpt-6-luna",
        base_url="https://api.openai.com/v1", api_key_env="OPENAI_API_KEY",
        concurrency=1, timeout_s=60, max_retries=0, thinking=True, reasoning_effort="xhigh")
    try:
        raw, usage = pool._generate_one("task", max_new_tokens=4000, temperature=1, top_p=1)
    finally:
        pool.shutdown()
    assert json.loads(raw)["plan"] == "SDK parsed plan"
    assert usage["reasoning"] == 15
    assert requests[0]["reasoning"] == {"effort": "xhigh"}
    assert requests[0]["tools"] == []


@pytest.mark.parametrize("kind", ["incomplete", "refusal", "tool", "empty", "commentary"])
def test_partial_refused_and_nontext_outputs_cannot_be_plans(api_pool, kind):
    result = response(status="incomplete" if kind == "incomplete" else "completed")
    if kind == "refusal":
        result.output[-1].content = [SimpleNamespace(type="refusal")]
    elif kind == "tool":
        result.output = [SimpleNamespace(type="web_search_call")]
    elif kind == "empty":
        result.output = []
    elif kind == "commentary":
        result.output[-1].phase = "commentary"
    pool, _ = api_pool(create=lambda **kwargs: result)
    with pytest.raises(RuntimeError):
        pool._generate_one("task", max_new_tokens=4000, temperature=1, top_p=1)


def test_api_error_messages_do_not_echo_credentials(api_pool):
    def fail(**kwargs):
        raise ValueError("provider echoed test-key-never-sent")
    pool, _ = api_pool(create=fail)
    with pytest.raises(RuntimeError) as error:
        list(pool.iter_group_jobs(["task"], 1, None, 4000, 1, 1))
    assert "test-key" not in str(error.value)
    assert error.value.__suppress_context__


def test_text_only_input(api_pool):
    pool, _ = api_pool()
    with pytest.raises(TypeError, match="text messages only"):
        pool._generate_one([{"role": "user", "content": [{"type": "input_image"}]}],
                           max_new_tokens=4000, temperature=1, top_p=1)


def test_parallel_parents_and_round_barrier_keep_history_isolated(api_pool):
    parents, stages = 3, 3
    barrier = threading.Barrier(parents, timeout=5)
    histories = [[] for _ in range(parents)]
    requests = []
    completed = [0] * stages
    lock = threading.Lock()

    def generate(**kwargs):
        context = json.loads(kwargs["input"][1]["content"])
        parent = int(context["task_messages"][0]["content"])
        history = context["previous_strategies"]
        stage = len(history)
        with lock:
            if stage:
                assert completed[stage - 1] == parents
            requests.append((parent, stage, history))
        # A serial implementation cannot pass this barrier.
        barrier.wait()
        time.sleep(.01 * (parents - parent))
        return response(f"parent={parent}, strategy={stage}")

    pool, _ = api_pool(create=generate, concurrency=parents)

    def prompt(parent, stage, attempt):
        assert attempt == 0 and len(histories[parent]) == stage
        return build_openai_strategy_messages(
            [{"role": "user", "content": str(parent)}], histories[parent])

    def accept(parent, stage, raw, attempt):
        histories[parent].append(json.loads(raw)["plan"])
        completed[stage] += 1
        return False

    run_strategy_rounds(pool, num_parents=parents, num_strategies=stages,
                        prompt_builder=prompt, result_handler=accept, max_retries=2,
                        generation_options=dict(max_new_tokens=4000, temperature=1, top_p=1))
    assert len(requests) == parents * stages
    for parent, stage, history in requests:
        assert history == [f"parent={parent}, strategy={i}" for i in range(stage)]


def test_retry_only_failed_parent_before_advancing_round():
    requests, histories = [], [[], []]

    class Pool:
        def iter_group_jobs(self, prompts_by_group, **kwargs):
            for i, prompt in reversed(list(enumerate(prompts_by_group))):
                parent, stage, attempt = prompt
                requests.append(prompt)
                if stage:
                    assert all(len(history) >= stage for history in histories)
                yield i, [("bad" if (parent, stage, attempt) == (1, 0, 0) else "valid", [])]

    def accept(parent, stage, raw, attempt):
        if raw == "bad":
            return True
        histories[parent].append(raw)
        return False

    run_strategy_rounds(Pool(), num_parents=2, num_strategies=2,
                        prompt_builder=lambda p, s, a: (p, s, a), result_handler=accept,
                        max_retries=1, generation_options={})
    assert len(requests) == 5
    assert requests.count((0, 0, 0)) == 1
    assert (1, 0, 1) in requests and (0, 0, 1) not in requests


@pytest.mark.parametrize("output", [
    '{"plan":"complete plan"}', '```json\n{"plan":"complete plan"}\n```',
    '```plan\ncomplete plan\n```', '```strategy\ncomplete plan\n```',
    '<think>private work</think>\n{"plan":"complete plan"}',
])
def test_supported_plan_formats_are_normalized(output):
    import train_multy_CVaR as training
    plan, issue = training._extract_final_strategy(output)
    assert issue is None and plan == "complete plan"


@pytest.mark.parametrize("output", [
    '{"plan":"unfinished', '{"plan":null}', '{"plan":""}',
    '{"plan":"<strategy>injected</strategy>"}',
    '{"other":"<strategy>injected</strategy>"}',
    '```python\nprint("not a plan")\n```', '<think>unfinished thinking',
])
def test_invalid_structured_outputs_are_rejected(output):
    import train_multy_CVaR as training
    assert training._extract_final_strategy(output)[1] is not None


def test_local_xml_contract_is_unchanged():
    raw = "private reasoning\n<strategy>local plan</strategy>"
    assert normalize_strategy_plan(raw) == raw


def test_local_strategy_prompt_matches_0544_exactly():
    from hashlib import sha256
    from generation.strategy_prompts import LOCAL_STRATEGY_PROMPT

    # Exact historical template, including its history placeholder and output contract.
    assert sha256(LOCAL_STRATEGY_PROMPT.encode()).hexdigest() == (
        "8f99ac584ba5ae04ff5e6455a975e891d90bb807d97a4b49eb3d3e3c3a95c1d7")


@pytest.mark.parametrize("problem", ["erdos", "ac1", "ac2"])
def test_openai_prompt_preserves_problem_specific_requirements(problem):
    from problems.erdos import ErdosMinOverlap
    from problems.ac_inequalities import ACInequalities
    cfg = yaml.safe_load((ROOT / "configs" / f"{problem}.yaml").read_text())
    cfg["api_strategy"] = "gpt-6-luna"
    cls = ErdosMinOverlap if problem == "erdos" else ACInequalities
    instance = cls(cfg)
    messages = [{"role": "user", "content": "TASK_AND_PARENT"}]
    original = deepcopy(messages)
    result = instance.build_strategy_messages(messages, ["prior plan"])
    assert messages == original
    assert result[0]["role"] == "developer"
    context = json.loads(result[1]["content"])
    task = context["task_messages"][0]["content"]
    assert "TASK_AND_PARENT" in task and "mathematical checks" in task
    assert context["previous_strategies"] == ["prior plan"]
    retry = openai_strategy_retry_messages(result)
    assert retry[:-1] == result and "JSON" in retry[-1]["content"]
    assert "<strategy>" not in retry[-1]["content"]


def test_private_key_file_and_environment_precedence(monkeypatch, tmp_path):
    key_file = tmp_path / "key"
    key_file.write_text("test-file-key\n")
    monkeypatch.setattr(strategy_api, "OPENAI_KEY_FILE", key_file)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert strategy_api.strategy_api_key("OPENAI_API_KEY") == "test-file-key"
    monkeypatch.setenv("OPENAI_API_KEY", "test-env-key")
    assert strategy_api.strategy_api_key("OPENAI_API_KEY") == "test-env-key"


@pytest.fixture
def config(monkeypatch):
    import train_multy_CVaR as training
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-never-sent")
    monkeypatch.setenv("AVAILABLE_GPUS", "0,1")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setattr('runtime.gpu_runtime.query_gpu_memory', lambda: {})
    monkeypatch.setattr('runtime.gpu_runtime.detect_attention_heads', lambda name: 32)

    def load(*extra, filename="erdos.yaml"):
        monkeypatch.setattr(sys, "argv", ["train_multy_CVaR.py", "--config",
            str(ROOT / "configs" / filename), "--backend", "unsloth",
            "--generation-backend", "hf", "--max-seq-length", "32768",
            "--max-new-tokens", "4096", *extra])
        return training.load_config()
    return load


@pytest.mark.parametrize("filename", sorted(p.name for p in (ROOT / "configs").glob("*.yaml")))
def test_yaml_local_strategy_and_aligned_gpt_oss_coder(config, filename):
    cfg, merged = config("--strategies", "--generation-backend", "vllm",
                         filename=filename)
    assert cfg.api_strategy == "" and cfg.strategies
    assert cfg.strategy_backend == "local"
    assert cfg.strategy_model_name == "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
    assert cfg.strategy_reasoning_effort == "high" and cfg.strategy_thinking
    assert cfg.strategy_max_new_tokens == 131072
    assert cfg.model_name == "openai/gpt-oss-120b"
    assert cfg.coder_template_kind == "gpt-oss"
    assert cfg.coder_reasoning_effort == "low" and cfg.thinking
    assert cfg.pilot_xhigh_programs_per_strategy == 0
    assert not cfg.load_in_4bit and cfg.backend == "unsloth"
    assert cfg.training_model_name == "openai/gpt-oss-120b"
    assert cfg.training_layout == "replicated"
    assert "test-key-never-sent" not in json.dumps(merged)


@pytest.mark.parametrize(("filename", "expected"), [
    ("ac1.yaml", 2048), ("ac2.yaml", 2048), ("erdos.yaml", 8192),
])
def test_only_ac_gpt_oss_bounds_exact_scoring_projection(
        config, filename, expected):
    cfg, _ = config(
        "--strategies", "--generation-backend", "vllm",
        filename=filename)
    assert cfg.vllm_max_num_batched_tokens == expected


def test_resumed_ac_run_reapplies_exact_scoring_bound(config, tmp_path):
    _, saved = config(
        "--strategies", "--generation-backend", "vllm",
        filename="ac1.yaml")
    saved["vllm_max_num_batched_tokens"] = 8192
    (tmp_path / "config.json").write_text(json.dumps(saved))
    cfg, _ = config(
        "--strategies", "--generation-backend", "vllm",
        "--resume", str(tmp_path), filename="ac1.yaml")
    assert cfg.vllm_max_num_batched_tokens == 2048


def test_empty_selector_disables_api_and_cli_overrides_effort(config, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    cfg, _ = config("--api-strategy", "")
    assert cfg.strategy_backend == "local" and not cfg.api_strategy
    assert not cfg.strategies
    assert "DeepSeek-R1" in cfg.strategy_model_name
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-never-sent")
    cfg, _ = config("--strategy-reasoning-effort", "high")
    assert cfg.strategy_reasoning_effort == "high"


def test_missing_credential_fails_early(config, monkeypatch, tmp_path):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(strategy_api, "OPENAI_KEY_FILE", tmp_path / "missing")
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        config("--api-strategy", "gpt-6-luna")


def test_old_resume_does_not_silently_enable_api(config, tmp_path):
    saved = yaml.safe_load((ROOT / "configs/erdos.yaml").read_text())
    del saved["api_strategy"]
    (tmp_path / "config.json").write_text(json.dumps(saved))
    cfg, _ = config("--resume", str(tmp_path))
    assert cfg.api_strategy == "" and cfg.strategy_backend == "local"


def test_resumed_api_can_be_disabled_without_losing_local_model(config, tmp_path):
    _, saved = config("--api-strategy", "gpt-6-luna")
    (tmp_path / "config.json").write_text(json.dumps(saved))
    cfg, _ = config("--resume", str(tmp_path), "--api-strategy", "")
    assert cfg.api_strategy == "" and cfg.strategy_backend == "local"
    assert "DeepSeek-R1" in cfg.strategy_model_name
