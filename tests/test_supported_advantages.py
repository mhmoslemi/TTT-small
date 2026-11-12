"""CPU regressions for the two supported discovery objectives."""

from contextlib import nullcontext
from concurrent.futures import Future
import json
import math
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

import train_multy_CVaR as training
from training.advantage import entropic_adaptive_advantages
from config_validation import validate_problem_config
from problems.base import RewardResult, SeedState
from evaluation.reward import extract_python_code
from search.sampler import make_sampler
from training.spo_rs import SPORSTracker, SPORSV2Tracker


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(("model_name", "expected"), [
    ("openai/gpt-oss-120b", "unsloth"),
    ("Qwen/Qwen3-8B", "hf"),
    ("Qwen/Qwen3.8-27B", "hf"),
])
def test_auto_training_backend_follows_coder_model(model_name, expected):
    assert training._resolve_training_backend("auto", model_name) == expected


@pytest.fixture
def load_config(monkeypatch):
    monkeypatch.setenv("AVAILABLE_GPUS", "0,1")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setattr('runtime.gpu_runtime.query_gpu_memory', lambda: {})
    monkeypatch.setattr('runtime.gpu_runtime.detect_attention_heads', lambda name: 32)

    def load(mode="entropic", filename="erdos.yaml", extra=()):
        monkeypatch.setattr(sys, "argv", [
            "train_multy_CVaR.py", "--config", str(ROOT / "configs" / filename),
            "--backend", "unsloth", "--generation-backend", "hf",
            "--max-seq-length", "32768", "--max-new-tokens", "4096",
            "--advantage-mode", mode, "--api-strategy", "", *extra,
        ])
        return training.load_config()

    return load


@pytest.mark.parametrize("mode", ["entropic", "spo-rs", "spo-rs-v2"])
@pytest.mark.parametrize("filename", sorted(p.name for p in (ROOT / "configs").glob("*.yaml")))
def test_all_presets_support_supported_objectives(load_config, mode, filename):
    path = ROOT / "configs" / filename
    validate_problem_config(yaml.safe_load(path.read_text()), source=path)
    cfg, values = load_config(mode, filename)
    assert cfg.advantage_mode == mode
    assert cfg.training_model_name not in {None, "", "None"}
    assert not any(key.startswith(("rank_", "x_grpo_", "cvar_", "binary_coder_"))
                   for key in values)
    if training._is_spo_rs_mode(mode):
        assert cfg.spo_rs_update_epochs == 1
        assert cfg.spo_rs_win_alpha == .25


@pytest.mark.parametrize("mode", ["grpo", "cvar", "rank", "x-grpo", "binary-coder"])
def test_removed_objectives_rejected_by_cli_and_yaml(mode):
    with pytest.raises(SystemExit):
        training._build_arg_parser().parse_args(["--advantage-mode", mode])
    values = yaml.safe_load((ROOT / "configs/erdos.yaml").read_text())
    values["advantage_mode"] = mode
    with pytest.raises(ValueError, match="advantage_mode"):
        validate_problem_config(values)
    with pytest.raises(ValueError, match="unsupported training objective"):
        training._normalize_saved_advantage_config(values)


@pytest.mark.parametrize("option", ["--rank-gamma", "--rank-update-epochs",
                                   "--rank-clip-epsilon", "--x-grpo-budgets",
                                   "--cvar-alpha", "--cvar-lambda"])
def test_removed_cli_options_are_not_accepted(option):
    with pytest.raises(SystemExit):
        training._build_arg_parser().parse_args([option, "1"])


def test_help_and_new_policy_controls(load_config):
    help_text = training._build_arg_parser().format_help()
    assert "{entropic,spo-rs,spo-rs-v2}" in help_text
    assert "--rank-" not in help_text
    cfg, _ = load_config("spo-rs", extra=("--spo-rs-update-epochs", "2",
                         "--spo-rs-clip-epsilon", "0.3"))
    assert cfg.spo_rs_update_epochs == 2
    assert cfg.spo_rs_clip_epsilon_low == cfg.spo_rs_clip_epsilon_high == .3


def test_resume_can_upgrade_spo_rs_to_sticky_v2(load_config, tmp_path):
    _, saved = load_config("spo-rs")
    (tmp_path / "config.json").write_text(json.dumps(saved))
    cfg, _ = load_config("spo-rs-v2", extra=("--resume", str(tmp_path)))
    assert cfg.advantage_mode == "spo-rs-v2"

    saved["advantage_mode"] = "spo-rs-v2"
    (tmp_path / "config.json").write_text(json.dumps(saved))
    cfg, _ = load_config("spo-rs", extra=("--resume", str(tmp_path)))
    assert cfg.advantage_mode == "spo-rs-v2"


@pytest.mark.parametrize("mode", ["puct", "uct", "llm"])
def test_parent_picker_cli_and_resumed_config(load_config, tmp_path, mode):
    cfg, values = load_config(extra=("--parent-selection", mode))
    assert cfg.parent_selection == mode and cfg.uct == (mode == "uct")
    saved_dir = tmp_path / "saved"
    saved_dir.mkdir()
    (saved_dir / "config.json").write_text(json.dumps(values))
    cfg, _ = load_config(extra=("--resume", str(saved_dir)))
    assert cfg.parent_selection == mode
    if mode != "llm":
        with pytest.raises(ValueError, match="requires training_state.pt"):
            load_config(extra=("--resume", str(saved_dir), "--parent-selection", "llm"))
        (saved_dir / "training_state.pt").touch()
        cfg, _ = load_config(extra=("--resume", str(saved_dir), "--parent-selection", "llm"))
        assert cfg.parent_selection == "llm" and not cfg.uct
    with pytest.raises(ValueError, match="cannot change parent_selection"):
        load_config(extra=("--resume", str(saved_dir), "--parent-selection",
                           "puct" if mode != "puct" else "uct"))


def test_legacy_uct_config_and_flag_keep_their_meaning(load_config, tmp_path):
    cfg, saved = load_config(extra=("--uct",))
    assert cfg.parent_selection == "uct" and cfg.uct
    saved.pop("parent_selection")
    (tmp_path / "config.json").write_text(json.dumps(saved))
    cfg, _ = load_config(extra=("--resume", str(tmp_path)))
    assert cfg.parent_selection == "uct" and cfg.uct


@pytest.mark.parametrize("mode", ["entropic", "spo-rs", "spo-rs-v2"])
def test_a3b_keeps_sequence_clipping_and_full_policy_sampling(load_config, mode):
    cfg, _ = load_config(mode, extra=(
        "--model-name", "Qwen/Qwen3-30B-A3B", "--temperature", ".7",
        "--top-p", ".9", "--sequence-clip-epsilon-low", ".15",
        "--sequence-clip-epsilon-high", ".25"))
    assert training._uses_sequence_level_policy_ratio(cfg)
    assert cfg.temperature == cfg.top_p == 1.
    assert cfg.sampling_top_k == cfg.sampling_min_p == 0
    _, low, high, kl = training._clipped_policy_options(cfg)
    if mode == "entropic":
        assert (low, high, kl) == (.15, .25, cfg.kl_penalty_coef)
    else:
        assert (low, high, kl) == (cfg.spo_rs_clip_epsilon_low,
                                   cfg.spo_rs_clip_epsilon_high, 0.)


def test_optional_values_retain_model_and_worker_defaults():
    assert training._resolve_training_model_name("Qwen/Qwen3-8B", None, False) == "Qwen/Qwen3-8B"
    assert training._model_basename(None) == ""
    cfg = SimpleNamespace(reward_workers=None, num_gpus=None, isolate_eval=False)
    problem = SimpleNamespace(eval_cpus=2)
    assert training._resolve_reward_workers(cfg, problem, cpu_count=8) >= 1


def test_qwen38_medium_pilots_and_phase2_with_saved_low():
    cfg = SimpleNamespace(coder_template_kind="qwen3.8", coder_reasoning_effort="low",
                          pilot_medium_programs_per_strategy=10,
                          pilot_xhigh_programs_per_strategy=1)
    entries = training._coder_effort_entries([], [0, 1], [11, 11], cfg, 7, "pilot")
    assert entries == [(0, 1, "xhigh"), (1, 1, "xhigh"),
                       (0, 10, "medium"), (1, 10, "medium")]
    assert training._coder_effort_entries([], [0], [5], cfg, 7, "adaptive") == [(0, 5, "medium")]
    jobs = [dict(messages=[], prompt_text="low", coder_reasoning_effort="low")]
    cache = {}
    render = lambda messages, **options: options["reasoning_effort"]
    pilot = training._phase_coder_prompt_job(jobs, 0, "pilot", cfg, render, cache, effort="medium")
    adaptive = training._phase_coder_prompt_job(jobs, 0, "adaptive", cfg, render, cache)
    assert pilot == adaptive != 0
    assert jobs[adaptive]["prompt_text"] == jobs[adaptive]["coder_reasoning_effort"] == "medium"
    assert jobs[0]["prompt_text"] == "low"  # saved/source prompts are never overwritten
    cfg.coder_template_kind = "gpt-oss"
    assert training._coder_effort_entries([], [0], [11], cfg, 7, "pilot") == [(0, 11, "low")]
    assert training._coder_effort_for_rollout_phase(cfg, "adaptive") == "low"


def test_qwen38_phase2_effort_counts_and_exact_prompt_retries():
    from generation.output_retries import coder_retry_prompt_job

    cfg = SimpleNamespace(
        coder_template_kind="qwen3.8", coder_reasoning_effort="low",
        phase2_low_programs_per_strategy=2,
        phase2_medium_programs_per_strategy=-1,
        phase2_xhigh_programs_per_strategy=1)
    entries = training._coder_effort_entries(
        [], [0, 1, 2], [7, 2, 0], cfg, 7, "adaptive")
    assert entries == [(0, 1, "xhigh"), (1, 1, "xhigh"),
                       (0, 4, "medium"), (0, 2, "low"), (1, 1, "low")]
    jobs = [dict(messages=[], prompt_text="low", coder_reasoning_effort="low")]
    cache = {}
    retry_cache = {}
    render = lambda messages, **options: options["reasoning_effort"]
    for effort in ("low", "medium", "xhigh"):
        index = training._phase_coder_prompt_job(
            jobs, 0, "adaptive", cfg, render, cache, effort=effort)
        assert jobs[index]["prompt_text"] == effort
        assert jobs[index]["coder_reasoning_effort"] == effort
        retry = coder_retry_prompt_job(
            jobs, dict(job_idx=index, strategy_rollout_phase="adaptive"),
            render, retry_cache, reasoning_effort=effort)
        assert jobs[retry]["prompt_text"] == effort
        assert jobs[retry]["coder_reasoning_effort"] == effort
    assert jobs[0]["prompt_text"] == "low"
    cfg.coder_template_kind = "gpt-oss"
    assert training._coder_effort_entries(
        [], [0], [7], cfg, 7, "adaptive") == [(0, 7, "low")]


def test_qwen38_phase2_remainders_and_small_allocations():
    for requested, count, expected in [
        ((-1, 2, 1), 8, [(0, 1, "xhigh"), (0, 2, "medium"), (0, 5, "low")]),
        ((1, 2, -1), 8, [(0, 5, "xhigh"), (0, 2, "medium"), (0, 1, "low")]),
        ((4, 3, 2), 1, [(0, 1, "xhigh")]),
        ((2, 1, 1), 7, [(0, 1, "xhigh"), (0, 1, "medium"), (0, 5, "low")]),
        ((0, -1, 0), 0, []),
    ]:
        cfg = SimpleNamespace(coder_template_kind="qwen3.8", coder_reasoning_effort="low")
        for effort, value in zip(("low", "medium", "xhigh"), requested):
            setattr(cfg, f"phase2_{effort}_programs_per_strategy", value)
        entries = training._coder_effort_entries([], [0], [count], cfg, 7, "adaptive")
        assert entries == expected
        assert sum(entry[1] for entry in entries) == count


def test_resume_uses_current_yaml_pilot_effort_counts(load_config, tmp_path):
    options = ("--strategies", "--coder-model-name", "Qwen/Qwen3.8-27B")
    _, saved = load_config(extra=options)
    saved.pop("pilot_medium_programs_per_strategy", None)
    saved.update(coder_reasoning_effort="low", pilot_xhigh_programs_per_strategy=2)
    saved.update(phase2_low_programs_per_strategy=-1,
                 phase2_medium_programs_per_strategy=0,
                 phase2_xhigh_programs_per_strategy=2)
    (tmp_path / "config.json").write_text(json.dumps(saved))
    resumed, _ = load_config(extra=(*options, "--resume", str(tmp_path)))
    current = yaml.safe_load((ROOT / "configs/erdos.yaml").read_text())
    for key in ("pilot_medium_programs_per_strategy", "pilot_xhigh_programs_per_strategy",
                "phase2_low_programs_per_strategy", "phase2_medium_programs_per_strategy",
                "phase2_xhigh_programs_per_strategy"):
        assert getattr(resumed, key) == current[key]
    assert training._coder_effort_for_rollout_phase(resumed, "adaptive") == "medium"


def test_pilot_effort_counts_reject_invalid_or_excess_budget():
    values = yaml.safe_load((ROOT / "configs/erdos.yaml").read_text())
    for count in (-1, True, values["pilot_programs_per_strategy"] + 1):
        invalid = dict(values, pilot_medium_programs_per_strategy=count)
        with pytest.raises(ValueError, match="pilot.*(medium|xhigh)"):
            validate_problem_config(invalid)


def test_phase2_effort_counts_reject_invalid_values():
    for effort in ("low", "medium", "xhigh"):
        for count in (-2, True, 1.5):
            with pytest.raises(ValueError, match="phase2"):
                validate_problem_config(
                    {"problem": "erdos", f"phase2_{effort}_programs_per_strategy": count},
                    require_complete=False)
    with pytest.raises(ValueError, match="only one phase2"):
        validate_problem_config(
            {"problem": "erdos", "phase2_low_programs_per_strategy": -1,
             "phase2_medium_programs_per_strategy": -1}, require_complete=False)


def test_resume_migrates_retained_controls_and_drops_unused_options():
    saved = dict(advantage_mode="spo-rs", rank_update_epochs=3, rank_gamma=1.,
                 binary_coder_training=False, x_grpo_budgets=[1, 2], cvar_alpha=.2)
    current = training._normalize_saved_advantage_config(saved)
    assert current == {"advantage_mode": "spo-rs", "spo_rs_update_epochs": 3}
    assert saved["rank_update_epochs"] == 3
    current = training._normalize_saved_advantage_config(dict(
        advantage_mode="entropic", rank_clip_epsilon=.2, rank_clip_epsilon_high=.4))
    assert current["sequence_clip_epsilon_low"] == .2
    assert current["sequence_clip_epsilon_high"] == .4
    with pytest.raises(ValueError, match="unsupported training objective"):
        training._normalize_saved_advantage_config(dict(
            advantage_mode="entropic", binary_coder_training=True))


@pytest.mark.parametrize("key", ["rank_gamma", "x_grpo_budgets", "cvar_alpha",
                                "binary_coder_training"])
def test_partial_configs_cannot_reactivate_removed_options(key):
    with pytest.raises(ValueError, match="removed objective options"):
        validate_problem_config({"problem": "erdos", key: 1}, require_complete=False)


def test_entropic_leave_one_out_and_tied_groups():
    rewards = np.array([0., .4, .8, 1.])
    actual, beta = entropic_adaptive_advantages(rewards)
    weights = np.exp(beta * (rewards - rewards.max()))
    expected = weights / ((weights.sum() - weights) / 3 + 1e-12) - 1
    np.testing.assert_allclose(actual, expected, rtol=1e-14)
    np.testing.assert_array_equal(entropic_adaptive_advantages(np.ones(4))[0], np.zeros(4))
    np.testing.assert_array_equal(entropic_adaptive_advantages(np.array([1.]))[0], [0.])


@pytest.mark.parametrize("sequence", [False, True])
@pytest.mark.parametrize("advantage", [-2., 0., .6])
@pytest.mark.parametrize("kl", [0., .1])
def test_clipped_loss_and_gradients(sequence, advantage, kl):
    old = torch.tensor([-2., -1., -3.], dtype=torch.float64)
    ref = torch.tensor([-2.3, -.9, -2.7], dtype=torch.float64)
    current = (old + torch.tensor([-.4, .15, .5])).requires_grad_()
    loss, _ = training.clipped_surrogate_loss(
        current, old, ref, advantage, clip_epsilon_low=.1,
        clip_epsilon_high=.3, kl_coef=kl, sequence_level_ratio=sequence)
    ratios = ((current - old).mean() if sequence else current - old).exp()
    expected = -torch.minimum(ratios * advantage, ratios.clamp(.9, 1.3) * advantage).mean()
    correction = kl * ((current - ref).detach().mean() - (current - ref))
    expected -= (ratios.detach() * correction.detach() * current).mean()
    torch.testing.assert_close(loss, expected, rtol=0, atol=0)
    torch.testing.assert_close(torch.autograd.grad(loss, current)[0],
                               torch.autograd.grad(expected, current)[0], rtol=0, atol=0)


class TinyTokenizer:
    pad_token_id = 0

    def __call__(self, text, **kwargs):
        return SimpleNamespace(input_ids=torch.tensor([[1, 2]]))

    def apply_chat_template(self, messages, **kwargs):
        return messages[-1]["content"]


class TinyProblem:
    name = "tiny"
    metric_name = "score"
    maximize = True
    saves_construction = False
    fail_score = 0.
    eval_cpus = 1
    target = 100.

    def build_prompt(self, parent):
        return [{"role": "user", "content": "CODE_REQUEST"}]

    def build_strategy_messages(self, messages, previous_strategies=None):
        return [{"role": "user", "content": "STRATEGY_REQUEST"}]

    def build_code_messages(self, messages, strategy):
        return [{"role": "user", "content": "CODE_REQUEST " + strategy}]

    def compute_reward(self, text, parent, timeout_s, **kwargs):
        index = int(text.split("candidate=")[1].split()[0])
        reward = float(index % 4)
        return RewardResult(reward=reward, raw_score=reward, valid=reward > 0,
                            parsed=True, ran=True, code=extract_python_code(text))


class CapturingTrainer:
    is_offloaded = False

    def __init__(self):
        self.calls = []

    def train_policy(self, examples, cfg, step):
        self.calls.append(("entropic", examples))
        return {"training_seconds": 0.}

    def train_spo_rs(self, examples, cfg, step):
        self.calls.append(("spo-rs", examples))
        return {"training_seconds": 0.}


@pytest.mark.parametrize("mode", ["entropic", "spo-rs"])
@pytest.mark.parametrize("allocation", [None, "fixed", "rule_based", "bandit", "hurdle"])
@pytest.mark.parametrize("remote_strategy,pooled_coder", [
    (False, False), (True, False), (True, True), (True, "live")])
def test_complete_step_with_cpu_generation_and_evaluation(
        monkeypatch, tmp_path, load_config, mode, allocation, remote_strategy, pooled_coder,
        parent_selection="puct", num_steps=2):
    cfg, _ = load_config(mode)
    cfg.num_steps = num_steps
    cfg.groups_per_step = 2
    cfg.group_size = 4
    cfg.strategies_per_parent = 2
    cfg.programs_per_strategy = 2
    cfg.pilot_programs_per_strategy = -1 if allocation == "fixed" else 1
    cfg.pilot_medium_programs_per_strategy = 0
    cfg.pilot_xhigh_programs_per_strategy = 0
    cfg.phase2_allocation_method = allocation if allocation not in {None, "fixed"} else "hurdle"
    if pooled_coder == "live" and allocation not in {None, "fixed"}:
        cfg.programs_per_strategy = 3
        cfg.group_size = 6
        cfg.pilot_programs_per_strategy = 2
        cfg.pilot_xhigh_programs_per_strategy = 1
        cfg.coder_template_kind = "qwen3.8"
    expected_total = cfg.groups_per_step * cfg.group_size
    cfg.reward_workers = 2
    cfg.max_seq_length = 128
    cfg.max_new_tokens = 16
    cfg.gen_micro_batch = 4
    cfg.logprob_chunk = 8
    cfg.measure_entropy = False
    cfg.no_train = False
    model = torch.nn.Linear(1, 1)
    model.device = torch.device("cpu")
    backend = SimpleNamespace(set_training_mode=model.train, set_inference_mode=model.eval,
                              disable_adapter=nullcontext)
    optimizer = torch.optim.SGD(model.parameters(), lr=.01)
    problem = TinyProblem()
    problem.two_stage_rollouts = allocation is not None
    strategy_pool = None
    remote_prompts = []
    api_started = threading.Event()
    coder_started = threading.Event()
    worker_prepared = threading.Event()
    if remote_strategy and allocation is not None:
        from generation.strategy_prompts_openai import build_openai_strategy_messages
        cfg.api_strategy = "gpt-6-luna"
        cfg.strategy_backend = "api"
        problem.build_strategy_messages = build_openai_strategy_messages

        class RemotePool:
            def iter_group_jobs(self, prompts_by_group, **kwargs):
                api_started.set()
                for position, (i, prompt) in enumerate(reversed(list(enumerate(prompts_by_group)))):
                    if pooled_coder == "live" and position == 1:
                        # The very first coder rollout must happen before the
                        # remaining API response can complete even round one.
                        assert coder_started.wait(5), "coder waited for all API plans"
                    remote_prompts.append(json.loads(prompt[1]["content"]))
                    yield i, [(json.dumps({"plan": "Use an explicit search plan."}), [])]

            def release(self):
                pass

        strategy_pool = RemotePool()
    seed_values = [2., 1.] if mode == "spo-rs-v2" else [1., 1.]
    sampler = make_sampler(
        parent_selection=parent_selection, num_seeds=2, group_size=cfg.group_size,
        exp_dir=tmp_path, problem=problem,
        seed=cfg.seed,
        seed_states=[SeedState(value=value, raw_score=value) for value in seed_values])
    prefetched = []
    if parent_selection == "llm":
        def select(request_path, response_path, tree_path, record):
            tree = json.loads(tree_path.read_text())
            ids = sorted(
                sampler._seed_ids,
                key=lambda sid: tree["nodes"][sid]["raw_problem_score"],
                reverse=True)
            result = {
                "selected_parent_ids": ids,
                "selections": [{"node_id": sid, "role": "exploit", "reason": "Seed anchor"}
                               for sid in ids],
                "tree_diagnosis": "Bootstrap", "confidence": "low"}
            response = {"status": "completed", "output": [{"type": "message", "content": [
                {"type": "output_text", "text": json.dumps(result)}]}]}
            response_path.write_text(json.dumps(response))
            return response
        monkeypatch.setattr(sampler, "_request", select)

        def prefetch(count, *, selection_step):
            metadata = [json.loads(path.read_text()) for path in
                        (tmp_path / "step00").glob("*rollout*.meta.json")]
            assert len(metadata) == expected_total
            assert all(row.get("status") != "evaluation_pending" for row in metadata)
            assert sampler.best_raw_state(maximize=True).raw_score == 3.
            assert not trainer.calls, "selection started after training finished"
            prefetched.append((count, selection_step))

        monkeypatch.setattr(sampler, "prefetch_states", prefetch)
    trainer = CapturingTrainer()
    adapter = tmp_path / "adapter_initial"
    adapter.mkdir()
    tracker = None
    if training._is_spo_rs_mode(mode):
        tracker_type = SPORSV2Tracker if mode == "spo-rs-v2" else SPORSTracker
        tracker = tracker_type(
            entropic_beta=2., d_half=.06, rho_min=.875, rho_max=.96,
            **({"win_alpha": .25} if mode == "spo-rs-v2" else {}))
        tracker.update(tracker.transformed_rewards(seed_values), divergence=0.,
                       policy_adapter=str(adapter), step=-1,
                       raw_rewards=seed_values, valid_flags=[True, True])
    initial_tracker = tracker.state_dict() if tracker else None

    generated = []

    def generate(model, tokenizer, prompts, counts, cfg, **kwargs):
        for job, (prompt, count) in enumerate(zip(prompts, counts)):
            if prompt == "STRATEGY_REQUEST":
                yield job, [("<strategy>Use an explicit search plan.</strategy>", [1, 2])]
            else:
                responses = []
                for _ in range(count):
                    index = len(generated)
                    text = f"```python\n# candidate={index}\ndef run(): return {index}\n```"
                    generated.append(index)
                    responses.append((text, [3, 4]))
                yield job, responses

    def scores(model, examples, with_grad=False, **kwargs):
        return [torch.full((ex["response_ids"].shape[1],), -.7) for ex in examples]

    monkeypatch.setattr(training, "generate_prompt_jobs", generate)
    monkeypatch.setattr(training, "compute_batched_token_logprobs", scores)
    coder_pool = None
    if pooled_coder:
        class CoderPool:
            sequential = False

            def iter_group_jobs(self, prompts_by_group, counts_by_group, **kwargs):
                assert kwargs["adapter_path"] == str(adapter)
                yield from generate(model, TinyTokenizer(), prompts_by_group,
                                    counts_by_group, cfg)

            def score_token_logprobs(self, pairs, **kwargs):
                return [[-.7] * len(response_ids) for _, response_ids in pairs]

        if pooled_coder == "live":
            class LiveCoderPool(CoderPool):
                def prepare(self):
                    assert api_started.wait(5), "worker startup blocked API submission"
                    assert not coder_started.is_set()
                    worker_prepared.set()

                def iter_ready_jobs(self, poll_jobs, **kwargs):
                    assert worker_prepared.is_set()
                    assert kwargs["adapter_path"] == str(adapter)
                    expires = time.monotonic() + 5
                    done = False
                    while not done:
                        assert time.monotonic() < expires, "pipeline stopped making progress"
                        jobs, done = poll_jobs()
                        for job in jobs:
                            coder_started.set()
                            for _, results in generate(
                                    model, TinyTokenizer(), [job["prompt_text"]],
                                    [job["count"]], cfg):
                                yield job["key"], [(text, ids, [-.7] * len(ids))
                                                  for text, ids in results]
                        if not jobs and not done:
                            threading.Event().wait(.001)

                def score_token_logprobs(self, pairs, **kwargs):
                    return [[-.7] * len(response_ids) for _, response_ids in pairs]

            CoderPool = LiveCoderPool
            if allocation:
                cfg.generation_backend = "vllm"

        coder_pool = CoderPool()
        monkeypatch.setattr(training, "_save_adapter", lambda *args, **kwargs: str(adapter))
    stats = training.train_step(
        backend, model, TinyTokenizer(), sampler, optimizer, 0, cfg, tmp_path,
        problem, parallel_trainer=trainer, spo_rs_tracker=tracker,
        strategy_pool=strategy_pool, gen_pool=coder_pool,
        sampling_adapter_path=str(adapter) if tracker else None)
    assert len(generated) == expected_total
    assert len(trainer.calls) == 1
    if parent_selection == "llm":
        assert prefetched == ([(2, 1)] if num_steps > 1 else [])
    assert trainer.calls[0][0] == ("spo-rs" if training._is_spo_rs_mode(mode) else mode)
    examples = trainer.calls[0][1]
    assert len(examples) == expected_total
    assert math.isclose(sum(ex["sample_weight"] for ex in examples), 1.)
    assert all(not any(key.startswith(("rank_", "x_grpo_", "binary_")) for key in ex)
               for ex in examples)
    metadata = [json.loads(p.read_text()) for p in (tmp_path / "step00").glob("*rollout*.meta.json")]
    assert len(metadata) == expected_total
    assert {row["advantage_mode"] for row in metadata} == {mode}
    assert stats["planned_rollouts"] == expected_total
    if training._is_spo_rs_mode(mode):
        assert stats["spo_rs_policy_updated"]
    if mode == "spo-rs":
        for row in metadata:
            expected = (math.exp(2 * row["reward"]) / math.exp(2) - 1) / 2
            assert row["advantage"] == pytest.approx(expected)
    if mode == "spo-rs-v2":
        spo_rs = [training._literal_metadata_mapping(row["spo_rs"])
                  for row in metadata]
        assert all(info["win_lambda"] > 0. for info in spo_rs)
        assert any(row["advantage"] != pytest.approx(
            (info["transformed_reward"] / info["value_before"] - 1.) / 2.)
            for row, info in zip(metadata, spo_rs))
    if allocation:
        if allocation != "fixed":
            assert len(stats["strategy_pilot_allocation"]) == 4
        assert len(list((tmp_path / "step00").glob("*fold00_strategy[0-9][0-9].txt"))) == 4
        if remote_strategy:
            assert len(remote_prompts) == 4
            assert [len(prompt["previous_strategies"]) for prompt in remote_prompts] == [0, 0, 1, 1]
            assert all(prompt["previous_strategies"] == ["Use an explicit search plan."]
                       for prompt in remote_prompts[2:])
            if pooled_coder == "live":
                assert worker_prepared.is_set() and coder_started.is_set()
    if parent_selection == "llm" and training._is_spo_rs_mode(mode):
        recovered = make_sampler(
            parent_selection="llm", num_seeds=2, exp_dir=tmp_path, problem=problem,
            seed_states=[SeedState(value=value, raw_score=value) for value in seed_values])
        monkeypatch.setattr(recovered, "_request", lambda *args: pytest.fail("repeated API call"))
        tracker_type = SPORSV2Tracker if mode == "spo-rs-v2" else SPORSTracker
        recovered_tracker = tracker_type(
            entropic_beta=2., d_half=.06, rho_min=.875, rho_max=.96,
            **({"win_alpha": .25} if mode == "spo-rs-v2" else {}))
        recovered_tracker.load_state_dict(initial_tracker)
        artifacts = training._recoverable_spo_rs_artifacts(tmp_path, 0, cfg)
        assert artifacts is not None
        training._prepare_recoverable_response_tokens(None, TinyTokenizer(), artifacts)
        cfg.strategies = allocation is not None
        training._resume_spo_rs_step_from_artifacts(
            backend, model, TinyTokenizer(), recovered, optimizer, 0, cfg, tmp_path,
            problem, CapturingTrainer(), recovered_tracker, str(adapter), artifacts)
        assert recovered.state_dict() == sampler.state_dict()


def test_spo_rs_v2_complete_step(monkeypatch, tmp_path, load_config):
    test_complete_step_with_cpu_generation_and_evaluation(
        monkeypatch, tmp_path, load_config, "spo-rs-v2", None,
        remote_strategy=False, pooled_coder=False,
        parent_selection="llm", num_steps=1)


@pytest.mark.parametrize("mode", ["entropic", "spo-rs"])
@pytest.mark.parametrize("allocation", [None, "hurdle"])
@pytest.mark.parametrize("num_steps", [1, 2])
def test_llm_full_step_and_spo_recovery(
        monkeypatch, tmp_path, load_config, mode, allocation, num_steps):
    test_complete_step_with_cpu_generation_and_evaluation(
        monkeypatch, tmp_path, load_config, mode, allocation,
        remote_strategy=True, pooled_coder=True, parent_selection="llm", num_steps=num_steps)


def test_behavior_cache_keeps_only_aligned_finite_scores():
    examples = [dict(response_ids=torch.ones(1, 2), behavior_logprobs=values)
                for values in (torch.tensor([-.2, -.3]), torch.tensor([-.2]),
                               torch.tensor([float("nan"), -.3]), None)]
    missing = training._initialize_behavior_logprobs(examples)
    assert missing == examples[1:]
    assert examples[0]["old_policy_logprobs"] is not None
    assert all(ex["old_policy_logprobs"] is None for ex in missing)


@pytest.mark.parametrize("epochs", [1, 2])
def test_spo_rs_cpu_optimizer_preserves_frozen_behavior(monkeypatch, load_config, epochs):
    cfg, _ = load_config("spo-rs", extra=("--spo-rs-update-epochs", str(epochs)))
    cfg.train_examples_per_microbatch = 2
    cfg.grad_clip = 100.
    model = torch.nn.Linear(1, 1, bias=False, dtype=torch.float64)
    model.device = torch.device("cpu")
    with torch.no_grad():
        model.weight.fill_(.1)
    backend = SimpleNamespace(set_training_mode=model.train)
    optimizer = torch.optim.SGD(model.parameters(), lr=.05)
    examples = [dict(prompt_ids=torch.tensor([[1, 2]]), response_ids=torch.tensor([[3, 4]]),
                     advantage=adv, sample_weight=weight, behavior_logprobs=behavior)
                for adv, weight, behavior in [
                    (1.2, .4, torch.tensor([-.95, -.9], dtype=torch.float64)),
                    (-.7, .6, None)]]
    fallback_calls = []

    def scores(model, batch, with_grad=False, **kwargs):
        if not with_grad:
            fallback_calls.extend(batch)
        with torch.set_grad_enabled(with_grad):
            return [model.weight[0, 0] * torch.tensor([.5, 1.], dtype=torch.float64) - 1
                    for ex in batch]

    monkeypatch.setattr("training.trainers.compute_batched_token_logprobs", scores)
    expected = model.weight.detach().clone().requires_grad_()
    old = torch.tensor([-.95, -.9], dtype=torch.float64)
    for _ in range(epochs):
        current = expected[0, 0] * torch.tensor([.5, 1.], dtype=torch.float64) - 1
        ratios = (current - old).exp()
        expected_loss = sum(
            -ex["sample_weight"] * torch.minimum(
                ratios * ex["advantage"],
                ratios.clamp(1 - cfg.spo_rs_clip_epsilon_low,
                             1 + cfg.spo_rs_clip_epsilon_high) * ex["advantage"]).mean()
            for ex in examples)
        expected = (expected - .05 * torch.autograd.grad(expected_loss, expected)[0]).detach().requires_grad_()
    result = training._train_spo_rs_examples(
        backend, model, TinyTokenizer(), optimizer, examples, cfg, 0)
    torch.testing.assert_close(model.weight, expected, rtol=1e-12, atol=1e-12)
    assert len(fallback_calls) == 1
    assert fallback_calls[0] is examples[1]
    assert len(result["policy_updates"]) == epochs
    assert all("old_policy_logprobs" not in ex for ex in examples)


@pytest.mark.parametrize("v2", [False, True])
def test_overlap_helpers_use_the_same_retained_advantages(v2):
    rewards = [0., 1., 2., 3.]
    futures = [Future() for _ in rewards]
    records = [dict(job_idx=0, token_ids=[3, 4], behavior_logprobs=[-.2, -.3])
               for _ in rewards]
    jobs = [{"prompt_text": "CODE_REQUEST"}]
    staged = training._prepare_spo_rs_overlap_inputs(
        TinyTokenizer(), {0: records}, {0: futures}, jobs)
    tracker_type = SPORSV2Tracker if v2 else SPORSTracker
    tracker = tracker_type(
        entropic_beta=2., d_half=.06, rho_min=.875, rho_max=.96,
        **({"win_alpha": .25} if v2 else {}))
    history = [.5, 1., 2.]
    tracker.update(
        tracker.transformed_rewards(history), divergence=0.,
        policy_adapter="unused-test-adapter", step=-1,
        raw_rewards=history, valid_flags=[True] * len(history))
    tracker.prepare_step()
    with pytest.raises(RuntimeError, match="before its reward completed"):
        training._complete_spo_rs_overlap_examples(staged, tracker)
    for future, reward in zip(futures, rewards):
        future.set_result(SimpleNamespace(reward=reward, valid=True))
    completed = training._complete_spo_rs_overlap_examples(staged, tracker)
    expected = tracker.rollout_advantages(rewards, [True] * len(rewards))
    np.testing.assert_allclose([ex["advantage"] for ex in completed], expected)
    assert sum(ex["sample_weight"] for ex in completed) == 1.
    assert all("_overlap_reward_future" not in ex for ex in completed)
    entropic = training._prepare_entropic_overlap_group(
        TinyTokenizer(), 0, records, futures, jobs, {}, 2)
    expected, _ = entropic_adaptive_advantages(np.asarray(rewards))
    np.testing.assert_allclose([ex["advantage"] for ex in entropic], expected)
    assert sum(ex["sample_weight"] for ex in entropic) == .5


@pytest.mark.parametrize("rounding_delta,discard", [
    (1e-15, False), (1e-4, True), (float("nan"), True),
])
def test_spo_rs_v2_overlap_preserves_gradients_after_roundoff(
        monkeypatch, tmp_path, load_config, capsys, rounding_delta, discard):
    trainers = []

    class OverlappingTrainer(CapturingTrainer):
        world_size = 2

        def __init__(self):
            super().__init__()
            self.queued = []
            self.finishes = []
            trainers.append(self)

        def begin_spo_rs_overlap(self, cfg, step):
            self.state = dict(queued_examples=0, supplied_old=0, scheduled_keys=set())
            return self.state

        def queue_spo_rs_overlap(self, examples):
            for example in examples:
                example["advantage"] += rounding_delta
                self.queued.append(dict(example))
                self.state["scheduled_keys"].add(example["_overlap_key"])
            self.state["queued_examples"] += len(examples)
            self.state["supplied_old"] += len(examples)

        def seal_spo_rs_overlap(self):
            pass

        def finish_spo_rs_overlap(self, cfg, step, count, *, apply_update=True):
            assert count == len(self.queued)
            self.finishes.append(apply_update)
            if apply_update:
                self.calls.append(("spo-rs", self.queued))
            return {"training_seconds": 0.}

    monkeypatch.setattr(sys.modules[__name__], "CapturingTrainer", OverlappingTrainer)

    def vllm_config(*args, **kwargs):
        cfg, values = load_config(*args, **kwargs)
        cfg.generation_backend = "vllm"
        return cfg, values

    test_complete_step_with_cpu_generation_and_evaluation(
        monkeypatch, tmp_path, vllm_config, "spo-rs-v2", None,
        remote_strategy=False, pooled_coder=True, num_steps=1)
    assert len(trainers) == 1
    assert trainers[0].finishes == [not discard]
    output = capsys.readouterr().out
    assert ("discarded streamed SPO-RS gradients" in output) == discard
    assert "[step 0] best-ever raw score:" in output


def test_spo_rs_v2_adds_frozen_win_signal_and_round_trips():
    tracker = SPORSV2Tracker(
        entropic_beta=2., d_half=.06, rho_min=.875, rho_max=.96,
        win_alpha=.25)
    seed_rewards = [1., 1.]
    tracker.update(
        tracker.transformed_rewards(seed_rewards), divergence=0.,
        policy_adapter="adapter_initial", step=-1,
        raw_rewards=seed_rewards, valid_flags=[True, True])
    tracker.prepare_step()
    np.testing.assert_allclose(
        tracker.rollout_advantages([0., 1.], [False, True]),
        tracker.normalized_advantages(tracker.transformed_rewards([0., 1.])))

    rewards = [0., .5, 1., 2.]
    valid = [False, True, True, True]
    tracker.update(
        tracker.transformed_rewards(rewards), divergence=.03,
        policy_adapter="adapter_step000", step=0,
        raw_rewards=rewards, valid_flags=valid)
    tracker.prepare_step()
    base = tracker.normalized_advantages(tracker.transformed_rewards(rewards))
    combined = tracker.rollout_advantages(rewards, valid)
    assert tracker.win_info()["win_lambda"] > 0.
    assert tracker.win_info()["win_scale"] > 0.
    assert not np.allclose(combined, base)
    assert combined[-1] - combined[-2] > base[-1] - base[-2]

    restored = SPORSV2Tracker(
        entropic_beta=2., d_half=.06, rho_min=.875, rho_max=.96,
        win_alpha=.25)
    restored.load_state_dict(tracker.state_dict())
    restored.prepare_step()
    np.testing.assert_allclose(
        restored.rollout_advantages(rewards, valid), combined,
        rtol=1e-14, atol=1e-14)


def test_spo_rs_v2_resume_reconstructs_exact_v1_history(tmp_path):
    options = dict(entropic_beta=2., d_half=.06, rho_min=.875, rho_max=.96)
    v1 = SPORSTracker(**options)
    seeds = [.2, .8]
    v1.update(
        v1.transformed_rewards(seeds), divergence=0.,
        policy_adapter="adapter_initial", step=-1)
    steps = [
        ([0., .4, 1.2], [False, True, True], .01),
        ([0., .7, 1.8, 2.1], [False, True, True, True], .08),
    ]
    for step, (rewards, valid, divergence) in enumerate(steps):
        info = v1.update(
            v1.transformed_rewards(rewards), divergence=divergence,
            policy_adapter=f"adapter_step{step:03d}", step=step)
        step_dir = tmp_path / f"step{step:02d}"
        step_dir.mkdir()
        for rollout, (reward, is_valid) in enumerate(zip(rewards, valid)):
            metadata = {
                "step": step,
                "reward": reward,
                "valid": is_valid,
                "advantage_mode": "spo-rs",
                "spo_rs": repr({**info, "used_for_policy_update": True}),
            }
            (step_dir / (
                f"step{step:02d}_group00_rollout{rollout:03d}.meta.json"
            )).write_text(json.dumps(metadata))

    sampler_state = {
        "seed_ids": ["seed-a", "seed-b"],
        "states": [
            {"id": "seed-a", "value": seeds[0]},
            {"id": "seed-b", "value": seeds[1]},
        ],
    }
    v2 = SPORSV2Tracker(**options, win_alpha=.25)
    migrated = training._migrate_spo_rs_v2_history(
        v2, v1.state_dict(), sampler_state, tmp_path, len(steps))
    assert migrated == {"seed_rewards": 2, "rollouts": 7}
    assert v2.baseline() == pytest.approx(v1.baseline(), rel=1e-13)
    assert v2.last_policy_adapter == v1.last_policy_adapter
    v2.prepare_step()
    assert v2.win_info()["win_lambda"] > 0.
