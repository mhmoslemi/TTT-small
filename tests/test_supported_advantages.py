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
from advantage import entropic_adaptive_advantages
from config_validation import validate_problem_config
from problems.base import RewardResult, SeedState
from reward import extract_python_code
from sampler import make_sampler
from spo_rs import SPORSTracker


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
    monkeypatch.setattr("gpu_runtime.query_gpu_memory", lambda: {})
    monkeypatch.setattr("gpu_runtime.detect_attention_heads", lambda name: 32)

    def load(mode="entropic", filename="erdos.yaml", extra=()):
        monkeypatch.setattr(sys, "argv", [
            "train_multy_CVaR.py", "--config", str(ROOT / "configs" / filename),
            "--backend", "unsloth", "--generation-backend", "hf",
            "--max-seq-length", "32768", "--max-new-tokens", "4096",
            "--advantage-mode", mode, "--api-strategy", "", *extra,
        ])
        return training.load_config()

    return load


@pytest.mark.parametrize("mode", ["entropic", "spo-rs"])
@pytest.mark.parametrize("filename", sorted(p.name for p in (ROOT / "configs").glob("*.yaml")))
def test_all_presets_support_both_objectives(load_config, mode, filename):
    path = ROOT / "configs" / filename
    validate_problem_config(yaml.safe_load(path.read_text()), source=path)
    cfg, values = load_config(mode, filename)
    assert cfg.advantage_mode == mode
    assert cfg.training_model_name not in {None, "", "None"}
    assert not any(key.startswith(("rank_", "x_grpo_", "cvar_", "binary_coder_"))
                   for key in values)
    if mode == "spo-rs":
        assert cfg.spo_rs_update_epochs == 1


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
    assert "{entropic,spo-rs}" in help_text
    assert "--rank-" not in help_text
    cfg, _ = load_config("spo-rs", extra=("--spo-rs-update-epochs", "2",
                         "--spo-rs-clip-epsilon", "0.3"))
    assert cfg.spo_rs_update_epochs == 2
    assert cfg.spo_rs_clip_epsilon_low == cfg.spo_rs_clip_epsilon_high == .3


@pytest.mark.parametrize("mode", ["puct", "uct", "thompson", "llm"])
def test_parent_picker_cli_and_resumed_config(load_config, tmp_path, mode):
    cfg, values = load_config(extra=("--parent-selection", mode))
    assert cfg.parent_selection == mode and cfg.uct == (mode == "uct")
    saved_dir = tmp_path / "saved"
    saved_dir.mkdir()
    (saved_dir / "config.json").write_text(json.dumps(values))
    cfg, _ = load_config(extra=("--resume", str(saved_dir)))
    assert cfg.parent_selection == mode
    if mode in ("puct", "uct"):
        with pytest.raises(ValueError, match="requires training_state.pt"):
            load_config(extra=("--resume", str(saved_dir), "--parent-selection", "thompson"))
        (saved_dir / "training_state.pt").touch()
        cfg, _ = load_config(extra=("--resume", str(saved_dir), "--parent-selection", "thompson"))
        assert cfg.parent_selection == "thompson" and not cfg.uct
    if mode != "llm":
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


@pytest.mark.parametrize("mode", ["entropic", "spo-rs"])
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
        from strategy_prompts_openai import build_openai_strategy_messages
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
    sampler = make_sampler(
        parent_selection=parent_selection, num_seeds=2, group_size=cfg.group_size,
        exp_dir=tmp_path, problem=problem,
        seed=cfg.seed, seed_states=[SeedState(value=1., raw_score=1.) for _ in range(2)])
    prefetched = []
    if parent_selection == "llm":
        def select(request_path, response_path, tree_path, record):
            ids = list(sampler._seed_ids)
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
    if mode == "spo-rs":
        tracker = SPORSTracker(entropic_beta=2., d_half=.06, rho_min=.875, rho_max=.96)
        tracker.update(tracker.transformed_rewards([1., 1.]), divergence=0.,
                       policy_adapter=str(adapter), step=-1)
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
    assert trainer.calls[0][0] == mode
    examples = trainer.calls[0][1]
    assert len(examples) == expected_total
    assert math.isclose(sum(ex["sample_weight"] for ex in examples), 1.)
    assert all(not any(key.startswith(("rank_", "x_grpo_", "binary_")) for key in ex)
               for ex in examples)
    metadata = [json.loads(p.read_text()) for p in (tmp_path / "step00").glob("*rollout*.meta.json")]
    assert len(metadata) == expected_total
    assert {row["advantage_mode"] for row in metadata} == {mode}
    assert stats["planned_rollouts"] == expected_total
    if mode == "spo-rs":
        assert stats["spo_rs_policy_updated"]
        for row in metadata:
            expected = (math.exp(2 * row["reward"]) / math.exp(2) - 1) / 2
            assert row["advantage"] == pytest.approx(expected)
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
    if parent_selection == "thompson":
        assert sampler._outcomes.attempts == expected_total
        assert sampler._outcomes.valid == sum(bool(row["valid"]) for row in metadata)
        for info in sampler.last_picks_info:
            parent_node = sampler._nodes[info["thompson"]["path"][-1]]
            assert parent_node.direct.attempts == cfg.group_size
        if mode == "spo-rs":
            # Recreate seeds with new UUIDs, as happens after a step-zero crash.
            # Recovery must bind their saved identities, rebuild every outcome,
            # and advance the RNG exactly as the ordinary step did.
            recovered = make_sampler(
                parent_selection="thompson", num_seeds=2, group_size=cfg.group_size,
                seed=cfg.seed, seed_states=[SeedState(value=1., raw_score=1.) for _ in range(2)])
            recovered_tracker = SPORSTracker(
                entropic_beta=2., d_half=.06, rho_min=.875, rho_max=.96)
            recovered_tracker.load_state_dict(initial_tracker)
            artifacts = training._recoverable_spo_rs_artifacts(tmp_path, 0, cfg)
            assert artifacts is not None
            training._prepare_recoverable_response_tokens(None, TinyTokenizer(), artifacts)
            cfg.strategies = allocation is not None
            training._resume_spo_rs_step_from_artifacts(
                backend, model, TinyTokenizer(), recovered, optimizer, 0, cfg, tmp_path,
                problem, CapturingTrainer(), recovered_tracker, str(adapter), artifacts)
            assert recovered.state_dict()["thompson"] == sampler.state_dict()["thompson"]
            assert [s.id for s in recovered.sample_states(2, selection_step=1)] == [
                s.id for s in sampler.sample_states(2, selection_step=1)]
    if parent_selection == "llm" and mode == "spo-rs":
        recovered = make_sampler(
            parent_selection="llm", num_seeds=2, exp_dir=tmp_path, problem=problem,
            seed_states=[SeedState(value=1., raw_score=1.) for _ in range(2)])
        monkeypatch.setattr(recovered, "_request", lambda *args: pytest.fail("repeated API call"))
        recovered_tracker = SPORSTracker(
            entropic_beta=2., d_half=.06, rho_min=.875, rho_max=.96)
        recovered_tracker.load_state_dict(initial_tracker)
        artifacts = training._recoverable_spo_rs_artifacts(tmp_path, 0, cfg)
        assert artifacts is not None
        training._prepare_recoverable_response_tokens(None, TinyTokenizer(), artifacts)
        cfg.strategies = allocation is not None
        training._resume_spo_rs_step_from_artifacts(
            backend, model, TinyTokenizer(), recovered, optimizer, 0, cfg, tmp_path,
            problem, CapturingTrainer(), recovered_tracker, str(adapter), artifacts)
        assert recovered.state_dict() == sampler.state_dict()


@pytest.mark.parametrize("mode", ["entropic", "spo-rs"])
@pytest.mark.parametrize("allocation", [None, "hurdle"])
def test_thompson_full_step_and_spo_recovery(
        monkeypatch, tmp_path, load_config, mode, allocation):
    test_complete_step_with_cpu_generation_and_evaluation(
        monkeypatch, tmp_path, load_config, mode, allocation,
        remote_strategy=True, pooled_coder=True, parent_selection="thompson")


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

    monkeypatch.setattr(training, "compute_batched_token_logprobs", scores)
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


def test_overlap_helpers_use_the_same_retained_advantages():
    rewards = [0., 1., 2., 3.]
    futures = [Future() for _ in rewards]
    records = [dict(job_idx=0, token_ids=[3, 4], behavior_logprobs=[-.2, -.3])
               for _ in rewards]
    jobs = [{"prompt_text": "CODE_REQUEST"}]
    staged = training._prepare_spo_rs_overlap_inputs(
        TinyTokenizer(), {0: records}, {0: futures}, jobs)
    tracker = SPORSTracker(entropic_beta=2., d_half=.06, rho_min=.875, rho_max=.96)
    tracker.update(tracker.transformed_rewards([1.]), divergence=0.,
                   policy_adapter="unused-test-adapter", step=-1)
    with pytest.raises(RuntimeError, match="before its reward completed"):
        training._complete_spo_rs_overlap_examples(staged, tracker)
    for future, reward in zip(futures, rewards):
        future.set_result(SimpleNamespace(reward=reward))
    completed = training._complete_spo_rs_overlap_examples(staged, tracker)
    expected = tracker.normalized_advantages(tracker.transformed_rewards(rewards))
    np.testing.assert_allclose([ex["advantage"] for ex in completed], expected)
    assert sum(ex["sample_weight"] for ex in completed) == 1.
    assert all("_overlap_reward_future" not in ex for ex in completed)
    entropic = training._prepare_entropic_overlap_group(
        TinyTokenizer(), 0, records, futures, jobs, {}, 2)
    expected, _ = entropic_adaptive_advantages(np.asarray(rewards))
    np.testing.assert_allclose([ex["advantage"] for ex in entropic], expected)
    assert sum(ex["sample_weight"] for ex in entropic) == .5
