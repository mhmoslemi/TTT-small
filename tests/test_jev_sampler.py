"""Semantic parent selection, durable replay, and transient-worker contracts."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import sys

import pytest
import yaml

from experiment_io import restart_changed_rollout_step, save_parent_selections
from search import jev_sampler as jev
from search.jev_worker import GlideWorker
from search.sampler import PUCTSampler, State, make_sampler


def problem(description="Find a construction minimizing the maximum overlap."):
    return SimpleNamespace(build_prompt=lambda parent: [
        {"role": "system", "content": "Mathematical construction problem."},
        {"role": "user", "content": description}])


def response(same=.01, variant=.01):
    return {"answers": {"relationship": {"probabilities": {
        "same_core_idea": same, "meaningful_variant": variant,
        "different_approach": 1 - same - variant}}}, "usage": {"input_tokens": 123}}


def sampler(tmp_path, codes=("A", "B", "C", "D", "E"), *, candidate_multiplier=3,
            same_core_threshold=0.68):
    result = make_sampler(parent_selection="puct-jev", num_seeds=0,
                          exp_dir=tmp_path, problem=problem(), puct_c=0.,
                          jev_candidate_multiplier=candidate_multiplier,
                          jev_same_core_threshold=same_core_threshold)
    result._states = [State.make(0, 20. - i, code) for i, code in enumerate(codes)]
    return result


def fake_judge(monkeypatch, decide):
    requests, lifecycle = [], []

    class Judge:
        def __init__(self, model_path, gpu_id, log_path, *, before_start, after_stop):
            self.before, self.after = before_start, after_stop

        def __enter__(self):
            if self.before:
                self.before()
            lifecycle.append("load")
            return self

        def compare(self, request):
            requests.append(request)
            return decide(request)

        def __exit__(self, *args):
            lifecycle.append("unload")
            if self.after:
                self.after()

    monkeypatch.setattr(jev, "GlideWorker", Judge)
    return requests, lifecycle


def test_step_zero_is_exactly_normal_puct_and_replays_new_seed_ids(tmp_path, monkeypatch):
    monkeypatch.setattr(jev, "GlideWorker", lambda *a, **kw: pytest.fail("step 0 loaded GLiDE"))
    baseline = PUCTSampler(4)
    expected = baseline.sample_states(3)
    actual = make_sampler(parent_selection="puct-jev", num_seeds=0,
                          exp_dir=tmp_path, problem=problem())
    actual.load_state_dict(dict(baseline.state_dict(), selection_mode="puct-jev"))
    assert actual.sample_states(3, selection_step=0) == expected
    for ordinary, gated in zip(baseline.last_picks_info, actual.last_picks_info):
        assert {key: value for key, value in gated.items() if key != "jev"} == ordinary
    fresh = make_sampler(parent_selection="puct-jev", num_seeds=4,
                         exp_dir=tmp_path, problem=problem())
    assert [s.id for s in fresh.sample_states(3, selection_step=0)] == [s.id for s in expected]
    assert fresh._n == {} and fresh._T == 0


def test_problem_and_two_complete_proposals_are_in_request():
    a, b = "def run():\n    return 'A'\n", "def run():\n    return 'B'\n"
    request = jev.comparison_request("The actual AC1 description.", a, b)
    assert "The actual AC1 description." in request["state"]
    assert a in request["state"] and b in request["state"]
    assert "Program A" in request["state"] and "Program B" in request["state"]
    question = request["questions"]["relationship"]
    assert "two proposed solutions" in question["instructions"]
    assert set(question["criteria"]) == {
        "same_core_idea", "meaningful_variant", "different_approach"}


def test_skips_duplicates_checks_every_accepted_parent_and_preserves_puct(tmp_path, monkeypatch):
    actual = sampler(tmp_path)
    ids = {s.code: s.id for s in actual._states}
    before = copy.deepcopy(actual.state_dict())

    def decide(req):
        text = req["state"]
        if "```python\nB\n```" in text or ("```python\nD\n```" in text and "```python\nC\n```" in text):
            return response(.95, .03)
        if "```python\nC\n```" in text:
            return response(.02, .95)
        return response(.67, .05)  # Uncertain same-core judgments remain eligible.

    requests, lifecycle = fake_judge(monkeypatch, decide)
    picked = actual.sample_states(3, selection_step=1)
    assert [s.code for s in picked] == ["A", "C", "E"]
    assert len(requests) == 6 and lifecycle == ["load", "unload"]
    assert all("maximum overlap" in r["state"] for r in requests)
    assert actual.state_dict() == before  # No rejected candidate accrues visits.
    saved = json.loads((tmp_path / "step01/step01.jev.json").read_text())
    d_matches = [r for r in saved["comparisons"] if r["candidate_id"] == ids["D"]]
    assert [r["parent_id"] for r in d_matches] == [ids["A"], ids["C"]]
    path = save_parent_selections(tmp_path, 1, type(actual).__name__, picked, actual.last_picks_info)
    assert all("jev" in r for r in json.loads(path.read_text())["parents"])


@pytest.mark.parametrize("threshold, expected", [
    (.79, ["A", "C"]), (.80, ["A", "C"]), (.90, ["A", "B"]),
])
def test_configured_threshold_controls_gate(tmp_path, monkeypatch, threshold, expected):
    actual = sampler(tmp_path, same_core_threshold=threshold)
    fake_judge(monkeypatch, lambda req:
        response(.80, .1) if "```python\nB\n```" in req["state"] else response())
    assert [s.code for s in actual.sample_states(2, selection_step=1)] == expected
    saved = json.loads((tmp_path / "step01/step01.jev.json").read_text())
    assert saved["threshold"] == threshold


@pytest.mark.parametrize("status", ["complete", "partial", "empty", "generated"])
def test_resume_threshold_overrides_cached_selection(tmp_path, monkeypatch, status):
    actual = sampler(tmp_path, same_core_threshold=.90)
    payload = copy.deepcopy(actual.state_dict())
    decide = lambda req: response(.70, .1) if "```python\nB\n```" in req["state"] else response()
    fake_judge(monkeypatch, decide)
    assert [s.code for s in actual.sample_states(2, selection_step=1)] == ["A", "B"]
    path = tmp_path / "step01/step01.jev.json"
    if status in ("partial", "empty"):
        saved = json.loads(path.read_text())
        saved["selected"] = None
        if status == "empty":
            saved["comparisons"], saved["sources"] = [], {}
        path.write_text(json.dumps(saved))
    if status == "generated":
        cfg = SimpleNamespace(parent_selection="puct-jev", jev_same_core_threshold=.68,
                              groups_per_step=2, group_size=8, pilot_programs_per_strategy=-1)
        (tmp_path / "config.json").write_text(json.dumps(vars(cfg)))
        save_parent_selections(tmp_path, 1, type(actual).__name__, actual._selected,
                               actual.last_picks_info, rollout_config=cfg)
        rollout = path.parent / "step01_group00_rollout000.meta.json"
        rollout.write_text('{"reward": 1}')
        backup = restart_changed_rollout_step(cfg, tmp_path, 1)
        assert (backup / "step01" / rollout.name).is_file()
        assert not rollout.exists()
        assert not (path.parent / "step01.parents.json").exists()
        assert restart_changed_rollout_step(cfg, tmp_path, 1) is None
    resumed = sampler(tmp_path, same_core_threshold=.68)
    resumed.load_state_dict(payload)
    requests, _ = fake_judge(monkeypatch, decide)
    expected = ["A", "C"]
    assert [s.code for s in resumed.sample_states(2, selection_step=1)] == expected
    assert len(requests) == (2 if status == "empty" else 1)
    if status != "empty":
        assert "```python\nC\n```" in requests[0]["state"]  # B's judgment was reused.
    assert json.loads(path.read_text())["threshold"] == .68
    resumed.load_state_dict(payload)
    monkeypatch.setattr(jev, "GlideWorker", lambda *a, **kw: pytest.fail("replayed GLiDE"))
    assert [s.code for s in resumed.sample_states(2, selection_step=1)] == expected
    fake_judge(monkeypatch, decide)
    assert [s.code for s in resumed.sample_states(2, selection_step=2)] == ["A", "C"]
    assert json.loads((tmp_path / "step02/step02.jev.json").read_text())["threshold"] == .68


@pytest.mark.parametrize("value", [-.1, 1.1, True, "0.8", None, float("nan"), float("inf")])
def test_threshold_requires_a_probability(tmp_path, value):
    from config_validation import validate_problem_config
    with pytest.raises(ValueError, match="jev_same_core_threshold"):
        sampler(tmp_path, same_core_threshold=value)
    with pytest.raises(ValueError, match="jev_same_core_threshold"):
        validate_problem_config({"problem": "erdos", "jev_same_core_threshold": value},
                                require_complete=False)


def test_saved_selection_and_partial_comparisons_replay_without_model(tmp_path, monkeypatch):
    actual = sampler(tmp_path)
    payload = copy.deepcopy(actual.state_dict())
    requests, lifecycle = fake_judge(monkeypatch, lambda req: response())
    picked = actual.sample_states(3, selection_step=1)
    expected_info = copy.deepcopy(actual.last_picks_info)
    assert requests
    monkeypatch.setattr(jev, "GlideWorker", lambda *a, **kw: pytest.fail("replayed GLiDE"))
    # The same step remains stable even after expansion counts begin changing.
    actual.record_expansion(picked[0], 64)
    assert actual.sample_states(3, selection_step=1) == picked
    actual.load_state_dict(payload)
    assert actual.sample_states(3, selection_step=1) == picked
    assert actual.last_picks_info == expected_info
    # A crash after the last comparison but before committing the parent list.
    path = tmp_path / "step01/step01.jev.json"
    saved = json.loads(path.read_text())
    saved["selected"] = None
    path.write_text(json.dumps(saved))
    actual.load_state_dict(payload)
    assert actual.sample_states(3, selection_step=1) == picked


def test_failure_unloads_worker_and_reuses_completed_comparisons(tmp_path, monkeypatch):
    actual = sampler(tmp_path)
    seen = []

    def fail_second(req):
        seen.append(req)
        if len(seen) == 2:
            raise RuntimeError("GPU failure")
        return response()

    requests, lifecycle = fake_judge(monkeypatch, fail_second)
    with pytest.raises(RuntimeError, match="GPU failure"):
        actual.sample_states(3, selection_step=1)
    assert lifecycle == ["load", "unload"]
    saved = json.loads((tmp_path / "step01/step01.jev.json").read_text())
    assert saved["selected"] is None and len(saved["comparisons"]) == 1
    resumed_requests, _ = fake_judge(monkeypatch, lambda req: response())
    assert [s.code for s in actual.sample_states(3, selection_step=1)] == ["A", "B", "C"]
    assert len(resumed_requests) == 2


def test_lineage_relaxes_before_duplicate_fallback_and_batch_always_fills(tmp_path, monkeypatch):
    actual = sampler(tmp_path, ("A", "B", "C", "D"))
    a, b, c, d = actual._states
    b.parents = [{"id": a.id, "timestep": 0}]
    fake_judge(monkeypatch, lambda req: response(.98, .01) if "```python\nC\n```" in req["state"] else response())
    assert [s.code for s in actual.sample_states(3, selection_step=1)] == ["A", "D", "B"]
    assert actual.last_picks_info[2]["jev"]["lineage_relaxed"] is True
    actual = sampler(tmp_path / "all_same", ("A", "B", "C"))
    fake_judge(monkeypatch, lambda req: response(.98, .01))
    assert [s.code for s in actual.sample_states(3, selection_step=1)] == ["A", "B", "C"]
    assert [i["jev"]["decision"] for i in actual.last_picks_info] == [
        "no_comparison", "duplicate_fallback", "duplicate_fallback"]


def test_no_judge_for_one_parent_or_empty_sources(tmp_path, monkeypatch):
    monkeypatch.setattr(jev, "GlideWorker", lambda *a, **kw: pytest.fail("unnecessary GLiDE load"))
    actual = sampler(tmp_path / "one")
    assert len(actual.sample_states(1, selection_step=1)) == 1
    actual = sampler(tmp_path / "empty", ("", "", ""))
    assert len(actual.sample_states(3, selection_step=1)) == 3


@pytest.mark.parametrize("multiplier", [1, 2, 3])
def test_candidate_window_caps_scan_and_fills_without_searching_later_ideas(tmp_path, monkeypatch, multiplier):
    codes = tuple("ABCDEFGH")
    actual = sampler(tmp_path, codes, candidate_multiplier=multiplier)
    before = copy.deepcopy(actual.state_dict())
    requests, _ = fake_judge(monkeypatch, lambda req:
        response() if "```python\nH\n```" in req["state"] else response(.98, .01))
    picks = actual.sample_states(2, selection_step=1)
    assert [s.code for s in picks] == ["A", "B"]
    assert len(requests) == 2 * multiplier - 1
    assert actual.state_dict() == before
    saved = json.loads((tmp_path / "step01/step01.jev.json").read_text())
    assert saved["candidate_multiplier"] == multiplier
    assert saved["candidate_limit"] == 2 * multiplier
    assert set(saved["sources"].values()) == set(codes[:2 * multiplier])
    assert actual.last_picks_info[-1]["jev"]["decision"] == "duplicate_fallback"


def test_lineage_blocked_nodes_count_toward_candidate_window(tmp_path, monkeypatch):
    actual = sampler(tmp_path, candidate_multiplier=2)
    a = actual._states[0]
    for child in actual._states[1:4]:
        child.parents = [{"id": a.id, "timestep": 0}]
    requests, _ = fake_judge(monkeypatch, lambda req: response(.98, .01))
    assert [s.code for s in actual.sample_states(2, selection_step=1)] == ["A", "B"]
    assert len(requests) == 3
    assert all("```python\nE\n```" not in req["state"] for req in requests)


def test_candidate_window_stops_early_and_does_not_cap_pairwise_questions(tmp_path, monkeypatch):
    actual = sampler(tmp_path, tuple(f"program-{i}" for i in range(20)), candidate_multiplier=1)
    requests, _ = fake_judge(monkeypatch, lambda req: response())
    assert [s.code for s in actual.sample_states(4, selection_step=1)] == [
        "program-0", "program-1", "program-2", "program-3"]
    assert len(requests) == 6  # Four candidates require all six pairwise judgments.
    saved = json.loads((tmp_path / "step01/step01.jev.json").read_text())
    assert saved["candidate_limit"] == len(saved["decisions"]) == 4
    actual = sampler(tmp_path / "early", tuple(f"program-{i}" for i in range(20)))
    requests, _ = fake_judge(monkeypatch, lambda req: response())
    assert len(actual.sample_states(2, selection_step=1)) == 2
    assert len(requests) == 1


def test_partial_selection_freezes_multiplier_and_next_step_uses_current_yaml(tmp_path, monkeypatch):
    codes = tuple("ABCDEFGH")
    actual = sampler(tmp_path, codes, candidate_multiplier=2)
    payload = copy.deepcopy(actual.state_dict())
    calls = []

    def fail_second(request):
        calls.append(request)
        if len(calls) == 2:
            raise RuntimeError("interrupted")
        return response(.98, .01)

    fake_judge(monkeypatch, fail_second)
    with pytest.raises(RuntimeError, match="interrupted"):
        actual.sample_states(2, selection_step=1)
    resumed = sampler(tmp_path, codes, candidate_multiplier=1)
    resumed.load_state_dict(payload)
    requests, _ = fake_judge(monkeypatch, lambda req: response(.98, .01))
    resumed.sample_states(2, selection_step=1)
    assert len(requests) == 2  # B is cached; C and D finish the saved window.
    saved = json.loads((tmp_path / "step01/step01.jev.json").read_text())
    assert saved["candidate_multiplier"] == 2 and saved["candidate_limit"] == 4
    requests, _ = fake_judge(monkeypatch, lambda req: response(.98, .01))
    resumed.sample_states(2, selection_step=2)
    assert len(requests) == 1
    saved = json.loads((tmp_path / "step02/step02.jev.json").read_text())
    assert saved["candidate_multiplier"] == 1 and saved["candidate_limit"] == 2


def test_legacy_selected_parents_replay_even_outside_new_window(tmp_path, monkeypatch):
    actual = sampler(tmp_path, tuple("ABCDEFGH"), candidate_multiplier=4)
    payload = copy.deepcopy(actual.state_dict())
    fake_judge(monkeypatch, lambda req:
        response() if "```python\nH\n```" in req["state"] else response(.98, .01))
    assert [s.code for s in actual.sample_states(2, selection_step=1)] == ["A", "H"]
    path = tmp_path / "step01/step01.jev.json"
    saved = json.loads(path.read_text())
    del saved["candidate_multiplier"], saved["candidate_limit"]
    path.write_text(json.dumps(saved))
    actual.candidate_multiplier = 1
    actual.load_state_dict(payload)
    monkeypatch.setattr(jev, "GlideWorker", lambda *a, **kw: pytest.fail("replayed GLiDE"))
    assert [s.code for s in actual.sample_states(2, selection_step=1)] == ["A", "H"]


@pytest.mark.parametrize("value", [0, -1, True, False, 1.5, "3", None])
def test_candidate_multiplier_requires_a_positive_integer(tmp_path, value):
    from config_validation import validate_problem_config
    with pytest.raises(ValueError, match="jev_candidate_multiplier"):
        sampler(tmp_path, candidate_multiplier=value)
    with pytest.raises(ValueError, match="jev_candidate_multiplier"):
        validate_problem_config({"problem": "erdos", "jev_candidate_multiplier": value},
                                require_complete=False)


def test_replay_rejects_changed_problem_count_or_archive(tmp_path, monkeypatch):
    actual = sampler(tmp_path)
    payload = actual.state_dict()
    fake_judge(monkeypatch, lambda req: response())
    actual.sample_states(2, selection_step=1)
    with pytest.raises(ValueError, match="parent count"):
        actual.sample_states(3, selection_step=1)
    actual.load_state_dict(payload)
    actual.problem = problem("A different task")
    with pytest.raises(ValueError, match="differs"):
        actual.sample_states(2, selection_step=1)
    actual.problem = problem()
    actual.record_expansion(actual._states[0])
    with pytest.raises(ValueError, match="differs"):
        actual.sample_states(2, selection_step=1)


@pytest.mark.parametrize("values", [
    {"same_core_idea": float("nan"), "meaningful_variant": 0., "different_approach": 1.},
    {"same_core_idea": .8, "meaningful_variant": .8, "different_approach": .1},
    {"same_core_idea": True, "meaningful_variant": 0., "different_approach": 0.},
    {"wrong_label": 1.},
])
def test_invalid_probabilities_do_not_silently_disable_gate(values):
    with pytest.raises(ValueError, match="invalid GLiDE decision"):
        jev._probabilities({"answers": {"relationship": {"probabilities": values}}})


@pytest.fixture
def worker_checkpoint(tmp_path):
    folder = tmp_path / "GLiDE-9B"
    (folder / "glide").mkdir(parents=True)
    for name in ("decision_config.json", "model.safetensors.index.json", "readout.safetensors"):
        (folder / name).write_text("{}")
    # Run the real subprocess protocol with an inexpensive fake CUDA runtime.
    (folder / "torch.py").write_text(
        "from contextlib import nullcontext\n"
        "from types import SimpleNamespace\n"
        "cuda=SimpleNamespace(is_available=lambda: True)\n"
        "bfloat16='bf16'\n"
        "inference_mode=nullcontext\n")
    (folder / "glide/__init__.py").write_text(
        "from types import SimpleNamespace\n"
        "class Glide:\n"
        " @classmethod\n"
        " def from_pretrained(cls, path, **kw):\n"
        "  assert kw==dict(device='cuda:0',dtype='bf16',vision=False,max_len=40000)\n"
        "  print('dependency output')\n"
        "  obj=cls()\n"
        "  obj.prompter=SimpleNamespace(text=lambda state,q,opts: state)\n"
        "  obj.tokenizer=lambda text,**kw: {'input_ids':list(range(len(text)))}\n"
        "  return obj\n"
        " def decide(self,**kw):\n"
        "  print('inference output')\n"
        "  return {'answers':{'relationship':{'probabilities':"
        "{'same_core_idea':0.1,'meaningful_variant':0.2,'different_approach':0.7}}}}\n")
    (folder / "glide/prompt.py").write_text("def options_for(q): return list(q['criteria'].items())\n")
    return folder


def test_real_worker_protocol_loads_once_and_exits_before_gpu_restore(tmp_path, worker_checkpoint):
    events = []
    log = tmp_path / "worker.log"
    with GlideWorker(worker_checkpoint, 3, log,
                     before_start=lambda: events.append("park"),
                     after_stop=lambda: events.append("released")) as worker:
        proc = worker.process
        for _ in range(2):
            result = worker.compare(jev.comparison_request("task", "A", "B"))
            assert jev._probabilities(result)["meaningful_variant"] == .2
        assert proc.poll() is None
        assert events == ["park"]
    assert proc.poll() is not None and worker.process is None
    assert events == ["park", "released"]
    assert "dependency output" in log.read_text() and "inference output" in log.read_text()


def test_oversized_worker_request_is_rejected_intact_and_worker_exits(tmp_path, worker_checkpoint):
    events = []
    with pytest.raises(RuntimeError, match="Both programs were kept intact"):
        with GlideWorker(worker_checkpoint, 0, tmp_path / "worker.log",
                         after_stop=lambda: events.append("released")) as worker:
            proc = worker.process
            worker.compare(jev.comparison_request("task", "A" * 40000, "B"))
    assert events == ["released"] and proc.poll() is not None


def test_missing_checkpoint_does_not_park_the_trainer(tmp_path):
    before = Mock()
    with pytest.raises(FileNotFoundError, match="complete local"):
        with GlideWorker(tmp_path / "missing", 0, tmp_path / "log", before_start=before):
            pytest.fail("worker unexpectedly loaded")
    before.assert_not_called()


def test_migration_preserves_archive_and_defers_already_selected_step(tmp_path, monkeypatch):
    import train_multy_CVaR as training
    old = PUCTSampler(2)
    old.record_expansion(old._states[0], 64)
    cfg = SimpleNamespace(parent_selection="puct-jev", jev_model_path="GLiDE-9B",
                          jev_candidate_multiplier=5, jev_same_core_threshold=.83)
    (tmp_path / "config.json").write_text('{"parent_selection":"puct"}')
    checkpoint = Mock()
    monkeypatch.setattr(training, "_save_training_checkpoint", checkpoint)
    args = (old, cfg, tmp_path, 2, tmp_path / "adapter", object(), object())
    (tmp_path / "step02").mkdir()
    selected = tmp_path / "step02/step02.parents.json"
    selected.write_text("{}")
    assert training._migrate_parent_selection(*args, recovery_pending=False, problem=problem()) is old
    checkpoint.assert_not_called()
    selected.unlink()
    migrated = training._migrate_parent_selection(*args, recovery_pending=False, problem=problem())
    assert migrated.state_dict() == dict(old.state_dict(), selection_mode="puct-jev")
    assert migrated.candidate_multiplier == 5
    assert migrated.same_core_threshold == .83
    assert checkpoint.call_args.args[3] is migrated
    assert json.loads((tmp_path / "config.json").read_text())["parent_selection"] == "puct-jev"


@pytest.mark.parametrize("profile", ["RTX6000", "B200", "B200_full_ram"])
def test_vllm_handoff_preserves_full_ram_pools_and_never_restores_early(tmp_path, profile):
    import train_multy_CVaR as training
    from generation.gen_workers import PhasedVLLMGenerationPool
    events = []
    pools = [Mock(spec=PhasedVLLMGenerationPool), Mock(spec=PhasedVLLMGenerationPool)]
    for name, pool in zip(("coder", "strategy"), pools):
        pool.shutdown.side_effect = lambda name=name: events.append(name)
        pool.release.side_effect = lambda name=name: events.append(f"sleep {name}")
    actual = sampler(tmp_path)
    cfg = SimpleNamespace(gpu_ids="6,7", generation_backend="vllm", hardware_profile=profile)
    training._configure_jev_sampler(actual, cfg, Mock(), Mock(), Mock(),
                                   *pools, lambda phase: events.append(phase))
    actual.before_judge()
    expected = (["sleep strategy", "sleep coder", "jev"] if profile == "B200_full_ram"
                else ["strategy", "coder", "jev"])
    assert events == expected and actual.gpu_id == 6
    actual.after_judge()
    assert events == expected
    if profile == "B200_full_ram":
        for pool in pools:
            pool.shutdown.assert_not_called()


@pytest.mark.parametrize("name", ["ac1", "ac2", "erdos"])
def test_real_problem_descriptions_are_in_comparison_prompt(tmp_path, name):
    from problems import get_problem
    from problems.base import ParentContext

    cfg = yaml.safe_load((Path(__file__).resolve().parents[1] / f"configs/{name}.yaml").read_text())
    task = get_problem(name, cfg)
    actual = make_sampler(parent_selection="puct-jev", num_seeds=0,
                          exp_dir=tmp_path, problem=task)
    description = actual._description()
    assert description and task.entrypoint in description
    for message in task.build_prompt(ParentContext()):
        assert message["content"] in description
    request = jev.comparison_request(description, "def program_a(): pass", "def program_b(): pass")
    assert description in request["state"]


def test_config_resolves_sibling_checkpoint_and_cli_override(tmp_path, monkeypatch):
    import train_multy_CVaR as training
    import runtime.gpu_runtime as gpu

    monkeypatch.setenv("AVAILABLE_GPUS", "0,1")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setattr(gpu, "query_gpu_memory", lambda: {})
    monkeypatch.setattr(gpu, "detect_attention_heads", lambda name: 32)
    root = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load((root / "configs/erdos.yaml").read_text())
    cfg["hardware_profile"] = "RTX6000"
    cfg["jev_candidate_multiplier"] = 5
    cfg["jev_same_core_threshold"] = .83
    cfg["model_name"] = cfg["coder_model_name"] = str(tmp_path / "Qwen3.8-27B")
    (tmp_path / "Qwen3.8-27B").mkdir()
    sibling = tmp_path / "GLiDE-9B"
    sibling.mkdir()
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump(cfg))
    argv = ["train_multy_CVaR.py", "--config", str(config), "--backend", "unsloth",
            "--generation-backend", "hf", "--max-seq-length", "32768",
            "--max-new-tokens", "4096", "--parent-selection", "puct-jev"]
    monkeypatch.setattr(sys, "argv", argv)
    resolved, _ = training.load_config()
    assert resolved.parent_selection == "puct-jev"
    assert resolved.jev_model_path == str(sibling)
    assert resolved.jev_candidate_multiplier == 5
    assert resolved.jev_same_core_threshold == .83
    monkeypatch.setattr(sys, "argv", argv + ["--jev-model-path", str(tmp_path / "explicit")])
    resolved, _ = training.load_config()
    assert resolved.jev_model_path == str(tmp_path / "explicit")


def test_hf_handoff_restores_the_live_coder_after_worker_exit(tmp_path, monkeypatch):
    import train_multy_CVaR as training
    events = []
    actual = sampler(tmp_path)
    backend, optimizer = Mock(), object()
    backend.offload_for_generation.side_effect = lambda **kw: events.append(("park", kw))
    backend.restore_after_generation.side_effect = lambda: events.append("restore")
    monkeypatch.setattr(training, "_move_optimizer_state", lambda opt, device, **kw: events.append(device))
    monkeypatch.setattr(training, "_restore_optimizer_state_to_parameters", lambda opt: events.append("optimizer"))
    cfg = SimpleNamespace(gpu_ids="2", generation_backend="hf", trainer_offload_dir="disk")
    training._configure_jev_sampler(actual, cfg, backend, optimizer, None, None, None, None)
    fake_judge(monkeypatch, lambda req: response())
    actual.sample_states(2, selection_step=1)
    assert events == ["cpu", ("park", {"disk_directory": "disk"}), "restore", "optimizer"]
    backend.set_training_mode.assert_called_once()
