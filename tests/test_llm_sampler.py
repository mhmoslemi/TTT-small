"""Mocked API and artifact replay checks for LLM parent selection."""

from copy import deepcopy
from concurrent.futures import Future
import json
import sys
from threading import Event, Thread
from types import SimpleNamespace

import pytest

from experiment_io import save_parent_selections
from llm_sampler import LLMSampler
from sampler import make_sampler


def problem(maximize=False):
    return SimpleNamespace(name="test-science", metric_name="bound", maximize=maximize,
                           cfg={"target_print": .25}, target=.3)


def response(ids, *, status="completed"):
    result = {
        "selected_parent_ids": ids,
        "selections": [{"node_id": sid, "role": "exploit", "reason": f"Evidence for {sid}"}
                       for sid in ids],
        "tree_diagnosis": "Revisit productive ancestors and cover other promising forks.",
        "confidence": "medium"}
    return {
        "id": "resp-local-test", "status": status,
        "output": [{"type": "reasoning"}, {"type": "message", "content": [
            {"type": "output_text", "text": json.dumps(result)}]}],
        "usage": {"input_tokens": 300, "output_tokens": 50,
                  "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 200},
                  "output_tokens_details": {"reasoning_tokens": 25}}}


@pytest.fixture
def api(monkeypatch):
    calls = {"uploads": [], "requests": [], "deletions": [], "clients": []}
    monkeypatch.setenv("OPENAI_API_KEY", "private-test-key")
    def upload(**kwargs):
        name, file, mime = kwargs["file"]
        calls["uploads"].append((name, mime, json.loads(file.read())))
        return SimpleNamespace(id="file-local-test")
    def create(**kwargs):
        calls["requests"].append(deepcopy(kwargs))
        value = calls["reply"]
        return SimpleNamespace(model_dump=lambda **unused: deepcopy(value))
    def client(**kwargs):
        calls["clients"].append(kwargs)
        return SimpleNamespace(
            files=SimpleNamespace(create=upload, delete=calls["deletions"].append),
            responses=SimpleNamespace(create=create), close=lambda: None)
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=client))
    return calls


def history(tmp_path, *, maximize=False):
    sampler = LLMSampler(num_seeds=1, exp_dir=tmp_path, problem=problem(maximize),
                         topk_children=1, seed_value=1.)
    root = sampler._states[0]
    root.raw_score = 1.
    save_parent_selections(tmp_path, 0, "PUCTSampler", [root], [])
    folder = tmp_path / "step00"
    for index, (sid, raw, valid) in enumerate([
            ("champion", .4, True), ("pruned", .6, True), ("failed", 0., False)]):
        meta = dict(step=0, group=0, parent_group=0, rollout=index, node_id=sid,
                    parent_id=root.id, parent_is_seed=True, parent_value=root.value,
                    parent_raw_score=root.raw_score, raw_score=raw, reward=1. / raw if valid else 0.,
                    valid=valid, construction="[0.1, 0.9]", archive_eligible=valid)
        path = folder / f"step00_group00_rollout{index:03d}.meta.json"
        path.write_text(json.dumps(meta))
        path.with_name(path.name.removesuffix(".meta.json") + ".txt").write_text(
            f"<think>draft</think>\n```python\n# {sid}\ndef run(): return [0.1, 0.9]\n```")
    (folder / "step00_group00_rollout003.meta.json").write_text(json.dumps({
        "step": 0, "group": 0, "rollout": 3, "status": "evaluation_pending"}))
    # Deliberately leave the stale public tree behind and a partially started next step.
    (tmp_path / "search_tree.json").write_text('{"stale": true}')
    save_parent_selections(tmp_path, 1, "PUCTSampler", [root], [])
    return sampler, root


def test_full_tree_selects_pruned_nodes_once_and_saves_exact_request(tmp_path, api):
    sampler, root = history(tmp_path)
    api["reply"] = response(["champion", "pruned"])
    picks = sampler.sample_states(2, selection_step=1)
    assert [state.id for state in picks] == ["champion", "pruned"]
    assert all(state.construction == [.1, .9] for state in picks)
    assert all(state.parents == [{"id": root.id, "timestep": 0}] for state in picks)
    assert "# pruned" in picks[1].code
    assert [state.id for state in sampler._states] == [root.id]
    assert len(api["requests"]) == len(api["uploads"]) == 1
    assert api["deletions"] == ["file-local-test"]
    request = api["requests"][0]
    assert request["model"] == "gpt-6-luna" and request["reasoning"] == {"effort": "max"}
    assert request["max_output_tokens"] == 128000 and request["store"] is False
    assert request["text"]["format"]["schema"]["properties"]["selected_parent_ids"]["minItems"] == 2
    assert not {"temperature", "top_p", "previous_response_id"} & request.keys()
    prompt = request["input"][0]["content"][0]["text"]
    assert "MINIMIZE" in prompt and "Number of parents to select: 2" in prompt
    assert "\"champion\"" in prompt and "Depth is not a generation timestamp" in prompt
    snapshot = api["uploads"][0][2]
    assert snapshot["updated_through_step"] == 0
    assert set(snapshot["nodes"]) == {root.id, "champion", "pruned", "failed"}
    assert json.loads((tmp_path / "search_tree.json").read_text()) == {"stale": True}
    assert api["clients"][0]["max_retries"] == 0
    for suffix in ("prompt.txt", "request.json", "tree.json", "response.json"):
        text = (tmp_path / "step01" / f"step01.sampler.{suffix}").read_text()
        assert "private-test-key" not in text
    assert (tmp_path / "step01/step01.sampler.prompt.txt").read_text() == prompt
    assert sampler.sample_states(2, selection_step=1) == picks
    restored = LLMSampler(num_seeds=0, exp_dir=tmp_path, problem=problem())
    restored.load_state_dict(sampler.state_dict())
    assert restored.sample_states(2, selection_step=1) == picks
    assert len(api["requests"]) == 1
    event = save_parent_selections(tmp_path, 1, "LLMSampler", picks, sampler.last_picks_info)
    assert json.loads(event.read_text())["parents"][1]["llm"]["rank"] == 2


def test_prefetch_overlaps_checkpoints_and_reuses_one_decision(tmp_path, api, monkeypatch):
    sampler, _ = history(tmp_path)
    api["reply"] = response(["champion", "pruned"])
    sampler.last_picks_info = [{"current_step": True}]
    before = sampler.state_dict()
    started, release = Event(), Event()
    request = sampler._request

    def slow_request(*args):
        started.set()
        assert release.wait(5), "training never released the API response"
        return request(*args)

    monkeypatch.setattr(sampler, "_request", slow_request)
    future = sampler.prefetch_states(2, selection_step=1)
    try:
        assert started.wait(5)
        assert not future.done()
        assert sampler.prefetch_states(2, selection_step=1) is future
        assert sampler.state_dict() == before
        assert not future.done(), "checkpoint waited for the API response"
        assert sampler._selection_step is None
        assert sampler.last_picks_info == [{"current_step": True}]
        with pytest.raises(ValueError, match="parent count"):
            sampler.sample_states(1, selection_step=1)

        consumed = Future()

        def next_step():
            try:
                consumed.set_result(sampler.sample_states(2, selection_step=1))
            except Exception as error:
                consumed.set_exception(error)

        Thread(target=next_step, daemon=True).start()
        release.set()
        picks = consumed.result(timeout=5)
        future.result(timeout=5)
    finally:
        release.set()
        future.result(timeout=5)

    assert [state.id for state in picks] == ["champion", "pruned"]
    assert len(api["requests"]) == 1
    assert api["uploads"][0][2]["updated_through_step"] == 0
    assert not any(key.startswith("prefetch") for key in before)
    restored = LLMSampler(num_seeds=0, exp_dir=tmp_path, problem=problem())
    restored.load_state_dict(before)
    assert restored.sample_states(2, selection_step=1) == picks
    assert len(api["requests"]) == 1


def test_prefetch_error_is_deferred_without_repeating_the_request(tmp_path, api):
    sampler, _ = history(tmp_path)
    api["reply"] = response(["pruned", "champion"])
    before = sampler.state_dict()
    future = sampler.prefetch_states(2, selection_step=1)
    with pytest.raises(ValueError, match="exact best"):
        future.result(timeout=5)
    assert sampler.state_dict() == before
    assert sampler.last_picks_info == []
    with pytest.raises(ValueError, match="exact best"):
        sampler.sample_states(2, selection_step=1)
    assert len(api["requests"]) == 1


@pytest.mark.parametrize("maximize,anchor", [(False, "champion"), (True, "seed")])
def test_metric_direction_and_exact_best_anchor(tmp_path, api, maximize, anchor):
    sampler, seed = history(tmp_path, maximize=maximize)
    chosen = "champion" if anchor == "champion" else seed.id
    api["reply"] = response([chosen])
    assert sampler.sample_states(1, selection_step=1)[0].id == chosen
    prompt = api["requests"][0]["input"][0]["content"][0]["text"]
    assert ("MAXIMIZE" if maximize else "MINIMIZE") in prompt


@pytest.mark.parametrize("ids,error", [
    (["pruned", "champion"], "exact best"),
    (["champion", "failed"], "unknown, failed"),
    (["champion", "invented"], "unknown, failed"),
    (["champion", "champion"], "unique"),
    (["champion"], "unique"),
])
def test_invalid_answer_is_saved_and_never_silently_retried(tmp_path, api, ids, error):
    sampler, _ = history(tmp_path)
    api["reply"] = response(ids)
    with pytest.raises(ValueError, match=error):
        sampler.sample_states(2, selection_step=1)
    with pytest.raises(ValueError, match=error):
        sampler.sample_states(2, selection_step=1)
    assert len(api["requests"]) == 1
    assert sampler.last_picks_info == []


def test_step_zero_resume_preserves_all_seed_ids_without_another_call(tmp_path, api):
    seeds = [SimpleNamespace(value=1., raw_score=.5, code="# one"),
             SimpleNamespace(value=.5, raw_score=1., code="# two")]
    sampler = make_sampler(parent_selection="llm", exp_dir=tmp_path, problem=problem(),
                           num_seeds=2, seed_states=seeds)
    ids = [state.id for state in sampler._states]
    api["reply"] = response(ids)
    before = sampler.sample_states(2, selection_step=0)
    resumed = LLMSampler(num_seeds=2, seed_states=seeds, exp_dir=tmp_path, problem=problem())
    after = resumed.sample_states(2, selection_step=0)
    assert after == before
    assert resumed._seed_ids == sampler._seed_ids
    assert len(api["requests"]) == 1
    with pytest.raises(ValueError, match="parent count"):
        resumed.sample_states(1, selection_step=0)


def test_unscored_seeds_can_bootstrap_but_generated_zero_cannot(tmp_path, api):
    sampler = LLMSampler(num_seeds=2, exp_dir=tmp_path, problem=problem())
    ids = [state.id for state in sampler._states]
    api["reply"] = response(ids)
    assert [state.id for state in sampler.sample_states(2, selection_step=0)] == ids


@pytest.mark.parametrize("status", ["incomplete", "failed"])
def test_incomplete_response_is_not_accepted(tmp_path, api, status):
    sampler, _ = history(tmp_path)
    api["reply"] = response(["champion"], status=status)
    with pytest.raises(ValueError, match="incomplete"):
        sampler.sample_states(1, selection_step=1)
    assert api["deletions"] == ["file-local-test"]


def test_resume_checks_snapshot_integrity(tmp_path, api):
    sampler, _ = history(tmp_path)
    api["reply"] = response(["champion"])
    sampler.sample_states(1, selection_step=1)
    (tmp_path / "step01/step01.sampler.tree.json").write_text("{}")
    restored = LLMSampler(num_seeds=0, exp_dir=tmp_path, problem=problem())
    restored.load_state_dict(sampler.state_dict())
    with pytest.raises(ValueError, match="snapshot changed"):
        restored.sample_states(1, selection_step=1)
    assert len(api["requests"]) == 1


def test_pruned_parent_cannot_silently_lose_its_warm_start(tmp_path, api):
    sampler, _ = history(tmp_path)
    sampler.problem.saves_construction = True
    path = tmp_path / "step00/step00_group00_rollout001.meta.json"
    meta = json.loads(path.read_text())
    meta["construction"] = None
    path.write_text(json.dumps(meta))
    api["reply"] = response(["champion", "pruned"])
    with pytest.raises(ValueError, match="missing its saved construction"):
        sampler.sample_states(2, selection_step=1)


def test_missing_checkpoint_cannot_relabel_a_legacy_tree(tmp_path):
    sampler = LLMSampler(num_seeds=1, exp_dir=tmp_path, problem=problem())
    with pytest.raises(ValueError, match="training_state.pt"):
        sampler.import_legacy_states([])


@pytest.mark.parametrize("source", ["puct", "uct", "thompson"])
def test_migration_preserves_training_state_and_defers_recovery(tmp_path, monkeypatch, source):
    import train_multy_CVaR as training

    old = make_sampler(parent_selection=source, num_seeds=1, seed_value=1.)
    cfg = SimpleNamespace(parent_selection="llm")
    before = old.state_dict()
    (tmp_path / "config.json").write_text(json.dumps({"parent_selection": source}))
    saved = []
    optimizer, tracker, adapter = object(), object(), tmp_path / "adapter_step000"
    monkeypatch.setattr(training, "_save_training_checkpoint",
                        lambda *args: saved.append(args))
    args = (old, cfg, tmp_path, 1, adapter, optimizer, tracker)
    assert training._migrate_parent_selection(
        *args, recovery_pending=True, problem=problem()) is old
    assert saved == []
    migrated = training._migrate_parent_selection(
        *args, recovery_pending=False, problem=problem())
    assert migrated.selection_mode == "llm"
    for key in ("states", "seed_ids", "n", "m", "T", "best_raw_min", "best_raw_max"):
        assert migrated.state_dict()[key] == before[key]
    assert old.state_dict() == before
    assert saved[0][2] == adapter and saved[0][4:6] == (optimizer, tracker)
    assert json.loads((tmp_path / "config.json").read_text())["parent_selection"] == "llm"
