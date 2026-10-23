"""CPU checks for parent selection, outcome accounting and reproducible resume."""

from copy import deepcopy
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from config_validation import validate_problem_config
from experiment_io import save_parent_selections, save_step_summary
from sampler import PUCTSampler, State, make_sampler, parent_selection_mode
from thompson_sampler import OutcomeStats, ThompsonSampler, expected_max_gain


def search(seeds=2, **kwargs):
    return ThompsonSampler(
        num_seeds=seeds, group_size=8, seed=42,
        seed_states=[SimpleNamespace(value=1.0, code=f"seed{i}") for i in range(seeds)],
        **kwargs)


@pytest.fixture
def legacy_history(tmp_path):
    old = PUCTSampler(num_seeds=1, seed_value=1., topk_children=1)
    root = old._states[0]

    def step(index, parent, rewards, retries=0):
        save_parent_selections(tmp_path, index, "PUCTSampler", [parent], [])
        batch = []
        for i, reward in enumerate(rewards):
            child = State.make(index, reward, f"code-{index}-{i}")
            if index == 0 and i == 3:
                child.code = batch[0].code  # A duplicate is still an observed attempt.
            row = dict(step=index, group=0, parent_group=0, rollout=i,
                       node_id=child.id, parent_id=parent.id, parent_value=parent.value,
                       reward=reward, valid=reward > 0,
                       retry_attempt=int(i >= len(rewards) - retries),
                       counts_toward_allocation=i < len(rewards) - retries)
            (tmp_path / f"step{index:02d}" / f"step{index:02d}_group00_rollout{i:03d}.meta.json").write_text(json.dumps(row))
            batch.append(child)
        old.record_expansion(parent, len(rewards) - retries)
        old.update([(child, parent) for child in batch if child.value > 0])
        save_step_summary(tmp_path, index, dict(step=index, completed=True,
                          planned_rollouts=len(rewards) - retries, coder_format_retries=retries))
        return batch[0]

    pruned = step(0, root, [1.2, 1.1, 0., 1.2])
    step(1, pruned, [1.3, 0., 1.25], retries=1)
    step(2, root, [1.4])
    assert pruned.id not in {s.id for s in old._states}
    (tmp_path / "config.json").write_text(json.dumps({"model_name": "unchanged", "uct": False}))
    return old, root, pruned


@pytest.mark.parametrize("source", ["puct", "uct"])
def test_migrate_preserves_archive_and_rebuilds_all_attempts(tmp_path, legacy_history, source):
    old, root, pruned = legacy_history
    old.use_uct = source == "uct"
    original = deepcopy(old.state_dict())
    migrated = search()
    assert migrated.migrate_state_dict(original, tmp_path, 3) == 8
    assert old.state_dict() == original
    assert migrated._outcomes.valid == 6
    assert migrated._nodes[root.id].subtree.attempts == 8
    assert migrated._nodes[root.id].direct.attempts == 5
    assert migrated._nodes[pruned.id].direct.attempts == 3
    assert migrated._best_reward == 1.4
    snapshot = migrated.state_dict()
    assert {k: snapshot[k] for k in original if k != "selection_mode"} == {
        k: v for k, v in original.items() if k != "selection_mode"}
    restored = search()
    restored.load_state_dict(snapshot)
    assert [s.id for s in migrated.sample_states(3)] == [s.id for s in restored.sample_states(3)]


@pytest.mark.parametrize("damage", ["missing", "pending", "genealogy", "identity", "summary"])
def test_migration_refuses_incomplete_or_inconsistent_history(tmp_path, legacy_history, damage):
    old, _, _ = legacy_history
    original = deepcopy(old.state_dict())
    path = tmp_path / "step01/step01_group00_rollout001.meta.json"
    if damage == "missing":
        path.unlink()
    elif damage in ("pending", "identity"):
        row = json.loads(path.read_text())
        row.update({"status": "evaluation_pending"} if damage == "pending" else {"parent_id": "wrong"})
        path.write_text(json.dumps(row))
    elif damage == "summary":
        (tmp_path / "step01/step01.summary.json").unlink()
    else:
        path = tmp_path / "step01/step01.parents.json"
        row = json.loads(path.read_text())
        row["parents"][0]["ancestor_ids"] = ["missing-root"]
        path.write_text(json.dumps(row))
    with pytest.raises((ValueError, FileNotFoundError)):
        search().migrate_state_dict(original, tmp_path, 3)
    assert old.state_dict() == original


def test_migration_defers_recovery_and_checkpoints_before_new_selection(tmp_path, legacy_history, monkeypatch):
    import train_multy_CVaR as training
    old, _, _ = legacy_history
    cfg = SimpleNamespace(parent_selection="thompson", group_size=8, seed=42)
    optimizer, tracker, adapter = object(), object(), tmp_path / "adapter_step002"
    saved = []

    def checkpoint(directory, next_step, path, sampler, opt, spo):
        assert (directory, next_step, path, opt, spo) == (tmp_path, 3, adapter, optimizer, tracker)
        assert json.loads((tmp_path / "config.json").read_text())["parent_selection"] == "thompson"
        saved.append(sampler.state_dict())
    monkeypatch.setattr(training, "_save_training_checkpoint", checkpoint)
    args = (old, cfg, tmp_path, 3, adapter, optimizer, tracker)
    assert training._migrate_parent_selection(*args, recovery_pending=True) is old
    assert saved == []
    assert "parent_selection" not in json.loads((tmp_path / "config.json").read_text())
    migrated = training._migrate_parent_selection(*args, recovery_pending=False)
    assert saved == [migrated.state_dict()]
    assert json.loads((tmp_path / "config.json").read_text())["model_name"] == "unchanged"
    assert training._migrate_parent_selection(migrated, *args[1:], recovery_pending=False) is migrated
    assert len(saved) == 1


def test_failed_checkpoint_can_repeat_migration_without_changing_training_state(tmp_path, legacy_history, monkeypatch):
    import train_multy_CVaR as training
    old, _, _ = legacy_history
    original = deepcopy(old.state_dict())
    cfg = SimpleNamespace(parent_selection="thompson", group_size=8, seed=42)
    states = []
    def checkpoint(*args):
        states.append(args[3].state_dict())
        if len(states) == 1:
            raise OSError("simulated interrupted checkpoint write")
    monkeypatch.setattr(training, "_save_training_checkpoint", checkpoint)
    args = (old, cfg, tmp_path, 3, tmp_path / "adapter_step002", object(), object())
    with pytest.raises(OSError):
        training._migrate_parent_selection(*args, recovery_pending=False)
    training._migrate_parent_selection(*args, recovery_pending=False)
    assert states[0] == states[1]
    assert old.state_dict() == original


def test_migration_saved_checkpoint_keeps_optimizer_and_spo_state(tmp_path, legacy_history):
    import torch
    import train_multy_CVaR as training
    old, _, _ = legacy_history
    parameter = torch.nn.Parameter(torch.tensor([1.]))
    optimizer = torch.optim.AdamW([parameter], lr=.01)
    parameter.square().sum().backward()
    optimizer.step()
    before = deepcopy(optimizer.state_dict())
    spo = {"version": 2, "value": 123., "history": [1., 2.]}
    tracker = SimpleNamespace(state_dict=lambda: spo)
    cfg = SimpleNamespace(parent_selection="thompson", group_size=8, seed=42)
    migrated = training._migrate_parent_selection(
        old, cfg, tmp_path, 3, tmp_path / "adapter_step002", optimizer, tracker,
        recovery_pending=False)
    checkpoint = training._load_training_checkpoint(tmp_path)
    assert checkpoint["next_step"] == 3
    assert checkpoint["adapter_dir"] == "adapter_step002"
    assert checkpoint["spo_rs_tracker"] == spo
    assert checkpoint["sampler"] == migrated.state_dict()
    assert checkpoint["optimizer"]["param_groups"] == before["param_groups"]
    for index, state in before["state"].items():
        for key, value in state.items():
            assert torch.equal(value, checkpoint["optimizer"]["state"][index][key])


def children(sampler, parent, rewards, step=0):
    batch = [State.make(timestep=step, value=r, code=f"{parent.id}:{step}:{i}")
             for i, r in enumerate(rewards)]
    sampler.record_expansion(parent, len(rewards))
    sampler.record_outcomes(parent, rewards, [True] * len(rewards))
    sampler.update([(child, parent) for child in batch])
    return batch


@pytest.mark.parametrize("mode", ["puct", "uct", "thompson"])
def test_archive_deduplicates_constructions_without_losing_outcomes(mode):
    sampler = make_sampler(
        parent_selection=mode, num_seeds=1, topk_children=0,
        seed_states=[SimpleNamespace(value=1., code="seed", construction=[.1, .2])])
    parent = sampler._states[0]
    seed_copy = State.make(0, 1., "different seed code", construction=[.1, .2])
    first = State.make(0, 2., "program A", construction=[.3, .4], raw_score=.5)
    copied = State.make(0, 2., "program B", construction=[.3, .4], raw_score=.4)
    # Equal source can produce distinct outcomes; exact nearby values and
    # sequence order must remain distinct too.
    changed = State.make(0, 2.1, "program A", construction=[.3, np.nextafter(.4, 1.)])
    reversed_values = State.make(0, 2.2, "program A", construction=[.4, .3])
    batch = [seed_copy, first, copied, changed, reversed_values]
    sampler.record_expansion(parent, len(batch))
    sampler.record_outcomes(parent, [s.value for s in batch], [True] * len(batch))
    result = sampler.update([(s, parent) for s in batch],
                            strategy_keys=[0, 0, 1, 1, 1], strategy_top_r=3)
    assert result["submitted"] == 5 and result["unique_new"] == 3
    assert {s.id for s in sampler._states} == {
        parent.id, first.id, changed.id, reversed_values.id}
    assert sampler._n[parent.id] == sampler._T == 5
    assert sampler._m[parent.id] == 2.2
    assert sampler.best_raw_state(maximize=False).id == copied.id
    if mode == "thompson":
        assert sampler._outcomes.attempts == 5

    # Resume retains the existing identities/counts. New copies of those
    # constructions cannot enter with a new zero-count identity.
    payload = sampler.state_dict()
    resumed = make_sampler(parent_selection=mode, num_seeds=1, topk_children=0)
    resumed.load_state_dict(payload)
    assert resumed.state_dict() == payload
    copied_again = State.make(1, 2., "yet another program", construction=[.3, .4])
    result = resumed.update([(copied_again, resumed._states[0])])
    assert result["unique_new"] == 0
    assert {s.id for s in resumed._states} == {s.id for s in sampler._states}
    assert resumed._n == sampler._n and resumed._T == sampler._T


def test_archive_code_fallback_and_legacy_construction_import():
    sampler = PUCTSampler(1, topk_children=0)
    parent = sampler._states[0]
    first = State.make(0, 1., "code-only", construction=None)
    duplicate = State.make(0, 1., "code-only", construction=[])
    assert sampler.update([(first, parent), (duplicate, parent)])["unique_new"] == 1
    imported = State.make(0, 2., "import A", construction=[1., 2.])
    copied = State.make(0, 2., "import B", construction=[1., 2.])
    sampler.import_legacy_states([imported, copied])
    assert imported.id in {s.id for s in sampler._states}
    assert copied.id not in {s.id for s in sampler._states}


def test_checkpoint_cleanup_preserves_training_and_backs_up_before_puct_switch(tmp_path):
    import torch
    from dataclasses import asdict
    from clean_sampler_archive import clean_archive

    sampler = PUCTSampler(1, seed_value=1.)
    parent = sampler._states[0]
    explored = State.make(0, 2., "explored", construction=[.1, .2],
                          parents=[dict(id=parent.id, timestep=0)])
    copied = State.make(1, 3., "fresh copy", construction=[.1, .2],
                        parents=[dict(id=parent.id, timestep=0)], raw_score=.3)
    saved = sampler.state_dict()
    saved.update(selection_mode="llm", states=saved["states"] + [asdict(copied), asdict(explored)],
                 n={parent.id: 64, explored.id: 32}, m={parent.id: 3.}, T=64,
                 best_raw_min=asdict(copied), best_raw_max=asdict(copied))
    payload = dict(version=2, next_step=2, adapter_dir="adapter_step001", sampler=saved,
                   optimizer={"state": {0: {"exp_avg": torch.tensor([1., 2.])}}},
                   spo_rs_tracker={"v": 1.7, "step": 1})
    path = tmp_path / "training_state.pt"
    torch.save(payload, path)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(dict(parent_selection="llm", uct=False, seed=42)))
    old_bytes = path.read_bytes()
    old_config = config_path.read_bytes()
    clean_archive(tmp_path)
    actual = torch.load(path, map_location="cpu", weights_only=False)
    assert actual["next_step"] == 2 and actual["adapter_dir"] == payload["adapter_dir"]
    assert actual["spo_rs_tracker"] == payload["spo_rs_tracker"]
    assert torch.equal(actual["optimizer"]["state"][0]["exp_avg"],
                       payload["optimizer"]["state"][0]["exp_avg"])
    assert [row["id"] for row in actual["sampler"]["states"]] == [parent.id, explored.id]
    for key in ("n", "m", "T", "best_raw_min", "best_raw_max"):
        assert actual["sampler"][key] == saved[key]
    assert actual["sampler"]["selection_mode"] == "puct"
    assert json.loads(config_path.read_text()) == dict(parent_selection="puct", uct=False, seed=42)
    backups = list(tmp_path.glob("archive_cleanup_backup_*"))
    assert len(backups) == 1
    assert (backups[0] / path.name).read_bytes() == old_bytes
    assert (backups[0] / config_path.name).read_bytes() == old_config
    clean_archive(tmp_path)  # Repeating a completed cleanup does not rewrite it.
    assert list(tmp_path.glob("archive_cleanup_backup_*")) == backups


def test_checkpoint_cleanup_refuses_to_change_an_interrupted_steps_parents(tmp_path):
    import torch
    from clean_sampler_archive import clean_archive

    checkpoint = tmp_path / "training_state.pt"
    payload = dict(version=2, next_step=2, adapter_dir="adapter_step001",
                   sampler=PUCTSampler(1).state_dict(), optimizer={})
    torch.save(payload, checkpoint)
    (tmp_path / "config.json").write_text('{"parent_selection":"llm"}')
    (tmp_path / "step02").mkdir()
    (tmp_path / "step02" / "step02.parents.json").write_text('{}')
    original = checkpoint.read_bytes()
    with pytest.raises(ValueError, match="already started"):
        clean_archive(tmp_path)
    assert checkpoint.read_bytes() == original
    assert not list(tmp_path.glob("archive_cleanup_backup_*"))


@pytest.mark.parametrize("mode", ["puct", "uct"])
def test_existing_pickers_and_checkpoint_format_unchanged(mode):
    original = PUCTSampler(3, use_uct=(mode == "uct"))
    for index, parent in enumerate(original._states):
        children(original, parent, [index + .1, index + .2])
    payload = original.state_dict()
    current = make_sampler(parent_selection=mode, num_seeds=3, group_size=64)
    current.load_state_dict(payload)
    current.record_outcomes(current._states[0], [999, float("nan")], [False, False])
    assert current.state_dict() == payload
    assert [s.id for s in current.sample_states(4, selection_step=1)] == [
        s.id for s in original.sample_states(4)]
    assert current.last_picks_info == original.last_picks_info
    assert "thompson" not in current.state_dict()


def test_all_yaml_presets_default_to_original_picker_and_accept_new_mode():
    paths = sorted((Path(__file__).resolve().parents[1] / "configs").glob("*.yaml"))
    assert len(paths) == 8
    for path in paths:
        config = yaml.safe_load(path.read_text())
        assert config["parent_selection"] == "puct"
        for mode in ("puct", "uct", "thompson", "llm"):
            config["parent_selection"] = mode
            validate_problem_config(config, source=path)
    assert parent_selection_mode({}) == "puct"
    assert parent_selection_mode({"uct": True}) == "uct"
    assert parent_selection_mode({"uct": True, "parent_selection": "thompson"}) == "thompson"
    for invalid in ("unknown", True, 3):
        with pytest.raises(ValueError, match="parent_selection"):
            validate_problem_config({"problem": "erdos", "parent_selection": invalid},
                                    require_complete=False)


@pytest.mark.parametrize("threshold", [-4., 0., .4, 5.])
def test_expected_gain_matches_analytic_one_rollout(threshold):
    p, mu, sigma = .31, .4, 1.3
    z = (mu - threshold) / sigma
    expected = p * ((mu - threshold) * .5 * math.erfc(-z / math.sqrt(2.))
                    + sigma * math.exp(-z * z / 2.) / math.sqrt(2. * math.pi))
    assert expected_max_gain(p, mu, sigma, threshold, 1) == pytest.approx(
        expected, rel=1e-8, abs=1e-12)


def test_pending_work_reduces_branch_utility_without_fake_outcomes():
    gains = [expected_max_gain(.4, .2, .3, .1, 64, pending=n)
             for n in (0, 64, 128, 512)]
    assert all(a > b > 0 for a, b in zip(gains, gains[1:]))
    assert expected_max_gain(1., 100., 1., 0., 64) > 100.
    sampler = search(seeds=2)
    sampler._draw_model = lambda *args: (.5, 0., .2)
    picks = sampler.sample_states(2, selection_step=0)
    assert len({s.id for s in picks}) == 2
    assert sampler._outcomes.attempts == 0
    assert all(n.subtree.attempts == 0 for n in sampler._nodes.values())


def test_siblings_share_root_pending_budget():
    sampler = search(seeds=2, topk_children=0)
    for root in list(sampler._states):
        children(sampler, root, [1.1] * 8)
    sampler._draw_model = lambda *args: (.5, 0., .2)
    sampler.sample_states(4, selection_step=1)
    roots = [row["thompson"]["path"][0] for row in sampler.last_picks_info]
    assert roots[0] != roots[1]
    assert roots.count(roots[0]) == roots.count(roots[1]) == 2
    assert max(row["thompson"]["pending_groups_on_path"][0]
               for row in sampler.last_picks_info) == 1


def test_all_outcomes_count_and_new_children_borrow_prior_only():
    sampler = search(topk_children=1)
    root = sampler._states[0]
    sampler.record_outcomes(root, [1.1, 1.1, 1.3, 1e9, float("nan")],
                            [True, True, True, False, True])
    candidates = [State.make(0, reward, "same code") for reward in (1.1, 1.3)]
    sampler.update([(s, root) for s in candidates])
    assert sampler._outcomes.attempts == 5
    assert sampler._outcomes.valid == 3
    assert sampler._best_reward == 1.3
    assert sampler.archive_size() == 3  # Only one duplicate is retained.
    child = next(s for s in sampler._states if not s.is_seed)
    assert sampler._nodes[child.id].subtree.attempts == 0
    assert sampler._context(child.id, False).attempts == 5
    sampler.record_outcomes(child, [1.2, 0], [True, False])
    assert sampler._outcomes.attempts == 7
    assert sampler._nodes[root.id].subtree.attempts == 7
    assert sampler._nodes[root.id].direct.attempts == 5
    assert sampler._context(child.id, False).attempts == 5  # Excludes its own data.


def test_sufficient_statistics_subtraction():
    all_stats, part, rest = OutcomeStats(), OutcomeStats(), OutcomeStats()
    for index, value in enumerate([-.4, .7, .1, -3., .9, 2.]):
        valid = index != 2
        all_stats.add(value, valid)
        (part if index < 3 else rest).add(value, valid)
    difference = all_stats.without(part)
    assert difference.attempts == rest.attempts
    assert difference.valid == rest.valid
    assert difference.mean == pytest.approx(rest.mean)
    assert difference.m2 == pytest.approx(rest.m2)


@pytest.mark.parametrize("batch_size", [1, 3, 8, 17, 80])
def test_widening_preserves_arbitrary_parent_budget(batch_size):
    sampler = search(seeds=1, topk_children=0)
    root = sampler._states[0]
    children(sampler, root, np.linspace(1.0, 1.1, 32))
    picks = sampler.sample_states(batch_size, selection_step=1)
    assert len(picks) == len({s.id for s in picks}) == min(batch_size, 33)
    if batch_size == 1:
        assert len(sampler._nodes[root.id].active_children) == 3
    if batch_size == 17:
        assert sampler.last_picks_info[0]["thompson"]["batch_capacity_widenings"] > 0


def test_pruned_ancestors_keep_evidence_and_reachable_descendants():
    sampler = search(seeds=1, topk_children=1)
    root = sampler._states[0]
    first = children(sampler, root, [1.1])[0]
    descendant = children(sampler, first, [1.2], step=1)[0]
    children(sampler, root, [1.3], step=2)  # Removes first from the live archive.
    assert first not in sampler._states and descendant in sampler._states
    assert sampler._nodes[first.id].subtree.attempts == 1
    assert descendant in sampler.sample_states(3, selection_step=3)
    for node in sampler.state_dict()["thompson"]["nodes"].values():
        assert "code" not in node and "construction" not in node


def test_checkpoint_restores_rng_pruning_and_next_selection(tmp_path):
    sampler = search(topk_children=2, max_buffer_size=5)
    roots = list(sampler._states)
    for root in roots:
        children(sampler, root, [.9, 1.1, 1.2, 1.3])
    sampler.sample_states(3, selection_step=1)
    payload = deepcopy(sampler.state_dict())
    # JSON round trip also checks that there are no tensors or NumPy scalars.
    restored = search(topk_children=2, max_buffer_size=5)
    restored.load_state_dict(json.loads(json.dumps(payload)))
    expected = sampler.sample_states(3, selection_step=2)
    actual = restored.sample_states(3, selection_step=2)
    assert [s.id for s in actual] == [s.id for s in expected]
    assert sampler.last_picks_info == restored.last_picks_info
    assert sampler.state_dict() == restored.state_dict()
    event = save_parent_selections(tmp_path, 2, "ThompsonSampler", actual,
                                   restored.last_picks_info)
    rows = json.loads(event.read_text())["parents"]
    assert all("thompson" in row and row["prior"] is None for row in rows)


def test_interrupted_step_zero_rebinds_ids_and_selects_only_once():
    original, recovered = search(seeds=3), search(seeds=3)
    selected = original.sample_states(3, selection_step=0)
    replayed = recovered.sample_states(3, selection_step=0)
    for old, fresh in zip(selected, replayed):
        assert old.code == fresh.code
        recovered.reassign_seed_id(fresh, old.id)
    state = deepcopy(recovered.state_dict())
    assert recovered.sample_states(3, selection_step=0) == replayed
    assert recovered.state_dict() == state
    assert recovered.state_dict() == original.state_dict()
    assert [s.id for s in original.sample_states(3, selection_step=1)] == [
        s.id for s in recovered.sample_states(3, selection_step=1)]


def test_incompatible_resume_is_rejected():
    with pytest.raises(ValueError, match="saved search statistics"):
        search().load_state_dict(PUCTSampler(2).state_dict())
    with pytest.raises(ValueError, match="uses thompson"):
        PUCTSampler(2).load_state_dict(search().state_dict())


def test_negative_rewards_and_no_valid_children_stay_finite():
    sampler = ThompsonSampler(num_seeds=2, seed_value=-5, group_size=64, seed=1)
    for seed in sampler._states:
        sampler.record_outcomes(seed, [1000.] * 64, [False] * 64)
    picks = sampler.sample_states(2, selection_step=1)
    assert len(picks) == 2 and sampler._best_reward == -5
    assert all(math.isfinite(row["score"]) for row in sampler.last_picks_info)
