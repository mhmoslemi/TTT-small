"""CPU checks for supported parent pickers and archive checkpointing."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from config_validation import validate_problem_config
from search.sampler import PUCTSampler, State, make_sampler, parent_selection_mode


def children(sampler, parent, rewards, step=0):
    batch = [State.make(timestep=step, value=r, code=f"{parent.id}:{step}:{i}")
             for i, r in enumerate(rewards)]
    sampler.record_expansion(parent, len(rewards))
    sampler.update([(child, parent) for child in batch])
    return batch



@pytest.mark.parametrize("mode", ["puct", "uct"])
def test_archive_deduplicates_code_without_losing_outcomes(mode):
    sampler = make_sampler(
        parent_selection=mode, num_seeds=1, topk_children=0,
        seed_states=[SimpleNamespace(value=1., code="seed", construction=[.1, .2])])
    parent = sampler._states[0]
    seed_copy = State.make(0, 1., "different seed code", construction=[.1, .2])
    first = State.make(0, 2., "program A", construction=[.3, .4], raw_score=.5)
    copied = State.make(0, 2., "program B", construction=[.3, .4], raw_score=.4)
    # 0544 rejects repeated source even when the returned construction changes.
    changed = State.make(0, 2.1, "program A", construction=[.3, np.nextafter(.4, 1.)])
    reversed_values = State.make(0, 2.2, "program A", construction=[.4, .3])
    batch = [seed_copy, first, copied, changed, reversed_values]
    sampler.record_expansion(parent, len(batch))
    result = sampler.update([(s, parent) for s in batch],
                            strategy_keys=[0, 0, 1, 1, 1], strategy_top_r=3)
    assert result["submitted"] == 5 and result["unique_new"] == 3
    assert {s.id for s in sampler._states} == {
        parent.id, seed_copy.id, first.id, copied.id}
    assert sampler._n[parent.id] == sampler._T == 5
    assert sampler._m[parent.id] == 2.2
    assert sampler.best_raw_state(maximize=False).id == copied.id

    # Resume retains identities/counts and the exact-code duplicate rule.
    payload = sampler.state_dict()
    resumed = make_sampler(parent_selection=mode, num_seeds=1, topk_children=0)
    resumed.load_state_dict(payload)
    assert resumed.state_dict() == payload
    copied_again = State.make(1, 2., "program A", construction=[.8, .9])
    result = resumed.update([(copied_again, resumed._states[0])])
    assert result["unique_new"] == 0
    assert {s.id for s in resumed._states} == {s.id for s in sampler._states}
    assert resumed._n == sampler._n and resumed._T == sampler._T
    different_code = State.make(1, 2., "yet another program", construction=[.3, .4])
    assert resumed.update([(different_code, resumed._states[0])])["unique_new"] == 1



def test_archive_code_identity_and_legacy_import():
    sampler = PUCTSampler(1, topk_children=0)
    parent = sampler._states[0]
    first = State.make(0, 1., "code-only", construction=None)
    duplicate = State.make(0, 1., "code-only", construction=[])
    assert sampler.update([(first, parent), (duplicate, parent)])["unique_new"] == 1
    imported = State.make(0, 2., "import A", construction=[1., 2.])
    copied = State.make(0, 2., "import B", construction=[1., 2.])
    duplicate_code = State.make(0, 2., "import A", construction=[3., 4.])
    sampler.import_legacy_states([imported, copied, duplicate_code])
    assert imported.id in {s.id for s in sampler._states}
    assert copied.id in {s.id for s in sampler._states}
    assert duplicate_code.id not in {s.id for s in sampler._states}



def test_checkpoint_cleanup_preserves_training_and_backs_up_before_puct_switch(tmp_path):
    import torch
    from dataclasses import asdict
    from _retire.clean_sampler_archive import clean_archive

    sampler = PUCTSampler(1, seed_value=1.)
    parent = sampler._states[0]
    explored = State.make(0, 2., "explored", construction=[.1, .2],
                          parents=[dict(id=parent.id, timestep=0)])
    copied = State.make(1, 3., "explored", construction=[.3, .4],
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
    from _retire.clean_sampler_archive import clean_archive

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
    assert current.state_dict() == payload
    assert [s.id for s in current.sample_states(4, selection_step=1)] == [
        s.id for s in original.sample_states(4)]
    assert current.last_picks_info == original.last_picks_info



def test_all_yaml_presets_accept_supported_pickers():
    paths = sorted((Path(__file__).resolve().parents[1] / "configs").glob("*.yaml"))
    assert len(paths) == 8
    for path in paths:
        config = yaml.safe_load(path.read_text())
        assert config["parent_selection"] in ("puct", "uct", "llm")
        for mode in ("puct", "uct", "llm"):
            config["parent_selection"] = mode
            validate_problem_config(config, source=path)
    assert parent_selection_mode({}) == "puct"
    assert parent_selection_mode({"uct": True}) == "uct"
    assert parent_selection_mode({"uct": True, "parent_selection": "llm"}) == "llm"
    for invalid in ("unknown", True, 3):
        with pytest.raises(ValueError, match="parent_selection"):
            validate_problem_config({"problem": "erdos", "parent_selection": invalid},
                                    require_complete=False)
