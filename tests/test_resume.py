import json
import sys
from dataclasses import asdict

from sampler import PUCTSampler, State
from experiment_io import make_experiment_dir
from train_multy import (Config, _legacy_resume_info, _restore_legacy_archive,
                         load_config)


def test_sampler_checkpoint_round_trip():
    sampler = PUCTSampler(num_seeds=2, puct_c=1.5)
    parent = sampler.sample_states(1)[0]
    sampler.record_expansion(parent, count=4)
    child = State.make(
        timestep=0, value=2.5, raw_score=2.5,
        code="def run(): return 2.5", construction=[1.0, 2.0],
    )
    sampler.update([(child, parent)])
    sampler.set_external_prior({child.id: 0.75}, alpha=0.4)

    restored = PUCTSampler(num_seeds=1)
    restored.load_state_dict(sampler.state_dict())

    assert restored.state_dict() == sampler.state_dict()
    assert restored.best_state().construction == [1.0, 2.0]


def test_best_raw_state_respects_problem_direction_and_includes_seed():
    class Seed:
        value = 2.0
        code = "seed"
        raw_score = 0.5
        construction = None

    sampler = PUCTSampler(num_seeds=0, seed_states=[Seed()],
                          topk_children=3)
    parent = sampler._states[0]
    low = State.make(1, 3.0, "low", raw_score=0.4)
    high = State.make(1, 1.0, "high", raw_score=0.8)
    missing = State.make(1, 4.0, "missing", raw_score=None)
    sampler.update([(low, parent), (high, parent), (missing, parent)])

    assert sampler.best_raw_state(maximize=False) is low
    assert sampler.best_raw_state(maximize=True) is high

    # A better stochastic result from identical code is not inserted into the
    # search archive, but it must still advance the best-ever raw record.
    duplicate = State.make(2, 5.0, "low", raw_score=0.3)
    sampler.update([(duplicate, parent)])
    assert sampler.best_raw_state(maximize=False) is duplicate

    restored = PUCTSampler(num_seeds=1)
    restored.load_state_dict(sampler.state_dict())
    assert restored.best_raw_state(maximize=False).raw_score == 0.3


def test_experiment_config_keeps_problem_specific_keys(tmp_path):
    cfg = Config()
    run_dir = make_experiment_dir(
        cfg, root=tmp_path,
        config_dict={"problem": "gpu_mode", "task_yaml": "task.yml"},
    )
    saved = json.loads((run_dir / "config.json").read_text())

    assert saved["task_yaml"] == "task.yml"
    assert saved["_max_seq_length_includes_memory_topup"] is False


def test_resume_uses_saved_config_and_cli_can_extend_steps(tmp_path, monkeypatch):
    run_dir = tmp_path / "old-run"
    run_dir.mkdir()
    saved = asdict(Config())
    saved.update({
        "problem": "erdos",
        "model_name": "saved/model",
        "num_steps": 12,
        "groups_per_step": 3,
        "_max_seq_length_includes_memory_topup": False,
    })
    (run_dir / "config.json").write_text(json.dumps(saved))

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", [
        "train_multy.py", "--resume", str(run_dir), "--num-steps", "20",
    ])
    cfg, merged = load_config()

    assert cfg.problem == "erdos"
    assert cfg.model_name == "saved/model"
    assert cfg.groups_per_step == 3
    assert cfg.num_steps == 20
    assert merged["_resume_dir"] == str(run_dir.resolve())


def test_old_saved_config_memory_topup_is_not_applied_twice(tmp_path, monkeypatch):
    run_dir = tmp_path / "old-run"
    run_dir.mkdir()
    saved = asdict(Config())
    saved.update({
        "memory": True,
        "memory_grant_context": True,
        "memory_token_budget": 1200,
        "max_seq_length": 33200,
    })
    (run_dir / "config.json").write_text(json.dumps(saved))

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", [
        "train_multy.py", "--resume", str(run_dir),
    ])
    cfg, _ = load_config()

    assert cfg.max_seq_length == 32000


def test_legacy_run_restarts_at_adapter_step_and_recovers_archive(tmp_path):
    (tmp_path / "adapter_step003").mkdir()
    step_dir = tmp_path / "step02"
    step_dir.mkdir()
    base = "step02_group00_rollout000"
    (step_dir / f"{base}.txt").write_text(
        "```python\ndef run():\n    return 7\n```"
    )
    (step_dir / f"{base}.meta.json").write_text(json.dumps({
        "step": 2, "valid": True, "reward": 7.0, "raw_score": 7.0,
        "construction": [3.0, 4.0],
    }))

    step, adapter = _legacy_resume_info(tmp_path)
    sampler = PUCTSampler(num_seeds=1)
    recovered, total = _restore_legacy_archive(sampler, tmp_path, step)

    assert step == 3
    assert adapter.name == "adapter_step003"
    assert recovered == total == 1
    assert sampler.best_state().construction == [3.0, 4.0]
