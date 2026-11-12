import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
from threading import Event
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from experiment_io import StepPlotter, save_best_state_snapshot
from analysis.plot_erdos_best_curve import load_best_curve
from analysis.plot_parent_tree import (_best_path, load_genealogy, load_plot_genealogy,
                              tree_layout, write_svg, write_tree_log)


def _parents(root, step, rows):
    folder = root / f"step{step:02d}"
    folder.mkdir(exist_ok=True)
    (folder / f"step{step:02d}.parents.json").write_text(json.dumps({
        "step": step, "parents": rows,
    }))


def test_selected_genealogy_is_vertical_and_honors_max_step(tmp_path):
    _parents(tmp_path, 0, [{
        "parent_id": "seed", "parent_timestep": 0,
        "parent_reward": 1.0, "parent_is_seed": True, "ancestor_ids": [],
    }])
    _parents(tmp_path, 1, [{
        "parent_id": "child", "parent_timestep": 0,
        "parent_reward": 2.0, "parent_is_seed": False,
        "ancestor_ids": ["seed"],
    }])
    _parents(tmp_path, 2, [{
        "parent_id": "grandchild", "parent_timestep": 1,
        "parent_reward": 3.0, "parent_is_seed": False,
        "ancestor_ids": ["child", "seed"],
    }])
    nodes, edges = load_genealogy(tmp_path, max_step=1)
    assert set(nodes) == {"seed", "child"}
    assert edges == {("seed", "child")}
    assert nodes["seed"]["level"] < nodes["child"]["level"]
    assert set(tree_layout(nodes, edges)) == set(nodes)
    output = write_svg(tmp_path, nodes, edges)
    text = output.read_text()
    assert "Generation step" in text
    assert "parent_child_tree" in output.name


def test_every_problem_refreshes_the_parent_tree(tmp_path):
    plotter = StepPlotter(tmp_path, "ac1")
    with patch("subprocess.run", return_value=SimpleNamespace(returncode=0)) as run:
        plotter.submit(4)
        plotter.close()
    plot_calls = [call for call in run.call_args_list if call.args[0][1] == "-m"]
    scripts = [call.args[0][2] for call in plot_calls]
    assert scripts == ["analysis.plot_ac", "analysis.plot_erdos_best_curve",
                       "analysis.plot_parent_tree"]
    assert all(call.args[0][1] == "-m" and call.args[0][4:6] == ["--max-step", "4"]
               for call in plot_calls)
    assert all(call.kwargs["cwd"] == Path(__file__).resolve().parents[1]
               for call in plot_calls)
    assert run.call_args.args[0] == ["./make_explorer.sh", "--all-code", "--unified"]
    assert not (tmp_path / "step04/step04.summary.json").exists()


def test_evaluation_plot_refresh_does_not_wait_for_rendering(tmp_path):
    started, release = Event(), Event()

    def render(*args, **kwargs):
        started.set()
        assert release.wait(5)
        return SimpleNamespace(returncode=0)

    plotter = StepPlotter(tmp_path, "ac1")
    try:
        with patch("subprocess.run", side_effect=render):
            plotter.submit(4)
            assert started.wait(5)
            assert not release.is_set()
            release.set()
            plotter.close()
    finally:
        release.set()


@pytest.mark.parametrize("problem,seed_score,child_score", [
    ("ac1", .4, .6), ("ac2", 1.5, 1.3),
])
def test_best_curve_uses_evaluated_snapshot_before_training(
        tmp_path, monkeypatch, problem, seed_score, child_score):
    (tmp_path / "config.json").write_text(json.dumps({"problem": problem}))
    state = SimpleNamespace(id="seed", timestep=-1, is_seed=True,
                            raw_score=seed_score, value=1., code="")
    save_best_state_snapshot(tmp_path, 0, state, metric_name="bound",
                             maximize=problem == "ac2")
    (tmp_path / "step00/step00_group00_rollout000.meta.json").write_text(
        json.dumps(dict(step=0, valid=True, raw_score=child_score)))
    steps, cumulative, per_step, count = load_best_curve(tmp_path, max_step=0)
    assert (steps, cumulative, per_step, count) == ([0], [seed_score], [child_score], 1)
    assert not (tmp_path / "step00/step00.summary.json").exists()
    from run_explorer.collect import collect
    monkeypatch.setattr("run_explorer.collect.cf.ProcessPoolExecutor", ThreadPoolExecutor)
    main, *_ = collect(str(tmp_path), progress=None)
    evaluated = main["steps"][0]
    assert evaluated["provisional"] and evaluated["rollouts_done"]
    assert not evaluated["completed"]
    assert evaluated["best_seen_raw_score"] == seed_score
    assert evaluated["step_best_raw_score"] == child_score


def test_evaluated_step_refreshes_unified_explorer(tmp_path):
    plotter = StepPlotter(tmp_path, "ac1")
    with patch("subprocess.run", return_value=SimpleNamespace(returncode=0)) as run:
        plotter.submit_explorer(4)
        plotter.close()
    assert run.call_args.args[0] == [
        "./make_explorer.sh", "--all-code", "--unified"]
    assert run.call_args.kwargs["cwd"].name == "run_explorer"
    assert run.call_args.kwargs["timeout"] == 600
    assert run.call_args.kwargs["env"]["TTT_EVALUATED_RUN"] == str(tmp_path.resolve())
    assert "evaluated step 4" in (tmp_path / "explorer.log").read_text()


def test_unified_explorer_includes_the_evaluated_run(tmp_path, monkeypatch):
    from run_explorer import build_explorer

    reference, current = tmp_path / "reference", tmp_path / "current"
    reference.mkdir()
    current.mkdir()
    monkeypatch.setattr(build_explorer, "default_unified_runs", lambda: [str(reference)])
    builds = []
    monkeypatch.setattr(build_explorer, "build_unified", lambda runs, *args: builds.append(runs))
    monkeypatch.setenv("TTT_EVALUATED_RUN", str(current))
    monkeypatch.setattr(sys, "argv", ["build_explorer.py", "--unified"])
    build_explorer.main()
    assert builds == [[str(reference), str(current)]]


def test_plot_keeps_completed_steps_when_selection_revisits_old_nodes(tmp_path):
    for step in range(3):
        _parents(tmp_path, step, [{
            "group": 0, "parent_id": "seed", "parent_timestep": 0,
            "parent_reward": 1.0, "parent_raw_score": 1.0,
            "parent_is_seed": True, "ancestor_ids": [],
        }])
    for step, node_id, reward, raw, valid, pending in [
            (0, "first-child", 2.0, .5, True, False),
            (1, "best-child", 4.0, .25, True, False),
            (1, "worse-child", 3.0, 1 / 3, True, False),
            (1, "failed-child", 999.0, None, False, False),
            (2, "pending-child", 10.0, .1, True, True)]:
        folder = tmp_path / f"step{step:02d}"
        (folder / f"step{step:02d}_group00_rollout_{node_id}.meta.json").write_text(
            json.dumps({
                "step": step, "group": 0, "node_id": node_id,
                "parent_id": "seed" if step == 0 else None,
                "reward": reward, "raw_score": raw, "valid": valid,
                "status": "evaluation_pending" if pending else "evaluated",
            }))
    selected, _ = load_genealogy(tmp_path)
    assert set(selected) == {"seed"}
    nodes, edges = load_plot_genealogy(tmp_path)
    assert set(nodes) == {"seed", "first-child", "best-child"}
    assert edges == {("seed", "first-child"), ("seed", "best-child")}
    assert nodes["best-child"]["level"] == 2
    assert nodes["best-child"]["raw_score"] == .25
    assert _best_path(nodes, edges)[0] == {"seed", "best-child"}
    earlier, _ = load_plot_genealogy(tmp_path, max_step=0)
    assert set(earlier) == {"seed", "first-child"}


def test_complete_tree_log_links_both_directions_and_zeros_failures(tmp_path):
    _parents(tmp_path, 0, [{
        "parent_id": "seed", "parent_timestep": 0,
        "parent_reward": 1.0, "parent_raw_score": .5,
        "parent_is_seed": True, "ancestor_ids": [],
    }])
    folder = tmp_path / "step00"
    for rollout, valid in enumerate((True, False)):
        (folder / f"step00_group00_rollout{rollout:03d}.meta.json").write_text(
            json.dumps({
                "step": 0, "group": 0, "rollout": rollout,
                "node_id": f"child-{rollout}",
                "parent_id": "seed" if valid else None, "parent_group": 0,
                "parent_value": 1.0, "parent_raw_score": .5,
                "parent_is_seed": True, "valid": valid,
                "reward": 2.0 if valid else 99.0,
                "raw_score": .4 if valid else None,
                "failure_kind": None if valid else "code",
            }))
    (folder / "step00_group00_rollout002.meta.json").write_text(json.dumps({
        "step": 0, "group": 0, "parent_group": 0, "rollout": 2,
        "status": "evaluation_pending",
    }))
    payload_path = write_tree_log(tmp_path, max_step=0)
    payload = json.loads(payload_path.read_text())
    assert payload["root_ids"] == ["seed"]
    assert payload["nodes"]["seed"]["child_ids"] == ["child-0", "child-1"]
    assert payload["nodes"]["child-0"]["parent_id"] == "seed"
    assert set(payload["nodes"]["child-0"]) == {
        "parent_id", "child_ids", "raw_problem_score",
    }
    assert payload["nodes"]["child-0"]["raw_problem_score"] == .4
    assert payload["nodes"]["child-1"]["raw_problem_score"] == 0.0
    assert len(payload["nodes"]) == 3
