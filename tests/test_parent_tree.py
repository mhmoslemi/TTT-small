import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from experiment_io import StepPlotter
from plot_parent_tree import (_best_path, load_genealogy, load_plot_genealogy,
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
    scripts = [Path(call.args[0][1]).name for call in run.call_args_list]
    assert scripts == ["plot_ac.py", "plot_erdos_best_curve.py",
                       "plot_parent_tree.py"]
    assert all(call.args[0][3:5] == ["--max-step", "4"]
               for call in run.call_args_list)


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
