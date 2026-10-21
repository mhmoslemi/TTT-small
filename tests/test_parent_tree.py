import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from experiment_io import StepPlotter
from plot_parent_tree import (load_genealogy, tree_layout, write_svg,
                              write_tree_log)


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
