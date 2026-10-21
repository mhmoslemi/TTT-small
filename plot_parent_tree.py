#!/usr/bin/env python3
"""Export the complete search tree and plot its selected-parent genealogy."""

import argparse
import html
import json
import math
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path


def _finite(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _score_label(value):
    return "" if value is None else f"{value:.6g}"


def _tree_node(node_id):
    return {
        "id": str(node_id), "parent_id": None, "child_ids": set(),
        "generation_step": None, "selected_as_parent_steps": set(),
        "is_seed": False, "valid": None, "raw_problem_score": None,
        "status": None, "failure_kind": None, "group": None, "rollout": None,
        "response_file": None,
    }


def load_genealogy(run_dir, max_step=None):
    """Return selected nodes and parent edges through ``max_step``."""
    nodes, edges = {}, set()
    for path in sorted(Path(run_dir).glob("step*/step*.parents.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            step = int(payload["step"])
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
        if max_step is not None and step > max_step:
            continue
        for row in payload.get("parents", []):
            if not isinstance(row, dict) or row.get("parent_id") is None:
                continue
            node_id = str(row["parent_id"])
            node = nodes.setdefault(node_id, {
                "id": node_id, "level": None, "reward": None,
                "raw_score": None, "is_seed": False, "selected_steps": set(),
            })
            is_seed = bool(row.get("parent_is_seed"))
            node["is_seed"] = node["is_seed"] or is_seed
            try:
                timestep = int(row.get("parent_timestep"))
            except (TypeError, ValueError):
                timestep = None
            if is_seed:
                node["level"] = 0
            elif timestep is not None:
                node["level"] = timestep + 1
            reward = _finite(row.get("parent_reward"))
            raw_score = _finite(row.get("parent_raw_score"))
            if reward is not None:
                node["reward"] = reward
            if raw_score is not None:
                node["raw_score"] = raw_score
            node["selected_steps"].add(step)

            ancestry = [str(value) for value in row.get("ancestor_ids", [])
                        if value is not None]
            chain = list(reversed(ancestry)) + [node_id]
            for ancestor_id in ancestry:
                nodes.setdefault(ancestor_id, {
                    "id": ancestor_id, "level": None, "reward": None,
                    "raw_score": None, "is_seed": False,
                    "selected_steps": set(),
                })
            for parent_id, child_id in zip(chain, chain[1:]):
                if parent_id != child_id:
                    edges.add((parent_id, child_id))

    if not nodes:
        raise ValueError(f"no selected-parent events found under {run_dir}")

    parent_of = {child: parent for parent, child in edges}

    def infer_level(node_id, visiting=None):
        node = nodes[node_id]
        if node["level"] is not None:
            return node["level"]
        visiting = set() if visiting is None else visiting
        if node_id in visiting or node_id not in parent_of:
            node["level"] = 0
        else:
            visiting.add(node_id)
            node["level"] = infer_level(parent_of[node_id], visiting) + 1
        return node["level"]

    for node_id in nodes:
        infer_level(node_id)
    return nodes, edges


def write_tree_log(run_dir, max_step=None):
    """Write an LLM-friendly, bidirectionally linked complete search tree."""
    run_dir = Path(run_dir)
    nodes = {}
    try:
        selected_nodes, selected_edges = load_genealogy(run_dir, max_step)
    except ValueError:
        selected_nodes, selected_edges = {}, set()

    for node_id, source in selected_nodes.items():
        node = nodes.setdefault(node_id, _tree_node(node_id))
        node.update({
            "generation_step": (None if source["is_seed"]
                                else int(source["level"]) - 1),
            "selected_as_parent_steps": set(source["selected_steps"]),
            "is_seed": bool(source["is_seed"]),
            "valid": True,
            "status": "seed" if source["is_seed"] else "valid",
            "raw_problem_score": source["raw_score"],
        })
    for parent_id, child_id in selected_edges:
        parent = nodes.setdefault(parent_id, _tree_node(parent_id))
        child = nodes.setdefault(child_id, _tree_node(child_id))
        parent["child_ids"].add(child_id)
        child["parent_id"] = parent_id

    selected_parent_ids = {}
    for path in sorted(run_dir.glob("step*/step*.parents.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            step = int(payload["step"])
        except (OSError, KeyError, TypeError, ValueError,
                json.JSONDecodeError):
            continue
        if max_step is not None and step > max_step:
            continue
        for row in payload.get("parents", []):
            if isinstance(row, dict) and row.get("parent_id") is not None:
                selected_parent_ids[(step, int(row.get("group", 0)))] = str(
                    row["parent_id"])

    errors = []
    pattern = "step*/step*_group*_rollout*.meta.json"
    for path in sorted(run_dir.glob(pattern)):
        try:
            meta = json.loads(path.read_text(encoding="utf-8"))
            step = int(meta["step"])
        except (OSError, KeyError, TypeError, ValueError,
                json.JSONDecodeError) as error:
            errors.append({"file": str(path.relative_to(run_dir)),
                           "error": str(error)})
            continue
        if max_step is not None and step > max_step:
            continue
        if meta.get("status") == "evaluation_pending":
            continue

        node_id = str(meta.get("node_id") or
                      f"rollout:{step}:{meta.get('group')}:{meta.get('rollout')}")
        parent_value = meta.get("parent_id")
        if parent_value is None:
            parent_group = int(meta.get("parent_group", meta.get("group", 0)))
            parent_value = selected_parent_ids.get((step, parent_group))
        parent_id = str(parent_value) if parent_value is not None else None
        valid = bool(meta.get("valid"))
        raw_score = _finite(meta.get("raw_score"))
        if valid is False:
            raw_score = 0.0
        node = nodes.setdefault(node_id, _tree_node(node_id))
        node.update({
            "parent_id": parent_id,
            "generation_step": step,
            "valid": valid,
            "status": "valid" if valid else "failed",
            "raw_problem_score": raw_score,
            "failure_kind": (None if valid is not False else
                             str(meta.get("failure_kind") or "unknown")),
            "group": meta.get("group"),
            "rollout": meta.get("rollout"),
            "response_file": str(path.relative_to(run_dir)).removesuffix(
                ".meta.json") + ".txt",
        })
        if parent_id is not None:
            parent = nodes.setdefault(parent_id, _tree_node(parent_id))
            parent["child_ids"].add(node_id)
            parent_raw_score = _finite(meta.get("parent_raw_score"))
            if parent_raw_score is not None:
                parent["raw_problem_score"] = parent_raw_score
            parent["is_seed"] = bool(
                parent["is_seed"] or meta.get("parent_is_seed"))

    def node_order(node_id):
        node = nodes[node_id]
        step = node["generation_step"]
        return (-1 if step is None else int(step),
                -1 if node["group"] is None else int(node["group"]),
                -1 if node["rollout"] is None else int(node["rollout"]),
                node_id)

    ordered_ids = sorted(nodes, key=node_order)
    serialized = {}
    for node_id in ordered_ids:
        node = nodes[node_id]
        serialized[node_id] = {
            "parent_id": node["parent_id"],
            "child_ids": sorted(node["child_ids"], key=node_order),
            "raw_problem_score": node["raw_problem_score"],
        }
    roots = [node_id for node_id in ordered_ids
             if nodes[node_id]["parent_id"] is None]
    included_steps = [node["generation_step"] for node in nodes.values()
                      if node["generation_step"] is not None]
    payload = {
        "updated_through_step": (max_step if max_step is not None else
                                 (max(included_steps)
                                  if included_steps else None)),
        "root_ids": roots,
        "nodes": serialized,
    }
    if errors:
        payload["source_errors"] = errors
    output = run_dir / "search_tree.json"
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                         encoding="utf-8")
    temporary.replace(output)
    return output


def tree_layout(nodes, edges):
    children = defaultdict(list)
    child_ids = set()
    for parent, child in edges:
        children[parent].append(child)
        child_ids.add(child)
    roots = sorted(set(nodes) - child_ids,
                   key=lambda value: (nodes[value]["level"], value))
    x_positions, visiting, next_leaf = {}, set(), [0]

    def place(node_id):
        if node_id in x_positions:
            return x_positions[node_id]
        if node_id in visiting:
            return float(next_leaf[0])
        visiting.add(node_id)
        descendants = sorted(set(children[node_id]),
                             key=lambda value: (nodes[value]["level"], value))
        if descendants:
            values = [place(child) for child in descendants]
            x_positions[node_id] = sum(values) / len(values)
        else:
            x_positions[node_id] = float(next_leaf[0])
            next_leaf[0] += 1
        visiting.remove(node_id)
        return x_positions[node_id]

    for root in roots:
        place(root)
        next_leaf[0] += 1
    for node_id in nodes:
        if node_id not in x_positions:
            place(node_id)
    return x_positions


def _best_path(nodes, edges):
    candidates = [node for node in nodes.values()
                  if node["reward"] is not None and node["selected_steps"]]
    if not candidates:
        return set(), set()
    current = max(candidates, key=lambda node: node["reward"])["id"]
    parent_of = {child: parent for parent, child in edges}
    path_nodes, path_edges = {current}, set()
    while current in parent_of:
        parent = parent_of[current]
        path_nodes.add(parent)
        path_edges.add((parent, current))
        current = parent
    return path_nodes, path_edges


def write_svg(run_dir, nodes, edges):
    x_positions = tree_layout(nodes, edges)
    path_nodes, path_edges = _best_path(nodes, edges)
    max_level = max(node["level"] for node in nodes.values())
    leaf_span = max(x_positions.values(), default=0.0) + 1.0
    width = max(760, int(110 + 38 * leaf_span))
    height = max(560, 110 + 138 * (max_level + 1))
    left, right, top, bottom = 72, 28, 30, 34
    plot_width = width - left - right
    plot_height = height - top - bottom

    def x(node_id):
        return left + ((x_positions[node_id] + .5) / leaf_span) * plot_width

    def y(node_id):
        if max_level == 0:
            return top + plot_height / 2
        return top + nodes[node_id]["level"] / max_level * plot_height

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<g font-family="Helvetica,Arial,sans-serif" font-size="12" '
        'fill="#334155">',
    ]
    for level in range(max_level + 1):
        yy = top + (plot_height / 2 if max_level == 0
                    else level / max_level * plot_height)
        label = "Seeds" if level == 0 else str(level - 1)
        lines.append(
            f'<line x1="{left}" y1="{yy:.2f}" x2="{width-right}" '
            f'y2="{yy:.2f}" stroke="#e2e8f0" stroke-width="1"/>')
        lines.append(
            f'<text x="{left-12}" y="{yy+4:.2f}" text-anchor="end">'
            f'{label}</text>')
    for parent, child in sorted(edges):
        selected = (parent, child) in path_edges
        lines.append(
            f'<line x1="{x(parent):.2f}" y1="{y(parent):.2f}" '
            f'x2="{x(child):.2f}" y2="{y(child):.2f}" '
            f'stroke="{"#dc2626" if selected else "#cbd5e1"}" '
            f'stroke-width="{3 if selected else 1.2}"/>')
    for node_id, node in sorted(nodes.items(), key=lambda item: item[0]):
        selected = node_id in path_nodes
        fill = "#f59e0b" if selected else (
            "#64748b" if node["is_seed"] else "#2563eb")
        stroke = "#dc2626" if selected else "white"
        tooltip = (f"reward={node['reward']!r}; raw={node['raw_score']!r}; "
                   f"selected steps={sorted(node['selected_steps'])}")
        if node["is_seed"]:
            xx, yy, radius = x(node_id), y(node_id), 6
            points = (f"{xx:.2f},{yy-radius:.2f} {xx+radius:.2f},{yy:.2f} "
                      f"{xx:.2f},{yy+radius:.2f} {xx-radius:.2f},{yy:.2f}")
            lines.append(
                f'<polygon points="{points}" fill="{fill}" stroke="{stroke}" '
                f'stroke-width="{2 if selected else 1}"><title>'
                f'{html.escape(tooltip)}</title></polygon>')
        else:
            lines.append(
                f'<circle cx="{x(node_id):.2f}" cy="{y(node_id):.2f}" r="5" '
                f'fill="{fill}" stroke="{stroke}" '
                f'stroke-width="{2 if selected else 1}"><title>'
                f'{html.escape(tooltip)}</title></circle>')
        score = _score_label(node["raw_score"])
        if score:
            xx, yy = x(node_id) + 6, y(node_id) + 9
            lines.append(
                f'<text x="{xx:.2f}" y="{yy:.2f}" font-size="8" '
                f'fill="#475569" transform="rotate(45 {xx:.2f} {yy:.2f})">'
                f'{html.escape(score)}</text>')
    lines.extend([
        f'<text x="18" y="{height/2:.2f}" text-anchor="middle" '
        f'transform="rotate(-90 18 {height/2:.2f})">Generation step</text>',
        '</g>', '</svg>',
    ])
    output = Path(run_dir) / "parent_child_tree.svg"
    output.write_text("\n".join(lines), encoding="utf-8")
    return output


def _export_static(svg_path, nodes, edges):
    converter = shutil.which("rsvg-convert")
    if converter:
        commands = [
            [converter, "-f", "png", "-o", str(svg_path.with_suffix(".png")),
             str(svg_path)],
            [converter, "-f", "pdf", "-o", str(svg_path.with_suffix(".pdf")),
             str(svg_path)],
        ]
    elif shutil.which("magick"):
        converter = shutil.which("magick")
        commands = [[converter, str(svg_path), str(svg_path.with_suffix(suffix))]
                    for suffix in (".png", ".pdf")]
    else:
        commands = None
    if commands is not None:
        succeeded = True
        for command in commands:
            result = subprocess.run(
                command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                timeout=60, check=False)
            succeeded = succeeded and result.returncode == 0
        if succeeded:
            return

    # The training environment already depends on matplotlib for its other
    # automatic plots; this keeps PNG/PDF output available without a new tool.
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    positions = tree_layout(nodes, edges)
    path_nodes, path_edges = _best_path(nodes, edges)
    leaf_span = max(positions.values(), default=0.0) + 1.0
    max_level = max(node["level"] for node in nodes.values())
    figure, axis = plt.subplots(
        figsize=(max(8, leaf_span * .34), max(5.5, (max_level + 1) * 1.35)))
    for parent, child in edges:
        selected = (parent, child) in path_edges
        axis.plot([positions[parent], positions[child]],
                  [nodes[parent]["level"], nodes[child]["level"]],
                  color="#dc2626" if selected else "#cbd5e1",
                  linewidth=2.2 if selected else .8, zorder=1)
    for node_id, node in nodes.items():
        selected = node_id in path_nodes
        axis.scatter(
            positions[node_id], node["level"], s=30,
            marker="D" if node["is_seed"] else "o",
            color="#f59e0b" if selected else (
                "#64748b" if node["is_seed"] else "#2563eb"),
            edgecolor="#dc2626" if selected else "white",
            linewidth=1.2, zorder=2)
        score = _score_label(node["raw_score"])
        if score:
            axis.annotate(score, (positions[node_id], node["level"]),
                          xytext=(4, 5), textcoords="offset points",
                          fontsize=5.5, color="#475569", rotation=45,
                          ha="left", va="top", zorder=3)
    axis.set_yticks(range(max_level + 1),
                    ["Seeds"] + [str(value) for value in range(max_level)])
    axis.set_ylabel("Generation step")
    axis.set_xticks([])
    axis.invert_yaxis()
    axis.grid(axis="y", color="#e2e8f0", linewidth=.7)
    for spine in axis.spines.values():
        spine.set_visible(False)
    figure.tight_layout()
    figure.savefig(svg_path.with_suffix(".png"), dpi=180,
                   bbox_inches="tight", facecolor="white")
    figure.savefig(svg_path.with_suffix(".pdf"), bbox_inches="tight",
                   facecolor="white")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--max-step", type=int)
    args = parser.parse_args()
    tree_log = write_tree_log(args.run_dir, args.max_step)
    nodes, edges = load_genealogy(args.run_dir, args.max_step)
    output = write_svg(args.run_dir, nodes, edges)
    _export_static(output, nodes, edges)
    print(f"Wrote {tree_log} and {output}")


if __name__ == "__main__":
    main()
