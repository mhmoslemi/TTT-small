#!/usr/bin/env python3
"""Reconstruct and plot the parent/child search tree saved by train_multy.py.

Examples:
    python3 plot_search_tree.py runs/my_run --out tree.png
    python3 plot_search_tree.py runs/my_run --step 12 --out step12_tree.png
    python3 plot_search_tree.py runs/my_run --max-children 0 --labels none

The plotter is sampler-agnostic. Selection events and stable node IDs are
written by the trainer, so UCB, PUCT, D-PUCT, or a custom sampler can all use
the same post-run format.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


BLUE = "#2a78d6"
ORANGE = "#e58b22"
RED = "#cf3c3c"
GRAY = "#9a9995"
INK = "#151515"
INK_SOFT = "#565550"
GRID = "#e7e5df"
SURFACE = "#fcfcfb"


def _number(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _integer(value, default=None):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _read_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _run_info(run_dir):
    config = _read_json(Path(run_dir) / "config.json") or {}
    problem = str(config.get("problem", "unknown")).strip().lower()
    kind = str(config.get("problem_type", "")).strip().lower()
    if problem in ("circle_packing", "circle", "circles"):
        return config, problem, "sum of radii", True
    if problem in ("erdos", "erdos_min_overlap", "erdos_minimum_overlap"):
        return config, "erdos", "C₅ bound", False
    if problem in ("denoising", "single_cell", "single_cell_analysis", "scrna"):
        return config, problem, "MSE", False
    if problem in ("ac1", "ac2", "ac_inequalities", "autocorrelation",
                   "autocorrelation_inequalities"):
        kind = problem if problem in ("ac1", "ac2") else (kind or "ac1")
        return config, problem, ("lower bound" if kind == "ac2" else "upper bound"), kind == "ac2"
    if problem in ("gpu_mode", "kernel", "kernel_engineering", "trimul",
                   "mla_decode_nvidia", "mla"):
        return config, problem, "runtime (microseconds)", False
    return config, problem, "reward", True


def _node(node_id):
    return {
        "id": str(node_id),
        "generated_step": None,
        "reward": None,
        "raw_score": None,
        "valid": None,
        "is_seed": False,
        "archive_eligible": None,
        "selected_steps": set(),
        "selection": {},
    }


def _set_if_number(node, key, value):
    value = _number(value)
    if value is not None:
        node[key] = value


def load_tree_events(run_dir):
    """Return ``(nodes, edges, sampler_types)`` from a completed/partial run.

    Edges contain every generated rollout, including invalid and pruned ones.
    Older runs without stable IDs remain readable, but their groups appear as
    independent one-level trees because their cross-step lineage was not saved.
    """
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        raise ValueError(f"not a directory: {run_dir}")

    nodes = {}
    edges = []
    sampler_types = set()

    # Parent event files exist even if a run stopped before producing children.
    for path in sorted(run_dir.glob("step*/step*.parents.json")):
        payload = _read_json(path)
        if not isinstance(payload, dict):
            continue
        step = _integer(payload.get("step"))
        sampler_name = str(payload.get("sampler_type", "")).strip()
        if sampler_name:
            sampler_types.add(sampler_name)
        for item in payload.get("parents", []):
            if not isinstance(item, dict) or item.get("parent_id") is None:
                continue
            parent_id = str(item["parent_id"])
            node = nodes.setdefault(parent_id, _node(parent_id))
            node["is_seed"] = bool(item.get("parent_is_seed", node["is_seed"]))
            _set_if_number(node, "reward", item.get("parent_reward"))
            _set_if_number(node, "raw_score", item.get("parent_raw_score"))
            if step is not None:
                node["selected_steps"].add(step)
                node["selection"][step] = {
                    "group": _integer(item.get("group")),
                    "visits": _integer(item.get("visit_count"), 0),
                    "q": _number(item.get("q_value")),
                    "prior": _number(item.get("prior")),
                    "bonus": _number(item.get("exploration_bonus")),
                    "score": _number(item.get("selection_score")),
                }

    pattern = "step*/step*_group*_rollout*.meta.json"
    for path in sorted(run_dir.glob(pattern)):
        meta = _read_json(path)
        if not isinstance(meta, dict):
            continue
        step = _integer(meta.get("step"))
        group = _integer(meta.get("group"), 0)
        rollout = _integer(meta.get("rollout"), 0)
        if step is None:
            continue
        sampler_name = str(meta.get("sampler_type", "")).strip()
        if sampler_name:
            sampler_types.add(sampler_name)

        # Synthetic IDs keep legacy runs plottable, though those old files did
        # not contain enough information to reconnect a child selected later.
        child_id = str(meta.get("node_id") or
                       f"rollout:{step}:{group}:{rollout}")
        parent_id = str(meta.get("parent_id") or
                        f"legacy-parent:{step}:{group}")

        child = nodes.setdefault(child_id, _node(child_id))
        child["generated_step"] = step
        child["valid"] = bool(meta.get("valid"))
        child["archive_eligible"] = bool(
            meta.get("archive_eligible", meta.get("valid")))
        _set_if_number(child, "reward", meta.get("reward"))
        _set_if_number(child, "raw_score", meta.get("raw_score"))

        parent = nodes.setdefault(parent_id, _node(parent_id))
        parent["is_seed"] = bool(meta.get("parent_is_seed", parent["is_seed"]))
        _set_if_number(parent, "reward", meta.get("parent_value"))
        _set_if_number(parent, "raw_score", meta.get("parent_raw_score"))
        parent["selected_steps"].add(step)
        if step not in parent["selection"]:
            parent["selection"][step] = {
                "group": group,
                "visits": _integer(meta.get("parent_visit_count"), 0),
                "q": _number(meta.get("parent_q_value")),
                "prior": _number(meta.get("parent_prior")),
                "bonus": _number(meta.get("parent_exploration_bonus")),
                "score": _number(meta.get("parent_selection_score")),
            }
        edges.append({
            "parent": parent_id,
            "child": child_id,
            "step": step,
            "group": group,
            "rollout": rollout,
        })

    if not edges and not nodes:
        raise ValueError(f"no tree events or rollout metadata found under {run_dir}")
    return nodes, edges, sampler_types


def visible_tree(nodes, edges, step=None, max_step=None, max_children=16,
                 valid_only=False):
    """Filter events and retain selected lineage within the displayed scope."""
    def selected_in_scope(node):
        if step is not None:
            return step in node["selected_steps"]
        if max_step is not None:
            return any(value <= max_step for value in node["selected_steps"])
        return bool(node["selected_steps"])

    kept_edges = []
    for edge in edges:
        if step is not None and edge["step"] != step:
            continue
        if max_step is not None and edge["step"] > max_step:
            continue
        child = nodes[edge["child"]]
        if valid_only and not child["valid"]:
            continue
        kept_edges.append(edge)

    omitted = defaultdict(int)
    if max_children and max_children > 0:
        grouped = defaultdict(list)
        for edge in kept_edges:
            grouped[edge["parent"]].append(edge)
        trimmed = []
        for parent_id, candidates in grouped.items():
            # A child used as a parent later is never trimmed: otherwise a
            # displayed lineage would become disconnected. Fill remaining
            # slots with the highest-reward generated children.
            lineage = [e for e in candidates if selected_in_scope(nodes[e["child"]])]
            rest = [e for e in candidates if e not in lineage]
            rest.sort(key=lambda e: (
                nodes[e["child"]]["reward"]
                if nodes[e["child"]]["reward"] is not None else -math.inf
            ), reverse=True)
            limit = max(int(max_children), len(lineage))
            chosen = lineage + rest[:max(0, limit - len(lineage))]
            omitted[parent_id] = len(candidates) - len(chosen)
            trimmed.extend(chosen)
        kept_edges = trimmed

    visible_ids = {item for edge in kept_edges
                   for item in (edge["parent"], edge["child"])}
    # A parent selection with no completed children should still be visible.
    for node_id, node in nodes.items():
        if selected_in_scope(node):
            visible_ids.add(node_id)
    return {node_id: nodes[node_id] for node_id in visible_ids}, kept_edges, omitted


def overview_tree(nodes, edges, step=None, max_step=None, valid_only=False):
    """Return a sparse overview: selected lineage plus each step's best child."""
    scoped_nodes, scoped_edges, _ = visible_tree(
        nodes, edges, step=step, max_step=max_step, max_children=0,
        valid_only=valid_only)

    def selected_in_scope(node):
        if step is not None:
            return step in node["selected_steps"]
        if max_step is not None:
            return any(value <= max_step for value in node["selected_steps"])
        return bool(node["selected_steps"])

    important = {node_id for node_id, node in scoped_nodes.items()
                 if selected_in_scope(node)}
    by_step = defaultdict(list)
    for edge in scoped_edges:
        child = scoped_nodes[edge["child"]]
        if child["valid"] and child["reward"] is not None:
            by_step[edge["step"]].append(edge["child"])
    for child_ids in by_step.values():
        important.add(max(child_ids,
                          key=lambda node_id: scoped_nodes[node_id]["reward"]))

    overview_edges = [edge for edge in scoped_edges if edge["child"] in important]
    visible_ids = set(important)
    visible_ids.update(edge["parent"] for edge in overview_edges)
    visible_ids.update(edge["child"] for edge in overview_edges)

    # Count collapsed rollout siblings for the optional single-step detail
    # annotation, while keeping the full-run overview visually quiet.
    total_by_parent = defaultdict(int)
    shown_by_parent = defaultdict(int)
    for edge in scoped_edges:
        total_by_parent[edge["parent"]] += 1
    for edge in overview_edges:
        shown_by_parent[edge["parent"]] += 1
    omitted = {parent_id: total - shown_by_parent[parent_id]
               for parent_id, total in total_by_parent.items()}
    return ({node_id: scoped_nodes[node_id] for node_id in visible_ids},
            overview_edges, omitted)


def tree_layout(nodes, edges):
    """Tidy-tree x positions with chronological training-step y positions."""
    children = defaultdict(list)
    child_ids = set()
    for edge in edges:
        children[edge["parent"]].append(edge["child"])
        child_ids.add(edge["child"])
    for parent_id in children:
        children[parent_id].sort(key=lambda nid: (
            nodes[nid]["generated_step"] if nodes[nid]["generated_step"] is not None else -1,
            -(nodes[nid]["reward"] if nodes[nid]["reward"] is not None else -math.inf),
            nid,
        ))

    roots = sorted(set(nodes) - child_ids)
    if not roots:
        roots = sorted(nodes)
    positions = {}
    next_x = 0.0
    visited = set()

    def place_root(root_id, start_x):
        order = []
        stack = [(root_id, False)]
        local_seen = set()
        while stack:
            node_id, expanded = stack.pop()
            if expanded:
                order.append(node_id)
                continue
            if node_id in local_seen:
                continue
            local_seen.add(node_id)
            stack.append((node_id, True))
            for child_id in reversed(children.get(node_id, [])):
                stack.append((child_id, False))
        leaf_x = start_x
        for node_id in order:
            placed_children = [cid for cid in children.get(node_id, [])
                               if cid in positions]
            if placed_children:
                x = sum(positions[cid][0] for cid in placed_children) / len(placed_children)
            else:
                x = leaf_x
                leaf_x += 1.0
            node = nodes[node_id]
            if node["is_seed"] or node["generated_step"] is None:
                y = 0.0
            else:
                y = -float(node["generated_step"] + 1)
            positions[node_id] = (x, y)
            visited.add(node_id)
        return leaf_x + 0.75

    for root_id in roots:
        if root_id not in visited:
            next_x = place_root(root_id, next_x)
    for node_id in nodes:
        if node_id not in visited:
            next_x = place_root(node_id, next_x)
    return positions


def _fmt(value):
    return "?" if value is None else f"{value:.6g}"


def plot_tree(run_dir, output=None, step=None, max_step=None, max_children=16,
              valid_only=False, labels="auto", title=None, view="auto"):
    run_dir = Path(run_dir)
    config, problem, metric_name, maximize = _run_info(run_dir)
    nodes, edges, sampler_types = load_tree_events(run_dir)
    actual_view = ("expanded" if step is not None else "overview") if view == "auto" else view
    if actual_view == "overview":
        shown, shown_edges, omitted = overview_tree(
            nodes, edges, step=step, max_step=max_step, valid_only=valid_only)
    else:
        shown, shown_edges, omitted = visible_tree(
            nodes, edges, step=step, max_step=max_step,
            max_children=max_children, valid_only=valid_only)
    if not shown:
        raise ValueError("the requested filters leave no nodes to plot")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        from matplotlib.colors import Normalize
    except ImportError as exc:
        raise ValueError(
            "matplotlib is required; install it with `python3 -m pip install matplotlib`"
        ) from exc

    positions = tree_layout(shown, shown_edges)
    node_count = len(shown)
    fig_width = min(24.0, max(13.0, 8.0 + math.sqrt(node_count) * 0.32))
    max_event_step = max((e["step"] for e in shown_edges), default=0)
    fig_height = min(24.0, max(7.5, 5.5 + max_event_step * 0.24))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=150,
                           facecolor="none")
    fig.subplots_adjust(left=0.075, right=0.91, top=0.86, bottom=0.08)
    ax.set_facecolor("none")
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)

    for edge in shown_edges:
        x0, y0 = positions[edge["parent"]]
        x1, y1 = positions[edge["child"]]
        ax.plot([x0, x1], [y0, y1], color="#c8c6c0", lw=0.65,
                alpha=0.7, zorder=1)

    score_key = "reward" if metric_name == "reward" else "raw_score"
    score_nodes = [node for node in shown.values()
                   if node["valid"] is not False and node[score_key] is not None]
    scores = [node[score_key] for node in score_nodes]
    if scores:
        low, high = min(scores), max(scores)
        if math.isclose(low, high):
            low, high = low - 0.5, high + 0.5
        norm = Normalize(low, high)
    else:
        norm = Normalize(0.0, 1.0)

    valid_ids = [nid for nid, node in shown.items()
                 if node["valid"] is not False and node[score_key] is not None]
    if valid_ids:
        scatter = ax.scatter(
            [positions[nid][0] for nid in valid_ids],
            [positions[nid][1] for nid in valid_ids],
            c=[shown[nid][score_key] for nid in valid_ids],
            cmap="viridis", norm=norm, s=34, edgecolors=SURFACE,
            linewidths=0.65, zorder=3,
        )
        colorbar = fig.colorbar(scatter, ax=ax, pad=0.012, shrink=0.82)
        colorbar.set_label(f"raw {metric_name}" if score_key == "raw_score" else "reward",
                           color=INK_SOFT)
        colorbar.ax.tick_params(colors=INK_SOFT, labelsize=8)

    invalid_ids = [nid for nid, node in shown.items() if node["valid"] is False]
    if invalid_ids:
        ax.scatter([positions[nid][0] for nid in invalid_ids],
                   [positions[nid][1] for nid in invalid_ids],
                   marker="x", c=GRAY, s=24, linewidths=0.8, zorder=2)

    seed_ids = [nid for nid, node in shown.items() if node["is_seed"]]
    if seed_ids:
        ax.scatter([positions[nid][0] for nid in seed_ids],
                   [positions[nid][1] for nid in seed_ids],
                   marker="D", facecolors=SURFACE, edgecolors=INK_SOFT,
                   s=72, linewidths=1.3, zorder=5)

    def selection_steps_in_scope(node):
        if step is not None:
            return [step] if step in node["selected_steps"] else []
        values = sorted(node["selected_steps"])
        return ([value for value in values if value <= max_step]
                if max_step is not None else values)

    selected_ids = [nid for nid, node in shown.items()
                    if selection_steps_in_scope(node)]
    if selected_ids:
        ax.scatter([positions[nid][0] for nid in selected_ids],
                   [positions[nid][1] for nid in selected_ids],
                   marker="s", facecolors="none", edgecolors=ORANGE,
                   s=115, linewidths=1.5, zorder=6)

    best_id = None
    if valid_ids:
        choose = max if maximize else min
        best_id = choose(valid_ids, key=lambda nid: shown[nid][score_key])
        bx, by = positions[best_id]
        ax.scatter([bx], [by], facecolors="none", edgecolors=RED,
                   s=245, linewidths=2.0, zorder=7)

    if labels == "all":
        label_ids = set(shown)
    elif labels == "none":
        label_ids = set()
    elif labels == "selected":
        label_ids = set(selected_ids)
        if best_id:
            label_ids.add(best_id)
    elif node_count <= 24:
        label_ids = set(shown)
    elif actual_view == "overview":
        # One numeric callout per generation step is enough to read progress;
        # every other raw score remains encoded by the shared color scale.
        label_ids = {best_id} if best_id else set()
        by_generated_step = defaultdict(list)
        for node_id, node in shown.items():
            if node["generated_step"] is not None and node["reward"] is not None:
                by_generated_step[node["generated_step"]].append(node_id)
        for node_ids in by_generated_step.values():
            label_ids.add(max(node_ids, key=lambda nid: shown[nid]["reward"]))
    else:
        label_ids = set(selected_ids)
        if best_id:
            label_ids.add(best_id)
        by_parent = defaultdict(list)
        for edge in shown_edges:
            by_parent[edge["parent"]].append(edge["child"])
        for child_ids in by_parent.values():
            scored = [nid for nid in child_ids if shown[nid]["reward"] is not None]
            if scored:
                label_ids.add(max(scored, key=lambda nid: shown[nid]["reward"]))

    focus_step = step if step is not None else max_step
    for node_id in label_ids:
        node = shown[node_id]
        x, y = positions[node_id]
        score = node[score_key]
        kind = ("seed" if node["is_seed"] else
                ("parent" if node["generated_step"] is None
                 else f"s{node['generated_step']}"))
        parts = [kind,
                 f"raw={_fmt(score)}" if score_key == "raw_score" else f"r={_fmt(score)}"]
        scoped_selections = selection_steps_in_scope(node)
        selection_step = (focus_step if focus_step in scoped_selections
                          else max(scoped_selections, default=None))
        if selection_step is not None:
            selection_score = node["selection"][selection_step].get("score")
            parts.append(f"pick={_fmt(selection_score)}")
        ax.annotate("\n".join(parts), (x, y), xytext=(3, 7),
                    textcoords="offset points", fontsize=6.2,
                    color=INK, ha="left", va="bottom", zorder=8,
                    bbox=dict(boxstyle="round,pad=0.16", fc=SURFACE,
                              ec=GRID, lw=0.5, alpha=0.88))

    for parent_id, count in omitted.items():
        if actual_view != "expanded" or step is None:
            continue
        if count <= 0 or parent_id not in positions:
            continue
        x, y = positions[parent_id]
        ax.annotate(f"+{count} generated", (x, y), xytext=(5, -13),
                    textcoords="offset points", fontsize=6.5,
                    color=INK_SOFT, ha="left", va="top")

    sampler_label = ", ".join(sorted(sampler_types)) or str(
        config.get("sampler", "legacy/unknown"))
    scope = f"step {step}" if step is not None else (
        f"through step {max_step}" if max_step is not None else "full run")
    view_label = "selected-lineage overview" if actual_view == "overview" else "expanded children"
    direction = "maximize" if maximize else "minimize"
    fig.suptitle(title or f"{problem}: post-run search tree",
                 x=0.075, y=0.955, ha="left", fontsize=14,
                 color=INK, fontweight="bold")
    fig.text(0.075, 0.91,
             f"{sampler_label} · {scope} · {view_label} · "
             f"{len(shown_edges):,} shown edges / "
             f"{len(edges):,} saved rollouts · raw {metric_name} ({direction})",
             ha="left", fontsize=9.2, color=INK_SOFT)

    handles = [
        Line2D([], [], marker="o", ls="", mfc=BLUE, mec=SURFACE,
               label="valid generated child"),
        Line2D([], [], marker="x", ls="", color=GRAY,
               label="invalid generated child"),
        Line2D([], [], marker="s", ls="", mfc="none", mec=ORANGE,
               mew=1.5, label="selected as parent"),
        Line2D([], [], marker="D", ls="", mfc=SURFACE, mec=INK_SOFT,
               label="seed parent"),
        Line2D([], [], marker="o", ls="", mfc="none", mec=RED,
               mew=2, label=f"best raw {metric_name}"),
    ]
    ax.legend(handles=handles, loc="upper right", frameon=True,
              facecolor=SURFACE, edgecolor=GRID, fontsize=8.2)
    ax.set_ylabel("generation step", fontsize=10, color=INK)
    ax.set_xticks([])
    event_steps = sorted({edge["step"] for edge in shown_edges})
    ticks = [0.0] + [-float(value + 1) for value in event_steps]
    tick_labels = ["initial"] + [str(value) for value in event_steps]
    ax.set_yticks(ticks)
    ax.set_yticklabels(tick_labels, fontsize=8, color=INK_SOFT)
    for side in ("top", "right", "bottom"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(GRAY)

    output_path = Path(output) if output else run_dir / "search_tree.png"
    if output_path.suffix.lower() != ".png":
        output_path = output_path.with_suffix(".png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = output_path.with_suffix(".pdf")
    fig.savefig(output_path, bbox_inches="tight", dpi=300, transparent=True)
    fig.savefig(pdf_path, bbox_inches="tight", transparent=True)
    plt.close(fig)

    valid_total = sum(1 for node in nodes.values() if node["valid"] is True)
    print(f"read {len(edges):,} generated children ({valid_total:,} valid) and "
          f"{sum(bool(n['selected_steps']) for n in nodes.values()):,} selected parents")
    print(f"showing {len(shown):,} nodes and {len(shown_edges):,} edges")
    print(f"wrote {output_path} and {pdf_path}")
    return output_path, pdf_path


def main():
    parser = argparse.ArgumentParser(
        description="Plot the saved sampler parent/child tree after a run.")
    parser.add_argument("run_dir", help="run directory containing stepXX folders")
    parser.add_argument("--out", default=None, help="output PNG path")
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument("--step", type=int, default=None,
                       help="show only parents and children from this step")
    scope.add_argument("--max-step", type=int, default=None,
                       help="show the tree only through this step")
    parser.add_argument("--max-children", type=int, default=16,
                        help="expanded-view children per parent; 0 shows all")
    parser.add_argument("--view", choices=("auto", "overview", "expanded"),
                        default="auto",
                        help="overview for lineage, expanded for rollout children")
    parser.add_argument("--valid-only", action="store_true",
                        help="hide invalid generated children")
    parser.add_argument("--labels", choices=("auto", "all", "selected", "none"),
                        default="auto", help="which nodes receive numeric labels")
    parser.add_argument("--title", default=None, help="custom plot title")
    args = parser.parse_args()
    if args.max_children < 0:
        parser.error("--max-children must be zero or positive")
    try:
        plot_tree(args.run_dir, output=args.out, step=args.step,
                  max_step=args.max_step, max_children=args.max_children,
                  valid_only=args.valid_only, labels=args.labels,
                  title=args.title, view=args.view)
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
