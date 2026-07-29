"""
Drawing the tree.

Four renderers, in increasing order of dependency:

    render_text   ASCII/box-drawing. No dependencies, works in a terminal.
    to_mermaid    Mermaid source; GitHub and many notebooks render it inline.
    to_dot        Graphviz source, for `dot -Tpng`.
    draw          a matplotlib figure, if matplotlib is installed.

All four take the same view controls, because a search tree gets unreadable
fast: `max_depth` and `max_children` trim the display (keeping the *best*
children, not the first ones), and truncation is always shown rather than
silently hiding nodes.

What each node shows by default is the triple D-PUCT actually selects on:

    Q  the node's own value
    W  W_m, the subtree max -- what the policy exploits
    m  m_s, the subtree size -- what damps the exploration bonus

Seeing W diverge from Q down a branch is seeing the max backup work.
"""

from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

from .tree import Node, Tree

# Marks
MARK_BEST = "*"
MARK_TARGET = ">"
MARK_PLAIN = "o"
MARK_ROOT = "@"


def _default_label(node: Node) -> str:
    """Q / W / m -- the three numbers selection is computed from."""
    return f"Q={node.value:.4g} W={node.subtree_max:.4g} m={node.subtree_size}"


def path_to_best(tree: Tree) -> Set[str]:
    """The best node and every ancestor of it. Handy as a highlight set."""
    best = tree.best()
    if best is None:
        return set()
    return {best.id, *tree.ancestors(best.id)}


def _visible_children(tree: Tree, node_id: str, max_children: Optional[int]
                      ) -> Tuple[List[str], int]:
    """Best-first children, trimmed. Returns (shown, hidden_count)."""
    kids = tree.child_ids(node_id)
    if max_children is None or len(kids) <= max_children:
        return kids, 0
    ranked = sorted(kids, key=lambda c: tree.get(c).subtree_max, reverse=True)
    return ranked[:max_children], len(kids) - max_children


# ======================================================================
# Text
# ======================================================================
def render_text(tree: Tree, root_id: Optional[str] = None,
                max_depth: Optional[int] = None,
                max_children: Optional[int] = None,
                label: Optional[Callable[[Node], str]] = None,
                highlight: Optional[Sequence[str]] = None,
                targets: Optional[Sequence] = None,
                mark_best: bool = True) -> str:
    """
    Box-drawing rendering of the tree.

        print(render_text(tree, max_depth=3, max_children=3))

    `targets` accepts what DPUCT.select() returned, so you can see which nodes
    the policy just picked. `highlight` is a plain id list -- path_to_best(tree)
    is a good one.
    """
    if len(tree) == 0:
        return "(empty tree)"

    label = label or _default_label
    highlight = set(highlight or ())
    target_ids = {getattr(t, "node_id", t) for t in (targets or ())}
    target_kind = {getattr(t, "node_id", t): getattr(t, "kind", "")
                   for t in (targets or ())}
    best_id = tree.best().id if (mark_best and tree.best()) else None

    roots = [root_id] if root_id else [r.id for r in tree.roots()]
    lines: List[str] = []

    # Explicit stack, so a deep chain cannot blow the recursion limit. Entries
    # are either a node to render or a literal line (used for the "... N more"
    # markers, which have to come AFTER the siblings they summarize).
    for root in roots:
        stack: List[tuple] = [("node", root, "", True, 0, True)]
        while stack:
            item = stack.pop()
            if item[0] == "line":
                lines.append(item[1])
                continue
            _, node_id, prefix, is_last, depth, is_root = item
            node = tree.get(node_id)

            if is_root:
                connector, child_prefix = "", ""
                mark = MARK_ROOT
            else:
                connector = "`-- " if is_last else "|-- "
                child_prefix = prefix + ("    " if is_last else "|   ")
                mark = MARK_PLAIN

            if node_id == best_id:
                mark = MARK_BEST
            if node_id in target_ids:
                mark = MARK_TARGET

            text = f"{prefix}{connector}{mark} {label(node)}"
            if node_id in target_ids:
                text += f"   <- selected ({target_kind[node_id]})"
            elif node_id in highlight:
                text += "   *"
            lines.append(text)

            if max_depth is not None and depth >= max_depth:
                hidden = len(tree.child_ids(node_id))
                if hidden:
                    lines.append(f"{child_prefix}`-- ... {hidden} more below")
                continue

            shown, hidden = _visible_children(tree, node_id, max_children)
            # Pushed first so it pops last and prints below the siblings it
            # summarizes; that also makes it the one carrying the `-- corner.
            if hidden:
                stack.append(
                    ("line", f"{child_prefix}`-- ... {hidden} weaker sibling(s)"))
            # Reversed, because a stack pops last-in first.
            for index, child in enumerate(reversed(shown)):
                last = (index == 0) and not hidden
                stack.append(("node", child, child_prefix, last, depth + 1, False))

    legend = (f"\n{MARK_ROOT} root   {MARK_BEST} best   "
              f"{MARK_TARGET} selected   {MARK_PLAIN} node")
    return "\n".join(lines) + legend


# ======================================================================
# Mermaid / Graphviz
# ======================================================================
def _short(node_id: str) -> str:
    return node_id[:6]


def to_mermaid(tree: Tree, max_depth: Optional[int] = None,
               max_children: Optional[int] = None,
               label: Optional[Callable[[Node], str]] = None,
               targets: Optional[Sequence] = None) -> str:
    """
    Mermaid `graph TD` source. GitHub renders this inline in markdown, and so
    do many notebook viewers.
    """
    label = label or (lambda n: f"Q={n.value:.3g}<br/>W={n.subtree_max:.3g}")
    target_ids = {getattr(t, "node_id", t) for t in (targets or ())}
    best = tree.best()
    best_id = best.id if best else None

    lines = ["graph TD"]
    styled: List[str] = []
    for node_id, depth in _walk(tree, max_depth, max_children):
        node = tree.get(node_id)
        lines.append(f'  {_short(node_id)}["{label(node)}"]')
        if node.is_root:
            styled.append(f"  style {_short(node_id)} fill:#e8e8e8,stroke:#666")
        elif node_id == best_id:
            styled.append(f"  style {_short(node_id)} fill:#c8e6c9,stroke:#2e7d32,stroke-width:3px")
        elif node_id in target_ids:
            styled.append(f"  style {_short(node_id)} fill:#fff3cd,stroke:#f57f17,stroke-width:2px")
        if node.parent_id:
            lines.append(f"  {_short(node.parent_id)} --> {_short(node_id)}")
    return "\n".join(lines + styled)


def to_dot(tree: Tree, max_depth: Optional[int] = None,
           max_children: Optional[int] = None,
           label: Optional[Callable[[Node], str]] = None,
           targets: Optional[Sequence] = None) -> str:
    """
    Graphviz source.

        Path("tree.dot").write_text(to_dot(tree))
        # dot -Tpng tree.dot -o tree.png
    """
    label = label or (lambda n: f"Q={n.value:.3g}\\nW={n.subtree_max:.3g}\\nm={n.subtree_size}")
    target_ids = {getattr(t, "node_id", t) for t in (targets or ())}
    best = tree.best()
    best_id = best.id if best else None

    lines = ["digraph dpuct {", "  rankdir=TB;",
             '  node [shape=box, style="rounded,filled", fontname="monospace", '
             'fontsize=10, fillcolor="#ffffff"];']
    for node_id, depth in _walk(tree, max_depth, max_children):
        node = tree.get(node_id)
        attrs = [f'label="{label(node)}"']
        if node.is_root:
            attrs.append('fillcolor="#e8e8e8"')
        elif node_id == best_id:
            attrs += ['fillcolor="#c8e6c9"', 'color="#2e7d32"', "penwidth=3"]
        elif node_id in target_ids:
            attrs += ['fillcolor="#fff3cd"', 'color="#f57f17"', "penwidth=2"]
        lines.append(f'  "{_short(node_id)}" [{", ".join(attrs)}];')
        if node.parent_id:
            lines.append(f'  "{_short(node.parent_id)}" -> "{_short(node_id)}";')
    lines.append("}")
    return "\n".join(lines)


def _walk(tree: Tree, max_depth: Optional[int], max_children: Optional[int]):
    """Iterative pre-order over the visible subset. Yields (node_id, depth)."""
    for root in tree.roots():
        stack = [(root.id, 0)]
        while stack:
            node_id, depth = stack.pop()
            yield node_id, depth
            if max_depth is not None and depth >= max_depth:
                continue
            shown, _ = _visible_children(tree, node_id, max_children)
            stack.extend((c, depth + 1) for c in reversed(shown))


# ======================================================================
# Matplotlib
# ======================================================================
def layout(tree: Tree, max_depth: Optional[int] = None,
           max_children: Optional[int] = None) -> Dict[str, Tuple[float, float]]:
    """
    Tidy-tree coordinates: leaves get consecutive x, parents sit centred over
    their children, y is negative depth. Iterative, so a deep chain cannot
    exhaust the recursion limit.
    """
    positions: Dict[str, Tuple[float, float]] = {}
    next_leaf_x = [0.0]

    for root in tree.roots():
        # Post-order via two stacks, so children are placed before parents.
        order: List[Tuple[str, int]] = []
        stack = [(root.id, 0)]
        while stack:
            node_id, depth = stack.pop()
            order.append((node_id, depth))
            if max_depth is not None and depth >= max_depth:
                continue
            shown, _ = _visible_children(tree, node_id, max_children)
            stack.extend((c, depth + 1) for c in shown)

        for node_id, depth in reversed(order):
            visible = [c for c in tree.child_ids(node_id) if c in positions]
            if visible:
                xs = [positions[c][0] for c in visible]
                x = sum(xs) / len(xs)
            else:
                x = next_leaf_x[0]
                next_leaf_x[0] += 1.0
            positions[node_id] = (x, -float(depth))
    return positions


def draw(tree: Tree, ax=None, max_depth: Optional[int] = None,
         max_children: Optional[int] = None,
         targets: Optional[Sequence] = None,
         annotate: bool = False, cmap: str = "viridis",
         figsize: Tuple[float, float] = (12.0, 6.0), title: Optional[str] = None):
    """
    Plot the tree. Node colour is the node's value, node size grows with its
    subtree; the best node is ringed, selected targets are squared.

        draw(tree, targets=policy.select(tree))

    Returns the matplotlib Axes. Raises ImportError if matplotlib is missing --
    use render_text or to_dot instead.
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except ImportError as e:
        raise ImportError(
            "draw() needs matplotlib (pip install matplotlib). "
            "render_text(), to_dot() and to_mermaid() have no dependencies."
        ) from e

    if len(tree) == 0:
        raise ValueError("cannot draw an empty tree")

    positions = layout(tree, max_depth, max_children)
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    # Edges first, so nodes sit on top of them.
    for node_id in positions:
        parent = tree.get(node_id).parent_id
        if parent in positions:
            x0, y0 = positions[parent]
            x1, y1 = positions[node_id]
            ax.plot([x0, x1], [y0, y1], color="#bbbbbb", lw=0.8, zorder=1)

    ids = list(positions)
    xs = [positions[i][0] for i in ids]
    ys = [positions[i][1] for i in ids]
    values = [tree.get(i).value for i in ids]
    sizes = [30 + 55 * (tree.get(i).subtree_size ** 0.5) for i in ids]

    scatter = ax.scatter(xs, ys, c=values, s=sizes, cmap=cmap, zorder=2,
                         edgecolors="#ffffff", linewidths=0.6)

    best = tree.best()
    if best and best.id in positions:
        bx, by = positions[best.id]
        ax.scatter([bx], [by], s=340, facecolors="none", edgecolors="#d32f2f",
                   linewidths=2.2, zorder=3)

    target_ids = {getattr(t, "node_id", t) for t in (targets or ())}
    shown_targets = [i for i in target_ids if i in positions]
    if shown_targets:
        ax.scatter([positions[i][0] for i in shown_targets],
                   [positions[i][1] for i in shown_targets],
                   s=240, marker="s", facecolors="none", edgecolors="#f57f17",
                   linewidths=2.0, zorder=3)

    if annotate:
        for node_id in ids:
            x, y = positions[node_id]
            ax.annotate(f"{tree.get(node_id).value:.2f}", (x, y),
                        textcoords="offset points", xytext=(0, 9),
                        ha="center", fontsize=7, color="#444444")

    handles = [Line2D([], [], marker="o", ls="", mfc="none", mec="#d32f2f",
                      ms=11, mew=2, label="best")]
    if shown_targets:
        handles.append(Line2D([], [], marker="s", ls="", mfc="none",
                              mec="#f57f17", ms=11, mew=2, label="selected"))
    ax.legend(handles=handles, loc="upper right", frameon=False, fontsize=9)

    plt.colorbar(scatter, ax=ax, label="node value", shrink=0.8)
    ax.set_ylabel("depth")
    ax.set_xticks([])
    depths = sorted({int(-y) for y in ys})
    ax.set_yticks([-d for d in depths])
    ax.set_yticklabels([str(d) for d in depths])
    ax.set_title(title or f"dpuct tree - {len(tree)} nodes, "
                          f"best {best.value:.4g}" if best else "dpuct tree")
    for side in ("top", "right", "bottom"):
        ax.spines[side].set_visible(False)
    return ax
