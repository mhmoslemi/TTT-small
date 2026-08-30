import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from dpuct import (DPUCT, DPUCTConfig, Tree, path_to_best, render_text,
                   to_dot, to_mermaid)
from dpuct.viz import layout


def sample_tree():
    t = Tree()
    r = t.add_root(payload="start")
    a = t.add_child(r.id, value=0.1, payload="A")
    for _ in range(4):
        t.add_child(a.id, value=0.0)
    t.add_child(a.id, value=9.0, payload="gem")
    b = t.add_child(r.id, value=1.0, payload="B")
    t.add_child(b.id, value=1.1)
    t.recompute()
    return t


# ---------------- text ----------------
def test_every_node_appears():
    t = sample_tree()
    out = render_text(t)
    assert out.count("Q=") == len(t)

def test_the_best_node_is_marked():
    out = render_text(sample_tree())
    assert "* Q=9" in out

def test_the_stats_selection_uses_are_shown():
    """Q, W and m -- seeing W diverge from Q is seeing the max backup work."""
    out = render_text(sample_tree())
    assert "Q=0.1 W=9" in out          # own value 0.1, subtree max 9

def test_max_depth_truncates_and_says_so():
    out = render_text(sample_tree(), max_depth=1)
    assert "more below" in out
    assert "Q=9" not in out            # the depth-2 gem is hidden

def test_max_children_keeps_the_best_not_the_first():
    """Trimming must not hide the gem just because it was added last."""
    out = render_text(sample_tree(), max_children=1)
    assert "Q=9" in out
    assert "weaker sibling" in out

def test_the_truncation_marker_comes_after_the_siblings():
    lines = render_text(sample_tree(), max_children=2).splitlines()
    kept = next(i for i, l in enumerate(lines) if "Q=9" in l)
    marker = next(i for i, l in enumerate(lines) if "weaker sibling" in l)
    assert marker > kept
    # Not .strip(): the indent is "|   ", and "|" is not whitespace.
    assert "`--" in lines[marker]                    # carries the corner

def test_selected_targets_are_shown():
    t = sample_tree()
    targets = DPUCT(DPUCTConfig(n_select=2, alpha=1.0)).select(t)
    out = render_text(t, targets=targets)
    assert "<- selected" in out
    assert out.count("<- selected") == len(targets)

def test_highlight_and_path_to_best():
    t = sample_tree()
    path = path_to_best(t)
    assert len(path) == 3                       # gem, A, root
    assert "*" in render_text(t, highlight=path)

def test_custom_label():
    t = sample_tree()
    out = render_text(t, label=lambda n: str(n.payload))
    assert "gem" in out and "start" in out

def test_empty_tree():
    assert render_text(Tree()) == "(empty tree)"

def test_a_deep_chain_does_not_recurse():
    """Iterative on purpose: a long lineage would exhaust the stack otherwise."""
    t = Tree()
    cur = t.add_root().id
    for i in range(3000):
        cur = t.add_child(cur, value=float(i)).id
    t.recompute()
    assert render_text(t, max_depth=4).count("Q=") == 5

def test_tree_render_and_show_methods(capsys):
    t = sample_tree()
    assert t.render() == render_text(t)
    t.show()
    assert "Q=" in capsys.readouterr().out


# ---------------- mermaid / dot ----------------
def test_mermaid_structure():
    t = sample_tree()
    out = to_mermaid(t)
    assert out.startswith("graph TD")
    assert out.count("-->") == len(t) - 1       # one edge per non-root
    assert "style" in out                       # best node gets styled

def test_dot_structure():
    t = sample_tree()
    out = to_dot(t)
    assert out.startswith("digraph dpuct {") and out.rstrip().endswith("}")
    assert out.count("->") == len(t) - 1

def test_dot_and_mermaid_respect_the_view_limits():
    t = sample_tree()
    assert to_dot(t, max_depth=1).count("->") == 2
    assert to_mermaid(t, max_depth=1).count("-->") == 2

def test_targets_are_styled():
    """A target that is neither the root nor the best node: those two have
    their own styles, which take precedence."""
    t = sample_tree()
    plain = next(n for n in t.expanded()
                 if n.value == 1.1)                  # not root, not best
    assert "f57f17" in to_dot(t, targets=[plain.id])
    assert "f57f17" in to_mermaid(t, targets=[plain.id])


# ---------------- layout ----------------
def test_layout_places_every_node_with_depth_as_y():
    t = sample_tree()
    pos = layout(t)
    assert len(pos) == len(t)
    for node in t.nodes():
        assert pos[node.id][1] == -node.depth

def test_parents_sit_centred_over_their_children():
    t = Tree(); r = t.add_root()
    a = t.add_child(r.id); b = t.add_child(r.id)
    t.recompute()
    pos = layout(t)
    assert pos[r.id][0] == pytest.approx((pos[a.id][0] + pos[b.id][0]) / 2)

def test_layout_is_iterative_for_deep_trees():
    t = Tree()
    cur = t.add_root().id
    for _ in range(3000):
        cur = t.add_child(cur).id
    t.recompute()
    assert len(layout(t)) == len(t)


# ---------------- matplotlib ----------------
def test_draw_reports_a_missing_dependency_clearly():
    pytest.importorskip  # noqa
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError, match="matplotlib"):
            sample_tree().draw()
    else:
        import matplotlib
        matplotlib.use("Agg")
        ax = sample_tree().draw()
        assert ax is not None

def test_draw_rejects_an_empty_tree():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    with pytest.raises(ValueError, match="empty"):
        Tree().draw()
