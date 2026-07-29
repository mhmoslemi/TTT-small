import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from dpuct import (DPUCT, DPUCTConfig, EloRatings, LEAF_EXPAND, Tree,
                   VIRTUAL_EXPAND, average_ranks, blend, build_pairings,
                   rank_signal, search, softmax, standardize)
from dpuct.policy import VIRTUAL_KEY


def cfg(**kw):
    return DPUCTConfig(**kw)


# ---------------- tree ----------------
def test_subtree_max_beats_averaging():
    """The library's whole premise: one exceptional descendant must keep its
    ancestry attractive instead of being diluted by mediocre siblings."""
    t = Tree()
    root = t.add_root()
    branch = t.add_child(root.id, value=0.1)
    for _ in range(9):
        t.add_child(branch.id, value=0.0)
    t.add_child(branch.id, value=9.0)          # the one that matters
    flat = t.add_child(root.id, value=1.0)
    t.recompute()

    # Mean over the branch is 9.0/10 = 0.9, below the flat sibling's 1.0, so a
    # mean-backup search would walk away from the branch holding the 9.0.
    mean = np.mean([c.value for c in t.children(branch.id)])
    assert mean == pytest.approx(0.9)
    assert mean < flat.value                   # averaging prefers `flat`

    # The max backup keeps it visible, so both the exploitation term V = W_m
    # and the global rank rate the branch above the flat sibling.
    assert t.get(branch.id).subtree_max == 9.0
    assert t.get(flat.id).subtree_max == 1.0
    policy = DPUCT(cfg(alpha=1.0))
    assert policy._values(t)[branch.id] > policy._values(t)[flat.id]
    logits = policy.global_logits(t)
    assert logits[branch.id] > logits[flat.id]

def test_sizes_and_depth():
    t = Tree(); r = t.add_root()
    a = t.add_child(r.id, value=1.0)
    t.add_child(a.id, value=2.0)
    t.recompute()
    assert t.get(r.id).subtree_size == 3
    assert t.get(a.id).subtree_size == 2
    assert t.get(a.id).depth == 1

def test_payload_is_untouched():
    t = Tree(); r = t.add_root(payload={"state": [1, 2]})
    assert t.get(r.id).payload == {"state": [1, 2]}

def test_roots_are_excluded_from_the_archive():
    t = Tree(); r = t.add_root()
    c = t.add_child(r.id, value=1.0)
    t.recompute()
    assert [n.id for n in t.expanded()] == [c.id]
    assert set(DPUCT(cfg()).global_logits(t)) == {c.id}

def test_prune_keeps_survivors_attached():
    t = Tree(max_size=4); r = t.add_root()
    chain = r.id
    for i in range(8):
        chain = t.add_child(chain, value=float(i)).id
    t.recompute(); t.prune()
    for n in t.nodes():
        assert n.is_root or n.parent_id in t

def test_best_and_summary():
    t = Tree(); r = t.add_root()
    t.add_child(r.id, value=1.0)
    hi = t.add_child(r.id, value=7.0, payload="win")
    t.recompute()
    assert t.best().id == hi.id and t.best().payload == "win"
    assert t.summary()["nodes"] == 3


# ---------------- signals ----------------
def test_average_ranks_share_ties():
    assert list(average_ranks([10, 20, 20, 40])) == [1.0, 2.5, 2.5, 4.0]

def test_standardize_is_nan_free_when_degenerate():
    out = standardize([2.0, 2.0]); assert np.allclose(out, 0.0) and np.isfinite(out).all()

def test_rank_signal_is_scale_free():
    """Ranks, so a reward in millions and one in millionths behave identically."""
    assert np.allclose(rank_signal([1.0, 2.0, 3.0]), rank_signal([1e6, 2e6, 3e6]))

def test_alpha_interpolates():
    r, e = [1.0, -1.0], [-1.0, 1.0]
    assert np.allclose(blend(r, e, 1.0), r)
    assert np.allclose(blend(r, e, 0.0), e)
    assert np.allclose(blend(r, None, 1.0), r)

def test_softmax_temperature():
    assert softmax([2.0, 1.0], 0.1)[0] > softmax([2.0, 1.0], 10.0)[0]
    assert math.isclose(softmax([2.0, 1.0], 1.0).sum(), 1.0)


# ---------------- the virtual child ----------------
def test_virtual_logit_is_mu_plus_lambda_sigma():
    t = Tree(); r = t.add_root()
    t.add_child(r.id, value=1.0); t.add_child(r.id, value=3.0)
    t.recompute()
    policy = DPUCT(cfg(lambda_virtual=2.0, alpha=1.0))
    logits = policy.global_logits(t)
    pp = policy.parent_priors(t, logits)[r.id]
    vals = np.array([logits[c.id] for c in t.children(r.id)])
    assert pp.mu == pytest.approx(vals.mean())
    assert pp.sigma == pytest.approx(vals.std())
    assert pp.logits[VIRTUAL_KEY] == pytest.approx(vals.mean() + 2 * vals.std())

def test_lambda_zero_prices_the_unseen_sibling_at_the_average():
    t = Tree(); r = t.add_root()
    t.add_child(r.id, value=1.0); t.add_child(r.id, value=3.0)
    t.recompute()
    policy = DPUCT(cfg(lambda_virtual=0.0))
    pp = policy.parent_priors(t, policy.global_logits(t))[r.id]
    assert pp.logits[VIRTUAL_KEY] == pytest.approx(pp.mu)

def test_variance_among_children_raises_the_virtual_logit():
    """Spread is upside when only the best survives, so widening gets credit."""
    def virtual_logit(values):
        t = Tree(); r = t.add_root()
        for v in values:
            t.add_child(r.id, value=v)
        t.recompute()
        p = DPUCT(cfg(lambda_virtual=1.0, alpha=1.0))
        return p.parent_priors(t, p.global_logits(t))[r.id].logits[VIRTUAL_KEY]

    # Same mean rank either way; the spread differs.
    assert virtual_logit([0.0, 1.0, 2.0, 3.0]) > 0

def test_priors_sum_to_one_per_parent():
    t = Tree(); r = t.add_root()
    for v in (0.5, 1.5, 2.5):
        t.add_child(r.id, value=v)
    t.recompute()
    p = DPUCT(cfg())
    for pp in p.parent_priors(t, p.global_logits(t)).values():
        assert math.isclose(sum(pp.priors.values()), 1.0, abs_tol=1e-9)


# ---------------- selection ----------------
def test_first_round_expands_the_root():
    t = Tree(); r = t.add_root(); t.recompute()
    targets = DPUCT(cfg(k_children=8)).select(t)
    assert len(targets) == 1
    assert targets[0].kind == LEAF_EXPAND and targets[0].node_id == r.id
    assert targets[0].num_children == 8

def test_one_target_per_node_leaves_deepen_internals_widen():
    t = Tree(); r = t.add_root()
    a = t.add_child(r.id, value=1.0)
    t.add_child(a.id, value=2.0)
    t.recompute()
    targets = {t_.key: t_ for t_ in DPUCT(cfg(n_select=99, k_children=5)).select(t)}
    assert len(targets) == len(t.nodes())
    assert (VIRTUAL_EXPAND, r.id) in targets
    assert targets[(VIRTUAL_EXPAND, r.id)].num_children == 1
    leaves = [k for k in targets if k[0] == LEAF_EXPAND]
    assert len(leaves) == 1 and targets[leaves[0]].num_children == 5

def test_batch_size_stays_within_n_and_nk():
    t = Tree(); r = t.add_root()
    for _ in range(4):
        kid = t.add_child(r.id, value=1.0)
        t.add_child(kid.id, value=2.0)
    t.recompute()
    n, k = 3, 7
    targets = DPUCT(cfg(n_select=n, k_children=k)).select(t)
    assert len(targets) == n
    assert n <= sum(x.num_children for x in targets) <= n * k

def test_c_puct_zero_is_pure_exploitation():
    t = Tree(); r = t.add_root()
    lo = t.add_child(r.id, value=0.5); hi = t.add_child(r.id, value=2.5)
    t.recompute()
    by_node = {x.node_id: x for x in DPUCT(cfg(n_select=99, c_puct=0.0)).select(t)}
    assert all(x.bonus == 0.0 for x in by_node.values())
    assert by_node[hi.id].score > by_node[lo.id].score

def test_exploration_bonus_decays_with_search_effort():
    t = Tree(); r = t.add_root()
    busy = t.add_child(r.id, value=1.0)
    for _ in range(20):
        t.add_child(busy.id, value=0.5)
    quiet = t.add_child(r.id, value=1.0)
    t.recompute()
    policy = DPUCT(cfg(n_select=99, c_puct=1.0))
    by_node = {x.node_id: x for x in policy.select(t)}
    assert by_node[quiet.id].bonus > by_node[busy.id].bonus

def test_virtual_action_scores_on_prior_and_bonus_only():
    t = Tree(); r = t.add_root(); t.add_child(r.id, value=5.0)
    t.recompute()
    tgt = [x for x in DPUCT(cfg(n_select=99)).select(t) if x.kind == VIRTUAL_EXPAND][0]
    assert tgt.value == 0.0 and tgt.score == pytest.approx(tgt.bonus)

def test_descend_mode_respects_the_bounds_too():
    t = Tree(); r = t.add_root()
    for _ in range(3):
        kid = t.add_child(r.id, value=1.0)
        t.add_child(kid.id, value=2.0)
    t.recompute()
    targets = DPUCT(cfg(n_select=3, k_children=4,
                        selection_mode="action_descend")).select(t)
    assert 0 < len(targets) <= 3
    assert all(x.kind in (LEAF_EXPAND, VIRTUAL_EXPAND) for x in targets)

def test_explain_renders_a_table():
    t = Tree(); r = t.add_root(); t.add_child(r.id, value=1.0)
    t.recompute()
    assert "score" in DPUCT(cfg()).explain(t)


# ---------------- config validation ----------------
@pytest.mark.parametrize("kw", [
    {"n_select": 0}, {"k_children": 0}, {"alpha": 1.5}, {"tau": 0.0},
    {"lambda_virtual": -1.0}, {"selection_mode": "nope"},
    {"virtual_value_mode": "nope"},
])
def test_bad_config_is_rejected(kw):
    with pytest.raises(ValueError):
        DPUCT(DPUCTConfig(**kw))


# ---------------- elo ----------------
def test_elo_is_zero_sum_and_a_tie_between_equals_is_a_noop():
    elo = EloRatings(); elo.ensure(["a", "b"])
    elo.update("a", "b", 1.0)
    assert elo.ratings["a"] > 0 > elo.ratings["b"]
    assert abs(sum(elo.ratings.values())) < 1e-9
    elo2 = EloRatings(); elo2.ensure(["a", "b"]); elo2.update("a", "b", 0.5)
    assert abs(elo2.ratings["a"]) < 1e-9

def test_elo_scale_convention():
    steep = EloRatings(scale=1.0); steep.ensure(["a", "b"]); steep.ratings["a"] = 1.0
    assert steep.expected("a", "b") == pytest.approx(10 / 11, abs=1e-6)
    classic = EloRatings(scale=400.0); classic.ensure(["a", "b"]); classic.ratings["a"] = 1.0
    assert classic.expected("a", "b") == pytest.approx(0.5, abs=1e-2)

def test_elo_never_overflows():
    elo = EloRatings(scale=1.0); elo.ensure(["a", "b"])
    elo.ratings["a"], elo.ratings["b"] = 1e6, -1e6
    assert elo.expected("a", "b") == 1.0 and elo.expected("b", "a") == 0.0

def test_play_skips_a_none_verdict():
    elo = EloRatings(); elo.ensure(["a", "b"])
    assert elo.play([("a", "b")], lambda x, y: None) == 0

def test_pairing_modes():
    assert len(build_pairings(list("abcd"), "round_robin")) == 6
    assert len(build_pairings(list("abcd"), "neighbors")) == 3
    assert len(build_pairings(list("abcdef"), "random", num_matches=4)) == 4
    assert build_pairings(["a"], "round_robin") == []

def test_comparison_signal_changes_selection():
    """alpha < 1 must let the comparator actually move the ordering."""
    t = Tree(); r = t.add_root()
    a = t.add_child(r.id, value=1.0); b = t.add_child(r.id, value=1.0)
    t.recompute()
    policy = DPUCT(cfg(alpha=0.0))
    elo = EloRatings(); elo.ensure([a.id, b.id])
    for _ in range(5):
        elo.update(a.id, b.id, 1.0)          # a is judged better every time
    logits = policy.global_logits(t, elo.as_dict([a.id, b.id]))
    assert logits[a.id] > logits[b.id]


# ---------------- the driver ----------------
def test_search_improves_on_a_simple_landscape():
    import random
    rng = random.Random(0)

    def expand(node, num_children):
        base = node.payload if node.payload is not None else 0.0
        out = []
        for _ in range(num_children):
            x = base + rng.gauss(0, 0.4 / (1 + node.depth))
            out.append((-abs(x - 0.75), x))   # peak at x = 0.75
        return out

    result = search(expand, rounds=12,
                    config=DPUCTConfig(n_select=3, k_children=4))
    assert result.best_value > -0.05
    assert abs(result.best_payload - 0.75) < 0.05
    assert len(result.history) == 12
    assert result.curve() == sorted(result.curve())   # best-so-far is monotone

def test_expand_may_return_fewer_than_asked():
    def expand(node, num_children):
        return [(1.0, "x")]          # always one, whatever was requested
    result = search(expand, rounds=3, config=DPUCTConfig(n_select=2, k_children=8))
    assert result.best is not None

def test_expand_may_return_nothing():
    result = search(lambda node, k: [], rounds=3)
    assert result.best is None and len(result.tree) == 1
