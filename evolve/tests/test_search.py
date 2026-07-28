import math, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pytest

from config import Config
from core.tree import SearchTree
from core.types import LEAF_EXPAND, VIRTUAL_EXPAND
from search.dpuct import DPUCT, VIRTUAL_KEY, softmax
from search.elo import EloRatings
from search.pairing import build_pairings
from search.signals import average_ranks, rank_signal, standardize, global_node_logits


# ---------------- tree ----------------
def test_subtree_max_and_size():
    t = SearchTree()
    root = t.add_root()
    a = t.add_child(root.id, reward=1.0, valid=True)
    b = t.add_child(root.id, reward=0.0)
    a1 = t.add_child(a.id, reward=5.0, valid=True)
    t.add_child(b.id, reward=2.0, valid=True)
    t.recompute()
    assert root.subtree_size == 5
    assert root.subtree_max == 5.0        # the exceptional descendant survives
    assert a.subtree_max == 5.0
    assert a.subtree_size == 2
    assert b.subtree_max == 2.0
    assert a1.subtree_size == 1

def test_prune_keeps_ancestors_and_roots():
    t = SearchTree(max_archive_size=3)
    root = t.add_root()
    chain = root.id
    for i in range(6):
        chain = t.add_child(chain, reward=float(i)).id
    t.recompute()
    t.prune()
    assert root.id in t
    for n in t.nodes():                    # every survivor stays attached
        assert n.is_root or n.parent_id in t

def test_roundtrip_serialization():
    t = SearchTree()
    r = t.add_root()
    c = t.add_child(r.id, reward=1.5, valid=True, code="x=1")
    t.recompute()
    t2 = SearchTree.from_dict(t.to_dict())
    assert len(t2) == 2
    assert t2.get(c.id).reward == 1.5
    assert t2.child_ids(r.id) == [c.id]


# ---------------- signals ----------------
def test_average_ranks_ties():
    assert list(average_ranks([10, 20, 20, 40])) == [1.0, 2.5, 2.5, 4.0]
    assert list(average_ranks([5, 5, 5])) == [2.0, 2.0, 2.0]

def test_standardize_degenerate_is_zero_not_nan():
    out = standardize([3.0, 3.0, 3.0])
    assert np.allclose(out, 0.0) and np.isfinite(out).all()

def test_rank_signal_is_monotone_in_reward():
    sig = rank_signal([0.1, 0.9, 0.5])
    assert sig[1] > sig[2] > sig[0]

def test_alpha_interpolates_eq3():
    r, e = [1.0, -1.0], [-1.0, 1.0]
    assert np.allclose(global_node_logits(r, e, 1.0), r)   # rank only
    assert np.allclose(global_node_logits(r, e, 0.0), e)   # elo only
    assert np.allclose(global_node_logits(r, e, 0.5), [0.0, 0.0])


# ---------------- elo ----------------
def test_elo_winner_gains_loser_loses_and_sum_conserved():
    elo = EloRatings(k_factor=24.0, initial_rating=0.0, scale=400.0)
    elo.ensure(["a", "b"])
    elo.update("a", "b", 1.0)
    assert elo.ratings["a"] > 0 > elo.ratings["b"]
    assert abs(elo.ratings["a"] + elo.ratings["b"]) < 1e-9

def test_elo_tie_between_equals_is_a_noop():
    elo = EloRatings(k_factor=24.0)
    elo.ensure(["a", "b"])
    elo.update("a", "b", 0.5)
    assert abs(elo.ratings["a"]) < 1e-9 and abs(elo.ratings["b"]) < 1e-9

def test_paper_scale_is_far_steeper_than_classic():
    steep = EloRatings(scale=1.0); steep.ensure(["a", "b"]); steep.ratings["a"] = 1.0
    classic = EloRatings(scale=400.0); classic.ensure(["a", "b"]); classic.ratings["a"] = 1.0
    assert steep.expected("a", "b") == pytest.approx(10 / 11, abs=1e-6)
    assert classic.expected("a", "b") == pytest.approx(0.5, abs=1e-2)

def test_elo_expected_never_overflows_at_scale_one():
    elo = EloRatings(scale=1.0); elo.ensure(["a", "b"])
    elo.ratings["a"], elo.ratings["b"] = 1e6, -1e6
    assert elo.expected("a", "b") == 1.0 and elo.expected("b", "a") == 0.0

def test_pairings():
    assert len(build_pairings(list("abcd"), "round_robin")) == 6
    assert len(build_pairings(list("abcd"), "neighbors")) == 3
    assert len(build_pairings(list("abcdef"), "random", num_matches=4)) == 4
    assert build_pairings(["a"], "round_robin") == []


# ---------------- d-puct ----------------
def _cfg(**kw):
    c = Config().search
    for k, v in kw.items():
        setattr(c, k, v)
    return c

def test_softmax_temperature_sharpens():
    hot = softmax([2.0, 1.0, 0.0], tau=10.0)
    cold = softmax([2.0, 1.0, 0.0], tau=0.1)
    assert cold[0] > hot[0]
    assert math.isclose(hot.sum(), 1.0) and math.isclose(cold.sum(), 1.0)

def test_virtual_logit_is_mu_plus_lambda_sigma():
    t = SearchTree()
    root = t.add_root()
    a = t.add_child(root.id, reward=1.0, valid=True)
    b = t.add_child(root.id, reward=3.0, valid=True)
    t.recompute()
    d = DPUCT(_cfg(lambda_virtual=2.0, alpha=1.0))
    logits = d.compute_logits(t)
    pp = d.parent_priors(t, logits)[root.id]
    vals = np.array([logits[a.id], logits[b.id]])
    assert pp.mu == pytest.approx(vals.mean())
    assert pp.sigma == pytest.approx(vals.std())
    assert pp.logits[VIRTUAL_KEY] == pytest.approx(vals.mean() + 2.0 * vals.std())

def test_lambda_zero_puts_virtual_at_the_child_mean():
    t = SearchTree(); root = t.add_root()
    t.add_child(root.id, reward=1.0, valid=True)
    t.add_child(root.id, reward=3.0, valid=True)
    t.recompute()
    d = DPUCT(_cfg(lambda_virtual=0.0))
    pp = d.parent_priors(t, d.compute_logits(t))[root.id]
    assert pp.logits[VIRTUAL_KEY] == pytest.approx(pp.mu)

def test_priors_are_a_distribution_per_parent():
    t = SearchTree(); root = t.add_root()
    for r in (0.5, 1.5, 2.5):
        t.add_child(root.id, reward=r, valid=True)
    t.recompute()
    d = DPUCT(_cfg())
    for pp in d.parent_priors(t, d.compute_logits(t)).values():
        assert math.isclose(sum(pp.priors.values()), 1.0, abs_tol=1e-9)

def test_roots_excluded_from_archive_logits():
    t = SearchTree(); root = t.add_root()
    c = t.add_child(root.id, reward=2.0, valid=True)
    t.recompute()
    logits = DPUCT(_cfg()).compute_logits(t)
    assert set(logits) == {c.id}          # the root's W_m would distort ranks

def test_step_zero_selects_the_root_as_a_leaf_expansion():
    t = SearchTree(); root = t.add_root()
    t.recompute()
    d = DPUCT(_cfg(n_select=8, k_children=8))
    targets = d.select(t, d.compute_logits(t))
    assert len(targets) == 1
    assert targets[0].kind == LEAF_EXPAND and targets[0].node_id == root.id
    assert targets[0].num_children == 8

def test_one_target_per_node_leaf_deepens_internal_widens():
    t = SearchTree(); root = t.add_root()
    a = t.add_child(root.id, reward=1.0, valid=True)
    t.add_child(a.id, reward=2.0, valid=True)
    t.recompute()
    d = DPUCT(_cfg(n_select=99, k_children=5))
    targets = {(x.kind, x.node_id): x for x in d.select(t, d.compute_logits(t))}
    assert len(targets) == len(t.nodes())
    assert (VIRTUAL_EXPAND, root.id) in targets      # root has a child -> widen
    assert (VIRTUAL_EXPAND, a.id) in targets         # a has a child   -> widen
    assert targets[(VIRTUAL_EXPAND, root.id)].num_children == 1
    leaves = [k for k in targets if k[0] == LEAF_EXPAND]
    assert len(leaves) == 1 and targets[leaves[0]].num_children == 5

def test_batch_size_bounds_n_to_nk():
    t = SearchTree(); root = t.add_root()
    for _ in range(4):
        kid = t.add_child(root.id, reward=1.0, valid=True)
        t.add_child(kid.id, reward=2.0, valid=True)
    t.recompute()
    n, k = 3, 7
    d = DPUCT(_cfg(n_select=n, k_children=k))
    targets = d.select(t, d.compute_logits(t))
    B = sum(x.num_children for x in targets)
    assert len(targets) == n
    assert n <= B <= n * k

def test_exploitation_ranks_leaves_by_subtree_max():
    t = SearchTree(); root = t.add_root()
    lo = t.add_child(root.id, reward=0.5, valid=True)
    hi = t.add_child(root.id, reward=2.5, valid=True)
    t.recompute()
    d = DPUCT(_cfg(n_select=99, c_puct=0.0))          # exploitation only
    by_node = {x.node_id: x for x in d.select(t, d.compute_logits(t))}
    assert by_node[hi.id].score > by_node[lo.id].score

def test_a_buried_gem_lifts_its_whole_ancestry():
    """The point of W_m over an average: one exceptional descendant keeps the
    branch attractive even though its own reward is poor."""
    t = SearchTree(); root = t.add_root()
    strong = t.add_child(root.id, reward=0.1, valid=True)
    t.add_child(strong.id, reward=9.0, valid=True)     # buried gem
    weak = t.add_child(root.id, reward=0.2, valid=True)
    t.recompute()
    assert t.get(strong.id).subtree_max == 9.0
    assert t.get(weak.id).subtree_max == 0.2
    d = DPUCT(_cfg(alpha=1.0))
    values = d._value_table(t)
    logits = d.compute_logits(t)
    assert values[strong.id] > values[weak.id]         # V(p, a) = W_m(a)
    assert logits[strong.id] > logits[weak.id]         # and so does its rank

def test_virtual_action_has_zero_value_by_default():
    t = SearchTree(); root = t.add_root()
    t.add_child(root.id, reward=5.0, valid=True)
    t.recompute()
    d = DPUCT(_cfg(n_select=99))
    tgt = [x for x in d.select(t, d.compute_logits(t)) if x.kind == VIRTUAL_EXPAND][0]
    assert tgt.value == 0.0
    assert tgt.score == pytest.approx(tgt.bonus)      # prior + bonus only

def test_c_puct_zero_removes_the_bonus():
    t = SearchTree(); root = t.add_root()
    t.add_child(root.id, reward=1.0, valid=True)
    t.recompute()
    d = DPUCT(_cfg(n_select=99, c_puct=0.0))
    assert all(x.bonus == 0.0 for x in d.select(t, d.compute_logits(t)))

def test_action_descend_mode_also_respects_bounds():
    t = SearchTree(); root = t.add_root()
    for _ in range(3):
        kid = t.add_child(root.id, reward=1.0, valid=True)
        t.add_child(kid.id, reward=2.0, valid=True)
    t.recompute()
    d = DPUCT(_cfg(n_select=3, k_children=4, selection_mode="action_descend"))
    targets = d.select(t, d.compute_logits(t))
    assert 0 < len(targets) <= 3
    B = sum(x.num_children for x in targets)
    assert len(targets) <= B <= len(targets) * 4
    assert all(x.kind in (LEAF_EXPAND, VIRTUAL_EXPAND) for x in targets)
