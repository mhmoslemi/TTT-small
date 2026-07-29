"""Configuration for the D-PUCT selection policy."""

from dataclasses import dataclass

LEAF_EXPAND = "leaf"
VIRTUAL_EXPAND = "virtual"

SELECTION_NODE = "node"
SELECTION_ACTION_DESCEND = "action_descend"


@dataclass
class DPUCTConfig:
    """
    Every knob, with the symbol it corresponds to in the equations.

    The defaults are a reasonable starting point for a search where you keep
    the single best result. Tune `c_puct` first -- it is the explore/exploit
    dial and the one that actually matters.
    """

    # --- batch shape -------------------------------------------------
    n_select: int = 8
    """n: how many targets to return per round."""

    k_children: int = 8
    """k: children generated when a leaf is selected. A selected virtual action
    always produces exactly 1, so a round's batch lands in [n, n*k]."""

    # --- the score, Eq. 6 --------------------------------------------
    c_puct: float = 1.0
    """c: exploration strength. 0 makes selection purely greedy on W_m."""

    normalize_exploitation: bool = True
    """Rescale W_m to [0, 1] by the archive spread before adding the bonus.

    V = W_m is in your reward units while the prior is a probability, so
    without this `c` has to be retuned for every problem. Set False for the
    literal equation."""

    # --- the prior, Eq. 3-5 ------------------------------------------
    alpha: float = 0.5
    """Mix between the two global signals: 1.0 = subtree-rank only,
    0.0 = comparison (Elo) only. Ignored when no ratings are supplied."""

    lambda_virtual: float = 1.0
    """lambda: optimism of the virtual child, mu_L(p) + lambda * sigma_L(p).

    0 prices an unseen sibling at the average of its siblings. Larger values
    favour parents whose children came out *varied*, on the logic that spread
    means untapped upside when you only keep the best."""

    tau: float = 1.0
    """tau: temperature of the parent-local softmax. Small concentrates the
    prior on the best action; large flattens it."""

    virtual_value_mode: str = "zero"
    """V(p, s-hat) for the virtual action: "zero" (its score is prior plus
    bonus only) or "parent_mean" (the mean V of its existing siblings)."""

    # --- how targets are enumerated ----------------------------------
    selection_mode: str = SELECTION_NODE
    """
    "node": every node offers exactly one target -- a leaf expands into k
        children, a node that already has children offers its virtual action.
        An internal node is never itself a target, so the batch size bound
        holds exactly and there is no ambiguous case.

    "action_descend": score (parent, action) pairs and walk down through a
        chosen internal child until reaching a leaf or a virtual action.
        Closer to textbook MCTS descent.
    """

    # --- archive ------------------------------------------------------
    max_archive_size: int = 0
    """Cap on |D|; 0 means unbounded."""

    def validate(self) -> "DPUCTConfig":
        if self.n_select < 1:
            raise ValueError("n_select must be >= 1")
        if self.k_children < 1:
            raise ValueError("k_children must be >= 1")
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        if self.lambda_virtual < 0:
            raise ValueError("lambda_virtual must be >= 0")
        if self.tau <= 0:
            raise ValueError("tau must be > 0")
        if self.selection_mode not in (SELECTION_NODE, SELECTION_ACTION_DESCEND):
            raise ValueError(
                f"selection_mode must be {SELECTION_NODE!r} or "
                f"{SELECTION_ACTION_DESCEND!r}, got {self.selection_mode!r}")
        if self.virtual_value_mode not in ("zero", "parent_mean"):
            raise ValueError(
                f"virtual_value_mode must be 'zero' or 'parent_mean', "
                f"got {self.virtual_value_mode!r}")
        return self
