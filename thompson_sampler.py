"""Problem-independent, hierarchical Thompson parent selection.

Each edge is a genealogy branch, not a problem-specific construction family.
All evaluated attempts contribute to a Beta validity model. Valid child-minus-
parent rewards contribute to a Normal-Inverse-Gamma model of improvement.
New nodes borrow a weak empirical prior from outside their own subtree; their
real observation counts remain zero. This is a search heuristic under a pooled
gain model, not a claim that adaptive LLM rollouts are stationary or Gaussian.

At each tree level draw one model per action and maximize the expected gain of
the next group of K rollouts above the best reward seen. Reuse those draws for
the batch. H pending rollouts in a branch change its marginal utility to
integral F(x)^H * (1-F(x)^K) dx above the incumbent. Pending work never becomes
fake reward evidence. Child branches widen as ceil(sqrt(1+n/K)); widening is
relaxed only when necessary to fill the requested batch with distinct parents.

Only scalar sufficient statistics and tree links are retained for pruned nodes.
There are no extra LLM calls, evaluations, GPU work, or problem descriptors.
"""

from copy import deepcopy
from dataclasses import asdict, dataclass, field
from functools import lru_cache
import json
import math
from pathlib import Path

import numpy as np

from sampler import PUCTSampler, State


@dataclass
class OutcomeStats:
    attempts: int = 0
    valid: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def add(self, gain, valid):
        self.attempts += 1
        if valid:
            self.valid += 1
            delta = gain - self.mean
            self.mean += delta / self.valid
            self.m2 += delta * (gain - self.mean)

    def without(self, other):
        """Subtract disjoint sufficient statistics without counting them twice."""
        attempts = self.attempts - other.attempts
        count = self.valid - other.valid
        if attempts < 0 or count < 0 or count > attempts:
            raise ValueError("inconsistent Thompson subtree statistics")
        if not count:
            return OutcomeStats(attempts=attempts)
        mean = self.mean + (self.mean - other.mean) * other.valid / count
        correction = (other.mean - mean) ** 2 * count * other.valid / max(self.valid, 1)
        return OutcomeStats(attempts, count, mean,
                            max(0.0, self.m2 - other.m2 - correction))


@dataclass
class SearchNode:
    parent_id: str | None
    value: float
    direct: OutcomeStats = field(default_factory=OutcomeStats)
    subtree: OutcomeStats = field(default_factory=OutcomeStats)
    active_children: list = field(default_factory=list)


@lru_cache(maxsize=1)
def _quadrature():
    return np.polynomial.legendre.leggauss(64)


def expected_max_gain(probability, mean, sigma, threshold, count, pending=0):
    """Conditional expected-best marginal, using bounded-size quadrature.

    Invalid attempts cannot improve a valid incumbent, whatever fail_score is.
    ``threshold`` is best_reward - candidate_reward, and mean/sigma describe
    signed child-minus-parent gains. The only approximation is numerical
    integration, with negligible normal tails beyond ten standard deviations.
    """
    if count <= 0 or probability <= 0.0:
        return 0.0
    z0 = (threshold - mean) / sigma
    if z0 >= 10.0:
        return 0.0
    low = max(z0, -10.0)
    nodes, weights = _quadrature()
    half_span = (10.0 - low) / 2.0
    z = low + half_span * (nodes + 1.0)
    survival = np.fromiter((0.5 * math.erfc(float(x) / math.sqrt(2.0))
                            for x in z), dtype=np.float64, count=len(z))
    with np.errstate(divide="ignore"):
        log_cdf = np.log1p(-probability * survival)
    pending_cdf = np.exp(pending * log_cdf) if pending else np.ones_like(z)
    marginal = pending_cdf * -np.expm1(count * log_cdf)
    result = sigma * half_span * float(np.dot(weights, marginal))
    if z0 < -10.0:
        failed = 1.0 - probability
        result += sigma * (-10.0 - z0) * failed ** pending * (1.0 - failed ** count)
    return max(0.0, float(result))


class ThompsonSampler(PUCTSampler):
    def __init__(self, *args, group_size=1, seed=0, **kwargs):
        super().__init__(*args, **kwargs)
        if int(group_size) < 1:
            raise ValueError("Thompson group_size must be positive")
        self.group_size = int(group_size)
        self._rng = np.random.default_rng(seed)
        self._nodes = {}
        self._outcomes = OutcomeStats()
        self._best_reward = max(float(s.value) for s in self._states)
        values = [float(s.value) for s in self._states]
        self._initial_scale = max(float(np.std(values)),
                                  0.05 * max(abs(v) for v in values), 1e-12)
        self._selection_step = None
        self._selected = []
        for state in self._states:
            self._register(state)

    @property
    def selection_mode(self):
        return "thompson"

    def _register(self, state):
        parent_id = state.parents[0]["id"] if state.parents else None
        if parent_id is not None and parent_id not in self._nodes:
            raise ValueError("Thompson archive is missing a genealogy ancestor")
        if state.id not in self._nodes:
            self._nodes[state.id] = SearchNode(parent_id, float(state.value))

    def record_outcomes(self, parent, rewards, valids):
        rewards, valids = list(rewards), list(valids)
        if len(rewards) != len(valids):
            raise ValueError("Thompson rewards and validity must align")
        with self._lock:
            self._register(parent)
            path = []
            sid = parent.id
            while sid is not None:
                path.append(self._nodes[sid])
                sid = self._nodes[sid].parent_id
            for reward, valid in zip(rewards, valids):
                reward = float(reward)
                gain = reward - float(parent.value)
                valid = bool(valid) and math.isfinite(reward) and math.isfinite(gain)
                self._outcomes.add(gain, valid)
                self._nodes[parent.id].direct.add(gain, valid)
                for node in path:
                    node.subtree.add(gain, valid)
                if valid:
                    self._best_reward = max(self._best_reward, reward)

    def update(self, children_with_parents, **kwargs):
        with self._lock:
            result = super().update(children_with_parents, **kwargs)
            # Register only retained candidates. Previously retained nodes and
            # their statistics remain even when their program leaves the archive.
            for state in self._states:
                self._register(state)
                self._best_reward = max(self._best_reward, float(state.value))
            return result

    def _context(self, sid, direct):
        node = self._nodes[sid]
        if direct:
            descendants = node.subtree.without(node.direct)
            if descendants.attempts:
                return descendants
        branch = node
        while branch.parent_id is not None:
            ancestor = self._nodes[branch.parent_id]
            outside = ancestor.subtree.without(branch.subtree)
            if outside.attempts:
                return outside
            branch = ancestor
        return self._outcomes.without(branch.subtree)

    def _draw_model(self, sid, direct):
        node = self._nodes[sid]
        evidence = node.direct if direct else node.subtree
        context = self._context(sid, direct)
        # At most four observations' worth of borrowed evidence. Neither a new
        # child nor an ancestor receives fabricated visits from this prior.
        weight = min(4.0, context.attempts) / max(1, context.attempts)
        valid_prior = weight * context.valid
        invalid_prior = weight * (context.attempts - context.valid)
        probability = float(self._rng.beta(
            1.0 + valid_prior + evidence.valid,
            1.0 + invalid_prior + evidence.attempts - evidence.valid))
        prior_mean = context.mean if context.valid else 0.0
        scale = (math.sqrt(context.m2 / (context.valid - 1))
                 if context.valid > 1 and context.m2 > 0.0 else self._initial_scale)
        scale = max(scale, abs(prior_mean) * 1e-12, 1e-12)
        prior_kappa = max(0.5, valid_prior)
        kappa = prior_kappa + evidence.valid
        mean = (prior_kappa * prior_mean + evidence.valid * evidence.mean) / kappa
        alpha = 2.5 + 0.5 * evidence.valid
        beta = (1.5 * scale * scale + 0.5 * evidence.m2
                + 0.5 * prior_kappa * evidence.valid / kappa
                * (evidence.mean - prior_mean) ** 2)
        variance = beta / max(float(self._rng.gamma(alpha)), np.finfo(float).tiny)
        sigma = math.sqrt(variance)
        mean = float(self._rng.normal(mean, sigma / math.sqrt(kappa)))
        return probability, mean, sigma

    def _eligible_tree(self, requested):
        """Widen persistent branches, preserving the caller's parent budget."""
        live = {s.id: s for s in self._states}
        reachable = set()
        best = {}
        for state in self._states:
            sid = state.id
            while sid is not None:
                reachable.add(sid)
                best[sid] = max(best.get(sid, -math.inf), float(state.value))
                sid = self._nodes[sid].parent_id
        children = {}
        roots = []
        # Dictionary insertion order makes seed-zero replay independent of UUIDs.
        for sid, node in self._nodes.items():
            if sid not in reachable:
                continue
            if node.parent_id is None:
                roots.append(sid)
            else:
                children.setdefault(node.parent_id, []).append(sid)
        for sid, child_ids in children.items():
            node = self._nodes[sid]
            node.active_children = [c for c in node.active_children if c in reachable]
            limit = max(1, math.ceil(math.sqrt(1.0 + node.subtree.attempts / self.group_size)))
            unopened = [c for c in child_ids if c not in node.active_children]
            unopened.sort(key=lambda c: best[c], reverse=True)
            node.active_children.extend(unopened[:max(0, limit - len(node.active_children))])

        def visible():
            seen = []
            stack = list(reversed(roots))
            while stack:
                sid = stack.pop()
                seen.append(sid)
                stack.extend(reversed([c for c in self._nodes[sid].active_children
                                       if c in reachable]))
            return seen

        visible_ids = visible()
        relaxed = 0
        while sum(sid in live for sid in visible_ids) < min(requested, len(live)):
            unopened = [(best[c], sid, c) for sid in visible_ids
                        for c in children.get(sid, ())
                        if c not in self._nodes[sid].active_children]
            if not unopened:
                raise RuntimeError("Thompson genealogy cannot reach the live archive")
            _, sid, child = max(unopened, key=lambda entry: entry[0])
            self._nodes[sid].active_children.append(child)
            relaxed += 1
            visible_ids = visible()
        return live, roots, visible_ids, relaxed

    def sample_states(self, num_states, *, selection_step=None):
        with self._lock:
            # Pending-evaluation recovery and training recovery both ask for
            # this same event. Consume randomness and widen branches only once.
            if selection_step is not None and selection_step == self._selection_step:
                if len(self._selected) != min(int(num_states), len(self._states)):
                    raise ValueError("Thompson recovery changed its parent batch size")
                return list(self._selected)
            if num_states <= 0 or not self._states:
                self.last_picks_info = []
                return []
            live, roots, visible_ids, relaxed = self._eligible_tree(int(num_states))
            available = {sid for sid in visible_ids if sid in live}
            pending = {}
            models = {}
            picks, infos = [], []
            for _ in range(min(int(num_states), len(live))):
                anchors = {}
                for sid in visible_ids:
                    if sid not in available:
                        continue
                    value = float(live[sid].value)
                    while sid is not None:
                        anchors[sid] = max(anchors.get(sid, -math.inf), value)
                        sid = self._nodes[sid].parent_id

                def choose(options):
                    scored = []
                    for sid, direct in options:
                        key = (sid, direct)
                        if key not in models:
                            models[key] = self._draw_model(sid, direct)
                        anchor = float(live[sid].value) if direct else anchors[sid]
                        queued = 0 if direct else pending.get(sid, 0)
                        score = expected_max_gain(
                            *models[key], self._best_reward - anchor,
                            self.group_size, pending=queued * self.group_size)
                        stats = self._nodes[sid].direct if direct else self._nodes[sid].subtree
                        scored.append((score, -queued, -stats.attempts, anchor, sid, direct))
                    chosen = max(scored, key=lambda entry: entry[:4])
                    return chosen[4], chosen[5], chosen[0]

                sid, _, root_score = choose([(r, False) for r in roots if r in anchors])
                path = []
                while True:
                    path.append(sid)
                    options = [(c, False) for c in self._nodes[sid].active_children
                               if c in anchors]
                    if sid in available:
                        options.append((sid, True))
                    chosen, direct, score = choose(options)
                    if direct:
                        break
                    sid = chosen
                state = live[sid]
                node = self._nodes[sid]
                infos.append({
                    "value": float(state.value), "n": node.subtree.attempts,
                    "Q": None, "P": None, "bonus": None, "score": score,
                    "selection_mode": "thompson", "is_seed": state.is_seed,
                    "thompson": {
                        "path": list(path), "root_expected_gain": root_score,
                        "expected_gain": score, "direct_attempts": node.direct.attempts,
                        "subtree_attempts": node.subtree.attempts,
                        "valid_probability_draw": models[(sid, True)][0],
                        "pending_groups_on_path": [pending.get(p, 0) for p in path],
                        "group_size": self.group_size,
                        "batch_capacity_widenings": relaxed,
                    },
                })
                picks.append(state)
                available.remove(sid)
                for ancestor in path:
                    pending[ancestor] = pending.get(ancestor, 0) + 1
            self._selection_step = selection_step
            self._selected = list(picks)
            self.last_picks_info = infos
            return picks

    def reassign_seed_id(self, state, new_id):
        with self._lock:
            old_id = state.id
            super().reassign_seed_id(state, new_id)
            # Keep insertion order; otherwise a recovered fresh run would use
            # different Thompson draws for its seed branches on the next step.
            self._nodes = {new_id if sid == old_id else sid: node
                           for sid, node in self._nodes.items()}
            for node in self._nodes.values():
                if node.parent_id == old_id:
                    node.parent_id = new_id
            for info in self.last_picks_info:
                detail = info["thompson"]
                detail["path"] = [new_id if sid == old_id else sid for sid in detail["path"]]

    def state_dict(self):
        with self._lock:
            payload = super().state_dict()
            payload["thompson"] = {
                "version": 1, "group_size": self.group_size,
                "rng_state": deepcopy(self._rng.bit_generator.state),
                "outcomes": asdict(self._outcomes), "best_reward": self._best_reward,
                "initial_scale": self._initial_scale,
                "nodes": {sid: asdict(node) for sid, node in self._nodes.items()},
            }
            return payload

    def migrate_state_dict(self, payload, exp_dir, before_step):
        """Keep a legacy archive intact; rebuild evidence from completed attempts.

        Called on a new sampler, so failed validation never mutates the live
        picker. Pruned, never-expanded leaves need no posterior of their own.
        """
        if payload.get("selection_mode", "puct") not in ("puct", "uct"):
            raise ValueError("Thompson migration requires a PUCT/UCT checkpoint")
        PUCTSampler.load_state_dict(self, dict(payload, selection_mode="thompson"))
        self._nodes = {}
        self._outcomes = OutcomeStats()
        self._selection_step, self._selected = None, []
        claims, history = {}, []

        def claim(sid, value, ancestors):
            value, ancestors = float(value), tuple(ancestors)
            if not math.isfinite(value) or sid in ancestors or len(set(ancestors)) != len(ancestors):
                raise ValueError(f"invalid Thompson migration genealogy: {sid}")
            row = (value, ancestors)
            if sid in claims and claims[sid] != row:
                raise ValueError(f"conflicting Thompson migration identity: {sid}")
            claims[sid] = row

        for state in self._states:
            claim(state.id, state.value, [p["id"] for p in state.parents])
        for step in range(int(before_step)):
            directory = Path(exp_dir) / f"step{step:02d}"
            summary = json.loads((directory / f"step{step:02d}.summary.json").read_text())
            event = json.loads((directory / f"step{step:02d}.parents.json").read_text())
            if (summary.get("completed") is not True or summary.get("step") != step
                    or event.get("step") != step):
                raise ValueError(f"Thompson migration needs completed step {step}")
            parents = {int(row["group"]): row for row in event["parents"]}
            if len(parents) != len(event["parents"]) or not parents:
                raise ValueError(f"invalid saved parent groups for step {step}")
            for row in parents.values():
                claim(row["parent_id"], row["parent_reward"], row["ancestor_ids"])
            history.append((step, directory, summary, parents))

        for sid, (value, ancestors) in sorted(claims.items(), key=lambda item: len(item[1][1])):
            if ancestors:
                parent = claims.get(ancestors[0])
                if parent is None or parent[1] != ancestors[1:]:
                    raise ValueError(f"missing or inconsistent migration ancestor for {sid}")
            elif sid not in self._seed_ids:
                raise ValueError(f"migration root is not a saved seed: {sid}")
            self._nodes[sid] = SearchNode(ancestors[0] if ancestors else None, value)
        values = [claims[sid][0] for sid in self._seed_ids]
        self._best_reward = max(values)
        self._initial_scale = max(float(np.std(values)), .05 * max(abs(v) for v in values), 1e-12)
        unseen = set(claims) - self._seed_ids
        for step, directory, summary, parents in history:
            seen, primary = set(), 0
            for path in sorted(directory.glob(f"step{step:02d}_group*_rollout*.meta.json")):
                row = json.loads(path.read_text())
                key = (int(row["group"]), int(row["rollout"]))
                parent = parents[int(row.get("parent_group", row["group"]))]
                reward, valid = float(row["reward"]), row["valid"]
                if (row.get("step") != step or key in seen
                        or row["parent_id"] != parent["parent_id"]
                        or float(row["parent_value"]) != float(parent["parent_reward"])
                        or not isinstance(valid, bool) or not math.isfinite(reward)
                        or row.get("status") == "evaluation_pending"):
                    raise ValueError(f"inconsistent evaluated rollout for migration: {path}")
                seen.add(key)
                primary += int(bool(row.get("counts_toward_allocation", True))
                               and int(row.get("retry_attempt", 0) or 0) == 0)
                sid = row["node_id"]
                if sid in claims:
                    value, ancestors = claims[sid]
                    if not valid or value != reward or not ancestors or ancestors[0] != row["parent_id"]:
                        raise ValueError(f"rollout does not match saved archive identity: {path}")
                    unseen.discard(sid)
                state = State(id=parent["parent_id"], timestep=parent["parent_timestep"],
                              value=parent["parent_reward"], code="",
                              parents=[{"id": sid} for sid in parent["ancestor_ids"]])
                self.record_outcomes(state, [reward], [valid])
            expected = int(summary["planned_rollouts"])
            retries = int(summary.get("coder_format_retries", 0))
            if primary != expected or len(seen) != expected + retries:
                raise ValueError(f"incomplete rollout history for Thompson migration at step {step}: "
                                 f"found {len(seen)}, expected {expected + retries}")
        if unseen:
            raise ValueError(f"Thompson migration cannot trace {len(unseen)} saved nodes to evaluated rollouts")
        return self._outcomes.attempts

    def load_state_dict(self, payload):
        data = payload.get("thompson", {})
        if payload.get("selection_mode") != "thompson" or data.get("version") != 1:
            raise ValueError("Thompson resume requires its saved search statistics")
        nodes = {}
        for sid, row in data["nodes"].items():
            nodes[sid] = SearchNode(
                parent_id=row["parent_id"], value=float(row["value"]),
                direct=OutcomeStats(**row["direct"]),
                subtree=OutcomeStats(**row["subtree"]),
                active_children=list(row["active_children"]))
        for sid, node in nodes.items():
            if node.parent_id is not None and node.parent_id not in nodes:
                raise ValueError("Thompson checkpoint is missing an ancestor")
            if any(c not in nodes or nodes[c].parent_id != sid for c in node.active_children):
                raise ValueError("Thompson checkpoint has an invalid active branch")
        if any(row["id"] not in nodes for row in payload["states"]):
            raise ValueError("Thompson checkpoint is missing live candidates")
        with self._lock:
            super().load_state_dict(payload)
            self._nodes = nodes
            self._outcomes = OutcomeStats(**data["outcomes"])
            self._best_reward = float(data["best_reward"])
            self._initial_scale = float(data["initial_scale"])
            self._rng.bit_generator.state = deepcopy(data["rng_state"])
            self._selection_step = None
            self._selected = []

    def import_legacy_states(self, *args, **kwargs):
        raise ValueError("Thompson cannot infer lost outcomes/genealogy from a legacy archive; start a fresh run")
