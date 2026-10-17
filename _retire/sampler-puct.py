"""
PUCT-style state archive.

For each candidate state s in the archive, the selection score is:

    score(s) = Q(s) + c * scale * P(s) * sqrt(1 + T) / (1 + n(s))

  Q(s)   = max child reward seen so far from s (or R(s) if never expanded)
  P(s)   = rank-based prior — high-reward states get more weight, but
           everyone gets nonzero mass
  scale  = max(R) - min(R) over the non-seed archive
  T      = total expansions performed
  n(s)   = expansions of s OR ANY DESCENDANT (so a successful lineage
           gets dampened, not just one node)

When selecting a batch of B parents, we also exclude the full ancestor
AND descendant set of each picked state from being picked again in the
same batch. This is the paper's "lineage blocking" trick that prevents
the batch from collapsing onto one promising thread.

After each batch:
  - Push the new children into the archive
  - Keep top-K children per parent (K=2 by default)
  - Cap the archive at MAX_BUFFER_SIZE by reward (seeds always kept)

State now also carries two problem-agnostic payloads, both optional:
  - raw_score:   the TRUE metric (e.g. C5 bound, runtime us) for prompt display,
                 separate from `value` which is always the higher-is-better reward.
  - construction: an injected global (height_sequence_1 / initial_h_values) that
                 is threaded parent -> child so a lineage can warm-start its search.
"""

import uuid
import numpy as np
from dataclasses import dataclass, field


@dataclass
class State:
    id: str
    timestep: int          # training step when this state was first created
    value: float           # reward of this state (always higher=better)
    code: str              # the code that produced it
    parents: list = field(default_factory=list)   # [{"id": ..., "timestep": ...}]
    is_seed: bool = False  # True for initial seed states
    raw_score: float = None      # true metric for prompt display (may equal value)
    construction: list = None    # injected global threaded parent -> child

    @staticmethod
    def make(timestep, value, code, parents=None, is_seed=False,
             raw_score=None, construction=None):
        return State(
            id=str(uuid.uuid4()),
            timestep=timestep,
            value=value,
            code=code,
            parents=parents or [], # all the parents not just one
            is_seed=is_seed,
            raw_score=raw_score,
            construction=construction,
        )


class PUCTSampler:
    def __init__(
        self,
        num_seeds: int,
        puct_c: float = 1.0,
        max_buffer_size: int = 1000,
        topk_children: int = 2,
        seed_value: float = 0.0,
        seed_states: list = None,
    ):
        self.puct_c = float(puct_c)
        self.max_buffer_size = max_buffer_size
        self.topk_children = topk_children

        # PUCT statistics
        self._n = {}   # state.id -> visit count (incl. descendants)
        self._m = {}   # state.id -> best child reward seen
        self._T = 0    # total expansions

        # Archive starts with seeds
        self._states = []
        self._seed_ids = set()

        if seed_states:
            # Rich seeds from the problem: each carries code/value/raw_score/construction.
            for ss in seed_states:
                s = State.make(
                    timestep=0,
                    value=getattr(ss, "value", seed_value),
                    code=getattr(ss, "code", "") or "",
                    is_seed=True,
                    raw_score=getattr(ss, "raw_score", None),
                    construction=getattr(ss, "construction", None),
                )
                self._states.append(s)
                self._seed_ids.add(s.id)
        else:
            # Fallback: num_seeds blank seeds at a constant value (circle-packing style).
            for _ in range(num_seeds):
                s = State.make(timestep=0, value=seed_value, code="", is_seed=True)
                self._states.append(s)
                self._seed_ids.add(s.id)

        # Stats from last sample_states call (for printing)
        self.last_picks_info = []

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _build_children_map(self):
        children = {}
        for s in self._states:
            for p in s.parents:
                children.setdefault(p["id"], set()).add(s.id)
        return children

    def _full_lineage(self, state, children_map):
        """Ancestors + descendants + self."""
        lineage = {state.id}
        for p in state.parents:
            lineage.add(p["id"])
        # BFS down through children
        queue = [state.id]
        visited = {state.id}
        while queue:
            sid = queue.pop(0)
            for c in children_map.get(sid, []):
                if c not in visited:
                    visited.add(c)
                    lineage.add(c)
                    queue.append(c)
        return lineage




    def _scale(self):

        vals = np.array([
            s.value for s in self._states
            if s.id not in self._seed_ids and s.value is not None
        ])
        if vals.size == 0:
            return 1.0
        return float(max(vals.max() - vals.min(), 1e-6))


    def _prior(self, values):
        """Rank-based: best gets (N), worst gets 1; normalize to a distribution."""
        if len(values) == 0:
            return np.array([])
        N = len(values)
        order = np.argsort(np.argsort(-values))  # rank 0 = best
        w = (N - order).astype(np.float64)
        return w / w.sum()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def sample_states(self, num_states: int):
        """Pick `num_states` parent states for the next batch."""
        if not self._states:
            return []

        values = np.array([s.value if s.value is not None else -np.inf
                           for s in self._states])
        scale = self._scale()
        P = self._prior(values)
        sqrtT = np.sqrt(1.0 + self._T)

        scored = []
        for i, s in enumerate(self._states):
            n = self._n.get(s.id, 0)
            m = self._m.get(s.id, values[i])
            Q = m if n > 0 else values[i]
            bonus = self.puct_c * scale * P[i] * sqrtT / (1.0 + n)
            scored.append((Q + bonus, values[i], s, n, Q, P[i], bonus))

        # Sort by (score, value) descending
        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)

        # Lineage blocking across the batch
        children_map = self._build_children_map()
        blocked = set()
        picks = []
        info = []
        for entry in scored:
            s = entry[2]
            if s.id in blocked:
                continue
            picks.append(s)
            info.append({
                "value": entry[1], "n": entry[3], "Q": entry[4],
                "P": entry[5], "bonus": entry[6], "score": entry[0],
                "is_seed": s.id in self._seed_ids,
            })
            blocked.update(self._full_lineage(s, children_map))
            if len(picks) >= num_states:
                break

        # If we couldn't fill the batch with blocking, top up without blocking
        if len(picks) < num_states:
            remaining = num_states - len(picks)
            picked_ids = {s.id for s in picks}
            for entry in scored:
                if len(picks) >= num_states:
                    break
                s = entry[2]
                if s.id in picked_ids:
                    continue
                picks.append(s)
                info.append({
                    "value": entry[1], "n": entry[3], "Q": entry[4],
                    "P": entry[5], "bonus": entry[6], "score": entry[0],
                    "is_seed": s.id in self._seed_ids,
                })

        self.last_picks_info = info
        return picks

    def update(self, children_with_parents):
        """
        Push new children into the archive and update PUCT stats.

        children_with_parents: list of (child_state, parent_state)
        Only children with valid (non-None) values should be passed.
        """
        # Update visit counts (m, n, T) for ALL attempts, even failed ones,
        # by calling record_failure separately. Here we only handle successes.
        # Actually, let's match the paper: n increments on any expansion,
        # m only on success.

        # Update m for parents
        parent_best = {}
        parent_objs = {}
        for child, parent in children_with_parents:
            if child.value is None:
                continue
            pid = parent.id
            parent_objs[pid] = parent
            parent_best[pid] = max(parent_best.get(pid, -np.inf), float(child.value))

        for pid, best in parent_best.items():
            self._m[pid] = max(self._m.get(pid, best), best)

        # Now incorporate children + dedup
        existing_codes = {s.code for s in self._states if s.code}
        new_states = []
        for child, parent in children_with_parents:
            if child.value is None:
                continue
            if child.code and child.code in existing_codes:
                continue  # exact dup
            # Set parent ref
            child.parents = (
                [{"id": parent.id, "timestep": parent.timestep}]
                + (parent.parents or [])
            )
            new_states.append(child)
            if child.code:
                existing_codes.add(child.code)

        self._states.extend(new_states)

        # Enforce top-K children per parent (excluding seeds)
        if self.topk_children > 0:
            by_parent = {}
            no_parent = []
            for s in self._states:
                if s.id in self._seed_ids or not s.parents:
                    no_parent.append(s)
                    continue
                pid = s.parents[0]["id"]
                by_parent.setdefault(pid, []).append(s)
            filtered = list(no_parent)
            for children in by_parent.values():
                children.sort(
                    key=lambda x: x.value if x.value is not None else -np.inf,
                    reverse=True,
                )
                filtered.extend(children[: self.topk_children])
            self._states = filtered

        # Cap archive size, always keeping seeds
        if len(self._states) > self.max_buffer_size:
            seeds = [s for s in self._states if s.id in self._seed_ids]
            non_seeds = [s for s in self._states if s.id not in self._seed_ids]
            non_seeds.sort(
                key=lambda x: x.value if x.value is not None else -np.inf,
                reverse=True,
            )
            keep_non_seeds = non_seeds[: self.max_buffer_size - len(seeds)]
            self._states = seeds + keep_non_seeds



    def record_expansion(self, parent: State, count: int = 1):
        """Called once per parent per step. Pass count=group_size so n and T
        grow per-ROLLOUT (n[p]+=64, T+=512 per step at 8x64), matching discover,
        which increments once per rollout in update_states / record_failed_rollout.
        count=1 gives per-parent growth (n[p]+=1, T+=8), which is the paper's
        Appendix A.2 prose but NOT what their code does."""
        anc_ids = [parent.id] + [p["id"] for p in (parent.parents or [])]
        for aid in anc_ids:
            self._n[aid] = self._n.get(aid, 0) + count
        self._T += count


    def best_state(self):
        non_seeds = [s for s in self._states if s.id not in self._seed_ids]
        if not non_seeds:
            return None
        return max(non_seeds, key=lambda s: s.value if s.value is not None else -np.inf)

    def archive_size(self):
        return len(self._states)


if __name__ == "__main__":
    sampler = PUCTSampler(num_seeds=3, puct_c=1.0)
    print("Initial archive:", sampler.archive_size())

    picks = sampler.sample_states(2)
    print(f"Picked {len(picks)} parents (all seeds):", [s.is_seed for s in picks])

    # Simulate 2 rollouts that produced new states
    children = []
    for p in picks:
        sampler.record_expansion(p)
        child = State.make(timestep=1, value=np.random.rand() * 2.5, code=f"code_{uuid.uuid4().hex[:6]}")
        children.append((child, p))
    sampler.update(children)
    print("After update:", sampler.archive_size())

    picks2 = sampler.sample_states(2)
    print(f"Pick info: {sampler.last_picks_info}")
    print("Best state value:", sampler.best_state().value)
