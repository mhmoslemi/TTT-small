"""
PUCT-style state archive.

For each candidate state s in the archive, the selection score is:

    score(s) = Q(s) + c * scale * P(s) * sqrt(1 + T) / (1 + n(s))

  Q(s)   = max child reward seen so far from s (or R(s) if never expanded)
  P(s)   = rank-based prior by default; the top states may be re-weighted by an
           external Elo prior produced asynchronously by a MultiAgentReRanker
           (see reranker/). The Elo prior only RESHUFFLES the prior mass already
           assigned to the elite set, so the exploration mass on the tail is
           preserved and P(s) still sums to 1.
  scale  = max(R) - min(R) over the non-seed archive
  T      = total expansions performed
  n(s)   = expansions of s OR ANY DESCENDANT (so a successful lineage
           gets dampened, not just one node)

When selecting a batch of B parents, we also exclude the full ancestor
AND descendant set of each picked state from being picked again in the
same batch ("lineage blocking").

After each batch:
  - Push the new children into the archive
  - Keep top-K children per parent (K=2 by default)
  - Cap the archive at MAX_BUFFER_SIZE by reward (seeds always kept)

State carries two optional problem-agnostic payloads:
  - raw_score:   the TRUE metric for prompt display (separate from `value`).
  - construction: an injected global threaded parent -> child for warm starts.

THREAD-SAFETY: a background re-ranker thread reads the buffer via
snapshot_top_states() and writes the Elo prior via set_external_prior(). Both,
plus the buffer-mutating section of update(), are guarded by self._prior_lock
(an RLock). The background thread NEVER holds the lock during LLM calls.
"""

import threading
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
            parents=parents or [],  # all the parents not just one
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

        self._current_step = 0   # most recently STARTED training step; the
                                 # re-ranker tags its cycles with this so its
                                 # logs line up with the step dirs.


        # Archive starts with seeds
        self._states = []
        self._seed_ids = set()

        # ---- Elo re-ranker hook (filled by a background MultiAgentReRanker) ----
        # Guards: buffer-rebuild in update(), snapshot_top_states(), and the
        # external prior read/write. RLock so _blended_prior (called from
        # sample_states) can re-enter from the same thread.
        self._prior_lock = threading.RLock()
        self._external_prior = {}          # state.id -> non-negative Elo weight
        self._external_prior_alpha = 1.0   # 1.0 = Elo fully replaces rank in elite set

        if seed_states:
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

    def _blended_prior(self, values: np.ndarray) -> np.ndarray:
        """Rank prior over the whole buffer, with the Elo prior (if any) applied
        ON TOP of the elite states. We only redistribute the prior mass the rank
        prior already gave the elite group, so the tail's exploration mass is
        untouched and the result still sums to 1."""
        P = self._prior(values)
        if P.size == 0:
            return P

        with self._prior_lock:
            ext = dict(self._external_prior)            # id -> Elo weight
            alpha = float(self._external_prior_alpha)
        if not ext:
            return P

        # Indices of buffer states that currently have an Elo weight.
        idx = [i for i, s in enumerate(self._states) if s.id in ext]
        if len(idx) < 2:
            return P
        idx = np.array(idx, dtype=int)

        m_K = float(P[idx].sum())                       # mass the rank prior gave this group
        if m_K <= 0.0:
            return P

        elo_w = np.array([ext[self._states[i].id] for i in idx], dtype=np.float64)
        elo_w = np.clip(elo_w, 0.0, None)
        if elo_w.sum() <= 0.0:
            return P
        q_elo = elo_w / elo_w.sum()                     # Elo distribution over the group
        q_rank = P[idx] / m_K                           # rank distribution over the same group

        q = (1.0 - alpha) * q_rank + alpha * q_elo      # alpha=1 -> pure Elo
        q = q / q.sum()

        P = P.copy()
        P[idx] = m_K * q                                # reshuffle within the group only
        return P

    # ------------------------------------------------------------------
    # Thread-safe hooks for the background re-ranker
    # ------------------------------------------------------------------
    def snapshot_top_states(self, k: int):
        """Thread-safe snapshot of the top-k states (by reward) that have actual
        code. Returns lightweight dicts so the caller can run a slow LLM
        tournament WITHOUT holding the sampler lock. (Top-k is by reward .value:
        the full PUCT score also depends on visit counts and changes intra-step.)"""
        with self._prior_lock:
            cands = [s for s in self._states if s.code and s.code.strip()]
            cands.sort(
                key=lambda s: (s.value if s.value is not None else -np.inf),
                reverse=True,
            )
            top = cands[: max(0, int(k))]
            return [
                {
                    "id": s.id,
                    "value": float(s.value) if s.value is not None else None,
                    "raw_score": (float(s.raw_score)
                                  if s.raw_score is not None else None),
                    "code": s.code,
                }
                for s in top
            ]

    def set_external_prior(self, weights_by_id: dict, alpha: float = None):
        """Install the Elo-derived prior. Only ids still in the buffer matter at
        read time (sample_states filters). `alpha` interpolates Elo vs rank
        within the elite set (1.0 = pure Elo)."""
        with self._prior_lock:
            self._external_prior = {str(k): float(v)
                                    for k, v in (weights_by_id or {}).items()}
            if alpha is not None:
                self._external_prior_alpha = float(alpha)

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
        P = self._blended_prior(values)        # <-- rank prior + (optional) Elo re-ranking
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
        children_with_parents: list of (child_state, parent_state).
        Only children with valid (non-None) values should be passed.

        The whole body is guarded by _prior_lock so the background
        snapshot_top_states() cannot iterate self._states while it is rebuilt.
        """
        with self._prior_lock:
            # Update m for parents (best child reward)
            parent_best = {}
            for child, parent in children_with_parents:
                if child.value is None:
                    continue
                pid = parent.id
                parent_best[pid] = max(parent_best.get(pid, -np.inf), float(child.value))
            for pid, best in parent_best.items():
                self._m[pid] = max(self._m.get(pid, best), best)

            # Incorporate children + dedup by exact code
            existing_codes = {s.code for s in self._states if s.code}
            new_states = []
            for child, parent in children_with_parents:
                if child.value is None:
                    continue
                if child.code and child.code in existing_codes:
                    continue  # exact dup
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
        grow per-ROLLOUT (matching discover). count=1 gives per-parent growth."""
        anc_ids = [parent.id] + [p["id"] for p in (parent.parents or [])]
        for aid in anc_ids:
            self._n[aid] = self._n.get(aid, 0) + count
        self._T += count

    def best_state(self):
        with self._prior_lock:
            non_seeds = [s for s in self._states if s.id not in self._seed_ids]
            if not non_seeds:
                return None
            return max(non_seeds,
                       key=lambda s: s.value if s.value is not None else -np.inf)

    def archive_size(self):
        return len(self._states)

    def set_current_step(self, step: int):
        self._current_step = int(step)

    def get_current_step(self) -> int:
        return self._current_step   


if __name__ == "__main__":
    sampler = PUCTSampler(num_seeds=3, puct_c=1.0)
    print("Initial archive:", sampler.archive_size())

    picks = sampler.sample_states(2)
    print(f"Picked {len(picks)} parents (all seeds):", [s.is_seed for s in picks])

    children = []
    for p in picks:
        sampler.record_expansion(p)
        child = State.make(timestep=1, value=np.random.rand() * 2.5,
                           code=f"code_{uuid.uuid4().hex[:6]}")
        children.append((child, p))
    sampler.update(children)
    print("After update:", sampler.archive_size())

    picks2 = sampler.sample_states(2)
    print(f"Pick info: {sampler.last_picks_info}")
    print("Best state value:", sampler.best_state().value)