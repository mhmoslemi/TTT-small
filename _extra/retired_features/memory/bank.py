"""
The memory bank M.

No vectors. The bank is a list of lessons plus an id index, and everything that
used to be a nearest-neighbour question is now either an exact id lookup or a
token-set comparison:

  selection    the model reads catalog() and names ids. See lookup.py.
  dedup        normalized-title equality, then token-set Jaccard. Measured on
               the v2 bank, identical-title pairs had cosine 0.45-0.95 under the
               old hashed embedder, so no threshold on that scale separated them;
               title normalization catches them outright.
  eviction     lowest importance, then least used, then oldest.

`uses` now means the model chose this lesson, not that a cosine ranked it top-5.
That makes the catalog's "used Nx" column a real signal for the selector, and it
makes the retrieval-concentration measurement causal rather than an artifact of
the query being program text.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from memory.types import (FAILURE, GLOBAL, LOCAL, SUCCESS, Lesson,
                          clamp_importance, jaccard, normalize_title)


class MemoryBank:
    def __init__(self, cfg):
        self.cfg = cfg
        if (bool(getattr(cfg, "is_v2", False))
                and int(getattr(cfg, "arm_comparison_n", 0) or 0) <= 0):
            raise ValueError(
                "memory V2 bank requires a resolved arm_comparison_n")
        self.lessons: List[Lesson] = []
        self._by_id: Dict[str, Lesson] = {}
        self._path: Optional[Path] = None
        self.stats = {"proposed": 0, "added": 0, "duplicates": 0,
                      "reinforced": 0, "evicted": 0, "rejected": 0,
                      "lookups": 0, "selections": 0, "empty_selections": 0,
                      "outcome_updates": 0, "tail_wins": 0,
                      "curations": 0, "curated_away": 0}

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.lessons)

    def counts(self) -> Dict[str, int]:
        return {
            "total": len(self.lessons),
            "success": sum(1 for l in self.lessons if l.outcome == SUCCESS),
            "failure": sum(1 for l in self.lessons if l.outcome == FAILURE),
            "local": sum(1 for l in self.lessons if l.scope == LOCAL),
            "global": sum(1 for l in self.lessons if l.scope == GLOBAL),
        }

    def by_id(self, lesson_id: str) -> Optional[Lesson]:
        return self._by_id.get((lesson_id or "").strip().lower())

    def ids(self) -> List[str]:
        return [l.id for l in self.lessons]

    @property
    def comparison_n(self) -> Optional[int]:
        if bool(getattr(self.cfg, "is_v2", False)):
            value = int(getattr(self.cfg, "arm_comparison_n", 0) or 0)
            return value if value > 0 else None
        return None

    def evidence_stats(self, lesson: Lesson) -> Dict:
        return lesson.outcome_stats(self.comparison_n)

    def evidence_mean(self, lesson: Lesson) -> float:
        return lesson.mean_tail_uplift(self.comparison_n)

    def fetch(self, ids: Sequence[str], count_use: bool = True) -> List[Lesson]:
        """
        Resolve chosen ids to lessons, in the order asked for. Unknown ids are
        skipped. This is the whole of what used to be Eq. 7.
        """
        out = []
        for lesson_id in ids or ():
            lesson = self.by_id(lesson_id)
            if lesson is None:
                continue
            if count_use:
                lesson.uses += 1
            out.append(lesson)
        return out

    # ------------------------------------------------------------------
    def catalog(self, limit: Optional[int] = None,
                chars: Optional[int] = None) -> List[str]:
        """
        The index the model reads, one line per lesson. limit=0 means the whole
        bank, which is the default and the intent: the model should see
        everything it has and decide for itself.

        When a limit does apply, order is importance then recency, so the
        entries hidden are the weakest rather than an arbitrary slice.
        """
        limit = int(self.cfg.catalog_max_lessons if limit is None else limit)
        chars = int(self.cfg.catalog_chars if chars is None else chars)
        if not self.lessons:
            return []
        items = self.lessons
        if limit > 0 and len(items) > limit:
            items = sorted(items, key=lambda l: (
                self.evidence_mean(l), l.importance, l.step),
                           reverse=True)[:limit]
        # Stable, readable order for the model: oldest first, so ids it has seen
        # before stay in the same place between steps.
        items = sorted(items, key=lambda l: (l.step, l.id))
        return [l.catalog_line(chars, self.comparison_n) for l in items]

    def catalog_ids(self, limit: Optional[int] = None) -> List[str]:
        """The ids that appear in catalog(), for validating a selection."""
        limit = int(self.cfg.catalog_max_lessons if limit is None else limit)
        items = self.lessons
        if limit > 0 and len(items) > limit:
            items = sorted(items, key=lambda l: (
                self.evidence_mean(l), l.importance, l.step),
                           reverse=True)[:limit]
        return [l.id for l in items]

    # ------------------------------------------------------------------
    def add_many(self, lessons: Sequence[Lesson]) -> int:
        """Insert new lessons, reinforcing rather than storing duplicates."""
        if not lessons:
            return 0
        self.stats["proposed"] += len(lessons)
        added = 0
        for lesson in lessons:
            dup = self._duplicate_of(lesson)
            if dup is not None:
                # Textual re-derivation is not causal evidence. Keep the legacy
                # behavior only when explicitly requested.
                if bool(getattr(self.cfg, "text_reinforce", True)):
                    self._bump(dup, float(self.cfg.reinforce_delta))
                self.stats["duplicates"] += 1
                continue
            self.lessons.append(lesson)
            self._by_id[lesson.id] = lesson
            added += 1
        self.stats["added"] += added
        self._evict()
        return added

    def reinforce(self, ids: Sequence[str], delta: Optional[float] = None) -> int:
        if not bool(getattr(self.cfg, "text_reinforce", True)):
            return 0
        delta = float(self.cfg.reinforce_delta if delta is None else delta)
        hit = 0
        for lesson_id in ids or ():
            lesson = self.by_id(lesson_id)
            if lesson is not None:
                self._bump(lesson, delta)
                hit += 1
        return hit

    def record_outcome(self, ids: Sequence[str], rollouts: int, valid: int,
                       improved: int, tail_uplift: float, step: int,
                       comparison_n: Optional[int] = None, arm: str = "",
                       context_id: str = "", parent_reward: Optional[float] = None,
                       control_rollouts: int = 0,
                       control_valid: int = 0, distinct_codes: int = 0,
                       control_distinct_codes: int = 0) -> int:
        """Attach matched null-arm evidence to every lesson in one prompt arm."""
        is_v2 = bool(getattr(self.cfg, "is_v2", False))
        if is_v2 and (comparison_n is None or int(comparison_n) <= 0):
            raise ValueError(
                "memory V2 outcome records require comparison_n >= 1")
        if (is_v2 and self.comparison_n is not None
                and int(comparison_n) != self.comparison_n):
            raise ValueError(
                f"memory V2 expected best@n={self.comparison_n}, got "
                f"best@n={comparison_n}")
        hit = 0
        for lesson_id in ids or ():
            lesson = self.by_id(lesson_id)
            if lesson is None:
                continue
            first_trial = lesson.arm_trials == 0
            lesson.arm_trials += 1
            lesson.arm_rollouts += int(rollouts)
            lesson.arm_valid += int(valid)
            lesson.arm_parent_improvements += int(improved)
            lesson.tail_uplift_sum += float(tail_uplift)
            lesson.tail_uplift_best = (
                float(tail_uplift) if first_trial
                else max(float(lesson.tail_uplift_best), float(tail_uplift)))
            lesson.last_outcome_step = int(step)
            if tail_uplift > 0:
                lesson.arm_tail_wins += 1
                self.stats["tail_wins"] += 1
            if is_v2:
                lesson.causal_history.append({
                    "n": int(comparison_n),
                    "arm": str(arm),
                    "step": int(step),
                    "context_id": str(context_id or ""),
                    "parent_reward": (float(parent_reward)
                                      if parent_reward is not None else None),
                    "rollouts": int(rollouts),
                    "valid": int(valid),
                    "improved": int(improved),
                    "tail_uplift": float(tail_uplift),
                    "control_rollouts": int(control_rollouts),
                    "control_valid": int(control_valid),
                    "distinct_codes": int(distinct_codes),
                    "control_distinct_codes": int(control_distinct_codes),
                })
            hit += 1
        self.stats["outcome_updates"] += hit
        return hit

    def exploration_lesson(self, excluded=(), step: int = 0,
                           c: float = 0.5,
                           reservations: Optional[Dict[str, int]] = None
                           ) -> Optional[Lesson]:
        """UCB choice for the under-tested memory arm, with novelty tie-breaks."""
        excluded = set(excluded or ())
        candidates = [l for l in self.lessons if l.id not in excluded]
        if not candidates:
            return None
        if not bool(getattr(self.cfg, "is_v2", False)):
            # Historical scoring is intentionally kept byte-for-byte equivalent
            # for V1 reproducibility.
            total = sum(l.arm_trials for l in candidates) + 1
            means = [abs(l.tail_uplift_sum / l.arm_trials) for l in candidates
                     if l.arm_trials > 0]
            reward_scale = max(means, default=1e-3)

            def v1_score(lesson):
                if lesson.arm_trials <= 0:
                    return (float("inf"), -lesson.uses, lesson.step)
                mean = lesson.tail_uplift_sum / lesson.arm_trials
                bonus = (float(c) * reward_scale
                         * math.sqrt(math.log(total + 1) / lesson.arm_trials))
                return (mean + bonus, -lesson.uses, lesson.step)

            return max(candidates, key=v1_score)

        reservations = dict(reservations or {})
        target_n = self.comparison_n
        # N covers the full bank, including a lesson excluded from this parent's
        # exploration arm, and includes provisional assignments in this batch.
        stats_by_id = {lesson.id: lesson.outcome_stats(target_n)
                       for lesson in self.lessons}
        total = (sum(stats["trials"] for stats in stats_by_id.values())
                 + sum(max(0, int(value)) for value in reservations.values()) + 1)
        means = [abs(stats["uplift_sum"] / stats["trials"])
                 for stats in stats_by_id.values() if stats["trials"] > 0]
        reward_scale = max([1e-3, *means])

        def score(lesson):
            stats = stats_by_id[lesson.id]
            reserved = max(0, int(reservations.get(lesson.id, 0)))
            if stats["trials"] <= 0:
                # Every genuinely untested lesson gets a path into the batch;
                # pending reservations break ties so assignments spread out.
                return (float("inf"), -reserved, -lesson.uses, lesson.step)
            effective_trials = stats["trials"] + reserved
            mean = stats["uplift_sum"] / stats["trials"]
            bonus = (float(c) * reward_scale
                     * math.sqrt(math.log(total + 1) / effective_trials))
            return (mean + bonus, -lesson.uses, lesson.step)

        return max(candidates, key=score)

    def _bump(self, lesson: Lesson, delta: float) -> None:
        lesson.importance = clamp_importance(lesson.importance + delta)
        lesson.confirmations += 1
        self.stats["reinforced"] += 1

    def _duplicate_of(self, lesson: Lesson) -> Optional[Lesson]:
        if lesson.id in self._by_id:
            return self._by_id[lesson.id]
        key = normalize_title(lesson.title)
        thr = float(getattr(self.cfg, "dedup_jaccard", 0.0) or 0.0)
        toks = lesson.tokens()
        best, best_j = None, 0.0
        for other in self.lessons:
            if other.outcome != lesson.outcome:
                continue
            if key and normalize_title(other.title) == key:
                return other          # same title once stopwords are stripped
            if thr <= 0:
                continue
            j = jaccard(toks, other.tokens())
            if j > best_j:
                best, best_j = other, j
        return best if best_j >= thr else None

    def _evict(self) -> None:
        cap = int(getattr(self.cfg, "max_lessons", 0) or 0)
        if cap <= 0 or len(self.lessons) <= cap:
            return
        order = sorted(range(len(self.lessons)),
                       key=lambda i: (self.evidence_mean(self.lessons[i]),
                                      self.lessons[i].importance,
                                      self.lessons[i].uses,
                                      self.lessons[i].step))
        drop = set(order[: len(self.lessons) - cap])
        self.lessons = [l for i, l in enumerate(self.lessons) if i not in drop]
        self._by_id = {l.id: l for l in self.lessons}
        self.stats["evicted"] += len(drop)

    # ------------------------------------------------------------------
    def usage_summary(self) -> str:
        """
        The §4.2 measurement, computed inline so a run reports its own retrieval
        concentration instead of needing a post-hoc script.
        """
        if not self.lessons:
            return "bank empty"
        uses = sorted((l.uses for l in self.lessons), reverse=True)
        total = sum(uses) or 1
        never = sum(1 for u in uses if u == 0)
        top5 = sum(uses[:5])
        return (f"{len(uses)} lessons, {100 * never / len(uses):.0f}% never "
                f"chosen, top-5 hold {100 * top5 / total:.0f}% of selections")

    # ------------------------------------------------------------------
    def attach(self, path) -> None:
        self._path = Path(path)
        if self._path.exists():
            self.load(self._path)

    def save(self, path=None) -> Optional[Path]:
        target = Path(path) if path is not None else self._path
        if target is None:
            return None
        is_v2 = bool(getattr(self.cfg, "is_v2", False))
        payload = {"counts": self.counts(), "stats": self.stats,
                   "usage": self.usage_summary(),
                   "lessons": [l.to_dict(include_v2=is_v2)
                               for l in self.lessons]}
        if is_v2:
            payload["memory_version"] = "V2"
            payload["comparison_n"] = self.comparison_n
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(target.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2))
        tmp.replace(target)          # atomic, so a crash cannot truncate it
        return target

    def load(self, path) -> int:
        data = json.loads(Path(path).read_text())
        self.lessons = [Lesson.from_dict(d) for d in data.get("lessons", [])]
        self._by_id = {l.id: l for l in self.lessons}
        if bool(getattr(self.cfg, "is_v2", False)):
            legacy_trials = sum(
                max(0, int(lesson.arm_trials) - len(lesson.causal_history))
                for lesson in self.lessons)
            if legacy_trials:
                print(f"[memory] V2 loaded {legacy_trials} legacy causal "
                      "trial(s) without an n/role/context ledger; they remain "
                      "in archival counters but are treated as untested by V2")
        # Carry counters forward while remaining compatible with old banks.
        for key, value in (data.get("stats") or {}).items():
            if key in self.stats:
                self.stats[key] = value
        return len(self.lessons)
