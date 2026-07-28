"""
Algorithm 1: memory-augmented D-PUCT search with failure-aware max-seeking
test-time RL.

Per step t:

    recompute W_m(s), m_s over the tree                          (Eq. 2)
    rank + Elo debate signals over the archive D                 (Eq. 3)
    D-PUCT scores for all expandable actions                     (Eq. 4, 5, 6)
    S_t <- top-n targets
    for each target: prompt = [d | parent | top-m memories | instruction]
                     sample 1 child (virtual) or k children (leaf)
    verify every child -> reward r, feedback f
    insert into the tree and archive
    2L lessons -> memory bank                                    (Sec. 2.2)
    one clipped LoRA update                                      (Eq. 8-11)

    return argmax_{s in D} R(s)

The two learning channels run on different timescales on purpose: memory is
non-parametric and additive (it changes what is in context), the LoRA update is
parametric (it changes the weights). Both are fed by the same verifier call.
"""

import time
import traceback
from typing import List, Optional

from core.registry import load_example
from core.tree import SearchTree
from core.types import LEAF_EXPAND, OUTCOME_SUCCESS, Rollout, VIRTUAL_EXPAND
from llm.judge import EloJudge
from memory.bank import MemoryBank
from memory.extractor import LessonExtractor
from prompting.builder import PromptBuilder
from runio.experiment import ExperimentIO
from search.dpuct import DPUCT
from search.elo import EloRatings
from search.pairing import build_pairings


class Engine:
    def __init__(self, cfg, resolution=None, backbone=None, example=None):
        self.cfg = cfg
        self.resolution = resolution

        self.example = example or load_example(cfg)
        self.backbone = backbone or self._build_backbone()

        self.tree = SearchTree(max_archive_size=cfg.search.max_archive_size,
                               fail_reward=cfg.verifier.fail_reward)
        self.dpuct = DPUCT(cfg.search)
        self.elo = EloRatings(k_factor=cfg.elo.k_factor,
                              initial_rating=cfg.elo.initial_rating,
                              scale=cfg.elo.scale)

        self.memory = (MemoryBank(cfg.memory, backbone=self.backbone)
                       if cfg.memory.enabled else None)
        self.builder = PromptBuilder(self.example, self.memory, cfg.memory.top_m)
        self.extractor = (LessonExtractor(cfg.memory, self.backbone,
                                          self.example.meta_description())
                          if cfg.memory.enabled else None)
        self.judge = (EloJudge(cfg.elo, self.backbone,
                               self.example.meta_description())
                      if cfg.elo.enabled else None)
        self.trainer = self._build_trainer()
        self.generator = self._build_generator()

        self.io = ExperimentIO(cfg, resolution)
        self.history: List[dict] = []

        import random
        self.rng = random.Random(cfg.run.seed)

    # ------------------------------------------------------------------
    def _build_backbone(self):
        if (self.cfg.model.backend or "").lower() == "mock":
            from llm.mock import MockBackbone
            print("[engine] using the MOCK backbone -- plumbing only, no real model")
            return MockBackbone(self.cfg.model, seed=self.cfg.run.seed).load()
        from llm.backbone import Backbone
        return Backbone(self.cfg.model).load()

    def _build_trainer(self):
        if not self.cfg.rl.enabled:
            return None
        if (self.cfg.model.backend or "").lower() == "mock":
            print("[engine] rl disabled: the mock backbone has no parameters")
            return None
        from rl.trainer import TestTimeTrainer
        return TestTimeTrainer(self.cfg, self.backbone, self.builder)

    def _build_generator(self):
        from llm.generation import InProcessGenerator, build_generator
        if (self.cfg.model.backend or "").lower() == "mock":
            return InProcessGenerator(self.backbone, self.cfg.generation,
                                      progress=self.cfg.run.progress)
        return build_generator(self.cfg, self.backbone)

    # ------------------------------------------------------------------
    def seed(self) -> None:
        for state in self.example.seed_nodes(self.cfg.search.num_seed_nodes):
            self.tree.add_root(reward=self.cfg.verifier.fail_reward, **state)
        self.tree.recompute()

    # ------------------------------------------------------------------
    # Elo debate (Sec. 2.1)
    # ------------------------------------------------------------------
    def run_debates(self, step: int) -> dict:
        # alpha = 1 means Eq. 3 ignores the Elo term, so the debates would be
        # paid for and then discarded.
        if self.judge is None or self.cfg.search.alpha >= 1.0:
            return {"matches": 0, "skipped": "alpha=1 uses the rank signal only"}

        candidates = self.tree.top_k(int(self.cfg.elo.candidate_top_k))
        if len(candidates) < 2:
            return {"matches": 0}

        ids = [n.id for n in candidates]
        self.elo.ensure(ids)
        pairs = build_pairings(ids, self.cfg.elo.pairing_mode,
                               self.cfg.elo.num_matches,
                               self.cfg.elo.rounds_per_step, self.rng)
        if not pairs:
            return {"matches": 0}

        renders = {n.id: self.example.render_for_judge(n) for n in candidates}
        verdicts = self.judge.judge(pairs, renders)
        self.elo.apply(verdicts)

        for idx, verdict in enumerate(verdicts):
            self.io.save_elo_match(step, idx, {
                "node_a": verdict.node_a, "node_b": verdict.node_b,
                "y_a": verdict.y,
                "reward_a": self.tree.get(verdict.node_a).reward,
                "reward_b": self.tree.get(verdict.node_b).reward,
            }, response_text=verdict.raw)
        self.io.save_elo_standings(step, self.elo.standings(ids))
        return {"matches": len(verdicts), "rated": len(ids)}

    # ------------------------------------------------------------------
    # One step
    # ------------------------------------------------------------------
    def run_step(self, step: int) -> dict:
        started = time.time()
        self.tree.recompute()

        elo_stats = self.run_debates(step)
        elo_standardized = None
        if self.cfg.elo.enabled and self.cfg.search.alpha < 1.0:
            ids = [n.id for n in self.tree.evaluated()]
            if ids:
                values = self.elo.standardized(ids)
                elo_standardized = {i: float(v) for i, v in zip(ids, values)}

        logits = self.dpuct.compute_logits(self.tree, elo_standardized)
        targets = self.dpuct.select(self.tree, logits)
        if not targets:
            return {"step": step, "error": "no expandable targets"}

        if self.cfg.run.progress:
            planned = sum(t.num_children for t in targets)
            leaves = sum(1 for t in targets if t.kind == LEAF_EXPAND)
            print(f"[step {step:02d}] {len(targets)} targets "
                  f"({leaves} leaf / {len(targets) - leaves} virtual) "
                  f"-> B_t = {planned} rollouts", flush=True)

        # ---- build the prompts of Fig. 1 -----------------------------
        jobs = []
        prompts = {}
        for group_id, target in enumerate(targets):
            parent = self.tree.get(target.node_id)
            messages = self.builder.build(parent, self.builder.retrieve(parent))
            prompts[group_id] = (target, messages, self.backbone.render(messages))
            jobs.append((group_id, messages, target.num_children))

        # ---- generate: B_t in [n, nk] --------------------------------
        adapter_path = None
        if getattr(self.generator, "name", "") == "pool":
            # The workers hold their own copy of the base model, so the freshly
            # updated LoRA has to reach them through the filesystem.
            saved = self.io.save_adapter(self.backbone, step)
            adapter_path = str(saved) if saved else None

        by_group = self.generator.generate(jobs, adapter_path)

        rollouts: List[Rollout] = []
        for group_id, samples in sorted(by_group.items()):
            target, messages, prompt_text = prompts[group_id]
            for text, token_ids in samples:
                rollouts.append(Rollout(
                    target_key=target.key, parent_id=target.node_id,
                    group_id=group_id, prompt_messages=messages,
                    prompt_text=prompt_text, response_text=text,
                    token_ids=token_ids,
                ))

        # ---- verify --------------------------------------------------
        # Serial subprocesses, each up to verifier.timeout_s, so this is the
        # second place a step can sit silent for a long time.
        from runio.progress import make_bar

        results = []
        bar = make_bar(len(rollouts), "verify", unit="rollout",
                       enabled=self.cfg.run.progress)
        num_valid = 0
        try:
            for rollout in rollouts:
                parent = self.tree.get(rollout.parent_id)
                try:
                    result = self.example.verify(rollout.response_text, parent)
                except Exception as e:
                    from core.types import VerifyResult
                    result = VerifyResult(
                        reward=self.cfg.verifier.fail_reward, valid=False,
                        msg=f"verifier_crashed: {e}",
                        feedback=f"The verifier itself raised:\n{traceback.format_exc()}")
                results.append(result)
                num_valid += bool(result.valid)
                best_so_far = max((r.reward for r in results), default=0.0)
                # Postfix before update: update() is what draws, and a throttled
                # redraw would otherwise pair the new count with stale text.
                bar.set_postfix_str(
                    f"valid={num_valid}/{len(results)} best={best_so_far:.4f}")
                bar.update(1)
        finally:
            bar.close()

        # ---- insert into the tree ------------------------------------
        for idx, (rollout, result) in enumerate(zip(rollouts, results)):
            node = self.tree.add_child(
                rollout.parent_id, step=step, reward=result.reward,
                raw_score=result.raw_score, valid=result.valid,
                feedback=result.feedback, msg=result.msg, code=result.code,
                response=rollout.response_text,
            )
            self.io.save_rollout(step, rollout.group_id, idx,
                                 rollout.response_text, {
                                     "node_id": node.id,
                                     "parent_id": rollout.parent_id,
                                     "group_id": rollout.group_id,
                                     "reward": result.reward,
                                     "raw_score": result.raw_score,
                                     "valid": result.valid, "msg": result.msg,
                                     "feedback": result.feedback[:2000],
                                     "stdout": result.stdout[:2000],
                                 })
        self.tree.recompute()
        self.tree.prune()

        # ---- memory (Sec. 2.2) ---------------------------------------
        from runio.progress import PhaseTimer
        show = self.cfg.run.progress
        if self.memory is not None:
            with PhaseTimer("extract lessons", show):
                memory_stats = self.update_memory(results, step)
        else:
            memory_stats = self.update_memory(results, step)

        # ---- test-time RL (Sec. 2.3) ---------------------------------
        rl_stats = {"updated": False, "reason": "rl disabled"}
        if self.trainer is not None:
            try:
                rl_stats = self.trainer.step(rollouts, results, step)
            except Exception as e:
                rl_stats = {"updated": False, "reason": f"crashed: {e!r}"}
                print(f"[rl] update failed at step {step}: {e!r}")
                traceback.print_exc()

        # ---- summary -------------------------------------------------
        rewards = [r.reward for r in results]
        valid = [r for r in results if r.valid]
        best = self.tree.best()
        summary = {
            "step": step,
            "num_targets": len(targets),
            "batch_size": len(rollouts),
            "leaf_targets": sum(1 for t in targets if t.kind == LEAF_EXPAND),
            "virtual_targets": sum(1 for t in targets if t.kind == VIRTUAL_EXPAND),
            "num_valid": len(valid),
            "valid_rate": len(valid) / max(1, len(results)),
            "mean_reward": sum(rewards) / max(1, len(rewards)),
            "max_reward": max(rewards) if rewards else 0.0,
            "best_so_far": best.reward if best else 0.0,
            "archive_size": len(self.tree),
            "elo": elo_stats,
            "memory": memory_stats,
            "rl": rl_stats,
            "seconds": round(time.time() - started, 2),
        }
        self.io.save_step_summary(step, summary)
        self.history.append(summary)
        return summary

    # ------------------------------------------------------------------
    def update_memory(self, results, step: int) -> dict:
        if self.memory is None or self.extractor is None:
            return {"added": 0, "total": 0}

        successes = [(r.code, r.reward) for r in results if not r.failed]
        failures = [(r.code, r.feedback or r.msg) for r in results if r.failed]
        if not successes and not failures:
            return {"added": 0, "total": len(self.memory)}

        try:
            lessons = self.extractor.extract(successes, failures, step)
        except Exception as e:
            print(f"[memory] extraction failed at step {step}: {e!r}")
            return {"added": 0, "total": len(self.memory), "error": repr(e)}

        added = self.memory.add(lessons)
        return {"added": added, "total": len(self.memory),
                "successes": len(successes), "failures": len(failures),
                **self.memory.counts()}

    # ------------------------------------------------------------------
    def run(self) -> int:
        cfg = self.cfg
        print(f"[engine] example={cfg.example.name} steps={cfg.run.max_steps} "
              f"n={cfg.search.n_select} k={cfg.search.k_children} "
              f"-> B_t in [{cfg.search.n_select}, "
              f"{cfg.search.n_select * cfg.search.k_children}]")
        print(f"[engine] run directory: {self.io.root}")

        self.seed()
        for step in range(int(cfg.run.max_steps)):
            summary = self.run_step(step)
            if summary.get("error"):
                print(f"[step {step}] {summary['error']}")
                break
            self._print_step(summary)
            self.io.save_tree(self.tree)
            self.io.save_memory(self.memory)

        self.generator.shutdown()
        best = self.tree.best()
        self.io.save_final(best, int(cfg.run.max_steps), self.history)
        if best is not None:
            print(f"\n[engine] best {self.example.metric_name} = "
                  f"{best.display_score():.6f} (step {best.step})")
            target = getattr(self.example, "target", None)
            if target is not None:
                print(f"[engine] target was {target}")
        else:
            print("\n[engine] no valid candidate was found")
        print(f"[engine] results in {self.io.root}")
        return 0

    @staticmethod
    def _print_step(s: dict) -> None:
        rl = s.get("rl", {})
        rl_note = "rl+" if rl.get("updated") else f"rl-({rl.get('reason', '')[:40]})"
        print(f"[step {s['step']:02d}] "
              f"B={s['batch_size']:3d} "
              f"({s['leaf_targets']}L/{s['virtual_targets']}V) "
              f"valid={s['valid_rate']:.0%} "
              f"max={s['max_reward']:.4f} best={s['best_so_far']:.4f} "
              f"|D|={s['archive_size']:4d} "
              f"mem={s.get('memory', {}).get('total', 0):3d} "
              f"elo={s.get('elo', {}).get('matches', 0):3d} "
              f"{rl_note} {s['seconds']:.1f}s")
