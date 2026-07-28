"""
Test-time RL: one clipped policy-gradient step on the LoRA parameters per
evolution step (Sec. 2.3).

Per step:

  1. group the rollouts by their generating prompt x_p
  2. Eq. 8  -- group-relative advantage from the exponentially tilted rewards
  3. Eq. 9  -- for each FAILED rollout, per-token advantage from the
               feedback-conditioned self-teacher (same weights, feedback in
               context), computed once and detached
  4. combine, then one clipped update (Eq. 10, 11) with a KL anchor to theta_0

Only the LoRA parameters move; the backbone is frozen throughout, which is what
makes adapting mid-search cheap enough to do every step.
"""

import numpy as np

from prompting.builder import PromptBuilder
from rl.advantage import (clip_advantages, combine, feedback_advantages,
                          group_relative_advantages)
from rl.objective import microbatch_loss


class TestTimeTrainer:
    def __init__(self, cfg, backbone, builder: PromptBuilder):
        self.cfg = cfg
        self.rl = cfg.rl
        self.backbone = backbone
        self.builder = builder
        self._optimizer = None

    # ------------------------------------------------------------------
    def optimizer(self):
        if self._optimizer is None:
            import torch
            trainable = [p for p in self.backbone.model.parameters()
                         if p.requires_grad]
            if not trainable:
                raise RuntimeError(
                    "no trainable parameters found -- is the LoRA adapter attached?")
            self._optimizer = torch.optim.AdamW(
                trainable, lr=float(self.rl.learning_rate))
        return self._optimizer

    # ------------------------------------------------------------------
    def _encode_prompt(self, text: str):
        tok = self.backbone.tokenizer
        ids = tok(text, add_special_tokens=False,
                  truncation=True, max_length=self.cfg.model.max_seq_length)
        return list(ids["input_ids"])

    def _feedback_advantage(self, rollout, feedback: str, response_ids):
        """
        Eq. 9: log q(y | x, f, y_<l) - log pi(y | x, y_<l).

        Both passes use the SAME parameters; the only difference is whether the
        verifier's feedback is in the context. Detached -- a target, not a path.
        """
        teacher_messages = self.builder.reprompt(rollout.prompt_messages, feedback)
        teacher_ids = self._encode_prompt(self.backbone.render(teacher_messages))
        policy_ids = rollout.prompt_token_ids or self._encode_prompt(
            rollout.prompt_text)

        teacher_lp = self.backbone.token_logprobs(teacher_ids, response_ids)
        policy_lp = self.backbone.token_logprobs(policy_ids, response_ids)
        return feedback_advantages(
            np.asarray(teacher_lp.detach().float().cpu()),
            np.asarray(policy_lp.detach().float().cpu()),
        )

    # ------------------------------------------------------------------
    def step(self, rollouts, results, step_idx: int) -> dict:
        """
        rollouts: list[Rollout]; results: list[VerifyResult], index-aligned.
        Returns a stats dict; a skipped update is reported, never silent.
        """
        import torch

        stats = {"updated": False, "reason": "", "tokens": 0,
                 "policy_loss": 0.0, "kl": 0.0, "num_rollouts": len(rollouts)}

        if not self.rl.enabled:
            stats["reason"] = "rl.enabled=false"
            return stats

        usable = [(r, v) for r, v in zip(rollouts, results) if r.token_ids]
        if not usable:
            stats["reason"] = "no rollouts carried token ids"
            return stats

        # ---- Eq. 8, per group ----------------------------------------
        groups = {}
        for idx, (rollout, _) in enumerate(usable):
            groups.setdefault(rollout.group_id, []).append(idx)

        adv_scalar = np.zeros(len(usable), dtype=np.float64)
        for indices in groups.values():
            rewards = [usable[i][1].reward for i in indices]
            for pos, i in enumerate(indices):
                adv_scalar[i] = group_relative_advantages(rewards, self.rl.beta)[pos]

        if self.rl.skip_degenerate_batches and np.allclose(adv_scalar, 0.0):
            stats["reason"] = ("every group had identical rewards -- Eq. 8 gives "
                               "zero advantage, so the update would be a no-op")
            return stats

        # ---- per-rollout tensors -------------------------------------
        self.backbone.set_training_mode()
        device = self.backbone.device
        prepared = []
        total_tokens = 0

        for i, (rollout, verdict) in enumerate(usable):
            response_ids = list(rollout.token_ids)
            prompt_ids = rollout.prompt_token_ids or self._encode_prompt(
                rollout.prompt_text)

            fb = None
            if (self.rl.use_feedback_signal and verdict.failed
                    and verdict.feedback and self.rl.lambda_feedback):
                try:
                    fb = self._feedback_advantage(rollout, verdict.feedback,
                                                  response_ids)
                except Exception as e:                    # never lose the batch
                    print(f"[rl] feedback advantage failed for rollout {i}: {e!r}")

            advantages = combine(adv_scalar[i], fb, verdict.failed,
                                 self.rl.lambda_feedback, len(response_ids))
            advantages = clip_advantages(advantages, self.rl.advantage_clip)

            with torch.no_grad():
                logp_old = self.backbone.token_logprobs(
                    prompt_ids, response_ids).detach()
                logp_ref = (self.backbone.reference_logprobs(
                    prompt_ids, response_ids).detach()
                    if self.rl.kl_coef else None)

            prepared.append({
                "prompt_ids": prompt_ids,
                "response_ids": response_ids,
                "advantages": torch.tensor(advantages, dtype=torch.float32,
                                           device=device),
                "logp_old": logp_old,
                "logp_ref": logp_ref,
            })
            total_tokens += len(response_ids)

        if total_tokens == 0:
            stats["reason"] = "no response tokens"
            return stats

        # ---- Eq. 10, 11 ----------------------------------------------
        optimizer = self.optimizer()
        micro = max(1, int(self.rl.microbatch_size))
        policy_sum = kl_total = 0.0

        for _ in range(max(1, int(self.rl.updates_per_step))):
            optimizer.zero_grad(set_to_none=True)
            policy_sum = kl_total = 0.0

            for start in range(0, len(prepared), micro):
                for item in prepared[start:start + micro]:
                    logp_new = self.backbone.token_logprobs(
                        item["prompt_ids"], item["response_ids"], with_grad=True)
                    loss, parts = microbatch_loss(
                        logp_new, item["logp_old"], item["logp_ref"],
                        item["advantages"],
                        clip_epsilon=float(self.rl.clip_epsilon),
                        kl_coef=float(self.rl.kl_coef),
                        normalizer=total_tokens,
                    )
                    loss.backward()
                    policy_sum += parts["policy_loss_sum"]
                    kl_total += parts["kl_sum"]

            if self.rl.grad_clip:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.backbone.model.parameters() if p.requires_grad],
                    float(self.rl.grad_clip))
            optimizer.step()

        stats.update({
            "updated": True,
            "tokens": total_tokens,
            "policy_loss": policy_sum / total_tokens,
            "kl": kl_total / total_tokens,
            "mean_abs_advantage": float(np.abs(adv_scalar).mean()),
            "num_failed": sum(1 for _, v in usable if v.failed),
        })
        return stats
