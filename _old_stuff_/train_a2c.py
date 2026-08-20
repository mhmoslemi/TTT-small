"""
TTT-Discover - multi-problem local runner.

Config():  defaults  <  YAML  <  CLI flags
"""

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


import os
import argparse
import time
from dataclasses import dataclass, field, fields
from typing import Tuple
import numpy as np
import yaml


# ======================================================================
# Config
# ======================================================================
@dataclass
class Config:
    # Problem selector
    problem: str = "circle_packing"

        # "circle_packing", "erdos", "ac1", "ac2",
        # "denoising", "gpu_mode", "ahc",


    problem_type: str = ""        # ac1/ac2, trimul/mla_decode_nvidia, etc.

    # Model
    model_name: str = "Qwen/Qwen3-8B"
    backend: str = "auto"        # "auto" | "unsloth" | "hf"
    max_seq_length: int = 32000
    load_in_4bit: bool = False
    lora_rank: int = 32
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    target_modules: Tuple[str, ...] = (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    )


    num_circles: int = 26
    target: float = 2.635983
    sandbox_timeout_s: float = 30.0

    # RL hyperparameters
    num_steps: int = 50
    groups_per_step: int = 8
    group_size: int = 64
    num_seed_states: int = 16
    learning_rate: float = 4e-5
    kl_penalty_coef: float = 0.1
    max_new_tokens: int = 4200
    grad_clip: float = 1.0

    train_examples_per_microbatch: int = 1

    # Sampling
    temperature: float = 1.0
    top_p: float = 1.0

    # PUCT
    puct_c: float = 1.0
    max_buffer_size: int = 1000
    topk_children_per_parent: int = 2

    # Misc
    seed: int = 42
    print_responses: int = 0           # how many rollouts to print per step

    # Multi-GPU generation
    num_gpus: int = 4
    # num_gpus: int = 1
    # gpu_ids: str = "0,1,2,3,4,5,6,7"
    gpu_ids: str = "1,2,4,6"
    # gpu_ids: str = "1"


# ======================================================================
# CLI parsing + config loading (defaults < YAML < CLI)
# ======================================================================
def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="TTT-Discover multi-problem runner")
    # Problem selection
    p.add_argument("--problem", default="circle_packing",
                   help="Problem name. Loads configs/<problem>.yaml unless --config is given. "
                        "One of: circle_packing, erdos, ac1, ac2, denoising, gpu_mode.")
    p.add_argument("--config", default=None,
                   help="Explicit path to a YAML config (overrides the --problem lookup).")
    p.add_argument("--problem-type", default=None,
                   help="Sub-type for multi-mode problems (ac1/ac2, trimul/mla_decode_nvidia).")

    # CLI overrides - all default None so we can tell 'not given' from 'given'.
    p.add_argument("--backend", choices=["auto", "unsloth", "hf"], default=None)
    p.add_argument("--model-name", default=None)
    p.add_argument("--load-in-4bit", action="store_const", const=True, default=None)
    p.add_argument("--max-seq-length", type=int, default=None)
    p.add_argument("--lora-rank", type=int, default=None)
    p.add_argument("--lora-alpha", type=int, default=None)
    p.add_argument("--lora-dropout", type=float, default=None)
    p.add_argument("--num-circles", type=int, default=None)
    p.add_argument("--target", type=float, default=None)
    p.add_argument("--sandbox-timeout-s", type=float, default=None)
    p.add_argument("--num-steps", type=int, default=None,
                   help="Number of TTT-Discover steps (paper: 50)")
    p.add_argument("--groups-per-step", type=int, default=None,
                   help="Number of parent states sampled per step (paper: 8)")
    p.add_argument("--group-size", type=int, default=None,
                   help="Rollouts per parent per step (paper: 64)")
    p.add_argument("--num-seed-states", type=int, default=None)
    p.add_argument("--max-new-tokens", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--kl-penalty-coef", type=float, default=None)
    p.add_argument("--grad-clip", type=float, default=None)
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--top-p", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--print-responses", type=int, default=None)
    p.add_argument("--num-gpus", type=int, default=None,
                   help="Number of GPUs for parallel generation. 1 = single-process "
                        "in-line generation (no worker pool). >1 spawns that many "
                        "plain-HF generation workers, one per GPU.")
    p.add_argument("--gpu-ids", type=str, default=None,
                   help="Comma-separated physical GPU ids for the workers, e.g. "
                        "'0,1,2,3,4,5,6,7'. Defaults to 0..num_gpus-1.")
    return p


# CLI arg name -> config key (only where they differ)
_CLI_TO_CFG = {"lr": "learning_rate"}


def load_config():
    """
    Merge Config() defaults < YAML(configs/<problem>.yaml or --config) < CLI flags.

    Returns (cfg, merged) where:
      cfg    is a Config built from the engine-level fields, and
      merged is the full dict (including problem-only keys like num_circles,
             problem_type, budget_s, score_scale, gpu_type, task_yaml, lib_dir),
             which is what the problem registry consumes.
    """
    args = _build_arg_parser().parse_args()

    # 1) defaults from the dataclass
    merged = {f.name: getattr(Config(), f.name) for f in fields(Config)}

    # 2) YAML overlay
    cfg_path = args.config or os.path.join("configs", f"{args.problem}.yaml")
    ydict = {}
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            ydict = yaml.safe_load(f) or {}
        merged.update(ydict)
        print(f"[config] loaded {cfg_path}")
    elif args.config is not None:
        raise FileNotFoundError(f"--config path not found: {cfg_path}")
    else:
        print(f"[config] no YAML at {cfg_path}; using Config() defaults + CLI")

    # The registry routing key is the YAML's `problem` field when present
    # (this lets e.g. configs/gpu_mode_trimul.yaml declare `problem: gpu_mode`
    # while --problem just selects the file). With no YAML, --problem is the key.
    merged["problem"] = ydict.get("problem", args.problem)

    # 3) CLI overlay (only explicitly-provided values)
    skip = {"problem", "config", "problem_type"}
    for arg_name, value in vars(args).items():
        if arg_name in skip or value is None:
            continue
        key = _CLI_TO_CFG.get(arg_name, arg_name)
        merged[key] = value
    if args.problem_type is not None:
        merged["problem_type"] = args.problem_type

    # 4) build the Config from the fields it knows; leave the rest in `merged`
    known = {f.name for f in fields(Config)}
    cfg_kwargs = {k: v for k, v in merged.items() if k in known}
    cfg = Config(**cfg_kwargs)
    return cfg, merged


# ======================================================================
# Generation
# ======================================================================
def _generate_batch(model, tokenizer, inputs, input_len, n_samples, cfg):
    """
    Generate n_samples completions for a SINGLE prompt in ONE batched
    model.generate() call (via num_return_sequences). Returns a list of
    (text, gen_token_ids).
    """
    import torch
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id or eos_id

    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=cfg.max_new_tokens,
            do_sample=True,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            pad_token_id=pad_id,
            num_return_sequences=n_samples,
        )
    results = []
    for i in range(out.shape[0]):
        gen_ids = out[i, input_len:].tolist()
        if eos_id is not None and eos_id in gen_ids:
            gen_ids = gen_ids[: gen_ids.index(eos_id) + 1]
        text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        results.append((text, gen_ids))
    return results


def generate_responses(model, tokenizer, prompt_text: str, group_size: int, cfg: Config):
    """
    Generate `group_size` responses from a single prompt, batched.

    Try to generate all `group_size` at once. If OOMs, halve the
    per-call batch size and retry, accumulating until we have group_size
    responses. This keeps the algorithm identical (still group_size IID
    samples from the same policy) while using the GPU in parallel.

    Returns (list of (text, gen_token_ids), prompt_len_in_tokens).
    """
    import torch
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    input_len = inputs.input_ids.shape[1]

    responses = []
    remaining = group_size
    # Start by trying the whole group in one call.
    batch = group_size

    while remaining > 0:
        n = min(batch, remaining)
        try:
            chunk = _generate_batch(model, tokenizer, inputs, input_len, n, cfg)
            responses.extend(chunk)
            remaining -= n
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if batch == 1:
                # Can't even do one - re-raise, nothing we can do
                raise
            batch = max(1, batch // 2)
            print(f"  [oom] halving generation batch size to {batch}")

    return responses, input_len


# ======================================================================
# Logprob computation
# ======================================================================
def compute_token_logprobs(model, prompt_ids, response_ids, with_grad: bool):
    """
    Returns the per-token log-probabilities of the response under the model.

    prompt_ids:   (1, P) tensor
    response_ids: (1, R) tensor
    Output:       (R,) tensor of token logprobs
    """
    import torch
    import torch.nn.functional as F

    full_ids = torch.cat([prompt_ids, response_ids], dim=1)
    context = torch.enable_grad() if with_grad else torch.no_grad()
    with context:
        out = model(full_ids)
        logits = out.logits  # (1, T, V)
        P = prompt_ids.shape[1]
        R = response_ids.shape[1]
        # Predict response token at position P+k from logits at position P+k-1
        pred_logits = logits[:, P - 1 : P - 1 + R, :]
        log_probs = F.log_softmax(pred_logits.float(), dim=-1)
        gathered = log_probs.gather(2, response_ids.unsqueeze(-1)).squeeze(-1)  # (1, R)
    return gathered.squeeze(0)


# ======================================================================
# LoRA adapter sync (main process -> generation workers)
# ======================================================================
def _adapter_dir(exp_dir, step_idx):
    from pathlib import Path
    return str(Path(exp_dir) / f"adapter_step{step_idx:03d}")


def _adapter_exists(exp_dir):
    from pathlib import Path
    p = Path(exp_dir)
    return any(p.glob("adapter_step*"))


def _save_adapter(model, exp_dir, step_idx):
    """
    Save the current LoRA adapter to disk so generation workers can load it.
    Returns the directory path. Cleans up the previous step's adapter to
    avoid filling the disk (we only ever need the latest).
    """
    import shutil
    from pathlib import Path

    out_dir = _adapter_dir(exp_dir, step_idx)
    # PEFT/Unsloth models support save_pretrained, which writes just the adapter
    model.save_pretrained(out_dir)

    # Remove older adapter dirs (keep only the current one)
    for old in Path(exp_dir).glob("adapter_step*"):
        if str(old) != out_dir:
            try:
                shutil.rmtree(old)
            except Exception:
                pass

    return out_dir



# =====================================================================
# RL PART -- A2C (learned value baseline)
# =====================================================================
# Same rollout/reward scaffolding as your TTT runner. The group-mean baseline
# is replaced by a LEARNED state value V(s): advantage A_i = R_i - V(s_parent).
# One-step (terminal-reward) actor-critic, so there is no bootstrapping and the
# critic target is just the realized reward; V(s) regresses toward E[R|s].
#
# Design choices that isolate the single axis (baseline source):
#   * The value head reads the policy trunk's hidden state for the prompt, but
#     the hidden state is DETACHED before the head, so the critic does not
#     perturb the actor's representation.
#   * V(s) is computed once per parent and shared across the group's rollouts.
#   * The value head + its own AdamW are lazily created and stashed on `model`,
#     so the signature and main() are unchanged. The `optimizer` arg (LoRA/actor
#     params) is used as-is for the policy update.
#
# CAVEAT: at group_size=64 the group sample mean is already a strong estimate of
# V(s), so the expected gain over the REINFORCE control is small; it shows up
# with small groups or thinly-expanded parents.
#
# Knobs (getattr): value_lr (default 1e-3)
# =====================================================================
def _a2c_get_value_head(model, cfg):
    """Lazily build a small value head + its own AdamW, stash on the model."""
    import torch
    import torch.nn as nn
    if getattr(model, "_a2c_value_head", None) is not None:
        return model._a2c_value_head, model._a2c_value_opt

    hidden = None
    c = getattr(model, "config", None)
    if c is not None and getattr(c, "hidden_size", None):
        hidden = c.hidden_size
    if hidden is None:
        base = getattr(model, "base_model", None)
        if base is not None and getattr(base, "config", None):
            hidden = base.config.hidden_size
    if hidden is None:
        raise RuntimeError("could not resolve hidden_size for the A2C value head")

    p = next(model.parameters())
    head = nn.Sequential(
        nn.Linear(hidden, hidden // 4),
        nn.Tanh(),
        nn.Linear(hidden // 4, 1),
    ).to(device=p.device, dtype=torch.float32)
    opt = torch.optim.AdamW(head.parameters(),
                            lr=float(getattr(cfg, "value_lr", 1e-3)))
    model._a2c_value_head = head
    model._a2c_value_opt = opt
    print(f"[init] A2C value head created (hidden={hidden}, "
          f"params={sum(x.numel() for x in head.parameters()):,})")
    return head, opt


def _a2c_state_value(model, head, prompt_ids):
    """V(s): last-layer hidden state of the prompt's last token -> scalar.
    Hidden state DETACHED so the critic does not perturb the actor."""
    import torch
    with torch.no_grad():
        out = model(prompt_ids, output_hidden_states=True)
        h_last = out.hidden_states[-1][:, -1, :].detach().float()   # (1, hidden)
    return head(h_last).squeeze(-1).squeeze(0)                       # scalar (grad in head)


def train_step(backend, model, tokenizer, sampler, optimizer, step_idx: int,
               cfg: Config, exp_dir, problem, gen_pool=None):
    import os
    import torch
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from sampler import State
    from experiment_io import save_rollout
    from problems.base import ParentContext
    from gen_workers import make_progress_bar

    step_t0 = time.time()
    sampler.set_current_step(step_idx)
    parents = sampler.sample_states(cfg.groups_per_step)
    print(f"\n[step {step_idx}] parents picked: {len(parents)}")
    for i, info in enumerate(sampler.last_picks_info):
        tag = "seed" if info["is_seed"] else "expanded"
        print(f"  parent {i} [{tag}]  value={info['value']:.4f}  n={info['n']}  "
              f"Q={info['Q']:.4f}  P={info['P']:.4f}  bonus={info['bonus']:.4f}  "
              f"score={info['score']:.4f}")

    head, value_opt = _a2c_get_value_head(model, cfg)

    all_examples = []
    all_children = []
    value_targets = []   # (prompt_ids, mean_reward) for the critic regression

    # ----- BUILD PROMPTS (one per parent/group) -----
    prompts_by_group = []
    parent_ctxs = []

    for g, parent in enumerate(parents):
        sampler.record_expansion(parent, count=cfg.group_size)
        pc = ParentContext(
            code=parent.code,
            value=parent.value if parent.value is not None else 0.0,
            raw_score=parent.raw_score,
            construction=parent.construction,
        )
        parent_ctxs.append(pc)
        messages = problem.build_prompt(pc)
        try:
            prompt_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            prompt_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
        prompts_by_group.append(prompt_text)

    num_groups = len(parents)
    total_rollouts = num_groups * cfg.group_size

    # ----- REWARD POOL (CPU), runs concurrently with generation -----
    n_reward_workers = getattr(cfg, "reward_workers", 0)
    if not n_reward_workers:
        n_reward_workers = max(1, (os.cpu_count() or 8) - max(0, cfg.num_gpus))
    reward_pool = ThreadPoolExecutor(max_workers=n_reward_workers)

    group_responses = {g: [] for g in range(num_groups)}
    reward_futures = {g: [] for g in range(num_groups)}

    def _submit_rollout(g, text, token_ids):
        group_responses[g].append((text, token_ids))
        fut = reward_pool.submit(
            problem.compute_reward, text, parent_ctxs[g], cfg.sandbox_timeout_s
        )
        reward_futures[g].append(fut)

    # ----- ROLLOUTS (streamed) + dispatch rewards as each rollout lands -----
    rollout_t0 = time.time()
    try:
        if gen_pool is not None:
            adapter_path = _save_adapter(model, exp_dir, step_idx)
            for group_idx, job_results in gen_pool.iter_group_jobs(
                    prompts_by_group=prompts_by_group,
                    group_size=cfg.group_size,
                    adapter_path=adapter_path,
                    max_new_tokens=cfg.max_new_tokens,
                    temperature=cfg.temperature,
                    top_p=cfg.top_p,
            ):
                for (text, token_ids) in job_results:
                    _submit_rollout(group_idx, text, token_ids)
        else:
            backend.set_inference_mode()
            gen_bar = make_progress_bar(total_rollouts, desc="rollouts")
            try:
                for g, prompt_text in enumerate(prompts_by_group):
                    responses, _ = generate_responses(
                        model, tokenizer, prompt_text, cfg.group_size, cfg
                    )
                    for (text, token_ids) in responses:
                        _submit_rollout(g, text, token_ids)
                    gen_bar.update(len(responses))
            finally:
                gen_bar.close()

        all_futs = [f for g in range(num_groups) for f in reward_futures[g]]
        eval_bar = make_progress_bar(len(all_futs), desc="evaluating")
        try:
            for _ in as_completed(all_futs):
                eval_bar.update(1)
        finally:
            eval_bar.close()
    finally:
        reward_pool.shutdown(wait=True)

    # ----- SCORE + ADVANTAGE + SAVE + COLLECT TRAINING EXAMPLES -----
    backend.set_training_mode()
    for g, parent in enumerate(parents):
        prompt_text = prompts_by_group[g]
        responses = group_responses[g]
        futs = reward_futures[g]
        pc = parent_ctxs[g]

        rewards = []
        codes = []
        valids = []
        outs = []
        for r_idx, (text, token_ids) in enumerate(responses):
            res = futs[r_idx].result()
            rewards.append(res.reward)
            codes.append(res.code or "")
            valids.append(res.valid)
            outs.append(res)

        rewards_np = np.array(rewards, dtype=np.float64)

        prompt_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(model.device)

        # >>> RL CHANGE: learned-baseline advantage  A_i = R_i - V(s)  <<<
        v_s = _a2c_state_value(model, head, prompt_ids)     # scalar tensor (grad in head)
        v_scalar = float(v_s.detach().item())
        advantages = rewards_np - v_scalar
        value_targets.append((prompt_ids, float(rewards_np.mean())))

        print(f"  group {g}: rewards min={rewards_np.min():.4f} "
              f"mean={rewards_np.mean():.4f} max={rewards_np.max():.4f}  "
              f"valid={sum(valids)}/{len(valids)}  V(s)={v_scalar:.4f}  (A2C)")

        for r_idx, (text, token_ids) in enumerate(responses):
            res = outs[r_idx]
            meta = {
                "step": step_idx,
                "group": g,
                "rollout": r_idx,
                "reward": float(rewards[r_idx]),
                "raw_score": (float(res.raw_score) if res.raw_score is not None else None),
                "valid": bool(valids[r_idx]),
                "parsed": bool(res.parsed),
                "ran": bool(res.ran),
                "msg": res.msg,
                "advantage": float(advantages[r_idx]) if hasattr(advantages, "__len__") else 0.0,
                "algo": "a2c",
                "value_baseline": v_scalar,
                "n_response_tokens": len(token_ids),
                "sandbox_stdout": (res.stdout or "")[:2000],
                "parent_value": float(parent.value) if parent.value is not None else None,
                "parent_is_seed": parent.id in sampler._seed_ids,
            }
            save_rollout(exp_dir, step_idx, g, r_idx, text, meta)

        for r_idx, (text, token_ids) in enumerate(responses):
            res = outs[r_idx]
            if valids[r_idx] and codes[r_idx]:
                child = State.make(
                    timestep=step_idx,
                    value=rewards[r_idx],
                    code=codes[r_idx],
                    raw_score=res.raw_score,
                    construction=res.construction,
                )
                all_children.append((child, parent))

        if float(rewards_np.max() - rewards_np.min()) < 1e-12:
            continue

        for (text, token_ids), adv in zip(responses, advantages):
            if len(token_ids) == 0:
                continue
            response_ids = torch.tensor([token_ids], device=model.device)
            all_examples.append({
                "prompt_ids": prompt_ids,
                "response_ids": response_ids,
                "advantage": float(adv),
            })

    rollout_time = time.time() - rollout_t0
    print(f"[step {step_idx}] rollout+eval time: {rollout_time:.1f}s  "
          f"training examples: {len(all_examples)}  new children: {len(all_children)}")

    sampler.update(all_children)

    # ----- CRITIC update: regress V(s) -> mean realized reward (MSE) -----
    if value_targets:
        value_opt.zero_grad()
        critic_loss_total = 0.0
        for prompt_ids, target_r in value_targets:
            v_pred = _a2c_state_value(model, head, prompt_ids)   # scalar (grad in head)
            target = torch.tensor(target_r, device=v_pred.device, dtype=v_pred.dtype)
            cl = (v_pred - target) ** 2
            (cl / len(value_targets)).backward()
            critic_loss_total += float(cl.detach().item())
        torch.nn.utils.clip_grad_norm_(head.parameters(), max_norm=cfg.grad_clip)
        value_opt.step()
        print(f"[step {step_idx}] critic MSE: {critic_loss_total / len(value_targets):.6f}")

    if not all_examples:
        print(f"[step {step_idx}] no policy signal (all groups had constant reward)")
        return

    # ----- ACTOR update (KL shaping identical to your TTT runner) -----
    backend.set_training_mode()
    optimizer.zero_grad()

    train_t0 = time.time()
    total_loss = 0.0
    total_logp_delta = 0.0
    n_examples = len(all_examples)

    for ex in all_examples:
        pid = ex["prompt_ids"]
        rid = ex["response_ids"]
        adv = ex["advantage"]

        cur_lp = compute_token_logprobs(model, pid, rid, with_grad=True)  # (R,)

        try:
            with backend.disable_adapter(), torch.no_grad():
                base_lp = compute_token_logprobs(model, pid, rid, with_grad=False)
        except Exception as e:
            if not hasattr(train_step, "_kl_warned"):
                print(f"[warn] disable_adapter failed ({e}); training without KL penalty")
                train_step._kl_warned = True
            base_lp = cur_lp.detach()

        logp_diff = (cur_lp - base_lp).detach()
        avg_logp_diff = logp_diff.mean()
        kl_adv = cfg.kl_penalty_coef * (avg_logp_diff - (cur_lp - base_lp))
        eff_adv = adv + kl_adv

        loss = -(eff_adv.detach() * cur_lp).mean()
        (loss / n_examples).backward()

        total_loss += float(loss.detach().item())
        total_logp_delta += float(logp_diff.mean().item())

    import torch as _torch
    _torch.nn.utils.clip_grad_norm_(
        [p for p in model.parameters() if p.requires_grad],
        max_norm=cfg.grad_clip,
    )
    optimizer.step()

    train_time = time.time() - train_t0
    print(f"[step {step_idx}] train time: {train_time:.1f}s  "
          f"avg loss: {total_loss / n_examples:.4f}  "
          f"avg logpi_theta - logpi_base: {total_logp_delta / n_examples:.4f}")

    best = sampler.best_state()
    if best is not None:
        raw = f" raw={best.raw_score:.6f}" if best.raw_score is not None else ""
        print(f"[step {step_idx}] best so far: value={best.value:.6f}{raw}  "
              f"(step total {time.time() - step_t0:.1f}s, archive={sampler.archive_size()})")



# ======================================================================
# Main
# ======================================================================
def main():
    cfg, merged = load_config()

    # Build the problem from the merged config (the registry reads problem-only
    # knobs like num_circles / problem_type / budget_s / score_scale from here).
    from problems.registry import get_problem
    problem = get_problem(cfg.problem, merged)

    print("=" * 70)
    print("TTT-Discover - local multi-problem implementation")
    print("=" * 70)
    print(f"Problem:            {cfg.problem}"
          + (f" ({cfg.problem_type})" if cfg.problem_type else ""))
    print(f"Entrypoint:         {getattr(problem, 'entrypoint', '?')}")
    print(f"Metric:             {getattr(problem, 'metric_name', '?')} "
          f"({'maximize' if getattr(problem, 'maximize', True) else 'minimize'})")
    print(f"Model:              {cfg.model_name}")
    print(f"Backend:            {cfg.backend}")
    print(f"Target:             {cfg.target}")
    print(f"Steps:              {cfg.num_steps}")
    print(f"Groups per step:    {cfg.groups_per_step}")
    print(f"Group size:         {cfg.group_size}")
    print(f"Total rollouts/step: {cfg.groups_per_step * cfg.group_size}")
    print(f"LR:                 {cfg.learning_rate}")
    print(f"KL coef:            {cfg.kl_penalty_coef}")
    print(f"Max new tokens:     {cfg.max_new_tokens}")
    print(f"Sandbox timeout:    {cfg.sandbox_timeout_s}s")
    print("=" * 70)

    # ---- experiment dir ----
    from experiment_io import make_experiment_dir, save_final_summary
    exp_dir = make_experiment_dir(cfg)
    print(f"[init] writing all rollouts to: {exp_dir}")

    # ---- seed states (problem-defined) ----
    seeds = problem.seed_states()
    print(f"[init] problem produced {len(seeds)} seed state(s)")

    # ---- backend + model ----
    # Load backend FIRST so Unsloth can patch transformers if used.
    from model_backend import load_backend
    backend = load_backend(cfg.backend, cfg)
    model, tokenizer = backend.load()

    import torch  # safe to import now
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.learning_rate,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.0
    )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[init] trainable params: {trainable:,} / total {total:,} "
          f"({100 * trainable / total:.2f}%)")

    # ---- sampler ----
    from sampler import PUCTSampler
    sampler = PUCTSampler(
        num_seeds=len(seeds) if seeds else cfg.num_seed_states,
        puct_c=cfg.puct_c,
        max_buffer_size=cfg.max_buffer_size,
        topk_children=cfg.topk_children_per_parent,
        seed_value=0.0,
        seed_states=seeds,
    )
    print(f"[init] sampler archive size = {sampler.archive_size()}")

    # ---- generation pool (multi-GPU) ----
    gen_pool = None
    if cfg.num_gpus and cfg.num_gpus > 1:
        from gen_workers import GenerationPool
        if cfg.gpu_ids:
            gpu_ids = [int(x) for x in cfg.gpu_ids.split(",")]
        else:
            gpu_ids = list(range(cfg.num_gpus))
        print(f"[init] starting generation pool: {cfg.num_gpus} GPUs {gpu_ids}")
        gen_pool = GenerationPool(
            model_name=cfg.model_name,
            num_workers=cfg.num_gpus,
            gpu_ids=gpu_ids,
            max_seq_length=cfg.max_seq_length,
            load_in_4bit=cfg.load_in_4bit,
        )
        print("[init] generation pool ready")
    else:
        print("[init] single-GPU generation (no worker pool)")


    # ---- Elo re-ranker (optional, background thread) ----
    reranker = None
    try:
        from reranker.config import RerankerConfig
        rcfg = RerankerConfig.from_dict(merged)
        if rcfg.enabled:
            from reranker.judges import make_judge
            from reranker.reranker import MultiAgentReRanker
            judge = make_judge(rcfg)
            if judge is not None:
                reranker = MultiAgentReRanker(
                    sampler=sampler,
                    judge=judge,
                    cfg=rcfg,
                    metric_name=getattr(problem, "metric_name", "score"),
                    maximize=getattr(problem, "maximize", True),
                    target=getattr(problem, "target", None),
                    exp_dir=exp_dir,
                )


                reranker.start()
                print(f"[init] Elo re-ranker started "
                      f"(backend={rcfg.backend}, model={rcfg.model}, "
                      f"top_k={rcfg.top_k}, debate={rcfg.debate})")
            else:
                print("[init] Elo re-ranker enabled but judge unavailable; "
                      "continuing with rank-based prior")
        else:
            print("[init] Elo re-ranker disabled")
    except Exception as e:
        print(f"[init] Elo re-ranker setup failed ({e!r}); "
              f"continuing with rank-based prior")
        reranker = None


    # ---- main loop ----
    try:
        for step in range(cfg.num_steps):
            train_step(backend, model, tokenizer, sampler, optimizer, step,
                       cfg, exp_dir, problem, gen_pool)
    finally:
        if reranker is not None:
            print("[shutdown] stopping Elo re-ranker ...")
            reranker.stop()
        if gen_pool is not None:
            print("[shutdown] stopping generation pool ...")
            gen_pool.shutdown()

    # ---- summary ----
    print("\n" + "=" * 70)
    print("TRAINING DONE")
    print("=" * 70)
    best = sampler.best_state()
    if best is not None:
        raw = f"  (raw {getattr(problem, 'metric_name', 'metric')} = {best.raw_score:.6f})" \
            if best.raw_score is not None else ""
        print(f"Best reward (higher=better): {best.value:.6f}{raw}")
        print(f"Found at step:     {best.timestep}")
        print(f"\n--- best code ---\n{best.code}\n--- end ---")
        save_final_summary(exp_dir, best.value, best.code, best.timestep)
    else:
        print("No valid solution was ever produced.")
        save_final_summary(exp_dir, None, None, None)
    print(f"\nAll outputs saved under: {exp_dir}")


if __name__ == "__main__":
    main()
