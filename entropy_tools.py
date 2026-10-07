"""Measurement-only diagnostics, independent of the RL objective and advantages.

Entropy is full-vocabulary Shannon entropy (nats, raw logits / temperature 1)
on the coder's *pre-update training forwards*, at generated response prefixes.
It includes reasoning and answer tokens, excludes prompt/padding, and is not
sampled-token surprisal or the entropy of the temperature/top-p sampler.
No extra transformer forward is launched. Unscored/filtered rollouts remain
missing, with explicit coverage; --no-train yields diversity but no entropy.

Code diversity is a lexical proxy, NOT entropy or mutual information: distance
between normalized Python token 5-gram sets, using deterministic bottom-k
sketches. Only evaluator-extracted, valid programs are compared, within the
same parent/fold; pilots alone are used when available to avoid allocation
changing the comparison. These observations are not a causal ablation.
"""

from __future__ import annotations

import functools
import hashlib
import heapq
import html
import io
import itertools
import json
import math
import os
from pathlib import Path
import random
import tempfile
import threading
import tokenize


def _atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=".entropy-", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(text)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _json(value):
    return json.dumps(value, allow_nan=False, sort_keys=True)


def begin_step(exp_dir, step):
    """A new attempt directory prevents resumed steps reusing stale scores."""
    parent = Path(exp_dir).resolve() / f"step{step:02d}"
    parent.mkdir(parents=True, exist_ok=True)
    return tempfile.mkdtemp(prefix=".entropy-", dir=parent)


def observation_descriptor(directory, group, rollout):
    return {"path": str(Path(directory) / f"g{group}_r{rollout}.json")}


_warning_lock = threading.Lock()
_warned = set()


def _warn_once(kind, error):
    with _warning_lock:
        if kind in _warned:
            return
        _warned.add(kind)
    print(f"[entropy] {kind}: {error}; missing measurements are reported "
          "as unavailable, never as zero", flush=True)


def token_entropy(log_probs, *, detached=False):
    """Bounded, detached diagnostics; retain existing differentiable loss path.

    The extra diagnostic workspace is at most 16 vocabulary rows, independent
    of sequence length. No graph/logit tensor is retained by the recorder.
    A diagnostic allocation failure must not trigger training OOM retries.
    """
    import torch

    if not detached:
        safe = log_probs.masked_fill(~torch.isfinite(log_probs), 0.0)
        return -(log_probs.exp() * safe).sum(dim=-1)
    with torch.no_grad():
        flat = log_probs.detach().reshape(-1, log_probs.shape[-1])
        result = torch.full((flat.shape[0],), float("nan"),
                            device=flat.device, dtype=torch.float32)
        try:
            for start in range(0, flat.shape[0], 16):
                rows = flat[start:start + 16]
                probabilities = rows.exp()
                probabilities.mul_(rows)
                # The entropy convention is 0 * log(0) = 0, but actual NaN
                # or +inf logits must remain invalid measurements.
                probabilities.masked_fill_(torch.isneginf(rows), 0.0)
                result[start:start + 16] = -probabilities.sum(dim=-1)
        except torch.cuda.OutOfMemoryError as error:
            _warn_once("diagnostic workspace unavailable", error)
            result.fill_(float("nan"))
        return result.reshape(log_probs.shape[:-1])


def _record_tensor(descriptor, values):
    import torch

    with torch.no_grad():
        flat = values.detach().reshape(-1)
        # One small CPU transfer per response, not a sync per vocabulary chunk.
        valid = torch.isfinite(flat) & (flat >= -1e-5)
        count, total = torch.stack((
            valid.sum().double(),
            flat.double().masked_fill(~valid, 0.0).clamp_min_(0).sum(),
        )).cpu().tolist()
    complete = int(count) == int(flat.numel()) and count > 0
    _atomic_text(descriptor["path"], _json({
        "token_count": int(flat.numel()),
        "measured": complete,
        "entropy_sum_nats": float(total) if complete else None,
        "reason": None if complete else "nonfinite_or_empty_entropy",
    }) + "\n")


def measure_policy_entropy(function):
    """Instrument successful policy forwards without changing their API result.

    Descriptors travel with examples through existing queues and device moves.
    Only with_grad policy calls participate; reference/feedback scoring cannot
    contaminate the metric. Multi-epoch callers disable this after epoch zero.
    The first completed forward is counted once, even if backward subsequently
    OOMs. Such a forward is still a valid pre-update policy observation.
    Checkpoint recomputation happens below this wrapper and cannot double count.
    """
    @functools.wraps(function)
    def wrapped(model, examples, with_grad, chunk=0, *, pad_token_id=0,
                return_entropy=False, measure_entropy=True):
        examples = list(examples)
        pending = []
        if with_grad and measure_entropy:
            for index, example in enumerate(examples):
                descriptor = example.get("_entropy_measurement")
                if descriptor and not Path(descriptor["path"]).exists():
                    pending.append((index, descriptor))
        if not pending:
            return function(model, examples, with_grad, chunk,
                            pad_token_id=pad_token_id,
                            return_entropy=return_entropy)
        # True preserves a requested entropy regularizer's existing graph.
        # 'measure' returns entropy detached without changing policy logprobs.
        result = function(model, examples, with_grad, chunk,
                          pad_token_id=pad_token_id,
                          return_entropy=True if return_entropy else "measure")
        logprobs, entropies = result
        for index, descriptor in pending:
            try:
                _record_tensor(descriptor, entropies[index])
            except (OSError, ValueError, RuntimeError) as error:
                _warn_once("could not record policy entropy", error)
        return result if return_entropy else logprobs
    return wrapped


def _code_signature(code, size=128):
    """Ignore formatting/comments; bounded uniform sketch of token 5-grams."""
    if not code:
        return None
    try:
        tokens = [item.string for item in tokenize.generate_tokens(
            io.StringIO(code).readline)
                  if item.type not in (tokenize.COMMENT, tokenize.NL,
                                       tokenize.NEWLINE, tokenize.INDENT,
                                       tokenize.DEDENT, tokenize.ENDMARKER)]
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return None
    if not tokens:
        return None
    normalized = "\x1f".join(tokens)
    width = min(5, len(tokens))
    shingles = {
        int.from_bytes(hashlib.blake2b(
            "\x1f".join(tokens[i:i + width]).encode(), digest_size=8
        ).digest(), "big")
        for i in range(len(tokens) - width + 1)
    }
    return {
        "digest": hashlib.sha256(normalized.encode()).hexdigest(),
        "sketch": heapq.nsmallest(size, shingles),
    }


def rollout_observation(descriptor, meta, code):
    """Consume existing verification results; never evaluate a rollout again."""
    return {
        "group": int(meta["group"]), "rollout": int(meta["rollout"]),
        "parent": int(meta.get("parent_group", meta["group"])),
        "strategy": meta.get("strategy_index"),
        "phase": meta.get("strategy_rollout_phase") or "ordinary",
        "valid": bool(meta["valid"]), "reward": float(meta["reward"]),
        "raw_score": meta.get("raw_score"),
        "response_tokens": int(meta["n_response_tokens"]),
        "failure_kind": meta.get("failure_kind"),
        "_measurement": descriptor,
        "_code": _code_signature(code) if meta["valid"] else None,
    }


def _aggregate(rows):
    measured = [row for row in rows if row.get("measured")]
    tokens = sum(row["response_tokens"] for row in rows)
    counted = sum(row["token_count"] for row in measured)
    total = math.fsum(row["entropy_sum_nats"] for row in measured)
    return {
        "rollouts": len(rows), "measured_rollouts": len(measured),
        "response_tokens": tokens, "measured_tokens": counted,
        "token_coverage": counted / tokens if tokens else None,
        "entropy_sum_nats": total,
        "entropy_nats": total / counted if counted else None,
        "valid_fraction": (sum(row["valid"] for row in rows) / len(rows)
                           if rows else None),
        "mean_response_tokens": tokens / len(rows) if rows else None,
    }


def _sketch_distance(a, b, size=128):
    left, right = set(a), set(b)
    sample = heapq.nsmallest(size, left | right)
    return (1.0 - sum(x in left and x in right for x in sample) / len(sample)
            if sample else 0.0)


def code_diversity(rows, max_pairs_per_parent=256):
    """Balanced-pilot, same-parent comparisons with bounded CPU-only work."""
    pilot = [row for row in rows if row["phase"] == "pilot"]
    selected = pilot or rows
    usable = [row for row in selected if row["valid"] and row.get("_code")]
    groups = {}
    for row in usable:
        groups.setdefault(row["group"], []).append(row)
    rng = random.Random(0)  # Does not consume training/generation RNG state.
    buckets = {"within_strategy": [], "across_strategies": []}
    eligible_counts = dict.fromkeys(buckets, 0)
    for group in sorted(groups):
        samples = {key: [] for key in buckets}
        seen = dict.fromkeys(buckets, 0)
        for left, right in itertools.combinations(groups[group], 2):
            if left["strategy"] is None or right["strategy"] is None:
                continue
            key = ("within_strategy" if left["strategy"] == right["strategy"]
                   else "across_strategies")
            seen[key] += 1
            sample = samples[key]
            if len(sample) < max_pairs_per_parent:
                sample.append((left, right))
            else:
                replacement = rng.randrange(seen[key])
                if replacement < max_pairs_per_parent:
                    sample[replacement] = (left, right)
        for key in buckets:
            eligible_counts[key] += seen[key]
            buckets[key].extend(_sketch_distance(
                left["_code"]["sketch"], right["_code"]["sketch"])
                for left, right in samples[key])
    return {
        "method": "python_token_5gram_bottom128_jaccard_distance",
        "population": "valid_pilot_programs" if pilot else "valid_programs",
        "valid_programs_compared": len(usable),
        "unique_program_fraction": (
            len({row["_code"]["digest"] for row in usable}) / len(usable)
            if usable else None),
        **{key: {
            "distance": math.fsum(values) / len(values) if values else None,
            "sampled_pairs": len(values), "eligible_pairs": eligible_counts[key],
        } for key, values in buckets.items()},
    }


def _line_plot(path, rows, series, ylabel, *, upper=None):
    """A small dependency-free SVG: axes, lines, and legend only."""
    width, height = 800, 440
    left, right, top, bottom = 85, 765, 75, 370
    values = [value for _label, _color, get in series
              for row in rows if (value := get(row)) is not None
              and math.isfinite(value)]
    ymax = upper or max([0.1] + [v * 1.12 for v in values])
    xmin = min((row["step"] for row in rows), default=0)
    xmax = max(xmin + 1, max((row["step"] for row in rows), default=1))
    def xy(step, value):
        return (left + (step - xmin) / (xmax - xmin) * (right - left),
                bottom - value / ymax * (bottom - top))
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
           f'height="{height}" viewBox="0 0 {width} {height}">',
           '<rect width="100%" height="100%" fill="white"/>',
           '<g font-family="sans-serif" font-size="15" fill="#333">']
    for i in range(5):
        value = ymax * i / 4
        _, y = xy(xmin, value)
        svg.extend([
            f'<path d="M{left},{y:.1f} H{right}" stroke="#e5e5e5"/>',
            f'<text x="{left-12}" y="{y+5:.1f}" text-anchor="end">'
            f'{value:.2f}</text>'])
    for tick in sorted({round(xmin + (xmax - xmin) * i / 4) for i in range(5)}):
        x, _ = xy(tick, 0)
        svg.append(f'<text x="{x:.1f}" y="{bottom+25}" '
                   f'text-anchor="middle">{tick}</text>')
    svg.extend([
        f'<path d="M{left},{top} V{bottom} H{right}" fill="none" stroke="#333"/>',
        '<text x="425" y="425" text-anchor="middle">Step</text>',
        f'<text transform="translate(23,225) rotate(-90)" text-anchor="middle">'
        f'{html.escape(ylabel)}</text>'])
    for index, (label, color, get) in enumerate(series):
        x = left + index * 325
        svg.extend([
            f'<path d="M{x},30 h25" stroke="{color}" stroke-width="3"/>',
            f'<text x="{x+34}" y="35">{html.escape(label)}</text>'])
        segment = []
        for row in rows + [None]:
            value = get(row) if row is not None else None
            if value is None or not math.isfinite(value):
                if segment:
                    svg.append(f'<polyline points="{" ".join(segment)}" '
                               f'fill="none" stroke="{color}" stroke-width="2.5"/>')
                    segment = []
                continue
            px, py = xy(row["step"], value)
            segment.append(f"{px:.1f},{py:.1f}")
            svg.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="3" fill="{color}"/>')
    if not values:
        svg.append('<text x="425" y="225" text-anchor="middle">'
                   'No measured values yet</text>')
    svg.append('</g></svg>\n')
    _atomic_text(path, "\n".join(svg))


def save_step(exp_dir, step, observations, cfg, stats):
    """Join all worker observations; atomically upsert history and refresh plots."""
    samples = []
    for observation in observations:
        row = dict(observation)
        descriptor = row.pop("_measurement", None)
        try:
            measured = json.loads(Path(descriptor["path"]).read_text())
            if int(measured["token_count"]) != row["response_tokens"]:
                raise ValueError("response token count mismatch")
        except (OSError, ValueError, KeyError, TypeError) as error:
            measured = {"measured": False, "entropy_sum_nats": None,
                        "token_count": 0,
                        "reason": "no_completed_policy_forward"}
        row.update(measured)
        samples.append(row)
    summary = {
        "schema_version": 1, "step": int(step),
        "measurement": "pre_update_training_policy_full_vocabulary_entropy",
        "units": "nats", "logit_temperature": 1.0,
        "includes": "all_response_tokens_including_reasoning_and_failed_outputs",
        "excludes": "prompt_padding_and_unscored_rollouts",
        "coder_model": getattr(cfg, "model_name", None),
        "strategy_model": getattr(cfg, "strategy_model_name", None),
        "strategies_enabled": bool(getattr(cfg, "strategies", False)),
        "training_disabled": bool(getattr(cfg, "no_train", False)),
        "lora_dropout": getattr(cfg, "lora_dropout", None),
        "sampling": {key: getattr(cfg, key, None)
                     for key in ("temperature", "top_p", "sampling_top_k",
                                 "sampling_min_p", "thinking")},
        "all": _aggregate(samples),
        "pilot": _aggregate([r for r in samples if r["phase"] == "pilot"]),
        "code_diversity": code_diversity(samples),
        "step_best_raw_score": stats.get("step_best_raw_score"),
        "best_seen_raw_score": stats.get("best_seen_raw_score"),
        "metric_name": stats.get("result_metric_name"),
        "maximize": stats.get("result_maximize"),
    }
    exp_dir = Path(exp_dir)
    _atomic_text(exp_dir / f"step{step:02d}" / "entropy_samples.jsonl",
                 "".join(_json(row) + "\n" for row in samples))
    history_path = exp_dir / "entropy.jsonl"
    history = {}
    if history_path.exists():
        for line in history_path.read_text().splitlines():
            row = json.loads(line)
            # A resumed unfinished step must not leave later/stale points.
            if int(row["step"]) < int(step):
                history[int(row["step"])] = row
    history[int(step)] = summary
    rows = [history[key] for key in sorted(history)]
    _atomic_text(history_path, "".join(_json(row) + "\n" for row in rows))
    _line_plot(exp_dir / "entropy.svg", rows, [
        ("All measured responses", "#702dbd", lambda r: r["all"]["entropy_nats"]),
        ("Pilot only", "#29a6b6", lambda r: r["pilot"]["entropy_nats"]),
    ], "Coder conditional entropy (nats)")
    _line_plot(exp_dir / "strategy_diversity.svg", rows, [
        ("Within strategy", "#702dbd", lambda r:
         r["code_diversity"]["within_strategy"]["distance"]),
        ("Across strategies", "#29a6b6", lambda r:
         r["code_diversity"]["across_strategies"]["distance"]),
    ], "Valid-code diversity (lexical distance)", upper=1.0)
    return summary
