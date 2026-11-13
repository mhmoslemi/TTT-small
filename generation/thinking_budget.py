"""Qwen coder reasoning budgets and sparse masks for externally forced tokens."""

from copy import deepcopy


def validate_thinking_budget(config):
    budget = config.setdefault("coder_thinking_budget", 0)
    if isinstance(budget, bool) or not isinstance(budget, int) or budget < 0:
        raise ValueError("coder_thinking_budget must be an integer >= 0 (0 disables it)")
    if not budget:
        return
    if config.get("coder_template_kind") != "qwen3.8":
        raise ValueError("coder_thinking_budget currently supports only Qwen3.8-27B")
    if config.get("generation_backend") != "vllm":
        raise ValueError("coder_thinking_budget requires the vLLM generation backend")
    if budget + 1 >= min(int(config["max_seq_length"]), int(config["max_new_tokens"])):
        raise ValueError("coder_thinking_budget must leave output space for the final answer")
    # The native template supplies xhigh when effort is omitted in budget mode.
    config["coder_reasoning_effort"] = "xhigh"
    # A shared frozen-model pool must not inherit the coder's limit.
    if config.get("strategies") and config.get("strategy_backend", "local") == "local":
        config["separate_strategy_inference_pool"] = True


def budget_messages(messages, budget):
    if not budget:
        return messages
    instruction = (f"Your thinking budget is at most {budget} tokens; finish reasoning "
                   "within that budget, then provide the complete final answer.")
    result = deepcopy(messages)
    for message in reversed(result):
        if message.get("role") == "user":
            if instruction not in str(message.get("content", "")):
                message["content"] = str(message.get("content", "")) + "\n" + instruction
            return result
    raise ValueError("a budgeted coder prompt must contain a user message")


def thinking_token_ids(tokenizer):
    ids = [tokenizer.encode(marker, add_special_tokens=False)
           for marker in ("<think>", "</think>")]
    if any(len(value) != 1 for value in ids) or ids[0] == ids[1]:
        raise ValueError("budget mode requires Qwen's single-token native thinking boundaries")
    return ids[0][0], ids[1][0]


def check_answer_space(prompt_ids, max_tokens, budget, tokenizer):
    """Validate each real prompt; never reduce the total answer/output limit."""
    if not budget:
        return
    start, end = thinking_token_ids(tokenizer)
    last_start = last_end = -1
    for index, token in enumerate(prompt_ids):
        if token == start:
            last_start = index
        elif token == end:
            last_end = index
    if last_start <= last_end:
        raise ValueError("budgeted coder prompt must end in an open native thinking block")
    already_thinking = len(prompt_ids) - last_start - 1
    if max_tokens <= max(0, budget - already_thinking) + 1:
        raise ValueError("coder thinking budget leaves no final-answer space after this prompt")


def thinking_budget_metadata(tokenizer, prompt_ids, response_ids, budget):
    """Replay native single-token boundary enforcement in one linear pass.

    Mirrors vLLM 0.30 V2's count from the latest unmatched <think> token.
    A closing token at the enforced position is external even if the model
    would independently have picked that same token.
    """
    if not budget:
        return {}
    start, end = thinking_token_ids(tokenizer)
    last_start = last_end = -1
    for index, token in enumerate(prompt_ids):
        if token == start:
            last_start = index
        elif token == end:
            last_end = index
    offset = len(prompt_ids)
    forced, reasoning_tokens = [], 0
    for index, token in enumerate(response_ids):
        position = offset + index
        in_think = last_start > last_end
        if in_think and position - last_start - 1 >= budget:
            if token != end:
                raise RuntimeError("vLLM did not enforce the configured coder thinking budget")
            forced.append(index)
        elif in_think and token != end:
            reasoning_tokens += 1
        if token == start:
            last_start = position
        elif token == end:
            last_end = position
    return {"coder_thinking_budget": budget,
            "n_reasoning_tokens": reasoning_tokens,
            "forced_token_indices": forced}


def policy_token_values(current, *others, forced_token_indices=()):
    """Keep full trajectories as context; remove external actions from losses."""
    if not forced_token_indices:
        return (current, *others)
    import torch

    count = current.numel()
    indices = list(forced_token_indices)
    if (any(type(index) is not int or not 0 <= index < count for index in indices)
            or indices != sorted(set(indices)) or len(indices) >= count):
        raise ValueError("invalid forced-token indices for policy loss")
    keep = torch.ones(count, dtype=torch.bool, device=current.device)
    keep[indices] = False
    return tuple(None if value is None else value.to(current.device)[keep]
                 for value in (current, *others))


def saved_budget_metadata(meta, response_ids, *, exact=True):
    """Old runs have no external actions; budgeted runs require exact replay."""
    if not meta.get("coder_thinking_budget"):
        return {}
    if not exact or "forced_token_indices" not in meta:
        raise RuntimeError("budgeted rollout recovery requires exact token IDs and forced-token metadata")
    indices = meta["forced_token_indices"]
    if (not isinstance(indices, list)
            or any(type(index) is not int or not 0 <= index < len(response_ids) for index in indices)
            or indices != sorted(set(indices))):
        raise ValueError("invalid saved forced-token indices")
    return {key: meta[key] for key in (
        "coder_thinking_budget", "n_reasoning_tokens", "forced_token_indices")}
