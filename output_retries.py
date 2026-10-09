"""Output-contract checks and immutable retry prompts (no model dependencies)."""

from copy import deepcopy
import re


def final_answer_scope(response, *, require_final_marker=False):
    text = str(response or "")
    lowered = text.lower()
    end = -1
    for marker in ("<|channel|>final<|message|>", "assistantfinal",
                   "</think>", "</analysis>", "</reasoning>"):
        position = lowered.rfind(marker)
        if position >= 0:
            end = max(end, position + len(marker))
    if end >= 0:
        text = text[end:]
    elif require_final_marker:
        return ""
    if re.search(r"<(?:think|analysis|reasoning)\b", text, re.IGNORECASE):
        return ""
    return text


def coder_output_issue(response, *, require_final_marker=False):
    """Check formatting only, never program quality, syntax, or reward."""
    text = final_answer_scope(response, require_final_marker=require_final_marker)
    openings = list(re.finditer(r"```python\b[^\S\n]*\n?", text, re.IGNORECASE))
    if not openings:
        return "missing final Python block"
    start = openings[-1].end()
    end = text.find("```", start)
    if end < 0:
        return "unclosed final Python block"
    if not text[start:end].strip():
        return "empty final Python block"
    return None


def output_retry_messages(messages, kind):
    """Add one reminder to the frozen original; never include failed reasoning."""
    if kind == "strategy":
        contract = (
            "one complete <strategy>...</strategy> block containing the full "
            "detailed implementation plan requested above. Preserve the "
            "mathematical specification, concrete algorithm, budget, and "
            "validation needed by the coder; do not reduce it to a summary. "
            "Only that block is handed to the next agent")
    elif kind == "coder":
        contract = (
            "one complete fenced Python block starting with ```python and "
            "ending with ```, containing the entire requested program. "
            "Python snippets inside your reasoning are not the final answer")
    else:
        raise ValueError(f"unknown output contract: {kind}")
    reminder = (
        "Your previous attempt did not provide the required complete final "
        "output. Retry the same task with all the requirements above unchanged. "
        "You may reason as needed, but you MUST finish with " + contract + ". "
        "Close the required block before the token limit and put nothing after it.")
    result = deepcopy(messages)
    if result and result[-1].get("role") == "user":
        result[-1]["content"] = str(result[-1].get("content", "")) + "\n\n" + reminder
    else:
        result.append({"role": "user", "content": reminder})
    return result


def retry_metadata(record):
    return {key: record.get(key) for key in (
        "retry_attempt", "retry_of_group", "retry_of_rollout",
        "output_format_issue", "counts_toward_allocation")}


def coder_retry_gate(records):
    """Return whether incomplete coder outputs need format recovery.

    Recovery is intentionally a batch-level safety net, not a requirement that
    every sample end in a complete block.  A batch triggers only when fewer
    than half of its original rollouts contain complete final code blocks.
    """
    originals = [record for record in records
                 if not record.get("retry_attempt", 0)]
    total = len(originals)
    missing = sum(
        record.get("output_format_issue") is not None
        for record in originals)
    valid = total - missing
    triggered = bool(total and missing and valid * 2 < total)
    # Once recovery is necessary, aim ten percentage points above the trigger
    # instead of trying to repair every malformed rollout.  ceil(0.60 * N) is
    # computed exactly with integers.
    target_valid = (3 * total + 4) // 5 if total else 0
    retry_budget = (
        min(missing, max(0, target_valid - valid)) if triggered else 0)
    return triggered, total, valid, missing, retry_budget


def coder_retry_prompt_job(prompt_jobs, record, render, cache):
    """Separate prompt ID so generation, reference scoring and training agree."""
    source_idx = int(record["job_idx"])
    phase = record.get("strategy_rollout_phase")
    key = (source_idx, phase)
    if key not in cache:
        source = prompt_jobs[source_idx]
        messages = output_retry_messages(source["messages"], "coder")
        cache[key] = len(prompt_jobs)
        prompt_jobs.append({
            **source, "messages": messages,
            "prompt_text": render(messages, rollout_phase=phase), "count": 0,
        })
    return cache[key]


def strategy_retry_needed(issue, attempt, max_retries, *, label):
    if issue is None:
        return False
    if int(attempt) >= int(max_retries):
        raise RuntimeError(
            f"{label}: {issue} after {int(max_retries)} format retries "
            f"({int(max_retries) + 1} attempts). All attempts were saved; "
            "stopping instead of passing a fallback strategy downstream.")
    return True
