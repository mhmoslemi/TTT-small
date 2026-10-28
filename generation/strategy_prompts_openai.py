"""OpenAI strategy message assembly and final-plan normalization.

Edit the main local/API instructions in strategy_prompts.py.

OpenAI requests use Responses Structured Outputs, not tool/function calling:
https://developers.openai.com/api/docs/guides/structured-outputs
https://developers.openai.com/api/docs/guides/latest-model
"""

from copy import deepcopy
import json
import re

from generation.output_retries import final_answer_scope
from generation.strategy_prompts import API_STRATEGY_PROMPT


STRATEGY_PLAN_FORMAT = {
    "type": "json_schema",
    "name": "strategy_plan",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "plan": {
                "type": "string",
                "description": (
                    "The complete, detailed, self-contained scientific plan "
                    "with actionable implementation guidance and all six "
                    "numbered sections."
                ),
            },
        },
        "required": ["plan"],
        "additionalProperties": False,
    },
}


def build_openai_strategy_messages(messages, previous_strategies=None):
    """Keep problem-specific context and only this parent's accepted history."""
    context = []
    for message in messages:
        content = message.get("content", "")
        if not isinstance(content, str):
            raise TypeError("OpenAI strategy context must contain text only")
        context.append({"role": str(message.get("role", "user")),
                        "content": content})
    history = list(previous_strategies or [])
    if not all(isinstance(plan, str) for plan in history):
        raise TypeError("previous strategies must be extracted plan strings")
    history_text = ""
    if history:
        rendered = []
        for index, strategy in enumerate(history, start=1):
            rendered.append(
                f"<previous_strategy_{index}>\n{strategy.strip()}\n"
                f"</previous_strategy_{index}>")
        history_text = (
            "The following strategies were already proposed for this same "
            "task and parent context:\n\n"
            + "\n\n".join(rendered)
            + "\n\nPropose a materially different approach. Do not merely "
              "rename variables, reorder steps, or make superficial "
              "parameter changes. Treat these plans as unverified proposals, "
              "not established facts. Explain the substantive difference "
              "in your final plan, and make that plan independently usable "
              "without referring back to earlier strategies.\n\n"
        )
    instruction = API_STRATEGY_PROMPT.replace(
        "{previous_strategies}", history_text, 1)
    return [
        {"role": "developer", "content": instruction},
        {"role": "user", "content": json.dumps({
            "task_messages": context,
            "previous_strategies": history,
        }, ensure_ascii=False)},
    ]


def openai_strategy_retry_messages(messages):
    result = deepcopy(messages)
    result.append({"role": "user", "content": (
        "The previous attempt did not deliver a complete valid plan. Retry "
        "the same task and return exactly the schema's JSON object with one "
        "plan string containing all six detailed sections. Finish the JSON; "
        "do not use XML tags or code fences, and do not shorten the plan into "
        "a summary. The task and previous accepted strategies are unchanged."
    )})
    return result


def normalize_strategy_plan(response):
    """Adapt explicit JSON/plan fences; leave the existing XML path intact.

    Also accepts these final-answer formats from compatible open-weight
    models. Arbitrary prose, Python, incomplete JSON, or reasoning alone is
    never promoted to a strategy. Existing validation/retries remain in charge.
    """
    original = str(response or "")
    text = final_answer_scope(original).strip()
    fence = re.fullmatch(r"```(json|plan|strategy)\s*\n(.*?)\n```",
                         text, flags=re.DOTALL | re.IGNORECASE)
    plan = None
    if fence:
        text = fence.group(2).strip()
        if fence.group(1).lower() in {"plan", "strategy"}:
            plan = text
    if plan is None:
        try:
            data = json.loads(text)
        except (ValueError, TypeError):
            return "" if fence or text.startswith(("{", "[")) else original
        if not isinstance(data, dict) or set(data) != {"plan"}:
            return ""
        plan = data["plan"]
    if not isinstance(plan, str) or not plan.strip():
        return ""
    # Reject embedded boundaries so an injected or nested block cannot
    # silently replace the complete plan during the existing extraction step.
    if re.search(r"</?(?:strategy|think|analysis|reasoning)\b|assistantfinal|"
                 r"<\|channel\|>", plan, re.IGNORECASE):
        return ""
    return f"<strategy>\n{plan.strip()}\n</strategy>"
