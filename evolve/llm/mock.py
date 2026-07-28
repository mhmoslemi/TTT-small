"""
A scripted stand-in for the backbone.

Lets the full engine loop -- selection, verification, memory, logging -- run end
to end on a laptop with no torch, no GPU and no model download, which is how the
integration tests exercise Algorithm 1. Set model.backend to "mock" to use it.

It is a test fixture, not a fallback: it produces canned programs, so any run
using it is measuring the plumbing, not the method.
"""

import random
import re
from typing import List, Optional, Sequence, Tuple


class MockBackbone:
    """Implements the subset of Backbone the engine actually calls."""

    def __init__(self, cfg=None, responses: Optional[Sequence[str]] = None,
                 seed: int = 0, judge_bias: float = 0.5):
        self.cfg = cfg
        self.responses = list(responses) if responses else None
        self.rng = random.Random(seed)
        self.judge_bias = judge_bias
        self.model = None
        self.tokenizer = None
        self.calls = {"sample_group": 0, "chat_batch": 0, "logprobs": 0}

    # ------------------------------------------------------------------
    def load(self):
        return self

    def set_inference_mode(self):
        pass

    def set_training_mode(self):
        pass

    def render(self, messages: Sequence[dict]) -> str:
        return "\n".join(m.get("content", "") for m in messages)

    # ------------------------------------------------------------------
    def _canned_program(self) -> str:
        """
        A VALID grid packing, with the fill fraction jittered per rollout so the
        batch has a spread of rewards -- Eq. 8 produces all-zero advantages when
        every response scores the same, and a fixture that never varies would
        make the RL path untestable.

        Occasionally emits a deliberately broken program so the failure path,
        the feedback signal and the negative-lesson extractor are exercised too.
        """
        if self.rng.random() < 0.25:
            return ("```python\n"
                    "def run_packing():\n"
                    "    raise ValueError('mock deliberate failure')\n"
                    "```\n")

        fill = self.rng.uniform(0.5, 0.99)
        return (
            "<strategy>Grid packing.</strategy>\n\n"
            "```python\n"
            "import math\n"
            "import numpy as np\n"
            "def run_packing():\n"
            "    n = N_CIRCLES\n"
            "    g = int(math.ceil(math.sqrt(n)))\n"
            "    cell = 1.0 / g\n"
            f"    r = {fill:.4f} * cell / 2.0\n"
            "    centers = np.zeros((n, 2))\n"
            "    radii = np.full(n, r)\n"
            "    for idx in range(n):\n"
            "        i, j = idx % g, idx // g\n"
            "        centers[idx] = ((i + 0.5) * cell, (j + 0.5) * cell)\n"
            "    return centers, radii, float(radii.sum())\n"
            "```\n"
        )

    def sample_group(self, messages, k: int, max_new_tokens: int = 0,
                     temperature: float = 1.0, top_p: float = 1.0, on_step=None
                     ) -> List[Tuple[str, List[int]]]:
        self.calls["sample_group"] += 1
        if on_step is not None:
            # Pretend to decode, so the progress path is exercised in tests.
            for _ in range(min(int(max_new_tokens or 0), 8)):
                on_step()
        prompt = self.render(messages)
        match = re.search(r"pack (\d+) circles", prompt)
        n = match.group(1) if match else "1"

        out = []
        for _ in range(k):
            if self.responses:
                text = self.responses[self.rng.randrange(len(self.responses))]
            else:
                text = self._canned_program().replace("N_CIRCLES", n)
            out.append((text, [self.rng.randrange(1000) for _ in range(12)]))
        return out

    def chat_batch(self, batch, max_new_tokens: int = 1024,
                   temperature: float = 0.7, top_p: float = 1.0,
                   batch_size: int = 4) -> List[str]:
        self.calls["chat_batch"] += 1
        replies = []
        for messages in batch:
            text = self.render(messages)
            if "VERDICT" in text or "which candidate" in text.lower():
                pick = self.rng.choice(["A", "B", "TIE"])
                replies.append(f"Both are plausible.\nVERDICT: {pick}")
            else:
                kind = "failure" if "FAILED" in text else "success"
                replies.append(
                    '[{"title": "mock lesson", "summary": "a mock rule", '
                    f'"lesson": "a mock {kind} lesson body for testing."}}]')
        return replies

    def chat(self, messages, **kwargs) -> str:
        return self.chat_batch([messages], **kwargs)[0]

    # ------------------------------------------------------------------
    def token_logprobs(self, prompt_ids, response_ids, with_grad=False,
                       adapter=True):
        self.calls["logprobs"] += 1
        import numpy as np
        return np.array([-self.rng.uniform(0.1, 3.0) for _ in response_ids])

    def reference_logprobs(self, prompt_ids, response_ids):
        return self.token_logprobs(prompt_ids, response_ids)

    def disable_adapter(self):
        from contextlib import nullcontext
        return nullcontext()
