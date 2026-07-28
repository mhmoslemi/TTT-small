"""
High-level wrapper over a loaded backend.

This is the interface every other component talks to, so the generator, the Elo
judge, the memory maker and the trainer all share one set of weights and one
tokenizer -- which is what Sec. 2.2's "same LLM backbone, dedicated prompts"
requires.

    chat / chat_batch     one reply per prompt      (judge, memory maker)
    sample_group          k samples from one prompt (rollouts)
    token_logprobs        per-token log pi(y | x)   (Eq. 9, 10, 11)
    reference_logprobs    the same with LoRA off    (theta_0 for the KL term)
"""

from contextlib import nullcontext
from typing import List, Optional, Sequence, Tuple

from llm.backend import as_tokenizer, load_backend


class Backbone:
    def __init__(self, cfg, backend=None):
        """cfg is a ModelConfig."""
        self.cfg = cfg
        self.backend = backend or load_backend(cfg)
        self.model = None
        self.tokenizer = None      # always the TEXT tokenizer
        self.processor = None      # the multimodal wrapper, when there is one
        self._loaded = False

    # ------------------------------------------------------------------
    def load(self):
        if not self._loaded:
            self.model, loaded = self.backend.load()
            # A multimodal checkpoint loads a Processor; we only ever do text.
            self.processor = loaded
            self.tokenizer = as_tokenizer(loaded)
            if self.tokenizer is not loaded:
                print(f"[backbone] {type(loaded).__name__} detected; using its "
                      f"{type(self.tokenizer).__name__} for text")
            self._loaded = True
        return self

    @property
    def device(self):
        return next(self.model.parameters()).device

    def set_inference_mode(self):
        self.backend.set_inference_mode()

    def set_training_mode(self):
        self.backend.set_training_mode()

    def disable_adapter(self):
        """Context manager giving pi_theta_0, the policy before adaptation."""
        return self.backend.disable_adapter()

    # ------------------------------------------------------------------
    # Prompt formatting
    # ------------------------------------------------------------------
    def render(self, messages: Sequence[dict]) -> str:
        # Some multimodal checkpoints carry the chat template on the processor
        # rather than on the tokenizer, so try both before giving up.
        for holder in (self.tokenizer, self.processor):
            if holder is not None and getattr(holder, "chat_template", None):
                return holder.apply_chat_template(
                    list(messages), tokenize=False, add_generation_prompt=True)
        parts = [f"{m.get('role', 'user').upper()}: {m.get('content', '')}"
                 for m in messages]
        return "\n\n".join(parts) + "\n\nASSISTANT:"

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------
    def _tick_criteria(self, on_step):
        """
        A StoppingCriteria that never stops -- it exists only to fire a callback
        once per decoding step, which is the only hook into an otherwise opaque
        blocking generate() call.
        """
        import torch
        from transformers import StoppingCriteria

        class _Tick(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                on_step()
                return torch.zeros(input_ids.shape[0], dtype=torch.bool,
                                   device=input_ids.device)

        return _Tick()

    def _encode_left_padded(self, texts: Sequence[List[int]]):
        """Left-pad ragged token-id rows into a batch, with an attention mask."""
        import torch

        pad_id = self.tokenizer.pad_token_id
        width = max(len(row) for row in texts)
        ids, mask = [], []
        for row in texts:
            padding = width - len(row)
            ids.append([pad_id] * padding + list(row))
            mask.append([0] * padding + [1] * len(row))
        device = self.model.device
        return (torch.tensor(ids, device=device),
                torch.tensor(mask, device=device))

    def _generate_budgeted(self, prompt_texts, max_new_tokens, temperature,
                           top_p, think_budget, close_tag, force_text, on_step):
        """
        Budget forcing: think for at most `think_budget` tokens, then close the
        block and spend the rest of the budget answering.

        A reasoning model left uncapped will spend the whole allowance inside
        <think> and emit no answer at all, which scores zero no matter how good
        the reasoning was. Phase 1 lets it reason; phase 2 injects the closing
        tag for whichever sequences are still thinking and lets every unfinished
        sequence continue. Sequences that closed the block on their own are
        continued untouched, so nothing is truncated that would have finished.
        """
        import torch

        tok = self.tokenizer
        self.set_inference_mode()

        prompt_rows = [tok(text=t, add_special_tokens=False,
                           truncation=True, max_length=self.cfg.max_seq_length
                           )["input_ids"] for t in prompt_texts]
        ids, mask = self._encode_left_padded(prompt_rows)
        input_len = ids.shape[1]

        extra = {}
        if on_step is not None:
            from transformers import StoppingCriteriaList
            extra["stopping_criteria"] = StoppingCriteriaList(
                [self._tick_criteria(on_step)])

        common = dict(
            do_sample=temperature is not None and temperature > 0,
            temperature=float(temperature),
            top_p=float(top_p),
            num_return_sequences=1,
            pad_token_id=tok.pad_token_id,
        )

        with torch.no_grad():
            phase1 = self.model.generate(
                input_ids=ids, attention_mask=mask,
                max_new_tokens=int(think_budget), **common, **extra)

        eos_id = tok.eos_token_id
        close_ids = tok(text=force_text, add_special_tokens=False)["input_ids"]

        produced, pending = [], []
        for row in range(phase1.shape[0]):
            raw = [int(t) for t in phase1[row, input_len:]]
            # pad_token is normally set to eos_token, so completion has to be
            # decided from the raw row -- filtering pads would delete the EOS
            # that proves the sequence finished on its own.
            finished = eos_id is not None and eos_id in raw
            # </think> is an added token on some checkpoints; skipping specials
            # would hide it and make every sequence look like it is still
            # thinking, injecting a second closing tag into a closed block.
            closed = close_tag in tok.decode(raw, skip_special_tokens=False)

            new_ids = [t for t in raw if t != tok.pad_token_id]
            injected = [] if (closed or finished) else list(close_ids)
            produced.append(new_ids + injected)
            if not finished:
                pending.append(row)

        remaining = int(max_new_tokens) - int(think_budget)
        if pending and remaining > 0:
            rows = [list(prompt_rows[r]) + produced[r] for r in pending]
            ids2, mask2 = self._encode_left_padded(rows)
            with torch.no_grad():
                phase2 = self.model.generate(
                    input_ids=ids2, attention_mask=mask2,
                    max_new_tokens=remaining, **common, **extra)
            for offset, row in enumerate(pending):
                tail = [int(t) for t in phase2[offset, ids2.shape[1]:]
                        if int(t) != tok.pad_token_id]
                produced[row] = produced[row] + tail

        return [[(tok.decode(ids_, skip_special_tokens=True), ids_)]
                for ids_ in produced]

    def _generate(self, prompt_texts: Sequence[str], num_return_sequences: int,
                  max_new_tokens: int, temperature: float, top_p: float,
                  on_step=None, think_budget: int = 0,
                  think_close_tag: str = "</think>",
                  think_force_text: str = "\n</think>\n\n",
                  ) -> List[List[Tuple[str, List[int]]]]:
        import torch

        # Budget forcing only applies one sample per prompt; with
        # num_return_sequences > 1 the rows do not map back 1:1 to prompts.
        budget = int(think_budget or 0)
        if 0 < budget < int(max_new_tokens) and int(num_return_sequences) == 1:
            return self._generate_budgeted(
                prompt_texts, max_new_tokens, temperature, top_p, budget,
                think_close_tag, think_force_text, on_step)

        tok = self.tokenizer
        self.set_inference_mode()

        # Left padding so every sequence's continuation starts at the same index.
        previous_side = tok.padding_side
        tok.padding_side = "left"
        try:
            # text= as a keyword: a Processor's first positional is `images`.
            enc = tok(text=list(prompt_texts), return_tensors="pt", padding=True,
                      truncation=True, max_length=self.cfg.max_seq_length,
                      add_special_tokens=False).to(self.model.device)
            input_len = enc["input_ids"].shape[1]

            extra = {}
            if on_step is not None:
                from transformers import StoppingCriteriaList
                extra["stopping_criteria"] = StoppingCriteriaList(
                    [self._tick_criteria(on_step)])

            with torch.no_grad():
                out = self.model.generate(
                    **enc,
                    max_new_tokens=int(max_new_tokens),
                    do_sample=temperature is not None and temperature > 0,
                    temperature=float(temperature),
                    top_p=float(top_p),
                    num_return_sequences=int(num_return_sequences),
                    pad_token_id=tok.pad_token_id,
                    **extra,
                )
        finally:
            tok.padding_side = previous_side

        results: List[List[Tuple[str, List[int]]]] = [
            [] for _ in range(len(prompt_texts))]
        for row in range(out.shape[0]):
            prompt_idx = row // int(num_return_sequences)
            new_tokens = out[row, input_len:]
            ids = [int(t) for t in new_tokens if int(t) != tok.pad_token_id]
            text = tok.decode(new_tokens, skip_special_tokens=True)
            results[prompt_idx].append((text, ids))
        return results

    def sample_group(self, messages: Sequence[dict], k: int, max_new_tokens: int,
                     temperature: float, top_p: float, on_step=None,
                     ) -> List[Tuple[str, List[int]]]:
        """k samples from one prompt -- the group g_p of Sec. 2.3."""
        return self._generate([self.render(messages)], k, max_new_tokens,
                              temperature, top_p, on_step=on_step)[0]

    def sample_batch(self, prompt_texts: Sequence[str], max_new_tokens: int,
                     temperature: float, top_p: float, on_step=None,
                     think_budget: int = 0, think_close_tag: str = "</think>",
                     think_force_text: str = "\n</think>\n\n",
                     ) -> List[Tuple[str, List[int]]]:
        """
        One sample per prompt, all in a single generate() call.

        Lets a whole step's rollouts share one call even though targets ask for
        different counts: the caller repeats a prompt `count` times rather than
        using num_return_sequences, which can only apply one count to the batch.
        """
        grouped = self._generate(list(prompt_texts), 1, max_new_tokens,
                                 temperature, top_p, on_step=on_step,
                                 think_budget=think_budget,
                                 think_close_tag=think_close_tag,
                                 think_force_text=think_force_text)
        return [g[0] if g else ("", []) for g in grouped]

    def chat(self, messages: Sequence[dict], max_new_tokens: int = 1024,
             temperature: float = 0.7, top_p: float = 1.0) -> str:
        return self.chat_batch([messages], max_new_tokens, temperature, top_p)[0]

    def chat_batch(self, batch: Sequence[Sequence[dict]], max_new_tokens: int = 1024,
                   temperature: float = 0.7, top_p: float = 1.0,
                   batch_size: int = 4) -> List[str]:
        """One reply per prompt, batched."""
        texts = [self.render(m) for m in batch]
        out: List[str] = []
        for start in range(0, len(texts), max(1, batch_size)):
            chunk = texts[start:start + max(1, batch_size)]
            grouped = self._generate(chunk, 1, max_new_tokens, temperature, top_p)
            out.extend(g[0][0] if g else "" for g in grouped)
        return out

    # ------------------------------------------------------------------
    # Log-probabilities
    # ------------------------------------------------------------------
    def token_logprobs(self, prompt_ids: Sequence[int], response_ids: Sequence[int],
                       with_grad: bool = False, adapter: bool = True):
        """
        log pi(y_l | x, y_<l) for each response token, shape (T,).

        adapter=False evaluates the base policy theta_0 (LoRA disabled), which
        is what the KL term of Eq. 11 regularizes toward.
        """
        import torch

        if not response_ids:
            return torch.zeros(0, device=self.model.device)

        ids = torch.tensor([list(prompt_ids) + list(response_ids)],
                           device=self.model.device)
        n_prompt, n_resp = len(prompt_ids), len(response_ids)

        grad_ctx = nullcontext() if with_grad else torch.no_grad()
        adapter_ctx = nullcontext() if adapter else self.disable_adapter()

        with adapter_ctx, grad_ctx:
            logits = self.model(ids).logits
            # Position i predicts token i+1, so the response is scored by the
            # logits at n_prompt-1 .. end-2.
            sliced = logits[0, n_prompt - 1: n_prompt + n_resp - 1, :]
            logprobs = torch.log_softmax(sliced.float(), dim=-1)
            targets = ids[0, n_prompt:]
            return logprobs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)

    def reference_logprobs(self, prompt_ids: Sequence[int],
                           response_ids: Sequence[int]):
        """log pi_theta_0 -- the policy before any test-time adaptation."""
        return self.token_logprobs(prompt_ids, response_ids,
                                   with_grad=False, adapter=False)
