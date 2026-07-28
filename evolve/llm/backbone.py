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
    def _generate(self, prompt_texts: Sequence[str], num_return_sequences: int,
                  max_new_tokens: int, temperature: float, top_p: float
                  ) -> List[List[Tuple[str, List[int]]]]:
        import torch

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

            with torch.no_grad():
                out = self.model.generate(
                    **enc,
                    max_new_tokens=int(max_new_tokens),
                    do_sample=temperature is not None and temperature > 0,
                    temperature=float(temperature),
                    top_p=float(top_p),
                    num_return_sequences=int(num_return_sequences),
                    pad_token_id=tok.pad_token_id,
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
                     temperature: float, top_p: float,
                     ) -> List[Tuple[str, List[int]]]:
        """k samples from one prompt -- the group g_p of Sec. 2.3."""
        return self._generate([self.render(messages)], k, max_new_tokens,
                              temperature, top_p)[0]

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
