import pickle
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from feedback import build_reprompt
from model_io import VisionPrompt, make_prompt, prompt_images, prompt_text


class FakeProcessor:
    def apply_chat_template(self, messages, tokenize=False,
                            add_generation_prompt=True, **kwargs):
        assert not tokenize
        blocks = []
        for message in messages:
            content = message["content"]
            if isinstance(content, str):
                blocks.append(content)
                continue
            for item in content:
                blocks.append("<image>" if item["type"] == "image"
                              else item["text"])
        return "|".join(blocks) + "|<assistant>"


class ModelIOTests(unittest.TestCase):
    def test_llm_prompt_remains_a_string(self):
        prompt = make_prompt(
            FakeProcessor(), [{"role": "user", "content": "solve"}])
        self.assertIsInstance(prompt, str)
        self.assertEqual(prompt, "solve|<assistant>")

    def test_vlm_prompt_is_picklable_and_carries_local_image(self):
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / "observation.png"
            image.write_bytes(b"not decoded while the prompt is built")
            prompt = make_prompt(
                FakeProcessor(), [{"role": "user", "content": "solve"}],
                model_kind="vlm", image_paths=[str(image)])

            self.assertIsInstance(prompt, VisionPrompt)
            self.assertEqual(prompt_images(prompt), (str(image.resolve()),))
            self.assertIn("<image>", prompt_text(prompt))
            self.assertEqual(pickle.loads(pickle.dumps(prompt)), prompt)

    def test_images_are_rejected_on_the_llm_path(self):
        with self.assertRaises(ValueError):
            make_prompt(
                FakeProcessor(), [{"role": "user", "content": "solve"}],
                image_paths=["unused.png"])

    def test_feedback_append_preserves_multimodal_blocks(self):
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "url": "/tmp/example.png"},
                {"type": "text", "text": "solve"},
            ],
        }]
        updated = build_reprompt(messages, "shape mismatch")
        self.assertEqual(updated[0]["content"][0]["type"], "image")
        self.assertIn("shape mismatch", updated[0]["content"][-1]["text"])
        self.assertEqual(len(messages[0]["content"]), 2)

    def test_vlm_logprob_forward_extends_text_aligned_masks(self):
        try:
            import torch
            from train_multy import compute_token_logprobs
        except ImportError:
            self.skipTest("PyTorch/NumPy are not installed in this interpreter")

        class FakeModel:
            def __call__(self, *args, **kwargs):
                self.args = args
                self.kwargs = kwargs
                length = kwargs["input_ids"].shape[1]
                return SimpleNamespace(logits=torch.zeros(1, length, 16))

        model = FakeModel()
        pixel_values = torch.ones(2, 3)
        prompt = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.ones(1, 3, dtype=torch.long),
            "cross_attention_mask": torch.tensor(
                [[[[1]], [[0]], [[1]]]], dtype=torch.long),
            "pixel_values": pixel_values,
        }
        response = torch.tensor([[4, 5]])

        logprobs = compute_token_logprobs(
            model, prompt, response, with_grad=False)

        self.assertEqual(tuple(logprobs.shape), (2,))
        self.assertEqual(tuple(model.kwargs["input_ids"].shape), (1, 5))
        self.assertEqual(tuple(model.kwargs["attention_mask"].shape), (1, 5))
        self.assertEqual(
            tuple(model.kwargs["cross_attention_mask"].shape), (1, 5, 1, 1))
        self.assertIs(model.kwargs["pixel_values"], pixel_values)

    def test_llm_logprob_forward_keeps_the_original_positional_call(self):
        try:
            import torch
            from train_multy import compute_token_logprobs
        except ImportError:
            self.skipTest("PyTorch/NumPy are not installed in this interpreter")

        class FakeModel:
            def __call__(self, *args, **kwargs):
                self.args = args
                self.kwargs = kwargs
                length = args[0].shape[1]
                return SimpleNamespace(logits=torch.zeros(1, length, 16))

        model = FakeModel()
        compute_token_logprobs(
            model, torch.tensor([[1, 2]]), torch.tensor([[3]]),
            with_grad=False)

        self.assertEqual(len(model.args), 1)
        self.assertEqual(model.kwargs, {})


if __name__ == "__main__":
    unittest.main()
