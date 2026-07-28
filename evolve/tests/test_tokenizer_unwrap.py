"""
Regression for the Qwen3.5-4B crash: a multimodal checkpoint loads a Processor,
whose FIRST POSITIONAL argument is `images`, not `text`. Passing the prompt
positionally routed it into the image pipeline and died in base64 decoding.
"""
import pytest

from llm.backend import as_tokenizer, _ensure_pad_token


class FakeTokenizer:
    """Stands in for the text tokenizer inside a processor."""
    def __init__(self):
        self.pad_token_id = None
        self.pad_token = None
        self.eos_token = "<eos>"
        self.padding_side = "right"
        self.chat_template = "{{ messages }}"
        self.seen = None

    def convert_tokens_to_ids(self, t):
        return 0

    def __call__(self, text=None, **kwargs):
        self.seen = text
        return {"input_ids": [[1, 2, 3]]}

    def apply_chat_template(self, messages, **kwargs):
        return "RENDERED"


class FakeProcessor:
    """Mimics Qwen3VLProcessor: images first, and it rejects non-image input."""
    def __init__(self):
        self.tokenizer = FakeTokenizer()
        self.chat_template = None

    def __call__(self, images=None, text=None, videos=None, **kwargs):
        if images is not None:
            raise ValueError(
                "Incorrect image source. Must be a valid URL ... Got "
                f"{str(images)[:40]}. Failed with Incorrect padding")
        return self.tokenizer(text=text, **kwargs)


def test_processor_unwraps_to_its_text_tokenizer():
    processor = FakeProcessor()
    assert as_tokenizer(processor) is processor.tokenizer

def test_a_plain_tokenizer_passes_through_untouched():
    tok = FakeTokenizer()
    assert as_tokenizer(tok) is tok

def test_pad_token_is_set_on_the_tokenizer_inside_a_processor():
    processor = FakeProcessor()
    _ensure_pad_token(processor)
    assert processor.tokenizer.pad_token == "<eos>"

def test_the_original_crash_positional_prompt_hits_the_image_pipeline():
    """Guards the failure mode itself, so a reintroduced regression is caught."""
    with pytest.raises(ValueError, match="Incorrect image source"):
        FakeProcessor()("a long circle packing prompt")

def test_calling_with_text_keyword_is_safe_on_both():
    prompt = "pack 26 circles"
    assert FakeProcessor()(text=[prompt])["input_ids"] == [[1, 2, 3]]
    assert FakeTokenizer()(text=[prompt])["input_ids"] == [[1, 2, 3]]

def test_backbone_uses_the_text_tokenizer_and_calls_it_by_keyword():
    from config import Config
    from llm.backbone import Backbone

    class StubBackend:
        name = "stub"
        def load(self):
            return object(), FakeProcessor()

    backbone = Backbone(Config().model, backend=StubBackend()).load()
    assert isinstance(backbone.tokenizer, FakeTokenizer)
    assert isinstance(backbone.processor, FakeProcessor)
    # The prompt must reach the tokenizer as text, never as images.
    backbone.tokenizer(text=["x"])
    assert backbone.tokenizer.seen == ["x"]

def test_render_falls_back_to_the_processor_chat_template():
    from config import Config
    from llm.backbone import Backbone

    class NoTemplateTokenizer(FakeTokenizer):
        def __init__(self):
            super().__init__()
            self.chat_template = None      # instance attr, or __init__ wins

    class ProcessorWithTemplate(FakeProcessor):
        def __init__(self):
            super().__init__()
            self.tokenizer = NoTemplateTokenizer()
            self.chat_template = "{{ messages }}"
        def apply_chat_template(self, messages, **kwargs):
            return "FROM_PROCESSOR"

    class StubBackend:
        name = "stub"
        def load(self):
            return object(), ProcessorWithTemplate()

    backbone = Backbone(Config().model, backend=StubBackend()).load()
    assert backbone.render([{"role": "user", "content": "hi"}]) == "FROM_PROCESSOR"

def test_render_prefers_the_tokenizer_template_when_it_has_one():
    from config import Config
    from llm.backbone import Backbone

    class StubBackend:
        name = "stub"
        def load(self):
            return object(), FakeProcessor()

    backbone = Backbone(Config().model, backend=StubBackend()).load()
    assert backbone.render([{"role": "user", "content": "hi"}]) == "RENDERED"
