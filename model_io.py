"""Shared text/VLM prompt encoding helpers.

The text path intentionally remains a rendered string.  VLM prompts carry the
rendered text, normalized multimodal messages, and local image paths together so
the same input can be reconstructed in the training process and in spawned
generation workers.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple, Union


@dataclass
class VisionPrompt:
    """Pickle-safe prompt passed between the trainer and generation workers."""

    messages: List[Dict[str, Any]]
    text: str
    image_paths: Tuple[str, ...]


PromptInput = Union[str, VisionPrompt]


def is_vlm(model_kind: str) -> bool:
    return str(model_kind or "llm").strip().lower() == "vlm"


def text_tokenizer(processor):
    """Return the decoder tokenizer from either a tokenizer or processor."""

    return getattr(processor, "tokenizer", processor)


def ensure_pad_token(processor):
    tokenizer = text_tokenizer(processor)
    if getattr(tokenizer, "pad_token_id", None) is None:
        tokenizer.pad_token = tokenizer.eos_token
    return processor


def decoder_token_ids(processor) -> Tuple[Any, Any]:
    tokenizer = text_tokenizer(processor)
    eos_id = getattr(tokenizer, "eos_token_id", None)
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        pad_id = eos_id
    return eos_id, pad_id


def decode_tokens(processor, token_ids: Sequence[int]) -> str:
    return text_tokenizer(processor).decode(token_ids, skip_special_tokens=True)


def _apply_chat_template(processor, messages, *, tokenize: bool,
                         return_tensors: str | None = None):
    kwargs = {
        "tokenize": tokenize,
        "add_generation_prompt": True,
    }
    if tokenize:
        kwargs.update(return_dict=True, return_tensors=return_tensors or "pt")
    try:
        return processor.apply_chat_template(
            messages, enable_thinking=False, **kwargs)
    except TypeError:
        return processor.apply_chat_template(messages, **kwargs)


def render_chat(processor, messages: List[Dict[str, Any]]) -> str:
    return _apply_chat_template(processor, messages, tokenize=False)


def _canonical_image_path(value: Any) -> str:
    if isinstance(value, Path):
        value = str(value)
    if not isinstance(value, str) or not value.strip():
        raise TypeError(
            "VLM images must be local file paths so generation workers can "
            "reopen them"
        )
    value = value.strip()
    if value.startswith(("http://", "https://")):
        raise ValueError(
            "remote VLM images are not supported; download the image and pass "
            "a local path"
        )
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"VLM image not found: {path}")
    return str(path)


def _normalize_multimodal_messages(messages: List[Dict[str, Any]],
                                    image_paths: Sequence[str]):
    """Convert string content to HF multimodal blocks and prepend images."""

    out = deepcopy(messages)
    for message in out:
        content = message.get("content", "")
        if isinstance(content, str):
            message["content"] = [{"type": "text", "text": content}]
        elif isinstance(content, list):
            message["content"] = deepcopy(content)
        else:
            raise TypeError("chat message content must be a string or block list")

    if image_paths:
        image_blocks = [
            {"type": "image", "url": path} for path in image_paths
        ]
        for message in out:
            if message.get("role") == "user":
                message["content"] = image_blocks + message["content"]
                break
        else:
            out.append({"role": "user", "content": image_blocks})
    return out


def make_prompt(processor, messages: List[Dict[str, Any]], *,
                model_kind: str = "llm",
                image_paths: Iterable[str] = ()) -> PromptInput:
    """Build a rendered LLM prompt or a reconstructable VLM prompt."""

    if not is_vlm(model_kind):
        paths = list(image_paths or ())
        if paths:
            raise ValueError("vision images require model_kind='vlm'")
        return render_chat(processor, messages)

    paths = tuple(_canonical_image_path(p) for p in (image_paths or ()))
    normalized = _normalize_multimodal_messages(messages, paths)
    text = render_chat(processor, normalized)
    return VisionPrompt(messages=normalized, text=text, image_paths=paths)


def prompt_text(prompt: PromptInput) -> str:
    return prompt.text if isinstance(prompt, VisionPrompt) else str(prompt)


def prompt_images(prompt: PromptInput) -> Tuple[str, ...]:
    return prompt.image_paths if isinstance(prompt, VisionPrompt) else ()


def _load_images(paths: Sequence[str]):
    if not paths:
        return []
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError("VLM image input requires Pillow") from exc
    images = []
    for path in paths:
        with Image.open(path) as image:
            images.append(image.convert("RGB").copy())
    return images


def _move_to_device(inputs, device):
    if device is None:
        return inputs
    if hasattr(inputs, "to"):
        return inputs.to(device)
    return {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in inputs.items()
    }


def encode_prompt(processor, prompt: PromptInput, device=None):
    """Tokenize one prompt and include VLM pixel tensors when applicable."""

    if not isinstance(prompt, VisionPrompt):
        return _move_to_device(
            processor(prompt, return_tensors="pt"), device)

    # Current Transformers processors can load the media references directly
    # from the structured chat.  The fallback supports older Qwen/Gemma
    # processors that only tokenize a rendered string plus PIL images.
    try:
        inputs = _apply_chat_template(
            processor, prompt.messages, tokenize=True, return_tensors="pt")
    except (TypeError, ValueError, KeyError):
        images = _load_images(prompt.image_paths)
        kwargs = {
            "text": [prompt.text],
            "return_tensors": "pt",
            "padding": False,
        }
        if images:
            kwargs["images"] = images
        inputs = processor(**kwargs)
    return _move_to_device(inputs, device)


def model_device(model):
    try:
        return model.device
    except AttributeError:
        return next(model.parameters()).device

