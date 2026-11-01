"""Bound GPT-OSS startup scratch without changing Marlin's packed weights.

vLLM's helper allocates a second tensor containing every expert. During
sleep-mode loading those large temporary allocations also remain in the
private allocator pool until loading finishes. Reuse the input's storage and
call the installed vLLM helper for one expert at a time instead. The resulting
weights remain in vLLM's sleep allocator and use the original Marlin kernel.
"""

from contextlib import contextmanager
from functools import wraps
import inspect


def repack_experts_in_place(weight, *args, repack, **kwargs):
    import torch

    if (weight.ndim != 3 or weight.shape[0] < 1
            or not weight.is_contiguous() or weight.requires_grad):
        raise RuntimeError("GPT-OSS bounded repacking requires contiguous frozen experts")
    out = None
    with torch.no_grad():
        for expert in range(weight.shape[0]):
            # Keep vLLM's actual packing kernel, permutation and precision.
            converted = repack(weight[expert:expert + 1], *args, **kwargs)
            if out is None:
                if (converted.shape[0] != 1 or not converted.is_contiguous()
                        or converted.device != weight.device
                        or converted.numel() * converted.element_size()
                        != weight[0].numel() * weight.element_size()):
                    raise RuntimeError(
                        "vLLM changed the Marlin packing storage contract; "
                        "cannot safely repack GPT-OSS in place")
                out = weight.view(converted.dtype).view(
                    weight.shape[0], *converted.shape[1:])
            # CUDA operations remain ordered on the same current stream.
            # Only this expert is overwritten, after its input was consumed.
            out[expert:expert + 1].copy_(converted)
            del converted
    return out


@contextmanager
def bounded_marlin_repacking():
    from vllm.model_executor.layers.quantization.utils import marlin_utils_fp4

    original = getattr(marlin_utils_fp4, "_repack_marlin_experts", None)
    if not callable(original):
        raise RuntimeError(
            "GPT-OSS bounded startup requires vLLM's _repack_marlin_experts "
            "helper")
    signature = inspect.signature(original)
    arguments = dict(weight=None, size_n=1, size_k=1, is_a_8bit=False)
    # vLLM 0.30 removed perm; forward the installed helper's arguments intact.
    if "perm" in signature.parameters:
        arguments["perm"] = None
    try:
        signature.bind(**arguments)
    except TypeError as error:
        raise RuntimeError(
            f"Unsupported vLLM _repack_marlin_experts signature: {signature}") from error

    @wraps(original)
    def bounded(weight, *args, **kwargs):
        return repack_experts_in_place(weight, *args, repack=original, **kwargs)

    marlin_utils_fp4._repack_marlin_experts = bounded
    try:
        yield
    finally:
        marlin_utils_fp4._repack_marlin_experts = original
