"""Bind training-capable Qwen DeltaNet kernels across Transformers interfaces.

Older releases call kernel attributes; newer ones call module-level functions.
Keep the installed layer forward (including placement hooks) in both cases.
Only opted-in trainer instances use this routing; other models stay untouched.
"""

from functools import wraps
import importlib
import inspect
from types import MethodType


def load_training_kernels():
    # find_spec only checks presence: an installed CUDA extension can still
    # fail to import, which Transformers catches and silently replaces.
    try:
        from causal_conv1d import causal_conv1d_fn
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule
        if not callable(causal_conv1d_fn) or not callable(chunk_gated_delta_rule):
            raise TypeError("the installed training kernels are not callable")
    except Exception as error:
        raise RuntimeError(
            "Qwen3.8 training kernels could not be imported: "
            f"{type(error).__name__}: {error}. Repair causal-conv1d / "
            "flash-linear-attention for this server's PyTorch/CUDA environment "
            "before launching. No packages are installed automatically; "
            "refusing silent reference-kernel training."
        ) from error
    return causal_conv1d_fn, chunk_gated_delta_rule


def _kernel_dispatch(original, attribute, parameters):
    @wraps(original)
    def dispatch(*args, **kwargs):
        layer = kwargs.pop("_ttt_training_kernel_layer", None)
        if layer is None:
            return original(*args, **kwargs)
        # Match Transformers' dispatch filtering: layer metadata is not a
        # kernel argument. Keep sequence boundaries and all supported options.
        return getattr(layer, attribute)(
            *args, **{key: value for key, value in kwargs.items() if key in parameters})

    dispatch._ttt_training_kernel_dispatch = True
    return dispatch


def _kernel_forward(layer, *args, **kwargs):
    kwargs["_ttt_training_kernel_layer"] = layer
    return layer._ttt_native_delta_forward(*args, **kwargs)


def install_training_kernels(model, kernels):
    """Bind actual native kernels, not a wrapper that may select a fallback."""
    convolution, delta = kernels
    installed = 0
    for layer in model.modules():
        if type(layer).__name__ != "Qwen3_5GatedDeltaNet":
            continue
        if getattr(layer, "_ttt_training_kernels_bound", False):
            installed += 1
            continue
        native = inspect.unwrap(layer.forward)
        names = set(getattr(getattr(native, "__code__", None), "co_names", ()))
        if "torch_chunk_gated_delta_rule" in names:
            implementation = importlib.import_module(type(layer).__module__)
            for name, attribute, kernel in (
                    ("causal_conv1d_fn", "causal_conv1d_fn", convolution),
                    ("torch_chunk_gated_delta_rule", "chunk_gated_delta_rule", delta)):
                original = getattr(implementation, name, None)
                if not callable(original):
                    raise RuntimeError(f"unsupported Qwen DeltaNet kernel interface: {name}")
                if not getattr(original, "_ttt_training_kernel_dispatch", False):
                    setattr(implementation, name, _kernel_dispatch(
                        original, attribute, tuple(inspect.signature(kernel).parameters)))
            layer._ttt_native_delta_forward = layer.forward
            layer.forward = MethodType(_kernel_forward, layer)
        elif not (callable(getattr(layer, "causal_conv1d_fn", None))
                  and callable(getattr(layer, "chunk_gated_delta_rule", None))):
            raise RuntimeError("unsupported Qwen DeltaNet forward; cannot bind exact training kernels")
        layer.causal_conv1d_fn = convolution
        layer.chunk_gated_delta_rule = delta
        layer._ttt_training_kernels_bound = True
        installed += 1
    if not installed:
        raise RuntimeError("Qwen3.8 training model contains no supported DeltaNet layers")
    return installed
