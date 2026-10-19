"""Stable tensor layouts at GPT-OSS's compiled attention boundaries."""

import functools
import importlib
import types

import torch


def _contiguous_inputs(function, names):
    if getattr(function, "_ttt_attention_layout", False):
        return function

    # Keep the layout conversion OUTSIDE the compiled graph. Otherwise an
    # outer compiler can fold it away and hand the backward a different
    # layout from its saved-input contract. The original compiled kernel
    # still runs, including its fused forward and backward implementations.
    @torch.compiler.disable(recursive=False)
    @functools.wraps(function)
    def call(*args, **kwargs):
        args = list(args)
        for index, name in enumerate(names):
            if index < len(args):
                args[index] = args[index].contiguous()
            elif name in kwargs:
                kwargs[name] = kwargs[name].contiguous()
        return function(*args, **kwargs)

    call._ttt_attention_layout = True
    return call


def install_attention_layout_guards(model):
    """Retain Unsloth's kernels/sinks/masks; normalize only Q/K/V storage.

    Unsloth 2026.9.3 captures a compiled rotary function in the patched
    attention forward's closure. Flex attention is a module-global compiled
    callable. Cover both boundaries: the failing [B, 8, L, 64] input can be
    a rotary key or a flex key/value. No weights or attention math change.
    """
    attention_sink = importlib.import_module(
        "unsloth_zoo.flex_attention.attention_sink")
    attention_sink.flex_attention = _contiguous_inputs(
        attention_sink.flex_attention, ("query", "key", "value"))
    visited = set()
    rotary_boundaries = 0

    def visit(function):
        nonlocal rotary_boundaries
        if isinstance(function, types.MethodType):
            function = function.__func__
        if not callable(function) or id(function) in visited:
            return
        visited.add(id(function))
        if getattr(function, "_ttt_attention_layout", False):
            return
        # Accelerate uses update_wrapper(partial(new_forward, module),
        # old_forward). Follow its metadata BEFORE filtering for plain Python
        # functions; otherwise every hooked attention layer is skipped.
        for name in ("__wrapped__", "_torchdynamo_orig_callable"):
            visit(getattr(function, name, None))
        if isinstance(function, functools.partial):
            visit(function.func)
        if not isinstance(function, types.FunctionType):
            return
        for name, cell in zip(function.__code__.co_freevars,
                              function.__closure__ or ()):
            try:
                value = cell.cell_contents
            except ValueError:
                continue
            if name == "apply_rotary_pos_emb" and callable(value):
                cell.cell_contents = _contiguous_inputs(value, ("q", "k"))
                rotary_boundaries += 1
            elif "forward" in name:
                visit(value)
        # Unsloth's generated forward can bind the same function as a global
        # instead of a closure. Inspect the actual loaded forward, not a
        # hard-coded package path or a generated cache filename.
        namespace = function.__globals__
        for name in function.__code__.co_names:
            if name == "apply_rotary_pos_emb" and callable(namespace.get(name)):
                namespace[name] = _contiguous_inputs(namespace[name], ("q", "k"))
                rotary_boundaries += 1
            elif "forward" in name:
                visit(namespace.get(name))

    for module in model.modules():
        if getattr(module, "sinks", None) is not None:
            visit(module.forward)
            # Accelerate restores this same method when hooks are removed
            # for CPU offload. Patch its rotary call, not the routing wrapper,
            # so protection survives offload and redispatch unchanged.
            visit(getattr(module, "_old_forward", None))
    if not rotary_boundaries:
        raise RuntimeError(
            "GPT-OSS attention layout guard could not locate the loaded "
            "rotary call; refusing to start an unprotected compiled backward")
    return rotary_boundaries
