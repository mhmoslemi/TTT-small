"""Untimed TriMul compliance checks; timing still uses the unwrapped submission."""

import re

import torch
from torch.utils._python_dispatch import TorchDispatchMode


# Pointer/index arithmetic and plain load/store copies do not implement TriMul.
_MATH = re.compile(
    r"\b(?:arith\.(?:addf|subf|mulf|divf|remf|negf|cmpf|maximumf|minimumf|maxnumf|minnumf)"
    r"|math\.\w+|tt\.dot)\b|\barith\.select\b[^\n]*\b(?:bf16|f16|f32|f64)\b"
)


class _InitializeEmpty(TorchDispatchMode):
    """Avoid accidentally reusing correct intermediates from the preceding run."""

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        result = func(*args, **(kwargs or {}))
        if func._schema.name in {
            "aten::empty", "aten::empty_like", "aten::empty_strided",
            "aten::new_empty", "aten::new_empty_strided",
        }:
            result.fill_(float("nan") if result.is_floating_point() else 0)
        return result


def check_triton_contribution(custom_kernel, data, clone_data, check_output):
    """Require correct output and numerical custom Triton work on this input.

    An untimed counterfactual disables numerical submission kernels while
    leaving PyTorch and copy kernels intact. Still-correct output means that
    the custom numerical work was unnecessary for this case at task tolerance.
    shortcut: this audits ordinary JIT launches, not adversarial Python or
    pre-captured graphs; unsupported paths fail closed and need a separate audit.
    """
    from triton.runtime.jit import JITFunction

    original_run = JITFunction.run
    launched = set()
    suppressed = set()
    probing = False

    def tracked_run(kernel, *args, grid, warmup, **kwargs):
        if kernel.fn.__module__ != custom_kernel.__module__ or warmup:
            return original_run(kernel, *args, grid=grid, warmup=warmup, **kwargs)
        if probing:
            compiled = original_run(kernel, *args, grid=grid, warmup=True, **kwargs)
            if compiled is None or "ttir" not in compiled.asm:
                raise RuntimeError("cannot inspect submitted Triton kernel")
            if _MATH.search(compiled.asm["ttir"]):
                suppressed.add(kernel.__name__)
                return compiled
        compiled = original_run(kernel, *args, grid=grid, warmup=False, **kwargs)
        if compiled is None or "ttir" not in compiled.asm:
            raise RuntimeError("cannot inspect submitted Triton kernel")
        if _MATH.search(compiled.asm["ttir"]):
            launched.add(kernel.__name__)
        return compiled

    devices = [data[0].device.index] if data[0].is_cuda else []
    try:
        JITFunction.run = tracked_run
        with torch.random.fork_rng(devices=devices):
            output = custom_kernel(clone_data(data))
            torch.cuda.synchronize()
            good, message = check_output(data, output)
        del output
        if not good:
            return False, message
        if not launched:
            return False, "No numerical submission Triton kernel executed (unused/copy-only code)."

        probing = True
        with torch.random.fork_rng(devices=devices), _InitializeEmpty():
            output = custom_kernel(clone_data(data))
            torch.cuda.synchronize()
            if not suppressed:
                return False, "Cannot verify Triton contribution: no interceptable launches on fresh inputs."
            still_correct, _ = check_output(data, output)
        if still_correct:
            return False, "Output remains correct without the submitted Triton computation."
        return True, "Triton contribution checked: " + ", ".join(sorted(launched))
    except Exception as exc:
        # An audit error is not evidence that a kernel contributed to the result.
        return False, f"Triton contribution check failed: {type(exc).__name__}: {exc}"
    finally:
        JITFunction.run = original_run
