"""
Example resolution.

The framework never imports a concrete example. It takes `example.module` (or
falls back to examples.<name>.env), imports it, and calls its `build(cfg)`.
Adding a use-case is a new directory plus a config.yaml -- this file does not
change.
"""

import importlib
from typing import List

from config import EXAMPLES_DIR


def available_examples() -> List[str]:
    if not EXAMPLES_DIR.is_dir():
        return []
    return sorted(d.name for d in EXAMPLES_DIR.iterdir()
                  if d.is_dir() and not d.name.startswith(("_", ".")))


def load_example(cfg):
    """Import example.module and build the Example instance."""
    name = cfg.example.name
    module_path = cfg.example.module or f"examples.{name}.env"

    try:
        module = importlib.import_module(module_path)
    except ModuleNotFoundError as e:
        raise ImportError(
            f"cannot import example module {module_path!r} for example {name!r}: {e}\n"
            f"Available example directories: {', '.join(available_examples()) or '(none)'}"
        ) from e

    builder = getattr(module, "build", None)
    if builder is None:
        raise ImportError(
            f"{module_path} must define build(cfg) -> Example "
            f"(found: {', '.join(n for n in dir(module) if not n.startswith('_'))})"
        )
    return builder(cfg)
