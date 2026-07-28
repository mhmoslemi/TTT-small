"""
EVOLVE — entry point.

Resolves the configuration, then hands it to the engine (Algorithm 1). Run it
through run.sh, or directly:

    python main.py --example circle_packing --print-config
    python main.py --example circle_packing --steps 3 --set example.params.num_circles=10
"""

import sys

from config import load_config


def main() -> int:
    resolution = load_config()
    cfg = resolution.config

    print(resolution.explain(changed_only=True))
    print()

    try:
        from core.engine import Engine
    except ModuleNotFoundError:
        print("[evolve] configuration resolved; core/engine.py is not implemented yet.")
        print("[evolve] re-run with --print-config to inspect the full resolution.")
        return 0

    return Engine(cfg).run()


if __name__ == "__main__":
    sys.exit(main())
