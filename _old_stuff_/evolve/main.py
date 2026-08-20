"""
EVOLVE — entry point.

Resolves the configuration, then runs Algorithm 1. Use run.sh, or directly:

    python main.py --example circle_packing --print-config
    python main.py --example circle_packing --steps 3 --set example.params.num_circles=10
    python main.py --backend mock --steps 2        # plumbing check, no model
"""

import sys

from config import ConfigError, load_config


def main() -> int:
    try:
        resolution = load_config()
    except ConfigError as e:
        print(f"[config] {e}", file=sys.stderr)
        return 2

    print(resolution.explain(changed_only=True))
    print()

    from core.engine import Engine
    return Engine(resolution.config, resolution).run()


if __name__ == "__main__":
    sys.exit(main())
