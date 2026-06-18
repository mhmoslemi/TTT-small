"""
Problem registry for TTT-local.

Each problem mirrors one example from the original TTT-Discover codebase
(examples/<name>/env.py) but is wrapped around the local, no-Ray/no-Tinker
engine (sandbox.py, sampler.py, advantage.py, model_backend.py, gen_workers.py).

A problem bundles four things the original split across Environment +
RewardEvaluator + create_initial_state:

  - build_prompt(parent)   -> chat messages (verbatim get_question template)
  - preprocess(code, ...)  -> full program injected into the sandbox
  - score(output, stdout)  -> RewardResult (reward is always higher-is-better)
  - seed_states()          -> initial archive states

Use problems.get_problem(name, cfg) to construct one.
"""

from problems.registry import get_problem, available_problems

__all__ = ["get_problem", "available_problems"]
