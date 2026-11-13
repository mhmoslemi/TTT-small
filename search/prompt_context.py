"""Recover the previous-program output of archived states from saved rollouts.

A checkpoint written before State.observation existed has no output for the
programs already in the archive. Every rollout meta keeps the first
SAVED_STDOUT_CHARS characters of its sandbox output, which is the whole output
for most programs. This module only reads files; it needs no GPU packages.
"""
import json
from pathlib import Path

SAVED_STDOUT_CHARS = 2000   # what train_multy_CVaR saves as meta["sandbox_stdout"]


def observations_from_rollout_metas(exp_dir, wanted):
    """wanted = {timestep: {state id, ...}} -> {state id: text}.

    Uses meta["observation"] when the run already wrote it. Otherwise the saved
    sandbox output is used; when it hit the 2000-character limit only the start
    exists, and the text says so instead of passing it off as the whole output.
    """
    found = {}
    for timestep, ids in wanted.items():
        directory = Path(exp_dir) / f"step{int(timestep):02d}"
        pending = set(ids)
        for path in sorted(directory.glob("step*_group*_rollout*.meta.json")):
            if not pending:
                break
            try:
                meta = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            node = meta.get("node_id")
            if node not in pending:
                continue
            pending.discard(node)
            text = str(meta.get("observation") or "").strip()
            if not text:
                saved = str(meta.get("sandbox_stdout") or "").strip()
                if saved and len(saved) >= SAVED_STDOUT_CHARS:
                    saved += (f"\n[... only the first {SAVED_STDOUT_CHARS} characters "
                              "of this output were saved ...]")
                text = saved
            if text:
                found[node] = text
    return found
