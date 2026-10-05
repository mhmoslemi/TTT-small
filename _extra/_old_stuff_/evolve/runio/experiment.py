"""
Per-run file I/O.

Every rollout is written to disk, including the ones that failed to parse or
failed validation -- a run's failures are the input to the memory module and to
Eq. 9, so a log that keeps only the successes hides half the training signal.

    runs/circle_packing_n26_Qwen3-8B_0728-1432/
      config.json                        full resolved config
      provenance.txt                     which layer set each key
      step00/
        step00_target00_rollout000.txt          raw response
        step00_target00_rollout000.prompt.txt   full rendered prompt
        step00_target00_rollout000.meta.json    reward, valid, feedback, advantage
        step00.summary.json                   per-step stats
        step00_elo/                           judge prompts, replies, standings
      memory.json                         the lesson bank
      tree.json                           the search tree + archive
      best_code.py / final.summary.json
"""

import json
import re
import time
from pathlib import Path
from typing import Optional


def _slug(text) -> str:
    return re.sub(r"[^A-Za-z0-9._\-]", "_", str(text)).strip("_")


def _coerce(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    for attr in ("item", "tolist"):
        if hasattr(value, attr):
            try:
                return getattr(value, attr)()
            except Exception:
                pass
    return str(value)


class ExperimentIO:
    def __init__(self, cfg, resolution=None):
        self.cfg = cfg
        self.enabled = bool(cfg.run.save_rollouts)
        self.root = Path(cfg.run.output_root) / (cfg.run.run_name or self._auto_name())
        self.root.mkdir(parents=True, exist_ok=True)

        (self.root / "config.json").write_text(
            json.dumps(cfg.to_dict(), indent=2, default=str))
        if resolution is not None:
            (self.root / "provenance.txt").write_text(resolution.explain())

    def _auto_name(self) -> str:
        cfg = self.cfg
        parts = [_slug(cfg.example.name)]
        for key in ("num_circles", "n"):
            if key in (cfg.example.params or {}):
                parts.append(f"n{cfg.example.params[key]}")
                break
        parts.append(_slug(str(cfg.model.name).split("/")[-1]))
        parts.append(time.strftime("%m%d-%H%M"))
        return "_".join(parts)

    # ------------------------------------------------------------------
    def step_dir(self, step: int) -> Path:
        path = self.root / f"step{step:02d}"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def save_rollout(self, step: int, target_idx: int, rollout_idx: int,
                     response_text: str, meta: dict,
                     prompt_text: str = "") -> None:
        if not self.enabled:
            return
        base = f"step{step:02d}_target{target_idx:02d}_rollout{rollout_idx:03d}"
        directory = self.step_dir(step)
        (directory / f"{base}.txt").write_text(response_text or "", errors="replace")
        (directory / f"{base}.meta.json").write_text(
            json.dumps({k: _coerce(v) for k, v in meta.items()}, indent=2))
        if prompt_text:
            (directory / f"{base}.prompt.txt").write_text(prompt_text,
                                                          errors="replace")

    def save_step_summary(self, step: int, summary: dict) -> None:
        (self.step_dir(step) / f"step{step:02d}.summary.json").write_text(
            json.dumps(summary, indent=2, default=str))

    def save_elo_match(self, step: int, idx: int, meta: dict,
                       prompt_text: str = "", response_text: str = "") -> None:
        if not self.enabled:
            return
        directory = self.step_dir(step) / f"step{step:02d}_elo"
        directory.mkdir(parents=True, exist_ok=True)
        base = f"match{idx:04d}"
        (directory / f"{base}.meta.json").write_text(
            json.dumps({k: _coerce(v) for k, v in meta.items()}, indent=2))
        (directory / f"{base}.response.txt").write_text(response_text or "",
                                                        errors="replace")
        if prompt_text:
            (directory / f"{base}.prompt.txt").write_text(prompt_text,
                                                          errors="replace")

    def save_elo_standings(self, step: int, standings: list) -> None:
        directory = self.step_dir(step) / f"step{step:02d}_elo"
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "standings.json").write_text(
            json.dumps(standings, indent=2, default=str))

    # ------------------------------------------------------------------
    def save_tree(self, tree) -> None:
        (self.root / "tree.json").write_text(
            json.dumps(tree.to_dict(), indent=2, default=str))

    def save_memory(self, bank) -> None:
        if bank is not None:
            bank.save(self.root / "memory.json")

    def save_final(self, best_node, step: int, history: Optional[list] = None) -> None:
        payload = {
            "best_reward": float(best_node.reward) if best_node else None,
            "best_raw_score": (best_node.raw_score if best_node else None),
            "best_step": int(best_node.step) if best_node else None,
            "final_step": int(step),
            "history": history or [],
        }
        (self.root / "final.summary.json").write_text(
            json.dumps(payload, indent=2, default=str))
        if best_node is not None and best_node.code:
            (self.root / "best_code.py").write_text(best_node.code)

    def save_adapter(self, backbone, step: int) -> Optional[Path]:
        """Persist the LoRA weights so a run can be resumed or the policy reused."""
        model = getattr(backbone, "model", None)
        if model is None or not hasattr(model, "save_pretrained"):
            return None
        path = self.root / "adapters" / f"step{step:02d}"
        path.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(path))
        return path
