#!/usr/bin/env python3
"""One-time exact archive deduplication and PUCT conversion for a stopped run."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil

from sampler import PUCTSampler, State


def deduplicate_sampler(saved):
    if saved.get("version") != 1 or not saved.get("states"):
        raise ValueError("Unsupported or empty sampler checkpoint")
    seeds = set(saved.get("seed_ids", []))
    visits = saved.get("n", {})
    states = saved["states"]
    ids = {row["id"] for row in states}
    if len(ids) != len(states) or not seeds <= ids:
        raise ValueError("Invalid saved state/seed identities")
    winners = {}
    for row in states:
        key = PUCTSampler._archive_key(State(**row))
        if key is None:
            key = ("unkeyed", row["id"])
        # Keep the original seed when present; otherwise favor the already
        # explored representative, rather than replacing it with a fresh ID.
        priority = (row["id"] in seeds, int(visits.get(row["id"], 0)),
                    float(row["value"]) if row["value"] is not None else float("-inf"))
        if key not in winners or priority > winners[key][0]:
            winners[key] = (priority, row["id"])
    retained = {sid for _, sid in winners.values()}
    cleaned = dict(saved, states=[row for row in states if row["id"] in retained],
                   seed_ids=sorted(seeds & retained), selection_mode="puct")
    # Keep every historical n/m entry and all original ancestry links. Adding
    # subtree counts across duplicate ancestors would double-count rollouts.
    probe = PUCTSampler(num_seeds=0)
    probe.load_state_dict(cleaned)
    return cleaned, len(states) - len(retained)


def require_step_boundary(run, step):
    directory = run / f"step{step:02d}"
    if ((directory / f"step{step:02d}.parents.json").exists()
            or next(directory.glob(f"step{step:02d}_group*_rollout*"), None) is not None):
        raise ValueError(
            f"Step {step} has already started. Finish and checkpoint that step "
            "with the original sampler before cleaning/switching to PUCT; "
            "changing its selected parents would break recovery.")


def clean_archive(run_dir):
    import torch

    run = Path(run_dir).expanduser().resolve()
    checkpoint = run / "training_state.pt"
    config_path = run / "config.json"
    original_stat = checkpoint.stat()
    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes)
    # Optimizer storage stays memory-mapped on CPU; no model/GPU load or
    # additional full in-memory copy of the optimizer is needed.
    payload = torch.load(str(checkpoint), map_location="cpu", weights_only=False, mmap=True)
    if (not isinstance(payload, dict) or payload.get("version") not in (1, 2)
            or not {"next_step", "adapter_dir", "optimizer", "sampler"} <= payload.keys()):
        raise ValueError("Unsupported or incomplete training checkpoint")
    step = int(payload["next_step"])
    require_step_boundary(run, step)
    cleaned, removed = deduplicate_sampler(payload["sampler"])
    old_count = len(payload["sampler"]["states"])
    old_mode = payload["sampler"].get("selection_mode", "puct")
    if (not removed and old_mode == "puct"
            and config.get("parent_selection") == "puct" and not config.get("uct", False)):
        print(f"Already clean: {old_count} archived states; PUCT; next step {step}.")
        return
    if shutil.disk_usage(run).free < 2 * original_stat.st_size + 64 * 1024**2:
        raise ValueError("Insufficient disk space for checkpoint backup plus replacement")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    backup = run / f"archive_cleanup_backup_{stamp}"
    backup.mkdir()
    shutil.copy2(checkpoint, backup / checkpoint.name)
    shutil.copy2(config_path, backup / config_path.name)
    temporary = run / f".training_state.cleanup_{stamp}.pt.tmp"
    config_temporary = run / f".config.cleanup_{stamp}.json.tmp"
    print(f"Backup: {backup}", flush=True)
    try:
        payload["sampler"] = cleaned
        torch.save(payload, temporary)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        config.update(parent_selection="puct", uct=False)
        config_temporary.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        require_step_boundary(run, step)
        current_stat = checkpoint.stat()
        if ((current_stat.st_ino, current_stat.st_size, current_stat.st_mtime_ns)
                != (original_stat.st_ino, original_stat.st_size, original_stat.st_mtime_ns)
                or config_path.read_bytes() != config_bytes):
            raise RuntimeError("Run files changed during cleanup; stop training before retrying")
        temporary.replace(checkpoint)
        try:
            config_temporary.replace(config_path)
        except BaseException:
            # Restore the old checkpoint if the companion config cannot commit.
            shutil.copy2(backup / checkpoint.name, temporary)
            temporary.replace(checkpoint)
            raise
    finally:
        temporary.unlink(missing_ok=True)
        config_temporary.unlink(missing_ok=True)
    print(f"Archive: {old_count} -> {len(cleaned['states'])}; removed {removed} duplicates.")
    print(f"Sampler: {old_mode} -> puct; next step remains {step}.")
    print("Optimizer, SPO-RS tracker, adapters and rollout/tree files preserved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", help="Stopped run directory containing training_state.pt")
    args = parser.parse_args()
    try:
        clean_archive(args.run_dir)
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"Cleanup stopped: {error}\n")
