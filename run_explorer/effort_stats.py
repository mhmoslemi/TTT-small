#!/usr/bin/env python3
"""
effort_stats.py - how long the coder thinks at every reasoning effort, and how fast tokens come out (read-only).

Used by build_explorer.py --unified for the hub tab "Reasoning effort & speed". Standard library only.

What is measured, and how
  * tokens per rollout      meta["n_response_tokens"] of every scored rollout, grouped by meta["coder_reasoning_effort"].
                            This is the whole response (thinking + code), exact.
  * thinking vs answer      for a sample of rollouts the saved response text is split at its reasoning boundary
                            ("</think>" for Qwen, "assistantfinal" for gpt-oss) and the character share of the reasoning is applied to the
                            token count. That is an ESTIMATE (code has more tokens per character than prose).
  * generation speed        the log's progress bar of the PILOT phase ("pilot rollouts: ... N/N [elapsed<...]"): the phase generates all its
                            rollouts at once and CPU / GPU evaluation of earlier programs overlaps without blocking it, so
                            tokens(pilot rollouts of that step) / elapsed is the effective aggregate speed of the generation GPUs, in
                            tokens per second per GPU. It includes ramp-up and the tail where only the longest sequences are still running.
                            Per-effort speed cannot be read from the logs (the efforts run together in one batch); it is derived from the
                            measured speed of the model, see `derived` below.
Which runs are used is listed in effort_runs.txt.
"""
import glob
import json
import os
import re
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
BAR = re.compile(r"^\[(\d\d):(\d\d):(\d\d)\] pilot rollouts:\s+\d+%\|[^|]*\|\s*(\d+)/(\d+)\s*\[([0-9:]+)<")
STEP = re.compile(r"\[step (\d+)\]")
ANSI = re.compile(r"\x1b\[[0-9;]*m")
MARKERS = (("</think>", 8), ("assistantfinal", 14))      # (text, length): the end of the reasoning in the saved response
SAMPLE_PER_GROUP = 120


def family(cfg, name):
    p = str(cfg.get("coder_model_profile") or cfg.get("coder_model_name") or cfg.get("model_name") or name).lower()
    if "gpt-oss" in p or "gpt_oss" in p:
        return "gpt-oss-120b"
    if "qwen" in p:
        return "Qwen3.8-27B"
    return p.split("/")[-1] or "model"


def _secs(s):
    p = [int(x) for x in s.split(":")]
    while len(p) < 3:
        p.insert(0, 0)
    return p[0] * 3600 + p[1] * 60 + p[2]


def _quant(v, q):
    v = sorted(v)
    return v[min(len(v) - 1, int(q * (len(v) - 1) + 0.5))]


def _summary(v):
    if not v:
        return None
    return {"n": len(v), "mean": round(st.mean(v)), "median": round(st.median(v)), "p10": round(_quant(v, .10)), "p90": round(_quant(v, .90)), "max": max(v)}


def _think_share(path):
    """Fraction of the saved response (in characters) that is reasoning, or None if no boundary is visible."""
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            t = fh.read()
    except OSError:
        return None
    for mk, ln in MARKERS:
        i = t.find(mk)
        if i > 0:
            return i / max(1, len(t) - ln)
    return None


def pilot_bars(log_path):
    """[(step, total, elapsed_s)] of every pilot progress bar that reached (almost) its total."""
    out, cur, step = [], None, None
    try:
        with open(log_path, encoding="utf-8", errors="replace") as fh:
            for ln in fh:
                ln = ANSI.sub("", ln.rstrip("\n"))
                ms = STEP.search(ln)
                if ms and "pilot rollouts:" not in ln:
                    step = int(ms.group(1))
                m = BAR.match(ln)
                if not m:
                    continue
                k, n, el = int(m.group(4)), int(m.group(5)), _secs(m.group(6))
                if cur is None or k < cur["k"] or n != cur["n"]:
                    cur = {"step": step, "k": k, "n": n, "el": el}
                    out.append(cur)
                else:
                    cur["k"], cur["el"] = k, el
    except OSError:
        return []
    return [(b["step"], b["n"], b["el"]) for b in out if b["k"] >= b["n"] - 2 and b["el"] > 0]


def scan_run(run_dir):
    name = os.path.basename(run_dir.rstrip("/"))
    try:
        with open(os.path.join(run_dir, "config.json"), encoding="utf-8") as fh:
            cfg = json.load(fh)
    except (OSError, ValueError):
        return None
    fam = family(cfg, name)
    toks = {}          # effort -> [ntok]
    pilot = {}         # step -> {effort: tokens} of pilot-phase rollouts (first attempt)
    files = {}         # effort -> [text path] for the thinking-share sample
    for mf in sorted(glob.glob(os.path.join(run_dir, "step*", "step*_group*_rollout*.meta.json"))):
        try:
            with open(mf, encoding="utf-8") as fh:
                m = json.load(fh)
        except (OSError, ValueError):
            continue
        n, eff = m.get("n_response_tokens"), m.get("coder_reasoning_effort")
        if not n or not eff or "valid" not in m:
            continue
        toks.setdefault(eff, []).append(n)
        if (m.get("strategy_rollout_phase") or "pilot") == "pilot" and not m.get("retry_attempt"):
            d = pilot.setdefault(int(m.get("step", -1)), {})
            d[eff] = d.get(eff, 0) + n
        files.setdefault(eff, []).append(mf[:-len(".meta.json")] + ".txt")
    if not toks:
        return None
    modes = {}
    for eff, v in toks.items():
        f = files[eff]
        pick = f[:: max(1, len(f) // SAMPLE_PER_GROUP)][:SAMPLE_PER_GROUP]
        shares = [s for s in (_think_share(p) for p in pick) if s is not None]
        d = _summary(v)
        d["think_share"] = round(st.mean(shares), 3) if shares else None
        d["think_sampled"] = len(shares)
        d["think_est"] = round(d["mean"] * d["think_share"]) if shares else None
        modes[eff] = d
    gpus = [g for g in str(cfg.get("gpu_ids") or cfg.get("available_gpu_ids") or "").split(",") if g.strip()]
    hw = cfg.get("hardware_profile") or cfg.get("gpu_type") or ""
    rates = []
    for step, total, el in pilot_bars(os.path.join(run_dir, "temirnal.log")):
        t = pilot.get(step)
        if not t or not gpus:
            continue
        tok = sum(t.values())
        rates.append({"step": step, "rollouts": total, "tokens": tok, "elapsed_s": el, "tps_total": round(tok / el, 1), "tps_gpu": round(tok / el / len(gpus), 1),
                      "mix": {e: round(x / tok, 3) for e, x in t.items()}})
    return {"run": name, "problem": str(cfg.get("problem") or "").lower(), "model": fam, "hardware": hw, "n_gpus": len(gpus), "modes": modes, "rates": rates,
            "b200": "b200" in str(hw).lower()}


def collect(run_dirs, log=print):
    runs = []
    for d in run_dirs:
        r = scan_run(d)
        if r:
            runs.append(r)
            log("  effort stats: %-46s %s, %d rollouts, %d pilot timings" % (r["run"], r["model"], sum(m["n"] for m in r["modes"].values()), len(r["rates"])))
        else:
            log("  effort stats: skipped %s (no config / no scored rollouts)" % os.path.basename(d))
    # pooled token statistics per (model, effort, problem)
    pooled = {}
    for r in runs:
        for eff, m in r["modes"].items():
            pooled.setdefault((r["model"], eff, r["problem"]), []).append((r, m))
    groups = []
    for (model, eff, prob), items in sorted(pooled.items()):
        n = sum(m["n"] for _, m in items)
        wmean = lambda key: (sum(m[key] * m["n"] for _, m in items if m.get(key) is not None) / max(1, sum(m["n"] for _, m in items if m.get(key) is not None))) if any(m.get(key) is not None for _, m in items) else None
        sh = [(m["think_share"], m["think_sampled"]) for _, m in items if m.get("think_share") is not None]
        share = sum(a * b for a, b in sh) / sum(b for _, b in sh) if sh else None
        mean = wmean("mean")
        groups.append({"model": model, "effort": eff, "problem": prob, "n": n, "mean": round(mean), "median": round(wmean("median")), "p90": round(wmean("p90")), "max": max(m["max"] for _, m in items),
                       "think_share": round(share, 3) if share is not None else None, "think_est": round(mean * share) if share is not None else None,
                       "runs": sorted({r["run"] for r, _ in items})})
    # measured speed per model, B200 runs only (the same hardware for both models)
    speed = {}
    for r in runs:
        if not r["b200"]:
            continue
        for x in r["rates"]:
            speed.setdefault(r["model"], []).append(dict(x, run=r["run"], problem=r["problem"]))
    speed_out = {}
    for model, xs in speed.items():
        pure = [x for x in xs if len(x["mix"]) == 1 or max(x["mix"].values()) > 0.97]
        use = pure or xs
        speed_out[model] = {"tps_gpu_median": round(st.median(x["tps_gpu"] for x in use), 1), "tps_gpu_min": min(x["tps_gpu"] for x in use), "tps_gpu_max": max(x["tps_gpu"] for x in use),
                            "n_timings": len(use), "n_runs": len({x["run"] for x in use}), "basis": "steps whose pilot phase was (almost) all one effort" if pure else "all pilot phases (mixed efforts)",
                            "samples": use}
    # derived cost per rollout: mean tokens of the pooled group / measured aggregate speed of that model on B200
    derived = []
    for g in groups:
        sp = speed_out.get(g["model"])
        if sp:
            derived.append({"model": g["model"], "effort": g["effort"], "problem": g["problem"], "gpu_seconds": round(g["mean"] / sp["tps_gpu_median"]), "tps_gpu": sp["tps_gpu_median"]})
    return {"runs": runs, "groups": groups, "speed": speed_out, "derived": derived}


def default_runs():
    lst = os.path.join(HERE, "effort_runs.txt")
    out = []
    if os.path.isfile(lst):
        for line in open(lst, encoding="utf-8").read().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                p = line if os.path.isabs(line) else os.path.normpath(os.path.join(HERE, line))
                if os.path.isdir(p):
                    out.append(p)
                else:
                    print("warning: effort_runs.txt lists a missing directory, skipped: %s" % p)
    return out


if __name__ == "__main__":
    res = collect(default_runs())
    print(json.dumps({"groups": res["groups"], "speed": {k: {kk: vv for kk, vv in v.items() if kk != "samples"} for k, v in res["speed"].items()}, "derived": res["derived"]}, indent=1))
