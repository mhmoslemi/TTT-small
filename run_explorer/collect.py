"""
collect.py - read a finished/in-flight run directory and turn it into one compact,
JSON-serialisable dict that the HTML front-end consumes.

READ-ONLY with respect to the run directory and the rest of the code base.
Standard library only (Python >= 3.8).
"""
from __future__ import annotations

import concurrent.futures as cf
import datetime as _dt
import glob
import json
import os
import re
import sys
import time
from collections import defaultdict

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m|\[(?:\d{1,3};)*\d{1,3}m")
CODE_RE = re.compile(r"```python\s*\n(.*?)```", re.S)
STRAT_RE = re.compile(r"<strategy>(.*?)</strategy>", re.S)
FALLBACK_MARK = "No usable strategy was returned"
ROLLOUT_META_RE = re.compile(r"step(\d+)_group(\d+)_rollout(\d+)\.meta\.json$")
ATTEMPT_META_RE = re.compile(r"step(\d+)_group(\d+)_fold(\d+)_strategy(\d+)_attempt(\d+)\.meta\.json$")
STRAT_FILE_RE = re.compile(r"step(\d+)_group(\d+)_fold(\d+)_strategy(\d+)\.txt$")


def _jload(path, default=None):
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return default


def _read(path, default=""):
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            return fh.read()
    except Exception:
        return default


def _sig(x, digits=6):
    """Shorten floats so the embedded JSON stays small."""
    if x is None:
        return None
    try:
        if x != x or x in (float("inf"), float("-inf")):
            return None
        return float("%.*g" % (digits, x))
    except Exception:
        return None


def extract_code(response_text):
    """Return the last ```python block of a coder response (the final program)."""
    blocks = CODE_RE.findall(response_text or "")
    return blocks[-1].strip("\n") if blocks else ""


def split_think(text):
    """Split '<think> ... </think> answer' (the opening tag is usually in the prompt)."""
    if "</think>" in text:
        a, b = text.split("</think>", 1)
        return a.replace("<think>", "", 1).strip(), b.strip()
    return "", text


def parse_construction(s, limit=None):
    if not s:
        return []
    try:
        arr = json.loads(s)
        return arr if limit is None else arr[:limit]
    except Exception:
        return []


# ----------------------------------------------------------------------------------------
# per-step worker (runs in a process pool)
# ----------------------------------------------------------------------------------------
def _scan_step(args):
    run_dir, step = args
    sd = os.path.join(run_dir, "step%02d" % step)
    out = {"step": step, "rollouts": [], "strategies": {}, "files": 0, "slots": {}}
    names = os.listdir(sd)
    out["files"] = len(names)

    # --- rollouts -------------------------------------------------------------------
    for name in names:
        m = ROLLOUT_META_RE.search(name)
        if not m:
            continue
        meta = _jload(os.path.join(sd, name))
        if not meta:
            continue
        cons = meta.get("construction") or ""
        n_len = (cons.count(",") + 1) if cons.startswith("[") and len(cons) > 2 else 0
        strat = meta.get("strategy") or ""
        rec = {
            "step": step,
            "file_group": int(m.group(2)),
            "file_rollout": int(m.group(3)),
            "group": meta.get("group"),
            "pgroup": meta.get("parent_group"),
            "rollout": meta.get("rollout"),
            "sidx": meta.get("strategy_index"),
            "phase": meta.get("strategy_rollout_phase") or "pilot",
            "node": meta.get("node_id"),
            "parent": meta.get("parent_id"),
            "raw": _sig(meta.get("raw_score"), 9),
            "reward": _sig(meta.get("reward"), 9),
            "valid": bool(meta.get("valid")),
            "parsed": bool(meta.get("parsed")),
            "ran": bool(meta.get("ran")),
            "fail": meta.get("failure_kind") or "",
            "msg": (meta.get("msg") or "")[:160],
            "adv": _sig(meta.get("advantage"), 6),
            "ntok": meta.get("n_response_tokens"),
            "n": n_len,
            "retry": meta.get("retry_attempt") or 0,
            "counts": bool(meta.get("counts_toward_allocation", True)),
            "pval": _sig(meta.get("parent_value"), 9),
            "praw": _sig(meta.get("parent_raw_score"), 9),
            "fallback": FALLBACK_MARK in strat,
            "stdout": (meta.get("sandbox_stdout") or "")[:1200],
            "arch": bool(meta.get("archive_eligible")),
        }
        out["rollouts"].append(rec)
        sk = "%s:%s" % (meta.get("parent_group"), meta.get("strategy_index"))
        if sk not in out["slots"] and meta.get("strategy_rollout_phase") is not None and not meta.get("retry_attempt"):
            out["slots"][sk] = {k: meta.get(k) for k in (
                "strategy_hurdle", "strategy_pilot_reward_mean", "strategy_pilot_reward_variance", "strategy_pilot_valid_count",
                "strategy_posterior_valid_probability", "strategy_initial_expected_improvement", "strategy_pilot_followup_count",
                "strategy_allocated_programs", "strategy_pilot_scenario", "strategy_phase2_allocation_method", "pilot_programs_per_strategy")}

    # --- strategist attempts + final strategies -----------------------------------------
    attempts = defaultdict(list)
    for name in names:
        m = ATTEMPT_META_RE.search(name)
        if not m:
            continue
        g, fo, s, a = (int(x) for x in m.groups()[1:])
        meta = _jload(os.path.join(sd, name), {})
        base = name[: -len(".meta.json")]
        try:
            resp_chars = os.path.getsize(os.path.join(sd, base + ".txt"))
            prompt_chars = os.path.getsize(os.path.join(sd, base + ".prompt.txt"))
        except OSError:
            resp_chars = prompt_chars = 0
        attempts[(g, s)].append({
            "a": a,
            "ok": bool(meta.get("accepted")),
            "issue": meta.get("extraction_issue"),
            "rc": resp_chars,
            "pc": prompt_chars,
        })
    for name in names:
        m = STRAT_FILE_RE.search(name)
        if not m:
            continue
        g, fo, s = (int(x) for x in m.groups()[1:])
        txt = _read(os.path.join(sd, name))
        think, ans = split_think(txt)
        sm = STRAT_RE.search(txt)
        block = sm.group(1).strip() if sm else ""
        fallback = FALLBACK_MARK in block
        # strategy prompt anatomy (character offsets of the pieces) from the last attempt
        att = sorted(attempts.get((g, s), []), key=lambda r: r["a"])
        anat = None
        if att:
            ptxt = _read(os.path.join(sd, "step%02d_group%02d_fold%02d_strategy%02d_attempt%02d.prompt.txt"
                                      % (step, g, fo, s, att[-1]["a"])))
            anat = prompt_anatomy(ptxt)
        out["strategies"]["%d:%d" % (g, s)] = {
            "g": g, "s": s,
            "block": block,
            "think_head": think[:2500],
            "think_tail": think[-1200:] if len(think) > 3700 else "",
            "think_chars": len(think),
            "chars": len(txt),
            "fallback": fallback,
            "attempts": att,
            "anat": anat,
        }
    return out


def _literal(txt):
    import ast
    try:
        v = ast.literal_eval(txt) if isinstance(txt, str) else txt
        return v if isinstance(v, dict) else {}
    except Exception:
        return {}


def alloc_from_slots(slots):
    out = []
    for key in sorted(slots, key=lambda k: tuple(int(x) for x in k.split(":"))):
        g, s = (int(x) for x in key.split(":"))
        m = slots[key]
        hur = _literal(m.get("strategy_hurdle"))
        hur.setdefault("parent_reward", None)
        for k in ("improvement_count", "posterior_improvement_probability", "mean_positive_improvement", "best_positive_improvement"):
            hur.setdefault(k, 0 if k == "improvement_count" else 0.0)
        out.append({
            "parent_group": g, "strategy_index": s, "pilot_count": m.get("pilot_programs_per_strategy") or 11,
            "pilot_reward_mean": m.get("strategy_pilot_reward_mean"), "pilot_reward_variance": m.get("strategy_pilot_reward_variance"),
            "pilot_valid_count": m.get("strategy_pilot_valid_count") or 0, "posterior_valid_probability": m.get("strategy_posterior_valid_probability"),
            "initial_expected_improvement": m.get("strategy_initial_expected_improvement"), "final_marginal_expected_improvement": None,
            "projected_expected_gain": None, "followup_count": m.get("strategy_pilot_followup_count") or 0,
            "allocated_programs": m.get("strategy_allocated_programs"), "scenario": m.get("strategy_pilot_scenario"),
            "allocation_method": m.get("strategy_phase2_allocation_method"), "hurdle": hur, "rebuilt_from_rollouts": True,
        })
    return out


def prompt_anatomy(p):
    """Split a strategist prompt into labelled pieces so the UI can show the 'anatomy'."""
    if not p:
        return None
    marks = [
        ("task", 0),
        ("sota_hint", p.find("A previous state of the art")),
        ("parent_program", p.find("Your previous program achieved")),
        ("rules", p.find("Rules:\n- You must define")),
        ("prev_strategies", p.find("## Strategy-stage output")),
        ("contract", p.find("## Mandatory final-output contract")),
    ]
    marks = [(k, v) for k, v in marks if v >= 0]
    marks.sort(key=lambda kv: kv[1])
    pieces = []
    for i, (k, start) in enumerate(marks):
        end = marks[i + 1][1] if i + 1 < len(marks) else len(p)
        pieces.append([k, end - start])
    prev = len(re.findall(r"<previous_strategy_\d+>", p))
    return {"total": len(p), "pieces": pieces, "prev_strategies": prev}


# ----------------------------------------------------------------------------------------
# log parsing
# ----------------------------------------------------------------------------------------
TS_RE = re.compile(r"^\[(\d\d):(\d\d):(\d\d)\] ")
NOISE_RE = re.compile(r"\[(ram-wait|ram-ready|warn)\]")


def parse_terminal_log(path, max_lines=4000):
    """Returns (filtered events, per-step wall-clock phases)."""
    if not os.path.exists(path):
        return [], {}, {}
    lines = []
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for i, ln in enumerate(fh):
            lines.append(ln.rstrip("\n"))
    events = []
    anchors = []  # (line_idx, secs, kind, step)
    resume_idx = []
    # The log only has HH:MM:SS. Walk it in order: a clock that jumps back by more than 5 minutes means a new day started.
    day, prev_secs, first_un, last_un = 0, None, None, None
    ckpt_un = {}
    for i, ln in enumerate(lines):
        mt = TS_RE.match(ln)
        if not mt:
            continue
        secs0 = int(mt.group(1)) * 3600 + int(mt.group(2)) * 60 + int(mt.group(3))
        if prev_secs is not None and secs0 < prev_secs - 300:
            day += 1
        prev_secs = secs0
        un = day * 86400 + secs0
        if first_un is None:
            first_un = un
        last_un = un
        if "[checkpoint] completed step" in ln:
            try:
                ckpt_un[int(re.search(r"completed step (\d+)", ln).group(1))] = un
            except Exception:
                pass
        if NOISE_RE.search(ln):
            continue
        secs = secs0
        clean = ANSI_RE.sub("", ln)
        tag = re.match(r"^\[\d\d:\d\d:\d\d\] \[([^\]]+)\]", clean)
        tag = tag.group(1) if tag else ""
        st = re.search(r"\[step (\d+)\]", clean)
        stn = int(st.group(1)) if st else None
        if tag in ("resume",):
            resume_idx.append((i, secs))
        kind = None
        if "[checkpoint] completed step" in clean:
            kind = "ckpt"
            stn = int(re.search(r"completed step (\d+)", clean).group(1))
        elif "[puct-jev] step" in clean and "parents;" in clean:
            kind = "jev"
            stn = int(re.search(r"\[puct-jev\] step (\d+)", clean).group(1))
        elif "two-stage coding" in clean or "continuous coder queue" in clean:
            kind = "coding_start"
        elif "vLLM logprobs" in clean:
            kind = "logprobs"
        elif "SPO-RS evaluation/training overlap started" in clean:
            kind = "train_start"
        elif "rollout pilot:" in clean and "programs x" in clean:
            kind = "pilot_start"
        elif "parents picked:" in clean:
            kind = "pick"
        elif "coder format recovery" in clean:
            kind = "gen_done"
        if kind and stn is not None:
            anchors.append((i, secs, kind, stn))
        # keep human-meaningful events
        if tag in ("train-memory", "gpu", "pool", "puct-jev", "checkpoint", "resume", "init", "train-fast",
                   "train-oom", "host-memory", "runtime", "startup", "precision") or st:
            events.append({"i": i, "t": mt.group(0)[1:9], "tag": tag, "s": stn, "txt": clean[11:][:420]})
    if len(events) > max_lines:
        events = events[:max_lines]

    # wall-clock phases per step, best effort (HH:MM:SS has no date: unwrap day rollovers)
    by_step = defaultdict(dict)
    for (i, secs, kind, stn) in anchors:
        by_step[stn][kind] = (i, secs)  # last occurrence wins (resumes repeat anchors)

    def delta(a, b):
        d = b - a
        if d < 0:
            d += 86400
        return d

    phases = {}
    prev_ckpt = None
    first_ts = None
    for i, ln in enumerate(lines[:5]):
        mt = TS_RE.match(ln)
        if mt:
            first_ts = (i, int(mt.group(1)) * 3600 + int(mt.group(2)) * 60 + int(mt.group(3)))
            break
    for stn in sorted(by_step):
        a = by_step[stn]
        start = prev_ckpt if prev_ckpt else first_ts
        jev = a.get("jev") or a.get("pick")
        cs = a.get("coding_start") or a.get("pilot_start")
        gd = a.get("gen_done")
        lp = a.get("logprobs")
        tr = a.get("train_start")
        ck = a.get("ckpt")
        if start and jev:
            # when the run was resumed mid-step, restart the clock at the last resume before the jev line
            rs = [r for r in resume_idx if start[0] < r[0] < jev[0]]
            if rs:
                start = rs[-1]
        ph = {}
        try:
            if start and jev:
                ph["select"] = delta(start[1], jev[1])
            if jev and cs:
                ph["strategist"] = delta(jev[1], cs[1])
            if cs and lp:
                ph["coder"] = delta(cs[1], lp[1])
            if jev and lp and not cs:
                ph["strategist+coder"] = delta(jev[1], (gd or lp)[1])
            if lp and tr:
                ph["scoring"] = delta(lp[1], tr[1])
            elif gd and lp:
                ph["scoring"] = delta(gd[1], lp[1])
            if (tr or lp) and ck:
                ph["training"] = delta((tr or lp)[1], ck[1])
        except Exception:
            pass
        ph = {k: v for k, v in ph.items() if 0 <= v < 20 * 3600}
        if ph:
            phases[stn] = ph
        if ck:
            prev_ckpt = ck
    elapsed = {}
    if first_un is not None:
        try:
            mtime = _dt.datetime.fromtimestamp(os.path.getmtime(path)).isoformat(timespec="minutes")
        except Exception:
            mtime = None
        tail = [ANSI_RE.sub("", x) for x in lines[-60:]]
        stopped = [x for x in tail if "KeyboardInterrupt" in x or "Traceback (most recent call last)" in x or "[train-oom]" in x and "fatal" in x]
        elapsed = {"total": last_un - first_un, "by_step": {str(k): v - first_un for k, v in ckpt_un.items()}, "log_mtime": mtime,
                   "stopped": bool(stopped), "stop_hint": (stopped[0][:140] if stopped else "")}
    return events, phases, elapsed


def _read_head(path, limit=8_000_000):
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            return fh.read(limit)
    except Exception:
        return ""


def parse_hardware(run_dir, config, steps):
    """What machine produced the run: GPU count / memory / capability from the config and the vLLM + trainer logs."""
    from collections import Counter
    c = config or {}
    ids = [x.strip() for x in str(c.get("gpu_ids") or c.get("available_gpu_ids") or "").split(",") if x.strip()]
    n = int(c.get("num_gpus") or len(ids) or 0)
    hw = {"n_gpus": n, "gpu_ids": ids, "profile": c.get("hardware_profile") or "", "profile_defaulted": not c.get("hardware_profile")}
    totals, caps = Counter(), Counter()
    for fn in ("vllm.log", "strategy_vllm.log"):
        txt = _read_head(os.path.join(run_dir, fn))
        for m in re.finditer(r"Free memory on device \(([\d.]+)/([\d.]+) GiB\)", txt):
            totals[float(m.group(2))] += 1
        for m in re.finditer(r"Device capability ([\d.]+)", txt):
            caps[m.group(1)] += 1
    hw["mem_gib"] = totals.most_common(1)[0][0] if totals else None
    hw["capability"] = caps.most_common(1)[0][0] if caps else None
    setting = _read_head(os.path.join(run_dir, "setting.log"), 400000)
    tlog = _read_head(os.path.join(run_dir, "temirnal.log"), 3_000_000)
    m = re.search(r"vLLM engines: (\d+) replica\(s\), TP=(\d+), PP=(\d+), (\d+) GPU\(s\)/engine", setting)
    if m:
        hw["coder_engines"], hw["coder_tp"] = int(m.group(1)), int(m.group(2))
    m = re.search(r"strategy model [^\n]*?: (\d+) vLLM replica\(s\), TP=(\d+)", setting)
    if m:
        hw["strategist_engines"], hw["strategist_tp"] = int(m.group(1)), int(m.group(2))
    m = re.search(r"steady-state total=([\d.]+) GiB/GPU", setting)
    if m:
        hw["assumed_mem_gib"] = float(m.group(1))
    m = re.search(r"(\d+) sandbox process\(es\) across (\d+) CPU\(s\)", tlog)
    if m:
        hw["eval_processes"], hw["eval_cpus_used"] = int(m.group(1)), int(m.group(2))
    else:
        for st in reversed(steps):
            if st.get("evaluation_workers"):
                hw["eval_processes"], hw["eval_cpus_used"] = st["evaluation_workers"], st.get("evaluation_cpu_count")
                break
    m = re.search(r"from a (\d+)-GiB", tlog)
    if m:
        hw["host_offload_gib"] = int(m.group(1))
    hw["training_layout"] = c.get("training_layout")
    hw["backend"] = c.get("generation_backend")
    # naming: the logs never print the card name, so infer it from the hardware profile and the device facts
    prof = (hw["profile"] or "").upper()
    mem = hw["mem_gib"]
    if prof.startswith("B200"):
        hw.update(name="NVIDIA B200", short="B200", family="Blackwell (data-centre)", basis="hardware profile %s; the device reports %s GiB" % (hw["profile"], ("%.0f" % mem) if mem else "?"))
    elif hw["capability"] == "12.0" or prof == "RTX6000" or (hw["profile_defaulted"] and mem and 90 <= mem <= 100):
        hw.update(name="NVIDIA RTX PRO 6000 Blackwell", short="RTX PRO 6000", family="Blackwell (workstation)",
                  basis="inferred - the log only shows compute capability %s and %s GiB per GPU, and this run predates the hardware_profile setting (repo default RTX6000)" % (hw["capability"] or "12.0", ("%.1f" % mem) if mem else "?"))
    else:
        hw.update(name="GPU (model not logged)", short="GPU", family="", basis="the logs do not name the card")
    hw["label"] = "%d × %s%s" % (n, hw["short"], (" · %.0f GiB" % mem) if mem else "") if n else hw["short"]
    return hw


def parse_setting_block(path):
    txt = _read(path)
    if not txt:
        return []
    out = []
    started = False
    for ln in txt.splitlines():
        if "Strategist Bandit" in ln and "=====" not in ln and not started:
            started = True
        if started:
            out.append(ANSI_RE.sub("", ln)[11:] if TS_RE.match(ln) else ANSI_RE.sub("", ln))
            if "Sandbox timeout" in ln:
                break
    return out


def parse_logged_step_results(path):
    """Steps whose rollout phase has finished according to temirnal.log, with the logged best-ever value.
    A step writes stepNN.summary.json / best.json / result.txt only after the adapter update, but the log already
    prints '[step N] best-ever raw ... : X (... found step=F)' when its rollouts are scored."""
    out = {}
    txt = _read(path)
    if not txt:
        return out
    for m in re.finditer(r"\[step (\d+)\] best-ever raw [^:\n]*: ([0-9.eE+\-]+)[^\n]*", txt):
        found = re.search(r"found step=(\d+)", m.group(0))
        out[int(m.group(1))] = {"best_ever": float(m.group(2)), "found": int(found.group(1)) if found else None}
    for m in re.finditer(r"\[step (\d+)\] rollout\+eval", txt):
        out.setdefault(int(m.group(1)), {"best_ever": None, "found": None})
    return out


def parse_result_txt(path):
    rows = []
    for ln in _read(path).splitlines():
        parts = dict(p.split("=", 1) for p in ln.split("\t") if "=" in p)
        if "step" in parts:
            try:
                rows.append({
                    "step": int(parts["step"]),
                    "step_best_raw": float(parts.get("step_best_raw", "nan")),
                    "best_seen_raw": float(parts.get("best_seen_raw", "nan")),
                    "best_seen_reward": float(parts.get("best_seen_reward", "nan")),
                    "found": int(parts.get("best_seen_found_step", "-1")),
                })
            except ValueError:
                pass
    return rows


# ----------------------------------------------------------------------------------------
# main collector
# ----------------------------------------------------------------------------------------
def collect(run_dir, code_per_step=25, all_code=False, think_full=False, progress=print):
    t0 = time.time()
    run_dir = os.path.abspath(run_dir)
    if not os.path.isdir(run_dir):
        raise SystemExit("run directory not found: %s" % run_dir)

    def log(msg):
        if progress:
            progress("[collect %5.1fs] %s" % (time.time() - t0, msg))

    config = _jload(os.path.join(run_dir, "config.json"), {})
    step_dirs = sorted(d for d in os.listdir(run_dir) if re.fullmatch(r"step\d+", d) and
                       os.path.isdir(os.path.join(run_dir, d)))
    steps_idx = [int(d[4:]) for d in step_dirs]
    log("run=%s  steps found: %s" % (os.path.basename(run_dir), steps_idx))

    # ---- parallel per-step scan ------------------------------------------------------
    workers = max(1, min(len(steps_idx), (os.cpu_count() or 4)))
    scans = {}
    with cf.ProcessPoolExecutor(max_workers=workers) as ex:
        for res in ex.map(_scan_step, [(run_dir, s) for s in steps_idx]):
            scans[res["step"]] = res
            log("scanned step %02d: %d files, %d rollouts, %d strategies" %
                (res["step"], res["files"], len(res["rollouts"]), len(res["strategies"])))

    # ---- assemble step records ----------------------------------------------------------
    entropy_rows = {}
    for ln in _read(os.path.join(run_dir, "entropy.jsonl")).splitlines():
        try:
            e = json.loads(ln)
            entropy_rows[e.get("step")] = e
        except Exception:
            pass

    steps = []
    rollouts = []
    strategies = {}
    jev_sources = {}  # node id -> code, from jev.json
    for s in steps_idx:
        sd = os.path.join(run_dir, "step%02d" % s)
        summ = _jload(os.path.join(sd, "step%02d.summary.json" % s))
        parents = _jload(os.path.join(sd, "step%02d.parents.json" % s))
        jev = _jload(os.path.join(sd, "step%02d.jev.json" % s))
        best = _jload(os.path.join(sd, "step%02d.best.json" % s))
        rec = {"step": s, "completed": bool(summ and summ.get("completed")), "partial": summ is None}
        if summ:
            keep = ["next_step", "archive_size", "groups_per_step", "group_size", "planned_rollouts",
                    "coder_format_retries", "coder_format_retries_still_missing", "step_best_raw_score",
                    "step_best_reward", "best_seen_raw_score", "best_seen_reward", "best_seen_step",
                    "valid_fraction", "code_valid_fraction", "evaluation_workers", "evaluation_cpu_count",
                    "spo_rs_divergence", "spo_rs_context_divergences", "training_seconds",
                    "cumulative_training_seconds", "oom_quarantined_examples", "fast_trained_examples_per_gpu",
                    "fast_padded_token_budgets", "fast_peak_memory_fraction", "training_parallel_gpus",
                    "spo_rs_policy_updated", "spo_rs_sampling_adapter", "result_metric_name", "result_maximize"]
            for k in keep:
                if k in summ:
                    rec[k] = summ[k]
            upd = summ.get("spo_rs_updates") or []
            if upd:
                u = upd[0]
                rec["spo"] = {k: u.get(k) for k in ("value_before", "value_after", "effective_count_before",
                                                    "effective_count_after", "group_mean", "group_size",
                                                    "divergence", "rho", "eta", "updates", "initialized",
                                                    "context_divergences", "missing_policy_scores")}
            rec["policy_updates"] = summ.get("policy_updates") or []
            alloc = []
            for a in summ.get("strategy_pilot_allocation") or []:
                alloc.append({k: v for k, v in a.items()})
            if not alloc and scans[s].get("slots"):
                alloc = alloc_from_slots(scans[s]["slots"])   # some runs / steps do not write strategy_pilot_allocation: rebuild it from the rollout metas
            rec["alloc"] = alloc
        if parents:
            rec["sampler_type"] = parents.get("sampler_type")
            rec["parents"] = parents.get("parents", [])
            rec["rollout_config"] = parents.get("rollout_config")
        if jev:
            for nid, code in (jev.get("sources") or {}).items():
                jev_sources.setdefault(nid, code)
            rec["jev"] = {
                "threshold": jev.get("threshold"),
                "requested": jev.get("requested"),
                "candidate_multiplier": jev.get("candidate_multiplier"),
                "candidate_limit": jev.get("candidate_limit"),
                "comparisons": jev.get("comparisons") or [],
                "decisions": jev.get("decisions") or [],
                "picks_info": [
                    {k: v for k, v in p.items() if k != "jev"} for p in (jev.get("picks_info") or [])
                ],
                "selected": jev.get("selected") or [],
                "question": jev.get("question"),
                "seeds": [x.get("id") for x in (jev.get("seeds") or [])],
            }
        if best:
            rec["result_metric_name"] = best.get("metric_name")
            rec["result_maximize"] = bool(best.get("maximize"))
            rec["best"] = {
                "state_id": best.get("state_id"),
                "found_step": best.get("found_step"),
                "raw": _sig(best.get("raw_score"), 12),
                "reward": _sig(best.get("reward"), 12),
                "n": len(best.get("construction") or []),
                "is_seed": best.get("is_seed"),
            }
            rec["_best_full"] = best
        ent = entropy_rows.get(s)
        if ent:
            rec["entropy"] = {
                "all": ent.get("all"),
                "pilot": ent.get("pilot"),
                "by_strategy": ent.get("by_strategy"),
                "by_parent_strategy": ent.get("by_parent_strategy"),
                "code_diversity": ent.get("code_diversity"),
            }
        steps.append(rec)
        rollouts.extend(scans[s]["rollouts"])
        for k, v in scans[s]["strategies"].items():
            strategies["%d:%s" % (s, k)] = v

    # global rollout ids, stable order
    rollouts.sort(key=lambda r: (r["step"], r["file_group"], r["file_rollout"]))
    for i, r in enumerate(rollouts):
        r["id"] = i
    node_to_rid = {r["node"]: r["id"] for r in rollouts if r.get("node")}

    # ---- search tree -----------------------------------------------------------------
    tree = _jload(os.path.join(run_dir, "search_tree.json"), {}) or {}
    nodes_raw = tree.get("nodes", {}) or {}
    root_ids = tree.get("root_ids", [])
    log("search tree: %d nodes, %d roots" % (len(nodes_raw), len(root_ids)))
    node_ids = list(nodes_raw.keys())
    nidx = {nid: i for i, nid in enumerate(node_ids)}
    tnodes = []  # [parent_idx, raw, step, rollout_id]
    for nid in node_ids:
        n = nodes_raw[nid]
        pidx = nidx.get(n.get("parent_id"), -1) if n.get("parent_id") else -1
        rid = node_to_rid.get(nid, -1)
        st = rollouts[rid]["step"] if rid >= 0 else -1
        tnodes.append([pidx, _sig(n.get("raw_problem_score"), 9), st, rid])

    # ---- which programs get their code/construction embedded ---------------------------
    want = set()  # rollout ids
    selected_nodes = set()
    for s in steps:
        for p in s.get("parents", []):
            selected_nodes.add(p["parent_id"])
        b = s.get("best")
        if b and b.get("state_id"):
            selected_nodes.add(b["state_id"])
    for nid in selected_nodes:
        if nid in node_to_rid:
            want.add(node_to_rid[nid])
    by_step_rollouts = defaultdict(list)
    for r in rollouts:
        by_step_rollouts[r["step"]].append(r)
    # a step whose rollouts are all scored but whose summary.json is not written yet (the adapter update is still running):
    # its programs are real, so its best counts for the best-so-far curve ("provisional")
    logged = parse_logged_step_results(os.path.join(run_dir, "temirnal.log"))
    maximize_run = any(bool(x.get("result_maximize")) for x in steps)
    prev_best = None
    for rec in steps:
        if rec.get("completed"):
            if rec.get("best_seen_raw_score") is not None:
                prev_best = (rec["best_seen_raw_score"], rec.get("best_seen_reward"), rec.get("best_seen_step"))
            continue
        lg = logged.get(rec["step"])
        snapshot = rec.get("best") or {}
        has_snapshot = snapshot.get("raw") is not None
        if not rec.get("partial") or (lg is None and not has_snapshot):
            continue
        rs = by_step_rollouts.get(rec["step"], [])
        good = [r for r in rs if r["valid"] and r["raw"] is not None]
        if not good and not has_snapshot:
            continue
        rec["provisional"] = True
        rec["rollouts_done"] = True
        if good:
            top = max(good, key=lambda r: r["raw"]) if maximize_run else min(good, key=lambda r: r["raw"])
            rec["step_best_raw_score"] = _sig(top["raw"], 12)
            rec["step_best_reward"] = _sig(top["reward"], 12)
            better = prev_best is None or (top["raw"] > prev_best[0] if maximize_run else top["raw"] < prev_best[0])
            prev_best = (rec["step_best_raw_score"], rec["step_best_reward"], rec["step"]) if better else prev_best
        if has_snapshot:
            prev_best = (snapshot["raw"], snapshot.get("reward"), snapshot.get("found_step"))
        rec["valid_fraction"] = len(good) / max(1, len(rs))
        rec["best_seen_raw_score"], rec["best_seen_reward"], rec["best_seen_step"] = prev_best
        rec["log_best_ever"] = (lg or {}).get("best_ever")
    for st, rs in by_step_rollouts.items():
        good = [r for r in rs if r["valid"] and r["raw"]]
        good.sort(key=lambda r: r["raw"])  # lower is better for AC1; harmless if maximise (we embed both ends)
        pool = good if all_code else good[:code_per_step]
        for r in pool:
            want.add(r["id"])
        if not all_code:
            for r in good[-3:]:
                want.add(r["id"])

    log("embedding code for %d programs ..." % len(want))
    code_blob = {}
    cons_blob = {}

    def load_code(rid):
        r = rollouts[rid]
        sd = os.path.join(run_dir, "step%02d" % r["step"])
        base = "step%02d_group%02d_rollout%03d" % (r["step"], r["file_group"], r["file_rollout"])
        txt = _read(os.path.join(sd, base + ".txt"))
        think, ans = split_think(txt)
        code = extract_code(ans or txt)
        meta = _jload(os.path.join(sd, base + ".meta.json"), {})
        cons = parse_construction(meta.get("construction") or "")
        return rid, code, cons, len(think), len(ans)

    with cf.ThreadPoolExecutor(max_workers=8) as ex:
        for rid, code, cons, tl, al in ex.map(load_code, sorted(want)):
            code_blob[rid] = code
            if cons:
                cons_blob[rid] = [_sig(x, 7) for x in cons]
            rollouts[rid]["think_chars"] = tl

    # parents that are not rollouts (seeds): code from jev sources
    seed_code = {}
    for nid in selected_nodes:
        if nid not in node_to_rid and nid in jev_sources:
            seed_code[nid] = jev_sources[nid]
    # also seeds' construction not available -> skip

    # best-so-far full snapshots (code + construction): from stepNN.best.json when it exists, otherwise
    # reconstructed from the rollouts (some runs only write best.json for a few steps)
    best_snaps = []
    run_best = None  # rollout record of the best valid program seen so far
    maximize = any(bool(s.get("result_maximize")) for s in steps)
    for s in steps:
        if not (s.get("completed") or s.get("provisional")):
            s.pop("_best_full", None)
            continue
        for r in by_step_rollouts.get(s["step"], []):
            if r["valid"] and r["raw"]:
                if run_best is None or (r["raw"] > run_best["raw"] if maximize else r["raw"] < run_best["raw"]):
                    run_best = r
        b = s.pop("_best_full", None)
        if b and b.get("construction"):
            best_snaps.append({
                "step": s["step"], "state_id": b.get("state_id"), "found_step": b.get("found_step"),
                "raw": _sig(b.get("raw_score"), 12), "reward": _sig(b.get("reward"), 12),
                "code": b.get("code") or "", "construction": [_sig(x, 9) for x in (b.get("construction") or [])],
            })
        elif run_best is not None:
            rid = run_best["id"]
            meta_b = _jload(os.path.join(run_dir, "step%02d" % run_best["step"],
                                         "step%02d_group%02d_rollout%03d.meta.json" % (run_best["step"], run_best["file_group"], run_best["file_rollout"])), {})
            cons9 = parse_construction(meta_b.get("construction") or "")
            best_snaps.append({
                "step": s["step"], "state_id": run_best.get("node"), "found_step": run_best["step"],
                "raw": _sig(run_best["raw"], 12), "reward": _sig(run_best["reward"], 12),
                "code": code_blob.get(rid, ""), "construction": [_sig(x, 9) for x in cons9],
            })
        if best_snaps and best_snaps[-1]["step"] == s["step"] and s.get("best") is None:
            s["best"] = {"state_id": best_snaps[-1]["state_id"], "found_step": best_snaps[-1]["found_step"], "raw": best_snaps[-1]["raw"],
                         "reward": best_snaps[-1]["reward"], "n": len(best_snaps[-1]["construction"]), "is_seed": False}

    # slim rollouts into columnar form
    cols = ["id", "step", "group", "pgroup", "rollout", "sidx", "phase", "node", "parent", "raw", "reward",
            "valid", "parsed", "ran", "fail", "msg", "adv", "ntok", "n", "retry", "counts", "pval", "praw",
            "fallback", "arch", "think_chars"]
    rows = []
    for r in rollouts:
        rows.append([r.get(c) for c in cols])
    stdout_map = {r["id"]: r["stdout"] for r in rollouts if r["id"] in want and r.get("stdout")}

    # ---- logs ------------------------------------------------------------------------
    events, phases, elapsed = parse_terminal_log(os.path.join(run_dir, "temirnal.log"))
    log("terminal log: %d events kept, phases for %d steps" % (len(events), len(phases)))
    setting = parse_setting_block(os.path.join(run_dir, "setting.log"))
    result_rows = parse_result_txt(os.path.join(run_dir, "result.txt"))
    runtime = _jload(os.path.join(run_dir, "runtime.json"), {})
    hardware = parse_hardware(run_dir, config, steps)

    meta = {
        "run_name": os.path.basename(run_dir),
        "run_dir": run_dir,
        "generated_at": _dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "steps_found": steps_idx,
        "steps_completed": [s["step"] for s in steps if s.get("completed")],
        "steps_provisional": [s["step"] for s in steps if s.get("provisional")],
        "tree_updated_through": tree.get("updated_through_step"),
        "n_rollouts": len(rollouts),
        "n_nodes": len(node_ids),
        "code_per_step": code_per_step, "all_code": all_code,
    }
    main = {
        "meta": meta,
        "config": config,
        "runtime": runtime,
        "setting": setting,
        "result": result_rows,
        "steps": steps,
        "rcols": cols,
        "rows": rows,
        "tree": {"ids": node_ids, "nodes": tnodes, "roots": [nidx[r] for r in root_ids if r in nidx]},
        "phases": phases,
        "elapsed": elapsed,
        "hardware": hardware,
        "best_snaps_meta": [{k: v for k, v in b.items() if k not in ("code", "construction")} for b in best_snaps],
    }
    heavy = {
        "code": {str(k): v for k, v in code_blob.items()},
        "cons": {str(k): v for k, v in cons_blob.items()},
        "stdout": {str(k): v for k, v in stdout_map.items()},
        "seed_code": seed_code,
        "best": best_snaps,
    }
    strat_blob = {"strategies": strategies}
    log_blob = {"events": events}
    log("done")
    return main, heavy, strat_blob, log_blob


if __name__ == "__main__":
    m, h, s, l = collect(sys.argv[1])
    print({k: len(json.dumps(v)) for k, v in dict(main=m, heavy=h, strat=s, log=l).items()})
