#!/usr/bin/env python3
"""
build_explorer.py - turn training-run directories into self-contained, interactive HTML files.

    python3 build_explorer.py RUN_DIR [-o out.html] [--code-per-step 25] [--all-code] [--open]
    python3 build_explorer.py RUN_A RUN_B ... --unified [-o out/all_problems.html]     # one HTML with several problems

* Read-only on the run directories and on the rest of the code base (writes only to ./out by default).
* Standard library only. The produced HTML has no external dependencies (works offline, file://).
* Data is embedded as gzip+base64 blobs and inflated lazily in the browser (DecompressionStream).
* --unified builds one page that holds a landing view (all problems side by side, with GPU specification, results
  vs published records) and the full explorer of every run (each in its own isolated frame).
"""
from __future__ import annotations

import argparse
import base64
import gzip
import json
import os
import sys
import webbrowser

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from collect import collect  # noqa: E402

JS_ORDER = [
    "core.js", "charts.js", "problems.js", "knowledge.js",
    "tab_overview.js", "tab_pipeline.js", "tab_search.js", "tab_strategist.js",
    "tab_rollouts.js", "tab_alloc.js", "tab_train.js", "tab_entropy.js",
    "tab_solution.js", "tab_published.js", "tab_concepts.js", "tab_logs.js", "main.js",
]
PROBLEM_TITLES = {"ac1": "First autocorrelation inequality (AC1)", "erdos": "Erdős minimum-overlap problem (C₅)"}
PROBLEM_SHORT = {"ac1": "AC1", "erdos": "Erdős C₅"}


def pack(obj) -> str:
    raw = json.dumps(obj, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
    return base64.b64encode(gzip.compress(raw, 9, mtime=0)).decode("ascii")


def pack_text(txt: str) -> str:
    return base64.b64encode(gzip.compress(txt.encode("utf-8"), 9, mtime=0)).decode("ascii")


def read(p):
    with open(p, "r", encoding="utf-8") as fh:
        return fh.read()


def summarize(main, pub):
    """Small, JSON-friendly digest of a run for the landing page of the unified HTML."""
    cfg = main.get("config") or {}
    steps = main["steps"]
    done = [s for s in steps if s.get("completed")]
    seen = [s for s in steps if s.get("completed") or s.get("provisional")]    # provisional = rollouts scored, adapter update still running
    curve = [[s["step"], s.get("best_seen_raw_score")] for s in seen if s.get("best_seen_raw_score") is not None]
    pend = [s for s in steps if not s.get("completed")]
    pending = None
    if pend:
        p = pend[-1]
        n_p = sum(1 for r in main["rows"] if r[main["rcols"].index("step")] == p["step"])
        pending = {"step": p["step"], "rollouts": n_p, "rollouts_done": bool(p.get("provisional"))}
    best = curve[-1][1] if curve else None
    cols = main["rcols"]
    iv, ir = cols.index("valid"), cols.index("raw")
    rows = main["rows"]
    n_valid = sum(1 for r in rows if r[iv])
    pid = str(cfg.get("problem") or "").lower()
    ref = None
    rank = None
    firm = []
    if pub and pub.get("results"):
        by_id = {r["id"]: r for r in pub["results"]}
        dr = by_id.get(pub.get("default_ref"))
        if dr:
            ref = {"label": dr["label"], "value": dr["reported"], "id": dr["id"]}
        firm = [r["reported"] for r in pub["results"] if r.get("kind") != "claim" and r.get("reported") is not None]
        if best is not None:
            rank = 1 + sum(1 for v in firm if v < best)
    el = main.get("elapsed") or {}
    hw = main.get("hardware") or {}
    metric = next((s.get("result_metric_name") for s in steps if s.get("result_metric_name")), "score")
    nm = lambda p: str(p or "").rstrip("/").split("/")[-1]
    return {
        "problem": pid, "title": PROBLEM_TITLES.get(pid, pid.upper() or "problem"), "short": PROBLEM_SHORT.get(pid, pid.upper() or "problem"),
        "metric": metric, "lower_is_better": True,
        "run_name": main["meta"]["run_name"], "generated_at": main["meta"]["generated_at"],
        "coder": nm(cfg.get("coder_model_name") or cfg.get("model_name")), "strategist": nm(cfg.get("strategy_model_name")),
        "steps_done": len(done), "steps_total": cfg.get("num_steps"), "rollouts": len(rows), "valid_fraction": (n_valid / len(rows)) if rows else None,
        "best": best, "curve": curve, "pending": pending, "reference": ref, "rank": rank, "n_ranked": len(firm) + 1, "target": cfg.get("target"),
        "stopped": bool(el.get("stopped")), "elapsed_s": el.get("total"), "hardware": hw,
        "gpu_hours": (el.get("total") / 3600 * hw.get("n_gpus", 0)) if el.get("total") is not None and hw.get("n_gpus") else None,
        "selection": "PUCT + JEV" if any(s.get("jev") for s in steps) else "PUCT",
        "dec": {"ac1": 6, "erdos": 8}.get(pid, 6),
    }


def render_run(run_dir, code_per_step=25, all_code=False):
    """Returns (html, summary, blob_sizes) of the single-run explorer."""
    main, heavy, strat, logs = collect(run_dir, code_per_step=code_per_step, all_code=all_code)
    tdir = os.path.join(HERE, "template")
    html = read(os.path.join(tdir, "index.html"))
    css = read(os.path.join(tdir, "style.css"))
    js = "\n".join("/* ---- %s ---- */\n%s" % (n, read(os.path.join(tdir, "js", n))) for n in JS_ORDER
                   if os.path.exists(os.path.join(tdir, "js", n)))
    problem = str((main.get("config") or {}).get("problem") or "").lower()
    pub_path = os.path.join(HERE, "data", "published_%s.json" % problem)
    pub = json.load(open(pub_path, encoding="utf-8")) if os.path.exists(pub_path) else {}
    if not pub:
        print("note: no data/published_%s.json - the explorer will work without published reference values" % (problem or "?"))
    blobs = {"main": pack(main), "heavy": pack(heavy), "strat": pack(strat), "logs": pack(logs), "pub": pack(pub)}
    data_tags = "\n".join('<script type="application/octet-stream" id="blob-%s">%s</script>' % (k, v) for k, v in blobs.items())
    html = (html.replace("/*__CSS__*/", css)
                .replace("/*__JS__*/", js.replace("</script>", "<\\/script>"))
                .replace("<!--__DATA__-->", data_tags)
                .replace("__TITLE__", "Run Explorer - " + main["meta"]["run_name"]))
    return html, summarize(main, pub), {k: len(v) for k, v in blobs.items()}


def build(run_dir, out_path, code_per_step=25, all_code=False):
    html, _, sizes = render_run(run_dir, code_per_step, all_code)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(html)
    print("wrote %s  (%.1f MB; blobs: %s)" % (out_path, os.path.getsize(out_path) / 1e6,
                                              ", ".join("%s=%.1fMB" % (k, v / 1e6) for k, v in sizes.items())))
    return out_path


def build_unified(run_dirs, out_path, code_per_step=25, all_code=False):
    """One HTML holding a landing page and every run's full explorer (each run in an isolated iframe)."""
    tdir = os.path.join(HERE, "template")
    runs, tags = [], []
    for i, rd in enumerate(run_dirs):
        html, summ, _ = render_run(rd, code_per_step, all_code)
        rid = "run%d" % i
        summ["id"] = rid
        runs.append(summ)
        tags.append('<script type="application/octet-stream" id="html-%s">%s</script>' % (rid, pack_text(html)))
        print("  embedded %s (%s, %.1f MB html)" % (summ["run_name"], summ["title"], len(html) / 1e6))
    page = read(os.path.join(tdir, "hub.html"))
    css = read(os.path.join(tdir, "style.css")) + "\n" + read(os.path.join(tdir, "hub.css"))
    js = read(os.path.join(tdir, "hub.js"))
    page = (page.replace("/*__CSS__*/", css).replace("/*__JS__*/", js.replace("</script>", "<\\/script>"))
                .replace("<!--__HUBDATA__-->", '<script type="application/json" id="hub-data">%s</script>' % json.dumps(runs, ensure_ascii=False, allow_nan=False).replace("</", "<\\/"))
                .replace("<!--__RUNS__-->", "\n".join(tags)))
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(page)
    print("wrote %s  (%.1f MB, %d runs)" % (out_path, os.path.getsize(out_path) / 1e6, len(runs)))
    return out_path


def default_unified_runs():
    """Runs for `--unified` without arguments: the list in unified_runs.txt, else the newest run of every problem."""
    lst = os.path.join(HERE, "unified_runs.txt")
    if os.path.isfile(lst):
        out = []
        for line in read(lst).splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            p = line if os.path.isabs(line) else os.path.normpath(os.path.join(HERE, line))
            if os.path.isdir(p):
                out.append(p)
            else:
                print("warning: unified_runs.txt lists a missing directory, skipped: %s" % p)
        if out:
            return out
    runs = os.path.join(os.path.dirname(HERE), "runs")
    best = {}
    for d in (os.listdir(runs) if os.path.isdir(runs) else []):
        p = os.path.join(runs, d)
        cfgp = os.path.join(p, "config.json")
        if " " in d or not os.path.isfile(cfgp):
            continue
        try:
            prob = str(json.load(open(cfgp)).get("problem") or "").lower()
        except Exception:
            continue
        if prob and (prob not in best or os.path.getmtime(p) > os.path.getmtime(best[prob])):
            best[prob] = p
    return [best[k] for k in sorted(best)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", nargs="*", help="run directory (contains config.json, stepNN/, search_tree.json ...). "
                    "Omit to use the most recently modified run under ../runs (or, with --unified, the runs listed in unified_runs.txt)")
    ap.add_argument("-o", "--out", help="output HTML path (default: out/<run name>.html, or out/all_problems.html with --unified)")
    ap.add_argument("--unified", action="store_true", help="build ONE html that contains all the given runs (different problems) with a landing page")
    ap.add_argument("--code-per-step", type=int, default=25, help="embed program source for the N best valid rollouts of each step (default 25)")
    ap.add_argument("--all-code", action="store_true", help="embed source for every valid rollout (large file)")
    ap.add_argument("--open", action="store_true", help="open the result in the default browser")
    a = ap.parse_args()
    if a.unified:
        runs = a.run_dir or default_unified_runs()
        evaluated_run = os.environ.get("TTT_EVALUATED_RUN")
        if (evaluated_run and os.path.isdir(evaluated_run)
                and not any(os.path.realpath(r) == os.path.realpath(evaluated_run) for r in runs)):
            runs.append(evaluated_run)
        if not runs:
            ap.error("--unified needs run directories (none given, none in unified_runs.txt, none found under ../runs)")
        print("unified build of: " + ", ".join(os.path.basename(os.path.abspath(r)) for r in runs))
        out = a.out or os.path.join(HERE, "out", "all_problems.html")
        build_unified([os.path.abspath(r) for r in runs], out, a.code_per_step, a.all_code)
    else:
        if len(a.run_dir) > 1:
            ap.error("several run directories need --unified (or run the command once per run)")
        rd = a.run_dir[0] if a.run_dir else None
        if not rd:
            runs = os.path.join(os.path.dirname(HERE), "runs")
            cands = [os.path.join(runs, d) for d in os.listdir(runs)] if os.path.isdir(runs) else []
            cands = [c for c in cands if os.path.isfile(os.path.join(c, "config.json"))]
            if not cands:
                ap.error("no run_dir given and no run found under %s" % runs)
            rd = max(cands, key=os.path.getmtime)
            print("no run_dir given - using the newest run: %s" % rd)
        out = a.out or os.path.join(HERE, "out", os.path.basename(os.path.abspath(rd)) + ".html")
        build(rd, out, a.code_per_step, a.all_code)
    if a.open:
        webbrowser.open("file://" + os.path.abspath(out))


if __name__ == "__main__":
    main()
