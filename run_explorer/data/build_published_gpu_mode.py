#!/usr/bin/env python3
"""
build_published_gpu_mode.py - (re)create data/published_gpu_mode.json: the reference results for the GPU Mode TriMul kernel task on B200.

    python3 build_published_gpu_mode.py                  # fetch the live leaderboard (needs internet)
    python3 build_published_gpu_mode.py --snapshot f.json  # use a saved copy of https://www.gpumode.com/api/leaderboard/496

Sources
  paper        arXiv:2601.16175 "Learning to Discover at Test Time" (TTT-Discover), Table 4 - quoted from the PDF (v2).
               B200 was NOT submitted to the official board: the authors ran 10 trials per kernel and report mean + 95% CI.
  leaderboard  the public GPU MODE leaderboard 496 (trimul), B200 ranking, fetched from the site's JSON endpoint.
               Scores there are seconds; they are converted to microseconds here.
The explorer never touches the network: build_explorer.py only embeds the JSON file this script writes.
Standard library only.
"""
import argparse
import datetime
import json
import os
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
API = "https://www.gpumode.com/api/leaderboard/496"
BOARD = "https://www.gpumode.com/leaderboard/496?tab=rankings"
ARXIV = "https://arxiv.org/abs/2601.16175"
PROJECT = "https://test-time-training.github.io/discover"
REPO = "https://github.com/test-time-training/discover"

# Table 4 of arXiv:2601.16175 (v2), TriMul runtime in microseconds, lower is better.
# (method, model, A100, H100, B200, B200 95% CI, MI300X, MI300X 95% CI)
PAPER_TABLE4 = [
    ("1st human", "-", 4531.5, 1371.1, 1038.9, "[1016.3, 1061.6]", 2515.8, "[2510.9, 2520.7]"),
    ("2nd human", "-", 4918.5, 2368.0, 2362.4, "[2335.7, 2389.1]", 5165.0, "[5163.0, 5167.0]"),
    ("3rd human", "-", 5182.2, 2545.7, 1931.0, "[1910.9, 1951.1]", 5359.3, "[5343.5, 5375.1]"),
    ("4th human", "-", 6097.8, 3654.8, 2248.9, "[2089.4, 2408.4]", 5981.4, "[5978.4, 5984.4]"),
    ("5th human", "-", 8345.0, 4233.1, 6503.8, "[6400.5, 6607.1]", 8365.1, "[8347.7, 8382.5]"),
    ("Best-of-25600", "gpt-oss-120b", 9219.7, 5390.3, 3254.9, "[3252.5, 3257.4]", 4902.0, "[4897.6, 4906.4]"),
    ("TTT-Discover", "gpt-oss-120b", 2198.2, 1161.2, 914.2, "[907.3, 921.1]", 1555.7, "[1550.8, 1560.6]"),
]


def load(snapshot):
    if snapshot:
        with open(snapshot, encoding="utf-8") as fh:
            return json.load(fh)["data"]
    req = urllib.request.Request(API, headers={"User-Agent": "Mozilla/5.0 (run-explorer data refresh)"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode("utf-8"))["data"]


def build(data, today):
    rows = data["rankings"]["B200"]
    us = lambda r: r["score"] * 1e6
    top = rows[0]
    results = [
        dict(id="ttt", label="TTT-Discover (gpt-oss-120b)", method="TTT-Discover", model="gpt-oss-120b", date="2026-01", reported=914.2, kind="reported",
             ci=[907.3, 921.1], url=ARXIV, hardware="B200",
             note="Table 4: mean of 10 trials on B200 (95% CI 907.3-921.1). Trained on H100 runtime only and evaluated on B200 afterwards; not an official leaderboard entry."),
        dict(id="human1", label="1st human (as printed in the paper)", method="best human", model="", date="", reported=1038.9, kind="reported",
             ci=[1016.3, 1061.6], url=ARXIV, hardware="B200",
             note="Table 4: the best human kernel of the leaderboard when the paper was written, re-timed on B200 (10 trials)."),
        dict(id="bo25600", label="Best-of-25600 (gpt-oss-120b)", method="best-of-N", model="gpt-oss-120b", date="2026-01", reported=3254.9, kind="reported",
             ci=[3252.5, 3257.4], url=ARXIV, hardware="B200",
             note="Table 4: the same model sampled 25,600 times without learning; shows what test-time training adds."),
    ]
    for r in rows[:5]:
        results.append(dict(
            id="gm%d" % r["rank"], label="GPU Mode #%d %s" % (r["rank"], r["user_name"]), method="GPU Mode leaderboard", model="human / unknown",
            date=r["submission_time"][:10], reported=round(us(r), 3), kind="reported", url=BOARD, hardware="B200",
            note="official B200 board, rank %d; %d lines, %d submissions by this user; fetched %s" % (r["rank"], r["line_count"], r["submission_count"], today)))
    for r in results:
        r["tier"] = "leaderboard" if r["id"].startswith("gm") else "paper"

    cols4 = [{"label": "method"}, {"label": "model"},
             {"label": "A100", "num": True, "best": "min", "digits": 1}, {"label": "H100", "num": True, "best": "min", "digits": 1},
             {"label": "B200", "num": True, "best": "min", "digits": 1}, {"label": "B200 95% CI"},
             {"label": "MI300X", "num": True, "best": "min", "digits": 1}, {"label": "MI300X 95% CI"}]
    rows4 = [[m, mo, a, h, b, bci, mi, mci] for (m, mo, a, h, b, bci, mi, mci) in PAPER_TABLE4]
    colsb = [{"label": "rank", "num": True}, {"label": "user"}, {"label": "runtime (µs)", "num": True, "best": "min", "digits": 3},
             {"label": "submitted"}, {"label": "lines", "num": True}, {"label": "submissions", "num": True}]
    rowsb = [[r["rank"], r["user_name"], round(us(r), 3), r["submission_time"][:10], r["line_count"], r["submission_count"]] for r in rows]
    h100 = data["rankings"]["H100"]
    ttt_h100 = next((r for r in h100 if r["user_name"] == "TTT"), None)
    tables = [
        dict(title="TTT-Discover paper, Table 4 - TriMul runtime (µs, lower is better)",
             subtitle="arXiv:2601.16175 v2; the model was trained with H100 runtime as its reward and the single best kernel was then timed on every GPU",
             columns=cols4, rows=rows4, tier="paper",
             note="A100 and H100 were submitted to the official leaderboard; B200 and MI300X could not be (an infrastructure problem on GPU Mode's server), so the authors report the mean of 10 trials with 95% confidence intervals measured with the same harness."),
        dict(title="GPU Mode leaderboard 496 (trimul) - B200 ranking, live snapshot",
             subtitle="https://www.gpumode.com/leaderboard/496 - geometric mean over %d benchmark shapes; competition deadline %s" % (len(data["benchmarks"]), data["deadline"][:10]),
             columns=colsb, rows=rowsb, tier="leaderboard",
             note="Fetched %s from the site's JSON endpoint. The task's score is in seconds there; shown here in microseconds. The run's prompt target (553.982 µs) is exactly rank 1." % today),
    ]
    events = [
        ["2025-06", "GPU MODE's TriMul task opens (triangular multiplicative update from AlphaFold 3). Every GPU type has its own leaderboard: A100, H100, B200, MI300X; the score is the geometric mean runtime over %d input shapes." % len(data["benchmarks"]), "paper"],
        ["2026-01", "TTT-Discover (arXiv:2601.16175) reports 914.2 µs on B200 against 1038.9 µs for the best human kernel. It trained on H100 runtime only and could not submit B200 results to the board.", "paper"],
        ["2026-01", "The 'TTT' account submits an H100 kernel to the official board (paper: 1161.2 µs; today's board shows %s)." % ("%.1f µs, rank %d" % (us(ttt_h100), ttt_h100["rank"]) if ttt_h100 else "n/a"), "leaderboard"],
        ["2026-02", "K-Search (arXiv:2602.19128) reaches 1030 µs on H100 with GPT-5.2 + Gemini-3-Pro (300 iterations). It reports no TriMul number on B200.", "paper"],
        ["2026-05", "The competition closes on %s. The B200 board ends with %s at %.3f µs (rank 1) and %s at %.3f µs (rank 2)." % (data["deadline"][:10], top["user_name"], us(top), rows[1]["user_name"], us(rows[1])), "leaderboard"],
    ]
    links = [
        ["TTT-Discover paper (arXiv:2601.16175), Table 4", ARXIV, "paper"],
        ["TTT-Discover project page", PROJECT, "paper"],
        ["TTT-Discover code and discovered kernels", REPO, "paper"],
        ["K-Search (arXiv:2602.19128): TriMul on H100, 1030 µs", "https://arxiv.org/abs/2602.19128", "paper"],
        ["GPU MODE leaderboard 496 (trimul), rankings", BOARD, "leaderboard"],
    ]
    paper_only = dict(
        lede_paper="Where this run stands against the <b>published</b> TriMul results on <b>B200</b> (runtime in microseconds, lower is better). The reference is Table 4 of the TTT-Discover paper (<a href=\"%s\" target=\"_blank\" rel=\"noopener\">arXiv:2601.16175</a>): its kernel runs in 914.2 µs on B200 against 1038.9 µs for the best human kernel. The live GPU Mode leaderboard is hidden - use <b>Results</b> in the top bar to show it." % ARXIV,
        league_note_paper="(published works only: the TTT-Discover paper, Table 4)",
        league_foot_paper="Values are quoted as printed (lower is better). The paper trained on H100 runtime only and timed the kernel on B200 afterwards (10 trials, not an official leaderboard entry); this run trains on B200 directly. The live GPU Mode board (rank 1: 553.982 µs) is hidden - top bar: <b>Results</b>.",
        caution_paper="Only published works are shown. The paper's B200 numbers are means of 10 trials measured by the authors with the GPU Mode harness, not official leaderboard entries.")
    out = dict(
        problem="gpu_mode", fetched=today, short="TriMul B200", long="GPU Mode TriMul kernel (B200)", default_ref="ttt",
        paper=dict(id="arxiv:2601.16175", title="Learning to Discover at Test Time", url=ARXIV, table="Table 4"),
        lede="Where this run stands against the TriMul results on <b>B200</b> (runtime in microseconds, lower is better): the TTT-Discover paper (Table 4) and the live GPU MODE leaderboard (B200 ranking; rank 1 is %.3f µs, which is also this run's prompt target)." % us(top),
        league_note="(the TTT-Discover paper and the live GPU MODE B200 ranking)",
        league_foot="Values are quoted as printed (lower is better). Leaderboard scores are the geometric mean over %d shapes, fetched %s." % (len(data["benchmarks"]), today),
        footnote="All numbers are the geometric-mean runtime in microseconds over the task's %d benchmark shapes (batch 1-2, sequence length 256-1024, dim 128 or 384, hidden dim 128; normal and Cauchy inputs, with and without mask). A kernel that fails the correctness check scores nothing." % len(data["benchmarks"]),
        caution="B200 entries marked 'reported' cannot be recomputed here - a kernel's runtime depends on the GPU, so only the published figure is shown.",
        results=results, tables=tables, events=events, links=links)
    out.update(paper_only)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--snapshot", help="use this saved JSON instead of fetching the leaderboard")
    ap.add_argument("--date", default=datetime.date.today().isoformat(), help="fetch date written into the file")
    a = ap.parse_args()
    out = build(load(a.snapshot), a.date)
    path = os.path.join(HERE, "published_gpu_mode.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, separators=(",", ":"), ensure_ascii=False)
    print("wrote %s (%.1f kB), %d results" % (path, os.path.getsize(path) / 1e3, len(out["results"])))


if __name__ == "__main__":
    main()
