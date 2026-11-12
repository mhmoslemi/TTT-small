#!/usr/bin/env python3
"""
build_published.py - (re)create the reference data the explorer ships with, one file per problem:

    data/published_ac1.json    first autocorrelation inequality (AC1)
    data/published_erdos.json  Erdos minimum-overlap problem

Each file holds the step-function constructions published by other methods (downloaded from pinned git commits,
with SHA-256), results that are only reported (no construction downloaded), tables, links and a short history.

Run it only when you want to refresh the data (needs internet). The explorer itself never touches the network:
`build_explorer.py` just embeds the JSON file that matches the run's problem. Standard library only.
"""
import hashlib
import json
import os
import re
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = "https://raw.githubusercontent.com/%s/%s/%s"
BLOB = "https://github.com/%s/blob/%s/%s"
TAO_1A = "https://github.com/teorth/optimizationproblems/blob/main/constants/1a.md"
TAO_1B = "https://github.com/teorth/optimizationproblems/blob/main/constants/1b.md"
TAO_ISSUE = "https://github.com/teorth/optimizationproblems/issues/348"


def fetch(repo, sha, path):
    url = RAW % (repo, sha, path)
    with urllib.request.urlopen(url, timeout=120) as r:
        raw = r.read()
    return raw, BLOB % (repo, sha, path), hashlib.sha256(raw).hexdigest()


def r10(vals):
    return [float("%.10g" % float(x)) for x in vals]


def numbers_after(text, marker):
    i = text.index(marker)
    j = text.index("[", i)
    k = text.index("]", j)
    return [float(x) for x in re.findall(r"-?\d+\.?\d*(?:[eE][-+]?\d+)?", text[j + 1:k])]



# ----------------------------------------------------------------------------------------------------------------
# Tiers: what counts as a "published work"?  (the explorer's top-bar key "Results" shows one tier or more)
#   paper       peer-reviewed papers / arXiv preprints with a public result (incl. the human record)
#   leaderboard Einstein Arena / community-curated list entries that are replayed but not a paper
#   claim       unreviewed claims (preprint listings, quoted numbers) - never ranked
PAPER_ONLY_TEXT = {
    "ac1": dict(
        lede_paper="Where this run stands against the <b>published</b> results on the same problem - the <b>first autocorrelation inequality</b> (lower is better). The reference table is Supplementary Table 17 of <a href=\"https://arxiv.org/abs/2604.19341\" target=\"_blank\" rel=\"noopener\">arXiv:2604.19341</a> (SimpleTES) plus the Einstein Arena paper (<a href=\"https://arxiv.org/abs/2606.10402\" target=\"_blank\" rel=\"noopener\">arXiv:2606.10402</a>). Community-leaderboard entries and unreviewed claims are hidden - use <b>Results</b> in the top bar to show them. For every method whose height sequence is public, the bound is <b>recomputed here</b> from the sequence itself.",
        league_note_paper="(published works only: the table of <a href=\"https://arxiv.org/abs/2604.19341\" target=\"_blank\" rel=\"noopener\">arXiv:2604.19341</a> and the Einstein Arena paper)",
        league_foot_paper="Values are quoted as printed (lower is better). The paper's own method (SimpleTES, gpt-oss-120b) reaches 1.503871 on AC1 and does not claim the AC1 record - the record in its table is the Together AI entry. Newer leaderboard entries and unreviewed claims are hidden (top bar: <b>Results</b>). Change the <b>Reference</b> selector to re-base every gap in this page.",
        caution_paper="Only published works are shown (papers and preprints with a public result). Entries marked 'reported' are taken from the paper as printed.",
    ),
    "erdos": dict(
        lede_paper="Where this run stands against the <b>published</b> results on <b>Erdős' minimum-overlap problem</b> - the constant <b>C₅</b> (lower is better). Published constructions come from Haugland, AlphaEvolve, TTT-Discover, Together AI and SimpleTES (<a href=\"https://arxiv.org/abs/2604.19341\" target=\"_blank\" rel=\"noopener\">arXiv:2604.19341</a>). Newer community-leaderboard entries are hidden - use <b>Results</b> in the top bar to show them. For every method whose step function is public, the bound is <b>recomputed here</b> from the function itself.",
        league_note_paper="(published works only: Haugland, AlphaEvolve, TTT-Discover, Together AI and SimpleTES)",
        league_foot_paper="Values are quoted as printed (lower is better). The SimpleTES paper presents its 0.380868 as the new record. Newer community-leaderboard entries are hidden (top bar: <b>Results</b>). Change the <b>Reference</b> selector to re-base every gap in this page.",
        caution_paper="Only published works are shown (papers and preprints with a public result).",
    ),
}
LEADERBOARD_IDS = {"agentdiscover", "atris", "codex"}
LEADERBOARD_EVENTS = ("Oct 2026", "Aug 2026", "Autumn 2026")


def tier_of_result(r):
    if r.get("kind") == "claim":
        return "claim"
    if r["id"] in LEADERBOARD_IDS:
        return "leaderboard"
    return "paper"


def annotate(d):
    """Add tiers (results, tables, events, links) and the text variants shown when only published works are visible."""
    for r in d["results"]:
        r["tier"] = tier_of_result(r)
    for t in d["tables"]:
        t["tier"] = "leaderboard" if "teorth" in t["title"] else "paper"
    ev = []
    for e in d["events"]:
        date, text = e[0], e[1]
        tier = "claim" if ("Unreviewed" in text or "claim" in text.lower() and date.startswith("Nov")) else ("leaderboard" if date in LEADERBOARD_EVENTS else "paper")
        ev.append([date, text, tier])
    d["events"] = ev
    ln = []
    for l in d["links"]:
        lab, url = l[0], l[1]
        tier = "claim" if "unreviewed" in lab.lower() else ("leaderboard" if ("teorth" in url or "issues/348" in url or "issues" in url) else "paper")
        ln.append([lab, url, tier])
    d["links"] = ln
    d.update(PAPER_ONLY_TEXT.get(d["problem"], {}))
    return d


def write(name, data):
    path = os.path.join(HERE, name)
    annotate(data)
    with open(path, "w") as fh:
        json.dump(data, fh, separators=(",", ":"))
    n = sum(1 for r in data["results"] if r.get("values"))
    print("wrote %s (%.0f kB), %d constructions, %d results" % (path, os.path.getsize(path) / 1e3, n, len(data["results"])))


# ----------------------------------------------------------------------------------------------------------------
def ac1():
    S = {
        "theta": ("ypwang61/ThetaEvolve", "7c12898", "Results/FirstAutoCorrIneq/data.json"),
        "simpletes": ("wq-will/SimpleTES", "47d3413", "best_results/mathematics_discovery/first_autocorrelation_inequality/first_autocorrelation_inequality_best_construction.json"),
        "ttt": ("test-time-training/discover", "6c40e82", "results/mathematics/ttt_ac1_sequence.json"),
        "together": ("togethercomputer/EinsteinArena-new-SOTA", "c388c6f", "first-autocorrelation/solutions/ours_2026.json"),
    }
    b = {k: fetch(*v) for k, v in S.items()}
    theta = {e["name"]: e["list"] for e in json.loads(b["theta"][0])}
    simple = json.loads(b["simpletes"][0])
    ttt = json.loads(b["ttt"][0])["sequence"]
    together = json.loads(b["together"][0])["values"]
    thu, ths = b["theta"][1], b["theta"][2]
    R = [
        dict(id="human", label="Best human (Matolcsi-Vinuesa)", method="human", model="", date="2010-01", reported=1.509730, exact_ref=None, kind="construction",
             values=r10(theta["HumanBest"]), url=thu, sha256=ths, note="208-step function stored as 'HumanBest' in ThetaEvolve's data file"),
        dict(id="ae1", label="AlphaEvolve (v1)", method="AlphaEvolve", model="Gemini-2.0 Pro + Flash", date="2025-05", reported=1.505300, exact_ref=1.50529396844016,
             kind="construction", values=r10(theta["AlphaEvolve-v1"]), url=thu, sha256=ths, note="600 equal steps; arXiv:2506.13131"),
        dict(id="ae2", label="AlphaEvolve V2", method="AlphaEvolve V2", model="Gemini-2.0 Pro + Flash", date="2025-11", reported=1.503170, exact_ref=1.50316355468156,
             kind="construction", values=r10(theta["AlphaEvolve-v2"]), url=thu, sha256=ths, note="1319 equal steps; arXiv:2511.02864"),
        dict(id="theta", label="ThetaEvolve (Distill-Qwen3-8B)", method="ThetaEvolve", model="Distill-Qwen3-8B", date="2025-11", reported=1.503133, exact_ref=1.50313243597978,
             kind="construction", values=r10(theta["8B-wo_RL@46-w_SOTA"]), url=thu, sha256=ths, note="entry '8B-wo_RL@46-w_SOTA'; arXiv:2511.23473"),
        dict(id="ttt", label="TTT-Discover (gpt-oss-120b)", method="TTT-Discover", model="gpt-oss-120b", date="2026-01", reported=1.502870, exact_ref=1.50286289825583,
             kind="construction", values=r10(ttt), url=b["ttt"][1], sha256=b["ttt"][2], note="30,000 steps; arXiv:2601.16175"),
        dict(id="together", label="Together AI (Einstein Arena)", method="Together AI", model="mixed agents", date="2026-03", reported=1.502862, exact_ref=1.50286285870531,
             kind="construction", values=r10(together), url=b["together"][1], sha256=b["together"][2],
             note="30,000 steps, refines the TTT-Discover construction; printed 1.502862 is slightly BELOW the exact 1.5028628587 (teorth/optimizationproblems#348)"),
        dict(id="simpletes", label="SimpleTES (gpt-oss-120b)", method="SimpleTES", model="gpt-oss-120b", date="2026-04", reported=1.503871, exact_ref=1.50387081393243,
             kind="construction", values=r10(simple), url=b["simpletes"][1], sha256=b["simpletes"][2], note="972 steps; the paper itself says this trails the leading public AC1 result"),
        dict(id="arena", label="Einstein Arena best (June 2026)", method="Einstein Arena", model="agents in the wild", date="2026-06", reported=1.5028609, exact_ref=None, kind="reported",
             url="https://arxiv.org/abs/2606.10402", note="arXiv:2606.10402 table of frontier problems: prior best 1.5028629 (TTT-Discover) -> Arena best 1.5028609; construction not downloaded"),
        dict(id="agentdiscover", label="AgentDiscover (Einstein Arena)", method="AgentDiscover", model="agents", date="2026-10-03", reported=1.5027435320, exact_ref=1.50274353195396,
             n=65536, kind="reported", url=TAO_1A, note="listed on teorth/optimizationproblems (1a), leaderboard solution 2603 with 65,536 heights; independently replayed by the page's curators (exact 1.50274353195396); construction not downloaded here"),
        dict(id="atris", label="Atris (refines AgentDiscover)", method="Atris", model="", date="2026-10", date_approx=True, reported=1.5027434868, exact_ref=1.50274348670958,
             n=65536, kind="reported", url=TAO_1A, note="65,536 equal steps with exact rational heights, refined from the AgentDiscover construction; replayed in two implementations on the curated page (exact 1.50274348670958). Date not stated - after 3 Oct 2026"),
        dict(id="milo", label="MILO (as quoted)", method="MILO", model="", date=None, reported=1.5027435984, exact_ref=None, kind="claim",
             url="https://www.alphaxiv.org/abs/2611.active-set-escape-circle-packing-autocorrelation", note="value quoted by the Cumes preprint as the Einstein Arena leader / closest predecessor; no primary source found - unverified"),
        dict(id="cumes", label="Cumes preprint (claim)", method="active-set escape", model="Claude agents", date="2026-11", reported=1.502664840776, exact_ref=None, kind="claim",
             url="https://www.alphaxiv.org/abs/2611.active-set-escape-circle-packing-autocorrelation", note="alphaXiv listing, submitted 3 Nov 2026, 262,144-cell grid; unreviewed - no construction release found"),
    ]
    t17 = dict(
        title="Table 17 of arXiv:2604.19341v2 (as printed)",
        subtitle="autocorrelation inequality results: AC1 and AC3 lower is better, AC2 higher is better · best per column in green",
        columns=[dict(label="method"), dict(label="model"), dict(label="AC1 ↓ (this page)", num=True, best="min", digits=6), dict(label="AC2 ↑", num=True, best="max", digits=6), dict(label="AC3 ↓", num=True, best="min", digits=6)],
        rows=[
            ["Best human [93, 18, 149]", "-", 1.509730, 0.901500, 1.458100], ["AlphaEvolve [100]", "Gemini-2.0 Pro + Flash", 1.505300, 0.896200, 1.455700],
            ["AlphaEvolve V2 [43]", "Gemini-2.0 Pro + Flash", 1.503170, 0.961000, None], ["ThetaEvolve [155]", "Distill-Qwen3-8B", 1.503133, 0.946900, 1.493000],
            ["OpenEvolve [6]", "gpt-oss-120b", 1.507190, 0.944900, None], ["OpenEvolve [6]", "Gemini-3.0-Pro", None, None, 1.460000],
            ["TTT-Discover [165]", "gpt-oss-120b", 1.502870, 0.959100, None], ["Together AI [143]", "Mixed", 1.502862, 0.961206, 1.454555],
            ["ShinkaEvolve [73]", "Gemini-3.0-Pro", None, None, 1.457800], ["EvoX [82]", "Gemini-3.0-Pro", None, None, 1.455800],
            ["AlphaResearch [164]", "o4-mini", None, None, 1.546000], ["SimpleTES", "gpt-oss-20b", 1.506791, 0.950494, 1.455324], ["SimpleTES", "gpt-oss-120b", 1.503871, 0.962694, 1.453675]],
        note="† The paper's AC3 entry for ThetaEvolve uses its corrected verifier - informative but not strictly head-to-head with AlphaEvolve. A dash means the source did not report a directly aligned value. Bracketed numbers are the paper's reference list.")
    tao = dict(
        title="Known upper bounds, curated by teorth/optimizationproblems (constant 1a)",
        subtitle="the community-maintained list of record bounds; newer than the paper above",
        columns=[dict(label="bound", num=True), dict(label="reference"), dict(label="comment")],
        rows=[["1.50992", "Matolcsi-Vinuesa 2009", "human"], ["1.5053", "AlphaEvolve", "May 2025 announcement"], ["1.503164", "AlphaEvolve", "Dec 2025 preprint release"], ["1.503133", "ThetaEvolve", ""],
              ["1.503871", "SimpleTES", ""], ["1.502870", "TTT-Discover", ""], ["1.502862", "Together AI", "printed value; exact 1.5028628587"],
              ["1.5027435320", "AgentDiscover (Einstein Arena)", "3 Oct 2026 - independently replayed, 65,536 heights, exact 1.50274353195396"],
              ["1.5027434868", "Atris", "65,536 equal steps, refined from AgentDiscover - replayed, exact 1.50274348670958"]],
        note="Source: constants/1a.md, fetched 2026-11-11. The page also lists speculative 'predicted intervals' from an unverified algebraic framework; they are deliberately not reproduced here.")
    events = [
        ["2010", "Best human bound (Matolcsi-Vinuesa): 1.50973 - the 208-step function is recomputed in the table above."],
        ["May 2025", "AlphaEvolve v1 (arXiv:2506.13131): 600 equal steps → 1.50529397."],
        ["Nov 2025", "AlphaEvolve V2 (arXiv:2511.02864) 1319 steps → 1.50316355; ThetaEvolve (arXiv:2511.23473, an 8B model with test-time learning) 1319 steps → 1.50313244."],
        ["Jan 2026", "TTT-Discover (arXiv:2601.16175, the test-time-training method this repo is about): 30,000 steps, printed 1.502870 (exact 1.5028629)."],
        ["Mar 2026", "Together AI / Einstein Arena refines the TTT construction: printed 1.502862 - but the construction's exact value is 1.5028628587, i.e. ~9e-7 above the printed figure."],
        ["Apr 2026", "SimpleTES (arXiv:2604.19341): 1.503871 with a 972-step function - the paper says it trails the leading public result."],
        ["Jun 2026", "Einstein Arena paper (arXiv:2606.10402): best arena entry 1.5028609 (prior best 1.5028629)."],
        ["Oct 2026", "Einstein Arena leaderboard: AgentDiscover 1.5027435320 (3 Oct, 65,536 heights) and then Atris 1.5027434868 - both independently replayed on the curated 1a page. Much smaller steps than before: ~1.2e-4 below Together AI."],
        ["Nov 2026", "Unreviewed preprint on alphaXiv (3 Nov): three agents certify 1.502664840776 using an exact rational certificate on a 262,144-cell grid; it also quotes MILO's 1.5027435984. No construction release was found, so these two stay claims and are not ranked."],
    ]
    return dict(
        problem="ac1", fetched="2026-11-11", short="AC1", long="first autocorrelation inequality", default_ref="together",
        paper=dict(id="arXiv:2604.19341v2", title="Structured Scaling of AI Discovery Across Diverse Scientific Domains (SimpleTES)", url="https://arxiv.org/abs/2604.19341", table="Supplementary Table 17"),
        lede="Where this run stands against what other methods reached on the same problem - the <b>first autocorrelation inequality</b> (lower is better). The reference table comes from <a href=\"https://arxiv.org/abs/2604.19341\" target=\"_blank\" rel=\"noopener\">arXiv:2604.19341</a> (SimpleTES), Supplementary Table 17; newer results were added from the community-curated list and the Einstein Arena. For every method whose height sequence is public, the bound is <b>recomputed here</b> from the sequence itself.",
        league_note="(the table of <a href=\"https://arxiv.org/abs/2604.19341\" target=\"_blank\" rel=\"noopener\">arXiv:2604.19341</a> plus the newer Einstein Arena / curated-list results)",
        league_foot="Values are quoted as printed (lower is better). The paper's own method (SimpleTES, gpt-oss-120b) reaches 1.503871 on AC1 and does not claim the AC1 record - the record in its table is the Together AI entry. Newer replayed results go lower still. Change the <b>Reference</b> selector (top bar) to re-base every gap in this page.",
        footnote="All numbers are for the discretised problem: a non-negative step function on [−¼, ¼] with n equal steps, score 2n·max(a*a)/(Σa)². Any such score is a valid upper bound for the continuous constant C₁; none of these results says what the true C₁ is.",
        caution="Entries marked 'claim' come from a preprint listing and a quoted number that I could read but not independently verify. Entries marked 'reported' are taken from the paper / curated list as printed (the curated list says its Oct-2026 rows were independently replayed).",
        results=R, tables=[t17, tao], events=events,
        links=[["Together AI - Einstein Arena new SOTA (AC1 solutions + analysis)", "https://github.com/togethercomputer/EinsteinArena-new-SOTA/tree/main/first-autocorrelation"],
               ["Curated list of record bounds (constant 1a)", TAO_1A], ["Exact replays of the AC1 rows (issue #348)", TAO_ISSUE],
               ["TTT-Discover paper (arXiv:2601.16175)", "https://arxiv.org/abs/2601.16175"], ["Einstein Arena paper (arXiv:2606.10402)", "https://arxiv.org/abs/2606.10402"],
               ["Cumes 2026 preprint on alphaXiv (unreviewed)", "https://www.alphaxiv.org/abs/2611.active-set-escape-circle-packing-autocorrelation"]])


# ----------------------------------------------------------------------------------------------------------------
def erdos():
    S = {
        "ae": ("togethercomputer/EinsteinArena-new-SOTA", "c388c6f", "erdos-minimum-overlap/solutions/alphaevolve_2025.py"),
        "haug": ("togethercomputer/EinsteinArena-new-SOTA", "c388c6f", "erdos-minimum-overlap/solutions/haugland_2016.py"),
        "together": ("togethercomputer/EinsteinArena-new-SOTA", "c388c6f", "erdos-minimum-overlap/solutions/together_ai_2026.py"),
        "ttt": ("test-time-training/discover", "6c40e82", "results/mathematics/ttt_erdos_sequence.json"),
        "simpletes": ("wq-will/SimpleTES", "47d3413", "best_results/mathematics_discovery/erdos_minimum_overlap/erdos_minimum_overlap_best_construction.json"),
    }
    b = {k: fetch(*v) for k, v in S.items()}
    txt = lambda k: b[k][0].decode("utf-8")
    ae_half = numbers_after(txt("ae"), "ae_half = ")
    ae = ae_half[:-1] + ae_half[::-1]
    hg = numbers_after(txt("haug"), "haugland_half = ")
    haug = hg + hg[-2::-1]
    together = numbers_after(txt("together"), "h_values = ")
    ttt = json.loads(b["ttt"][0])["sequence"]
    simple = json.loads(b["simpletes"][0])["items"][0]["data"]
    R = [
        dict(id="haugland", label="Haugland 2016", method="Haugland", model="", date="2016-09", reported=0.380927, exact_ref=0.380926853433, kind="construction", values=r10(haug),
             url=b["haug"][1], sha256=b["haug"][2], note="51 steps (mirrored half); arXiv:1609.08000. The curated list prints 0.380926 (truncated) - the exact value of the construction is 0.3809268534"),
        dict(id="ae1", label="AlphaEvolve", method="AlphaEvolve", model="Gemini-2.0 Pro + Flash", date="2025-05", reported=0.380924, exact_ref=0.380923035108, kind="construction", values=r10(ae),
             url=b["ae"][1], sha256=b["ae"][2], note="95 steps (48 numbers mirrored); arXiv:2506.13131 / 2511.02864"),
        dict(id="ttt", label="TTT-Discover (gpt-oss-120b)", method="TTT-Discover", model="gpt-oss-120b", date="2026-01", reported=0.380876, exact_ref=0.380875323218, kind="construction", values=r10(ttt),
             url=b["ttt"][1], sha256=b["ttt"][2], note="600 steps; arXiv:2601.16175"),
        dict(id="together", label="Together AI (Einstein Arena)", method="Together AI", model="mixed agents", date="2026-03", reported=0.380871, exact_ref=0.380870310586, kind="construction", values=r10(together),
             url=b["together"][1], sha256=b["together"][2], note="600 steps, sequential linear programming started from the best known solutions"),
        dict(id="simpletes", label="SimpleTES (gpt-oss-120b)", method="SimpleTES", model="gpt-oss-120b", date="2026-04", reported=0.380868, exact_ref=0.380867675827, kind="construction", values=r10(simple),
             url=b["simpletes"][1], sha256=b["simpletes"][2], note="2400 steps; arXiv:2604.19341 - the paper presents it as the new record for this problem"),
        dict(id="codex", label="CodexProLong (Einstein Arena)", method="CodexProLong", model="agent", date="2026-08-15", reported=0.3808585749, exact_ref=0.380858574858, n=3584, kind="reported",
             url=TAO_1B, note="Einstein Arena leaderboard solution 2507, 3,584 heights; independently replayed on the curated 1b page (exact 0.380858574858, after rescaling to sum n/2); construction not downloaded here"),
        dict(id="atris", label="Atris (refines CodexProLong)", method="Atris", model="", date="2026-09", date_approx=True, reported=0.3808585504, exact_ref=0.380858550316, n=114688, kind="reported",
             url=TAO_1B, note="114,688 equal steps, total mass exactly 1, refined from the CodexProLong construction; replayed in two implementations on the curated page (exact 0.380858550316). Date not stated - after 15 Aug 2026"),
    ]
    tao = dict(
        title="Known upper bounds, curated by teorth/optimizationproblems (constant 1b)",
        subtitle="the community-maintained list of record bounds",
        columns=[dict(label="bound", num=True), dict(label="reference"), dict(label="comment")],
        rows=[["0.385694", "Haugland", "unpublished, 1993"], ["0.382002", "Haugland 1996", ""], ["0.380926", "Haugland 2016", "51 steps"], ["0.380924", "AlphaEvolve", "95 steps"], ["0.380876", "TTT-Discover", "600 steps"],
              ["0.380871", "Together AI", "600 steps"], ["0.380868", "SimpleTES", "2400 steps"],
              ["0.3808585749", "CodexProLong (Einstein Arena)", "15 Aug 2026 - independently replayed, 3,584 heights"],
              ["0.3808585504", "Atris", "114,688 equal steps, refined from CodexProLong - replayed"]],
        note="Known LOWER bound on the constant: 0.379005 (White 2022, convex programming). Source: constants/1b.md, fetched 2026-11-11. The page also lists speculative 'predicted intervals' from an unverified algebraic framework; they are deliberately not reproduced here.")
    events = [
        ["1955", "Erdős poses the minimum-overlap problem; the constant C₅ has been squeezed from above by explicit step functions ever since (0.5 → 0.4 → 0.3857 → 0.3820)."],
        ["2016", "Haugland (arXiv:1609.08000): 51-step function, 0.380927 (exact 0.3809268534)."],
        ["May 2025", "AlphaEvolve: 95 steps (48 numbers mirrored) → 0.380924."],
        ["Jan 2026", "TTT-Discover (arXiv:2601.16175): 600 steps → 0.380876 (exact 0.3808753)."],
        ["Mar 2026", "Together AI / Einstein Arena: sequential linear programming from the best known solutions, 600 steps → 0.380871."],
        ["Apr 2026", "SimpleTES (arXiv:2604.19341): 2400 steps → 0.380868, presented as the new record."],
        ["Jun 2026", "Einstein Arena paper (arXiv:2606.10402): arena best 0.380871 (prior best 0.380876)."],
        ["Aug 2026", "Einstein Arena leaderboard: CodexProLong 0.3808585749 (3,584 heights, 15 Aug) - independently replayed on the curated 1b page."],
        ["Autumn 2026", "Atris refines it to 0.3808585504 with 114,688 equal steps - also replayed. The curated page lists no newer verified result."],
    ]
    return dict(
        problem="erdos", fetched="2026-11-11", short="Erdős", long="Erdős minimum-overlap problem", default_ref="simpletes",
        lower_bound=dict(value=0.379005, who="White (2022)"),
        paper=dict(id="arXiv:2604.19341v2", title="Structured Scaling of AI Discovery Across Diverse Scientific Domains (SimpleTES)", url="https://arxiv.org/abs/2604.19341", table="Figure 6 / main text"),
        lede="Where this run stands against what other methods reached on <b>Erdős' minimum-overlap problem</b> - the constant <b>C₅</b> (lower is better). Published constructions come from the Together AI Einstein Arena repo, TTT-Discover, SimpleTES and Haugland; the newest records come from the community-curated list (<a href=\"" + TAO_1B + "\" target=\"_blank\" rel=\"noopener\">teorth/optimizationproblems, constant 1b</a>). For every method whose step function is public the bound is <b>recomputed here</b> from the sequence itself.",
        league_note="(Haugland, AlphaEvolve, TTT-Discover, Together AI, SimpleTES and the newer Einstein Arena entries from the curated list)",
        league_foot="Values are quoted as printed (lower is better). The SimpleTES paper presents its 0.380868 as the new record; the community-curated list has since added two lower, independently replayed entries (0.3808585749 and 0.3808585504). Change the <b>Reference</b> selector (top bar) to re-base every gap in this page.",
        footnote="All numbers are for the discretised problem: a step function h on [0, 2] with n equal steps, 0 ≤ h ≤ 1, Σh = n/2, score dx·max_s Σ_i h_i(1 − h_{i+s}) with dx = 2/n (overlap over all shifts, zero outside the support). Any such score is a valid upper bound for C₅; the best known lower bound is 0.379005 (White).",
        caution="Entries marked 'reported' (CodexProLong, Atris) are taken from the curated list as printed; the list states it independently replayed them, but their constructions are not downloaded here.",
        results=R, tables=[tao], events=events,
        links=[["Together AI - Einstein Arena new SOTA (Erdős solutions + analysis)", "https://github.com/togethercomputer/EinsteinArena-new-SOTA/tree/main/erdos-minimum-overlap"],
               ["Curated list of record bounds (constant 1b)", TAO_1B], ["Exact replays of the Erdős rows (issue #348)", TAO_ISSUE],
               ["TTT-Discover paper (arXiv:2601.16175)", "https://arxiv.org/abs/2601.16175"], ["SimpleTES paper (arXiv:2604.19341)", "https://arxiv.org/abs/2604.19341"],
               ["Haugland 2016 (arXiv:1609.08000)", "https://arxiv.org/abs/1609.08000"], ["Einstein Arena paper (arXiv:2606.10402)", "https://arxiv.org/abs/2606.10402"]])


def main():
    which = sys.argv[1:] or ["ac1", "erdos"]
    if "ac1" in which:
        write("published_ac1.json", ac1())
    if "erdos" in which:
        write("published_erdos.json", erdos())


if __name__ == "__main__":
    main()
