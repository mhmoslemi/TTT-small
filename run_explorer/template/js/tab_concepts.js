"use strict";
/* ---- Concepts tab: architecture, glossary, gotchas, file map ---------------------------------- */
const FILE_MAP = [
  ["config.json", "Resolved run configuration snapshot (stale for the JEV threshold - see stepNN.jev.json)."],
  ["setting.log · temirnal.log", "Startup settings; timestamped tee of stdout (HH:MM:SS only, no dates)."],
  ["result.txt", "One line per finished step: step best / best seen raw and reward."],
  ["search_tree.json", "Every program as a node {parent_id, child_ids, raw_problem_score}; failed programs have score 0."],
  ["entropy.jsonl", "One summary per step: entropy, tokens, valid fraction, code diversity, by strategy and by parent×strategy."],
  ["stepNN/stepNN.parents.json", "The 8 selected parents with Q, P, bonus, score and the JEV verdicts."],
  ["stepNN/stepNN.jev.json", "Every GLiDE judgment, the scan order with accept/reject, threshold, and the candidate sources."],
  ["stepNN/stepNN.summary.json", "Counts, best scores, hurdle allocation (32 entries), SPO² tracker, policy-update stats, timings. Its presence marks a finished step."],
  ["stepNN/stepNN.best.json", "Best program so far: code + construction + score."],
  ["stepNN/…_groupGG_rolloutRRR.txt", "Raw coder output: reasoning, </think>, then the python block."],
  ["…rolloutRRR.prompt.txt", "The fully rendered coder prompt (task, parent code, strategy, system line)."],
  ["…rolloutRRR.strategy.txt", "The strategist's full response (with reasoning) that produced the strategy of this rollout."],
  ["…rolloutRRR.meta.json", "Score, validity, failure kind, advantage, parent/PUCT snapshot, strategy and allocation info. Dict fields are Python-repr strings."],
  ["…_foldFF_strategySS.txt", "The accepted strategist response for the slot (or the fallback sentence)."],
  ["…_strategySS_attemptAA.{txt,prompt.txt,meta.json}", "Every strategist attempt, its prompt and the format verdict."],
  ["vllm.log · strategy_vllm.log", "Engine logs of the coder and strategist pools."],
  ["plots.log · *.png/pdf/svg", "Per-step plots written by the analysis helpers."],
];

function architectureSvg(host) {
  registerChart(host, () => {
    const W = 1130, H = 476;
    host.innerHTML = "";
    const svg = sv("svg", { viewBox: `0 0 ${W} ${H}`, width: "100%", style: "max-width:1130px" }, host);
    const defs = sv("defs", null, svg);
    ["a", "b"].forEach(k => { const m = sv("marker", { id: "ar" + k, viewBox: "0 0 10 10", refX: 9, refY: 5, markerWidth: 7, markerHeight: 7, orient: "auto-start-reverse" }, defs); sv("path", { d: "M0,0L10,5L0,10z", fill: k === "a" ? cssVar("--ink-3") : cssVar("--s4") }, m); });
    const col = { search: cssVar("--s4"), strat: cssVar("--s2"), coder: cssVar("--s1"), cpu: cssVar("--s3"), train: cssVar("--s8"), jev: cssVar("--s7") };
    const box = (x, y, w, hh, title, sub, c, stage) => {
      const g = sv("g", { style: stage ? "cursor:pointer" : "" }, svg);
      sv("rect", { x, y, width: w, height: hh, rx: 10, fill: cssVar("--surface"), stroke: c, "stroke-width": 2 }, g);
      sv("rect", { x, y, width: 6, height: hh, rx: 3, fill: c }, g);
      const t = sv("text", { x: x + 16, y: y + 24, style: "font-weight:600;font-size:13px;fill:" + cssVar("--ink") }, g); t.textContent = title;
      sub.split("\n").forEach((ln, i) => { const s = sv("text", { x: x + 16, y: y + 42 + i * 14, style: "font-size:11px;fill:" + cssVar("--ink-2") }, g); s.textContent = ln; });
      if (stage) g.addEventListener("click", () => { _stageSel = stage; setState({ tab: "pipeline" }); });
      return g;
    };
    const arrow = (d, label, dashed, lx, ly, c) => {
      sv("path", { d, fill: "none", stroke: c || cssVar("--ink-3"), "stroke-width": 1.6, "stroke-dasharray": dashed ? "5 4" : "", "marker-end": `url(#ar${dashed ? "b" : "a"})` }, svg);
      if (label) { const t = sv("text", { x: lx, y: ly, "text-anchor": "middle", style: "font-size:10.5px;fill:" + cssVar("--ink-3") }, svg); t.textContent = label; }
    };
    // row A
    box(20, 24, 200, 74, "Archive", "valid, de-duplicated programs\n8 seeds + ~63 per step · cap 2000", col.search, "archive");
    box(285, 24, 190, 74, "PUCT ranking", "Q + c·scale·P·√(1+T)/(1+n)\nrank-based prior P", col.search, "select");
    if (RUN.hasJev) box(540, 24, 220, 74, "JEV gate · GLiDE-9B", "reject if P(same core idea) ≥ " + (((ST[STEPS[STEPS.length - 1]] || {}).jev || {}).threshold || 0.68) + "\nlooks at top 3×8 = 24 states", col.jev, "select");
    else box(540, 24, 220, 74, "Lineage blocking", "skip ancestors / descendants of\nevery picked parent (no JEV gate)", col.search, "select");
    box(825, 24, 200, 74, "8 parents", "one per group\n+ their code, score, construction", col.search, "select");
    arrow("M220,61H283", "", false); arrow("M475,61H538", "", false); arrow("M760,61H823", "", false);
    // row B (right to left)
    box(825, 168, 200, 82, "Strategist chains", String(CFG.strategy_model_name || "?").split("/").pop() + " (frozen)\n" + (CFG.strategies_per_parent || 4) + " sequential plans per parent\nup to " + ((CFG.strategy_format_max_retries != null ? CFG.strategy_format_max_retries : 3) + 1) + " attempts, else fallback", col.strat, "strategist");
    box(540, 168, 220, 82, "32 strategy plans", "<strategy> blocks: approach, math,\nprocedure, search, budget, contract\n(thinking never reaches the coder)", col.strat, "strategist");
    { const nm = p => String(p || "").split("/").filter(Boolean).pop(); const L = STEPS[STEPS.length - 1], pil = CFG.pilot_programs_per_strategy || 11;
      box(285, 168, 190, 82, "Coder", nm(CFG.coder_model_name || CFG.model_name) + " + LoRA r=" + (CFG.lora_rank || "?") + "\nvLLM, thinking '" + (CFG.coder_reasoning_effort || "?") + "'\n" + pil + " pilot programs / strategy", col.coder, "pilot"); }
    { const per = (CFG.groups_per_step || 8) * (CFG.group_size || 64), pilot = (CFG.groups_per_step || 8) * (CFG.strategies_per_parent || 4) * (CFG.pilot_programs_per_strategy || 11);
      box(20, 168, 200, 82, per + " programs / step", pilot + " pilot + " + (per - pilot) + " phase 2\npython `" + PROB.entry + "`\n" + (CFG.budget_s || "?") + " s search budget each", col.coder, "pilot"); }
    arrow("M925,98V166", "", false); arrow("M825,209H762", "plans", false, 793, 203); arrow("M540,209H477", "", false); arrow("M285,209H222", "", false);
    // row C
    box(20, 316, 170, 74, "CPU sandbox", "reward = 1 / raw " + PROB.metric + "\nfailure ⇒ 0" + ((ST[STEPS[STEPS.length - 1]] || {}).evaluation_workers ? " · " + ST[STEPS[STEPS.length - 1]].evaluation_workers + " parallel" : ""), col.cpu, "eval");
    box(230, 316, 170, 74, "Hurdle allocator", "20 spare programs / parent\nto the strategy likeliest to beat it", col.search, "alloc");
    box(440, 316, 190, 74, "SPO² advantage", "A = (exp(2r)/v − 1)/2\nrun-wide baseline v", col.train, "adv");
    box(670, 316, 170, 74, "LoRA update", "PPO-clip, 1 AdamW step\nlr 4e-5 · 2 GPUs", col.train, "train");
    box(880, 316, 215, 74, "Archive update", "dedupe · top-2 / (parent,strategy)\ntop-8 / parent · cap 2000", col.search, "archive");
    arrow("M120,250V314", "evaluate", false, 148, 286);
    arrow("M190,353H228", "", false);
    arrow("M315,314V252", "+20 programs / parent", true, 396, 290, cssVar("--s4"));
    // bypasses along the bottom
    arrow("M105,392V420H535V392", "all rewards", false, 320, 436);
    arrow("M755,314V282H400V252", "updated LoRA adapter → samples the next step", true, 640, 276, cssVar("--s8"));
    arrow("M630,353H668", "", false);
    arrow("M150,392V446H987V392", "valid programs with code", false, 760, 462);
    // loop notes (adapter loop is drawn above)
    const note = (x, y, t) => { const s = sv("text", { x, y, style: "font-size:11px;font-style:italic;fill:" + cssVar("--s4") }, svg); s.textContent = t; };
        note(880, 409, "→ becomes next step's archive");
  });
}

registerTab({
  id: "concepts", label: "How it works", ico: "ⓘ",
  async render(root) {
    root.append(pageHead("How the whole system fits together",
      `Written from a read-through of the code base (<code>train_multy_CVaR.py</code>, <code>search/</code>, <code>training/</code>, <code>generation/</code>, <code>problems/ac_inequalities.py</code>) and checked against the files of this run. ` +
      `Click any box to open that stage in the <b>Pipeline</b> tab. Solid arrows happen inside one step; dashed arrows are the loops that tie steps together.`));
    const host = h("div", { class: "chart" });
    root.append(card("Architecture of one training step", "colour = which model/subsystem does the work", h("div", null, host,
      legend([{ color: cssVar("--s4"), label: "search / bookkeeping" }, { color: cssVar("--s7"), label: "GLiDE-9B judge" }, { color: cssVar("--s2"), label: "strategist (frozen)" }, { color: cssVar("--s1"), label: "coder (trained)" }, { color: cssVar("--s3"), label: "CPU sandboxes" }, { color: cssVar("--s8"), label: "trainer" }]))));
    architectureSvg(host);

    // the three loops
    root.append(h("div", { class: "grid g3", style: { marginTop: "14px" } },
      conceptCard("Search loop (across steps)", (RUN.hasJev ? "PUCT + JEV pick parents" : "PUCT picks parents (lineage-blocked)") + " → their children (valid, new, high reward) enter the archive → next step's parents. This is what drives the best-so-far curve and keeps the 8 parents of a step on different branches."),
      conceptCard("Allocation loop (inside a step)", "A strategy's 11 pilot programs reveal how likely it is to beat the parent; the hurdle bandit then funnels the 20 spare programs to the most promising plan. It measures the strategist indirectly - the strategist itself is never trained."),
      conceptCard("Learning loop (across steps)", "SPO² turns rewards into advantages against one slowly-moving baseline; a single PPO-clipped LoRA step updates the coder; the new adapter samples step N+1. Mostly it learns to produce valid, well-formed programs.")));

    // stage deep dives
    root.append(h("h2", { style: { margin: "22px 0 10px" } }, "Stage reference"));
    const wrap = h("div", { class: "cols2" });
    STAGES.map(stageView).forEach(sg => wrap.append(h("div", { class: "concept", style: { breakInside: "avoid" } },
      h("h2", null, `${sg.n}. ${sg.title}`), h("div", { html: sg.body }), h("div", { class: "eq" }, sg.eq), h("div", null, sg.refs.map(r => h("div", { class: "cite" }, r))))));
    root.append(wrap);

    // glossary
    root.append(h("h2", { style: { margin: "22px 0 10px" } }, "Glossary"));
    root.append(card(null, null, h("div", { class: "kv", style: { gridTemplateColumns: "190px 1fr", gap: "8px 14px" }, html: glossary().map(([k, v]) => `<div class="k"><b style="color:var(--ink)">${esc(k)}</b></div><div class="v">${v}</div>`).join("") })));

    // gotchas
    root.append(h("h2", { style: { margin: "22px 0 10px" } }, "Things that are easy to get wrong (found while reading the code and the data)"));
    root.append(card(null, null, h("div", null, gotchas().map(([t, fl, b]) => h("div", { style: { padding: "8px 0", borderBottom: "1px solid var(--line)" } }, h("div", { style: { fontWeight: 600 } }, t), h("div", { html: b, style: { color: "var(--ink-2)", fontSize: "13px" } }))))));

    // file map
    root.append(h("h2", { style: { margin: "22px 0 10px" } }, "What each file in the run directory is"));
    root.append(card(null, null, h("div", { class: "kv", style: { gridTemplateColumns: "300px 1fr", gap: "6px 14px" }, html: FILE_MAP.filter(([k]) => RUN.hasJev || !/jev/.test(k)).map(([k, v]) => `<div class="k mono" style="color:var(--ink-2)">${esc(k)}</div><div class="v">${esc(v)}</div>`).join("") })));
  }
});

function conceptCard(t, b) { return h("div", { class: "card" }, h("h2", null, t), h("p", { style: { color: "var(--ink-2)", margin: "4px 0 0" } }, b)); }
