"use strict";
let _ovRefs = "main";
/* ---- Overview tab ------------------------------------------------------------------ */
const _ssCache = {};
function stepStats(step) {
  if (_ssCache[step]) return _ssCache[step];
  const rs = RS[step] || [];
  const o = { n: rs.length, valid: 0, code: 0, constraint: 0, timeout: 0, improved: 0, retries: 0, tokens: 0, fallback: 0 };
  const raws = [];
  rs.forEach(r => {
    if (r.valid) { o.valid++; raws.push(r.raw); } else if (r.fail === "timeout") o.timeout++; else if (r.fail === "constraint") o.constraint++; else o.code++;
    if (r.improved) o.improved++;
    if (r.isRetry) o.retries++;
    o.tokens += r.ntok || 0;
  });
  o.raws = raws.sort((a, b) => a - b);
  o.best = raws.length ? raws[0] : null;
  _ssCache[step] = o;
  return o;
}
function bestSeen(step) { const s = ST[step]; return s && s.best_seen_raw_score != null ? s.best_seen_raw_score : null; }

registerTab({
  id: "overview", label: "Overview", ico: "◎",
  async render(root) {
    const done = STEPS;                                   // finished steps (summary written): training, timing, archive
    const seen = BSTEPS;                                  // + steps whose rollouts are scored: best-so-far, validity, score distribution
    const last = ST[done[done.length - 1]], lastSeen = ST[seen[seen.length - 1]];
    const nSteps = CFG.num_steps || "?";
    const best = lastSeen.best_seen_raw_score, tgt = TARGET();
    const totalRoll = R.length;
    const valids = R.filter(r => r.valid).length;
    const cum = last.cumulative_training_seconds;
    const partial = ALL_STEPS.filter(s => !ST[s].completed);
    const HW = D.main.hardware || {};

    root.append(pageHead("Run overview",
      `<b>${esc(D.main.meta.run_name)}</b> — ${PROB.intro}. ` +
      `A frozen <b>strategist</b> proposes plans, a LoRA-trained <b>coder</b> turns them into programs, a <b>${RUN.selection}</b> search decides which programs to extend, and <b>SPO²</b> updates the coder from the scored programs. ` +
      `Use the tabs on the left to follow one step through every stage; the step selector (top right) is shared by all tabs.`));

    root.append(h("div", { class: "kpis" },
      kpi("Best " + PROB.metric, fmt.raw(best), `vs ${esc(TARGET_LABEL())} ${fmt.rep(tgt)} · gap <b>${fmt.gap(best - tgt)}</b> (${fmt.pctGap((best - tgt) / tgt)})` + (lastSeen.provisional ? ` · found in step ${lastSeen.best_seen_step}, update pending` : "")),
      kpi("Steps completed", `${done.length} <span style="font-size:14px;color:var(--ink-3)">/ ${nSteps}</span>`, partial.length ? partial.map(p => ST[p].provisional ? `step ${p}: rollouts scored, adapter update pending` : `step ${p} in progress (${(RS[p] || []).length} programs so far)`).join(" · ") : "all steps on disk finished"),
      kpi("Programs evaluated", fmt.n(totalRoll), `${fmt.n(valids)} valid (${fmt.pct(valids / totalRoll)})`),
      kpi("Improvement vs seed", fmt.f(PROB.seedRaw - best, 4), `seeds score ${fmt.f(PROB.seedRaw, 4)} → ${fmt.raw(best)}`),
      kpi("Archive size", fmt.n(last.archive_size), `${fmt.n(D.main.meta.n_nodes)} nodes in the search tree`),
      kpi("Hardware", HW.n_gpus ? `${HW.n_gpus} × ${esc(HW.short)}` : "–", HW.n_gpus ? `${HW.mem_gib ? HW.mem_gib.toFixed(0) + " GiB each · " : ""}${esc(HW.family || "")}` : "not recorded"),
      kpi("Training compute", fmt.dur(cum), `coder LoRA updates on ${last.training_parallel_gpus || String(CFG.training_gpu_ids || CFG.gpu_ids || CFG.available_gpu_ids || "").split(",").filter(Boolean).length || "several"} GPUs`)));

    hardwareCard(root, HW, done);

    // ---- best curve + recipe
    const curve = card("Best " + PROB.metric + " per step", "lower is better · click a point to open that step in <b>Pipeline</b>", h("div", null,
      h("div", { class: "row-flex", style: { marginBottom: "6px" } }, seg([{ value: "main", label: "reference only" }, { value: "all", label: "all published results" }], _ovRefs, v => { _ovRefs = v; drawBest(); }), h("span", { class: "sub" }, "reference = " + TARGET_LABEL())),
      h("div", { class: "chart", id: "chBest" })));
    const recipe = card("Run recipe", "from config.json", recipeNode());
    root.append(h("div", { class: "grid g-main" }, curve, recipe));
    const drawBest = () => lineChart($("#chBest"), {
      height: 300, xLabel: "step", yLabel: PROB.short + " " + PROB.metric, margin: { l: 78 },
      series: [
        { name: "best so far", color: S_COL(0), pts: seen.map(s => [s, bestSeen(s)]), step: false, width: 2.4 },
        { name: "best of this step", color: S_COL(1), pts: seen.map(s => [s, ST[s].step_best_raw_score]), width: 0, dotR: 5 },
      ],
      hlines: refLines(_ovRefs === "all"), includeH: true,
      xFmt: v => String(Math.round(v)), xTicks: seen,
      tipHead: x => `<b>step ${x}</b>` + (ST[x].provisional ? " (update pending)" : ""),
      tipVal: v => fmt.raw(v),
      tipExtra: x => `<div class="mut">valid ${fmt.pct(ST[x].valid_fraction)}${ST[x].archive_size != null ? " · archive " + ST[x].archive_size : ""}</div>`,
      onClick: x => { setState({ tab: "pipeline", step: x }); },
    });
    drawBest();
    root.append(legend([{ color: S_COL(0), label: "best so far (best_seen)", line: true }, { color: S_COL(1), label: "best of the step" }, { color: cssVar("--good-ink"), label: "reference: " + TARGET_LABEL(), line: true }]));

    // ---- league table vs published results
    leagueSection(root, best);

    // ---- validity + failure breakdown, score quantiles
    const stats = seen.map(stepStats);
    const failCard = card("What happens to the programs?", "each bar = all rollouts of a step", h("div", { class: "chart", id: "chFail" }));
    const quantCard = card("Score distribution of valid programs", "distance of valid programs to the reference (symmetric log) - the whole population moves, not just the best", h("div", { class: "chart", id: "chQuant" }));
    root.append(h("div", { class: "grid g2", style: { marginTop: "14px" } }, failCard, quantCard));
    const COLS = { valid: cssVar("--good"), code: cssVar("--s2"), timeout: cssVar("--s4"), constraint: cssVar("--s7") };
    barChart($("#chFail"), {
      height: 300, cats: seen.map(String), stacked: true, xLabel: "step", yLabel: "programs", bands: true,
      segLabels: (i, si, v, t) => (v / t >= 0.06 ? Math.round(100 * v / t) + "%" : ""),
      totalLabel: (i, t) => String(t),
      colTip: i => { const s = stats[i]; const p = x => fmt.pct(x / s.n, 1); return `<b>step ${seen[i]}</b> · ${s.n} programs<div><span class="dot" style="background:${COLS.valid}"></span>valid ${s.valid} (${p(s.valid)})</div><div><span class="dot" style="background:${COLS.code}"></span>code failure ${s.code} (${p(s.code)})</div><div><span class="dot" style="background:${COLS.timeout}"></span>timeout ${s.timeout} (${p(s.timeout)})</div><div><span class="dot" style="background:${COLS.constraint}"></span>invalid score ${s.constraint} (${p(s.constraint)})</div>`; },
      series: [
        { name: "valid", color: COLS.valid, vals: stats.map(s => s.valid) },
        { name: "no / broken code block, runtime error", color: COLS.code, vals: stats.map(s => s.code) },
        { name: "timeout (1900 s)", color: COLS.timeout, vals: stats.map(s => s.timeout) },
        { name: "invalid score", color: COLS.constraint, vals: stats.map(s => s.constraint) },
      ],
      tip: (i, si, v) => { const s = stats[i]; return `<b>step ${seen[i]}</b><div>valid ${s.valid} (${fmt.pct(s.valid / s.n)})</div><div>code failure ${s.code}</div><div>timeout ${s.timeout}</div><div>invalid score ${s.constraint}</div><div class="mut">${s.n} programs</div>`; },
      onClick: i => setState({ tab: "rollouts", step: seen[i] }),
    });
    root.append(legend([{ color: COLS.valid, label: "valid" }, { color: COLS.code, label: "code failure (no / broken python block, runtime error)" }, { color: COLS.timeout, label: "timeout" }, { color: COLS.constraint, label: "invalid score" }]));
    const qs = [["best", 0, 0], ["10th pct", 0.1, 1], ["median", 0.5, 2], ["90th pct", 0.9, 3]];
    const qpts = qs.map(([n, q, i]) => seen.map((s, k) => [s, gapOf(quantile(stats[k].raws, q))]));
    const qd = gapDomain(qpts.flatMap(arr => arr.slice(2).map(p => p[1])));
    lineChart($("#chQuant"), {
      height: 280, xLabel: "step", yLabel: "gap to reference " + tgt.toFixed(6) + " (symlog)", margin: { l: 62 }, clip: true, yDomain: qd,
      yTicks: gapTicks(), yFmt: gapFmt,
      series: qs.map(([n, q, i], qi) => ({ name: n, color: S_COL(i), pts: qpts[qi], width: 1.8, dotR: 2.6, fmt: v => (tgt + gapInv(v)).toFixed(5) })),
      xFmt: v => String(Math.round(v)), xTicks: seen,
      tipHead: x => `<b>step ${x}</b> (n valid = ${stats[seen.indexOf(x)].raws.length})`,
    });
    root.append(legend(qs.map(([n, , i]) => ({ color: S_COL(i), label: n, line: true }))));

    // ---- wall clock
    const phases = D.main.phases || {};
    const PH = [["select", "Parent selection (" + RUN.selection + ")", 0], ["strategist", "Strategist chains", 1], ["coder", "Coder generation + CPU eval", 2], ["strategist+coder", "Strategist + coder (not split in old log format)", 6], ["scoring", "Log-prob scoring", 3], ["training", "LoRA training", 4]];
    const pst = done.filter(s => phases[s]);
    if (pst.length) {
      const c = card("Wall-clock time per step", "approximate, reconstructed from timestamps in terminal log (resumed steps include restart time)", h("div", { class: "chart", id: "chPhase" }));
      const tab = card("Throughput", "per completed step", h("div", { id: "thr" }));
      root.append(h("div", { class: "grid g-main", style: { marginTop: "14px" } }, c, tab));
      barChart($("#chPhase"), {
        height: 300, cats: pst.map(String), stacked: true, xLabel: "step", yLabel: "hours", yFmt: v => v.toFixed(1) + " h", margin: { l: 62 }, bands: true,
      segLabels: (i, si, v) => (v >= 0.2 ? v.toFixed(1) + " h" : ""),
      totalLabel: (i, t) => t.toFixed(1) + " h",
      colTip: i => { const ph = phases[pst[i]]; return `<b>step ${pst[i]}</b> · total ${fmt.dur(sum(Object.values(ph)))}` + PH.filter(([k]) => ph[k]).map(([k, n, ci]) => `<div><span class="dot" style="background:${S_COL(ci)}"></span>${esc(n)}: <b>${(ph[k] / 3600).toFixed(2)} h</b> (${fmt.dur(ph[k])})</div>`).join(""); },
        series: PH.map(([k, n, i]) => ({ name: n, color: S_COL(i), vals: pst.map(s => (phases[s][k] || 0) / 3600) })),
        tip: (i, si, v) => `<b>step ${pst[i]}</b><div>${esc(PH[si][1])}: <b>${fmt.dur(v * 3600)}</b></div><div class="mut">total ${fmt.dur(sum(Object.values(phases[pst[i]])))}</div>`,
      });
      root.append(legend(PH.filter(([k]) => pst.some(s => phases[s][k])).map(([k, n, i]) => ({ color: S_COL(i), label: n }))));
      const lastS = done[done.length - 1], tot = sum(Object.values(phases[lastS] || {}));
      const E = D.main.elapsed || {}, byStep = E.by_step || {};
      const stepSecs = s => { const a = byStep[String(s)], b = s === done[0] ? 0 : byStep[String(done[done.indexOf(s) - 1])]; return a != null && b != null ? a - b : null; };
      const durs = done.map(stepSecs).filter(x => x != null && x > 0);
      const recent = durs.slice(-6), perStep = recent.length ? median(recent) : tot;
      const remainSteps = Math.max(0, nSteps - done.length), left = perStep * remainSteps;
      const computeSum = sum(done.map(s => sum(Object.values(phases[s] || {}))));
      const spent = E.total != null ? E.total : computeSum;
      const stopped = !!E.stopped;
      const eta = E.log_mtime && !stopped ? new Date(new Date(E.log_mtime).getTime() + left * 1000) : null;
      const rows = [
        ["Run status", stopped ? `<span class="pill warn">stopped</span> <span class="sub">${esc(E.stop_hint || "interrupted")} - the log ends after step ${done[done.length - 1]}</span>` : `<span class="pill good">running or finished</span> <span class="sub">no interruption in the last log lines</span>`],
        ["Time spent so far", `<b>${fmt.dh(spent)}</b> <span class="sub">(${(spent / 3600).toFixed(1)} h${E.total != null ? ", wall clock incl. restarts" : ""})</span>`],
        ["…of which in step phases", `${fmt.dh(computeSum)} <span class="sub">(${(computeSum / 3600).toFixed(1)} h, sum of the bars - only steps whose log phases could be parsed)</span>`],
        [stopped ? "Time left if resumed" : "Time left (estimate)", `<b>${fmt.dh(left)}</b> <span class="sub">(${(left / 3600).toFixed(1)} h · ${remainSteps} steps × ${fmt.dur(perStep)})</span>`],
        ["Estimated finish", eta ? `${eta.toLocaleString("en-US", { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" })} <span class="sub">(from the log file's last write time)</span>` : (stopped ? "– (run is stopped)" : "–")],
        ["Total run time", `<b>${fmt.dh(spent + left)}</b> <span class="sub">for ${nSteps} steps${stopped ? " if it were resumed" : ""}</span>`],
        ["Training / step", fmt.dur(mean(done.map(s => ST[s].training_seconds).filter(x => x)))],
        ["Tokens / step", `${fmt.tok(Math.round(mean(done.slice(1).map(s => stepStats(s).tokens))))} response tokens`],
        [`Step ${lastS} wall clock`, fmt.dur(stepSecs(lastS) != null ? stepSecs(lastS) : tot)],
      ];
      $("#thr").append(h("div", { class: "kv", style: { gridTemplateColumns: "170px 1fr" }, html: rows.map(([k, v]) => `<div class="k">${k}</div><div class="v">${v}</div>`).join("") }));
      $("#thr").append(h("p", { class: "sub", style: { marginTop: "10px" } }, "Time left = median of the last ≤ 6 step durations × remaining steps; the log has no dates, so elapsed time is unwrapped from the clock (gaps longer than a day while the run was stopped are not visible). The coder phase dominates: ~20k reasoning tokens per program, 512 programs per step; training is ~1 h."));
    }

    // ---- step table
    const rows = seen.map(s => ({ s }));
    const tb = table([
      { label: "step", get: r => String(r.s), num: true },
      { label: "step best", get: r => fmt.raw(ST[r.s].step_best_raw_score), num: true },
      { label: "best so far", get: r => fmt.raw(bestSeen(r.s)), num: true },
      { label: "new record?", get: r => { const i = seen.indexOf(r.s); return (i === 0 || bestSeen(r.s) < bestSeen(seen[i - 1]) - 1e-12) ? pill("new best", "good") : ""; } },
      { label: "valid", get: r => fmt.pct(ST[r.s].valid_fraction), num: true },
      { label: "beat parent", get: r => stepStats(r.s).improved, num: true, title: "valid programs with a higher reward than their parent" },
      { label: "archive", get: r => ST[r.s].archive_size != null ? fmt.n(ST[r.s].archive_size) : "–", num: true },
      ...(RUN.hasJev ? [{ label: "JEV thr.", get: r => ST[r.s].jev ? ST[r.s].jev.threshold : "–", num: true }] : []),
      { label: "train time", get: r => ST[r.s].training_seconds != null ? fmt.dur(ST[r.s].training_seconds) : "pending", num: true },
      { label: "elapsed", get: r => { const e = ((D.main.elapsed || {}).by_step || {})[String(r.s)]; return e != null ? fmt.dh(e) : "–"; }, num: true, title: "wall-clock time since the first log line, at the end of this step" },
      { label: "SPO² v", get: r => ST[r.s].spo ? fmt.f(ST[r.s].spo.value_before, 3) : "–", num: true },
    ], rows, { onRow: r => setState({ tab: "pipeline", step: r.s }) });
    root.append(h("div", { style: { marginTop: "14px" } }, card("All steps", "click a row to open it in Pipeline", h("div", { class: "scroll" }, tb))));
  }
});

function recipeNode() {
  const c = CFG;
  const nm = p => (p || "").split("/").filter(Boolean).pop();
  const rows = [
    ["Problem", `${c.problem} · ${PROB.recipeProblem} · prompt target ${c.target}`],
    ["Coder (trained)", `${nm(c.coder_model_name || c.model_name)} + LoRA r=${c.lora_rank}, lr ${c.learning_rate}, reasoning "${c.coder_reasoning_effort}"`],
    ["Strategist (frozen)", `${nm(c.strategy_model_name)}${c.strategy_reasoning_effort ? ` · reasoning "${c.strategy_reasoning_effort}"` : ""}${c.strategy_format_max_retries != null ? ` · up to ${c.strategy_format_max_retries + 1} attempts` : ""}`],
    ...(RUN.hasJev ? [["JEV judge", nm((c.jev_model_path || "").replace(/\/snapshots.*/, "")).replace(/^models--/, "").replace(/--/g, "/")]] : []),
    ["Parent selection", `${c.parent_selection || RUN.sampler} · c=${c.puct_c} · ${c.num_seed_states} seeds${RUN.hasJev ? ` · window ×${c.jev_candidate_multiplier}` : ""}`],
    ["Per step", `${c.groups_per_step} parents × (${c.strategies_per_parent} strategies × ${c.programs_per_strategy}) = ${c.groups_per_step * c.group_size} programs`],
    ["Pilot / phase 2", `${c.pilot_programs_per_strategy} pilot per strategy, then ${c.strategies_per_parent * (c.programs_per_strategy - c.pilot_programs_per_strategy)} per parent by "${c.phase2_allocation_method}"`],
    ["Advantage", `${c.advantage_mode} · β=${c.spo_rs_beta} · clip [${c.spo_rs_clip_epsilon_low}, ${c.spo_rs_clip_epsilon_high}]`],
    ["Archive", `top-${c.strategy_archive_top_r}/strategy · top-${c.topk_children_per_parent}/parent · cap ${c.max_buffer_size}`],
    ["Program budget", `${c.budget_s}s search${c.sandbox_timeout_s ? `, sandbox kill at ${c.sandbox_timeout_s}s` : ""}${c.max_new_tokens ? `, up to ${fmt.n(c.max_new_tokens)} tokens` : ""}`],
    ["Hardware", ((D.main.hardware || {}).label || [c.hardware_profile, c.gpu_ids ? "GPUs " + c.gpu_ids : ""].filter(Boolean).join(" · ")) + (c.generation_backend ? " · vLLM " + c.generation_backend : "")],
  ];
  return h("div", { class: "kv", html: rows.map(([k, v]) => `<div class="k">${esc(k)}</div><div class="v">${esc(v)}</div>`).join("") });
}


/* hlines for the reference value(s) */
function refLines(all) {
  const main = REF(), out = [{ y: main.val, label: `${main.label} ${main.val.toFixed(6)}`, color: cssVar("--good-ink") }];
  const others = all ? refList().filter(r => r.id !== main.id && r.id !== "prompt") : refList().filter(r => r.id === "ttt" && main.id !== "ttt");
  others.forEach((r, i) => out.push({ y: r.val, label: `${r.label} ${r.val.toFixed(6)}`, color: cssVar("--ink-3"), dash: "2 4", anchor: "start", dy: (i % 2 ? 14 : 0) }));
  return out;
}

function leagueSection(root, best) {
  const pub = D.pub || {};
  const items = refList().filter(r => r.id !== "prompt").map(r => ({ label: r.label, val: r.val, ref: r, claim: r.kind === "claim" }));
  if (!items.length) return;
  items.push({ label: "THIS RUN (best so far)", val: best, me: true });
  const prompt = refList().find(r => r.id === "prompt");
  items.sort((a, b) => a.val - b.val);
  const firm = items.filter(x => !x.claim);
  const rank = firm.findIndex(x => x.me) + 1, lead = firm[0];
  const beaten = firm.filter(x => !x.me && x.val > best).map(x => x.label);
  const claims = items.filter(x => x.claim);
  const host = h("div", { class: "chart" });
  const txt = h("div", { html:
    `<p style="margin:0 0 8px;color:var(--ink-2)">This run's best ${esc(PROB.metric)} is <b>${fmt.raw(best)}</b> - rank <b>${rank}</b> of ${firm.length} among the published ${esc(PROB.short)} results ${pubText("league_note")}. ${claims.length ? `${claims.length} unreviewed claims (hollow dots) are drawn but not ranked. ` : ""}Details and sources: <b>Published SOTA</b> tab.</p>` +
    `<div class="kv" style="grid-template-columns:150px 1fr">` +
    `<div class="k">${SHOW === "paper" ? "Leading published" : "Leading result shown"}</div><div class="v">${esc(lead.label)} · ${fmt.rep(lead.val)}</div>` +
    `<div class="k">Gap to leading</div><div class="v"><b>${fmt.gap(best - lead.val)}</b> (${fmt.pctGap((best - lead.val) / lead.val)})</div>` +
    `<div class="k">Already beaten</div><div class="v">${beaten.length ? esc(beaten.join(", ")) : "none yet"}</div>` +
    (claims.length ? `<div class="k">Unreviewed claims</div><div class="v">${claims.map(c => esc(c.label) + " " + fmt.rep(c.val)).join("; ")}</div>` : "") +
    (pub.lower_bound ? `<div class="k">Known lower bound</div><div class="v">${pub.lower_bound.value} (${esc(pub.lower_bound.who)})</div>` : "") +
    (prompt ? `<div class="k">Run's prompt target</div><div class="v">${prompt.val} (shown to the coder; not part of the reward)</div>` : "") + `</div>` +
    (hiddenNote() ? `<p class="sub" style="margin-top:8px"><span class="pill warn">filtered</span> ${hiddenNote()}</p>` : "") +
    `<p class="sub" style="margin-top:10px">${pubText("league_foot") || "Values are quoted as printed (lower is better)."}</p>` });
  root.append(h("div", { style: { marginTop: "14px" } }, card("How this run compares with published results", "reference lines and gaps everywhere follow the top-bar selector", h("div", { class: "grid g-main" }, h("div", null, host), txt))));
  registerChart(host, () => {
    const W = Math.max(380, host.clientWidth || 600), rowH = 27, H = 34 + items.length * rowH, lw = 226, rw = 96;
    host.innerHTML = "";
    const svg = sv("svg", { viewBox: `0 0 ${W} ${H}`, width: W, height: H }, host);
    const minV0 = Math.min(...items.map(i => i.val)), maxV0 = Math.max(...items.map(i => i.val));
    const hiV = Math.min(maxV0, best + Math.max(5 * (best - minV0), 0.25 * (maxV0 - minV0)));
    const lo = minV0 - 0.04 * (hiV - minV0), hi = hiV + 0.04 * (hiV - minV0);
    const xs = linear(lo, hi, lw, W - rw), tk = niceTicks(lo, hi, 5), f = tickFmt(tk);
    tk.forEach(t => { sv("line", { x1: xs(t), x2: xs(t), y1: 18, y2: H - 6, stroke: cssVar("--line"), "stroke-dasharray": "2 3" }, svg); const tx = sv("text", { x: xs(t), y: 12, "text-anchor": "middle" }, svg); tx.textContent = f(t); });
    const lx = xs(lead.val), vfmt = v => String(+v.toFixed(PROB.dec + 4));
    items.forEach((it, i) => {
      const y = 30 + i * rowH;
      if (it.me) sv("rect", { x: 0, y: y - 14, width: W, height: rowH - 2, rx: 6, fill: cssVar("--accent-soft") }, svg);
      const t = sv("text", { x: lw - 10, y: y + 4, "text-anchor": "end", style: `fill:${it.me ? cssVar("--accent") : it.claim ? cssVar("--ink-3") : cssVar("--ink")};font-weight:${it.me ? 700 : 400};font-style:${it.claim ? "italic" : "normal"}` }, svg); t.textContent = it.label;
      const off = xs(it.val) > W - rw + 4, px = Math.min(xs(it.val), W - rw + 4);
      sv("line", { x1: lx, x2: px, y1: y, y2: y, stroke: it.me ? cssVar("--accent") : cssVar("--line-2"), "stroke-width": it.me ? 2.4 : 1.4, "stroke-dasharray": it.claim ? "3 3" : "" }, svg);
      const c = sv("circle", { cx: px, cy: y, r: it.me ? 7 : 5, fill: it.claim ? cssVar("--surface") : it.me ? cssVar("--accent") : (it === lead ? cssVar("--good") : cssVar("--ink-3")), stroke: it.claim ? cssVar("--s2") : cssVar("--surface"), "stroke-width": 2, class: "refdot" }, svg);
      bindTip(c, () => `<b>${esc(it.label)}</b><div>${vfmt(it.val)}</div>${it.ref ? `<div class="mut">${esc(it.ref.note)}</div>` : ""}<div class="mut">${it.me ? "gap to leading " + fmt.gap(it.val - lead.val) : "this run is " + fmt.gap(best - it.val) + " vs it"}</div>`);
      const v = sv("text", { x: px + 11, y: y + 4, style: `font-size:11px;fill:${cssVar("--ink-2")}` }, svg); v.textContent = vfmt(it.val) + (off ? " →" : "");
    });
    const r = REF(); if (xs(r.val) >= lw && xs(r.val) <= W - rw + 4) sv("line", { x1: xs(r.val), x2: xs(r.val), y1: 18, y2: H - 6, stroke: cssVar("--good"), "stroke-width": 1.4, "stroke-dasharray": "5 4" }, svg);
  });
}


/* GPU / machine specification of the run */
function hardwareCard(root, HW, done) {
  if (!HW || !HW.n_gpus) return;
  const E = D.main.elapsed || {}, spentH = (E.total != null ? E.total : 0) / 3600;
  const rows = [
    ["GPUs", `<b>${HW.n_gpus} × ${esc(HW.name)}</b> <span class="sub">· ${esc(HW.family || "")}</span>`],
    ["Memory per GPU", HW.mem_gib ? `${HW.mem_gib.toFixed(2)} GiB <span class="sub">as reported by vLLM at start-up</span>` : "–"],
    ["Compute capability", HW.capability ? `${esc(HW.capability)} <span class="sub">(logged by vLLM)</span>` : "<span class='sub'>not logged</span>"],
    ["Hardware profile", `${esc(HW.profile || "RTX6000")}${HW.profile_defaulted ? ' <span class="sub">(not in this run\'s config - the repo default for runs of that time)</span>' : ""}`],
    ["Model name basis", `<span class="sub">${esc(HW.basis || "")}</span>`],
    ["GPU ids used", `${(HW.gpu_ids || []).join(", ")} <span class="sub">- all ${HW.n_gpus} train the coder (replicated LoRA) and generate (sequential sharing)</span>`],
    ["Coder generation", HW.coder_engines ? `${HW.coder_engines} vLLM engine${HW.coder_engines > 1 ? "s" : ""}, tensor-parallel ${HW.coder_tp} (${HW.coder_engines * HW.coder_tp} GPUs)` : "vLLM"],
    ["Strategist generation", HW.strategist_engines ? `${HW.strategist_engines} vLLM engine${HW.strategist_engines > 1 ? "s" : ""}, tensor-parallel ${HW.strategist_tp}` : "–"],
    ["Evaluation (CPU)", HW.eval_processes ? `${HW.eval_processes} sandbox processes on ${HW.eval_cpus_used || "?"} CPUs (config eval_cpus = ${CFG.eval_cpus || "?"})` : `${CFG.eval_cpus || "?"} CPUs configured`],
    HW.host_offload_gib ? ["Host RAM for offload", `${HW.host_offload_gib} GiB budget for activation / weight offload`] : null,
    E.total != null ? ["GPU-hours so far", `<b>≈ ${Math.round(spentH * HW.n_gpus).toLocaleString("en-US")} GPU-h</b> <span class="sub">(${HW.n_gpus} GPUs × ${spentH.toFixed(1)} h wall clock incl. restarts)</span>`] : null,
  ].filter(Boolean);
  const chips = h("div", { class: "row-flex", style: { gap: "6px", margin: "10px 0 2px" } }, (HW.gpu_ids || []).map(id => h("span", { class: "pill acc", title: `${HW.name} · ${HW.mem_gib ? HW.mem_gib.toFixed(1) + " GiB" : ""}` }, `GPU ${id}`)));
  root.append(h("div", { style: { marginBottom: "14px" } }, card("Hardware used", "machine specification of this run", h("div", null,
    h("div", { class: "kv", style: { gridTemplateColumns: "170px 1fr" }, html: rows.map(([k, v]) => `<div class="k">${k}</div><div class="v">${v}</div>`).join("") }), chips))));
}
