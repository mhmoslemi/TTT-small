"use strict";
/* ---- Search tab: PUCT selection, JEV diversity gate, genealogy tree, score landscape ---------- */
let _searchOpts = { label: "same_core_idea", landscape: "valid", treeColor: "raw", treeGrow: "all" };

/* signed symmetric-log axis for "gap to the reference value": 0 = equal to the reference, negative = better than it */
function gapOf(raw) { const g = raw - TARGET(), K = PROB.K; return Math.sign(g) * Math.log10(1 + Math.abs(g) / K); }
function gapInv(y) { return Math.sign(y) * (Math.pow(10, Math.abs(y)) - 1) * PROB.K; }
function gapTicks() { const K = PROB.K, ms = [0.3, 1, 3, 10, 30, 100, 300, 1000, 3000, 1e4, 3e4, 1e5]; return [...ms.slice().reverse().map(m => -m * K), 0, ...ms.map(m => m * K)].map(g => Math.sign(g) * Math.log10(1 + Math.abs(g) / K)); }
function gapFmt(y) { const g = gapInv(y), a = Math.abs(g); if (a < 1e-12) return "0"; return (g < 0 ? "−" : "+") + (+a.toPrecision(1)); }
function gapDomain(ys) { const v = ys.filter(isFinite); if (!v.length) return [0, 1]; return [Math.min(...v) - 0.25, Math.max(...v) + 0.25]; }

registerTab({
  id: "search", label: "Search: PUCT + JEV", ico: "⌥",
  async render(root) {
    const step = S.step, st = ST[step];
    root.append(pageHead("Search: choosing which programs to extend",
      `The search keeps an <b>archive</b> of valid programs and picks 8 parents per step. <b>PUCT</b> ranks the archive (exploit the best known child, explore via a rank prior). ` +
      (RUN.hasJev
        ? `<b>JEV</b> then acts as a gate: GLiDE-9B reads two programs and says whether they share the <i>same core idea</i>; a candidate that is too similar to an already-accepted parent is skipped, so the 8 parents cover different ideas.`
        : `The sampler then takes states in rank order but <b>blocks the lineage</b> of every pick (its ancestors and descendants), so the 8 parents sit on different branches of the tree. <i>This run has no JEV diversity gate.</i>`)));
    selectionSection(root, step);
    jevTreeSection(root, step);
    await jevSection(root, step);
    treeSection(root);
    landscapeSection(root);
  }
});

/* ---------------------------------------------------------------- shared helpers ---------- */
function pucContext(step) {
  const ps = parentsOf(step), N = step > 0 && ST[step - 1] ? ST[step - 1].archive_size : 8, T = 512 * step, c0 = CFG.puct_c || 1;
  const sc = ps.length ? mean(ps.map(p => p.exploration_bonus * (1 + p.visit_count) / (c0 * p.prior * Math.sqrt(1 + T)))) : 1; // recovered 'scale'
  return { ps, N, T, c0, sc };
}
/* the 8 parents re-scored at the current what-if exploration constant */
function pucRows(step) {
  const ctx = pucContext(step), c = P.c != null ? P.c : ctx.c0;
  const rows = ctx.ps.map(p => {
    const bonus = c * ctx.sc * p.prior * Math.sqrt(1 + ctx.T) / (1 + p.visit_count);
    return Object.assign({}, p, { bonus2: bonus, score2: p.q_value + bonus, rank: ctx.N - p.prior * ctx.N * (ctx.N + 1) / 2 });
  });
  const order = rows.slice().sort((a, b) => b.score2 - a.score2).map(r => r.group);
  return { rows, order, ctx, c };
}
/* replay the JEV scan of a step at a given threshold (null = the recorded one).
   With the rebuilt PUCT window (collect -> archive_sim.py, validated against every recorded scan) the replay is complete and follows
   search/jev_sampler.py: pass 1 skips candidates whose lineage an accepted parent blocks, pass 2 relaxes that, and whatever is still
   missing is filled from the rest of the window without comparisons.  Candidates the run never compared have no GLiDE judgment:
   they are accepted provisionally and flagged `unjudged` (substitutes).  Without a window the old record-only replay is used. */
let _lin = null;
function lineageOf(id) {            // ancestors + descendants + itself, from the search tree
  if (!_lin) _lin = {};
  if (_lin[id]) return _lin[id];
  const out = new Set([id]), i0 = TREE.idx[id];
  if (i0 != null) {
    let k = TREE.nodes[i0][0]; while (k != null && k >= 0) { out.add(TREE.ids[k]); k = TREE.nodes[k][0]; }
    const st = [i0]; while (st.length) { const x = st.pop(); (TREE.kids[x] || []).forEach(c => { if (!out.has(TREE.ids[c])) { out.add(TREE.ids[c]); st.push(c); } }); }
  }
  return (_lin[id] = out);
}
function windowReplay(jev, t, cmp) {
  const win = jev.window, REQ = jev.requested || 8, scanned = {};
  jev.decisions.forEach(d => { scanned[d.candidate_id] = d; });
  const picks = [], rejected = new Set(), blocked = new Set(), sim = [];
  const compare = id => {
    let verdict = "accept", match = null, unknown = false;
    for (const a of picks) { const p = cmp[id + "|" + a]; if (!p) { unknown = true; continue; } if (p.same_core_idea >= t) { verdict = "reject"; match = a; break; } }
    return { verdict, match, unknown };
  };
  for (const relax of [false, true]) {
    for (const id of win) {
      if (picks.length >= REQ) break;
      if (picks.includes(id) || rejected.has(id) || (!relax && blocked.has(id))) continue;
      const d = scanned[id], c = compare(id), unjudged = !d;
      const e = { id, actual: d ? d.accepted : null, actualMatch: d ? d.matched_parent_id : null, verdict: c.verdict, match: c.match, unknown: c.verdict === "accept" && c.unknown && !unjudged, unjudged: unjudged && c.verdict === "accept", pass: relax ? 2 : 1 };
      sim.push(e);
      if (c.verdict === "accept") { picks.push(id); lineageOf(id).forEach(x => blocked.add(x)); } else rejected.add(id);
    }
  }
  for (const id of win) {           // window exhausted: fill in PUCT order without any comparison
    if (picks.length >= REQ) break;
    if (picks.includes(id)) continue;
    const d = scanned[id]; picks.push(id);
    sim.push({ id, actual: d ? d.accepted : null, actualMatch: d ? d.matched_parent_id : null, verdict: "accept", match: null, unknown: false, unjudged: !d, fallback: true, pass: 3 });
  }
  return { sim, accepted: picks };
}
function windowTrusted(step) {      // the window must reproduce the recorded parents at the recorded threshold, otherwise ignore it
  const jev = ST[step].jev; if (jev._wt !== undefined) return jev._wt;
  let ok = false;
  if (jev.window_ok && jev.window && jev.window.length) {
    const cmp = {}; jev.comparisons.forEach(c => { cmp[c.candidate_id + "|" + c.parent_id] = c.probabilities; });
    const r = windowReplay(jev, jev.threshold, cmp), sel = new Set(jev.selected || []);
    ok = r.accepted.length === sel.size && r.accepted.every(x => sel.has(x));
  }
  return (jev._wt = ok);
}
function jevReplay(step, thr) {
  const jev = ST[step] && ST[step].jev;
  if (!jev || !jev.decisions || !jev.decisions.length) return null;
  const t = thr != null ? thr : jev.threshold, cmp = {};
  jev.comparisons.forEach(c => { cmp[c.candidate_id + "|" + c.parent_id] = c.probabilities; });
  const meta = { thr: t, recorded: jev.threshold, recRej: jev.decisions.filter(d => !d.accepted).length, recAcc: jev.decisions.filter(d => d.accepted).length, cmp, decs: jev.decisions, jev };
  if (windowTrusted(step)) {
    const r = windowReplay(jev, t, cmp), nAcc = r.sim.filter(x => x.verdict === "accept").length;
    return Object.assign(meta, { sim: r.sim, accepted: r.accepted, nAcc, nRej: r.sim.length - nAcc, hasWindow: true, nUnjudged: r.sim.filter(x => x.unjudged).length });
  }
  const accepted = [], sim = [];
  for (const d of jev.decisions) {
    let verdict = "accept", match = null, unknown = false;
    for (const a of accepted) {
      const p = cmp[d.candidate_id + "|" + a];
      if (!p) { unknown = true; continue; }
      if (p.same_core_idea >= t) { verdict = "reject"; match = a; break; }
    }
    if (verdict === "accept" && accepted.length >= jev.requested) break;
    sim.push({ id: d.candidate_id, actual: d.accepted, actualMatch: d.matched_parent_id, verdict, match, unknown: verdict === "accept" && unknown });
    if (verdict === "accept") accepted.push(d.candidate_id);
  }
  const nAcc = sim.filter(x => x.verdict === "accept").length;
  return Object.assign(meta, { sim, accepted, nAcc, nRej: sim.length - nAcc, hasWindow: false, nUnjudged: 0 });
}
function thrNow(step) { return P.thr != null ? P.thr : (ST[step].jev ? ST[step].jev.threshold : 0.68); }

/* ---------------------------------------------------------------- A. PUCT ------------------ */
function selectionSection(root, step) {
  const ctx0 = pucContext(step), ps = ctx0.ps;
  if (!ps.length) { root.append(card("Parent selection", null, h("div", { class: "empty" }, "No parents recorded."))); return; }
  const { N, T, c0, sc } = ctx0;
  const box = h("div"), chart = h("div", { class: "chart" });
  const holder = card(`PUCT: the 8 parents of step ${step}`, `archive N = ${N} · rollouts so far T = ${fmt.n(T)} · recovered scale = ${fmt.f(sc, 4)} (non-seed reward range)`, h("div", null,
    paramSlider("exploration constant c", 0, 20, 0.5, v => v.toFixed(1), () => P.c != null ? P.c : c0, (v, row) => setParam({ c: v }, row)), box));
  const side = card("Rank prior P(rank)", "linear in rank, so the 8 picks differ mostly through P", chart);
  root.append(h("div", { class: "grid g-main", style: { marginBottom: "14px" } }, holder, side));
  const draw = () => {
    box.innerHTML = "";
    const { rows, order, c } = pucRows(step);
    box.append(h("div", { class: "eq", html: esc(`score = Q + c · scale · P · √(1+T) / (1+n)   =   Q + ${c.toFixed(1)} · ${sc.toFixed(4)} · P · ${Math.sqrt(1 + T).toFixed(1)} / (1+n)`) }));
    const t = table([
      { label: "grp", get: p => String(p.group), num: true },
      { label: "parent", get: p => h("a", { href: "javascript:void 0", class: "mono", onclick: e => { e.stopPropagation(); openNode(p.parent_id); } }, fmt.id(p.parent_id)) },
      { label: "made in", get: p => p.parent_is_seed ? "seed" : "step " + p.parent_timestep, num: true },
      { label: PROB.metric, get: p => fmt.raw(p.parent_raw_score), num: true },
      { label: "Q", get: p => fmt.f(p.q_value, 5), num: true, title: "best valid child reward seen (= own reward while n=0)" },
      { label: "n", get: p => String(p.visit_count), num: true, title: "rollouts spent below this state" },
      { label: "rank", get: p => String(Math.round(p.rank)), num: true, title: "rank in the archive by reward, recovered from P" },
      { label: "P", get: p => fmt.f(p.prior, 5), num: true },
      { label: "bonus", get: p => fmt.f(p.bonus2, 4), num: true },
      { label: "score", get: p => fmt.f(p.score2, 5), num: true },
      { label: "order", get: p => String(order.indexOf(p.group) + 1), num: true, title: "rank by score at the chosen c (among these 8)" },
      ...(RUN.hasJev ? [{ label: "JEV", get: p => { const d = (p.jev || {}).decision; return pill(d === "accepted" ? "accepted" : d === "no_comparison" ? "first pick" : d === "step_zero_puct" ? "PUCT only" : d || "?", d === "accepted" || d === "no_comparison" ? "good" : d === "duplicate_fallback" ? "warn" : ""); } }] : []),
    ], rows, { selected: rows.findIndex(r => r.group === S.group), onRow: p => { S.group = p.group; syncHash(); draw(); } });
    box.append(h("div", { class: "scroll" }, t));
    box.append(h("p", { class: "sub", style: { marginTop: "8px" } }, ps.every(p => !p.parent_is_seed && p.visit_count === 0) ? "All 8 picks are fresh children of the previous step (n = 0), so Q differs by ~1e-5 and the rank prior P decides the ordering. A larger c lets exploration dominate; c = 0 would be pure greed. Only the 8 accepted parents are re-ranked here - rejected candidates' PUCT numbers are not stored." : "At step 0 the 8 seeds all have reward 0.5, so Q ties and only the rank prior P separates them. A larger c lets exploration dominate; c = 0 would be pure greed. Only the 8 accepted parents are re-ranked here."));
    if (P.c != null && P.c !== c0) box.append(h("div", { class: "what-if" }, `what-if: c = ${P.c.toFixed(1)} (run used ${c0}) - also applied in the Pipeline tab`));
  };
  draw();
  onParam(patch => { if ("c" in patch) draw(); });

  lineChart(chart, {
    height: 250, clip: true, xLabel: "rank in archive (0 = best reward)", yLabel: "prior P", margin: { l: 56 }, xDomain: [0, Math.max(40, Math.max(...ps.map(p => N - p.prior * N * (N + 1) / 2)) * 1.4)],
    yFmt: v => v.toFixed(4),
    series: [
      { name: "P(rank)", color: S_COL(0), pts: range(0, Math.min(N, 400)).map(r => [r, (N - r) / (N * (N + 1) / 2)]), dots: false, width: 2 },
      { name: "selected parents", color: S_COL(1), width: 0, dotR: 5, pts: ps.map(p => [N - p.prior * N * (N + 1) / 2, p.prior]) },
    ],
    tipHead: x => `rank ${Math.round(x)}`,
  });
}

/* ---------------------------------------------------------------- B. JEV ------------------- */
async function jevSection(root, step) {
  if (!RUN.hasJev) return;
  const jev = ST[step].jev;
  const wrap = h("div", { class: "card", style: { marginBottom: "14px" } });
  root.append(wrap);
  if (!jev || !jev.decisions || !jev.decisions.length) {
    wrap.append(h("div", { class: "card-head" }, h("h2", null, "JEV diversity gate")),
      h("div", { class: "hint" }, step === 0 ? "Step 0 uses plain PUCT on the 8 seeds - there is nothing to compare yet." : "No JEV decisions stored for this step."));
    return;
  }
  const thr0 = jev.threshold;
  wrap.append(h("div", { class: "card-head" }, h("h2", null, `JEV judgments in detail — step ${step}`), h("span", { class: "sub" }, `GLiDE-9B · window = top ${jev.candidate_limit} PUCT states · recorded threshold ${thr0}`)));
  wrap.append(paramSlider("same-core threshold", 0.3, 1, 0.01, v => v.toFixed(2), () => thrNow(step), (v, row) => setParam({ thr: v }, row)));
  const body = h("div"); wrap.append(body);
  let reveal = null, timer = null;
  const lbl = () => _searchOpts.label;
  const draw = () => {
    body.innerHTML = "";
    const R0 = jevReplay(step, thrNow(step)), thr = R0.thr, decs = R0.decs, cmp = R0.cmp;
    const nShow = reveal == null ? decs.length : reveal;
    body.append(h("div", { class: "row-flex", style: { margin: "4px 0 8px" } },
      pill(`recorded @ ${thr0}: ${R0.recAcc} accepted · ${R0.recRej} rejected`, "acc"),
      pill(`replay @ ${thr.toFixed(2)}: ${R0.nAcc} accepted · ${R0.nRej} rejected`, thr === thr0 ? "good" : "warn"),
      h("button", { class: "btn sm" + (timer ? " on" : ""), onclick: () => scan() }, timer ? "■ stop" : "▶ animate the scan"),
      P.thr != null ? h("button", { class: "btn sm", onclick: () => setParam({ thr: null }) }, "use recorded threshold") : null,
      P.thr != null && P.thr !== thr0 ? h("span", { class: "what-if" }, "what-if threshold - applied to all steps, also in Pipeline") : null));
    const accOrder = decs.filter(d => d.accepted).map(d => d.candidate_id);
    const rowsC = decs.map(d => d.candidate_id).slice(0, nShow);
    const host = h("div", { class: "chart", style: { overflowX: "auto" } });
    body.append(host);
    const pv = (i, j) => { const p = cmp[rowsC[i] + "|" + accOrder[j]]; return p ? p[lbl()] : null; };
    heatmap(host, {
      rows: rowsC.map((id, i) => `#${i + 1} ${fmt.id(id)}${decs[i].accepted ? " ✓" : " ✗"}`), cols: accOrder.map((_, j) => "P" + (j + 1)), rowLabelW: 116, maxCell: 54, cellH: 24,
      val: (i, j) => pv(i, j),
      color: v => (lbl() === "same_core_idea" && v >= thr) ? cssVar("--crit") : seqRamp(v), text: v => v == null ? null : Math.round(v * 100),
      textColor: v => (lbl() === "same_core_idea" && v >= thr) || v > 0.5 ? "#fff" : cssVar("--ink"),
      tip: (i, j, v) => { if (v == null) return `<b>candidate #${i + 1} vs parent P${j + 1}</b><div class="mut">not compared (it was rejected earlier, or P${j + 1} was accepted after it)</div>`; const p = cmp[rowsC[i] + "|" + accOrder[j]]; return `<b>candidate ${fmt.id(rowsC[i])} vs parent ${fmt.id(accOrder[j])}</b><div>same core idea: <b>${fmt.f(p.same_core_idea, 3)}</b></div><div>meaningful variant: ${fmt.f(p.meaningful_variant, 3)}</div><div>different approach: ${fmt.f(p.different_approach, 3)}</div><div class="mut">${p.input_tokens} input tokens</div>`; },
      onClick: (i) => openNode(rowsC[i]),
    });
    body.append(h("div", { class: "row-flex", style: { margin: "6px 0" } }, seg([{ value: "same_core_idea", label: "same core idea" }, { value: "meaningful_variant", label: "meaningful variant" }, { value: "different_approach", label: "different approach" }], lbl(), v => { _searchOpts.label = v; draw(); }),
      h("span", { class: "sub" }, "cell = P(label) judged by GLiDE for (row candidate vs column accepted parent) · click a row to open the candidate")));
    const rt = table([
      { label: "#", get: (r, i) => String(i + 1), num: true },
      { label: "candidate", get: r => h("a", { href: "javascript:void 0", class: "mono", onclick: () => openNode(r.id) }, fmt.id(r.id)) },
      { label: "raw", get: r => RN[r.id] ? fmt.raw(RN[r.id].raw) : (TREE.idx[r.id] != null ? fmt.raw(TREE.nodes[TREE.idx[r.id]][1]) : "–"), num: true },
      { label: "recorded", get: r => r.actual == null ? pill("not scanned", "") : pill(r.actual ? "accepted" : "rejected", r.actual ? "good" : "bad") },
      { label: "matched parent", get: r => r.actualMatch ? fmt.id(r.actualMatch) : "" },
      { label: `replay @ ${thr.toFixed(2)}`, get: r => r.unjudged ? pill(r.fallback ? "substitute (window fill)" : "substitute (never judged)", "acc") : pill(r.verdict === "accept" ? (r.unknown ? "accept (partly unknown)" : "accept") : "reject", r.verdict === "accept" ? (r.unknown ? "warn" : "good") : "bad") },
      { label: "differs?", get: r => ((r.verdict === "accept") !== !!r.actual) ? pill("changes", "warn") : "" },
    ], R0.sim.slice(0, reveal == null ? R0.sim.length : reveal));
    body.append(h("div", { class: "scroll", style: { maxHeight: "300px" } }, rt));
    body.append(h("p", { class: "sub", style: { marginTop: "6px" } }, "The replay walks candidates in the recorded scan order and rejects one when P(same core idea) ≥ threshold against any already-accepted parent. Pairs that the original run never computed (because it had already stopped on an earlier match) are skipped and flagged as 'partly unknown' - raising the threshold above the recorded one can therefore under-estimate rejections."));
  };
  const scan = () => {
    if (timer) { clearInterval(timer); timer = null; reveal = null; draw(); return; }
    reveal = 1; draw();
    timer = setInterval(() => {
      if (!body.isConnected) { clearInterval(timer); timer = null; return; }
      reveal++;
      if (reveal > jev.decisions.length) { clearInterval(timer); timer = null; reveal = null; }
      draw();
    }, 650);
  };
  draw();
  onParam(patch => { if ("thr" in patch && !timer) draw(); });

  // global distribution of judgments + rejections per step (recorded vs replay at the current what-if threshold)
  const all = [];
  STEPS.concat(ALL_STEPS.filter(s => !ST[s].completed)).forEach(s => (ST[s].jev ? ST[s].jev.comparisons : []).forEach(c => all.push(Object.assign({ step: s }, c.probabilities))));
  if (all.length) {
    const hc = h("div", { class: "chart" }), rc = h("div", { class: "chart" });
    const hsel = h("span");
    root.append(h("div", { class: "grid g2", style: { marginBottom: "14px" } },
      card(`All ${all.length} JEV judgments in the run`, "distribution of P(same core idea) with the current threshold marked", hc),
      card("Candidates rejected per step", "recorded vs. replay at the current what-if threshold", h("div", null, rc, h("div", { id: "rejLegend" })))));
    const drawH = () => {
      const bins = 20, hist = new Array(bins).fill(0); all.forEach(a => hist[Math.min(bins - 1, Math.floor(a.same_core_idea * bins))]++);
      const t = thrNow(step);
      barChart(hc, { height: 220, cats: hist.map((_, i) => (i / bins).toFixed(2)), series: [{ name: "pairs", color: S_COL(6), colors: hist.map((_, i) => (i + 1) / bins > t ? cssVar("--crit") : S_COL(6)), vals: hist }], xLabel: `P(same core idea) - red = at/above ${t.toFixed(2)}`, yLabel: "pairs", tip: (i, si, v) => `<b>${(i / bins).toFixed(2)} – ${((i + 1) / bins).toFixed(2)}</b><div>${v} pairs</div>` });
    };
    const drawR = () => {
      const stepsJ = ALL_STEPS.filter(s => ST[s].jev && ST[s].jev.decisions && ST[s].jev.decisions.length);
      const rec = stepsJ.map(s => ST[s].jev.decisions.filter(d => !d.accepted).length), rep = stepsJ.map(s => jevReplay(s, thrNow(s)).nRej);
      const thrKnown = P.thr != null;
      barChart(rc, { height: 220, cats: stepsJ.map(String), series: [{ name: "recorded", color: S_COL(7), vals: rec }].concat(thrKnown ? [{ name: "replay @ " + P.thr.toFixed(2), color: S_COL(3), vals: rep }] : []), xLabel: "step", yLabel: "rejected candidates",
        tip: (i, si, v) => `<b>step ${stepsJ[i]}</b><div>${si === 0 ? "recorded" : "replay"}: ${v} rejected · recorded threshold ${ST[stepsJ[i]].jev.threshold}</div>`, onClick: i => setState({ step: stepsJ[i] }) });
      const lg = $("#rejLegend"); if (lg) { lg.innerHTML = ""; lg.append(legend([{ color: S_COL(7), label: "recorded" }].concat(thrKnown ? [{ color: S_COL(3), label: "replay @ " + P.thr.toFixed(2) }] : []))); }
    };
    drawH(); drawR();
    onParam(patch => { if ("thr" in patch) { drawH(); drawR(); } });
  }
}

/* ---------------------------------------------------------------- C. genealogy ------------- */
function genealogy(limit) {
  // nodes: selected parents + best-of-step + all their ancestors, optionally only those created up to step `limit`
  const keep = new Set(), selCount = {};
  const add = id => { const i = TREE.idx[id]; if (i == null) return; let k = i; while (k >= 0 && !keep.has(k)) { keep.add(k); k = TREE.nodes[k][0]; } };
  ALL_STEPS.filter(s => s <= limit).forEach(s => parentsOf(s).forEach(p => { add(p.parent_id); selCount[p.parent_id] = (selCount[p.parent_id] || 0) + 1; }));
  STEPS.filter(s => s <= limit).forEach(s => { const b = ST[s].best; if (b && b.state_id) add(b.state_id); });
  [...keep].forEach(i => { if (TREE.nodes[i][2] > limit) keep.delete(i); });
  const lastS = STEPS.filter(s => s <= limit).pop();
  const lastBest = lastS != null ? ST[lastS].best : null;
  const bestIdx = lastBest ? TREE.idx[lastBest.state_id] : null;
  const path = new Set(); { let k = bestIdx; while (k != null && k >= 0) { path.add(k); k = TREE.nodes[k][0]; } }
  const kids = {}; keep.forEach(i => { const p = TREE.nodes[i][0]; if (p >= 0 && keep.has(p)) (kids[p] = kids[p] || []).push(i); });
  const roots = [...keep].filter(i => TREE.nodes[i][0] < 0 || !keep.has(TREE.nodes[i][0]));
  let leafY = 0; const pos = {};
  const lay = i => {
    const ch = (kids[i] || []).sort((a, b) => (path.has(b) ? 1 : 0) - (path.has(a) ? 1 : 0) || TREE.nodes[a][2] - TREE.nodes[b][2]);
    if (!ch.length) pos[i] = leafY++; else { ch.forEach(lay); pos[i] = mean(ch.map(c => pos[c])); }
  };
  roots.forEach(lay);
  return { keep, kids, roots, pos, leaves: leafY, path, bestIdx, selCount };
}

function treeSection(root) {
  const grow = _searchOpts.treeGrow === "step";
  const limit = grow ? S.step : Infinity;
  const G = genealogy(limit);
  const nowSel = new Set(grow ? parentsOf(S.step).map(p => p.parent_id) : []);
  const host = h("div", { style: { overflow: "auto", maxHeight: "680px", border: "1px solid var(--line)", borderRadius: "10px", background: "var(--surface)" } });
  const ctl = h("div", { class: "row-flex", style: { marginBottom: "8px" } },
    seg([{ value: "raw", label: "color: upper bound" }, { value: "step", label: "color: step created" }], _searchOpts.treeColor, v => { _searchOpts.treeColor = v; drawTree(); }),
    seg([{ value: "all", label: "whole run" }, { value: "step", label: `grow: up to step ${S.step}` }], _searchOpts.treeGrow || "all", v => { _searchOpts.treeGrow = v; renderTab(); }),
    h("span", { class: "sub" }, `${G.keep.size} states: every state chosen as a parent, each step's best, and their ancestors (of ${fmt.n(TREE.ids.length)} programs in the full tree). Orange path leads to the best${grow ? " so far" : ""}${grow ? "; pulsing rings = the 8 parents chosen at the selected step - press ▶ play steps to watch the search grow" : ""}.`));
  root.append(card("Genealogy of the search", "left → right = step created · click a node to open its program", h("div", null, ctl, host)));
  const drawTree = () => {
    host.innerHTML = "";
    if (!G.keep.size) { host.append(h("div", { class: "empty" }, "nothing to show yet")); return; }
    const dx = 96, dy = 20, ml = 40, mt = 30;
    const maxLevel = Math.max(...[...G.keep].map(i => TREE.nodes[i][2] + 1));
    const W = ml + (maxLevel + 1) * dx + 40, H = mt + G.leaves * dy + 24;
    const svg = sv("svg", { width: W, height: H, viewBox: `0 0 ${W} ${H}` });
    host.append(svg);
    const X = i => ml + (TREE.nodes[i][2] + 1) * dx, Y = i => mt + G.pos[i] * dy;
    for (let l = 0; l <= maxLevel; l++) { const t = sv("text", { x: ml + l * dx, y: 16, "text-anchor": "middle", style: "font-size:11px;fill:" + cssVar("--ink-3") }, svg); t.textContent = l === 0 ? "seeds" : "step " + (l - 1); sv("line", { x1: ml + l * dx, x2: ml + l * dx, y1: 22, y2: H - 8, stroke: cssVar("--line"), "stroke-dasharray": "2 4" }, svg); }
    const nonSeed = [...G.keep].filter(i => TREE.nodes[i][0] >= 0).map(i => TREE.nodes[i][1]).filter(x => x > 0);
    const lo = nonSeed.length ? Math.min(...nonSeed) : 0, hi = nonSeed.length ? Math.max(...nonSeed) : 1;
    const colOf = i => { const n = TREE.nodes[i]; if (n[0] < 0) return cssVar("--ink-3"); if (_searchOpts.treeColor === "step") return seqRamp(n[2] / Math.max(1, maxLevel - 1)); return seqRamp(1 - clamp((n[1] - lo) / Math.max(1e-9, hi - lo), 0, 1)); };
    G.keep.forEach(i => { const p = TREE.nodes[i][0]; if (p >= 0 && G.keep.has(p)) {
      const x0 = X(p), y0 = Y(p), x1 = X(i), y1 = Y(i), mx = (x0 + x1) / 2, onP = G.path.has(i) && G.path.has(p);
      const e = sv("path", { d: `M${x0},${y0}C${mx},${y0} ${mx},${y1} ${x1},${y1}`, fill: "none", stroke: onP ? cssVar("--s2") : cssVar("--line-2"), "stroke-width": onP ? 2.6 : 1.2 }, svg);
      if (ANIM) { e.setAttribute("pathLength", "1"); e.setAttribute("class", "a-line"); }
    } });
    G.keep.forEach(i => {
      const n = TREE.nodes[i], id = TREE.ids[i], isSel = G.selCount[id], r = n[3] >= 0 ? R[n[3]] : null;
      if (nowSel.has(id)) sv("circle", { cx: X(i), cy: Y(i), r: 8, fill: "none", stroke: cssVar("--accent"), "stroke-width": 2, class: "ring-pulse" }, svg);
      const c = sv("circle", { cx: X(i), cy: Y(i), r: n[0] < 0 ? 6 : (isSel ? 6 : 4), fill: colOf(i), stroke: G.path.has(i) ? cssVar("--s2") : (isSel ? cssVar("--ink") : cssVar("--surface")), "stroke-width": G.path.has(i) ? 2.4 : (isSel ? 1.4 : 1), style: "cursor:pointer" }, svg);
      if (ANIM) { c.setAttribute("class", "a-fade"); c.style.animationDelay = (0.05 * (n[2] + 1)) + "s"; }
      c.addEventListener("mousemove", ev => tipShow(`<b>${n[0] < 0 ? "seed" : "step " + n[2]}</b> <span class="mut">${fmt.id(id)}</span><div>upper bound <b>${fmt.raw(n[1])}</b></div>${r ? `<div class="mut">parent group ${r.pg} · S${r.sidx} · ${r.phase}</div>` : ""}${isSel ? `<div>chosen as parent ${isSel}×</div>` : ""}<div class="mut">${(TREE.kids[i] || []).length} programs derived (total)</div>`, ev));
      c.addEventListener("mouseleave", tipHide);
      c.addEventListener("click", () => openNode(id));
    });
    if (G.bestIdx != null && G.keep.has(G.bestIdx)) {
      const bt = sv("text", { x: X(G.bestIdx) + 10, y: Y(G.bestIdx) + 4, style: "font-size:11px;font-weight:600;fill:" + cssVar("--s2") }, svg); bt.textContent = "best " + fmt.f(TREE.nodes[G.bestIdx][1], 5);
      requestAnimationFrame(() => { host.scrollLeft = Math.max(0, X(G.bestIdx) - host.clientWidth + 140); host.scrollTop = Math.max(0, Y(G.bestIdx) - host.clientHeight / 2); });
    }
  };
  drawTree();
}

/* ---------------------------------------------------------------- D. landscape -------------- */
function landscapeSection(root) {
  const host = h("div", { class: "chart" });
  const ctl = h("div", { class: "row-flex", style: { marginBottom: "8px" } },
    seg([{ value: "valid", label: "all valid" }, { value: "improved", label: "beat their parent" }, { value: "adaptive", label: "phase-2 only" }], _searchOpts.landscape, v => { _searchOpts.landscape = v; draw(); }),
    h("span", { class: "sub" }, `colour = parent group of the current step selection is emphasised: group ${S.group}`));
  root.append(card("Score landscape of every valid program", "x = step (jittered), y = gap to the reference value (log scale) · click a dot to open the program", h("div", null, ctl, host)));
  const draw = () => {
    const f = _searchOpts.landscape;
    const pts = R.filter(r => r.valid && r.raw > 0 && r.raw < 5 && (f === "valid" || (f === "improved" && r.improved) || (f === "adaptive" && r.phase === "adaptive"))).map(r => ({
      x: r.step + ((r.id * 2654435761 % 1000) / 1000 - 0.5) * 0.7, y: gapOf(r.raw), c: S_COL(r.pg), a: r.pg === S.group ? 0.85 : 0.35, r: r.pg === S.group ? 3.2 : 2.4, id: r.id, rr: r,
    }));
    scatterChart(host, {
      height: 380, points: pts, xLabel: "step", yLabel: "gap to reference (symlog)", margin: { l: 62 }, yTicks: gapTicks(), yFmt: gapFmt, xDomain: [-0.7, Math.max(...ALL_STEPS) + 0.7], yDomain: gapDomain(pts.map(p => p.y)),
      xTicks: ALL_STEPS, xFmt: v => String(Math.round(v)),
      tip: p => `<b>step ${p.rr.step} · parent ${p.rr.pg} · S${p.rr.sidx}</b><div>upper bound ${fmt.raw(p.rr.raw)}</div><div class="mut">${p.rr.phase}${p.rr.improved ? " · beat parent" : ""}</div>`,
      onPoint: p => openRollout(p.id), hlines: [],
    });
  };
  draw();
  root.append(legend(range(0, 8).map(g => ({ color: S_COL(g), label: "parent " + g }))));
}
