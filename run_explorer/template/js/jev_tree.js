"use strict";
/* ---- JEV on the search tree --------------------------------------------------------------------------------------------
   The candidates the JEV gate scanned in a step, drawn as nodes of the search tree (x = step created, labelled with their upper
   bound), coloured by the verdict at the CURRENT what-if threshold; a second strip shows the same candidates on a zoomed
   upper-bound axis (recorded run vs replay), and two sweeps show how the whole decision changes with the threshold.
   Everything is driven by jevReplay() (tab_search.js) and the shared threshold P.thr, so the matrix below and the Pipeline tab follow too. */
let _jt = { ctx: true, gens: 3 };

function jevTreeSection(root, step) {
  if (!RUN.hasJev) return;
  const jev = ST[step].jev;
  if (!jev || !jev.decisions || !jev.decisions.length) return;

  const decs = jev.decisions, HASWIN = windowTrusted(step), cand = HASWIN ? jev.window.slice() : decs.map(d => d.candidate_id), K = cand.length, REQ = jev.requested || 8;
  const scannedSet = new Set(decs.map(d => d.candidate_id));
  const idx = id => TREE.idx[id];
  const rawOf = id => { const i = idx(id); return i != null ? TREE.nodes[i][1] : null; };
  const stepOf = id => { const i = idx(id); return i != null ? TREE.nodes[i][2] : null; };
  const recAcc = new Set(decs.filter(d => d.accepted).map(d => d.candidate_id));
  const recMatch = {}; decs.forEach(d => { if (!d.accepted && d.matched_parent_id) recMatch[d.candidate_id] = d.matched_parent_id; });
  const thr0 = jev.threshold;
  const rankOf = {}; cand.forEach((id, k) => { rankOf[id] = k + 1; });
  const cmp = {}; (jev.comparisons || []).forEach(c => { cmp[c.candidate_id + "|" + c.parent_id] = c.probabilities; });
  const simOf = (a, b) => { const p = cmp[a + "|" + b]; return p ? p.same_core_idea : null; };

  /* ------------------------------------------------------------ card skeleton */
  const treeHost = h("div", { class: "jt-scroll" });
  const stripHost = h("div", { class: "chart jt-strip" });
  const chips = h("div", { class: "row-flex", style: { margin: "6px 0 2px" } });
  const diffBox = h("div", { class: "jt-diff" });
  const noteBox = h("div", { class: "sub", style: { marginTop: "4px" } });
  const sweepA = h("div", { class: "chart" }), sweepB = h("div", { class: "chart" });
  let timer = null, restore = null;
  const sweepBtn = h("button", { class: "btn sm", onclick: () => sweep() }, "▶ sweep the threshold 0.30 → 1.00");
  const slider = paramSlider("same-core threshold", 0.3, 1, 0.01, v => v.toFixed(2), () => thrNow(step), (v, row) => setParam({ thr: v }, row));
  const ctl = h("div", { class: "row-flex", style: { margin: "2px 0 6px" } }, sweepBtn,
    h("button", { class: "btn sm", onclick: () => { stopSweep(); setParam({ thr: null }); } }, "use recorded " + thr0),
    h("span", { class: "sub" }, "move the slider (or click a sweep chart below) - every node flips live; the same threshold drives the matrix further down and the Pipeline tab"));
  const keyLegend = h("div", { class: "legend", style: { marginTop: "6px" } },
    h("span", { class: "it" }, h("span", { class: "sw", style: { background: "var(--good)", borderRadius: "50%" } }), "accepted → becomes a parent (P1…)"),
    h("span", { class: "it" }, h("span", { class: "sw", style: { background: "var(--crit)", borderRadius: "50%" } }), "rejected - too similar to an accepted parent"),
    HASWIN ? h("span", { class: "it" }, h("span", { class: "sw", style: { background: "var(--accent)", borderRadius: "50%" } }), "substitute - next in the PUCT window, GLiDE never judged it (assumed to pass)") : null,
    h("span", { class: "it" }, h("span", { class: "sw", style: { background: "var(--surface-3)", borderRadius: "50%", border: "1px solid var(--line-2)" } }), "not reached (scan already full)"),
    h("span", { class: "it" }, h("span", { class: "sw", style: { border: "2px solid var(--s4)", borderRadius: "50%", background: "none" } }), "verdict differs from the recorded run"),
    h("span", { class: "it", html: '<svg width="22" height="10" style="display:inline-block"><path d="M1,8 C10,8 14,2 21,2" fill="none" stroke="var(--crit)" stroke-width="1.8" stroke-dasharray="4 3"/></svg> similarity ≥ threshold (P of "same core idea")' }),
    h("span", { class: "it" }, "small dots = ancestors (colour = upper bound)"));
  const gensSel = h("label", { class: "ctl", style: { marginLeft: "4px" } }, "ancestors shown", (() => { const sel = h("select", null, [[2, "2 generations"], [3, "3 generations"], [5, "5 generations"], [Infinity, "all the way to the seeds"]].map(([v, t]) => h("option", { value: String(v), selected: v === _jt.gens }, t))); sel.addEventListener("change", () => { _jt.gens = sel.value === "Infinity" ? Infinity : +sel.value; buildTree(); update(); }); return sel; })());
  const ctxChk = h("label", { class: "ctl", style: { marginLeft: "6px" } }, (() => { const c = h("input", { type: "checkbox" }); c.checked = _jt.ctx; c.addEventListener("change", () => { _jt.ctx = c.checked; drawStrip(); update(); }); return c; })(), "show the whole archive behind the strip");

  root.append(card(`JEV on the search tree — step ${step}`, HASWIN ? `the top ${K} PUCT candidates (the run scanned ${decs.length} of them) · recorded threshold ${thr0} · ${REQ} become parents` : `the ${K} candidates the gate scanned (top of the PUCT ranking) · recorded threshold ${thr0} · up to ${REQ} become parents`,
    h("div", null, slider, ctl, chips, h("div", { class: "row-flex", style: { margin: "4px 0" } }, gensSel, h("span", { class: "sub" }, "left → right = step in which the program was created · circle = a candidate of the PUCT window (#rank in PUCT order; the matrix further down numbers the scan order instead), small dot = an ancestor")), treeHost, keyLegend,
      h("h3", { style: { margin: "14px 0 2px" } }, "The same candidates on the upper-bound axis"),
      h("div", { class: "sub" }, "dot colour = verdict at the current threshold · orange ring = the verdict differs from the recorded run · triangles = mean upper bound of the parent set (hollow: recorded, filled: replay) · equal values are stacked vertically"), stripHost, ctxChk, diffBox, noteBox,
      h("h3", { style: { margin: "16px 0 4px" } }, "How the decision changes with the threshold"),
      h("div", { class: "grid g2" },
        h("div", null, sweepA, legend([{ color: S_COL(7), label: "candidates rejected", line: true }, { color: S_COL(0), label: "recorded parents that get replaced", line: true }, { color: S_COL(3), label: "substitutes in the parent set (never judged)", line: true }])),
        h("div", null, sweepB, legend([{ color: S_COL(1), label: "mean upper bound of the accepted parents", line: true }, { color: S_COL(6), label: "worst parent", line: true }]))))));
  root.lastChild.style.marginBottom = "14px";

  /* ------------------------------------------------------------ the tree: candidates + their ancestors (built by buildTree, restyled by update()) */
  const arcEls = {}, nodeEls = {};
  const rawLab = v => (v == null ? "–" : Number(v).toFixed(PROB.dec + 1));
  function buildTree() {
    treeHost.innerHTML = "";
    Object.keys(arcEls).forEach(k => delete arcEls[k]); Object.keys(nodeEls).forEach(k => delete nodeEls[k]);
    // ancestors: up to _jt.gens generations above every candidate (Infinity = all the way to the seeds)
    const keep = new Set(), cut = new Set();
    cand.forEach(id => { let k = idx(id), g = 0; while (k != null && k >= 0 && g <= _jt.gens) { keep.add(k); const p = TREE.nodes[k][0]; if (g === _jt.gens && p >= 0) cut.add(k); k = p; g++; } });
    const kids = {}; keep.forEach(i => { const p = TREE.nodes[i][0]; if (p >= 0 && keep.has(p)) (kids[p] = kids[p] || []).push(i); });
    const roots = [...keep].filter(i => TREE.nodes[i][0] < 0 || !keep.has(TREE.nodes[i][0]));
    const minRank = {}; const calc = i => { let m = rankOf[TREE.ids[i]] || 1e9; (kids[i] || []).forEach(c => { m = Math.min(m, calc(c)); }); return (minRank[i] = m); };
    roots.forEach(calc);
    let leaf = 0; const pos = {};
    const lay = i => { const ch = (kids[i] || []).slice().sort((a, b) => minRank[a] - minRank[b]); if (!ch.length) pos[i] = leaf++; else { ch.forEach(lay); pos[i] = mean(ch.map(c => pos[c])); } };
    roots.sort((a, b) => minRank[a] - minRank[b]).forEach(lay);
    const minStep = Math.min(...[...keep].map(i => TREE.nodes[i][2])), maxStep = Math.max(...[...keep].map(i => TREE.nodes[i][2]));
    const dx = 100, dy = 34, ml = 60, mt = 48, cols = maxStep - minStep + 1;
    const W = ml + cols * dx + 150, H = mt + Math.max(1, leaf) * dy + 16;
    const svg = sv("svg", { width: W, height: H, viewBox: `0 0 ${W} ${H}`, class: "jt-svg" }, treeHost);
    const X = i => ml + (TREE.nodes[i][2] - minStep) * dx, Y = i => mt + pos[i] * dy;
    for (let c = 0; c < cols; c++) {
      const st0 = minStep + c; sv("line", { x1: ml + c * dx, x2: ml + c * dx, y1: 24, y2: H - 6, stroke: cssVar("--line"), "stroke-dasharray": "2 4" }, svg);
      const t = sv("text", { x: ml + c * dx, y: 14, "text-anchor": "middle", style: "font-size:11px;fill:" + cssVar("--ink-3") }, svg); t.textContent = st0 < 0 ? "seeds" : "step " + st0;
    }
    const ancRaw = [...keep].filter(i => TREE.nodes[i][0] >= 0).map(i => TREE.nodes[i][1]).filter(x => x > 0);
    const lo = Math.min(...ancRaw), hi = Math.max(...ancRaw);
    const colOf = i => { const n = TREE.nodes[i]; if (n[0] < 0) return cssVar("--ink-3"); return seqRamp(1 - clamp((n[1] - lo) / Math.max(1e-12, hi - lo), 0, 1)); };
    const gEdges = sv("g", {}, svg), gArcs = sv("g", {}, svg), gNodes = sv("g", {}, svg);
    keep.forEach(i => {
      const p = TREE.nodes[i][0];
      if (p >= 0 && keep.has(p)) { const x0 = X(p), y0 = Y(p), x1 = X(i), y1 = Y(i), mx = (x0 + x1) / 2; sv("path", { d: `M${x0},${y0}C${mx},${y0} ${mx},${y1} ${x1},${y1}`, fill: "none", stroke: cssVar("--line-2"), "stroke-width": 1.2 }, gEdges); }
      if (cut.has(i)) { const x1 = X(i), y1 = Y(i); sv("path", { d: `M${x1 - 34},${y1}H${x1}`, stroke: cssVar("--line-2"), "stroke-width": 1.2, "stroke-dasharray": "3 3" }, gEdges); const t = sv("text", { x: x1 - 36, y: y1 - 6, "text-anchor": "start", style: "font-size:10.5px;fill:" + cssVar("--ink-3") }, gEdges); t.textContent = "… older"; }
    });
    const candIdx = new Set(cand.map(idx));
    keep.forEach(i => {
      if (candIdx.has(i)) return;
      const n = TREE.nodes[i], id = TREE.ids[i];
      const c = sv("circle", { cx: X(i), cy: Y(i), r: n[0] < 0 ? 5 : 4.2, fill: colOf(i), stroke: cssVar("--surface"), "stroke-width": 1, style: "cursor:pointer" }, gNodes);
      c.addEventListener("mousemove", ev => tipShow(`<b>${n[0] < 0 ? "seed" : "step " + n[2]}</b> <span class="mut">${fmt.id(id)}</span><div>upper bound <b>${fmt.raw(n[1])}</b></div><div class="mut">ancestor of a scanned candidate · click to open</div>`, ev));
      c.addEventListener("mouseleave", tipHide); c.addEventListener("click", () => openNode(id));
    });
    // arcs: one hidden path per computed (candidate, accepted parent) pair; update() shows the ones that cause a rejection
    (jev.comparisons || []).forEach(c => {
      const a = idx(c.candidate_id), b = idx(c.parent_id); if (a == null || b == null || !keep.has(a) || !keep.has(b)) return;
      const x1 = X(a) + 8, y1 = Y(a), x2 = X(b) + 8, y2 = Y(b), xb = Math.max(x1, x2) + 46 + 5 * (rankOf[c.candidate_id] % 4);
      const path = sv("path", { d: `M${x1},${y1}C${xb},${y1} ${xb},${y2} ${x2},${y2}`, fill: "none", stroke: cssVar("--crit"), "stroke-width": 1.9, "stroke-dasharray": "5 4", class: "jt-arc", opacity: 0 }, gArcs);
      const lab = sv("text", { x: xb - 6, y: (y1 + y2) / 2 + 4, "text-anchor": "start", class: "jt-arclab", opacity: 0, style: "font-size:11px;font-weight:600;fill:" + cssVar("--crit") }, gArcs);
      arcEls[c.candidate_id + "|" + c.parent_id] = { path, lab };
    });
    cand.forEach((id, k) => {
      const i = idx(id); if (i == null) return;
      const ring = sv("circle", { cx: X(i), cy: Y(i), r: 14, fill: "none", stroke: cssVar("--s4"), "stroke-width": 2.2, class: "jt-ring", opacity: 0 }, gNodes);
      const c = sv("circle", { cx: X(i), cy: Y(i), r: 10.5, class: "jt-n unseen", style: "cursor:pointer" }, gNodes);
      const num = sv("text", { x: X(i), y: Y(i) + 3.6, "text-anchor": "middle", class: "jt-num" }, gNodes); num.textContent = "#" + (k + 1);
      const lab = sv("text", { x: X(i), y: Y(i) - 16, "text-anchor": "middle", class: "jt-lab" }, gNodes);
      c.addEventListener("click", () => openNode(id));
      c.addEventListener("mousemove", ev => tipShow(tipFor(id), ev)); c.addEventListener("mouseleave", tipHide);
      nodeEls[id] = { c, num, lab, ring };
    });
    requestAnimationFrame(() => { treeHost.scrollLeft = Math.max(0, W - treeHost.clientWidth); });
  }

  /* ------------------------------------------------------------ the zoomed strip: recorded vs replay */
  const raws = cand.map(rawOf).filter(x => x != null);
  const sLo = Math.min(...raws), sHi = Math.max(...raws), sPad = (sHi - sLo) * 0.07 || 1e-5;
  const stripState = {};
  function drawStrip() {
    stripHost.innerHTML = "";
    const SW = 1000, l = 40, r = 24, top = 10, GAP = 19;
    const xs = linear(sLo - sPad, sHi + sPad, l, SW - r);
    // candidates with (almost) equal values would hide each other: a small beeswarm - each dot takes the free vertical slot nearest to the lane
    const order = cand.map(id => ({ id, v: rawOf(id) })).filter(o => o.v != null).sort((a, b) => a.v - b.v);
    const placed = [], yoff = {}; let maxOff = 0;
    order.forEach(o => {
      const px = xs(o.v); let k = 0, y = 0;
      for (;;) { y = k === 0 ? 0 : (k % 2 ? -1 : 1) * Math.ceil(k / 2) * GAP; if (!placed.some(q => Math.hypot(q.x - px, q.y - y) < GAP)) break; k++; }
      placed.push({ x: px, y }); yoff[o.id] = y; maxOff = Math.max(maxOff, Math.abs(y));
    });
    const laneY = top + 18 + maxOff, axisY = laneY + maxOff + 26, SH = axisY + 24 + 62;
    const sg = sv("svg", { viewBox: `0 0 ${SW} ${SH}`, width: "100%", style: "display:block;overflow:visible" }, stripHost);
    if (_jt.ctx) {          // context: every archive node created before this step that falls inside the zoomed range
      let d = ""; for (let i = 0; i < TREE.nodes.length; i++) { const n = TREE.nodes[i]; if (n[0] < 0 || n[2] >= step || !(n[1] > 0) || n[1] < sLo - sPad || n[1] > sHi + sPad) continue; d += `M${xs(n[1]).toFixed(1)},${axisY - 9}h0`; }
      sv("path", { d, stroke: cssVar("--ink-3"), "stroke-width": 3, "stroke-linecap": "round", opacity: 0.25, fill: "none" }, sg);
    }
    const tk = niceTicks(sLo - sPad, sHi + sPad, 6), tf = tickFmt(tk);
    sv("line", { x1: l, x2: SW - r, y1: axisY, y2: axisY, stroke: cssVar("--line-2") }, sg);
    tk.forEach(t => { sv("line", { x1: xs(t), x2: xs(t), y1: top, y2: axisY, stroke: cssVar("--line"), "stroke-dasharray": "2 3" }, sg); const tx = sv("text", { x: xs(t), y: axisY + 15, "text-anchor": "middle", style: "font-size:11px;fill:" + cssVar("--ink-3") }, sg); tx.textContent = tf(t); });
    const ax = sv("text", { x: SW - r, y: axisY + 15 + 14, "text-anchor": "end", style: "font-size:11px;fill:" + cssVar("--ink-3") }, sg); ax.textContent = "upper bound of the candidate  (← better)";
    const rf = REF(); if (rf && rf.val >= sLo - sPad && rf.val <= sHi + sPad) { sv("line", { x1: xs(rf.val), x2: xs(rf.val), y1: top, y2: axisY, stroke: cssVar("--good"), "stroke-width": 1.4, "stroke-dasharray": "5 4" }, sg); const t = sv("text", { x: xs(rf.val) + 4, y: top + 9, style: "font-size:10.5px;fill:" + cssVar("--good-ink") }, sg); t.textContent = "reference"; }
    sv("line", { x1: l, x2: SW - r, y1: laneY, y2: laneY, stroke: cssVar("--line-2") }, sg);
    stripState.sg = sg; stripState.xs = xs; stripState.dots = {}; stripState.rings = {}; stripState.means = {}; stripState.laneY = laneY; stripState.maxOff = maxOff; stripState.axisY = axisY;
    const gDots = sv("g", {}, sg);
    cand.forEach((id, k) => {
      const v = rawOf(id); if (v == null) return;
      const cy = laneY + yoff[id], g = sv("g", { style: "cursor:pointer" }, gDots);
      stripState.rings[id] = sv("circle", { cx: xs(v), cy, r: 13.5, fill: "none", stroke: cssVar("--s4"), "stroke-width": 2.2, class: "jt-ring", opacity: 0 }, g);
      stripState.dots[id] = sv("circle", { cx: xs(v), cy, r: 9.5, class: "jt-n unseen" }, g);
      const t = sv("text", { x: xs(v), y: cy + 3.5, "text-anchor": "middle", class: "jt-num", style: "font-size:9.5px" }, g); t.textContent = k + 1;
      g.addEventListener("mousemove", ev => tipShow(tipFor(id), ev)); g.addEventListener("mouseleave", tipHide); g.addEventListener("click", () => openNode(id));
    });
    // markers for the mean upper bound of the parent set: recorded (hollow) and replay (filled)
    [["rec", "var(--ink-3)", true], ["rep", "var(--ink)", false]].forEach(([k, col, hollow]) => {
      const tri = sv("path", { d: "M0,-8 L6,2 L-6,2 Z", fill: hollow ? "var(--surface)" : col, stroke: col, "stroke-width": 1.6, opacity: 0 }, sg);
      const tx = sv("text", { y: 0, "text-anchor": "middle", style: "font-size:10.5px;font-weight:600;fill:" + (hollow ? "var(--ink-3)" : "var(--ink)"), opacity: 0 }, sg);
      stripState.means[k] = { tri, tx };
    });
  }

  /* ------------------------------------------------------------ sweep data (once) */
  const thrs = []; for (let t = 30; t <= 100; t++) thrs.push(t / 100);
  const sweepRows = thrs.map(t => {
    const rp = jevReplay(step, t), acc = rp.accepted, raws2 = acc.map(rawOf).filter(x => x != null);
    return { t, rej: rp.nRej, repl: [...recAcc].filter(id => !acc.includes(id)).length, subs: rp.sim.filter(x => x.unjudged && x.verdict === "accept").length, mean: raws2.length ? mean(raws2) : null, worst: raws2.length ? Math.max(...raws2) : null, acc };
  });
  function drawSweeps() {
    const t = thrNow(step), vl = [{ x: thr0, label: "recorded " + thr0, dash: "5 4", color: cssVar("--ink-3"), anchor: t > thr0 ? "end" : "start" }, ...(Math.abs(t - thr0) > 0.004 ? [{ x: t, label: "now " + t.toFixed(2), color: cssVar("--accent"), width: 2.2, dy: 14, anchor: t > thr0 ? "start" : "end" }] : [])];
    lineChart(sweepA, {
      height: 230, xLabel: "same-core threshold", yLabel: "count", margin: { l: 48 }, xDomain: [0.3, 1], xTicks: [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], xFmt: v => v.toFixed(1), vlines: vl, yTicks: [0, 2, 4, 6, 8, 10, 12, 14].filter(v => v <= Math.max(8, ...sweepRows.map(r => Math.max(r.rej, r.repl, r.subs))) + 1),
      series: [
        { name: "candidates rejected", color: S_COL(7), pts: sweepRows.map(r => [r.t, r.rej]), step: true, dots: false, width: 2.2 },
        { name: "recorded parents replaced", color: S_COL(0), pts: sweepRows.map(r => [r.t, r.repl]), step: true, dots: false, width: 2.2 },
        { name: "substitutes among the parents (never judged by GLiDE)", color: S_COL(3), pts: sweepRows.map(r => [r.t, r.subs]), step: true, dots: false, width: 1.8, dash: "4 3" },
      ],
      xFmtTip: v => v.toFixed(2), tipHead: x => `threshold ${x.toFixed(2)}`, onClick: x => { stopSweep(); setParam({ thr: x }); },
    });
    const ys = sweepRows.flatMap(r => [r.mean, r.worst]).filter(x => x != null);
    const yl = Math.min(...ys), yh = Math.max(...ys), pd = (yh - yl) * 0.15 || 1e-5;
    lineChart(sweepB, {
      height: 230, xLabel: "same-core threshold", yLabel: "upper bound of the parents", margin: { l: 70 }, xDomain: [0.3, 1], xTicks: [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], xFmt: v => v.toFixed(1), vlines: vl, yDomain: [yl - pd, yh + pd], yFmt: v => v.toFixed(5),
      series: [
        { name: "mean of the parents", color: S_COL(1), pts: sweepRows.filter(r => r.mean != null).map(r => [r.t, r.mean]), step: true, dots: false, width: 2.2 },
        { name: "worst parent", color: S_COL(6), pts: sweepRows.filter(r => r.worst != null).map(r => [r.t, r.worst]), step: true, dots: false, width: 1.8, dash: "4 3" },
      ],
      tipHead: x => `threshold ${x.toFixed(2)}`, tipVal: v => fmt.raw(v), onClick: x => { stopSweep(); setParam({ thr: x }); },
    });
  }

  /* ------------------------------------------------------------ tooltips */
  let cur = null;       // last replay
  function tipFor(id) {
    const k = rankOf[id], e = cur && cur.byId[id], thr = cur ? cur.thr : thr0;
    const rec = !scannedSet.has(id) ? "not scanned (the run stopped after its 8th acceptance)" : recAcc.has(id) ? "accepted" : (recMatch[id] ? `rejected - same core idea as ${fmt.id(recMatch[id])} (${fmt.f(simOf(id, recMatch[id]), 2)})` : "rejected");
    let rep = "not reached";
    if (e) rep = e.verdict === "accept" ? (e.unjudged ? `substitute${cur.pnum[id] ? " P" + cur.pnum[id] : ""} - GLiDE never compared it, assumed to pass${e.fallback ? " (window fill)" : ""}` : `accepted${e.unknown ? " (some pairs were never computed)" : ""}${cur.pnum[id] ? " as P" + cur.pnum[id] : ""}`) : `rejected - ≥ ${thr.toFixed(2)} similar to ${fmt.id(e.match)} (${fmt.f(simOf(id, e.match), 2)})`;
    const g = rawOf(id) - TARGET();
    return `<b>#${k} · ${fmt.id(id)}</b> <span class="mut">created in step ${stepOf(id)}</span><div>upper bound <b>${fmt.raw(rawOf(id))}</b> <span class="mut">(${fmt.gap(g)} vs reference)</span></div><div>recorded @ ${thr0}: <b>${rec}</b></div><div>replay @ ${thr.toFixed(2)}: <b>${rep}</b></div><div class="mut">click to open the program</div>`;
  }

  /* ------------------------------------------------------------ restyle everything for the current threshold */
  function update() {
    const thr = thrNow(step), rp = jevReplay(step, thr); if (!rp) return;
    const byId = {}; rp.sim.forEach(e => { byId[e.id] = e; });
    const pnum = {}; rp.accepted.forEach((id, j) => { pnum[id] = j + 1; });
    cur = { thr, byId, pnum, rp };
    const stOf = id => { const e = byId[id]; return !e ? "unseen" : e.verdict === "reject" ? "rej" : e.unjudged ? "sub" : "acc"; };
    const changed = id => { const e = byId[id]; return (e ? e.verdict === "accept" : false) !== recAcc.has(id); };
    Object.values(arcEls).forEach(a => { a.path.setAttribute("opacity", 0); a.lab.setAttribute("opacity", 0); });
    cand.forEach(id => {
      const st = stOf(id), e = byId[id], N = nodeEls[id];
      if (N) {
        N.c.setAttribute("class", "jt-n " + st); N.num.setAttribute("class", "jt-num"); N.ring.setAttribute("opacity", changed(id) ? 1 : 0); N.ring.setAttribute("class", "jt-ring");
        N.lab.textContent = (pnum[id] ? "P" + pnum[id] + (byId[id] && byId[id].unjudged ? "*" : "") + " · " : "") + rawLab(rawOf(id));
        N.lab.setAttribute("class", "jt-lab " + st);
      }
      const dEl = stripState.dots[id]; if (dEl) dEl.setAttribute("class", "jt-n " + st);
      const rEl = stripState.rings && stripState.rings[id]; if (rEl) { rEl.setAttribute("opacity", changed(id) ? 1 : 0); rEl.setAttribute("class", "jt-ring"); }
      if (e && e.verdict === "reject" && e.match) { const a = arcEls[id + "|" + e.match]; if (a) { a.path.setAttribute("opacity", 1); a.lab.setAttribute("opacity", 1); a.lab.textContent = `≈ ${pnum[e.match] ? "P" + pnum[e.match] : "#" + rankOf[e.match]} · ${Math.round(100 * (simOf(id, e.match) || 0))} %`; } }
    });
    // strip: parent means
    const meanOf = ids => { const v = ids.map(rawOf).filter(x => x != null); return v.length ? mean(v) : null; };
    const recMean = meanOf([...recAcc]), repMean = meanOf(rp.accepted);
    {
      const MS = stripState.means; if (MS) {
        const same = recMean != null && repMean != null && Math.abs(recMean - repMean) < 1e-10, base = stripState.axisY + 46;
        const put = (k, m, text, row) => { const M = MS[k]; if (!M) return; if (m == null || text == null) { M.tri.setAttribute("opacity", 0); M.tx.setAttribute("opacity", 0); return; }
          const px = stripState.xs(m); M.tri.setAttribute("transform", `translate(${px},${base})`); M.tri.setAttribute("opacity", 1);
          M.tx.setAttribute("x", px); M.tx.setAttribute("y", base + 14 + row * 14); M.tx.setAttribute("text-anchor", px > 760 ? "end" : px < 240 ? "start" : "middle"); M.tx.setAttribute("opacity", 1); M.tx.textContent = text; };
        const f = v => v.toFixed(PROB.dec + 1);
        if (same) { put("rep", repMean, `mean of the parents ${f(repMean)} (same as the recorded run)`, 0); put("rec", null, null, 0); }
        else { put("rec", recMean, recMean != null ? `recorded: mean of the parents ${f(recMean)}` : null, 0); put("rep", repMean, repMean != null ? `replay: mean of the parents ${f(repMean)}` : null, 1); }
      }
    }
    // chips + diff
    const nAcc = rp.nAcc, nRej = rp.nRej, unk = rp.sim.filter(x => x.unknown).length;
    chips.innerHTML = "";
    chips.append(pill(`recorded @ ${thr0}: ${rp.recAcc} accepted · ${rp.recRej} rejected`, "acc"), pill(`replay @ ${thr.toFixed(2)}: ${nAcc} accepted · ${nRej} rejected`, Math.abs(thr - thr0) < 1e-9 ? "good" : "warn"));
    if (P.thr != null && Math.abs(thr - thr0) > 1e-9) chips.append(h("span", { class: "what-if" }, "what-if threshold - applied to all steps"));
    const enter = rp.accepted.filter(id => !recAcc.has(id)), leave = [...recAcc].filter(id => !rp.accepted.includes(id));
    diffBox.innerHTML = "";
    const chip = (id, sign) => { const sb = sign === "+" && byId[id] && byId[id].unjudged; return h("a", { href: "javascript:void 0", class: "jt-chip " + (sign === "+" ? (sb ? "sub" : "in") : "out"), onclick: () => openNode(id), title: sb ? "substitute from the PUCT window - GLiDE never judged it (assumed to pass) - click to open the program" : "click to open the program" }, `${sign} #${rankOf[id]} ${fmt.id(id)}  ${fmt.raw(rawOf(id))}${sb ? "  (unjudged)" : ""}`); };
    if (enter.length || leave.length) {
      diffBox.append(h("div", { class: "row-flex", style: { margin: "8px 0 2px" } }, h("b", null, `At ${thr.toFixed(2)} the ${REQ} parents differ from the recorded run:`), enter.map(id => chip(id, "+")), leave.map(id => chip(id, "−"))));
      const mr = (meanOf(rp.accepted) != null && recMean != null) ? meanOf(rp.accepted) - recMean : null;
      if (mr != null) diffBox.append(h("div", { class: "sub" }, `mean upper bound of the parents: ${fmt.raw(recMean)} → ${fmt.raw(meanOf(rp.accepted))} (${fmt.gap(mr)}; negative = better)`));
    } else diffBox.append(h("div", { class: "sub", style: { margin: "8px 0 2px" } }, "The parent set is identical to the recorded run at this threshold."));
    noteBox.innerHTML = "";
    const notes = [];
    if (rp.hasWindow) {
      const subs = rp.sim.filter(x => x.unjudged && x.verdict === "accept");
      if (subs.length) notes.push(`${subs.length} of the ${REQ} parents at this threshold ${subs.length > 1 ? "are substitutes" : "is a substitute"} taken from further down the PUCT window (blue, marked *): the original run stopped scanning after its 8th acceptance, so GLiDE never compared ${subs.length > 1 ? "them" : "it"} with the accepted parents. They are shown as if they pass; if one were judged too similar the next candidate would take its place.`);
    } else if (nAcc < REQ) notes.push(`At this threshold only ${nAcc} of ${REQ} slots can be filled from the ${K} candidates the run scanned (the candidate window could not be rebuilt for this run, so the substitutes are unknown).`);
    if (unk) notes.push(`${unk} acceptance${unk > 1 ? "s rely" : " relies"} on pairs the original run never computed (it had already stopped at an earlier match); they are counted as 'no match'.`);
    notes.forEach(t => noteBox.append(h("div", { class: "warnbox", style: { marginTop: "6px" } }, t)));
    drawSweeps();
  }

  /* ------------------------------------------------------------ animated sweep */
  function stopSweep() { if (timer) { clearInterval(timer); timer = null; } sweepBtn.classList.remove("on"); sweepBtn.textContent = "▶ sweep the threshold 0.30 → 1.00"; }
  function sweep() {
    if (timer) { stopSweep(); return; }
    restore = P.thr; let v = 0.3; sweepBtn.classList.add("on"); sweepBtn.textContent = "■ stop";
    setParam({ thr: v });
    timer = setInterval(() => {
      if (!treeHost.isConnected) { stopSweep(); return; }
      v = Math.round((v + 0.02) * 100) / 100;
      if (v > 1.0001) { stopSweep(); setParam({ thr: restore }); return; }
      setParam({ thr: v });
    }, 260);
  }

  buildTree(); drawStrip(); update();
  onParam(patch => { if ("thr" in patch) update(); });
}
