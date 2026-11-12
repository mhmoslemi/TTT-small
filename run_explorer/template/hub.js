"use strict";
/* ---- hub.js: landing page for several problems + lazy, isolated explorers ------------------------------------------ */
const RUNS = JSON.parse(document.getElementById("hub-data").textContent);
const $ = (s, r = document) => r.querySelector(s);
const NS = "http://www.w3.org/2000/svg";
const COLORS = ["var(--s1)", "var(--s2)", "var(--s3)", "var(--s4)", "var(--s5)", "var(--s7)", "var(--s6)", "var(--s8)"];
RUNS.forEach((r, i) => { r.color = COLORS[i % COLORS.length]; });

function esc(s) { return String(s == null ? "" : s).replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])); }
function h(tag, attrs, ...kids) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v == null || v === false) continue;
    if (k === "class") el.className = v;
    else if (k === "style" && typeof v === "object") Object.assign(el.style, v);
    else if (k === "html") el.innerHTML = v;
    else if (k.startsWith("on") && typeof v === "function") el.addEventListener(k.slice(2), v);
    else el.setAttribute(k, v === true ? "" : v);
  }
  (function add(list) { for (const c of list) { if (c == null || c === false) continue; if (Array.isArray(c)) add(c); else el.append(c.nodeType ? c : document.createTextNode(c)); } })(kids);
  return el;
}
function svg(tag, attrs, ...kids) {
  const el = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs || {})) if (v != null) el.setAttribute(k, v);
  for (const c of kids) if (c != null) el.append(c.nodeType ? c : document.createTextNode(c));
  return el;
}

/* ---------------------------------------------------------------- formatting */
const fx = (v, d) => (v == null || !isFinite(v) ? "–" : (+v).toFixed(d));
const int = n => (n == null ? "–" : Math.round(n).toLocaleString("en-US"));
const sgn = (v, d) => (v >= 0 ? "+" : "−") + Math.abs(v).toFixed(d);
function dh(sec) {            // day:hour as "3d 14h"
  if (sec == null) return "–";
  let d = Math.floor(sec / 86400), hh = Math.round((sec % 86400) / 3600);
  if (hh === 24) { d += 1; hh = 0; }
  return d ? `${d}d ${hh}h` : `${hh}h`;
}
const nameOf = r => r.short || r.title;
const better = (r, a, b) => (r.lower_is_better === false ? a > b : a < b);   // is a better than b?
function status(r) {
  if (r.steps_total && r.steps_done >= r.steps_total) return { t: "complete", cls: "good", tip: "all configured steps finished" };
  if (r.stopped) return { t: "stopped", cls: "warn", tip: "the log of this run ends in a stop / traceback - it can be resumed" };
  return { t: "in progress", cls: "acc", tip: "fewer steps than configured and no stop / traceback in the log - the run was still going when this page was generated" };
}
function gapInfo(r) {
  if (!r.reference || r.best == null) return null;
  const g = r.best - r.reference.value, pct = g / r.reference.value * 100, ok = better(r, r.best, r.reference.value);
  return { g, pct, ok, txt: `${sgn(g, r.dec)} (${sgn(pct, Math.abs(pct) < 0.01 ? 4 : 3)} %)` };
}
function progress(r) {          // 0 % = the first step's best, 100 % = the published reference
  if (!r.reference || !r.curve.length) return null;
  const s0 = r.curve[0][1], ref = r.reference.value;
  if (s0 === ref) return null;
  return r.curve.map(([st, v]) => [st, (s0 - v) / (s0 - ref) * 100]);
}
function hwLabel(r) {
  const hw = r.hardware || {};
  const inf = hw.basis && /^inferred/.test(hw.basis);
  return h("span", null, hw.label || "not recorded", inf ? h("span", { class: "inf", title: hw.basis }, "model inferred") : null);
}

/* ---------------------------------------------------------------- tooltip */
const tip = $("#tip");
function tipShow(html, ev) {
  tip.innerHTML = html; tip.classList.add("on");
  const w = tip.offsetWidth, hh = tip.offsetHeight;
  let x = ev.clientX + 14, y = ev.clientY + 14;
  if (x + w > innerWidth - 8) x = ev.clientX - w - 14;
  if (y + hh > innerHeight - 8) y = ev.clientY - hh - 14;
  tip.style.left = Math.max(4, x) + "px"; tip.style.top = Math.max(4, y) + "px";
}
const tipHide = () => tip.classList.remove("on");

/* ---------------------------------------------------------------- generic line chart (SVG, hover crosshair) */
function lineChart(o) {
  const W = o.w, H = o.h, m = Object.assign({ l: 46, r: 10, t: 8, b: 22 }, o.m || {});
  if (o.xLabel) m.b += 12;
  const iw = W - m.l - m.r, ih = H - m.t - m.b;
  const [x0, x1] = o.xDomain, [y0, y1] = o.yDomain;
  const X = x => m.l + (x1 === x0 ? iw / 2 : (x - x0) / (x1 - x0) * iw), Y = y => m.t + (1 - (y - y0) / (y1 - y0)) * ih;
  const s = svg("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": o.label || "line chart" });
  (o.yTicks || []).forEach(t => {
    s.append(svg("line", { class: "grid-line", x1: m.l, x2: W - m.r, y1: Y(t), y2: Y(t) }));
    s.append(svg("text", { x: m.l - 6, y: Y(t) + 3.5, "text-anchor": "end" }, o.yFmt ? o.yFmt(t) : t));
  });
  s.append(svg("line", { class: "axis-line", x1: m.l, x2: W - m.r, y1: H - m.b, y2: H - m.b }));
  (o.xTicks || []).forEach(t => s.append(svg("text", { x: X(t), y: H - m.b + 14, "text-anchor": "middle" }, t)));
  if (o.xLabel) s.append(svg("text", { x: W - m.r, y: H - 2, "text-anchor": "end" }, o.xLabel));
  (o.hlines || []).forEach(l => {
    if (l.y < y0 || l.y > y1) return;
    s.append(svg("line", { class: "ref-line", x1: m.l, x2: W - m.r, y1: Y(l.y), y2: Y(l.y) }));
    if (l.label) s.append(svg("text", { x: W - m.r - 4, y: Y(l.y) + (l.below ? 13 : -5), "text-anchor": "end" }, l.label));
  });
  const ends = [];
  o.series.forEach(se => {
    if (!se.pts.length) return;
    s.append(svg("path", { d: se.pts.map((p, i) => (i ? "L" : "M") + X(p[0]).toFixed(1) + " " + Y(p[1]).toFixed(1)).join(""), fill: "none", stroke: se.color, "stroke-width": 2, "stroke-linejoin": "round", "stroke-linecap": "round" }));
    const last = se.pts[se.pts.length - 1];
    s.append(svg("circle", { cx: X(last[0]), cy: Y(last[1]), r: 4.5, fill: se.color, stroke: "var(--surface)", "stroke-width": 2 }));
    ends.push({ se, x: X(last[0]), y: Y(last[1]), v: last[1] });
  });
  if (o.endLabels) {        // direct labels at the line ends, nudged apart
    ends.sort((a, b) => a.y - b.y);
    let prev = -1e9;
    ends.forEach(e => { e.ly = Math.max(e.y, prev + 15); prev = e.ly; });
    ends.forEach(e => s.append(svg("text", { class: "lbl", x: Math.min(e.x + 9, W - 4), y: e.ly + 4, "text-anchor": "start" }, o.endLabels(e.se, e.v))));
  }
  const cross = svg("line", { x1: 0, x2: 0, y1: m.t, y2: H - m.b, stroke: "var(--ink-3)", "stroke-width": 1, visibility: "hidden" });
  const dots = o.series.map(se => svg("circle", { r: 4, fill: se.color, stroke: "var(--surface)", "stroke-width": 2, visibility: "hidden" }));
  s.append(cross); dots.forEach(d => s.append(d));
  const xs = [...new Set(o.series.flatMap(se => se.pts.map(p => p[0])))].sort((a, b) => a - b);
  const hit = svg("rect", { x: m.l, y: m.t, width: iw, height: ih, fill: "transparent" });
  hit.addEventListener("mousemove", ev => {
    const r = s.getBoundingClientRect(), px = (ev.clientX - r.left) / r.width * W;
    let best = xs[0]; for (const x of xs) if (Math.abs(X(x) - px) < Math.abs(X(best) - px)) best = x;
    cross.setAttribute("x1", X(best)); cross.setAttribute("x2", X(best)); cross.setAttribute("visibility", "visible");
    o.series.forEach((se, i) => { const p = se.pts.find(q => q[0] === best); if (p) { dots[i].setAttribute("cx", X(best)); dots[i].setAttribute("cy", Y(p[1])); dots[i].setAttribute("visibility", "visible"); } else dots[i].setAttribute("visibility", "hidden"); });
    tipShow(o.tip(best), ev);
  });
  hit.addEventListener("mouseleave", () => { cross.setAttribute("visibility", "hidden"); dots.forEach(d => d.setAttribute("visibility", "hidden")); tipHide(); });
  s.append(hit);
  return s;
}
function niceTicks(lo, hi, n) { return Array.from({ length: n }, (_, i) => lo + (hi - lo) * i / (n - 1)); }

/* ---------------------------------------------------------------- landing page */
function sparkline(r) {
  const pts = r.curve.map(p => [p[0], p[1]]);
  if (!pts.length) return h("div", { class: "sub" }, "no finished step yet");
  const ref = r.reference ? r.reference.value : null;
  const vals = pts.map(p => p[1]).concat(ref != null ? [ref] : []);
  let lo = Math.min(...vals), hi = Math.max(...vals); const pad = (hi - lo) * 0.1 || Math.abs(hi) * 1e-4;
  const labelBelow = ref != null && pts[pts.length - 1][1] > ref;      // the reference label goes on the side the curve is not on - leave room for it
  lo -= pad * (labelBelow ? 2.6 : 1); hi += pad * (labelBelow ? 1 : 1);
  const d = Math.min(10, Math.max(2, Math.ceil(-Math.log10(hi - lo)) + 1));
  const xmax = Math.max(pts[pts.length - 1][0], 1);
  const wrap = h("div", { class: "hub-spark" });
  wrap.append(lineChart({
    w: 420, h: 150, m: { l: 52, r: 12, t: 8, b: 22 }, xDomain: [0, xmax], yDomain: [lo, hi], yTicks: niceTicks(Math.min(...vals), Math.max(...vals), 3), yFmt: v => v.toFixed(d),
    xTicks: [0, ...(xmax > 3 ? [Math.round(xmax / 2)] : []), xmax].filter((v, i, a) => a.indexOf(v) === i), xLabel: "step",
    hlines: ref != null ? [{ y: ref, label: "reference " + ref, below: labelBelow }] : [],
    series: [{ id: r.id, name: nameOf(r), color: r.color, pts }], label: "best so far by step, " + nameOf(r),
    tip: x => { const p = pts.find(q => q[0] === x); return `<b>step ${x}</b><br>best so far <b>${fx(p[1], r.dec)}</b>` + (ref != null ? `<br><span class="mut">${sgn(p[1] - ref, r.dec)} vs reference</span>` : ""); },
  }));
  return wrap;
}

function runCard(r) {
  const st = status(r), gi = gapInfo(r), hw = r.hardware || {};
  const kv = [
    ["Progress", `${r.steps_done} of ${r.steps_total || "?"} steps · ${int(r.rollouts)} programs (${fx(r.valid_fraction * 100, 0)} % valid)` +
      (r.pending ? `<br><span class="sub">step ${r.pending.step}: ${r.pending.rollouts_done ? "all rollouts scored, adapter update pending - its best is included" : int(r.pending.rollouts) + " programs scored so far" + (r.stopped ? " (run stopped here)" : " (running)")}</span>` : "")],
    ["Models", `${esc(r.coder)} <span class="sub">coder</span> · ${esc(r.strategist)} <span class="sub">strategist</span>`],
    ["Selection", esc(r.selection || "PUCT")],
    ["GPUs", null],
    ["Wall clock", r.elapsed_s != null ? `<b>${dh(r.elapsed_s)}</b>${r.gpu_hours ? ` <span class="sub">≈ ${int(r.gpu_hours)} GPU-hours</span>` : ""}` : "–"],
  ];
  const facts = h("div", { class: "facts" });
  kv.forEach(([k, v]) => { facts.append(h("div", { class: "k" }, k)); const d = h("div", { class: "v" }); if (k === "GPUs") d.append(hwLabel(r)); else d.innerHTML = v; facts.append(d); });
  return h("div", { class: "card pcard" },
    h("div", { class: "ptitle" }, h("span", { class: "sw", style: { background: r.color } }), r.title, h("span", { class: "pill " + st.cls, title: st.tip, style: { marginLeft: "auto" } }, st.t)),
    h("div", { class: "rname" }, r.run_name),
    h("div", { class: "big" }, h("span", { class: "v" }, fx(r.best, r.dec)), h("span", { class: "u" }, (r.metric || "best") + " · best so far")),
    h("div", { class: "gaprow" },
      gi ? [h("span", { class: "pill " + (gi.ok ? "good" : "") }, gi.ok ? "beats the reference" : "above the reference"), h("span", null, `${gi.txt} vs ${r.reference.label}`)] : h("span", { class: "sub" }, "no published reference for this problem"),
      r.rank ? h("span", { class: "sub", title: "rank among the published results shipped with the explorer, plus this run" }, `· rank ${r.rank} of ${r.n_ranked}`) : null),
    sparkline(r), facts,
    h("div", { class: "go" }, h("button", { class: "btn primary", onclick: () => { location.hash = "#" + r.id; } }, "Open explorer →"),
      h("span", { class: "sub" }, "pipeline · search · strategist · training · constructions · published SOTA")));
}

function kpiTile(k, v, d) { return h("div", { class: "kpi" }, h("div", { class: "k" }, k), h("div", { class: "v" }, v), h("div", { class: "d" }, d || "")); }

function progressCard() {
  const rs = RUNS.filter(r => progress(r));
  if (rs.length < 1) return null;
  const series = rs.map(r => ({ id: r.id, name: nameOf(r), color: r.color, pts: progress(r) }));
  const allV = series.flatMap(s => s.pts.map(p => p[1])), xmax = Math.max(...series.flatMap(s => s.pts.map(p => p[0])));
  const lo = Math.min(0, ...allV), hi = Math.max(110, Math.max(...allV) + 8), stp = hi > 200 ? 50 : 25;
  const yt = []; for (let t = Math.ceil(lo / stp) * stp; t <= hi; t += stp) yt.push(t);
  const host = h("div", { class: "hub-chart" });
  host.append(lineChart({
    w: 940, h: 300, m: { l: 50, r: 150, t: 10, b: 26 }, xDomain: [0, xmax], yDomain: [lo, hi], yTicks: yt, yFmt: v => v + " %",
    xTicks: Array.from({ length: Math.min(xmax, 20) + 1 }, (_, i) => Math.round(i * xmax / Math.min(xmax, 20))).filter((v, i, a) => a.indexOf(v) === i), xLabel: "step",
    hlines: [{ y: 100, label: "100 % = the published reference of each problem", below: true }],
    series, endLabels: (se, v) => `${se.name}  ${v.toFixed(0)} %`, label: "progress toward the published reference",
    tip: x => `<b>step ${x}</b>` + series.map(se => { const p = se.pts.find(q => q[0] === x); return p ? `<br><span class="dot" style="background:${se.color}"></span>${esc(se.name)}: <b>${p[1].toFixed(1)} %</b>` : ""; }).join(""),
  }));
  const legend = h("div", { class: "legend" }, series.map(se => h("span", { class: "it" }, h("span", { class: "sw", style: { background: se.color } }), se.name + " – " + RUNS.find(r => r.id === se.id).reference.label)));
  const rows = []; for (let x = 0; x <= xmax; x++) rows.push([x, ...series.map(se => { const p = se.pts.find(q => q[0] === x); return p ? p[1].toFixed(1) + " %" : "–"; })]);
  const table = h("details", { style: { marginTop: "8px" } }, h("summary", { class: "sub", style: { cursor: "pointer" } }, "show as table"),
    h("div", { class: "scroll", style: { maxHeight: "240px", marginTop: "6px" } }, h("table", { class: "t" },
      h("thead", null, h("tr", null, h("th", null, "step"), series.map(se => h("th", { class: "num" }, se.name)))),
      h("tbody", null, rows.map(r => h("tr", null, r.map((c, i) => h("td", { class: i ? "num" : "" }, c))))))));
  return h("div", { class: "card", style: { marginTop: "14px" } },
    h("div", { class: "card-head" }, h("h2", null, "Progress toward each problem's published reference"), h("span", { class: "sub" }, "0 % = the best of the first step · 100 % = the reference value · above 100 % = the run beats it")),
    host, legend, table);
}

function compareCard() {
  const rows = [
    ["Problem", r => r.title],
    ["Run", r => h("span", { class: "mono", style: { fontSize: "11.5px", wordBreak: "break-all" } }, r.run_name)],
    ["Status", r => { const s = status(r); return h("span", { class: "pill " + s.cls, title: s.tip }, s.t); }],
    ["Steps finished", r => `${r.steps_done} / ${r.steps_total || "?"}`],
    ["Programs · valid", r => `${int(r.rollouts)} · ${fx(r.valid_fraction * 100, 1)} %`],
    ["Coder / strategist", r => h("span", null, r.coder, h("span", { class: "sub" }, r.strategist))],
    ["Selection", r => r.selection || "PUCT"],
    ["Best so far", r => h("b", null, fx(r.best, r.dec), h("span", { class: "sub" }, r.metric))],
    ["Reference", r => r.reference ? h("span", null, fx(r.reference.value, r.dec), h("span", { class: "sub" }, r.reference.label)) : "–"],
    ["Gap to reference", r => { const g = gapInfo(r); return g ? h("span", null, g.txt, h("span", { class: "sub" }, g.ok ? "better than the reference" : "worse than the reference")) : "–"; }],
    ["Progress to reference", r => { const p = progress(r); return p ? p[p.length - 1][1].toFixed(0) + " %" : "–"; }],
    ["Rank", r => r.rank ? `${r.rank} of ${r.n_ranked}` : "–"],
    ["Wall clock", r => dh(r.elapsed_s)],
    ["GPU-hours", r => r.gpu_hours ? "≈ " + int(r.gpu_hours) : "–"],
  ];
  return tableCard("Side by side", "same pipeline, different problems", rows);
}
function hardwareCardHub() {
  const H = r => r.hardware || {};
  const rows = [
    ["GPUs", r => hwLabel(r)],
    ["Card", r => h("span", null, H(r).name || "–", h("span", { class: "sub" }, H(r).family || ""))],
    ["Memory per GPU", r => H(r).mem_gib ? H(r).mem_gib.toFixed(2) + " GiB" : "–"],
    ["Compute capability", r => H(r).capability || "not logged"],
    ["Hardware profile", r => H(r).profile ? H(r).profile : h("span", null, "RTX6000", h("span", { class: "sub" }, "repo default - not set in this run's config"))],
    ["Coder generation", r => H(r).coder_engines ? `${H(r).coder_engines} vLLM engines, TP ${H(r).coder_tp}` : "vLLM"],
    ["Strategist generation", r => H(r).strategist_engines ? `${H(r).strategist_engines} vLLM engines, TP ${H(r).strategist_tp}` : "–"],
    ["Training layout", r => H(r).training_layout || "–"],
    ["Evaluation (CPU)", r => H(r).eval_processes ? `${H(r).eval_processes} sandbox processes on ${H(r).eval_cpus_used || "?"} CPUs` : "–"],
    ["Host RAM for offload", r => H(r).host_offload_gib ? H(r).host_offload_gib + " GiB" : "–"],
    ["Basis of the card name", r => h("span", { class: "sub", style: { display: "inline" } }, H(r).basis || "")],
  ];
  return tableCard("Hardware used", "GPU specification of every run (read from its config and the vLLM / trainer logs)", rows);
}
function tableCard(title, sub, rows) {
  const head = h("tr", null, h("th", { class: "rowh" }, ""), RUNS.map(r => h("th", null, h("span", { class: "dot", style: { background: r.color } }), nameOf(r))));
  const body = rows.map(([k, f]) => h("tr", null, h("td", { class: "rowh" }, k), RUNS.map(r => { const v = f(r); const td = h("td", null); if (typeof v === "string") td.innerHTML = esc(v); else td.append(v); return td; })));
  return h("div", { class: "card", style: { marginTop: "14px" } }, h("div", { class: "card-head" }, h("h2", null, title), h("span", { class: "sub" }, sub)),
    h("div", { class: "scroll", style: { maxHeight: "none" } }, h("table", { class: "t cmp" }, h("thead", null, head), h("tbody", null, body))));
}

function renderLanding() {
  const root = $("#landing"); root.innerHTML = "";
  const totalS = RUNS.reduce((a, r) => a + (r.elapsed_s || 0), 0), gh = RUNS.reduce((a, r) => a + (r.gpu_hours || 0), 0), prog = RUNS.reduce((a, r) => a + r.rollouts, 0);
  root.append(h("h1", { class: "hub-h1" }, "All problems"),
    h("p", { class: "hub-lede" }, `${RUNS.length} problem${RUNS.length > 1 ? "s" : ""}, one pipeline (coder + strategist, PUCT selection, SPO² training). Each card is a full interactive explorer of one run - open it for the pipeline, search tree, strategist plans, training curves, every construction and the published SOTA. For both problems below a smaller number is better.`),
    h("div", { class: "kpis" },
      kpiTile("Problems", String(RUNS.length), RUNS.map(nameOf).join(" · ")),
      kpiTile("Programs generated", int(prog), "across all runs"),
      kpiTile("Wall clock", dh(totalS), "all runs together"),
      kpiTile("GPU-hours", "≈ " + int(gh), RUNS.map(r => (r.hardware || {}).label).filter(Boolean).join("  |  "))),
    h("div", { class: "grid g2" }, RUNS.map(runCard)));
  const pc = progressCard(); if (pc) root.append(pc);
  root.append(compareCard(), hardwareCardHub(),
    h("div", { class: "hub-foot" }, "Generated ", RUNS[0].generated_at, " by ", h("code", null, "make_explorer.sh --unified"), ". Add a problem by building its run into the same page: ", h("code", null, "./make_explorer.sh --unified RUN_A RUN_B RUN_C"), " (see README). Everything here is embedded - the file works offline."));
}

/* ---------------------------------------------------------------- explorers (one isolated iframe per run, inflated on demand) */
const frames = {};      // id -> { promise, iframe, url }
function b64bytes(b64) { const bin = atob(b64); const a = new Uint8Array(bin.length); for (let i = 0; i < bin.length; i++) a[i] = bin.charCodeAt(i); return a; }
async function inflateHtml(id) {
  const el = document.getElementById("html-" + id);
  const bytes = b64bytes(el.textContent.trim());
  const ds = new Blob([bytes]).stream().pipeThrough(new DecompressionStream("gzip"));
  return await new Response(ds).text();
}
function loadRun(r) {
  if (frames[r.id]) return frames[r.id].promise;
  const rec = frames[r.id] = {};
  rec.promise = (async () => {
    const txt = await inflateHtml(r.id);
    rec.url = URL.createObjectURL(new Blob([txt], { type: "text/html" }));
    const f = h("iframe", { title: r.title + " - " + r.run_name, src: rec.url, "data-run": r.id });
    rec.iframe = f;
    await new Promise(res => { f.addEventListener("load", res, { once: true }); $("#frames").append(f); });
    return rec;
  })();
  return rec.promise;
}
async function openRun(r) {
  $("#hubMain").hidden = true; $("#frames").hidden = false; $("#btnNewTab").hidden = false;
  document.querySelectorAll("#frames iframe").forEach(f => f.classList.remove("on"));
  document.title = nameOf(r) + " - Run Explorer";
  $("#hubSub").textContent = r.run_name + "  ·  " + ((r.hardware || {}).label || "");
  const first = !frames[r.id];
  let spin = null;
  if (first) { spin = h("div", { class: "loading" }, h("div", { class: "spinner" }), h("div", null, "Inflating " + r.run_name + " …")); $("#frames").append(spin); }
  try {
    const rec = await loadRun(r);
    if (currentView() === r.id) rec.iframe.classList.add("on");
  } catch (e) {
    console.error(e);
    $("#frames").append(h("div", { class: "warnbox", style: { margin: "30px" } }, "Could not load this explorer: " + (e && e.message || e) + ". Open the standalone file for this run instead."));
  } finally { if (spin) spin.remove(); }
}
function openOverview() {
  $("#hubMain").hidden = false; $("#frames").hidden = true; $("#btnNewTab").hidden = true;
  document.title = "Run Explorer - all problems"; $("#hubSub").textContent = "all problems";
}
const currentView = () => { const v = (location.hash || "#overview").slice(1); return RUNS.some(r => r.id === v) ? v : "overview"; };
function buildNav() {
  const nav = $("#hubNav"); nav.innerHTML = "";
  nav.append(h("button", { class: "hub-pill", "data-view": "overview", onclick: () => { location.hash = "#overview"; } }, "◧ All problems"));
  RUNS.forEach(r => nav.append(h("button", { class: "hub-pill", "data-view": r.id, title: r.title + " - " + r.run_name, onclick: () => { location.hash = "#" + r.id; } },
    h("span", { class: "sw", style: { background: r.color } }), nameOf(r), h("small", null, r.best != null ? fx(r.best, r.dec) : ""))));
}
function route() {
  const v = currentView();
  document.querySelectorAll(".hub-pill").forEach(p => p.classList.toggle("active", p.dataset.view === v));
  if (v === "overview") openOverview(); else openRun(RUNS.find(r => r.id === v));
  window.scrollTo(0, 0);
}

/* ---------------------------------------------------------------- theme (shared with every explorer) */
const curTheme = () => document.documentElement.getAttribute("data-theme") || (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
let explicitTheme = null;
try { explicitTheme = localStorage.getItem("rx-theme"); if (explicitTheme) document.documentElement.setAttribute("data-theme", explicitTheme); } catch (e) { /* ignore */ }
function pushTheme(t) { Object.values(frames).forEach(f => { try { if (f.iframe) f.iframe.contentWindow.postMessage({ rxTheme: t }, "*"); } catch (e) { /* ignore */ } }); }
$("#btnTheme").addEventListener("click", () => {
  const nxt = curTheme() === "dark" ? "light" : "dark";
  explicitTheme = nxt; document.documentElement.setAttribute("data-theme", nxt);
  try { localStorage.setItem("rx-theme", nxt); } catch (e) { /* ignore */ }
  pushTheme(nxt);
});
window.addEventListener("message", e => {
  const d = e.data || {};
  if (d.rxThemeSet) {                                  // the explorer's own toggle was used: follow it
    explicitTheme = d.rxThemeSet; document.documentElement.setAttribute("data-theme", d.rxThemeSet);
    try { localStorage.setItem("rx-theme", d.rxThemeSet); } catch (err) { /* ignore */ }
    Object.values(frames).forEach(f => { try { if (f.iframe && f.iframe.contentWindow !== e.source) f.iframe.contentWindow.postMessage({ rxTheme: d.rxThemeSet }, "*"); } catch (err) { /* ignore */ } });
  } else if (d.rxReady && explicitTheme && d.theme !== explicitTheme) {
    try { e.source.postMessage({ rxTheme: explicitTheme }, "*"); } catch (err) { /* ignore */ }
  }
});
$("#btnNewTab").addEventListener("click", async () => {
  const r = RUNS.find(x => x.id === currentView()); if (!r) return;
  const rec = await loadRun(r); window.open(rec.url, "_blank");
});

buildNav();
renderLanding();
window.addEventListener("hashchange", route);
route();
