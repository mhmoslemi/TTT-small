"use strict";
/* =====================================================================================
   core.js - data loading, global state, small DOM / formatting helpers, drawer, tooltip
   ===================================================================================== */
const $ = (s, r) => (r || document).querySelector(s);
const $$ = (s, r) => Array.from((r || document).querySelectorAll(s));
const esc = s => String(s == null ? "" : s).replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
const sum = a => a.reduce((s, x) => s + x, 0);
const mean = a => a.length ? sum(a) / a.length : NaN;
const median = a => { if (!a.length) return NaN; const b = a.slice().sort((x, y) => x - y); const m = b.length >> 1; return b.length % 2 ? b[m] : (b[m - 1] + b[m]) / 2; };
const quantile = (a, q) => { if (!a.length) return NaN; const b = a.slice().sort((x, y) => x - y); const p = (b.length - 1) * q, i = Math.floor(p), f = p - i; return i + 1 < b.length ? b[i] * (1 - f) + b[i + 1] * f : b[i]; };
const range = (a, b) => { const r = []; for (let i = a; i < b; i++) r.push(i); return r; };
const uniq = a => Array.from(new Set(a));

const fmt = {
  n: (x, d = 0) => x == null || isNaN(x) ? "–" : Number(x).toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d }),
  f: (x, d = 4) => x == null || isNaN(x) ? "–" : Number(x).toFixed(d),
  raw: x => x == null || isNaN(x) ? "–" : Number(x).toFixed(PROB ? PROB.dec : 6),
  rep: x => x == null || isNaN(x) ? "–" : String(+Number(x).toFixed((PROB ? PROB.dec : 6) + 4)),
  rawx: x => x == null || isNaN(x) ? "–" : Number(x).toFixed((PROB ? PROB.dec : 6) + 2),
  pct: (x, d = 1) => x == null || isNaN(x) ? "–" : (100 * x).toFixed(d) + "%",
  sci: (x, d = 2) => x == null || isNaN(x) ? "–" : Number(x).toExponential(d),
  dur: s => { if (s == null || isNaN(s)) return "–"; s = Math.round(s); const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60); return h ? `${h}h ${String(m).padStart(2, "0")}m` : m ? `${m}m ${String(s % 60).padStart(2, "0")}s` : `${s}s`; },
  id: s => s ? String(s).slice(0, 8) : "–",
  dh: s => { if (s == null || isNaN(s)) return "–"; let h = Math.round(s / 3600); const d = Math.floor(h / 24); h -= d * 24; return d ? `${d}d ${String(h).padStart(2, "0")}h` : `${h}h`; },
  tok: x => x == null ? "–" : x >= 1e6 ? (x / 1e6).toFixed(1) + "M" : x >= 1000 ? (x / 1000).toFixed(1) + "k" : String(x),
  gap: x => x == null || isNaN(x) ? "–" : (x >= 0 ? "+" : "−") + Math.abs(x).toPrecision(3),
  pctGap: x => x == null || isNaN(x) ? "–" : (x >= 0 ? "+" : "−") + Math.abs(x * 100).toPrecision(2) + "%",
  sign: (x, d = 3) => x == null || isNaN(x) ? "–" : (x >= 0 ? "+" : "") + Number(x).toFixed(d),
};
const cssVar = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const S_COL = i => cssVar("--s" + ((i % 8) + 1));
const STRAT_LABEL = ["S0", "S1", "S2", "S3"];

/* hyperscript */
function h(tag, attrs, ...kids) {
  const e = document.createElement(tag);
  if (attrs) for (const k in attrs) {
    const v = attrs[k];
    if (v == null || v === false) continue;
    if (k === "class") e.className = v;
    else if (k === "html") e.innerHTML = v;
    else if (k.startsWith("on") && typeof v === "function") e.addEventListener(k.slice(2), v);
    else if (k === "style" && typeof v === "object") Object.assign(e.style, v);
    else e.setAttribute(k, v === true ? "" : v);
  }
  for (const c of kids.flat(Infinity)) {
    if (c == null || c === false) continue;
    e.append(c.nodeType ? c : document.createTextNode(String(c)));
  }
  return e;
}

/* ------------------------------------------------------------------ data loading -- */
const D = { main: null, heavy: null, strat: null, logs: null };
const _pending = {};
async function inflate(id) {
  const tag = document.getElementById("blob-" + id);
  if (!tag) throw new Error("missing data blob " + id);
  const b64 = tag.textContent.trim();
  const bin = Uint8Array.from(atob(b64), c => c.charCodeAt(0));
  if (typeof DecompressionStream === "undefined") throw new Error("This browser lacks DecompressionStream; please use a current Chrome, Edge, Safari 16.4+, or Firefox.");
  const stream = new Blob([bin]).stream().pipeThrough(new DecompressionStream("gzip"));
  return JSON.parse(await new Response(stream).text());
}
function lazy(id) {
  if (D[id]) return Promise.resolve(D[id]);
  if (!_pending[id]) _pending[id] = inflate(id).then(x => (D[id] = x));
  return _pending[id];
}

/* derived structures, filled in by prepare() */
let R = [];            // rollout objects
let RS = {};           // step -> rollouts
let RN = {};           // node id -> rollout
let ST = {};           // step -> step record
let STEPS = [];        // completed step numbers
let BSTEPS = [];       // steps that count for "best so far": completed ones + steps whose rollouts are scored but whose update is still running (provisional)
let ALL_STEPS = [];    // every step directory (incl. partial)
let TREE = null;
let CFG = {};

function prepare() {
  const m = D.main;
  CFG = m.config || {};
  const cols = m.rcols;
  R = m.rows.map(r => { const o = {}; cols.forEach((c, i) => (o[c] = r[i])); return o; });
  RS = {}; RN = {};
  for (const r of R) {
    (RS[r.step] = RS[r.step] || []).push(r);
    if (r.node) RN[r.node] = r;
    r.reward = r.reward == null ? 0 : r.reward;
    r.improved = !!(r.valid && r.pval != null && r.reward > r.pval + 1e-12);
    r.pg = r.pgroup != null ? r.pgroup : r.group;
    r.isRetry = (r.retry || 0) > 0 || r.counts === false;
  }
  ST = {};
  for (const s of m.steps) ST[s.step] = s;
  ALL_STEPS = m.steps.map(s => s.step);
  STEPS = m.steps.filter(s => s.completed).map(s => s.step);
  BSTEPS = m.steps.filter(s => s.completed || s.provisional).map(s => s.step);
  initParams();
  TREE = m.tree;
  TREE.idx = {};
  TREE.ids.forEach((id, i) => (TREE.idx[id] = i));
  // children lists
  TREE.kids = TREE.ids.map(() => []);
  TREE.nodes.forEach((n, i) => { if (n[0] >= 0) TREE.kids[n[0]].push(i); });
  setupProblem();
  { // the reference value: default from the published data, remembered per problem
    let saved = null; try { saved = localStorage.getItem("rx-ref-" + PROB.id); } catch (e) { /* ignore */ }
    const l = refList().map(r => r.id);
    P.ref = l.includes(saved) ? saved : ((D.pub && l.includes(D.pub.default_ref)) ? D.pub.default_ref : (l.includes("prompt") ? "prompt" : l[0]));
  }
}

/* ------------------------------------------------------------------ global state -- */
const S = { tab: "overview", step: 0, group: 0, sidx: 0, listeners: [] };
function setState(patch, silent) {
  Object.assign(S, patch);
  syncHash();
  if (!silent) renderTab();
}
function syncHash() {
  try { history.replaceState(null, "", `#${S.tab}/${S.step}/${S.group}/${S.sidx}`); } catch (e) { /* file:// quirks */ }
}
function readHash() {
  const p = (location.hash || "").replace(/^#/, "").split("/");
  if (p[0]) S.tab = p[0];
  if (p[1] != null && p[1] !== "" && !isNaN(+p[1])) S.step = +p[1];
  if (p[2] != null && p[2] !== "" && !isNaN(+p[2])) S.group = +p[2];
  if (p[3] != null && p[3] !== "" && !isNaN(+p[3])) S.sidx = +p[3];
}

/* ------------------------------------------------------------------ tooltip -- */
const tipEl = () => $("#tip");
function tipShow(html, ev) {
  const t = tipEl(); t.innerHTML = html; t.classList.add("on");
  const pad = 14, w = t.offsetWidth, hgt = t.offsetHeight;
  let x = ev.clientX + pad, y = ev.clientY + pad;
  if (x + w > innerWidth - 8) x = ev.clientX - w - pad;
  if (y + hgt > innerHeight - 8) y = ev.clientY - hgt - pad;
  t.style.left = Math.max(4, x) + "px"; t.style.top = Math.max(4, y) + "px";
}
function tipHide() { tipEl().classList.remove("on"); }
function bindTip(el, fn) {
  el.addEventListener("mousemove", ev => { const html = fn(ev); if (html) tipShow(html, ev); else tipHide(); });
  el.addEventListener("mouseleave", tipHide);
}

/* ------------------------------------------------------------------ drawer -- */
function openDrawer(title, node) {
  $("#drawerTitle").textContent = title;
  const b = $("#drawerBody"); b.innerHTML = ""; b.append(node);
  const d = $("#drawer"); d.classList.add("open"); d.setAttribute("aria-hidden", "false");
}
function closeDrawer() { const d = $("#drawer"); d.classList.remove("open"); d.setAttribute("aria-hidden", "true"); }

/* ------------------------------------------------------------------ syntax highlight (python) -- */
const PY_KW = "def|class|import|from|return|if|elif|else|for|while|in|not|and|or|is|None|True|False|try|except|finally|with|as|lambda|pass|break|continue|raise|global|nonlocal|yield|assert";
const PY_RE = new RegExp("(#[^\\n]*)|(\"\"\"[\\s\\S]*?\"\"\"|'''[\\s\\S]*?'''|\"(?:\\\\.|[^\"\\\\\\n])*\"|'(?:\\\\.|[^'\\\\\\n])*')|\\b(" + PY_KW + ")\\b|\\b(\\d+\\.?\\d*(?:[eE][-+]?\\d+)?)\\b", "g");
function highlightPy(src) {
  let out = "", last = 0, m;
  PY_RE.lastIndex = 0;
  while ((m = PY_RE.exec(src))) {
    out += esc(src.slice(last, m.index));
    const cls = m[1] ? "cm" : m[2] ? "str" : m[3] ? "kw" : "num-c";
    out += `<span class="${cls}">${esc(m[0])}</span>`;
    last = m.index + m[0].length;
    if (m[0].length === 0) PY_RE.lastIndex++;
  }
  return out + esc(src.slice(last));
}
function codeBlock(src, maxLines) {
  const pre = h("pre", { class: "code" });
  pre.innerHTML = highlightPy(src);
  return pre;
}
/* line diff (LCS) used by the drawer to show what the coder changed relative to the parent program */
function lineDiff(a, b) {
  const A = a.split("\n"), B = b.split("\n"), n = A.length, m = B.length;
  if (n * m > 6e6) return null;
  const W = m + 1, dp = new Uint16Array((n + 1) * W);
  for (let i = n - 1; i >= 0; i--) for (let j = m - 1; j >= 0; j--)
    dp[i * W + j] = A[i] === B[j] ? dp[(i + 1) * W + j + 1] + 1 : Math.max(dp[(i + 1) * W + j], dp[i * W + j + 1]);
  const ops = []; let i = 0, j = 0;
  while (i < n && j < m) {
    if (A[i] === B[j]) { ops.push([" ", A[i]]); i++; j++; }
    else if (dp[(i + 1) * W + j] >= dp[i * W + j + 1]) { ops.push(["-", A[i]]); i++; }
    else { ops.push(["+", B[j]]); j++; }
  }
  while (i < n) ops.push(["-", A[i++]]);
  while (j < m) ops.push(["+", B[j++]]);
  return ops;
}
function diffBlock(parentCode, childCode) {
  const ops = lineDiff(parentCode, childCode);
  if (!ops) return h("div", { class: "hint" }, "Programs are too long to diff in the browser.");
  const add = ops.filter(o => o[0] === "+").length, del = ops.filter(o => o[0] === "-").length, same = ops.length - add - del;
  const keep = new Array(ops.length).fill(false), CTX = 3;
  ops.forEach((o, i) => { if (o[0] !== " ") for (let k = Math.max(0, i - CTX); k <= Math.min(ops.length - 1, i + CTX); k++) keep[k] = true; });
  const pre = h("pre", { class: "code diff" });
  let html = "", skipped = 0;
  const flush = () => { if (skipped) { html += `<span class="df-sk">  … ${skipped} unchanged lines …</span>`; skipped = 0; } };
  ops.forEach((o, i) => {
    if (!keep[i]) { skipped++; return; }
    flush();
    html += `<span class="${o[0] === "+" ? "df-add" : o[0] === "-" ? "df-del" : "df-eq"}">${esc(o[0] + " " + o[1])}</span>`;
  });
  flush();
  pre.innerHTML = html;
  const sim = same / Math.max(1, Math.max(ops.length - add, ops.length - del));
  return h("div", null, h("div", { class: "sub", style: { marginBottom: "6px" } }, `+${add} / −${del} lines · ${fmt.pct(same / Math.max(1, same + del), 0)} of the parent's lines survive unchanged`), pre);
}
function textBlock(txt) { const p = h("pre", { class: "text" }); p.textContent = txt; return p; }

/* ------------------------------------------------------------------ misc components -- */
function card(title, sub, body, opts) {
  opts = opts || {};
  return h("div", { class: "card" + (opts.flat ? " flat" : "") },
    (title || sub) ? h("div", { class: "card-head" }, title ? h("h2", null, title) : null, sub ? h("div", { class: "sub", html: sub }) : null) : null,
    body);
}
function kpi(k, v, d) { return h("div", { class: "kpi" }, h("div", { class: "k" }, k), h("div", { class: "v", html: String(v) }), d ? h("div", { class: "d", html: d }) : null); }
function pill(text, cls) { return h("span", { class: "pill " + (cls || "") }, text); }
function seg(options, value, onChange) {
  const wrap = h("div", { class: "seg" });
  options.forEach(o => {
    const b = h("button", { class: o.value === value ? "on" : "", onclick: () => { $$("button", wrap).forEach(x => x.classList.remove("on")); b.classList.add("on"); onChange(o.value); } }, o.label);
    wrap.append(b);
  });
  return wrap;
}
function slider(label, min, max, step, value, onInput, fmtv) {
  const val = h("span", { class: "val" }, (fmtv || (x => x))(value));
  const inp = h("input", { type: "range", min, max, step, value });
  inp.addEventListener("input", () => { const v = +inp.value; val.textContent = (fmtv || (x => x))(v); onInput(v); });
  const row = h("div", { class: "slider" }, h("span", null, label), inp, val);
  row.setValue = v => { inp.value = v; val.textContent = (fmtv || (x => x))(v); };
  return row;
}
function stepSuffix(s) { const st = ST[s]; return st.completed ? "" : st.provisional ? " (rollouts done)" : " (in progress)"; }
function stepSelect(onChange, includePartial) {
  const sel = h("select", null, (includePartial ? ALL_STEPS : STEPS).map(s => h("option", { value: s, selected: s === S.step }, "step " + s + stepSuffix(s))));
  sel.addEventListener("change", () => onChange(+sel.value));
  return sel;
}
function table(cols, rows, opts) {
  opts = opts || {};
  const t = h("table", { class: "t" });
  const head = h("tr", null, cols.map((c, ci) => h("th", {
    class: (c.num ? "num " : "") + (opts.sortable ? "sortable" : ""), title: c.title || "",
    onclick: opts.sortable ? () => opts.onSort && opts.onSort(ci) : null
  }, c.label + (opts.sortCol === ci ? (opts.sortDir > 0 ? " ▲" : " ▼") : ""))));
  t.append(h("thead", null, head));
  const tb = h("tbody");
  rows.forEach((r, ri) => {
    const tr = h("tr", { class: "row" + (opts.selected === ri ? " sel" : ""), onclick: opts.onRow ? () => opts.onRow(r, ri) : null });
    cols.forEach(c => {
      const v = c.get(r, ri);
      const td = h("td", { class: c.num ? "num" : "" });
      if (v && v.nodeType) td.append(v); else if (c.html) td.innerHTML = v; else td.textContent = v == null ? "" : v;
      tr.append(td);
    });
    tb.append(tr);
  });
  t.append(tb);
  return t;
}
function legend(items, onToggle) {
  const el = h("div", { class: "legend" });
  items.forEach(it => {
    const x = h("span", { class: "it" + (onToggle ? " clk" : "") + (it.off ? " off" : ""), onclick: onToggle ? () => { onToggle(it); x.classList.toggle("off"); } : null },
      h("span", { class: "sw", style: { background: it.color, borderRadius: it.line ? "2px" : "3px", height: it.line ? "3px" : "11px" } }), it.label);
    el.append(x);
  });
  return el;
}
function explain(title, body) {
  return h("details", { class: "fold" }, h("summary", null, title), h("div", { html: body, style: { padding: "6px 2px 2px", color: "var(--ink-2)" } }));
}
function pageHead(title, lede) {
  return h("div", null, h("h1", null, title), lede ? h("p", { class: "lede", html: lede }) : null);
}
function rewardToRaw(r) { return r > 0 ? 1 / r : 0; }

/* ------------------------------------------------------------------ rollout / node drawer -- */
function strategyKey(step, pg, s) { return `${step}:${pg}:${s}`; }
function rolloutTitle(r) { return `Rollout · step ${r.step} · parent ${r.pg} · S${r.sidx} · #${r.rollout}`; }

async function openRollout(rid) {
  const r = R[rid]; if (!r) return;
  const root = h("div", null, h("div", { class: "sub" }, "loading…"));
  openDrawer(rolloutTitle(r), root);
  const [heavy, strat] = await Promise.all([lazy("heavy"), lazy("strat")]);
  root.innerHTML = "";
  const st = ST[r.step] || {};
  const spo = st.spo;
  const beta = 2.0;
  const G = Math.exp(beta * r.reward);
  const v = spo ? spo.value_before : null;
  const advCalc = v ? (G / v - 1) / beta : null;
  const kv = [
    ["Outcome", r.valid ? `<span class="pill good">valid</span> ${esc(PROB.metric)} <b>${fmt.raw(r.raw)}</b> (reward ${fmt.f(r.reward, 6)})` : `<span class="pill bad">invalid</span> ${esc(r.fail || "failed")} - ${esc(r.msg)}`],
    ["Phase", `${r.phase === "adaptive" ? "phase 2 (hurdle-allocated)" : "pilot"} · strategy S${r.sidx}${r.isRetry ? ' · <span class="pill warn">format retry</span>' : ""}${r.fallback ? ' · <span class="pill warn">strategist fallback</span>' : ""}`],
    ["Node / parent", `<span class="mono">${esc(fmt.id(r.node))}</span> ← <a href="javascript:void 0" class="mono" id="lnkParent">${esc(fmt.id(r.parent))}</a>`],
    ["Parent score", `raw ${fmt.raw(r.praw)} (reward ${fmt.f(r.pval, 6)}) ${r.valid ? (r.improved ? '<span class="pill good">beats parent</span>' : '<span class="pill">not better than parent</span>') : ""}`],
    ["Response tokens", `${fmt.n(r.ntok)}${r.think_chars != null ? ` · reasoning ${fmt.n(r.think_chars)} chars` : ""}`],
    ["Sequence length n", r.n ? fmt.n(r.n) : "–"],
    ["SPO² advantage", `logged <b>${fmt.f(r.adv, 5)}</b>` + (advCalc != null ? ` · recomputed (exp(2·r)/v − 1)/2 with v=${fmt.f(v, 4)} → ${fmt.f(advCalc, 5)}` : "") +
      (advIsWhatIf() ? ` · <span class="what-if">what-if</span> β=${P.beta.toFixed(1)}, v=${fmt.f(advParams(r.step).v, 4)} → <b>${fmt.f(advWhatIf(r), 5)}</b>` : "")],
  ];
  root.append(h("div", { class: "kv", html: kv.map(([k, vv]) => `<div class="k">${k}</div><div class="v">${vv}</div>`).join("") }));
  const lp = $("#lnkParent", root);
  if (lp) lp.addEventListener("click", () => openNode(r.parent));
  // strategy
  const sk = strat.strategies[`${r.step}:${r.pg}:${r.sidx}`];
  if (sk) {
    root.append(h("h3", { style: { marginTop: "16px" } }, "Strategy given to the coder (from the strategist)"));
    root.append(sk.fallback ? h("div", { class: "warnbox" }, "The strategist failed all 4 format attempts - the coder got the fallback text and had to derive the approach itself.") : null);
    root.append(textBlock(sk.block || "(not available)"));
  }
  // construction
  const cons = heavy.cons[String(rid)];
  if (cons) {
    if (PROB.prep) {
      root.append(h("h3", { style: { marginTop: "16px" } }, "Construction (step function) and its " + PROB.gShort));
      const host = h("div", { class: "chart" }); root.append(host);
      const dfm = PROB.variants ? displayForm(cons) : { vals: cons, name: "", changed: false };
      requestAnimationFrame(() => solutionPlot(host, [{ name: "this program", vals: dfm.vals, color: S_COL(0) }], { compact: true }));
      if (dfm.changed) root.append(h("div", { class: "sub" }, `drawn as ${dfm.name} - an equivalent form with exactly the same score (standard orientation; toggle it on the Solution tab)`));
    }
  }
  // code
  const code = heavy.code[String(rid)];
  root.append(h("h3", { style: { marginTop: "16px" } }, "Program written by the coder"));
  root.append(code ? codeBlock(code) : h("div", { class: "hint" }, "Source for this rollout is not embedded (only the best 25 per step plus every selected parent are). Rebuild with ", h("code", null, "--all-code"), " to include everything."));
  const pr = RN[r.parent], pcode = pr ? heavy.code[String(pr.id)] : heavy.seed_code[r.parent];
  if (code && pcode) {
    root.append(h("h3", { style: { marginTop: "16px" } }, "What changed relative to its parent program"));
    root.append(diffBlock(pcode, code));
  }
  const so = heavy.stdout[String(rid)];
  if (so) {
    root.append(h("h3", { style: { marginTop: "16px" } }, "Sandbox stdout (first 1200 chars)"));
    root.append(textBlock(so));
  }
}

async function openNode(nodeId) {
  const r = RN[nodeId];
  if (r) return openRollout(r.id);
  const heavy = await lazy("heavy");
  const root = h("div", null);
  const idx = TREE.idx[nodeId];
  const n = idx != null ? TREE.nodes[idx] : null;
  root.append(h("div", { class: "kv", html: `<div class="k">Node</div><div class="v mono">${esc(nodeId)}</div><div class="k">Raw score</div><div class="v">${n ? fmt.raw(n[1]) : "–"}</div><div class="k">Kind</div><div class="v">${n && n[0] < 0 ? "seed (root state)" : "unknown"}</div>` }));
  const code = heavy.seed_code[nodeId];
  root.append(h("h3", { style: { marginTop: "14px" } }, "Source"));
  root.append(code ? codeBlock(code) : h("div", { class: "hint" }, "Source not embedded."));
  openDrawer("Node " + fmt.id(nodeId), root);
}

/* ------------------------------------------------------------------ tab registry -- */
const TABS = [];
function registerTab(t) { TABS.push(t); }


/* ======================================================================================
   Shared "what-if" parameter store. Every interactive control in every tab reads and writes
   P, and every view that depends on a parameter subscribes with onParam(), so a change made
   in one place is reflected everywhere (also in tabs opened later).
   ====================================================================================== */
const P = { ref: "together" };
let DEFAULT_P = null;
const _pl = [];
function initParams() {
  Object.assign(P, {
    beta: CFG.spo_rs_beta != null ? CFG.spo_rs_beta : 2, dHalf: CFG.spo_rs_d_half != null ? CFG.spo_rs_d_half : 0.06,
    rhoMin: CFG.spo_rs_rho_min != null ? CFG.spo_rs_rho_min : 0.875, rhoMax: CFG.spo_rs_rho_max != null ? CFG.spo_rs_rho_max : 0.96,
    vOver: {}, c: null, thr: null,
    clipLo: CFG.spo_rs_clip_epsilon_low != null ? CFG.spo_rs_clip_epsilon_low : 0.2, clipHi: CFG.spo_rs_clip_epsilon_high != null ? CFG.spo_rs_clip_epsilon_high : 0.38,
  });
  DEFAULT_P = JSON.parse(JSON.stringify(P));
}
function onParam(fn) { _pl.push(fn); }
function clearParamListeners() { _pl.length = 0; }
function setParam(patch, source) {
  Object.assign(P, patch);
  _trkKey = null;
  _pl.slice().forEach(fn => { try { fn(patch, source); } catch (e) { console.error(e); } });
  updateResetBadge();
}
function paramsDirty() {
  if (!DEFAULT_P) return false;
  return ["beta", "dHalf", "rhoMin", "rhoMax", "c", "thr", "clipLo", "clipHi"].some(k => P[k] !== DEFAULT_P[k]) || Object.keys(P.vOver).length > 0;
}
function updateResetBadge() { const b = $("#btnReset"); if (b) b.hidden = !paramsDirty(); }
function resetParams() {
  const ref = P.ref; Object.assign(P, JSON.parse(JSON.stringify(DEFAULT_P)), { ref }); _trkKey = null;
  updateResetBadge(); renderTab();
}
/* a slider bound to the store: getVal() reads, setVal(v, rowEl) writes via setParam */
function paramSlider(label, min, max, step, fmtv, getVal, setVal) {
  const row = slider(label, min, max, step, getVal(), v => setVal(v, row), fmtv);
  onParam((patch, src) => { if (src !== row) row.setValue(getVal()); });
  return row;
}

/* published reference values (from data/published_<problem>.json) + the run's own prompt target ------------ */
function refList() {
  const pub = D.pub, l = [];
  if (pub && pub.results) pub.results.filter(r => r.reported != null).forEach(r => l.push({ id: r.id, label: r.label, val: r.reported, kind: r.kind === "claim" ? "claim" : r.kind === "reported" ? "reported" : "record", note: r.note || "" }));
  l.sort((a, b) => a.val - b.val);
  if (CFG.target) l.push({ id: "prompt", label: "this run's prompt target", val: CFG.target, kind: "record", note: "the target number shown to the coder in the prompt" });
  return l;
}
function REF() { const l = refList(); return l.find(r => r.id === P.ref) || l[0]; }
const TARGET = () => REF().val;
const TARGET_LABEL = () => REF().label;

/* ------------------------------------------------------------------------------------------
   Simulation of the SPO² baseline tracker under the current what-if parameters.
   With the logged parameters it reproduces the run's v_before/v_after (checked in the UI).
   ------------------------------------------------------------------------------------------ */
let _trkKey = null;            // (kept so setParam can invalidate; caches below are keyed by the parameters themselves)
const _simCache = {};
/* pure simulation of the baseline tracker for given constants, starting from the mean exp(beta*r) of the seed rewards */
function simulateTracker(beta, dHalf, rhoMin, rhoMax) {
  const out = { byStep: {}, order: [] };
  const seeds = PROB.seedRewards && PROB.seedRewards.length ? PROB.seedRewards : [PROB.seedReward];
  let v = mean(seeds.map(r => Math.exp(beta * r))), N = 8 / (1 - rhoMin);
  STEPS.forEach(s => {
    const sp = ST[s].spo; if (!sp) return;
    const rs = RS[s] || [], M = sp.group_size || rs.length;
    let sumG = 0; rs.forEach(r => { sumG += Math.exp(beta * r.reward); });
    const D = Math.max(0, sp.divergence || 0);
    const rho = clamp(Math.pow(2, -D / dHalf), rhoMin, rhoMax), kept = rho * N, Nc = kept + M;
    const v2 = (kept * v + sumG) / Nc;
    out.byStep[s] = { vb: v, va: v2, Nb: N, Na: Nc, eta: M / Nc, rho, meanG: sumG / Math.max(1, rs.length), M, D };
    out.order.push(s);
    v = v2; N = Nc;
  });
  return out;
}
function trackerSimRaw() { const k = [P.beta, P.dHalf, P.rhoMin, P.rhoMax].join("|"); return _simCache[k] || (_simCache[k] = simulateTracker(P.beta, P.dHalf, P.rhoMin, P.rhoMax)); }
function trackerSimDefault() { const k = "def|" + [DEFAULT_P.beta, DEFAULT_P.dHalf, DEFAULT_P.rhoMin, DEFAULT_P.rhoMax].join("|"); return _simCache[k] || (_simCache[k] = simulateTracker(DEFAULT_P.beta, DEFAULT_P.dHalf, DEFAULT_P.rhoMin, DEFAULT_P.rhoMax)); }
function trackerLogged() {
  if (_simCache.logged) return _simCache.logged;
  const out = { byStep: {}, order: [] };
  STEPS.forEach(s => {
    const sp = ST[s].spo; if (!sp) return;
    out.byStep[s] = { vb: sp.value_before, va: sp.value_after, Nb: sp.effective_count_before, Na: sp.effective_count_after, eta: sp.eta, rho: sp.rho, meanG: sp.group_mean, M: sp.group_size, D: sp.divergence };
    out.order.push(s);
  });
  return (_simCache.logged = out);
}
function trackerParamsChanged() { return P.beta !== DEFAULT_P.beta || P.dHalf !== DEFAULT_P.dHalf || P.rhoMin !== DEFAULT_P.rhoMin || P.rhoMax !== DEFAULT_P.rhoMax; }
/* the baselines used everywhere: exactly the logged ones until a tracker constant is changed, then a re-simulation of the whole run */
function trackerSim() { return trackerParamsChanged() ? trackerSimRaw() : trackerLogged(); }
function advParams(step) {
  const t = trackerSim().byStep[step];
  const v = P.vOver[step] != null ? P.vOver[step] : (t ? t.vb : null);
  return { beta: P.beta, v, simV: t ? t.vb : null };
}
function advWhatIf(r) { const a = advParams(r.step); return a.v ? (Math.exp(a.beta * r.reward) / a.v - 1) / a.beta : null; }
function advIsWhatIf() { return P.beta !== DEFAULT_P.beta || P.dHalf !== DEFAULT_P.dHalf || P.rhoMin !== DEFAULT_P.rhoMin || P.rhoMax !== DEFAULT_P.rhoMax || Object.keys(P.vOver).length > 0; }

/* ------------------------------------------------------------------ entry animations / playback -- */
let ANIM = false;
const REDUCED = typeof matchMedia === "function" && matchMedia("(prefers-reduced-motion: reduce)").matches;
function animWindow() { ANIM = !REDUCED; clearTimeout(animWindow._t); animWindow._t = setTimeout(() => { ANIM = false; }, 1600); }
let _play = null;
function playToggle() {
  const btn = $("#btnPlay");
  if (_play) { clearInterval(_play); _play = null; if (btn) { btn.classList.remove("on"); btn.textContent = "▶ play steps"; } return; }
  if (btn) { btn.classList.add("on"); btn.textContent = "■ stop"; }
  if (STEPS.indexOf(S.step) === STEPS.length - 1) S.step = STEPS[0];
  setState({ step: S.step });
  _play = setInterval(() => {
    const i = STEPS.indexOf(S.step);
    if (i >= STEPS.length - 1) { playToggle(); return; }
    setState({ step: STEPS[i + 1] });
  }, 2600);
}
