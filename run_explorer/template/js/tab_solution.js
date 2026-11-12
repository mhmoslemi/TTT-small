"use strict";
/* ---- Solution tab: the construction itself, every step, and the published constructions ----------- */
let _sol = { a: "run:last", b: "pub:together", morph: null, align: true };
const _prepCache = {};

/* normalised step function f, profile g (f*f for AC1, overlap profile for Erdős) and its peak, via the problem profile */
function prepProgram(key, vals) {
  if (key && _prepCache[key]) return _prepCache[key];
  const out = PROB.prep(vals);
  if (key) _prepCache[key] = out;
  return out;
}
/* bucket-max downsample of an array to at most `w` points */
function bucketMax(arr, w) {
  const L = arr.length; if (L <= w) return Array.from(arr);
  const out = new Array(w);
  for (let b = 0; b < w; b++) { const i0 = Math.floor(b * L / w), i1 = Math.max(i0 + 1, Math.floor((b + 1) * L / w)); let m = -Infinity; for (let i = i0; i < i1; i++) if (arr[i] > m) m = arr[i]; out[b] = m; }
  return out;
}

function miniStep(vals, w, hgt, color) {
  const arr = bucketMax(vals, 220), n = arr.length, mx = Math.max(...arr) || 1;
  const svg = sv("svg", { viewBox: `0 0 ${w} ${hgt}`, width: "100%", height: hgt });
  let d = ""; arr.forEach((v, i) => { const x = i / n * w, y = hgt - 3 - (v / mx) * (hgt - 8); d += (i ? `V${y}` : `M${x},${y}`) + `H${(i + 1) / n * w}`; });
  sv("path", { d: d + `V${hgt - 3}H0Z`, fill: color, opacity: 0.18 }, svg);
  sv("path", { d, fill: "none", stroke: color, "stroke-width": 1 }, svg);
  return svg;
}
/* tiny autoconvolution panel on a fixed y range so every step is comparable; dashed line = reference value */
function miniAuto(g, w, hgt, color, ymax) {
  const arr = bucketMax(g, 220), n = arr.length, svg = sv("svg", { viewBox: `0 0 ${w} ${hgt}`, width: "100%", height: hgt });
  const Y = v => hgt - 3 - (clamp(v, 0, ymax) / ymax) * (hgt - 8);
  let d = ""; arr.forEach((v, i) => { d += (i ? "L" : "M") + `${(i + 0.5) / n * w},${Y(v)}`; });
  sv("path", { d: d + `L${w},${hgt - 3}L0,${hgt - 3}Z`, fill: color, opacity: 0.16 }, svg);
  sv("path", { d, fill: "none", stroke: color, "stroke-width": 1.1, "stroke-linejoin": "round" }, svg);
  sv("line", { x1: 0, x2: w, y1: Y(TARGET()), y2: Y(TARGET()), stroke: cssVar("--good"), "stroke-width": 1, "stroke-dasharray": "3 3" }, svg);
  return svg;
}

registerTab({
  id: "solution", label: "Solution (AC1)", ico: "▲",
  async render(root) {
    const heavy = await lazy("heavy");
    const pub = await lazy("pub");
    const snaps = heavy.best.filter(b => b.construction && b.construction.length);
    const pubs = ((pub && pub.results) || []).filter(r => r.values && r.values.length);
    root.append(pageHead("The construction the search found",
      PROB.solIntro + ` Everything is recomputed in your browser from the saved sequences - for this run <b>and</b> for the published constructions (${pubs.length ? pubs.map(p => esc(p.method)).filter((x, i, a) => a.indexOf(x) === i).join(", ") : "none embedded"}) - so numbers are verified, not just read from a log.`));
    if (!PROB.prep) { root.append(h("div", { class: "hint" }, "No plot is defined for this problem.")); return; }
    /* ---------- program catalogue for the two selectors ---------- */
    const lastSnap = snaps[snaps.length - 1];
    const options = [];
    snaps.forEach(s => options.push({ key: "run:" + s.step, group: "This run", label: `this run · best through step ${s.step} · ${fmt.f(s.raw, 6)} · n=${s.construction.length}`, vals: s.construction, rep: s.raw, kind: "run", snap: s }));
    pubs.forEach(r => options.push({ key: "pub:" + r.id, group: "Published", label: `${r.label} · ${fmt.f(r.reported, 6)} · n=${r.values.length}`, vals: r.values, rep: r.reported, kind: "pub", entry: r }));
    const resolve = k => k === "none" ? null : (k === "run:last" ? options.find(o => o.key === "run:" + lastSnap.step) : options.find(o => o.key === k));
    const df = v => _sol.align !== false && PROB.variants ? displayForm(v) : { vals: v, name: "", changed: false };
    if (!resolve(_sol.a)) _sol.a = "run:last";
    if (_sol.b !== "none" && !resolve(_sol.b)) _sol.b = "none";
    const mkSel = (val, withNone, onChange) => {
      const s = h("select", { style: { maxWidth: "380px" } }, withNone ? h("option", { value: "none" }, "none") : null);
      ["This run", "Published"].forEach(g => { const og = h("optgroup", { label: g }); options.filter(o => o.group === g).forEach(o => og.append(h("option", { value: o.key }, o.label))); if (og.children.length) s.append(og); });
      s.value = val === "run:last" ? "run:" + lastSnap.step : val; s.addEventListener("change", () => onChange(s.value)); return s;
    };

    /* ---------- main comparison plot ---------- */
    const host = h("div", { class: "chart" }), info = h("div");
    let timer = null;
    const morphBtn = h("button", { class: "btn sm", onclick: () => toggleMorph() }, "▶ morph through the steps");
    const stopMorph = () => { if (timer) { clearInterval(timer); timer = null; morphBtn.classList.remove("on"); morphBtn.textContent = "▶ morph through the steps"; } };
    const selA = mkSel(_sol.a, false, v => { stopMorph(); _sol.a = v; drawMain(); });
    const selB = mkSel(_sol.b, true, v => { _sol.b = v; drawMain(); });
    const toggleMorph = () => {
      if (timer) { stopMorph(); return; }
      let i = 0; morphBtn.classList.add("on"); morphBtn.textContent = "■ stop";
      _sol.a = "run:" + snaps[0].step; selA.value = _sol.a; drawMain();
      timer = setInterval(() => {
        if (!host.isConnected) { stopMorph(); return; }
        i++; if (i >= snaps.length) { stopMorph(); return; }
        _sol.a = "run:" + snaps[i].step; selA.value = _sol.a; drawMain();
      }, 1100);
    };
    root.append(card("Step function and autoconvolution", "hover to read values · pick any two constructions", h("div", { class: "grid g-main" },
      h("div", null, h("div", { class: "row-flex", style: { marginBottom: "8px" } }, h("span", { class: "sub" }, "A"), selA, h("span", { class: "sub" }, "B"), selB, morphBtn,
        PROB.variants ? h("label", { class: "ctl", title: "Equivalent forms of the same construction (" + (PROB.id === "erdos" ? "h, its complement 1 − h, and the mirror images" : "f and its mirror image") + ") score exactly the same. Draw everything in one orientation" + (PROB.canon ? " (edges at 0, like most published figures; mirrored to match this run's best)" : " (mirrored to match this run's best)") + " so shapes can be compared by eye." },
          (() => { const cb = h("input", { type: "checkbox" }); cb.checked = _sol.align !== false; cb.addEventListener("change", () => { _sol.align = cb.checked; renderTab(); }); return cb; })(), "standard orientation") : null), host, h("div", { id: "solLegend" })), info)));
    function drawMain() {
      const A = resolve(_sol.a), B = resolve(_sol.b);
      const da = df(A.vals), db = B ? df(B.vals) : null;
      const progs = [{ name: A.label.split(" · ").slice(0, 2).join(" · "), vals: da.vals, color: S_COL(0), opt: A, shownAs: da.changed ? da.name : "" }];
      if (B) progs.push({ name: B.label.split(" · ").slice(0, 2).join(" · "), vals: db.vals, color: S_COL(1), opt: B, shownAs: db.changed ? db.name : "" });
      solutionPlot(host, progs);
      const lg = $("#solLegend"); lg.innerHTML = ""; lg.append(legend(progs.map(p => ({ color: p.color, label: p.name, line: true }))));
      info.innerHTML = "";
      const vals = progs.map(p => constructionScore(p.vals));
      progs.forEach((p, i) => {
        const o = p.opt, val = vals[i];
        info.append(h("div", { class: "kv", style: { marginBottom: "10px" }, html:
          `<div class="k"><span class="dot" style="background:${p.color}"></span><b>${i ? "B" : "A"}</b></div><div class="v">${esc(p.name)}</div>` +
          `<div class="k">length n</div><div class="v">${fmt.n(p.vals.length)}</div>` +
          `<div class="k">${o.kind === "run" ? "logged bound" : "reported bound"}</div><div class="v">${fmt.rawx(o.rep)}</div>` +
          `<div class="k">recomputed here</div><div class="v"><b>${fmt.rawx(val)}</b> <span class="sub">(Δ ${fmt.sci(Math.abs(val - o.rep), 1)}${o.kind === "run" ? "; construction stored to 9 significant digits" : ""})</span></div>` +
          `<div class="k">gap to reference</div><div class="v">${fmt.gap(val - TARGET())} (${fmt.pctGap(val / TARGET() - 1)}) <span class="sub">${esc(TARGET_LABEL())}</span></div>` +
          (o.kind === "run" ? `<div class="k">found in step</div><div class="v">${o.snap.found_step}</div>` : `<div class="k">source</div><div class="v"><a href="${esc(o.entry.url)}" target="_blank" rel="noopener">${esc(o.entry.method)} · ${esc(o.entry.date || "")}</a> <span class="sub">${esc(o.entry.note || "")}</span></div>`) +
          (p.shownAs ? `<div class="k">drawn as</div><div class="v"><b>${esc(p.shownAs)}</b> <span class="sub">- an equivalent form with exactly the same score; untick "standard orientation" to see it as stored</span></div>` : "") +
          `<div class="k">max / mean height</div><div class="v">${fmt.f(Math.max(...p.vals) / mean(Array.from(p.vals)), 2)}× <span class="sub">(peaky = large)</span></div>` }));
      });
      if (progs.length === 2) info.append(h("div", { class: "hint" }, `A − B = ${fmt.gap(vals[0] - vals[1])} - ${vals[0] > vals[1] ? "B is lower (better)" : "A is lower (better)"}.`));
      info.append(h("p", { class: "sub" }, PROB.solNote + " Constructions with thousands of steps are drawn as min/max envelopes."));
    }
    drawMain();

    /* ---------- small multiples: step function AND autoconvolution after every step ---------- */
    const alignPub = r => df(r.values);
    const YM = PROB.profileYmax || Math.max(...snaps.map(x => prepProgram("run:" + x.step, x.construction).mx)) * 1.1;
    const mult = (title, sub, build) => {
      const grid = h("div", { style: { display: "grid", gridTemplateColumns: "repeat(auto-fill,minmax(190px,1fr))", gap: "10px" } });
      snaps.forEach(s => grid.append(build({ key: "run:" + s.step + "@" + df(s.construction).name, vals: df(s.construction).vals, head: "step " + s.step, val: s.raw, tag: `n=${s.construction.length} · found in step ${s.found_step}`, color: S_COL(0), run: true, click: () => { stopMorph(); _sol.a = "run:" + s.step; renderTab(); } })));
      const pubGrid = h("div", { style: { display: "grid", gridTemplateColumns: "repeat(auto-fill,minmax(190px,1fr))", gap: "10px", marginTop: "12px" } });
      pubs.slice().sort((a, b) => a.reported - b.reported).forEach(r => pubGrid.append(build({ key: "pub:" + r.id + "@" + alignPub(r).name, vals: alignPub(r).vals, head: r.label.replace(/ \(.*\)/, ""), val: r.reported, tag: `n=${fmt.n(r.values.length)} · ${r.date || ""}`, color: S_COL(1), run: false, click: () => { _sol.b = "pub:" + r.id; renderTab(); } })));
      return card(title, sub, h("div", null, grid, pubs.length ? h("div", { class: "sub", style: { margin: "14px 0 -4px", fontWeight: 600 } }, "Published constructions (click to put them in slot B above" + (PROB.variants && _sol.align !== false ? "; drawn in the standard orientation" : "") + ")") : null, pubs.length ? pubGrid : null));
    };
    const panel = (o, draw) => h("div", { class: "card flat", style: { padding: "8px 10px", cursor: "pointer", borderColor: o.run && _sol.a === o.key ? cssVar("--accent") : "" }, onclick: o.click },
      h("div", { class: "row-flex", style: { justifyContent: "space-between", fontSize: "12px" } }, h("b", null, o.head), h("span", { class: "mono" }, fmt.f(o.val, 6))), draw(o), h("div", { class: "sub" }, o.tag));
    root.append(h("div", { style: { marginTop: "14px" } }, mult("Best-so-far construction after each step", PROB.fShort + " (heights normalised per panel); click to inspect", o => panel(o, x => miniStep(x.vals, 180, 56, x.color)))));
    root.append(h("div", { style: { marginTop: "14px" } }, mult(PROB.gShort.charAt(0).toUpperCase() + PROB.gShort.slice(1) + " after each step", `same y-range (0 … ${+YM.toFixed(2)}) in every panel; dashed green = the reference value - watch the profile flatten and its peak come down`, o => panel(o, x => miniAuto(prepProgram(x.key, x.vals).g, 180, 64, x.color, YM)))));

    /* ---------- overlay of all steps ---------- */
    const hostO = h("div", { class: "chart" });
    root.append(h("div", { style: { marginTop: "14px" } }, card("All steps on one axis", PROB.gShort + " of every best-so-far construction: light = early step, dark = late step; orange dashed = a published construction (the best one embedded)",
      h("div", null, hostO, h("div", { class: "sub", style: { marginTop: "4px" } }, PROB.overlayNote)))));
    registerChart(hostO, () => {
      const F = frame(hostO, { height: 300, margin: { l: 48, b: 34, t: 10, r: 14 } }), { svg, m, iw, ih } = F;
      const xs = linear(0, 1, m.l, m.l + iw), ys = linear(0, YM, m.t + ih, m.t);
      axes(F, xs, ys, { xLabel: PROB.gLabel, xTicks: [], yFmt: v => v.toFixed(YM < 1 ? 2 : 1) });
      hline(F, ys, TARGET(), `${TARGET_LABEL()} ${+TARGET().toFixed(PROB.dec + 2)}`, cssVar("--good-ink"), "5 4", "end");
      const draw1 = (g, color, w, dash) => { const arr = bucketMax(g, Math.floor(iw)); let d = ""; arr.forEach((v, i) => { d += (i ? "L" : "M") + `${m.l + (i + 0.5) / arr.length * iw},${ys(clamp(v, 0, YM))}`; }); sv("path", { d, fill: "none", stroke: color, "stroke-width": w, "stroke-dasharray": dash || "", "stroke-linejoin": "round", opacity: 0.95 }, svg); };
      snaps.forEach((s, k) => draw1(prepProgram("run:" + s.step + "@" + df(s.construction).name, df(s.construction).vals).g, seqRamp(0.12 + 0.88 * (k + 1) / snaps.length), 1.1));
      const rec = pubs.filter(r => r.kind === "construction").sort((a, b) => a.reported - b.reported)[0];
      if (rec) { const al = alignPub(rec); draw1(prepProgram("pub:" + rec.id + "@" + al.name, al.vals).g, S_COL(1), 1.4, "4 3"); }
    });

    /* ---------- sequence lengths ---------- */
    const hostN = h("div", { class: "chart" }), hostT = h("div", { class: "chart" });
    root.append(h("div", { class: "grid g2", style: { marginTop: "14px" } },
      card("Sequence length chosen by the coder", "every valid program: step vs n (the coder is free to resample the parent's sequence to another length)", hostN),
      card("Which lengths produce the best programs?", "valid-program count per sequence length", hostT)));
    const vp = R.filter(r => r.valid && r.n > 0 && r.raw > 0 && r.raw < 5 * Math.max(1, PROB.seedRaw || 1));
    const gq = vp.map(r => gapOf(r.raw)).sort((a, b) => a - b), glo = gq[Math.floor(gq.length * 0.02)] || 0, ghi = gq[Math.floor(gq.length * 0.9)] || 1;
    scatterChart(hostN, { height: 280, points: vp.map(r => ({ x: r.step + ((r.id * 2654435761 % 1000) / 1000 - 0.5) * 0.7, y: Math.log2(r.n), c: seqRamp(1 - clamp((gapOf(r.raw) - glo) / Math.max(1e-9, ghi - glo), 0, 1)), a: 0.65, r: 2.6, rr: r })), xLabel: "step", yLabel: "n (log₂ scale)", margin: { l: 56 }, xTicks: ALL_STEPS, xFmt: v => String(Math.round(v)), yTicks: range(4, 18), yFmt: v => String(Math.pow(2, v)),
      tip: p => `<b>step ${p.rr.step} · n=${p.rr.n}</b><div>${esc(PROB.metric)} ${fmt.raw(p.rr.raw)}</div>`, onPoint: p => openRollout(p.rr.id) });
    const byN = {}; vp.forEach(r => { (byN[r.n] = byN[r.n] || []).push(r); });
    const ns = Object.keys(byN).map(Number).sort((a, b) => b - a).filter(n => byN[n].length >= 3).sort((a, b) => byN[b].length - byN[a].length).slice(0, 14).sort((a, b) => a - b);
    barChart(hostT, { height: 280, cats: ns.map(String), series: [{ name: "valid programs", color: S_COL(0), vals: ns.map(n => byN[n].length) }], xLabel: "sequence length n (≥3 programs)", yLabel: "valid programs", margin: { l: 50 },
      tip: (i) => { const rs = byN[ns[i]]; return `<b>n = ${ns[i]}</b><div>${rs.length} valid programs</div><div>best ${fmt.raw(Math.min(...rs.map(r => r.raw)))}</div><div class="mut">median ${fmt.raw(median(rs.map(r => r.raw)))}</div>`; } });
    root.append(legend([{ color: seqRamp(0.1), label: "worse bound" }, { color: seqRamp(0.95), label: "closer to the reference" }]));

    /* ---------- code of the selected run snapshot ---------- */
    const A0 = resolve(_sol.a);
    const code = A0.kind === "run" ? A0.snap.code : "";
    root.append(h("div", { style: { marginTop: "14px" } }, card(A0.kind === "run" ? `Source of the best program through step ${A0.snap.step}` : "Source", "as written by the coder (docstring, algorithm, search loop)",
      code ? codeBlock(code) : h("div", { class: "empty" }, "Slot A holds a published construction: only the height sequence is published (no source shown). Select a run snapshot in slot A to see the program that produced it."))));
  }
});
