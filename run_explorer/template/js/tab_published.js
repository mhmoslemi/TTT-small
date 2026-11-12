"use strict";
/* ---- Published SOTA tab: reference table, timeline and constructions of other methods (any problem) ---------- */
const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
function monthIdx(d) { if (!d) return null; const [y, m] = d.split("-").map(Number); return (y - 2025) * 12 + (m - 1) + (d.length > 7 ? (Number(d.slice(8)) - 1) / 31 : 0); }
function monthLabel(v) { return `${MONTHS[((Math.round(v) % 12) + 12) % 12]} ${25 + Math.floor(Math.round(v) / 12)}`; }

registerTab({
  id: "published", label: "Published SOTA", ico: "★",
  async render(root) {
    const pub = await lazy("pub");
    root.append(pageHead(`Published ${PROB.short} results (state of the art)`, (pub && pubText("lede")) || "Published results for this problem."));
    if (!pub || !pub.results || !pub.results.length) { root.append(h("div", { class: "hint" }, `No published reference data is embedded for the problem "${esc(CFG.problem)}" (data/published_${esc(String(CFG.problem).toLowerCase())}.json is missing).`)); return; }

    const done = BSTEPS, best = ST[done[done.length - 1]].best_seen_raw_score;
    const rows = pub.results.filter(pubVisible).map(r => Object.assign({}, r, { calc: r.values ? constructionScore(r.values) : null }));
    if (hiddenNote()) root.append(h("div", { class: "hint", style: { marginBottom: "12px" }, html: `<b>Showing ${SHOW === "paper" ? "published works only" : SHOW === "board" ? "published works + Arena leaderboard" : "everything"}.</b> ${hiddenNote()}` }));
    const me = { label: "THIS RUN (best so far)", method: "this run", reported: best, calc: best, kind: "run", date: (D.main.meta.generated_at || "").slice(0, 10), n: heavyBestN(), me: true };
    const all = rows.concat([me]).sort((a, b) => a.reported - b.reported);
    const firm = all.filter(r => r.kind !== "claim");
    const rank = firm.findIndex(r => r.me) + 1, lead = firm[0];
    const cons = rows.filter(r => r.kind === "construction" && r.calc != null).sort((a, b) => a.calc - b.calc);
    const claims = rows.filter(r => r.kind === "claim");

    root.append(h("div", { class: "kpis" },
      cons.length ? kpi("Best verified bound", fmt.rawx(cons[0].calc), `recomputed from the public sequence (${esc(cons[0].label)})`) : null,
      kpi("Best reported", fmt.rep(lead.reported), esc(lead.label) + " · as published"),
      claims.length ? kpi("Unreviewed claims", String(claims.length), "lowest: " + fmt.rawx(Math.min(...claims.map(r => r.reported)))) : (pub.lower_bound ? kpi("Known lower bound", String(pub.lower_bound.value), esc(pub.lower_bound.who)) : null),
      kpi("This run", fmt.raw(best), `rank ${rank} of ${firm.length} · gap to leading <b>${fmt.gap(best - lead.reported)}</b>`),
      kpi("Reference in plots", fmt.rep(TARGET()), esc(TARGET_LABEL()) + " (top-bar selector)")));

    /* ------------------------------------------------ timeline (symmetric-log gap to the reference) */
    const hostT = h("div", { class: "chart" });
    const dated = rows.filter(r => r.date && monthIdx(r.date) >= 0), older = rows.filter(r => r.date && monthIdx(r.date) < 0);
    root.append(card("Record progression", `published ${PROB.short} ${PROB.metric} over time (lower is better) · y-axis = gap to the reference (symmetric log) · filled = construction available and recomputed · ring = reported only · diamond = unreviewed claim`,
      h("div", null, hostT, legend([{ color: S_COL(0), label: "this run" }, { color: cssVar("--ink-2"), label: "verified construction" }, rows.some(r => r.kind === "reported") ? { color: cssVar("--s4"), label: "reported" } : null, rows.some(r => r.kind === "claim") ? { color: cssVar("--s2"), label: "unreviewed claim" } : null, { color: cssVar("--ink-3"), label: "record so far", line: true }].filter(Boolean)),
        older.length ? h("div", { class: "sub", style: { marginTop: "4px" } }, "Older results not on this time axis: " + older.map(r => `${r.label} ${+r.reported.toFixed(PROB.dec + 2)} (${r.date.slice(0, 4)})`).join("; ") + ".") : null)));
    registerChart(hostT, () => {
      const F = frame(hostT, { height: 420, margin: { l: 70, b: 40, t: 18, r: 30 } }), { svg, m, iw, ih } = F;
      const pts = dated.map(r => Object.assign({ x: monthIdx(r.date), y: gapOf(r.kind === "construction" ? r.calc : r.reported) }, r));
      const nowX = monthIdx(me.date), myY = gapOf(best);
      const x0 = Math.min(...pts.map(p => p.x)) - 1.5, x1 = Math.max(nowX, ...pts.map(p => p.x)) + 1.8;
      const dom = gapDomain(pts.map(p => p.y).concat([myY, 0]));
      const xs = linear(x0, x1, m.l, m.l + iw), ys = linear(dom[0], dom[1], m.t + ih, m.t);
      const xt = []; for (let v = Math.ceil(x0 / 3) * 3; v <= x1; v += 3) xt.push(v);
      axes(F, xs, ys, { xLabel: "date", yLabel: "gap to reference (symlog)", yTicks: gapTicks().filter(t => t >= dom[0] && t <= dom[1]), yFmt: gapFmt, xTicks: xt, xFmt: monthLabel });
      hline(F, ys, 0, `${TARGET_LABEL()} ${+TARGET().toFixed(PROB.dec + 3)}`, cssVar("--good-ink"), "5 4", "start", -2);
      // record step line (verified + reported only)
      const rec = pts.filter(p => p.kind !== "claim").sort((a, b) => a.x - b.x || a.y - b.y); let cur = Infinity, d = "";
      rec.forEach(p => { if (p.y < cur) { d += (d ? `H${xs(p.x)}V${ys(p.y)}` : `M${xs(p.x)},${ys(p.y)}`); cur = p.y; } });
      if (d) d += `H${xs(nowX)}`;
      sv("path", { d, fill: "none", stroke: cssVar("--ink-3"), "stroke-width": 1.6, "stroke-dasharray": "5 3" }, svg);
      const order = pts.slice().sort((a, b) => a.x - b.x || a.y - b.y);
      const mx = xs(nowX), my = ys(myY);
      const boxes = [{ x0: mx - 150, x1: mx + 10, y0: my - 12, y1: my + 8 }];       // 'this run' label
      const overlaps = b => boxes.some(q => b.x0 < q.x1 && b.x1 > q.x0 && b.y0 < q.y1 && b.y1 > q.y0);
      order.forEach(p => {
        const x = xs(p.x), y = ys(p.y);
        const col = p.kind === "claim" ? cssVar("--s2") : p.kind === "reported" ? cssVar("--s4") : cssVar("--ink-2");
        let el;
        if (p.kind === "claim") el = sv("path", { d: `M${x},${y - 7}L${x + 7},${y}L${x},${y + 7}L${x - 7},${y}Z`, fill: cssVar("--surface"), stroke: col, "stroke-width": 2 }, svg);
        else el = sv("circle", { cx: x, cy: y, r: 6, fill: p.kind === "reported" ? cssVar("--surface") : col, stroke: p.kind === "reported" ? col : cssVar("--surface"), "stroke-width": 2.2 }, svg);
        const txt = p.label.replace(/ \(.*\)/, "") + " " + +(p.kind === "construction" ? p.calc : p.reported).toFixed(PROB.dec + 2), w = txt.length * 6.1;
        // greedy label placement: first free slot among above / below / further above / further below, centred or flipped to the left
        const cands = [[0, -13], [0, 20], [0, -30], [0, 37], [0, -47], [0, 54]];
        let pick = null;
        for (const [, dy] of cands) {
          for (const anchor of x > m.l + iw - w ? ["end"] : ["middle", "start", "end"]) {
            const x0 = anchor === "middle" ? x - w / 2 : anchor === "start" ? x + 9 : x - 9 - w, bb = { x0, x1: x0 + w, y0: y + dy - 10, y1: y + dy + 3 };
            if (!overlaps(bb)) { pick = { dy, anchor, bb }; break; }
          }
          if (pick) break;
        }
        if (!pick) pick = { dy: -13, anchor: "middle", bb: { x0: x - w / 2, x1: x + w / 2, y0: y - 23, y1: y - 10 } };
        boxes.push(pick.bb);
        const tx = pick.anchor === "middle" ? x : pick.anchor === "start" ? x + 9 : x - 9;
        const t = sv("text", { x: tx, y: y + pick.dy, "text-anchor": pick.anchor, style: `font-size:11px;fill:${cssVar("--ink")}` }, svg);
        t.textContent = txt;
        bindTip(el, () => `<b>${esc(p.label)}</b><div>${+(p.kind === "construction" ? p.calc : p.reported).toFixed(PROB.dec + 4)}</div><div class="mut">${esc(p.date)}${p.date_approx ? " (approximate)" : ""} · ${p.kind === "claim" ? "unreviewed claim" : p.kind === "reported" ? "reported" : "construction recomputed here"}</div><div class="mut">${esc(p.note || "")}</div>`);
      });
      sv("circle", { cx: mx, cy: my, r: 8, fill: S_COL(0), stroke: cssVar("--surface"), "stroke-width": 2.5 }, svg);
      const mt = sv("text", { x: mx - 12, y: my + 4, "text-anchor": "end", style: `font-size:12px;font-weight:700;fill:${S_COL(0)}` }, svg); mt.textContent = `this run ${fmt.raw(best)}`;
    });

    /* ------------------------------------------------ results table */
    const stat = r => r.me ? pill("this run", "acc") : r.kind === "construction" ? ((r.exact_ref != null ? Math.abs(r.calc - r.exact_ref) < 5e-9 : Math.abs(r.calc - r.reported) < 5e-5) ? pill("construction verified", "good") : pill("construction differs", "warn")) : r.kind === "reported" ? pill("reported", "acc") : pill("unreviewed claim", "warn");
    const val = r => r.calc != null ? r.calc : r.reported;
    const tb = table([
      { label: "method", get: r => h("span", { style: { fontWeight: r.me ? 700 : 400, fontStyle: r.kind === "claim" ? "italic" : "normal" } }, r.label) },
      { label: "date", get: r => r.date ? (r.date_approx ? "≈" : "") + r.date : "–" },
      { label: "steps n", get: r => r.values ? fmt.n(r.values.length) : (r.n ? fmt.n(r.n) : "–"), num: true },
      { label: "reported", get: r => fmt.rep(r.reported), num: true },
      { label: "recomputed here", get: r => r.calc != null && !r.me ? fmt.rawx(r.calc) : "–", num: true, title: "the problem's score evaluated from the public height sequence, in your browser" },
      { label: "Δ recomputed − reported", get: r => r.calc != null && !r.me && r.values ? (r.calc - r.reported >= 0 ? "+" : "−") + Math.abs(r.calc - r.reported).toExponential(1) : "", num: true, title: "positive = the printed bound is slightly BELOW what the construction really gives" },
      { label: "this run − it", get: r => r.me ? "" : fmt.gap(best - val(r)), num: true },
      { label: "status", get: stat },
      { label: "source", get: r => r.url ? h("a", { href: r.url, target: "_blank", rel: "noopener", onclick: e => e.stopPropagation() }, "link ↗") : "" },
    ], all, { onRow: r => { if (r.values) { _sol.b = "pub:" + r.id; setState({ tab: "solution" }); } } });
    root.append(h("div", { style: { marginTop: "14px" } }, card(`${PROB.short} results, lowest first`, "click a row with a construction to compare it with this run in the Solution tab", h("div", null, h("div", { class: "scroll", style: { maxHeight: "none" } }, tb),
      h("div", { class: "sub", style: { marginTop: "8px" } }, "'reported' is the figure as printed by the source; 'recomputed' is evaluated here from the public height sequence (stored to 10 significant digits). Where the community-curated list replayed the same construction with exact arithmetic, the recomputed values agree with it to ~1e-9 (see ", h("a", { href: "https://github.com/teorth/optimizationproblems/issues/348", target: "_blank", rel: "noopener" }, "teorth/optimizationproblems#348"), ").")))));

    /* ------------------------------------------------ source tables (paper table, curated list ...) */
    (pub.tables || []).filter(t => tierVisible(t.tier || "paper")).forEach(t => {
      const bestOf = (i, dir) => { const v = t.rows.map(r => r[i]).filter(x => typeof x === "number"); return v.length ? (dir === "max" ? Math.max(...v) : Math.min(...v)) : null; };
      const bests = t.columns.map((c, i) => c.best ? bestOf(i, c.best) : null);
      const cell = (v, ci) => {
        const c = t.columns[ci];
        if (v == null) return h("span", { style: { color: "var(--ink-3)" } }, "–");
        if (typeof v === "number") return h("span", { style: { fontWeight: v === bests[ci] ? 700 : 400, color: v === bests[ci] ? "var(--good-ink)" : "" } }, v.toFixed(c.digits || 6));
        return String(v);
      };
      const el = table(t.columns.map((c, ci) => ({ label: c.label, num: c.num, get: r => cell(r[ci], ci) })), t.rows);
      root.append(h("div", { style: { marginTop: "14px" } }, card(t.title, t.subtitle, h("div", null, h("div", { class: "scroll", style: { maxHeight: "none" } }, el), t.note ? h("div", { class: "sub", style: { marginTop: "6px" } }, t.note) : null))));
    });

    /* ------------------------------------------------ history + sources */
    const links = (pub.links || []).filter(l => tierVisible(l[2] || "paper")).map(([t, u]) => h("div", null, h("a", { href: u, target: "_blank", rel: "noopener" }, t + " ↗")));
    root.append(h("div", { class: "grid g2", style: { marginTop: "14px" } },
      card("What happened, in order", null, h("div", { class: "kv", style: { gridTemplateColumns: "92px 1fr", gap: "8px 12px" }, html: (pub.events || []).filter(e => tierVisible(e[2] || "paper")).map(([d, t]) => `<div class="k"><b style="color:var(--ink)">${esc(d)}</b></div><div class="v">${esc(t)}</div>`).join("") })),
      card("Sources and how to read them", null, h("div", null, h("div", null, links), h("div", { class: "warnbox", style: { marginTop: "10px" } }, (pubText("caution") ? pubText("caution") + " " : "") + "Reference data fetched " + esc(pub.fetched || "") + "; to refresh it run data/build_published.py."),
        h("p", { class: "sub", style: { marginTop: "8px" } }, pub.footnote || "")))));
  }
});

function heavyBestN() { try { const b = D.heavy && D.heavy.best; return b && b.length ? b[b.length - 1].construction.length : null; } catch (e) { return null; } }
