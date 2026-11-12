"use strict";
/* =====================================================================================
   charts.js - tiny dependency-free chart kit (SVG + canvas), hover layer included.
   Every chart re-draws itself on window resize via the registry below.
   ===================================================================================== */
const SVGNS = "http://www.w3.org/2000/svg";
const _charts = [];
function registerChart(el, draw) { const k = _charts.findIndex(c => c.el === el); if (k >= 0) _charts[k] = { el, draw }; else _charts.push({ el, draw }); draw(); }
function clearCharts() { _charts.length = 0; }
let _rz = null;
window.addEventListener("resize", () => { clearTimeout(_rz); _rz = setTimeout(() => _charts.forEach(c => { if (c.el.isConnected) c.draw(); }), 120); });

function sv(tag, attrs, parent) {
  const e = document.createElementNS(SVGNS, tag);
  if (attrs) for (const k in attrs) e.setAttribute(k, attrs[k]);
  if (parent) parent.appendChild(e);
  return e;
}
function linear(d0, d1, r0, r1) {
  if (d0 === d1) { d0 -= 1; d1 += 1; }
  const f = x => r0 + (x - d0) / (d1 - d0) * (r1 - r0);
  f.invert = y => d0 + (y - r0) / (r1 - r0) * (d1 - d0);
  f.domain = [d0, d1]; f.range = [r0, r1];
  return f;
}
function niceStep(span, n) {
  const raw = span / Math.max(1, n), p = Math.pow(10, Math.floor(Math.log10(raw))), f = raw / p;
  return (f < 1.5 ? 1 : f < 3 ? 2 : f < 7 ? 5 : 10) * p;
}
function niceTicks(lo, hi, n) {
  if (!isFinite(lo) || !isFinite(hi)) return [];
  if (lo === hi) return [lo];
  const st = niceStep(hi - lo, n || 5), out = [];
  for (let v = Math.ceil(lo / st - 1e-9) * st; v <= hi + st * 1e-9; v += st) out.push(+v.toFixed(12));
  out.step = st;
  return out;
}
function tickFmt(ticks) {
  const st = ticks.step || (ticks.length > 1 ? ticks[1] - ticks[0] : 1);
  const d = clamp(Math.ceil(-Math.log10(st) - 1e-9), 0, 9);
  return v => Math.abs(v) >= 1e5 ? fmt.n(v) : v.toFixed(d);
}
function hexToRgb(h6) { h6 = h6.replace("#", ""); if (h6.length === 3) h6 = h6.split("").map(c => c + c).join(""); const n = parseInt(h6, 16); return [(n >> 16) & 255, (n >> 8) & 255, n & 255]; }
function mixColor(a, b, t) { const A = hexToRgb(a), B = hexToRgb(b); return `rgb(${A.map((x, i) => Math.round(x + (B[i] - x) * t)).join(",")})`; }
function seqRamp(t) {
  const stops = ["--seq-100", "--seq-250", "--seq-400", "--seq-550", "--seq-700"].map(cssVar);
  t = clamp(t, 0, 1) * (stops.length - 1);
  const i = Math.min(stops.length - 2, Math.floor(t));
  return mixColor(stops[i], stops[i + 1], t - i);
}
function divRamp(t) { // -1..1  blue(neg) - mid - red(pos)
  const mid = cssVar("--mid"), neg = cssVar("--s1"), pos = cssVar("--s8");
  t = clamp(t, -1, 1);
  return t < 0 ? mixColor(mid, neg, -t) : mixColor(mid, pos, t);
}

/* ---------------------------------------------------------------- frame / axes ----- */
function frame(el, o) {
  const W = Math.max(280, el.clientWidth || 600), H = o.height || 260;
  const m = Object.assign({ t: 12, r: 14, b: 38, l: 54 }, o.margin || {});
  el.innerHTML = "";
  const svg = sv("svg", { viewBox: `0 0 ${W} ${H}`, width: W, height: H, role: "img", "aria-label": o.aria || "chart" });
  el.appendChild(svg);
  return { svg, W, H, m, iw: W - m.l - m.r, ih: H - m.t - m.b };
}
function axes(F, xs, ys, o) {
  const { svg, m, W, H, iw, ih } = F;
  const gy = sv("g", { class: "grid" }, svg);
  const yt = o.yTicks || niceTicks(ys.domain[0], ys.domain[1], Math.max(3, Math.round(ih / 46)));
  const yf = o.yFmt || tickFmt(yt);
  let lastLabelY = Infinity;
  yt.slice().sort((a, b) => ys(a) - ys(b)).forEach(v => {
    const y = ys(v); if (y < m.t - 1 || y > m.t + ih + 1) return;
    sv("line", { x1: m.l, x2: m.l + iw, y1: y, y2: y }, gy);
    if (Math.abs(y - lastLabelY) < 16) return;          // skip labels that would overlap the previous one
    lastLabelY = y;
    const t = sv("text", { x: m.l - 7, y: y + 3.5, "text-anchor": "end" }, svg); t.textContent = yf(v);
  });
  const xt = o.xTicks || niceTicks(xs.domain[0], xs.domain[1], Math.max(3, Math.round(iw / 80)));
  const xf = o.xFmt || tickFmt(xt);
  const ax = sv("g", { class: "axis" }, svg);
  sv("line", { x1: m.l, x2: m.l + iw, y1: m.t + ih, y2: m.t + ih }, ax);
  xt.forEach(v => {
    const x = xs(v); if (x < m.l - 1 || x > m.l + iw + 1) return;
    sv("line", { x1: x, x2: x, y1: m.t + ih, y2: m.t + ih + 4 }, ax);
    const t = sv("text", { x, y: m.t + ih + 16, "text-anchor": "middle" }, svg); t.textContent = xf(v);
  });
  if (o.xLabel) { const t = sv("text", { x: m.l + iw / 2, y: H - 4, "text-anchor": "middle", class: "muted" }, svg); t.textContent = o.xLabel; }
  if (o.yLabel) { const t = sv("text", { x: 12, y: m.t + ih / 2, "text-anchor": "middle", transform: `rotate(-90 12 ${m.t + ih / 2})`, class: "muted" }, svg); t.textContent = o.yLabel; }
}
function hline(F, ys, y, label, color, dash, anchor, dy) {
  const { svg, m, iw, ih } = F;
  const py = ys(y); if (py < m.t || py > m.t + ih) return;
  sv("line", { x1: m.l, x2: m.l + iw, y1: py, y2: py, stroke: color || cssVar("--ink-3"), "stroke-width": 1.2, "stroke-dasharray": dash || "5 4" }, svg);
  if (label) { const t = sv("text", { x: anchor === "start" ? m.l + 4 : m.l + iw - 3, y: py - 4 + (dy || 0), "text-anchor": anchor === "start" ? "start" : "end", style: "fill:" + (color || cssVar("--ink-2")) }, svg); t.textContent = label; }
}

/* ---------------------------------------------------------------- line chart -------- */
function lineChart(el, o) {
  registerChart(el, () => {
    const F = frame(el, o), { svg, m, iw, ih } = F;
    const all = o.series.flatMap(s => s.pts.filter(p => p[1] != null && isFinite(p[1])));
    if (!all.length) { el.innerHTML = '<div class="empty">no data</div>'; return; }
    let x0 = o.xDomain ? o.xDomain[0] : Math.min(...all.map(p => p[0])), x1 = o.xDomain ? o.xDomain[1] : Math.max(...all.map(p => p[0]));
    let y0 = o.yDomain ? o.yDomain[0] : Math.min(...all.map(p => p[1])), y1 = o.yDomain ? o.yDomain[1] : Math.max(...all.map(p => p[1]));
    (o.hlines || []).forEach(l => { if (o.includeH && l.y != null) { y0 = Math.min(y0, l.y); y1 = Math.max(y1, l.y); } });
    if (!o.yDomain) { const pad = (y1 - y0) * 0.08 || 0.5; y0 -= pad; y1 += pad; }
    if (!o.xDomain && x0 === x1) { x0 -= 1; x1 += 1; }
    const xs = linear(x0, x1, m.l + (o.xPad || 8), m.l + iw - (o.xPad || 8)), ys = linear(y0, y1, m.t + ih, m.t);
    axes(F, xs, ys, o);
    (o.bands || []).forEach(b => sv("rect", { x: xs(b.x0), width: Math.max(1, xs(b.x1) - xs(b.x0)), y: m.t, height: ih, fill: b.color || cssVar("--surface-2"), opacity: 0.6 }, svg));
    (o.hlines || []).forEach(l => hline(F, ys, l.y, l.label, l.color, l.dash, l.anchor, l.dy));
    (o.vlines || []).forEach(l => {
      const px = xs(l.x); if (px < m.l - 1 || px > m.l + iw + 1) return;
      sv("line", { x1: px, x2: px, y1: m.t, y2: m.t + ih, stroke: l.color || cssVar("--ink-3"), "stroke-width": l.width || 1.4, "stroke-dasharray": l.dash || "" }, svg);
      if (l.label) { const t = sv("text", { x: px + (l.anchor === "end" ? -5 : 5), y: m.t + 12 + (l.dy || 0), "text-anchor": l.anchor === "end" ? "end" : "start", style: "font-size:11px;font-weight:600;fill:" + (l.color || cssVar("--ink-2")) }, svg); t.textContent = l.label; }
    });
    let G = svg;
    if (o.clip) {
      const cid = "cp" + Math.random().toString(36).slice(2, 8), cp = sv("clipPath", { id: cid }, svg);
      sv("rect", { x: m.l, y: m.t, width: iw, height: ih }, cp);
      G = sv("g", { "clip-path": `url(#${cid})` }, svg);
    }
    o.series.forEach(s => {
      const pts = s.pts.filter(p => p[1] != null && isFinite(p[1]));
      if (!pts.length) return;
      let d = "";
      pts.forEach((p, i) => {
        const X = xs(p[0]), Y = ys(p[1]);
        d += i === 0 ? `M${X},${Y}` : (s.step ? `H${X}V${Y}` : `L${X},${Y}`);
      });
      if (s.fill) sv("path", { d: d + `L${xs(pts[pts.length - 1][0])},${m.t + ih}L${xs(pts[0][0])},${m.t + ih}Z`, fill: s.color, opacity: 0.12 }, G);
      const animL = ANIM && !s.dash && s.width !== 0;
      const pe = sv("path", { d, fill: "none", stroke: s.color, "stroke-width": s.width != null ? s.width : 2, "stroke-dasharray": s.dash || "", "stroke-linejoin": "round", "stroke-linecap": "round" }, G);
      if (animL) { pe.setAttribute("pathLength", "1"); pe.setAttribute("class", "a-line"); }
      if (s.dots !== false && pts.length <= 80) pts.forEach((p, pi) => { const c = sv("circle", { cx: xs(p[0]), cy: ys(p[1]), r: s.dotR || 3.2, fill: s.color, stroke: cssVar("--surface"), "stroke-width": 1.5 }, G); if (ANIM) { c.setAttribute("class", "a-fade"); c.style.animationDelay = (0.25 + 0.6 * pi / pts.length) + "s"; } });
      if (s.label) { const lp = pts[pts.length - 1]; const t = sv("text", { x: xs(lp[0]) - 4, y: ys(lp[1]) - 8, "text-anchor": "end", style: "fill:" + cssVar("--ink") }, svg); t.textContent = s.name; }
    });
    // hover layer: crosshair + nearest x
    const xsAll = uniq(all.map(p => p[0])).sort((a, b) => a - b);
    const cross = sv("line", { y1: m.t, y2: m.t + ih, stroke: cssVar("--ink-3"), "stroke-width": 1, opacity: 0 }, svg);
    const dots = o.series.map(s => sv("circle", { r: 5, fill: s.color, stroke: cssVar("--surface"), "stroke-width": 2, opacity: 0 }, svg));
    const hit = sv("rect", { x: m.l, y: m.t, width: iw, height: ih, fill: "transparent", style: o.onClick ? "cursor:pointer" : "" }, svg);
    const nearest = ev => {
      const r = svg.getBoundingClientRect(), px = (ev.clientX - r.left) * (F.W / r.width), xv = xs.invert(px);
      let best = xsAll[0]; xsAll.forEach(v => { if (Math.abs(v - xv) < Math.abs(best - xv)) best = v; });
      return best;
    };
    hit.addEventListener("mousemove", ev => {
      const xv = nearest(ev), X = xs(xv);
      cross.setAttribute("x1", X); cross.setAttribute("x2", X); cross.setAttribute("opacity", 1);
      let rows = "";
      o.series.forEach((s, i) => {
        const p = s.pts.find(q => q[0] === xv);
        if (p && p[1] != null && isFinite(p[1])) { dots[i].setAttribute("cx", X); dots[i].setAttribute("cy", ys(p[1])); dots[i].setAttribute("opacity", 1); rows += `<div><span class="dot" style="background:${s.color}"></span>${esc(s.name)}: <b>${(s.fmt || o.tipVal || o.yFmt || (v => v.toFixed(4)))(p[1])}</b></div>`; }
        else dots[i].setAttribute("opacity", 0);
      });
      const head = o.tipHead ? o.tipHead(xv) : `${o.xLabel || "x"} ${(o.xFmtTip || (v => v))(xv)}`;
      tipShow(`<div class="mut">${head}</div>${rows}${o.tipExtra ? o.tipExtra(xv) : ""}`, ev);
    });
    hit.addEventListener("mouseleave", () => { cross.setAttribute("opacity", 0); dots.forEach(d => d.setAttribute("opacity", 0)); tipHide(); });
    if (o.onClick) hit.addEventListener("click", ev => o.onClick(nearest(ev)));
  });
}

/* ---------------------------------------------------------------- scatter (canvas) -- */
function scatterChart(el, o) {
  registerChart(el, () => {
    const F = frame(el, o), { svg, m, iw, ih, W, H } = F;
    const pts = o.points.filter(p => isFinite(p.x) && isFinite(p.y));
    if (!pts.length) { el.innerHTML = '<div class="empty">no points match the filters</div>'; return; }
    let x0 = o.xDomain ? o.xDomain[0] : Math.min(...pts.map(p => p.x)), x1 = o.xDomain ? o.xDomain[1] : Math.max(...pts.map(p => p.x));
    let y0 = o.yDomain ? o.yDomain[0] : Math.min(...pts.map(p => p.y)), y1 = o.yDomain ? o.yDomain[1] : Math.max(...pts.map(p => p.y));
    if (!o.xDomain) { const pd = (x1 - x0) * 0.04 || 1; x0 -= pd; x1 += pd; }
    if (!o.yDomain) { const pd = (y1 - y0) * 0.06 || 1; y0 -= pd; y1 += pd; }
    const xs = linear(x0, x1, m.l, m.l + iw), ys = linear(y0, y1, m.t + ih, m.t);
    // canvas under the svg
    const cv = document.createElement("canvas"), dpr = window.devicePixelRatio || 1;
    cv.width = W * dpr; cv.height = H * dpr; cv.style.cssText = `position:absolute;left:0;top:0;width:${W}px;height:${H}px`;
    el.style.position = "relative"; el.insertBefore(cv, svg);
    const ctx = cv.getContext("2d"); ctx.scale(dpr, dpr);
    ctx.save(); ctx.beginPath(); ctx.rect(m.l, m.t, iw, ih); ctx.clip();
    const vis = [];
    pts.forEach(p => {
      const X = xs(p.x), Y = ys(p.y);
      if (X < m.l - 4 || X > m.l + iw + 4 || Y < m.t - 4 || Y > m.t + ih + 4) return;
      vis.push({ p, X, Y });
    });
    (o.drawOrder ? vis.sort(o.drawOrder) : vis).forEach(({ p, X, Y }) => {
      ctx.globalAlpha = p.a == null ? (o.alpha || 0.75) : p.a;
      ctx.fillStyle = p.c || cssVar("--accent");
      ctx.beginPath();
      if (p.shape === "diamond") { const r = (p.r || 3.5) + 1.5; ctx.moveTo(X, Y - r); ctx.lineTo(X + r, Y); ctx.lineTo(X, Y + r); ctx.lineTo(X - r, Y); ctx.closePath(); }
      else ctx.arc(X, Y, p.r || 3, 0, 6.283);
      ctx.fill();
      if (p.ring) { ctx.globalAlpha = 1; ctx.strokeStyle = cssVar("--ink"); ctx.lineWidth = 1.4; ctx.stroke(); }
    });
    ctx.restore();
    axes(F, xs, ys, o);
    (o.hlines || []).forEach(l => hline(F, ys, l.y, l.label, l.color, l.dash));
    const ring = sv("circle", { r: 7, fill: "none", stroke: cssVar("--ink"), "stroke-width": 1.8, opacity: 0 }, svg);
    const hit = sv("rect", { x: m.l, y: m.t, width: iw, height: ih, fill: "transparent", style: "cursor:crosshair" }, svg);
    const find = ev => {
      const r = svg.getBoundingClientRect(), k = F.W / r.width, px = (ev.clientX - r.left) * k, py = (ev.clientY - r.top) * k;
      let best = null, bd = 100;
      for (const v of vis) { const d = (v.X - px) ** 2 + (v.Y - py) ** 2; if (d < bd) { bd = d; best = v; } }
      return best;
    };
    hit.addEventListener("mousemove", ev => {
      const b = find(ev);
      if (b) { ring.setAttribute("cx", b.X); ring.setAttribute("cy", b.Y); ring.setAttribute("opacity", 1); tipShow(o.tip ? o.tip(b.p) : `${b.p.x}, ${b.p.y}`, ev); } else { ring.setAttribute("opacity", 0); tipHide(); }
    });
    hit.addEventListener("mouseleave", () => { ring.setAttribute("opacity", 0); tipHide(); });
    hit.addEventListener("click", ev => { const b = find(ev); if (b && o.onPoint) o.onPoint(b.p); });
    el.setAttribute("data-npts", vis.length);
  });
}

/* ---------------------------------------------------------------- bar chart --------- */
function barChart(el, o) {
  registerChart(el, () => {
    const F = frame(el, o), { svg, m, iw, ih } = F;
    const n = o.cats.length, ns = o.series.length;
    let tot = o.cats.map((_, i) => o.stacked ? sum(o.series.map(s => s.vals[i] || 0)) : Math.max(...o.series.map(s => s.vals[i] || 0)));
    const y0 = o.yDomain ? o.yDomain[0] : Math.min(0, ...o.series.flatMap(s => s.vals.map(v => v || 0)));
    const y1 = o.yDomain ? o.yDomain[1] : Math.max(...tot) * 1.08 || 1;
    const ys = linear(y0, y1, m.t + ih, m.t);
    const band = iw / n, gap = Math.min(o.bands ? 12 : 10, band * (o.bands ? 0.17 : 0.22)), bw = Math.max(2, (band - gap) / (o.stacked ? 1 : ns));
    const xs = linear(0, n, m.l, m.l + iw);
    axes(F, linear(0, n, m.l, m.l + iw), ys, Object.assign({}, o, { xTicks: [], xFmt: null }));
    // zebra bands + separators make every category (step) a clearly delimited column
    const bandEls = [];
    if (o.bands) {
      const first = svg.firstChild;
      for (let i = 0; i < n; i++) {
        const b = sv("rect", { x: m.l + band * i, y: m.t, width: band, height: ih, fill: i % 2 ? "transparent" : cssVar("--surface-2"), opacity: i % 2 ? 1 : 0.7 }, null);
        svg.insertBefore(b, first); bandEls.push(b);
      }
      for (let i = 0; i <= n; i++) sv("line", { x1: m.l + band * i, x2: m.l + band * i, y1: m.t, y2: m.t + ih + 5, stroke: cssVar("--line-2"), "stroke-width": 1, opacity: 0.8 }, svg);
    }
    // category labels
    const every = Math.ceil(n / Math.max(1, Math.floor(iw / 34)));
    o.cats.forEach((c, i) => { if (i % every) return; const t = sv("text", { x: m.l + band * (i + 0.5), y: m.t + ih + 17, "text-anchor": "middle", style: o.bands ? "font-weight:600;fill:" + cssVar("--ink") : "" }, svg); t.textContent = c; });
    (o.hlines || []).forEach(l => hline(F, ys, l.y, l.label, l.color, l.dash));
    if (o.colTip || o.bands) {
      for (let i = 0; i < n; i++) {
        const hit = sv("rect", { x: m.l + band * i, y: m.t, width: band, height: ih, fill: "transparent", style: o.onClick ? "cursor:pointer" : "" }, svg);
        hit.addEventListener("mousemove", ev => { if (bandEls[i]) { bandEls[i].setAttribute("fill", cssVar("--accent-soft")); bandEls[i].setAttribute("opacity", 1); } if (o.colTip) tipShow(o.colTip(i), ev); });
        hit.addEventListener("mouseleave", () => { if (bandEls[i]) { bandEls[i].setAttribute("fill", i % 2 ? "transparent" : cssVar("--surface-2")); bandEls[i].setAttribute("opacity", i % 2 ? 1 : 0.7); } tipHide(); });
        if (o.onClick) hit.addEventListener("click", () => o.onClick(i, 0));
      }
    }
    const rects = [];
    o.cats.forEach((c, i) => {
      let acc = 0;
      o.series.forEach((s, si) => {
        const v = s.vals[i] || 0; if (!v && !o.showZero) return;
        const x = m.l + band * i + gap / 2 + (o.stacked ? 0 : si * bw);
        let yTop, hh;
        if (o.stacked) { yTop = ys(acc + v); hh = Math.max(0, ys(acc) - yTop - (acc ? 2 : 0)); acc += v; }
        else { yTop = ys(Math.max(v, 0)); hh = Math.abs(ys(0) - ys(v)); }
        const col = (s.colors && s.colors[i]) || s.color;
        const r = sv("rect", { x, y: yTop, width: bw - (o.stacked ? 0 : 1.5), height: Math.max(hh, 0.5), rx: 3, fill: col, style: o.onClick ? "cursor:pointer" : "" }, svg);
        if (ANIM) { r.setAttribute("class", "a-bar"); r.style.animationDelay = (0.03 * i) + "s"; }
        r.addEventListener("mousemove", ev => tipShow(o.tip ? o.tip(i, si, v) : `<b>${esc(c)}</b><br>${esc(s.name)}: ${v}`, ev));
        r.addEventListener("mouseleave", tipHide);
        if (o.onClick) r.addEventListener("click", () => o.onClick(i, si));
        rects.push(r);
        // value label inside the segment (percent, hours, ...) when it is tall and wide enough
        if (o.segLabels && hh >= 14 && bw >= 21) {
          const lab = o.segLabels(i, si, v, tot[i]);
          if (lab != null && lab !== "") {
            const tx = sv("text", { x: x + (bw - (o.stacked ? 0 : 1.5)) / 2, y: yTop + hh / 2 + 4, "text-anchor": "middle", style: `font-size:10.5px;font-weight:700;pointer-events:none;fill:${textOn(col)}` }, svg);
            tx.textContent = lab;
            if (ANIM) { tx.setAttribute("class", "a-fade"); tx.style.animationDelay = (0.35 + 0.03 * i) + "s"; }
          }
        }
      });
      if (o.totalLabel) {
        const tl = sv("text", { x: m.l + band * (i + 0.5), y: ys(tot[i]) - 7, "text-anchor": "middle", style: "font-size:11px;font-weight:700;pointer-events:none;fill:" + cssVar("--ink") }, svg);
        tl.textContent = o.totalLabel(i, tot[i]);
      }
    });
  });
}

/* readable label colour on top of a fill (hex or rgb()) */
function textOn(col) {
  let r = 0, g = 0, b = 0;
  const m = /^#?([0-9a-f]{6})$/i.exec(col || "");
  if (m) { const n = parseInt(m[1], 16); r = (n >> 16) & 255; g = (n >> 8) & 255; b = n & 255; }
  else { const q = /rgb\((\d+),\s*(\d+),\s*(\d+)/.exec(col || ""); if (q) { r = +q[1]; g = +q[2]; b = +q[3]; } }
  const L = (0.299 * r + 0.587 * g + 0.114 * b) / 255;
  return L > 0.62 ? "#0b0b0b" : "#ffffff";
}

/* ---------------------------------------------------------------- heat map ---------- */
function heatmap(el, o) {
  registerChart(el, () => {
    const nr = o.rows.length, nc = o.cols.length;
    const lw = o.rowLabelW || 56, topH = o.colLabelH || 26;
    const W = Math.max(280, el.clientWidth || 600);
    const cw = Math.max(o.minCell || 6, Math.min(o.maxCell || 44, (W - lw - 8) / nc)), ch = o.cellH || Math.max(14, Math.min(34, 420 / nr));
    const H = topH + nr * ch + 8;
    el.innerHTML = "";
    const svg = sv("svg", { viewBox: `0 0 ${lw + nc * cw + 8} ${H}`, width: lw + nc * cw + 8, height: H, style: `max-width:${lw + nc * cw + 8}px` }, el);
    const every = Math.ceil(nc / Math.max(1, Math.floor(nc * cw / 26)));
    o.cols.forEach((c, j) => { if (j % every) return; const t = sv("text", { x: lw + cw * (j + 0.5), y: 15, "text-anchor": "middle" }, svg); t.textContent = c; });
    o.rows.forEach((r, i) => { const t = sv("text", { x: lw - 8, y: topH + ch * (i + 0.5) + 4, "text-anchor": "end" }, svg); t.textContent = r; });
    for (let i = 0; i < nr; i++) for (let j = 0; j < nc; j++) {
      const v = o.val(i, j);
      const col = v == null ? cssVar("--surface-2") : o.color(v, i, j);
      const rc = sv("rect", { x: lw + cw * j + 1, y: topH + ch * i + 1, width: cw - 2, height: ch - 2, rx: 3, fill: col, style: o.onClick ? "cursor:pointer" : "" }, svg);
      if (o.text) { const tx = o.text(v, i, j); if (tx != null && cw > 22) { const t = sv("text", { x: lw + cw * (j + 0.5), y: topH + ch * (i + 0.5) + 3.5, "text-anchor": "middle", style: "fill:" + (o.textColor ? o.textColor(v, i, j) : "#fff") + ";font-size:10px;pointer-events:none" }, svg); t.textContent = tx; } }
      rc.addEventListener("mousemove", ev => { const html = o.tip ? o.tip(i, j, v) : null; if (html) tipShow(html, ev); });
      rc.addEventListener("mouseleave", tipHide);
      if (o.onClick) rc.addEventListener("click", () => o.onClick(i, j, v));
    }
  });
}

/* ---------------------------------------------------------------- AC1 solution plot - */
function fftInPlace(re, im, inv) {
  const N = re.length;
  for (let i = 1, j = 0; i < N; i++) { let bit = N >> 1; for (; j & bit; bit >>= 1) j ^= bit; j ^= bit; if (i < j) { let t = re[i]; re[i] = re[j]; re[j] = t; t = im[i]; im[i] = im[j]; im[j] = t; } }
  for (let len = 2; len <= N; len <<= 1) {
    const ang = 2 * Math.PI / len * (inv ? 1 : -1), wr = Math.cos(ang), wi = Math.sin(ang), half = len >> 1;
    for (let i = 0; i < N; i += len) {
      let cr = 1, ci = 0;
      for (let j = 0; j < half; j++) {
        const a = i + j, b = a + half, vr = re[b] * cr - im[b] * ci, vi = re[b] * ci + im[b] * cr;
        re[b] = re[a] - vr; im[b] = im[a] - vi; re[a] += vr; im[a] += vi;
        const t = cr * wr - ci * wi; ci = cr * wi + ci * wr; cr = t;
      }
    }
  }
}
/* linear autoconvolution: direct for short sequences, FFT for long ones (the published 30,000-step constructions) */
function autoConv(a) {
  const n = a.length, out = new Float64Array(2 * n - 1);
  if (n <= 1500) {
    for (let i = 0; i < n; i++) { const ai = a[i]; if (!ai) continue; for (let j = 0; j < n; j++) out[i + j] += ai * a[j]; }
    return out;
  }
  let N = 1; while (N < 2 * n - 1) N <<= 1;
  const re = new Float64Array(N), im = new Float64Array(N);
  for (let i = 0; i < n; i++) re[i] = a[i];
  fftInPlace(re, im, false);
  for (let i = 0; i < N; i++) { const r = re[i], m = im[i]; re[i] = r * r - m * m; im[i] = 2 * r * m; }
  fftInPlace(re, im, true);
  for (let i = 0; i < out.length; i++) out[i] = Math.max(0, re[i] / N);
  return out;
}
/* peak-preserving min/max envelope path for long arrays (xOf(i) -> px, yOf(v) -> py) */
function envelopePath(arr, xOf, yOf, buckets) {
  const L = arr.length, B = Math.max(2, Math.min(buckets, L));
  const top = [], bot = [];
  for (let b = 0; b < B; b++) {
    const i0 = Math.floor(b * L / B), i1 = Math.max(i0 + 1, Math.floor((b + 1) * L / B));
    let mn = Infinity, mx = -Infinity; for (let i = i0; i < i1; i++) { const v = arr[i]; if (v < mn) mn = v; if (v > mx) mx = v; }
    const x = xOf((i0 + i1) / 2); top.push(`${x},${yOf(mx)}`); bot.push(`${x},${yOf(mn)}`);
  }
  return "M" + top.join("L") + "L" + bot.reverse().join("L") + "Z";
}
function ac1Value(a) { return PROB ? PROB.score(a) : NaN; }   // kept for older call sites; the score depends on the problem
/* programs: [{name, vals, color}] ; draws the step function (top) + the problem's profile (bottom):
   autoconvolution f*f for AC1, overlap profile C(s) for Erdos - see problems.js */
function solutionPlot(el, programs, o) {
  o = o || {};
  const hgt = o.compact ? 170 : 230;
  el.innerHTML = '<div class="sf"></div><div class="ac"></div>';
  const sfEl = $(".sf", el), acEl = $(".ac", el);
  if (!PROB || !PROB.prep) { el.innerHTML = '<div class="empty">no plot is defined for this problem</div>'; return []; }
  const prep = programs.map(p => Object.assign({}, p, PROB.prep(p.vals)));
  const mk = (host, which) => registerChart(host, () => {
    const F = frame(host, { height: hgt, margin: { l: 48, b: 34, t: 10, r: 12 } });
    const { svg, m, iw, ih } = F;
    let ymax = 0; prep.forEach(p => (which === "f" ? p.f : p.g).forEach(v => { if (v > ymax) ymax = v; }));
    ymax = which === "f" && PROB.fYmax ? PROB.fYmax : ymax * 1.1;
    const xs = linear(0, 1, m.l, m.l + iw), ys = linear(0, ymax, m.t + ih, m.t);
    axes(F, xs, ys, { xLabel: which === "f" ? PROB.fLabel : PROB.gLabel, xTicks: [], yFmt: v => v >= 10 ? v.toFixed(0) : v.toFixed(which === "g" && ymax < 1 ? 2 : 1) });
    if (which === "g") hline(F, ys, TARGET(), `${TARGET_LABEL()} ${TARGET().toFixed(PROB.dec + 1)}`, cssVar("--good-ink"), "5 4", "end");
    prep.forEach((p, pi) => {
      const arr = which === "f" ? p.f : p.g;
      const xOf = which === "f" ? (i => xs(i / p.n)) : (i => xs((i + 1) / (2 * p.n)));
      if (arr.length > 2500) {
        sv("path", { d: envelopePath(arr, xOf, v => ys(v), Math.floor(iw)), fill: p.color, "fill-opacity": 0.35, stroke: p.color, "stroke-width": 0.8, "stroke-linejoin": "round" }, svg);
      } else {
        let d = "";
        if (which === "f") { arr.forEach((v, i) => { const x0 = xs(i / p.n), x1 = xs((i + 1) / p.n); d += (i ? `V${ys(v)}` : `M${x0},${ys(v)}`) + `H${x1}`; }); }
        else { arr.forEach((v, i) => { d += (i ? "L" : "M") + `${xOf(i)},${ys(v)}`; }); }
        sv("path", { d, fill: "none", stroke: p.color, "stroke-width": which === "f" ? 1.2 : 1.6, "stroke-linejoin": "round" }, svg);
      }
      if (which === "g") hline(F, ys, p.mx, `max = ${p.mx.toFixed(PROB.dec + 1)}`, p.color, "2 3", "start", pi * 13);
    });
    const hit = sv("rect", { x: m.l, y: m.t, width: iw, height: ih, fill: "transparent" }, svg);
    hit.addEventListener("mousemove", ev => {
      const r = svg.getBoundingClientRect(), px = (ev.clientX - r.left) * (F.W / r.width), xv = clamp(xs.invert(px), 0, 1);
      const rows = prep.map(p => { const arr = which === "f" ? p.f : p.g; const i = clamp(Math.floor(xv * arr.length), 0, arr.length - 1); return `<div><span class="dot" style="background:${p.color}"></span>${esc(p.name)}: <b>${arr[i].toFixed(which === "g" && ymax < 1 ? 5 : 4)}</b> <span class="mut">(i=${i})</span></div>`; }).join("");
      tipShow(rows, ev);
    });
    hit.addEventListener("mouseleave", tipHide);
  });
  mk(sfEl, "f"); mk(acEl, "g");
  return prep;
}

/* ---------------------------------------------------------------- Sankey / flow diagram -- */
function sankey(el, o) {
  registerChart(el, () => {
    const W = Math.max(520, el.clientWidth || 760), H = o.height || 380, ml = 6, mr = 168, mt = 28, mb = 8, nw = 14, pad = 9;
    const iw = W - ml - mr, ih = H - mt - mb, nc = o.cols.length;
    el.innerHTML = "";
    const svg = sv("svg", { viewBox: `0 0 ${W} ${H}`, width: W, height: H }, el);
    const val = {}; o.links.forEach(l => { val[l.s] = val[l.s] || { i: 0, o: 0 }; val[l.t] = val[l.t] || { i: 0, o: 0 }; val[l.s].o += l.v; val[l.t].i += l.v; });
    const nodeV = n => { const x = val[n.id] || { i: 0, o: 0 }; return Math.max(x.i, x.o); };
    let k = Infinity;
    o.cols.forEach(c => { const ns = c.nodes.filter(n => nodeV(n) > 0), tot = sum(ns.map(nodeV)); if (tot > 0) k = Math.min(k, (ih - pad * Math.max(0, ns.length - 1)) / tot); });
    if (!isFinite(k)) { el.innerHTML = '<div class="empty">no data</div>'; return; }
    const pos = {};
    o.cols.forEach((c, ci) => {
      const ns = c.nodes.filter(n => nodeV(n) > 0), hh = sum(ns.map(nodeV)) * k + pad * Math.max(0, ns.length - 1);
      let y = mt + (ih - hh) / 2; const x = ml + (nc > 1 ? ci * (iw - nw) / (nc - 1) : 0);
      ns.forEach(n => { pos[n.id] = { x, y, h: nodeV(n) * k, n, oo: 0, io: 0, ci }; y += nodeV(n) * k + pad; });
      const t = sv("text", { x: x + (ci === nc - 1 ? nw : 0), y: 15, "text-anchor": ci === nc - 1 ? "end" : "start", style: "font-weight:600;fill:" + cssVar("--ink-2") }, svg); t.textContent = c.title;
    });
    const order = id => o.cols[pos[id].ci].nodes.findIndex(n => n.id === id);
    const links = o.links.filter(l => l.v > 0 && pos[l.s] && pos[l.t]);
    const bySrc = links.slice().sort((a, b) => order(a.t) - order(b.t));
    bySrc.forEach(l => { const p = pos[l.s]; l._y0 = p.y + p.oo; p.oo += l.v * k; });
    links.slice().sort((a, b) => order(a.s) - order(b.s)).forEach(l => { const p = pos[l.t]; l._y1 = p.y + p.io; p.io += l.v * k; });
    const clipId = "sk" + Math.random().toString(36).slice(2, 7), cp = sv("clipPath", { id: clipId }, svg);
    const clipRect = sv("rect", { x: 0, y: 0, width: W, height: H }, cp);
    const gl = sv("g", { "clip-path": `url(#${clipId})` }, svg);
    links.forEach(l => {
      const a = pos[l.s], b = pos[l.t], x0 = a.x + nw, x1 = b.x, th = l.v * k, xm = (x0 + x1) / 2;
      const d = `M${x0},${l._y0}C${xm},${l._y0} ${xm},${l._y1} ${x1},${l._y1}L${x1},${l._y1 + th}C${xm},${l._y1 + th} ${xm},${l._y0 + th} ${x0},${l._y0 + th}Z`;
      const p = sv("path", { d, fill: l.color || cssVar("--accent"), "fill-opacity": l.op || 0.42, class: "sk-link" }, gl);
      bindTip(p, () => o.tipLink ? o.tipLink(l) : `<b>${esc(a.n.label)} → ${esc(b.n.label)}</b><div>${fmt.n(l.v)} programs</div>`);
      if (th > 2.5 && !REDUCED) sv("path", { d: `M${x0},${l._y0 + th / 2}C${xm},${l._y0 + th / 2} ${xm},${l._y1 + th / 2} ${x1},${l._y1 + th / 2}`, class: "flowdash" }, gl);
    });
    Object.values(pos).forEach(p => {
      const r = sv("rect", { x: p.x, y: p.y, width: nw, height: Math.max(2, p.h), rx: 3, fill: p.n.color || cssVar("--ink-3") }, gl);
      bindTip(r, () => `<b>${esc(p.n.label)}</b><div>${fmt.n(nodeV(p.n))} programs</div>`);
      if (o.onNode) { r.style.cursor = "pointer"; r.addEventListener("click", () => o.onNode(p.n)); }
      const last = p.ci === nc - 1;
      const t = sv("text", { x: last ? p.x + nw + 6 : p.x + nw + 5, y: p.y + Math.max(2, p.h) / 2 + 4, class: "sk-lbl", style: "paint-order:stroke;stroke:" + cssVar("--surface") + ";stroke-width:3px;stroke-linejoin:round" }, gl);
      t.textContent = `${p.n.label} · ${fmt.n(nodeV(p.n))}`;
    });
    el._skPlay = (ms) => {
      if (REDUCED) return;
      const t0 = performance.now(), dur = ms || 3800;
      const tick = now => { const f = clamp((now - t0) / dur, 0, 1); clipRect.setAttribute("width", Math.max(2, W * f * 1.0)); if (o.onProgress) o.onProgress(f); if (f < 1 && el.isConnected) requestAnimationFrame(tick); };
      clipRect.setAttribute("width", 2); requestAnimationFrame(tick);
    };
    if (ANIM && o.autoPlay) el._skPlay(2600);
  });
}
