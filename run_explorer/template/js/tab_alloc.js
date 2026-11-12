"use strict";
/* ---- Allocation tab: hurdle bandit for phase-2 rollouts ------------------------------------ */
let _allocSb = { step: null, group: null, s: null, budget: 20, sims: 3000 };

function mulberry32(a) { return function () { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
function randBeta(rng, a, b) { // integer a,b: a-th order statistic of (a+b-1) uniforms
  const u = []; for (let i = 0; i < a + b - 1; i++) u.push(rng()); u.sort((x, y) => x - y); return u[a - 1];
}
function randNorm(rng) { return Math.sqrt(-2 * Math.log(rng() + 1e-12)) * Math.cos(2 * Math.PI * rng()); }

/* Monte-Carlo replica of the allocator's model (rollout_allocation.py): gains are normalised by the pooled maximum; the future outcome of an arm is a
   Dirichlet mixture of {atom at 0 (weight n-s+1), pooled kernel (1), one kernel per own positive pilot (1 each)}; kernels are reflected Gaussians with a
   Silverman bandwidth. The 'budget' units are given greedily to the arm with the largest marginal expected improvement over the best pilot. */
function silverman(vals) {
  if (vals.length < 2) return vals.length ? Math.max(1e-3, mean(vals) * 0.2) : 0.1;
  const m = mean(vals), sd = Math.sqrt(sum(vals.map(v => (v - m) ** 2)) / (vals.length - 1));
  const iqr = (quantile(vals, 0.75) - quantile(vals, 0.25)) / 1.349;
  const sp = Math.min(sd, iqr > 0 ? iqr : sd);
  return sp > 1e-12 ? 0.9 * sp * Math.pow(vals.length, -0.2) : Math.max(1e-3, m);
}
function mcAllocate(arms, budget, nSims, seed, bwMult) {
  const rng = mulberry32(seed || 7), A = arms.length;
  const pooled = arms.flatMap(a => a.gains);
  if (!pooled.length) { const q = Math.floor(budget / A); const al = new Array(A).fill(q); for (let i = 0; i < budget - q * A; i++) al[i]++; return { alloc: al, ei: 0, fallback: true }; }
  const pmax = Math.max(...pooled), pn = pooled.map(x => x / pmax), h = silverman(pn) * (bwMult || 1);
  const expo = () => -Math.log(rng() + 1e-12);
  const gam = k => { let t = 0; for (let i = 0; i < k; i++) t += expo(); return t; };
  const pre = arms.map(() => []);
  arms.forEach((a, ai) => {
    const own = a.gains.map(x => x / pmax), K = own.length;
    pre[ai].push(new Float32Array(nSims));
    const arr = Array.from({ length: budget }, () => new Float32Array(nSims));
    const w = new Float64Array(K + 2);
    for (let k = 0; k < nSims; k++) {
      w[0] = gam(a.n - a.s + 1); w[1] = gam(1); let tot = w[0] + w[1];
      for (let j = 0; j < K; j++) { w[j + 2] = gam(1); tot += w[j + 2]; }
      let run = 0;
      for (let j = 0; j < budget; j++) {
        let u = rng() * tot, c = 0; while (c < K + 1 && u > w[c]) { u -= w[c]; c++; }
        if (c > 0) {
          const center = c === 1 ? pn[Math.floor(rng() * pn.length)] : own[c - 2];
          const g = Math.abs(center + h * randNorm(rng));
          if (g > run) run = g;
        }
        arr[j][k] = run;
      }
    }
    arr.forEach(x => pre[ai].push(x));
  });
  const cur = new Float32Array(nSims).fill(1), alloc = new Array(A).fill(0), trace = [];
  for (let u = 0; u < budget; u++) {
    let bestA = -1, bestMarg = -1;
    for (let ai = 0; ai < A; ai++) {
      const nxt = pre[ai][alloc[ai] + 1]; if (!nxt) continue;
      let tot = 0; for (let k = 0; k < nSims; k++) tot += nxt[k] > cur[k] ? nxt[k] - cur[k] : 0;
      const marg = tot / nSims;
      if (marg > bestMarg + 1e-12 || (Math.abs(marg - bestMarg) <= 1e-12 && bestA >= 0 && alloc[ai] < alloc[bestA])) { bestMarg = marg; bestA = ai; }
    }
    if (bestA < 0) break;
    alloc[bestA]++;
    const nxt = pre[bestA][alloc[bestA]]; for (let k = 0; k < nSims; k++) if (nxt[k] > cur[k]) cur[k] = nxt[k];
    trace.push(bestMarg);
  }
  let ei = 0; for (let k = 0; k < nSims; k++) ei += cur[k] - 1; ei = ei / nSims * pmax;
  return { alloc, ei, trace, bandwidth: h };
}

/* arms of one (step, parent) built from the logged pilots */
function realArms(step, g) {
  const al = [0, 1, 2, 3].map(s => allocOf(step, g, s)).filter(Boolean);
  if (!al.length) return null;
  const hurdle = al[0].hurdle.parent_reward;
  return al.map(a => {
    const rs = (RS[step] || []).filter(r => r.pg === g && r.sidx === a.strategy_index && r.phase === "pilot" && !r.isRetry);
    return { idx: a.strategy_index, s: a.hurdle.improvement_count, n: a.pilot_count, gains: rs.filter(r => r.valid && r.reward - hurdle > 1e-12).map(r => r.reward - hurdle), followups: a.followup_count, rec: a };
  });
}

function yBand(z, valid) { return !valid ? -1 : z > 1e-12 ? Math.min(6, 1 + Math.log10(z / 1e-9)) : 0; }
function yBandFmt(v) { return v === -1 ? "invalid" : v === 0 ? "valid, no gain" : (v >= 1 && Number.isInteger(v)) ? "+" + (1e-9 * Math.pow(10, v - 1)).toExponential(0).replace("e-", "e−") : ""; }

registerTab({
  id: "alloc", label: "Allocation (hurdle)", ico: "⚖",
  async render(root) {
    const step = S.step, g = S.group, al = ST[step].alloc || [];
    root.append(pageHead("Allocation: where do the 20 spare programs go?",
      `After each parent's 4 × 11 pilot programs are scored, a <b>hurdle bandit</b> decides how to spend the remaining <b>20 programs</b>. A program only counts if it <b>beats the parent's reward</b> (the hurdle); the allocator estimates, per strategy, <i>how likely</i> and <i>by how much</i> further programs would exceed the best pilot, then hands out the 20 units greedily by marginal expected improvement. ` +
      `Here you can inspect the real decisions and re-run a Monte-Carlo replica of the allocator with your own numbers.`));
    if (!al.length) { root.append(h("div", { class: "hint" }, `Step ${step} has no allocation record (not finished, or no summary).`)); return; }
    root.append(h("div", { class: "row-flex", style: { marginBottom: "12px" } }, h("span", { class: "sub" }, "Parent:"), seg(range(0, 8).map(x => ({ value: x, label: String(x) })), g, v => setState({ group: v }))));

    const arms = range(0, 4).map(s => allocOf(step, g, s)).filter(Boolean);
    const pil = s => (RS[step] || []).filter(r => r.pg === g && r.sidx === s && r.phase === "pilot" && !r.isRetry);
    const hurdle = arms[0].hurdle.parent_reward;

    // --- strip plot of pilot outcomes
    const hostS = h("div", { class: "chart" });
    root.append(h("div", { class: "grid g-main" },
      card(`Pilot outcomes of parent ${g} (step ${step})`, `hurdle = parent reward ${fmt.f(hurdle, 7)} (${PROB.metric} ${fmt.raw(arms[0].hurdle.parent_reward ? 1 / arms[0].hurdle.parent_reward : NaN)})`, h("div", null, hostS,
        legend(range(0, 4).map(s => ({ color: S_COL(s), label: "S" + s }))))),
      card("Reading the plot", null, h("div", { html: `<p style="color:var(--ink-2)">Each dot is one pilot program. <b>invalid</b> programs score 0. <b>valid, no gain</b> ran fine but did not beat the parent. Dots above that line beat the hurdle by <code>Z = R − R_parent</code> (log scale) - these are the only ones the allocator's gain model learns from.</p><p style="color:var(--ink-2)">Differences between a "good" and a "great" strategy are tiny in absolute terms (≈1e-7 in reward), which is why the allocator works with normalised gains and a kernel bandwidth.</p>` }))));
    const pts = [];
    arms.forEach(a => pil(a.strategy_index).forEach((r, k) => pts.push({ x: a.strategy_index + ((k * 37 % 11) / 11 - 0.5) * 0.55, y: yBand(r.reward - hurdle, r.valid), c: S_COL(a.strategy_index), id: r.id, r: 4.2, a: 0.85, rr: r })));
    scatterChart(hostS, {
      height: 300, points: pts, xDomain: [-0.5, 3.5], yDomain: [-1.6, 6.4], yTicks: [-1, 0, 1, 2, 3, 4, 5, 6], yFmt: yBandFmt, xTicks: [0, 1, 2, 3], xFmt: v => "S" + v, margin: { l: 96 },
      tip: p => `<b>S${p.rr.sidx} · rollout ${p.rr.rollout}</b><div>${p.rr.valid ? PROB.metric + " " + fmt.raw(p.rr.raw) : "invalid: " + esc(p.rr.fail)}</div>${p.rr.valid ? `<div>gain over parent: ${fmt.sci(p.rr.reward - hurdle, 2)}</div>` : ""}`,
      onPoint: p => openRollout(p.id),
    });

    // --- table
    const rows = arms;
    const winner = Math.max(...rows.map(a => a.followup_count));
    root.append(h("div", { style: { marginTop: "14px" } }, card("What the allocator saw and decided", "per strategy of this parent", h("div", { class: "scroll", style: { maxHeight: "none" } }, table([
      { label: "strategy", get: a => h("span", null, h("span", { class: "dot", style: { background: S_COL(a.strategy_index) } }), "S" + a.strategy_index) },
      { label: "valid", get: a => `${a.pilot_valid_count}/${a.pilot_count}`, num: true },
      { label: "improved s", get: a => a.hurdle.improvement_count, num: true, title: "valid pilots with reward > parent" },
      { label: "P(improve)", get: a => fmt.f(a.hurdle.posterior_improvement_probability, 3), num: true, title: "(s+1)/(n+2)" },
      { label: "mean + gain", get: a => fmt.sci(a.hurdle.mean_positive_improvement, 2), num: true },
      { label: "best + gain", get: a => fmt.sci(a.hurdle.best_positive_improvement, 2), num: true },
      { label: "initial EI", get: a => fmt.sci(a.initial_expected_improvement, 2), num: true, title: "expected improvement over the best pilot if this arm gets the whole budget's first unit" },
      { label: "final marginal EI", get: a => fmt.sci(a.final_marginal_expected_improvement, 2), num: true },
      { label: "follow-ups", get: a => a.followup_count ? pill(a.followup_count + (a.followup_count === winner ? "  ★" : ""), "good") : "0", num: true },
      { label: "total programs", get: a => a.allocated_programs, num: true },
    ], rows)))));
    root.append(h("p", { class: "sub", style: { marginTop: "6px" } }, `Projected expected gain for the whole parent: ${fmt.sci(arms[0].projected_expected_gain, 2)} (reward units). Scenario "${esc(arms[0].scenario)}", method "${esc(arms[0].allocation_method)}".`));

    // --- Monte-Carlo sandbox
    sandbox(root, step, g, arms, pil, hurdle);

    // --- run-wide: winner-take-all + regression to the mean
    runWide(root);
  }
});

function sandbox(root, step, g, arms, pil, hurdle) {
  const wrap = h("div"), outHost = h("div");
  root.append(h("div", { style: { marginTop: "14px" } }, card("Allocation sandbox (Monte-Carlo replica)", "same model family as <code>rollout_allocation.py</code> but sampled instead of integrated", wrap)));
  const real = realArms(step, g);
  const state = { s: real.map(a => a.s), budget: 20, bw: 1 };
  const sliders = h("div");
  wrap.append(h("div", { class: "grid g-main-r" }, sliders, outHost));
  const run = () => {
    const A = real.map((a, i) => {
      const gains = a.gains.slice(0, state.s[i]);
      const pool = real.flatMap(x => x.gains);
      for (let k = gains.length; k < state.s[i] && pool.length; k++) gains.push(pool[(k * 7 + i) % pool.length]);
      return { s: state.s[i], n: a.n, gains };
    });
    const res = mcAllocate(A, state.budget, _allocSb.sims, 11 + step * 13 + g, state.bw);
    outHost.innerHTML = "";
    const bars = h("div", { style: { display: "grid", gap: "8px" } });
    const mx = Math.max(state.budget, 1);
    real.forEach((a, i) => {
      bars.append(h("div", { style: { display: "grid", gridTemplateColumns: "34px 1fr 118px", gap: "8px", alignItems: "center", fontSize: "12.5px" } },
        h("b", null, "S" + a.idx),
        h("div", null,
          h("div", { style: { background: "var(--surface-2)", height: "14px", borderRadius: "4px", marginBottom: "3px" } }, h("div", { style: { width: (res.alloc[i] / mx * 100) + "%", height: "100%", background: S_COL(a.idx), borderRadius: "4px" } })),
          h("div", { style: { background: "var(--surface-2)", height: "8px", borderRadius: "3px" } }, h("div", { style: { width: (a.followups / 20 * 100) + "%", height: "100%", background: S_COL(a.idx), opacity: 0.4, borderRadius: "3px" } }))),
        h("span", { class: "mono", style: { whiteSpace: "nowrap" } }, `${res.alloc[i]}  (run: ${a.followups})`)));
    });
    outHost.append(h("h3", null, "Replica allocation vs. what the run did"), bars,
      h("p", { class: "sub", style: { marginTop: "8px" } }, "Thick bar = Monte-Carlo greedy allocation with your settings; thin bar = the recorded follow-ups. ", res.fallback ? "No pilot beat the parent, so the allocator falls back to an equal split." : `Expected improvement of the replica ≈ ${fmt.sci(res.ei, 2)} reward units (the log's projected gain: ${fmt.sci(arms[0].projected_expected_gain, 2)}); kernel bandwidth ${res.bandwidth.toFixed(3)} (normalised).`));
  };
  state.s.forEach((v, i) => sliders.append(slider(`S${real[i].idx} improvements s`, 0, 11, 1, v, x => { state.s[i] = x; run(); }, x => `${x}/11`)));
  sliders.append(slider("budget (programs)", 4, 40, 1, state.budget, x => { state.budget = x; run(); }, x => String(x)));
  sliders.append(slider("kernel bandwidth ×", 0.2, 3, 0.1, 1, x => { state.bw = x; run(); }, x => x.toFixed(1)));
  const val = h("div", { class: "sub", style: { marginTop: "8px" } });
  sliders.append(h("div", { class: "row-flex", style: { marginTop: "6px" } },
    h("button", { class: "btn sm", onclick: () => renderTab() }, "reset"),
    h("button", { class: "btn sm", onclick: () => { val.textContent = "validating on every parent of the run…"; setTimeout(() => { const r = validateReplica(); val.innerHTML = r; }, 30); } }, "validate replica on the whole run")));
  sliders.append(val);
  sliders.append(h("p", { class: "sub", style: { marginTop: "8px" } }, "More pilots that beat the parent ⇒ higher P(improve) ⇒ the arm attracts budget; but what matters most is whether an arm's gains reach the best pilot - an arm with few but near-maximal gains can beat one with many small ones."));
  run();
}

function validateReplica() {
  let n = 0, same = 0, l1 = 0, split = 0;
  STEPS.forEach(step => range(0, 8).forEach(g => {
    const arms = realArms(step, g); if (!arms || arms.length < 4) return;
    const A = arms.map(a => ({ s: a.s, n: a.n, gains: a.gains }));
    const res = mcAllocate(A, 20, 1000, 5 + step * 17 + g, 1);
    const realMax = Math.max(...arms.map(a => a.followups));
    const top = res.alloc.indexOf(Math.max(...res.alloc));
    n++; if (arms[top].followups === realMax) same++;
    if (realMax < 20) split++;
    l1 += sum(arms.map((a, i) => Math.abs(a.followups - res.alloc[i]))) / 2;
  }));
  return `Replica vs. run over ${n} parents: it gives the largest share to the same strategy as the real allocator in <b>${same}</b> (${fmt.pct(same / n, 0)}); mean number of programs moved: ${(l1 / n).toFixed(1)} of 20. ${split} real decisions were split.`;
}

function runWide(root) {
  const slots = [];
  STEPS.forEach(s => (ST[s].alloc || []).forEach(a => slots.push(Object.assign({ step: s }, a))));
  // per parent winner share
  const per = {};
  slots.forEach(a => { const k = a.step + ":" + a.parent_group; (per[k] = per[k] || []).push(a); });
  const maxFu = Object.values(per).map(v => Math.max(...v.map(a => a.followup_count)));
  const hist = {}; maxFu.forEach(m => hist[m] = (hist[m] || 0) + 1);
  const keys = Object.keys(hist).map(Number).sort((a, b) => b - a);
  const hostH = h("div", { class: "chart" }), hostR = h("div", { class: "chart" });
  const fav = slots.filter(a => a.followup_count >= 10);
  root.append(h("div", { class: "grid g2", style: { marginTop: "14px" } },
    card("How concentrated is the allocation?", `across ${maxFu.length} (step, parent) decisions: the largest share any one strategy received`, hostH),
    card("Do favoured strategies keep delivering?", "improvement rate in phase 2 vs. in the pilot (strategies that received ≥10 follow-ups)", hostR)));
  barChart(hostH, { height: 240, cats: keys.map(k => k + " of 20"), series: [{ name: "parents", color: S_COL(0), vals: keys.map(k => hist[k]) }], yLabel: "(step, parent) pairs", tip: (i, si, v) => `<b>${keys[i]} of the 20 follow-ups to one strategy</b><div>${v} decisions (${fmt.pct(v / maxFu.length, 0)})</div>` });
  const pts = fav.map(a => {
    const rs = (RS[a.step] || []).filter(r => r.pg === a.parent_group && r.sidx === a.strategy_index && !r.isRetry);
    const pi = rs.filter(r => r.phase === "pilot"), ad = rs.filter(r => r.phase === "adaptive");
    if (!pi.length || !ad.length) return null;
    return { x: pi.filter(r => r.improved).length / pi.length, y: ad.filter(r => r.improved).length / ad.length, c: S_COL(a.strategy_index), a: 0.8, r: 4, info: a };
  }).filter(Boolean);
  scatterChart(hostR, { height: 240, points: pts, xLabel: "fraction of pilot programs beating the parent", yLabel: "fraction of phase-2 programs", xDomain: [0, 1], yDomain: [0, 1], xFmt: v => v.toFixed(1), yFmt: v => v.toFixed(1), margin: { l: 50 },
    hlines: [], tip: p => `<b>step ${p.info.step} · parent ${p.info.parent_group} · S${p.info.strategy_index}</b><div>pilot ${fmt.pct(p.x, 0)} → phase 2 ${fmt.pct(p.y, 0)}</div>`, onPoint: p => setState({ step: p.info.step, group: p.info.parent_group }) });
  const mp = mean(pts.map(p => p.x)), ma = mean(pts.map(p => p.y));
  root.append(h("p", { class: "sub", style: { marginTop: "6px" } }, `Mean improvement rate of favoured strategies: pilot ${fmt.pct(mp, 0)} → phase 2 ${fmt.pct(ma, 0)} (n=${pts.length}). Dots below the diagonal show regression to the mean: part of the pilot's apparent advantage was luck.`));
}
