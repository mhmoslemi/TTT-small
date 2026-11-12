"use strict";
/* ---- Entropy & diversity tab ------------------------------------------------------------- */
registerTab({
  id: "entropy", label: "Entropy & diversity", ico: "≈",
  async render(root) {
    const done = STEPS.filter(s => ST[s].entropy && ST[s].entropy.all);
    root.append(pageHead("Entropy and diversity: is the coder collapsing?",
      `Entropy is the coder policy's mean full-vocabulary token entropy (nats, temperature 1) over <b>every response token</b> - reasoning included - measured in the pre-update training forward pass. ` +
      `Falling entropy means the policy is getting more deterministic. <b>Code diversity</b> is a separate lexical proxy: 1 − Jaccard similarity of 5-gram sketches of valid pilot programs of the same parent.`));
    if (!done.length) { root.append(h("div", { class: "hint" }, "No entropy.jsonl found for this run.")); return; }
    const E = s => ST[s].entropy;
    const e0 = E(done[0]).all.entropy_nats, e1 = E(done[done.length - 1]).all.entropy_nats;
    const div = s => E(s).code_diversity && E(s).code_diversity.within_strategy ? E(s).code_diversity.within_strategy.distance : null;
    root.append(h("div", { class: "kpis" },
      kpi("Entropy now", fmt.f(e1, 3) + " <span style='font-size:13px;color:var(--ink-3)'>nats/token</span>", `step ${done[0]}: ${fmt.f(e0, 3)} (${fmt.sign((e1 / e0 - 1) * 100, 0)}%)`),
      kpi("Mean response length", fmt.tok(Math.round(E(done[done.length - 1]).all.mean_response_tokens)) + " tokens", `step ${done[0]}: ${fmt.tok(Math.round(E(done[0]).all.mean_response_tokens))}`),
      kpi("Within-strategy diversity", fmt.f(div(done[done.length - 1]), 3), `step ${done[0]}: ${fmt.f(div(done[0]), 3)} (distance, 0 = identical)`),
      kpi("Tokens measured", fmt.tok(sum(done.map(s => E(s).all.measured_tokens))), "all steps, coverage " + fmt.pct(mean(done.map(s => E(s).all.token_coverage)), 0))));

    const a = h("div", { class: "chart" }), b = h("div", { class: "chart" }), c = h("div", { class: "chart" }), d = h("div", { class: "chart" });
    root.append(h("div", { class: "grid g2" },
      card("Policy entropy per step", "token-weighted mean, all rollouts of the step", a),
      card("Entropy by strategy position", "pooled over parents and phases", b),
      card("Mean response length", "tokens per rollout (reasoning + code)", c),
      card("Code diversity of valid pilot programs", "1 − Jaccard distance, 128-hash sketches of Python 5-grams", d)));
    const base = { xLabel: "step", margin: { l: 54 }, xTicks: done, xFmt: v => String(Math.round(v)), height: 240 };
    lineChart(a, Object.assign({}, base, { yLabel: "nats / token", yFmt: v => v.toFixed(2), series: [{ name: "all rollouts", color: S_COL(0), pts: done.map(s => [s, E(s).all.entropy_nats]), fmt: v => v.toFixed(4) }, { name: "valid fraction", color: S_COL(2), pts: done.map(s => [s, E(s).all.valid_fraction]), dash: "4 3", width: 1.5, fmt: v => v.toFixed(3) }], onClick: x => setState({ step: x }) }));
    lineChart(b, Object.assign({}, base, { yLabel: "nats / token", yFmt: v => v.toFixed(2), series: range(0, 4).map(k => ({ name: "S" + k, color: S_COL(k), pts: done.map(s => [s, (E(s).by_strategy[String(k)] || {}).entropy_nats]).filter(p => p[1] != null), fmt: v => v.toFixed(4) })) }));
    lineChart(c, Object.assign({}, base, { yLabel: "tokens", yFmt: v => (v / 1000).toFixed(0) + "k", series: [{ name: "mean tokens", color: S_COL(1), pts: done.map(s => [s, E(s).all.mean_response_tokens]), fmt: v => fmt.n(v) }] }));
    const ds = done.filter(s => E(s).code_diversity && E(s).code_diversity.within_strategy);
    lineChart(d, Object.assign({}, base, { yLabel: "distance", yFmt: v => v.toFixed(2), series: [
      { name: "within a strategy", color: S_COL(0), pts: ds.map(s => [s, E(s).code_diversity.within_strategy.distance]), fmt: v => v.toFixed(3) },
      { name: "across strategies", color: S_COL(3), pts: ds.map(s => [s, E(s).code_diversity.across_strategies.distance]), fmt: v => v.toFixed(3) }] }));
    root.append(legend(range(0, 4).map(k => ({ color: S_COL(k), label: "S" + k }))));

    // heat map parent x strategy for the selected step
    const step = S.step, bp = ST[step].entropy && ST[step].entropy.by_parent_strategy;
    if (bp) {
      const val = (i, j) => { const x = bp[`p${i}:s${j}`]; return x ? x.entropy_nats : null; };
      const vals = range(0, 8).flatMap(i => range(0, 4).map(j => val(i, j))).filter(x => x != null);
      const lo = Math.min(...vals), hi = Math.max(...vals);
      const hostH = h("div", { class: "chart" }), hostS = h("div", { class: "chart" });
      root.append(h("div", { class: "grid g2", style: { marginTop: "14px" } },
        card(`Entropy by parent × strategy · step ${step}`, "pilot rollouts of each cell (11 programs); darker = higher entropy", hostH),
        card("Does entropy relate to validity?", "one dot per (step, parent, strategy) cell, all steps", hostS)));
      heatmap(hostH, { rows: range(0, 8).map(i => "parent " + i), cols: ["S0", "S1", "S2", "S3"], rowLabelW: 62, maxCell: 90, cellH: 30,
        val, color: v => seqRamp((v - lo) / Math.max(1e-9, hi - lo)), text: v => v == null ? null : v.toFixed(2), textColor: v => (v - lo) / Math.max(1e-9, hi - lo) > 0.55 ? "#fff" : cssVar("--ink"),
        tip: (i, j, v) => { const x = bp[`p${i}:s${j}`]; return x ? `<b>parent ${i} · S${j}</b><div>${fmt.f(x.entropy_nats, 4)} nats/token</div><div>${x.measured_rollouts} rollouts · ${fmt.tok(Math.round(x.mean_response_tokens))} tokens each</div><div>valid ${fmt.pct(x.valid_fraction, 0)}</div>` : null; },
        onClick: (i, j) => setState({ tab: "strategist", group: i, sidx: j }) });
      const pts = [];
      done.forEach(s => { const m = E(s).by_parent_strategy || {}; for (const k in m) { const x = m[k]; pts.push({ x: x.entropy_nats, y: x.valid_fraction, c: S_COL(x.strategy), a: 0.7, r: 3.2, s, k, tok: x.mean_response_tokens }); } });
      scatterChart(hostS, { height: 240, points: pts, xLabel: "entropy (nats / token)", yLabel: "valid fraction", margin: { l: 48 }, xFmt: v => v.toFixed(2), yFmt: v => v.toFixed(1), yDomain: [-0.05, 1.05],
        tip: p => `<b>step ${p.s} · ${p.k}</b><div>entropy ${fmt.f(p.x, 3)}</div><div>valid ${fmt.pct(p.y, 0)} · ${fmt.tok(Math.round(p.tok))} tokens</div>`, onPoint: p => setState({ step: p.s }) });
      const mx = mean(pts.map(p => p.x)), my = mean(pts.map(p => p.y));
      const r = sum(pts.map(p => (p.x - mx) * (p.y - my))) / Math.sqrt(sum(pts.map(p => (p.x - mx) ** 2)) * sum(pts.map(p => (p.y - my) ** 2)));
      root.append(h("p", { class: "sub", style: { marginTop: "6px" } }, `Correlation between cell entropy and valid fraction across ${pts.length} cells: r = ${fmt.f(r, 3)}. (Step 0 cells have much higher entropy and lower validity - the fast early adaptation shows up here.)`));
    }
    root.append(explain("How entropy is measured", `Taken from the exact forward pass the trainer already runs before the update (no extra compute): <code>H_t = −Σ_v π(v|ctx) log π(v|ctx)</code> at logit temperature 1, summed over all response tokens of a rollout and divided by the (token-weighted) number of measured tokens. Prompt and padding are excluded; failed outputs are included. Code: <span class="cite">analysis/entropy_tools.py:76-124</span>.`));
  }
});
