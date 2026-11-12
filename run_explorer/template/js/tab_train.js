"use strict";
/* ---- SPO² & training tab ------------------------------------------------------------------- */
/* All parameters (β, d_half, ρ bounds, per-step baseline override, clip ε) live in the shared store P,
   so every change here is also visible in the Pipeline tab (stage 6/7 + flow diagram) and the Rollouts tab. */
let _trainLocal = { D: null, Dstep: null };

function spoAdv(r, v, beta) { return (Math.exp(beta * r) / v - 1) / beta; }
function entropicAdv(r) {
  const K = r.length; if (K < 2) return r.map(() => 0);
  const mx = Math.max(...r), mn = Math.min(...r); if (mx - mn < 1e-12) return r.map(() => 0);
  const kl = beta => { const z = r.map(x => beta * (x - mx)), lse = Math.log(sum(z.map(Math.exp))); let s = 0; z.forEach(v => { const p = Math.exp(v - lse); if (p > 0) s += p * Math.log(p * K); }); return s; };
  let lo = 0, hi = 1; while (kl(hi) < Math.LN2 && hi < 1e6) hi *= 2;
  for (let i = 0; i < 60; i++) { const mid = (lo + hi) / 2; if (kl(mid) < Math.LN2) lo = mid; else hi = mid; }
  const beta = (lo + hi) / 2, e = r.map(x => Math.exp(beta * (x - mx))), tot = sum(e);
  const out = e.map(x => x / ((tot - x) / (K - 1)) - 1); out.beta = beta; return out;
}
function trackerUpdate(sumG, M, v, Neff, D, dHalf, rhoMin, rhoMax) {
  const rho = clamp(Math.pow(2, -D / dHalf), rhoMin, rhoMax), kept = rho * Neff, Nc = kept + M;
  return { rho, Nc, v: (kept * v + sumG) / Nc, eta: M / Nc };
}

registerTab({
  id: "train", label: "SPO² & training", ico: "∇",
  async render(root) {
    const step = S.step, st = ST[step];
    root.append(pageHead("Learning: SPO² advantages and the LoRA update",
      `<b>SPO²</b> (<code>spo-rs</code> in the code: <i>single-stream policy optimisation with a risk-sensitive exponential reward</i>) turns a program's reward into a training signal with <b>one run-wide baseline</b> <i>v</i> instead of comparing programs inside a group. ` +
      `Only the coder's LoRA adapter is trained - one PPO-clipped AdamW step per training step. ` +
      `<span class="pill acc">linked what-ifs</span> β, the tracker constants, the baseline override and ε are shared with the Pipeline and Rollouts tabs: change them here and the flow diagram, advantages and tables there follow.`));
    if (!st.spo) { root.append(h("div", { class: "hint" }, `Step ${step} has not finished training - pick a completed step.`)); }
    else {
      if (_trainLocal.Dstep !== step) { _trainLocal.D = st.spo.divergence; _trainLocal.Dstep = step; }
      playground(root, step);
      compareModes(root, step);
      trackerCalc(root, step);
    }
    trackerHistory(root);
    ppoSection(root, step);
    metricsSection(root);
  }
});

/* ------------------------------------------------------------ advantage playground ------ */
function playground(root, step) {
  const st = ST[step], rs = RS[step] || [];
  const host = h("div", { class: "chart" }), info = h("div"), ctl = h("div");
  root.append(card(`Advantage function A(r) · step ${step}`, "A = (exp(β·r)/v − 1)/β · drag the sliders; dots are real programs with their logged advantage", h("div", { class: "grid g-main" }, h("div", null, host, legend([{ color: S_COL(0), label: "A(r) with your β, v", line: true }, { color: S_COL(1), label: "this step's programs (logged A)" }])), h("div", null, ctl, info))));
  const draw = () => {
    const { beta, v, simV } = advParams(step);
    const sample = []; const seen = new Set();
    rs.forEach(r => { const k = Math.round(r.reward * 200) + ":" + Math.round((r.adv || 0) * 500); if (!seen.has(k) && r.adv != null) { seen.add(k); sample.push([r.reward, r.adv]); } });
    lineChart(host, {
      height: 300, xLabel: "reward r = 1 / " + PROB.metric, yLabel: "advantage A", margin: { l: 56 }, xDomain: [-0.02 * PROB.rewardMax, PROB.rewardMax], xTicks: niceTicks(0, PROB.rewardMax, 7), yFmt: x => x.toFixed(2),
      series: [
        { name: "A(r)", color: S_COL(0), pts: range(0, 73).map(i => { const r = i * PROB.rewardMax / 72; return [r, spoAdv(r, v, beta)]; }), dots: false, width: 2.4 },
        { name: "programs", color: S_COL(1), pts: sample.sort((a, b) => a[0] - b[0]).slice(0, 80), width: 0, dotR: 4 },
      ],
      hlines: [{ y: 0, label: "A = 0", color: cssVar("--ink-3") }], tipHead: x => `reward ${x.toFixed(2)}`,
    });
    const rstar = Math.log(v) / beta, valid = rs.filter(r => r.valid).map(r => r.reward), med = median(valid), best = Math.max(...valid, 0);
    info.innerHTML = "";
    info.append(h("div", { class: "kv", html:
      `<div class="k">Break-even r*</div><div class="v"><b>${fmt.f(rstar, 4)}</b> = ln(v)/β ⇒ ${PROB.metric} ${fmt.f(1 / rstar, 4)}</div>` +
      `<div class="k">A(failure, r=0)</div><div class="v">${fmt.f(spoAdv(0, v, beta), 4)}</div>` +
      `<div class="k">A(seed level, r=${fmt.f(PROB.seedReward, 2)})</div><div class="v">${fmt.f(spoAdv(PROB.seedReward, v, beta), 4)}</div>` +
      `<div class="k">A(median valid)</div><div class="v">${fmt.f(spoAdv(med, v, beta), 4)} <span class="sub">(r=${fmt.f(med, 4)})</span></div>` +
      `<div class="k">A(best of step)</div><div class="v">${fmt.f(spoAdv(best, v, beta), 4)} <span class="sub">(r=${fmt.f(best, 5)})</span></div>` +
      `<div class="k">Best − median</div><div class="v"><b>${fmt.f(spoAdv(best, v, beta) - spoAdv(med, v, beta), 5)}</b> <span class="sub">compare with the failure row: the signal is mostly valid-vs-invalid, not good-vs-great</span></div>` +
      `<div class="k">Baseline used</div><div class="v">v = ${fmt.f(v, 4)} <span class="sub">(${P.vOver[step] != null ? "your override" : "simulated by the tracker"}; logged ${fmt.f(st.spo.value_before, 4)})</span></div>` }));
    const logged = Math.abs(beta - 2) < 1e-9 && Math.abs(v - st.spo.value_before) < 1e-6;
    if (rs.length && logged) {
      const err = Math.max(...rs.filter(r => r.adv != null).map(r => Math.abs(spoAdv(r.reward, v, 2) - r.adv)));
      info.append(h("div", { class: "hint", style: { marginTop: "8px" } }, `Check: recomputing every logged advantage of step ${step} with β = 2 and v = ${fmt.f(v, 4)} reproduces them to max |Δ| = ${fmt.sci(err, 1)}.`));
    } else info.append(h("div", { class: "what-if", style: { display: "inline-block", marginTop: "8px" } }, "what-if values - also shown in the Pipeline flow, stage 6 and the Rollouts tab"));
  };
  ctl.append(paramSlider("β (risk sensitivity)", 0.1, 8, 0.1, x => x.toFixed(1), () => P.beta, (x, row) => setParam({ beta: x }, row)));
  const vmax = Math.max(4, st.spo.value_before * 1.8);
  ctl.append(paramSlider("baseline v (this step)", 1, vmax, vmax > 30 ? 0.1 : 0.01, x => x.toFixed(3), () => advParams(step).v, (x, row) => { const o = Object.assign({}, P.vOver); o[step] = x; setParam({ vOver: o }, row); }));
  ctl.append(h("div", { class: "row-flex", style: { margin: "4px 0 8px" } },
    h("button", { class: "btn sm", onclick: () => { const o = Object.assign({}, P.vOver); delete o[step]; setParam({ vOver: o }); } }, "use simulated v"),
    h("button", { class: "btn sm", onclick: () => resetParams() }, "reset all what-ifs")));
  draw();
  onParam(draw);
}

/* ------------------------------------------------------------ SPO² vs entropic group baseline ---- */
function compareModes(root, step) {
  const st = ST[step], g = S.group;
  const rs = (RS[step] || []).filter(r => r.pg === g && !r.isRetry);
  if (!rs.length) return;
  const rewards = rs.map(r => r.reward);
  const ent = entropicAdv(rewards);
  const order = rs.map((r, i) => i).sort((a, b) => rewards[a] - rewards[b]);
  const host = h("div", { class: "chart" }), lg = h("div");
  const allSame = Math.max(...ent) - Math.min(...ent) < 1e-9;
  root.append(h("div", { style: { marginTop: "14px" } }, card(`Same ${rs.length} programs, two ways to learn from them · parent ${g}`, "the other mode in the code base (entropic, per-group) vs the one this run uses (SPO², run-wide baseline)", h("div", null,
    h("div", { class: "row-flex", style: { marginBottom: "8px" } }, h("span", { class: "sub" }, "Parent:"), seg(range(0, 8).map(x => ({ value: x, label: String(x) })), g, x => setState({ group: x }))),
    host, lg,
    h("p", { class: "sub", style: { marginTop: "6px" } }, allSame ? "All programs in this group tie, so the entropic rule gives zero advantage everywhere." : "x = programs sorted by reward (failures at the left). The entropic rule normalises inside the group, so a handful of winners get large positive advantages and everything else ≈ −1; SPO² uses the same smooth transform for everyone, so advantages stay bounded and comparable across groups and steps. The SPO² series follows your β and v.")))));
  const draw = () => {
    const { beta, v } = advParams(step);
    const pts = [];
    order.forEach((i, k) => {
      pts.push({ x: k + 1, y: spoAdv(rewards[i], v, beta), c: S_COL(0), r: 3.4, a: 0.85, tag: "SPO²", rr: rs[i], r0: rewards[i] });
      pts.push({ x: k + 1, y: ent[i], c: S_COL(1), r: 3.4, a: 0.85, tag: "entropic (per-group)", rr: rs[i], r0: rewards[i] });
    });
    lg.innerHTML = "";
    lg.append(legend([{ color: S_COL(0), label: `SPO² (β=${beta.toFixed(1)}, v=${v.toFixed(3)})` }, { color: S_COL(1), label: `entropic: β solved per group to KL(softmax(βr)‖U)=ln2 ${ent.beta ? "→ β=" + (ent.beta > 1e4 ? ent.beta.toExponential(1) + " (near-tied rewards blow β up)" : ent.beta.toFixed(1)) : ""}` }]));
    scatterChart(host, {
      height: 280, points: pts, xLabel: "programs of this group, sorted by reward", yLabel: "advantage", margin: { l: 52 }, xFmt: v => String(Math.round(v)), yFmt: v => v.toFixed(1),
      tip: p => `<b>${p.tag}</b><div>reward ${fmt.f(p.r0, 4)}${p.rr.valid ? "" : " (failed)"}</div><div>advantage <b>${fmt.f(p.y, 4)}</b></div>`, onPoint: p => openRollout(p.rr.id),
    });
  };
  draw(); onParam(draw);
}

/* ------------------------------------------------------------ tracker update calculator ----- */
function trackerCalc(root, step) {
  const st = ST[step], sp = st.spo;
  const box = h("div"), res = h("div");
  root.append(h("div", { style: { marginTop: "14px" } }, card("Baseline tracker update", "after the step's advantages are formed, v moves toward the batch's mean exp(β·r) · constants are shared (they re-simulate the baseline of every step below)", h("div", { class: "grid g2" }, box, res))));
  const draw = () => {
    const t = trackerSim().byStep[step];
    if (!t) return;
    const D = _trainLocal.D;
    const u = trackerUpdate(t.meanG * t.M, t.M, t.vb, t.Nb, D, P.dHalf, P.rhoMin, P.rhoMax);
    res.innerHTML = "";
    res.append(h("div", { class: "eq", html: esc(
      `ρ   = clip(2^(−D/d_half), ρ_min, ρ_max) = clip(2^(−${D.toFixed(2)}/${P.dHalf.toFixed(2)}), ${P.rhoMin.toFixed(3)}, ${P.rhoMax.toFixed(3)}) = ${u.rho.toFixed(3)}\n` +
      `N_c = ρ·N_eff + M = ${u.rho.toFixed(3)}·${fmt.n(t.Nb, 1)} + ${t.M} = ${fmt.n(u.Nc, 1)}\n` +
      `v'  = (ρ·N_eff·v + ΣG)/N_c = ${u.v.toFixed(4)}          (ΣG = M·mean G = ${t.M}·${t.meanG.toFixed(4)}, G = exp(${P.beta.toFixed(1)}·r))\n` +
      `η   = M/N_c = ${u.eta.toFixed(4)}`) }));
    res.append(h("div", { class: "kv", html: `<div class="k">Logged result</div><div class="v">v ${fmt.f(sp.value_before, 4)} → <b>${fmt.f(sp.value_after, 4)}</b>, N_eff ${fmt.n(sp.effective_count_after, 1)}, η ${fmt.f(sp.eta, 4)}, ρ ${sp.rho}</div><div class="k">Your parameters give</div><div class="v">v ${fmt.f(t.vb, 4)} → <b>${u.v.toFixed(4)}</b>, N_eff ${fmt.n(u.Nc, 1)}, η ${u.eta.toFixed(4)}${trackerParamsChanged() ? ' <span class="what-if">what-if</span>' : ""}</div>` }));
    res.append(h("p", { class: "sub", style: { marginTop: "8px" } }, "D is the policy's sequence-level KL shift since the previous step (a SUM over ~20k tokens, so ≈4–5 nats). 2^(−D/0.06) is therefore ~0 and ρ always sits at its floor 0.875 after step 0 - d_half and ρ_max only mattered at step 0 (D=0 ⇒ ρ=0.96). In steady state N_eff → M/(1−ρ) = 4096 and η → 0.125."));
  };
  box.append(slider("divergence D (nats, this step only)", 0, 6, 0.05, _trainLocal.D, x => { _trainLocal.D = x; draw(); }, x => x.toFixed(2)));
  box.append(paramSlider("d_half", 0.01, 5, 0.01, x => x.toFixed(2), () => P.dHalf, (x, row) => setParam({ dHalf: x }, row)));
  box.append(paramSlider("ρ_min", 0.5, 0.99, 0.005, x => x.toFixed(3), () => P.rhoMin, (x, row) => setParam({ rhoMin: x }, row)));
  box.append(paramSlider("ρ_max", 0.5, 0.999, 0.005, x => x.toFixed(3), () => P.rhoMax, (x, row) => setParam({ rhoMax: x }, row)));
  box.append(h("div", { class: "sub" }, `Inputs from step ${step}: batch size M = ${sp.group_size}; mean G and the incoming baseline are re-simulated from the shared parameters (β = ${P.beta.toFixed(1)}).`));
  draw(); onParam(draw);
}

/* ------------------------------------------------------------ tracker history ----------------- */
function trackerHistory(root) {
  const done = STEPS.filter(s => ST[s].spo);
  if (!done.length) return;
  const a = h("div", { class: "chart" }), b = h("div", { class: "chart" }), c = h("div", { class: "chart" }), note = h("div", { class: "sub", style: { marginTop: "6px" } });
  root.append(h("div", { class: "grid g3", style: { marginTop: "14px" } },
    card("Baseline v: logged vs your parameters", `starts at the mean of exp(β·r) over the 8 seed rewards (= ${mean((PROB.seedRewards || [PROB.seedReward]).map(r => Math.exp(2 * r))).toFixed(2)} for β=2)`, h("div", null, a, note)),
    card("Update rate η and forgetting ρ", "solid = logged · dashed = your parameters", b),
    card("Policy shift D per step", "sequence-level KL to previous adapter, nats (held fixed in the simulation)", c)));
  const draw = () => {
    const sim = trackerSim().byStep, simDef = trackerSimDefault().byStep;
    const relErr = Math.max(...done.map(s => Math.abs(simDef[s].vb - ST[s].spo.value_before) / Math.abs(ST[s].spo.value_before)));
    const dirty = trackerParamsChanged();
    lineChart(a, { height: 230, xLabel: "step", yLabel: "v", margin: { l: 48 }, xTicks: done, xFmt: v => String(Math.round(v)), yFmt: v => v.toFixed(2),
      series: [{ name: "v before (logged)", color: S_COL(0), pts: done.map(s => [s, ST[s].spo.value_before]) }, { name: "v after (logged)", color: S_COL(1), pts: done.map(s => [s, ST[s].spo.value_after]), dash: "4 3" }]
        .concat(dirty ? [{ name: "v before (your parameters)", color: S_COL(3), pts: done.map(s => [s, sim[s].vb]), width: 2.2 }] : []) });
    note.innerHTML = dirty ? `<span class="what-if">what-if</span> the amber line re-simulates the whole run's baseline with your β / d_half / ρ (policy shift D held at its logged values).` : (relErr < 1e-6 ? `Simulating the tracker with the logged constants reproduces the run's v<sub>before</sub> (relative error ${fmt.sci(relErr, 1)}) - the model is exact.` : `Simulating the tracker with the logged constants reproduces the run's v<sub>before</sub> to ${fmt.pct(relErr, 3)} - close, not exact (resumed steps can shift the logged baseline slightly). All numbers shown are the logged ones until you change a constant.`);
    lineChart(b, { height: 230, xLabel: "step", yLabel: "η / ρ", margin: { l: 48 }, xTicks: done, xFmt: v => String(Math.round(v)), yFmt: v => v.toFixed(2),
      series: [{ name: "η logged", color: S_COL(3), pts: done.map(s => [s, ST[s].spo.eta]) }, { name: "ρ logged", color: S_COL(4), pts: done.map(s => [s, ST[s].spo.rho]) }]
        .concat(dirty ? [{ name: "η (yours)", color: S_COL(3), dash: "4 3", pts: done.map(s => [s, sim[s].eta]) }, { name: "ρ (yours)", color: S_COL(4), dash: "4 3", pts: done.map(s => [s, sim[s].rho]) }] : []) });
    lineChart(c, { height: 230, xLabel: "step", yLabel: "D (nats)", margin: { l: 48 }, xTicks: done, xFmt: v => String(Math.round(v)), yFmt: v => v.toFixed(1),
      series: [{ name: "mean D", color: S_COL(5), pts: done.map(s => [s, ST[s].spo.divergence]) }] });
  };
  draw(); onParam(draw);
}

/* ------------------------------------------------------------ PPO clip --------------------------- */
function ppoSection(root, step) {
  const host = h("div", { class: "chart" }), ctl = h("div"), info = h("div");
  root.append(h("div", { style: { marginTop: "14px" } }, card("PPO-clip objective per token", "L = −min(ρ·A, clip(ρ, 1−ε_low, 1+ε_high)·A) - why clipping is almost inert here", h("div", { class: "grid g-main" }, h("div", null, host, legend([{ color: S_COL(0), label: "A > 0", line: true }, { color: S_COL(1), label: "A < 0", line: true }])), h("div", null, ctl, info)))));
  const draw = () => {
    const lo = P.clipLo, hi = P.clipHi, f = (rho, A) => Math.min(rho * A, clamp(rho, 1 - lo, 1 + hi) * A);
    lineChart(host, { height: 270, xLabel: "ratio ρ = π_new / π_old", yLabel: "surrogate min(ρA, clip·A)", margin: { l: 56 }, xDomain: [0.5, 1.7], xFmt: v => v.toFixed(1), yFmt: v => v.toFixed(1),
      series: [{ name: "A=+1", color: S_COL(0), pts: range(0, 61).map(i => [0.5 + i * 0.02, f(0.5 + i * 0.02, 1)]), dots: false, width: 2.2 }, { name: "A=−1", color: S_COL(1), pts: range(0, 61).map(i => [0.5 + i * 0.02, f(0.5 + i * 0.02, -1)]), dots: false, width: 2.2 }],
      bands: [{ x0: 1 - lo, x1: 1 + hi, color: cssVar("--accent-soft") }], tipHead: x => "ρ = " + x.toFixed(2) });
    const u = (ST[step].policy_updates || [])[0];
    info.innerHTML = "";
    info.append(h("div", { class: "sub" }, `Shaded band = unclipped region [${(1 - lo).toFixed(2)}, ${(1 + hi).toFixed(2)}]. Outside it the gradient vanishes on the favourable side.`));
    if (u) info.append(h("div", { class: "kv", style: { marginTop: "8px" }, html: `<div class="k">Step ${step}: mean ρ</div><div class="v">${fmt.f(u.ratio, 6)}</div><div class="k">max ρ</div><div class="v">${fmt.f(u.ratio_max, 4)}</div><div class="k">clipped tokens</div><div class="v">${fmt.pct(u.clipped_fraction, 3)}</div>` }));
    info.append(h("p", { class: "sub", style: { marginTop: "8px" } }, "With one epoch and one optimiser step the new policy equals the old one when the gradient is computed, so ρ ≈ 1 for almost every token (the tiny spread is numerical noise between vLLM and the trainer). The clip only matters if the run is changed to several epochs."));
  };
  ctl.append(paramSlider("ε_low", 0.05, 0.5, 0.01, x => x.toFixed(2), () => P.clipLo, (x, row) => setParam({ clipLo: x }, row)));
  ctl.append(paramSlider("ε_high", 0.05, 0.6, 0.01, x => x.toFixed(2), () => P.clipHi, (x, row) => setParam({ clipHi: x }, row)));
  draw(); onParam(p => { if ("clipLo" in p || "clipHi" in p) draw(); });
}

/* ------------------------------------------------------------ metrics per step ------------------ */
function metricsSection(root) {
  const done = STEPS.filter(s => ST[s].policy_updates && ST[s].policy_updates.length);
  if (!done.length) return;
  const U = s => ST[s].policy_updates[0];
  const hosts = range(0, 4).map(() => h("div", { class: "chart" }));
  root.append(h("div", { class: "grid g2", style: { marginTop: "14px" } },
    card("Policy loss", "−mean_t min(ρA, clip·A), averaged over trajectories", hosts[0]),
    card("Gradient norm (log10)", "clip threshold is 1.0 - never reached", hosts[1]),
    card("Clipped-token fraction & max ratio", "almost no token reaches the clip boundary", hosts[2]),
    card("Training time & examples per GPU", "2 GPUs, adaptive process-per-GPU scheduler", hosts[3])));
  const base = { xLabel: "step", margin: { l: 56 }, xTicks: done, xFmt: v => String(Math.round(v)), height: 220 };
  lineChart(hosts[0], Object.assign({}, base, { yLabel: "loss", yFmt: v => v.toFixed(3), series: [{ name: "loss", color: S_COL(0), pts: done.map(s => [s, U(s).loss]), fmt: v => v.toFixed(5) }], hlines: [{ y: 0, color: cssVar("--ink-3") }], includeH: true }));
  lineChart(hosts[1], Object.assign({}, base, { yLabel: "log10 ‖g‖", yFmt: v => v.toFixed(1), series: [{ name: "grad norm", color: S_COL(3), pts: done.map(s => [s, Math.log10(U(s).grad_norm)]), fmt: v => fmt.sci(Math.pow(10, v), 2) }], hlines: [{ y: 0, label: "clip = 1.0", color: cssVar("--crit") }], includeH: true }));
  lineChart(hosts[2], Object.assign({}, base, { yLabel: "clipped %", yFmt: v => v.toFixed(2), series: [{ name: "clipped tokens %", color: S_COL(2), pts: done.map(s => [s, U(s).clipped_fraction * 100]), fmt: v => v.toFixed(3) + "%" }, { name: "(max ρ − 1)·100", color: S_COL(1), pts: done.map(s => [s, (U(s).ratio_max - 1) * 100]), fmt: v => v.toFixed(3) }] }));
  barChart(hosts[3], { height: 220, cats: done.map(String), xLabel: "step", yLabel: "minutes", margin: { l: 52 }, yFmt: v => v.toFixed(0), bands: true, segLabels: (i, si, v) => (v >= 20 ? v.toFixed(0) + "m" : ""), series: [{ name: "training time", color: S_COL(7), vals: done.map(s => (ST[s].training_seconds || 0) / 60) }], tip: (i, si, v) => `<b>step ${done[i]}</b><div>${fmt.dur(v * 60)} · examples/GPU ${(ST[done[i]].fast_trained_examples_per_gpu || []).join(" + ")}</div><div class="mut">peak mem ${(ST[done[i]].fast_peak_memory_fraction || []).map(x => fmt.pct(x, 0)).join(" / ")}</div>` });
  root.append(h("p", { class: "sub", style: { marginTop: "6px" } }, "Reading these curves: a flat, tiny gradient norm and a vanishing clipped fraction are what a single on-policy step looks like. The learning rate (4e-5 on rank-32 LoRA) and the advantage scale set the actual step size; the run has no KL-to-reference term (coefficient forced to 0 for SPO²)."));
}
