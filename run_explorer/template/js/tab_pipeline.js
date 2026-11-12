"use strict";
/* ---- Pipeline tab: one training step, stage by stage ----------------------------------- */
let _stageSel = "select", _walk = null;

function parentsOf(step) { return (ST[step] && ST[step].parents) || []; }
function allocOf(step, g, s) { return ((ST[step] && ST[step].alloc) || []).find(a => a.parent_group === g && a.strategy_index === s); }
function slotsOf(step) { // from the strategist blob: keyed "step:g:s"
  const out = [];
  if (!D.strat) return out;
  for (const k in D.strat.strategies) { const [st, g, s] = k.split(":").map(Number); if (st === step) out.push(D.strat.strategies[k]); }
  return out;
}

registerTab({
  id: "pipeline", label: "Pipeline (one step)", ico: "⛓",
  async render(root) {
    await lazy("strat");
    const step = S.step, st = ST[step], rs = RS[step] || [];
    root.append(pageHead(`How step ${step} was produced`,
      `One training step runs <b>eight stages</b>. The numbers below are the real counts for the selected step; click a stage for the mechanism, the exact formula, the code that does it, and the data it produced. ` +
      (st.completed ? "" : st.provisional ? `<span class="pill warn">step ${step}: all rollouts are scored (its best counts for best-so-far), the adapter update and summary are not written yet - training numbers are missing</span>` : `<span class="pill warn">step ${step} is still in progress on disk - only the stages that already wrote files are filled in</span>`)));

    const planned = rs.filter(r => !r.isRetry), retries = rs.filter(r => r.isRetry);
    const slots = slotsOf(step);
    const attempts = sum(slots.map(s => s.attempts.length)), fallbacks = slots.filter(s => s.fallback).length;
    const jev = st.jev, rejected = jev ? (jev.decisions || []).filter(d => !d.accepted).length : 0;
    const nums = {
      select: [parentsOf(step).length || "–", jev && jev.candidate_limit ? `${jev.comparisons.length} JEV judgments · ${rejected} rejected` : (RUN.hasJev ? "plain PUCT on the seeds" : "PUCT with lineage blocking")],
      strategist: [slots.length ? `${slots.length}` : "–", `${attempts} attempts · ${fallbacks} fallbacks`],
      pilot: [planned.filter(r => r.phase === "pilot").length || "–", "programs · 11 per strategy"],
      alloc: [planned.filter(r => r.phase === "adaptive").length || "–", "programs · 20 per parent"],
      eval: [rs.length ? fmt.pct(rs.filter(r => r.valid).length / rs.length, 0) : "–", `valid of ${rs.length}${retries.length ? ` (${retries.length} retries)` : ""}`],
      adv: [st.spo ? "v " + fmt.f(st.spo.value_before, 3) : "–", st.spo ? `→ ${fmt.f(st.spo.value_after, 3)} · η ${fmt.f(st.spo.eta, 2)}` : ""],
      train: [st.training_seconds ? fmt.dur(st.training_seconds) : "–", st.policy_updates && st.policy_updates[0] ? `grad ‖g‖ ${fmt.sci(st.policy_updates[0].grad_norm, 1)}` : ""],
      archive: [st.archive_size || "–", st.archive_size ? `states after step ${step}` : ""],
    };

    const flow = h("div", { class: "flow" });
    STAGES.map(stageView).forEach((sg, i) => {
      if (i) flow.append(h("div", { class: "arrow" + (_walk && _stageSel === sg.id ? " playing" : "") }, "›"));
      const [big, sub] = nums[sg.id];
      flow.append(h("div", { class: "stage" + (_stageSel === sg.id ? " sel" : "") + (_walk && _stageSel === sg.id ? " playing" : ""), "data-stage": sg.id, onclick: () => { _stageSel = sg.id; renderTab(); } },
        h("div", { class: "n" }, "STAGE " + sg.n), h("div", { class: "t" }, sg.title),
        h("div", { class: "big" }, String(big)), h("div", { class: "m" }, sub),
        h("div", { class: "model-chip " + sg.chip[1], title: sg.chip[2] }, sg.chip[0])));
    });
    const walkBtn = h("button", { class: "btn" + (_walk ? " on" : ""), onclick: () => walkToggle() }, _walk ? "■ stop walkthrough" : "▶ walk through the stages");
    root.append(card(null, null, h("div", null, h("div", { class: "row-flex", style: { marginBottom: "6px", justifyContent: "space-between" } }, h("div", { class: "sub" }, `Step ${step}: 8 parents → 32 strategies → 512 programs → one LoRA update`), walkBtn), flow)));

    // ---- flow diagram of where the programs go (animated)
    if (rs.length) {
      const sk = h("div", { class: "chart" });
      const playBtn = h("button", { class: "btn sm", onclick: () => sk._skPlay && sk._skPlay(4600) }, "▶ replay the flow");
      root.append(h("div", { style: { marginTop: "14px" } }, card(`Where did the programs of step ${step} go?`, "band width = number of programs · hover a band · the right-hand column follows your what-if β / v", h("div", null,
        h("div", { class: "row-flex", style: { marginBottom: "6px" } }, playBtn, h("span", { class: "sub" }, "the flow reveals left → right and lights up the matching stage above")), sk))));
      const drawSk = () => { const fd = flowData(step); sankey(sk, { cols: fd.cols, links: fd.links, height: 410, autoPlay: true, onProgress: highlightStageByProgress, tipLink: l => fd.tip(l) }); };
      drawSk();
      onParam(p => { if ("beta" in p || "vOver" in p || "dHalf" in p || "rhoMin" in p || "rhoMax" in p) { const keepAnim = ANIM; ANIM = false; drawSk(); ANIM = keepAnim; } });
    }

    const sg = stageView(STAGES.find(x => x.id === _stageSel) || STAGES[0]);
    const left = h("div", { class: "card" },
      h("div", { class: "card-head" }, h("h2", null, `${sg.n}. ${sg.title}`), h("span", { class: "model-chip " + sg.chip[1] }, sg.chip[0])),
      h("div", { html: sg.body, style: { color: "var(--ink-2)" } }),
      h("h3", { style: { marginTop: "12px" } }, "Mechanism"), h("div", { class: "eq" }, sg.eq),
      h("h3", { style: { marginTop: "10px" } }, "Where in the code"), h("div", null, sg.refs.map(r => h("div", { class: "cite" }, r))),
      h("h3", { style: { marginTop: "10px" } }, "Files written"), h("div", null, sg.files.map(f => h("div", { class: "cite" }, f))));
    const right = h("div", { class: "card" }, h("div", { class: "card-head" }, h("h2", null, `Step ${step} in numbers`), h("span", { class: "sub" }, "live data from this run")));
    root.append(h("div", { class: "grid g-main-r", style: { marginTop: "14px", alignItems: "start" } }, left, right));
    await STAGE_VIZ[sg.id](right, step);
  }
});

function walkToggle() {
  if (_walk) { clearInterval(_walk); _walk = null; renderTab(); return; }
  let i = STAGES.findIndex(s => s.id === _stageSel);
  _walk = setInterval(() => {
    if (S.tab !== "pipeline") { clearInterval(_walk); _walk = null; return; }
    i = (i + 1) % STAGES.length; _stageSel = STAGES[i].id; renderTab();
  }, 3200);
  renderTab();
}

/* ---------------------------------------------------------------------------------------- */
const STAGE_VIZ = {
  async select(box, step) {
    const ps = parentsOf(step);
    if (!ps.length) { box.append(h("div", { class: "empty" }, "No parents recorded for this step.")); return; }
    const jev = ST[step].jev;
    box.append(h("div", { class: "sub", html: `Sampler <b>${esc(ST[step].sampler_type || "")}</b>${jev && jev.candidate_limit ? ` · looked at the top ${jev.candidate_limit} PUCT states, accepted ${jev.selected.length}` : (RUN.hasJev ? " · step 0: no JEV, plain PUCT over the seeds" : " · picks are lineage-blocked (no JEV gate in this run)")}` }));
    const c0 = CFG.puct_c || 1;
    box.append(paramSlider("exploration constant c", 0, 20, 0.5, v => v.toFixed(1), () => P.c != null ? P.c : c0, (v, row) => setParam({ c: v }, row)));
    const tbl = h("div"), jv = h("div");
    const draw = () => {
      tbl.innerHTML = "";
      const { rows, order } = pucRows(step);
      tbl.append(h("div", { class: "scroll" }, table([
        { label: "grp", get: p => String(p.group), num: true },
        { label: "parent", get: p => h("a", { href: "javascript:void 0", class: "mono", onclick: e => { e.stopPropagation(); openNode(p.parent_id); } }, fmt.id(p.parent_id)) },
        { label: "made in", get: p => p.parent_is_seed ? "seed" : "step " + p.parent_timestep, num: true },
        { label: PROB.metric, get: p => fmt.raw(p.parent_raw_score), num: true },
        { label: "Q", get: p => fmt.f(p.q_value, 5), num: true },
        { label: "explore", get: p => fmt.f(p.bonus2, 4), num: true, title: "c·scale·P·√(1+T)/(1+n)" },
        { label: "score", get: p => fmt.f(p.score2, 5), num: true },
        { label: "order", get: p => String(order.indexOf(p.group) + 1), num: true },
        ...(RUN.hasJev ? [{ label: "JEV", get: p => { const d = (p.jev || {}).decision; return pill(d === "accepted" ? "accepted" : d === "no_comparison" ? "first pick" : d === "step_zero_puct" ? "PUCT only" : d || "?", d === "accepted" || d === "no_comparison" ? "good" : d === "duplicate_fallback" ? "warn" : ""); } }] : []),
      ], rows, { onRow: p => { setState({ tab: "search", group: p.group }); } })));
    };
    box.append(tbl);
    if (jev && jev.decisions && jev.decisions.length) {
      box.append(h("div", { style: { marginTop: "10px" } }, paramSlider("JEV same-core threshold", 0.3, 1, 0.01, v => v.toFixed(2), () => thrNow(step), (v, row) => setParam({ thr: v }, row))));
      box.append(jv);
      const drawJ = () => {
        const r = jevReplay(step, thrNow(step)); jv.innerHTML = "";
        jv.append(h("div", { class: "row-flex" }, pill(`recorded @ ${r.recorded}: ${r.recAcc} accepted · ${r.recRej} rejected`, "acc"), pill(`replay @ ${r.thr.toFixed(2)}: ${r.nAcc} accepted · ${r.nRej} rejected`, r.thr === r.recorded ? "good" : "warn"),
          P.thr != null && P.thr !== r.recorded ? h("span", { class: "what-if" }, "what-if") : null));
      };
      drawJ(); onParam(p => { if ("thr" in p) drawJ(); });
    }
    draw(); onParam(p => { if ("c" in p) draw(); });
    box.append(h("p", { class: "sub", style: { marginTop: "8px" } }, (RUN.hasJev ? "These two controls are shared with the Search tab: moving them here changes the replay there and vice-versa. " : "The exploration constant is shared with the Search tab. ") + "Click a row to open the selection in the Search tab."));
  },

  async strategist(box, step) {
    const slots = slotsOf(step);
    if (!slots.length) { box.append(h("div", { class: "empty" }, "No strategist files for this step.")); return; }
    const by = {}; slots.forEach(s => (by[s.g + ":" + s.s] = s));
    const host = h("div", { class: "chart" }); box.append(host);
    const fallbacks = slots.filter(s => s.fallback).length, tries = sum(slots.map(s => s.attempts.length));
    box.append(h("div", { class: "kv", style: { marginTop: "10px" }, html:
      `<div class="k">Slots</div><div class="v">${slots.length} (8 chains × 4)</div>` +
      `<div class="k">Generation calls</div><div class="v">${tries} (${fmt.f(tries / slots.length, 2)} per slot)</div>` +
      `<div class="k">Accepted first try</div><div class="v">${slots.filter(s => s.attempts.length && s.attempts[0].ok).length}</div>` +
      `<div class="k">Fallback (no plan)</div><div class="v">${fallbacks} (${fmt.pct(fallbacks / slots.length, 0)})</div>` }));
    const col = s => !s ? null : s.fallback ? cssVar("--crit") : s.attempts.length > 1 ? cssVar("--warn") : cssVar("--good");
    heatmap(host, {
      rows: range(0, 8).map(g => "parent " + g), cols: ["S0", "S1", "S2", "S3"], rowLabelW: 62, maxCell: 90, cellH: 30,
      val: (i, j) => by[i + ":" + j] ? by[i + ":" + j].attempts.length || 1 : null,
      color: (v, i, j) => col(by[i + ":" + j]),
      text: (v, i, j) => by[i + ":" + j] ? (by[i + ":" + j].fallback ? "✕" : String(by[i + ":" + j].attempts.length)) : null,
      textColor: () => "#0b0b0b",
      tip: (i, j) => { const s = by[i + ":" + j]; if (!s) return null; return `<b>parent ${i} · strategy S${j}</b><div>${s.attempts.length} attempt(s): ${s.attempts.map(a => a.ok ? "✓" : "✗").join(" ")}</div><div>${s.fallback ? "fallback: coder got no plan" : "plan length " + fmt.n(s.block.length) + " chars"}</div><div class="mut">reasoning ${fmt.n(s.think_chars)} chars</div>`; },
      onClick: (i, j) => setState({ tab: "strategist", group: i, sidx: j }),
    });
    box.append(legend([{ color: cssVar("--good"), label: "accepted first try" }, { color: cssVar("--warn"), label: "accepted after retries (number = attempts)" }, { color: cssVar("--crit"), label: "fallback ✕" }]));
    box.append(h("p", { class: "sub" }, "Click a cell to read that strategy, its attempts and the prompt anatomy in the Strategist tab."));
  },

  async pilot(box, step) {
    const rs = (RS[step] || []).filter(r => !r.isRetry && r.phase === "pilot");
    if (!rs.length) { box.append(h("div", { class: "empty" }, "No coder rollouts for this step yet.")); return; }
    const cell = (g, s) => rs.filter(r => r.pg === g && r.sidx === s);
    const host = h("div", { class: "chart" }); box.append(host);
    heatmap(host, {
      rows: range(0, 8).map(g => "parent " + g), cols: ["S0", "S1", "S2", "S3"], rowLabelW: 62, maxCell: 90, cellH: 30,
      val: (i, j) => { const c = cell(i, j); return c.length ? c.filter(r => r.valid).length / c.length : null; },
      color: v => seqRamp(v), text: (v, i, j) => { const c = cell(i, j); return c.length ? c.filter(r => r.valid).length + "/" + c.length : null; },
      textColor: v => v > 0.55 ? "#fff" : cssVar("--ink"),
      tip: (i, j) => { const c = cell(i, j); const v = c.filter(r => r.valid); return `<b>parent ${i} · S${j}</b><div>${v.length}/${c.length} valid</div><div>beat parent: ${c.filter(r => r.improved).length}</div><div>best bound: ${v.length ? fmt.raw(Math.min(...v.map(r => r.raw))) : "–"}</div><div class="mut">mean reward ${fmt.f(mean(c.map(r => r.reward)), 4)}</div>`; },
      onClick: (i, j) => setState({ tab: "rollouts", group: i, sidx: j }),
    });
    box.append(h("p", { class: "sub", style: { marginTop: "6px" } }, "Color = fraction of the 11 pilot programs that were valid. Each strategy's pilot decides how the 20 spare programs get spent (next stage)."));
    const v = rs.filter(r => r.valid);
    box.append(h("div", { class: "kv", style: { marginTop: "8px" }, html:
      `<div class="k">Pilot programs</div><div class="v">${rs.length}</div><div class="k">Valid</div><div class="v">${v.length} (${fmt.pct(v.length / rs.length)})</div>` +
      `<div class="k">Mean response</div><div class="v">${fmt.tok(Math.round(mean(rs.map(r => r.ntok || 0))))} tokens</div>` +
      `<div class="k">Best pilot bound</div><div class="v">${v.length ? fmt.raw(Math.min(...v.map(r => r.raw))) : "–"}</div>` }));
  },

  async alloc(box, step) {
    const al = (ST[step].alloc) || [];
    if (!al.length) { box.append(h("div", { class: "empty" }, "No allocation recorded for this step.")); return; }
    box.append(h("div", { class: "sub" }, "Each bar = the 64 programs of one parent: grey = 11 pilot programs per strategy, bright = follow-ups granted by the hurdle allocator."));
    const wrap = h("div", { style: { display: "grid", gap: "8px", marginTop: "8px" } });
    for (let g = 0; g < 8; g++) {
      const row = range(0, 4).map(s => allocOf(step, g, s)).filter(Boolean);
      if (!row.length) continue;
      const bar = h("div", { class: "sbar", style: { height: "20px" } });
      row.forEach(a => {
        const pil = h("span", { style: { width: (a.pilot_count / 64 * 100) + "%", background: S_COL(a.strategy_index), opacity: 0.38 } });
        bar.append(pil);
        if (a.followup_count) bar.append(h("span", { style: { width: (a.followup_count / 64 * 100) + "%", background: S_COL(a.strategy_index) } }));
        bindTip(pil, () => tipHtmlAlloc(a));
      });
      wrap.append(h("div", { style: { display: "grid", gridTemplateColumns: "58px 1fr 120px", gap: "8px", alignItems: "center", fontSize: "12px" }, onclick: () => setState({ tab: "alloc", group: g }), }, h("span", null, "parent " + g), bar,
        h("span", { class: "sub" }, row.map(a => a.allocated_programs).join(" / "))));
    }
    box.append(wrap);
    box.append(legend(range(0, 4).map(s => ({ color: S_COL(s), label: "S" + s }))));
    const winners = range(0, 8).map(g => { const r = range(0, 4).map(s => allocOf(step, g, s)).filter(Boolean); return r.length ? Math.max(...r.map(a => a.followup_count)) : null; }).filter(x => x != null);
    box.append(h("div", { class: "kv", style: { marginTop: "8px" }, html:
      `<div class="k">Winner-take-all parents</div><div class="v">${winners.filter(x => x === 20).length} of ${winners.length} gave all 20 to one strategy</div>` +
      `<div class="k">Split parents</div><div class="v">${winners.filter(x => x < 20).length}</div>` }));
    box.append(h("p", { class: "sub" }, "Click a bar to open the allocation sandbox for that parent."));
  },

  async eval(box, step) {
    const rs = RS[step] || [];
    if (!rs.length) { box.append(h("div", { class: "empty" }, "No scored programs yet.")); return; }
    const total = rs.length;
    const parsed = rs.filter(r => r.parsed).length, ran = rs.filter(r => r.ran).length, valid = rs.filter(r => r.valid).length, imp = rs.filter(r => r.improved).length;
    const steps = [["generated", total], ["code block extracted", parsed], ["ran to completion", ran], ["valid score", valid], ["beat their parent", imp]];
    box.append(h("div", { style: { display: "grid", gap: "6px" } }, steps.map(([n, c], i) =>
      h("div", { style: { display: "grid", gridTemplateColumns: "150px 1fr 90px", gap: "8px", alignItems: "center", fontSize: "12.5px" } },
        h("span", null, n), h("div", { style: { background: "var(--surface-2)", borderRadius: "5px", height: "18px" } }, h("div", { style: { width: (c / total * 100) + "%", height: "100%", borderRadius: "5px", background: i === 4 ? cssVar("--s3") : S_COL(0), opacity: 0.4 + 0.15 * (4 - i) } })),
        h("span", { class: "mono" }, `${c} · ${fmt.pct(c / total, 0)}`)))));
    const st = stepStats(step);
    box.append(h("div", { class: "kv", style: { marginTop: "12px" }, html:
      `<div class="k">Failure kinds</div><div class="v">code ${st.code} · timeout ${st.timeout} · invalid score ${st.constraint}</div>` +
      `<div class="k">Best valid bound</div><div class="v">${fmt.raw(st.best)}</div>` +
      `<div class="k">Median valid bound</div><div class="v">${fmt.raw(median(st.raws))}</div>` +
      `<div class="k">Evaluation pool</div><div class="v">${ST[step].evaluation_workers || "–"} sandboxes on ${ST[step].evaluation_cpu_count || "–"} CPUs</div>` }));
    const msgs = {}; rs.filter(r => !r.valid).forEach(r => { const k = (r.msg || "").replace(/[0-9.]+/g, "#").slice(0, 70); msgs[k] = (msgs[k] || 0) + 1; });
    const top = Object.entries(msgs).sort((a, b) => b[1] - a[1]).slice(0, 6);
    box.append(h("h3", { style: { marginTop: "12px" } }, "Most common failure messages"));
    box.append(h("div", null, top.map(([k, c]) => h("div", { class: "row-flex", style: { fontSize: "12px", justifyContent: "space-between" } }, h("span", { class: "mono", style: { color: "var(--ink-2)" } }, k || "(empty)"), h("b", null, c)))));
    box.append(h("p", { class: "sub", style: { marginTop: "8px" } }, "Browse every program in the Rollouts tab."));
  },

  async adv(box, step) {
    const st = ST[step];
    if (!st.spo) { box.append(h("div", { class: "empty" }, "No SPO² update recorded for this step yet.")); return; }
    const rs = RS[step] || [];
    const host = h("div", { class: "chart" }), info = h("div");
    box.append(paramSlider("β (risk sensitivity)", 0.1, 8, 0.1, v => v.toFixed(1), () => P.beta, (v, row) => setParam({ beta: v }, row)));
    const vmax = Math.max(4, st.spo.value_before * 1.8);
    box.append(paramSlider("baseline v (this step)", 1, vmax, 0.01, v => v.toFixed(3), () => advParams(step).v, (v, row) => { const o = Object.assign({}, P.vOver); o[step] = v; setParam({ vOver: o }, row); }));
    box.append(host, info);
    const draw = () => {
      const { beta, v, simV } = advParams(step), A = r => (Math.exp(beta * r) / v - 1) / beta;
      lineChart(host, {
        height: 220, xLabel: "reward r = 1 / " + PROB.metric, yLabel: "advantage A(r)", margin: { l: 56 }, xDomain: [0, PROB.rewardMax], xTicks: niceTicks(0, PROB.rewardMax, 7), yFmt: x => x.toFixed(2),
        series: [{ name: "A(r)", color: S_COL(0), pts: range(0, 73).map(i => { const r = i * PROB.rewardMax / 72; return [r, A(r)]; }), dots: false, width: 2.2 }],
        hlines: [{ y: 0, label: "A = 0", color: cssVar("--ink-3") }], tipHead: x => `reward ${x.toFixed(2)}`,
      });
      const rstar = Math.log(v) / beta;
      const mk = (name, f) => { const a = rs.filter(f); return a.length ? `<div class="k">${name}</div><div class="v">${a.length} programs · mean A = <b>${fmt.f(mean(a.map(r => advWhatIf(r))), 4)}</b>${advIsWhatIf() ? ` <span class="sub">(logged ${fmt.f(mean(a.map(r => r.adv)), 4)})</span>` : ""}</div>` : ""; };
      info.innerHTML = "";
      info.append(h("div", { class: "kv", style: { marginTop: "8px" }, html:
        `<div class="k">Baseline v</div><div class="v">${fmt.f(v, 4)}${Math.abs(v - st.spo.value_before) > 1e-6 ? ` <span class="sub">(logged ${fmt.f(st.spo.value_before, 4)})</span>` : ""} → after ${fmt.f(trackerSim().byStep[step] ? trackerSim().byStep[step].va : st.spo.value_after, 4)}</div>` +
        `<div class="k">Break-even reward r*</div><div class="v">ln(v)/β = <b>${fmt.f(rstar, 4)}</b> (${PROB.metric} ${fmt.f(1 / rstar, 4)}) — programs worse than this get A &lt; 0</div>` +
        mk("Failed programs (r=0)", r => !r.valid) + mk("Valid programs", r => r.valid) + mk("Valid, beat parent", r => r.improved) +
        `<div class="k">Tracker (logged)</div><div class="v">N_eff ${fmt.n(st.spo.effective_count_before)} → ${fmt.n(st.spo.effective_count_after)} · η=${fmt.f(st.spo.eta, 3)} · ρ=${st.spo.rho} · D=${fmt.f(st.spo.divergence, 2)} nats</div>` }));
      if (advIsWhatIf()) info.append(h("div", { class: "what-if", style: { display: "inline-block", marginTop: "6px" } }, "what-if values - the Sankey above, the Rollouts tab and the SPO² tab use the same numbers"));
      info.append(h("p", { class: "sub", style: { marginTop: "8px" } }, `Because r is ≈0 (fail) or ≈${PROB.validReward.toFixed(2)} (valid), advantages are almost binary. The baseline v of every step is re-simulated from the tracker parameters in the SPO² tab.`));
    };
    draw(); onParam(draw);
  },

  async train(box, step) {
    const st = ST[step];
    if (!st.policy_updates || !st.policy_updates.length) { box.append(h("div", { class: "empty" }, "Training metrics are not available for this step yet.")); return; }
    const u = st.policy_updates[0];
    box.append(h("div", { class: "kv", html:
      `<div class="k">Policy loss</div><div class="v">${fmt.f(u.loss, 5)}</div>` +
      `<div class="k">Mean token ratio π/π_old</div><div class="v">${fmt.f(u.ratio, 6)} (max ${fmt.f(u.ratio_max, 4)})</div>` +
      `<div class="k">Clipped tokens</div><div class="v">${fmt.pct(u.clipped_fraction, 3)}</div>` +
      `<div class="k">Gradient norm</div><div class="v">${fmt.sci(u.grad_norm, 2)} (clip at 1.0 never reached)</div>` +
      `<div class="k">Training time</div><div class="v">${fmt.dur(st.training_seconds)} on ${st.training_parallel_gpus} GPUs</div>` +
      `<div class="k">Examples / GPU</div><div class="v">${(st.fast_trained_examples_per_gpu || []).join(" + ")}</div>` +
      `<div class="k">Peak GPU memory</div><div class="v">${(st.fast_peak_memory_fraction || []).map(x => fmt.pct(x, 0)).join(" / ")}</div>` +
      `<div class="k">OOM-quarantined</div><div class="v">${st.oom_quarantined_examples || 0}</div>` }));
    box.append(h("h3", { style: { marginTop: "12px" } }, "Clipped objective (shared with the SPO² tab)"));
    box.append(paramSlider("ε_low", 0.05, 0.5, 0.01, x => x.toFixed(2), () => P.clipLo, (v, row) => setParam({ clipLo: v }, row)));
    box.append(paramSlider("ε_high", 0.05, 0.6, 0.01, x => x.toFixed(2), () => P.clipHi, (v, row) => setParam({ clipHi: v }, row)));
    const host = h("div", { class: "chart" }); box.append(host);
    const draw = () => {
      const lo = P.clipLo, hi = P.clipHi, f = (rho, A) => Math.min(rho * A, clamp(rho, 1 - lo, 1 + hi) * A);
      lineChart(host, { height: 190, xLabel: "ratio ρ", yLabel: "surrogate", margin: { l: 46 }, xDomain: [0.5, 1.7], xFmt: v => v.toFixed(1), yFmt: v => v.toFixed(1),
        series: [{ name: "A=+1", color: S_COL(0), pts: range(0, 61).map(i => [0.5 + i * 0.02, f(0.5 + i * 0.02, 1)]), dots: false }, { name: "A=−1", color: S_COL(1), pts: range(0, 61).map(i => [0.5 + i * 0.02, f(0.5 + i * 0.02, -1)]), dots: false }],
        bands: [{ x0: 1 - lo, x1: 1 + hi, color: cssVar("--accent-soft") }], tipHead: x => "ρ = " + x.toFixed(2) });
    };
    draw(); onParam(p => { if ("clipLo" in p || "clipHi" in p) draw(); });
  },

  async archive(box, step) {
    const st = ST[step], rs = RS[step] || [];
    const host = h("div", { class: "chart" }); box.append(host);
    lineChart(host, {
      height: 200, xLabel: "step", yLabel: "archive states", xTicks: STEPS, xFmt: v => String(Math.round(v)), margin: { l: 52 }, yFmt: v => v.toFixed(0),
      series: [{ name: "archive size", color: S_COL(3), pts: STEPS.map(s => [s, ST[s].archive_size]), fmt: v => String(v) }],
      hlines: [{ y: CFG.max_buffer_size || 2000, label: "cap " + (CFG.max_buffer_size || 2000), color: cssVar("--crit") }], includeH: true,
    });
    const elig = rs.filter(r => r.arch).length;
    box.append(h("div", { class: "kv", style: { marginTop: "10px" }, html:
      `<div class="k">Eligible this step</div><div class="v">${elig} valid programs with code</div>` +
      `<div class="k">Archive after step</div><div class="v">${fmt.n(st.archive_size)} states (cap ${CFG.max_buffer_size})</div>` +
      `<div class="k">Best of step</div><div class="v">${fmt.raw(st.step_best_raw_score)}${st.best_seen_step === step ? " " + '<span class="pill good">new global best</span>' : ""}</div>` +
      `<div class="k">Best so far</div><div class="v">${fmt.raw(st.best_seen_raw_score)} (found in step ${st.best_seen_step})</div>` }));
    box.append(h("p", { class: "sub", style: { marginTop: "8px" } }, "Only ≈63 of ~210 valid programs per step survive dedup + top-2 per (parent, strategy) + top-8 per parent. See the Search tab for how these states become tomorrow's parents."));
  },
};

function tipHtmlAlloc(a) {
  return `<b>parent ${a.parent_group} · S${a.strategy_index}</b><div>pilot valid ${a.pilot_valid_count}/${a.pilot_count} · improved ${a.hurdle ? a.hurdle.improvement_count : "–"}</div>` +
    `<div>mean reward ${fmt.f(a.pilot_reward_mean, 4)}</div><div>posterior P(improve) ${fmt.f(a.hurdle ? a.hurdle.posterior_improvement_probability : NaN, 3)}</div>` +
    `<div>expected improvement ${fmt.sci(a.initial_expected_improvement, 2)}</div><div><b>follow-ups ${a.followup_count}</b> → ${a.allocated_programs} programs</div>`;
}


let _lastHi = -1;
function highlightStageByProgress(f) {
  const idx = f >= 1 ? -1 : Math.min(STAGES.length - 1, Math.floor(f * STAGES.length));
  if (idx === _lastHi) return;
  _lastHi = idx;
  $$(".stage").forEach((e, i) => e.classList.toggle("playing", i === idx));
  $$(".arrow").forEach((e, i) => e.classList.toggle("playing", i === idx - 1 && idx > 0));
}

/* columns / links of the programs flow for one step (counts from the real rollouts) */
function flowData(step) {
  const rs = RS[step] || [], strat = (D.strat && D.strat.strategies) || {};
  const adv = advIsWhatIf() ? advWhatIf : r => r.adv;
  const plan = r => { const sl = strat[`${step}:${r.pg}:${r.sidx}`]; if (!sl) return "first"; return sl.fallback ? "fallback" : sl.attempts.length > 1 ? "retry" : "first"; };
  const phase = r => r.isRetry ? "retry" : r.phase === "adaptive" ? "adaptive" : "pilot";
  const result = r => !r.valid ? (r.fail === "timeout" ? "timeout" : "fail") : r.improved ? "beat" : "valid";
  const signal = r => { const a = adv(r); return a == null ? null : a < 0 ? "neg" : "pos"; };
  const NODES = {
    plan: { first: ["Plan accepted 1st try", cssVar("--good")], retry: ["Plan after retries", cssVar("--warn")], fallback: ["No plan (fallback)", cssVar("--crit")] },
    phase: { pilot: ["Pilot (11 / strategy)", S_COL(0)], adaptive: ["Phase 2 (hurdle)", S_COL(3)], retry: ["Format retry", S_COL(5)] },
    result: { fail: ["Failed: no / broken code", S_COL(1)], timeout: ["Timeout", S_COL(3)], valid: ["Valid, not better", cssVar("--ink-3")], beat: ["Valid, beats parent", cssVar("--good")] },
    signal: { neg: ["A < 0: discouraged", cssVar("--crit")], pos: ["A > 0: reinforced", cssVar("--good")] },
  };
  const cols = [
    { title: "Parent", nodes: range(0, 8).map(g => ({ id: "p" + g, label: "parent " + g, color: S_COL(g) })) },
    { title: "Strategist plan", nodes: Object.entries(NODES.plan).map(([k, [l, c]]) => ({ id: "plan:" + k, label: l, color: c })) },
    { title: "Budget phase", nodes: Object.entries(NODES.phase).map(([k, [l, c]]) => ({ id: "phase:" + k, label: l, color: c })) },
    { title: "Result", nodes: Object.entries(NODES.result).map(([k, [l, c]]) => ({ id: "res:" + k, label: l, color: c })) },
    { title: "Training signal" + (advIsWhatIf() ? " (what-if)" : ""), nodes: Object.entries(NODES.signal).map(([k, [l, c]]) => ({ id: "sig:" + k, label: l, color: c })) },
  ];
  const agg = {};
  const add = (s, t, v, color) => { const k = s + ">" + t; (agg[k] = agg[k] || { s, t, v: 0, color }).v += v; };
  rs.forEach(r => {
    const a = "p" + r.pg, b = "plan:" + plan(r), c = "phase:" + phase(r), d = "res:" + result(r), sg = signal(r);
    add(a, b, 1, S_COL(r.pg));
    add(b, c, 1, NODES.plan[plan(r)][1]);
    add(c, d, 1, NODES.phase[phase(r)][1]);
    if (sg) add(d, "sig:" + sg, 1, NODES.result[result(r)][1]);
  });
  const nameOf = id => { for (const c of cols) { const n = c.nodes.find(x => x.id === id); if (n) return n.label; } return id; };
  const totals = {};
  Object.values(agg).forEach(l => { totals[l.s] = (totals[l.s] || 0) + l.v; });
  return { cols, links: Object.values(agg), tip: l => `<b>${esc(nameOf(l.s))} → ${esc(nameOf(l.t))}</b><div>${fmt.n(l.v)} programs (${fmt.pct(l.v / totals[l.s], 0)} of the source)</div>` };
}
