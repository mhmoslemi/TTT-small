"use strict";
/* ---- Rollouts explorer ---------------------------------------------------------------------- */
let _rl = { step: "cur", group: "all", sidx: "all", phase: "all", status: "all", q: "", x: "ntok", y: "gap", page: 0, sortCol: 6, sortDir: 1, retries: "include" };

registerTab({
  id: "rollouts", label: "Rollouts explorer", ico: "▦",
  async render(root) {
    root.append(pageHead("Rollouts: every program the coder wrote",
      `Filter the ${fmt.n(R.length)} scored programs by step, parent, strategy, phase and outcome; plot any two quantities; click a dot or a row to read the program, the strategy it was given, its sandbox output, its construction and how its advantage was computed.`));
    const bar = h("div", { class: "card", style: { marginBottom: "14px" } });
    const mkSel = (label, key, opts) => {
      const s = h("select", null, opts.map(([v, l]) => h("option", { value: v, selected: String(_rl[key]) === String(v) }, l)));
      s.addEventListener("change", () => { _rl[key] = s.value; _rl.page = 0; renderTab(); });
      return h("label", { class: "ctl" }, label, s);
    };
    const q = h("input", { type: "search", placeholder: "node id / message…", value: _rl.q, style: { width: "180px" } });
    q.addEventListener("change", () => { _rl.q = q.value.trim(); _rl.page = 0; renderTab(); });
    bar.append(h("div", { class: "row-flex" },
      mkSel("Step", "step", [["cur", "selected (" + S.step + ")"], ["all", "all steps"]].concat(ALL_STEPS.filter(s => RS[s]).map(s => [String(s), "step " + s]))),
      mkSel("Parent", "group", [["all", "all"]].concat(range(0, 8).map(g => [String(g), "parent " + g]))),
      mkSel("Strategy", "sidx", [["all", "all"]].concat(range(0, 4).map(s => [String(s), "S" + s]))),
      mkSel("Phase", "phase", [["all", "all"], ["pilot", "pilot"], ["adaptive", "phase 2"]]),
      mkSel("Outcome", "status", [["all", "all"], ["valid", "valid"], ["improved", "beat parent"], ["invalid", "invalid"], ["timeout", "timeout"]]),
      mkSel("Retries", "retries", [["include", "include"], ["exclude", "exclude"], ["only", "only format retries"]]),
      q));
    root.append(bar);

    const curStep = _rl.step === "cur" ? S.step : _rl.step;
    let rows = R.filter(r => {
      if (_rl.step !== "all" && r.step !== +curStep) return false;
      if (_rl.group !== "all" && r.pg !== +_rl.group) return false;
      if (_rl.sidx !== "all" && r.sidx !== +_rl.sidx) return false;
      if (_rl.phase !== "all" && r.phase !== _rl.phase) return false;
      if (_rl.retries === "exclude" && r.isRetry) return false;
      if (_rl.retries === "only" && !r.isRetry) return false;
      const st = _rl.status;
      if (st === "valid" && !r.valid) return false;
      if (st === "improved" && !r.improved) return false;
      if (st === "invalid" && r.valid) return false;
      if (st === "timeout" && r.fail !== "timeout") return false;
      if (_rl.q && !((r.node || "").startsWith(_rl.q) && true) && !(r.msg || "").toLowerCase().includes(_rl.q.toLowerCase()) && !(r.parent || "").startsWith(_rl.q)) return false;
      return true;
    });
    const valid = rows.filter(r => r.valid);
    root.append(h("div", { class: "kpis" },
      kpi("Programs", fmt.n(rows.length), `${valid.length} valid (${fmt.pct(rows.length ? valid.length / rows.length : NaN)})`),
      kpi("Best " + PROB.metric, valid.length ? fmt.raw(Math.min(...valid.map(r => r.raw))) : "–", valid.length ? `median valid ${fmt.raw(median(valid.map(r => r.raw)))}` : ""),
      kpi("Beat their parent", fmt.n(rows.filter(r => r.improved).length), `${fmt.pct(rows.length ? rows.filter(r => r.improved).length / rows.length : NaN)} of programs`),
      kpi("Advantage (valid / failed)", `${fmt.f(mean(valid.map(r => advIsWhatIf() ? advWhatIf(r) : r.adv)), 3)} / ${fmt.f(mean(rows.filter(r => !r.valid).map(r => advIsWhatIf() ? advWhatIf(r) : r.adv)), 3)}`, advIsWhatIf() ? "<span class='what-if'>what-if β, v</span> (see SPO² tab)" : "logged SPO² advantages"),
      kpi("Mean response", fmt.tok(Math.round(mean(rows.map(r => r.ntok || 0)))) + " tok", `valid ${fmt.tok(Math.round(mean(valid.map(r => r.ntok || 0))))} · invalid ${fmt.tok(Math.round(mean(rows.filter(r => !r.valid).map(r => r.ntok || 0))))}`)));

    // ----- scatter
    const AX = {
      ntok: ["response tokens", r => r.ntok, v => (v / 1000).toFixed(0) + "k"], rollout: ["rollout # in group", r => r.rollout, v => String(Math.round(v))], step: ["step (jittered)", r => r.step + ((r.id * 2654435761 % 1000) / 1000 - 0.5) * 0.7, v => String(Math.round(v))],
      sidx: ["strategy (jittered)", r => r.sidx + ((r.id * 40503 % 1000) / 1000 - 0.5) * 0.6, v => "S" + Math.round(v)], n: ["sequence length n", r => r.n, v => String(Math.round(v))],
      gap: ["gap to reference (symlog)", r => r.valid && r.raw > 0 ? gapOf(r.raw) : null, gapFmt], reward: ["reward", r => r.reward, v => v.toFixed(2)], advw: ["advantage (what-if β, v)", r => advWhatIf(r), v => v.toFixed(2)], adv: ["advantage", r => r.adv, v => v.toFixed(2)],
    };
    const axSel = (key, opts) => { const s = h("select", null, opts.map(k => h("option", { value: k, selected: _rl[key] === k }, AX[k][0]))); s.addEventListener("change", () => { _rl[key] = s.value; renderTab(); }); return s; };
    const host = h("div", { class: "chart" });
    root.append(card("Scatter", "colour = strategy · ring = beats parent · click a dot to open the program",
      h("div", null, h("div", { class: "row-flex", style: { marginBottom: "8px" } }, h("span", { class: "sub" }, "x"), axSel("x", ["ntok", "rollout", "step", "sidx", "n", "reward", "adv", "advw"]), h("span", { class: "sub" }, "y"), axSel("y", ["gap", "reward", "adv", "advw", "ntok", "n"])), host,
        legend(range(0, 4).map(s => ({ color: S_COL(s), label: "S" + s }))))));
    const ax = AX[_rl.x], ay = AX[_rl.y];
    const pts = rows.map(r => ({ x: ax[1](r), y: ay[1](r), c: S_COL(r.sidx), a: 0.7, r: r.valid ? 3.2 : 2.4, ring: r.improved, id: r.id, rr: r })).filter(p => p.x != null && p.y != null);
    scatterChart(host, { height: 380, points: pts, xLabel: ax[0], yLabel: ay[0], margin: { l: 62 }, xFmt: ax[2], yFmt: ay[2], yTicks: _rl.y === "gap" ? gapTicks() : null, yDomain: _rl.y === "gap" && pts.length ? gapDomain(pts.map(p => p.y)) : null,
      tip: p => `<b>step ${p.rr.step} · parent ${p.rr.pg} · S${p.rr.sidx} · #${p.rr.rollout}</b><div>${p.rr.valid ? PROB.metric + " " + fmt.raw(p.rr.raw) : "<span style='opacity:.8'>invalid: " + esc(p.rr.fail) + "</span>"}</div><div class="mut">${p.rr.phase} · ${fmt.tok(p.rr.ntok)} tokens · A=${fmt.f(p.rr.adv, 3)}</div>`, onPoint: p => openRollout(p.id) });

    // ----- table
    const cols = [
      { label: "step", get: r => String(r.step), num: true, key: r => r.step },
      { label: "parent", get: r => String(r.pg), num: true, key: r => r.pg },
      { label: "S", get: r => h("span", null, h("span", { class: "dot", style: { background: S_COL(r.sidx) } }), String(r.sidx)), key: r => r.sidx },
      { label: "phase", get: r => r.phase === "adaptive" ? "phase 2" : "pilot" + (r.isRetry ? " (retry)" : ""), key: r => r.phase },
      { label: "#", get: r => String(r.rollout), num: true, key: r => r.rollout },
      { label: "result", get: r => r.valid ? (r.improved ? pill("beats parent", "good") : pill("valid")) : pill(r.fail || "failed", "bad"), key: r => r.valid ? (r.improved ? 0 : 1) : 2 },
      { label: PROB.metric, get: r => r.valid ? fmt.raw(r.raw) : "–", num: true, key: r => r.valid ? r.raw : 99 },
      { label: "reward", get: r => fmt.f(r.reward, 5), num: true, key: r => -r.reward },
      { label: "advantage", get: r => fmt.f(r.adv, 4), num: true, key: r => -(r.adv || 0) },
      { label: "A what-if", get: r => advIsWhatIf() ? fmt.f(advWhatIf(r), 4) : "·", num: true, key: r => -(advWhatIf(r) || 0), title: "advantage under the current what-if β / baseline (set in the SPO² or Pipeline tab)" },
      { label: "tokens", get: r => fmt.n(r.ntok), num: true, key: r => r.ntok || 0 },
      { label: "n", get: r => r.n || "–", num: true, key: r => r.n || 0 },
      { label: "node", get: r => h("span", { class: "mono" }, fmt.id(r.node)), key: r => r.node || "" },
    ];
    const sc = _rl.sortCol, sd = _rl.sortDir;
    rows = rows.slice().sort((a, b) => { const ka = cols[sc].key(a), kb = cols[sc].key(b); return (ka < kb ? -1 : ka > kb ? 1 : 0) * sd; });
    const per = 100, pages = Math.max(1, Math.ceil(rows.length / per)); _rl.page = Math.min(_rl.page, pages - 1);
    const pageRows = rows.slice(_rl.page * per, (_rl.page + 1) * per);
    const tb = table(cols, pageRows, { sortable: true, sortCol: sc, sortDir: sd, onSort: ci => { if (_rl.sortCol === ci) _rl.sortDir *= -1; else { _rl.sortCol = ci; _rl.sortDir = 1; } renderTab(); }, onRow: r => openRollout(r.id) });
    const pager = h("div", { class: "row-flex", style: { marginTop: "8px" } },
      h("button", { class: "btn sm", disabled: _rl.page === 0 ? "" : null, onclick: () => { _rl.page--; renderTab(); } }, "‹ prev"),
      h("span", { class: "sub" }, `page ${_rl.page + 1} / ${pages} · rows ${rows.length ? _rl.page * per + 1 : 0}–${Math.min(rows.length, (_rl.page + 1) * per)} of ${fmt.n(rows.length)}`),
      h("button", { class: "btn sm", disabled: _rl.page >= pages - 1 ? "" : null, onclick: () => { _rl.page++; renderTab(); } }, "next ›"));
    root.append(h("div", { style: { marginTop: "14px" } }, card("Programs", "click a column to sort, a row to open", h("div", null, h("div", { class: "scroll tall" }, tb), pager))));
  }
});
