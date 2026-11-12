"use strict";
/* ---- Logs & config tab ------------------------------------------------------------------------ */
let _lg = { tag: "all", step: "all", q: "" };

registerTab({
  id: "logs", label: "Config & event log", ico: "☰",
  async render(root) {
    const logs = await lazy("logs");
    root.append(pageHead("Configuration and event log",
      `The resolved configuration, the startup banner, and the terminal events of the run (RAM-wait and torch warnings filtered out). Timestamps carry no date; the log spans several resumes of the run.`));

    // config table
    const cfg = Object.entries(CFG).filter(([k, v]) => typeof v !== "object" || v === null);
    const cq = h("input", { type: "search", placeholder: "filter config keys…", style: { width: "220px" } });
    const cbody = h("div");
    const drawCfg = () => {
      const f = cq.value.toLowerCase();
      const rows = cfg.filter(([k, v]) => !f || k.toLowerCase().includes(f) || String(v).toLowerCase().includes(f));
      cbody.innerHTML = "";
      cbody.append(h("div", { class: "scroll" }, table([{ label: "key", get: r => h("span", { class: "mono" }, r[0]) }, { label: "value", get: r => h("span", { class: "mono" }, JSON.stringify(r[1])) }], rows)));
    };
    cq.addEventListener("input", drawCfg); drawCfg();
    const setting = D.main.setting || [];
    root.append(h("div", { class: "grid g2" },
      card("config.json", (RUN.hasJev ? "scalar settings · note: stale w.r.t. the JEV threshold (see Search tab)" : "scalar settings"), h("div", null, h("div", { style: { marginBottom: "8px" } }, cq), cbody)),
      card("Startup banner", "from setting.log", setting.length ? h("pre", { class: "code", style: { maxHeight: "560px" } }, setting.join("\n")) : h("div", { class: "empty" }, "no setting.log"))));

    // event log
    const ev = logs.events;
    const tags = uniq(ev.map(e => e.tag).filter(Boolean)).sort();
    const sel = (key, opts) => { const s = h("select", null, opts.map(([v, l]) => h("option", { value: v, selected: String(_lg[key]) === String(v) }, l))); s.addEventListener("change", () => { _lg[key] = s.value; draw(); }); return s; };
    const q = h("input", { type: "search", placeholder: "search text…", value: _lg.q, style: { width: "220px" } });
    q.addEventListener("input", () => { _lg.q = q.value; draw(); });
    const out = h("div"), cnt = h("span", { class: "sub" });
    const draw = () => {
      const f = _lg.q.toLowerCase();
      const rows = ev.filter(e => (_lg.tag === "all" || e.tag === _lg.tag) && (_lg.step === "all" || String(e.s) === String(_lg.step)) && (!f || e.txt.toLowerCase().includes(f)));
      cnt.textContent = `${fmt.n(rows.length)} of ${fmt.n(ev.length)} events`;
      out.innerHTML = "";
      const shown = rows.slice(0, 600);
      out.append(h("div", { class: "scroll tall" }, table([
        { label: "line", get: e => String(e.i + 1), num: true }, { label: "time", get: e => h("span", { class: "mono" }, e.t) },
        { label: "tag", get: e => e.tag ? pill(e.tag) : "" }, { label: "step", get: e => e.s == null ? "" : String(e.s), num: true },
        { label: "message", get: e => h("span", { style: { whiteSpace: "normal", color: "var(--ink-2)" } }, e.txt) },
      ], shown)));
      if (rows.length > shown.length) out.append(h("div", { class: "sub", style: { marginTop: "6px" } }, `showing the first ${shown.length}; narrow the filters to see more`));
    };
    root.append(h("div", { style: { marginTop: "14px" } }, card("Event log", null, h("div", null,
      h("div", { class: "row-flex", style: { marginBottom: "8px" } }, h("label", { class: "ctl" }, "tag", sel("tag", [["all", "all"]].concat(tags.map(t => [t, t])))), h("label", { class: "ctl" }, "step", sel("step", [["all", "all"]].concat(ALL_STEPS.map(s => [String(s), "step " + s])))), q, cnt), out))));
    draw();
  }
});
