"use strict";
/* ---- main.js: boot, navigation, theme ---------------------------------------------------- */
const TAB_GROUPS = [
  ["Understand", ["overview", "pipeline"]],
  ["Search & strategy", ["search", "strategist", "alloc"]],
  ["Learning", ["train", "entropy"]],
  ["Data", ["rollouts", "solution"]],
  ["Reference", ["published", "concepts", "logs"]],
];

function buildNav() {
  const nav = $("#tabs"); nav.innerHTML = "";
  TAB_GROUPS.forEach(([g, ids]) => {
    nav.append(h("div", { class: "tab-sep" }, g));
    ids.forEach(id => {
      const t = TABS.find(x => x.id === id); if (!t) return;
      nav.append(h("button", { class: "tab" + (S.tab === id ? " active" : ""), role: "tab", "data-tab": id, onclick: () => setState({ tab: id }) },
        h("span", { class: "ico" }, t.ico), t.label));
    });
  });
}

let _renderToken = 0;
async function renderTab() {
  const tok = ++_renderToken;
  const t = TABS.find(x => x.id === S.tab) || TABS[0];
  S.tab = t.id;
  $$(".tab").forEach(b => b.classList.toggle("active", b.dataset.tab === t.id));
  const g = $("#gStep"); if (g) g.value = S.step;
  clearCharts(); clearParamListeners(); tipHide(); animWindow();
  const root = h("div", { class: "page" });
  const main = $("#main"), prevY = window.scrollY, sameTab = S._lastTab === t.id;
  S._lastTab = t.id;
  main.innerHTML = ""; main.append(root);
  try {
    await t.render(root);
  } catch (e) {
    console.error(e);
    root.append(h("div", { class: "warnbox" }, "Could not render this tab: " + (e && e.message || e)));
  }
  if (tok !== _renderToken) return;
  window.scrollTo(0, sameTab ? prevY : 0);
}

function initControls() {
  const sel = $("#gStep");
  sel.innerHTML = "";
  ALL_STEPS.forEach(s => sel.append(h("option", { value: s }, "step " + s + stepSuffix(s))));
  sel.value = S.step;
  sel.addEventListener("change", () => setState({ step: +sel.value }));
  const move = d => { const i = ALL_STEPS.indexOf(S.step); const j = clamp(i + d, 0, ALL_STEPS.length - 1); setState({ step: ALL_STEPS[j] }); };
  const rs = $("#gRef"); rs.innerHTML = "";
  refList().forEach(r => rs.append(h("option", { value: r.id }, `${r.label} · ${+r.val.toFixed(PROB.dec + 4)}`)));
  if (!refList().some(r => r.id === P.ref)) P.ref = "together";
  rs.value = P.ref;
  rs.addEventListener("change", () => { P.ref = rs.value; try { localStorage.setItem("rx-ref-" + PROB.id, P.ref); } catch (e) { /* ignore */ } renderTab(); });
  $("#btnPlay").addEventListener("click", playToggle);
  $("#btnReset").addEventListener("click", resetParams);
  $("#btnPrev").addEventListener("click", () => move(-1));
  $("#btnNext").addEventListener("click", () => move(1));
  $("#btnTheme").addEventListener("click", () => {
    const cur = document.documentElement.getAttribute("data-theme") || (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
    const nxt = cur === "dark" ? "light" : "dark";
    document.documentElement.setAttribute("data-theme", nxt);
    try { localStorage.setItem("rx-theme", nxt); } catch (e) { /* ignore */ }
    try { if (window.parent !== window) window.parent.postMessage({ rxThemeSet: nxt }, "*"); } catch (e) { /* ignore */ }   // unified page: keep every explorer in sync
    renderTab();
  });
  $("#drawerClose").addEventListener("click", closeDrawer);
  document.addEventListener("keydown", e => {
    if (/INPUT|TEXTAREA|SELECT/.test((e.target.tagName || ""))) return;
    if (e.key === "Escape") closeDrawer();
    else if (e.key === " ") { e.preventDefault(); playToggle(); }
    else if (e.key === "ArrowLeft") move(-1);
    else if (e.key === "ArrowRight") move(1);
  });
  window.addEventListener("hashchange", () => { readHash(); renderTab(); });
  try { const th = localStorage.getItem("rx-theme"); if (th) document.documentElement.setAttribute("data-theme", th); } catch (e) { /* ignore */ }
}

async function boot() {
  try {
    if (window.parent !== window) document.documentElement.classList.add("embedded");
    D.main = await inflate("main");
    try { D.pub = await inflate("pub"); } catch (e) { D.pub = {}; }
    prepare();
    { const st = TABS.find(t => t.id === "search"), so = TABS.find(t => t.id === "solution"), pb = TABS.find(t => t.id === "published");
      if (st) st.label = RUN.hasJev ? "Search: PUCT + JEV" : "Search: PUCT";
      if (so) so.label = "Solution (" + PROB.short + ")";
      if (pb) pb.label = "Published SOTA"; }
    $("#runName").textContent = D.main.meta.run_name + "  ·  " + ((D.main.hardware || {}).label || "") + "  ·  generated " + D.main.meta.generated_at;
    window.addEventListener("message", e => { if (e.data && e.data.rxTheme) { document.documentElement.setAttribute("data-theme", e.data.rxTheme); renderTab(); } });
    S.step = STEPS.length ? STEPS[STEPS.length - 1] : (ALL_STEPS[0] || 0);
    readHash();
    if (!ST[S.step]) S.step = S.step in ST ? S.step : ALL_STEPS[ALL_STEPS.length - 1];
    initControls();
    updateResetBadge();
    buildNav();
    await renderTab();
    try { if (window.parent !== window) window.parent.postMessage({ rxReady: true, theme: document.documentElement.getAttribute("data-theme") || (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light") }, "*"); } catch (e) { /* ignore */ }
  } catch (e) {
    console.error(e);
    $("#main").innerHTML = `<div class="warnbox" style="margin:30px">Failed to load run data: ${esc(e && e.message || e)}</div>`;
  }
}
boot();
