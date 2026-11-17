"use strict";
/* ---- Solution tab for GPU-kernel problems: the solution is a program, so we show runtimes, references and source ------------ */
const KERNEL_FEATURES = [
  ["Triton kernels", /@triton\.jit/g, "number of @triton.jit functions"],
  ["tl.dot", /\btl\.dot\b/g, "tensor-core matmul inside a Triton kernel"],
  ["autotune", /@triton\.autotune/g, "Triton configuration search at run time"],
  ["torch matmul / bmm / einsum", /\btorch\.(?:matmul|bmm|mm|einsum)\b|\.matmul\(|\beinsum\(/g, "the big contraction delegated to cuBLAS"],
  ["fp16 / bf16", /\.half\(\)|float16|bfloat16|\.to\(torch\.(?:float16|bfloat16)\)/g, "reduced-precision activations"],
  ["layer norm", /layer_norm|LayerNorm/g, ""],
  ["torch.compile", /torch\.compile/g, ""],
  ["CUDA graph / stream", /CUDAGraph|cuda\.graph|torch\.cuda\.Stream/g, ""],
  ["inline CUDA", /load_inline|cpp_extension/g, ""],
];
function kernelFeatures(code) { return KERNEL_FEATURES.map(([, re]) => (String(code || "").match(re) || []).length); }

async function kernelSolution(root, heavy, pub) {
  const snaps = (heavy.best || []).filter(b => b.code);
  const refs = ((pub && pub.results) || []).filter(pubVisible).filter(r => r.reported != null).sort((a, b) => a.reported - b.reported);
  root.append(pageHead("The kernel the search found",
    PROB.solIntro + " Nothing is recomputed in the browser here - a kernel's runtime only exists on the GPU it was timed on - so every number below is the one logged by the run or printed by the published source."));
  if (!snaps.length) { root.append(h("div", { class: "hint" }, "No best-so-far kernel snapshot is saved yet (stepNN.best.json is written when a step finishes).")); return; }

  const last = snaps[snaps.length - 1], first = snaps[0], best = last.raw;
  const valid = R.filter(r => r.valid), failed = R.filter(r => !r.valid && !r.isRetry);
  const constraint = R.filter(r => r.fail === "constraint").length;
  const lead = refs[0], sel = REF();
  const ratio = (v, w) => (v / w >= 10 ? (v / w).toFixed(1) : (v / w).toFixed(2)) + "×";
  root.append(h("div", { class: "kpis" },
    kpi("Best kernel", `${fmt.raw(best)} <span style="font-size:14px;color:var(--ink-3)">µs</span>`, `found in step ${last.found_step} · geometric mean over the benchmark shapes`),
    kpi("Since the first step", ratio(first.raw, best) + " faster", `${fmt.raw(first.raw)} µs after step ${first.step} → ${fmt.raw(best)} µs`),
    lead ? kpi("vs fastest visible reference", ratio(best, lead.reported) + " slower", `${esc(lead.label)} · ${fmt.rep(lead.reported)} µs`) : null,
    sel && sel.id !== (lead && lead.id) ? kpi("vs selected reference", (best <= sel.val ? ratio(sel.val, best) + " faster" : ratio(best, sel.val) + " slower"), `${esc(sel.label)} · ${fmt.rep(sel.val)} µs (top-bar selector)`) : null,
    kpi("Kernels that passed", fmt.pct(valid.length / Math.max(1, R.length)), `${fmt.n(valid.length)} of ${fmt.n(R.length)} programs · ${fmt.n(constraint)} built but failed the correctness tests`)));

  /* ---------- best-so-far curve with the published references ---------- */
  const hostC = h("div", { class: "chart", id: "kBest" });
  const seen = BSTEPS.map(s => [s, ST[s].best_seen_raw_score]).filter(p => p[1] != null);
  const hl = refs.slice(0, 6).map((r, i) => ({ y: r.reported, label: `${r.label.replace(/ \(.*\)/, "")} ${fmt.rep(r.reported)}`, color: r.id === sel.id ? cssVar("--good-ink") : cssVar("--ink-3"), dash: r.id === sel.id ? "5 4" : "2 4", anchor: i % 2 ? "end" : "start" }));
  root.append(card("Best runtime per step vs the references", `runtime in µs (lower is better) · dashed lines are the published results currently shown (top bar: <b>Results</b>); the selected reference is green`, hostC));
  lineChart(hostC, { height: 330, xLabel: "step", yLabel: "best runtime so far (µs)", margin: { l: 70 }, xTicks: ALL_STEPS, xFmt: v => String(Math.round(v)), includeH: true, hlines: hl,
    series: [{ name: "best so far", color: S_COL(0), pts: seen, width: 2.4 }, { name: "best of this step", color: S_COL(1), pts: BSTEPS.map(s => [s, ST[s].step_best_raw_score]), width: 0, dotR: 5 }],
    tipHead: x => `<b>step ${x}</b>` + (ST[x].provisional ? " (update pending)" : ""), tipVal: v => fmt.raw(v) + " µs",
    onClick: x => { setState({ tab: "pipeline", step: x }); } });

  /* ---------- every valid kernel ---------- */
  const hostS = h("div", { class: "chart" }), gl = valid.map(r => Math.log10(r.raw)).filter(isFinite);
  const lo = Math.min(...gl), hi = Math.max(...gl), ticks = [];
  for (let e = Math.floor(lo) - 1; e <= Math.ceil(hi); e++) [1, 2, 3, 5].forEach(m => { const v = Math.log10(m * Math.pow(10, e)); if (v >= lo - 0.05 && v <= hi + 0.05) ticks.push(v); });
  const g0 = Math.log10(first.raw), g1 = Math.log10(best);
  root.append(h("div", { style: { marginTop: "14px" } }, card("Every kernel that passed the tests", "one dot per valid program, runtime on a log scale; click a dot to open the program · the line at the bottom is the best kernel so far",
    h("div", null, hostS, h("div", { class: "sub", style: { marginTop: "4px" } }, "Most kernels are far slower than the best of their step: the search keeps the occasional fast one, and the coder is updated towards the programs whose runtime beat the others in their group.")))));
  scatterChart(hostS, { height: 320, points: valid.filter(r => r.raw > 0).map(r => ({ x: r.step + ((r.id * 2654435761 % 1000) / 1000 - 0.5) * 0.7, y: Math.log10(r.raw), c: seqRamp(1 - clamp((Math.log10(r.raw) - g1) / Math.max(1e-9, Math.log10(first.raw * 3) - g1), 0, 1)), a: 0.6, r: 2.6, rr: r })),
    xLabel: "step", yLabel: "runtime µs (log scale)", margin: { l: 62 }, xTicks: ALL_STEPS, xFmt: v => String(Math.round(v)), yTicks: ticks, yFmt: v => String(Math.round(Math.pow(10, v))),
    tip: p => `<b>step ${p.rr.step}</b><div>${fmt.raw(p.rr.raw)} µs</div><div class="mut">${p.rr.improved ? "faster than its parent" : "not faster than its parent"}</div>`, onPoint: p => openRollout(p.rr.id) });

  /* ---------- best kernel per step: runtime + what the code contains ---------- */
  const feat = snaps.map(s => kernelFeatures(s.code));
  const cols = [
    { label: "step", get: (s, i) => String(s.step), num: true },
    { label: "best so far (µs)", get: s => h("b", null, fmt.raw(s.raw)), num: true },
    { label: "Δ vs previous", get: (s, i) => i ? fmt.f(s.raw - snaps[i - 1].raw, 1) + " (" + fmt.pct((s.raw - snaps[i - 1].raw) / snaps[i - 1].raw, 1) + ")" : "–", num: true },
    { label: "found in step", get: s => String(s.found_step), num: true },
    { label: "lines", get: s => String(s.code.split("\n").length), num: true },
  ].concat(KERNEL_FEATURES.slice(0, 6).map(([lab, , tip], k) => ({ label: lab, title: tip, get: (s, i) => feat[i][k] ? h("span", { style: { color: "var(--good-ink)", fontWeight: 600 } }, feat[i][k] > 1 ? "✓ ×" + feat[i][k] : "✓") : h("span", { style: { color: "var(--ink-3)" } }, "–") })));
  root.append(h("div", { style: { marginTop: "14px" } }, card("Best kernel after each step", "runtime and what the source contains (found by text search in the code - a hint, not a profile) · click a row to read that kernel below",
    h("div", { class: "scroll", style: { maxHeight: "none" } }, table(cols.map(c => Object.assign({}, c, { get: r => c.get(r, snaps.indexOf(r)) })), snaps, { onRow: s => { _kern.step = s.step; renderTab(); } })))));

  /* ---------- source of one snapshot, full or as a diff against the previous best ---------- */
  if (!snaps.some(s => s.step === _kern.step)) _kern.step = last.step;
  const idx = snaps.findIndex(s => s.step === _kern.step), cur = snaps[idx], prev = snaps[idx - 1];
  const selStep = h("select", null); snaps.forEach(s => selStep.append(h("option", { value: s.step }, `best through step ${s.step} · ${fmt.raw(s.raw)} µs`)));
  selStep.value = String(cur.step); selStep.addEventListener("change", () => { _kern.step = +selStep.value; renderTab(); });
  const mode = prev ? _kern.mode : "full";
  const body = mode === "diff" && prev ? diffBlock(prev.code, cur.code) : codeBlock(cur.code);
  root.append(h("div", { style: { marginTop: "14px" } }, card(`Kernel source - best through step ${cur.step}`, "as written by the coder; the diff shows what changed against the previous best kernel",
    h("div", null, h("div", { class: "row-flex", style: { marginBottom: "8px" } }, selStep, prev ? seg([{ value: "full", label: "full source" }, { value: "diff", label: "diff vs step " + prev.step }], mode, v => { _kern.mode = v; renderTab(); }) : null,
      h("span", { class: "sub" }, `${cur.code.split("\n").length} lines · found in step ${cur.found_step}`)), body))));
}
const _kern = { step: null, mode: "full" };
