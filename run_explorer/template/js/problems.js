"use strict";
/* =====================================================================================
   problems.js - everything that depends on WHICH problem the run optimises.
   A profile says how to turn a saved height sequence into the picture and the number of that problem,
   which words to use, and at which scale gaps are measured. Add a profile here to support another problem.
   ===================================================================================== */
let PROB = null;           // active profile (set by setupProblem)
let RUN = {};              // facts about the run itself (does it use JEV, sampler, seeds, reward range ...)

/* full linear convolution of two sequences: direct for short inputs, FFT for long ones */
function convolve(a, b) {
  const n = a.length, m = b.length, out = new Float64Array(n + m - 1);
  if (n * m <= 3e6) {
    for (let i = 0; i < n; i++) { const ai = a[i]; if (!ai) continue; for (let j = 0; j < m; j++) out[i + j] += ai * b[j]; }
    return out;
  }
  let N = 1; while (N < n + m - 1) N <<= 1;
  const ar = new Float64Array(N), ai = new Float64Array(N), br = new Float64Array(N), bi = new Float64Array(N);
  for (let i = 0; i < n; i++) ar[i] = a[i];
  for (let i = 0; i < m; i++) br[i] = b[i];
  fftInPlace(ar, ai, false); fftInPlace(br, bi, false);
  for (let i = 0; i < N; i++) { const r = ar[i] * br[i] - ai[i] * bi[i], q = ar[i] * bi[i] + ai[i] * br[i]; ar[i] = r; ai[i] = q; }
  fftInPlace(ar, ai, true);
  for (let i = 0; i < out.length; i++) out[i] = ar[i] / N;
  return out;
}

const PROFILES = {
  /* ------------------------------------------------------------------ AC1 -------------------- */
  ac1: {
    id: "ac1", short: "AC1", long: "first autocorrelation inequality", metric: "upper bound", dec: 6, K: 1e-4, profileYmax: 1.6, fYmax: null,
    arrName: "height_sequence_1", entry: "propose_candidate", file: "problems/ac_inequalities.py",
    intro: "an LLM-guided search for step-function constructions that minimise the <b>AC1</b> autocorrelation upper bound",
    recipeProblem: "minimise the AC1 upper bound",
    fLabel: "position x in [-1/4, 1/4] (shifted)  ·  step function f(x)", gLabel: "position in [-1/2, 1/2] (shifted)  ·  autoconvolution (f*f)(x)",
    gShort: "autoconvolution f*f", fShort: "step function f",
    variants(vals) {        // equivalent forms: reflecting x -> -x leaves the autoconvolution peak unchanged
      const a = Array.from(vals);
      return [{ name: "as published", vals: a }, { name: "mirrored (x → −x)", vals: a.slice().reverse() }];
    },
    prep(vals) {
      const a = Array.from(vals, x => Math.max(0, x)), n = a.length; let s = 0; for (const x of a) s += x;
      const c = autoConv(a), k = 2 * n / (s * s);
      const g = new Float64Array(c.length); let mx = 0;
      for (let i = 0; i < c.length; i++) { g[i] = c[i] * k; if (g[i] > mx) mx = g[i]; }
      const f = a.map(x => x * 2 * n / s);               // density: integral over the length-1/2 support = 1
      return { n, f, g, mx };
    },
    solIntro: "AC1 asks for a non-negative step function <i>f</i> on an interval of length ½ whose autoconvolution <i>f*f</i> has the smallest possible peak (relative to (∫f)²). A program returns the step heights <code>a₀…a_{n−1}</code>; the score is <code>2·n·max(a*a)/(Σa)²</code> - exactly the highest point of the lower curve below (after normalising ∫f = 1).",
    solNote: "The bound is a peak-to-mean statement: a perfectly flat autoconvolution would give the lowest possible value for the given support. The search is polishing the shape of f so that the plateau of f*f gets flatter and its highest point drops; late records lower that peak by ~1e-4 or less.",
    overlayNote: "Peaks are what matter - the curve tops out at the upper bound. Early constructions overshoot the plateau; later ones squeeze all peaks down to a nearly flat line just under the reference.",
    evalEq: `raw    = 2 · n · max(convolve(a, a, "full")) / (Σ a)²      a_i ← clip(a_i, 0, 1000),  Σa ≥ 0.01
reward = 1 / raw          (failure ⇒ 0)     seeds: raw = 2.0 ⇒ reward 0.5`,
    evalBody: `<p>Each program runs in its own subprocess (pinned to one CPU, BLAS threads = 1, memory-limited, hard timeout 1900 s). It must return a non-empty finite list; the evaluator computes the AC1 upper bound.</p>
      <p><b>Reward = 1 / raw upper bound</b> (higher is better); every failure gets reward 0. Failure kinds: <code>code</code> (no code block / unclosed fence / runtime error / bad return), <code>constraint</code> (invalid score), <code>timeout</code>.</p>
      <p>The target upper bound only appears in the prompt; it does not enter the reward.</p>`,
    evalRefs: ["problems/ac_inequalities.py:71-94 (raw score), :376-409 (score), :405-406 (reward = 1/raw)", "evaluation/sandbox.py:211", "train_multy_CVaR.py:1126 (sandbox timeout)"],
    glossRaw: "2·n·max(a*a)/(Σa)² - the quantity to minimise.",
    sizeHint: "n",
  },
  /* ------------------------------------------------------------------ Erdős ------------------ */
  erdos: {
    id: "erdos", short: "Erdős", long: "Erdős minimum-overlap problem", metric: "C₅ bound", dec: 8, K: 1e-5, profileYmax: 0.43, fYmax: 1.05,
    arrName: "initial_h_values", entry: "run", file: "problems/erdos.py",
    intro: "an LLM-guided search for step functions <b>h : [0, 2] → [0, 1]</b> that minimise Erdős' minimum-overlap constant <b>C₅</b>",
    recipeProblem: "minimise the Erdős overlap C₅",
    fLabel: "position x in [0, 2]  ·  step function h(x) ∈ [0, 1]", gLabel: "shift s in [−2, 2]  ·  overlap profile C(s) = ∫ h(x)(1 − h(x+s)) dx",
    gShort: "overlap profile C(s)", fShort: "step function h",
    canon(vals) {           // standard orientation used by most published figures: h close to 0 at both edges
      const n = vals.length; let s = 0; for (const x of vals) s += clamp(x, 0, 1);
      const h = Array.from(vals, x => clamp(x, 0, 1) * (n / 2) / (s || 1));
      const k = Math.max(1, Math.round(n * 0.06)); let e = 0; for (let i = 0; i < k; i++) e += h[i] + h[n - 1 - i];
      return e / (2 * k) > 0.5 ? { vals: h.map(x => 1 - x), name: "complement 1 − h" } : { vals: h, name: "" };
    },
    variants(vals) {        // h and 1 - h (both have mean 1/2) and the mirror image x -> 2 - x all give exactly the same C5 bound
      const n = vals.length; let s = 0; for (const x of vals) s += clamp(x, 0, 1);
      const h = Array.from(vals, x => clamp(x, 0, 1) * (n / 2) / (s || 1)), c = h.map(x => 1 - x);
      return [{ name: "as published", vals: Array.from(vals) }, { name: "complement 1 − h", vals: c }, { name: "mirrored (x → 2 − x)", vals: h.slice().reverse() }, { name: "complement + mirrored", vals: c.slice().reverse() }];
    },
    prep(vals) {
      const n = vals.length, h = Float64Array.from(vals, x => clamp(x, 0, 1)); let s = 0; for (const x of h) s += x;
      const sc = (n / 2) / (s || 1); for (let i = 0; i < n; i++) h[i] *= sc;     // the evaluator rescales to sum n/2
      const pre = new Float64Array(n + 1); for (let i = 0; i < n; i++) pre[i + 1] = pre[i] + h[i];
      const rev = h.slice().reverse(), R = convolve(h, rev), dx = 2 / n;
      const g = new Float64Array(2 * n - 1); let mx = 0;
      for (let lag = -(n - 1); lag <= n - 1; lag++) {
        const i0 = Math.max(0, -lag), i1 = Math.min(n - 1, n - 1 - lag);
        const v = (pre[i1 + lag + 1] - pre[i0 + lag] - R[n - 1 - lag]) * dx;
        g[lag + n - 1] = v; if (v > mx) mx = v;
      }
      return { n, f: h, g, mx };
    },
    solIntro: "Erdős' minimum-overlap problem asks for a step function <i>h</i> on [0, 2] with 0 ≤ h ≤ 1 and ∫h = 1 whose <b>worst overlap with a shifted copy of its complement</b> is as small as possible: <code>C(h) = max_s ∫ h(x)(1 − h(x+s)) dx</code>. A program returns the heights <code>h₀…h_{n−1}</code>; the evaluator rescales them to Σh = n/2 and scores <code>dx·max_s Σ_i h_i(1 − h_{i+s})</code> with dx = 2/n - exactly the highest point of the lower curve below.",
    solNote: "The bound is the height of the highest peak of the overlap profile. The search flattens the profile so that many shifts are (almost) simultaneously maximal - an equioscillation structure - which is what pushes the maximum down. Good constructions are near-binary: most steps sit at 0 or 1.",
    overlayNote: "The curve tops out at the bound. Early constructions have one dominant peak; later ones flatten the top of the profile so many shifts share the maximum.",
    evalEq: `h ← h · (n/2)/Σh                           must satisfy 0 ≤ h ≤ 1 before and after the rescaling
raw    = dx · max_s Σ_i h_i (1 − h_{i+s}),   dx = 2/n      all 2n−1 lags of np.correlate(h, 1−h, "full")
reward = 1 / raw          (failure ⇒ 0)     seeds: raw ≈ 0.5 ⇒ reward ≈ 2.0`,
    evalBody: `<p>Each program runs in its own subprocess (pinned to one CPU, memory-limited, hard timeout). It must return the step heights <code>h</code> (a 1-D array of n values in [0, 1]); the evaluator rescales them to Σh = n/2, rejects NaN / out-of-range values, and computes the overlap objective over all shifts.</p>
      <p><b>Reward = 1 / raw bound</b> (higher is better); every failure gets reward 0. Failure kinds: <code>code</code> (no code block / unclosed fence / runtime error / bad return), <code>constraint</code> (h outside [0, 1], wrong length, mismatching score), <code>timeout</code>.</p>
      <p>The target (0.3808) only appears in the prompt; it does not enter the reward.</p>`,
    evalRefs: ["problems/erdos.py:12-61 (verify_c5_solution: bounds, rescaling to n/2, np.correlate objective)", "problems/erdos.py (ErdosMinOverlap: metric 'C₅ bound', saves_construction)", "evaluation/sandbox.py"],
    glossRaw: "dx·max_s Σ h_i(1 − h_{i+s}) - the overlap constant C₅ bound to minimise.",
    sizeHint: "n",
  },
};

function genericProfile(id) {
  return {
    id: id || "generic", short: (id || "problem").toUpperCase(), long: id || "this problem", metric: "score", dec: 6, K: 1e-4, profileYmax: null, fYmax: null, arrName: "the parent array", entry: "solve", file: "problems/<problem>.py",
    intro: "an LLM-guided program search on the <b>" + esc(id || "task") + "</b> problem", recipeProblem: "minimise the score",
    prep: null, solIntro: "No geometry is known for this problem, so only the numeric results are shown.", solNote: "", overlayNote: "",
    evalEq: "reward = 1 / raw   (failure ⇒ 0)", evalBody: "<p>Each program runs in a sandbox; the problem's evaluator turns it into a raw score. <b>Reward = 1 / raw</b>; failures get 0.</p>", evalRefs: ["problems/<problem>.py (score)"],
    glossRaw: "the task's raw score (lower is better).", sizeHint: "n",
  };
}

/* called by prepare() once D.main / D.pub, rows and the tree are available */
function setupProblem() {
  const id = String(CFG.problem || (D.pub && D.pub.problem) || "").toLowerCase();
  PROB = Object.assign({}, PROFILES[id] || genericProfile(id));
  const sm = ALL_STEPS.map(s => ST[s]).find(s => s && s.result_metric_name);
  if (sm && sm.result_metric_name) PROB.metric = sm.result_metric_name;
  PROB.score = vals => PROB.prep ? PROB.prep(vals).mx : NaN;
  // facts about the run
  const roots = (TREE.roots || []).map(i => TREE.nodes[i][1]).filter(x => x > 0);
  PROB.seedRaw = roots.length ? mean(roots) : NaN;
  PROB.seedRewards = roots.map(x => 1 / x);
  PROB.seedReward = roots.length ? mean(PROB.seedRewards) : 0.5;
  const rew = R.filter(r => r.valid && r.reward > 0).map(r => r.reward);
  const rmax = rew.length ? Math.max(...rew) : PROB.seedReward;
  PROB.rewardMax = niceCeil(Math.max(rmax, PROB.seedReward) * 1.08);
  PROB.validReward = rew.length ? median(rew) : rmax;
  RUN = {
    hasJev: ALL_STEPS.some(s => ST[s].jev && ST[s].jev.decisions && ST[s].jev.decisions.length),
    sampler: (ALL_STEPS.map(s => ST[s].sampler_type).find(Boolean) || "PUCTSampler"),
  };
  RUN.selection = RUN.hasJev ? "PUCT + JEV" : "PUCT";
}
function niceCeil(x) { const p = Math.pow(10, Math.floor(Math.log10(x))), f = x / p; return (f <= 1 ? 1 : f <= 1.5 ? 1.5 : f <= 2 ? 2 : f <= 3 ? 3 : f <= 5 ? 5 : f <= 7.5 ? 7.5 : 10) * p; }
/* tiny helper: fill {{TASK}} / {{ARR}} / {{METRIC}} placeholders of the explanatory texts */
function fillText(t) {
  return String(t || "").replace(/\{\{TASK\}\}/g, PROB.short + " task").replace(/\{\{ARR\}\}/g, PROB.arrName).replace(/\{\{METRIC\}\}/g, PROB.metric)
    .replace(/\{\{SHORT\}\}/g, PROB.short).replace(/\{\{ENTRY\}\}/g, PROB.entry).replace(/\{\{BUDGET\}\}/g, String(CFG.budget_s || "?")).replace(/\{\{PROBFILE\}\}/g, PROB.file)
    .replace(/\{\{SEEDDESC\}\}/g, "score ≈ " + (isNaN(PROB.seedRaw) ? "?" : PROB.seedRaw.toFixed(3))).replace(/\{\{VALIDR\}\}/g, PROB.validReward ? PROB.validReward.toFixed(2) : "?");
}
function constructionScore(vals) { return PROB.score(vals); }

/* Many different step functions describe the SAME construction (for Erdős: h, its complement 1 − h and their mirror images; for AC1: f and its mirror image).
   To compare two programs by eye, draw B in the equivalent form that is closest to A. */
function resampleMean(vals, m) {
  const n = vals.length, out = new Float64Array(m);
  for (let b = 0; b < m; b++) { const i0 = Math.floor(b * n / m), i1 = Math.max(i0 + 1, Math.floor((b + 1) * n / m)); let s = 0; for (let i = i0; i < i1; i++) s += vals[i]; out[b] = s / (i1 - i0); }
  return out;
}
function shapeVec(vals) {
  const r = resampleMean(vals, 240);
  if (PROB && PROB.id === "ac1") { let s = 0; for (const x of r) s += x; for (let i = 0; i < r.length; i++) r[i] = r[i] / (s || 1) * r.length; }   // AC1 is scale-free
  return r;
}
function alignVariant(refVals, vals, mirrorOnly) {
  if (!PROB || !PROB.variants) return { vals, name: "", changed: false };
  const ref = shapeVec(refVals); let best = null;
  PROB.variants(vals).forEach((v, idx) => {
    if (mirrorOnly && /complement/.test(v.name)) return;
    const x = shapeVec(v.vals); let d = 0; for (let i = 0; i < ref.length; i++) d += (x[i] - ref[i]) * (x[i] - ref[i]);
    if (!best || d < best.d) best = { vals: v.vals, name: v.name, changed: idx !== 0, d, idx };
  });
  return best;
}

/* How every construction is DRAWN: first in the standard orientation of the problem (Erdős: edges at 0, like most published figures),
   then mirrored if that matches this run's own best construction better. Scores are unchanged - these are equivalent forms. */
let _dispRef = null;
function displayRef() {
  if (_dispRef) return _dispRef;
  const b = D.heavy && D.heavy.best && D.heavy.best.filter(x => x.construction && x.construction.length).pop();
  if (!b) return null;
  return (_dispRef = PROB.canon ? PROB.canon(b.construction).vals : Array.from(b.construction));
}
function displayForm(vals) {
  if (!PROB || !PROB.variants) return { vals, name: "", changed: false };
  const c = PROB.canon ? PROB.canon(vals) : { vals: Array.from(vals), name: "" };
  const ref = displayRef();
  if (!ref) return { vals: c.vals, name: c.name, changed: !!c.name };
  const al = alignVariant(ref, c.vals, !!PROB.canon);
  const name = [c.name, al.changed ? al.name : ""].filter(Boolean).join(" + ");
  return { vals: al.vals, name, changed: !!name };
}
