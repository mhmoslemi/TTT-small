"use strict";
/* =====================================================================================
   knowledge.js - the written explanation of the pipeline, distilled from reading the code
   base (file:line references are to the repository the run was produced with).
   ===================================================================================== */
const STAGES = [
  {
    id: "select", n: 1, title: "Select parents", chip: ["search + GLiDE-9B", "mc-search", "PUCT ranking, JEV gate"],
    short: "8 parent programs",
    body: `
      <p>Every step starts by choosing <b>8 parent programs</b> (one per <i>group</i>) from the archive of everything the search has kept so far. The archive begins with 8 trivial <i>seed</i> programs ({{SEEDDESC}}).</p>
      <p><b>PUCT ranks</b> all archive states. <b>JEV</b> (the <code>PUCTJevSampler</code>) then walks down the top <b>3 × 8 = 24</b> states and rejects any candidate that a small judge model (<b>GLiDE-9B</b>) thinks is the <i>same core idea</i> as a parent already accepted in this step. JEV never changes a score - it is a pure diversity gate on PUCT's ranking.</p>
      <p>Step 0 is plain PUCT on the seeds (no comparisons possible). From step 1, accepted candidates cost 0+1+…+7 = 28 pairwise judgments when nothing is rejected.</p>`,
    eq: `score(s) = Q(s) + c · scale · P(s) · √(1+T) / (1+n(s))
Q(s)   = best reward among VALID children ever seen from s   (value(s) while n=0)
P(s)   = (N − rank(s)) / (N(N+1)/2)        rank-based prior, best state ⇒ 2/(N+1)
scale  = max(1e-6, max value − min value over non-seed states)
n(s)   = rollouts spent below s, T = total rollouts so far (= 512 · step)
JEV:   reject candidate if  P(same_core_idea | accepted parent) ≥ threshold   (0.9 in steps 0–3, 0.68 from step 4)`,
    refs: ["search/sampler.py:234-263 (PUCT score)", "search/sampler.py:274-308 (sample_states, lineage blocking)", "search/jev_sampler.py:198-287 (_diverse_picks)", "search/jev_worker.py:17-111 (GLiDE worker)", "train_multy_CVaR.py:1782 (selection), :1825 (record_expansion)"],
    files: ["stepNN.parents.json", "stepNN.jev.json"],
  },
  {
    id: "strategist", n: 2, title: "Strategist chains", chip: ["R1-Qwen3-8B · frozen", "mc-strat", "DeepSeek-R1-0528-Qwen3-8B, LoRA disabled"],
    short: "8 chains × 4 strategies",
    body: `
      <p>For each parent, a <b>reasoning model</b> writes a natural-language <b>implementation plan</b> (a <code>&lt;strategy&gt;</code> block with six sections: approach, math spec, procedure, search choices, compute budget, validation/return contract). No code is allowed; the coder only ever sees the extracted plan, never the strategist's thinking.</p>
      <p>The 4 strategies of a parent are generated <b>sequentially</b>: strategy <i>k</i> receives <code>&lt;previous_strategy_1..k&gt;</code> and is told to propose a <i>materially different</i> approach. The 8 parent chains run in parallel on two vLLM engines.</p>
      <p>Each slot gets up to <b>4 attempts</b> (<code>strategy_format_max_retries=3</code>). An attempt fails if there is no complete <code>&lt;strategy&gt;…&lt;/strategy&gt;</code> at the end of the response (usually the model ran out of tokens or never closed the tag). If all 4 fail, the slot gets the fallback text <i>"No usable strategy was returned. Derive the implementation directly…"</i> and the coder works without a plan.</p>
      <p>The strategist is <b>never trained</b>. Its quality only matters through the rewards of the coder programs it spawns (that is what the hurdle allocator measures).</p>`,
    eq: `chain(parent):  S0 → S1 | S0 → S2 | S0,S1 → S3 | S0,S1,S2
attempt a ∈ {0..3}; accepted iff a complete <strategy> block closes the response
fallback slot ⇒ coder prompt contains the fixed "No usable strategy…" sentence`,
    refs: ["train_multy_CVaR.py:2014-2075 (strategy prompt + retries + fallback)", "train_multy_CVaR.py:92-134 (_extract_final_strategy)", "problems/base.py:150-185 (previous-strategy blocks)", "generation/strategy_prompts.py:29-168 (LOCAL_STRATEGY_PROMPT)", "generation/output_retries.py:44-163"],
    files: ["stepNN_groupGG_fold00_strategySS.txt", "…_strategySS_attemptAA.{txt,prompt.txt,meta.json}"],
  },
  {
    id: "pilot", n: 3, title: "Pilot coding", chip: ["Qwen3.8-27B + LoRA", "mc-coder", "vLLM, reasoning effort low"],
    short: "11 programs / strategy",
    body: `
      <p>The <b>coder</b> receives the {{TASK}}, the parent's source and score, the <code>{{ARR}}</code> hint (the parent's actual array is injected at run time, never printed in the prompt) and the strategist's plan, and writes <b>one Python program</b> (<code>{{ENTRY}}</code>) with a {{BUDGET}} s search budget.</p>
      <p>Each of the 32 strategies gets an equal <b>pilot</b> of <b>11 programs</b> (= 352 per step). Each program is sent to a CPU sandbox the instant it is generated, so evaluation overlaps generation.</p>
      <p>Programs are sampled at temperature 1.0 with thinking on at "low" reasoning effort. Only the part after the last <code>&lt;/think&gt;</code> counts as the answer; a response without a closed <code>python</code> fence scores as a failure.</p>`,
    eq: `pilot programs per step = groups × strategies × pilot = 8 × 4 × 11 = 352`,
    refs: ["train_multy_CVaR.py:1850-1894 (coder prompt rendering)", "train_multy_CVaR.py:2237-2265 (submit rollout to sandbox)", "{{PROBFILE}} (build_prompt)", "problems/base.py:187-216 (strategy → coder prompt)"],
    files: ["stepNN_groupGG_rolloutRRR.{txt,prompt.txt,strategy.txt,meta.json}"],
  },
  {
    id: "alloc", n: 4, title: "Hurdle allocation", chip: ["hurdle bandit", "mc-search", "rollout_allocation.py, expected-best EI"],
    short: "+20 programs / parent",
    body: `
      <p>A parent's budget is 64 rollouts = 4 × 16. The pilots used 4 × 11 = 44, so <b>20 programs per parent are still unspent</b>. As soon as a parent's 44 pilots are scored, the <b>hurdle allocator</b> pools those 20 and gives them out greedily to the strategies most likely to <i>beat the parent</i> by the most.</p>
      <p>"Hurdle" = the parent's own reward. For every pilot, the gain is <code>Z = max(R − R_parent, 0)</code> (invalid programs count as 0). Each strategy is modelled as <b>P(improve) × size of improvement</b>; the 20 units are assigned one by one to the arm with the largest marginal <b>expected improvement</b> over the best pilot.</p>
      <p>In practice it is almost <b>winner-take-all</b>: one strategy usually gets all 20 (11 + 20 = 31 programs) while the other three keep their 11.</p>`,
    eq: `Z_a,j     = max(R − r_parent, 0)  for pilot j of strategy a   (0 if invalid)
p_a       ~ Beta(1 + s_a, 1 + n − s_a),  s_a = #pilots with valid ∧ R > r_parent   (logged: (s+1)/(n+2))
future outcome of arm a = Dirichlet mixture of an atom at 0 and reflected-Gaussian kernels
                          (pooled kernel + one per own positive pilot, Silverman bandwidth)
EI(m)     = E[ max(0, best of m_a new draws per arm − best pilot) ]
greedy:   for each of the 20 units → arm with largest marginal EI   (tie → fewer allocations)
fallback: if nobody beat the parent → 5 / 5 / 5 / 5`,
    refs: ["search/rollout_allocation.py:369-522 (allocate_hurdle_expected_best)", "train_multy_CVaR.py:2500-2620 (per-parent streaming phase 2)", "models/coder_profiles.py:290-350 (phase2_*_programs_per_strategy = reasoning-effort split, not an allocation knob)"],
    files: ["stepNN.summary.json → strategy_pilot_allocation[32]"],
  },
  {
    id: "eval", n: 5, title: "Sandbox evaluation", chip: ["CPU sandboxes", "mc-cpu", "isolated, 264 concurrent"],
    short: "512 scored programs",
    body: `
      <p>Each program runs in its own subprocess (pinned to one CPU, BLAS threads = 1, memory-limited, hard timeout 1900 s). It must return a non-empty finite list; the evaluator computes the AC1 upper bound.</p>
      <p><b>Reward = 1 / raw upper bound</b> (higher is better); every failure gets reward 0. Failure kinds: <code>code</code> (no code block / unclosed fence / runtime error / bad return), <code>constraint</code> (invalid score), <code>timeout</code>.</p>
      <p>The target upper bound 1.5027 only appears in the prompt; it does not enter the reward.</p>`,
    eq: `raw    = 2 · n · max(convolve(a, a, "full")) / (Σ a)²      a_i ← clip(a_i, 0, 1000),  Σa ≥ 0.01
reward = 1 / raw          (failure ⇒ 0)     seeds: raw = 2.0 ⇒ reward 0.5
target 1.5027  ⇔  reward ≈ 0.66547`,
    refs: ["problems/ac_inequalities.py:71-94 (raw score), :376-409 (score), :405-406 (reward = 1/raw)", "evaluation/sandbox.py:211", "train_multy_CVaR.py:1126 (sandbox timeout)"],
    files: ["…rolloutRRR.meta.json (reward, raw_score, valid, failure_kind, msg, sandbox_stdout)"],
  },
  {
    id: "adv", n: 6, title: "SPO² advantage", chip: ["SPO-RS · β=2", "mc-train", "single-stream, risk-sensitive"],
    short: "exp(β·r) vs baseline v",
    body: `
      <p>Instead of GRPO-style group normalisation, SPO-RS keeps <b>one run-wide scalar baseline <code>v</code></b> (a persistent "value" of the exponentiated reward) shared by all 512 rollouts of a step. The reward is first passed through an exponential <b>risk-sensitive transform</b> <code>G = exp(β·r)</code>, which emphasises the upper tail (objective ≈ E[R] + β/2·Var R).</p>
      <p>Advantages are <b>not centred inside a group</b>. Because almost every program is either a failure (r=0) or a valid program with r≈{{VALIDR}}, the learning signal is essentially <i>valid vs. invalid</i>: failures get a negative advantage, valid programs a positive one, and the difference between a 1.5050 and 1.5048 solution is tiny (≈0.006).</p>
      <p>After the advantages are formed the baseline moves toward the new batch: <code>v ← (ρ·N_eff·v + ΣG)/(ρ·N_eff + M)</code>. The forgetting factor ρ depends on how far the policy moved (divergence D); in this run D≈4–5 nats so ρ sits at its floor 0.875.</p>`,
    eq: `G_i   = exp(β r_i)                       β = spo_rs_beta = 2.0 (fixed)
A_i   = (G_i / v − 1) / β                v = baseline BEFORE this step (one value for all rollouts)
break-even reward   r* = ln(v) / β
D     = mean over contexts of Σ_tokens (logπ_cur − logπ_prev)   (clamped ≥ 0)
ρ     = clip(2^(−D / d_half), ρ_min, ρ_max) = clip(2^(−D/0.06), 0.875, 0.96)
N_c   = ρ·N_eff + M                      v' = (ρ·N_eff·v + Σ G_i) / N_c      η = M / N_c      N_eff' = N_c`,
    refs: ["training/spo_rs.py:57-82 (advantages), :136-196 (tracker update)", "train_multy_CVaR.py:4053-4083 (advantage assembly)", "train_multy_CVaR.py:5245-5273 (tracker init from seed rewards: v0 = e = 2.718, N_eff = 64)", "_retire/bandit_strategist_spo_rs_method.txt:580 (\"SPO-RS, called SPO² in the slides\")"],
    files: ["…rolloutRRR.meta.json (advantage, beta, spo_rs)", "stepNN.summary.json → spo_rs_updates"],
  },
  {
    id: "train", n: 7, title: "LoRA policy update", chip: ["LoRA r=32 · 2 GPUs", "mc-train", "1 AdamW step per training step"],
    short: "PPO-clip, lr 4e-5",
    body: `
      <p>Only the <b>coder</b> is trained (LoRA rank 32, α 32 on all 7 projection modules; 233 M trainable of 27.1 B parameters). Behaviour log-probs come from vLLM; the trainer recomputes current log-probs with exact long-context kernels (sequences up to 262 k tokens).</p>
      <p>The loss is a per-token PPO-clip surrogate with <b>asymmetric clipping [0.8, 1.38]</b>, averaged per trajectory so long and short responses weigh the same, then summed with weight 1/N over all trained trajectories. It is <b>one epoch and one AdamW step per training step</b>, so the policy ratio ≈ 1 and the clip is almost inert (clipped fraction ≈ 0.1%). The KL-to-reference term and gradient clipping are effectively off.</p>
      <p>Training of the previous step overlaps with CPU evaluation of the current rollouts. Step N samples with the adapter from step N−1.</p>`,
    eq: `ρ_t  = exp(logπ_cur,t − logπ_behavior,t)                      (per token)
L_i  = − mean_t min( ρ_t · A_i ,  clip(ρ_t, 1−0.2, 1+0.38) · A_i )
Loss = Σ_i L_i / N                                                 N = all trained trajectories (incl. format retries)
AdamW: lr 4e-5, β=(0.9, 0.95), wd 0 — kl_penalty_coef=0.1 has no effect for SPO-RS (kl_coef forced to 0)`,
    refs: ["training/trainers.py:988-1077 (loss), :1104-1110 (kl_coef = 0 for SPO-RS), :1564 (_train_spo_rs_examples)", "train_multy_CVaR.py:3674 (release_for_training), :3739 (begin_spo_rs_overlap), :4567 (finish)"],
    files: ["stepNN.summary.json → policy_updates, training_seconds, fast_*"],
  },
  {
    id: "archive", n: 8, title: "Archive update", chip: ["sampler.update", "mc-search", "dedupe, top-r, top-k"],
    short: "+~63 states / step",
    body: `
      <p>Only <b>valid programs with code</b> are submitted to the archive. The sampler (1) updates each parent's <i>Q</i> (best valid child reward) before any pruning, (2) drops children whose <b>exact source</b> is already archived, (3) keeps the top <b>2</b> per (parent, strategy) — at most 8 per parent per step — and (4) keeps the top <b>8</b> children per direct parent over the whole run. Seeds are never evicted. The cap <code>max_buffer_size=2000</code> is not binding yet (≈ +64 states per step ⇒ reached near step 31).</p>
      <p><code>record_expansion(parent, 64)</code> adds 64 visits to the parent and all its ancestors; these visit counts <i>n</i> shrink PUCT's exploration bonus for heavily-used lineages.</p>
      <p>The "best so far" curve is tracked separately: <code>best_seen</code> uses every valid child (even ones later dedup-ed or pruned) and never regresses.</p>`,
    eq: `archive_size(step k) ≈ 8 seeds + 63·(k+1)        (71, 133, 197, … in this run)
Q(parent) ← max(Q, best valid child reward)       before dedupe
child.parents = [parent] + parent.parents          (nearest-first ancestor chain)`,
    refs: ["search/sampler.py:310-441 (update)", "search/sampler.py:443-449 (record_expansion)", "train_multy_CVaR.py:4159-4166 (what gets submitted)"],
    files: ["stepNN.summary.json → archive_size", "stepNN.best.json", "search_tree.json"],
  },
];

const GOTCHAS = [
  ["config.json is a stale snapshot", "jev", "It has no JEV threshold. The real value is stored per step in <code>stepNN.jev.json</code>: <b>0.9 for steps 0–3, 0.68 from step 4</b> (the run was resumed from a step-3 copy and the threshold changed; step 4's selection shows 0.68 → 0.9 → 0.68 across resumes)."],
  ["\"puct-jev\" is not a blended score", "jev", "PUCT ranks, JEV only vetoes. No JEV probability enters any score. The window is the top 3×8 = 24 PUCT states."],
  ["phase2_low_programs_per_strategy = 20 is not the allocation", "It is a reasoning-effort split (all 20 follow-ups are 'low' effort). The 20 follow-ups per <i>parent</i> come from (16 − 11) × 4."],
  ["kl_penalty_coef and grad_clip are inert", "For SPO-RS the KL coefficient is forced to 0 (\"reference: off\"), and logged gradient norms are ~0.001–0.002, so the 1.0 clip never triggers."],
  ["Only the coder is trained", "The strategist (DeepSeek-R1-0528-Qwen3-8B) runs with LoRA disabled. The DeepSeek API settings in the config are unused (<code>strategy_backend: local</code>)."],
  ["Advantages are not group-centred", "One run-wide baseline <i>v</i> serves all 512 rollouts, so learning is dominated by valid-vs-invalid."],
  ["Step 0 has more rollouts than planned", "retry0", "Extra coder 'format-retry' rollouts were generated because fewer than 50% of the planned ones had a complete code block. They count for training/archive but not for allocation."],
  ["rolloutRRR is arrival order", "The number in a rollout file name is not the strategy order. Group by <code>strategy_index</code> and <code>strategy_rollout_phase</code> (pilot / adaptive)."],
  ["'fold00' is a legacy tag", "It is hard-coded in the file name layout and always reads fold00."],
  ["A sizeable share of strategy slots fall back", "", "After 4 failed attempts the coder gets a generic sentence instead of a plan; most failures are 'missing complete final strategy' (the model never closed the tag)."],
  ["search_tree.json lags one step and includes failures", "It is rebuilt post-hoc; every rollout is a node (a large share have score 0.0 = failed programs)."],
  ["llm_sampler.py is not used here", "It is an alternative <code>parent_selection=llm</code> mode where an LLM picks parents from the tree JSON."],
  ["Test expectation looks stale", "jev", "<code>tests/test_parent_picker_contract.py</code> asserts a same-core threshold of 0.80 while the code default is 0.68."],
];

const GLOSSARY = [
  ["Group", "One parent slot g ∈ [0,8) in a step: a parent program plus the 64 rollouts spawned from it (4 strategies × 16)."],
  ["Strategy / chain", "A natural-language plan from the strategist. 4 per parent, generated sequentially; later ones see earlier ones."],
  ["Attempt", "One strategist generation call. A failed format check triggers a retry (max 4 per slot)."],
  ["Pilot (phase 1)", "The first 11 coder programs for every strategy."],
  ["Phase 2 (adaptive)", "The 20 follow-up programs per parent, allocated by the hurdle bandit after that parent's pilots are scored. Called 'adaptive' in the files."],
  ["Hurdle", "The parent's own reward - a program only counts as an improvement if it beats it."],
  ["Seed", "One of the 8 initial constant-sequence programs (upper bound 2.0, reward 0.5)."],
  ["Archive", "The set of valid, de-duplicated states the sampler can pick parents from (capped at 2000)."],
  ["PUCT", "Predictor + UCT: exploitation (best child reward Q) plus an exploration bonus scaled by a rank prior and shrunk by visit counts."],
  ["JEV", "The GLiDE-9B semantic diversity gate in <code>jev_sampler.py</code>. Judges Program A vs B as same_core_idea / meaningful_variant / different_approach. (The code never expands the acronym.)"],
  ["SPO² / SPO-RS", "Single-stream policy optimisation with a Risk-Sensitive exponential reward transform and a persistent baseline."],
  ["v, N_eff, η, ρ", "Baseline value, its effective sample count, its per-step update rate (M/N_c) and its forgetting factor."],
  ["Valid", "Program parsed, ran, returned a finite non-empty list and scored a finite positive bound. It does not need to beat the parent."],
  ["Upper bound (raw)", "2·n·max(a*a)/(Σa)² - the quantity to minimise (target 1.5027)."],
  ["Reward", "1 / raw (0 for failures); higher is better."],
  ["Entropy (nats)", "Mean full-vocabulary token entropy of the coder's policy over all response tokens, measured in the pre-update training forward pass."],
];

/* GOTCHAS rows are [title, flag, body] where flag is "" (always), "jev" (only runs with the JEV gate) or "retry0" (only if step 0 has retries) */
function gotchas() {
  const out = GOTCHAS.map(g => g.length === 3 ? g : [g[0], "", g[1]]);
  const f = out.filter(g => g[1] === "" || (g[1] === "jev" && RUN.hasJev) || (g[1] === "retry0" && (RS[ALL_STEPS[0]] || []).length > (CFG.groups_per_step || 8) * (CFG.group_size || 64)));
  if (!RUN.hasJev) f.unshift(["This run has no JEV diversity gate", "", "It was produced by an earlier version of the pipeline: parents are chosen by the plain <code>PUCTSampler</code> only (rank-based prior, best-child Q, lineage blocking). The GLiDE-9B judge, the 3×8 candidate window and the same-core threshold of the newer runs do not exist here."]);
  return f;
}
function glossary() {
  return GLOSSARY.filter(g => RUN.hasJev || g[0] !== "JEV").map(g => g[0] === "Seed" ? ["Seed", "One of the 8 initial trivial programs (" + fillText("{{SEEDDESC}}") + ", reward ≈ " + (PROB.seedReward ? PROB.seedReward.toFixed(2) : "?") + ")."]
    : g[0] === "Upper bound (raw)" ? ["Raw score (" + PROB.metric + ")", PROB.glossRaw] : g[0] === "Reward" ? ["Reward", "1 / raw (0 for failures); higher is better."] : g);
}
/* the per-stage text depends on the problem and on whether the run used the JEV gate */
function stageView(sg) {
  let v = Object.assign({}, sg);
  if (sg.id === "select" && !RUN.hasJev) {
    v.chip = ["PUCT sampler", "mc-search", "rank-based prior, lineage blocking"];
    v.body = `
      <p>Every step starts by choosing <b>8 parent programs</b> (one per <i>group</i>) from the archive of everything the search has kept so far. The archive begins with 8 trivial <i>seed</i> programs ({{SEEDDESC}}).</p>
      <p><b>PUCT ranks</b> all archive states. The sampler walks down the ranking, takes a state and then <b>blocks its whole lineage</b> (the state, all its ancestors and all its descendants - siblings and cousins stay allowed) so the 8 parents sit on different branches of the tree. If blocking leaves fewer than 8, the rest are topped up in score order.</p>
      <p><i>This run used the plain <code>PUCTSampler</code>: there is no semantic diversity gate (JEV / GLiDE-9B) in this earlier version of the pipeline.</i></p>`;
    v.eq = `score(s) = Q(s) + c · scale · P(s) · √(1+T) / (1+n(s))
Q(s)   = best reward among VALID children ever seen from s   (value(s) while n=0)
P(s)   = (N − rank(s)) / (N(N+1)/2)        rank-based prior, best state ⇒ 2/(N+1)
scale  = max(1e-6, max value − min value over non-seed states)
n(s)   = rollouts spent below s, T = total rollouts so far (= 512 · step)
pick:  best-scoring state not in the lineage of an already picked parent`;
    v.refs = ["search/sampler.py:234-263 (PUCT score)", "search/sampler.py:274-308 (sample_states, lineage blocking)", "train_multy_CVaR.py:1782 (selection), :1825 (record_expansion)"];
    v.files = ["stepNN.parents.json"];
  }
  if (sg.id === "eval") { v.body = PROB.evalBody; v.eq = PROB.evalEq; v.refs = PROB.evalRefs; }
  v.body = fillText(v.body); v.eq = fillText(v.eq); v.refs = (v.refs || []).map(fillText);
  return v;
}
