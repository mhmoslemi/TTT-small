# Run audit: `erdos_Qwen3.8-27B_1006-0453`

## Bottom line

The run is **mechanically healthy through completed step 6**, and the search is genuinely improving the best construction. The adapter updates use all eight GPUs, exact rollout/reference log-probabilities are present for every example, no examples are OOM-quarantined, and all completed updates are checkpointed.

However, the run is **not healthy at the population-quality level**. After step 3, coder reliability degrades sharply: valid yield falls from roughly 65% to 55%, 57%, and then 34% at step 6. The dominant failure is `no_code_block`, which rises from 52 at step 0 to 218 at step 6. Inspection shows many of these are long responses that end midway through code without a completed fenced answer. The run still finds tiny improvements because 544 attempts are enough to expose a few strong outliers, but the policy as a whole is getting less reliable.

The bandit is operational and has produced real useful descendants, but its printed expected-gain model is badly miscalibrated. The strategist is definitely wired into the coder and causes measurable behavioral steering, but there is still no convincing evidence in this run that it consistently improves reward or validity.

Step 7 finished rollout generation and evaluation and provisionally found `0.380890805017`, but an external `KeyboardInterrupt` stopped its adapter update. It is therefore **not a completed or checkpointed step**: `result.txt` and the completed summaries end at step 6.

## Search progress

Lower C₅ is better.

| Step | Best raw C₅ | Improvement from prior step | Valid | Mean reward | Main failures |
|---:|---:|---:|---:|---:|---|
| 0 | 0.380985889328 | — | 340/544 (62.5%) | 1.559936924 | 142 code, 53 constraint, 9 timeout |
| 1 | 0.380937100281 | 4.8789e-5 | 354/544 (65.1%) | 1.694373287 | 156 code, 33 constraint, 1 timeout |
| 2 | 0.380910790668 | 2.6310e-5 | 358/544 (65.8%) | 1.711233051 | 154 code, 22 constraint, 10 timeout |
| 3 | 0.380901383934 | 9.4067e-6 | 354/544 (65.1%) | 1.680942798 | 168 code, 14 constraint, 8 timeout |
| 4 | 0.380893442546 | 7.9414e-6 | 301/544 (55.3%) | 1.289407254 | 225 code, 9 constraint, 9 timeout |
| 5 | 0.380892742879 | 6.9967e-7 | 310/544 (57.0%) | 1.371771746 | 220 code, 10 constraint, 4 timeout |
| 6 | 0.380890913075 | 1.8298e-6 | 184/544 (33.8%) | 0.872354763 | 324 code, 36 constraint |
| 7, provisional | 0.380890805017 | 1.0806e-7 | 206/544 (37.9%) | 0.978397972 | 289 code, 47 constraint, 2 timeout |

Across completed steps 0–6, the best raw score improved by `9.4976e-5`. The completed best is `1.4913e-5` above the configured `0.380876` target. The provisional step-7 candidate is `1.4805e-5` above it.

The gains are real but strongly diminishing. Steps 0–4 did almost all the movement; steps 5–7 collectively add only about `2.64e-6`.

## Training health

The completed adapter updates look correct:

- Every completed step reports exact vLLM scores for `544/544` rollout and `544/544` reference sequences, with zero HF fallbacks.
- Every step trains all 544 examples, distributed almost perfectly evenly: 66–70 examples per GPU.
- Peak training memory is 93.0–96.1% across the eight GPUs.
- `OOM-quarantined=0` for all seven completed updates.
- There are no CUDA OOM failures, NaNs, failed exact-scoring workers, or internal training tracebacks in steps 0–6.
- IS-ratio means stay essentially 1.0. Their maxima vary from 4.226 to 29.034, but there is no accompanying numerical failure or discarded update.
- The policy change is small and cumulative: average `logpi_theta - logpi_base` moves from `-7.87e-5` at step 0 to `-3.05e-4` at step 6.

Training time rises from 1,223 seconds at step 0 to 1,631 seconds at step 6, consistent with longer responses and heavier long-sequence handling. The response-token median rises from 15,220 at step 0 to 17,468 at step 6. The proactive long-sequence/checkpoint/offload path is doing its job; it is slow, but it is not silently dropping data.

The losses are small and can be positive or negative under the clipped policy objective; that sign alone is not an error. Advantages are finite in every saved example. At step 6 there are 183 positive and 361 negative advantages, which is consistent with the much lower valid yield rather than a missing gradient signal.

### The real training-side concern

The output distribution is degrading even though the update mechanics are sound:

| Step | `no_code_block` | Median tokens, valid | Median tokens, no-code |
|---:|---:|---:|---:|
| 0 | 52 | 14,764 | 18,151 |
| 1 | 81 | 15,452 | 17,382 |
| 2 | 70 | 14,330 | 18,072 |
| 3 | 91 | 14,806 | 17,719 |
| 4 | 127 | 16,036 | 18,100 |
| 5 | 133 | 16,052 | 17,410 |
| 6 | 218 | 16,516 | 18,426 |

A representative step-6 failure contains extensive reasoning and a large partial implementation, then ends inside `run` without closing the code fence. This is not an evaluator problem: it is a coder completion/format-reliability regression. It is also not being hidden—these responses correctly receive zero reward and negative advantages.

The evidence does not prove that LoRA training alone caused the regression, because parent contexts and sampled tasks also evolve. The temporal trend is nevertheless too large and monotonic to dismiss. The best-outlier score is improving while average usability is deteriorating.

## Bandit allocation audit

For completed steps 0–6, the bandit made 56 parent-level allocation decisions. Each parent used 44 pilot rollouts and 24 phase-2 rollouts.

Observed behavior:

- Phase 2 beat that parent's pilot best in 16/56 decisions (28.6%).
- The maximally allocated arm contained the best phase-2 response in 48/56 decisions (85.7%). This is directionally good, although the most-sampled arm also has more chances to win.
- The arm with the highest initial expected improvement received the largest allocation in 41/56 decisions (73.2%); later posterior updates explain the remainder.
- The bandit put all 24 follow-ups on one strategy in 20/56 decisions (35.7%).
- Pilot: 2,464 rollouts, 57.63% valid, mean reward 1.44666.
- Phase 2: 1,344 rollouts, 58.11% valid, mean reward 1.46828.

So phase 2 is not reducing average quality. Its validity is +0.48 percentage points and its mean reward is about 1.5% higher than the pilot pool.

### Did phase 2 produce actual gain?

Yes, but not on every step:

- The global winner came from phase 2 on steps 1 and 3; it came from the pilot on steps 0, 2, 4, 5, and 6.
- The step-1 phase-2 winner `b55532dc...` became the parent of the step-2 winner and an ancestor of the step-3 winner.
- A separate step-3 phase-2 node `46781474...` became the ancestor of every winner from steps 4 through 7.

That lineage is strong evidence that phase 2 added real search value. Without those adaptive nodes, both major improving branches in the current archive would be absent.

What this run cannot establish is that the **bandit allocation rule** is better than uniform or the old rule-based allocation. That requires an otherwise identical ablation. Only two of seven immediate step winners came from the 35.3% phase-2 share, while its important archival descendants show that immediate-winner rate is not the whole story.

### The bandit's expected-gain numbers are not trustworthy

Across steps 0–6:

- Sum of printed projected phase-2 gains: `15.0845` reward units.
- Sum of realized parent-best gains: `0.007365` reward units.
- Spearman rank correlation between projected and realized gain: only `0.115`.

The approximately 2,000x aggregate overprediction is not a display rounding issue. The implementation samples future valid rewards from an unbounded Gaussian posterior and clips only the lower end at the failure reward. In this near-saturated regime, tiny observed variance can generate unrealistically optimistic upper tails for the expected maximum. The allocation mechanism still functions as an exploration heuristic, but `projected phase-2 gain` should not be interpreted literally or used as an efficiency estimate.

## Does the strategy actually affect the coder?

Yes—the strategy is not ignored—but its effect is modest and its reward benefit is unproven.

### Wiring and behavioral evidence

Across the 3,808 completed-step rollouts:

- The exact selected strategy text is present in 3,808/3,808 coder prompts.
- 3,803 responses contain a Python fence somewhere in the raw response; 3,791 nonempty extracted code bodies are exact-text unique.
- 3,366/3,808 raw coder responses explicitly mention "strategy" in their reasoning.
- Strategy plans are substantial: 224 unique plans for 224 parent/strategy slots, median 605 words.
- A curated algorithm-term comparison finds 38.02% adherence to the assigned strategy's named techniques versus 34.54% adherence to sibling strategies, a +3.49 percentage-point lift.
- Sampled code-token cosine similarity is 0.6420 for code from the same strategy versus 0.6326 across different strategies under the same parent. The effect is small (`+0.0094`) but in the expected direction.

Qualitative inspection agrees with the statistics. Winning programs often carry broad ideas from their plans—multiresolution interpolation, FFT correlation, SLSQP, simulated annealing, coordinate/local search—but the coder frequently rejects or replaces incorrect details. For example, the step-4 winning plan proposes sparse filters and group LASSO, while the coder actually implements FFT correlation plus binary/local search. The step-5 plan proposes a "double-node network" and physics-inspired geometry, while the coder discards that and writes FFT/local-search code. The coder is using the plan as guidance, not blindly transcribing it, exactly as the prompt permits.

### Reward-level evidence

Using only the equal-size pilot samples, with parent effects removed, the fraction of within-parent reward variation associated with strategy identity is:

| Step | Strategy effect (eta-squared) | Permutation p |
|---:|---:|---:|
| 0 | 0.1168 | 0.018 |
| 1 | 0.0490 | 0.862 |
| 2 | 0.0733 | 0.388 |
| 3 | 0.0420 | 0.941 |
| 4 | 0.0743 | 0.375 |
| 5 | 0.0573 | 0.716 |
| 6 | 0.0579 | 0.699 |
| All | 0.0672 | 0.628 |

Only step 0 shows a detectable strategy-level reward effect. Across the full run, strategy identity explains about 6.7% of within-parent reward variation, but this is entirely compatible with sampling noise (`p=0.628`). Validity gives the same conclusion: overall eta-squared `0.0685`, permutation `p=0.560`.

Therefore:

- **The strategy changes what the coder attempts.**
- **The coder uses and critiques the plan rather than ignoring it.**
- **This run does not show that the strategist consistently improves reward or validity.**

A no-strategy control with the same parents, coder checkpoint, sampling, and rollout budget is still required for a causal claim.

### Strategy-output defects

The strategy pipeline has a few real quality-control holes:

- Two of 224 plans fell back to `No usable strategy was returned...` (step 0 group 1 strategy 0; step 2 group 5 strategy 3).
- Two accepted plans contain literal tokenizer artifacts such as `Ġ` and `Ċ` (step 2 group 4 strategy 1; step 5 group 5 strategy 3).
- One step-4 plan is only 15 words and ends with an ellipsis, yet was accepted as usable.

These are rare, but they show that checking only for a `<strategy>...</strategy>` block is insufficient. The corruption reached coder prompts and consumed rollout budget.

## Step 7 interruption

Step 7 completed all 544 rollouts, evaluations, exact log-prob scoring, and artifact saving. It provisionally found `0.380890805017`. At 22:37:07, while `finish_entropic_overlap()` was waiting for the training future, the process received `KeyboardInterrupt`. The subsequent tracebacks are cleanup paths also interrupted by the same signal; they are not evidence of a CUDA, vLLM, or algorithmic crash.

Because the optimizer update and checkpoint finalization did not complete:

- `result.txt` stops at step 6.
- There is no completed `step07.summary.json`.
- The step-7 adapter update must not be treated as applied.
- The safe durable state is completed step 6, even though step-7 rollout artifacts and its provisional best candidate are on disk.

## Final verdict

1. **Training implementation:** healthy and exact through step 6.
2. **Search:** genuinely improving, but now deep in diminishing returns.
3. **Coder population:** unhealthy downward trend in completion/runnability; this is the most important issue in the new data.
4. **Bandit:** functioning and demonstrably useful through archive lineage, but expected-gain calibration is severely wrong and superiority to simpler allocation is not yet established.
5. **Strategist:** definitely influences code, but the influence is modest and has no consistent measured reward benefit in this run.
6. **Latest state:** step 7 was externally interrupted during training, so only steps 0–6 are committed.

The most honest summary is: **the system is finding better needles while the haystack is getting worse**. The best-score curve is healthy; the overall policy-quality curve is not.
