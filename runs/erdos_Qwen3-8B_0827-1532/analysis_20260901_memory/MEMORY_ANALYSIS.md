# Memory effect in `erdos_Qwen3-8B_0827-1532`

## Comparison used

Steps 1--33 contain 264 same-parent comparisons. For each parent, 38 rollouts used the selected lesson, 13 used no memory, and 13 used an exploratory lesson. Outcome rates can therefore be compared within the same search state. Maximum quality is compared at equal budget using best-of-13 on both sides. This is strong within-run evidence, but it is still one run rather than a replicated memory-on versus memory-off experiment.

## Main findings

- **Memory clearly improves feasibility.** Selected memory raised scientific validity from 42.0% to 65.5%: **+23.5 percentage points** (matched bootstrap 95% interval: +21.1 to +25.9). The effect was positive in 32 of 33 steps.

- **It also improves code robustness.** Code validity rose from 74.2% to 81.8%: **+7.6 points** (95% interval: +5.7 to +9.5). The effect was positive in 31 of 33 steps.

- **More rollouts improve on their parent.** The rate rose from 17.6% to 32.7%: **+15.1 points** (95% interval: +12.8 to +17.4).

- **The average maximum-quality benefit is not reliable.** With both arms restricted to best-of-13, selected memory had mean reward uplift +0.028, but its 95% interval was -0.009 to +0.070. Across the 264 comparisons it won 103, tied 50, and lost 111.

- **Memory changes role over time.** In steps 26--33 it still raised scientific validity by 11.9 points, but its beats-parent gain fell to 1.1 points. In equal-budget tail comparisons it won only 2/64, tied 22/64, and lost 40/64. Mean tail uplift was negative in every one of these eight steps.

- **Exploratory memory mattered through rare events.** Its average matched tail effect was not positive, yet it generated the two large record improvements at steps 2 and 3. Descriptively, selected and exploratory memories each account for about half of the post-step-0 raw improvement; this is not a fair rate comparison because the selected arm had three times as many attempts.

- **The selector became concentrated on safe advice.** One lesson, “Use SLSQP with equality constraints,” received 219/264 selections; another received 45/264; the other seven lessons were never selected by retrieval. This explains the late pattern: memory continued to prevent invalid programs but stopped redirecting search toward a new scientific basin.

## Verdict

Memory is working as a feasibility and repair mechanism, and memory exploration contributed important early discoveries. It is not yet working as a reliable late-stage max-seeking mechanism. The next design should maintain separate recent credit for (1) validity repair and (2) equal-budget tail improvement, discount old success, demote lessons after repeated recent tail losses, and permanently reserve an exploratory-memory fraction.

## Figure captions

**Figure 1 -- Overall memory effect.** Selected memory substantially improves scientific validity, code validity, and the chance of beating the parent. However, the confidence interval for equal-budget best-of-13 reward crosses zero, so the run does not establish a reliable average improvement in the maximum.

**Figure 2 -- Memory effect over time.** The feasibility benefit persists throughout training, while the max-seeking benefit disappears. During steps 26--33, selected memory still improves validity but loses most equal-budget tail comparisons.

**Figure 3 -- Origin of record improvements.** Selected memory produces many incremental improvements, while exploratory memory produces two unusually large early jumps. The attribution is descriptive because the selected arm received more attempts.
