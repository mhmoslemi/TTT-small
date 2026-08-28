# Forensic effectiveness analysis: `erdos_Qwen3-8B_0827-1532`

## Bottom line

The method is **partially working, but this run does not validate it as a robust max-seeking discovery system**.

- The search found a strong basin quickly and improved it for many steps.
- Selected memory has strong within-run causal evidence for making programs executable, valid, and more likely to beat their parent.
- The memory portfolio's exploration arm produced two of the largest early record jumps, so it did contribute genuine extreme-value discoveries.
- Neither selected nor exploratory memory shows a reliable continuing tail advantage. Selected memory is tail-negative in every late step, even while it remains validity-positive.
- Feedback eligibility, adaptive activation, and caps operated as designed. Its causal benefit is not identifiable because there is no simultaneous feedback-off control.
- PUCT did not behave like a sequential tree bandit. It became a one-shot rolling beam, collapsed to one seed lineage by step 6, and never selected a previously visited node.
- Late max-seeking GRPO lost its scientific ranking signal: at step 33, 208 different valid programs all produced one construction and one reward. GRPO could only learn valid versus invalid.

The most defensible claim from this run is:

> Memory improves feasibility and helped some early discoveries, but the combined system eventually closes an exploitation loop around one local basin. Feedback may improve code robustness, while PUCT and GRPO do not preserve enough semantic diversity to keep seeking new maxima.

## Evidence boundary

The configured run has 50 steps, but only steps 0–33 completed. `step34` contains a parent-selection file and no rollout or summary artifacts. The completed evidence therefore contains 34 steps and 17,408 primary rollouts.

This is one non-deterministic combined-treatment run. It can diagnose behavior and use memory's randomized null arm for causal comparisons, but it cannot estimate the total method effect against a plain run. The copied run also omits training logs, token-level feedback statistics, adapter checkpoints, and trustworthy wall-clock timing.

Derived data and plots:

- [Search dynamics](run_dynamics.svg)
- [Memory effects](memory_effects.svg)
- [Tree efficiency](tree_efficiency.svg)
- [Per-step metrics](step_metrics.csv)
- [Matched memory contrasts](memory_group_contrasts.csv)
- [Feedback signatures](feedback_signatures.csv)
- [Lineage metrics](lineage_metrics.csv)
- [Machine-readable summary](metrics.json)

## 1. Did the search improve the result?

Yes, substantially at first, but it did not reach the requested threshold and it stalled.

| Quantity | Result |
|---|---:|
| Best initial seed | 0.4896288712 |
| Best rollout at step 0 | 0.3825711346 |
| Best final result | **0.3809593361** |
| Configured target | 0.3808760000 |
| Gap above target | **8.3336e-5** |
| Stated record in prompt | 0.3809200000 |
| Gap above stated record | **3.9336e-5** |
| Target or record hits | 0 |

The best saved construction was independently rescored with the same full-correlation definition. The recomputed result is `0.38095933606372034`, differing from the saved value by `1.1e-16`, so the result is genuine rather than a reporting or verifier error.

The time profile is strongly front-loaded:

- 98.47% of the step-0-to-final improvement was already achieved by step 9.
- The practically final value, within `1e-12`, appeared at step 28.
- After step 22, another 5,632 rollouts improved the incumbent by only `1.256e-7` in total.
- Steps 31–33 reported zero distinct children beating their parent.

Overall outcomes were:

| Outcome | Count | Fraction |
|---|---:|---:|
| Scientifically valid | 9,674 | 55.57% |
| Code failure | 3,765 | 21.63% |
| Scientific constraint failure | 2,972 | 17.07% |
| Timeout | 997 | 5.73% |

Validity itself was non-monotone. It rose from 37.30% at step 0 to a peak of 76.17% at step 25, then fell to 46.29% at step 32 and ended at 51.56%.

## 2. Is memory working?

### 2.1 The experimental mechanism is working

Step 0 had an empty bank. In every later step, each parent kept the same total budget of 64 rollouts:

| Arm | Rollouts per parent | Total, steps 1–33 | Share |
|---|---:|---:|---:|
| Selected lesson | 38 | 10,032 | 59.4% |
| No-memory control | 13 | 3,432 | 20.3% |
| UCB exploratory lesson | 13 | 3,432 | 20.3% |

This produced 264 same-parent matched comparisons for each treatment arm. The memory intervention did not increase the total rollout budget.

### 2.2 Selected memory causally improves feasibility

The same-parent comparisons show a large and consistent selected-memory benefit:

| Outcome | No memory | Selected memory | Matched effect |
|---|---:|---:|---:|
| Scientifically valid | 41.96% | 65.50% | **+23.54 pp** |
| Code-valid | 74.18% | 81.79% | **+7.60 pp** |
| Logged `reward > parent` | 22.44% | 38.58% | **+16.14 pp** |

Exploratory step-cluster bootstrap intervals remain clearly positive for the selected arm: approximately +19 to +28 points for validity, +5 to +10 points for code validity, and +12 to +20 points for the logger's parent-improvement test. The exploratory arm is approximately equal to control on these average feasibility measures.

One lesson explains much of the benefit. `e29ed4f8`, “Use SLSQP with equality constraints,” was selected in 219 of 264 groups. The recurring “COBYLA cannot handle equality constraints” error occurred in 4.43% of selected-memory rollouts versus 11.71% of matched controls.

### 2.3 Selected memory does not show robust max-tail improvement

The appropriate max-seeking diagnostic is the matched best-of-13 reward uplift. It compares treatment and control at the same effective sample size.

| Arm | Mean uplift | Median | Wins / ties / losses |
|---|---:|---:|---:|
| Selected lesson | +0.02765 | 0 | 103 / 50 / 111 |
| Exploratory lesson | -0.03523 | 0 | 65 / 120 / 79 |

The selected-arm mean looks positive, but its bootstrap interval crosses zero. It is driven by a handful of comparisons where memory produced at least one valid program and the control produced none. Restricting to the 259 comparisons where both arms had a valid result gives:

- mean uplift: **-0.00159**;
- median uplift: 0;
- 99 wins, 50 ties, and 110 losses.

The late evidence is worse. Across steps 26–33, selected memory had only 2 tail wins, 22 ties, and 40 losses, with mean uplift **-0.00915**. Every one of those eight steps had negative mean selected-memory uplift. Yet it still increased validity by 11.88 points. Memory was therefore making the model safer inside the existing basin rather than finding a better basin.

### 2.4 The exploratory arm mattered early despite poor average uplift

Average tail uplift is not the entire max-seeking story. Assigning each meaningful new incumbent to its winning arm gives the following descriptive decomposition of the post-step-0 improvement:

| Winning arm | Share of total raw improvement |
|---|---:|
| Selected lesson | 48.9% |
| Exploratory lesson | 48.6% |
| No-memory control | 2.5% |

The exploratory share is dominated by two rare large jumps at steps 2 and 3. This attribution is not a causal arm comparison—the selected arm had 38 attempts and the other arms had 13—but it shows that under-tested memories did produce exactly the rare discoveries a max objective values. The exploration arm should be preserved; its selection and credit need to become non-stationary and novelty-aware.

### 2.5 The memory bank becomes stale and concentrated

The final bank has 9 lessons:

- 76 lessons proposed, 9 added, and 67 deduplicated;
- 7 of 9 lessons never selected by the retrieval model;
- all 264 selected-arm choices went to only two lessons: 219 to `e29ed4f8` and 45 to `14e87b21`;
- all lessons did receive exploration trials, so “never selected” does not mean “never tested”;
- zero curation events occurred.

The critical failure is not lack of outcome logging. The logger recorded 528 matched updates. The failure is that early cumulative rescue wins continue to dominate allocation after their recent tail value becomes negative. Lifetime credit is masking a change in regime.

## 3. Is feedback working?

### 3.1 The implementation behavior matches the intended design

Only code failures were eligible. Constraint failures and all 997 timeouts were excluded, as requested.

For `G=8, K=64`, the automatic caps resolved to 103 teacher forwards per step and 26 per failure signature. Feedback activated only when code validity fell below the adaptive threshold:

`0–7, 26–28`

It performed 1,079 teacher forwards over 1,763 eligible failures in active steps, covering 61.2% of them and 153 normalized failure signatures. No active step exceeded its cap.

### 3.2 The effectiveness evidence is encouraging but not causal

The temporal pattern is consistent with repair:

- code validity fell as low as 45.31% at step 3;
- it reached 81.45% at step 8, causing feedback to switch off;
- it averaged 82.86% during steps 8–25;
- it relapsed to a 77.47% average during steps 26–28, reactivating feedback;
- it averaged 83.52% in steps 29–33 and ended at 89.06%.

This cannot be attributed to feedback alone. Feedback activates precisely after low-validity observations, so regression to the mean is expected; GRPO and memory also update simultaneously; and selected memory itself raises code validity by 7.60 points.

The artifacts record which failures received teacher forwards, but not the token-level feedback advantages, clipping rates, or evaluated corrected programs. Therefore the run proves that the controller and budget operated, not that the feedback gradient caused the recovery.

### 3.3 Persistent failures and the new late bottleneck

The most common code error, “equality constraints not handled by COBYLA,” occurred 1,224 times, in every completed step. Only 200 occurrences received teacher treatment, and it still appeared 18 times in the final step. Feedback did not eliminate this recurrent mode.

Late failure shifted toward runtime rather than coding. Of 997 total timeouts, 703 occurred in steps 28–33. This explains how code validity could end at 89.06% while scientific validity ended at only 51.56%. Excluding timeouts from textual feedback is behaving as intended, but a separate execution-budget controller is now necessary; otherwise feedback correctly turns itself off while runtime waste grows.

## 4. Is max-seeking GRPO working?

It works when rollouts have meaningful quality differences. Late in the run, its input signal degenerates.

- Entropic beta saturated near `1.05e6` in all eight groups during steps 20–26, indicating that the optimizer was trying to amplify extremely small reward differences.
- At step 33 there were only two reward values: zero for 248 invalid rollouts and `2.624952...` for all 264 valid rollouts.
- Exactly the 264 valid rollouts had positive scalar reward advantage; the 248 invalid rollouts had negative advantage.
- All 264 valid rollouts at step 33 produced one construction and one raw value, despite containing 208 distinct valid program texts.

Thus the late GRPO update was max-seeking only in a formal sense. There was no remaining scientific ranking signal, so it optimized “return the accepted local construction” versus “fail.” A larger beta cannot recover information absent from the reward samples.

The current `distinct_good` diagnostic also overstates progress near floating-point equality. Step 30 reports 184 distinct good programs, but only one child improves its parent by more than `1e-14`, none improves by `1e-7`, and the largest raw improvement is about `1.77e-8`. A meaningful, scale-aware improvement threshold is needed.

## 5. Is PUCT providing useful tree search?

Early exploitation is useful, but the recorded behavior is a rolling beam rather than a sequential PUCT allocation.

### 5.1 Rollout funnel

| Stage | Count | Fraction of all rollouts |
|---|---:|---:|
| Generated | 17,408 | 100% |
| Scientifically valid | 9,674 | 55.57% |
| Present in final non-seed archive | 544 | 3.125% |
| Expanded as a parent in a completed later step | 264 | 1.517% |
| Selected including unfinished step 34 | 272 | 1.562% |

The run generated 102.08 million response tokens. Only 264 generated nodes were subsequently expanded in a completed step. This does not make every other rollout useless: all 272 reward groups were nonconstant, so their samples could still train GRPO, feed memory, and supply feedback. It does show that fixed batches of 64 are very coarse for tree allocation.

Top-2 pruning is not the main problem. Reconstructing the archive shows that it retained the best child for 250 of 272 groups and retained the best child of the whole step in 33 of 34 completed steps. The problem is semantic duplication before and after pruning.

### 5.2 The visit and backup terms never govern a chosen node

Across all 280 saved parent selections, including step 34:

- every selected parent had `visit_count = 0`;
- every selected `q_value` equaled its own reward rather than a backed-up child value;
- all 280 parent IDs were unique;
- 94.85% of completed selections used a parent from the immediately preceding step.

Each node receives one fixed 64-rollout expansion, then a fresh unvisited descendant replaces it. The visit denominator and max-child backup never decide whether a promising parent deserves another small allocation. Operationally, the score is a best-first prior bonus over fresh leaves.

### 5.3 Lineage diversity collapses by step 6

The number of active seed roots among the eight selected parents was:

- step 0: 8;
- step 1: 5;
- step 2: 3;
- steps 3–5: 2;
- step 6 onward: 1.

The winning root received 15,360 of 17,408 rollouts, or 88.24% of the complete budget. By step 34, all parents shared one seed, one step-0 branch, and 28 ancestors; 86.15% of the median path was common to all eight parents.

Lineage blocking prevents selecting a direct ancestor and descendant together, but it does not keep cousin branches that share almost their entire history from occupying every group. The historical archive looks broad while the active beam is one basin.

### 5.4 Collapse is scientific, not textual

Across all valid rollouts, 86.52% of exact program bodies are unique, but only 2,357 of 9,674 saved constructions are distinct at 12-digit precision. Late behavior is decisive:

| Step | Valid rollouts | Distinct constructions | Same reward as parent |
|---:|---:|---:|---:|
| 17 | 351 | 39 | 0 |
| 22 | 381 | 31 | 11 |
| 24 | 369 | 11 | 260 |
| 26 | 357 | 6 | 349 |
| 31 | 312 | 7 | 305 |
| 32 | 237 | 2 | 236 |
| 33 | 264 | **1** | **264** |

The archive still grows mechanically by 16 nodes per step—from 24 after step 0 to 552 after step 33—because two textually distinct children are retained per parent. Archive growth is therefore not evidence of scientific exploration.

## 6. The collapse mechanism supported by this run

The artifacts support the following closed loop:

1. SLSQP and error-repair memory makes more programs executable and valid.
2. Many valid programs warm-start from `initial_h_values` and return the same locally optimized construction.
3. Exact-code deduplication treats those different programs as different scientific states.
4. Top-2 retention fills the archive with fresh IDs for the same basin.
5. PUCT gives every fresh ID zero-visit exploration credit and follows its descendants instead of revisiting or restarting other basins.
6. Entropic GRPO sees a large valid-versus-invalid gap but no difference among valid constructions, so it further reinforces feasibility and local reproduction.
7. Lifetime memory credit keeps selecting the historically safe lesson even after its recent tail effect becomes negative.

This explains the apparently contradictory observations: code remains diverse, code validity improves, the archive keeps growing, and yet the scientific best is flat.

## 7. Component-level verdict

| Component | Verdict from this run |
|---|---|
| End-to-end search | **Partial:** strong early improvement, target missed, late stall |
| Memory portfolio mechanics | **Working:** budget-neutral matched control and exploration |
| Selected memory feasibility | **Working strongly:** large causal validity benefit |
| Memory max-tail behavior | **Not robust:** interval crosses zero and late effect is negative |
| Exploratory memory | **Useful early:** rare large discoveries; ineffective late |
| Feedback controller/caps | **Working as designed** |
| Feedback causal efficacy | **Unknown:** suggestive trajectory, no control arm |
| Max-seeking GRPO | **Works only while valid rewards differ meaningfully** |
| PUCT/tree diversity | **Not working as intended:** one-shot leaves and one-lineage collapse |
| SOTA/target claim | **Not achieved** |

## 8. Highest-priority theoretical/design changes

1. **Define nodes by scientific state, not source code.** Deduplicate near-identical constructions or correlation profiles before assigning a fresh prior and visit count. Keep at most one representative of a local basin.

2. **Make tree allocation progressive.** Give a candidate parent a small pilot allocation, update its probability of producing a meaningful incumbent improvement, and allocate additional chunks only when its extreme-value posterior remains promising. This makes visits and max backups operational.

3. **Retain one quality child and one novelty child.** Top-2 by reward often preserves two programs returning the same construction. The second slot should maximize basin novelty subject to minimum quality.

4. **Reserve global exploration.** Keep 15–25% of groups for different roots, semantically distant archive states, or fresh restarts. No single root should own all groups merely because it won early.

5. **Use meaningful improvement and stagnation tests.** A child should count as better only if it is construction-novel and exceeds a scale-aware epsilon, for example `max(1e-10, 1e-4 × remaining target gap)`. Trigger diversification when unique-construction yield collapses or parent-return rate rises.

6. **Make memory credit recency-sensitive and objective-specific.** Separate feasibility credit from extreme-tail credit. Discount old evidence or use a sliding window, demote a lesson after sustained recent tail losses, and retain an explicit novelty/exploration arm. The early exploratory wins show why exploration should not be removed.

7. **Evaluate feedback causally.** Run the same starting checkpoint and seed with feedback off, or branch matched checkpoints before each repair episode. Save feedback-advantage magnitudes, clipping statistics, and whether a sampled corrected program actually runs.

8. **Handle timeouts outside textual feedback.** Preserve the requested no-timeout feedback policy, but add a separate resource controller that downweights parents or strategies with excessive timeout risk and enforces early best-so-far return.

The first three changes are the most important. Adding a stronger DPUCT bonus before semantic deduplication and progressive allocation would probably concentrate budget even faster on duplicate IDs representing the same local optimum.
