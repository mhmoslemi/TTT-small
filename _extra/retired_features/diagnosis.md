# Causal Memory Implementation Diagnosis

## Purpose

This document records a read-only audit of the causal-memory implementation for
a future LLM or developer. It diagnoses the current behavior; it does not record
any code changes.

The intended theory is described in `CAUSAL_MEMORY.md`. The implementation was
checked against these core requirements:

1. selected memory, no-memory control, and exploration share one fixed budget;
2. treatment and control use the same parent and policy snapshot;
3. memory receives credit from matched equal-budget tail uplift;
4. textual repetition and selection frequency do not create causal credit;
5. under-tested lessons retain an efficient route to evaluation;
6. causal evidence remains attached to the exact intervention that produced it.

## Overall verdict

The **immediate per-parent causal comparison is implemented correctly**. The
fixed budget is divided into prompt arms, the no-memory rollouts are concurrent,
and the expected best-of-$n$ estimator is correct.

The **long-lived causal memory bandit is not fully correct**. Three issues can
corrupt or waste the evidence used by UCB, lookup, retention, and curation:

1. curation can transfer causal evidence to rewritten, untested lessons;
2. all parents can receive the same exploration lesson in one step;
3. effects measured at different best-of-$n$ budgets are averaged together.

There are also two secondary issues involving deterministic seed separation and
the exact interpretation of the treatment prompt.

---

## What is correct

### Fixed rollout budget

`memory/bandit.py::allocate_memory_arms` divides the existing group budget $K$
among the selected, no-memory, and exploration arms. The counts sum to exactly
$K$; memory does not launch a second complete run.

Relevant code:

- `memory/bandit.py:30-88`
- `train_multy.py:1300-1342`

A read-only diagnostic checked every $K$ from 1 through 100 and found exact
budget conservation with no negative counts.

### Matched parent and policy snapshot

All prompt arms are constructed from the same selected parent. The adapter is
saved before lookup and rollout generation, and outcome credit is computed
before the policy update.

Relevant code:

- `train_multy.py:1262-1308`
- `train_multy.py:1493-1544`

### Correct equal-budget tail estimator

For treatment rewards $\mathbf r_a$ and control rewards $\mathbf r_0$, the code
uses

$$
n=\min(|\mathbf r_a|,|\mathbf r_0|)
$$

and computes

$$
\widehat\Delta_a
=
\widehat M_n(\mathbf r_a)-\widehat M_n(\mathbf r_0).
$$

`expected_subsample_max` correctly computes the expected maximum of a uniformly
chosen size-$n$ subset. It was checked against brute-force enumeration for every
subset size on small arrays.

Relevant code:

- `memory/bandit.py:91-127`

### Arm identity is preserved

Each rollout retains its `memory_arm`, `memory_ids`, and actual prompt. Treatment
outcomes can therefore be separated from control outcomes after generation.

Relevant code:

- `train_multy.py:1531-1535`
- `train_multy.py:1597-1600`
- `memory/types.py:103-104`

### Lesson-level identifiability is guarded

When outcome credit is enabled, configuration validation requires one lesson per
selected arm and requires a nonzero control fraction.

Relevant code:

- `memory/config.py:195-216`

### Textual confirmation is separated from causal evidence

The bank stores selection/confirmation counts separately from controlled outcome
statistics. In the Erdos configuration, textual reinforcement is disabled.

Relevant code:

- `memory/types.py:173-188`
- `memory/bank.py:114-147`
- `configs/erdos.yaml:120-121`

---

## Finding 1 — causal evidence survives a change of intervention

**Severity: high**

### Behavior

Curation may rewrite, refine, or merge lessons. The resulting lesson has new
content and usually a new ID, but `_carry_counters` copies all controlled outcome
statistics from one or more old lessons:

- `arm_trials`;
- `arm_rollouts`;
- `arm_valid`;
- `arm_parent_improvements`;
- `arm_tail_wins`;
- `tail_uplift_sum`;
- `tail_uplift_best`;
- `last_outcome_step`.

Explicit `merged_from` IDs are trusted without verifying that the new lesson is
the same intervention. If no IDs are supplied, token overlap of only 0.4 can be
enough to transfer the evidence.

Relevant code:

- `memory/curator.py:209-265`
- `memory/curator.py:41-86`

### Why this is causally invalid

The observed uplift belongs to the exact prompt content that generated the
rollouts. Rewriting the lesson changes the treatment. Old outcomes cannot be
treated as trials of the new treatment unless equivalence is guaranteed.

Consequently, an untested rewritten lesson can enter the catalog with apparently
strong causal evidence and immediately influence lookup, UCB, retention, and
future curation.

### Confirming diagnostic

A read-only construction changed the lesson from “use coordinate descent” to
“use a different global optimizer,” declared the old ID in `merged_from`, and
called `_carry_counters`. The new lesson inherited all seven old trials and the
complete old uplift sum despite its content being different.

### When it is active

The Erdos configuration enables curation every three steps once the bank contains
at least twenty lessons:

- `configs/erdos.yaml:105-109`

The default five-step run may finish before the threshold is reached, but longer
runs can trigger the problem.

### Required invariant for a future fix

Causal outcome statistics must remain attached to an unchanged intervention.
Lineage and usage metadata may be inherited separately, but a rewritten or
merged treatment needs a new causal ledger unless equivalence is established by
a strict, non-generative rule. A merge of several tested lessons does not become
a tested intervention merely by summing their statistics.

---

## Finding 2 — same-step exploration repeatedly selects one lesson

**Severity: high for efficiency and coverage; individual matched estimates remain valid**

### Behavior

`train_step` allocates the arms for every parent before any rollout is evaluated.
Each independent call to `allocate_memory_arms` asks the bank for an exploration
lesson. The bank has not yet received a provisional reservation or a new trial,
so every call sees identical UCB statistics.

Untested lessons receive an infinite score. Their deterministic tie-break favors
the least selected and then the newest lesson. Therefore, several parents often
receive the same exploration lesson.

Relevant code:

- `train_multy.py:1303-1339`
- `memory/bandit.py:66-75`
- `memory/bank.py:173-193`

The `step` argument accepted by `exploration_lesson` is not used in its score or
tie-breaking.

### Confirming diagnostic

With five untested lessons, one selected lesson, four parents, and the current
20% exploration fraction, four consecutive same-step allocations returned the
same exploration ID. Only one unique lesson received the entire exploration
budget.

### Consequence

The code still collects valid causal evidence for the repeated lesson, but it
spends several simultaneous exploration arms measuring the same hypothesis
before any one of those measurements can update UCB. If extraction adds lessons
faster than this process tests unique lessons, a growing fraction of memory is
never tested. This undermines the purpose of the exploration arm and can recreate
concentration around a small subset of the bank.

### Required invariant for a future fix

Exploration must be allocated jointly across the parents of a step, or each
allocation must create a provisional reservation/trial count before the next
parent is assigned. The batch policy may revisit a lesson deliberately, but it
should not do so accidentally because all selections observed stale statistics.

Contextual relevance should also be considered: the current UCB exploration
choice receives no parent context, so an under-tested but irrelevant lesson can
be tested against an arbitrary parent and accumulate misleadingly negative
aggregate evidence.

---

## Finding 3 — the evidence ledger mixes different best-of-$n$ effects

**Severity: high for UCB and retention ranking**

### Behavior

The local estimator correctly chooses

$$
n=\min(K_a,K_0),
$$

but `record_outcome` receives only the resulting scalar uplift. It does not store
$n$, arm role, parent context, or the comparison budget. `mean_tail_uplift` then
averages all stored uplifts equally.

Relevant code:

- `memory/bandit.py:104-127`
- `memory/bank.py:148-171`
- `memory/types.py:181-220`

### Current configuration example

The checked-in Erdos configuration has $K=8$ with nominal fractions 60% selected,
20% control, and 20% exploration. Largest-remainder allocation produces:

$$
K_{\mathrm{sel}}=5,\qquad K_0=2,\qquad K_{\mathrm{exp}}=1.
$$

Therefore:

- selected-memory uplift estimates $\Delta^{(2)}$;
- exploration uplift estimates $\Delta^{(1)}$.

If the same lesson is first explored and later selected, its stored mean combines
best-of-1 and best-of-2 effects. Different lessons may also receive systematically
different $n$ distributions depending on whether they were usually selected or
explored. Adaptive changes to group size introduce additional values of $n$.

Relevant configuration:

- `configs/erdos.yaml:52-57`
- `configs/erdos.yaml:116-120`

### Why this matters

$\Delta^{(1)}$ and $\Delta^{(2)}$ are different causal estimands. A memory can
improve typical output while suppressing rare high-tail outcomes, so its effect
can change sign with $n$. Averaging them as interchangeable samples gives UCB and
retention no coherent quantity to optimize.

### Required invariant for a future fix

Either:

1. enforce one common comparison $n$ across selected and exploration arms and
   across the evidence being ranked; or
2. store evidence indexed by $n$ and compare/aggregate it under an explicitly
   chosen target rollout budget.

At minimum, $n$ and arm role must remain in the causal evidence record. Changing
group size must not silently change the estimand while preserving one global
mean.

---

## Finding 4 — lookup does not use its reserved deterministic seed offset

**Severity: medium; active only in seeded deterministic runs**

### Behavior

`LOOKUP_STEP_OFFSET` is defined, and the memory LLM documentation says lookup,
extraction, curation, and rollouts use different seed namespaces. Extraction and
curation apply their offsets, but lookup passes the raw search step directly.

Relevant code:

- `memory/llm.py:11-22`
- `memory/lookup.py:97-102`
- `memory/extractor.py:139-145`
- `memory/curator.py:173-179`
- `train_multy.py:1411-1414`

### Consequence

In deterministic operation, stochastic lesson selection and rollout generation
can reuse the same step seed namespace. This contradicts the documented RNG
separation and can couple treatment selection randomness with outcome randomness.
It also makes it harder to reason about reproducibility when memory lookup is
toggled.

The checked-in Erdos YAML currently has `deterministic: false`, so this defect is
dormant unless determinism is enabled from the command line or a saved run.

### Required invariant for a future fix

Lookup must use a seed namespace disjoint from rollout, extraction, and curation
generation.

---

## Finding 5 — lesson credit identifies a composite prompt treatment

**Severity: interpretation limitation; medium if isolated lesson effects are claimed**

### Behavior

For memory-aware problems, adding memory can change both:

1. the lesson text shown to the policy; and
2. the surrounding instructions describing how to reason about memory.

The no-memory control receives neither. In the Erdos prompt, the memory arm is
told to work through the lessons, identify irrelevant lessons, avoid spent
avenues, and seek a different algorithmic direction. The control receives a
shorter generic improvement instruction.

Relevant code:

- `problems/erdos.py:138-168`
- `problems/circle_packing.py:66-97`
- `problems/circle_packing.py:118-122`
- `problems/gpu_mode.py:405-414`

### Correct interpretation

The current estimator identifies the causal effect of the **complete memory
prompt protocol**—wrapper plus lesson—relative to the no-memory prompt. It does
not isolate the textual lesson's contribution from the generic instruction to
analyze memory or diversify.

This is acceptable if the intervention is formally defined as the complete
prompt arm. It is not acceptable to claim that the individual lesson alone
caused the measured uplift.

### Required invariant if lesson-only attribution is desired

Treatment and control must share the same generic reasoning wrapper and differ
only in the lesson content, possibly using an empty or placebo lesson in the
control. Otherwise, documentation and reports must explicitly name the treatment
as a composite prompt intervention.

---

## Finding 6 — very small groups can silently lose required arms

**Severity: low for the current $K=8$ configuration; important for future speed reductions**

### Behavior

Configuration validation requires a positive control fraction when causal credit
is enabled, but it does not verify that integer allocation produces an actual
control or exploration rollout.

For 60%/20%/20% allocation, diagnostics produced:

| $K$ | Actual arms |
|---:|---|
| 1 | selected 1 |
| 2 | selected 1, control 1 |
| 3 | selected 2, control 1 |
| 4 | selected 2, control 1, exploration 1 |
| 8 | selected 5, control 2, exploration 1 |
| 16 | selected 10, control 3, exploration 3 |
| 64 | selected 38, control 13, exploration 13 |

At $K=1$, no matched control exists. At $K=2$ or $K=3$, the configured exploration
arm disappears.

Relevant code:

- `memory/config.py:195-216`
- `memory/bandit.py:30-88`

### Required invariant for a future fix

When causal outcome credit is enabled, the effective integer allocation must be
validated, not only the fractions. The run should either guarantee the required
arms or state that the chosen group size is too small for the requested design.

---

## Configuration scope

The checked-in Erdos configuration activates the causal design:

- one selected lesson per arm;
- 20% no-memory control;
- 20% UCB exploration;
- matched outcome credit;
- no textual reinforcement.

See `configs/erdos.yaml:88-126`.

Most other problem configurations have memory disabled. However,
`configs/gpu_mode_trimul.yaml` and `configs/gpy_mode.yaml` enable memory while
retaining the legacy noncausal settings:

- control fraction 0;
- exploration fraction 0;
- outcome credit off;
- textual reinforcement on;
- up to five selected lessons.

Therefore, “the codebase supports causal memory” is true, but “every active
memory configuration is causal” is false.

---

## Evidence and test coverage

Read-only diagnostics performed during the audit established that:

- `expected_subsample_max` matched brute-force subset enumeration;
- allocated arm counts summed exactly to $K$ for $K=1,\ldots,100$;
- with $K=8$, allocation was selected 5, control 2, exploration 1;
- four same-step parent allocations selected one identical exploration lesson;
- a rewritten lesson inherited seven old causal trials and their uplift sum.

The repository currently has no dedicated tests for:

- `memory/bandit.py`;
- same-step exploration diversity;
- preservation of intervention identity through curation;
- heterogeneous-$n$ outcome accounting;
- the lookup RNG namespace.

Existing tests focus primarily on resume and vLLM runtime behavior. Passing those
tests would not establish causal-memory correctness.

---

## Priority order for a future implementation task

1. **Stop causal-statistic inheritance across rewritten or merged lesson text.**
2. **Allocate exploration jointly across parents or reserve lessons within the
   step.**
3. **Make the causal evidence ledger aware of comparison budget $n$ and arm
   role, or enforce a single common $n$.**
4. **Apply the lookup seed offset in deterministic runs.**
5. **Decide whether the estimand is the full memory prompt or isolated lesson
   content, then align the control prompt and claims accordingly.**
6. **Validate effective integer arm counts for small group sizes.**
7. **Add focused tests for every invariant above before interpreting another
   memory experiment.**

## Short handoff summary

The code correctly performs a concurrent, budget-neutral best-of-$n$ comparison
against no memory. The estimator itself is not the problem. The weaknesses occur
after and around that comparison: curation can attach evidence to a new treatment,
same-step UCB allocation over-tests one lesson, and the bank averages effects from
different tail budgets. Until those are corrected, individual per-parent uplift
records are credible, but long-term lesson rankings and retention decisions are
not fully causally identified.
