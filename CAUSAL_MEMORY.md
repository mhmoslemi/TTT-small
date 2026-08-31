# Causal Memory for Max-Seeking Scientific Discovery

## Purpose of this document

This document gives the conceptual model of memory used in this project. It is
written for an agent that needs to reason about the method, interpret an
experiment, or propose a change without first reading the implementation.

The central idea is:

> A memory is not useful because it sounds plausible, appears often, or was
> extracted from a successful rollout. It is useful only when exposing the
> policy to that memory improves search relative to a matched no-memory
> counterfactual.

The system therefore treats memory as a **causal intervention** and allocates a
fixed part of the existing search budget to estimating its effect.

---

## 1. Why ordinary memory is unsafe in discovery

In ordinary retrieval-augmented generation, a relevant-looking memory is added
to the prompt and a better-looking answer is taken as evidence that retrieval
helped. That reasoning is unreliable in an adaptive scientific search.

Suppose a lesson is extracted after one unusually good result. Reusing that
lesson makes later proposals resemble the result that produced it. Those
similar proposals generate more text supporting the same idea, so the lesson
is retrieved even more often. Eventually, exposure is mistaken for evidence:

$$
\text{lucky result}
\rightarrow
\text{lesson reuse}
\rightarrow
\text{similar rollouts}
\rightarrow
\text{apparent confirmation}
\rightarrow
\text{more lesson reuse}.
$$

This loop can rapidly improve a familiar family of candidates while preventing
the search from discovering a different, better family. We call this
**memory-induced collapse**.

Several observations that look like evidence are not causal evidence:

- the lesson came from a high-reward candidate;
- the selector frequently chooses the lesson;
- new rollouts repeat or paraphrase the lesson;
- rollouts using the lesson are more valid;
- the largest observed reward came from the arm with the most samples.

Each observation can be useful diagnostically, but none answers the required
counterfactual question: *What would have happened from the same search state
if the memory had not been shown?*

---

## 2. The objective is the best discovery, not the average response

Let $p$ be a selected parent state and let $m$ be a memory lesson. Showing $m$
changes the policy's conditional proposal distribution:

$$
Y \sim \pi_t(\cdot\mid p,m),
$$

where $\pi_t$ is the current policy at search step $t$. A verifier assigns each
proposal a reward $R(Y)$ and determines whether it is valid.

The final result of scientific search is normally the best valid candidate
found within a finite budget, not a randomly selected candidate. Memory must
therefore be judged by its effect on the attainable upper tail. For $n$
rollouts, define

$$
M_n(m\mid p,t)
=
\mathbb E\!\left[
\max_{1\le j\le n}R(Y_j)
\;\middle|\;
Y_j\sim\pi_t(\cdot\mid p,m)
\right].
$$

The local causal quantity of interest is

$$
\Delta_m^{(n)}(p,t)
=
M_n(m\mid p,t)-M_n(\varnothing\mid p,t),
$$

where $\varnothing$ means that no memory is injected.

- $\Delta_m^{(n)}(p,t)>0$: the memory improves the expected best result under
  an equal budget from this parent at this time.
- $\Delta_m^{(n)}(p,t)<0$: the memory makes the attainable best result worse.
- $\Delta_m^{(n)}(p,t)\approx0$: there is no demonstrated tail benefit at this
  context and budget.

This is deliberately different from comparing mean rewards. A lesson that
usually produces safe, ordinary candidates but suppresses rare breakthroughs
may raise the mean while lowering $M_n$.

---

## 3. The matched three-arm experiment

For every selected parent, the existing rollout budget $K$ is divided among
three arms:

1. **Selected-memory arm.** Test the lesson currently judged most relevant.
2. **No-memory control arm.** Generate from the same parent without memory.
3. **Exploration arm.** Test a lesson whose effect is still uncertain.

Their budgets satisfy

$$
K_{\mathrm{sel}}+K_0+K_{\mathrm{exp}}=K.
$$

This is one budget split into simultaneous comparisons, not two complete runs
with and without memory. Consequently, causal memory need not double the
rollout or evaluation budget.

The comparison is matched because all arms use:

- the same parent;
- the same policy snapshot;
- the same search step;
- the same evaluator and reward definition;
- comparable rollout randomness.

Matching matters. Comparing a memory rollout from an easy parent against a
no-memory rollout from a difficult parent would mix the effect of the memory
with the effect of the parent. Comparing arms before and after a policy update
would similarly mix memory with policy improvement.

The no-memory arm is permanent. It is not only an initial baseline: without a
concurrent control, later changes in the policy, parent distribution, and
search difficulty become indistinguishable from memory effects.

---

## 4. Why raw arm maxima cannot be compared

The maximum of more samples is larger in expectation, even if all samples come
from exactly the same distribution. Therefore, if one arm has more rollouts,
its raw maximum receives a statistical advantage unrelated to memory quality.

Let the observed rewards in arm $a$ be

$$
\mathbf r_a=(r_{a,1},\ldots,r_{a,K_a})
$$

and let $\mathbf r_0$ denote the control rewards. Set the common comparison
budget to

$$
n=\min(K_a,K_0).
$$

For an arm with $K_a$ rewards sorted as
$r_{(1)}\le\cdots\le r_{(K_a)}$, define the expected maximum of a uniformly
chosen size-$n$ subset:

$$
\widehat M_n(\mathbf r_a)
=
\frac{1}{\binom{K_a}{n}}
\sum_{j=n}^{K_a}
\binom{j-1}{n-1}r_{(j)}.
$$

The matched estimate is then

$$
\widehat\Delta_a
=
\widehat M_n(\mathbf r_a)
-
\widehat M_n(\mathbf r_0).
$$

Both terms now describe a best-of-$n$ outcome. The selected arm cannot win
merely because it received more attempts.

The estimator should be understood intuitively as follows: repeatedly imagine
drawing only $n$ results from each arm, record the best result in each imagined
draw, and compare the two expected best results. The formula computes that
quantity exactly from the observed rewards.

---

## 5. What receives credit

The intervention is the memory content shown in one arm. Arm identity must
therefore remain attached to its outcomes even if all sibling rollouts are used
together for policy learning.

The primary credit signal is matched tail uplift $\widehat\Delta_a$. Other
statistics answer different questions:

| Statistic | What it tells us | What it does **not** establish |
|---|---|---|
| Matched tail uplift | Whether memory improved the equal-budget best outcome | Whether it helps every parent |
| Validity change | Whether memory makes proposals more feasible or executable | Whether scientific quality improved |
| Parent improvement | Whether a rollout beat its immediate starting point | Whether memory caused the improvement |
| Tail win count | How often the memory arm won a matched contest | The magnitude or generality of the effect |
| Selection count | How often the selector exposed the lesson | Any outcome benefit |
| Textual repetition | Whether the policy reproduced the idea | Any causal effect |

Validity is especially easy to misinterpret. A lesson may reliably produce
valid candidates while narrowing the proposal distribution to a mediocre
region. That lesson is a useful feasibility aid but has not demonstrated a
max-seeking discovery benefit.

For clean lesson-level attribution, one tested arm should contain one lesson.
If several lessons are injected together, the observed uplift identifies the
effect of the **set**, not the individual contribution of each lesson. Credit
must not be duplicated across all members as though each had independently
caused the result.

---

## 6. Exploitation, exploration, and the role of UCB

Always selecting the lesson with the largest current estimated uplift would
recreate the original collapse problem. Early estimates are noisy, and an
under-tested lesson may be better than the current winner.

For lesson $\ell$, let

- $N_\ell$ be the number of matched trials;
- $\bar\Delta_\ell$ be its average matched tail uplift;
- $N_t$ be the total controlled memory evidence collected so far.

An upper-confidence score has the form

$$
\operatorname{UCB}_t(\ell)
=
\bar\Delta_\ell
+
c_m s_t
\sqrt{\frac{\log(1+N_t)}{N_\ell}},
$$

where $c_m$ controls exploration and $s_t$ places uncertainty on the observed
uplift scale.

The two terms have separate meanings:

- $\bar\Delta_\ell$ exploits lessons with demonstrated value;
- the uncertainty bonus tests lessons that have not received enough evidence.

Untested lessons must receive high priority rather than being assigned a
default value of zero and forgotten. As evidence accumulates, the uncertainty
bonus shrinks, so continued exposure must eventually be justified by measured
uplift.

The selected-memory arm and exploration arm therefore serve different roles.
The first spends budget on the current best contextual belief; the second buys
information that can overturn that belief.

---

## 7. The complete memory lifecycle

Memory has four logically separate stages. They must not be collapsed into one
score or one language-model judgment.

### 7.1 Formation

A lesson is extracted from evaluated search experience. Strong formation is
contrastive:

- compare a child with its parent to identify what actually changed;
- compare successful and failed attempts to distinguish useful operations from
  recurring mistakes;
- express a transferable principle rather than copy a complete construction.

Being extracted makes a lesson a **hypothesis**, not established knowledge.

### 7.2 Contextual selection

Given a parent, the selector chooses a small relevant lesson set or chooses no
lesson. Selection proposes which hypothesis to test in this context. It does
not award credit.

### 7.3 Controlled evaluation

The fixed parent budget is split into selected, control, and exploration arms.
Outcomes are compared using equal-budget best-of-$n$ uplift.

### 7.4 Updating and retention

The lesson accumulates controlled evidence across uses. Positive uplift
increases confidence that it deserves future search budget; negative uplift is
evidence that it may be harmful. When memory capacity is limited, controlled
outcome evidence should dominate textual importance or retrieval frequency.

These separations prevent three invalid shortcuts:

$$
\text{extracted}\not\Rightarrow\text{useful},\qquad
\text{selected}\not\Rightarrow\text{credited},\qquad
\text{valid}\not\Rightarrow\text{max-seeking}.
$$

---

## 8. What “causal” means here

The design supports a **local, on-policy causal estimate**. It asks whether
showing a particular memory changed the attainable reward tail for a particular
parent under the current policy and search conditions.

The interpretation relies on the following assumptions:

1. Arms differ in the intended memory intervention, not in parent, policy,
   evaluator, or evaluation budget after matching.
2. Rollout randomness is comparable across arms.
3. Outcomes in one arm do not alter generation or evaluation in another arm
   before the matched comparison is completed.
4. The reward and validity measurements are applied consistently.
5. The tested memory identity is known; if memories are bundled, only the
   bundle's effect is identifiable.

The estimate is not automatically universal. A lesson can help one parent and
hurt another, and its value can change as the policy learns. Aggregating uplift
across trials estimates performance over the contexts in which the lesson was
tested. It does not prove a context-free law.

Accordingly, valid claims include:

- “This lesson improved matched best-of-$n$ reward in these tested contexts.”
- “Selected memory improved code validity relative to its concurrent control.”
- “The current evidence is uncertain because the lesson has few matched
  trials.”

Invalid claims include:

- “Memory works because the best rollout happened to use memory.”
- “The lesson is causal because it was selected frequently.”
- “Higher validity proves better scientific discovery.”
- “A positive aggregate estimate means the lesson helps every parent.”

---

## 9. Relationship to the rest of the discovery system

Causal memory is one budget-allocation mechanism inside a larger max-seeking
loop. It should not be confused with the other signals:

| Component | Decision being learned | Evidence used |
|---|---|---|
| Search policy | Which parent or branch deserves more evaluations | Search-tree value and uncertainty |
| Max-seeking policy optimization | Which sampled responses should become more probable | Upper-tail reward among siblings |
| Failure feedback | Which token choices caused a malformed or crashing program | Code-level verifier diagnostics |
| Causal memory | Which prior lesson deserves future prompt exposure | Matched best-of-$n$ uplift over no memory |

All rollout arms may contribute to the policy update because every evaluated
response contains information about the proposal policy. However, memory credit
uses the retained arm labels and the matched control. Policy learning and
memory attribution are related but not interchangeable.

Memory should also remain subordinate to the overall discovery objective. It
is a mechanism for reallocating future probability mass and search budget, not
a second objective that rewards agreement with the memory bank.

---

## 10. Common failure modes

### Exposure is treated as evidence

A lesson becomes important because it appears often. This creates a
self-confirming loop. Only controlled outcomes should establish value.

### The control arm disappears

Once the system believes memory is useful, it stops generating without memory.
The system can then observe performance but can no longer attribute it.

### Unequal raw maxima are compared

The larger arm wins because it has more chances. Compare equal-budget
best-of-$n$ quantities instead.

### Validity replaces discovery reward

Memory produces safe programs and looks successful, while the scientific tail
stagnates. Report validity separately from causal reward uplift.

### Too many lessons are injected together

The arm may work, but individual lesson credit becomes unidentified. Keep the
intervention sparse or credit only the complete set.

### Early winners monopolize testing

No alternative receives enough trials to disprove the incumbent. Preserve an
explicit uncertainty-driven exploration arm.

### Evidence is pooled across incompatible contexts

A lesson's average looks neutral because strong positive and negative local
effects cancel. Inspect contextual heterogeneity rather than assuming every
lesson has one universal value.

### Policy drift is mistaken for memory improvement

Historical no-memory outcomes are compared with current memory outcomes after
the policy has changed. Use concurrent matched controls.

---

## 11. How to read a memory experiment

Analyze the experiment in this order:

1. **Verify the comparison.** Were treatment and control generated from the
   same parent and policy state?
2. **Verify the budget.** Are maxima matched to the same effective sample size?
3. **Measure scientific uplift.** Is best-of-$n$ reward better with memory?
4. **Measure feasibility separately.** Did validity improve even if reward did
   not?
5. **Check uncertainty.** How many matched trials support each conclusion?
6. **Check heterogeneity.** Does the effect change across parents, steps, or
   lesson identities?
7. **Check allocation.** Are uncertain lessons still tested, or has one lesson
   monopolized exposure?
8. **Check search diversity.** Even with positive local uplift, did memory
   reduce the number of meaningfully different regions explored?

The correct conclusion may be mixed. For example, memory can causally improve
validity while having uncertain or negative scientific tail uplift. That does
not mean the controlled method failed; it means the method successfully
distinguished a feasibility benefit from a discovery benefit.

---

## 12. Rules an agent should preserve when reasoning about changes

Any proposed change to causal memory should preserve these invariants unless it
explicitly replaces them with a stronger identification argument:

1. Memory use must compete inside a fixed total search budget.
2. A concurrent no-memory control must remain available.
3. Treatment and control must be matched by parent and policy state.
4. Comparisons must use an equal effective rollout count.
5. The primary outcome must reflect the max-seeking discovery objective.
6. Retrieval frequency and textual agreement must not create causal credit.
7. Uncertain lessons must retain a deliberate path to being tested.
8. Multi-lesson interventions must receive set-level, not fabricated
   lesson-level, attribution.
9. Validity, reward uplift, and search diversity must be reported separately.
10. Claims must remain local and on-policy unless broader evidence supports
    generalization.

When discussing this system, the shortest accurate summary is:

> Extracted lessons are hypotheses. Retrieval chooses which hypotheses to test.
> Matched no-memory rollouts provide the counterfactual. Equal-budget tail
> uplift supplies causal credit. UCB prevents uncertain ideas from being
> eliminated before they receive evidence.
