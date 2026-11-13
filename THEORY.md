# TTT-Discover: Mathematical Theory, Search Logic, and Design Rationale

This document describes the theory of the discovery system: what probability distributions it defines, what objectives it optimizes, how search and learning interact, and why each major design choice exists. It deliberately excludes software architecture, APIs, command-line usage, storage formats, device management, and other implementation instructions.

The names **SPO-RS** and **SPO-RS-v2** are treated as the project’s algorithm names. The codebase does not define a canonical long-form expansion, so none is invented here.

---

## 1. The problem the system is solving

The system searches for a program that constructs a high-quality solution to a fixed scientific or combinatorial problem. A candidate is not merely text: it is an executable generator of a construction.

Let:

- \(x\) be the problem statement and evaluator;
- \(p\) be a selected parent program or parent construction;
- \(z\) be a natural-language strategy;
- \(y\) be a sampled candidate program;
- \(a=E_x(y)\) be the construction obtained by executing \(y\) under the problem evaluator;
- \(s_x(a)\) be the problem’s native scientific score;
- \(r_x(y)\) be the internal reward used by search and learning;
- \(v_x(y)\in\{0,1\}\) indicate whether the candidate is valid.

The scientific goal is a best-of-search objective:

$$
y^* \in \arg\max_{y\in\mathcal Y_{\mathrm{valid}}} r_x(y).
$$

This is different from maximizing the average quality of all sampled programs. A single exceptional construction can be the desired outcome. That distinction explains several otherwise unusual choices:

- parent values use the best child rather than the average child;
- strategy allocation estimates improvement in the eventual maximum;
- the archive retains high-scoring branches;
- training emphasizes relative winners;
- semantic diversity is preserved because another conceptual basin may contain the rare breakthrough.

The system is therefore an **online discovery process**, not a stationary reinforcement-learning benchmark. Its prompts, parents, strategies, and sampling allocation change in response to previous observations.

---

## 2. Native score, internal reward, validity, and failure

### 2.1 A single higher-is-better convention

Different problems report different native quantities. Some are maximized; others, such as runtime, error, or a bound, are minimized. The learning system converts every task to a higher-is-better reward through a monotone transformation \(h_x\):

$$
r_x(y)=h_x\!\left(s_x(E_x(y))\right).
$$

Typical examples are:

$$
h(s)=s \quad\text{for a maximization score},
$$

$$
h(s)=\frac{1}{s}\quad\text{for a positive quantity to minimize},
$$

or, for a runtime \(t\),

$$
h(t)=\frac{c}{t}
$$

for a fixed scale \(c>0\).

A monotone transform preserves which valid candidate is best, but it does **not** preserve the learning dynamics. Exponentials, differences, and posterior models operate in reward space, so reward normalization is part of the objective rather than mere presentation.

### 2.2 Validity is not the same object as reward

A rollout can fail because its output is malformed, its program does not run, execution times out, the returned construction violates constraints, or the evaluator rejects it. These cases receive a common failure reward, normally

$$
r_{\mathrm{fail}}=0.
$$

The binary validity variable is nevertheless retained:

$$
v_x(y)=
\begin{cases}
1,&\text{the evaluator accepts the candidate},\\
0,&\text{otherwise}.
\end{cases}
$$

This separation matters. Base SPO-RS uses only scalar reward, so two zero-reward outcomes are equivalent to it. SPO-RS-v2 and the hurdle allocator also use validity or improvement events and can distinguish an accepted but weak result from an invalid one when the evaluator exposes that distinction.

### 2.3 The evaluator is the authority

The sampled program cannot establish its own success by claiming a score. Its text is executed and independently checked. This creates a hard trust boundary:

$$
\text{program text}\longrightarrow\text{execution}\longrightarrow
\text{validated construction}\longrightarrow\text{reward}.
$$

The parent program is best viewed as a **genotype** and its evaluated construction as a **phenotype**. Descendants may inherit the parent construction as a warm start while changing the generative procedure that produced it.

### 2.4 Best-ever and current archive are different statistics

The best valid result ever observed is recorded before deduplication or pruning. The current archive is a bounded search structure. Thus:

$$
\text{best-ever record}\neq\max(\text{current archive})
$$

in general. Archive pressure must never be allowed to erase the scientific result already found.

---

## 3. The three nested decision layers

Each discovery step contains three distinct decisions:

1. **Parent selection:** which existing search states deserve another expansion?
2. **Strategy selection:** which conceptual approaches should be tried from each parent?
3. **Program sampling and learning:** which concrete programs realize those approaches, and how should their likelihood change?

For a parent \(p\), sequential strategy \(z_k\), and program \(y_{k,j}\), the conceptual sampling law is

$$
p\sim\alpha_t(p),
$$

$$
z_k\sim q_\phi\!\left(z\mid x,p,z_{<k}\right),
$$

$$
y_{k,j}\sim \pi_{\theta_t}\!\left(y\mid x,p,z_k\right).
$$

Here:

- \(\alpha_t\) is the parent picker induced by PUCT, UCT, an LLM picker, or PUCT-JEV;
- \(q_\phi\) is the frozen strategist;
- \(\pi_{\theta_t}\) is the trainable coder;
- the number of programs assigned to strategy \(k\) may depend on pilot observations.

Only the coder distribution is trained. No gradient passes through archive pruning, parent selection, the strategist, or the rollout allocator. These mechanisms form an adaptive curriculum around the learned policy.

This separation avoids pretending that the whole discovery loop is a differentiable objective. It also makes the source of each bias explicit.

---

## 4. The search state and archive

### 4.1 A state in the search tree

A search state contains a valid program, its construction, its reward, its direct parent, its ancestry, visit statistics, and the strategy under which it was produced. Seeds are root states supplied before learning begins.

The tree is not assumed to be a faithful model of causal scientific descent. It is an operational genealogy: a child was generated while the parent and its construction were in context.

### 4.2 What enters the archive

Only valid candidates can become future parents. Invalid rollouts can still contribute to policy learning, but they do not become search states.

Archive admission is progressively filtered by four principles:

1. **Exact program deduplication.** Identical source programs represent the same proposal, even if stochastic execution could produce different outcomes.
2. **Optional per-strategy pressure.** Only the best \(r\) results for a given parent-strategy pair may remain eligible. This prevents one strategy from flooding the archive.
3. **Per-parent child pressure.** Only the top \(K\) direct children of a parent are retained. This bounds local branching.
4. **Global capacity pressure.** If the archive exceeds its capacity, states are ranked by reward and the weakest non-seeds are removed. Seeds are permanent anchors.

These filters do not define the scientific objective; they approximate a finite-memory search. Their main bias is toward exploitation. Strategy caps and semantic gates partly restore diversity.

### 4.3 Exact deduplication versus conceptual diversity

Exact source equality and semantic equality are different notions:

- exact deduplication asks whether two program texts are identical;
- JEV asks whether two programs implement the same core method;
- different random seeds, constants, budgets, formatting, or variable names may be textually different but semantically redundant;
- two programs may share syntax while implementing genuinely different representations or optimization mechanisms.

The system uses both levels because neither alone is sufficient.

---

## 5. Parent picking

Parent selection determines the distribution of prompts seen by the coder. It is therefore a major source of curriculum and search bias even though it is not learned by gradient descent.

### 5.1 Shared search statistics

For each state \(s\), define:

- \(R(s)\): the state’s own reward;
- \(n(s)\): the number of expansions attributed to the state or its descendant lineage;
- \(m(s)\): the best reward among its observed direct children;
- \(T\): the total number of archive expansion events;
- \(P(s)\): a rank-based prior;
- \(\Delta\): an empirical reward scale.

The exploitation value is

$$
Q(s)=
\begin{cases}
m(s),&n(s)>0,\\
R(s),&n(s)=0.
\end{cases}
$$

Using the best child rather than the mean child matches the best-of-search goal. It is deliberately optimistic and can be noisy: one lucky child can make a branch attractive. Visit penalties and bounded archive capacity provide the counterpressure.

Visits propagate through the ancestry. Expanding a descendant therefore reduces the future exploration bonus of the whole exploited lineage, rather than treating each node as unrelated.

The empirical scale is

$$
\Delta=\max\!\left(\max R-\min R,\,10^{-6}\right)
$$

over the relevant non-seed archive, with a neutral fallback when no range exists. It places reward and exploration bonuses in comparable units.

### 5.2 Rank prior

If there are \(N\) eligible states, sorted best to worst, the state at zero-based rank \(k\) gets unnormalized weight \(N-k\). Thus

$$
P_k=\frac{N-k}{\sum_{j=1}^{N}j}.
$$

Rank is robust to extreme numerical reward gaps and preserves a preference for quality. The cost is that it discards how much better one state is than another.

### 5.3 PUCT

The PUCT score is

$$
U_{\mathrm{PUCT}}(s)
=Q(s)+c\,\Delta\,P(s)\frac{\sqrt{1+T}}{1+n(s)}.
$$

The terms play separate roles:

- \(Q(s)\) exploits a branch that has already produced a strong child;
- \(P(s)\) favors higher-ranked states before evidence accumulates;
- \(\sqrt{1+T}\) slowly increases global pressure to revisit underexplored states;
- \(1+n(s)\) suppresses repeatedly used lineages;
- \(c\) controls the exploration-exploitation balance.

PUCT is appropriate when rank is a meaningful prior belief about which states deserve expansion.

### 5.4 UCT

The UCT alternative is

$$
U_{\mathrm{UCT}}(s)
=Q(s)+c\,\Delta
\sqrt{\frac{\log(1+T)}{1+n(s)}}.
$$

UCT omits the explicit rank prior. Its exploration bonus is symmetric across states with equal visit counts. It is useful when the search should rely more on empirical visits and less on current reward rank.

### 5.5 Lineage blocking inside a parent batch

Selecting several nearly identical points on one lineage wastes a parallel batch. During the first selection pass, choosing state \(s\) blocks:

- \(s\) itself;
- all ancestors of \(s\);
- all descendants of \(s\).

This encourages each parent slot to represent a different branch. If strict blocking prevents a full batch, a later pass relaxes the constraint and fills the remaining slots by search score. Diversity is therefore a strong preference, not a reason to leave compute idle.

### 5.6 LLM parent picker

The LLM picker sees a structured view of the tree and selects a set jointly rather than assigning independent numeric bonuses.

Its first choice is constrained to be a state with the exact best native scientific score. This preserves a guaranteed exploitation anchor. The remaining choices may mix:

- exploitation of strong or improving branches;
- exploration of underused roots or lineages;
- continuation of branches with informative progress;
- deliberate revisitation, because the coder has changed since old attempts.

There is no fixed exploit/explore quota. Distinct eligible state identifiers are required, but the picker may choose related nodes when it judges that useful. It reasons from tree evidence rather than program contents and must not rely on timestamps, generator versions, or information it was not shown.

The LLM picker’s strength is qualitative joint reasoning. Its weaknesses are nondeterminism, calibration difficulty, possible inconsistency, and the absence of a simple regret bound. The exact-best first choice is a guardrail against losing the strongest known anchor.

Native-score direction is respected when identifying that anchor: “best” means the exact minimum for a minimization problem and the exact maximum for a maximization problem. Generated states carrying the failure-sentinel raw score are not eligible merely because that number would look attractive under the wrong direction; zero or missing seed scores may still be allowed during bootstrap when no generated evidence exists.

### 5.7 PUCT-JEV: semantic diversity as a gate

PUCT-JEV begins with ordinary PUCT ranking, then applies a semantic redundancy classifier. JEV is treated here as the project’s classifier name; no unrecorded expansion of the acronym is assumed.

The first discovery step uses ordinary PUCT because no established selected-parent set yet exists to justify semantic filtering. Later steps apply the gate.

If the desired batch size is \(B\), only the top \(X B\) PUCT candidates are considered, where \(X\ge1\) is a window multiplier. Candidates are visited in PUCT order. A candidate is accepted only if it is not classified as “same core approach” as every already accepted parent.

For a pair of programs, the classifier produces

$$
p_{\mathrm{same}}+p_{\mathrm{variant}}+p_{\mathrm{different}}=1.
$$

The candidate is rejected as redundant when

$$
p_{\mathrm{same}}\ge\tau,
$$

for a threshold such as \(\tau=0.68\). A meaningful variant is acceptable; only core-method equivalence is the exclusion target.

The semantic judgment ignores cosmetic differences such as:

- renaming;
- formatting;
- refactoring without algorithmic change;
- random seeds;
- constant or budget tuning.

Differences in representation, construction rule, neighborhood, optimization mechanism, or mathematical decomposition count as substantive.

JEV is a **gate**, not a reranker. Among accepted candidates, the original PUCT order remains intact. This has three rationales:

1. search value remains the primary quality signal;
2. semantic judgment is used only for the question it can answer—redundancy;
3. restricting comparisons to a bounded high-quality window limits cost and prevents “different but hopeless” candidates from displacing all strong ones.

Lineage blocking is still applied around accepted parents. If diversity and lineage constraints cannot fill the batch, a final fallback fills from the bounded PUCT window, even if that re-admits a semantically rejected duplicate. Full utilization is guaranteed; semantic diversity is not absolute.

The threshold controls asymmetric risks:

- a lower threshold rejects more redundancy but can suppress real innovation;
- a higher threshold preserves more variants but spends more rollouts on duplicated ideas.

### 5.8 Parent-picking comparison

| Picker | Exploitation signal | Exploration/diversity signal | Main strength | Main risk |
|---|---|---|---|---|
| PUCT | Best direct child plus rank prior | Visit bonus and lineage blocking | Stable quality-aware search | Optimistic outliers and rank bias |
| UCT | Best direct child | Visit bonus and lineage blocking | Less prior dependence | May spend on weak states uniformly |
| LLM | Exact-best anchor plus tree reasoning | Joint qualitative selection | Can reason about branch structure | Hard to calibrate or reproduce |
| PUCT-JEV | PUCT order | Lineage plus semantic gate | High-quality semantic variety | Classifier errors and extra cost |

---

## 6. The strategist–coder decomposition

### 6.1 Why split planning from coding?

A single model asked to invent an approach and express flawless executable code in one pass must solve two different problems simultaneously:

1. search over mathematical ideas;
2. search over concrete program realizations.

The strategist–coder split introduces strategy text \(z\) as an explicit latent action:

$$
p(y\mid x,p)
=\int q_\phi(z\mid x,p)\,
\pi_\theta(y\mid x,p,z)\,dz.
$$

The decomposition allows several programs to test the same idea and several ideas to branch from the same parent. It also makes strategy-level credit and allocation possible without requiring the coder to expose a faithful internal chain of thought.

### 6.2 Sequential strategy sampling

For each parent, strategies are sampled sequentially:

$$
z_k\sim q_\phi(z\mid x,p,z_1,\ldots,z_{k-1}).
$$

The strategist is explicitly asked for a materially different approach from earlier strategies for that parent. The diversity history is local to the parent. A strategy proposed for one parent does not prevent a similar strategy for another parent, because its usefulness can depend on the inherited construction.

The sequential conditioning is on earlier **plan text**, not the evaluation results of earlier strategies. Strategy generation is therefore a diversity mechanism, not an inner feedback loop.

### 6.3 What a strategy represents

A useful strategy must be self-contained enough to function as a one-shot scientific proposal. Conceptually it should specify:

- why the idea might improve the parent;
- the mathematical representation or objective;
- the constructive or optimization principle;
- important variations or robustness considerations;
- how success should be validated.

These are not software requirements. They reduce ambiguity in the latent action \(z\): two coders conditioned on the same strategy should at least be attempting the same scientific idea.

### 6.4 Information boundary

The coder receives the original problem, the parent, and one extracted strategy plan. It may adapt or reject the plan if necessary. Hidden strategist deliberation is not part of the coder’s conditioning information.

This boundary matters theoretically. The observable strategy \(z\), not an unrecorded reasoning trace, is the variable that explains the strategy arm and supports later allocation analysis.

### 6.5 Frozen strategist, trainable coder

The strategist distribution \(q_\phi\) is fixed while the coder \(\pi_\theta\) changes. This choice provides a stable source of high-level variation and concentrates gradient updates on code realization.

The tradeoff is a distribution mismatch over time: a fixed strategist may continue proposing ideas that the improved coder has mastered or cannot exploit. Adaptive rollout allocation and parent changes provide partial compensation, but there is no learned strategist credit assignment.

### 6.6 Direct mode and no-strategy fallback

Strategy use is optional. In direct mode, the coder distribution is simply

$$
y\sim\pi_\theta(y\mid x,p).
$$

If a strategy response remains malformed after bounded recovery attempts, that strategy slot becomes an explicit direct-solve/no-strategy condition rather than aborting the whole discovery step. This preserves the fixed rollout opportunity while honestly marking that no usable latent plan was obtained.

### 6.7 Output recovery as a sampling policy

Malformed strategy text is retried up to a fixed limit because a missing plan would invalidate an entire strategy arm. Exhaustion falls back to direct solving.

Coder recovery is deliberately weaker. Original malformed programs keep their failure reward; they are not erased or relabeled. A batch-level recovery gate activates only if fewer than half of original programs contain a complete final code block. It adds only enough recovery attempts to target approximately 60% complete outputs, and those extra attempts do not consume the allocator’s planned rollout counts.

This policy encodes two beliefs:

- malformed output is real evidence about the coder and should remain negative training data;
- a catastrophically malformed batch may be an interface failure rather than useful scientific evidence, so limited extra sampling protects the step.

Trying to repair every failure would distort the sampling distribution and hide reliability problems.

### 6.8 Reasoning effort and rollout count are separate resources

Pilot and continuation rollouts may use different inference effort. This changes the conditional quality and cost of samples, but not the mathematical identity of the strategy arm. Rollout allocation counts samples; reasoning effort allocates computation within a sample.

Comparisons are cleanest when arms in the same phase receive the same effort. If efforts differ systematically, observed reward combines strategy quality with inference-budget effects.

---

## 7. Adaptive allocation among strategies

Suppose each parent has \(S\) strategies and a nominal budget of \(M\) programs per strategy. First sample \(H\) pilots from every strategy, where \(0<H\le M\). The remaining budget per parent is

$$
B=S(M-H).
$$

Let \(a_k\) be the additional rollouts assigned to strategy \(k\). The fixed-budget constraint is

$$
a_k\in\mathbb Z_{\ge0},
\qquad
\sum_{k=1}^{S}a_k=B.
$$

Thus every parent still receives exactly \(SM\) planned programs. Adaptation changes only their distribution across strategies. This preserves per-parent batch comparability and prevents the allocator from silently changing the training volume.

All pilots and continuations within a step are sampled before a policy update. Therefore arm comparisons are not confounded by coder learning during the allocation decision.

### 7.1 Rule-based allocation

For each strategy \(k\), compute the pilot mean \(\mu_k\) and variance \(\sigma_k^2\). Let \(\tilde\mu\) and \(\widetilde{\sigma^2}\) be the medians across strategies. Assign a heuristic weight

$$
w_k=
\begin{cases}
4,&\mu_k>\tilde\mu,\ \sigma_k^2>\widetilde{\sigma^2},\\
3,&\mu_k>\tilde\mu,\ \sigma_k^2\le\widetilde{\sigma^2},\\
2,&\mu_k\le\tilde\mu,\ \sigma_k^2>\widetilde{\sigma^2},\\
0,&\text{otherwise}.
\end{cases}
$$

High mean is interpreted as exploitation value. High variance is interpreted as upside potential. Remaining rollouts are distributed proportionally to \(w_k\), using largest remainders to recover integer counts.

An arm whose pilots all have approximately zero mean and zero variance receives a small exploration floor when the budget permits. If all weights vanish, allocation is equal.

This rule is cheap and understandable, but it has no posterior semantics. Variance can mean upside, instability, or failure mixture; means are reward-scale dependent; and median thresholds discard magnitude.

### 7.2 Posterior expected-best bandit

The bandit allocator directly targets the expected best reward after the remaining samples.

#### Validity model

For strategy \(k\), if \(n_k^{\mathrm{valid}}\) of \(H\) pilots are valid, use a uniform-prior Beta posterior:

$$
\theta_k\mid\mathcal D
\sim\operatorname{Beta}
\left(n_k^{\mathrm{valid}}+1,
H-n_k^{\mathrm{valid}}+1\right).
$$

Here \(\theta_k\) is the probability that a future program from the arm is valid.

#### Conditional reward model

Valid rewards use a Normal-Inverse-Gamma model with weak pooled empirical-Bayes hyperparameters. For valid observations \(x_1,\ldots,x_n\), prior parameters \((\mu_0,\kappa_0,\alpha_0,\beta_0)\), and sample mean \(\bar x\), the posterior is

$$
\kappa_n=\kappa_0+n,
$$

$$
\mu_n=\frac{\kappa_0\mu_0+n\bar x}{\kappa_n},
$$

$$
\alpha_n=\alpha_0+\frac n2,
$$

$$
\beta_n
=\beta_0+rac12\sum_i(x_i-\bar x)^2
+\frac{\kappa_0n(\bar x-\mu_0)^2}{2\kappa_n}.
$$

The pooled prior shares a weak location and scale across strategies, allowing arms with few valid pilots to borrow limited information without forcing them to be identical.

A posterior scenario draws an arm validity rate, a conditional reward distribution, and future outcomes. Invalid future outcomes receive \(r_{\mathrm{fail}}\). Valid Gaussian draws are truncated below at the failure reward, because a modeled valid score should not be worse than the failure sentinel.

The concrete weak prior is

$$
\kappa_0=0.5,
\qquad
\alpha_0=2.5,
\qquad
\beta_0=(\alpha_0-1)s_{\mathrm{pool}}^2,
$$

centered at the pooled valid-pilot mean. If the pooled sample standard deviation degenerates, the scale falls back to 5% of a reference scale formed from the pooled mean, its distance from failure, and the observed reward span. This prevents one valid observation from implying zero predictive variance.

#### Allocation objective

For scenario \(\omega\), let \(M_\omega(a_1,\ldots,a_S)\) be the best reward among the incumbent pilots and the additional simulated samples. The objective is

$$
F(a_1,\ldots,a_S)
=\mathbb E_\omega[M_\omega(a_1,\ldots,a_S)].
$$

Each budget unit is assigned greedily to the strategy with greatest marginal gain

$$
\Delta_k(a)=F(a+e_k)-F(a).
$$

Expected maximum is monotone and exhibits diminishing returns under the scenario construction, making greedy allocation a natural fixed-budget approximation.

The expectation is represented by 4,096 shared posterior scenarios. Sharing scenarios across marginal comparisons is important: it reduces comparison noise and makes each proposed allocation face the same posterior worlds.

If every pilot is invalid, the arms are treated symmetrically and the allocation is equal. If numerical marginal gains vanish, remaining units are balanced rather than assigned by arbitrary floating-point noise.

This allocator matches the global best-of-search objective better than a mean-reward bandit. Its main weakness is model dependence: Gaussian tails and pooled priors can be poor descriptions of sparse, multimodal program rewards.

### 7.3 Hurdle allocation

The hurdle allocator focuses specifically on **improvement over the selected parent**.

For parent reward \(r_p\), define nonnegative gain

$$
Z=\max(R-r_p,0)
$$

for valid programs, and \(Z=0\) otherwise. The distribution has a point mass at zero and a positive continuous component:

$$
Z_k\sim
\begin{cases}
0,&\text{with probability }1-\pi_k,\\
G_k^+,&\text{with probability }\pi_k.
\end{cases}
$$

This separates two scientific questions:

1. how likely is the strategy to beat the parent?
2. conditional on beating it, how large is the gain?

#### Improvement probability

If \(m_k\) of \(H\) pilots have strictly positive gain, a uniform-prior posterior gives

$$
\pi_k\mid\mathcal D
\sim\operatorname{Beta}(m_k+1,H-m_k+1),
$$

with posterior mean

$$
\mathbb E[\pi_k\mid\mathcal D]
=\frac{m_k+1}{H+2}.
$$

This is improvement probability, not validity probability. A valid program that fails to beat the parent belongs to the zero hurdle.

#### Positive-gain model

Positive gains are normalized by the largest observed positive gain so that allocation is invariant to a global multiplicative reward scale. Their magnitudes are represented by a reflected Gaussian-kernel mixture on the positive half-line, with a shared Silverman-style bandwidth.

For pooled normalized positive gains \(g_1,\ldots,g_n\), let

$$
\hat s=\min\!\left(operatorname{sd}(g),
\frac{Q_{0.75}-Q_{0.25}}{1.349}\right)
$$

when both terms are informative, with the positive mean as the zero-spread fallback. The bandwidth is

$$
h=0.9\,\hat s\,n^{-1/5}.
$$

A kernel centered at \(c>0\) uses reflection at zero. Its CDF is

$$
K_c(z)=
\Phi\!\left(\frac{z-c}{h}\right)
+\Phi\!\left(\frac{z+c}{h}\right)-1,
\qquad z\ge0.
$$

Reflection gives positive support without creating a hard upper endpoint.

The posterior mixture contains:

- a zero component;
- a weak pooled positive-gain component;
- components centered on the arm’s own positive pilots.

A Bayesian-bootstrap/Dirichlet construction gives these components uncertainty rather than fixing one empirical CDF. The concentration is chosen so that the implied expected positive mass agrees with the Beta posterior mean above.

More exactly, with \(m_k\) positive pilots out of \(H\), the Dirichlet concentrations are

$$
(H-m_k+1,\ 1,\ \underbrace{1,\ldots,1}_{m_k\text{ own positive pilots}}).
$$

The first entry is the zero component, the second is the pooled positive component, and the remainder are the arm’s observations. Their total positive concentration is \(m_k+1\), so the positive-mass marginal is exactly \(\operatorname{Beta}(m_k+1,H-m_k+1)\).

#### Why the CDF moment matters

Future observations from one arm share the same uncertain arm parameters. If \(F_k(z)\) is a posterior-random CDF, the probability that \(a\) future draws all fail to exceed \(z\) is

$$
M_{k,a}(z)=\mathbb E\!\left[F_k(z)^a\right],
$$

not

$$
\left(\mathbb E[F_k(z)]\right)^a.
$$

The distinction preserves posterior dependence between future draws. Replacing the moment of a power by the power of a mean would incorrectly treat the unknown arm distribution as independently redrawn for each rollout.

#### Expected improvement of the final maximum

Let the incumbent be the better of the parent and all valid pilot outcomes:

$$
r_{\mathrm{inc}}
=\max\!\left(r_p,\max_{k,i}R_{k,i}^{\mathrm{valid}}\right).
$$

In gain coordinates, let \(z_{\mathrm{inc}}=r_{\mathrm{inc}}-r_p\). For allocation vector \(a\), expected improvement of the final maximum is

$$
\operatorname{EI}(a)
=\int_{z_{\mathrm{inc}}}^{\infty}
\left[1-\prod_{k=1}^{S}M_{k,a_k}(z)\right]dz.
$$

The next unit goes to the arm with the greatest marginal increase in this integral. Numerical quadrature evaluates the positive tail deterministically.

The integral uses 128-point Gauss-Legendre quadrature from the best observed normalized gain to twelve bandwidths beyond it. The omitted reflected-Gaussian tail is negligible under the model. Tied marginal gains are broken in favor of the currently less-allocated arm, then by stable arm order.

If no pilot beats the parent, the allocator does not infer a preference from degrees of failure or mere validity. It allocates equally. This is a deliberate conservatism: without a positive signal, apparent differences among failures can be artifacts of the reward sentinel or formatting reliability rather than evidence of scientific upside.

The hurdle model is the most directly aligned with local discovery. Its cost is greater modeling complexity, and it can be conservative when improvements are extremely rare but pilot failures still contain useful ordinal information.

### 7.4 Allocation comparison

| Method | Event modeled | Quantity optimized | Best property | Main limitation |
|---|---|---|---|---|
| Rule-based | Pilot mean and variance | Heuristic weighted count | Simple and transparent | No probabilistic semantics |
| Bandit | Validity plus absolute reward | Expected final maximum | Uses all absolute reward evidence | Parametric tail assumptions |
| Hurdle | Beat-parent event plus positive gain | Expected improvement of maximum | Matches local discovery goal | Conservative with no positive pilots |

---

## 8. The coder as a stochastic policy

The coder defines an autoregressive distribution over response tokens:

$$
\pi_\theta(y\mid c)
=\prod_{t=1}^{L}
\pi_\theta(y_t\mid c,y_{<t}),
$$

where context \(c\) contains the problem, parent, and optional strategy.

Only generated response tokens are policy actions. Prompt tokens are conditioning information. If a token is externally forced to enforce a reasoning boundary or output transition, it is excluded from policy likelihood, divergence, entropy, and loss. A forced token was not sampled from \(\pi_\theta\) and therefore cannot carry an honest policy gradient.

### 8.1 Why clipped-policy modes use untruncated unit-temperature sampling

For behavior-policy correction and PPO ratios to be valid, the logged sampling distribution must match the policy probabilities later scored by the model. Therefore clipped SPO-RS modes use the native model distribution:

$$
T=1,\qquad p_{\mathrm{top}}=1,
\qquad k_{\mathrm{top}}=0,
\qquad p_{\min}=0.
$$

Temperature scaling or nucleus/top-k truncation would define a different behavior distribution. Using raw model log probabilities as if they described that altered sampler would produce incorrect importance ratios unless the truncation normalization were modeled exactly.

### 8.2 One-shot generation and external evaluation

The coder does not observe execution feedback while constructing a response. It samples a complete program, after which evaluation occurs. Thus the action is a long sequence with terminal reward, and every sampled token receives the same rollout-level advantage. This is coarse credit assignment, but it avoids inventing intermediate rewards that may not reflect scientific correctness.

---

## 9. Entropic leave-one-out training

Before SPO-RS, the system’s baseline objective is a within-parent entropic comparison.

Suppose a parent group contains \(K\) rollout rewards \(r_1,\ldots,r_K\). Define the exponentially tilted distribution

$$
q_\beta(i)
=\frac{\exp(\beta r_i)}{\sum_{j=1}^{K}\exp(\beta r_j)}.
$$

### 9.1 Adaptive inverse temperature

Rather than fixing \(\beta\), choose it so that the tilt has a target divergence from the uniform distribution:

$$
D_{\mathrm{KL}}\!\left(q_\beta\,\|\,U_K\right)=\gamma,
$$

with the default target

$$
\gamma=\log 2.
$$

The solution is found over \(\beta\ge0\). This makes the concentration of credit approximately invariant to affine reward scale: large reward ranges produce smaller \(\beta\), while tiny ranges produce larger \(\beta\).

The search doubles an upper bracket and then bisects it. If the requested \(\gamma\) is unattainable because the tilted distribution has saturated—for example, because several samples tie for the maximum—\(\beta\) is capped at an effectively argmax value. The target must also respect the theoretical ceiling \(\log K\), reached only by a unique point mass.

If all rewards are equal, no \(\beta\) can create meaningful ordering and every advantage is zero.

### 9.2 Leave-one-out advantage

Let \(r_{\max}=\max_jr_j\) and

$$
e_i=\exp\!\left(\beta(r_i-r_{\max})\right).
$$

For candidate \(i\), normalize against the other \(K-1\) samples:

$$
Z_{-i}=\frac{1}{K-1}\sum_{j\ne i}e_j.
$$

The advantage is

$$
A_i=\frac{e_i}{Z_{-i}+\varepsilon}-1.
$$

Leave-one-out normalization prevents a sample from helping to define its own baseline. A strong winner is compared against its siblings rather than against a denominator it substantially inflated.

### 9.3 Meaning and limitations

Entropic training answers:

> Which rollouts were unusually good relative to other rollouts from this same parent in this same step?

Its strengths are local scale adaptation and direct sibling comparison. Its weaknesses are equally important:

- a uniformly bad group can still have a positive local winner;
- a uniformly excellent tied group produces no gradient;
- results from different parents have no shared baseline;
- invalid rollouts may be skipped when the group has no variation;
- the objective changes with group composition.

### 9.4 Reference regularization

In standard entropic mode, token-level drift from a fixed reference model may be regularized. Let

$$
d_t=\log\pi_\theta(y_t\mid h_t)
-\log\pi_{\mathrm{ref}}(y_t\mid h_t).
$$

A centered per-token correction is

$$
A^{\mathrm{KL}}_t
=\lambda_{\mathrm{KL}}\left(\bar d-d_t\right),
\qquad
\bar d=\frac1L\sum_{u=1}^{L}d_u.
$$

The effective token advantage is \(A_i+A^{\mathrm{KL}}_t\). Centering avoids adding a uniform trajectory-level offset while discouraging tokens whose reference log-ratio is unusually large.

This reference regularizer is not used by SPO-RS. SPO-RS instead uses clipped policy updates and drift-adaptive historical memory.

### 9.5 Behavior-policy correction in entropic mode

If the policy has changed between generation and optimization, standard entropic training uses a detached token importance weight

$$
I_t=\exp\!\left(
\log\pi_\theta(y_t\mid h_t)
-\log\pi_b(y_t\mid h_t)
\right).
$$

The token objective is approximately

$$
\mathcal L_{mathrm{entropic}}
=-\mathbb E\left[
I_t\,\operatorname{stopgrad}
\left(A_i+A_t^{\mathrm{KL}}\right)
\log\pi_\theta(y_t\mid h_t)
\right].
$$

The importance weight is detached: it corrects weighting but is not itself differentiated. If exact behavior scores are unavailable, the weight is one. For the sequence-ratio model variant, token ratios are replaced by the trajectory geometric-mean ratio and the reward term uses the clipped sequence surrogate described later.

---

## 10. SPO-RS

SPO-RS replaces the within-group adaptive baseline with one frozen, run-wide historical baseline.

Let \(\beta>0\) be fixed and define transformed reward

$$
g(r)=\exp(\beta r).
$$

Before a discovery step, let \(V_t>0\) be the historical mean of transformed reward. Every rollout in the step receives

$$
A_t(r)
=\frac{g(r)/V_t-1}{\beta}
=\frac{\exp(\beta r)/V_t-1}{\beta}.
$$

### 10.1 Interpretation

Define the certainty-equivalent historical level

$$
c_t=\frac{1}{\beta}\log V_t.
$$

Then

$$
A_t(r)=\frac{\exp(\beta(r-c_t))-1}{\beta}.
$$

Therefore:

- \(A_t(r)=0\) when \(r=c_t\);
- \(A_t(r)>0\) above the historical entropic level;
- \(A_t(r)<0\) below it;
- the positive tail grows exponentially.

As \(\beta\to0\), a Taylor expansion gives

$$
A_t(r)\approx r-c_t,
$$

so SPO-RS approaches an ordinary centered reward advantage. Larger \(\beta\) places progressively more credit on exceptional outcomes.

An additive reward shift \(r\mapsto r+c\) multiplies both \(g(r)\) and \(V_t\) by \(e^{\beta c}\), leaving the ratio unchanged. A multiplicative reward rescaling changes the effective temperature and therefore changes the objective.

The certainty equivalent \(c_t=\beta^{-1}\log V_t\) is the positive-temperature entropic risk functional. Because reward is being maximized and \(\beta>0\), it is **upper-tail seeking**: unusually high rewards matter more than equally sized downward deviations. As \(\beta\to\infty\), log-sum-exp approaches a maximum over the retained empirical support. At finite \(\beta\), however, SPO-RS is neither a hard maximum nor Conditional Value at Risk (CVaR); it applies a smooth exponential utility to every reward.

### 10.2 Why fixed \(\beta\)?

A fixed \(\beta\) gives “one reward unit” a stable meaning throughout the run. Adaptive within-batch temperature keeps rank concentration stable but changes the utility curve from group to group. Fixed exponential utility instead allows comparisons across parents and time.

The cost is sensitivity to reward scale. If \(\beta\) is too small, rare improvements receive little extra weight. If it is too large, a few rewards dominate and numerical overflow becomes possible.

### 10.3 Global rather than group-local credit

Every rollout uses the same \(V_t\), regardless of parent or strategy. Consequently:

- a weak winner from a poor group can remain negative;
- a tied group can still train if its common reward differs from history;
- failures always carry their historical meaning;
- rewards from different parents are comparable in the objective.

All rollouts, including invalid ones, remain training examples. Failure reward is not filtered away.

### 10.4 Freeze-before-use principle

The baseline for step \(t\) is fixed before the current rollouts are assigned advantages. Current rewards update history only after all current advantages are formed.

This prevents a rollout from changing the baseline used to judge itself. Formally, \(V_t\) is measurable with respect to history \(\mathcal H_{t-1}\), while rewards \(R_t\) are newly sampled:

$$
V_t\in\sigma(\mathcal H_{t-1}),
\qquad
A_t=A(R_t;V_t).
$$

### 10.5 Initialization from verified seeds

A fresh run needs a baseline before step 0. Verified seed rewards \(r_1^{(0)},\ldots,r_M^{(0)}\) supply it:

$$
V_0=\frac1M\sum_{i=1}^{M}e^{\beta r_i^{(0)}}.
$$

This enables a genuine first update without leaking step-0 rewards into their own advantages.

The initial effective count is inflated to

$$
N_0=\frac{M}{1-\rho_{\min}}.
$$

This treats the seed batch as an established baseline rather than a single fragile minibatch. It is equivalent to giving each seed the steady-state weight induced by minimum retention.

---

## 11. Policy-drift-adaptive historical memory

The coder changes after each update, so old rewards came from a different policy. SPO-RS discounts history according to consecutive-policy divergence.

### 11.1 Forward-KL estimate

Let \(\pi_t\) be the current sampling policy and \(\pi_{t-1}\) the preceding one. On a trajectory \(y\sim\pi_t(\cdot\mid c)\), the log-ratio sum

$$
\widehat D(y,c)
=\sum_{u\in\mathcal A(y)}
\left[
\log\pi_t(y_u\mid h_u)
-\log\pi_{t-1}(y_u\mid h_u)
\right]
$$

estimates

$$
D_{\mathrm{KL}}
\left(\pi_t(\cdot\mid c)\,\|\,\pi_{t-1}(\cdot\mid c)\right).
$$

The action index set \(\mathcal A(y)\) excludes forced tokens.

Trajectories are averaged within each selected parent context, then contexts are averaged equally:

$$
\widehat D_t
=\frac1{|\mathcal C_t|}
\sum_{c\in\mathcal C_t}
\frac1{n_c}\sum_{i=1}^{n_c}\widehat D(y_i,c).
$$

Equal context weighting prevents a parent with more allocated rollouts from dominating the drift estimate. Because Monte Carlo noise can make the estimate slightly negative, the finite result is clamped below by zero.

If any required context lacks a valid consecutive-policy score, divergence is treated as infinite. Missing evidence triggers conservative forgetting rather than silently preserving stale history.

### 11.2 Half-life retention law

Let \(D_{1/2}>0\) be the divergence at which historical mass should halve. Raw retention is

$$
\rho_t^{\mathrm{raw}}
=2^{-\widehat D_t/D_{1/2}}.
$$

Then

$$
\rho_t
=\operatorname{clip}
\left(\rho_t^{\mathrm{raw}},
\rho_{\min},
\rho_{\max}\right).
$$

This has an interpretable scale:

$$
\widehat D_t=D_{1/2}
\quad\Longrightarrow\quad
\rho_t^{\mathrm{raw}}=\frac12.
$$

The bounds prevent two pathologies:

- \(\rho_{\min}>0\) prevents a single noisy divergence estimate from erasing all history;
- \(\rho_{\max}<1\) guarantees eventual forgetting even when consecutive policies are nearly identical.

Missing or infinite divergence uses \(\rho_{\min}\).

### 11.3 Baseline update

Let \(N_t\) be effective historical count before the update and let the current step contain \(M_t\) rewards. After advantages have been computed,

$$
N_{t+1}=\rho_tN_t+M_t,
$$

$$
V_{t+1}
=\frac{\rho_tN_tV_t+
\sum_{i=1}^{M_t}e^{\beta r_{t,i}}}
{N_{t+1}}.
$$

The effective contribution of the new step is

$$
\eta_t=\frac{M_t}{N_{t+1}}.
$$

Policy drift therefore controls memory length in behavior space rather than wall-clock steps. A nearly unchanged policy retains a long baseline; a substantially changed policy forgets faster.

### 11.4 Why not a fixed moving average?

A fixed time decay assumes that one training step always changes the sampling distribution by the same amount. That is false when gradient magnitude, clipping, data quality, or optimizer state vary. Divergence-based decay asks the more relevant question: “How comparable are old rewards to samples from the current coder?”

The forward-KL estimate is still local to sampled contexts. It cannot measure changes in unvisited prompt regions, so it is a pragmatic nonstationarity signal rather than a global policy metric.

Because token log-ratios are summed, this is a full-response divergence rather than a per-token normalized divergence. Longer action sequences can therefore contribute more drift. That is appropriate if memory should respond to the change in the probability of complete programs, but it also couples the half-life scale to typical response length.

---

## 12. Bounded historical shaping variant

An optional SPO-RS variant replaces raw exponential reward with a bounded, history-calibrated upper-tail utility. It is selected under the SPO-RS family but is mathematically a different surrogate.

Using only previously observed successful rewards \(r>r_{\mathrm{fail}}\), define weighted historical quantiles

$$
b=Q_{0.90},
\qquad
s=Q_{0.90}-Q_{0.75}.
$$

If this scale degenerates, progressively broader historical gaps or a neutral positive scale are used. The frozen utility is

$$
u(r)=\frac12+rac1\pi
\arctan\!\left(\frac{r-b}{s}\right),
$$

so that

$$
0<u(r)<1.
$$

The transformed reward is

$$
g_{\mathrm{bounded}}(r)
=\exp\!\left(1.25\,\beta\,u(r)\right),
$$

and advantage remains

$$
A(r)=\frac{g_{\mathrm{bounded}}(r)/V-1}{\beta}.
$$

Failures are excluded only from quantile calibration. They remain in the historical mean and remain training examples.

The scale fallback order is: \(Q_{0.90}-Q_{0.75}\), the preceding valid scale when available, \(Q_{0.75}-Q_{0.25}\), the full successful-reward range, and finally \(\max(1,|b-r_{\mathrm{fail}}|)\). This prevents a tied upper tail from collapsing the transform.

### 12.1 Rationale

This transform has three useful properties:

- it is strictly monotone, so better rewards always receive higher utility;
- its sensitivity is concentrated near the historical upper tail;
- its bounded range prevents extreme raw reward scales from exploding the exponential.

The method is not the original \(e^{\beta r}\) objective and is not an exact maximum operator. It optimizes a moving, rank-like upper-tail surrogate. Because \(b\) and \(s\) depend on history, the meaning of a raw reward can change across steps.

The entire retained raw history is reinterpreted under the newly frozen calibration. This avoids averaging numbers produced by incompatible transforms.

---

## 13. SPO-RS-v2

SPO-RS-v2 keeps fixed-exponential SPO-RS and adds a validity-aware, historical pairwise-win advantage.

It is not combined with bounded shaping. Its base term is

$$
A^{\mathrm{base}}_t(r)
=\frac{e^{\beta r}/V_t-1}{\beta}.
$$

### 13.1 Decayed historical validity mass

Let historical observations have drift-decayed weights \(w_j\), rewards \(R_j\), and validity flags \(V_j\in\{0,1\}\). Define

$$
m_v
=\frac{\sum_jw_jV_j}{\sum_jw_j}.
$$

This is the effective historical probability of validity under the same memory system as the base baseline.

Only valid historical rewards form the pairwise reference distribution:

$$
R^{\mathrm{ref}}\sim
\frac{w_jV_j}{\sum_\ell w_\ell V_\ell}.
$$

### 13.2 Soft pairwise win probability

For a candidate reward \(r\), define

$$
p_{\mathrm{win}}(r)
=\mathbb E_{R^{\mathrm{ref}}}
\left[
\sigma\!\left(\frac{r-R^{\mathrm{ref}}}{s_w}\right)
\right],
$$

where

$$
\sigma(a)=\frac{1}{1+e^{-a}}.
$$

This is a smoothed probability of beating a valid historical draw. It is ordinal in spirit but differentiates near-ties continuously.

Statistically, it is a smoothed analogue of a Mann–Whitney win probability or area-under-the-ranking-curve statistic. Pure pairwise order would be invariant under any monotone reward transform; the logistic smoothing and quantile scale make the implemented feature approximately invariant to affine rescaling, not to every nonlinear transform.

### 13.3 Win scale

From valid historical rewards, compute weighted \(Q_{0.25}\), \(Q_{0.75}\), and \(Q_{0.90}\). Select the first numerically meaningful positive gap among

$$
Q_{0.90}-Q_{0.75},
\qquad
Q_{0.75}-Q_{0.25},
\qquad
\max R-\min R.
$$

Then set

$$
s_w=\frac{\mathrm{gap}}{\log3}.
$$

A reward difference equal to the selected gap therefore yields

$$
\sigma(\log3)=0.75.
$$

A machine-precision floor prevents meaningless scales, and the scale may change by at most a factor of two per historical update. This avoids violently changing the ordinal term when quantiles move or the valid history is sparse.

If there are fewer than two distinct valid references, the win term is disabled.

### 13.4 Centered validity-aware win feature

For candidate validity \(v\), define

$$
W(r,v)=v\,p_{\mathrm{win}}(r)-\frac{m_v}{2}.
$$

An invalid candidate receives

$$
W(r,0)=-\frac{m_v}{2}.
$$

A valid candidate that is a neutral match to history has approximately

$$
W(r,1)\approx\frac12-\frac{m_v}{2}.
$$

The centering is not arbitrary. If \(R\) and \(R'\) are independent draws from the same valid reference distribution, symmetry gives

$$
\mathbb E\left[
\sigma\!\left(\frac{R-R'}{s_w}\right)
\right]=\frac12.
$$

Including historical validity probability,

$$
\mathbb E[V\,p_{\mathrm{win}}(R)]
=\frac{m_v}{2},
$$

so the historical expectation of \(W\) is approximately zero.

### 13.5 RMS calibration

The base advantage and win feature have unrelated units. SPO-RS-v2 chooses their mixing coefficient by historical root-mean-square magnitude.

Let

$$
\operatorname{RMS}_{\mathrm{base}}
=\sqrt{
\frac{\sum_jw_j(A_j^{\mathrm{base}})^2}
{\sum_jw_j}
},
$$

and

$$
\operatorname{RMS}_{\mathrm{win}}
=\sqrt{\mathbb E[W^2]}.
$$

Then

$$
\lambda_w
=\alpha\,
\frac{\operatorname{RMS}_{\mathrm{base}}}
{\operatorname{RMS}_{\mathrm{win}}},
$$

when both RMS values are positive. The final advantage is

$$
A^{\mathrm{v2}}(r,v)
=A^{\mathrm{base}}(r)+\lambda_wW(r,v).
$$

Thus \(\alpha\) is interpretable as the win term’s target RMS size relative to the base advantage, not as a coefficient in raw reward units.

The win RMS is approximated with deterministic weighted-quantile representatives of valid history, while explicitly including the invalid historical mass in the second moment. This keeps calibration bounded in cost without random Monte Carlo noise.

At most 256 equally spaced weighted-quantile representatives are used for this calibration.

If \(\alpha=0\), the valid reference is degenerate, or either RMS vanishes, then \(\lambda_w=0\) and v2 reduces to base SPO-RS for that step.

### 13.6 What v2 adds

Base SPO-RS asks:

> How large is this reward under a fixed exponential utility relative to historical transformed reward?

The v2 term additionally asks:

> Was it valid, and how often would its reward beat a valid historical result?

The base term preserves cardinal magnitude. The win term adds a robust ordinal signal and an explicit invalidity penalty. Neither subsumes the other.

### 13.7 Tradeoffs

- When historical validity \(m_v\) is high, invalid outputs receive a stronger negative win feature.
- When \(m_v\) is low, the invalid penalty is smaller because validity itself is historically rare.
- A very small win scale can exaggerate tiny reward differences; quantile floors and factor-two smoothing limit this.
- Pairwise probability depends on the retained history, so the same raw reward can receive a different ordinal meaning later.
- A candidate can have a positive cardinal advantage but negative win contribution, or vice versa. That disagreement is intentional information.

---

## 14. Comparing entropic training, SPO-RS, bounded SPO-RS, and SPO-RS-v2

| Property | Entropic LOO | SPO-RS | Bounded SPO-RS | SPO-RS-v2 |
|---|---|---|---|---|
| Comparison set | Current sibling group | Decayed global history | Decayed global history | Decayed global history |
| Reward transform | Adaptive \(e^{\beta r}\) | Fixed \(e^{\beta r}\) | Bounded quantile-calibrated transform | Fixed \(e^{\beta r}\) plus soft win |
| Temperature | Chosen per group by target KL | Fixed | Fixed after bounded utility | Fixed |
| Validity used explicitly | No | No | Only to exclude failures from calibration | Yes |
| Tied current group can train | Usually no | Yes | Yes | Yes |
| Invalid rollouts retained | Depends on group signal | Yes | Yes | Yes |
| Historical nonstationarity | None | Policy-KL decay | Policy-KL decay | Policy-KL decay |
| Reference-model KL | Optional | Off | Off | Off |
| Main bias | Local relative winners | Cardinal upper tail | Historical upper-tail rank | Cardinal reward plus validity-aware rank |

No method is universally superior:

- Entropic LOO is scale-adaptive and local, useful when reward scales vary unpredictably across groups.
- Fixed SPO-RS has the clearest run-wide cardinal semantics when reward calibration is trustworthy.
- Bounded shaping is robust when raw magnitudes are unstable or heavy-tailed.
- SPO-RS-v2 is useful when validity and “beat historical work” carry information not captured by reward magnitude alone.

---

## 15. The clipped coder update

SPO-RS variants use an asymmetric PPO-style clipped surrogate.

Let \(\pi_b\) be the frozen behavior policy that generated a rollout and \(\pi_\theta\) the policy being optimized. For sampled token \(y_t\),

$$
\rho_t(\theta)
=\exp\!\left(
\log\pi_\theta(y_t\mid h_t)
-\log\pi_b(y_t\mid h_t)
\right).
$$

With lower and upper clipping tolerances \(\epsilon_-,\epsilon_+>0\), define

$$
\bar\rho_t
=\operatorname{clip}
\left(\rho_t,1-\epsilon_-,1+\epsilon_+\right).
$$

For rollout advantage \(A\), the token surrogate is

$$
L_t^{\mathrm{clip}}(\theta)
=\min\!\left(ho_tA,\bar\rho_tA\right).
$$

The loss minimizes

$$
\mathcal L(\theta)
=-\mathbb E_{\mathrm{rollouts}}
\left[
\frac1L\sum_{t=1}^{L}L_t^{\mathrm{clip}}(\theta)
\right].
$$

Trajectories are weighted equally in the global SPO-RS batch, regardless of parent or strategy. Token averaging prevents long programs from receiving proportionally more total influence merely because they contain more tokens.

### 15.1 Why asymmetric clipping?

The interval

$$
[1-\epsilon_-,1+\epsilon_+]
$$

allows separate limits on decreasing and increasing sampled-token probability. A wider upper tolerance can permit faster reinforcement of rare successful programs while a tighter lower tolerance limits how aggressively failures suppress behavior. The exact tradeoff depends on the chosen values.

For \(A>0\), clipping limits the benefit of increasing probability too far. For \(A<0\), clipping limits the benefit of decreasing probability too far. This creates a local trust region around the generating policy.

### 15.2 Sequence-level ratio variant

For a model whose tokenwise routing makes a stable token ratio unsuitable, the ratio may be formed from the geometric mean sequence likelihood:

$$
\rho_{\mathrm{seq}}
=\exp\!\left[
\frac1L\sum_{t=1}^{L}
\left(log\pi_\theta(y_t\mid h_t)
-\log\pi_b(y_t\mid h_t)\right)
\right].
$$

The same ratio is then used for all tokens in the trajectory. This sacrifices token-local correction for a stable sequence-level trust signal.

### 15.3 Why no reference KL in SPO-RS?

SPO-RS already has two stabilizers:

1. the update is clipped relative to the behavior policy;
2. history is discounted when the policy moves.

A fixed-reference KL would add a separate objective to stay near the original model, potentially opposing discovery-specific specialization. The design chooses local update control and adaptive memory instead.

This does not guarantee global closeness to the base model. Repeated clipped steps can accumulate substantial drift.

---

## 16. How search, allocation, and learning interact

The full process is a feedback system:

$$
\text{archive}
\rightarrow\text{parents}
\rightarrow\text{strategies}
\rightarrow\text{programs}
\rightarrow\text{evaluations}
\rightarrow\text{policy update and archive}.
$$

### 16.1 Different layers optimize related but nonidentical quantities

- Parent selection approximates which branch may produce a future best child.
- Strategy allocation approximates which plan may improve the final maximum for one parent.
- SPO-RS raises likelihood according to transformed historical reward.
- SPO-RS-v2 also raises likelihood for validity-aware historical wins.
- Archive pruning retains a bounded set of strong valid states.

These objectives are aligned toward discovery, but they are not one exact scalar objective optimized end to end.

### 16.2 Adaptive data distribution

The training distribution at step \(t\) is

$$
d_t(c,y)
=d_t(c)\pi_{\theta_t}(y\mid c),
$$

where \(d_t(c)\) is itself induced by the current archive, parent picker, strategist, and allocator. PPO ratios correct the change from the behavior policy to the updated coder **conditional on sampled contexts**. They do not importance-correct the adaptive context distribution.

That omission is intentional. The evolving context distribution is the discovery curriculum, not a nuisance distribution to be held fixed.

### 16.3 Positive feedback and collapse risk

A strong branch receives more expansions, creates more training data, and may become even more likely under the coder. The same loop can cause premature collapse around one family of methods.

Countermeasures act at different layers:

- PUCT or UCT visit bonuses revisit underexplored states;
- lineage blocking diversifies a batch;
- sequential strategist prompts request materially different plans;
- strategy allocation reserves pilots for every plan;
- JEV filters same-core parents;
- PPO clipping limits a single update;
- archive caps prevent unbounded flooding.

None guarantees global exploration. Together they make collapse less immediate while retaining strong exploitation.

### 16.4 Revisiting old parents is rational

Failure to improve a parent under policy \(\pi_t\) does not prove the parent is exhausted under \(\pi_{t+k}\). The coder changes, so the child distribution changes:

$$
\pi_{\theta_t}(y\mid x,p,z)
\ne
\pi_{\theta_{t+k}}(y\mid x,p,z).
$$

Parent pickers therefore may revisit old states. Visit penalties discourage immediate repetition but do not permanently blacklist a branch.

### 16.5 Local improvement versus absolute quality

The hurdle allocator rewards beating a particular parent, while archive pruning and SPO-RS operate on absolute internal reward. A modest improvement to an elite parent can be globally excellent; a large improvement to a weak parent can remain mediocre.

Using both perspectives is useful:

- local gain detects whether a strategy works in context;
- absolute reward determines scientific quality and historical policy credit.

---

## 17. Diversity and entropy diagnostics

Token entropy, source-code diversity, strategy diversity, and semantic parent diversity measure different things.

For token distribution \(\pi_\theta(\cdot\mid h_t)\), full-vocabulary entropy is

$$
H_t=-\sum_{a}\pi_\theta(a\mid h_t)
\log\pi_\theta(a\mid h_t).
$$

High token entropy means the next token is uncertain. It does not imply scientifically different programs. Conversely, two low-entropy generations can implement different algorithms if their contexts differ.

Useful distinctions are:

- **token diversity:** uncertainty in local wording or syntax;
- **program diversity:** textual difference among complete outputs;
- **strategy diversity:** different stated high-level plans;
- **semantic diversity:** different core algorithms;
- **outcome diversity:** different constructions or rewards.

These quantities are diagnostics, not direct optimization objectives in SPO-RS. JEV is the component that explicitly acts on semantic diversity.

---

## 18. Hyperparameters as theoretical controls

### 18.1 Search controls

| Symbol or name | Meaning | Increasing it tends to… |
|---|---|---|
| \(c\) | PUCT/UCT exploration constant | Favor less-visited states |
| \(K\) | Children retained per parent | Preserve more local alternatives |
| Archive capacity | Total non-seed memory | Preserve more branches but weaken pruning |
| Strategy top-\(r\) | Per-parent/strategy archive limit | Let one strategy occupy more archive slots |
| JEV window \(X\) | PUCT candidates considered per desired parent | Search farther for semantic variety |
| JEV threshold \(\tau\) | Same-core rejection threshold | With larger \(\tau\), reject fewer candidates |

### 18.2 Strategy controls

| Symbol | Meaning | Increasing it tends to… |
|---|---|---|
| \(S\) | Strategies per parent | Broaden high-level search |
| \(M\) | Total planned programs per strategy before reallocation | Increase evidence and cost |
| \(H\) | Pilots per strategy | Improve arm estimates but leave fewer adaptive units |
| \(B=S(M-H)\) | Adaptive continuation budget | Increase exploitation after pilots |

Small \(H\) makes posterior allocation reactive but noisy. Large \(H\) makes it stable but nearly uniform because little budget remains to adapt.

### 18.3 SPO-RS controls

| Symbol | Meaning | Increasing it tends to… |
|---|---|---|
| \(\beta\) | Exponential reward sensitivity | Concentrate credit on exceptional rewards |
| \(D_{1/2}\) | KL half-life scale | Retain history despite larger policy movement |
| \(\rho_{\min}\) | Minimum retention | Make catastrophic forgetting less possible |
| \(\rho_{\max}\) | Maximum retention | Lengthen memory under stable policies |
| \(\epsilon_-\) | Allowed probability decrease | Permit stronger suppression of negative samples |
| \(\epsilon_+\) | Allowed probability increase | Permit stronger reinforcement of positive samples |
| \(\alpha\) | v2 win term relative RMS | Give validity/rank more influence |

These parameters interact. For example, large \(\beta\) and large \(\epsilon_+\) can strongly reinforce a rare winner, while low \(\rho_{\max}\) makes the historical baseline adapt quickly afterward.

### 18.4 Entropic controls

| Symbol | Meaning | Increasing it tends to… |
|---|---|---|
| \(\gamma\) | Target KL from uniform | Concentrate within-group credit more sharply |
| \(\lambda_{\mathrm{KL}}\) | Reference correction | Resist token-level drift from the base model |
| Group size \(K\) | Siblings compared | Improve local ranking evidence at higher cost |

---

## 19. Key invariants

The theory relies on several invariants. Violating them changes the meaning of the algorithms.

1. **Higher internal reward always means better.** Native minimization scores must be transformed.
2. **Validity is evaluator-defined.** Model claims do not determine success.
3. **Current rewards do not define their own SPO-RS baseline.** History is frozen before advantage computation.
4. **Consecutive-policy divergence is measured on current-policy samples.** Otherwise it is not the intended forward KL.
5. **Forced tokens are not actions.** They are excluded from ratios, KL, loss, and entropy.
6. **Behavior probabilities match the actual sampler.** Clipped modes avoid unmodeled truncation and temperature changes.
7. **Pilot allocation is fixed-budget.** It redistributes programs without changing the parent’s planned total.
8. **Every strategy gets pilots.** Adaptive exploitation begins only after common initial evidence.
9. **Invalid rollouts remain policy evidence.** Recovery does not erase originals.
10. **Only valid candidates become parents.** Search state and training data have different admission rules.
11. **Best-ever survives archive pruning.** Memory management cannot revoke a discovery.
12. **SPO-RS-v2 calibrates from prior history only.** Current rewards do not set their own win scale or mixing weight.

---

## 20. Failure modes and what each safeguard addresses

### 20.1 Reward-scale failure

**Risk:** fixed \(\beta\) is too large or small for the task.

**Safeguards:** task-level reward normalization; bounded shaping as an alternative; monitoring transformed-reward and advantage magnitudes.

### 20.2 Historical staleness

**Risk:** rewards from an old policy set an irrelevant baseline.

**Safeguard:** forward-KL half-life decay with conservative forgetting on missing scores.

### 20.3 Winner overfitting

**Risk:** one noisy high reward dominates learning and parent selection.

**Safeguards:** PPO clipping, visit penalties, archive bounds, multiple parents, and semantic diversity. No mechanism fully removes stochastic winner’s curse.

### 20.4 Invalid-output collapse

**Risk:** malformed code consumes most rollout budget.

**Safeguards:** invalids receive failure reward; v2 adds validity-aware penalty; limited format recovery protects catastrophic batches without hiding failures.

### 20.5 Strategy monoculture

**Risk:** one plan family occupies most samples and archive states.

**Safeguards:** sequential different-plan prompting, common pilots, per-strategy archive limits, posterior allocation rather than immediate commitment.

### 20.6 Parent monoculture

**Risk:** all selected parents represent the same algorithmic basin.

**Safeguards:** lineage blocking, UCT/PUCT exploration, LLM joint selection, and PUCT-JEV.

### 20.7 Misleading diversity

**Risk:** textual variation is mistaken for algorithmic novelty.

**Safeguard:** JEV judges core method rather than surface form. Its decisions remain probabilistic and can be wrong.

### 20.8 Bandit model misspecification

**Risk:** sparse program rewards are multimodal, heavy-tailed, or non-Gaussian.

**Safeguards:** explicit validity hurdle, pooled weak priors, posterior uncertainty, and a separate nonparametric positive-gain hurdle model.

### 20.9 Adaptive-curriculum bias

**Risk:** the observed dataset is dominated by what earlier search already preferred.

**Safeguards:** exploration bonuses, diversity gates, and mandatory pilots. There is no claim that the resulting data are unbiased samples from a fixed distribution.

---

## 21. End-to-end mathematical summary

At discovery step \(t\):

### 21.1 Pick parents

Select a batch \(\mathcal P_t\) using PUCT, UCT, joint LLM reasoning, or PUCT filtered by semantic diversity.

### 21.2 Propose strategies

For each \(p\in\mathcal P_t\), sample

$$
z_{p,k}\sim q_\phi
\left(z\mid x,p,z_{p,<k}\right),
\qquad k=1,\ldots,S.
$$

### 21.3 Sample pilots

For each strategy, sample \(H\) programs from the frozen behavior policy:

$$
y_{p,k,j}\sim\pi_{\theta_t}
\left(y\mid x,p,z_{p,k}\right).
$$

Evaluate each to obtain \((r_{p,k,j},v_{p,k,j})\).

### 21.4 Allocate continuations

Choose integers \(a_{p,k}\) satisfying

$$
\sum_ka_{p,k}=S(M-H)
$$

using a heuristic, posterior expected maximum, or hurdle expected improvement. Sample and evaluate those continuations under the same behavior policy.

### 21.5 Form advantages

For fixed SPO-RS:

$$
A_i=\frac{e^{\beta r_i}/V_t-1}{\beta}.
$$

For SPO-RS-v2:

$$
A_i
=\frac{e^{\beta r_i}/V_t-1}{\beta}
+\lambda_w
\left[
v_i\,\mathbb E_{R^{\mathrm{ref}}}
\sigma\!\left(\frac{r_i-R^{\mathrm{ref}}}{s_w}\right)
-\frac{m_v}{2}
\right].
$$

### 21.6 Update the coder

Optimize the clipped surrogate

$$
\max_\theta
\mathbb E_i\left[
\frac1{L_i}\sum_t
\min\left(ho_{i,t}(\theta)A_i,
\operatorname{clip}(\rho_{i,t},1-\epsilon_-,1+\epsilon_+)A_i
\right)
\right].
$$

### 21.7 Update historical memory

Estimate consecutive-policy forward KL \(\widehat D_t\), calculate

$$
\rho_t=operatorname{clip}
\left(2^{-\widehat D_t/D_{1/2}},
\rho_{\min},\rho_{\max}\right),
$$

then update effective count and transformed-reward baseline:

$$
N_{t+1}=\rho_tN_t+M_t,
$$

$$
V_{t+1}
=\frac{\rho_tN_tV_t+\sum_ie^{\beta r_i}}
{N_{t+1}}.
$$

The bounded variant substitutes its frozen transformed reward. V2 also appends raw reward and validity to the same decayed historical measure.

### 21.8 Update search

Record the best valid result before pruning. Add valid, novel candidates; apply strategy, parent, and global archive pressure; and propagate visits through selected lineages. The resulting archive defines the next parent distribution.

---

## 22. What the system is—and is not—claiming

The system can be understood as a practical synthesis of:

- tree search over program lineages;
- latent-plan diversification;
- Bayesian best-arm allocation;
- risk-sensitive historical policy learning;
- validity-aware ordinal credit;
- trust-region-like policy updates;
- semantic redundancy control.

It does **not** claim that:

- the whole loop is the unbiased gradient of one stationary objective;
- PUCT values are calibrated estimates of long-run return;
- the strategist’s text is a complete causal explanation of generated code;
- semantic classification is infallible;
- posterior reward models are the true generative distributions;
- PPO clipping prevents cumulative drift;
- any finite diversity mechanism guarantees discovery of a globally optimal method.

Its central rationale is more modest and operational:

> Preserve strong constructions, explore genuinely different ways to improve them, allocate finite rollouts toward the chance of a new best result, and train the coder from all observed outcomes using a stable historical comparison whose memory adapts to policy change.

---

## Appendix A. Assumptions behind the theory

The mathematical interpretation relies on the following assumptions:

1. **Finite comparable rewards.** Every evaluated rollout produces a finite internal reward on one consistent higher-is-better scale.
2. **Trusted validity.** Validity comes from the evaluator, not the generated program’s claims.
3. **Verified seeds.** Historical initialization uses accepted seed constructions whose rewards live on the same scale as future rollouts.
4. **Correct behavior likelihoods.** Logged action probabilities describe the distribution that actually sampled the response.
5. **Within-step stationarity.** Pilot and continuation programs for an allocation decision use the same coder policy; otherwise arm evidence is confounded by learning.
6. **Conditionally comparable strategy arms.** Within one parent, rollout effort and evaluator conditions are similar enough that reward differences can be attributed mainly to strategy and sampling noise.
7. **Useful genealogy.** Conditioning on a parent creates enough inheritance that its descendants form a meaningful search branch.
8. **Local drift represents memory relevance.** Forward KL on selected contexts is assumed to be a useful proxy for how quickly old reward history should be forgotten.
9. **Best-of-budget relevance.** Scientific value is dominated by the best valid construction found, justifying expected-maximum allocation rather than average-reward allocation.
10. **Semantic judgments are advisory.** JEV is assumed useful enough to reduce redundancy, but not reliable enough to override the guarantee of filling a parent batch.

Violating an assumption does not necessarily make the system unusable, but it changes the meaning of its scores and guarantees. For example, stochastic evaluators make best-child values more susceptible to winner’s curse, and differing pilot effort makes a “strategy” posterior partly an effort posterior.

---

## Appendix B. Glossary

| Term | Meaning in this system |
|---|---|
| Advantage | A signed scalar controlling whether sampled actions become more or less likely |
| Archive | Bounded set of valid programs eligible to become parents |
| Behavior policy | Frozen coder distribution that generated a rollout |
| CDF | Cumulative distribution function |
| CVaR | Conditional Value at Risk; not the active SPO-RS objective |
| Direct child | Candidate generated with a particular state as its immediate parent |
| Effective count | Drift-decayed historical mass attached to the SPO-RS baseline |
| Entropic certainty equivalent | \(\beta^{-1}\log\mathbb E[e^{\beta R}]\) |
| Forward KL | \(D_{\mathrm{KL}}(\pi_{\mathrm{current}}\|\pi_{\mathrm{previous}})\) |
| Hurdle model | A distribution with a point mass at no improvement and a separate positive-gain component |
| JEV | Project name for the semantic same-core/variant/different classifier gate |
| LLM | Large language model |
| LOO | Leave one out |
| Native score | Problem-specific scientific metric before higher-is-better normalization |
| PPO | Proximal Policy Optimization; here, specifically its clipped surrogate idea |
| PUCT | Prior-guided upper-confidence search over tree states |
| RMS | Root mean square |
| Strategy arm | One plan and the coder distribution conditioned on that plan and parent |
| UCT | Upper Confidence bounds applied to Trees |
| Valid rollout | A program whose executed construction passes evaluator checks |
| Win probability | Smoothed probability that a valid candidate beats a valid historical draw |

