# Adaptive upper-tail (CVaR) advantages: findings so far

*2026-09-04. Based on the Erdos runs under `runs/` (Qwen3-8B, 512 rollouts per step) and the analysis in `plot_reward_pdf.py` / `advantage.py`.*

## Setup

The trainer's default objective is the entropic one from the paper: per group of K rollouts, advantages are `w_beta - 1` with `w_beta = exp(beta R) / E[exp(beta R)]`, and beta is chosen per group by bisection so that the tilted distribution has KL = log 2 from uniform. As beta grows this concentrates on the single best rollout. `train_multy_CVaR.py` adds two alternatives selected by `--advantage-mode`: plain GRPO (`(r - mean) / std`) and `cvar`, which blends GRPO with a sparse upper-tail term `(r - q_{1-alpha}) / std` for rollouts above the `(1-alpha)`-quantile, mixed with weight lambda. The open question is how to set alpha per step instead of fixing it.

## Observation 1: the reward distribution collapses to a few discrete values

Within a run the per-step reward distribution starts broad and quickly degenerates. At step 0 of `erdos_Qwen3-8B_0827-1532` the 191 valid rollouts take 134 distinct reward values; by step 20 there are 37 distinct values among 367 valid rollouts; by step 26 six values with 98 % of rollouts on one of them; by step 33 every valid rollout has the identical reward. The model is replaying a handful of constructions. Invalid rollouts sit at reward 0 and make up 30–75 % of each batch throughout, so every step is a spike at zero plus a narrow cluster near the maximum. The reward `1 / c5_bound` compresses the whole late-run progress (a gap to target of 1e-1 down to 1e-4 in raw score) into the fifth decimal of the reward, which is where all the late-step structure lives.

## Observation 2: EVT threshold selection is unstable on this data

We implemented the textbook approach: fit a Generalized Pareto Distribution by maximum likelihood to the excesses above each candidate threshold `R_(k+1)` and pick the `k*` minimising the Kolmogorov–Smirnov distance (Clauset–Shalizi–Newman). The implementation is verified on synthetic GPD samples (shape 0.3 and −0.4 recovered within 0.01, scale within 1 %) and on a synthetic body-plus-Pareto-tail batch, where it finds the tail start correctly. On real steps it is erratic: `k*` jumps between e.g. 19, 96, 150, 25, 134 on consecutive steps, most KS distances (0.15–0.5) fail the 5 % goodness-of-fit test (critical ≈ 1.36/√k), fitted shapes reach −14 or +5 with scales of 1e-8, and many steps hit the `max_tail_frac = 0.5` cap, meaning no interior tail was found. The cause is Observation 1: a GPD is a continuous model and the excesses are a staircase of ties on a 1e-6 scale, so the KS argmin is choosing between equally bad fits and flips at the slightest change. On fully tied steps the fit is impossible and the code falls back to "tail = rollouts tied at the maximum" (reported as `ties_at_max`, with NaN shape/scale/KS). The plain Hill estimator is worse still, since it can only represent heavy (positive-shape) tails while bounded rewards have a negative shape.

## Observation 3: the Gini-modulated alpha is stable but carries no tail information

`alpha = alpha_base (1 - Gini)^gamma` behaves smoothly, but for the wrong reason. Computed over all rewards, Gini is dominated by the zeros and is essentially `1 - valid_fraction`, so alpha tracks the validity rate rather than the shape of the good rollouts. Computed over positive rewards only, Gini is 0.00–0.07 at every step (the valid rewards are nearly equal), so alpha sits at `alpha_base` and the rule reduces to a fixed quantile. Neither variant distinguishes a step with one standout discovery from a step with a flat plateau, which is exactly what it was meant to do.

## Observation 4: thresholds and ties

Because the tail threshold is itself one of the observed values and the values are heavily tied, "rollouts strictly above the threshold" and "the top k rollouts" disagree, sometimes by a lot (late steps report 0 rollouts above a threshold that k = 60 rollouts sit on). Any CVaR implementation on this data must define explicitly whether ties at the boundary belong to the tail; the top-k-with-ties convention is the only one that gives a non-empty tail once the run converges.

## What this implies for an adaptive alpha

The failure is not in the estimators but in the space they are applied to. Three directions look viable, in order of expected payoff: (1) run the tail analysis on the raw-score gap to target in log space, where late-run progress spans orders of magnitude instead of the fifth decimal of the reward; (2) treat rewards as discrete levels, cluster ties, and define the tail as the top m distinct levels so ties are the object of analysis rather than a failure mode; (3) if a per-step alpha is needed regardless, gate the GPD fit on the KS test, fall back to the previous step's alpha when it fails, and smooth across steps. Fully tied steps have no tail in any statistical sense and should be reported as such rather than assigned an alpha.
