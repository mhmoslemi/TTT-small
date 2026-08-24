# Max-seeking memory and feedback

The redesigned paths are enabled in `configs/erdos.yaml`. Generic dataclass
defaults retain the previous behavior, so other problem configurations do not
change unless their new options are enabled.

## Memory: one budget, three arms

For each selected parent, the existing group of `K` rollouts is split among:

- `selected`: the lesson chosen by the model;
- `no_memory`: a permanent matched control;
- `explore`: an under-tested lesson selected with a UCB score.

The counts always sum to exactly `K`. There is no second full run and no extra
program evaluation. For the Erdos settings, a group of 16 becomes 10 selected,
3 no-memory, and 3 exploration rollouts.

Every response is still trained against its actual prompt, while the
max-seeking reward advantages are calculated across all `K` siblings from the
same parent. Memory credit uses the expected best reward of equal-size
subsamples, so the larger selected arm does not win merely because it has more
draws. The bank persists trials, valid programs, parent improvements, tail
uplift, and tail wins for each lesson. These counters survive curation and are
shown to both lookup and curation. When the bank or catalog must be capped,
matched mean tail uplift ranks ahead of static importance, so demonstrated
max-seeking value—not textual repetition—controls retention.

Important controls:

- `memory_arm_control_fraction`: no-memory share of each group;
- `memory_arm_explore_fraction`: under-tested-memory share;
- `memory_arm_exploration_c`: UCB exploration strength;
- `memory_outcome_credit`: enable matched outcome attribution;
- `memory_text_reinforce`: whether textual re-derivation changes importance.

Outcome credit requires a nonzero control fraction. Keeping
`memory_arm_max_lessons: 1`, `memory_lookup_mode: select`, and
`memory_lookup_max_select: 1` makes lesson-level attribution identifiable; the
configuration validator enforces these conditions when outcome credit is on.

## Feedback: repair only while repair is useful

Feedback remains one additional teacher forward for a retained failed rollout;
it does not generate or evaluate another program. Its effective coefficient is
controlled by the current step's validity rate:

- full strength at or below `feedback_validity_floor`;
- linearly reduced between the floor and target;
- zero at or above `feedback_validity_target`.

Teacher examples are selected round-robin across normalized verifier failure
signatures. `feedback_max_per_step` bounds total extra forwards and
`feedback_max_per_signature` prevents one repeated crash from dominating. The
mean magnitude of the already-weighted feedback advantage is capped relative
to the max-seeking reward advantage by `feedback_max_reward_ratio`. A small
`feedback_reward_scale_floor` still allows repair learning in an all-failed,
constant-reward group.

Each step summary records `valid_fraction`, `feedback_lambda_effective`,
`feedback_teacher_rollouts`, `memory_arm_rollouts`, and the matched per-parent
`memory_arm_updates`. Individual rollout metadata records its `memory_arm` and
`memory_ids`.

## Compute behavior

Memory keeps the rollout/evaluation budget exactly unchanged. It adds prompt
variants and a short lookup call, but null prompts are shorter, so total rollout
tokens may be slightly lower than injecting memory everywhere. Feedback is the
material extra model cost. The adaptive validity gate and caps make that cost
largest when invalid programs are the bottleneck and zero once validity reaches
the target.
