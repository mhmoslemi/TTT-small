"""
Lesson extraction (Sec. 2.2).

Planned: partition the step's B_t rollouts into S_t (r > 0) and F_t (r = 0 or
failed), then ONE LLM call per group -- lessons summarize patterns across the
whole group, not per-response notes. Exactly 2L lessons per step.
"""
