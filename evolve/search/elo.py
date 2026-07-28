"""
Elo debate signal (Sec. 2.1).

Planned: pairwise LLM-judge comparisons y_ij in {0, 0.5, 1};
p_ij = 1 / (1 + 10^((E_j - E_i) / elo.scale)); K-factor updates; standardize
to E~(s) over D. elo.scale is configurable because the paper writes the
logistic without the classic /400.
"""
