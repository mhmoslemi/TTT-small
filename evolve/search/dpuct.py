"""
The D-PUCT selection rule (Eq. 4, 5, 6).

Planned: virtual child logit mu_L(p) + lambda*sigma_L(p); parent-local
temperature softmax pi_D(a|p); score V(p,a) + c*pi_D(a|p)*sqrt(m_p)/(1+m_p,a);
top-n over all expandable actions; virtual -> 1 child, leaf -> k children.
"""
