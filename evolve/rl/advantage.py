"""
Advantages.

Planned: Eq. 8 group-relative tilt omega_i = G_p*softmax(beta*r)_i,
A_rew = omega_i - 1 (sums to zero by construction); Eq. 9 feedback-conditioned
self-teacher A_fb, detached, failures only; combine A = A_rew + lambda_f*d_i*A_fb.
"""
