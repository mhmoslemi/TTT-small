"""
Search tree + flat archive D (paper Sec. 1).

Planned: insert children under a parent; recompute per step the subtree size
m_s = |T(s)| and the subtree max W_m(s) = max_{y in T(s)} Q(y); expose the flat
archive for dataset-level statistics; cap |D| at search.max_archive_size.
"""
