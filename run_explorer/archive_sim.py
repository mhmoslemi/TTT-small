"""
archive_sim.py - rebuild the PUCT candidate window of every step from the files of a run (read-only).

The JEV gate scans the top `requested x candidate_multiplier` states of the PUCT ranking, but jev.json only stores the candidates it
actually compared (it stops after the 8th acceptance).  To show which programs would take the place of rejected candidates when the
threshold is lowered, the explorer needs the REST of that window.  The window is a deterministic function of the archive, so it is
rebuilt here by replaying the rules of search/sampler.py on the saved rollouts:

    update():  dedupe exact code -> top-r children per (parent, strategy) -> top-k children per parent -> global cap
    ranking :  score = Q + c * scale * P(rank) * sqrt(1 + T) / (1 + n),   Q = best child reward if n > 0 else own reward
    visits  :  every selected parent and all of its ancestors get +group_size per step, T += group_size

Validation (printed by collect): the rebuilt archive size must equal summary.json's archive_size for every finished step, and the first
candidates of the rebuilt ranking must equal the recorded scan sequence (candidates with exactly equal reward may swap places - numpy's
argsort is not stable - which does not change who is selected).  Steps that do not validate get no window and fall back to the old,
record-only replay in the browser.
"""
import math
import os
import re
from concurrent.futures import ThreadPoolExecutor


class _S:
    __slots__ = ("id", "value", "parents", "seed", "code")

    def __init__(self, id_, value, parents, seed, code):
        self.id, self.value, self.parents, self.seed, self.code = id_, value, parents, seed, code


def _lineage(state, children):
    lin = {state.id} | set(state.parents)
    queue, seen = [state.id], {state.id}
    while queue:
        x = queue.pop()
        for c in children.get(x, ()):
            if c not in seen:
                seen.add(c)
                lin.add(c)
                queue.append(c)
    return lin


def reconstruct_windows(run_dir, config, root_ids, steps, rollouts, read_code, log=print):
    """Returns {step: {"window": [ids], "score": [..], "ok": bool, "seq_match": int, "n_decisions": int}}.

    steps    - the step records of collect() (parents, jev, archive_size, completed / provisional)
    rollouts - collect()'s rollout dicts (step, file_group, file_rollout, node, parent, sidx, reward_full, arch, valid, pval)
    read_code(rollout) -> source text of the program (used only to detect exact duplicates)
    """
    cfg = config or {}
    try:
        K = int(cfg["topk_children_per_parent"])
        CAP = int(cfg["max_buffer_size"])
        C = float(cfg["puct_c"])
        GS = int(cfg.get("group_size") or 0)
    except (KeyError, TypeError, ValueError):
        log("window reconstruction skipped: config lacks topk_children_per_parent / max_buffer_size / puct_c")
        return {}
    if not GS:
        return {}
    by_step = {}
    for r in rollouts:
        by_step.setdefault(r["step"], []).append(r)
    for rs in by_step.values():
        rs.sort(key=lambda r: (r["file_group"], r["file_rollout"]))
    two_stage = any(r.get("sidx") is not None for r in rollouts)
    TOPR = int(cfg.get("strategy_archive_top_r") or 0) if two_stage else 0

    # seeds: their reward is the parent_value logged by the rollouts that expanded them
    seed_val = {}
    for r in by_step.get(min(by_step) if by_step else 0, []):
        if r["parent"] in root_ids and r.get("pval") is not None:
            seed_val.setdefault(r["parent"], float(r["pval"]))
    if len(seed_val) != len(root_ids):
        log("window reconstruction skipped: seed rewards not recoverable")
        return {}
    states = [_S(r, seed_val[r], [], True, "") for r in root_ids]
    n, m, T = {}, {}, 0

    def ranked():
        vals = [s.value for s in states]
        nonseed = [s.value for s in states if not s.seed]
        scale = max(max(nonseed) - min(nonseed), 1e-6) if nonseed else 1.0
        N = len(vals)
        order = sorted(range(N), key=lambda i: -vals[i])
        rank = [0] * N
        for rk, i in enumerate(order):
            rank[i] = rk
        w = [N - rank[i] for i in range(N)]
        tw = float(sum(w))
        sq = math.sqrt(1.0 + T)
        out = []
        for i, s in enumerate(states):
            nn = n.get(s.id, 0)
            q = m.get(s.id, vals[i]) if nn > 0 else vals[i]
            bonus = C * scale * (w[i] / tw) * sq / (1.0 + nn)
            out.append((q + bonus, vals[i], s))
        out.sort(key=lambda x: (x[0], x[1]), reverse=True)
        return out

    out_windows = {}
    ok_steps = bad_steps = 0
    for rec in steps:
        step = rec["step"]
        parents = rec.get("parents")
        if not parents:
            break
        jev = rec.get("jev") or {}
        decisions = jev.get("decisions") or []
        if step > 0 and decisions:
            req = int(jev.get("requested") or 8)
            limit = int(jev.get("candidate_limit") or req * int(jev.get("candidate_multiplier") or 3))
            rk = ranked()
            window = rk[:limit]
            recseq = [d["candidate_id"] for d in decisions]
            rec_val = {e[2].id: e[1] for e in rk}
            # strict prefix match of the recorded scan sequence against a replay of the scan on the rebuilt window
            children = {}
            for s in states:
                for p in s.parents:
                    children.setdefault(p, set()).add(s.id)
            accepted = {d["candidate_id"]: d["accepted"] for d in decisions}
            blocked, picks, rejected, seq = set(), [], set(), []
            stop = False
            for relax in (False, True):
                for e in window:
                    if len(picks) >= req or stop:
                        break
                    s = e[2]
                    if s.id in picks or s.id in rejected or (not relax and s.id in blocked):
                        continue
                    seq.append(s.id)
                    v = accepted.get(s.id)
                    if v is None:
                        stop = True
                        break
                    if v:
                        picks.append(s.id)
                        blocked |= _lineage(s, children)
                    else:
                        rejected.add(s.id)
            match = 0
            for a, b in zip(seq, recseq):
                if a == b or (rec_val.get(a) is not None and rec_val.get(a) == rec_val.get(b)):
                    match += 1
                else:
                    break
            sel = jev.get("selected") or []
            ok = match == len(recseq) and set(picks) == set(sel)
            out_windows[step] = {"window": [e[2].id for e in window], "score": [round(e[0], 9) for e in window], "ok": bool(ok),
                                 "seq_match": match, "n_decisions": len(recseq)}
            ok_steps += ok
            bad_steps += (not ok)
        # ---- bookkeeping of this step: visits, then the archive update with its children
        by_id = {s.id: s for s in states}
        sel_ids = [p["parent_id"] for p in parents]
        for pid in sel_ids:
            p = by_id.get(pid)
            if p is None:
                continue
            for aid in [p.id] + list(p.parents):
                n[aid] = n.get(aid, 0) + GS
            T += GS
        if not (rec.get("completed") or rec.get("provisional")):
            break
        kids = []
        for r in by_step.get(step, []):
            if not r.get("arch") or r.get("reward_full") is None:
                continue
            parent = by_id.get(r["parent"])
            if parent is None:
                continue
            kids.append((_S(r["node"], float(r["reward_full"]), None, False, read_code(r) or ""), parent, r.get("sidx")))
        pb = {}
        for c, p, _sk in kids:
            pb[p.id] = max(pb.get(p.id, -1e18), c.value)
        for pid, b in pb.items():
            m[pid] = max(m.get(pid, b), b)
        existing = {s.code for s in states if s.code}
        new = []
        for c, p, sk in kids:
            if c.code and c.code in existing:
                continue
            c.parents = [p.id] + list(p.parents)
            new.append((c, p, sk))
            if c.code:
                existing.add(c.code)
        if TOPR > 0:
            groups = {}
            for c, p, sk in new:
                groups.setdefault((p.id, sk), []).append((c, p, sk))
            adm = []
            for lst in groups.values():
                lst.sort(key=lambda e: e[0].value, reverse=True)
                adm.extend(lst[:TOPR])
        else:
            adm = new
        states.extend(e[0] for e in adm)
        byp, nop = {}, []
        for s in states:
            if s.seed or not s.parents:
                nop.append(s)
                continue
            byp.setdefault(s.parents[0], []).append(s)
        filt = list(nop)
        for lst in byp.values():
            lst.sort(key=lambda x: x.value, reverse=True)
            filt.extend(lst[:K])
        states[:] = filt
        if len(states) > CAP:
            seeds = [s for s in states if s.seed]
            rest = sorted([s for s in states if not s.seed], key=lambda x: x.value, reverse=True)
            states[:] = seeds + rest[:CAP - len(seeds)]
        # the archive size is logged for finished steps: an exact-match check of the whole replay
        if rec.get("archive_size") is not None and rec["archive_size"] != len(states):
            log("window reconstruction: archive size after step %d is %d but the run logged %d - stopping the replay here"
                % (step, len(states), rec["archive_size"]))
            for later in [s["step"] for s in steps if s["step"] > step]:
                out_windows.pop(later, None)
            break
    log("candidate windows rebuilt for %d steps (%d validated against the recorded scans, %d not)" % (len(out_windows), ok_steps, bad_steps))
    return out_windows


def code_reader(run_dir, extract_code, split_think, read):
    """Factory for read_code(rollout) used by collect()."""
    def read_code(r):
        sd = os.path.join(run_dir, "step%02d" % r["step"])
        base = "step%02d_group%02d_rollout%03d.txt" % (r["step"], r["file_group"], r["file_rollout"])
        txt = read(os.path.join(sd, base))
        think, ans = split_think(txt)
        return extract_code(ans or txt)
    return read_code
