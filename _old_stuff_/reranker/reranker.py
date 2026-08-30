"""MultiAgentReRanker: background Elo tournament over the top-K buffer states.

Runs in a daemon thread. Each cycle it:
  1. snapshots the top-K non-empty states from the PUCT buffer (thread-safe),
  2. plays pairwise LLM-judge matches (single-turn comparison or simulated
     debate) to obtain win/loss records,
  3. converts the resulting Elo ratings into a probability distribution,
  4. writes that distribution back as the buffer's external P(s) prior.

The training loop keeps sampling from the buffer the entire time; it just picks
up whatever the latest prior is. Judge calls are HTTP / I/O bound, so the GIL is
released during them and this thread does not block generation or scoring.
"""
from __future__ import annotations
import random
import threading
import time
import traceback
# from concurrent.futures import ThreadPoolExecutor, as_completed

import random
import threading
import time
import traceback

from reranker.elo import build_pairings, update_pair, softmax_from_ratings
from reranker.prompts import (
    build_comparison_messages, parse_verdict, default_goal, DEFAULT_CRITERIA,
)


from reranker.elo import build_pairings, update_pair, softmax_from_ratings
from reranker.prompts import (
    build_comparison_messages, parse_verdict, default_goal, DEFAULT_CRITERIA,
)
from experiment_io import save_elo_match, save_elo_cycle_summary


class MultiAgentReRanker:
    def __init__(self, sampler, judge, cfg, metric_name="score", exp_dir=None,
                 maximize=True, target=None, criteria=None):
        self.sampler = sampler          # PUCTSampler; we use its thread-safe hooks
        self.judge = judge              # BaseJudge instance
        self.cfg = cfg                  # RerankerConfig
        self.criteria = criteria
        self.goal = cfg.goal or default_goal(metric_name, maximize, target)
        self.exp_dir = exp_dir          # where to write step{N}_Elo/ logs

        self.metric_name = metric_name


        self._stop = threading.Event()
        self._thread = None
        self._pair_rng = random.Random(1234)   # used only in the single bg thread
        self._cycle = 0

    # ---- lifecycle -------------------------------------------------------
    def start(self):
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._loop, name="elo-reranker", daemon=True
        )
        self._thread.start()


    def stop(self, join_timeout: float = 10.0):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=join_timeout)
            self._thread = None
        close = getattr(self.judge, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass

    # ---- background loop -------------------------------------------------
    def _loop(self):
        while not self._stop.is_set():
            t0 = time.time()
            try:
                self._run_once()
            except Exception:
                print("[reranker] tournament cycle failed:\n"
                      + traceback.format_exc())
            # Sleep the remainder of the poll interval, staying responsive to stop().
            elapsed = time.time() - t0
            self._stop.wait(max(0.0, self.cfg.poll_interval_s - elapsed))



    def _run_once(self):
        # 1) thread-safe snapshot of the elite states (no lock held during LLM work)
        snap = self.sampler.snapshot_top_states(self.cfg.top_k)
        if len(snap) < max(2, self.cfg.min_states_to_rank):
            return
        by_id = {s["id"]: s for s in snap}
        ids = list(by_id.keys())

        pairings = build_pairings(
            ids,
            mode=self.cfg.tournament_mode,
            num_random_matches=self.cfg.num_random_matches,
            both_orders=self.cfg.both_orders,
            rng=self._pair_rng,
        )
        if not pairings:
            return

        # 2) build all match prompts up front, recording which displayed side
        #    (1 vs 2) maps to which real id. Per-pair seeded flip reduces position
        #    bias and is deterministic, so it stays thread-safe.
        specs = []  # (pair, messages, id_shown_as_1, id_shown_as_2)
        for pair in pairings:
            a_id, b_id = pair
            seed = (hash(a_id) ^ (hash(b_id) << 1)) & 0xFFFFFFFF
            flip = random.Random(seed).random() < 0.5
            s1, s2 = (by_id[b_id], by_id[a_id]) if flip else (by_id[a_id], by_id[b_id])
            messages = build_comparison_messages(
                goal=self.goal, code1=s1["code"], code2=s2["code"],
                criteria=self.criteria, debate=self.cfg.debate,
                max_code_chars=self.cfg.max_code_chars,
                metric_name=self.metric_name,
            )
            specs.append((pair, messages, s1["id"], s2["id"]))



        # snapshot the step this cycle belongs to (for log dir naming)
        step_tag = self.sampler.get_current_step()

        # 3) play matches in GPU-sized batches; map each verdict back to a winner
        results = {}        # pair -> winner_id | None
        saved = []          # per-match log rows, written after Elo is computed
        bs = max(1, int(self.cfg.judge_batch_size))
        for start in range(0, len(specs), bs):
            if self._stop.is_set():
                return
            chunk = specs[start:start + bs]
            try:
                texts = self.judge.complete_batch([m for (_, m, _, _) in chunk])
            except Exception as e:
                print(f"[reranker] judge batch failed ({e!r})")
                texts = ["" for _ in chunk]
            for (pair, messages, id1, id2), text in zip(chunk, texts):
                v = parse_verdict(text)
                winner = id1 if v == 1 else (id2 if v == 2 else None)
                results[pair] = winner
                saved.append({
                    "pair": pair, "messages": messages, "response": text,
                    "id_shown_as_1": id1, "id_shown_as_2": id2,
                    "verdict": v, "winner": winner,
                })

        if self._stop.is_set():
            return

        # 4) Elo updates in a fixed pairing order for stability
        ratings = {i: float(self.cfg.elo_init) for i in ids}
        wins = {i: 0 for i in ids}
        for pair in pairings:
            winner = results.get(pair)
            if winner is None:
                continue
            a_id, b_id = pair
            score_a = 1.0 if winner == a_id else 0.0
            update_pair(ratings, a_id, b_id, score_a, self.cfg.elo_k)
            wins[winner] += 1

        # 5) normalize -> distribution -> write back as the buffer's external prior
        weights = softmax_from_ratings(ratings, temp=self.cfg.elo_softmax_temp)
        self.sampler.set_external_prior(weights, alpha=self.cfg.prior_weight)
        self._cycle += 1

        # 6) persist this cycle's matches + standings under step{N}_Elo/
        self._save_cycle(step_tag, self._cycle, saved, ratings, wins, weights, by_id)

        best_id = max(ratings, key=ratings.get)
        print(f"[reranker] cycle {self._cycle} (step {step_tag}): ranked "
              f"{len(ids)} states over {len(pairings)} matches; "
              f"top Elo={ratings[best_id]:.0f}")


    def _save_cycle(self, step_tag, cycle, saved, ratings, wins, weights, by_id):
        """Write each match (prompt + raw judge response + parsed verdict) and a
        per-cycle standings summary. No-op if no exp_dir was provided."""
        if not self.exp_dir:
            return
        try:
            for match_idx, row in enumerate(saved):
                a_id, b_id = row["pair"]
                meta = {
                    "step": step_tag,
                    "cycle": cycle,
                    "match": match_idx,
                    "judge_model": self.cfg.model,
                    "judge_backend": self.cfg.backend,
                    "debate": self.cfg.debate,
                    "id_a": a_id,
                    "id_b": b_id,
                    "value_a": by_id[a_id]["value"],
                    "value_b": by_id[b_id]["value"],
                    "id_shown_as_1": row["id_shown_as_1"],
                    "id_shown_as_2": row["id_shown_as_2"],
                    "verdict_1or2": row["verdict"],
                    "winner": row["winner"],
                }
                # The prompt is the user turn; include system for completeness.
                prompt_text = "\n\n".join(
                    f"=== {m['role'].upper()} ===\n{m['content']}"
                    for m in row["messages"]
                )
                save_elo_match(self.exp_dir, step_tag, cycle, match_idx,
                               meta, prompt_text=prompt_text,
                               response_text=row["response"])

            standings = sorted(
                ratings.keys(), key=lambda i: ratings[i], reverse=True
            )
            summary = {
                "step": step_tag,
                "cycle": cycle,
                "num_states": len(ratings),
                "num_matches": len(saved),
                "standings": [
                    {
                        "rank": r + 1,
                        "id": sid,
                        "elo": round(ratings[sid], 2),
                        "wins": wins[sid],
                        "prior_weight": round(weights.get(sid, 0.0), 6),
                        "reward_value": by_id[sid]["value"],
                        "raw_score": by_id[sid]["raw_score"],
                    }
                    for r, sid in enumerate(standings)
                ],
            }
            save_elo_cycle_summary(self.exp_dir, step_tag, cycle, summary)
        except Exception as e:
            print(f"[reranker] failed to save cycle logs ({e!r})")



    # def _run_once(self):
    #     # 1) thread-safe snapshot of the elite states (no lock held during LLM work)
    #     snap = self.sampler.snapshot_top_states(self.cfg.top_k)
    #     if len(snap) < max(2, self.cfg.min_states_to_rank):
    #         return

    #     by_id = {s["id"]: s for s in snap}
    #     ids = list(by_id.keys())

    #     # 2) build matches
    #     pairings = build_pairings(
    #         ids,
    #         mode=self.cfg.tournament_mode,
    #         num_random_matches=self.cfg.num_random_matches,
    #         both_orders=self.cfg.both_orders,
    #         rng=self._pair_rng,
    #     )
    #     if not pairings:
    #         return

    #     # 3) play matches concurrently (I/O bound). Result = winner id or None.
    #     results = {}  # (a_id, b_id) -> winner_id | None

    #     def play(pair):
    #         a_id, b_id = pair
    #         return pair, self._judge_pair(by_id[a_id], by_id[b_id])

    #     workers = max(1, int(self.cfg.judge_concurrency))
    #     with ThreadPoolExecutor(max_workers=workers) as pool:
    #         futs = [pool.submit(play, p) for p in pairings]
    #         for fut in as_completed(futs):
    #             if self._stop.is_set():
    #                 break
    #             try:
    #                 pair, winner = fut.result()
    #                 results[pair] = winner
    #             except Exception:
    #                 pass  # a single bad match must never kill the tournament

    #     if self._stop.is_set():
    #         return

    #     # 4) apply Elo updates in a fixed pairing order for stability
    #     ratings = {i: float(self.cfg.elo_init) for i in ids}
    #     for pair in pairings:
    #         winner = results.get(pair)
    #         if winner is None:
    #             continue
    #         a_id, b_id = pair
    #         score_a = 1.0 if winner == a_id else 0.0
    #         update_pair(ratings, a_id, b_id, score_a, self.cfg.elo_k)

    #     # 5) normalize -> distribution -> write back as the buffer's external prior
    #     weights = softmax_from_ratings(ratings, temp=self.cfg.elo_softmax_temp)
    #     self.sampler.set_external_prior(weights, alpha=self.cfg.prior_weight)

    #     self._cycle += 1
    #     best_id = max(ratings, key=ratings.get)
    #     print(f"[reranker] cycle {self._cycle}: ranked {len(ids)} states over "
    #           f"{len(pairings)} matches; top Elo={ratings[best_id]:.0f}")

    # # ---- one match -------------------------------------------------------
    # def _judge_pair(self, state_a: dict, state_b: dict):
    #     """Return the winning state id, or None. Randomize which state is shown
    #     as '1' vs '2' (per-pair seeded RNG -> thread-safe) to reduce position
    #     bias, then map the verdict back to the real id."""
    #     seed = (hash(state_a["id"]) ^ (hash(state_b["id"]) << 1)) & 0xFFFFFFFF
    #     flip = random.Random(seed).random() < 0.5
    #     s1, s2 = (state_b, state_a) if flip else (state_a, state_b)

    #     messages = build_comparison_messages(
    #         goal=self.goal,
    #         code1=s1["code"],
    #         code2=s2["code"],
    #         criteria=self.criteria,
    #         debate=self.cfg.debate,
    #         max_code_chars=self.cfg.max_code_chars,
    #     )
    #     try:
    #         text = self.judge.complete(messages)
    #     except Exception as e:
    #         print(f"[reranker] judge call failed ({e!r})")
    #         return None

    #     verdict = parse_verdict(text)
    #     if verdict == 1:
    #         return s1["id"]
    #     if verdict == 2:
    #         return s2["id"]
    #     return None  # unparseable -> no-op match