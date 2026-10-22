"""Select parents from the evaluated tree with one saved GPT-6 Luna decision."""

from __future__ import annotations

import ast
import hashlib
import json
import math
from concurrent.futures import Future
from pathlib import Path
from string import Template
from threading import Thread

from plot_parent_tree import load_genealogy, write_tree_log
from sampler import PUCTSampler, State
from strategy_api import (OPENAI_MAX_OUTPUT_TOKENS, OPENAI_STRATEGY_MODEL,
                          _openai_strategy_cost, strategy_api_key)


PROMPT_FILE = Path(__file__).with_name("llm_sampler_prompt.txt")


def _write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                    allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _finite_score(value):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value))


def _selection_key(num_states, selection_step):
    if (isinstance(num_states, bool) or int(num_states) != num_states or num_states < 1
            or selection_step is None or isinstance(selection_step, bool)
            or int(selection_step) != selection_step
            or selection_step < 0):
        raise ValueError("LLM parent selection needs a positive count and explicit nonnegative step")
    return int(num_states), int(selection_step)


def _selection_format(count):
    return {
        "type": "json_schema", "name": "parent_selection", "strict": True,
        "schema": {
            "type": "object", "additionalProperties": False,
            "required": ["selected_parent_ids", "selections", "tree_diagnosis", "confidence"],
            "properties": {
                "selected_parent_ids": {
                    "type": "array", "minItems": count, "maxItems": count,
                    "items": {"type": "string"}},
                "selections": {
                    "type": "array", "minItems": count, "maxItems": count,
                    "items": {
                        "type": "object", "additionalProperties": False,
                        "required": ["node_id", "role", "reason"],
                        "properties": {
                            "node_id": {"type": "string"},
                            "role": {"type": "string", "enum": ["exploit", "explore"]},
                            "reason": {"type": "string"}}}},
                "tree_diagnosis": {"type": "string"},
                "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
            },
        },
    }


class LLMSampler(PUCTSampler):
    def __init__(self, *args, exp_dir, problem, **kwargs):
        if exp_dir is None or problem is None:
            raise ValueError("LLM parent selection requires the run directory and problem")
        super().__init__(*args, **kwargs)
        self.exp_dir = Path(exp_dir)
        self.problem = problem
        self._selection_step, self._selected = None, []
        self._prefetch = None

    @property
    def selection_mode(self):
        return "llm"

    def load_state_dict(self, payload):
        super().load_state_dict(payload)
        self._selection_step, self._selected = None, []
        self._prefetch = None

    def import_legacy_states(self, *args, **kwargs):
        raise ValueError("LLM resume requires training_state.pt to preserve the saved node identities")

    def _seed_identities(self):
        return [{"id": state.id, "value": state.value, "raw_score": state.raw_score,
                 "code_sha256": hashlib.sha256(state.code.encode()).hexdigest()}
                for state in self._states if state.id in self._seed_ids]

    def _restore_initial_seed_ids(self, saved):
        # A step-zero interruption can predate the first training checkpoint.
        current = self._seed_identities()
        if len(saved) != len(current):
            raise ValueError("LLM step-zero replay has a different seed count")
        for old, new in zip(saved, current):
            if any(old[key] != new[key] for key in ("value", "raw_score", "code_sha256")):
                raise ValueError("LLM step-zero replay has different seed contents")
        states = {state.id: state for state in self._states}
        for old, new in zip(saved, current):
            if old["id"] != new["id"]:
                self.reassign_seed_id(states[new["id"]], old["id"])

    def _tree(self, step, path):
        # Use a separate snapshot so asynchronous plotting cannot race this write.
        write_tree_log(self.exp_dir, max_step=step - 1, output_path=path)
        tree = json.loads(path.read_text(encoding="utf-8"))
        if tree.get("source_errors"):
            raise ValueError("LLM selection cannot use a tree with unreadable rollout metadata")
        for seed in self._states:
            if seed.id not in self._seed_ids:
                continue
            node = tree["nodes"].setdefault(seed.id, {
                "parent_id": None, "child_ids": [], "raw_problem_score": seed.raw_score})
            node["raw_problem_score"] = seed.raw_score
            if seed.id not in tree["root_ids"]:
                tree["root_ids"].append(seed.id)
        _write_json(path, tree)
        return tree

    def _eligible(self, tree):
        scored = {sid: row["raw_problem_score"] for sid, row in tree["nodes"].items()
                  if _finite_score(row["raw_problem_score"])
                  and row["raw_problem_score"] != 0}
        eligible = set(scored) | self._seed_ids
        if not scored:
            return eligible, set(self._seed_ids), None
        best = (max if self.problem.maximize else min)(scored.values())
        return eligible, {sid for sid, score in scored.items() if score == best}, best

    def _prompt(self, tree, step, count):
        _, best_ids, best = self._eligible(tree)
        cfg = getattr(self.problem, "cfg", {})
        target = cfg.get("target_print", getattr(self.problem, "target", None))
        return Template(PROMPT_FILE.read_text(encoding="utf-8")).substitute(
            problem_name=self.problem.name, metric_name=self.problem.metric_name,
            direction="MAXIMIZE" if self.problem.maximize else "MINIMIZE",
            score_order="Higher valid raw scores are better." if self.problem.maximize
                        else "Lower valid raw scores are better.",
            target=repr(target), selection_step=step, num_parents=count,
            best_score=repr(best), best_node_ids=json.dumps(sorted(best_ids)),
            seed_ids=json.dumps(sorted(self._seed_ids)))

    def _request(self, request_path, response_path, tree_path, request_record):
        from openai import OpenAI

        key = strategy_api_key("OPENAI_API_KEY")
        client = OpenAI(api_key=key, base_url="https://api.openai.com/v1",
                        timeout=1800.0, max_retries=0)
        uploaded = None
        try:
            with tree_path.open("rb") as tree_file:
                uploaded = client.files.create(
                    file=("search_tree.json", tree_file, "application/json"),
                    purpose="user_data")
            body = request_record["api_request"]
            body["input"][0]["content"][1] = {"type": "input_file", "file_id": uploaded.id}
            _write_json(request_path, request_record)
            print(f"[llm-sampler] step {request_record['step']}: requesting "
                  f"{request_record['num_parents']} parents from {OPENAI_STRATEGY_MODEL} "
                  "(reasoning=max)", flush=True)
            response = client.responses.create(**body)
            payload = response.model_dump(mode="json")
            # Persist before validation so an interrupted step never rebuys a saved decision.
            _write_json(response_path, payload)
            return payload
        except Exception as error:
            status = getattr(error, "status_code", None)
            detail = f", HTTP {status}" if status is not None else ""
            raise RuntimeError(
                f"LLM parent request failed ({type(error).__name__}{detail}); "
                f"request saved at {request_path}") from None
        finally:
            if uploaded is not None:
                try:
                    client.files.delete(uploaded.id)
                except Exception:
                    print(f"[warn] LLM sampler could not delete uploaded file {uploaded.id}",
                          flush=True)
            client.close()

    def _validate_response(self, response, tree, count):
        if response.get("status") != "completed":
            raise ValueError("LLM parent response is incomplete; saved response retained")
        parts = []
        for item in response.get("output", []):
            if item.get("type") == "reasoning":
                continue
            if item.get("type") != "message":
                raise ValueError("LLM parent response contains an unexpected output")
            for content in item.get("content", []):
                if content.get("type") != "output_text":
                    raise ValueError("LLM parent response was refused or contains non-text output")
                parts.append(content["text"])
        result = json.loads("".join(parts))
        ids = result.get("selected_parent_ids")
        if (not isinstance(ids, list) or len(ids) != count
                or any(not isinstance(sid, str) for sid in ids)
                or len(set(ids)) != count):
            raise ValueError(f"LLM must return exactly {count} unique node IDs")
        eligible, best_ids, _ = self._eligible(tree)
        if any(sid not in tree["nodes"] or sid not in eligible for sid in ids):
            raise ValueError("LLM selected an unknown, failed, or unscored generated node")
        if ids[0] not in best_ids:
            raise ValueError("LLM first parent must have the tree's exact best valid raw score")
        selections = result.get("selections")
        if (not isinstance(selections, list) or len(selections) != count
                or [row.get("node_id") for row in selections] != ids
                or any(row.get("role") not in ("exploit", "explore")
                       or not isinstance(row.get("reason"), str) for row in selections)):
            raise ValueError("LLM selection rationales do not match its ordered parent IDs")
        return result

    def _states_from_tree(self, ids, tree, step):
        known = {state.id: state for state in self._states}
        missing = set(ids) - set(known)
        if missing:
            from reward import extract_python_code

            genealogy, _ = load_genealogy(self.exp_dir, max_step=step - 1)
            for path in sorted(self.exp_dir.glob("step*/step*_group*_rollout*.meta.json")):
                if int(path.parent.name.removeprefix("step")) >= step:
                    continue
                meta = json.loads(path.read_text(encoding="utf-8"))
                sid = meta.get("node_id")
                if sid not in missing:
                    continue
                if (not meta.get("valid") or meta.get("status") == "evaluation_pending"
                        or not _finite_score(meta.get("reward"))
                        or meta.get("raw_score") != tree["nodes"][sid]["raw_problem_score"]
                        or meta.get("parent_id") != tree["nodes"][sid]["parent_id"]):
                    raise ValueError(f"LLM parent does not match its evaluated artifact: {path}")
                response_path = path.with_name(path.name.removesuffix(".meta.json") + ".txt")
                code = extract_python_code(response_path.read_text(encoding="utf-8", errors="replace"))
                if not code:
                    raise ValueError(f"LLM parent has no recoverable saved program: {response_path}")
                ancestors, seen = [], {sid}
                ancestor = tree["nodes"][sid]["parent_id"]
                while ancestor is not None:
                    if ancestor in seen or ancestor not in genealogy:
                        raise ValueError(f"LLM parent has inconsistent saved ancestry: {sid}")
                    seen.add(ancestor)
                    source = genealogy[ancestor]
                    ancestors.append({"id": ancestor, "timestep": (
                        0 if source["is_seed"] else int(source["level"]) - 1)})
                    ancestor = tree["nodes"][ancestor]["parent_id"]
                construction = meta.get("construction")
                if isinstance(construction, str):
                    construction = ast.literal_eval(construction)
                if construction is None and getattr(self.problem, "saves_construction", False):
                    raise ValueError(f"LLM parent is missing its saved construction: {sid}")
                if construction is not None:
                    construction = [float(value) for value in construction]
                    if not all(math.isfinite(value) for value in construction):
                        raise ValueError(f"LLM parent has a non-finite saved construction: {sid}")
                known[sid] = State(
                    id=sid, timestep=int(meta["step"]), value=float(meta["reward"]),
                    code=code, parents=ancestors, raw_score=float(meta["raw_score"]),
                    construction=construction)
                missing.remove(sid)
                if not missing:
                    break
        if missing:
            raise ValueError(f"LLM selection is missing saved programs: {sorted(missing)}")
        return [known[sid] for sid in ids]

    def _selection_response(self, count, step):
        directory = self.exp_dir / f"step{step:02d}"
        directory.mkdir(parents=True, exist_ok=True)
        base = directory / f"step{step:02d}.sampler"
        tree_path, request_path, response_path = (
            Path(str(base) + suffix) for suffix in
            (".tree.json", ".request.json", ".response.json"))
        if request_path.is_file():
            record = json.loads(request_path.read_text(encoding="utf-8"))
            if (record["step"] != step or record["num_parents"] != count
                    or record["problem"] != self.problem.name
                    or record["maximize"] != bool(self.problem.maximize)):
                raise ValueError("saved LLM request differs from this step's problem or parent budget")
            if step == 0:
                self._restore_initial_seed_ids(record["seed_identities"])
            tree_bytes = tree_path.read_bytes()
            if hashlib.sha256(tree_bytes).hexdigest() != record["tree_sha256"]:
                raise ValueError("saved LLM tree snapshot changed after submission")
            tree = json.loads(tree_bytes)
        else:
            if response_path.exists():
                raise ValueError("saved LLM response has lost its request/tree snapshot")
            tree = self._tree(step, tree_path)
            eligible, _, _ = self._eligible(tree)
            if len(eligible) < count:
                raise ValueError(f"LLM requested {count} parents but only {len(eligible)} are eligible")
            prompt = self._prompt(tree, step, count)
            Path(str(base) + ".prompt.txt").write_text(prompt, encoding="utf-8")
            record = {
                "step": step, "num_parents": count, "problem": self.problem.name,
                "maximize": bool(self.problem.maximize),
                "seed_identities": self._seed_identities(),
                "tree_sha256": hashlib.sha256(tree_path.read_bytes()).hexdigest(),
                "api_request": {
                    "model": OPENAI_STRATEGY_MODEL, "reasoning": {"effort": "max"},
                    "max_output_tokens": OPENAI_MAX_OUTPUT_TOKENS, "store": False,
                    "tools": [], "tool_choice": "none", "parallel_tool_calls": False,
                    "text": {"format": _selection_format(count)},
                    "input": [{"role": "user", "content": [
                        {"type": "input_text", "text": prompt},
                        {"type": "input_file", "filename": "search_tree.json"}]}],
                },
            }
            _write_json(request_path, record)
        response = (json.loads(response_path.read_text(encoding="utf-8"))
                    if response_path.is_file() else self._request(
                        request_path, response_path, tree_path, record))
        return tree, response

    def prefetch_states(self, num_states, *, selection_step):
        """Save the next decision during training without changing current picks."""
        count, step = _selection_key(num_states, selection_step)
        with self._lock:
            if self._prefetch is not None:
                previous_step, previous_count, future = self._prefetch
                if previous_step == step:
                    if previous_count != count:
                        raise ValueError("cannot change the saved LLM parent count within a step")
                    return future
                if not future.done():
                    raise ValueError("another LLM parent decision is still running")
            future = Future()
            self._prefetch = step, count, future

        def prepare():
            try:
                tree, response = self._selection_response(count, step)
                self._validate_response(response, tree, count)
            except Exception as error:
                future.set_exception(error)
                print(f"[warn] LLM sampler step {step}: background decision failed "
                      f"({type(error).__name__}); current training can finish; "
                      "the error will be raised before next-step selection", flush=True)
            else:
                future.set_result(None)
                print(f"[llm-sampler] step {step}: next-parent response saved "
                      "and validated while training continues", flush=True)

        # A slow API request must not delay checkpoint saving or process shutdown.
        Thread(target=prepare, name=f"llm-sampler-step{step}", daemon=True).start()
        return future

    def sample_states(self, num_states, *, selection_step=None):
        count, step = _selection_key(num_states, selection_step)
        prefetched = self._prefetch
        if prefetched is not None and prefetched[0] == step:
            if prefetched[1] != count:
                raise ValueError("cannot change the saved LLM parent count within a step")
            if not prefetched[2].done():
                print(f"[llm-sampler] step {step}: waiting for the parent "
                      "request started during previous-step training", flush=True)
            prefetched[2].result()
        with self._lock:
            if self._selection_step == step:
                if len(self._selected) != count:
                    raise ValueError("cannot change the saved LLM parent count within a step")
                return list(self._selected)
            tree, response = self._selection_response(count, step)
            result = self._validate_response(response, tree, count)
            picks = self._states_from_tree(result["selected_parent_ids"], tree, step)
            infos = []
            for rank, (state, row) in enumerate(zip(picks, result["selections"]), 1):
                infos.append({
                    "value": float(state.value), "n": self._n.get(state.id, 0),
                    "Q": None, "P": None, "bonus": None, "score": None,
                    "selection_mode": "llm", "is_seed": state.id in self._seed_ids,
                    "llm": {"rank": rank, "role": row["role"], "reason": row["reason"],
                            "response_id": response.get("id")},
                })
            self.last_picks_info = infos
            self._selection_step, self._selected = step, list(picks)
            usage = response.get("usage") or {}
            details = usage.get("input_tokens_details") or {}
            input_usd, output_usd = _openai_strategy_cost({
                "prompt": usage.get("input_tokens", 0),
                "cached": details.get("cached_tokens", 0),
                "cache_write": details.get("cache_write_tokens", 0),
                "completion": usage.get("output_tokens", 0)})
            print(f"[llm-sampler] step {step}: {count} parents validated; "
                  f"estimated request cost=${input_usd + output_usd:.6f}; "
                  "prompt/tree/request/response saved under "
                  f"{self.exp_dir / f'step{step:02d}'}", flush=True)
            return picks
