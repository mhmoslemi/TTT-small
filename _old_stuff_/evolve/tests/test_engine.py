"""End-to-end: Algorithm 1 on the mock backbone, no torch and no GPU."""
import json

import pytest

from config import Config, load_config
from core.engine import Engine
from core.types import LEAF_EXPAND, VIRTUAL_EXPAND
from llm.judge import parse_verdict
from llm.generation import split_jobs
from prompting.builder import PromptBuilder
from core.types import Lesson, Node


def _cfg(tmp_path, over=None):
    """Resolved config for a fast mock run; `over` takes dotted paths."""
    cfg = load_config([
        "--example", "circle_packing", "--backend", "mock",
        "--steps", "3", "--n-select", "3", "--k-children", "3",
        "--verifier-timeout", "30", "--output-root", str(tmp_path),
        "--set", "example.params.num_circles=4",
    ]).config
    for path, value in (over or {}).items():
        section, _, leaf = path.partition(".")
        setattr(getattr(cfg, section), leaf, value)
    return cfg


def test_full_run_produces_a_valid_best_candidate(tmp_path):
    engine = Engine(_cfg(tmp_path))
    assert engine.run() == 0
    best = engine.tree.best()
    assert best is not None and best.valid and best.reward > 0
    assert len(engine.history) == 3

def test_batch_size_stays_within_n_to_nk(tmp_path):
    engine = Engine(_cfg(tmp_path))
    engine.run()
    n, k = engine.cfg.search.n_select, engine.cfg.search.k_children
    for step in engine.history:
        assert 1 <= step["num_targets"] <= n
        assert step["num_targets"] <= step["batch_size"] <= step["num_targets"] * k

def test_tree_grows_and_every_node_stays_attached(tmp_path):
    engine = Engine(_cfg(tmp_path))
    engine.run()
    sizes = [s["archive_size"] for s in engine.history]
    assert sizes == sorted(sizes) and sizes[-1] > sizes[0]
    for node in engine.tree.nodes():
        assert node.is_root or node.parent_id in engine.tree

def test_search_widens_as_well_as_deepens(tmp_path):
    """Both expansion modes must actually fire, or the virtual child is dead code."""
    engine = Engine(_cfg(tmp_path, {"run.max_steps": 6}))
    engine.run()
    assert sum(s["leaf_targets"] for s in engine.history) > 0
    assert sum(s["virtual_targets"] for s in engine.history) > 0

def test_memory_accumulates_additively(tmp_path):
    engine = Engine(_cfg(tmp_path))
    engine.run()
    totals = [s["memory"]["total"] for s in engine.history]
    assert totals == sorted(totals) and totals[-1] > 0

def test_memory_disabled_leaves_prompts_clean(tmp_path):
    engine = Engine(_cfg(tmp_path, {"memory.enabled": False}))
    engine.run()
    assert engine.memory is None
    assert all(s["memory"]["total"] == 0 for s in engine.history)

def test_elo_runs_and_is_skipped_when_alpha_is_one(tmp_path):
    engine = Engine(_cfg(tmp_path))
    engine.run()
    assert sum(s["elo"]["matches"] for s in engine.history) > 0

    other = Engine(_cfg(tmp_path, {"search.alpha": 1.0}))
    other.run()
    assert all(s["elo"]["matches"] == 0 for s in other.history)

def test_run_directory_contains_the_full_record(tmp_path):
    engine = Engine(_cfg(tmp_path))
    engine.run()
    root = engine.io.root
    for name in ("config.json", "tree.json", "memory.json",
                 "final.summary.json", "best_code.py"):
        assert (root / name).exists(), name
    step0 = root / "step00"
    assert list(step0.glob("*.txt")) and list(step0.glob("*.meta.json"))
    # Failed rollouts are logged too -- they are the input to Eq. 9.
    metas = [json.loads(p.read_text()) for p in step0.glob("*.meta.json")]
    assert any(not m["valid"] for m in metas)
    assert all(m["feedback"] for m in metas if not m["valid"])

def test_selection_mode_action_descend_also_completes(tmp_path):
    engine = Engine(_cfg(tmp_path, {"search.selection_mode": "action_descend"}))
    assert engine.run() == 0
    assert engine.tree.best() is not None

def test_resolved_config_is_saved_with_provenance(tmp_path):
    resolution = load_config([
        "--example", "circle_packing", "--backend", "mock", "--steps", "1",
        "--n-select", "1", "--k-children", "1", "--output-root", str(tmp_path),
        "--alpha", "0.75"])
    engine = Engine(resolution.config, resolution)
    engine.run()
    saved = json.loads((engine.io.root / "config.json").read_text())
    assert saved["search"]["alpha"] == 0.75
    assert "cli" in (engine.io.root / "provenance.txt").read_text()


# ---------------- judge parsing ----------------
@pytest.mark.parametrize("text,expected", [
    ("A is stronger.\nVERDICT: A", 1.0),
    ("blah\nVERDICT: B", 0.0),
    ("VERDICT: TIE", 0.5),
    ("verdict: a", 1.0),
    ("VERDICT: **B**", 0.0),
    ("I cannot decide.", 0.5),
    ("", 0.5),
])
def test_verdict_parsing(text, expected):
    assert parse_verdict(text)[0] == expected

def test_last_verdict_wins_when_the_judge_reconsiders():
    assert parse_verdict("VERDICT: A\nActually no.\nVERDICT: B")[0] == 0.0


# ---------------- prompt assembly ----------------
class _StubExample:
    metric_name = "score"
    def meta_description(self): return "META_D"
    def instruction(self): return "INSTRUCTION"
    def render_parent(self, node): return "PARENT" if node else "NO_PARENT"

def test_prompt_sections_follow_figure_1_order():
    builder = PromptBuilder(_StubExample(), None, 0)
    content = builder.build(Node(code="x"), [Lesson(title="L", summary="S")])[0]["content"]
    assert (content.index("META_D") < content.index("PARENT")
            < content.index("S") < content.index("INSTRUCTION"))

def test_prompt_omits_the_memory_block_when_there_are_no_lessons():
    content = PromptBuilder(_StubExample(), None, 0).build(None, [])[0]["content"]
    assert "Lessons from earlier attempts" not in content

def test_reprompt_appends_feedback_and_preserves_the_prefix():
    builder = PromptBuilder(_StubExample(), None, 0)
    base = builder.build(None, [])
    with_fb = builder.reprompt(base, "circles 2 and 3 overlap")
    assert base[0]["content"] not in ("",)
    assert with_fb[0]["content"].startswith(base[0]["content"])
    assert "circles 2 and 3 overlap" in with_fb[0]["content"]
    assert base[0]["content"] == builder.build(None, [])[0]["content"]  # not mutated


# ---------------- job splitting ----------------
def test_jobs_split_by_sample_count_not_by_target():
    jobs = [(0, "p", 8), (1, "p", 1), (2, "p", 1)]
    buckets = split_jobs(jobs, 2)
    loads = [sum(c for _, _, c in b) for b in buckets]
    assert sum(loads) == 10
    assert max(loads) - min(loads) <= 4      # the 8-sample job is split up
