import sys
import pytest

from runio.progress import PhaseTimer, _FallbackBar, _NullBar, _fmt_seconds, make_bar


def test_null_bar_when_disabled():
    bar = make_bar(10, "x", enabled=False)
    assert isinstance(bar, _NullBar)
    bar.update(3); bar.set_postfix_str("hi"); bar.close()   # all no-ops

def test_bar_counts_and_closes(capsys):
    bar = make_bar(4, "verify", unit="rollout")
    for _ in range(4):
        bar.update(1)
    bar.close()
    err = capsys.readouterr().err
    assert "verify" in err and "4" in err

def test_fallback_bar_reports_a_percentage(capsys):
    bar = _FallbackBar(total=10, desc="gen", min_interval=0.0)
    bar.update(5)
    bar.close()
    assert "50.0%" in capsys.readouterr().err

def test_fallback_bar_survives_a_zero_total(capsys):
    bar = _FallbackBar(total=0, desc="unknown", min_interval=0.0)
    bar.update(3)
    bar.close()
    assert "unknown" in capsys.readouterr().err      # no ZeroDivisionError

def test_fallback_bar_throttles_redraws(capsys):
    bar = _FallbackBar(total=1000, desc="gen", min_interval=60.0)
    for _ in range(500):
        bar.update(1)
    bar.close()
    # One draw at construction, one at close; the 500 updates are throttled out.
    assert capsys.readouterr().err.count("gen") <= 3

def test_postfix_is_shown(capsys):
    bar = _FallbackBar(total=2, desc="verify", min_interval=0.0)
    bar.set_postfix_str("valid=1/2")
    bar.close()
    assert "valid=1/2" in capsys.readouterr().err

@pytest.mark.parametrize("seconds,expected", [
    (0, "00:00"), (61, "01:01"), (3661, "1:01:01"), (float("inf"), "--:--"),
])
def test_time_formatting(seconds, expected):
    assert _fmt_seconds(seconds) == expected

def test_phase_timer_announces_and_reports(capsys):
    with PhaseTimer("extract lessons"):
        pass
    err = capsys.readouterr().err
    assert "extract lessons ..." in err and "done in" in err

def test_phase_timer_is_silent_when_disabled(capsys):
    with PhaseTimer("quiet", enabled=False):
        pass
    assert capsys.readouterr().err == ""

def test_phase_timer_still_reports_when_the_body_raises(capsys):
    with pytest.raises(RuntimeError):
        with PhaseTimer("boom"):
            raise RuntimeError("x")
    assert "done in" in capsys.readouterr().err


def test_generation_drives_a_token_bar(capsys, tmp_path):
    """The tick callback must reach the backbone, not be silently dropped."""
    from config import Config
    from llm.generation import InProcessGenerator
    from llm.mock import MockBackbone

    cfg = Config().generation
    cfg.max_new_tokens = 8
    gen = InProcessGenerator(MockBackbone(seed=0), cfg, progress=True)
    out = gen.generate([(0, [{"role": "user", "content": "pack 4 circles"}], 3)])
    assert len(out[0]) == 3
    assert "generate 1/1" in capsys.readouterr().err

def test_progress_can_be_turned_off_end_to_end(capsys, tmp_path):
    from config import load_config
    from core.engine import Engine

    cfg = load_config([
        "--example", "circle_packing", "--backend", "mock", "--steps", "1",
        "--n-select", "1", "--k-children", "2", "--progress", "false",
        "--output-root", str(tmp_path), "--set", "example.params.num_circles=4",
    ]).config
    Engine(cfg).run()
    err = capsys.readouterr().err
    assert "verify" not in err and "generate" not in err
