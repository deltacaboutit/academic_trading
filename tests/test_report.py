"""Checks of the plain-English text, the Pattern Guide and its example shapes."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from atrade.config import DEFAULTS
from atrade.demo import demo_prices
from atrade.examples import CASES, GAP_CASES, example_chart
from atrade.pattern_info import PATTERN_INFO, STATUS_HELP

S = dict(DEFAULTS)


@pytest.fixture(scope="module")
def res():
    from atrade.analysis import analyze
    from atrade.cleaning import clean
    df, _, _ = clean(demo_prices(), S)
    return analyze("DEMO", S, prices=df.iloc[:1940])   # a day on which two patterns are forming


def test_every_pattern_has_a_description_and_an_example_the_tool_finds():
    assert set(PATTERN_INFO) == set(CASES) | set(GAP_CASES)
    assert len(PATTERN_INFO) == 21
    for rule, info in PATTERN_INFO.items():
        assert {"category", "bias", "looks", "means", "rule", "target", "pattern_stop"} <= set(info)
        df, sg = example_chart(rule)
        assert sg is not None, f"the detectors did not find the {rule} example"
        assert sg.target is not None and sg.time_limit is not None


def test_every_status_is_explained():
    from atrade import forming as F
    for st in (F.FORMING, F.FAILED, F.EXPIRED, F.CONFIRMED, F.TARGET, F.STOPPED, F.TIME):
        assert st in STATUS_HELP


def test_summary_and_explanations(res):
    from atrade.report import describe, indicator_sentences, summary_text
    txt = summary_text(res)
    assert res["rating_now"] in txt and "forming now" in txt and "not financial advice" in txt
    assert any(x.startswith("**Trend") for x in indicator_sentences(res))
    assert res["forming"], "the fixture day should have forming patterns"
    for u in res["forming"]:
        t = describe(u, res)
        assert "Status: Forming" in t and "would be a breakout" in t
    for sg in res["patterns"][-5:]:
        t = describe(sg, res)
        assert sg.rule in t and "Status:" in t and "4%" in t
    failed = [u for u in res["setups"] if u.status in ("Failed", "Expired")]
    assert failed and "Status: " + failed[0].status in describe(failed[0], res)
