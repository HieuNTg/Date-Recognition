"""Unit tests for DateParser — pure logic, no models or datasets required.

Run from the repository root:  pytest
"""

from datetime import datetime, timedelta

import pytest

from src.utils import DateParser


@pytest.fixture(scope="module")
def parser():
    return DateParser()


# ── clean(): prefix stripping & OCR misread repair ──────────────────────────

@pytest.mark.parametrize("raw,expected", [
    ("EXP: 01/02/2026", "01/02/2026"),
    ("HSD 31/12/2025", "31/12/2025"),
    ("NSX.10/05/2024", "10/05/2024"),
    ("BB 2026", "2026"),
])
def test_clean_strips_prefixes(parser, raw, expected):
    assert parser.clean(raw) == expected


def test_clean_fixes_letter_o_to_zero(parser):
    # 'O' adjacent to a digit should become '0'
    assert parser.clean("O1/O2/2O26") == "01/02/2026"


# ── try_parse(): valid vs garbage ───────────────────────────────────────────

def test_try_parse_valid_date(parser):
    assert parser.try_parse("01/02/2026") is not None


def test_try_parse_garbage_returns_none(parser):
    assert parser.try_parse("not-a-date-zzz") is None


# ── get_max_date(): picks the latest parsable date ──────────────────────────

def test_get_max_date_picks_latest(parser):
    dates = ["01/01/2024", "EXP 01/01/2026", "01/01/2025"]
    assert parser.get_max_date(dates) == "EXP 01/01/2026"


def test_get_max_date_falls_back_when_none_parse(parser):
    dates = ["garbage1", "garbage2"]
    assert parser.get_max_date(dates) == "garbage1"


def test_get_max_date_empty(parser):
    assert parser.get_max_date([]) is None


# ── evaluate_expiry(): valid / warning / expired ────────────────────────────

def _fmt(dt):
    return dt.strftime("%Y-%m-%d")


def test_evaluate_expiry_valid(parser):
    future = datetime.now() + timedelta(days=400)
    status, delta = parser.evaluate_expiry(_fmt(future))
    assert status == "valid"
    assert delta > parser.warning_days


def test_evaluate_expiry_warning(parser):
    soon = datetime.now() + timedelta(days=10)
    status, delta = parser.evaluate_expiry(_fmt(soon))
    assert status == "warning"
    assert 0 <= delta <= parser.warning_days


def test_evaluate_expiry_expired(parser):
    past = datetime.now() - timedelta(days=10)
    status, delta = parser.evaluate_expiry(_fmt(past))
    assert status == "expired"
    assert delta < 0


def test_evaluate_expiry_unparsable(parser):
    status, delta = parser.evaluate_expiry("???")
    assert status is None and delta is None
