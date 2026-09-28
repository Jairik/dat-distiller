"""PII detection: the regex baseline, the optional Presidio layer, the actions.

The baseline is the layer that must never be wrong in the expensive direction:
a false positive costs a user a redaction, a false negative ships real PII into
a Card. So every detector is pinned with both positive and negative fixtures,
including the near-misses (a dotted version string, a 999.x address, a 16-digit
number that fails Luhn) that a looser regex would happily report.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pandas as pd
import pytest

from dat_distiller.pii import (
    ACTIONS,
    DETECTORS_BY_NAME,
    PII_CHECK_KIND,
    REDACTION,
    _reset_presidio_engine,
    apply_actions,
    luhn,
    mask_span,
    presidio_findings,
    register_pii_checks,
    scan,
    scan_dataframe,
    summarize_findings,
)
from dat_distiller.store.provenance import PROVENANCE_COLUMN

# -- the baseline, one detector at a time -------------------------------------

#: detector -> strings that MUST be reported.
POSITIVES: dict[str, list[str]] = {
    "email": [
        "jane.doe@example.com",
        "a+tag@sub.domain.co.uk",
        "write to SUPPORT@Example.COM please",
        "first.last+tag@xn--80ak6aa92e.com",
    ],
    "phone": [
        "555-123-4567",
        "(555) 123-4567",
        "555.123.4567",
        "+1 555 123 4567",
        "5551234567",
    ],
    "ssn": ["123-45-6789", "078-05-1120"],
    "credit_card": [
        "4111111111111111",  # Visa
        "4111 1111 1111 1111",
        "4111-1111-1111-1111",
        "5500 0000 0000 0004",  # Mastercard
        "378282246310005",  # Amex, 15 digits
    ],
    "ip": ["192.168.1.1", "8.8.8.8", "10.0.0.255", "172.16.254.3"],
}

#: detector -> strings that MUST NOT be reported, most of them near-misses.
NEGATIVES: dict[str, list[str]] = {
    "email": [
        "jane.doe",
        "@example.com",
        "no-at-sign.example.com",
        "a@b",
        "user@localhost",
    ],
    "phone": [
        "1.212.555.0187",  # a version string
        "version 2.1.0 released",
        "555-1234",  # too short
        "123-456-7890",  # area code cannot start with 1
        "call 55512345678901234 for the order",  # embedded in a longer number
    ],
    "ssn": [
        "000-45-6789",  # never issued
        "666-45-6789",  # never issued
        "900-45-6789",  # withheld
        "123456789",  # no dashes
        "123-456-7890",
    ],
    "credit_card": [
        "4111111111111112",  # fails Luhn
        "1234567890123",  # 13 digits, fails Luhn
        "1",
    ],
    "ip": [
        "999.1.1.1",  # octet out of range
        "1.2.3.4.5",  # five parts
        "1.2.3",  # too few
        "256.100.50.1",
        "1.212.555.0187",  # a version string: one octet is too big
    ],
}


@pytest.mark.parametrize("detector", sorted(POSITIVES))
def test_positive_fixtures_are_detected(detector: str) -> None:
    for value in POSITIVES[detector]:
        assert DETECTORS_BY_NAME[detector].spans(value), f"{value!r} should match {detector}"


@pytest.mark.parametrize("detector", sorted(NEGATIVES))
def test_negative_fixtures_are_not_detected(detector: str) -> None:
    for value in NEGATIVES[detector]:
        assert not DETECTORS_BY_NAME[detector].spans(
            value
        ), f"{value!r} should not match {detector}"


def test_a_version_string_with_an_impossible_octet_is_not_a_phone_or_ip() -> None:
    # the specific bug this guards: release notes full of dotted version
    # numbers. `1.212.555.0187` has an out-of-range octet, so no IP or phone
    # pattern can claim it.
    text = "upgraded to 1.212.555.0187 and 3.0.12 today"
    assert scan_dataframe(pd.DataFrame({"notes": [text]})) == []


def test_a_well_formed_dotted_quad_is_reported_as_an_ip() -> None:
    # `2.4.0.1` is indistinguishable from an IPv4 address, and a false negative
    # ships real PII into a Card — so the baseline reports it and lets the user
    # mask or drop. Documented tradeoff, not an oversight.
    findings = scan_dataframe(pd.DataFrame({"notes": ["released 2.4.0.1 today"]}))
    assert [(f["column"], f["detector"]) for f in findings] == [("notes", "ip")]


def test_detectors_do_not_fire_on_each_others_examples() -> None:
    # each positive belongs to exactly one detector, so counts stay honest
    for detector, values in POSITIVES.items():
        for value in values:
            hits = {name for name, d in DETECTORS_BY_NAME.items() if d.spans(value)}
            assert hits == {detector}, f"{value!r} also matched {hits - {detector}}"


def test_luhn() -> None:
    assert luhn("4111111111111111")
    assert not luhn("4111111111111112")
    # separators are tolerated, a lone digit is not a card
    assert luhn("4111-1111-1111-1111")
    assert not luhn("4")


def test_mask_span_never_leaks_the_value() -> None:
    assert mask_span("jane.doe@example.com") == "j***.com"
    assert mask_span("555-123-4567") == "5***"
    assert "jane" not in mask_span("jane.doe@example.com")


# -- scanning a frame ---------------------------------------------------------


def frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "email": ["jane@example.com", "bob@example.org", "clean@example.com"],
            "note": ["call 555-123-4567", "nothing here", "ssn 123-45-6789"],
            "age": [30, 40, 50],
            "score": [1.5, 2.5, 3.5],
        }
    )


def test_scan_reports_one_finding_per_detector_per_column() -> None:
    findings = scan_dataframe(frame())
    assert {(f["column"], f["detector"]) for f in findings} == {
        ("email", "email"),
        ("note", "phone"),
        ("note", "ssn"),
    }
    by_key = {(f["column"], f["detector"]): f for f in findings}
    assert by_key[("email", "email")]["count"] == 3
    assert by_key[("note", "phone")]["count"] == 1


def test_numeric_columns_are_skipped() -> None:
    # scanning numbers would be slow and PII lives in text
    assert all(f["column"] not in {"age", "score"} for f in scan_dataframe(frame()))


def test_provenance_is_never_scanned() -> None:
    df = pd.DataFrame({"email": ["jane@example.com"], PROVENANCE_COLUMN: ["jane@example.com"]})
    assert [f["column"] for f in scan_dataframe(df)] == ["email"]


def test_findings_carry_masked_examples_not_raw_values() -> None:
    findings = scan_dataframe(pd.DataFrame({"email": ["jane@example.com"]}))
    example = findings[0]["example_cells"][0]
    assert "jane@example.com" not in example
    assert "j***.com" in example
    assert findings[0]["row_examples"] == [0]


def test_missing_values_are_skipped_not_stringified() -> None:
    df = pd.DataFrame({"email": ["jane@example.com", None, float("nan")]})
    findings = scan_dataframe(df)
    assert findings[0]["count"] == 1


def test_a_clean_frame_reports_nothing() -> None:
    assert scan_dataframe(pd.DataFrame({"note": ["all clean", "nothing here"]})) == []


def test_summarize() -> None:
    summary = summarize_findings(scan_dataframe(frame()))
    assert summary["columns"] == ["email", "note"]
    assert summary["total_findings"] == 5
    assert summary["detectors"] == {"email": 3, "phone": 1, "ssn": 1}


# -- Presidio is optional -----------------------------------------------------


class FakePresidioEngine:
    """Stands in for `presidio_analyzer.AnalyzerEngine`."""

    def __init__(self, results=None, explode: bool = False) -> None:
        self._results = results or {}
        self._explode = explode

    def analyze(self, text, entities=None, language="en"):
        if self._explode:
            raise RuntimeError("spaCy model missing")
        return [
            SimpleNamespace(entity_type=entity, start=start, end=end)
            for entity, start, end in self._results.get(text, [])
        ]


@pytest.fixture(autouse=True)
def _clean_engine():
    _reset_presidio_engine()
    yield
    _reset_presidio_engine()


def test_presidio_contributes_nothing_when_it_is_not_installed() -> None:
    # the real import fails in this environment; the baseline still stands
    assert presidio_findings(pd.DataFrame({"note": ["Jane Doe lives at 1 Main St"]})) == []


def test_presidio_is_used_only_when_installed(monkeypatch) -> None:
    module = SimpleNamespace(
        AnalyzerEngine=lambda: FakePresidioEngine(
            {"Jane Doe lives in Boston": [("PERSON", 0, 8)]}
        )
    )
    monkeypatch.setitem(sys.modules, "presidio_analyzer", module)
    _reset_presidio_engine()
    findings = presidio_findings(pd.DataFrame({"bio": ["Jane Doe lives in Boston", "anon"]}))
    assert [(f["column"], f["detector"], f["count"]) for f in findings] == [
        ("bio", "presidio:PERSON", 1)
    ]
    assert findings[0]["example_cells"] == ["J*** lives in Boston"]


def test_a_presidio_that_cannot_build_its_engine_is_treated_as_absent(monkeypatch) -> None:
    monkeypatch.setitem(
        sys.modules, "presidio_analyzer", SimpleNamespace(AnalyzerEngine=_boom)
    )
    _reset_presidio_engine()
    assert presidio_findings(pd.DataFrame({"bio": ["Jane Doe"]})) == []


def test_one_unreadable_cell_does_not_stop_a_presidio_scan(monkeypatch) -> None:
    monkeypatch.setitem(
        sys.modules,
        "presidio_analyzer",
        SimpleNamespace(AnalyzerEngine=lambda: FakePresidioEngine(explode=True)),
    )
    _reset_presidio_engine()
    assert presidio_findings(pd.DataFrame({"bio": ["Jane Doe"]})) == []


def _boom():
    raise ValueError("no spaCy model installed")


def test_scan_combines_both_layers(monkeypatch) -> None:
    monkeypatch.setitem(
        sys.modules,
        "presidio_analyzer",
        SimpleNamespace(AnalyzerEngine=lambda: FakePresidioEngine({"Jane Doe": [("PERSON", 0, 8)]})),
    )
    _reset_presidio_engine()
    df = pd.DataFrame({"bio": ["Jane Doe", "jane@example.com"]})
    assert {(f["column"], f["detector"]) for f in scan(df)} == {
        ("bio", "presidio:PERSON"),
        ("bio", "email"),
    }


# -- actions: mask and drop ---------------------------------------------------


def test_mask_redacts_every_span_in_place_and_keeps_the_column() -> None:
    df = pd.DataFrame({"email": ["jane@example.com", "bob@example.org"], "age": [1, 2]})
    out, summary = apply_actions(df, {"email": "mask"})
    assert list(out.columns) == ["email", "age"]
    assert out["email"].tolist() == [REDACTION, REDACTION]
    assert summary["columns"]["email"]["replacements"] == 2
    assert summary["masked"] == ["email"]
    # the input frame is untouched
    assert df["email"].tolist() == ["jane@example.com", "bob@example.org"]


def test_mask_only_touches_the_named_column() -> None:
    df = pd.DataFrame({"email": ["jane@example.com"], "note": ["jane@example.com"]})
    out, _ = apply_actions(df, {"email": "mask"})
    assert out["note"].tolist() == ["jane@example.com"]


def test_mask_leaves_text_around_the_span_intact() -> None:
    df = pd.DataFrame({"note": ["reach me at jane@example.com ok?"]})
    out, _ = apply_actions(df, {"note": "mask"})
    assert out["note"].tolist() == [f"reach me at {REDACTION} ok?"]


def test_mask_on_a_column_with_no_findings_is_a_reported_noop() -> None:
    df = pd.DataFrame({"note": ["nothing here"]})
    out, summary = apply_actions(df, {"note": "mask"})
    assert out["note"].tolist() == ["nothing here"]
    assert summary["columns"]["note"]["replacements"] == 0
    assert summary["noops"] == ["note"]


def test_drop_removes_the_whole_column() -> None:
    df = pd.DataFrame({"email": ["jane@example.com"], "age": [1]})
    out, summary = apply_actions(df, {"email": "drop"})
    assert list(out.columns) == ["age"]
    assert summary["dropped"] == ["email"]


def test_both_actions_in_one_pass() -> None:
    df = pd.DataFrame(
        {"email": ["jane@example.com"], "ssn": ["123-45-6789"], "age": [1]}
    )
    out, summary = apply_actions(df, {"email": "mask", "ssn": "drop"})
    assert list(out.columns) == ["email", "age"]
    assert summary["masked"] == ["email"] and summary["dropped"] == ["ssn"]


def test_provenance_is_protected_from_both_actions() -> None:
    df = pd.DataFrame({"x": ["a"], PROVENANCE_COLUMN: ["jane@example.com"]})
    out, summary = apply_actions(df, {PROVENANCE_COLUMN: "drop"})
    assert PROVENANCE_COLUMN in out.columns
    assert "protected" in summary["columns"][PROVENANCE_COLUMN]["note"]


def test_an_unknown_action_is_rejected() -> None:
    with pytest.raises(ValueError, match="pii action"):
        apply_actions(pd.DataFrame({"a": ["b"]}), {"a": "encrypt"})
    assert ACTIONS == ("mask", "drop")


# -- the warn action is the Check --------------------------------------------


class FakeChecks:
    def __init__(self) -> None:
        self.registered: list[dict] = []
        self.rows: list[SimpleNamespace] = []

    def register(self, **kwargs):
        self.registered.append(kwargs)
        check = SimpleNamespace(kind=kwargs["kind"], details=kwargs["details"])
        self.rows.append(check)
        return check

    def list_for_subject(self, subject_type, subject_id):
        return self.rows


def test_warn_raises_one_warning_check_per_affected_column() -> None:
    checks = FakeChecks()
    findings = scan_dataframe(frame())
    registered = register_pii_checks(checks, "v1", findings)

    assert [c.details["column"] for c in registered] == ["email", "note"]
    first = checks.registered[0]
    assert first["kind"] == PII_CHECK_KIND
    assert first["severity"] == "warning"
    assert first["subject_type"] == "dataset_version"
    assert first["subject_id"] == "v1"
    assert first["details"]["detectors"] == {"email": 3}
    assert "email" in first["message"]


def test_check_details_never_carry_raw_pii() -> None:
    checks = FakeChecks()
    register_pii_checks(checks, "v1", scan_dataframe(frame()))
    assert "jane@example.com" not in str(checks.registered)


def test_rescanning_the_same_version_does_not_duplicate_checks() -> None:
    checks = FakeChecks()
    findings = scan_dataframe(frame())
    assert len(register_pii_checks(checks, "v1", findings)) == 2
    # a second scan (an upload retry) adds nothing new
    assert register_pii_checks(checks, "v1", findings) == []
    assert len(checks.registered) == 2
