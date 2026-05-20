import pytest

from app.services.label_parser import LabelParser

parser = LabelParser()


@pytest.mark.parametrize(
    "text, expected_lot",
    [
        ("LOT: RN620 EXP: 10/2026", "RN620"),
        ("LOT: N3661\nPER: 10/2027", "N3661"),
        ("LOT: CIX31\nEXP 07/25", "CIX31"),
        ("LOT 251391 1\nEXP 04 2028", "251391 1"),
        ("LOT 24040 51 PER: 04/2027", "24040 51"),
        ("LOT: PF1122001 PER: 10/2025", "PF1122001"),
        ("LOT: 2JV0951 EXP: 09/2025 FAB: 09/2023", "2JV0951"),
        ("K4S598 05/24", "K4S598"),
    ],
)
def test_extract_lot(text: str, expected_lot: str) -> None:
    result = parser.parse(text)
    assert result.lot_number == expected_lot, (
        f"Expected LOT={expected_lot!r}, got {result.lot_number!r}\nText: {text!r}"
    )


@pytest.mark.parametrize(
    "text, expected_iso",
    [
        ("EXP: 10/2026", "2026-10"),
        ("PER: 10/27", "2027-10"),
        ("EXP 04 2028", "2028-04"),
        ("EXP: 10-2022", "2022-10"),
        ("EXP 05/27", "2027-05"),
        ("EXP: 09-2025", "2025-09"),
        ("EXP 04/2027", "2027-04"),
        ("EXP: 08-2025", "2025-08"),
    ],
)
def test_extract_date(text: str, expected_iso: str) -> None:
    result = parser.parse(text)
    assert result.expiration_date == expected_iso, (
        f"Expected EXP={expected_iso!r}, got {result.expiration_date!r}\nText: {text!r}"
    )


def test_ignore_dom_keep_exp() -> None:
    text = "DOM: 01/2023 EXP: 10/2026"
    result = parser.parse(text)
    assert result.expiration_date == "2026-10"


def test_ignore_ppv_keep_lot() -> None:
    text = "PPV: 45.50 MAD\nLOT: RN620"
    result = parser.parse(text)
    assert result.lot_number == "RN620"


def test_ignore_ppc_keep_lot() -> None:
    text = "PPC: 139.00 DH\nLot: CIX31\n: 07/25"
    result = parser.parse(text)
    assert result.lot_number == "CIX31"


def test_multiple_dates_keep_latest() -> None:
    # DOM 2023, EXP 2026 — should keep 2026
    text = "EXP: 10/2026\nDOM: 10/2023"
    result = parser.parse(text)
    assert result.expiration_date == "2026-10"


def test_fab_line_ignored_for_lot() -> None:
    text = "LOT: 2JV0951\nEXP: 09-2025\nFAB: 10-2022"
    result = parser.parse(text)
    assert result.lot_number == "2JV0951"
    assert result.expiration_date == "2025-09"


def test_confidence_global_average() -> None:
    result = parser.parse("LOT: RN620 EXP: 10/2026")
    assert result.confidence_global == (result.confidence_lot + result.confidence_date) / 2


def test_no_match_returns_none() -> None:
    result = parser.parse("some random text without lot or date")
    assert result.lot_number is None
    assert result.expiration_date is None
    assert result.confidence_global == 0.0
