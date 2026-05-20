import pytest

from app.utils.date_normalizer import normalize_to_iso


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("10/2025", "2025-10"),
        ("05/27", "2027-05"),
        ("04-2028", "2028-04"),
        ("10 2026", "2026-10"),
        ("04 2027", "2027-04"),
        ("09-2025", "2025-09"),
        ("07/25", "2025-07"),
        ("08-2025", "2025-08"),
        ("10/27", "2027-10"),
        ("04/2027", "2027-04"),
    ],
)
def test_normalize_valid(raw: str, expected: str) -> None:
    assert normalize_to_iso(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [
        "13/2025",   # month 13 invalid
        "00/2025",   # month 0 invalid
        "2025/10",   # reversed (4-digit first)
        "abc",       # not digits
        "10/20/25",  # too many parts
        "",
    ],
)
def test_normalize_invalid(raw: str) -> None:
    assert normalize_to_iso(raw) is None


def test_two_digit_year_boundary_below_50() -> None:
    assert normalize_to_iso("01/49") == "2049-01"


def test_two_digit_year_boundary_50_and_above() -> None:
    assert normalize_to_iso("01/50") == "1950-01"
