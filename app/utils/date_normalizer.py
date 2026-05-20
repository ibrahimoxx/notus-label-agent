import re


def normalize_to_iso(raw: str) -> str | None:
    """
    Normalize raw date string to YYYY-MM format.

    Examples:
      '10/2025' -> '2025-10'
      '05/27'   -> '2027-05'  (2-digit year: <50 -> 20XX, else 19XX)
      '10 2026' -> '2026-10'
      '04-2028' -> '2028-04'
    """
    cleaned = re.sub(r"[\s\-]", "/", raw.strip())
    parts = cleaned.split("/")
    if len(parts) != 2:
        return None

    month_str, year_str = parts
    if not month_str.isdigit() or not year_str.isdigit():
        return None

    month = int(month_str)
    if not 1 <= month <= 12:
        return None

    if len(year_str) == 2:
        y = int(year_str)
        year_str = f"20{year_str}" if y < 50 else f"19{year_str}"

    if len(year_str) != 4:
        return None

    return f"{year_str}-{month:02d}"
