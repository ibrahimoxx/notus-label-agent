import re
from datetime import date

from app.models.schemas import ParseResult
from app.utils.date_normalizer import normalize_to_iso

# Lines containing these tokens are stripped before LOT extraction
_LOT_NOISE_TOKENS = ("PPV", "PPC", "PPH", "ACL", "DOM", "FAB")

# LOT patterns — ordered by confidence (high to low). Stop at first match.
LOT_PATTERNS: list[tuple[str, float]] = [
    (r"LOT\s*[:\s]\s*([A-Z]{1,3}\d{3,}[A-Z]?\d*)", 1.0),   # LOT: RN620, LOT: PF1122001
    (r"LOT\s*[:\s]\s*([A-Z0-9]{4,12})", 1.0),                # LOT: CIX31, LOT: N3661
    (r"LOT\s+(\d{4,}\s?\d*)", 0.8),                          # LOT 251391 1, LOT 24040 51
    (r"\bLOT[:\s]+([A-Z0-9\s]{3,15})", 0.6),                 # generic
    (r"^([A-Z]{1,2}\d{3,}[A-Z]?\d*)\s+\d{2}[/\-]", 0.4),   # K4S598 05/24 without prefix
]

# DATE patterns — ordered by confidence (high to low). Stop at first pattern that matches.
DATE_PATTERNS: list[tuple[str, float]] = [
    (r"(?:EXP|PER|PEREMPTION)[:\s]+(\d{2}[/\s\-]\d{4})", 1.0),  # EXP: 04/2027
    (r"(?:EXP|PER)[:\s]+(\d{2}[/\s\-]\d{2})\b", 1.0),           # EXP 05/27
    (r"(?:EXP|PER)[:\s]+(\d{2}\s\d{4})", 1.0),                   # EXP 04 2028
    (r"(?:EXP|PER)[:\s]+(\d{2}[/\-]\d{2,4})", 0.8),              # generic EXP
    (r"\b(\d{2}[/\-]\d{4})\b(?!.*LOT)", 0.6),                    # fallback isolated date
]


def validate_parsed(lot: str | None, exp_iso: str | None) -> list[str]:
    warnings: list[str] = []
    if lot:
        clean = lot.replace(" ", "")
        if not 3 <= len(clean) <= 15:
            warnings.append(f"LOT longueur suspecte: {lot!r}")
        if re.match(r"^\d+\.\d+$", clean):
            warnings.append(f"LOT ressemble à un prix: {lot!r}")
    if exp_iso:
        try:
            y, m = exp_iso.split("-")
            exp_date = date(int(y), int(m), 1)
            if exp_date < date.today():
                warnings.append(f"Date expiration dans le passé: {exp_iso}")
        except ValueError:
            warnings.append(f"Date ISO malformée: {exp_iso!r}")
    return warnings


class LabelParser:
    def parse(self, raw_text: str) -> ParseResult:
        text = raw_text.upper()
        lot, conf_lot, lot_matches = self._extract_lot(text)
        exp_iso, conf_date, date_matches = self._extract_date(text)
        warnings = validate_parsed(lot, exp_iso)
        return ParseResult(
            lot_number=lot,
            expiration_date=exp_iso,
            confidence_lot=conf_lot,
            confidence_date=conf_date,
            confidence_global=(conf_lot + conf_date) / 2,
            raw_matches={"lot": lot_matches, "date": date_matches},
            warnings=warnings,
        )

    def _extract_lot(self, text: str) -> tuple[str | None, float, list[str]]:
        # Strip lines containing noise tokens before matching
        lines = [
            line for line in text.splitlines()
            if not any(tok in line for tok in _LOT_NOISE_TOKENS)
        ]
        clean = "\n".join(lines)
        all_matches: list[str] = []

        for pattern, conf in LOT_PATTERNS:
            matches = re.findall(pattern, clean, re.MULTILINE)
            all_matches.extend(matches)
            if matches:
                return matches[0].strip(), conf, all_matches

        return None, 0.0, all_matches

    def _extract_date(self, text: str) -> tuple[str | None, float, list[str]]:
        # If multiple dates found, keep the LATEST (expiration, not manufacture)
        for pattern, conf in DATE_PATTERNS:
            candidates: list[tuple[str, float]] = []
            for raw_match in re.findall(pattern, text):
                iso = normalize_to_iso(raw_match)
                if iso:
                    candidates.append((iso, conf))
            if candidates:
                candidates.sort(key=lambda x: x[0], reverse=True)
                best_iso, best_conf = candidates[0]
                return best_iso, best_conf, [c[0] for c in candidates]

        return None, 0.0, []
