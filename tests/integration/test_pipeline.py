import json
from pathlib import Path

import pytest

from app.services.image_preprocessor import ImagePreprocessor
from app.services.label_parser import LabelParser
from app.services.ocr_service import OCRService
from app.utils.image_utils import read_image

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures" / "images"
GROUND_TRUTH_FILE = Path(__file__).parent.parent / "fixtures" / "ground_truth.json"

GROUND_TRUTH = json.loads(GROUND_TRUTH_FILE.read_text(encoding="utf-8"))


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.parametrize("sample", GROUND_TRUTH)
async def test_full_pipeline_real_images(sample: dict) -> None:
    image_path = FIXTURES_DIR / sample["image"]
    assert image_path.exists(), f"Fixture image missing: {image_path}"

    img = read_image(image_path)
    variants = ImagePreprocessor().process(img)
    ocr_result = await OCRService().extract(variants)
    parsed = LabelParser().parse(ocr_result.raw_text)

    expected_lot = sample["expected"]["lot"]
    expected_date = sample["expected"]["expiration_date"]

    assert parsed.lot_number == expected_lot, (
        f"[{sample['image']}] LOT mismatch: expected={expected_lot!r} got={parsed.lot_number!r}\n"
        f"OCR text:\n{ocr_result.raw_text}"
    )
    assert parsed.expiration_date == expected_date, (
        f"[{sample['image']}] DATE mismatch: expected={expected_date!r} got={parsed.expiration_date!r}\n"
        f"OCR text:\n{ocr_result.raw_text}"
    )
