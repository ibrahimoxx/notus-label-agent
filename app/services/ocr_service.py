import asyncio
import time
from typing import Any

import pytesseract

from app.models.schemas import BoundingBox, OCRResult
from app.services.image_preprocessor import PreprocessedVariants

import numpy as np

TESSERACT_CONFIGS: dict[str, str] = {
    "standard": "--oem 3 --psm 6 -l fra+eng",
    "sparse": "--oem 3 --psm 11 -l fra+eng",
    "whitelist": (
        '--oem 3 --psm 6 -c tessedit_char_whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789/:.- " -l fra'
    ),
    "arabic": "--oem 3 --psm 6 -l ara+fra",
}

PADDLE_CONFIG: dict[str, Any] = {
    "use_angle_cls": True,
    "lang": "fr",
    "det_db_thresh": 0.3,
    "det_db_box_thresh": 0.5,
    "rec_batch_num": 6,
    "use_gpu": False,
    "show_log": False,
}


class OCRService:
    def __init__(self) -> None:
        self._paddle: Any = None

    def _get_paddle(self) -> Any:
        if self._paddle is None:
            from paddleocr import PaddleOCR  # lazy import — heavy init

            self._paddle = PaddleOCR(**PADDLE_CONFIG)
        return self._paddle

    async def extract(self, variants: PreprocessedVariants) -> OCRResult:
        """4-pass cascade. Returns fused best result."""
        loop = asyncio.get_running_loop()
        results: list[OCRResult] = []

        results.append(
            await loop.run_in_executor(
                None, self._paddle_extract, variants.original_gray, "paddle_gray"
            )
        )
        results.append(
            await loop.run_in_executor(
                None, self._tesseract_extract, variants.thresholded, "standard", "tesseract_psm6"
            )
        )
        results.append(
            await loop.run_in_executor(
                None, self._tesseract_extract, variants.inverted, "sparse", "tesseract_psm11"
            )
        )
        results.append(
            await loop.run_in_executor(
                None, self._paddle_extract, variants.clahe_enhanced, "paddle_clahe"
            )
        )

        return self._fuse(results)

    def _paddle_extract(self, gray: np.ndarray, label: str) -> OCRResult:
        start = time.perf_counter()
        try:
            paddle = self._get_paddle()
            result = paddle.ocr(gray, cls=True)
            elapsed = int((time.perf_counter() - start) * 1000)

            if not result or not result[0]:
                return OCRResult(
                    raw_text="",
                    confidence=0.0,
                    engine_used=label,
                    processing_time_ms=elapsed,
                )

            lines: list[str] = []
            boxes: list[BoundingBox] = []
            confidences: list[float] = []

            for line in result[0]:
                coords, (text, conf) = line
                lines.append(text)
                confidences.append(float(conf))
                boxes.append(
                    BoundingBox(
                        text=text,
                        confidence=float(conf),
                        coordinates=[(float(p[0]), float(p[1])) for p in coords],
                    )
                )

            avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
            return OCRResult(
                raw_text="\n".join(lines),
                confidence=min(avg_conf, 1.0),
                engine_used=label,
                processing_time_ms=elapsed,
                bounding_boxes=boxes,
            )
        except Exception:
            elapsed = int((time.perf_counter() - start) * 1000)
            return OCRResult(
                raw_text="", confidence=0.0, engine_used=label, processing_time_ms=elapsed
            )

    def _tesseract_extract(
        self, gray: np.ndarray, config_key: str, label: str
    ) -> OCRResult:
        start = time.perf_counter()
        try:
            config = TESSERACT_CONFIGS[config_key]
            text: str = pytesseract.image_to_string(gray, config=config)
            data = pytesseract.image_to_data(
                gray, config=config, output_type=pytesseract.Output.DICT
            )
            elapsed = int((time.perf_counter() - start) * 1000)

            confs = [
                int(c) for c in data["conf"] if str(c).lstrip("-").isdigit() and int(c) >= 0
            ]
            avg_conf = (sum(confs) / len(confs) / 100.0) if confs else 0.0

            return OCRResult(
                raw_text=text.strip(),
                confidence=min(avg_conf, 1.0),
                engine_used=label,
                processing_time_ms=elapsed,
            )
        except Exception:
            elapsed = int((time.perf_counter() - start) * 1000)
            return OCRResult(
                raw_text="", confidence=0.0, engine_used=label, processing_time_ms=elapsed
            )

    def _fuse(self, results: list[OCRResult]) -> OCRResult:
        """Keep best confidence result; concat unique texts for max parser recall."""
        best = max(results, key=lambda r: r.confidence)
        # dedup-preserving-order concat of all non-empty texts
        seen: dict[str, None] = {}
        for r in results:
            if r.raw_text:
                seen[r.raw_text] = None
        all_text = "\n".join(seen.keys())

        return OCRResult(
            raw_text=all_text,
            confidence=best.confidence,
            engine_used=f"fusion({best.engine_used})",
            processing_time_ms=sum(r.processing_time_ms for r in results),
            bounding_boxes=best.bounding_boxes,
        )
