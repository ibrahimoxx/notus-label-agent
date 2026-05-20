import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Annotated

import aiofiles
import numpy as np
from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, UploadFile
from loguru import logger

from app.api.dependencies import get_batch_repo
from app.models.schemas import AnalyzeResponse
from app.repositories.batch_repo import BatchRepository
from app.services.camera_service import validate_image
from app.services.image_preprocessor import ImagePreprocessor
from app.services.label_parser import LabelParser
from app.services.ocr_service import OCRService
from app.utils.image_utils import read_image

router = APIRouter(prefix="/api", tags=["analyze"])

_preprocessor = ImagePreprocessor()
_ocr = OCRService()
_parser = LabelParser()


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(
    background: BackgroundTasks,
    image: UploadFile = File(...),
    product_name: str = Form(...),
    batch_repo: Annotated[BatchRepository, Depends(get_batch_repo)] = ...,  # type: ignore[assignment]
) -> AnalyzeResponse:
    start = time.perf_counter()
    logger.info("analyze_start", product_name=product_name, filename=image.filename)

    # 1. Validate + save image
    raw_bytes = await validate_image(image)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    filename = f"scan_{ts}_{uuid.uuid4().hex[:8]}.jpeg"
    upload_path = Path("uploads") / filename
    async with aiofiles.open(upload_path, "wb") as f:
        await f.write(raw_bytes)
    logger.info("image_saved", path=str(upload_path))

    # 2. Preprocessing
    img_array: np.ndarray = read_image(upload_path)
    variants = _preprocessor.process(img_array)
    logger.info("preprocessing_done", ms=int((time.perf_counter() - start) * 1000))

    # 3. OCR multi-pass
    ocr_result = await _ocr.extract(variants)
    logger.info(
        "ocr_done", engine=ocr_result.engine_used, confidence=ocr_result.confidence
    )

    # 4. Parse LOT + date
    parsed = _parser.parse(ocr_result.raw_text)
    logger.info(
        "parsing_done",
        lot=parsed.lot_number,
        exp=parsed.expiration_date,
        confidence=parsed.confidence_global,
    )

    # 5. Low-confidence warning
    if parsed.confidence_global < 0.5:
        parsed.warnings.append(
            "Résultat incertain — vérification manuelle recommandée"
        )

    # 6. Persist to DB in background (non-blocking)
    if parsed.lot_number and parsed.expiration_date:
        background.add_task(
            batch_repo.save_from_scan, product_name, parsed, str(upload_path)
        )

    total_ms = int((time.perf_counter() - start) * 1000)
    return AnalyzeResponse(
        product_name=product_name,
        lot_number=parsed.lot_number,
        expiration_date=parsed.expiration_date,
        confidence_global=parsed.confidence_global,
        confidence_lot=parsed.confidence_lot,
        confidence_date=parsed.confidence_date,
        raw_ocr_text=ocr_result.raw_text,
        engine_used=ocr_result.engine_used,
        processing_time_ms=total_ms,
        image_path=str(upload_path),
        warnings=parsed.warnings,
    )
