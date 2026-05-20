from datetime import date, datetime

from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    text: str
    confidence: float
    coordinates: list[tuple[float, float]]


class OCRResult(BaseModel):
    raw_text: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    engine_used: str
    processing_time_ms: int
    bounding_boxes: list[BoundingBox] = []


class ParseResult(BaseModel):
    lot_number: str | None
    expiration_date: str | None
    confidence_lot: float
    confidence_date: float
    confidence_global: float
    raw_matches: dict[str, list[str]] = {}
    warnings: list[str] = []


class ProductRead(BaseModel):
    id: int
    name: str
    price: float
    created_at: datetime

    model_config = {"from_attributes": True}


class ProductCreate(BaseModel):
    name: str
    price: float


class BatchRead(BaseModel):
    id: int
    product_id: int
    lot_number: str
    expiration_date: date
    scan_image_path: str | None
    confidence_score: float
    created_at: datetime

    model_config = {"from_attributes": True}


class BatchCreate(BaseModel):
    product_id: int
    lot_number: str
    expiration_date: date


class AnalyzeResponse(BaseModel):
    product_name: str
    lot_number: str | None
    expiration_date: str | None
    confidence_global: float
    confidence_lot: float
    confidence_date: float
    raw_ocr_text: str
    engine_used: str
    processing_time_ms: int
    image_path: str
    warnings: list[str] = []
