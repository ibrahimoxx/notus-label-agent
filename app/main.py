from pathlib import Path

import pytesseract
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from prometheus_fastapi_instrumentator import Instrumentator

from app.api.dependencies import get_redis
from app.api.routes import analyze, batches, products
from app.core.config import settings
from app.core.logging import configure_logging

configure_logging()

app = FastAPI(title=settings.app_name, version=settings.app_version)

Instrumentator().instrument(app).expose(app)

app.include_router(products.router)
app.include_router(batches.router)
app.include_router(analyze.router)


@app.get("/api/health")
async def health() -> dict[str, object]:
    status: dict[str, object] = {"status": "ok"}

    # Tesseract binary check
    try:
        pytesseract.get_tesseract_version()
        status["tesseract"] = "ok"
    except Exception as e:
        status["tesseract"] = f"error: {e}"
        status["status"] = "degraded"

    # Redis check
    try:
        redis = get_redis()
        if redis:
            await redis.ping()
            status["redis"] = "ok"
        else:
            status["redis"] = "unavailable"
    except Exception as e:
        status["redis"] = f"error: {e}"

    return status


Path(settings.uploads_dir).mkdir(exist_ok=True)

_static = Path("static")
if _static.exists():
    app.mount("/", StaticFiles(directory="static", html=True), name="static")
