from fastapi import HTTPException, UploadFile

ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB


async def validate_image(image: UploadFile, max_mb: int = 10) -> bytes:
    """Read + validate uploaded image. Returns raw bytes."""
    if image.content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Format non supporté: {image.content_type}. Accepted: jpeg, png, webp",
        )
    data = await image.read()
    limit = max_mb * 1024 * 1024
    if len(data) > limit:
        raise HTTPException(
            status_code=413,
            detail=f"Image trop grande ({len(data) // 1024}KB > {max_mb}MB)",
        )
    return data
