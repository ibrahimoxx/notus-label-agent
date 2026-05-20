from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from app.api.dependencies import get_batch_repo
from app.models.schemas import BatchCreate, BatchRead
from app.repositories.batch_repo import BatchRepository

router = APIRouter(prefix="/api/batches", tags=["batches"])

RepoDep = Annotated[BatchRepository, Depends(get_batch_repo)]


@router.post("", response_model=BatchRead, status_code=201)
async def create_batch(data: BatchCreate, repo: RepoDep) -> BatchRead:
    batch = await repo.create(data)
    return BatchRead.model_validate(batch)


@router.get("", response_model=list[BatchRead])
async def list_batches(
    repo: RepoDep,
    product_id: int | None = None,
    limit: int = 50,
) -> list[BatchRead]:
    if product_id is not None:
        return await repo.list_by_product(product_id, limit)
    return await repo.list_all(limit)


@router.get("/{batch_id}", response_model=BatchRead)
async def get_batch(batch_id: int, repo: RepoDep) -> BatchRead:
    batch = await repo.get_by_id(batch_id)
    if not batch:
        raise HTTPException(status_code=404, detail="Batch not found")
    return BatchRead.model_validate(batch)
