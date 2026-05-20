from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from app.api.dependencies import get_product_repo
from app.models.schemas import ProductRead
from app.repositories.product_repo import ProductRepository

router = APIRouter(prefix="/api/products", tags=["products"])

RepoDep = Annotated[ProductRepository, Depends(get_product_repo)]


@router.get("", response_model=list[ProductRead])
async def list_products(
    repo: RepoDep,
    cursor: int | None = None,
    limit: int = 20,
) -> list[ProductRead]:
    return await repo.get_all_cursor(cursor, limit)


@router.get("/{product_id}", response_model=ProductRead)
async def get_product(product_id: int, repo: RepoDep) -> ProductRead:
    product = await repo.get_by_id(product_id)
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")
    return ProductRead.model_validate(product)
