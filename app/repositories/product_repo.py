import json
from datetime import date

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.domain import Product
from app.models.schemas import ProductCreate, ProductRead


class ProductRepository:
    def __init__(self, session: AsyncSession, redis: "redis.Redis | None" = None) -> None:  # type: ignore[name-defined]
        self._session = session
        self._redis = redis

    async def get_all_cursor(self, cursor: int | None, limit: int) -> list[ProductRead]:
        cache_key = f"products:cursor:{cursor}:limit:{limit}"
        if self._redis:
            cached = await self._redis.get(cache_key)
            if cached:
                data: list[dict] = json.loads(cached)
                return [ProductRead.model_validate(d) for d in data]

        stmt = select(Product).order_by(Product.id).limit(limit)
        if cursor:
            stmt = stmt.where(Product.id > cursor)
        result = await self._session.execute(stmt)
        products = [ProductRead.model_validate(p) for p in result.scalars().all()]

        if self._redis:
            await self._redis.setex(cache_key, 300, json.dumps([p.model_dump(mode="json") for p in products]))

        return products

    async def get_by_id(self, product_id: int) -> Product | None:
        result = await self._session.execute(select(Product).where(Product.id == product_id))
        return result.scalar_one_or_none()

    async def create(self, data: ProductCreate) -> Product:
        product = Product(name=data.name, price=data.price)
        self._session.add(product)
        await self._session.commit()
        await self._session.refresh(product)
        if self._redis:
            await self._redis.delete("products:*")
        return product
