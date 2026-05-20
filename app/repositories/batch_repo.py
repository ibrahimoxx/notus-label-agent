from datetime import date

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.domain import Product, ProductBatch
from app.models.schemas import BatchCreate, BatchRead, ParseResult


class BatchRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(self, data: BatchCreate) -> ProductBatch:
        batch = ProductBatch(
            product_id=data.product_id,
            lot_number=data.lot_number,
            expiration_date=data.expiration_date,
        )
        self._session.add(batch)
        await self._session.commit()
        await self._session.refresh(batch)
        return batch

    async def get_by_id(self, batch_id: int) -> ProductBatch | None:
        result = await self._session.execute(
            select(ProductBatch).where(ProductBatch.id == batch_id)
        )
        return result.scalar_one_or_none()

    async def list_by_product(self, product_id: int, limit: int = 50) -> list[BatchRead]:
        result = await self._session.execute(
            select(ProductBatch)
            .where(ProductBatch.product_id == product_id)
            .order_by(ProductBatch.created_at.desc())
            .limit(limit)
        )
        return [BatchRead.model_validate(b) for b in result.scalars().all()]

    async def list_all(self, limit: int = 50) -> list[BatchRead]:
        result = await self._session.execute(
            select(ProductBatch).order_by(ProductBatch.created_at.desc()).limit(limit)
        )
        return [BatchRead.model_validate(b) for b in result.scalars().all()]

    async def save_from_scan(
        self,
        product_name: str,
        parsed: ParseResult,
        image_path: str,
    ) -> None:
        """Background task: find or create product, then save batch."""
        from sqlalchemy import func

        result = await self._session.execute(
            select(Product).where(func.lower(Product.name) == product_name.lower())
        )
        product = result.scalar_one_or_none()

        if not product:
            product = Product(name=product_name, price=0.0)
            self._session.add(product)
            await self._session.flush()

        if parsed.lot_number and parsed.expiration_date:
            exp_parts = parsed.expiration_date.split("-")
            exp_date = date(int(exp_parts[0]), int(exp_parts[1]), 1)
            batch = ProductBatch(
                product_id=product.id,
                lot_number=parsed.lot_number,
                expiration_date=exp_date,
                scan_image_path=image_path,
                confidence_score=parsed.confidence_global,
            )
            self._session.add(batch)

        await self._session.commit()
