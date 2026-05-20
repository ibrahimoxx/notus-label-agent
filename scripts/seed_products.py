"""Seed sample pharmaceutical product catalogue."""
import asyncio

from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.core.config import settings
from app.models.domain import Base, Product

SAMPLE_PRODUCTS = [
    {"name": "Doliprane 1000mg", "price": 28.50},
    {"name": "Amoxicilline 500mg", "price": 45.00},
    {"name": "Ibuprofene 400mg", "price": 32.00},
    {"name": "Metformine 500mg", "price": 18.00},
    {"name": "Omeprazole 20mg", "price": 55.00},
    {"name": "Melatonine 1mg", "price": 89.00},
    {"name": "Doliprane 500mg", "price": 22.00},
    {"name": "Augmentin 1g", "price": 120.00},
    {"name": "Ventoline 100mcg", "price": 75.00},
]


async def main() -> None:
    engine = create_async_engine(settings.database_url)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as session:
        for data in SAMPLE_PRODUCTS:
            session.add(Product(name=data["name"], price=data["price"]))
        await session.commit()
    print(f"Seeded {len(SAMPLE_PRODUCTS)} products.")
    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
