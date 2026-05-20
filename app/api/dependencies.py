from collections.abc import AsyncGenerator
from typing import Annotated

import redis.asyncio as aioredis
from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.config import settings
from app.repositories.batch_repo import BatchRepository
from app.repositories.product_repo import ProductRepository

_engine = create_async_engine(settings.database_url, echo=settings.debug)
_session_factory = async_sessionmaker(_engine, expire_on_commit=False)

_redis_pool: aioredis.Redis | None = None


def get_redis() -> aioredis.Redis | None:
    global _redis_pool
    if _redis_pool is None:
        try:
            _redis_pool = aioredis.from_url(settings.redis_url, decode_responses=True)
        except Exception:
            return None
    return _redis_pool


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    async with _session_factory() as session:
        yield session


SessionDep = Annotated[AsyncSession, Depends(get_session)]


async def get_product_repo(session: SessionDep) -> ProductRepository:
    return ProductRepository(session, get_redis())


async def get_batch_repo(session: SessionDep) -> BatchRepository:
    return BatchRepository(session)
