"""Async engine, session scope and schema initialisation."""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import AsyncIterator

from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from memforge.storage.models import Base

DEFAULT_DATABASE_URL = os.environ.get(
    "MEMFORGE_DATABASE_URL",
    "postgresql+asyncpg://memforge:memforge@localhost:55432/memforge",
)


def build_engine(database_url: str | None = None) -> AsyncEngine:
    return create_async_engine(database_url or DEFAULT_DATABASE_URL, echo=False, pool_pre_ping=True)


def build_sessionmaker(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(engine, expire_on_commit=False)


@asynccontextmanager
async def session_scope(
    session_factory: async_sessionmaker[AsyncSession],
) -> AsyncIterator[AsyncSession]:
    """Provide a transactional session; rollback on exception."""
    async with session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def init_db(engine: AsyncEngine) -> None:
    """Create the vector extension and all tables."""
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)