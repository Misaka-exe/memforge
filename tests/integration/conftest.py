"""Shared fixtures for integration tests.

Integration tests require a reachable PostgreSQL + pgvector instance.
When the database is not available they are skipped automatically.
"""
from __future__ import annotations

import os

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import create_async_engine

from memforge.storage.database import build_engine, build_sessionmaker, init_db

DATABASE_URL = os.environ.get(
    "MEMFORGE_DATABASE_URL",
    "postgresql+asyncpg://memforge:memforge@localhost:55432/memforge",
)


async def _check_db() -> bool:
    engine = create_async_engine(DATABASE_URL, echo=False)
    try:
        async with engine.connect() as conn:
            await conn.execute(__import__("sqlalchemy").text("SELECT 1"))
        return True
    except Exception:
        return False
    finally:
        await engine.dispose()


@pytest.fixture(scope="session")
async def db_available() -> bool:
    return await _check_db()


@pytest_asyncio.fixture
async def session_factory(db_available):
    if not db_available:
        pytest.skip("PostgreSQL not available; skipping integration test")
    engine = build_engine(DATABASE_URL)
    await init_db(engine)
    factory = build_sessionmaker(engine)
    yield factory
    await engine.dispose()