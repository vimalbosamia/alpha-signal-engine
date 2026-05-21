from __future__ import annotations

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from libs.core.config.settings import get_settings

_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def get_engine() -> AsyncEngine:
    global _engine
    if _engine is None:
        settings = get_settings()
        # Auto-create data directory if using SQLite
        db_url = settings.storage.database_url
        if "sqlite" in db_url:
            import os
            # Extract path from URL like "sqlite+aiosqlite:///./data/signals.db"
            db_path = db_url.split("///")[-1] if "///" in db_url else ""
            if db_path:
                os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        _engine = create_async_engine(
            db_url,
            echo=False,
            future=True,
        )
    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    global _session_factory
    if _session_factory is None:
        _session_factory = async_sessionmaker(
            get_engine(), expire_on_commit=False
        )
    return _session_factory


async def init_db() -> None:
    """Create all tables. Call once at startup."""
    import libs.paper_trading.models  # noqa: F401
    from libs.data.storage.models import Base

    async with get_engine().begin() as conn:
        await conn.run_sync(Base.metadata.create_all, checkfirst=True)
