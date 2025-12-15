"""
PostgreSQL database connection pool
"""
import logging
from typing import Optional, List, Dict, Any
from app.config import settings

logger = logging.getLogger(__name__)

# Try to import asyncpg, fallback to SQLAlchemy if not available
try:
    import asyncpg
    ASYNCPG_AVAILABLE = True
except ImportError:
    ASYNCPG_AVAILABLE = False
    logger.warning("asyncpg not available, using SQLAlchemy async instead")

if not ASYNCPG_AVAILABLE:
    from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
    from sqlalchemy import text


class PostgreSQLService:
    """Service for PostgreSQL database connections"""
    
    def __init__(self):
        # Initialize both attributes regardless of which mode we use
        # This prevents AttributeError when checking self.engine in get_pool()
        self.pool: Optional[asyncpg.Pool] = None
        self.engine = None
        self.async_session = None
        
        if ASYNCPG_AVAILABLE:
            self._use_asyncpg = True
        else:
            self._use_asyncpg = False
    
    async def create_pool(self):
        """Create connection pool"""
        if self._use_asyncpg:
            if self.pool is None:
                try:
                    self.pool = await asyncpg.create_pool(
                        host=settings.database_host,
                        port=settings.database_port,
                        database=settings.database_name,
                        user=settings.database_user,
                        password=settings.database_password,
                        min_size=2,
                        max_size=10,
                        command_timeout=60
                    )
                    logger.info(f"Pool de connexions PostgreSQL créé (asyncpg): {settings.database_host}:{settings.database_port}/{settings.database_name}")
                except Exception as e:
                    logger.error(f"Erreur lors de la création du pool PostgreSQL: {e}", exc_info=True)
                    raise
        else:
            # Use SQLAlchemy async
            database_url = f"postgresql+psycopg://{settings.database_user}:{settings.database_password}@{settings.database_host}:{settings.database_port}/{settings.database_name}"
            self.engine = create_async_engine(database_url, echo=False)
            self.async_session = async_sessionmaker(self.engine, class_=AsyncSession, expire_on_commit=False)
            logger.info(f"Pool de connexions PostgreSQL créé (SQLAlchemy): {settings.database_host}:{settings.database_port}/{settings.database_name}")
    
    async def get_pool(self):
        """Get connection pool, creating it if necessary"""
        if self.pool is None and self.engine is None:
            await self.create_pool()
        if self._use_asyncpg:
            return self.pool
        return self.async_session
    
    async def execute(self, query: str, *args):
        """Execute a query"""
        if self._use_asyncpg:
            pool = await self.get_pool()
            async with pool.acquire() as conn:
                return await conn.execute(query, *args)
        else:
            async with self.async_session() as session:
                result = await session.execute(text(query), args if args else {})
                await session.commit()
                return result
    
    async def fetch(self, query: str, *args) -> List[Dict[str, Any]]:
        """Fetch rows from a query"""
        if self._use_asyncpg:
            pool = await self.get_pool()
            async with pool.acquire() as conn:
                rows = await conn.fetch(query, *args)
                return [dict(row) for row in rows]
        else:
            async with self.async_session() as session:
                result = await session.execute(text(query), args if args else {})
                rows = result.fetchall()
                return [dict(row._mapping) for row in rows]
    
    async def fetchrow(self, query: str, *args) -> Optional[Dict[str, Any]]:
        """Fetch a single row from a query"""
        if self._use_asyncpg:
            pool = await self.get_pool()
            async with pool.acquire() as conn:
                row = await conn.fetchrow(query, *args)
                return dict(row) if row else None
        else:
            async with self.async_session() as session:
                result = await session.execute(text(query), args if args else {})
                row = result.fetchone()
                return dict(row._mapping) if row else None
    
    async def fetchval(self, query: str, *args):
        """Fetch a single value from a query"""
        if self._use_asyncpg:
            pool = await self.get_pool()
            async with pool.acquire() as conn:
                return await conn.fetchval(query, *args)
        else:
            async with self.async_session() as session:
                result = await session.execute(text(query), args if args else {})
                row = result.fetchone()
                return row[0] if row else None
    
    async def close(self):
        """Close the connection pool"""
        if self._use_asyncpg:
            if self.pool:
                await self.pool.close()
                self.pool = None
                logger.info("Pool de connexions PostgreSQL fermé")
        else:
            if self.engine:
                await self.engine.dispose()
                self.engine = None
                logger.info("Pool de connexions PostgreSQL fermé")


# Global instance
_postgresql_service: Optional[PostgreSQLService] = None


async def get_postgresql_service() -> PostgreSQLService:
    """Get global PostgreSQL service instance"""
    global _postgresql_service
    if _postgresql_service is None:
        _postgresql_service = PostgreSQLService()
        await _postgresql_service.create_pool()
    return _postgresql_service

