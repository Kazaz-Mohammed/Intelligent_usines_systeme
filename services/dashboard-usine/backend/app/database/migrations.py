"""
Database migration scripts
"""
import logging
from app.database.postgresql import PostgreSQLService
from app.database.postgis import PostGISService

logger = logging.getLogger(__name__)


async def run_migrations():
    """Run all database migrations"""
    logger.info("Exécution des migrations de base de données...")
    
    try:
        # Create PostgreSQL service
        postgresql_service = PostgreSQLService()
        await postgresql_service.create_pool()
        
        # Create PostGIS service and tables (optional, may fail if PostGIS not installed)
        try:
            postgis_service = PostGISService(postgresql_service)
            await postgis_service.create_tables()
        except Exception as e:
            logger.warning(f"PostGIS tables creation skipped: {e}")
        
        # Create additional tables
        await create_kpi_tables(postgresql_service)
        await create_asset_tables(postgresql_service)
        await create_interventions_table(postgresql_service)
        await create_anomalies_detected_table(postgresql_service)
        await insert_sample_assets(postgresql_service)
        
        logger.info("Migrations terminées avec succès")
        
    except Exception as e:
        logger.error(f"Erreur lors des migrations: {e}", exc_info=True)
        raise


async def create_kpi_tables(postgresql_service: PostgreSQLService):
    """Create KPI history tables"""
    await postgresql_service.execute("""
        CREATE TABLE IF NOT EXISTS kpis_history (
            id SERIAL PRIMARY KEY,
            asset_id VARCHAR(255),
            metric_name VARCHAR(100) NOT NULL,
            value FLOAT NOT NULL,
            unit VARCHAR(50),
            timestamp TIMESTAMP NOT NULL,
            metadata JSONB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    
    # Create indexes
    await postgresql_service.execute("""
        CREATE INDEX IF NOT EXISTS idx_kpis_history_asset_timestamp 
        ON kpis_history(asset_id, timestamp DESC);
    """)
    
    await postgresql_service.execute("""
        CREATE INDEX IF NOT EXISTS idx_kpis_history_metric_timestamp 
        ON kpis_history(metric_name, timestamp DESC);
    """)
    
    logger.info("Table kpis_history créée")


async def create_asset_tables(postgresql_service: PostgreSQLService):
    """Create asset metadata tables"""
    await postgresql_service.execute("""
        CREATE TABLE IF NOT EXISTS assets_metadata (
            asset_id VARCHAR(255) PRIMARY KEY,
            name VARCHAR(255) NOT NULL,
            type VARCHAR(100),
            status VARCHAR(50) DEFAULT 'operational',
            location VARCHAR(255),
            coordinates JSONB,
            metadata JSONB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    
    logger.info("Table assets_metadata créée")


async def create_interventions_table(postgresql_service: PostgreSQLService):
    """Create interventions table"""
    await postgresql_service.execute("""
        CREATE TABLE IF NOT EXISTS interventions (
            id SERIAL PRIMARY KEY,
            asset_id VARCHAR(255) NOT NULL,
            title VARCHAR(255) NOT NULL,
            description TEXT,
            type VARCHAR(100),
            status VARCHAR(50) DEFAULT 'scheduled',
            priority VARCHAR(50) DEFAULT 'medium',
            scheduled_start TIMESTAMP,
            scheduled_end TIMESTAMP,
            actual_start TIMESTAMP,
            actual_end TIMESTAMP,
            assigned_to VARCHAR(255),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    
    # Create indexes
    await postgresql_service.execute("""
        CREATE INDEX IF NOT EXISTS idx_interventions_asset ON interventions(asset_id);
    """)
    await postgresql_service.execute("""
        CREATE INDEX IF NOT EXISTS idx_interventions_status ON interventions(status);
    """)
    
    logger.info("Table interventions créée")


async def create_anomalies_detected_table(postgresql_service: PostgreSQLService):
    """Create anomalies_detected table for KPI calculations"""
    await postgresql_service.execute("""
        CREATE TABLE IF NOT EXISTS anomalies_detected (
            id SERIAL PRIMARY KEY,
            asset_id VARCHAR(255) NOT NULL,
            sensor_id VARCHAR(255),
            timestamp TIMESTAMP NOT NULL,
            severity VARCHAR(50) DEFAULT 'medium',
            is_anomaly BOOLEAN DEFAULT true,
            anomaly_score FLOAT,
            model_used VARCHAR(100),
            features JSONB,
            metadata JSONB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    
    # Create indexes
    await postgresql_service.execute("""
        CREATE INDEX IF NOT EXISTS idx_anomalies_asset ON anomalies_detected(asset_id);
    """)
    await postgresql_service.execute("""
        CREATE INDEX IF NOT EXISTS idx_anomalies_timestamp ON anomalies_detected(timestamp);
    """)
    await postgresql_service.execute("""
        CREATE INDEX IF NOT EXISTS idx_anomalies_severity ON anomalies_detected(severity);
    """)
    
    logger.info("Table anomalies_detected créée")


async def insert_sample_assets(postgresql_service: PostgreSQLService):
    """Insert sample assets from NASA C-MAPSS data"""
    sample_assets = [
        ('ENGINE_FD001_000', 'Turbofan Engine FD001-000', 'turbofan', 'operational'),
        ('ENGINE_FD002_000', 'Turbofan Engine FD002-000', 'turbofan', 'operational'),
        ('ENGINE_FD002_019', 'Turbofan Engine FD002-019', 'turbofan', 'operational'),
        ('ENGINE_FD004_000', 'Turbofan Engine FD004-000', 'turbofan', 'operational'),
    ]
    
    for asset_id, name, asset_type, status in sample_assets:
        try:
            await postgresql_service.execute("""
                INSERT INTO assets_metadata (asset_id, name, type, status)
                VALUES ($1, $2, $3, $4)
                ON CONFLICT (asset_id) DO NOTHING;
            """, asset_id, name, asset_type, status)
        except Exception as e:
            logger.warning(f"Could not insert asset {asset_id}: {e}")
    
    logger.info("Sample assets inserted")


if __name__ == "__main__":
    import asyncio
    asyncio.run(run_migrations())

