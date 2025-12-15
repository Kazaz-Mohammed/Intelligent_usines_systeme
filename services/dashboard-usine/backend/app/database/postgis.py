"""
PostGIS spatial database operations
"""
import logging
from typing import List, Optional, Dict, Any
from app.database.postgresql import PostgreSQLService
from app.models.gis import Point, AssetLocation

logger = logging.getLogger(__name__)


class PostGISService:
    """Service for PostGIS spatial operations"""
    
    def __init__(self, postgresql_service: PostgreSQLService):
        self.db = postgresql_service
    
    async def initialize_extension(self):
        """Initialize PostGIS extension if not already enabled"""
        try:
            await self.db.execute("CREATE EXTENSION IF NOT EXISTS postgis;")
            logger.info("Extension PostGIS vérifiée/activée")
        except Exception as e:
            logger.warning(f"Impossible d'activer PostGIS (peut-être déjà activé): {e}")
    
    async def create_tables(self):
        """Create spatial tables"""
        await self.initialize_extension()
        
        # Create asset_locations table with PostGIS geometry
        await self.db.execute("""
            CREATE TABLE IF NOT EXISTS asset_locations (
                id SERIAL PRIMARY KEY,
                asset_id VARCHAR(255) UNIQUE NOT NULL,
                asset_name VARCHAR(255),
                location GEOMETRY(POINT, 4326),
                floor_level INTEGER,
                building VARCHAR(100),
                zone VARCHAR(100),
                metadata JSONB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        
        # Create spatial index
        await self.db.execute("""
            CREATE INDEX IF NOT EXISTS idx_asset_locations_geom 
            ON asset_locations USING GIST (location);
        """)
        
        # Create floor_plans table
        await self.db.execute("""
            CREATE TABLE IF NOT EXISTS floor_plans (
                id VARCHAR(255) PRIMARY KEY,
                name VARCHAR(255) NOT NULL,
                image_url TEXT,
                bounds_north_east_lat FLOAT,
                bounds_north_east_lon FLOAT,
                bounds_south_west_lat FLOAT,
                bounds_south_west_lon FLOAT,
                scale FLOAT,
                metadata JSONB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        
        logger.info("Tables PostGIS créées")
    
    async def add_asset_location(
        self,
        asset_id: str,
        asset_name: str,
        point: Point,
        floor_level: Optional[int] = None,
        building: Optional[str] = None,
        zone: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Add or update asset location"""
        try:
            import json
            metadata_json = json.dumps(metadata) if metadata else None
            
            await self.db.execute("""
                INSERT INTO asset_locations 
                (asset_id, asset_name, location, floor_level, building, zone, metadata, updated_at)
                VALUES ($1, $2, ST_SetSRID(ST_MakePoint($3, $4), 4326), $5, $6, $7, $8::jsonb, CURRENT_TIMESTAMP)
                ON CONFLICT (asset_id) 
                DO UPDATE SET
                    asset_name = EXCLUDED.asset_name,
                    location = EXCLUDED.location,
                    floor_level = EXCLUDED.floor_level,
                    building = EXCLUDED.building,
                    zone = EXCLUDED.zone,
                    metadata = EXCLUDED.metadata,
                    updated_at = CURRENT_TIMESTAMP;
            """, asset_id, asset_name, point.lon, point.lat, floor_level, building, zone, metadata_json)
            
            return True
        except Exception as e:
            logger.error(f"Erreur lors de l'ajout de la localisation de l'actif {asset_id}: {e}", exc_info=True)
            return False
    
    async def get_asset_locations(self, building: Optional[str] = None, zone: Optional[str] = None) -> List[AssetLocation]:
        """Get all asset locations"""
        try:
            query = """
                SELECT 
                    asset_id,
                    asset_name,
                    ST_X(location) as lon,
                    ST_Y(location) as lat,
                    floor_level,
                    building,
                    zone,
                    metadata
                FROM asset_locations
                WHERE 1=1
            """
            params = []
            
            if building:
                query += " AND building = $" + str(len(params) + 1)
                params.append(building)
            
            if zone:
                query += " AND zone = $" + str(len(params) + 1)
                params.append(zone)
            
            rows = await self.db.fetch(query, *params)
            
            locations = []
            for row in rows:
                locations.append(AssetLocation(
                    asset_id=row['asset_id'],
                    asset_name=row['asset_name'],
                    point=Point(lat=row['lat'], lon=row['lon']),
                    floor_level=row['floor_level'],
                    building=row['building'],
                    zone=row['zone'],
                    metadata=row['metadata']
                ))
            
            return locations
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des localisations: {e}", exc_info=True)
            return []
    
    async def get_assets_within_radius(self, center: Point, radius_meters: float) -> List[AssetLocation]:
        """Get assets within a radius of a point"""
        try:
            rows = await self.db.fetch("""
                SELECT 
                    asset_id,
                    asset_name,
                    ST_X(location) as lon,
                    ST_Y(location) as lat,
                    floor_level,
                    building,
                    zone,
                    metadata,
                    ST_Distance(
                        location::geography,
                        ST_SetSRID(ST_MakePoint($1, $2), 4326)::geography
                    ) as distance
                FROM asset_locations
                WHERE ST_DWithin(
                    location::geography,
                    ST_SetSRID(ST_MakePoint($1, $2), 4326)::geography,
                    $3
                )
                ORDER BY distance;
            """, center.lon, center.lat, radius_meters)
            
            locations = []
            for row in rows:
                locations.append(AssetLocation(
                    asset_id=row['asset_id'],
                    asset_name=row['asset_name'],
                    point=Point(lat=row['lat'], lon=row['lon']),
                    floor_level=row['floor_level'],
                    building=row['building'],
                    zone=row['zone'],
                    metadata=row['metadata']
                ))
            
            return locations
        except Exception as e:
            logger.error(f"Erreur lors de la recherche d'actifs dans un rayon: {e}", exc_info=True)
            return []

