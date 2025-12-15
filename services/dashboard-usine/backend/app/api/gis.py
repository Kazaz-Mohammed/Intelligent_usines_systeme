"""
API endpoints for GIS (Geographic Information System)
"""
import logging
from typing import Optional
from fastapi import APIRouter, HTTPException, Query, Depends
from app.models.gis import AssetLocationListResponse, AssetLocation, FloorPlan, Point
from app.database.postgis import PostGISService
from app.database.postgresql import get_postgresql_service, PostgreSQLService

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/assets", response_model=AssetLocationListResponse)
async def get_asset_locations(
    building: Optional[str] = Query(None),
    zone: Optional[str] = Query(None),
    db: PostgreSQLService = Depends(get_postgresql_service)
):
    """Get asset locations with GIS data"""
    try:
        postgis_service = PostGISService(db)
        locations = await postgis_service.get_asset_locations(building, zone)
        
        # Get floor plan if available
        floor_plan = None
        try:
            row = await db.fetchrow("SELECT * FROM floor_plans LIMIT 1")
            if row:
                floor_plan = FloorPlan(
                    id=row['id'],
                    name=row['name'],
                    image_url=row.get('image_url'),
                    bounds={
                        "north_east": Point(
                            lat=row['bounds_north_east_lat'],
                            lon=row['bounds_north_east_lon']
                        ),
                        "south_west": Point(
                            lat=row['bounds_south_west_lat'],
                            lon=row['bounds_south_west_lon']
                        )
                    },
                    scale=row.get('scale'),
                    metadata=row.get('metadata')
                )
        except Exception as e:
            logger.warning(f"Impossible de récupérer le plan d'étage: {e}")
        
        return AssetLocationListResponse(
            locations=locations,
            total=len(locations),
            floor_plan=floor_plan
        )
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des localisations: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/assets/within-radius")
async def get_assets_within_radius(
    lat: float = Query(..., ge=-90, le=90),
    lon: float = Query(..., ge=-180, le=180),
    radius_meters: float = Query(100, ge=1, le=10000),
    db: PostgreSQLService = Depends(get_postgresql_service)
):
    """Get assets within a radius of a point"""
    try:
        postgis_service = PostGISService(db)
        center = Point(lat=lat, lon=lon)
        locations = await postgis_service.get_assets_within_radius(center, radius_meters)
        return {"locations": [loc.model_dump() for loc in locations], "count": len(locations)}
    except Exception as e:
        logger.error(f"Erreur lors de la recherche d'actifs dans un rayon: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/floor-plan")
async def get_floor_plan(
    floor_plan_id: Optional[str] = Query(None),
    db: PostgreSQLService = Depends(get_postgresql_service)
):
    """Get factory floor plan"""
    try:
        if floor_plan_id:
            row = await db.fetchrow("SELECT * FROM floor_plans WHERE id = $1", floor_plan_id)
        else:
            row = await db.fetchrow("SELECT * FROM floor_plans LIMIT 1")
        
        if not row:
            raise HTTPException(status_code=404, detail="Floor plan not found")
        
        return FloorPlan(
            id=row['id'],
            name=row['name'],
            image_url=row.get('image_url'),
            bounds={
                "north_east": Point(
                    lat=row['bounds_north_east_lat'],
                    lon=row['bounds_north_east_lon']
                ),
                "south_west": Point(
                    lat=row['bounds_south_west_lat'],
                    lon=row['bounds_south_west_lon']
                )
            },
            scale=row.get('scale'),
            metadata=row.get('metadata')
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur lors de la récupération du plan d'étage: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

