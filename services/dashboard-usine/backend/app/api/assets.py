"""
API endpoints for assets
"""
import logging
from typing import List, Optional
from datetime import datetime
from fastapi import APIRouter, HTTPException, Query, Depends
from app.models.asset import Asset, AssetDetail, AssetListResponse
from app.services.extraction_service import ExtractionService
from app.services.anomaly_service import AnomalyService
from app.services.rul_service import RULService
from app.database.postgresql import get_postgresql_service, PostgreSQLService

logger = logging.getLogger(__name__)

router = APIRouter()

# Service instances
extraction_service = ExtractionService()
anomaly_service = AnomalyService()
rul_service = RULService()


@router.get("/", response_model=AssetListResponse)
async def get_assets(
    page: int = Query(1, ge=1),
    page_size: int = Query(100, ge=1, le=1000),
    status: Optional[str] = None,
    asset_type: Optional[str] = None,
    db: PostgreSQLService = Depends(get_postgresql_service)
):
    """Get list of assets"""
    try:
        # Check if assets_metadata table exists
        try:
            table_check = await db.fetchval(
                "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'assets_metadata')"
            )
            if not table_check:
                logger.warning("Table 'assets_metadata' does not exist, returning empty list")
                return AssetListResponse(
                    assets=[],
                    total=0,
                    page=page,
                    page_size=page_size
                )
        except Exception as check_error:
            logger.warning(f"Could not check for assets_metadata table: {check_error}, returning empty list")
            return AssetListResponse(
                assets=[],
                total=0,
                page=page,
                page_size=page_size
            )
        
        # Query assets from database
        query = "SELECT * FROM assets_metadata WHERE 1=1"
        params = []
        
        if status:
            query += f" AND status = ${len(params) + 1}"
            params.append(status)
        
        if asset_type:
            query += f" AND type = ${len(params) + 1}"
            params.append(asset_type)
        
        query += f" LIMIT ${len(params) + 1} OFFSET ${len(params) + 2}"
        params.extend([page_size, (page - 1) * page_size])
        
        logger.debug(f"Executing query: {query} with params: {params}")
        rows = await db.fetch(query, *params)
        logger.debug(f"Fetched {len(rows)} rows from database")
        
        assets = []
        for row in rows:
            try:
                # Handle coordinates - it might be a dict or JSONB string
                coordinates = row.get('coordinates')
                if isinstance(coordinates, str):
                    import json
                    coordinates = json.loads(coordinates)
                
                asset = Asset(
                    id=row.get('asset_id') or row.get('id', ''),
                    name=row.get('name', 'Unknown'),
                    type=row.get('type', 'unknown'),
                    status=row.get('status', 'offline'),
                    location=row.get('location'),
                    coordinates=coordinates,
                    metadata=row.get('metadata'),
                    created_at=row.get('created_at'),
                    updated_at=row.get('updated_at')
                )
                assets.append(asset)
            except Exception as row_error:
                logger.warning(f"Error converting row to Asset: {row_error}, row: {row}")
                continue
        
        # Get total count
        count_query = "SELECT COUNT(*) FROM assets_metadata WHERE 1=1"
        count_params = []
        if status:
            count_query += f" AND status = ${len(count_params) + 1}"
            count_params.append(status)
        if asset_type:
            count_query += f" AND type = ${len(count_params) + 1}"
            count_params.append(asset_type)
        
        total = await db.fetchval(count_query, *count_params) if count_params else await db.fetchval("SELECT COUNT(*) FROM assets_metadata")
        
        return AssetListResponse(
            assets=assets,
            total=total or 0,
            page=page,
            page_size=page_size
        )
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des actifs: {e}", exc_info=True)
        logger.error(f"Exception type: {type(e).__name__}, message: {str(e)}")
        # Return empty list instead of 500 error to allow frontend to work
        return AssetListResponse(
            assets=[],
            total=0,
            page=page,
            page_size=page_size
        )


@router.get("/{asset_id}", response_model=AssetDetail)
async def get_asset_detail(
    asset_id: str,
    db: PostgreSQLService = Depends(get_postgresql_service)
):
    """Get detailed information about an asset"""
    try:
        # Get asset metadata
        row = await db.fetchrow(
            "SELECT * FROM assets_metadata WHERE asset_id = $1",
            asset_id
        )
        
        if not row:
            raise HTTPException(status_code=404, detail=f"Asset {asset_id} not found")
        
        asset = Asset(
            id=row['asset_id'],
            name=row['name'],
            type=row['type'],
            status=row['status'],
            location=row.get('location'),
            metadata=row.get('metadata'),
            created_at=row.get('created_at'),
            updated_at=row.get('updated_at')
        )
        
        # Get additional data
        # Latest RUL
        latest_rul = await rul_service.get_latest_rul(asset_id)
        current_rul = latest_rul.get('rul_prediction') if latest_rul else None
        
        # Anomaly count
        anomalies_result = await anomaly_service.get_anomalies(
            asset_id=asset_id,
            limit=1
        )
        anomaly_count = anomalies_result.get('total', 0)
        last_anomaly = None
        if anomalies_result.get('items'):
            last_anomaly_str = anomalies_result['items'][0].get('timestamp')
            if last_anomaly_str:
                last_anomaly = datetime.fromisoformat(last_anomaly_str.replace('Z', '+00:00'))
        
        # Features
        features_result = await extraction_service.get_features(asset_id, limit=10)
        features = features_result.get('features', [])
        
        return AssetDetail(
            **asset.model_dump(),
            current_rul=current_rul,
            anomaly_count=anomaly_count,
            last_anomaly=last_anomaly,
            features=features
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des détails de l'actif: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{asset_id}/features")
async def get_asset_features(
    asset_id: str,
    start_time: Optional[datetime] = Query(None),
    end_time: Optional[datetime] = Query(None),
    limit: int = Query(100, ge=1, le=1000)
):
    """Get features for an asset"""
    try:
        result = await extraction_service.get_features(
            asset_id=asset_id,
            start_time=start_time,
            end_time=end_time,
            limit=limit
        )
        return result
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des features: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

