"""
API endpoints for maintenance interventions
"""
import logging
from typing import Optional
from fastapi import APIRouter, HTTPException, Query
from app.models.intervention import InterventionListResponse
from app.services.orchestrator_service import OrchestratorService

logger = logging.getLogger(__name__)

router = APIRouter()
orchestrator_service = OrchestratorService()


@router.get("/", response_model=InterventionListResponse)
async def get_interventions(
    asset_id: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0)
):
    """Get maintenance interventions"""
    try:
        result = await orchestrator_service.get_interventions(
            asset_id=asset_id,
            status=status,
            limit=limit,
            offset=offset
        )
        
        # Convert to Intervention models
        from app.models.intervention import Intervention
        from datetime import datetime
        
        interventions = []
        raw_interventions = result.get('interventions', [])
        logger.info(f"Processing {len(raw_interventions)} interventions from database")
        
        # Parse datetime strings helper
        def parse_datetime(dt_str):
            if not dt_str:
                return None
            if isinstance(dt_str, str):
                return datetime.fromisoformat(dt_str.replace('Z', '+00:00'))
            return dt_str
        
        for interv in raw_interventions:
            try:
                # Convert id to string if it's an integer
                intervention_id = str(interv.get('id')) if interv.get('id') is not None else None
                
                interventions.append(Intervention(
                    id=intervention_id,
                    asset_id=interv.get('asset_id'),
                    work_order_id=interv.get('work_order_id'),
                    title=interv.get('title', 'Untitled Intervention'),
                    description=interv.get('description'),
                    status=interv.get('status', 'planned'),
                    priority=interv.get('priority', 'medium'),
                    scheduled_start=parse_datetime(interv.get('scheduled_start')),
                    scheduled_end=parse_datetime(interv.get('scheduled_end')),
                    actual_start=parse_datetime(interv.get('actual_start')),
                    actual_end=parse_datetime(interv.get('actual_end')),
                    assigned_to=interv.get('assigned_to'),
                    estimated_duration=interv.get('estimated_duration'),
                    actual_duration=interv.get('actual_duration'),
                    cost=interv.get('cost'),
                    metadata=interv.get('metadata'),
                    created_at=parse_datetime(interv.get('created_at')),
                    updated_at=parse_datetime(interv.get('updated_at'))
                ))
                logger.debug(f"Successfully converted intervention: {intervention_id}")
            except Exception as e:
                logger.warning(f"Erreur lors de la conversion d'une intervention: {e}, data: {interv}")
                continue
        
        return InterventionListResponse(
            interventions=interventions,
            total=result.get('total', 0),
            limit=limit,
            offset=offset
        )
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des interventions: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/active")
async def get_active_interventions():
    """Get all active interventions"""
    try:
        interventions = await orchestrator_service.get_active_interventions()
        return {"interventions": interventions, "count": len(interventions)}
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des interventions actives: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

