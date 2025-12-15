"""
Service client for orchestrateur-maintenance service
"""
import logging
import httpx
from typing import List, Dict, Any, Optional
from datetime import datetime
from app.config import settings

logger = logging.getLogger(__name__)


class OrchestratorService:
    """Client for orchestrateur-maintenance service"""
    
    def __init__(self):
        self.base_url = settings.orchestrator_url
        self.timeout = 30.0
    
    async def get_work_orders(
        self,
        asset_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> Dict[str, Any]:
        """Get work orders"""
        try:
            params = {"limit": limit, "offset": offset}
            if asset_id:
                params["asset_id"] = asset_id
            if status:
                params["status"] = status
            
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(
                    f"{self.base_url}/api/v1/work-orders",
                    params=params
                )
                response.raise_for_status()
                return response.json()
        except httpx.HTTPError as e:
            logger.error(f"Erreur HTTP lors de la récupération des ordres de travail: {e}")
            return {"work_orders": [], "total": 0, "limit": limit, "offset": offset}
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des ordres de travail: {e}", exc_info=True)
            return {"work_orders": [], "total": 0, "limit": limit, "offset": offset}
    
    async def get_interventions(
        self,
        asset_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> Dict[str, Any]:
        """Get interventions - first tries external orchestrator, falls back to local database"""
        # First try external orchestrator service
        try:
            params = {"limit": limit, "offset": offset}
            if asset_id:
                params["asset_id"] = asset_id
            if status:
                params["status"] = status
            
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(
                    f"{self.base_url}/api/v1/interventions",
                    params=params
                )
                response.raise_for_status()
                return response.json()
        except Exception as e:
            logger.debug(f"Orchestrator service not available, falling back to local database: {e}")
        
        # Fallback: query local database
        try:
            from app.database.postgresql import get_postgresql_service
            db = await get_postgresql_service()
            
            query = "SELECT * FROM interventions WHERE 1=1"
            params_list = []
            
            if asset_id:
                query += f" AND asset_id = ${len(params_list) + 1}"
                params_list.append(asset_id)
            
            if status:
                query += f" AND status = ${len(params_list) + 1}"
                params_list.append(status)
            
            query += f" ORDER BY scheduled_start DESC LIMIT ${len(params_list) + 1} OFFSET ${len(params_list) + 2}"
            params_list.extend([limit, offset])
            
            rows = await db.fetch(query, *params_list)
            
            # Get total count
            count_query = "SELECT COUNT(*) FROM interventions WHERE 1=1"
            count_params = []
            if asset_id:
                count_query += f" AND asset_id = ${len(count_params) + 1}"
                count_params.append(asset_id)
            if status:
                count_query += f" AND status = ${len(count_params) + 1}"
                count_params.append(status)
            
            total = await db.fetchval(count_query, *count_params) if count_params else await db.fetchval("SELECT COUNT(*) FROM interventions")
            
            interventions = []
            for row in rows:
                interventions.append({
                    "id": row.get("id"),
                    "asset_id": row.get("asset_id"),
                    "title": row.get("title"),
                    "description": row.get("description"),
                    "type": row.get("type"),
                    "status": row.get("status"),
                    "priority": row.get("priority"),
                    "scheduled_start": row.get("scheduled_start").isoformat() if row.get("scheduled_start") else None,
                    "scheduled_end": row.get("scheduled_end").isoformat() if row.get("scheduled_end") else None,
                    "actual_start": row.get("actual_start").isoformat() if row.get("actual_start") else None,
                    "actual_end": row.get("actual_end").isoformat() if row.get("actual_end") else None,
                    "assigned_to": row.get("assigned_to"),
                    "created_at": row.get("created_at").isoformat() if row.get("created_at") else None,
                    "updated_at": row.get("updated_at").isoformat() if row.get("updated_at") else None,
                })
            
            return {"interventions": interventions, "total": total or 0, "limit": limit, "offset": offset}
            
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des interventions depuis la base locale: {e}", exc_info=True)
            return {"interventions": [], "total": 0, "limit": limit, "offset": offset}
    
    async def get_active_interventions(self) -> List[Dict[str, Any]]:
        """Get all active interventions"""
        try:
            result = await self.get_interventions(status="in_progress", limit=1000)
            return result.get("interventions", [])
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des interventions actives: {e}", exc_info=True)
            return []
    
    async def get_status(self) -> Dict[str, Any]:
        """Get orchestrateur-maintenance service status"""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.base_url}/api/v1/health")
                response.raise_for_status()
                return response.json()
        except Exception as e:
            logger.warning(f"Impossible de récupérer le statut du service orchestrateur-maintenance: {e}")
            return {"status": "unknown"}

