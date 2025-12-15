"""
API endpoints for Grafana integration
"""
import logging
from typing import Optional
from fastapi import APIRouter, HTTPException, Query
from app.config import settings

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/dashboard-url")
async def get_grafana_dashboard_url(
    asset_id: Optional[str] = Query(None),
    dashboard_id: Optional[str] = Query(None)
):
    """Get Grafana dashboard URL with embedded panels"""
    try:
        if not settings.grafana_url:
            raise HTTPException(status_code=503, detail="Grafana integration not configured")
        
        # Construct Grafana dashboard URL
        base_url = settings.grafana_url.rstrip('/')
        
        if dashboard_id:
            url = f"{base_url}/d/{dashboard_id}"
        else:
            # Default dashboard
            url = f"{base_url}/d/maintenance-dashboard"
        
        # Add asset filter if provided
        if asset_id:
            url += f"?var-asset={asset_id}"
        
        # Add embed parameters
        url += "&kiosk=tv"  # TV mode (no navigation)
        
        return {
            "url": url,
            "embed_url": url + "&theme=light",
            "asset_id": asset_id
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur lors de la génération de l'URL Grafana: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

