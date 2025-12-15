"""
Service for calculating KPIs (MTBF, MTTR, OEE)
"""
import logging
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
from app.database.postgresql import PostgreSQLService
from app.models.kpi import KPISummary, KPI

logger = logging.getLogger(__name__)


class KPIService:
    """Service for KPI calculations"""
    
    def __init__(self, postgresql_service: PostgreSQLService):
        self.db = postgresql_service
    
    async def calculate_mtbf(self, asset_id: Optional[str] = None, days: int = 30) -> Optional[float]:
        """
        Calculate MTBF (Mean Time Between Failures) in hours
        
        MTBF = Total Operating Time / Number of Failures
        """
        try:
            start_date = datetime.now() - timedelta(days=days)
            
            # Get failure events (anomalies with critical criticality or maintenance events)
            # Note: Uses anomaly_detections table from detection-anomalies service
            query = """
                SELECT COUNT(*) as failure_count,
                       MIN(timestamp) as first_failure,
                       MAX(timestamp) as last_failure
                FROM (
                    SELECT timestamp
                    FROM anomaly_detections
                    WHERE criticality = 'critical' AND is_anomaly = true
                    AND timestamp >= $1
                    """ + (f"AND asset_id = $2" if asset_id else "") + """
                ) failures
            """
            
            params = [start_date]
            if asset_id:
                params.append(asset_id)
                result = await self.db.fetchrow(query, *params)
            else:
                result = await self.db.fetchrow(query.replace("AND asset_id = $2", ""), *params)
            
            failure_count = result['failure_count'] if result else 0
            
            if failure_count == 0:
                return None  # No failures, MTBF is undefined
            
            # Calculate total operating time
            # For simplicity, assume 24/7 operation minus maintenance time
            total_hours = days * 24
            
            # Subtract maintenance time (if we have maintenance records)
            # For now, we'll use a simple calculation
            mtbf = total_hours / failure_count if failure_count > 0 else None
            
            return mtbf
        except Exception as e:
            logger.error(f"Erreur lors du calcul du MTBF: {e}", exc_info=True)
            return None
    
    async def calculate_mttr(self, asset_id: Optional[str] = None, days: int = 30) -> Optional[float]:
        """
        Calculate MTTR (Mean Time To Repair) in hours
        
        MTTR = Total Repair Time / Number of Repairs
        """
        try:
            start_date = datetime.now() - timedelta(days=days)
            
            # Get repair events from interventions
            query = """
                SELECT 
                    COUNT(*) as repair_count,
                    SUM(EXTRACT(EPOCH FROM (actual_end - actual_start)) / 3600) as total_repair_time
                FROM interventions
                WHERE status = 'completed'
                AND actual_start IS NOT NULL
                AND actual_end IS NOT NULL
                AND actual_start >= $1
                """ + (f"AND asset_id = $2" if asset_id else "") + """
            """
            
            params = [start_date]
            if asset_id:
                params.append(asset_id)
                result = await self.db.fetchrow(query, *params)
            else:
                result = await self.db.fetchrow(query.replace("AND asset_id = $2", ""), *params)
            
            repair_count = result['repair_count'] if result else 0
            total_repair_time = result['total_repair_time'] if result and result['total_repair_time'] else 0
            
            if repair_count == 0:
                return None  # No repairs, MTTR is undefined
            
            mttr = total_repair_time / repair_count if repair_count > 0 else None
            
            return mttr
        except Exception as e:
            logger.error(f"Erreur lors du calcul du MTTR: {e}", exc_info=True)
            return None
    
    async def calculate_oee(self, asset_id: str, days: int = 30) -> Optional[float]:
        """
        Calculate OEE (Overall Equipment Effectiveness) as percentage
        
        OEE = Availability × Performance × Quality
        
        For now, simplified calculation:
        - Availability = (Operating Time - Downtime) / Operating Time
        - Performance = Actual Production / Ideal Production (simplified to 1.0)
        - Quality = Good Parts / Total Parts (simplified to 1.0)
        """
        try:
            start_date = datetime.now() - timedelta(days=days)
            
            # Calculate downtime from maintenance interventions
            query = """
                SELECT 
                    SUM(EXTRACT(EPOCH FROM (COALESCE(actual_end, scheduled_end) - COALESCE(actual_start, scheduled_start))) / 3600) as downtime_hours
                FROM interventions
                WHERE asset_id = $1
                AND (status = 'in_progress' OR status = 'completed')
                AND scheduled_start >= $2
            """
            
            result = await self.db.fetchrow(query, asset_id, start_date)
            downtime_hours = result['downtime_hours'] if result and result['downtime_hours'] else 0
            
            # Total operating time
            total_hours = days * 24
            
            # Availability
            availability = ((total_hours - downtime_hours) / total_hours * 100) if total_hours > 0 else 0
            
            # Simplified OEE (assuming perfect performance and quality)
            oee = availability
            
            return min(100.0, max(0.0, oee))
        except Exception as e:
            logger.error(f"Erreur lors du calcul de l'OEE: {e}", exc_info=True)
            return None
    
    async def get_kpi_summary(self, asset_id: Optional[str] = None, days: int = 30) -> KPISummary:
        """Get KPI summary"""
        mtbf = await self.calculate_mtbf(asset_id, days)
        mttr = await self.calculate_mttr(asset_id, days)
        
        # Calculate availability (works for all assets if no specific asset_id)
        availability = None
        oee = None
        try:
            start_date = datetime.now() - timedelta(days=days)
            
            if asset_id:
                query = """
                    SELECT 
                        SUM(EXTRACT(EPOCH FROM (COALESCE(actual_end, scheduled_end) - COALESCE(actual_start, scheduled_start))) / 3600) as downtime_hours
                    FROM interventions
                    WHERE asset_id = $1
                    AND (status = 'in_progress' OR status = 'completed')
                    AND scheduled_start >= $2
                """
                result = await self.db.fetchrow(query, asset_id, start_date)
            else:
                # Calculate for all assets
                query = """
                    SELECT 
                        SUM(EXTRACT(EPOCH FROM (COALESCE(actual_end, scheduled_end) - COALESCE(actual_start, scheduled_start))) / 3600) as downtime_hours,
                        COUNT(DISTINCT asset_id) as asset_count
                    FROM interventions
                    WHERE (status = 'in_progress' OR status = 'completed')
                    AND scheduled_start >= $1
                """
                result = await self.db.fetchrow(query, start_date)
            
            downtime_hours = result['downtime_hours'] if result and result['downtime_hours'] else 0
            total_hours = days * 24
            
            # For global calculation, average across all assets
            if not asset_id and result and result.get('asset_count', 0) > 0:
                # Total available hours across all assets
                asset_count = await self.db.fetchval("SELECT COUNT(*) FROM assets_metadata") or 1
                total_hours = days * 24 * asset_count
            
            availability = ((total_hours - downtime_hours) / total_hours * 100) if total_hours > 0 else 100.0
            availability = round(min(100.0, max(0.0, availability)), 2)
            
            # OEE is simplified as availability (assuming 100% performance and quality)
            oee = round(availability, 2)
            
        except Exception as e:
            logger.error(f"Erreur lors du calcul de l'availability/OEE: {e}", exc_info=True)
            availability = 100.0  # Default to 100% if calculation fails
            oee = 100.0
        
        # Calculate reliability based on anomaly rate
        reliability = 100.0
        try:
            start_date = datetime.now() - timedelta(days=days)
            anomaly_count = await self.db.fetchval(
                "SELECT COUNT(*) FROM anomaly_detections WHERE is_anomaly = true AND timestamp >= $1",
                start_date
            ) or 0
            # Reliability decreases with more anomalies (simple model)
            # 0 anomalies = 100%, 10+ anomalies = ~50%
            reliability = round(max(50.0, 100.0 - (anomaly_count * 5)), 2)
        except Exception as e:
            logger.debug(f"Could not calculate reliability: {e}")
            reliability = 100.0
        
        return KPISummary(
            mtbf=mtbf,
            mttr=mttr,
            oee=oee,
            availability=availability,
            reliability=reliability,
            timestamp=datetime.now()
        )
    
    async def save_kpi(self, kpi: KPI):
        """Save KPI to history"""
        try:
            import json
            metadata_json = json.dumps(kpi.metadata) if kpi.metadata else None
            
            await self.db.execute("""
                INSERT INTO kpis_history 
                (asset_id, metric_name, value, unit, timestamp, metadata)
                VALUES ($1, $2, $3, $4, $5, $6::jsonb)
            """, kpi.asset_id, kpi.metric_name, kpi.value, kpi.unit, kpi.timestamp, metadata_json)
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde du KPI: {e}", exc_info=True)

